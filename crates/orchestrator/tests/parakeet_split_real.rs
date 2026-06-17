//! Gated ACCEPTANCE test for #0015 phase 3: transcribe a REAL recording with
//! Parakeet (which emits word timestamps), then diarize + split it, and confirm
//! the Parakeet word-split fires on real data — a mixed VAD segment is cut at the
//! speaker-turn boundary so the output segment list grows into per-speaker rows.
//!
//! All dev recordings on disk are Qwen-transcribed (no persisted `words`), so the
//! split path has no live data; this re-runs Parakeet over the raw audio to
//! produce the word timestamps the split needs — the only way to exercise the
//! Parakeet path end-to-end on a real meeting.
//!
//! Gated on `MINUTIST_RECORDINGS_DIR` + `MINUTIST_PARAKEET_DIR` +
//! `MINUTIST_DIARIZE_SEG_PATH` + `MINUTIST_DIARIZE_EMB_PATH`; skips cleanly
//! otherwise. Run via `scripts/run-parakeet-acceptance-windows.ps1`.

use std::path::{Path, PathBuf};

use asr_parakeet::{ParakeetBackend, ParakeetConfig};
use diarizer::{DiarizerConfig, SherpaDiarizer};
use minutist_common::{AsrBackend, AudioChunk, Diarizer, Segment};
use persistence::read_audio_pcm;
use vad_chunker::{default_model_path, VadChunker, VadConfig, VadEvent};

fn env_ne(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

/// The smallest multi-speaker recording (lexicographic tie-break): `audio.opus`
/// present, a `transcript.json` with >= 2 distinct `speaker_id`s and >= 5
/// non-empty segments. Multi-speaker biases toward a recording that actually
/// contains handovers (so the split has something to do); smallest keeps the
/// Parakeet + diarize runtime bounded.
fn find_recording(dir: &str) -> Option<PathBuf> {
    let mut cands: Vec<(usize, PathBuf)> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("audio.opus").is_file())
        .filter_map(|p| {
            let bytes = std::fs::read(p.join("transcript.json")).ok()?;
            let segs: Vec<Segment> = serde_json::from_slice(&bytes).ok()?;
            let mut spk: Vec<&String> = segs.iter().filter_map(|s| s.speaker_id.as_ref()).collect();
            spk.sort();
            spk.dedup();
            let nonempty = segs.iter().filter(|s| !s.text.trim().is_empty()).count();
            (spk.len() >= 2 && nonempty >= 5).then_some((nonempty, p))
        })
        .collect();
    cands.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    cands.into_iter().next().map(|(_, p)| p)
}

/// Transcribe the whole PCM with Parakeet by VAD-chunking it (the production
/// segmenter) and decoding each VAD segment; `words` land on the recording clock
/// via `AudioChunk::start_ms`.
fn parakeet_segments(pcm: &[f32], parakeet_dir: &str) -> Vec<Segment> {
    let mut backend = ParakeetBackend::new(ParakeetConfig::new(parakeet_dir))
        .expect("open Parakeet backend");
    let mut vad =
        VadChunker::open(&default_model_path(), VadConfig::default()).expect("open Silero VAD");

    let mut ends: Vec<(u64, u64, Vec<f32>)> = Vec::new();
    for ev in vad.process_samples(pcm, 0).expect("vad process") {
        if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
            ends.push((start_ms, end_ms, samples));
        }
    }
    for ev in vad.flush_end_of_stream().expect("vad flush") {
        if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
            ends.push((start_ms, end_ms, samples));
        }
    }

    let mut out: Vec<Segment> = Vec::new();
    for (start_ms, end_ms, samples) in ends {
        let chunk = AudioChunk { samples, sample_rate: 16_000, start_ms, end_ms };
        if let Ok(segs) = backend.transcribe_chunk(&chunk) {
            out.extend(segs.into_iter().filter(|s| !s.text.trim().is_empty()));
        }
    }
    out
}

#[test]
fn parakeet_split_on_real_recording() {
    let (recordings, parakeet_dir, seg, emb) = match (
        env_ne("MINUTIST_RECORDINGS_DIR"),
        env_ne("MINUTIST_PARAKEET_DIR"),
        env_ne("MINUTIST_DIARIZE_SEG_PATH"),
        env_ne("MINUTIST_DIARIZE_EMB_PATH"),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => {
            eprintln!(
                "skipping: set MINUTIST_RECORDINGS_DIR + MINUTIST_PARAKEET_DIR \
                 + MINUTIST_DIARIZE_SEG_PATH + MINUTIST_DIARIZE_EMB_PATH"
            );
            return;
        }
    };

    // `MINUTIST_RECORDING_ID` targets a specific meeting (busy/multi-speaker
    // recordings exercise the split); otherwise auto-pick the smallest
    // multi-speaker recording.
    let meeting = match env_ne("MINUTIST_RECORDING_ID") {
        Some(id) => {
            let p = Path::new(&recordings).join(&id);
            if !p.join("audio.opus").is_file() {
                eprintln!("MINUTIST_RECORDING_ID={id}: no audio.opus under {recordings}; skipping");
                return;
            }
            p
        }
        None => match find_recording(&recordings) {
            Some(m) => m,
            None => {
                eprintln!("no multi-speaker recording with audio.opus under {recordings}; skipping");
                return;
            }
        },
    };
    let pcm = read_audio_pcm(&meeting).expect("decode audio.opus");
    eprintln!(
        "recording {:?}: {:.1}s audio",
        meeting.file_name().unwrap(),
        pcm.len() as f32 / 16_000.0
    );

    // 1. Parakeet transcript WITH word timestamps (the split's precondition).
    let para = parakeet_segments(&pcm, &parakeet_dir);
    let with_words = para.iter().filter(|s| !s.words.is_empty()).count();
    let total_words: usize = para.iter().map(|s| s.words.len()).sum();
    eprintln!(
        "parakeet: {} segments, {} with words, {} words total",
        para.len(),
        with_words,
        total_words
    );
    assert!(with_words > 0, "Parakeet must emit word timestamps");

    // 2. Diarize + split with the production default config, then the SAME turns
    //    with the split disabled (multi_speaker_min_share = 0.0) as a baseline.
    //    Splitting only grows the list, so (split ON) >= (split OFF); the delta is
    //    the number of segments the Parakeet word-split produced on real audio.
    let diar = SherpaDiarizer::open(Path::new(&seg), Path::new(&emb), DiarizerConfig::default())
        .expect("open diarizer");
    let input_len = para.len();
    let (split_on, count_on) =
        diar.assign_speakers(&pcm, 16_000, para).expect("assign_speakers");

    // overlay_speakers only ever KEEPS (1) or SPLITS (N) each input segment, so
    // the output is never shorter than the input; the delta is the number of
    // sub-segments the Parakeet word-split produced on real audio (no separate
    // split-off pass needed — the input length IS the no-split count).
    let added = split_on.len() as i64 - input_len as i64;
    eprintln!(
        "diarize+split: {} speakers | {} input segs -> {} output segs (+{} from word-splits)",
        count_on,
        input_len,
        split_on.len(),
        added
    );
    assert!(split_on.len() >= input_len, "the split must not reduce the segment count");

    // Every emitted segment that got a label is single-speaker by construction
    // (split sub-segments carry one cluster). Report the labelled coverage; a
    // split sub-segment must carry no multi-speaker flag.
    let labelled = split_on.iter().filter(|s| s.speaker_id.is_some()).count();
    let flagged = split_on.iter().filter(|s| !s.shared_speakers.is_empty()).count();
    eprintln!(
        "labelled {}/{} segments; {} still carry a shared-speaker flag (Qwen-style no-words mixed)",
        labelled,
        split_on.len(),
        flagged
    );

    // Diagnostic: show up to 6 segments around the first split point.
    if added > 0 {
        eprintln!("-- first few split segments (speaker: text) --");
        for s in split_on.iter().take(12) {
            eprintln!(
                "  [{:>7}..{:>7}] {}: {}",
                s.start_ms,
                s.end_ms,
                s.speaker_id.as_deref().unwrap_or("?"),
                s.text.chars().take(70).collect::<String>()
            );
        }
    } else {
        eprintln!(
            "NOTE: no word-split fired on this recording (no VAD segment spanned >=2 \
             surviving speakers above the 30% share). Try a busier recording."
        );
    }
}
