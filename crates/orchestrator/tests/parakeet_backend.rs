//! Model-gated integration test: NVIDIA Parakeet TDT v3 via the production
//! [`asr_parakeet::ParakeetBackend`] on a real recording.
//!
//! Validates the hybrid-ASR primary engine end-to-end: it loads through the
//! sherpa-onnx binding, transcribes our 16 kHz mono audio, populates per-word
//! timestamps (the gap the Qwen mtmd path has), and runs near real-time on CPU.
//!
//! Gated on env (skips otherwise):
//!   MEETING_APP_PARAKEET_DIR   = extracted sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/
//!   MEETING_APP_RECORDINGS_DIR = the app meetings/ dir
//! Optional: MEETING_APP_SPIKE_MEETING_ID (defaults to the known test recording).
//!
//! Run: source tests-local.env, set MEETING_APP_PARAKEET_DIR, then
//!   cargo test -p orchestrator --test parakeet_backend -- --include-ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use asr_parakeet::{ParakeetBackend, ParakeetConfig};
use meeting_app_common::{AsrBackend, AudioChunk};

const DEFAULT_MEETING: &str = "f63ed109-f492-476a-ad71-fed93ae64669";

fn is_cjk(s: &str) -> bool {
    s.chars().any(|c| {
        let u = c as u32;
        (0x4E00..=0x9FFF).contains(&u) || (0x3040..=0x30FF).contains(&u) || (0xAC00..=0xD7AF).contains(&u)
    })
}

#[test]
#[ignore = "model-gated: set MEETING_APP_PARAKEET_DIR + MEETING_APP_RECORDINGS_DIR"]
fn parakeet_backend_transcribes_real_recording_with_timestamps() {
    let Some(model_dir) = std::env::var("MEETING_APP_PARAKEET_DIR").ok().filter(|s| !s.is_empty())
    else {
        eprintln!("skip: set MEETING_APP_PARAKEET_DIR to the extracted parakeet-tdt-0.6b-v3 dir");
        return;
    };
    let Some(recordings) = std::env::var("MEETING_APP_RECORDINGS_DIR").ok().filter(|s| !s.is_empty())
    else {
        eprintln!("skip: set MEETING_APP_RECORDINGS_DIR");
        return;
    };
    let meeting_id = std::env::var("MEETING_APP_SPIKE_MEETING_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_MEETING.to_string());
    let meeting_dir = PathBuf::from(&recordings).join(&meeting_id);

    // Decode to 16 kHz mono f32 (what Parakeet expects) via the app's reader.
    let samples = persistence::read_audio_pcm(&meeting_dir).expect("read_audio_pcm");
    let secs = samples.len() as f32 / 16_000.0;
    let chunk = AudioChunk {
        samples,
        sample_rate: 16_000,
        start_ms: 0,
        end_ms: (secs * 1000.0) as u64,
    };

    let mut backend =
        ParakeetBackend::new(ParakeetConfig::new(PathBuf::from(model_dir))).expect("ParakeetBackend::new");

    let t0 = Instant::now();
    let segments = backend.transcribe_chunk(&chunk).expect("transcribe_chunk");
    let rtf = t0.elapsed().as_secs_f32() / secs.max(0.001);

    assert_eq!(segments.len(), 1, "one segment per chunk (mirrors asr-runtime)");
    let seg = &segments[0];
    eprintln!("RTF {rtf:.3}, {} words, any_cjk={}", seg.words.len(), is_cjk(&seg.text));
    eprintln!("--- text ---\n{}", seg.text);
    for w in seg.words.iter().take(15) {
        eprintln!("  [{:6.2}-{:6.2}s] {}", w.start_ms as f32 / 1000.0, w.end_ms as f32 / 1000.0, w.text);
    }

    assert!(!seg.text.trim().is_empty(), "non-empty transcript");
    assert!(!seg.words.is_empty(), "per-word timestamps must be populated");
    assert!(!is_cjk(&seg.text), "English recording must produce no CJK");
    assert!(rtf < 1.5, "expected near real-time on CPU, got RTF {rtf:.3}");
    // Words are ordered and bounded by the chunk.
    assert!(seg.words.first().unwrap().start_ms <= seg.words.last().unwrap().end_ms);
    assert!(seg.words.last().unwrap().end_ms <= chunk.end_ms);
}
