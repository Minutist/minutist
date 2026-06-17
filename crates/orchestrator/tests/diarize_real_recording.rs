//! Gated diagnostic + regression guard: diarize a REAL recording and sweep the
//! agglomerative `cluster_threshold`.
//!
//! Reproduces the field bug where a single-speaker recording was split into 3
//! speakers, and shows how the count moves with the threshold so the production
//! default can be chosen from data rather than guessed. The `single_speaker_*`
//! synthetic fixture is too clean to catch this — a real recording's natural
//! vocal variation is what over-splits at a low threshold.
//!
//! Gated on `MINUTIST_RECORDINGS_DIR` + `MINUTIST_DIARIZE_SEG_PATH` +
//! `MINUTIST_DIARIZE_EMB_PATH`; skips cleanly otherwise. Run via:
//!   make test-integration-diarize ARGS=diarize_real_recording
//! (add `-- --nocapture` to see the sweep).

use std::path::{Path, PathBuf};

use diarizer::{DiarizerConfig, SherpaDiarizer};
use minutist_common::{Diarizer, Segment};
use persistence::read_audio_pcm;

fn env_ne(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

/// First recording (lexicographic) with `audio.opus` + a `transcript.json`
/// holding >= 3 non-empty segments — the directory whose audio we diarize.
fn find_recording(dir: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for d in dirs {
        if !d.join("audio.opus").is_file() {
            continue;
        }
        let Ok(bytes) = std::fs::read(d.join("transcript.json")) else {
            continue;
        };
        let Ok(segs) = serde_json::from_slice::<Vec<Segment>>(&bytes) else {
            continue;
        };
        if segs.iter().filter(|s| !s.text.trim().is_empty()).count() >= 3 {
            return Some(d);
        }
    }
    None
}

fn diarize_at(seg_path: &Path, emb_path: &Path, threshold: f32, audio: &[f32], segs: &[Segment]) -> u32 {
    // A raw threshold-only sweep: disable the temporal smoothing and the
    // post-cluster prune/cap (issue #63) so this curve shows the unpruned
    // agglomerative count vs `cluster_threshold` alone. The pruned shipped curve
    // lives in `crates/diarizer/tests/oversplit_eval.rs`.
    let d = SherpaDiarizer::open(
        seg_path,
        emb_path,
        DiarizerConfig {
            num_clusters: None,
            cluster_threshold: threshold,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            min_cluster_share: 0.0,
            min_cluster_segments: 0,
            max_speakers: None,
            multi_speaker_min_share: 0.0,
        },
    )
    .expect("open diarizer");
    let copy = segs.to_vec();
    let (_segs, count) = d.assign_speakers(audio, 16_000, copy).expect("assign_speakers");
    count
}

#[test]
fn diarize_real_recording_threshold_sweep() {
    let (recordings, seg, emb) = match (
        env_ne("MINUTIST_RECORDINGS_DIR"),
        env_ne("MINUTIST_DIARIZE_SEG_PATH"),
        env_ne("MINUTIST_DIARIZE_EMB_PATH"),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            eprintln!(
                "skipping: set MINUTIST_RECORDINGS_DIR + MINUTIST_DIARIZE_SEG_PATH \
                 + MINUTIST_DIARIZE_EMB_PATH"
            );
            return;
        }
    };

    // `MINUTIST_RECORDING_ID` targets a specific meeting; else auto-pick the first.
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
                eprintln!("no recording with audio.opus + usable transcript under {recordings}; skipping");
                return;
            }
        },
    };
    let audio = read_audio_pcm(&meeting).expect("decode audio.opus");
    let segs: Vec<Segment> =
        serde_json::from_slice(&std::fs::read(meeting.join("transcript.json")).unwrap()).unwrap();

    let old_speakers = {
        let mut s: Vec<&String> = segs.iter().filter_map(|x| x.speaker_id.as_ref()).collect();
        s.sort();
        s.dedup();
        s.len()
    };
    eprintln!(
        "recording {:?}: {:.1}s audio, {} segments, {} distinct speaker_id in the old transcript",
        meeting.file_name().unwrap(),
        audio.len() as f32 / 16_000.0,
        segs.len(),
        old_speakers
    );
    let seg_p = PathBuf::from(&seg);
    let emb_p = PathBuf::from(&emb);

    // Threshold sweep with the prune + smoothing OFF, so this isolates the
    // agglomerative count vs `cluster_threshold` alone.
    for t in [0.30f32, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95] {
        let n = diarize_at(&seg_p, &emb_p, t, &audio, &segs);
        eprintln!("  cluster_threshold={t:.2} (no prune) -> {n} speaker(s)");
    }

    // The SHIPPED default (0.75 + smoothing 0.3/0.5 + prune 0.02). Comparing this
    // to the no-prune count at 0.75 above isolates the prune's effect: if the
    // no-prune 0.75 count is higher, the prune collapsed those clusters; if it
    // already matches, the threshold/embedding did the merging.
    let d = SherpaDiarizer::open(&seg_p, &emb_p, DiarizerConfig::default()).expect("open default");
    let (_s, def_count) = d
        .assign_speakers(&audio, 16_000, segs.clone())
        .expect("assign default");
    eprintln!("  DEFAULT config (0.75 + smoothing + prune 0.02) -> {def_count} speaker(s)");
}
