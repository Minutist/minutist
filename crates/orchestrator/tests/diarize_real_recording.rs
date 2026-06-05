//! Gated diagnostic + regression guard: diarize a REAL recording and sweep the
//! agglomerative `cluster_threshold`.
//!
//! Reproduces the field bug where a single-speaker recording was split into 3
//! speakers, and shows how the count moves with the threshold so the production
//! default can be chosen from data rather than guessed. The `single_speaker_*`
//! synthetic fixture is too clean to catch this — a real recording's natural
//! vocal variation is what over-splits at a low threshold.
//!
//! Gated on `MEETING_APP_RECORDINGS_DIR` + `MEETING_APP_DIARIZE_SEG_PATH` +
//! `MEETING_APP_DIARIZE_EMB_PATH`; skips cleanly otherwise. Run via:
//!   make test-integration-diarize ARGS=diarize_real_recording
//! (add `-- --nocapture` to see the sweep).

use std::path::{Path, PathBuf};

use diarizer::{DiarizerConfig, SherpaDiarizer};
use meeting_app_common::{Diarizer, Segment};
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
    let d = SherpaDiarizer::open(
        seg_path,
        emb_path,
        DiarizerConfig {
            num_clusters: None,
            cluster_threshold: threshold,
        },
    )
    .expect("open diarizer");
    let mut copy = segs.to_vec();
    d.assign_speakers(audio, 16_000, &mut copy).expect("assign_speakers")
}

#[test]
fn diarize_real_recording_threshold_sweep() {
    let (recordings, seg, emb) = match (
        env_ne("MEETING_APP_RECORDINGS_DIR"),
        env_ne("MEETING_APP_DIARIZE_SEG_PATH"),
        env_ne("MEETING_APP_DIARIZE_EMB_PATH"),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            eprintln!(
                "skipping: set MEETING_APP_RECORDINGS_DIR + MEETING_APP_DIARIZE_SEG_PATH \
                 + MEETING_APP_DIARIZE_EMB_PATH"
            );
            return;
        }
    };

    let Some(meeting) = find_recording(&recordings) else {
        eprintln!("no recording with audio.opus + usable transcript under {recordings}; skipping");
        return;
    };
    let audio = read_audio_pcm(&meeting).expect("decode audio.opus");
    let segs: Vec<Segment> =
        serde_json::from_slice(&std::fs::read(meeting.join("transcript.json")).unwrap()).unwrap();

    eprintln!(
        "recording {:?}: {:.1}s audio, {} segments",
        meeting.file_name().unwrap(),
        audio.len() as f32 / 16_000.0,
        segs.len()
    );
    let seg_p = PathBuf::from(&seg);
    let emb_p = PathBuf::from(&emb);
    for t in [0.30f32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95] {
        let n = diarize_at(&seg_p, &emb_p, t, &audio, &segs);
        eprintln!("  cluster_threshold={t:.2} -> {n} speaker(s)");
    }
}
