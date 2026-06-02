//! Env-var-gated diarization accuracy tests.
//!
//! These need the two real ONNX models, so they are gated on
//! `MEETING_APP_DIARIZE_SEG_PATH` (pyannote segmentation) and
//! `MEETING_APP_DIARIZE_EMB_PATH` (speaker embedding). When either env var is
//! unset the test logs a skip line and returns `Ok` — the default `cargo test
//! -p diarizer` suite (no model, no GPU) passes with these skipped, per
//! `architecture/cross-cutting.md` "Automated-testing policy".
//!
//! To run:
//!   MEETING_APP_DIARIZE_SEG_PATH=/path/to/segmentation.onnx \
//!   MEETING_APP_DIARIZE_EMB_PATH=/path/to/embedding.onnx \
//!   cargo test -p diarizer --test accuracy
//!
//! Fixtures (committed, synthetic real-speech — see `tests/fixtures/README.md`):
//!   * `two_speakers_synth.wav`     — A then gap then pitch-shifted-A (= B).
//!   * `single_speaker_control.wav` — A then gap then A (one speaker).

use std::path::{Path, PathBuf};

use diarizer::{DiarizerConfig, SherpaDiarizer};
use meeting_app_common::{Diarizer, Segment};

const SAMPLE_RATE: u32 = 16_000;
/// Speaker A occupies `[0, A_END_MS)` of the two-speaker fixture; speaker B the
/// remainder after the gap. See `tests/fixtures/README.md`.
const A_END_MS: u64 = 5_855;
const GAP_END_MS: u64 = 6_255;
const SEGMENT_MS: u64 = 1_000;

/// Resolve the gated model paths, or `None` (→ skip) when either is unset.
fn model_paths() -> Option<(PathBuf, PathBuf)> {
    let seg = std::env::var("MEETING_APP_DIARIZE_SEG_PATH").ok()?;
    let emb = std::env::var("MEETING_APP_DIARIZE_EMB_PATH").ok()?;
    Some((PathBuf::from(seg), PathBuf::from(emb)))
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Decode a committed 16 kHz mono s16 WAV fixture to f32 PCM in [-1, 1].
fn read_fixture_pcm(name: &str) -> Vec<f32> {
    let path = fixture_path(name);
    let mut reader =
        hound::WavReader::open(&path).unwrap_or_else(|e| panic!("open {:?}: {e}", path));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "fixture must be mono");
    assert_eq!(spec.sample_rate, SAMPLE_RATE, "fixture must be 16 kHz");
    reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<_, _>>()
        .expect("reading samples")
}

/// Tile `[0, total_ms)` into `SEGMENT_MS` windows, skipping any window whose
/// midpoint falls in the silence gap (no ground-truth speaker there).
/// Returns the segments plus a parallel ground-truth label list ("A"/"B").
fn build_segments(total_ms: u64) -> (Vec<Segment>, Vec<&'static str>) {
    let mut segments = Vec::new();
    let mut truth = Vec::new();
    let mut start = 0;
    while start + SEGMENT_MS <= total_ms {
        let end = start + SEGMENT_MS;
        let mid = (start + end) / 2;
        // Skip the gap region — there's no speech to attribute there.
        if (A_END_MS..GAP_END_MS).contains(&mid) {
            start = end;
            continue;
        }
        let label = if mid < A_END_MS { "A" } else { "B" };
        segments.push(Segment {
            start_ms: start,
            end_ms: end,
            text: String::new(),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
        });
        truth.push(label);
        start = end;
    }
    (segments, truth)
}

/// Permutation-invariant accuracy: the predicted labels are a clustering, so
/// the assignment "A"/"B" may be swapped vs ground truth. Try both the identity
/// and the swapped mapping of the two ground-truth labels and take the best.
fn permutation_invariant_accuracy(segments: &[Segment], truth: &[&str]) -> f64 {
    assert_eq!(segments.len(), truth.len());
    let n = segments.len();
    if n == 0 {
        return 1.0;
    }
    // Two ground-truth classes: A↔predicted-X, B↔predicted-Y. Score both
    // direct and swapped, since cluster labels are arbitrary.
    let predicted: Vec<Option<&str>> =
        segments.iter().map(|s| s.speaker_id.as_deref()).collect();

    // Collect the distinct predicted labels (≤2 expected for this fixture).
    let mut labels: Vec<&str> = Vec::new();
    for p in predicted.iter().flatten() {
        if !labels.contains(p) {
            labels.push(p);
        }
    }

    let score_with = |map_a: Option<&str>, map_b: Option<&str>| -> usize {
        truth
            .iter()
            .zip(predicted.iter())
            .filter(|(t, p)| {
                let expect = if **t == "A" { map_a } else { map_b };
                **p == expect
            })
            .count()
    };

    // Map ground-truth A/B onto each ordering of the predicted labels.
    let (l0, l1) = (labels.first().copied(), labels.get(1).copied());
    let direct = score_with(l0, l1);
    let swapped = score_with(l1, l0);
    direct.max(swapped) as f64 / n as f64
}

#[test]
fn two_speaker_accuracy_at_least_80pct() {
    let Some((seg_path, emb_path)) = model_paths() else {
        eprintln!(
            "skipping two_speaker_accuracy_at_least_80pct: set \
             MEETING_APP_DIARIZE_SEG_PATH and MEETING_APP_DIARIZE_EMB_PATH to run"
        );
        return;
    };

    let pcm = read_fixture_pcm("two_speakers_synth.wav");
    let total_ms = (pcm.len() as u64 * 1000) / SAMPLE_RATE as u64;
    let (mut segments, truth) = build_segments(total_ms);

    // Two known speakers → exact-cluster mode for a fair accuracy measure.
    let config = DiarizerConfig {
        num_clusters: Some(2),
        cluster_threshold: 0.5,
    };
    let diarizer =
        SherpaDiarizer::open(&seg_path, &emb_path, config).expect("open diarizer with valid models");

    let count = diarizer
        .assign_speakers(&pcm, SAMPLE_RATE, &mut segments)
        .expect("assign_speakers succeeds with valid models");
    assert_eq!(count, 2, "expected exactly 2 distinct speakers, got {count}");

    let acc = permutation_invariant_accuracy(&segments, &truth);
    assert!(
        acc >= 0.80,
        "permutation-invariant accuracy {acc:.3} below 0.80 threshold"
    );
}

#[test]
fn single_speaker_control_one_label() {
    let Some((seg_path, emb_path)) = model_paths() else {
        eprintln!(
            "skipping single_speaker_control_one_label: set \
             MEETING_APP_DIARIZE_SEG_PATH and MEETING_APP_DIARIZE_EMB_PATH to run"
        );
        return;
    };

    let pcm = read_fixture_pcm("single_speaker_control.wav");
    let total_ms = (pcm.len() as u64 * 1000) / SAMPLE_RATE as u64;
    // One window per second across the whole clip (the gap windows just get
    // None; they don't affect the distinct-label count).
    let mut segments = Vec::new();
    let mut start = 0;
    while start + SEGMENT_MS <= total_ms {
        segments.push(Segment {
            start_ms: start,
            end_ms: start + SEGMENT_MS,
            text: String::new(),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
        });
        start += SEGMENT_MS;
    }

    // Conservative single-speaker path: num_clusters = Some(1) so one speaker
    // is not over-split.
    let config = DiarizerConfig {
        num_clusters: Some(1),
        cluster_threshold: 0.5,
    };
    let diarizer =
        SherpaDiarizer::open(&seg_path, &emb_path, config).expect("open diarizer with valid models");

    let count = diarizer
        .assign_speakers(&pcm, SAMPLE_RATE, &mut segments)
        .expect("assign_speakers succeeds with valid models");
    assert_eq!(
        count, 1,
        "single-speaker control must yield exactly one distinct label, got {count}"
    );
}

#[test]
fn rejects_non_16khz() {
    // No model needed: the sample-rate guard fires before any sherpa call.
    // We still need a constructed diarizer, which needs models — so this only
    // asserts the guard when models are present; otherwise it skips.
    let Some((seg_path, emb_path)) = model_paths() else {
        eprintln!("skipping rejects_non_16khz: model env vars unset");
        return;
    };
    let diarizer =
        SherpaDiarizer::open(&seg_path, &emb_path, DiarizerConfig::default()).expect("open");
    let mut segments = vec![Segment {
        start_ms: 0,
        end_ms: 1_000,
        text: String::new(),
        speaker_id: None,
        confidence: None,
        words: Vec::new(),
    }];
    let err = diarizer
        .assign_speakers(&[0.0f32; 8_000], 8_000, &mut segments)
        .expect_err("8 kHz must be rejected");
    match err {
        meeting_app_common::AppError::InvalidInput { .. } => {}
        other => panic!("expected InvalidInput, got {other}"),
    }
}
