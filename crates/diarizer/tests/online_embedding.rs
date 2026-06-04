//! Env-var-gated online-diarizer embedding tests.
//!
//! The live path needs only the real speaker-embedding ONNX model, so these are
//! gated on `MEETING_APP_DIARIZE_EMB_PATH` alone (no segmentation model — VAD
//! upstream supplies the segment boundaries). When the var is unset the test
//! logs a skip line and returns — the default `cargo test -p diarizer` suite
//! (no model, no GPU) passes with these skipped, per
//! `architecture/cross-cutting.md` "Automated-testing policy".
//!
//! To run:
//!   MEETING_APP_DIARIZE_EMB_PATH=/path/to/embedding.onnx \
//!   cargo test -p diarizer --test online_embedding
//!
//! Fixtures (committed, real speech — see `tests/fixtures/README.md`): two
//! genuinely distinct LibriSpeech readers.
//!   * `speaker_a.wav` — reader 1089.
//!   * `speaker_b.wav` — reader 1221.
//!   * `single_speaker_control.wav` — reader 1089 repeated with a silence gap.

use std::path::{Path, PathBuf};

use diarizer::{OnlineDiarizer, OnlineDiarizerConfig};
use meeting_app_common::AppError;

const SAMPLE_RATE: u32 = 16_000;

/// Resolve the gated embedding-model path, or `None` (→ skip) when unset. An
/// empty-string env var (`VAR=""`) is treated as unset → skip.
fn embedding_path() -> Option<PathBuf> {
    std::env::var("MEETING_APP_DIARIZE_EMB_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
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

#[test]
fn distinct_speakers_get_distinct_sticky_labels() {
    let Some(emb_path) = embedding_path() else {
        eprintln!(
            "skipping distinct_speakers_get_distinct_sticky_labels: set \
             MEETING_APP_DIARIZE_EMB_PATH to run"
        );
        return;
    };

    let diarizer = OnlineDiarizer::open(&emb_path, OnlineDiarizerConfig::default())
        .expect("open online diarizer with a valid embedding model");

    let a = read_fixture_pcm("speaker_a.wav");
    let b = read_fixture_pcm("speaker_b.wav");

    // Feed A, then B, then A again. A vs B must differ; the repeat of A must
    // return the SAME label as its first occurrence (sticky — no retroactive
    // relabel).
    let label_a1 = diarizer
        .assign_segment(&a, SAMPLE_RATE)
        .expect("assign speaker A");
    let label_b = diarizer
        .assign_segment(&b, SAMPLE_RATE)
        .expect("assign speaker B");
    let label_a2 = diarizer
        .assign_segment(&a, SAMPLE_RATE)
        .expect("re-assign speaker A");

    assert_ne!(label_a1, label_b, "distinct speakers must get distinct labels");
    assert_eq!(
        label_a1, label_a2,
        "the repeat of speaker A must reuse its first sticky label"
    );
    assert_eq!(
        diarizer.speaker_count().expect("speaker_count"),
        2,
        "exactly two distinct speakers expected"
    );
}

#[test]
fn single_speaker_yields_one_label() {
    let Some(emb_path) = embedding_path() else {
        eprintln!(
            "skipping single_speaker_yields_one_label: set \
             MEETING_APP_DIARIZE_EMB_PATH to run"
        );
        return;
    };

    let diarizer = OnlineDiarizer::open(&emb_path, OnlineDiarizerConfig::default())
        .expect("open online diarizer with a valid embedding model");

    // One speaker fed as several ~1 s chunks must yield exactly one label
    // throughout (no over-splitting).
    let pcm = read_fixture_pcm("single_speaker_control.wav");
    let chunk = SAMPLE_RATE as usize; // ~1 s
    let mut first_label: Option<String> = None;
    let mut chunks_fed = 0;
    for window in pcm.chunks(chunk) {
        // Skip the trailing partial chunk — too short to be a meaningful VAD
        // segment, and the silence gap can yield a degenerate embedding.
        if window.len() < chunk {
            break;
        }
        let label = diarizer
            .assign_segment(window, SAMPLE_RATE)
            .expect("assign a single-speaker chunk");
        chunks_fed += 1;
        match &first_label {
            None => first_label = Some(label),
            Some(first) => assert_eq!(
                &label, first,
                "single-speaker audio must stay on one label across chunks"
            ),
        }
    }

    assert!(chunks_fed > 1, "expected several full chunks from the control clip");
    assert_eq!(
        diarizer.speaker_count().expect("speaker_count"),
        1,
        "single-speaker control must yield exactly one distinct label"
    );
}

#[test]
fn guard_rejects_bad_input() {
    let Some(emb_path) = embedding_path() else {
        eprintln!("skipping guard_rejects_bad_input: set MEETING_APP_DIARIZE_EMB_PATH to run");
        return;
    };

    let diarizer = OnlineDiarizer::open(&emb_path, OnlineDiarizerConfig::default())
        .expect("open online diarizer with a valid embedding model");

    let a = read_fixture_pcm("speaker_a.wav");

    // A non-16 kHz buffer is rejected by the shared guard before any FFI.
    let bad_rate = diarizer
        .assign_segment(&a, 44_100)
        .expect_err("non-16 kHz must be rejected");
    assert!(matches!(bad_rate, AppError::InvalidInput { .. }));

    // An empty samples slice is rejected (an empty VAD segment is a caller bug).
    let empty = diarizer
        .assign_segment(&[], SAMPLE_RATE)
        .expect_err("empty segment must be rejected");
    assert!(matches!(empty, AppError::InvalidInput { .. }));
}
