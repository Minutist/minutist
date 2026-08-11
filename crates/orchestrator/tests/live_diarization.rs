//! Phase B — live diarization wiring integration tests.
//!
//! Two sub-cases, mirroring the crate's model-free + env-gated conventions:
//!
//! 1. **None-path regression guard (always runs).** With no live diarizer wired
//!    (`online_diarizer = None`), the full pipeline must emit `TranscriptSegment`
//!    events and persist a `transcript.json` whose segments all carry
//!    `speaker_id == None` — proving the live path is additive and transcription
//!    is unchanged when live diarization is off. This is the regression guard for
//!    the "must not break transcription" hard constraint.
//!
//! 2. **Positive case (env-gated on `MINUTIST_DIARIZE_EMB_PATH`, skip on
//!    unset — mirrors `diarizer/tests/online_embedding.rs`).** With a real
//!    `OnlineDiarizer` built from the embedding model, the emitted segments carry
//!    non-None sticky labels. The default `cargo test -p orchestrator` suite
//!    passes with this skipped (no model download), per
//!    `architecture/cross-cutting.md` "Automated-testing policy".
//!
//! The Silero VAD model is vendored at `resources/silero/silero_vad_v4.onnx` and
//! is always present, so the VAD → Accumulator → flush path runs for real.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use audio_capture::{AudioFrameBatch, AudioStreams};
use diarizer::{OnlineDiarizer, OnlineDiarizerConfig};
use minutist_common::{AppEvent, Segment};
use model_registry::ModelRegistry;
use orchestrator::test_support::{load_fixture_wav, StubAsrBackend};
use orchestrator::Orchestrator;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::{broadcast, mpsc};

// ---------------------------------------------------------------------------
// Fixtures + helpers (shared shape with pipeline_stub_test.rs)
// ---------------------------------------------------------------------------


fn samples_to_streams(
    samples: Vec<f32>,
    batch_size: usize,
) -> (AudioStreams, tokio::task::JoinHandle<()>) {
    let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(256);
    let (_meter_tx, meter_rx) = mpsc::channel(64);

    let streams = AudioStreams {
        samples: sample_rx,
        meter: meter_rx,
    };

    let handle = tokio::task::spawn_blocking(move || {
        let sample_rate = 16_000u64;
        let mut start_ms = 0u64;
        for chunk in samples.chunks(batch_size) {
            let duration_ms = (chunk.len() as u64 * 1000) / sample_rate;
            let batch = AudioFrameBatch {
                samples: chunk.to_vec(),
                start_ms,
                end_ms: start_ms + duration_ms,
            };
            if sample_tx.blocking_send(batch).is_err() {
                break;
            }
            start_ms += duration_ms;
        }
    });

    (streams, handle)
}

fn test_orchestrator(
    persistence_root: PathBuf,
) -> (Orchestrator, broadcast::Receiver<AppEvent>) {
    let settings_path = persistence_root.join(".live_diar_settings.json");
    let store = JsonFileStore::new(settings_path);
    let settings = SettingsHandle::new(store).expect("SettingsHandle construction");

    let (event_tx, event_rx) = broadcast::channel::<AppEvent>(512);

    let model_cache = persistence_root.join(".model_cache");
    let registry = ModelRegistry::new(model_cache, Vec::new(), event_tx.clone())
        .expect("ModelRegistry construction");

    let orch =
        Orchestrator::with_event_tx(settings, persistence_root, Arc::new(registry), event_tx);

    (orch, event_rx)
}

/// Resolve the gated embedding-model path, or `None` (→ skip) when unset.
fn embedding_path() -> Option<PathBuf> {
    std::env::var("MINUTIST_DIARIZE_EMB_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Drive a recording to completion and collect the emitted `TranscriptSegment`s
/// plus the persisted transcript.
async fn run_and_collect(
    orch: &Orchestrator,
    mut event_rx: broadcast::Receiver<AppEvent>,
    online_diarizer: Option<Arc<OnlineDiarizer>>,
    persistence_root: &Path,
) -> (Vec<Segment>, Vec<Segment>) {
    let samples = load_fixture_wav();
    let (streams, _feeder) = samples_to_streams(samples, 1600);

    let stub = Box::new(StubAsrBackend::new("hello from stub asr"));
    let meeting_id = orch
        .start_with_streams_and_backend(streams, stub, online_diarizer)
        .await
        .expect("start_with_streams_and_backend");

    // Let the feeder drain before stopping so the runner sees end-of-stream.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let meta = orch.stop().await.expect("stop recording");

    // Collect emitted TranscriptSegment events (allow up to 15 s).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut emitted: Vec<Segment> = Vec::new();
    while tokio::time::Instant::now() < deadline {
        match event_rx.try_recv() {
            Ok(AppEvent::TranscriptSegment { meeting_id: mid, segment }) => {
                assert_eq!(mid, meeting_id, "TranscriptSegment meeting_id must match");
                emitted.push(segment);
            }
            Ok(_) => {}
            Err(broadcast::error::TryRecvError::Empty) => {
                if !emitted.is_empty() {
                    // Give a brief grace window for trailing events, then stop.
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {}
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }

    assert!(
        !emitted.is_empty(),
        "expected at least one TranscriptSegment (is the Silero VAD model present?)"
    );

    let meeting_dir = persistence_root.join(meta.uuid.0.to_string());
    let transcript_json = std::fs::read_to_string(meeting_dir.join("transcript.json"))
        .expect("read transcript.json");
    let persisted: Vec<Segment> =
        serde_json::from_str(&transcript_json).expect("parse transcript.json");

    (emitted, persisted)
}

// ---------------------------------------------------------------------------
// 1. None-path regression guard (always runs)
// ---------------------------------------------------------------------------

/// With NO live diarizer wired, every emitted and persisted segment must carry
/// `speaker_id == None`. Proves the live path is additive and transcription is
/// untouched when live diarization is off.
#[tokio::test(flavor = "multi_thread")]
async fn none_diarizer_yields_all_none_speaker_ids() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");

    let (orch, event_rx) = test_orchestrator(persistence_root.clone());

    let (emitted, persisted) =
        run_and_collect(&orch, event_rx, None, &persistence_root).await;

    assert!(
        emitted.iter().all(|s| s.speaker_id.is_none()),
        "all emitted segments must have speaker_id == None when live diarization is off"
    );
    assert!(
        !persisted.is_empty(),
        "transcript.json must contain at least one segment"
    );
    assert!(
        persisted.iter().all(|s| s.speaker_id.is_none()),
        "all persisted segments must have speaker_id == None when live diarization is off"
    );
}

// ---------------------------------------------------------------------------
// 2. Positive case (env-gated)
// ---------------------------------------------------------------------------

/// With a real `OnlineDiarizer`, the emitted segments carry non-None sticky
/// labels. Skipped (no failure) when `MINUTIST_DIARIZE_EMB_PATH` is unset.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "model-gated: set MINUTIST_DIARIZE_EMB_PATH"]
async fn live_diarizer_populates_speaker_ids() {
    let _ = tracing_subscriber::fmt::try_init();

    let Some(emb_path) = embedding_path() else {
        eprintln!(
            "skipping live_diarizer_populates_speaker_ids: set \
             MINUTIST_DIARIZE_EMB_PATH to run"
        );
        return;
    };

    let diarizer = OnlineDiarizer::open(&emb_path, OnlineDiarizerConfig::default())
        .expect("open online diarizer with a valid embedding model");
    let diarizer = Arc::new(diarizer);

    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");

    let (orch, event_rx) = test_orchestrator(persistence_root.clone());

    let (emitted, persisted) =
        run_and_collect(&orch, event_rx, Some(diarizer), &persistence_root).await;

    // At least one emitted segment must carry a live label.
    assert!(
        emitted.iter().any(|s| s.speaker_id.is_some()),
        "expected at least one non-None live speaker_id with a real diarizer; got {emitted:?}"
    );
    // And the persisted transcript reflects the live labels (the on-stop pass is
    // gated on `diarization_enabled`, which is false here, so the live labels
    // are what reach disk).
    assert!(
        persisted.iter().any(|s| s.speaker_id.is_some()),
        "expected at least one non-None persisted live speaker_id; got {persisted:?}"
    );
}
