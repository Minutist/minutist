//! Non-gated integration test for the live transcription pipeline using a
//! stub `AsrBackend`.
//!
//! Exercises the full path without needing a ~1 GB model file:
//!
//!   WAV fixture → AudioStreams → orchestrator → VadChunker → batched-VAD
//!   accumulator → flush dispatch → StubAsrBackend → TranscriptSegment events
//!   + transcript.json on disk.
//!
//! The Silero VAD model is vendored at `resources/silero/silero_vad_v4.onnx`
//! and is always present in the repo, so this test runs unconditionally in CI.
//!
//! The stub `AsrBackend` returns a single canned `Segment` per call; the test
//! asserts that at least one `AppEvent::TranscriptSegment` is emitted and that
//! `transcript.json` is written to disk after stop.
//!
//! Integration tests for the orchestrator live in `crates/orchestrator/tests/`
//! per `architecture/cross-cutting.md` — Testing section.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use audio_capture::{AudioFrameBatch, AudioStreams};
use minutist_common::{AppEvent, Segment};
use model_registry::ModelRegistry;
use orchestrator::test_support::StubAsrBackend;
use orchestrator::Orchestrator;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::{broadcast, mpsc};

// ---------------------------------------------------------------------------
// Test fixture helpers
// ---------------------------------------------------------------------------

/// Read `tests/fixtures/librispeech_0.wav` and return all samples as f32.
///
/// The fixture is 16 kHz mono i16 PCM, ~5.86 s.
fn load_fixture_wav() -> Vec<f32> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/librispeech_0.wav");
    let mut reader = hound::WavReader::open(&fixture)
        .unwrap_or_else(|e| panic!("cannot open {:?}: {e}", fixture));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "fixture must be mono");
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
    reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<_, _>>()
        .expect("reading samples")
}

/// Build `AudioStreams` from pre-loaded samples, chunked into batches of
/// `batch_size` samples each.
///
/// Returns the streams (passed to the orchestrator) and a join handle for the
/// feeder task. Drop the `AudioStreams` sender after feeding to signal
/// end-of-stream to the runner.
fn samples_to_streams(
    samples: Vec<f32>,
    batch_size: usize,
) -> (AudioStreams, tokio::task::JoinHandle<()>) {
    let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(256);
    // Meter channel: send nothing; the runner ignores empty meter.
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
            // Block until the channel accepts the batch; if the receiver
            // is gone, break.
            if sample_tx.blocking_send(batch).is_err() {
                break;
            }
            start_ms += duration_ms;
        }
        // Dropping `sample_tx` signals end-of-stream to the runner.
    });

    (streams, handle)
}

// ---------------------------------------------------------------------------
// Orchestrator constructor for this test
// ---------------------------------------------------------------------------

fn pipeline_orchestrator(
    persistence_root: std::path::PathBuf,
) -> (Orchestrator, broadcast::Receiver<AppEvent>) {
    let settings_path = persistence_root.join(".pipeline_test_settings.json");
    let store = JsonFileStore::new(settings_path);
    let settings = SettingsHandle::new(store).expect("SettingsHandle construction");

    let (event_tx, event_rx) = broadcast::channel::<AppEvent>(512);

    let model_cache = persistence_root.join(".model_cache");
    let registry = ModelRegistry::new(model_cache, Vec::new(), event_tx.clone())
        .expect("ModelRegistry construction");

    let orch = Orchestrator::with_event_tx(
        settings,
        persistence_root,
        Arc::new(registry),
        event_tx,
    );

    (orch, event_rx)
}

// ---------------------------------------------------------------------------
// Test: stub-backend pipeline emits TranscriptSegment and writes transcript.json
// ---------------------------------------------------------------------------

/// Full pipeline with a stub ASR backend (no model download required).
///
/// This test always runs — it is NOT gated on any env var.
///
/// Verifies:
/// 1. At least one `AppEvent::TranscriptSegment` is emitted.
/// 2. The meeting folder contains `audio.opus`, `metadata.json`, and
///    `transcript.json` after stop.
/// 3. `transcript.json` parses as `Vec<Segment>` with ≥1 segment.
#[tokio::test(flavor = "multi_thread")]
async fn stub_backend_pipeline_emits_transcript_and_writes_transcript_json() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");

    let (orch, mut event_rx) = pipeline_orchestrator(persistence_root.clone());

    // Subscribe before start so we don't miss any events.
    // (event_rx is already subscribed from pipeline_orchestrator.)

    // Load the fixture WAV and create streams.
    let samples = load_fixture_wav();
    let sample_count = samples.len();
    tracing::info!(
        target: "pipeline-stub-test",
        samples = sample_count,
        "loaded fixture WAV"
    );

    // Feed in 1600-sample batches (100 ms at 16 kHz).
    let (streams, _feeder) = samples_to_streams(samples, 1600);

    // Inject a stub backend that returns a fixed transcript text.
    let stub = Box::new(StubAsrBackend::new("hello from stub asr"));

    let meeting_id = orch
        .start_with_streams_and_backend(streams, stub, None)
        .await
        .expect("start_with_streams_and_backend");

    // Wait for the feeder to exhaust the samples channel. Since
    // `generate_streams` pre-fills the channel, we give it time to drain.
    // The runner will process batches and the accumulator will fill.
    // On stop, the remaining accumulator content is flushed.
    //
    // Allow up to 15 s for the stub backend to process the flush.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);

    // Stop the recording. This triggers the end-of-stream VAD flush and
    // dispatches remaining accumulator content to the ASR worker.
    //
    // We need to wait until the sample feeder has finished before stopping,
    // so the runner sees the disconnected samples channel and processes
    // everything. Give the feeder a moment to complete.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let meta = orch.stop().await.expect("stop recording");

    // Wait for the transcript event (comes from the ASR worker after stop).
    let mut got_transcript_event = false;
    while tokio::time::Instant::now() < deadline {
        match event_rx.try_recv() {
            Ok(AppEvent::TranscriptSegment { meeting_id: mid, .. }) => {
                assert_eq!(
                    mid, meeting_id,
                    "TranscriptSegment meeting_id must match the active recording"
                );
                got_transcript_event = true;
                break;
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(target: "pipeline-stub-test", lagged = n, "subscriber lagged");
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }

    assert!(
        got_transcript_event,
        "expected at least one AppEvent::TranscriptSegment; \
         the pipeline may not have received VAD segments (check that the Silero \
         VAD model is present at resources/silero/silero_vad_v4.onnx)"
    );

    // Derive the meeting directory.
    let meeting_dir = persistence_root.join(meta.uuid.0.to_string());

    assert!(meeting_dir.exists(), "meeting directory must exist");

    let audio_path = meeting_dir.join("audio.opus");
    assert!(audio_path.exists(), "audio.opus must exist");
    assert!(
        std::fs::metadata(&audio_path).unwrap().len() > 0,
        "audio.opus must not be empty"
    );

    let meta_path = meeting_dir.join("metadata.json");
    assert!(meta_path.exists(), "metadata.json must exist");

    let transcript_path = meeting_dir.join("transcript.json");
    assert!(
        transcript_path.exists(),
        "transcript.json must exist after recording with stub backend"
    );

    let transcript_json =
        std::fs::read_to_string(&transcript_path).expect("read transcript.json");
    let segments: Vec<Segment> =
        serde_json::from_str(&transcript_json).expect("transcript.json must parse as Vec<Segment>");

    assert!(
        !segments.is_empty(),
        "transcript.json must contain at least one segment"
    );

    // Verify the stub text appears in the transcript.
    let has_stub_text = segments.iter().any(|s| s.text.contains("hello"));
    assert!(
        has_stub_text,
        "at least one segment must contain the stub text; segments: {segments:?}"
    );
}
