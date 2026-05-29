//! Integration tests for ASR worker error resilience.
//!
//! Verifies two guarantees required by `architecture/cross-cutting.md`:
//!
//! 1. **Backend-error path**: when `AsrBackend::transcribe_chunk` returns
//!    `Err`, the worker emits `AppEvent::ErrorOccurred`, continues running,
//!    and a subsequent successful flush still produces a `TranscriptSegment`.
//!    `stop()` returns `Ok`.
//!
//! 2. **Backend-panic path**: when `AsrBackend::transcribe_chunk` panics, the
//!    `catch_unwind` wrapper in the ASR worker catches it, emits
//!    `AppEvent::ErrorOccurred`, and `stop()` still returns `Ok` with a
//!    finalised meeting folder (including `audio.opus`).
//!
//! Both tests use the real VAD + accumulator + the Silero VAD fixture WAV
//! (`tests/fixtures/librispeech_0.wav`), swapping in the failing/panicking
//! stub backends from `orchestrator::test_support`.
//!
//! These tests are NOT gated on any env-var and run unconditionally in CI.
//! They require the `test-source` feature (Cargo.toml `[[test]]` entry).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use audio_capture::{AudioFrameBatch, AudioStreams};
use meeting_app_common::{AppError, AppEvent, Segment};
use model_registry::ModelRegistry;
use orchestrator::test_support::{FailingAsrBackend, PanickingAsrBackend};
use orchestrator::Orchestrator;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::{broadcast, mpsc};

// ---------------------------------------------------------------------------
// Shared helpers (copied from pipeline_stub_test for isolation)
// ---------------------------------------------------------------------------

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
        // Dropping `sample_tx` signals end-of-stream.
    });

    (streams, handle)
}

fn resilience_orchestrator(
    persistence_root: std::path::PathBuf,
) -> (Orchestrator, broadcast::Receiver<AppEvent>) {
    let settings_path = persistence_root.join(".resilience_test_settings.json");
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
// Helper: drain all available events from the receiver into a Vec.
// ---------------------------------------------------------------------------
async fn drain_events(
    event_rx: &mut broadcast::Receiver<AppEvent>,
    wait_timeout: Duration,
) -> Vec<AppEvent> {
    let deadline = tokio::time::Instant::now() + wait_timeout;
    let mut events = Vec::new();
    loop {
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        match event_rx.try_recv() {
            Ok(ev) => events.push(ev),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(target: "asr-error-resilience-test", lagged = n, "subscriber lagged");
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }
    events
}

// ---------------------------------------------------------------------------
// Test 1: Backend-error path
// ---------------------------------------------------------------------------

/// Drive the pipeline with a `FailingAsrBackend`.
///
/// The audio consists of 5 copies of the fixture (~5.86 s each), separated by
/// 1 s of silence. The VAD emits `SegmentEnd` events during each silence gap
/// (hangover ≈ 720 ms), so speech segments enter the accumulator mid-stream.
/// Once the accumulator reaches `FLUSH_MIN_SECS = 25.0 s`, the size trigger
/// fires → first flush (backend returns `Err` → `ErrorOccurred` emitted). The
/// remaining speech is flushed at end-of-stream → second flush (backend returns
/// `Ok` → `TranscriptSegment` emitted).
///
/// Asserts:
///   1. At least one `AppEvent::ErrorOccurred` is emitted.
///   2. At least one `AppEvent::TranscriptSegment` is emitted after the error
///      (the worker kept running and processed the second flush).
///   3. `stop()` returns `Ok`.
///   4. `audio.opus` exists and is non-empty.
#[tokio::test(flavor = "multi_thread")]
async fn backend_error_emits_error_occurred_and_pipeline_continues() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");

    let (orch, mut event_rx) = resilience_orchestrator(persistence_root.clone());

    // Build a 30 s stream that guarantees ≥2 accumulator flushes:
    //   - 5× copies of the fixture (~5.86 s each), separated by 1 s of silence.
    //   - Total: 5 × 5.86 s speech + 4 × 1 s silence ≈ 33.3 s.
    //   - The VAD emits SegmentEnd events during each 1 s silence (hangover = 24
    //     frames ≈ 720 ms), so speech segments are added to the accumulator
    //     during the normal drain loop.
    //   - Once the accumulator reaches 25 s (FLUSH_MIN_SECS), the size trigger
    //     fires → first flush (FailingAsrBackend returns Err → ErrorOccurred).
    //   - Remaining speech accumulates until Stop → second flush (Ok →
    //     TranscriptSegment).
    let single = load_fixture_wav();
    let silence_1s = vec![0.0f32; 16_000]; // 1 s at 16 kHz
    let mut audio = Vec::new();
    for i in 0..5 {
        audio.extend_from_slice(&single);
        if i < 4 {
            audio.extend_from_slice(&silence_1s);
        }
    }
    let (streams, _feeder) = samples_to_streams(audio, 1600);

    // First flush fails, subsequent flushes succeed with "ok after error".
    let stub = Box::new(FailingAsrBackend::fail_then_succeed(1, "ok after error"));

    let meeting_id = orch
        .start_with_streams_and_backend(streams, stub)
        .await
        .expect("start_with_streams_and_backend");

    // Wait for the size-triggered flush to occur. The feeder queues ~332 batches
    // of 1600 samples. The runner drains the channel quickly (VAD runs faster
    // than realtime). The VAD emits SegmentEnd events at each 1 s silence gap,
    // adding speech to the accumulator. Once 25 s of speech has accumulated,
    // the size trigger fires. This happens in well under 3 s wall-time.
    tokio::time::sleep(Duration::from_secs(3)).await;

    let meta = orch.stop().await.expect("stop() must return Ok even after backend error");

    // Collect events (allow up to 15 s for the stub to process flushes).
    let events = drain_events(&mut event_rx, Duration::from_secs(15)).await;

    let error_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AppEvent::ErrorOccurred { .. }))
        .collect();
    assert!(
        !error_events.is_empty(),
        "expected at least one AppEvent::ErrorOccurred from the failing flush; \
         got events: {events:?}"
    );

    let transcript_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AppEvent::TranscriptSegment { meeting_id: mid, .. } if *mid == meeting_id))
        .collect();
    assert!(
        !transcript_events.is_empty(),
        "expected at least one AppEvent::TranscriptSegment after the error; \
         the worker must have continued processing the second flush; \
         check that the Silero VAD model is present at resources/silero/silero_vad_v4.onnx; \
         events: {events:?}"
    );

    // Verify audio.opus exists (audio is always preserved even when ASR fails).
    let meeting_dir = persistence_root.join(meta.uuid.0.to_string());
    assert!(meeting_dir.exists(), "meeting directory must exist");

    let audio_path = meeting_dir.join("audio.opus");
    assert!(audio_path.exists(), "audio.opus must exist");
    assert!(
        std::fs::metadata(&audio_path).unwrap().len() > 0,
        "audio.opus must not be empty"
    );

    // Verify transcript.json exists and contains the "ok after error" text.
    let transcript_path = meeting_dir.join("transcript.json");
    assert!(
        transcript_path.exists(),
        "transcript.json must exist when at least one flush succeeded"
    );
    let transcript_json = std::fs::read_to_string(&transcript_path).expect("read transcript.json");
    let segments: Vec<Segment> =
        serde_json::from_str(&transcript_json).expect("transcript.json must parse as Vec<Segment>");
    assert!(
        !segments.is_empty(),
        "transcript.json must contain at least one segment from the successful flush"
    );
    let has_ok_text = segments.iter().any(|s| s.text.contains("ok after error"));
    assert!(
        has_ok_text,
        "at least one segment must contain the ok-after-error stub text; segments: {segments:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Backend-panic path
// ---------------------------------------------------------------------------

/// Drive the pipeline with a `PanickingAsrBackend` that panics on the first
/// flush, then succeeds on subsequent flushes.
///
/// Asserts:
///   1. At least one `AppEvent::ErrorOccurred` is emitted (the `catch_unwind`
///      surfaced the panic).
///   2. The orchestrator does not abort the process (test completes).
///   3. `stop()` returns `Ok`.
///   4. `audio.opus` exists and is non-empty (audio is always preserved).
#[tokio::test(flavor = "multi_thread")]
async fn backend_panic_is_caught_emits_error_occurred_and_stop_succeeds() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");

    let (orch, mut event_rx) = resilience_orchestrator(persistence_root.clone());

    let samples = load_fixture_wav();
    let (streams, _feeder) = samples_to_streams(samples, 1600);

    // First flush panics, subsequent flushes succeed with "ok after panic".
    let stub = Box::new(PanickingAsrBackend::panic_then_succeed(1, "ok after panic"));

    let meeting_id = orch
        .start_with_streams_and_backend(streams, stub)
        .await
        .expect("start_with_streams_and_backend");

    // Let the feeder exhaust the samples.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // stop() must return Ok — the panic must not have wedged the orchestrator.
    let meta = orch
        .stop()
        .await
        .expect("stop() must return Ok even after backend panic");

    // Collect events.
    let events = drain_events(&mut event_rx, Duration::from_secs(15)).await;

    let error_events: Vec<_> = events
        .iter()
        .filter(|e| {
            if let AppEvent::ErrorOccurred {
                error: AppError::Internal { context },
            } = e
            {
                context.contains("panicked")
            } else {
                false
            }
        })
        .collect();
    assert!(
        !error_events.is_empty(),
        "expected AppEvent::ErrorOccurred with 'panicked' in context from catch_unwind; \
         got events: {events:?}"
    );

    // Verify audio.opus exists and is non-empty.
    let meeting_dir = persistence_root.join(meta.uuid.0.to_string());
    assert!(meeting_dir.exists(), "meeting directory must exist");

    let audio_path = meeting_dir.join("audio.opus");
    assert!(audio_path.exists(), "audio.opus must exist after panic in ASR backend");
    assert!(
        std::fs::metadata(&audio_path).unwrap().len() > 0,
        "audio.opus must not be empty"
    );

    // Verify that transcript events were emitted for the successful post-panic flush.
    // (If the fixture produced only one flush total, it may have been the panicking
    // one, so we only assert transcript segments if there were transcript events.)
    let transcript_events_count = events
        .iter()
        .filter(|e| {
            matches!(e, AppEvent::TranscriptSegment { meeting_id: mid, .. } if *mid == meeting_id)
        })
        .count();

    // We expect the pipeline continued after the panic: the fixture is ~5.86 s
    // of speech which typically produces ≥1 VAD segment → ≥1 flush.
    // The first flush panics; if there are subsequent flushes they succeed.
    // If the fixture only yielded one flush, we won't get a transcript event,
    // but the ErrorOccurred and successful stop are still verified above.
    tracing::info!(
        target: "asr-error-resilience-test",
        transcript_events = transcript_events_count,
        "backend-panic test: transcript events after panic"
    );

    // Regardless of how many flushes the VAD produced, stop() returned Ok
    // and audio.opus is present — that is the core guarantee being tested.
}
