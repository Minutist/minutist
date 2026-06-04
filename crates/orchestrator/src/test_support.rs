//! Test utilities for `orchestrator` tests.
//!
//! Gated behind `#[cfg(any(test, feature = "test-source"))]` so this code
//! never ships in production builds.
//!
//! Provides helpers to build a test `Orchestrator` with a tempdir-backed
//! persistence root, a default `SettingsHandle`, and a minimal `ModelRegistry`
//! backed by an empty manifest (no real models; ASR is skipped during tests).
//!
//! Also provides the `FlushBackpressureHarness` for directly exercising the
//! drop-oldest flush queue path without needing a real VAD model.

use std::path::PathBuf;
use std::sync::Arc;

use meeting_app_common::{AppError, AppEvent, AppResult, AsrBackend, AudioChunk, MeetingId, Segment};
use model_registry::ModelRegistry;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::broadcast;

use crate::runner::{dispatch_flush_pub, FlushPayload, FlushQueue};
use crate::Orchestrator;

/// Build an `Orchestrator` suitable for unit tests.
///
/// Uses a caller-supplied `persistence_root` so tests can use `tempfile`.
/// The `ModelRegistry` is initialised with an empty manifest and a tempdir
/// cache root — no models are present, so ASR is skipped during test runs.
pub fn test_orchestrator(persistence_root: PathBuf) -> Orchestrator {
    // Create a settings store that has no file (uses defaults).
    let settings_path = persistence_root.join(".test_settings.json");
    let store = JsonFileStore::new(settings_path);
    let handle =
        SettingsHandle::new(store).expect("test SettingsHandle construction should not fail");

    // Build a ModelRegistry with an empty manifest. The empty manifest means
    // list_models() returns [] and ensure() will return ModelNotFound, causing
    // the ASR worker to skip transcription. This is the correct test behaviour:
    // no real model present → no transcript segments → recording still finalises.
    let model_cache = persistence_root.join(".test_model_cache");
    let (event_tx, _) = broadcast::channel::<AppEvent>(256);
    let registry = ModelRegistry::new(model_cache, Vec::new(), event_tx)
        .expect("test ModelRegistry construction should not fail");
    let registry = Arc::new(registry);

    Orchestrator::new(handle, persistence_root, registry)
}

// ---------------------------------------------------------------------------
// Backpressure harness — direct flush queue access for tests
// ---------------------------------------------------------------------------

/// Harness for testing the drop-oldest backpressure behaviour of the flush
/// queue without going through the full VAD pipeline.
///
/// Lets tests push `FlushPayload`s directly and observe `ErrorOccurred` events.
pub struct FlushBackpressureHarness {
    flush_queue: FlushQueue,
    event_tx: broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
}

impl Default for FlushBackpressureHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl FlushBackpressureHarness {
    /// Create a new harness with a fresh flush queue and event bus.
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel::<AppEvent>(256);
        Self {
            flush_queue: FlushQueue::new(),
            event_tx,
            meeting_id: MeetingId::new(),
        }
    }

    /// Subscribe to the event bus to observe `ErrorOccurred` events.
    pub fn subscribe(&self) -> broadcast::Receiver<AppEvent> {
        self.event_tx.subscribe()
    }

    /// Dispatch one flush payload to the queue.
    ///
    /// If the queue is full, drops the oldest and emits `ErrorOccurred` on the
    /// shared event bus.
    pub fn dispatch(&self, start_ms: u64) {
        let payload = FlushPayload {
            samples: vec![0.0f32; 480],
            vad_segments: vec![(start_ms, start_ms + 500)],
            speaker_ids: vec![None],
            meeting_id: self.meeting_id,
        };
        dispatch_flush_pub(&self.flush_queue, payload, &self.event_tx);
    }

    /// Return the current queue length.
    pub fn queue_len(&self) -> usize {
        self.flush_queue
            .deque
            .lock()
            .expect("mutex poisoned")
            .len()
    }

    /// Return the `start_ms` values of all pending payloads (oldest first).
    pub fn queued_start_ms(&self) -> Vec<u64> {
        self.flush_queue
            .deque
            .lock()
            .expect("mutex poisoned")
            .iter()
            .map(|p| p.vad_segments[0].0)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Stub AsrBackend for non-gated pipeline tests
// ---------------------------------------------------------------------------

/// A minimal `AsrBackend` stub that returns a single canned `Segment` per call.
///
/// Used by `pipeline_stub_test` to exercise the full orchestrator pipeline
/// (VAD → Accumulator → flush dispatch → ASR worker → TranscriptSegment event
/// + transcript.json) without a real model.
pub struct StubAsrBackend {
    /// Text to include in each returned segment.
    pub canned_text: String,
    /// Optional artificial delay to simulate slow inference (for backpressure tests).
    pub delay_ms: u64,
}

impl StubAsrBackend {
    /// Create a stub that returns `text` in every segment with no artificial delay.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            canned_text: text.into(),
            delay_ms: 0,
        }
    }

    /// Create a stub with an artificial per-call delay (for backpressure tests).
    pub fn with_delay(text: impl Into<String>, delay_ms: u64) -> Self {
        Self {
            canned_text: text.into(),
            delay_ms,
        }
    }
}

impl AsrBackend for StubAsrBackend {
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
        if self.delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
        }
        Ok(vec![Segment {
            start_ms: chunk.start_ms,
            end_ms: chunk.end_ms,
            text: self.canned_text.clone(),
            speaker_id: None,
            confidence: None,
            words: vec![],
        }])
    }
}

// ---------------------------------------------------------------------------
// Error-returning stub — drives the Err(e) → ErrorOccurred path
// ---------------------------------------------------------------------------

/// An `AsrBackend` stub that always returns `Err(AppError::Inference { .. })`.
///
/// Used to verify that the ASR worker emits `AppEvent::ErrorOccurred` on a
/// backend error and that the pipeline continues (subsequent `Ok` flushes still
/// produce `TranscriptSegment` events).
pub struct FailingAsrBackend {
    /// After `fail_count` failing calls, subsequent calls succeed and return
    /// `ok_text`. Set to `usize::MAX` for an always-failing backend.
    pub fail_count: usize,
    calls: usize,
    pub ok_text: String,
}

impl FailingAsrBackend {
    /// Create a backend that always fails.
    pub fn always_failing() -> Self {
        Self {
            fail_count: usize::MAX,
            calls: 0,
            ok_text: String::new(),
        }
    }

    /// Create a backend that fails for the first `fail_count` calls, then
    /// succeeds with `ok_text`.
    pub fn fail_then_succeed(fail_count: usize, ok_text: impl Into<String>) -> Self {
        Self {
            fail_count,
            calls: 0,
            ok_text: ok_text.into(),
        }
    }
}

impl AsrBackend for FailingAsrBackend {
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
        let call = self.calls;
        self.calls += 1;
        if call < self.fail_count {
            Err(AppError::Inference {
                backend: "stub-failing".into(),
                context: format!("stub error on call {call}"),
            })
        } else {
            Ok(vec![Segment {
                start_ms: chunk.start_ms,
                end_ms: chunk.end_ms,
                text: self.ok_text.clone(),
                speaker_id: None,
                confidence: None,
                words: vec![],
            }])
        }
    }
}

// ---------------------------------------------------------------------------
// Panicking stub — drives the catch_unwind path
// ---------------------------------------------------------------------------

/// An `AsrBackend` stub that panics inside `transcribe_chunk`.
///
/// Used to verify that the `catch_unwind` wrapper in the ASR worker catches
/// the panic, emits `AppEvent::ErrorOccurred`, and allows the worker to
/// continue (and `stop()` to return `Ok`).
///
/// After `panic_count` panicking calls, subsequent calls succeed and return
/// `ok_text`. Set `panic_count` to `usize::MAX` for an always-panicking backend.
pub struct PanickingAsrBackend {
    pub panic_count: usize,
    calls: usize,
    pub ok_text: String,
}

impl PanickingAsrBackend {
    /// Create a backend that always panics.
    pub fn always_panicking() -> Self {
        Self {
            panic_count: usize::MAX,
            calls: 0,
            ok_text: String::new(),
        }
    }

    /// Create a backend that panics for the first `panic_count` calls, then
    /// succeeds with `ok_text`.
    pub fn panic_then_succeed(panic_count: usize, ok_text: impl Into<String>) -> Self {
        Self {
            panic_count,
            calls: 0,
            ok_text: ok_text.into(),
        }
    }
}

impl AsrBackend for PanickingAsrBackend {
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
        let call = self.calls;
        self.calls += 1;
        if call < self.panic_count {
            panic!("stub panic on call {call}");
        }
        Ok(vec![Segment {
            start_ms: chunk.start_ms,
            end_ms: chunk.end_ms,
            text: self.ok_text.clone(),
            speaker_id: None,
            confidence: None,
            words: vec![],
        }])
    }
}
