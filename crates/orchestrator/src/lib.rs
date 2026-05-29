//! `orchestrator` — Phase 2 live recording pipeline.
//!
//! Owns the recording lifecycle (start / pause / resume / stop), wires
//! `audio-capture → vad-chunker → asr-runtime → persistence`, and fans out
//! `AppEvent` to subscribers.
//!
//! Phase 2 adds the live pipeline: VAD → batched-VAD accumulator → ASR →
//! transcript events. See `runner.rs` for implementation details and
//! `architecture/cross-cutting.md` "ASR chunking constraint" for constraints.
//!
//! ## Threading model
//!
//! - `Orchestrator` is `Send + Sync` and intended to live behind an `Arc`.
//! - A `tokio::sync::Mutex<OrchestratorInner>` serialises state transitions.
//! - The capture-drain runner runs as one `tokio::task::spawn_blocking` task
//!   per recording session. It owns `AudioStreams`, `MeetingWriter`, and
//!   `VadChunker`.
//! - A separate `spawn_blocking` ASR worker drains flush payloads from the
//!   runner via a shared bounded queue (`Arc<Mutex<VecDeque>>` + `Notify`,
//!   capacity 4) with drop-oldest backpressure: on overflow the runner pops
//!   the OLDEST pending flush and emits `AppEvent::ErrorOccurred` (audio is
//!   preserved in `audio.opus`; only the live transcript for the dropped
//!   flush is lost). See `runner.rs`.
//! - Events are broadcast via `tokio::sync::broadcast::channel(256)`.
//!   Slow subscribers drop old events (broadcast semantics); a `tracing::warn!`
//!   fires when a subscriber reports lag.
//!
//! ## Tracing
//!
//! All log calls use `target: "orchestrator"`.
//!
//! ## No Tauri
//!
//! This crate does **not** import `tauri::*`. Tauri glue lives only in
//! `ipc-bridge` and `app-main`.

pub mod error;
mod runner;
mod state;

#[cfg(any(test, feature = "test-source"))]
pub mod test_support;

#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;

use audio_capture::AudioCaptureManager;
use chrono::{DateTime, Utc};
use meeting_app_common::{
    AppError, AppEvent, AppResult, AudioFormat, MeetingId, MeetingMeta, ModelId, ModelStatus,
    RecordingState,
};
use model_registry::ModelRegistry;
use persistence::MeetingWriter;
use settings::SettingsHandle;
use state::{
    transition_idle, transition_pause, transition_resume, transition_start, transition_stop,
    InternalState,
};
use tokio::sync::{broadcast, Mutex};

pub use error::Error;

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// The recording orchestrator. `Send + Sync`; intended to live in `Arc<Orchestrator>`.
pub struct Orchestrator {
    settings: SettingsHandle,
    persistence_root: PathBuf,
    model_registry: Arc<ModelRegistry>,
    /// Internal state machine, serialised by a mutex.
    inner: Mutex<OrchestratorInner>,
    /// Broadcast channel for `AppEvent`. Capacity 256 (~8 s of meter at 30 Hz).
    event_tx: broadcast::Sender<AppEvent>,
}

struct OrchestratorInner {
    state: InternalState,
    /// `AudioCaptureManager` held alive during a session. `AudioCaptureManager`
    /// is `Send` but not `Sync`; keeping it inside the mutex satisfies both.
    capture: Option<AudioCaptureManager>,
    /// Handle to the running drain task, present while Recording or Paused.
    runner: Option<runner::RunnerHandle>,
    /// Wall-clock instant the current recording started.
    started_at: Option<DateTime<Utc>>,
}

impl Orchestrator {
    /// Construct an `Orchestrator`.
    ///
    /// `persistence_root` is the directory under which per-meeting folders are
    /// created (typically `{app-data}/meetings/`). The caller resolves the
    /// platform app-data path; this crate carries no `tauri::*` dependency.
    ///
    /// `model_registry` is used by the live pipeline to locate and lazy-load
    /// the ASR model on the first flush.
    ///
    /// **Breaking change from Phase 1**: this constructor now requires a
    /// `model_registry` parameter. Callers in `src-tauri` must be updated
    /// (Stream E/F responsibility).
    pub fn new(
        settings: SettingsHandle,
        persistence_root: PathBuf,
        model_registry: Arc<ModelRegistry>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self::with_event_tx(settings, persistence_root, model_registry, event_tx)
    }

    /// Construct an `Orchestrator` sharing an externally-owned event bus.
    ///
    /// Used by `app-main` so the `ModelRegistry` and the orchestrator emit
    /// `AppEvent`s onto the *same* broadcast channel — the IPC forwarder
    /// subscribes once via [`subscribe_events`](Self::subscribe_events) and
    /// sees both orchestrator events (meter, state, transcript) and registry
    /// events (`ModelDownloadProgress`). Construct the channel, pass
    /// `event_tx.clone()` to `ModelRegistry::new`, and pass `event_tx` here.
    pub fn with_event_tx(
        settings: SettingsHandle,
        persistence_root: PathBuf,
        model_registry: Arc<ModelRegistry>,
        event_tx: broadcast::Sender<AppEvent>,
    ) -> Self {
        Orchestrator {
            settings,
            persistence_root,
            model_registry,
            inner: Mutex::new(OrchestratorInner {
                state: InternalState::Idle,
                capture: None,
                runner: None,
                started_at: None,
            }),
            event_tx,
        }
    }

    // ------------------------------------------------------------------
    // Model registry surface
    // ------------------------------------------------------------------

    /// Snapshot the runtime status of all known models.
    ///
    /// Thin wrapper over `ModelRegistry::list_models` so that the IPC bridge
    /// does not need a direct `model-registry` dependency.
    pub fn list_models(&self) -> Vec<ModelStatus> {
        self.model_registry.list_models()
    }

    /// Ensure a model is downloaded and hash-verified.
    ///
    /// Wraps `ModelRegistry::ensure` for the webview first-run flow. Returns
    /// `Ok(())` when the model is ready for use.
    pub async fn ensure_model(&self, model_id: &ModelId) -> AppResult<()> {
        self.model_registry.ensure(model_id).await.map(|_| ())
    }

    // ------------------------------------------------------------------
    // Public command surface
    // ------------------------------------------------------------------

    /// Start a new recording session.
    ///
    /// Opens an audio capture device (`device_id = None` → OS default),
    /// creates a per-meeting folder under `persistence_root`, and starts the
    /// drain runner task.
    ///
    /// Returns the new `MeetingId`.
    ///
    /// # Errors
    ///
    /// `AppError::InvalidInput` if not in `Idle` state.
    pub async fn start(&self, device_id: Option<String>) -> AppResult<MeetingId> {
        let mut guard = self.inner.lock().await;
        let (meeting_id, started_at_ms) = transition_start(&mut guard.state)?;

        let started_at =
            DateTime::<Utc>::from_timestamp_millis(started_at_ms as i64).unwrap_or_else(Utc::now);
        guard.started_at = Some(started_at);

        // Resolve device: caller wins; fall back to settings; then OS default.
        let resolved_device = device_id.or_else(|| self.settings.current().input_device_id);

        // Open and start audio capture.
        let mut capture = match AudioCaptureManager::open(resolved_device) {
            Ok(c) => c,
            Err(e) => {
                guard.state = InternalState::Idle;
                return Err(e);
            }
        };

        let streams = match capture.start(32, 64) {
            Ok(s) => s,
            Err(e) => {
                guard.state = InternalState::Idle;
                return Err(e);
            }
        };

        let audio_format = AudioFormat {
            codec: "opus".into(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        };

        let writer = match MeetingWriter::open(&self.persistence_root, meeting_id, audio_format) {
            Ok(w) => w,
            Err(e) => {
                guard.state = InternalState::Idle;
                // Stop the capture stream we already started.
                let _ = capture.stop();
                return Err(e);
            }
        };

        let runner_handle = runner::spawn_runner(
            streams,
            writer,
            self.event_tx.clone(),
            Arc::clone(&self.model_registry),
            meeting_id,
        );

        guard.capture = Some(capture);
        guard.runner = Some(runner_handle);

        let new_state = guard.state.as_public();
        self.emit(AppEvent::StateChanged { state: new_state });

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            "recording started"
        );

        Ok(meeting_id)
    }

    /// Pause the current recording.
    ///
    /// Pauses the capture stream and instructs the runner to pause the
    /// `MeetingWriter` (option-b gap encoding per `architecture/components.md`).
    ///
    /// # Errors
    ///
    /// `AppError::InvalidInput` if not in `Recording` state.
    pub async fn pause(&self) -> AppResult<()> {
        let mut guard = self.inner.lock().await;
        let _meeting_id = transition_pause(&mut guard.state)?;

        // Pause the audio capture stream first so no new samples arrive while
        // the writer is transitioning.
        if let Some(capture) = &mut guard.capture {
            capture.pause()?;
        }

        // Instruct the runner to call `MeetingWriter::pause`.
        if let Some(runner) = &guard.runner {
            let _ = runner
                .cmd_tx
                .try_send(runner::RunnerCommand::WriterPause)
                .map_err(
                    |_| tracing::warn!(target: "orchestrator", "WriterPause command send failed"),
                );
        }

        let new_state = guard.state.as_public();
        self.emit(AppEvent::StateChanged { state: new_state });

        tracing::info!(target: "orchestrator", "recording paused");
        Ok(())
    }

    /// Resume after a pause.
    ///
    /// Instructs the runner to resume the `MeetingWriter` (inserts a
    /// granule-position gap), then resumes the capture stream.
    ///
    /// # Errors
    ///
    /// `AppError::InvalidInput` if not in `Paused` state.
    pub async fn resume(&self) -> AppResult<()> {
        let mut guard = self.inner.lock().await;
        let _meeting_id = transition_resume(&mut guard.state)?;

        // Instruct the runner to resume the writer before new samples arrive.
        if let Some(runner) = &guard.runner {
            let _ = runner
                .cmd_tx
                .try_send(runner::RunnerCommand::WriterResume)
                .map_err(
                    |_| tracing::warn!(target: "orchestrator", "WriterResume command send failed"),
                );
        }

        // Resume audio capture so samples start flowing again.
        if let Some(capture) = &mut guard.capture {
            capture.resume()?;
        }

        let new_state = guard.state.as_public();
        self.emit(AppEvent::StateChanged { state: new_state });

        tracing::info!(target: "orchestrator", "recording resumed");
        Ok(())
    }

    /// Stop the current recording and finalise the meeting.
    ///
    /// Stops audio capture, waits for the runner to flush and finalise the
    /// `MeetingWriter`, then returns the completed `MeetingMeta`.
    ///
    /// # Errors
    ///
    /// `AppError::InvalidInput` if not in `Recording` or `Paused` state.
    pub async fn stop(&self) -> AppResult<MeetingMeta> {
        // Extract runner + capture + timestamps while holding the lock, then
        // release it so the runner can complete without deadlocking.
        let (meeting_id, runner_handle, started_at, capture_opt) = {
            let mut guard = self.inner.lock().await;
            let meeting_id = transition_stop(&mut guard.state)?;

            let new_state = guard.state.as_public();
            self.emit(AppEvent::StateChanged { state: new_state });

            let runner = guard.runner.take();
            let started_at = guard.started_at.take().unwrap_or_else(Utc::now);
            let capture = guard.capture.take();
            (meeting_id, runner, started_at, capture)
        };

        // Stop audio capture (drops the cpal stream → signals forwarder to exit).
        if let Some(mut capture) = capture_opt {
            if let Err(e) = capture.stop() {
                tracing::warn!(
                    target: "orchestrator",
                    "capture.stop() error during recording stop: {e}"
                );
            }
        }

        let ended_at = Utc::now();
        let duration_ms = (ended_at - started_at).num_milliseconds().max(0) as u64;

        let meta = MeetingMeta {
            uuid: meeting_id,
            title: format!("Recording {}", started_at.format("%Y-%m-%dT%H:%M:%SZ")),
            started_at: started_at.to_rfc3339(),
            ended_at: Some(ended_at.to_rfc3339()),
            duration_ms,
            speaker_count: 0,
            audio_format: AudioFormat {
                codec: "opus".into(),
                sample_rate: 16_000,
                channels: 1,
                bitrate_kbps: Some(32),
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        // Send stop command to runner and await the finalised metadata.
        let finalised_meta = if let Some(handle) = runner_handle {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            handle
                .cmd_tx
                .send(runner::RunnerCommand::Stop {
                    meta: Box::new(meta.clone()),
                    reply: reply_tx,
                })
                .await
                .map_err(|_| AppError::Internal {
                    context: "runner command channel closed before stop could be sent".into(),
                })?;

            reply_rx.await.map_err(|_| AppError::Internal {
                context: "runner reply channel closed before finalise completed".into(),
            })??
        } else {
            // No runner (e.g. test path with no real audio device).
            meta
        };

        // Transition Stopping → Idle.
        {
            let mut guard = self.inner.lock().await;
            if let Err(e) = transition_idle(&mut guard.state) {
                tracing::warn!(target: "orchestrator", "unexpected state after stop: {e:?}");
            }
        }

        self.emit(AppEvent::StateChanged {
            state: RecordingState::Idle,
        });

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            "recording stopped and finalised"
        );

        Ok(finalised_meta)
    }

    /// Return a snapshot of the current recording state.
    pub async fn state(&self) -> RecordingState {
        self.inner.lock().await.state.as_public()
    }

    /// Enumerate available audio-input devices.
    ///
    /// Thin wrapper over `audio_capture::AudioCaptureManager::list_devices()`
    /// so that the IPC layer does not need a direct `audio-capture`
    /// dependency. Runs on `spawn_blocking` because cpal's device
    /// enumeration is FFI-bound (especially on Linux/PulseAudio cold-start).
    pub async fn list_devices(&self) -> AppResult<Vec<meeting_app_common::AudioDevice>> {
        tokio::task::spawn_blocking(audio_capture::AudioCaptureManager::list_devices)
            .await
            .map_err(|join_err| AppError::Internal {
                context: format!("list_devices spawn_blocking join: {join_err}"),
            })?
    }

    /// Subscribe to `AppEvent` broadcasts.
    ///
    /// Emitted events include `StateChanged` on every transition and
    /// `AudioMeter` frames at ~30 Hz while recording.
    ///
    /// If the receiver falls behind (> 256 events), tokio will skip frames and
    /// return `RecvError::Lagged`. Callers should handle this with a
    /// `tracing::warn!` and continue consuming.
    pub fn subscribe_events(&self) -> broadcast::Receiver<AppEvent> {
        let rx = self.event_tx.subscribe();
        tracing::debug!(
            target: "orchestrator",
            "new AppEvent subscriber (broadcast capacity 256)"
        );
        rx
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn emit(&self, event: AppEvent) {
        match self.event_tx.send(event) {
            Ok(n) => {
                tracing::trace!(target: "orchestrator", receivers = n, "AppEvent broadcast");
            }
            Err(_) => {
                tracing::trace!(target: "orchestrator", "AppEvent dropped (no subscribers)");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test-only constructor (bypasses real AudioCaptureManager)
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-source"))]
impl Orchestrator {
    /// Test-only constructor that accepts pre-built `AudioStreams` instead of
    /// opening a real capture device.
    ///
    /// The caller provides streams (e.g. from `DummyAudioSource::generate_streams`);
    /// the orchestrator opens a `MeetingWriter` and spawns the runner directly.
    ///
    /// This bypasses `AudioCaptureManager` entirely so tests don't need a
    /// real microphone or cpal.
    pub async fn start_with_streams(
        &self,
        streams: audio_capture::AudioStreams,
    ) -> AppResult<MeetingId> {
        let mut guard = self.inner.lock().await;
        let (meeting_id, started_at_ms) = transition_start(&mut guard.state)?;

        let started_at =
            DateTime::<Utc>::from_timestamp_millis(started_at_ms as i64).unwrap_or_else(Utc::now);
        guard.started_at = Some(started_at);

        let audio_format = AudioFormat {
            codec: "opus".into(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        };

        let writer = match MeetingWriter::open(&self.persistence_root, meeting_id, audio_format) {
            Ok(w) => w,
            Err(e) => {
                guard.state = InternalState::Idle;
                return Err(e);
            }
        };

        let runner_handle = runner::spawn_runner(
            streams,
            writer,
            self.event_tx.clone(),
            Arc::clone(&self.model_registry),
            meeting_id,
        );
        guard.runner = Some(runner_handle);
        // No AudioCaptureManager to store (guard.capture stays None).

        let new_state = guard.state.as_public();
        self.emit(AppEvent::StateChanged { state: new_state });

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            "recording started (test-source path)"
        );

        Ok(meeting_id)
    }

    /// Test-only constructor that accepts pre-built `AudioStreams` **and** a
    /// pre-built `AsrBackend` stub.
    ///
    /// This allows integration tests to inject a fake ASR backend (returning
    /// canned `Segment`s) without needing a 1 GB model file. The pipeline
    /// wiring (VAD → Accumulator → flush dispatch → worker) runs for real.
    ///
    /// Available only under the `test-source` feature.
    pub async fn start_with_streams_and_backend(
        &self,
        streams: audio_capture::AudioStreams,
        backend: Box<dyn meeting_app_common::AsrBackend + Send>,
    ) -> AppResult<MeetingId> {
        let mut guard = self.inner.lock().await;
        let (meeting_id, started_at_ms) = transition_start(&mut guard.state)?;

        let started_at =
            DateTime::<Utc>::from_timestamp_millis(started_at_ms as i64).unwrap_or_else(Utc::now);
        guard.started_at = Some(started_at);

        let audio_format = AudioFormat {
            codec: "opus".into(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        };

        let writer = match MeetingWriter::open(&self.persistence_root, meeting_id, audio_format) {
            Ok(w) => w,
            Err(e) => {
                guard.state = InternalState::Idle;
                return Err(e);
            }
        };

        let runner_handle = runner::spawn_runner_with_backend(
            streams,
            writer,
            self.event_tx.clone(),
            Arc::clone(&self.model_registry),
            meeting_id,
            backend,
        );
        guard.runner = Some(runner_handle);

        let new_state = guard.state.as_public();
        self.emit(AppEvent::StateChanged { state: new_state });

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            "recording started (test-source + stub backend path)"
        );

        Ok(meeting_id)
    }
}
