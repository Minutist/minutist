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
    AppError, AppEvent, AppResult, AudioFormat, Diarizer, MeetingId, MeetingListEntry, MeetingMeta,
    ModelDescriptor, ModelId, ModelStatus, RecordingState, Segment,
};
use model_registry::ModelRegistry;
use persistence::{MeetingIndex, MeetingWriter};
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
    /// Test-only override for the on-stop diarization pass: when `Some`, the
    /// next `stop()` whose `diarization_enabled` is true consumes this injected
    /// diarizer instead of building the production `SherpaDiarizer` via
    /// `runner::build_diarizer`. This lets a default-suite test drive the FULL
    /// toggle-ON gate branch of `stop()` model-free, mirroring how
    /// `rediarize_with_diarizer` injects a diarizer into the re-diarize seam.
    /// `None` in production (the field is only ever set by the test-only
    /// `inject_on_stop_diarizer`).
    #[cfg(any(test, feature = "test-source"))]
    on_stop_diarizer_override: Option<Box<dyn Diarizer + Send>>,
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
                #[cfg(any(test, feature = "test-source"))]
                on_stop_diarizer_override: None,
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

    /// Ensure a model is present and return its on-disk model **directory**.
    ///
    /// A thin wrapper over `ModelRegistry::ensure` (which downloads + verifies
    /// when absent and resolves to the per-model directory under
    /// `{app-data}/models/{kind}/{model-id}/`). `ipc-bridge`'s
    /// `summarise_meeting` calls this to resolve the selected LLM directory
    /// before locating the `.gguf` and opening the summariser — keeping the
    /// `model-registry` edge inside the orchestrator (there is **no**
    /// `orchestrator → summariser` edge; the summariser is loaded in
    /// `ipc-bridge`).
    pub async fn ensure_model_path(&self, model_id: &ModelId) -> AppResult<PathBuf> {
        self.model_registry.ensure(model_id).await
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

        // On-stop diarization pass (FR-11), gated on `settings.diarization_enabled`
        // (default OFF → `stop()` behaviour is unchanged). The meeting is fully
        // finalised on disk and the recorder is back to `Idle`, so this runs the
        // same diarization pass the user-triggered re-diarize uses, INLINE
        // (awaited) so the returned `MeetingMeta.speaker_count` is correct. The
        // `stop_recording` command upserts the index from the returned meta, so
        // no index handle is needed here.
        let mut finalised_meta = finalised_meta;
        if self.settings.current().diarization_enabled {
            match self.diarize_on_stop(meeting_id).await {
                Ok(speaker_count) => {
                    finalised_meta.speaker_count = speaker_count;
                    finalised_meta.diarizer = Some(diarizer_descriptor());
                }
                Err(e) => {
                    // Diarization is a best-effort post-pass; a failure must not
                    // turn a successful stop into an error (the recording is
                    // safely on disk). Surface it as a recoverable event.
                    tracing::warn!(
                        target: "orchestrator",
                        meeting_id = %meeting_id.0,
                        "on-stop diarization failed: {e}; recording is finalised without speaker labels"
                    );
                    self.emit(AppEvent::ErrorOccurred { error: e });
                }
            }
        }

        Ok(finalised_meta)
    }

    /// Run the on-stop diarization pass for `meeting_id` and return the distinct
    /// speaker count.
    ///
    /// Builds the bundled `SherpaDiarizer` off the async worker threads (or, in
    /// tests, consumes a `StubDiarizer` injected via `inject_on_stop_diarizer`),
    /// then hands the owned `Box<dyn Diarizer + Send>` to the shared
    /// [`Self::diarize_on_stop_inner`]. This build → dispatch split mirrors the
    /// [`Self::rediarize`] → [`Self::rediarize_inner`] seam so a default-suite
    /// test can drive the toggle-ON `stop()` branch model-free.
    async fn diarize_on_stop(&self, meeting_id: MeetingId) -> AppResult<u32> {
        // Test seam: a `StubDiarizer` injected via `inject_on_stop_diarizer`
        // takes precedence over building the production `SherpaDiarizer`, so a
        // default-suite test drives the full toggle-ON gate branch (build →
        // dispatch → finalise) with no sherpa model. In production this is
        // always `None`, so the build path below runs unchanged.
        #[cfg(any(test, feature = "test-source"))]
        {
            // Take the override in its own scope so the `inner` guard drops
            // before the multi-second blocking dispatch below — never hold the
            // central mutex across `diarize_on_stop_inner`.
            let injected = self.inner.lock().await.on_stop_diarizer_override.take();
            if let Some(diarizer) = injected {
                return self.diarize_on_stop_inner(meeting_id, diarizer).await;
            }
        }

        let registry = Arc::clone(&self.model_registry);
        let diarizer: Box<dyn Diarizer + Send> =
            tokio::task::spawn_blocking(move || -> AppResult<Box<dyn Diarizer + Send>> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| AppError::Internal {
                        context: format!("on-stop diarize runtime build failed: {e}"),
                    })?;
                let diarizer = rt.block_on(runner::build_diarizer(&registry))?;
                Ok(Box::new(diarizer))
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("on-stop diarizer-build join failed: {e}"),
            })??;

        self.diarize_on_stop_inner(meeting_id, diarizer).await
    }

    /// Shared on-stop diarization core: decode + diarize the meeting with the
    /// supplied owned `diarizer`, then persist the result via
    /// [`Self::finalise_diarization`] (rewriting `transcript.json`, updating
    /// `metadata.json`'s `{ speaker_count, diarizer }`, and emitting
    /// `AppEvent::DiarizationComplete`). Returns the distinct speaker count. No
    /// index `upsert` — the `stop_recording` command upserts from the returned
    /// `MeetingMeta`, whose `speaker_count` this pass has updated.
    ///
    /// Driven by [`Self::diarize_on_stop`] with the production `SherpaDiarizer`
    /// (or a test-injected `StubDiarizer`), exactly as
    /// [`Self::rediarize_inner`] is driven for the user-triggered re-diarize.
    async fn diarize_on_stop_inner(
        &self,
        meeting_id: MeetingId,
        diarizer: Box<dyn Diarizer + Send>,
    ) -> AppResult<u32> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        let (segments, speaker_count) =
            run_diarization_blocking(meeting_dir.clone(), diarizer).await?;

        self.finalise_diarization(meeting_id, &meeting_dir, &segments, speaker_count)
            .await?;

        Ok(speaker_count)
    }

    /// Return a snapshot of the current recording state.
    pub async fn state(&self) -> RecordingState {
        self.inner.lock().await.state.as_public()
    }

    /// Re-run transcription for a previously-recorded meeting offline.
    ///
    /// Decodes the meeting's `audio.opus` (pause-INCLUDING 16 kHz mono PCM via
    /// `persistence::reader::read_audio_pcm`) and runs it through the **same**
    /// batched-VAD accumulator + `AsrBackend` machinery the live pipeline uses
    /// (`runner::re_transcribe_buffer`), so the 30 s encoder window and the
    /// silence-preservation constraint hold identically. The refreshed
    /// transcript replaces `transcript.json`; `AppEvent::TranscriptSegment`
    /// events are emitted as segments are produced, and the supplied
    /// [`MeetingIndex`] row is refreshed (`upsert`) so the meeting-list excerpt
    /// reflects the new first segment. The heavy decode + inference runs on
    /// `spawn_blocking`.
    ///
    /// # Errors
    ///
    /// - `AppError::InvalidInput` if a recording is in progress (state is not
    ///   `Idle`) — re-transcribe is an offline operation and must not contend
    ///   with the live pipeline for the ASR model.
    /// - `AppError::ModelLoad` / `AppError::Inference` if the ASR model is not
    ///   available or fails to load (unlike the live path's best-effort skip, an
    ///   explicit user-triggered re-transcribe with no model is an error).
    pub async fn re_transcribe(&self, index: &MeetingIndex, meeting_id: MeetingId) -> AppResult<()> {
        self.ensure_idle_for_retranscribe().await?;

        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        // Decode audio + run VAD + ASR on a blocking thread. Build the ASR
        // backend inside the blocking closure so the heavy model load is off the
        // async worker threads. The model is resolved via the same registry path
        // the live pipeline uses.
        let registry = Arc::clone(&self.model_registry);
        let event_tx = self.event_tx.clone();
        let meeting_dir_for_blocking = meeting_dir.clone();

        let segments: Vec<Segment> = tokio::task::spawn_blocking(move || -> AppResult<Vec<Segment>> {
            // Decode pause-INCLUDING PCM.
            let pcm = persistence::read_audio_pcm(&meeting_dir_for_blocking)?;

            // Build the production AsrRuntime. `init_asr_runtime` is async; drive
            // it on a current-thread runtime inside this blocking context (the
            // same approach the ASR worker uses). Model resolution itself is the
            // only async step.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| AppError::Internal {
                    context: format!("re_transcribe runtime build failed: {e}"),
                })?;

            let mut runtime = match rt.block_on(runner::build_asr_runtime_for_retranscribe(&registry))? {
                Some(r) => r,
                None => {
                    return Err(AppError::ModelLoad {
                        model_id: "qwen3-asr-0.6b-q8_0".into(),
                        context: "ASR model not available; cannot re-transcribe".into(),
                    });
                }
            };

            runner::re_transcribe_buffer(&pcm, &mut runtime, &event_tx, meeting_id)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("re_transcribe spawn_blocking join failed: {e}"),
        })??;

        self.finalise_retranscribe(index, meeting_id, &meeting_dir, segments)
            .await
    }

    /// Refuse a re-transcribe unless the recorder is `Idle`.
    ///
    /// Shared by [`Self::re_transcribe`] and the test-only
    /// `re_transcribe_with_backend` so both honour the same offline-only
    /// invariant.
    async fn ensure_idle_for_retranscribe(&self) -> AppResult<()> {
        let guard = self.inner.lock().await;
        if !matches!(guard.state, InternalState::Idle) {
            return Err(AppError::InvalidInput {
                context: "re_transcribe requires the recorder to be Idle".into(),
            });
        }
        Ok(())
    }

    /// Rewrite `transcript.json` from the refreshed `segments` and refresh the
    /// supplied index row so the meeting-list excerpt reflects the new first
    /// segment.
    ///
    /// Shared by the production [`Self::re_transcribe`] and the test-only
    /// `re_transcribe_with_backend`: both produce a `Vec<Segment>` via the same
    /// `runner::re_transcribe_buffer` machinery, then persist + index it
    /// identically. The blocking `std::fs` writes run on `spawn_blocking`; the
    /// async index `upsert` is awaited (never `block_on`).
    async fn finalise_retranscribe(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
        segments: Vec<Segment>,
    ) -> AppResult<()> {
        // Rewrite transcript.json from the refreshed segments (blocking fs).
        let meeting_dir_for_write = meeting_dir.to_path_buf();
        let segments_for_write = segments.clone();
        tokio::task::spawn_blocking(move || -> AppResult<()> {
            persistence::write_transcript(&meeting_dir_for_write, &segments_for_write)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("re_transcribe transcript write join failed: {e}"),
        })??;

        // Refresh the index row so the list excerpt reflects the new transcript.
        let meeting_dir_for_meta = meeting_dir.to_path_buf();
        let entry: MeetingListEntry = tokio::task::spawn_blocking(move || -> AppResult<MeetingListEntry> {
            let meta = persistence::read_metadata(&meeting_dir_for_meta)?;
            let transcript = persistence::read_transcript(&meeting_dir_for_meta)?;
            Ok(MeetingListEntry {
                id: meta.uuid,
                title: meta.title,
                started_at: meta.started_at,
                duration_ms: meta.duration_ms,
                speaker_count: meta.speaker_count,
                excerpt: transcript.first().map(|s| s.text.clone()),
            })
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("re_transcribe metadata read join failed: {e}"),
        })??;

        index.upsert(&entry).await?;

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            segments = segments.len(),
            "re_transcribe completed"
        );

        Ok(())
    }

    /// Re-run speaker diarization for a previously-recorded meeting offline
    /// (FR-11 user action).
    ///
    /// Mirrors [`Self::re_transcribe`]'s offline one-shot idiom: it refuses
    /// unless the recorder is `Idle` (an offline diarization pass must not
    /// contend with the live pipeline), then on a `spawn_blocking` thread it
    /// decodes the meeting's `audio.opus` (pause-INCLUDING 16 kHz mono PCM via
    /// `persistence::reader::read_audio_pcm`), reads `transcript.json`
    /// (`persistence::read_transcript`), and runs the bundled `SherpaDiarizer`
    /// over the segment array. The diarizer overlays `speaker_id` onto the
    /// segments in place (`Diarizer::assign_speakers`, returning the distinct
    /// speaker count); the refreshed transcript replaces `transcript.json`
    /// (`persistence::write_transcript`), `metadata.json` is updated
    /// (`persistence::write_metadata`, setting `speaker_count` + the `diarizer`
    /// [`ModelDescriptor`]), the supplied [`MeetingIndex`] row's `speaker_count`
    /// is refreshed (`upsert`), and `AppEvent::DiarizationComplete` is emitted on
    /// the shared bus.
    ///
    /// The diarizer is built lazily inside the blocking closure (resolving both
    /// model directories via `model-registry` and opening sherpa over the two
    /// `.onnx` files), mirroring the re-transcribe lazy ASR-runtime pattern so
    /// the heavy model load is off the async worker threads.
    ///
    /// # Errors
    ///
    /// - `AppError::InvalidInput` if a recording is in progress (state is not
    ///   `Idle`).
    /// - `AppError::ModelLoad` / `AppError::ModelDownload` if a diarize model is
    ///   not available or fails to load.
    /// - `AppError::Inference` if sherpa diarization fails.
    pub async fn rediarize(&self, index: &MeetingIndex, meeting_id: MeetingId) -> AppResult<()> {
        self.ensure_idle_for_rediarize().await?;

        // Build the production diarizer off the async worker threads (the model
        // load is heavy). It is then handed to the shared inner path as an owned
        // `Box<dyn Diarizer>`, exactly like the test seam supplies a StubDiarizer.
        let registry = Arc::clone(&self.model_registry);
        let diarizer: Box<dyn Diarizer + Send> =
            tokio::task::spawn_blocking(move || -> AppResult<Box<dyn Diarizer + Send>> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| AppError::Internal {
                        context: format!("rediarize runtime build failed: {e}"),
                    })?;
                let diarizer = rt.block_on(runner::build_diarizer(&registry))?;
                Ok(Box::new(diarizer))
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("rediarize diarizer-build join failed: {e}"),
            })??;

        self.rediarize_inner(index, meeting_id, diarizer).await
    }

    /// Refuse a re-diarize unless the recorder is `Idle`.
    ///
    /// Shared by [`Self::rediarize`] and the test-only `rediarize_with_diarizer`
    /// so both honour the same offline-only invariant.
    async fn ensure_idle_for_rediarize(&self) -> AppResult<()> {
        let guard = self.inner.lock().await;
        if !matches!(guard.state, InternalState::Idle) {
            return Err(AppError::InvalidInput {
                context: "rediarize requires the recorder to be Idle".into(),
            });
        }
        Ok(())
    }

    /// Shared diarization-and-persist core for the user-triggered re-diarize.
    ///
    /// Driven by the production [`Self::rediarize`] (with the bundled
    /// `SherpaDiarizer`) and the test-only `rediarize_with_diarizer` (with a
    /// `StubDiarizer`). On a `spawn_blocking` thread it decodes the meeting's
    /// pause-INCLUDING PCM (`persistence::read_audio_pcm`), reads
    /// `transcript.json` (`persistence::read_transcript`), and runs the supplied
    /// diarizer's `assign_speakers` (which overlays `speaker_id` in place and
    /// returns the distinct speaker count). It then calls
    /// [`Self::finalise_diarization`] to rewrite `transcript.json` + update
    /// `metadata.json`, and finally refreshes the supplied index row's
    /// `speaker_count` (`upsert`).
    async fn rediarize_inner(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        diarizer: Box<dyn Diarizer + Send>,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        let (segments, speaker_count) =
            run_diarization_blocking(meeting_dir.clone(), diarizer).await?;

        self.finalise_diarization(meeting_id, &meeting_dir, &segments, speaker_count)
            .await?;

        // Refresh the index row's speaker_count (and keep the excerpt current).
        let meeting_dir_for_meta = meeting_dir.clone();
        let entry: MeetingListEntry =
            tokio::task::spawn_blocking(move || -> AppResult<MeetingListEntry> {
                let meta = persistence::read_metadata(&meeting_dir_for_meta)?;
                let transcript = persistence::read_transcript(&meeting_dir_for_meta)?;
                Ok(MeetingListEntry {
                    id: meta.uuid,
                    title: meta.title,
                    started_at: meta.started_at,
                    duration_ms: meta.duration_ms,
                    speaker_count: meta.speaker_count,
                    excerpt: transcript.first().map(|s| s.text.clone()),
                })
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("rediarize metadata read join failed: {e}"),
            })??;

        index.upsert(&entry).await?;

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            speaker_count,
            "rediarize completed"
        );

        Ok(())
    }

    /// Persist a diarization result and announce it (no index touch).
    ///
    /// Shared by [`Self::rediarize_inner`] and the on-stop pass. Rewrites
    /// `transcript.json` (with the overlaid `speaker_id`s) and updates
    /// `metadata.json`'s `{ speaker_count, diarizer }` via
    /// `persistence::write_metadata`, keeping `persistence` the sole writer under
    /// `meetings/{uuid}/` (the diarizer never touches disk), then emits
    /// `AppEvent::DiarizationComplete` on the shared `event_tx`. The blocking
    /// `std::fs` writes run on `spawn_blocking`.
    ///
    /// The user-triggered re-diarize layers an index `upsert` on top of this; the
    /// on-stop pass does not (the `stop_recording` command upserts from the
    /// returned `MeetingMeta`, whose `speaker_count` this pass has updated).
    async fn finalise_diarization(
        &self,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
        segments: &[Segment],
        speaker_count: u32,
    ) -> AppResult<()> {
        let meeting_dir_for_write = meeting_dir.to_path_buf();
        let segments_for_write = segments.to_vec();
        let descriptor = diarizer_descriptor();
        tokio::task::spawn_blocking(move || -> AppResult<()> {
            persistence::write_transcript(&meeting_dir_for_write, &segments_for_write)?;
            let mut meta = persistence::read_metadata(&meeting_dir_for_write)?;
            meta.speaker_count = speaker_count;
            meta.diarizer = Some(descriptor);
            persistence::write_metadata(&meeting_dir_for_write, &meta)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("diarization transcript/metadata write join failed: {e}"),
        })??;

        self.emit(AppEvent::DiarizationComplete {
            meeting_id,
            speaker_count,
        });

        Ok(())
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
// Diarization helpers (shared by the user-triggered re-diarize + on-stop pass)
// ---------------------------------------------------------------------------

/// Decode the meeting's PCM + transcript and run `diarizer` over the segments,
/// all on a `spawn_blocking` thread.
///
/// Returns the segments with `speaker_id` overlaid and the distinct speaker
/// count `assign_speakers` reported. The diarizer is consumed (moved into the
/// blocking closure) so a `SherpaDiarizer` or a test `StubDiarizer` both work.
async fn run_diarization_blocking(
    meeting_dir: PathBuf,
    diarizer: Box<dyn Diarizer + Send>,
) -> AppResult<(Vec<Segment>, u32)> {
    tokio::task::spawn_blocking(move || -> AppResult<(Vec<Segment>, u32)> {
        let pcm = persistence::read_audio_pcm(&meeting_dir)?;
        let mut segments = persistence::read_transcript(&meeting_dir)?;
        let speaker_count = diarizer.assign_speakers(&pcm, 16_000, &mut segments)?;
        Ok((segments, speaker_count))
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("diarization spawn_blocking join failed: {e}"),
    })?
}

/// The `ModelDescriptor` recorded in `metadata.json` after a diarization pass.
///
/// Identifies the bundled segmentation model (the diarizer is a two-model
/// pipeline; the segmentation model is the diarizer's primary identity for the
/// `MeetingMeta.diarizer` field). The `version` mirrors the manifest id.
fn diarizer_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        name: runner::DIARIZE_SEG_MODEL_ID.to_string(),
        quantisation: None,
        version: runner::DIARIZE_SEG_MODEL_ID.to_string(),
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

    /// Offline re-transcribe driven by a caller-supplied [`AsrBackend`] stub,
    /// mirroring [`Self::start_with_streams_and_backend`] for the offline path.
    ///
    /// This is the stub-injectable seam for the offline re-transcribe pipeline:
    /// it decodes the meeting's `audio.opus` to pause-INCLUDING PCM via
    /// `persistence::reader::read_audio_pcm` and drives the **same**
    /// `runner::re_transcribe_buffer` machinery the production
    /// [`Self::re_transcribe`] uses (real Silero VAD + the batched-VAD
    /// accumulator + `transcribe_one_flush`), but with the injected `backend`
    /// instead of a real `AsrRuntime`. It then rewrites `transcript.json` and
    /// refreshes the index row exactly as the production path does
    /// ([`Self::finalise_retranscribe`]).
    ///
    /// This lets a DEFAULT-suite test exercise the whole offline path —
    /// real VAD over a real-speech fixture, `transcript.json` rewrite,
    /// `AppEvent::TranscriptSegment` emission, and index-excerpt refresh —
    /// without a ~1 GB ASR model.
    ///
    /// Honours the same `Idle`-only invariant as the production path.
    ///
    /// Available only under the `test-source` feature.
    pub async fn re_transcribe_with_backend(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        backend: Box<dyn meeting_app_common::AsrBackend + Send>,
    ) -> AppResult<()> {
        self.ensure_idle_for_retranscribe().await?;

        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        let event_tx = self.event_tx.clone();
        let meeting_dir_for_blocking = meeting_dir.clone();

        let segments: Vec<Segment> = tokio::task::spawn_blocking(move || -> AppResult<Vec<Segment>> {
            let pcm = persistence::read_audio_pcm(&meeting_dir_for_blocking)?;
            let mut backend = backend;
            runner::re_transcribe_buffer(&pcm, backend.as_mut(), &event_tx, meeting_id)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("re_transcribe_with_backend spawn_blocking join failed: {e}"),
        })??;

        self.finalise_retranscribe(index, meeting_id, &meeting_dir, segments)
            .await
    }

    /// Offline re-diarize driven by a caller-supplied [`Diarizer`], mirroring
    /// [`Self::re_transcribe_with_backend`] for the diarization path.
    ///
    /// This is the stub-injectable seam for the re-diarize pipeline: it drives
    /// the **same** [`Self::rediarize_inner`] core the production
    /// [`Self::rediarize`] uses (decode PCM + transcript → `assign_speakers` →
    /// `transcript.json` rewrite + `metadata.json` `{ speaker_count, diarizer }`
    /// update → index `upsert` → `AppEvent::DiarizationComplete`), but with the
    /// injected `diarizer` instead of building a real `SherpaDiarizer`. This lets
    /// the DEFAULT test suite exercise the whole wiring with a `StubDiarizer`
    /// (NO model).
    ///
    /// Honours the same `Idle`-only invariant as the production path.
    ///
    /// Available only under the `test-source` feature.
    pub async fn rediarize_with_diarizer(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        diarizer: Box<dyn Diarizer + Send>,
    ) -> AppResult<()> {
        self.ensure_idle_for_rediarize().await?;
        self.rediarize_inner(index, meeting_id, diarizer).await
    }

    /// Inject a `diarizer` for the next on-stop diarization pass, so a real
    /// [`Self::stop`] call with `diarization_enabled = true` exercises the FULL
    /// toggle-ON gate branch (`settings.diarization_enabled == true → build →
    /// dispatch → finalise`) without building a real `SherpaDiarizer`.
    ///
    /// [`Self::diarize_on_stop`] consumes this override (via `take()`) in place
    /// of `runner::build_diarizer`; everything downstream — decode, diarize,
    /// `transcript.json` / `metadata.json` rewrite, and the
    /// `AppEvent::DiarizationComplete` emission — is the production path. The
    /// override is one-shot: once consumed, the next `stop()` builds the real
    /// diarizer again.
    ///
    /// This is the on-stop analogue of [`Self::rediarize_with_diarizer`].
    /// Available only under the `test-source` feature (or in `#[cfg(test)]`).
    pub async fn inject_on_stop_diarizer(&self, diarizer: Box<dyn Diarizer + Send>) {
        self.inner.lock().await.on_stop_diarizer_override = Some(diarizer);
    }

    /// Borrow the orchestrator's `SettingsHandle` so a test can flip a setting
    /// (e.g. `diarization_enabled`) on the same handle `stop()` reads.
    ///
    /// `test_orchestrator` builds the handle internally; this accessor lets the
    /// W3 on-stop test enable diarization without a parallel handle.
    /// Available only under the `test-source` feature (or in `#[cfg(test)]`).
    pub fn settings_handle_for_test(&self) -> &SettingsHandle {
        &self.settings
    }
}
