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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use audio_capture::AudioCaptureManager;
use chrono::{DateTime, Utc};
use diarizer::{OnlineDiarizer, SpeakerTurn};
use minutist_common::{
    AppError, AppEvent, AppResult, AsrBackend, AsrEngine, AudioFormat, MeetingId,
    MeetingListEntry, MeetingMeta, ModelDescriptor, ModelId, ModelStatus, RecordingState, Segment,
};
use model_registry::ModelRegistry;
use persistence::{MeetingIndex, MeetingWriter};
use settings::SettingsHandle;
use state::{
    transition_finalising, transition_idle, transition_offline_claim, transition_offline_release,
    transition_pause, transition_resume, transition_start, transition_stop, InternalState,
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
    /// Whether the most recent recording's live transcript was incomplete (the
    /// drop-oldest flush queue lost audio, or the stop-time drain timed out).
    /// `stop()` copies the runner's flag here; `ipc-bridge` reads it via
    /// `take_transcript_incomplete()` to decide whether to run a background
    /// re-transcribe of the complete `audio.opus`.
    last_transcript_incomplete: Arc<AtomicBool>,
    /// Process-held prewarmed ASR backend (live-test UX T2). [`Self::prewarm_asr`]
    /// resolves the routed engine and loads the backend on `spawn_blocking`,
    /// storing the `(engine, backend)` pair here so the FIRST `start()` hands it
    /// to the runner instead of paying the cold ~29 s model load at record time.
    /// `start()` takes it (leaving `None`) only when the cached engine matches the
    /// session's engine; a mismatch (the user changed the transcription language)
    /// or an empty cache falls back to the existing lazy worker-init path, so the
    /// lazy path is never regressed. A `std::sync::Mutex` because the held value
    /// (`Box<dyn AsrBackend + Send>`) is `Send` but not `Sync`, and every access
    /// is a brief, non-awaiting take/insert.
    #[allow(clippy::type_complexity)]
    prewarmed_asr: Arc<StdMutex<Option<(AsrEngine, Box<dyn AsrBackend + Send>)>>>,
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
            last_transcript_incomplete: Arc::new(AtomicBool::new(false)),
            prewarmed_asr: Arc::new(StdMutex::new(None)),
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

    /// Pre-load the routed ASR backend into a process-held cache so the FIRST
    /// `start()` does not pay the cold model load (~29 s for the Parakeet model)
    /// at record time (live-test UX T2).
    ///
    /// Resolves the engine from `settings.transcription_language` (+ the GPU-model
    /// opt-in) exactly as `start()` does, then — when the routed (or a fallback)
    /// model is locally `Available` — builds the backend on `spawn_blocking` (the
    /// heavy GGUF / sherpa load is synchronous) and stores the `(engine, backend)`
    /// pair. `start()` consumes it when its engine matches; otherwise it falls
    /// back to the existing lazy worker-init path, so the lazy path is never
    /// regressed.
    ///
    /// **Idempotent + non-blocking-at-start.** Calling it again for an
    /// already-cached engine is a no-op; if the model is not downloaded it warms
    /// nothing and returns `Ok(())` (no download, no block — the prewarm must
    /// never stall a fresh install). Wired from `app-main`'s `setup` after the
    /// event bus is up. A build failure is logged and swallowed (the lazy path
    /// remains the safety net), so prewarm can never fail a startup.
    /// VRAM-aware GPU plan for an ASR/model load, from the current settings.
    ///
    /// Computes ONE [`minutist_common::GpuPlan`] from the live VRAM probe +
    /// the user's `gpu_acceleration` mode. The large ASR tier is requested
    /// unconditionally — [`minutist_common::resolve_gpu_plan`] applies the VRAM
    /// clamp and downgrades to the small tier when the large one would not fit.
    /// Call it once per model-load decision and read `plan.asr_gpu` /
    /// `plan.effective_prefer_large`; probing twice in one decision would risk
    /// the two reads disagreeing. See `architecture/cross-cutting.md` — "GPU
    /// portability" and "ASR engine routing".
    fn gpu_plan(&self) -> minutist_common::GpuPlan {
        let s = self.settings.current();
        minutist_common::resolve_gpu_plan(
            minutist_common::probe_primary_gpu().as_ref(),
            s.gpu_acceleration,
            true, // always request the large tier; the VRAM clamp in resolve_gpu_plan decides
        )
    }

    pub async fn prewarm_asr(&self) {
        let plan = self.gpu_plan();
        let engine = minutist_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            plan.effective_prefer_large,
        );

        // Idempotent: skip if we already hold a backend for this engine.
        {
            let guard = self.prewarmed_asr.lock().expect("prewarm mutex poisoned");
            if guard.as_ref().map(|(e, _)| *e == engine).unwrap_or(false) {
                return;
            }
        }

        let n_gpu_layers = runner::resolve_gpu_layers(plan.asr_gpu);
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        let registry = Arc::clone(&self.model_registry);

        // Build off the async executor (the model load is heavy + synchronous).
        // Drive the async `init_asr_backend` on a current-thread runtime inside
        // the blocking closure, exactly as the re-transcribe path does.
        let built = tokio::task::spawn_blocking(move || -> Option<(AsrEngine, Box<dyn AsrBackend + Send>)> {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(target: "orchestrator", "prewarm runtime build failed: {e}");
                    return None;
                }
            };
            match rt.block_on(runner::build_asr_backend_for_retranscribe(
                &registry,
                engine,
                n_gpu_layers,
                language,
            )) {
                Ok(Some(backend)) => Some((engine, backend)),
                Ok(None) => {
                    tracing::info!(
                        target: "orchestrator",
                        "ASR prewarm skipped: routed model not downloaded yet"
                    );
                    None
                }
                Err(e) => {
                    tracing::warn!(target: "orchestrator", "ASR prewarm build failed: {e}");
                    None
                }
            }
        })
        .await;

        match built {
            Ok(Some(pair)) => {
                let mut guard = self.prewarmed_asr.lock().expect("prewarm mutex poisoned");
                *guard = Some(pair);
                tracing::info!(target: "orchestrator", "ASR backend prewarmed");
            }
            Ok(None) => {}
            Err(join_err) => {
                tracing::warn!(
                    target: "orchestrator",
                    "ASR prewarm join failed: {join_err}; the lazy path remains the fallback"
                );
            }
        }
    }

    /// Take the prewarmed backend if it matches `engine` (live-test UX T2).
    ///
    /// Returns the held backend (leaving the cache empty) only on an engine
    /// match; a mismatch leaves the cached backend in place for a later matching
    /// `start()` and returns `None` so the caller uses the lazy path.
    fn take_prewarmed_asr(&self, engine: AsrEngine) -> Option<Box<dyn AsrBackend + Send>> {
        let mut guard = self.prewarmed_asr.lock().expect("prewarm mutex poisoned");
        match guard.take() {
            Some((cached_engine, backend)) if cached_engine == engine => Some(backend),
            // Mismatch: put it back; a later start() for that engine can use it.
            Some(pair) => {
                *guard = Some(pair);
                None
            }
            None => None,
        }
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

        // When `capture_system_audio` is on, also capture + mix the system/call
        // (loopback) audio so all participants are transcribed; mic-only
        // otherwise. On non-Windows / loopback-open failure the capture layer
        // falls back to mic-only (logged) rather than failing the recording.
        let capture_system_audio = self.settings.current().capture_system_audio;
        let streams = match capture.start(32, 64, capture_system_audio) {
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

        // GPU offload is a VRAM-aware runtime decision: a single plan probes the
        // GPU once and decides per model (see `architecture/cross-cutting.md` —
        // "GPU portability"). `resolve_gpu_layers` maps `plan.asr_gpu` to the
        // compile-time ceiling (GPU) or `0` (force CPU).
        let plan = self.gpu_plan();
        let n_gpu_layers = runner::resolve_gpu_layers(plan.asr_gpu);

        // Resolve the ASR language hint from the `transcription_language`
        // setting (see `runner::resolve_transcription_language`): a full English
        // name forces that language via the assistant-turn prefix; the `"auto"`
        // sentinel resolves to `None` (auto-detect, byte-identical to the
        // pre-feature behaviour).
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);

        // Hybrid ASR (Phase 8): pick the engine from the transcription-language
        // setting — Parakeet for the languages it covers, else a Qwen tier. The
        // tier honours `plan.effective_prefer_large` (the requested large tier
        // only when it ALSO fits the VRAM budget), so a CPU-bound large request
        // downgrades to the 0.6B default. The `language` hint above only affects
        // the Qwen tiers. See `common::asr_engine_for_language`.
        let engine = minutist_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            plan.effective_prefer_large,
        );

        // Live diarization (Phase B): build the additive `OnlineDiarizer` BEFORE
        // spawning the runner, gated on the `diarization_enabled` setting AND the
        // embedding model being locally `Available` (no download, no block — see
        // `runner::build_online_diarizer`). The heavy `EmbeddingExtractor::new`
        // load runs on `spawn_blocking` so it never stalls the async runtime,
        // mirroring the on-stop diarizer build. Every failure mode degrades to
        // `None` → no live label; recording/transcription proceed identically.
        let online_diarizer =
            self.build_live_diarizer(self.settings.current().diarization_enabled).await;

        // Live-test UX T2: hand the runner the prewarmed backend if one was
        // warmed for this session's engine; else `None` → the worker lazy-inits
        // it on the first flush (the pre-existing path, never regressed).
        let prewarmed_backend = self.take_prewarmed_asr(engine);
        if prewarmed_backend.is_some() {
            tracing::info!(
                target: "orchestrator",
                "using prewarmed ASR backend for this recording"
            );
        }

        let runner_handle = runner::spawn_runner(
            streams,
            writer,
            self.event_tx.clone(),
            Arc::clone(&self.model_registry),
            meeting_id,
            n_gpu_layers,
            language,
            engine,
            online_diarizer,
            prewarmed_backend,
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
        // The cmd_tx sender is cloned while the lock is held (for the state
        // transition), then the lock is released before awaiting the send. The
        // runner thread never takes this mutex, so there is no deadlock risk,
        // but holding the lock across an `await` would block every other async
        // caller of the orchestrator for the duration of the send.
        let (cmd_tx, new_state) = {
            let mut guard = self.inner.lock().await;
            let _meeting_id = transition_pause(&mut guard.state)?;

            // Pause the audio capture stream first so no new samples arrive while
            // the writer is transitioning.
            if let Some(capture) = &mut guard.capture {
                capture.pause()?;
            }

            let cmd_tx = guard.runner.as_ref().map(|r| r.cmd_tx.clone());
            let new_state = guard.state.as_public();
            (cmd_tx, new_state)
        };

        // Send the WriterPause command with back-pressure: if the channel is
        // full (runner busy under load), this yields until space is available
        // rather than silently dropping the command. A dropped WriterPause would
        // desynchronise the encoder-pause silence vs the pause-excluding timeline.
        if let Some(tx) = cmd_tx {
            if let Err(e) = tx.send(runner::RunnerCommand::WriterPause).await {
                tracing::warn!(target: "orchestrator", "WriterPause send failed: {e}");
            }
        }

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
        let (cmd_tx, new_state) = {
            let mut guard = self.inner.lock().await;
            let _meeting_id = transition_resume(&mut guard.state)?;

            let cmd_tx = guard.runner.as_ref().map(|r| r.cmd_tx.clone());
            let new_state = guard.state.as_public();
            (cmd_tx, new_state)
        };

        // Send WriterResume before resuming the capture stream so the writer is
        // ready to accept samples before the capture callback pushes them. A
        // dropped WriterResume would strand the encoder in Paused, causing every
        // subsequent push_samples call to return Err (silently swallowed) and
        // the audio tail to be lost. Using send().await makes delivery reliable.
        if let Some(tx) = cmd_tx {
            if let Err(e) = tx.send(runner::RunnerCommand::WriterResume).await {
                tracing::warn!(target: "orchestrator", "WriterResume send failed: {e}");
            }
        }

        // Resume audio capture so samples start flowing again.
        {
            let mut guard = self.inner.lock().await;
            if let Some(capture) = &mut guard.capture {
                capture.resume()?;
            }
        }

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
            speaker_names: std::collections::BTreeMap::new(),
            notes_format: 0,
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

            // Capture has stopped and the Stop command is dispatched; the runner
            // now finalises (drain + writes) on its own thread. Mark the recorder
            // Finalising and broadcast it so the UI stays responsive while the
            // (possibly slow) drain runs — `Stopping` was momentary; only a NEW
            // recording is gated until finalise completes.
            {
                let mut guard = self.inner.lock().await;
                if let Err(e) = transition_finalising(&mut guard.state) {
                    tracing::warn!(
                        target: "orchestrator",
                        "unexpected state entering finalise: {e:?}"
                    );
                }
            }
            self.emit(AppEvent::StateChanged {
                state: RecordingState::Finalising { meeting_id },
            });
            // Live-test UX T4(c): the finalise drain (ASR-backlog drain + the
            // transcript/metadata/audio writes) is opaque, so emit an
            // INDETERMINATE progress event for it. The terminal
            // `AppEvent::MeetingFinalised` below clears the per-row indicator.
            self.emit(AppEvent::OperationProgress {
                meeting_id,
                op: minutist_common::OperationKind::Finalise,
                fraction: None,
                label: "Finalising…".to_string(),
            });

            let finalised = reply_rx.await.map_err(|_| AppError::Internal {
                context: "runner reply channel closed before finalise completed".into(),
            })??;

            // Record whether the live transcript fell behind (drop-oldest loss
            // or a stop-drain timeout). The runner sets this before sending the
            // reply, so it is visible now. `ipc-bridge` reads it via
            // `take_transcript_incomplete()` to trigger a background re-transcribe.
            self.last_transcript_incomplete.store(
                handle.transcript_incomplete.load(Ordering::Acquire),
                Ordering::Release,
            );

            finalised
        } else {
            // No runner (e.g. test path with no real audio device).
            self.last_transcript_incomplete
                .store(false, Ordering::Release);
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

        // The meeting is now fully on disk. Announce it so the webview refreshes
        // the meeting list (the row appears via `reconcile_orphans`/`upsert`);
        // distinct from the `Idle` transition, which also fires in other paths.
        self.emit(AppEvent::MeetingFinalised { meeting_id });

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            "recording stopped and finalised"
        );

        // Diarization is NOT run inline here. The on-stop diarization pass
        // (FR-11) is now a DECOUPLED background job: `stop()` returns the
        // finalised meeting un-diarized (`speaker_count` 0, `diarizer` None) the
        // instant the recording is on disk and the recorder is back to `Idle`,
        // so the `stop_recording` command can index it immediately — in-session
        // visibility no longer waits on diarization. When `diarization_enabled`
        // is set, `ipc-bridge` spawns `Orchestrator::rediarize` in the
        // background; that pass re-writes the transcript + metadata, refreshes
        // the index row, and emits `AppEvent::DiarizationComplete` when done. A
        // slow or hung diarization can therefore never wedge `stop()` or hide
        // the meeting (the original failure mode). See
        // `architecture/components.md`, orchestrator "on-stop diarization".
        Ok(finalised_meta)
    }

    /// Whether the on-stop diarization pass should run (the user's
    /// `diarization_enabled` setting). `ipc-bridge` reads this after `stop()` to
    /// decide whether to spawn the background [`Self::rediarize`] pass; the
    /// decision lives here so the orchestrator stays the single owner of the
    /// settings read, but the *execution* is decoupled from `stop()`.
    pub fn diarization_enabled(&self) -> bool {
        self.settings.current().diarization_enabled
    }

    /// Take (read + reset) whether the most recent `stop()`'s live transcript
    /// was incomplete — the drop-oldest flush queue lost audio during recording,
    /// or the stop-time ASR drain timed out. `ipc-bridge` calls this once after
    /// `stop()`; when true it runs a background re-transcribe of the complete
    /// `audio.opus` (the authoritative repair, since the audio is fully captured
    /// regardless of live-ASR speed). Read-and-reset so it is consumed once.
    pub fn take_transcript_incomplete(&self) -> bool {
        self.last_transcript_incomplete.swap(false, Ordering::AcqRel)
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
        // Atomically CLAIM the recorder (not just check Idle) so a concurrent
        // start / re_transcribe / rediarize is rejected and cannot clobber the
        // same `transcript.json` (TIMELINE-DRIFT #5). Released on every exit.
        self.claim_offline(meeting_id).await?;
        let result = self.re_transcribe_claimed(index, meeting_id).await;
        self.release_offline().await;
        result
    }

    /// The re-transcribe body, run while the offline claim is held. Split out so
    /// [`Self::re_transcribe`] can guarantee the claim is released on every exit
    /// path (success and error).
    async fn re_transcribe_claimed(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());
        let segments = self.re_transcribe_segments(meeting_id, &meeting_dir).await?;
        self.finalise_retranscribe(index, meeting_id, &meeting_dir, segments)
            .await
    }

    /// Decode `audio.opus` + run VAD + ASR and return the FRESH segment list,
    /// WITHOUT persisting or finalising it.
    ///
    /// This is the post-claim re-transcribe COMPUTE step, factored out so it is
    /// shared, with no re-claim, by both [`Self::re_transcribe_claimed`] (which
    /// finalises the result on its own) and [`Self::reprocess_claimed`] (which
    /// feeds the fresh transcript straight into the diarize/split step before a
    /// single finalise). The caller owns persistence + the offline claim; this
    /// method neither writes `transcript.json` nor touches the claim.
    async fn re_transcribe_segments(
        &self,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
    ) -> AppResult<Vec<Segment>> {
        // Decode audio + run VAD + ASR on a blocking thread. Build the ASR
        // backend inside the blocking closure so the heavy model load is off the
        // async worker threads. The model is resolved via the same registry path
        // the live pipeline uses.
        let registry = Arc::clone(&self.model_registry);
        let event_tx = self.event_tx.clone();
        let meeting_dir_for_blocking = meeting_dir.to_path_buf();
        // Resolve the VRAM-aware GPU plan before entering the blocking closure
        // (it cannot read `self.settings`). The offline re-transcribe honours the
        // same GPU policy + ASR tier as the live path.
        let plan = self.gpu_plan();
        let n_gpu_layers = runner::resolve_gpu_layers(plan.asr_gpu);
        // Resolve the ASR language hint before entering the blocking closure (it
        // cannot read `self.settings`). The offline re-transcribe honours the
        // same `transcription_language` setting as the live path.
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        // Hybrid ASR (Phase 8): same engine routing as the live path, so a
        // re-transcribe of an English/EU meeting uses Parakeet (timestamps) and
        // others use the resolved Qwen tier (`plan.effective_prefer_large`).
        let engine = minutist_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            plan.effective_prefer_large,
        );
        let missing_model_id = runner::engine_model_id(engine);

        // Bound the (uninterruptible) offline ASR run with a length-relative
        // timeout sized for ASR (slower than diarization), mirroring the
        // diarization timeout in `rediarize_inner`: a wedged or pathologically
        // slow re-transcribe must not hold the offline claim — and thereby block
        // the next recording — without bound. On timeout we return before any
        // transcript write; the abandoned `spawn_blocking` thread's result is
        // discarded (tokio cannot cancel it).
        let duration_dir = meeting_dir.to_path_buf();
        let duration_ms = tokio::task::spawn_blocking(move || {
            persistence::read_metadata(&duration_dir).map(|m| m.duration_ms)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("re_transcribe metadata read join failed: {e}"),
        })??;
        let budget = retranscribe_timeout(duration_ms);

        let segments: Vec<Segment> = match tokio::time::timeout(
            budget,
            tokio::task::spawn_blocking(move || -> AppResult<Vec<Segment>> {
            // Decode pause-INCLUDING PCM.
            let pcm = persistence::read_audio_pcm(&meeting_dir_for_blocking)?;

            // Build the production ASR backend for the routed engine.
            // `init_asr_backend` is async; drive it on a current-thread runtime
            // inside this blocking context (the same approach the ASR worker
            // uses). Model resolution itself is the only async step.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| AppError::Internal {
                    context: format!("re_transcribe runtime build failed: {e}"),
                })?;

            let mut runtime = match rt.block_on(runner::build_asr_backend_for_retranscribe(&registry, engine, n_gpu_layers, language))? {
                Some(r) => r,
                None => {
                    return Err(AppError::ModelLoad {
                        model_id: missing_model_id.into(),
                        context: "ASR model not available; cannot re-transcribe".into(),
                    });
                }
            };

            runner::re_transcribe_buffer(&pcm, runtime.as_mut(), &event_tx, meeting_id)
            }),
        )
        .await
        {
            Ok(joined) => joined.map_err(|e| AppError::Internal {
                context: format!("re_transcribe spawn_blocking join failed: {e}"),
            })??,
            Err(_elapsed) => {
                return Err(AppError::Inference {
                    backend: "asr".to_string(),
                    context: format!(
                        "re-transcribe exceeded its {} s budget (recording {duration_ms} ms); \
                         transcript left unchanged",
                        budget.as_secs(),
                    ),
                });
            }
        };

        Ok(segments)
    }

    /// Re-run transcription THEN speaker diarization for a previously-recorded
    /// meeting in ONE offline operation (#0015 phase 5).
    ///
    /// Merges [`Self::re_transcribe`] + [`Self::rediarize`] under a SINGLE
    /// `claim_offline`/`release_offline`: a concurrent `start` / re-transcribe /
    /// re-diarize is rejected for the WHOLE pass (no `Idle` window opens between
    /// the two sub-steps), so the fresh transcript can never be clobbered by a
    /// racing op.
    ///
    /// Internal order is **re-transcribe FIRST, then diarize/split/merge over the
    /// fresh transcript, then finalise ONCE** (with `finalise_diarization`
    /// semantics). Diarize-first would be a guaranteed lost-update: the
    /// re-transcribe finalise's `write_transcript` would clobber the just-written
    /// split.
    ///
    /// **Clears `speaker_names` on every call.** The merged op always diarizes, so
    /// `finalise_diarization` clears `MeetingMeta.speaker_names` (a re-diarize can
    /// re-letter speakers, invalidating the old label→name map) every time — even
    /// a text-only repair loses user-assigned names. This is the accepted product
    /// default (accept-and-warn, 2026-06-17): consistent with re-diarize's existing
    /// behaviour; the durable fix is embedding-anchored retention (#0003
    /// voiceprints). The merged tool carries the "RESETS speaker names" warning
    /// (WU4). See `architecture/cross-cutting.md` — "Offline reprocessing".
    ///
    /// Does NOT summarise (parity with today; `Summarise` stays a separate
    /// post-stop pass, gated by `recorder_is_live`).
    ///
    /// # Errors
    ///
    /// - `AppError::InvalidInput` if a recording is in progress / another offline
    ///   op holds the claim.
    /// - `AppError::ModelLoad` / `AppError::Inference` if the ASR or diarize model
    ///   is unavailable or fails (the re-transcribe model is required, as in
    ///   [`Self::re_transcribe`]; the re-ASR split backend degrades to keep-whole
    ///   when absent, as in [`Self::rediarize`]).
    pub async fn reprocess(&self, index: &MeetingIndex, meeting_id: MeetingId) -> AppResult<()> {
        // ONE claim for the whole serial pass (re-transcribe → diarize). Both
        // sub-steps share it: the standalone commands each take their own claim,
        // but `reprocess` claims once and drives their CLAIMED bodies so no `Idle`
        // window opens mid-pass (TIMELINE-DRIFT #5). Released on every exit.
        self.claim_offline(meeting_id).await?;
        let result = self.reprocess_claimed(index, meeting_id).await;
        self.release_offline().await;
        result
    }

    /// The reprocess body, run while the SINGLE offline claim is held. Split out
    /// so [`Self::reprocess`] can guarantee the claim is released on every exit
    /// path (success and error).
    ///
    /// Composes the re-transcribe + diarize CLAIMED bodies WITHOUT re-claiming:
    /// 1. [`Self::re_transcribe_segments`] — decode + VAD + ASR → the FRESH
    ///    transcript (no finalise here).
    /// 2. Persist the fresh transcript so the diarize funnel reads it from disk;
    ///    `run_diarization_blocking` re-reads `transcript.json`, so the fresh text
    ///    must land on disk before the diarize step (NOT the stale one).
    /// 3. The re-diarize path ([`Self::rediarize_inner`] over a
    ///    `DiarizationJob::Production`) diarizes/splits/merges over THAT fresh
    ///    transcript and finalises ONCE via [`Self::finalise_diarization`] (write
    ///    transcript + `speaker_count` + diarizer descriptor + `speaker_names`
    ///    clear), then refreshes the index row.
    async fn reprocess_claimed(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        // Timeout budget under the single claim: each sub-step keeps its own
        // `retranscribe_timeout(duration_ms)` watchdog (the ASR run inside
        // `re_transcribe_segments`, and the diarize+split run inside
        // `rediarize_inner` — already on `retranscribe_timeout` from WU2). They
        // run SERIALLY, so the budgets compose additively (≈2× the per-step ASR
        // budget) and neither blocking pass is cut off mid-run; no single
        // watchdog straddles both. A wedged sub-step still releases the claim
        // (each `spawn_blocking` is individually bounded).

        // (a) Re-transcribe FIRST: produce the fresh segment list (no finalise).
        let segments = self.re_transcribe_segments(meeting_id, &meeting_dir).await?;

        // (b) Persist the fresh transcript so the diarize funnel — which re-reads
        // `transcript.json` from disk in `run_diarization_blocking` — diarizes the
        // re-transcribed text, not the stale one. This intermediate write is the
        // ONLY transcript write before the diarize finalise; the diarize step
        // rewrites it once more with the speaker labels (so the re-transcribe is
        // not separately finalised — that would clobber the split).
        let meeting_dir_for_write = meeting_dir.clone();
        tokio::task::spawn_blocking(move || -> AppResult<()> {
            persistence::write_transcript(&meeting_dir_for_write, &segments)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("reprocess transcript write join failed: {e}"),
        })??;

        // (c) Diarize/split/merge over the fresh transcript, then finalise ONCE
        // (write transcript + speaker_count + diarizer + speaker_names.clear()).
        // Build the production diarizer + best-effort re-ASR split backend off the
        // async worker threads, exactly as `rediarize_claimed` does.
        let registry = Arc::clone(&self.model_registry);
        let diarizer =
            tokio::task::spawn_blocking(move || -> AppResult<diarizer::SherpaDiarizer> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| AppError::Internal {
                        context: format!("reprocess diarizer runtime build failed: {e}"),
                    })?;
                rt.block_on(runner::build_diarizer(&registry))
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("reprocess diarizer-build join failed: {e}"),
            })??;
        let backend = self.build_split_backend().await;

        self.rediarize_inner(
            index,
            meeting_id,
            DiarizationJob::Production { diarizer, backend },
        )
        .await
    }

    /// Atomically claim the recorder for an offline operation (re-transcribe /
    /// re-diarize), under the state lock (TIMELINE-DRIFT #5).
    ///
    /// Returns `AppError::InvalidInput` if the recorder is not `Idle` — i.e. a
    /// live recording is in progress OR another offline op already holds the
    /// claim. On success the state is `Offline { meeting_id }` until
    /// [`Self::release_offline`] is called. This is the single seam both the
    /// production offline ops and their test-only `*_with_*` variants go through,
    /// so two concurrent offline ops (or an offline op racing a `start`) can no
    /// longer both pass the gate and clobber the same `transcript.json`.
    async fn claim_offline(&self, meeting_id: MeetingId) -> AppResult<()> {
        let mut guard = self.inner.lock().await;
        transition_offline_claim(&mut guard.state, meeting_id)?;
        // No StateChanged broadcast: `Offline` reports the public `Idle` (the
        // recorder stays READY — a `start` preempts the pass), so claiming the
        // slot does not change the public state. The repair's progress surfaces
        // per-meeting via `OperationProgress`, not the transport state.
        Ok(())
    }

    /// Release an offline claim, returning the recorder to `Idle`.
    ///
    /// Called on every exit path of an offline op (success and error) so a
    /// failed op never wedges the recorder out of `Idle`. PREEMPTION-SAFE: if a
    /// new recording has preempted the slot (`start` took it while this op ran),
    /// the release is a no-op and the `Idle` broadcast is suppressed — otherwise
    /// the late release would clobber the live session's `Recording` state in
    /// both the internal state and the UI.
    async fn release_offline(&self) {
        let released = {
            let mut guard = self.inner.lock().await;
            transition_offline_release(&mut guard.state)
        };
        // Only broadcast Idle when we actually released the claim. If the slot
        // was preempted, the `start` path already broadcast `Recording`; emitting
        // `Idle` here would wrongly tell the UI the new recording had stopped.
        if released {
            self.emit(AppEvent::StateChanged {
                state: RecordingState::Idle,
            });
        }
    }

    /// True when a live recording session holds the recorder (`Recording`,
    /// `Paused`, or `Stopping`) — i.e. a new meeting has started, possibly
    /// preempting a post-stop repair pass.
    ///
    /// The post-stop chain uses this to skip its remaining best-effort passes
    /// once the user has started the next meeting (the re-transcribe / re-diarize
    /// passes self-skip because their offline claim now fails, but the
    /// auto-summarise pass does not take the claim, so it checks this gate
    /// explicitly to avoid contending with the new recording's GPU use).
    pub async fn recorder_is_live(&self) -> bool {
        matches!(
            self.inner.lock().await.state,
            InternalState::Recording { .. }
                | InternalState::Paused { .. }
                | InternalState::Stopping { .. }
        )
    }

    /// Rewrite `transcript.json` from the refreshed `segments`, carry forward
    /// any prior diarization via a time-overlap join so speaker names remain
    /// valid, and refresh the supplied index row so the meeting-list excerpt
    /// reflects the new first segment.
    ///
    /// Shared by the production [`Self::re_transcribe`] and the test-only
    /// `re_transcribe_with_backend`: both produce a `Vec<Segment>` via the same
    /// `runner::re_transcribe_buffer` machinery, then persist + index it
    /// identically. The blocking `std::fs` writes run on `spawn_blocking`; the
    /// async index `upsert` is awaited (never `block_on`).
    ///
    /// Before writing, the function reads the OLD `transcript.json` (if any) and
    /// calls [`diarizer::overlay_speakers_from_prior`] to assign each new segment
    /// the `speaker_id` of the prior segment that covers the majority of its time.
    /// New segments with no prior overlap keep `speaker_id = None`. Because the
    /// prior label strings ("A", "B", …) are carried verbatim, any user-set
    /// `speaker_names` in metadata remain keyed correctly. A meeting that was never
    /// diarized (all prior `speaker_id = None`) leaves the new transcript as `None`
    /// with no regression.
    async fn finalise_retranscribe(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
        segments: Vec<Segment>,
    ) -> AppResult<()> {
        // Read the existing transcript (prior diarization) and apply its labels
        // onto the new segments before writing. All done in one blocking task to
        // avoid two round-trips.
        let meeting_dir_for_write = meeting_dir.to_path_buf();
        let (segments_written, speaker_count) = tokio::task::spawn_blocking(move || -> AppResult<(Vec<Segment>, u32)> {
            // Build the prior (start_ms, end_ms, speaker_id) triples from the
            // existing transcript.json, if present. Missing or unreadable transcript
            // is treated as an empty prior (first-ever transcription).
            let prior_triples: Vec<(u64, u64, Option<String>)> = persistence::read_transcript(&meeting_dir_for_write)
                .unwrap_or_default()
                .into_iter()
                .map(|s| (s.start_ms, s.end_ms, s.speaker_id))
                .collect();

            let mut new_segments = segments;
            diarizer::overlay_speakers_from_prior(&mut new_segments, &prior_triples);

            // Count the distinct non-None labels now present in the new transcript.
            let mut seen_labels: Vec<String> = Vec::new();
            for seg in &new_segments {
                if let Some(ref label) = seg.speaker_id {
                    if !seen_labels.contains(label) {
                        seen_labels.push(label.clone());
                    }
                }
            }
            let speaker_count = seen_labels.len() as u32;

            persistence::write_transcript(&meeting_dir_for_write, &new_segments)?;
            Ok((new_segments, speaker_count))
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("re_transcribe transcript write join failed: {e}"),
        })??;

        // Update metadata.speaker_count to reflect the distinct labels in the new
        // transcript (may differ from before if ASR boundary changes moved some
        // segments off their prior speaker).
        let meeting_dir_for_meta = meeting_dir.to_path_buf();
        let entry: MeetingListEntry = tokio::task::spawn_blocking(move || -> AppResult<MeetingListEntry> {
            let mut meta = persistence::read_metadata(&meeting_dir_for_meta)?;
            meta.speaker_count = speaker_count;
            persistence::write_metadata(&meeting_dir_for_meta, &meta)?;
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

        // Announce the refreshed transcript so the webview re-reads it (the
        // meeting-list excerpt + any open-meeting view), mirroring how
        // `finalise_diarization` emits `DiarizationComplete`. Without this, a
        // background re-transcribe with diarization OFF (the default) would leave
        // the UI showing the stale/truncated transcript until a manual refresh.
        self.emit(AppEvent::TranscriptReady { meeting_id });

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            segments = segments_written.len(),
            "re_transcribe completed"
        );

        Ok(())
    }

    /// Re-run ASR over a bounded audio window and return what was actually said
    /// there (Phase 9 — backs the `agent-tools` `relisten_section` tool).
    ///
    /// This is a **read-only** compute op: it never rewrites `transcript.json`
    /// and never takes the offline claim, so it is safe to run during a live
    /// recording (at a transient second-ASR-model memory cost — the backend is
    /// built fresh and dropped after the call). The heavy decode + inference run
    /// on `spawn_blocking`.
    ///
    /// `start_ms`/`end_ms` are **transcript timestamps** (the pause-EXCLUDING
    /// `Segment` clock — the only timeline an agent reading a transcript has).
    /// The mapping onto the pause-INCLUDING decoded PCM is documented in
    /// [`pcm_window_for_excluding_range`].
    ///
    /// Resolution of the ASR backend stays inside the orchestrator (it owns the
    /// `model-registry` edge); `agent-tools` never reaches `model-registry`.
    ///
    /// # Errors
    ///
    /// - `AppError::InvalidInput` if `end_ms <= start_ms`.
    /// - `AppError::ModelLoad` if the routed ASR model is not available.
    /// - `AppError::Inference` if the slice transcription fails.
    pub async fn transcribe_pcm_window(
        &self,
        meeting_id: MeetingId,
        start_ms: u64,
        end_ms: u64,
        language: Option<String>,
    ) -> AppResult<Vec<Segment>> {
        if end_ms <= start_ms {
            return Err(AppError::InvalidInput {
                context: format!("relisten window end_ms ({end_ms}) must exceed start_ms ({start_ms})"),
            });
        }

        // Re-listen is defined only over FINALISED audio. Reject the meeting that
        // is currently being recorded/finalised (W2): its `audio.opus` is still
        // being appended, so a full-file decode can hit a truncated OGG page and
        // the window may fall past what has been flushed.
        let active = match self.state().await {
            RecordingState::Recording { meeting_id, .. }
            | RecordingState::Paused { meeting_id, .. }
            | RecordingState::Stopping { meeting_id }
            | RecordingState::Finalising { meeting_id } => Some(meeting_id),
            RecordingState::Idle => None,
        };
        if active == Some(meeting_id) {
            return Err(AppError::InvalidInput {
                context: "cannot re-listen to a meeting that is still recording or finalising; \
                          finish it first"
                    .into(),
            });
        }

        // Resolve GPU plan + engine + language hint before entering the
        // blocking closure (it cannot read `self.settings`), mirroring
        // `re_transcribe_claimed`. An explicit caller-supplied `language`
        // overrides the setting-derived hint (the agent may force a re-listen in
        // a known language); the setting-derived engine routing is unchanged.
        let plan = self.gpu_plan();
        let n_gpu_layers = runner::resolve_gpu_layers(plan.asr_gpu);
        let setting_language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        let effective_language = language.or(setting_language);
        let engine = minutist_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            plan.effective_prefer_large,
        );
        let missing_model_id = runner::engine_model_id(engine);
        let registry = Arc::clone(&self.model_registry);

        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        // Bound the (uninterruptible) windowed ASR run with a length-relative
        // timeout sized for the requested span (S2): unlike `re_transcribe`, a
        // re-listen does NOT hold the offline claim, but an MCP/bridged caller
        // could still pin a blocking-pool thread + the second ASR model's memory
        // indefinitely on a wedged or pathologically slow decode. On timeout we
        // return before any result; the abandoned `spawn_blocking` thread's
        // result is discarded (tokio cannot cancel it).
        let budget = relisten_timeout(end_ms.saturating_sub(start_ms));

        match tokio::time::timeout(
            budget,
            tokio::task::spawn_blocking(move || -> AppResult<Vec<Segment>> {
                let pcm = persistence::read_audio_pcm(&meeting_dir)?;

                // Build the production ASR backend for the routed engine on a
                // current-thread runtime (model resolution is the only async step),
                // exactly as `re_transcribe_claimed` does.
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| AppError::Internal {
                        context: format!("transcribe_pcm_window runtime build failed: {e}"),
                    })?;
                let mut backend = match rt.block_on(runner::build_asr_backend_for_retranscribe(
                    &registry,
                    engine,
                    n_gpu_layers,
                    effective_language,
                ))? {
                    Some(b) => b,
                    None => {
                        return Err(AppError::ModelLoad {
                            model_id: missing_model_id.into(),
                            context: "ASR model not available; cannot re-listen to the section"
                                .into(),
                        });
                    }
                };

                transcribe_pcm_window_blocking(&pcm, backend.as_mut(), start_ms, end_ms)
            }),
        )
        .await
        {
            Ok(joined) => joined.map_err(|e| AppError::Internal {
                context: format!("transcribe_pcm_window spawn_blocking join failed: {e}"),
            })?,
            Err(_elapsed) => Err(AppError::Inference {
                backend: "asr".to_string(),
                context: format!(
                    "re-listen exceeded its {} s budget (window [{start_ms}, {end_ms}) ms)",
                    budget.as_secs(),
                ),
            }),
        }
    }

    /// Re-run speaker diarization for a previously-recorded meeting offline
    /// (FR-11 user action).
    ///
    /// Mirrors [`Self::re_transcribe`]'s offline one-shot idiom: it refuses
    /// unless the recorder is `Idle` (an offline diarization pass must not
    /// contend with the live pipeline), then on a `spawn_blocking` thread it
    /// decodes the meeting's `audio.opus` (pause-INCLUDING 16 kHz mono PCM via
    /// `persistence::reader::read_audio_pcm`), reads `transcript.json`
    /// (`persistence::read_transcript`), runs the bundled `SherpaDiarizer`'s
    /// `compute_turns` over the audio, overlays first-seen speaker labels, and
    /// (#0015 phase 4) re-ASRs each kept mixed Qwen segment into single-speaker
    /// sub-clips via the best-effort routed Qwen backend. The result is the
    /// speaker-labelled, possibly split segment list plus the distinct speaker
    /// count; the refreshed transcript replaces `transcript.json`
    /// (`persistence::write_transcript`), `metadata.json` is updated
    /// (`persistence::write_metadata`, setting `speaker_count` + the `diarizer`
    /// [`ModelDescriptor`]), the supplied [`MeetingIndex`] row's `speaker_count`
    /// is refreshed (`upsert`), and `AppEvent::DiarizationComplete` is emitted on
    /// the shared bus.
    ///
    /// The diarizer + re-ASR backend are built lazily off the async worker
    /// threads (resolving model directories via `model-registry`), mirroring the
    /// re-transcribe lazy ASR-runtime pattern so the heavy model loads do not
    /// stall the runtime. An absent re-ASR model degrades the split to
    /// keep-whole-and-flag (no regression).
    ///
    /// # Errors
    ///
    /// - `AppError::InvalidInput` if a recording is in progress (state is not
    ///   `Idle`).
    /// - `AppError::ModelLoad` / `AppError::ModelDownload` if a diarize model is
    ///   not available or fails to load.
    /// - `AppError::Inference` if sherpa diarization fails.
    pub async fn rediarize(&self, index: &MeetingIndex, meeting_id: MeetingId) -> AppResult<()> {
        // Atomic claim/release (TIMELINE-DRIFT #5): rediarize also rewrites
        // `transcript.json`, so it must claim the slot just like re_transcribe.
        self.claim_offline(meeting_id).await?;
        let result = self.rediarize_claimed(index, meeting_id).await;
        self.release_offline().await;
        result
    }

    /// The re-diarize body, run while the offline claim is held (so the claim is
    /// released on every exit path).
    async fn rediarize_claimed(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
    ) -> AppResult<()> {
        // Build the production diarizer off the async worker threads (the model
        // load is heavy). It is handed to the shared inner path inside a
        // `DiarizationJob::Production`, alongside the best-effort routed Qwen
        // re-ASR backend used to split kept mixed Qwen segments (#0015 phase 4).
        let registry = Arc::clone(&self.model_registry);
        let diarizer =
            tokio::task::spawn_blocking(move || -> AppResult<diarizer::SherpaDiarizer> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| AppError::Internal {
                        context: format!("rediarize runtime build failed: {e}"),
                    })?;
                rt.block_on(runner::build_diarizer(&registry))
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("rediarize diarizer-build join failed: {e}"),
            })??;

        // Build the routed Qwen re-ASR backend best-effort (honours `gpu_plan` +
        // the same engine routing + language hint as `re_transcribe`). An absent
        // model yields `None` → the split degrades to keep-whole-and-flag (no
        // regression vs. the pre-split behaviour). The VRAM cost is bounded: the
        // backend is dropped at the end of the split loop (Qwen GGUF co-resident
        // with the sherpa diarizer models).
        let backend = self.build_split_backend().await;

        self.rediarize_inner(
            index,
            meeting_id,
            DiarizationJob::Production { diarizer, backend },
        )
        .await
    }

    /// Build the routed re-ASR backend for the #0015-phase-4 split, best-effort.
    ///
    /// Resolves the same GPU plan + ASR engine + language hint as the live and
    /// re-transcribe paths (so a split re-ASRs with the same tier the meeting was
    /// transcribed with) and builds the backend off the async worker threads.
    /// Returns `None` on ANY failure (model absent, build error, join failure) so
    /// a split simply degrades to keep-whole — a missing re-ASR model must never
    /// fail the whole diarization pass.
    async fn build_split_backend(&self) -> Option<Box<dyn AsrBackend + Send>> {
        let plan = self.gpu_plan();
        let n_gpu_layers = runner::resolve_gpu_layers(plan.asr_gpu);
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        let engine = minutist_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            plan.effective_prefer_large,
        );
        let registry = Arc::clone(&self.model_registry);

        let built = tokio::task::spawn_blocking(
            move || -> Option<Box<dyn AsrBackend + Send>> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .ok()?;
                match rt.block_on(runner::build_asr_backend_for_retranscribe(
                    &registry,
                    engine,
                    n_gpu_layers,
                    language,
                )) {
                    Ok(opt) => opt,
                    Err(e) => {
                        tracing::warn!(
                            target: "orchestrator",
                            "split re-ASR backend build failed ({e}); keeping mixed Qwen segments whole"
                        );
                        None
                    }
                }
            },
        )
        .await;

        match built {
            Ok(opt) => opt,
            Err(join_err) => {
                tracing::warn!(
                    target: "orchestrator",
                    "split re-ASR backend build join failed ({join_err}); keeping mixed Qwen segments whole"
                );
                None
            }
        }
    }

    /// Shared diarization-and-persist core for the user-triggered re-diarize.
    ///
    /// Driven by the production [`Self::rediarize`] (a
    /// `DiarizationJob::Production` with the bundled `SherpaDiarizer` + Qwen
    /// backend) and the test-only `rediarize_with_split_inputs` (a
    /// `DiarizationJob::Stub` with caller-supplied turns + backend). On a
    /// `spawn_blocking` thread it decodes the meeting's
    /// pause-INCLUDING PCM (`persistence::read_audio_pcm`), reads
    /// `transcript.json` (`persistence::read_transcript`), and resolves the
    /// turns + re-ASR backend from `job` (production `SherpaDiarizer` +
    /// best-effort Qwen backend, or stub-supplied turns + backend) to drive the
    /// [`diarize_split_merge`] core — which overlays speaker labels, splits each
    /// kept mixed Qwen segment into single-speaker sub-clips by re-ASR (#0015
    /// phase 4), and returns the (possibly longer) segment list plus the distinct
    /// speaker count. It then calls
    /// [`Self::finalise_diarization`] to rewrite `transcript.json` + update
    /// `metadata.json`, and finally refreshes the supplied index row's
    /// `speaker_count` (`upsert`).
    async fn rediarize_inner(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        job: DiarizationJob,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        // Read the recording length to size the timeout (a small metadata read on
        // a blocking thread).
        let duration_dir = meeting_dir.clone();
        let duration_ms = tokio::task::spawn_blocking(move || {
            persistence::read_metadata(&duration_dir).map(|m| m.duration_ms)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("diarize metadata read join failed: {e}"),
        })??;
        // #0015 phase 4: the pass now budgets for the sherpa compute AND the N
        // `transcribe_chunk` re-ASR passes of the split, so a split-heavy meeting
        // is not cut off mid-split. Sized like `retranscribe_timeout` (ASR is the
        // slow part once the split runs) rather than the diarize-only budget.
        let budget = retranscribe_timeout(duration_ms);

        // Live-test UX T4(c): the sherpa diarization compute is one opaque FFI
        // call with no progress callback, so emit a single INDETERMINATE
        // (`fraction = None`) progress event before it runs. The terminal
        // `AppEvent::DiarizationComplete` (emitted by `finalise_diarization`)
        // clears the per-row indicator. Labelled with the user-facing
        // "Identifying speakers…" wording (T5 — internal name unchanged).
        self.emit(AppEvent::OperationProgress {
            meeting_id,
            op: minutist_common::OperationKind::Rediarize,
            fraction: None,
            label: "Identifying speakers…".to_string(),
        });

        // Bound the (uninterruptible) sherpa `compute` + re-ASR split: a
        // pathologically slow or hung pass on a long recording must not block
        // forever (the original on-stop hang). On timeout we return BEFORE
        // `finalise_diarization`, so nothing is written — the meeting is left
        // un-diarized and the abandoned blocking thread's result (if it ever
        // completes) is dropped. `tokio` cannot cancel a `spawn_blocking` thread,
        // so a true infinite hang leaks one thread until process exit; the budget
        // bounds the wait, not the thread.
        let (segments, speaker_count) = match tokio::time::timeout(
            budget,
            run_diarization_blocking(meeting_dir.clone(), job, self.event_tx.clone(), meeting_id),
        )
        .await
        {
            Ok(result) => result?,
            Err(_elapsed) => {
                return Err(AppError::Inference {
                    backend: "diarizer".to_string(),
                    context: format!(
                        "diarization exceeded its {} s budget (recording {duration_ms} ms); \
                         left un-diarized",
                        budget.as_secs(),
                    ),
                });
            }
        };

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
            // Phase 9 (§4.4): a (re-)diarization pass can re-letter speakers, so
            // any user-set `speaker_names` keyed on the OLD letters is now
            // potentially wrong. Clear it in this same metadata write (no second
            // write) so the map can never silently mis-label a re-lettered
            // speaker. The chat tool's description states this; an MCP client
            // cannot re-map the way the UI could, so clearing is the only safe
            // cross-consumer behaviour. See `cross-cutting.md` "Agent chat loop".
            meta.speaker_names.clear();
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
    pub async fn list_devices(&self) -> AppResult<Vec<minutist_common::AudioDevice>> {
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

    /// Build the live [`OnlineDiarizer`] for a record session (Phase B),
    /// best-effort.
    ///
    /// Returns `None` immediately (no `spawn_blocking`) when `enabled` is false
    /// — the `diarization_enabled` setting gates entry, so the heavy model load
    /// only runs when a real load is warranted. When enabled, the
    /// local-only/no-download `runner::build_online_diarizer` resolver +
    /// `OnlineDiarizer::open` run inside `spawn_blocking` (the `EmbeddingExtractor`
    /// load is heavy) so the async runtime is never stalled at record start,
    /// mirroring the on-stop diarizer build. Any failure (model absent, locate
    /// fail, open fail, join fail) degrades to `None` → no live label; recording
    /// proceeds identically.
    async fn build_live_diarizer(&self, enabled: bool) -> Option<Arc<OnlineDiarizer>> {
        if !enabled {
            tracing::debug!(
                target: "orchestrator",
                "live diarization disabled (diarization_enabled = false); skipping"
            );
            return None;
        }

        let registry = Arc::clone(&self.model_registry);
        match tokio::task::spawn_blocking(move || runner::build_online_diarizer(&registry)).await {
            Ok(opt) => opt,
            Err(join_err) => {
                tracing::warn!(
                    target: "orchestrator",
                    "live diarizer build join failed: {join_err}; skipping (recording unaffected)"
                );
                None
            }
        }
    }

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

/// #0015 phase 1 merge threshold: a same-speaker inter-segment gap up to this
/// many ms is rejoined into one segment. Kept strictly below the live
/// accumulator's `MAX_GAP_MS` (3 s) and far below `PAUSE_MIN_MS` (4 s) so a merge
/// never bridges a region the timeline treats as a pause; it comfortably covers
/// the 720 ms VAD hangover and the zero-gap 10 s force-split.
const MERGE_GAP_MS: u64 = 1500;

/// How a diarization+split pass obtains its turns + re-ASR backend (#0015 phase
/// 4). Both variants converge on [`diarize_split_merge`] (the model-free core);
/// the variant only decides where the turns + backend come from.
///
/// `Production` carries the bundled `SherpaDiarizer` (its `compute_turns` runs on
/// the decoded PCM, on the blocking thread) + the best-effort routed Qwen
/// backend (`None` when the model is absent → degrade to keep-whole). `Stub`
/// supplies the turns + backend + config directly, the seam the default suite
/// uses to exercise the split with no `SherpaDiarizer` and no Qwen GGUF.
enum DiarizationJob {
    Production {
        diarizer: diarizer::SherpaDiarizer,
        backend: Option<Box<dyn minutist_common::AsrBackend + Send>>,
    },
    #[cfg(any(test, feature = "test-source"))]
    Stub {
        turns: Vec<SpeakerTurn>,
        backend: Option<Box<dyn minutist_common::AsrBackend + Send>>,
        config: diarizer::DiarizerConfig,
    },
}

/// Decode the meeting's PCM + transcript, resolve the turns + config + backend
/// from `job`, and run the [`diarize_split_merge`] core, all on a
/// `spawn_blocking` thread.
///
/// Returns the (possibly split) segments with `speaker_id` overlaid and the
/// distinct speaker count. The `job` carries either the production
/// `SherpaDiarizer` (+ best-effort Qwen backend) or stub-supplied turns + backend
/// (the default-suite seam), so a `SherpaDiarizer` and a model-free stub both
/// drive the SAME split core.
async fn run_diarization_blocking(
    meeting_dir: PathBuf,
    job: DiarizationJob,
    event_tx: broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) -> AppResult<(Vec<Segment>, u32)> {
    tokio::task::spawn_blocking(move || -> AppResult<(Vec<Segment>, u32)> {
        let pcm = persistence::read_audio_pcm(&meeting_dir)?;
        let segments = persistence::read_transcript(&meeting_dir)?;

        let (turns, config, backend): (
            Vec<SpeakerTurn>,
            diarizer::DiarizerConfig,
            Option<Box<dyn minutist_common::AsrBackend + Send>>,
        ) = match job {
            DiarizationJob::Production { diarizer, backend } => {
                // `compute_turns` runs over the pause-INCLUDING PCM → turn ms are
                // on the INCLUDING clock the split funnel maps onto.
                let turns = diarizer.compute_turns(&pcm, 16_000)?;
                (turns, diarizer.config().clone(), backend)
            }
            #[cfg(any(test, feature = "test-source"))]
            DiarizationJob::Stub {
                turns,
                backend,
                config,
            } => (turns, config, backend),
        };

        // `Box<dyn AsrBackend + Send>` → `Box<dyn AsrBackend>` for the core (the
        // split runs on this one thread; the `Send` bound is only needed to move
        // the backend into the closure).
        let backend = backend.map(|b| b as Box<dyn minutist_common::AsrBackend>);
        diarize_split_merge(
            &turns,
            segments,
            &pcm,
            backend,
            &config,
            &event_tx,
            meeting_id,
        )
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("diarization spawn_blocking join failed: {e}"),
    })?
}

/// Distinct-label count over the segments (`speaker_id` first-seen order).
///
/// The merge + split preserve labels, but recomputing from the final list is
/// robust and self-documenting — never trust an upstream count after the list
/// has been transformed.
fn distinct_label_count(segments: &[Segment]) -> u32 {
    let mut seen: Vec<&str> = Vec::new();
    for seg in segments {
        if let Some(label) = seg.speaker_id.as_deref() {
            if !seen.contains(&label) {
                seen.push(label);
            }
        }
    }
    seen.len() as u32
}

/// Dominant cluster of `turns` over `[start_ms, end_ms)`: the cluster id with the
/// greatest total temporal overlap, lower id breaking a tie (matching
/// `diarizer::overlay_speakers`' tie orientation). `None` when no turn overlaps.
///
/// Used to letter a re-ASR'd sub-clip via the WU1 cluster→letter map. The
/// orchestrator computes this from the public [`SpeakerTurn`] fields rather than
/// reaching into the diarizer's private overlap helper.
fn dominant_cluster(turns: &[SpeakerTurn], start_ms: u64, end_ms: u64) -> Option<i32> {
    if end_ms <= start_ms {
        return None;
    }
    // Per-cluster overlap totals, then argmax (greatest overlap; lower id on a tie).
    let mut totals: Vec<(i32, u64)> = Vec::new();
    for t in turns {
        let lo = start_ms.max(t.start_ms);
        let hi = end_ms.min(t.end_ms);
        let overlap = hi.saturating_sub(lo);
        if overlap == 0 {
            continue;
        }
        match totals.iter_mut().find(|(id, _)| *id == t.cluster) {
            Some((_, sum)) => *sum += overlap,
            None => totals.push((t.cluster, overlap)),
        }
    }
    totals
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(id, _)| id)
}

/// Re-ASR split core (#0015 phase 4) — the BLOCKING, model-free-testable funnel.
///
/// A free fn taking EXPLICIT params (mirroring [`transcribe_pcm_window_blocking`]
/// rather than dispatching through the `common::Diarizer` trait) so the default
/// suite can drive the whole split with a stub-supplied `turns` + stub `AsrBackend` —
/// no `SherpaDiarizer`, no Qwen GGUF:
/// - `turns` are the raw [`SpeakerTurn`]s from `compute_turns`, on the
///   pause-INCLUDING clock the `pcm` shares.
/// - `segments` is the ASR transcript (pause-EXCLUDING `start_ms`).
/// - `pcm` is the pause-INCLUDING decoded audio.
/// - `backend` is the routed Qwen re-ASR backend, or `None` (model absent /
///   degrade to keep-whole — no regression vs. the pre-split behaviour).
/// - `config` is the `DiarizerConfig` the overlay + flag use.
///
/// Steps:
/// 1. [`diarizer::overlay_speakers`] labels segments + flags mixed Qwen segments
///    (keep-whole, `shared_speakers` set, empty `words`) + returns the
///    cluster→letter map.
/// 2. [`diarizer::merge_adjacent_speakers`] collapses VAD/force-split fragments.
/// 3. For each KEPT mixed Qwen segment (non-empty `shared_speakers` AND empty
///    `words`) with a `backend`: take [`diarizer::turn_boundaries_within`] cuts
///    on the SAME pause-INCLUDING clock (mapped via
///    [`runner::pcm_window_for_excluding_range`]), energy-snap each cut, slice the
///    PCM, re-ASR each single-speaker sub-clip, letter it from the map by its
///    dominant [`SpeakerTurn`] cluster, and stamp its `start_ms` on the EXCLUDING
///    clock via [`runner::excluding_ms_for_pcm_sample`]. Keep-whole if the cuts
///    are empty, any snap returns `None`, or `backend` is `None`.
/// 4. Re-run [`diarizer::merge_adjacent_speakers`] (the split may have produced
///    adjacent same-letter sub-clips across segments) and recompute the count.
///
/// The clock discipline is the #1 blocking fix: turn cuts are taken on the
/// pause-INCLUDING clock the turns + PCM share, and a sub-clip's `start_ms` is
/// mapped back to the EXCLUDING transcript clock by the inverse. INCLUDING-clock
/// turns are NEVER compared against EXCLUDING-clock segment bounds.
fn diarize_split_merge(
    turns: &[SpeakerTurn],
    segments: Vec<Segment>,
    pcm: &[f32],
    mut backend: Option<Box<dyn minutist_common::AsrBackend>>,
    config: &diarizer::DiarizerConfig,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) -> AppResult<(Vec<Segment>, u32)> {
    // 1. Overlay labels + flag mixed Qwen segments; keep the cluster→letter map.
    let (mut segments, _count, cluster_letters) = diarizer::overlay_speakers(turns, segments, config);

    // 2. Collapse fragments so a turn reads as one row (#0015 phase 1).
    diarizer::merge_adjacent_speakers(&mut segments, MERGE_GAP_MS);

    // 3. Re-ASR split, only when a backend is present.
    if backend.is_some() {
        let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
        for seg in segments.into_iter() {
            // A kept mixed Qwen segment: flagged by `overlay_speakers` with
            // non-empty `shared_speakers` and no per-word timestamps. Everything
            // else passes through unchanged.
            let is_kept_mixed_qwen = !seg.shared_speakers.is_empty() && seg.words.is_empty();
            if !is_kept_mixed_qwen {
                out.push(seg);
                continue;
            }

            match split_mixed_qwen_segment(
                &seg,
                turns,
                pcm,
                backend.as_deref_mut().expect("backend present in this branch"),
                &cluster_letters,
                event_tx,
                meeting_id,
            )? {
                Some(sub_segments) => out.extend(sub_segments),
                // Keep-whole: empty cuts, a snap with no clear minimum, or a
                // re-ASR that produced nothing — leave the overlay's dominant
                // label + `shared_speakers` flag intact.
                None => out.push(seg),
            }
        }
        segments = out;
    }

    // 4. Re-merge (the split can yield adjacent same-letter sub-clips that should
    // read as one row) and recompute the distinct-label count.
    diarizer::merge_adjacent_speakers(&mut segments, MERGE_GAP_MS);
    let count = distinct_label_count(&segments);

    // Drop the re-ASR backend promptly: the Qwen GGUF is co-resident with the
    // sherpa diarizer models, so free its VRAM as soon as the split loop is done
    // (it lives no longer than this fn).
    drop(backend);

    Ok((segments, count))
}

/// Split one kept mixed Qwen segment into single-speaker sub-segments by
/// re-ASR'ing each speaker turn's audio (#0015 phase 4), or `None` to keep-whole.
///
/// Returns `None` (caller keeps the segment whole) when:
/// - the segment maps to no pause-INCLUDING PCM range, or
/// - [`diarizer::turn_boundaries_within`] yields no interior cut, or
/// - any cut's [`runner::snap_to_energy_min`] finds no clear minimum (continuous
///   / overlapping speech), or
/// - the resulting sub-clips re-ASR to nothing.
///
/// Otherwise returns one sub-segment per single-speaker slice, lettered from
/// `cluster_letters` by its dominant [`SpeakerTurn`] cluster, with empty
/// `shared_speakers` (no longer mixed) and `start_ms` on the EXCLUDING clock.
fn split_mixed_qwen_segment(
    seg: &Segment,
    turns: &[SpeakerTurn],
    pcm: &[f32],
    backend: &mut dyn minutist_common::AsrBackend,
    cluster_letters: &[(i32, String)],
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) -> AppResult<Option<Vec<Segment>>> {
    // Map the segment's pause-EXCLUDING [start_ms, end_ms) to the single
    // pause-INCLUDING PCM range the turns share (the clamp matches the offline
    // pause model — a mixed Qwen segment never straddles a ≥4 s pause).
    let seg_range = match runner::pcm_window_for_excluding_range(pcm, seg.start_ms, seg.end_ms) {
        Some(r) => r,
        None => return Ok(None),
    };

    // Interior speaker-change cuts on the SAME pause-INCLUDING clock the turns +
    // PCM share. `turn_boundaries_within` takes a synthetic segment whose bounds
    // are on the INCLUDING clock (the PCM range's ms), NEVER the excluding bounds.
    let incl_start_ms = (seg_range.start as u64 * 1000) / 16_000;
    let incl_end_ms = (seg_range.end as u64 * 1000) / 16_000;
    let incl_seg = Segment {
        start_ms: incl_start_ms,
        end_ms: incl_end_ms,
        ..seg.clone()
    };
    let cut_ms = diarizer::turn_boundaries_within(&incl_seg, turns);
    if cut_ms.is_empty() {
        return Ok(None);
    }

    // Convert each interior cut (INCLUDING ms) to a PCM sample, then energy-snap
    // it. Any snap with no clear minimum abandons the whole split (keep-whole).
    let mut cut_samples: Vec<usize> = Vec::with_capacity(cut_ms.len());
    for ms in &cut_ms {
        let sample = (*ms as usize * 16_000) / 1000;
        match runner::snap_to_energy_min(pcm, sample, SNAP_SEARCH_WINDOW_MS) {
            Some(snapped) => cut_samples.push(snapped),
            None => return Ok(None),
        }
    }
    cut_samples.sort_unstable();
    cut_samples.dedup();

    // Slice boundaries inside the segment's PCM range: [seg_start, c0, c1, …, seg_end].
    let mut bounds: Vec<usize> = Vec::with_capacity(cut_samples.len() + 2);
    bounds.push(seg_range.start);
    for c in &cut_samples {
        // A snapped cut can land just outside the segment range; clamp + skip a
        // degenerate slice.
        let c = (*c).clamp(seg_range.start, seg_range.end);
        if c > *bounds.last().unwrap() && c < seg_range.end {
            bounds.push(c);
        }
    }
    bounds.push(seg_range.end);
    if bounds.len() < 3 {
        // No usable interior cut survived the clamp — keep-whole.
        return Ok(None);
    }

    let mut sub_segments: Vec<Segment> = Vec::with_capacity(bounds.len() - 1);
    for pair in bounds.windows(2) {
        let (lo, hi) = (pair[0], pair[1]);
        if hi <= lo {
            continue;
        }
        // Stamp each sub-clip's start on the EXCLUDING transcript clock (the
        // inverse map); the chunk's own clock is the INCLUDING ms so the backend's
        // word offsets stay self-consistent within the clip.
        let excl_start_ms = runner::excluding_ms_for_pcm_sample(pcm, lo);
        let chunk_incl_start_ms = (lo as u64 * 1000) / 16_000;
        let chunk_incl_end_ms = (hi as u64 * 1000) / 16_000;
        let chunk = minutist_common::AudioChunk {
            samples: pcm[lo..hi].to_vec(),
            sample_rate: 16_000,
            start_ms: chunk_incl_start_ms,
            end_ms: chunk_incl_end_ms,
        };
        let re_asr = backend.transcribe_chunk(&chunk)?;

        // Letter this sub-clip by its dominant turn cluster via the WU1 map, so it
        // lands in the EXISTING scheme (no rename). A cluster the overlay pruned
        // away has no map entry → leave `None`.
        let cluster = dominant_cluster(turns, chunk_incl_start_ms, chunk_incl_end_ms);
        let letter = cluster.and_then(|c| {
            cluster_letters
                .iter()
                .find(|(id, _)| *id == c)
                .map(|(_, l)| l.clone())
        });

        let text: String = re_asr
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let sub = Segment {
            start_ms: excl_start_ms,
            // The sub-clip's excluding end is the next slice's excluding start;
            // compute it from `hi` directly (the inverse clamps a trailing edge).
            end_ms: runner::excluding_ms_for_pcm_sample(pcm, hi),
            text,
            speaker_id: letter,
            confidence: seg.confidence,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        };
        let _ = event_tx.send(AppEvent::TranscriptSegment {
            meeting_id,
            segment: sub.clone(),
        });
        sub_segments.push(sub);
    }

    if sub_segments.is_empty() {
        return Ok(None);
    }
    Ok(Some(sub_segments))
}

/// `± window_ms` energy-snap search span for a speaker-change cut (#0015 phase 4).
const SNAP_SEARCH_WINDOW_MS: u64 = 150;

/// Slice the decoded `pcm` to the requested pause-EXCLUDING window, run the
/// `backend` over the slice, and re-map the chunk-relative timestamps back onto
/// the meeting timeline (Phase 9 — the body of
/// [`Orchestrator::transcribe_pcm_window`], factored out so the test seam can
/// drive it with a stub backend).
///
/// The pause-clock mapping (and its clamp-at-pause decision) lives in
/// [`runner::pcm_window_for_excluding_range`]. The backend stamps each returned
/// segment at the start of the chunk it is handed (`AudioChunk::start_ms`, with
/// word offsets absolutized from it), so we set the chunk's `start_ms` to the
/// requested `start_ms` and the returned segments land on the meeting timeline
/// directly.
fn transcribe_pcm_window_blocking(
    pcm: &[f32],
    backend: &mut dyn minutist_common::AsrBackend,
    start_ms: u64,
    end_ms: u64,
) -> AppResult<Vec<Segment>> {
    let range = runner::pcm_window_for_excluding_range(pcm, start_ms, end_ms).ok_or_else(|| {
        AppError::InvalidInput {
            context: format!(
                "relisten window [{start_ms}, {end_ms}) ms is outside the recorded audio"
            ),
        }
    })?;

    // The clamp may shorten the window (a window straddling a pause stops at the
    // kept-region boundary); the chunk's `end_ms` reflects the actual slice.
    let slice_len_ms = (range.len() as u64 * 1000) / 16_000;
    let chunk = minutist_common::AudioChunk {
        samples: pcm[range].to_vec(),
        sample_rate: 16_000,
        start_ms,
        end_ms: start_ms + slice_len_ms,
    };

    backend.transcribe_chunk(&chunk)
}

/// Length-relative timeout budget for a diarize-only pass (no re-ASR split).
///
/// The offline sherpa `compute` is a single uninterruptible FFI call with no
/// progress callback, so a true per-progress watchdog isn't available at that
/// boundary; instead we bound it by wall-clock relative to the recording
/// length: ≈1× real-time, floored at `FLOOR_SECS` (so short meetings still get
/// a sane minimum) and capped at `CAP_SECS`. A normal diarization runs well
/// under real-time, so this only fires on a pathologically slow or wedged pass.
///
/// The production re-diarize pass now also re-ASRs split sub-clips (#0015 phase
/// 4), so it budgets with [`retranscribe_timeout`] (ASR is the slow part) rather
/// than this diarize-only curve. This curve is retained as the documented
/// diarize-compute baseline its `timeout_helpers_clamp_to_documented_bounds`
/// test guards.
#[cfg_attr(not(test), allow(dead_code))]
fn diarize_timeout(recording_duration_ms: u64) -> Duration {
    const FLOOR_SECS: u64 = 120; // 2 min
    const CAP_SECS: u64 = 600; // 10 min
    let secs = (recording_duration_ms / 1000).clamp(FLOOR_SECS, CAP_SECS);
    Duration::from_secs(secs)
}

/// Length-relative timeout budget for an offline re-transcribe.
///
/// Like [`diarize_timeout`] but sized for ASR, which re-runs the model over the
/// full audio and is slower than diarization: ~3x real-time, floored at 5 min,
/// capped at 30 min. Deliberately generous so a legitimate long re-transcribe is
/// not cut short; the bound exists so a wedged ASR run cannot hold the offline
/// claim (and block the next recording) forever.
fn retranscribe_timeout(recording_duration_ms: u64) -> Duration {
    const FLOOR_SECS: u64 = 300; // 5 min
    const CAP_SECS: u64 = 1800; // 30 min
    let secs = (recording_duration_ms / 1000 * 3).clamp(FLOOR_SECS, CAP_SECS);
    Duration::from_secs(secs)
}

/// Length-relative timeout budget for a single `relisten_section` window (S2).
///
/// `transcribe_pcm_window` re-runs ASR over a BOUNDED span, not the whole
/// recording, so its budget is relative to the requested window length
/// (`end_ms - start_ms`) rather than the recording duration. Like
/// [`retranscribe_timeout`] it allows ~3x real-time but with a much smaller
/// floor/cap (a re-listen is meant to be a quick spot-check, and — unlike a
/// re-transcribe — it does NOT hold the offline claim, but an MCP/bridged caller
/// could still pin a `spawn_blocking` thread + the second ASR model's memory
/// indefinitely on a wedged decode). On timeout the call returns
/// `AppError::Inference` cleanly; the abandoned `spawn_blocking` thread's result
/// is discarded (tokio cannot cancel it).
fn relisten_timeout(window_ms: u64) -> Duration {
    const FLOOR_SECS: u64 = 60; // 1 min (covers model build + a short window)
    const CAP_SECS: u64 = 300; // 5 min
    let secs = (window_ms / 1000 * 3).clamp(FLOOR_SECS, CAP_SECS);
    Duration::from_secs(secs)
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

        let plan = self.gpu_plan();
        let n_gpu_layers = runner::resolve_gpu_layers(plan.asr_gpu);
        // Resolve the ASR language hint, exactly as the production `start()`
        // path (this test-source path is production-equivalent).
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        // Hybrid ASR (Phase 8): same engine routing as the production `start()`.
        let engine = minutist_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            plan.effective_prefer_large,
        );
        // Phase B: build the live diarizer (gated on diarization_enabled +
        // local model availability), exactly as the production `start()` path.
        let online_diarizer =
            self.build_live_diarizer(self.settings.current().diarization_enabled).await;
        // The test-source path uses the lazy worker-init path; prewarm is a
        // production-startup concern, so no prewarmed backend is threaded here.
        let runner_handle = runner::spawn_runner(
            streams,
            writer,
            self.event_tx.clone(),
            Arc::clone(&self.model_registry),
            meeting_id,
            n_gpu_layers,
            language,
            engine,
            online_diarizer,
            None,
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
    /// `online_diarizer` lets a test drive the Phase-B live-labelling path with a
    /// real `OnlineDiarizer` (env-gated positive case) or `None` (the always-on
    /// regression guard proving transcription is unchanged when live diarization
    /// is off).
    ///
    /// Available only under the `test-source` feature.
    pub async fn start_with_streams_and_backend(
        &self,
        streams: audio_capture::AudioStreams,
        backend: Box<dyn minutist_common::AsrBackend + Send>,
        online_diarizer: Option<Arc<OnlineDiarizer>>,
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
            online_diarizer,
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
        backend: Box<dyn minutist_common::AsrBackend + Send>,
    ) -> AppResult<()> {
        // Same atomic claim/release as the production path (TIMELINE-DRIFT #5).
        self.claim_offline(meeting_id).await?;
        let result = self
            .re_transcribe_with_backend_claimed(index, meeting_id, backend)
            .await;
        self.release_offline().await;
        result
    }

    /// The `re_transcribe_with_backend` body, run while the offline claim is
    /// held (so the claim is released on every exit path).
    async fn re_transcribe_with_backend_claimed(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        backend: Box<dyn minutist_common::AsrBackend + Send>,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());
        let segments = self
            .re_transcribe_segments_with_backend(meeting_id, &meeting_dir, backend)
            .await?;
        self.finalise_retranscribe(index, meeting_id, &meeting_dir, segments)
            .await
    }

    /// Stub-backend mirror of [`Self::re_transcribe_segments`]: decode + VAD +
    /// ASR (via the injected `backend`) → the FRESH segment list, WITHOUT
    /// persisting or finalising. Shared by [`Self::re_transcribe_with_backend`]
    /// and the [`Self::reprocess_with_inputs`] test seam so the model-free
    /// reprocess test drives the SAME re-transcribe → diarize composition the
    /// production [`Self::reprocess`] does.
    async fn re_transcribe_segments_with_backend(
        &self,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
        backend: Box<dyn minutist_common::AsrBackend + Send>,
    ) -> AppResult<Vec<Segment>> {
        let event_tx = self.event_tx.clone();
        let meeting_dir_for_blocking = meeting_dir.to_path_buf();

        tokio::task::spawn_blocking(move || -> AppResult<Vec<Segment>> {
            let pcm = persistence::read_audio_pcm(&meeting_dir_for_blocking)?;
            let mut backend = backend;
            runner::re_transcribe_buffer(&pcm, backend.as_mut(), &event_tx, meeting_id)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("re_transcribe_with_backend spawn_blocking join failed: {e}"),
        })?
    }

    /// Offline re-diarize driven by caller-supplied turns + re-ASR backend,
    /// mirroring [`Self::re_transcribe_with_backend`] for the diarization path.
    ///
    /// This is the stub-injectable seam for the re-diarize + #0015-phase-4 split
    /// pipeline: it drives the **same** [`Self::rediarize_inner`] →
    /// [`diarize_split_merge`] core the production [`Self::rediarize`] uses
    /// (decode PCM + transcript → `overlay_speakers` → merge → re-ASR split →
    /// re-merge → `transcript.json` rewrite + `metadata.json` `{ speaker_count,
    /// diarizer }` update → index `upsert` → `AppEvent::DiarizationComplete`), but
    /// with caller-supplied `turns` + `config` instead of a real
    /// `SherpaDiarizer::compute_turns`, and a caller-supplied `backend`
    /// (`Some(stub)` to exercise the split, `None` to assert keep-whole). This
    /// lets the DEFAULT test suite cover the whole split with NO sherpa model and
    /// NO Qwen GGUF.
    ///
    /// Honours the same `Idle`-only invariant as the production path.
    ///
    /// Available only under the `test-source` feature.
    pub async fn rediarize_with_split_inputs(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        turns: Vec<SpeakerTurn>,
        backend: Option<Box<dyn minutist_common::AsrBackend + Send>>,
        config: diarizer::DiarizerConfig,
    ) -> AppResult<()> {
        // Same atomic claim/release as the production path (TIMELINE-DRIFT #5).
        self.claim_offline(meeting_id).await?;
        let result = self
            .rediarize_inner(
                index,
                meeting_id,
                DiarizationJob::Stub {
                    turns,
                    backend,
                    config,
                },
            )
            .await;
        self.release_offline().await;
        result
    }

    /// Model-free reprocess driven by stub inputs (#0015 phase 5).
    ///
    /// The stub-injectable seam for [`Self::reprocess`]: it takes ONE
    /// `claim_offline`/`release_offline` and drives the SAME re-transcribe →
    /// diarize/split/merge → finalise-once composition [`Self::reprocess_claimed`]
    /// uses, but with stub inputs instead of real models:
    /// - `asr_backend` re-transcribes (via `runner::re_transcribe_buffer`, real
    ///   VAD + accumulator) → the FRESH segment list, which is persisted so the
    ///   diarize step reads it;
    /// - `turns` + `split_backend` + `config` drive the diarize/split via the
    ///   `DiarizationJob::Stub` path, finalising ONCE
    ///   ([`Self::finalise_diarization`] — transcript + `speaker_count` + diarizer
    ///   + `speaker_names` clear) and refreshing the index row.
    ///
    /// No `Idle` window opens between the two sub-steps (a concurrent op is
    /// rejected until the single release), exactly as production. This is the seam
    /// the order + one-claim + names-cleared tests use — no real ASR model, no
    /// sherpa model, no Qwen GGUF.
    ///
    /// Available only under the `test-source` feature.
    pub async fn reprocess_with_inputs(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        asr_backend: Box<dyn minutist_common::AsrBackend + Send>,
        turns: Vec<SpeakerTurn>,
        split_backend: Option<Box<dyn minutist_common::AsrBackend + Send>>,
        config: diarizer::DiarizerConfig,
    ) -> AppResult<()> {
        // ONE claim for the whole serial pass, exactly as production `reprocess`.
        self.claim_offline(meeting_id).await?;
        let result = self
            .reprocess_with_inputs_claimed(index, meeting_id, asr_backend, turns, split_backend, config)
            .await;
        self.release_offline().await;
        result
    }

    /// The `reprocess_with_inputs` body, run while the SINGLE offline claim is
    /// held — the model-free mirror of [`Self::reprocess_claimed`].
    async fn reprocess_with_inputs_claimed(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        asr_backend: Box<dyn minutist_common::AsrBackend + Send>,
        turns: Vec<SpeakerTurn>,
        split_backend: Option<Box<dyn minutist_common::AsrBackend + Send>>,
        config: diarizer::DiarizerConfig,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        // (a) Re-transcribe FIRST via the stub backend (no finalise).
        let segments = self
            .re_transcribe_segments_with_backend(meeting_id, &meeting_dir, asr_backend)
            .await?;

        // (b) Persist the fresh transcript so the diarize funnel reads it.
        let meeting_dir_for_write = meeting_dir.clone();
        tokio::task::spawn_blocking(move || -> AppResult<()> {
            persistence::write_transcript(&meeting_dir_for_write, &segments)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("reprocess_with_inputs transcript write join failed: {e}"),
        })??;

        // (c) Diarize/split/merge over the fresh transcript, finalise ONCE.
        self.rediarize_inner(
            index,
            meeting_id,
            DiarizationJob::Stub {
                turns,
                backend: split_backend,
                config,
            },
        )
        .await
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

    /// `transcribe_pcm_window` driven by a caller-supplied [`AsrBackend`] stub,
    /// mirroring [`Self::re_transcribe_with_backend`] for the relisten path.
    ///
    /// Decodes the meeting's `audio.opus` to pause-INCLUDING PCM, applies the
    /// SAME pause-clock window mapping ([`runner::pcm_window_for_excluding_range`])
    /// the production path uses, and runs the injected `backend` over the slice —
    /// without a ~1 GB ASR model. Read-only (no claim, no transcript rewrite),
    /// exactly like the production method.
    ///
    /// Available only under the `test-source` feature.
    pub async fn transcribe_pcm_window_with_backend(
        &self,
        meeting_id: MeetingId,
        start_ms: u64,
        end_ms: u64,
        backend: Box<dyn minutist_common::AsrBackend + Send>,
    ) -> AppResult<Vec<Segment>> {
        if end_ms <= start_ms {
            return Err(AppError::InvalidInput {
                context: format!(
                    "relisten window end_ms ({end_ms}) must exceed start_ms ({start_ms})"
                ),
            });
        }
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());
        tokio::task::spawn_blocking(move || -> AppResult<Vec<Segment>> {
            let pcm = persistence::read_audio_pcm(&meeting_dir)?;
            let mut backend = backend;
            transcribe_pcm_window_blocking(&pcm, backend.as_mut(), start_ms, end_ms)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("transcribe_pcm_window_with_backend spawn_blocking join failed: {e}"),
        })?
    }
}
