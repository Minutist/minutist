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
//!   the OLDEST pending flush (a self-healing, log-only WARN — not a UI error).
//!   Audio is preserved in `audio.opus`, and the `incomplete` flag drives a
//!   post-stop re-transcribe that restores the dropped flush's transcript. See
//!   `runner.rs`.
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
pub mod matcher;
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
use tokio::sync::{broadcast, Mutex, Semaphore};

pub use error::Error;

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Maximum span of a single transcript "play segment" / re-listen clip
/// (`extract_segment_wav`). A transcript segment is a VAD chunk — seconds — so
/// this is generous; its purpose is to cap the WAV a `meetingrecording:` request
/// can produce, so a crafted oversized `end_ms` cannot be amplified (via the
/// pause-window clamp) into a whole-region decode/encode.
const MAX_RELISTEN_CLIP_MS: u64 = 300_000;

/// Concurrent whole-file decodes permitted on the re-listen cache-miss path.
const RELISTEN_DECODE_CONCURRENCY: usize = 2;

/// Depth of the resampled-sample channel passed to [`AudioManager::start`]. Deep
/// (≈ several seconds of batches) on purpose: the consumer stalls during the
/// model-load burst at record start, and a shallow channel would back-pressure
/// the capture forwarder into drop-flooding (truncating the recording). Deep
/// buffering rides the stall — the backlog drains once models load, losing no
/// audio. Partner of `audio-capture`'s `RAW_RING_CAPACITY` (the raw-frame ring).
const CAPTURE_SAMPLE_CHANNEL_DEPTH: usize = 512;

/// Depth of the level-meter channel passed to [`AudioManager::start`]; small —
/// meter updates are advisory and only the newest value matters.
const CAPTURE_METER_CHANNEL_DEPTH: usize = 64;

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
    /// Single-entry decoded-PCM cache for the transcript "play segment" /
    /// re-listen path ([`Self::extract_segment_wav`]). Holds the most-recently-
    /// played meeting's pause-INCLUDING f32 PCM **plus its pause-excluding
    /// kept-region table**, so rapid clicking through a meeting's segments (the
    /// manual-labelling workflow, #0023) decodes `audio.opus` AND runs the
    /// O(samples) pause scan once rather than once per click — each clip request
    /// then does only the cheap region→slice math. Single-entry bounds it to one
    /// meeting's PCM; `audio.opus` is never rewritten by reprocess, so an entry
    /// stays valid for its meeting id (UUIDs are never reused). A
    /// `std::sync::Mutex` because every access is a brief, non-awaiting
    /// check/insert.
    #[allow(clippy::type_complexity)]
    relisten_pcm_cache:
        StdMutex<Option<(MeetingId, Arc<Vec<f32>>, Arc<Vec<runner::KeptRegion>>)>>,
    /// Caps concurrent whole-file decodes on the re-listen cache-miss path so a
    /// flood of `meetingrecording:` requests for distinct meetings cannot pile up
    /// unbounded full-file PCM allocations (defence-in-depth — the normal UI uses
    /// a single shared `<audio>`).
    relisten_decode_sem: Semaphore,
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
    /// User-typed meeting title captured DURING the live recording (via
    /// `set_pending_title`), consumed by `stop()` in place of the synthesized
    /// `Recording <timestamp>` default. `None` = unnamed → the default is used.
    /// Reset at `start()` so a title never bleeds across meetings.
    pending_title: Option<String>,
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
                pending_title: None,
            }),
            event_tx,
            last_transcript_incomplete: Arc::new(AtomicBool::new(false)),
            prewarmed_asr: Arc::new(StdMutex::new(None)),
            relisten_pcm_cache: StdMutex::new(None),
            relisten_decode_sem: Semaphore::new(RELISTEN_DECODE_CONCURRENCY),
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
        // Fresh recording: drop any title left over from a prior meeting so it
        // cannot bleed across. A live title arrives later via `set_pending_title`.
        guard.pending_title = None;

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
        let streams = match capture.start(
            CAPTURE_SAMPLE_CHANNEL_DEPTH,
            CAPTURE_METER_CHANNEL_DEPTH,
            capture_system_audio,
        ) {
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

        // Surface the live state NOW — capture is already running and the writer
        // is open, so the recording is genuinely live. Emitting here, BEFORE the
        // GPU probe + (optional) live-diarizer model load + runner spawn below,
        // makes the meeting screen and the rec indicator appear immediately
        // instead of waiting on that setup. The runner (drain → transcribe) and
        // the lazy ASR model load follow in the background; no audio is lost (the
        // capture ring buffers until the runner drains it, exactly as before —
        // only the event moved earlier). No state rollback follows this point:
        // the remaining steps degrade rather than fail.
        self.emit(AppEvent::StateChanged {
            state: guard.state.as_public(),
        });

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

        // StateChanged(Recording) was already emitted above (right after capture +
        // writer came up) so the UI switched to the meeting screen immediately;
        // do not re-emit here.
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
        let (meeting_id, runner_handle, started_at, capture_opt, pending_title) = {
            let mut guard = self.inner.lock().await;
            let meeting_id = transition_stop(&mut guard.state)?;

            let new_state = guard.state.as_public();
            self.emit(AppEvent::StateChanged { state: new_state });

            let runner = guard.runner.take();
            let started_at = guard.started_at.take().unwrap_or_else(Utc::now);
            let capture = guard.capture.take();
            // The title the user typed during the recording (if any), consumed here.
            let pending_title = guard.pending_title.take();
            (meeting_id, runner, started_at, capture, pending_title)
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

        // Use the title the user typed during recording when present + non-blank;
        // otherwise fall back to the synthesized `Recording <timestamp>` default.
        let title = pending_title
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| format!("Recording {}", started_at.format("%Y-%m-%dT%H:%M:%SZ")));

        let meta = MeetingMeta {
            uuid: meeting_id,
            title,
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
            collection_id: None,
            // Recorded and processed on this device → the `Local` default.
            processing: Default::default(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        // Send stop command to runner and await the finalised metadata. The whole
        // handshake below is one fallible unit: a channel-send failure, a dropped
        // reply, or `MeetingWriter::finalise` returning an I/O error must all
        // release the recorder back to `Idle` rather than leaving it wedged in
        // `Stopping`/`Finalising` — see `Self::abort_finalise`.
        let finalised_meta = if let Some(handle) = runner_handle {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            let handshake: AppResult<MeetingMeta> = async {
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

                Ok(finalised)
            }
            .await;

            match handshake {
                Ok(finalised) => finalised,
                Err(err) => return self.abort_finalise(meeting_id, err).await,
            }
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

        // Announce the finalised meeting BEFORE the `Idle` transition. The webview
        // opens the just-recorded meeting on `MeetingFinalised` (so the user stays
        // on it instead of bouncing to the list); emitting it first means the
        // meeting is already opening by the time `Idle` arrives, so the workspace
        // never flashes the list as the recorder goes idle. The meeting is fully
        // on disk at this point (the runner reply was awaited above), so the row
        // also appears via `reconcile_orphans`/`upsert` on the list refresh.
        self.emit(AppEvent::MeetingFinalised { meeting_id });

        self.emit(AppEvent::StateChanged {
            state: RecordingState::Idle,
        });

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

    /// Recover from a failure in the [`Self::stop`] finalise handshake (the
    /// runner-command channel closing before `Stop` could be sent, the finalise
    /// reply channel dropping, or `MeetingWriter::finalise` itself returning an
    /// I/O error): release the recorder back to `Idle` and surface the failure,
    /// then return it as `Err` so the caller can `return
    /// self.abort_finalise(...).await;` directly from the handshake's error arm.
    ///
    /// Every exit path out of the handshake MUST go through here rather than
    /// propagating with a bare `?` — without it the state machine stays in
    /// `Stopping`/`Finalising` forever (no further recording is possible until
    /// the process restarts), because only the happy path drove the
    /// `Stopping`/`Finalising` → `Idle` transition.
    async fn abort_finalise(&self, meeting_id: MeetingId, err: AppError) -> AppResult<MeetingMeta> {
        {
            let mut guard = self.inner.lock().await;
            if let Err(e) = transition_idle(&mut guard.state) {
                tracing::warn!(
                    target: "orchestrator",
                    "unexpected state aborting finalise: {e:?}"
                );
            }
        }
        self.emit(AppEvent::StateChanged {
            state: RecordingState::Idle,
        });
        // Surfaces the failure as a global notification. This does NOT clear the
        // per-row `Finalise` `OperationProgress` indicator started above:
        // `ErrorOccurred` carries no `meeting_id`, and the webview's
        // per-meeting progress store (`ui/src/state/operation-progress.ts`)
        // only clears on `transcript_ready` / `diarization_complete` /
        // `summary_ready` / `summary_unavailable` / `meeting_finalised` /
        // `translation_ready` — `error_occurred` is not among them, so the
        // "Finalising…" spinner is left showing on this meeting's row.
        // Clearing it requires a UI-side change (treating `error_occurred` as
        // terminal, or a new meeting-scoped failure event); tracked as a
        // follow-up outside this crate.
        self.emit(AppEvent::ErrorOccurred { error: err.clone() });

        tracing::warn!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            "finalise handshake failed, recorder released to Idle: {err:?}"
        );

        Err(err)
    }

    /// Whether the on-stop diarization pass should run (the user's
    /// `diarization_enabled` setting). `ipc-bridge` reads this after `stop()` to
    /// decide whether to spawn the background [`Self::rediarize`] pass; the
    /// decision lives here so the orchestrator stays the single owner of the
    /// settings read, but the *execution* is decoupled from `stop()`.
    pub fn diarization_enabled(&self) -> bool {
        self.settings.current().diarization_enabled
    }

    /// Set the title the user typed for the LIVE recording, captured in-progress
    /// and consumed by [`Self::stop`] in place of the `Recording <timestamp>`
    /// default. A no-op unless `meeting_id` is the currently recording/paused
    /// meeting (guards against a late set racing a stop, or a stale id). An empty
    /// / whitespace title clears it (revert to the default). `metadata.json` is
    /// untouched here — it does not exist until finalise — so this never violates
    /// the "metadata present ⇒ finalised" invariant.
    ///
    /// Privacy (issue #0014): the title is user content; like `rename_meeting`
    /// this MUST NOT log the title — only the meeting id.
    pub async fn set_pending_title(&self, meeting_id: MeetingId, title: String) -> AppResult<()> {
        let mut guard = self.inner.lock().await;
        if guard.state.live_meeting_id() != Some(meeting_id) {
            // Not the live meeting (already stopping/finalised, or a stale id).
            return Ok(());
        }
        let trimmed = title.trim();
        guard.pending_title = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            "pending meeting title set"
        );
        Ok(())
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

    /// Enrol a speaker voiceprint for `label` in `meeting_id` into `store`.
    ///
    /// Runs on a `spawn_blocking` thread, holding the per-meeting offline claim
    /// for the duration so that a concurrent `reprocess` cannot rewrite
    /// `transcript.json` and invalidate the label→segment mapping mid-enrolment.
    ///
    /// Steps (all on the blocking thread):
    /// 1. Read `transcript.json` for `meeting_id`.
    /// 2. Read and decode `audio.opus` (pause-INCLUDING PCM).
    /// 3. Filter to **clean** segments for `label` (§2.3.1 filter): `speaker_id ==
    ///    label`, `shared_speakers` empty, duration ≥ 1.0 s.
    /// 4. Map each clean segment through the pause-excl → incl clock mapper
    ///    (`runner::pcm_window_for_excluding_range`) to get the correct PCM slice.
    /// 5. Build a centroid via `diarizer::VoiceprintExtractor::centroid`.
    /// 6. Call `VoiceprintStore::enrol` with the centroid vector, `name`, `model_id`,
    ///    `meeting_id`, and `label`.
    ///
    /// Skips silently (returns `Ok(())`) when there are no clean windows above the
    /// minimum duration — enrolment is not forced. Uses `claim_offline` / `release_offline`
    /// so a `reprocess` racing this call is rejected until enrolment finishes.
    ///
    /// Returns the `VoiceprintIdentityId` on success, or `Ok(None)` when skipped.
    /// Errors from the extractor or store are propagated; the caller (ipc-bridge)
    /// is expected to log-and-swallow so a rename never fails because of enrolment.
    pub async fn enrol_voiceprint(
        &self,
        meeting_id: MeetingId,
        label: String,
        name: String,
        store: &persistence::VoiceprintStore,
    ) -> AppResult<Option<minutist_common::VoiceprintIdentityId>> {
        // Hold the claim so that a concurrent reprocess cannot rewite
        // transcript.json while we read it. Best-effort: if the claim cannot
        // be taken (a reprocess or re-transcribe is already running), we skip
        // enrolment rather than blocking the rename.
        if self.claim_offline(meeting_id).await.is_err() {
            tracing::debug!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                label = %label,
                "enrol_voiceprint: offline claim unavailable, skipping enrolment"
            );
            return Ok(None);
        }
        let result = self
            .enrol_voiceprint_claimed(meeting_id, label, name, store)
            .await;
        self.release_offline().await;
        result
    }

    /// The enrol_voiceprint body under the held offline claim.
    async fn enrol_voiceprint_claimed(
        &self,
        meeting_id: MeetingId,
        label: String,
        name: String,
        store: &persistence::VoiceprintStore,
    ) -> AppResult<Option<minutist_common::VoiceprintIdentityId>> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());
        let registry = Arc::clone(&self.model_registry);

        // Resolve the embedding model path synchronously (no download — must be
        // Available for the offline diarizer to have run at all). Mirrors the
        // `build_online_diarizer` local-only resolve.
        let emb_id = minutist_common::ModelId::from(runner::DIARIZE_EMB_MODEL_ID);
        let emb_dir_opt = {
            use minutist_common::ModelStatusState;
            registry
                .list_models()
                .into_iter()
                .find(|s| s.id == emb_id)
                .and_then(|s| match s.status {
                    ModelStatusState::Available { local_dir } => Some(local_dir),
                    _ => None,
                })
        };
        let emb_dir = match emb_dir_opt {
            Some(d) => d,
            None => {
                tracing::debug!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    label = %label,
                    "enrol_voiceprint: embedding model not available; skipping"
                );
                return Ok(None);
            }
        };

        let label_clone = label.clone();
        let name_clone = name.clone();
        let model_id_str = runner::DIARIZE_EMB_MODEL_ID.to_string();

        // The heavy work (read PCM + transcript, extract embeddings) runs on
        // spawn_blocking. The store async operations are awaited afterwards.
        // Returns `(centroid, window_count)` so the caller can use the count
        // as the contribution weight when refining an existing identity.
        let centroid_result = tokio::task::spawn_blocking(move || -> AppResult<Option<(diarizer::Voiceprint, u64)>> {
            use runner::find_file_in_dir;

            let emb_onnx = find_file_in_dir(
                std::path::Path::new(&emb_dir),
                |name| name.ends_with(".onnx"),
            )?;

            let extractor = diarizer::VoiceprintExtractor::open(&emb_onnx)?;

            let pcm = persistence::read_audio_pcm(&meeting_dir)?;
            let segments = persistence::read_transcript(&meeting_dir)?;

            // §2.3.1 cleanliness filter: speaker_id == label, shared_speakers empty,
            // duration >= 1.0 s (1000 ms).
            let clean_segs = clean_segments_for_label(&segments, &label_clone);

            if clean_segs.is_empty() {
                tracing::debug!(
                    target: "orchestrator",
                    label = %label_clone,
                    "enrol_voiceprint: no clean segments above minimum duration; skipping"
                );
                return Ok(None);
            }

            // Map each clean segment through the pause-excl → incl clock mapper
            // (§2.3 clock-mismatch hazard), against ONE pause scan for this meeting
            // (B2). Collect only those windows that the mapper resolves to a
            // non-empty PCM slice.
            let regions = runner::pause_excluding_segments(&pcm);
            let windows = segment_windows(&pcm, &regions, &clean_segs);

            if windows.is_empty() {
                tracing::debug!(
                    target: "orchestrator",
                    label = %label_clone,
                    "enrol_voiceprint: clock mapper yielded no usable PCM windows; skipping"
                );
                return Ok(None);
            }

            let (centroid, window_count) = centroid_from_windows(&extractor, &windows, 16_000)?;

            tracing::debug!(
                target: "orchestrator",
                label = %label_clone,
                windows = window_count,
                dim = centroid.dim(),
                "enrol_voiceprint: centroid built"
            );

            Ok(Some((centroid, window_count)))
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("enrol_voiceprint spawn_blocking join failed: {e}"),
        })??;

        let (centroid, window_count) = match centroid_result {
            Some(pair) => pair,
            None => return Ok(None),
        };

        // If an identity already exists for this display_name + model, this is a
        // confirmed subsequent association: call refine rather than enrol.
        // The UI rename path is always a confirmed association (§2.9.3 trigger (a)).
        let existing_id = store
            .find_identity_by_name_and_model(&name_clone, &model_id_str)
            .await?;

        let identity_id = if let Some(id) = existing_id {
            store
                .refine(id, &centroid.vector, window_count, &model_id_str, meeting_id, &label)
                .await?;
            tracing::info!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                label = %label,
                identity_id = %id.0,
                windows = window_count,
                "voiceprint refined (confirmed subsequent association)"
            );
            id
        } else {
            let dim = centroid.dim();
            let id = store
                .enrol(&name_clone, &centroid.vector, dim, &model_id_str, meeting_id, &label)
                .await?;
            tracing::info!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                label = %label,
                identity_id = %id.0,
                "voiceprint enrolled"
            );
            id
        };

        Ok(Some(identity_id))
    }

    /// Correct a false-accept match: remove the name overlay for `label` in
    /// `meeting_id` and drop that meeting/label's contribution from
    /// `identity_id`'s gallery, then recompute the affected centroid (§2.4
    /// correction path — `reject_match`).
    ///
    /// Steps:
    /// 1. Clear `speaker_names[label]` for `meeting_id` via
    ///    `persistence::meeting_ops::set_speaker_name(..., "")` (empty name =
    ///    remove the mapping — the same write as "unname a speaker").
    /// 2. Load all centroids for `identity_id` via `VoiceprintStore::all`, filter
    ///    to the identity, then call `VoiceprintStore::forget_contribution` on each
    ///    centroid. The method is idempotent when no matching contribution exists.
    ///
    /// The name-clear is best-effort: if the meeting folder is absent (already
    /// deleted) it returns `MeetingNotFound`; the caller (`ipc-bridge`) decides
    /// whether to surface this. The contribution drop is also best-effort.
    pub async fn reject_match(
        &self,
        meeting_id: MeetingId,
        label: String,
        identity_id: minutist_common::VoiceprintIdentityId,
        model_id: &str,
        store: &persistence::VoiceprintStore,
    ) -> AppResult<()> {
        // Step 1: clear the name overlay — empty string removes the mapping.
        let _ = persistence::meeting_ops::set_speaker_name(
            &self.persistence_root,
            meeting_id,
            &label,
            "",
        )
        .await;
        // Ignore MeetingNotFound (folder may have been deleted already); propagate
        // other errors.

        // Step 2: iterate the identity's gallery centroids and drop the
        // (meeting_id, label) contribution from each. The forget_contribution
        // call is idempotent when no row matches.
        let gallery = store.all(model_id).await?;
        for entry in gallery.iter().filter(|e| e.identity_id == identity_id) {
            store
                .forget_contribution(entry.centroid_id, meeting_id, &label)
                .await?;
        }

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            label = %label,
            identity_id = %identity_id.0,
            "reject_match: name overlay cleared and contribution dropped"
        );

        Ok(())
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

        // §2.6 snapshot: capture old speaker_names + transcript BEFORE the
        // re-transcribe overwrites transcript.json. The snapshot is passed to
        // `rediarize_inner_with_snapshot` so it can restore matched names after
        // the fresh diarize pass. Best-effort: a snapshot failure is logged and
        // the re-map is skipped for this run; the reprocess itself continues.
        let pre_snapshot = self.capture_reprocess_snapshot(meeting_id, &meeting_dir).await;

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

        // The reprocess path does not pre-load the gallery; the prune-veto requires
        // the store which is held by ipc-bridge. Pass Vec::new() here — the veto
        // will be a no-op for the reprocess path. The ipc-bridge calls
        // apply_voiceprint_matches after reprocess to handle post-diarize matching.
        self.rediarize_inner_with_snapshot_and_gallery(
            index,
            meeting_id,
            DiarizationJob::Production { diarizer, backend },
            pre_snapshot,
            Vec::new(),
        )
        .await
    }

    /// Capture an optional pre-reprocess snapshot of `(old_speaker_names, old_segments)`.
    ///
    /// Used by the `reprocess` path to snapshot the old state BEFORE re-transcribe
    /// overwrites `transcript.json`. Returns `None` when enrolment is disabled, when
    /// old names are empty, or when the read fails (best-effort).
    async fn capture_reprocess_snapshot(
        &self,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
    ) -> Option<(std::collections::BTreeMap<String, String>, Vec<Segment>)> {
        if !self.settings.current().voiceprint_enrolment_enabled {
            return None;
        }
        let snap_dir = meeting_dir.to_path_buf();
        match tokio::task::spawn_blocking(move || -> AppResult<(std::collections::BTreeMap<String, String>, Vec<Segment>)> {
            let meta = persistence::read_metadata(&snap_dir)?;
            let segs = persistence::read_transcript(&snap_dir)?;
            Ok((meta.speaker_names, segs))
        })
        .await
        {
            Ok(Ok(pair)) => {
                if pair.0.is_empty() {
                    None
                } else {
                    Some(pair)
                }
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "reprocess pre-snapshot read failed; ephemeral re-map skipped"
                );
                None
            }
            Err(e) => {
                tracing::warn!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "reprocess pre-snapshot join failed; ephemeral re-map skipped"
                );
                None
            }
        }
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
        let root = self.persistence_root.clone();
        let entry: MeetingListEntry = tokio::task::spawn_blocking(move || -> AppResult<MeetingListEntry> {
            // Guarded RMW of `speaker_count` (issue 0025): the per-meeting metadata
            // lock serialises this against the sync lifecycle subscriber so neither
            // reverts the other's field. Returns the post-write meta for the entry.
            let meta = persistence::meeting_ops::update_metadata(&root, meeting_id, |meta| {
                meta.speaker_count = speaker_count;
                meta.clone()
            })?;
            let transcript = persistence::read_transcript(&meeting_dir_for_meta)?;
            Ok(MeetingListEntry {
                id: meta.uuid,
                title: meta.title,
                started_at: meta.started_at,
                duration_ms: meta.duration_ms,
                speaker_count: meta.speaker_count,
                excerpt: transcript.first().map(|s| s.text.clone()),
                collection_id: meta.collection_id,
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
        // background re-transcribe with diarization OFF would leave the UI
        // showing the stale/truncated transcript until a manual refresh.
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

    /// Extract the audio for a transcript window as a self-contained 16 kHz mono
    /// PCM16 WAV, for the in-app transcript "play segment" affordance (re-listen
    /// without re-ASR).
    ///
    /// Unlike [`Self::transcribe_pcm_window`] this runs **no inference**: it
    /// decodes `audio.opus`, maps the pause-EXCLUDING `[start_ms, end_ms)`
    /// transcript window onto the pause-INCLUDING PCM via
    /// [`runner::pcm_window_for_excluding_range`] (same clamp-across-a-pause
    /// rule), and returns a WAV of exactly that slice. Serving a pre-cut WAV lets
    /// the webview play the clip start-to-finish with no seeking, so the Ogg
    /// granule-position non-conformance (#0024) and container quirks never enter
    /// the playback path.
    ///
    /// Read-only and cheap (decode-once, then cache). Rejects a window outside
    /// the recording, a window longer than [`MAX_RELISTEN_CLIP_MS`] (so the
    /// pause-window clamp can never amplify a crafted oversized `end_ms` into a
    /// whole-region WAV), and — mirroring `transcribe_pcm_window`'s W2 guard — a
    /// meeting that is still recording/finalising (its `audio.opus` is mid-write).
    ///
    /// **Reprocess concurrency.** A background re-transcribe/re-diarize reports
    /// the recorder as `Idle` (it holds the separate offline claim), so this is
    /// reachable during a reprocess. That is safe — it only reads `audio.opus`,
    /// which reprocess never rewrites — but a `[start_ms, end_ms)` the caller
    /// captured before the rewrite may cut a window stale relative to the new
    /// `transcript.json`. The displayed transcript is likewise stale until the
    /// reprocess completes, so the clip still matches what the user sees.
    ///
    /// # Errors
    ///
    /// - `AppError::InvalidInput` if `end_ms <= start_ms`, the window exceeds the
    ///   clip cap, the meeting is still active, or the window falls outside the
    ///   recording.
    /// - The `persistence` decode error if `audio.opus` is missing/corrupt.
    pub async fn extract_segment_wav(
        &self,
        meeting_id: MeetingId,
        start_ms: u64,
        end_ms: u64,
    ) -> AppResult<Vec<u8>> {
        if end_ms <= start_ms {
            return Err(AppError::InvalidInput {
                context: format!(
                    "segment window end_ms ({end_ms}) must exceed start_ms ({start_ms})"
                ),
            });
        }
        // Cap the REQUESTED span before any decode. The pause-window clamp only
        // ever shrinks the mapped slice, so a bounded request → bounded output —
        // this is what closes the amplification path (a crafted huge `end_ms`).
        if end_ms - start_ms > MAX_RELISTEN_CLIP_MS {
            return Err(AppError::InvalidInput {
                context: format!(
                    "segment window [{start_ms}, {end_ms}) ms exceeds the {MAX_RELISTEN_CLIP_MS} ms re-listen cap"
                ),
            });
        }

        // Re-listen is defined only over FINALISED audio (W2): reject the meeting
        // currently being recorded/finalised — its `audio.opus` is still being
        // appended, so a full-file decode can hit a truncated OGG page.
        let active = match self.state().await {
            RecordingState::Recording { meeting_id, .. }
            | RecordingState::Paused { meeting_id, .. }
            | RecordingState::Stopping { meeting_id }
            | RecordingState::Finalising { meeting_id } => Some(meeting_id),
            RecordingState::Idle => None,
        };
        if active == Some(meeting_id) {
            return Err(AppError::InvalidInput {
                context: "cannot play audio from a meeting that is still recording or finalising"
                    .into(),
            });
        }

        // Reuse the cached decode + pause-region table for this meeting (the
        // rapid-click labelling path); else decode `audio.opus` once under the
        // decode-concurrency semaphore, run the O(samples) pause scan once, and
        // cache both (single-entry, so memory is bounded to one meeting's PCM).
        // The lock is held only for the brief check/insert, never across the
        // decode or `.await`.
        let cached = {
            let guard = self.relisten_pcm_cache.lock().unwrap();
            guard
                .as_ref()
                .filter(|(id, _, _)| *id == meeting_id)
                .map(|(_, pcm, regions)| (Arc::clone(pcm), Arc::clone(regions)))
        };
        let (pcm, regions) = match cached {
            Some(hit) => hit,
            None => {
                let _permit = self.relisten_decode_sem.acquire().await.map_err(|e| {
                    AppError::Internal {
                        context: format!("relisten decode semaphore closed: {e}"),
                    }
                })?;
                let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());
                // Decode + the O(samples) pause scan together off the async
                // threads; both are cached so subsequent clips skip them.
                let (pcm, regions) = tokio::task::spawn_blocking(move || {
                    persistence::read_audio_pcm(&meeting_dir)
                        .map(|pcm| {
                            let regions = runner::pause_excluding_segments(&pcm);
                            (pcm, regions)
                        })
                })
                .await
                .map_err(|e| AppError::Internal {
                    context: format!("extract_segment_wav decode join failed: {e}"),
                })??;
                let pcm = Arc::new(pcm);
                let regions = Arc::new(regions);
                *self.relisten_pcm_cache.lock().unwrap() =
                    Some((meeting_id, Arc::clone(&pcm), Arc::clone(&regions)));
                (pcm, regions)
            }
        };

        // Map the window + encode the WAV off the async threads. The per-request
        // region→slice math is cheap; the O(samples) pause scan was done at decode
        // time (and cached). The mapped slice is ≤ the requested span, hence bounded.
        tokio::task::spawn_blocking(move || -> AppResult<Vec<u8>> {
            let range = runner::excluding_range_to_pcm_slice(&regions, start_ms, end_ms)
                .ok_or_else(|| AppError::InvalidInput {
                    context: format!(
                        "segment window [{start_ms}, {end_ms}) ms is outside the recording"
                    ),
                })?;
            Ok(runner::pcm16_wav(&pcm[range]))
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("extract_segment_wav encode join failed: {e}"),
        })?
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
    /// (`notes_crdt::write_metadata`, setting `speaker_count` + the `diarizer`
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
    /// Re-diarize a completed meeting, overlaying fresh speaker labels on the
    /// existing transcript.
    ///
    /// `voiceprint_store` is optional. When `Some`, the prune-veto pass (§2.5)
    /// loads the enrolled gallery and may rescue low-share clusters that match an
    /// enrolled speaker above `T_ACCEPT`. When `None` (or when
    /// `voiceprint_enrolment_enabled` is OFF), the prune runs normally.
    pub async fn rediarize(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        voiceprint_store: Option<&persistence::VoiceprintStore>,
    ) -> AppResult<()> {
        // Atomic claim/release (TIMELINE-DRIFT #5): rediarize also rewrites
        // `transcript.json`, so it must claim the slot just like re_transcribe.
        self.claim_offline(meeting_id).await?;
        let result = self.rediarize_claimed(index, meeting_id, voiceprint_store).await;
        self.release_offline().await;
        result
    }

    /// The re-diarize body, run while the offline claim is held (so the claim is
    /// released on every exit path).
    async fn rediarize_claimed(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        voiceprint_store: Option<&persistence::VoiceprintStore>,
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

        // Pre-load the voiceprint gallery for the §2.5 prune-veto pass. The gallery
        // is a plain Vec so it can be moved into the spawn_blocking closure.
        let gallery = load_voiceprint_gallery(voiceprint_store, meeting_id).await;

        self.rediarize_inner_with_gallery(
            index,
            meeting_id,
            DiarizationJob::Production { diarizer, backend },
            gallery,
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

    /// Build a `VoiceprintExtractor` for the §2.5 prune-veto pass, best-effort.
    ///
    /// Returns `None` on ANY failure (model not downloaded, `.onnx` not located,
    /// open error) so the diarization pass degrades to no-veto — missing model
    /// files must never block the diarization. The extractor is opened on a
    /// `spawn_blocking` thread (the heavy `EmbeddingExtractor::new` load is
    /// synchronous). The gallery is pre-loaded by the caller and passed separately.
    async fn build_prune_veto_extractor(
        &self,
        meeting_id: MeetingId,
    ) -> Option<diarizer::VoiceprintExtractor> {
        use minutist_common::ModelStatusState;
        use runner::{find_file_in_dir, DIARIZE_EMB_MODEL_ID};

        let emb_id = minutist_common::ModelId::from(DIARIZE_EMB_MODEL_ID);
        let registry = Arc::clone(&self.model_registry);

        let emb_onnx_opt: Option<std::path::PathBuf> =
            tokio::task::spawn_blocking(move || -> Option<std::path::PathBuf> {
                let local_dir = registry
                    .list_models()
                    .into_iter()
                    .find(|s| s.id == emb_id)
                    .and_then(|s| match s.status {
                        ModelStatusState::Available { local_dir } => Some(local_dir),
                        _ => None,
                    });
                let local_dir = local_dir?;
                find_file_in_dir(std::path::Path::new(&local_dir), |name| {
                    name.ends_with(".onnx")
                })
                .ok()
            })
            .await
            .unwrap_or(None);

        let emb_onnx = match emb_onnx_opt {
            Some(p) => p,
            None => {
                tracing::debug!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    "prune-veto: embedding model not available; skipping"
                );
                return None;
            }
        };

        // Open the extractor on spawn_blocking (heavy ONNX load).
        match tokio::task::spawn_blocking(move || diarizer::VoiceprintExtractor::open(&emb_onnx))
            .await
        {
            Ok(Ok(ext)) => Some(ext),
            Ok(Err(e)) => {
                tracing::warn!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "prune-veto: VoiceprintExtractor open failed; skipping"
                );
                None
            }
            Err(join_err) => {
                tracing::warn!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    error = %join_err,
                    "prune-veto: extractor build join failed; skipping"
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
    /// Convenience wrapper with no pre-snapshot and no gallery (stub/test paths).
    #[cfg_attr(not(any(test, feature = "test-source")), allow(dead_code))]
    async fn rediarize_inner(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        job: DiarizationJob,
    ) -> AppResult<()> {
        self.rediarize_inner_with_snapshot_and_gallery(index, meeting_id, job, None, Vec::new())
            .await
    }

    /// Convenience wrapper with a pre-loaded gallery but no pre-snapshot (production
    /// rediarize path where the store is available).
    async fn rediarize_inner_with_gallery(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        job: DiarizationJob,
        gallery: Vec<persistence::StoredVoiceprint>,
    ) -> AppResult<()> {
        self.rediarize_inner_with_snapshot_and_gallery(index, meeting_id, job, None, gallery)
            .await
    }

    /// Inner re-diarize body that accepts an optional pre-computed snapshot of
    /// `(old_speaker_names, old_segments)`.
    ///
    /// When `pre_snapshot` is `Some`, it was captured before a re-transcribe step
    /// overwrote `transcript.json` (the `reprocess` path). When `None`, this
    /// method reads the snapshot itself from the current on-disk state — correct
    /// for the standalone `rediarize` path where the transcript has not yet been
    /// overwritten.
    ///
    /// `gallery` is the voiceprint library for the §2.5 prune-veto pass. An empty
    /// Vec disables the veto pass (the prune runs normally). The gallery is
    /// pre-loaded by the caller (from `VoiceprintStore::all(model_id)`) so the
    /// spawn_blocking thread has a plain Vec with no DB handle.
    async fn rediarize_inner_with_snapshot_and_gallery(
        &self,
        index: &MeetingIndex,
        meeting_id: MeetingId,
        job: DiarizationJob,
        pre_snapshot: Option<(std::collections::BTreeMap<String, String>, Vec<Segment>)>,
        gallery: Vec<persistence::StoredVoiceprint>,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());

        // §2.6 re-map: when voiceprint enrolment is enabled, ensure we have a
        // snapshot of old speaker_names + old transcript BEFORE the fresh diarize
        // rewrites them. On the reprocess path the caller provides `pre_snapshot`
        // (captured before re-transcribe); on the standalone rediarize path we read
        // it here while the transcript still holds the pre-diarize state.
        let old_snapshot: Option<(
            std::collections::BTreeMap<String, String>,
            Vec<Segment>,
        )> = if self.settings.current().voiceprint_enrolment_enabled {
            match pre_snapshot {
                Some(snap) => {
                    if snap.0.is_empty() { None } else { Some(snap) }
                }
                None => {
                    let snap_dir = meeting_dir.clone();
                    match tokio::task::spawn_blocking(move || -> AppResult<(std::collections::BTreeMap<String, String>, Vec<Segment>)> {
                        let meta = persistence::read_metadata(&snap_dir)?;
                        let segs = persistence::read_transcript(&snap_dir)?;
                        Ok((meta.speaker_names, segs))
                    })
                    .await
                    .map_err(|e| AppError::Internal {
                        context: format!("rediarize pre-snapshot join failed: {e}"),
                    })? {
                        Ok(pair) => {
                            if pair.0.is_empty() {
                                None
                            } else {
                                Some(pair)
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "orchestrator",
                                meeting_id = %meeting_id.0,
                                error = %e,
                                "rediarize pre-snapshot failed; proceeding without ephemeral re-map"
                            );
                            None
                        }
                    }
                }
            }
        } else {
            None
        };

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

        // §2.5 prune-veto: build the second VoiceprintExtractor best-effort, gated
        // on voiceprint_enrolment_enabled and model availability. The gallery was
        // pre-loaded by the caller from VoiceprintStore::all(model_id) and passed in
        // as `gallery`; the extractor is built here (heavy ONNX load, blocking).
        // When either is absent (model not downloaded, gallery empty), the veto pass
        // is a no-op and the prune runs normally.
        let prune_veto_extractor = if self.settings.current().voiceprint_enrolment_enabled
            && !gallery.is_empty()
        {
            self.build_prune_veto_extractor(meeting_id).await
        } else {
            None
        };

        // Bound the (uninterruptible) sherpa `compute` + re-ASR split: a
        // pathologically slow or hung pass on a long recording must not block
        // forever (the original on-stop hang). On timeout we return BEFORE
        // `finalise_diarization`, so nothing is written — the meeting is left
        // un-diarized and the abandoned blocking thread's result (if it ever
        // completes) is dropped. `tokio` cannot cancel a `spawn_blocking` thread,
        // so a true infinite hang leaks one thread until process exit; the budget
        // bounds the wait, not the thread.
        let (segments, speaker_count, veto_names) = match tokio::time::timeout(
            budget,
            run_diarization_blocking(
                meeting_dir.clone(),
                job,
                prune_veto_extractor,
                gallery,
                self.event_tx.clone(),
                meeting_id,
            ),
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

        // Write any prune-veto names (enrolled quiet speakers rescued by the veto
        // pass) to speaker_names. This is a second metadata write after
        // finalise_diarization's clear, the same pattern as apply_voiceprint_matches.
        // The offline claim is still held throughout, so no concurrent op races this.
        if !veto_names.is_empty() {
            let root = self.persistence_root.clone();
            let veto_names_copy = veto_names.clone();
            let write_result = tokio::task::spawn_blocking(move || -> AppResult<()> {
                // Guarded RMW (issue 0025): serialises with the lifecycle subscriber.
                persistence::meeting_ops::update_metadata(&root, meeting_id, |meta| {
                    for (letter, name) in &veto_names_copy {
                        meta.speaker_names.insert(letter.clone(), name.clone());
                    }
                })
            })
            .await;
            match write_result {
                Ok(Ok(())) => {
                    tracing::info!(
                        target: "orchestrator",
                        meeting_id = %meeting_id.0,
                        count = veto_names.len(),
                        "prune-veto: wrote enrolled-speaker names to speaker_names"
                    );
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        target: "orchestrator",
                        meeting_id = %meeting_id.0,
                        error = %e,
                        "prune-veto: names write failed (best-effort); continuing"
                    );
                }
                Err(join_err) => {
                    tracing::warn!(
                        target: "orchestrator",
                        meeting_id = %meeting_id.0,
                        error = %join_err,
                        "prune-veto: names write join failed (best-effort); continuing"
                    );
                }
            }
        }

        // §2.6 ephemeral re-map: if we snapshotted old names before the diarize,
        // try to restore names to the fresh clusters via timeline-coherence and
        // (when the embedding model is available) centroid matching. Best-effort:
        // errors are logged; the meeting is left with cleared names, not failed.
        if let Some((old_names, old_segments)) = old_snapshot {
            self.apply_ephemeral_remap(meeting_id, &meeting_dir, old_names, old_segments, &segments)
                .await;
        }

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
                    collection_id: meta.collection_id,
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
    /// `notes_crdt::write_metadata`, keeping `persistence` the sole writer under
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
        let root = self.persistence_root.clone();
        tokio::task::spawn_blocking(move || -> AppResult<()> {
            persistence::write_transcript(&meeting_dir_for_write, &segments_for_write)?;
            // Guarded RMW (issue 0025): the per-meeting metadata lock serialises
            // this against the sync lifecycle subscriber, so a concurrent
            // `Claimed`/`Processed` write cannot be reverted (and vice versa).
            persistence::meeting_ops::update_metadata(&root, meeting_id, |meta| {
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
            })
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

    /// Apply voiceprint matches for `meeting_id` after a completed diarisation pass.
    ///
    /// Runs the §2.4 global greedy assignment against the stored gallery. For each
    /// accepted match (sim >= `T_ACCEPT` with margin), restores the matched name
    /// into `speaker_names`. For each uncertain match (`T_REJECT <= sim <
    /// T_ACCEPT`), emits `AppEvent::VoiceprintSuggestions` so the UI can present
    /// the "is this \<Name\>?" affordance.
    ///
    /// Called by ipc-bridge after `reprocess` or a standalone `rediarize`
    /// completes, gated on `voiceprint_enrolment_enabled`. Best-effort: errors
    /// are logged and the meeting is left with cleared names rather than
    /// propagated.
    ///
    /// This is a second write to `metadata.json` (after `finalise_diarization`'s
    /// transcript + meta write). The offline claim is still held by the caller,
    /// so no concurrent op can race this write.
    pub async fn apply_voiceprint_matches(
        &self,
        meeting_id: MeetingId,
        store: &persistence::VoiceprintStore,
    ) -> AppResult<()> {
        let meeting_dir = self.persistence_root.join(meeting_id.0.to_string());
        let model_id = runner::DIARIZE_EMB_MODEL_ID.to_string();

        // Read the fresh transcript to know which labels exist.
        let meeting_dir_r = meeting_dir.clone();
        let segments = tokio::task::spawn_blocking(move || persistence::read_transcript(&meeting_dir_r))
            .await
            .map_err(|e| AppError::Internal {
                context: format!("apply_voiceprint_matches transcript read join failed: {e}"),
            })??;

        let (matched_names, uncertain) = match self
            .compute_voiceprint_matches(meeting_id, &meeting_dir, &segments, store, &model_id)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                tracing::warn!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "apply_voiceprint_matches: matching failed (best-effort); speaker names unchanged"
                );
                return Ok(());
            }
        };

        if matched_names.is_empty() && uncertain.is_empty() {
            return Ok(());
        }

        // Update speaker_names: restore accepted matches on top of the cleared map.
        if !matched_names.is_empty() {
            let root = self.persistence_root.clone();
            let names_copy = matched_names.clone();
            tokio::task::spawn_blocking(move || -> AppResult<()> {
                // Guarded RMW (issue 0025): serialises with the lifecycle subscriber.
                // speaker_names was cleared by finalise_diarization; restore accepted ones.
                persistence::meeting_ops::update_metadata(&root, meeting_id, |meta| {
                    for (label, name) in &names_copy {
                        meta.speaker_names.insert(label.clone(), name.clone());
                    }
                })
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("apply_voiceprint_matches metadata write join failed: {e}"),
            })??;

            tracing::info!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                accepted_count = matched_names.len(),
                "voiceprint matching restored accepted names"
            );
        }

        // Emit uncertain suggestions for the UI affordance.
        if !uncertain.is_empty() {
            self.emit(AppEvent::VoiceprintSuggestions {
                meeting_id,
                suggestions: uncertain,
            });
        }

        Ok(())
    }

    /// Compute voiceprint matches for all fresh diariser labels in `segments`.
    ///
    /// For each unique label, extracts a centroid from clean PCM windows (the
    /// same §2.3.1 filter and clock mapper used by `enrol_voiceprint_claimed`),
    /// then runs `matcher::assign_identities` against the stored gallery.
    ///
    /// Returns `(accepted_names, uncertain_suggestions)` where:
    /// - `accepted_names`: label → display_name for matches in the Accept band.
    /// - `uncertain_suggestions`: one entry per Uncertain-band match.
    ///
    /// Best-effort: if the embedding model is absent, returns two empty maps so
    /// the caller falls back to `speaker_names.clear()`.
    async fn compute_voiceprint_matches(
        &self,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
        segments: &[Segment],
        store: &persistence::VoiceprintStore,
        model_id: &str,
    ) -> AppResult<(
        std::collections::BTreeMap<String, String>,
        Vec<minutist_common::VoiceprintSuggestion>,
    )> {
        use crate::matcher::{assign_identities, MatchBand, QueryCluster};

        // Collect the distinct labels from the fresh segments.
        let labels: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            segments
                .iter()
                .filter_map(|s| s.speaker_id.clone())
                .filter(|l| seen.insert(l.clone()))
                .collect()
        };

        if labels.is_empty() {
            return Ok(Default::default());
        }

        // Load the gallery filtered to the active model. If no enrolled speakers,
        // there is nothing to match against.
        let gallery = store.all(model_id).await?;
        if gallery.is_empty() {
            return Ok(Default::default());
        }

        // Resolve the embedding model path; skip if not locally available.
        let registry = Arc::clone(&self.model_registry);
        let emb_id = minutist_common::ModelId::from(runner::DIARIZE_EMB_MODEL_ID);
        let emb_dir_opt = {
            use minutist_common::ModelStatusState;
            registry
                .list_models()
                .into_iter()
                .find(|s| s.id == emb_id)
                .and_then(|s| match s.status {
                    ModelStatusState::Available { local_dir } => Some(local_dir),
                    _ => None,
                })
        };
        let emb_dir = match emb_dir_opt {
            Some(d) => d,
            None => {
                tracing::debug!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    "compute_voiceprint_matches: embedding model not available; skipping"
                );
                return Ok(Default::default());
            }
        };

        let meeting_dir_owned = meeting_dir.to_path_buf();
        let segments_owned = segments.to_vec();
        let model_id_str = model_id.to_string();

        // Extract a centroid for every label on spawn_blocking.
        let query_clusters: Vec<QueryCluster> = tokio::task::spawn_blocking(move || -> AppResult<Vec<QueryCluster>> {
            use runner::find_file_in_dir;

            let emb_onnx = find_file_in_dir(
                std::path::Path::new(&emb_dir),
                |name| name.ends_with(".onnx"),
            )?;
            let extractor = diarizer::VoiceprintExtractor::open(&emb_onnx)?;
            let pcm = persistence::read_audio_pcm(&meeting_dir_owned)?;
            // ONE pause scan for the whole meeting (B2), shared across every label.
            let regions = runner::pause_excluding_segments(&pcm);

            let mut clusters = Vec::new();

            for label in &labels {
                let clean_segs = clean_segments_for_label(&segments_owned, label);

                if clean_segs.is_empty() {
                    tracing::debug!(
                        target: "orchestrator",
                        label = %label,
                        "compute_voiceprint_matches: no clean segments for label, skipping"
                    );
                    continue;
                }

                let windows = segment_windows(&pcm, &regions, &clean_segs);

                if windows.is_empty() {
                    continue;
                }

                match centroid_from_windows(&extractor, &windows, 16_000) {
                    Ok((centroid, window_count)) => clusters.push(QueryCluster {
                        label: label.clone(),
                        centroid: centroid.vector,
                        window_count,
                    }),
                    Err(e) => {
                        tracing::warn!(
                            target: "orchestrator",
                            label = %label,
                            error = %e,
                            "compute_voiceprint_matches: centroid extraction failed for label"
                        );
                    }
                }
            }

            let _ = model_id_str; // silence dead-code warning — used in the outer scope
            Ok(clusters)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("voiceprint match extraction join failed: {e}"),
        })??;

        if query_clusters.is_empty() {
            return Ok(Default::default());
        }

        // Run the global greedy assignment.
        let assignments = assign_identities(&query_clusters, &gallery);

        let mut accepted: std::collections::BTreeMap<String, String> = Default::default();
        let mut uncertain_out: Vec<minutist_common::VoiceprintSuggestion> = Vec::new();

        // Label -> (centroid, window_count) for refining auto-accepted matches.
        let by_label: std::collections::HashMap<&str, (&[f32], u64)> = query_clusters
            .iter()
            .map(|q| (q.label.as_str(), (q.centroid.as_slice(), q.window_count)))
            .collect();

        for m in assignments {
            match m.band {
                MatchBand::Accept => {
                    // Refinement-on-confirm (§2.9.3 trigger b): an Accept-band match
                    // has cleared T_accept *and* the assignment margin, so fold this
                    // meeting's centroid into the matched identity to strengthen it.
                    // refine is idempotent per (identity, meeting), so re-running on a
                    // later reprocess replaces rather than double-counts. Best-effort:
                    // a refine failure must not block applying the name.
                    if let Some((centroid, window_count)) = by_label.get(m.query_label.as_str()) {
                        if let Err(e) = store
                            .refine(
                                m.identity_id,
                                centroid,
                                *window_count,
                                model_id,
                                meeting_id,
                                &m.query_label,
                            )
                            .await
                        {
                            tracing::warn!(
                                target: "orchestrator",
                                meeting_id = %meeting_id.0,
                                identity_id = %m.identity_id.0,
                                error = %e,
                                "auto-accept refinement failed (best-effort; name still applied)"
                            );
                        }
                    }
                    accepted.insert(m.query_label, m.display_name);
                }
                MatchBand::Uncertain => {
                    uncertain_out.push(minutist_common::VoiceprintSuggestion {
                        label: m.query_label,
                        display_name: m.display_name,
                        identity_id: m.identity_id,
                        model_id: model_id.to_string(),
                        similarity: m.similarity,
                    });
                }
                MatchBand::Reject => {} // nothing to do
            }
        }

        if !accepted.is_empty() {
            tracing::info!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                accepted_count = accepted.len(),
                uncertain_count = uncertain_out.len(),
                "voiceprint matching applied accepted names"
            );
        }

        Ok((accepted, uncertain_out))
    }

    /// Restore speaker names from the pre-reprocess snapshot after a fresh diarize
    /// has re-lettered the clusters (§2.6 clear-then-restore-matched).
    ///
    /// Attempts two matching strategies for each fresh label, in order:
    /// 1. **Centroid matching** (requires the embedding model): extracts centroids
    ///    for both the fresh clusters and the old named labels from the same meeting,
    ///    then runs `assign_identities` to find accept-band matches.
    /// 2. **Timeline coherence fallback**: computes Jaccard temporal overlap between
    ///    each fresh label's total speech span and each old label's total speech
    ///    span. If the overlap clears [`TIMELINE_JACCARD_THRESHOLD`], the name is
    ///    restored — this makes the re-map testable without a real embedding model
    ///    and guards against mis-transfer when segment boundaries shift across
    ///    reprocess runs.
    ///
    /// Only accept-band centroid matches or timeline-coherent pairs produce a
    /// name transfer. Ambiguous pairs (two old labels with equal best overlap for
    /// the same fresh label, or `< TIMELINE_JACCARD_THRESHOLD`) are skipped —
    /// the fresh label stays unlabelled. This is **clear-then-restore-matched**,
    /// not blanket retention.
    ///
    /// Best-effort: any failure (read, extractor, write) is logged but does not
    /// propagate — the meeting is left with cleared names, not errored.
    async fn apply_ephemeral_remap(
        &self,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
        old_names: std::collections::BTreeMap<String, String>,
        old_segments: Vec<Segment>,
        new_segments: &[Segment],
    ) {
        // Collect the distinct fresh labels.
        let new_labels: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            new_segments
                .iter()
                .filter_map(|s| s.speaker_id.clone())
                .filter(|l| seen.insert(l.clone()))
                .collect()
        };
        if new_labels.is_empty() || old_names.is_empty() {
            return;
        }

        // --- Strategy 1: centroid matching (requires embedding model) ----------
        let centroid_matches: std::collections::BTreeMap<String, String> =
            self.ephemeral_centroid_match(meeting_id, meeting_dir, &old_names, &old_segments, new_segments)
                .await
                .unwrap_or_default();

        // --- Strategy 2: timeline coherence fallback ---------------------------
        // For any fresh label not already covered by centroid matching, check
        // whether its total speech span overlaps an old label's span clearly enough
        // (Jaccard >= TIMELINE_JACCARD_THRESHOLD) to accept the name transfer.
        let mut restored: std::collections::BTreeMap<String, String> = centroid_matches;

        for new_label in &new_labels {
            if restored.contains_key(new_label.as_str()) {
                continue; // already matched by centroid
            }
            let new_span_ms = total_speech_ms(new_segments, new_label);
            if new_span_ms == 0 {
                continue;
            }
            let mut best: Option<(&str, f64)> = None;
            for (old_label, name) in &old_names {
                let old_span_ms = total_speech_ms(&old_segments, old_label);
                let jaccard = timeline_jaccard(new_segments, new_label, &old_segments, old_label);
                if old_span_ms == 0 || jaccard < TIMELINE_JACCARD_THRESHOLD {
                    continue;
                }
                match best {
                    None => best = Some((name.as_str(), jaccard)),
                    Some((_, prev_j)) => {
                        if (jaccard - prev_j).abs() < 1e-9 {
                            // Tie: two old labels overlap equally — ambiguous, skip.
                            best = None;
                            break;
                        } else if jaccard > prev_j {
                            best = Some((name.as_str(), jaccard));
                        }
                    }
                }
            }
            if let Some((name, jaccard)) = best {
                tracing::debug!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    new_label = %new_label,
                    name = %name,
                    jaccard,
                    "ephemeral re-map: timeline-coherence accepted"
                );
                restored.insert(new_label.clone(), name.to_string());
            }
        }

        if restored.is_empty() {
            tracing::debug!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                "ephemeral re-map: no names restored"
            );
            return;
        }

        // Write the restored names back into metadata.json (on top of the cleared map).
        let root = self.persistence_root.clone();
        let names_copy = restored.clone();
        let write_result = tokio::task::spawn_blocking(move || -> AppResult<()> {
            // Guarded RMW (issue 0025): serialises with the lifecycle subscriber.
            // speaker_names was cleared by finalise_diarization; restore matched ones.
            persistence::meeting_ops::update_metadata(&root, meeting_id, |meta| {
                for (label, name) in &names_copy {
                    meta.speaker_names.insert(label.clone(), name.clone());
                }
            })
        })
        .await;

        match write_result {
            Ok(Ok(())) => {
                tracing::info!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    restored_count = restored.len(),
                    "ephemeral re-map: restored names after reprocess"
                );
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "ephemeral re-map: metadata write failed"
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "ephemeral re-map: metadata write join failed"
                );
            }
        }
    }

    /// Attempt centroid-based name matching between fresh clusters and old named
    /// labels from the same meeting (§2.6 strategy 1).
    ///
    /// Extracts a centroid for each old named label (from the pre-reprocess
    /// transcript + audio) and for each fresh label (from the post-diarize
    /// transcript + the same audio), then runs `assign_identities` treating the
    /// old named centroids as the "gallery". Returns accept-band matches only.
    ///
    /// Returns an empty map — not an error — when the embedding model is absent
    /// or when either side has no extractable centroids.
    async fn ephemeral_centroid_match(
        &self,
        meeting_id: MeetingId,
        meeting_dir: &std::path::Path,
        old_names: &std::collections::BTreeMap<String, String>,
        old_segments: &[Segment],
        new_segments: &[Segment],
    ) -> AppResult<std::collections::BTreeMap<String, String>> {
        use crate::matcher::{assign_identities, MatchBand, QueryCluster};
        use persistence::StoredVoiceprint;

        // Resolve the embedding model path; skip gracefully if absent.
        let registry = Arc::clone(&self.model_registry);
        let emb_id = minutist_common::ModelId::from(runner::DIARIZE_EMB_MODEL_ID);
        let emb_dir_opt = {
            use minutist_common::ModelStatusState;
            registry
                .list_models()
                .into_iter()
                .find(|s| s.id == emb_id)
                .and_then(|s| match s.status {
                    ModelStatusState::Available { local_dir } => Some(local_dir),
                    _ => None,
                })
        };
        let emb_dir = match emb_dir_opt {
            Some(d) => d,
            None => {
                tracing::debug!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    "ephemeral_centroid_match: embedding model not available; skipping"
                );
                return Ok(Default::default());
            }
        };

        let meeting_dir_owned = meeting_dir.to_path_buf();
        let old_names_owned = old_names.clone();
        let old_segments_owned = old_segments.to_vec();
        let new_segments_owned = new_segments.to_vec();

        let result = tokio::task::spawn_blocking(move || -> AppResult<std::collections::BTreeMap<String, String>> {
            use runner::find_file_in_dir;
            use minutist_common::VoiceprintIdentityId;

            let emb_onnx = find_file_in_dir(
                std::path::Path::new(&emb_dir),
                |name| name.ends_with(".onnx"),
            )?;
            let extractor = diarizer::VoiceprintExtractor::open(&emb_onnx)?;
            let pcm = persistence::read_audio_pcm(&meeting_dir_owned)?;
            // ONE pause scan for the whole meeting (B2), shared across both loops
            // below (old labels, then fresh labels).
            let regions = runner::pause_excluding_segments(&pcm);

            // Build an ephemeral gallery from the old named labels.
            // Each old label becomes a `StoredVoiceprint` (with a placeholder
            // `VoiceprintIdentityId`) so `assign_identities` can score them.
            let mut gallery: Vec<StoredVoiceprint> = Vec::new();
            // Map from ephemeral identity id back to the display name.
            let mut id_to_name: std::collections::HashMap<VoiceprintIdentityId, String> = Default::default();

            for (old_label, name) in &old_names_owned {
                let clean_segs = clean_segments_for_label(&old_segments_owned, old_label);
                if clean_segs.is_empty() {
                    continue;
                }
                let windows = segment_windows(&pcm, &regions, &clean_segs);
                if windows.is_empty() {
                    continue;
                }
                match centroid_from_windows(&extractor, &windows, 16_000) {
                    Ok((vp, window_count)) => {
                        let eph_id = VoiceprintIdentityId::new();
                        let dim = vp.vector.len();
                        id_to_name.insert(eph_id, name.clone());
                        gallery.push(StoredVoiceprint {
                            identity_id: eph_id,
                            centroid_id: minutist_common::VoiceprintCentroidId::new(),
                            display_name: name.clone(),
                            embedding: vp.vector,
                            dim,
                            sample_count: window_count,
                            model_id: runner::DIARIZE_EMB_MODEL_ID.to_string(),
                            condition_label: None,
                        });
                    }
                    Err(e) => {
                        tracing::debug!(
                            target: "orchestrator",
                            old_label = %old_label,
                            error = %e,
                            "ephemeral_centroid_match: centroid extraction failed for old label"
                        );
                    }
                }
            }

            if gallery.is_empty() {
                return Ok(Default::default());
            }

            // Extract centroids for fresh labels.
            let mut queries: Vec<QueryCluster> = Vec::new();
            for label in new_segments_owned.iter().filter_map(|s| s.speaker_id.as_ref()).collect::<std::collections::HashSet<_>>() {
                let clean_segs = clean_segments_for_label(&new_segments_owned, label);
                if clean_segs.is_empty() {
                    continue;
                }
                let windows = segment_windows(&pcm, &regions, &clean_segs);
                if windows.is_empty() {
                    continue;
                }
                match centroid_from_windows(&extractor, &windows, 16_000) {
                    Ok((vp, window_count)) => queries.push(QueryCluster {
                        label: label.clone(),
                        centroid: vp.vector,
                        window_count,
                    }),
                    Err(e) => {
                        tracing::debug!(
                            target: "orchestrator",
                            label = %label,
                            error = %e,
                            "ephemeral_centroid_match: centroid extraction failed for fresh label"
                        );
                    }
                }
            }

            if queries.is_empty() {
                return Ok(Default::default());
            }

            let assignments = assign_identities(&queries, &gallery);
            let mut result: std::collections::BTreeMap<String, String> = Default::default();
            for m in assignments {
                if m.band == MatchBand::Accept {
                    if let Some(name) = id_to_name.get(&m.identity_id) {
                        result.insert(m.query_label, name.clone());
                    }
                }
            }
            Ok(result)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("ephemeral_centroid_match spawn_blocking join failed: {e}"),
        })?;

        result
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

/// Minimum Jaccard temporal overlap required between a fresh cluster's total
/// speech span and an old named cluster's total speech span for the
/// timeline-coherence fallback in `apply_ephemeral_remap` to accept the name
/// transfer. A value of 0.50 means the intersection must cover at least half the
/// union of the two spans.
///
/// **Placeholder — WU6 calibrates** this alongside the cosine thresholds once a
/// multi-session corpus is available. It is intentionally named as a constant
/// (not a magic literal) so WU6 can swap the value without changing call sites.
const TIMELINE_JACCARD_THRESHOLD: f64 = 0.50;

/// Total duration of non-overlapping speech attributed to `label` in `segments` (ms).
///
/// Sums the duration of every segment whose `speaker_id == label`. Does not
/// de-duplicate overlapping time ranges (the diarizer produces non-overlapping
/// segments per label in practice, so this is exact for the use case).
fn total_speech_ms(segments: &[Segment], label: &str) -> u64 {
    segments
        .iter()
        .filter(|s| s.speaker_id.as_deref() == Some(label))
        .map(|s| s.end_ms.saturating_sub(s.start_ms))
        .sum()
}

/// Jaccard temporal overlap between the speech spans of `new_label` (in
/// `new_segs`) and `old_label` (in `old_segs`).
///
/// Computes the intersection and union of the two sets of `[start_ms, end_ms)`
/// intervals (as continuous coverage), then returns `intersection / union`.
/// Returns `0.0` when either set is empty or the union is zero.
///
/// Uses a merge-then-scan approach: each set of intervals is sorted and merged
/// into non-overlapping coverage; intersection and union are derived from the
/// merged representations.
fn timeline_jaccard(new_segs: &[Segment], new_label: &str, old_segs: &[Segment], old_label: &str) -> f64 {
    fn merged_intervals(segs: &[Segment], label: &str) -> Vec<(u64, u64)> {
        let mut ivs: Vec<(u64, u64)> = segs
            .iter()
            .filter(|s| s.speaker_id.as_deref() == Some(label) && s.end_ms > s.start_ms)
            .map(|s| (s.start_ms, s.end_ms))
            .collect();
        ivs.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in ivs {
            match merged.last_mut() {
                Some((_, prev_hi)) if lo <= *prev_hi => {
                    if hi > *prev_hi {
                        *prev_hi = hi;
                    }
                }
                _ => merged.push((lo, hi)),
            }
        }
        merged
    }

    let new_ivs = merged_intervals(new_segs, new_label);
    let old_ivs = merged_intervals(old_segs, old_label);
    if new_ivs.is_empty() || old_ivs.is_empty() {
        return 0.0;
    }

    // Compute total coverage of each set.
    let new_total: u64 = new_ivs.iter().map(|(a, b)| b - a).sum();
    let old_total: u64 = old_ivs.iter().map(|(a, b)| b - a).sum();

    // Intersection: sorted two-pointer scan.
    let mut intersection: u64 = 0;
    let mut ni = 0;
    let mut oi = 0;
    while ni < new_ivs.len() && oi < old_ivs.len() {
        let (nlo, nhi) = new_ivs[ni];
        let (olo, ohi) = old_ivs[oi];
        let lo = nlo.max(olo);
        let hi = nhi.min(ohi);
        if hi > lo {
            intersection += hi - lo;
        }
        if nhi < ohi {
            ni += 1;
        } else {
            oi += 1;
        }
    }

    let union = new_total + old_total - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

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
/// Returns the (possibly split) segments with `speaker_id` overlaid, the distinct
/// speaker count, and any vetoed-cluster names `(letter, display_name)` that the
/// caller must write to `speaker_names` after `finalise_diarization` clears the map.
///
/// `prune_veto_extractor` is `Some` when the embedding model is locally available;
/// `gallery` is the active-model voiceprint library. When either is absent, the
/// veto pass is skipped (graceful degradation — the prune runs normally). When
/// both are present, a second `VoiceprintExtractor` pass runs over each low-share
/// candidate cluster's PCM windows (after `compute_turns`, before `diarize_split_merge`)
/// to build a centroid and match against the gallery via `matcher::assign_identities`.
/// Accept-band matches (with the query-side noise guard) produce veto verdicts that
/// are passed into `diarize_split_merge` so those clusters survive the prune.
///
/// The `job` carries either the production `SherpaDiarizer` (+ best-effort Qwen
/// backend) or stub-supplied turns + backend (the default-suite seam), so a
/// `SherpaDiarizer` and a model-free stub both drive the SAME split core.
///
/// VRAM sequencing (§2.5): the `SherpaDiarizer` holds the segmentation model; the
/// `prune_veto_extractor` holds the embedding model (same as the diarizer's internal
/// one, but a separate instance opened before the diarizer drop). The Qwen re-ASR
/// `backend` is the VRAM-heavy component; it is moved into `diarize_split_merge` and
/// dropped there. The extractor is a lightweight embedding-only model and is dropped
/// at the end of the veto pass, before `diarize_split_merge` starts — so the peak
/// VRAM budget is (sherpa-seg + sherpa-emb + extractor) for the veto pass, then
/// (sherpa-seg + sherpa-emb + Qwen) for the split — never all three simultaneously.
async fn run_diarization_blocking(
    meeting_dir: PathBuf,
    job: DiarizationJob,
    prune_veto_extractor: Option<diarizer::VoiceprintExtractor>,
    gallery: Vec<persistence::StoredVoiceprint>,
    event_tx: broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) -> AppResult<(Vec<Segment>, u32, Vec<(String, String)>)> {
    tokio::task::spawn_blocking(move || -> AppResult<(Vec<Segment>, u32, Vec<(String, String)>)> {
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

        // §2.5 prune-veto + #0023 library-informed merge: when the embedding model
        // is available, run a single extractor pass over all clusters to build
        // centroids and match against the gallery. The pass produces:
        // - veto_verdicts: low-share clusters that match an enrolled identity (rescued
        //   from the prune), passed to diarize_split_merge as veto_ids.
        // - merge_map: (source→canonical) pairs for clusters that both match the same
        //   enrolled identity; passed to overlay_speakers so the prune/cap sees the
        //   combined speech mass.
        let (veto_verdicts, merge_map): (Vec<(i32, String)>, Vec<(i32, i32)>) =
            if let Some(extractor) = prune_veto_extractor {
                compute_prune_veto_verdicts(
                    &turns, &pcm, &config, &extractor, &gallery, meeting_id,
                )
            } else {
                (Vec::new(), Vec::new())
            };
        // Drop the extractor before the split: its VRAM is freed before the Qwen
        // backend enters (§2.5 VRAM sequencing note above).
        // (prune_veto_extractor is moved into `extractor` above or already None)

        // `Box<dyn AsrBackend + Send>` → `Box<dyn AsrBackend>` for the core (the
        // split runs on this one thread; the `Send` bound is only needed to move
        // the backend into the closure).
        let backend = backend.map(|b| b as Box<dyn minutist_common::AsrBackend>);
        let ctx = SplitMergeContext {
            turns: &turns,
            pcm: &pcm,
            event_tx: &event_tx,
            meeting_id,
        };
        diarize_split_merge(&ctx, segments, backend, &config, &veto_verdicts, &merge_map)
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

/// Read-only per-meeting context shared by [`diarize_split_merge`] and
/// [`split_mixed_qwen_segment`]: the pause-INCLUDING [`SpeakerTurn`]s + decoded
/// PCM, the event bus, and the meeting id. Bundled into one struct rather than
/// four parallel parameters so a future meeting-wide input (like the
/// per-segment cache work in B2) does not grow either function's argument
/// list further.
struct SplitMergeContext<'a> {
    /// Raw [`SpeakerTurn`]s from `compute_turns`, on the pause-INCLUDING clock
    /// `pcm` shares.
    turns: &'a [SpeakerTurn],
    /// The pause-INCLUDING decoded audio.
    pcm: &'a [f32],
    event_tx: &'a broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
}

/// Re-ASR split core (#0015 phase 4) — the BLOCKING, model-free-testable funnel.
///
/// A free fn taking EXPLICIT params (mirroring [`transcribe_pcm_window_blocking`]
/// rather than dispatching through the `common::Diarizer` trait) so the default
/// suite can drive the whole split with a stub-supplied `turns` + stub `AsrBackend` —
/// no `SherpaDiarizer`, no Qwen GGUF:
/// - `ctx.turns` are the raw [`SpeakerTurn`]s from `compute_turns`, on the
///   pause-INCLUDING clock `ctx.pcm` shares.
/// - `segments` is the ASR transcript (pause-EXCLUDING `start_ms`).
/// - `ctx.pcm` is the pause-INCLUDING decoded audio.
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
///    [`runner::excluding_range_to_pcm_slice`] against one meeting-wide
///    [`runner::pause_excluding_segments`] scan), energy-snap each cut, slice the
///    PCM, re-ASR each single-speaker sub-clip, letter it from the map by its
///    dominant [`SpeakerTurn`] cluster, and stamp its `start_ms` on the EXCLUDING
///    clock via [`runner::excluding_ms_for_pcm_sample_in_regions`]. Keep-whole if the cuts
///    are empty, any snap returns `None`, or `backend` is `None`.
/// 4. Re-run [`diarizer::merge_adjacent_speakers`] (the split may have produced
///    adjacent same-letter sub-clips across segments) and recompute the count.
///
/// The clock discipline is the #1 blocking fix: turn cuts are taken on the
/// pause-INCLUDING clock the turns + PCM share, and a sub-clip's `start_ms` is
/// mapped back to the EXCLUDING transcript clock by the inverse. INCLUDING-clock
/// turns are NEVER compared against EXCLUDING-clock segment bounds.
/// `veto_verdicts` is a list of `(cluster_id, display_name)` pairs for low-share
/// clusters the orchestrator has vetoed from pruning (§2.5 prune-veto). Each
/// cluster in the list matched an enrolled voiceprint above `T_ACCEPT` (with the
/// query-side noise guard) and must survive the share prune + cap. After
/// `overlay_speakers` assigns letters, the cluster→letter map is used to convert
/// each vetoed cluster id to its final letter and collect `(letter, name)` pairs
/// returned as the third tuple element so the caller can write them to
/// `speaker_names` after `finalise_diarization` clears the map.
///
/// An empty `veto_verdicts` is the no-veto baseline (the prune runs normally).
///
/// `merge_map` is a slice of `(source, canonical)` pairs from the library-informed
/// merge pass (#0023). An empty slice is the no-merge baseline (bit-identical to
/// the pre-merge behaviour).
fn diarize_split_merge(
    ctx: &SplitMergeContext<'_>,
    segments: Vec<Segment>,
    mut backend: Option<Box<dyn minutist_common::AsrBackend>>,
    config: &diarizer::DiarizerConfig,
    veto_verdicts: &[(i32, String)],
    merge_map: &[(i32, i32)],
) -> AppResult<(Vec<Segment>, u32, Vec<(String, String)>)> {
    // Extract cluster ids for the veto; names are resolved below once the
    // cluster→letter map is available.
    let veto_ids: Vec<i32> = veto_verdicts.iter().map(|(id, _)| *id).collect();

    // 1. Overlay labels + flag mixed Qwen segments; keep the cluster→letter map.
    // Pass veto_ids (enrolled cluster rescue) and merge_map (same-identity cluster
    // unification, #0023) — applied together in a single overlay pass.
    let (mut segments, _count, cluster_letters) =
        diarizer::overlay_speakers(ctx.turns, segments, config, &veto_ids, merge_map);

    // 2. Collapse fragments so a turn reads as one row (#0015 phase 1).
    diarizer::merge_adjacent_speakers(&mut segments, MERGE_GAP_MS);

    // 3. Re-ASR split, only when a backend is present. The pause-excluding kept
    // regions are computed ONCE for the whole meeting here — `pause_excluding_segments`
    // is an O(samples) scan — rather than once per segment (or per interior cut)
    // inside `split_mixed_qwen_segment`, which is what made this pass
    // O(segments × cuts × samples) for a long recording (B2).
    if backend.is_some() {
        let regions = runner::pause_excluding_segments(ctx.pcm);
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
                ctx,
                &seg,
                &regions,
                backend.as_deref_mut().expect("backend present in this branch"),
                &cluster_letters,
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

    // 5. Resolve vetoed cluster ids to their assigned letters so the caller can
    // write the pre-matched names to speaker_names after finalise_diarization clears
    // the map. Clusters that did not survive (e.g. were absent from the turns and
    // got no segments) are silently dropped — only present letters carry names.
    let veto_names: Vec<(String, String)> = veto_verdicts
        .iter()
        .filter_map(|(cluster_id, name)| {
            cluster_letters
                .iter()
                .find(|(cid, _)| *cid == *cluster_id)
                .map(|(_, letter)| (letter.clone(), name.clone()))
        })
        .collect();

    Ok((segments, count, veto_names))
}

/// Minimum "clean" segment duration for voiceprint extraction (§2.3.1): long
/// enough that a single short interjection does not distort a speaker's
/// average embedding. Shared by every "segments → PCM windows → centroid"
/// extraction site: enrolment, cross-meeting matching, and ephemeral
/// name-carry matching.
const MIN_CLEAN_VOICEPRINT_DURATION_MS: u64 = 1000;

/// Segments belonging to `label`: excludes mixed (non-empty `shared_speakers`)
/// segments and any shorter than [`MIN_CLEAN_VOICEPRINT_DURATION_MS`] (the
/// §2.3.1 cleanliness filter every voiceprint-extraction call site applies
/// before mapping segments to PCM windows).
fn clean_segments_for_label<'a>(segments: &'a [Segment], label: &str) -> Vec<&'a Segment> {
    segments
        .iter()
        .filter(|seg| {
            seg.speaker_id.as_deref() == Some(label)
                && seg.shared_speakers.is_empty()
                && seg.end_ms.saturating_sub(seg.start_ms) >= MIN_CLEAN_VOICEPRINT_DURATION_MS
        })
        .collect()
}

/// Map `segs` through the pause-excl → incl clock mapper against the caller's
/// precomputed `regions`, collecting the non-empty PCM windows.
///
/// `regions` is one [`runner::pause_excluding_segments`] scan for the whole
/// meeting, computed ONCE by the caller and shared across every label's
/// mapping — the O(samples) scan must not re-run per segment (B2). `segs`
/// should already be filtered to one label/cleanliness criterion (see
/// [`clean_segments_for_label`]); a segment the mapper cannot resolve (out of
/// range) is silently skipped, matching
/// [`runner::excluding_range_to_pcm_slice`]'s documented behaviour.
fn segment_windows(pcm: &[f32], regions: &[runner::KeptRegion], segs: &[&Segment]) -> Vec<Vec<f32>> {
    segs.iter()
        .filter_map(|seg| {
            let range = runner::excluding_range_to_pcm_slice(regions, seg.start_ms, seg.end_ms)?;
            let window = pcm[range].to_vec();
            (!window.is_empty()).then_some(window)
        })
        .collect()
}

/// Build a speaker centroid from non-empty PCM `windows`.
///
/// The common tail of every "segments/turns → PCM windows → centroid"
/// extraction site (voiceprint enrolment, cross-meeting matching, ephemeral
/// name-carry matching, and the library-merge prune-veto pass) — those sites
/// differ only in how `windows` is gathered and in what they do with an
/// extraction `Err` (propagate vs. log-and-skip), both left to the caller.
/// The caller checks `windows.is_empty()` itself: the two possible "no usable
/// audio" cases (no clean segments vs. every clean segment mapping to nothing)
/// log distinct messages at several call sites.
fn centroid_from_windows(
    extractor: &diarizer::VoiceprintExtractor,
    windows: &[Vec<f32>],
    sample_rate: u32,
) -> AppResult<(diarizer::Voiceprint, u64)> {
    let window_count = windows.len() as u64;
    let window_refs: Vec<&[f32]> = windows.iter().map(|w| w.as_slice()).collect();
    let centroid = extractor.centroid(&window_refs, sample_rate)?;
    Ok((centroid, window_count))
}

/// Second embedding pass: compute prune-veto verdicts AND the library-informed
/// merge map in a single extractor pass.
///
/// Embeds every cluster from `turns` (not only low-share candidates) so that
/// `match_each_cluster` can detect when two clusters match the same enrolled
/// identity. The single extractor pass is reused for both outputs:
///
/// - **Veto verdicts** `Vec<(i32, String)>`: clusters that ARE low-share (would
///   be pruned) AND match an enrolled identity above the accept threshold. These
///   are returned as `(cluster_id, display_name)` pairs and forwarded to
///   `overlay_speakers` as `veto_ids`.
/// - **Merge map** `Vec<(i32, i32)>`: `(source, canonical)` pairs from the
///   library-informed merge pass (issue #0023). For each enrolled identity matched
///   by ≥2 clusters, the group merges: canonical = the member with the greatest
///   speech mass (tie-break: lowest cluster id). Non-canonical members become
///   sources. The invariant that only same-identity clusters share a canonical is
///   enforced here: groups are keyed by `identity_id`, so two different identities
///   can never share a canonical.
///
/// `veto_ids` and `merge_map` are derived from the SAME per-cluster match results;
/// no third extractor pass is added.
///
/// When the prune is not active, no cluster is at risk of pruning — veto verdicts
/// are skipped, but the merge pass still runs (a dominant cluster and another
/// cluster can both match the same identity even without a prune floor).
///
/// Errors in individual cluster embeddings are logged and skipped (best-effort).
fn compute_prune_veto_verdicts(
    turns: &[SpeakerTurn],
    pcm: &[f32],
    config: &diarizer::DiarizerConfig,
    extractor: &diarizer::VoiceprintExtractor,
    gallery: &[persistence::StoredVoiceprint],
    meeting_id: MeetingId,
) -> (Vec<(i32, String)>, Vec<(i32, i32)>) {
    use crate::matcher::{match_each_cluster, QueryCluster};

    if gallery.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Tally per-cluster total turn duration + turn count (pause-INCLUDING ms).
    // All clusters are tallied — not only low-share ones — because the merge pass
    // needs centroids for every cluster that might share an identity.
    let mut cluster_ids: Vec<i32> = Vec::new();
    let mut cluster_dur: Vec<u64> = Vec::new();
    let mut cluster_turn_count: Vec<usize> = Vec::new();
    for t in turns {
        let dur = t.end_ms.saturating_sub(t.start_ms);
        match cluster_ids.iter().position(|&id| id == t.cluster) {
            Some(i) => {
                cluster_dur[i] += dur;
                cluster_turn_count[i] += 1;
            }
            None => {
                cluster_ids.push(t.cluster);
                cluster_dur.push(dur);
                cluster_turn_count.push(1);
            }
        }
    }

    if cluster_ids.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let total_dur: u64 = cluster_dur.iter().sum();

    // Identify which clusters are low-share (would be pruned without a veto).
    let prune_active = config.min_cluster_share > 0.0 || config.min_cluster_segments > 0;
    let is_low_share: Vec<bool> = if prune_active {
        cluster_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let share = if total_dur > 0 {
                    cluster_dur[i] as f32 / total_dur as f32
                } else {
                    0.0
                };
                let below_share =
                    config.min_cluster_share > 0.0 && share < config.min_cluster_share;
                let below_count = config.min_cluster_segments > 0
                    && cluster_turn_count[i] < config.min_cluster_segments;
                below_share || below_count
            })
            .collect()
    } else {
        vec![false; cluster_ids.len()]
    };

    tracing::debug!(
        target: "orchestrator",
        meeting_id = %meeting_id.0,
        total_clusters = cluster_ids.len(),
        low_share_count = is_low_share.iter().filter(|&&b| b).count(),
        "library-merge/prune-veto: running embedding pass for all clusters"
    );

    // Embed every cluster (not only low-share ones) in one extractor pass.
    // Turns are on the pause-INCLUDING clock; PCM is pause-INCLUDING — no clock mapper.
    const SR: u32 = 16_000;
    let mut queries: Vec<QueryCluster> = Vec::new();

    for &cluster_id in &cluster_ids {
        let windows: Vec<Vec<f32>> = turns
            .iter()
            .filter(|t| t.cluster == cluster_id)
            .filter_map(|t| {
                let start_sample = (t.start_ms as usize * SR as usize) / 1000;
                let end_sample = (t.end_ms as usize * SR as usize) / 1000;
                if start_sample >= end_sample || end_sample > pcm.len() {
                    return None;
                }
                let window = pcm[start_sample..end_sample].to_vec();
                if window.is_empty() {
                    None
                } else {
                    Some(window)
                }
            })
            .collect();

        if windows.is_empty() {
            tracing::debug!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                cluster_id,
                "library-merge/prune-veto: no usable PCM windows for cluster; skipping"
            );
            continue;
        }

        // The noise guard threshold (T_ACCEPT_NOISY vs T_ACCEPT) depends on
        // window_count; it is applied inside match_each_cluster via the
        // per-cluster QueryCluster below.
        let (centroid, window_count) = match centroid_from_windows(extractor, &windows, SR) {
            Ok(pair) => pair,
            Err(e) => {
                tracing::debug!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    cluster_id,
                    error = %e,
                    "library-merge/prune-veto: centroid extraction failed; skipping"
                );
                continue;
            }
        };

        queries.push(QueryCluster {
            label: cluster_id.to_string(),
            centroid: centroid.vector,
            window_count,
        });
    }

    if queries.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Per-cluster best-identity match (collisions allowed) for the merge pass.
    let per_cluster = match_each_cluster(&queries, gallery);

    // -------------------------------------------------------------------
    // Derive veto verdicts: low-share clusters with an accept-band match.
    // -------------------------------------------------------------------
    // `is_low_share` is indexed by cluster_ids; look up by cluster_id.
    let veto_verdicts: Vec<(i32, String)> = per_cluster
        .iter()
        .filter_map(|(label, matched)| {
            let (identity_id, _sim) = matched.as_ref()?;
            let cluster_id: i32 = label.parse().ok()?;
            // Only veto if this cluster is low-share (would be pruned).
            let idx = cluster_ids.iter().position(|&id| id == cluster_id)?;
            if !is_low_share[idx] {
                return None;
            }
            // Resolve display name from gallery.
            let display_name = gallery
                .iter()
                .find(|e| e.identity_id == *identity_id)
                .map(|e| e.display_name.clone())?;
            Some((cluster_id, display_name))
        })
        .collect();

    if !veto_verdicts.is_empty() {
        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            vetoed_count = veto_verdicts.len(),
            "prune-veto: enrolled quiet speakers rescued from pruning"
        );
    }

    // -------------------------------------------------------------------
    // Derive merge map: group clusters by matched identity; emit
    // (source, canonical) pairs for non-canonical members of each group.
    //
    // Canonical = the group member with the greatest speech mass
    // (turn-duration sum from cluster_dur); tie-break = lowest cluster id.
    // Invariant: groups are keyed by identity_id, so only same-identity
    // clusters are ever merged.
    // -------------------------------------------------------------------
    use minutist_common::VoiceprintIdentityId;
    // Build identity → Vec<cluster_id> groups from accept-band matches.
    let mut identity_groups: Vec<(VoiceprintIdentityId, Vec<i32>)> = Vec::new();
    for (label, matched) in &per_cluster {
        let Some((identity_id, _)) = matched else { continue };
        let cluster_id: i32 = match label.parse() {
            Ok(id) => id,
            Err(_) => continue,
        };
        match identity_groups.iter_mut().find(|(id, _)| id == identity_id) {
            Some((_, members)) => members.push(cluster_id),
            None => identity_groups.push((*identity_id, vec![cluster_id])),
        }
    }

    let mut merge_map: Vec<(i32, i32)> = Vec::new();
    for (_identity_id, members) in &identity_groups {
        if members.len() < 2 {
            // Single cluster matched this identity — nothing to merge.
            continue;
        }
        // Canonical = largest speech mass; tie-break = lowest cluster id.
        let canonical = *members
            .iter()
            .max_by(|&&a, &&b| {
                let dur_a = cluster_ids
                    .iter()
                    .position(|&id| id == a)
                    .map(|i| cluster_dur[i])
                    .unwrap_or(0);
                let dur_b = cluster_ids
                    .iter()
                    .position(|&id| id == b)
                    .map(|i| cluster_dur[i])
                    .unwrap_or(0);
                dur_a.cmp(&dur_b).then(b.cmp(&a)) // tie: lower id wins (b.cmp(&a) = a < b)
            })
            .expect("members is non-empty");
        for &source in members {
            if source != canonical {
                merge_map.push((source, canonical));
            }
        }
    }

    if !merge_map.is_empty() {
        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            merge_count = merge_map.len(),
            "library-merge: merging same-identity diarizer clusters"
        );
    }

    (veto_verdicts, merge_map)
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
    ctx: &SplitMergeContext<'_>,
    seg: &Segment,
    regions: &[runner::KeptRegion],
    backend: &mut dyn minutist_common::AsrBackend,
    cluster_letters: &[(i32, String)],
) -> AppResult<Option<Vec<Segment>>> {
    let turns = ctx.turns;
    let pcm = ctx.pcm;
    // Map the segment's pause-EXCLUDING [start_ms, end_ms) to the single
    // pause-INCLUDING PCM range the turns share (the clamp matches the offline
    // pause model — a mixed Qwen segment never straddles a ≥4 s pause). `regions`
    // is the caller's ONE pause scan for the whole meeting (B2) — this is the
    // cheap per-segment half, [`runner::excluding_range_to_pcm_slice`].
    let seg_range = match runner::excluding_range_to_pcm_slice(regions, seg.start_ms, seg.end_ms) {
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
        let excl_start_ms = runner::excluding_ms_for_pcm_sample_in_regions(regions, lo);
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
            end_ms: runner::excluding_ms_for_pcm_sample_in_regions(regions, hi),
            text,
            speaker_id: letter,
            confidence: seg.confidence,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        };
        let _ = ctx.event_tx.send(AppEvent::TranscriptSegment {
            meeting_id: ctx.meeting_id,
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

/// Load the voiceprint gallery for the §2.5 prune-veto pass.
///
/// Calls `VoiceprintStore::all(model_id)` and returns the result as a plain Vec
/// so the spawn_blocking thread can own it without a DB handle. Returns an empty
/// Vec when the store is `None` (enrolment not enabled or store not open) or on
/// any read error (best-effort — a gallery read failure degrades to no-veto, not
/// a hard error).
async fn load_voiceprint_gallery(
    store: Option<&persistence::VoiceprintStore>,
    meeting_id: MeetingId,
) -> Vec<persistence::StoredVoiceprint> {
    let Some(store) = store else {
        return Vec::new();
    };
    match store.all(runner::DIARIZE_EMB_MODEL_ID).await {
        Ok(gallery) => gallery,
        Err(e) => {
            tracing::warn!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                error = %e,
                "prune-veto: gallery load failed (best-effort); skipping veto pass"
            );
            Vec::new()
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

        // §2.6 snapshot: capture old speaker_names + transcript BEFORE the
        // re-transcribe overwrites transcript.json. Mirrors `reprocess_claimed`.
        let pre_snapshot = self.capture_reprocess_snapshot(meeting_id, &meeting_dir).await;

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
        // Stub path (test-only): no gallery for the veto pass.
        self.rediarize_inner_with_snapshot_and_gallery(
            index,
            meeting_id,
            DiarizationJob::Stub {
                turns,
                backend: split_backend,
                config,
            },
            pre_snapshot,
            Vec::new(),
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
