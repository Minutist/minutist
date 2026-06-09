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
use std::sync::Arc;
use std::time::Duration;

use audio_capture::AudioCaptureManager;
use chrono::{DateTime, Utc};
use diarizer::OnlineDiarizer;
use meeting_app_common::{
    AppError, AppEvent, AppResult, AudioFormat, Diarizer, MeetingId, MeetingListEntry, MeetingMeta,
    ModelDescriptor, ModelId, ModelStatus, RecordingState, Segment,
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

        // GPU offload is a runtime decision: only when BOTH the build has a GPU
        // feature AND the `gpu_acceleration` setting is on (see
        // `architecture/cross-cutting.md` — "GPU portability"). `resolve_gpu_layers`
        // returns the compile-time ceiling when on, `0` (force CPU) when off.
        let n_gpu_layers = runner::resolve_gpu_layers(self.settings.current().gpu_acceleration);

        // Resolve the ASR language hint from the `transcription_language`
        // setting (see `runner::resolve_transcription_language`): a full English
        // name forces that language via the assistant-turn prefix; the `"auto"`
        // sentinel resolves to `None` (auto-detect, byte-identical to the
        // pre-feature behaviour).
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);

        // Hybrid ASR (Phase 8): pick the engine from the transcription-language
        // setting — Parakeet for the languages it covers, else a Qwen tier
        // (1.7B when `prefer_large_asr_model` is set, else 0.6B). The `language`
        // hint above only affects the Qwen tiers. See
        // `common::asr_engine_for_language`.
        let engine = meeting_app_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            self.settings.current().prefer_large_asr_model,
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
            speaker_names: std::collections::BTreeMap::new(),
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

        // Decode audio + run VAD + ASR on a blocking thread. Build the ASR
        // backend inside the blocking closure so the heavy model load is off the
        // async worker threads. The model is resolved via the same registry path
        // the live pipeline uses.
        let registry = Arc::clone(&self.model_registry);
        let event_tx = self.event_tx.clone();
        let meeting_dir_for_blocking = meeting_dir.clone();
        // Resolve the runtime GPU-offload count from the `gpu_acceleration`
        // setting before entering the blocking closure (it cannot read
        // `self.settings`). The offline re-transcribe honours the same GPU
        // toggle as the live path.
        let n_gpu_layers = runner::resolve_gpu_layers(self.settings.current().gpu_acceleration);
        // Resolve the ASR language hint before entering the blocking closure (it
        // cannot read `self.settings`). The offline re-transcribe honours the
        // same `transcription_language` setting as the live path.
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        // Hybrid ASR (Phase 8): same engine routing as the live path, so a
        // re-transcribe of an English/EU meeting uses Parakeet (timestamps) and
        // others use the resolved Qwen tier.
        let engine = meeting_app_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            self.settings.current().prefer_large_asr_model,
        );
        let missing_model_id = runner::engine_model_id(engine);

        // Bound the (uninterruptible) offline ASR run with a length-relative
        // timeout sized for ASR (slower than diarization), mirroring the
        // diarization timeout in `rediarize_inner`: a wedged or pathologically
        // slow re-transcribe must not hold the offline claim — and thereby block
        // the next recording — without bound. On timeout we return before any
        // transcript write; the abandoned `spawn_blocking` thread's result is
        // discarded (tokio cannot cancel it).
        let duration_dir = meeting_dir.clone();
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

        self.finalise_retranscribe(index, meeting_id, &meeting_dir, segments)
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
        {
            let mut guard = self.inner.lock().await;
            transition_offline_claim(&mut guard.state, meeting_id)?;
        }
        // Broadcast the busy state (Offline maps to the public `Finalising`) so
        // the webview gates Start while the offline op holds the claim, rather
        // than enabling Start into an `InvalidInput` failure.
        self.emit(AppEvent::StateChanged {
            state: RecordingState::Finalising { meeting_id },
        });
        Ok(())
    }

    /// Release an offline claim, returning the recorder to `Idle`.
    ///
    /// Called on every exit path of an offline op (success and error) so a
    /// failed op never wedges the recorder out of `Idle`.
    async fn release_offline(&self) {
        {
            let mut guard = self.inner.lock().await;
            transition_offline_release(&mut guard.state);
        }
        // Return the public state to Idle so the UI re-enables Start.
        self.emit(AppEvent::StateChanged {
            state: RecordingState::Idle,
        });
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

        // Announce the refreshed transcript so the webview re-reads it (the
        // meeting-list excerpt + any open-meeting view), mirroring how
        // `finalise_diarization` emits `DiarizationComplete`. Without this, a
        // background re-transcribe with diarization OFF (the default) would leave
        // the UI showing the stale/truncated transcript until a manual refresh.
        self.emit(AppEvent::TranscriptReady { meeting_id });

        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            segments = segments.len(),
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

        // Resolve GPU layers + engine + language hint before entering the
        // blocking closure (it cannot read `self.settings`), mirroring
        // `re_transcribe_claimed`. An explicit caller-supplied `language`
        // overrides the setting-derived hint (the agent may force a re-listen in
        // a known language); the setting-derived engine routing is unchanged.
        let n_gpu_layers = runner::resolve_gpu_layers(self.settings.current().gpu_acceleration);
        let setting_language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        let effective_language = language.or(setting_language);
        let engine = meeting_app_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            self.settings.current().prefer_large_asr_model,
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

        // Read the recording length to size the diarization timeout (a small
        // metadata read on a blocking thread).
        let duration_dir = meeting_dir.clone();
        let duration_ms = tokio::task::spawn_blocking(move || {
            persistence::read_metadata(&duration_dir).map(|m| m.duration_ms)
        })
        .await
        .map_err(|e| AppError::Internal {
            context: format!("diarize metadata read join failed: {e}"),
        })??;
        let budget = diarize_timeout(duration_ms);

        // Bound the (uninterruptible) sherpa `compute`: a pathologically slow or
        // hung diarization on a long recording must not block forever (the
        // original on-stop hang). On timeout we return BEFORE
        // `finalise_diarization`, so nothing is written — the meeting is left
        // un-diarized and the abandoned blocking thread's result (if it ever
        // completes) is dropped. `tokio` cannot cancel a `spawn_blocking` thread,
        // so a true infinite hang leaks one thread until process exit; the budget
        // bounds the wait, not the thread.
        let (segments, speaker_count) = match tokio::time::timeout(
            budget,
            run_diarization_blocking(meeting_dir.clone(), diarizer),
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
    backend: &mut dyn meeting_app_common::AsrBackend,
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
    let chunk = meeting_app_common::AudioChunk {
        samples: pcm[range].to_vec(),
        sample_rate: 16_000,
        start_ms,
        end_ms: start_ms + slice_len_ms,
    };

    backend.transcribe_chunk(&chunk)
}

/// Length-relative timeout budget for a diarization pass.
///
/// The offline sherpa `compute` is a single uninterruptible FFI call with no
/// progress callback, so a true per-progress watchdog isn't available at that
/// boundary; instead we bound it by wall-clock relative to the recording
/// length: ≈1× real-time, floored at `FLOOR_SECS` (so short meetings still get
/// a sane minimum) and capped at `CAP_SECS` (so a hang on a long recording
/// can't hold the offline claim — and thereby block starting a new recording —
/// for too long). A normal diarization runs well under real-time, so this only
/// fires on a pathologically slow or wedged pass; because visibility is
/// decoupled the meeting is already indexed, so a fired timeout merely leaves it
/// un-diarized for a manual re-diarize.
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

        let n_gpu_layers = runner::resolve_gpu_layers(self.settings.current().gpu_acceleration);
        // Resolve the ASR language hint, exactly as the production `start()`
        // path (this test-source path is production-equivalent).
        let language =
            runner::resolve_transcription_language(&self.settings.current().transcription_language);
        // Hybrid ASR (Phase 8): same engine routing as the production `start()`.
        let engine = meeting_app_common::asr_engine_for_language(
            &self.settings.current().transcription_language,
            self.settings.current().prefer_large_asr_model,
        );
        // Phase B: build the live diarizer (gated on diarization_enabled +
        // local model availability), exactly as the production `start()` path.
        let online_diarizer =
            self.build_live_diarizer(self.settings.current().diarization_enabled).await;
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
        backend: Box<dyn meeting_app_common::AsrBackend + Send>,
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
        backend: Box<dyn meeting_app_common::AsrBackend + Send>,
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
        backend: Box<dyn meeting_app_common::AsrBackend + Send>,
    ) -> AppResult<()> {
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
        // Same atomic claim/release as the production path (TIMELINE-DRIFT #5).
        self.claim_offline(meeting_id).await?;
        let result = self.rediarize_inner(index, meeting_id, diarizer).await;
        self.release_offline().await;
        result
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
        backend: Box<dyn meeting_app_common::AsrBackend + Send>,
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
