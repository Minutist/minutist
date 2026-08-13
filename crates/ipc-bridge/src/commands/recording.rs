//! Recording lifecycle commands: start / pause / resume / stop, prewarm, and current-state read.
use super::*;


/// Create a "New meeting" prep draft: a real, durable meeting folder
/// (`metadata.json` + `notes.ydoc`) with no audio yet, so a title, notes, and
/// attachments can be added before recording ever starts.
///
/// Routes DIRECTLY to `persistence::writer::create_draft` (no orchestrator —
/// the meeting has no live recording state until `start_recording` promotes
/// it), mirroring `add_attachment`'s direct-to-persistence routing.
#[tauri::command]
#[specta::specta]
pub async fn create_meeting(state: State<'_, IpcState>) -> AppResult<MeetingId> {
    let meetings_dir = state.meetings_dir.clone();
    let meeting_id = MeetingId::new();
    tokio::task::spawn_blocking(move || persistence::writer::create_draft(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("create_meeting join failed: {e}"),
        })??;

    // Index it now (best-effort) so it appears in `list_meetings` in this
    // session, mirroring `stop_recording`'s stop-time upsert — an absent row
    // self-heals via `reconcile_orphans` on the next `list_meetings` call.
    let meetings_dir = state.meetings_dir.clone();
    let entry = tokio::task::spawn_blocking(move || {
        let meta = persistence::read_metadata(&meetings_dir.join(meeting_id.0.to_string()))?;
        Ok::<_, AppError>(meeting_list_entry_for_meta(&meetings_dir, &meta))
    })
    .await;
    match entry {
        Ok(Ok(entry)) => {
            if let Err(e) = state.index.upsert(&entry).await {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "index upsert after create_meeting failed: {e}; will self-heal on next list"
                );
            }
        }
        Ok(Err(e)) => tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "building index entry after create_meeting failed: {e}; will self-heal on next list"
        ),
        Err(join_err) => tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "building index entry after create_meeting failed (join error): {join_err}; \
             will self-heal on next list"
        ),
    }

    Ok(meeting_id)
}

/// Start recording into an existing "New meeting" prep draft (created via
/// [`create_meeting`]).
///
/// `device_id = None` → use the device configured in settings, or the OS
/// default if none is configured.
///
/// Returns `meeting_id` on success.
#[tauri::command]
#[specta::specta]
pub async fn start_recording(
    meeting_id: MeetingId,
    device_id: Option<String>,
    state: State<'_, IpcState>,
) -> AppResult<MeetingId> {
    state.orchestrator.start(meeting_id, device_id).await
}

/// Pre-load the routed ASR model so the first record is not a cold ~29 s load
/// (live-test UX T2).
///
/// Routes to `Orchestrator::prewarm_asr`, which resolves the engine from the
/// `transcription_language` setting and builds the backend on `spawn_blocking`
/// into a process-held cache the first `start()` consumes. Idempotent and
/// non-blocking-at-start (no download; a not-yet-downloaded model warms nothing).
/// The webview calls this when the recording/meeting workspace opens so the model
/// is ready before the user presses Start. Returns `()` even on a build failure —
/// the lazy worker-init path remains the fallback, so prewarm is best-effort.
#[tauri::command]
#[specta::specta]
pub async fn prewarm_asr(state: State<'_, IpcState>) -> AppResult<()> {
    state.orchestrator.prewarm_asr().await;
    Ok(())
}

/// Pause the current recording.
#[tauri::command]
#[specta::specta]
pub async fn pause_recording(state: State<'_, IpcState>) -> AppResult<()> {
    state.orchestrator.pause().await
}

/// Resume after a pause.
#[tauri::command]
#[specta::specta]
pub async fn resume_recording(state: State<'_, IpcState>) -> AppResult<()> {
    state.orchestrator.resume().await
}

/// Set the meeting title WHILE actively recording. Held in the orchestrator's
/// in-progress state (not written to `metadata.json` immediately) and
/// consumed by `stop()`, which prefers it over whatever title is already on
/// disk (e.g. one set via `rename_meeting` during the "New meeting" prep
/// phase, before recording started); a no-op if `meeting_id` is not the
/// meeting currently recording/paused. Trimmed + capped so the UI cannot
/// persist an unbounded value (mirrors the speaker-name / collection-name
/// caps).
#[tauri::command]
#[specta::specta]
pub async fn set_recording_title(
    meeting_id: MeetingId,
    title: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    const MAX_TITLE_LEN: usize = 512;
    let title = title.trim().to_string();
    if title.chars().count() > MAX_TITLE_LEN {
        return Err(AppError::InvalidInput {
            context: format!("meeting title too long (max {MAX_TITLE_LEN} characters)"),
        });
    }
    state
        .orchestrator
        .set_pending_title(meeting_id, title)
        .await
}

/// Stop the current recording and finalise the meeting.
///
/// Returns the completed `MeetingMeta` on success.
///
/// After the orchestrator finalises the meeting folder, this also **upserts the
/// libsql index** (FR-33) so the just-recorded meeting appears in
/// `list_meetings` **in the same session** — without waiting for the next
/// app-start `rebuild_from_disk`. The orchestrator stays decoupled from the
/// index (it does not own one); the `ipc-bridge` owns the index handle in
/// `IpcState`, so the upsert lives here. See `architecture/components.md`,
/// `ipc-bridge` "Phase 4 additions — stop-upsert".
///
/// The blocking transcript read (for the list excerpt) runs on `spawn_blocking`
/// per the threading model; the async index `upsert` is awaited, never
/// `block_on`'d (the no-`block_on`-in-command-handlers rule).
///
/// An index-upsert failure is logged and swallowed — the recording itself is
/// safely persisted on disk and the index is a derived cache that the next
/// startup `rebuild_from_disk` will reconcile, so a failed upsert must not turn
/// a successful stop into an error.
#[tauri::command]
#[specta::specta]
pub async fn stop_recording(state: State<'_, IpcState>) -> AppResult<MeetingMeta> {
    let meta = state.orchestrator.stop().await?;

    // Build the meeting-list row for the freshly-stopped meeting. The excerpt is
    // the first transcript segment (if any), read on a blocking thread.
    let meetings_dir = state.meetings_dir.clone();
    let meta_for_entry = meta.clone();
    let entry = tokio::task::spawn_blocking(move || {
        meeting_list_entry_for_meta(&meetings_dir, &meta_for_entry)
    })
    .await;

    match entry {
        Ok(entry) => {
            if let Err(e) = state.index.upsert(&entry).await {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meta.uuid.0,
                    "index upsert after stop_recording failed: {e}; the recording is on disk \
                     and will be re-indexed on next startup"
                );
            }
        }
        Err(join_err) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meta.uuid.0,
                "building index entry after stop_recording failed (join error): {join_err}; \
                 the recording is on disk and will be re-indexed on next startup"
            );
        }
    }

    // Producer-gate S2 — capture-only delegation. When this device delegates
    // processing (a GPU-less / capture-only desktop), OFFER the meeting for an
    // eligible host to process rather than running the pipeline locally: mark it
    // `PendingProcessing` synchronously (before returning, so the next discovery
    // exchange advertises the offer) and skip the local post-stop passes. Default
    // OFF, gated by the `MINUTIST_DELEGATE_PROCESSING` env knob (a user-facing
    // Settings toggle + UI is the productisation follow-up). Only meaningful
    // alongside an eligible host running the election loop (S4) — on a lone device
    // the meeting simply waits as `PendingProcessing` until a host appears; the
    // audio is safely on disk regardless.
    if should_delegate_processing() {
        match persistence::meeting_ops::apply_processing_lifecycle(
            &state.meetings_dir,
            meta.uuid,
            minutist_common::ProcessingLifecycle::PendingProcessing,
        )
        .await
        {
            Ok(()) => {
                tracing::info!(
                    target: "ipc-bridge",
                    meeting_id = %meta.uuid.0,
                    "capture-only: offered the meeting for host processing (PendingProcessing); skipping local passes"
                );
                return Ok(meta);
            }
            // The mark failed, so the meeting is still `Local` — a state no election
            // host will ever claim (`election::claimable` is false for `Local`).
            // Returning here would strand it with no transcript repair / diarize /
            // summary and no automatic recovery, so fall THROUGH to the local
            // post-stop passes: a failed delegation degrades to processing here,
            // exactly as if delegation were off.
            Err(e) => tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meta.uuid.0,
                error = %e,
                "delegation mark failed; falling back to local post-stop passes"
            ),
        }
    }

    // Decoupled background post-processing: the meeting is already indexed and
    // visible, so any heavy passes run OFF the stop path (a slow/hung pass can
    // never wedge stop or hide the meeting). Up to two passes run, in order, in
    // one fire-and-forget task:
    //   1. Reprocess (FR — ASR repair + FR-11 diarize): pushed when the live
    //      transcript fell behind (drop-oldest loss during recording, or a
    //      stop-drain timeout) OR diarization is enabled. `Orchestrator::reprocess`
    //      takes ONE offline claim and re-runs ASR over the COMPLETE `audio.opus`
    //      — the authoritative transcript, since the audio is captured in full
    //      regardless of live-ASR speed — then (when diarization is on) diarizes
    //      the repaired transcript and emits `AppEvent::DiarizationComplete`. It
    //      carries its own length-relative timeout budget for the whole pass.
    // The reprocess pass claims the offline slot internally; errors are logged
    // (the recording is safely on disk). NOTE: `take_transcript_incomplete`
    // is consumed here, so a reprocess that fails or is skipped (recorder busy
    // with another op) is NOT auto-retried — the user-triggered reprocess action
    // is the recovery path. (Only the meeting INDEX self-heals, via
    // `reconcile_orphans` on `list_meetings`.)
    //
    //   3. Auto-summarise (#68): if `settings.auto_summarise_on_stop` is on (the
    //      default), run AFTER any reprocess so it summarises the FINAL
    //      transcript. Uses the held-summariser path ([`run_held_summarise`])
    //      and emits the determinate `OperationProgress` + `SummaryReady` events,
    //      exactly like the user-triggered `summarise_meeting`. Best-effort: an
    //      error is logged, the recording is on disk regardless.
    //
    // The gating/ordering ([`post_stop_passes`]) and per-pass error tolerance
    // ([`run_post_stop_passes`]) are extracted so they are unit-testable without
    // a Tauri runtime or a real orchestrator.
    let passes = post_stop_passes(
        state.orchestrator.take_transcript_incomplete(),
        state.orchestrator.diarization_enabled(),
        state.settings.current().auto_summarise_on_stop,
    );
    // Always spawn the post-stop task: it runs the gated/ordered passes (a no-op
    // when none apply) and then unconditionally indexes the final transcript into
    // meeting.db for retrieval.
    {
        let orchestrator = std::sync::Arc::clone(&state.orchestrator);
        let index = std::sync::Arc::clone(&state.index);
        let handles = state.chat_handles();
        let voiceprints = std::sync::Arc::clone(&state.voiceprints);
        let vp_enrolment_enabled = state.settings.current().voiceprint_enrolment_enabled;
        let meeting_id = meta.uuid;
        // If an auto-summary is planned, tell the webview NOW — before the
        // background task even starts — so the just-opened summary pane shows a
        // busy state for the whole queued → (reprocess) → summarising window
        // instead of the manual "Summarise" button (which invites a redundant,
        // racing second run). The terminal `SummaryReady` (success) or
        // `SummaryUnavailable` (deferred/failed, emitted in the Summarise arm
        // below) clears it; without the latter a deferred/failed auto-summary
        // would leave the pane spinning forever.
        if passes.contains(&PostStopPass::Summarise) {
            emit_summary_queued(&handles.event_tx, meeting_id);
        }
        tokio::spawn(async move {
            run_post_stop_passes(&passes, meeting_id, |pass| {
                let orchestrator = std::sync::Arc::clone(&orchestrator);
                let index = std::sync::Arc::clone(&index);
                let handles = handles.clone();
                let voiceprints = std::sync::Arc::clone(&voiceprints);
                async move {
                    match pass {
                        PostStopPass::Reprocess => {
                            orchestrator.reprocess(&index, meeting_id).await?;
                            // After a successful reprocess, apply voiceprint matches
                            // (best-effort — errors are logged, not propagated).
                            if vp_enrolment_enabled {
                                if let Some(store) = voiceprints.as_ref().as_ref() {
                                    if let Err(e) = orchestrator
                                        .apply_voiceprint_matches(meeting_id, store)
                                        .await
                                    {
                                        tracing::warn!(
                                            target: "ipc-bridge",
                                            meeting_id = %meeting_id.0,
                                            error = %e,
                                            "post-stop reprocess: voiceprint matching failed (best-effort)"
                                        );
                                    }
                                }
                            }
                            Ok(())
                        }
                        // The markdown result is discarded here (the summary is
                        // persisted + `SummaryReady` emitted inside
                        // `run_held_summarise`); only the `AppError` feeds the
                        // shared per-pass error logging.
                        //
                        // Unlike reprocess, this pass does NOT take the
                        // orchestrator's offline claim, so it cannot self-skip
                        // when a new recording preempts the slot. Gate it explicitly:
                        // if the user has started the next meeting, defer this
                        // meeting's auto-summarise (recoverable via the manual
                        // Summarise action) rather than contending with the live
                        // recording's GPU/LLM use.
                        PostStopPass::Summarise => {
                            if orchestrator.recorder_is_live().await {
                                // A new recording claimed the model; defer this
                                // meeting's auto-summary (the manual Summarise
                                // action is the recovery). Clear the queued busy
                                // state so the pane offers that action.
                                emit_summary_unavailable(&handles.event_tx, meeting_id);
                                Err(AppError::InvalidInput {
                                    context: "auto-summarise deferred: a new recording started"
                                        .into(),
                                })
                            } else {
                                match run_held_summarise(&handles, meeting_id).await {
                                    Ok(_) => Ok(()),
                                    Err(e) => {
                                        // Best-effort: a failed auto-summary wrote
                                        // no `summary.md`. Clear the queued busy
                                        // state (run_held_summarise emits no
                                        // terminal on the error path) so the pane
                                        // falls back to the manual action instead
                                        // of spinning forever.
                                        emit_summary_unavailable(&handles.event_tx, meeting_id);
                                        Err(AppError::from(e))
                                    }
                                }
                            }
                        }
                    }
                }
            })
            .await;

            // Always index the final transcript into meeting.db for retrieval
            // (best-effort; runs after any reprocess so it indexes the repaired
            // transcript). A failure leaves retrieval over this meeting incomplete,
            // never the meeting itself.
            if let Err(e) = crate::rag_index::index_transcript(&handles, meeting_id).await {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "post-stop transcript RAG-index failed (best-effort; rebuilt on next reprocess)"
                );
            }
        });
    }

    Ok(meta)
}

/// A background pass `stop_recording` may run after a stop, off the stop path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostStopPass {
    /// Re-run ASR over the complete `audio.opus` (the live transcript fell
    /// behind) and/or diarize, under one offline claim. Pushed when the live
    /// transcript is incomplete OR diarization is enabled; the merged
    /// `Orchestrator::reprocess` re-transcribes then diarizes the repaired
    /// transcript in a single pass.
    Reprocess,
    /// Auto-summarise the meeting (`settings.auto_summarise_on_stop`, #68). Runs
    /// LAST so it summarises the final (reprocessed) transcript.
    Summarise,
}

/// The ordered post-stop passes to run, derived from the three gating flags.
///
/// A single `Reprocess` pass is pushed when the transcript needs repair OR
/// diarization is enabled — `Orchestrator::reprocess` re-transcribes first so
/// diarization labels the **repaired** transcript rather than a truncated one.
/// Auto-summarise (#68) runs LAST so it summarises the final transcript after
/// any reprocess. An empty result means no background task is spawned. Pure +
/// unit-tested so the gating/ordering is verified without a Tauri runtime.
pub(crate) fn post_stop_passes(
    needs_retranscribe: bool,
    needs_diarize: bool,
    needs_summarise: bool,
) -> Vec<PostStopPass> {
    let mut passes = Vec::with_capacity(2);
    if needs_retranscribe || needs_diarize {
        passes.push(PostStopPass::Reprocess);
    }
    if needs_summarise {
        passes.push(PostStopPass::Summarise);
    }
    passes
}

/// Whether this device delegates processing to an eligible host (producer-gate S2)
/// instead of running the pipeline locally, from the `MINUTIST_DELEGATE_PROCESSING`
/// env knob. Default OFF. Read per `stop_recording` call (not cached at startup) so
/// an operator can toggle delegation without a restart — deliberately asymmetric
/// with S4's GPU `Capability`, which is probed once at sync-bind. (A user-facing
/// Settings toggle + UI is the productisation follow-up; the env knob is the v1
/// mechanism.)
fn should_delegate_processing() -> bool {
    is_delegate_value(std::env::var("MINUTIST_DELEGATE_PROCESSING").ok().as_deref())
}

/// Pure truthiness for the delegate knob — separated from the env read so the
/// gating is unit-testable without mutating the process environment. Trimmed and
/// case-insensitive; accepts `1` / `true` / `yes` / `on`.
pub(crate) fn is_delegate_value(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Run the planned post-stop `passes` in order, invoking `run_pass` for each.
///
/// Each pass's error is caught and logged — `AppError::InvalidInput` (the offline
/// slot is held by another op, e.g. the user started a new recording) at info as
/// a skip, anything else at warn — and **never aborts the remaining passes**: the
/// recording is already safely persisted, so a failed reprocess must not prevent
/// the auto-summarise pass. `run_pass` is a closure (rather than a direct
/// `Orchestrator` call) so a stub can drive the gating/ordering/error tolerance
/// in tests without models or audio.
pub(crate) async fn run_post_stop_passes<F, Fut>(
    passes: &[PostStopPass],
    meeting_id: MeetingId,
    mut run_pass: F,
) where
    F: FnMut(PostStopPass) -> Fut,
    Fut: std::future::Future<Output = Result<(), AppError>>,
{
    for &pass in passes {
        if let Err(e) = run_pass(pass).await {
            let busy = matches!(e, AppError::InvalidInput { .. });
            match (pass, busy) {
                (PostStopPass::Reprocess, true) => tracing::info!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "background reprocess skipped: recorder busy with another op"
                ),
                (PostStopPass::Reprocess, false) => tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "background reprocess after stop failed: {e}; keeping the live \
                     transcript (not auto-retried — use the reprocess action)"
                ),
                // #68 — auto-summarise is best-effort: a failure (model load, an
                // empty/unsummarisable transcript, etc.) leaves the meeting without
                // a summary; the user can still press Summarise. The `busy` arm is
                // not distinguished — `run_held_summarise` does not claim the
                // offline slot — so both are logged as a warning.
                (PostStopPass::Summarise, _) => tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "background auto-summarise after stop failed: {e}; no summary written \
                     (use the Summarise action)"
                ),
            }
        }
    }
}

/// Build the [`MeetingListEntry`] for a just-stopped meeting from its
/// [`MeetingMeta`] plus the first transcript segment (the list excerpt).
///
/// Blocking `std::fs` read of `transcript.json` (via
/// `persistence::reader::read_transcript`); an absent/empty transcript yields
/// `excerpt: None`. Extracted so it can be unit-tested without a Tauri runtime.
pub(crate) fn meeting_list_entry_for_meta(meetings_dir: &Path, meta: &MeetingMeta) -> MeetingListEntry {
    let meeting_dir = meetings_dir.join(meta.uuid.0.to_string());
    let excerpt = persistence::read_transcript(&meeting_dir)
        .ok()
        .and_then(|segs| segs.first().map(|s| s.text.clone()));

    MeetingListEntry {
        id: meta.uuid,
        title: meta.title.clone(),
        started_at: meta.started_at.clone(),
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
        collection_id: meta.collection_id,
        recording_started: meta.recording_started,
        deleted_at: meta.deletion.deleted_at(),
    }
}

/// Return a snapshot of the current recording state.
#[tauri::command]
#[specta::specta]
pub async fn get_recording_state(state: State<'_, IpcState>) -> AppResult<RecordingState> {
    Ok(state.orchestrator.state().await)
}

