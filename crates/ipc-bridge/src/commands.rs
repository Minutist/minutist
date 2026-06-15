//! Tauri command handlers for the Phase 1 IPC surface.
//!
//! Every command is annotated with both `#[tauri::command]` (wires it into
//! Tauri's invoke handler) and `#[specta::specta]` (registers its signature
//! for TypeScript generation).
//!
//! Commands are `async fn` because the orchestrator's methods are async.
//!
//! `list_devices` routes through `Orchestrator::list_devices` (which wraps
//! the cpal enumeration in `spawn_blocking`), preserving the dependency-table
//! invariant that `ipc-bridge` depends only on `orchestrator + settings +
//! common`.
//!
//! All commands return `Result<T, IpcError>`.  The `?` operator on
//! `AppResult<T>` automatically converts via `IpcError::from(AppError)`.
//!
//! ## Specta types
//!
//! `common` and `settings` derive `specta::Type` directly (gated on the
//! `specta` feature, which this crate enables on both deps). The mirror
//! layer that Phase 1 carried in `specta_types.rs` was removed in P0a;
//! commands return `common` / `settings` types directly.
//!
//! ## Tauri State
//!
//! Each command that needs the orchestrator or settings receives its handles
//! as `tauri::State<'_, IpcState>`.

use std::collections::HashMap;
use std::path::Path;

use std::sync::Arc;

use agent_tools::{ToolContext, ToolOutput};
use chat_agent::{LlamaTurnBackend, LlamaTurnConfig, TurnEngine};
use minutist_common::{
    AppError, AppEvent, AppResult, AudioDevice, ChatMessage, ChatRole, ChatSession, ChatSessionId,
    MeetingId, MeetingListEntry, MeetingMeta, MeetingState, ModelId, ModelStatus, NotesDocument,
    OperationKind, RecordingState, Summariser,
};
use persistence::{meeting_ops, ChatStore, NotesStore};
use settings::Settings;
use summariser::{LlamaSummariser, SummariseProgress, SummariserConfig};
use tauri::State;
use tokio::sync::broadcast;

use crate::chat::{
    engine_message_from_wire, initial_history, run_chat_turn, wire_role, CHAT_N_CTX,
};
use crate::chat_runtime::ChatHandles;
use crate::output_language::resolve_output_language;
use crate::{error::IpcError, IpcState};

/// Append an output-language instruction to a system prompt when
/// `settings.output_language` resolves to a concrete language name.
///
/// Applies to both the summariser system prompt and the chat system prompt:
/// when [`resolve_output_language`] returns `Some(lang)`, appends
/// `"\n\nRespond entirely in {lang}."` after the full prompt (including any
/// user-customised text). Appending AFTER any user-customised prompt ensures
/// the explicit output-language setting is honoured even when the user's
/// custom prompt itself says something different. Returns the prompt unchanged
/// when the setting resolves to `None` (e.g. `"auto"` on an unmapped locale).
pub(crate) fn apply_output_language(prompt: &str, output_language_setting: &str) -> String {
    match resolve_output_language(output_language_setting) {
        Some(lang) => format!("{prompt}\n\nRespond entirely in {lang}."),
        None => prompt.to_string(),
    }
}

/// The bundled default LLM model id used when `settings.llm_model_id` is unset.
///
/// Matches the `gemma-4-e4b-it-q4_k_m` entry in `resources/models.json`
/// (`kind = "llm"`). The model is settings-selected — never hard-coded into the
/// summariser — so a user override is honoured first; this constant is only the
/// fallback. See `architecture/components.md` — `summariser` "Bundled default
/// model".
///
/// `pub` so the manifest-consistency guard test
/// (`crates/ipc-bridge/tests/default_model_manifest.rs`) can assert this id
/// stays a real `kind = Llm` entry in `resources/models.json` — turning a
/// manifest rename into a failing test rather than a silently-broken default
/// summarise path.
pub const DEFAULT_LLM_MODEL_ID: &str = "gemma-4-e4b-it-q4_k_m";

/// Resolve the LLM model id used by [`summarise_meeting`]: the user-selected
/// `settings.llm_model_id` if set, else the bundled default
/// [`DEFAULT_LLM_MODEL_ID`].
///
/// Extracted (rather than inlined in the command) so the settings-override /
/// fallback decision is unit-testable without a Tauri runtime or an
/// orchestrator (a Phase 5 design decision).
pub(crate) fn resolve_llm_model_id(settings: &Settings) -> ModelId {
    settings
        .llm_model_id
        .clone()
        .unwrap_or_else(|| ModelId::from(DEFAULT_LLM_MODEL_ID))
}

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

/// List all available audio-input devices.
///
/// Routes through `Orchestrator::list_devices`, which wraps the FFI-bound
/// cpal enumeration in `spawn_blocking`. This keeps `ipc-bridge`'s
/// dependency table honest: it depends on `orchestrator`, not directly on
/// `audio-capture`.
#[tauri::command]
#[specta::specta]
pub async fn list_devices(state: State<'_, IpcState>) -> Result<Vec<AudioDevice>, IpcError> {
    state
        .orchestrator
        .list_devices()
        .await
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Recording lifecycle
// ---------------------------------------------------------------------------

/// Start a new recording session.
///
/// `device_id = None` → use the device configured in settings, or the OS
/// default if none is configured.
///
/// Returns the new `MeetingId` on success.
#[tauri::command]
#[specta::specta]
pub async fn start_recording(
    device_id: Option<String>,
    state: State<'_, IpcState>,
) -> Result<MeetingId, IpcError> {
    state
        .orchestrator
        .start(device_id)
        .await
        .map_err(IpcError::from)
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
pub async fn prewarm_asr(state: State<'_, IpcState>) -> Result<(), IpcError> {
    state.orchestrator.prewarm_asr().await;
    Ok(())
}

/// Pause the current recording.
#[tauri::command]
#[specta::specta]
pub async fn pause_recording(state: State<'_, IpcState>) -> Result<(), IpcError> {
    state.orchestrator.pause().await.map_err(IpcError::from)
}

/// Resume after a pause.
#[tauri::command]
#[specta::specta]
pub async fn resume_recording(state: State<'_, IpcState>) -> Result<(), IpcError> {
    state.orchestrator.resume().await.map_err(IpcError::from)
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
pub async fn stop_recording(state: State<'_, IpcState>) -> Result<MeetingMeta, IpcError> {
    let meta = state.orchestrator.stop().await.map_err(IpcError::from)?;

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

    // Decoupled background post-processing: the meeting is already indexed and
    // visible, so any heavy passes run OFF the stop path (a slow/hung pass can
    // never wedge stop or hide the meeting). Up to two passes run, in order, in
    // one fire-and-forget task:
    //   1. Re-transcribe (FR — ASR repair): if the live transcript fell behind
    //      (drop-oldest loss during recording, or a stop-drain timeout), re-run
    //      ASR over the COMPLETE `audio.opus` — the authoritative transcript,
    //      since the audio is captured in full regardless of live-ASR speed.
    //   2. Diarize (FR-11): if enabled, run AFTER any re-transcribe so it labels
    //      the repaired transcript (`rediarize` carries its own length-relative
    //      timeout). Emits `AppEvent::DiarizationComplete` when done.
    // Both claim the offline slot internally and run sequentially; errors are
    // logged (the recording is safely on disk). NOTE: `take_transcript_incomplete`
    // is consumed here, so a re-transcribe that fails or is skipped (recorder
    // busy with another op) is NOT auto-retried — the user-triggered re-transcribe
    // action is the recovery path. (Only the meeting INDEX self-heals, via
    // `reconcile_orphans` on `list_meetings`.)
    //
    //   3. Auto-summarise (#68): if `settings.auto_summarise_on_stop` is on (the
    //      default), run AFTER any re-transcribe / re-diarize so it summarises the
    //      FINAL transcript. Uses the held-summariser path ([`run_held_summarise`])
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
    if !passes.is_empty() {
        let orchestrator = std::sync::Arc::clone(&state.orchestrator);
        let index = std::sync::Arc::clone(&state.index);
        let handles = state.chat_handles();
        let meeting_id = meta.uuid;
        tokio::spawn(async move {
            run_post_stop_passes(&passes, meeting_id, |pass| {
                let orchestrator = std::sync::Arc::clone(&orchestrator);
                let index = std::sync::Arc::clone(&index);
                let handles = handles.clone();
                async move {
                    match pass {
                        PostStopPass::ReTranscribe => {
                            orchestrator.re_transcribe(&index, meeting_id).await
                        }
                        PostStopPass::Rediarize => orchestrator.rediarize(&index, meeting_id).await,
                        // Map the held-summarise `IpcError` back to `AppError` for
                        // the shared per-pass error logging; the markdown result is
                        // discarded (the summary is persisted + `SummaryReady`
                        // emitted inside `run_held_summarise`).
                        //
                        // Unlike re-transcribe / re-diarize, this pass does NOT take
                        // the orchestrator's offline claim, so it cannot self-skip
                        // when a new recording preempts the slot. Gate it explicitly:
                        // if the user has started the next meeting, defer this
                        // meeting's auto-summarise (recoverable via the manual
                        // Summarise action) rather than contending with the live
                        // recording's GPU/LLM use.
                        PostStopPass::Summarise => {
                            if orchestrator.recorder_is_live().await {
                                Err(AppError::InvalidInput {
                                    context: "auto-summarise deferred: a new recording started"
                                        .into(),
                                })
                            } else {
                                run_held_summarise(&handles, meeting_id)
                                    .await
                                    .map(|_| ())
                                    .map_err(AppError::from)
                            }
                        }
                    }
                }
            })
            .await;
        });
    }

    Ok(meta)
}

/// A background pass `stop_recording` may run after a stop, off the stop path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PostStopPass {
    /// Re-run ASR over the complete `audio.opus` (the live transcript fell behind).
    ReTranscribe,
    /// Run speaker diarization (`settings.diarization_enabled`).
    Rediarize,
    /// Auto-summarise the meeting (`settings.auto_summarise_on_stop`, #68). Runs
    /// LAST so it summarises the final (re-transcribed + re-diarized) transcript.
    Summarise,
}

/// The ordered post-stop passes to run, derived from the three gating flags.
///
/// Re-transcribe always precedes diarize so diarization labels the **repaired**
/// transcript rather than a truncated one; auto-summarise (#68) runs LAST so it
/// summarises the final transcript after any re-transcribe / re-diarize. An empty
/// result means no background task is spawned. Pure + unit-tested so the
/// gating/ordering is verified without a Tauri runtime.
fn post_stop_passes(
    needs_retranscribe: bool,
    needs_diarize: bool,
    needs_summarise: bool,
) -> Vec<PostStopPass> {
    let mut passes = Vec::with_capacity(3);
    if needs_retranscribe {
        passes.push(PostStopPass::ReTranscribe);
    }
    if needs_diarize {
        passes.push(PostStopPass::Rediarize);
    }
    if needs_summarise {
        passes.push(PostStopPass::Summarise);
    }
    passes
}

/// Run the planned post-stop `passes` in order, invoking `run_pass` for each.
///
/// Each pass's error is caught and logged — `AppError::InvalidInput` (the offline
/// slot is held by another op, e.g. the user started a new recording) at info as
/// a skip, anything else at warn — and **never aborts the remaining passes**: the
/// recording is already safely persisted, so a failed re-transcribe must not
/// prevent the diarize pass (or vice versa). `run_pass` is a closure (rather than
/// a direct `Orchestrator` call) so a stub can drive the gating/ordering/error
/// tolerance in tests without models or audio.
async fn run_post_stop_passes<F, Fut>(
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
                (PostStopPass::ReTranscribe, true) => tracing::info!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "background re-transcribe skipped: recorder busy with another op"
                ),
                (PostStopPass::ReTranscribe, false) => tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "background re-transcribe after stop failed: {e}; keeping the live \
                     transcript (not auto-retried — use the re-transcribe action)"
                ),
                (PostStopPass::Rediarize, true) => tracing::info!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "background diarization skipped: recorder busy with another op"
                ),
                (PostStopPass::Rediarize, false) => tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "background on-stop diarization failed: {e}; meeting left un-diarized"
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
fn meeting_list_entry_for_meta(meetings_dir: &Path, meta: &MeetingMeta) -> MeetingListEntry {
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
    }
}

/// Return a snapshot of the current recording state.
#[tauri::command]
#[specta::specta]
pub async fn get_recording_state(state: State<'_, IpcState>) -> Result<RecordingState, IpcError> {
    Ok(state.orchestrator.state().await)
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/// List all known models with their current runtime status.
///
/// Routes through `Orchestrator::list_models`, which wraps `ModelRegistry::list_models`
/// so that `ipc-bridge` does not need a direct `model-registry` dependency.
#[tauri::command]
#[specta::specta]
pub async fn list_models(state: State<'_, IpcState>) -> Result<Vec<ModelStatus>, IpcError> {
    Ok(state.orchestrator.list_models())
}

/// Ensure a model is downloaded and hash-verified.
///
/// Returns `Ok(())` when the model is ready for use. Starts a download if the
/// model is absent; the webview tracks granular progress via
/// `AppEvent::ModelDownloadProgress` events emitted on the broadcast channel.
///
/// Routes through `Orchestrator::ensure_model`, preserving the dependency-table
/// invariant that `ipc-bridge` does not depend directly on `model-registry`.
#[tauri::command]
#[specta::specta]
pub async fn ensure_model(model_id: ModelId, state: State<'_, IpcState>) -> Result<(), IpcError> {
    state
        .orchestrator
        .ensure_model(&model_id)
        .await
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Return the current application settings.
#[tauri::command]
#[specta::specta]
pub async fn get_settings(state: State<'_, IpcState>) -> Result<Settings, IpcError> {
    Ok(state.settings.current())
}

/// Persist updated application settings.
///
/// Broadcasts the change to all `SettingsHandle` subscribers.
#[tauri::command]
#[specta::specta]
pub async fn update_settings(
    settings: Settings,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    state
        .settings
        .update(|s| *s = settings)
        .await
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Notes persistence (Phase 3)
// ---------------------------------------------------------------------------

// The notes wire type returned by [`load_notes`] is `common::NotesDocument`
// (`notes_json` + `notes_markdown`, both `String`). It carries the
// Tiptap/ProseMirror document as a `String`, not a `serde_json::Value`: a bare
// `serde_json::Value` does not derive `specta::Type`, so it cannot cross the
// tauri-specta boundary directly. The webview owns the (de)serialisation of this
// opaque document; `persistence` stores it verbatim (the Phase-4 transcript-chip
// opacity guarantee). The `String`-over-the-wire choice keeps the IPC contract
// typed without forcing a Rust-side Tiptap model.
//
// `ipc-bridge` re-uses `common::NotesDocument` directly rather than mirroring a
// local copy: a duplicate would emit a second, identical TypeScript type into
// `bindings.ts` (the `NotesDoc`/`NotesDocument` divergence #19, removed here).

/// Persist a meeting's notes (`notes.json` + `notes.md`).
///
/// Routes **directly** to `persistence::NotesStore` against
/// `IpcState::meetings_dir` — notes I/O is independent of the live recording
/// pipeline (see `architecture/components.md`, `persistence` "Phase 3 surface
/// growth — notes"), so the orchestrator is not involved. The blocking
/// filesystem write runs on `spawn_blocking` per the threading model.
///
/// `notes_json` is parsed from a `String` into a `serde_json::Value`; an
/// invalid JSON string is rejected as `AppError::InvalidInput`.
#[tauri::command]
#[specta::specta]
pub async fn save_notes(
    meeting_id: MeetingId,
    notes_json: String,
    notes_markdown: String,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        save_notes_inner(&meetings_dir, meeting_id, &notes_json, &notes_markdown)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("save_notes task join failed: {e}"),
    })?
    .map_err(IpcError::from)
}

/// Load a meeting's persisted notes, or `None` when no notes have been saved.
///
/// Routes directly to `persistence::NotesStore`; the loaded opaque
/// `serde_json::Value` is re-serialised back to a `String` for the wire (see
/// [`NotesDocument`]).
#[tauri::command]
#[specta::specta]
pub async fn load_notes(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<Option<NotesDocument>, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || load_notes_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("load_notes task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

/// Apply an incremental Yjs update from the editor's local `Y.Doc` (the
/// `'update'` event) onto the meeting's authoritative `notes.ydoc`, then
/// re-derive `notes.json` + write the caller-supplied `notes.md`.
///
/// This is the **primary write path for an open editor** (D-O2.1): with
/// `@tiptap/extension-collaboration` the editor is Yjs-native, so its edits
/// arrive as CRDT updates that MERGE onto the stored doc — preserving the CRDT
/// history that the JSON-rebuild `save_notes` discards. `update` is a lib0 **v1**
/// update (the format the JS `yjs` library emits); the wire type is `Vec<u8>`,
/// exported as `number[]` (matching `save_note_image`'s `bytes` — no base64
/// hop). Routes directly to `persistence::NotesStore::apply_update` on
/// `spawn_blocking`, mirroring `save_notes`.
#[tauri::command]
#[specta::specta]
pub async fn apply_notes_update(
    meeting_id: MeetingId,
    update: Vec<u8>,
    notes_markdown: String,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        NotesStore::apply_update(&meetings_dir, meeting_id, &update, &notes_markdown)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("apply_notes_update task join failed: {e}"),
    })?
    .map_err(IpcError::from)
}

/// Read the meeting's current `notes.ydoc` state as a lib0 **v1** update for the
/// editor to apply with `Y.applyUpdate` on open.
///
/// Returns `None` when the meeting has no `notes.ydoc` (the editor then starts
/// empty and its first edit seeds the doc). The wire type is `Option<Vec<u8>>`,
/// exported as `number[] | null`. The stored blob is v2 (durable); persistence
/// re-encodes it as v1 because the JS `yjs` library only accepts v1 over
/// `applyUpdate` (the v1/v2 hops must not be crossed — see
/// `persistence::ydoc`). Routes directly to
/// `persistence::NotesStore::read_ydoc_state` on `spawn_blocking`.
#[tauri::command]
#[specta::specta]
pub async fn load_notes_ydoc(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<Option<Vec<u8>>, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || NotesStore::read_ydoc_state(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("load_notes_ydoc task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Notes command bodies — extracted so they can be unit-tested without a
// running Tauri runtime (the round-trip test calls these directly).
// ---------------------------------------------------------------------------

/// Inner body of [`save_notes`]: parse the JSON string and write via
/// `NotesStore`. Returns `AppError` so both the command and the unit test
/// share one error path.
fn save_notes_inner(
    meetings_dir: &std::path::Path,
    meeting_id: MeetingId,
    notes_json: &str,
    notes_markdown: &str,
) -> Result<(), AppError> {
    let value: serde_json::Value =
        serde_json::from_str(notes_json).map_err(|e| AppError::InvalidInput {
            context: format!("notes_json is not valid JSON: {e}"),
        })?;
    NotesStore::save(meetings_dir, meeting_id, &value, notes_markdown)
}

/// Inner body of [`load_notes`]: read via `NotesStore` and re-serialise the
/// opaque document back to a `String` for the wire.
fn load_notes_inner(
    meetings_dir: &std::path::Path,
    meeting_id: MeetingId,
) -> Result<Option<NotesDocument>, AppError> {
    let loaded = NotesStore::load(meetings_dir, meeting_id)?;
    match loaded {
        None => Ok(None),
        Some(data) => {
            let notes_json = serde_json::to_string(&data.json).map_err(|e| AppError::Internal {
                context: format!("failed to re-serialise loaded notes.json: {e}"),
            })?;
            Ok(Some(NotesDocument {
                notes_json,
                notes_markdown: data.markdown,
            }))
        }
    }
}

/// The image extensions a pasted/dropped note image may carry.
///
/// Lower-cased, no leading dot. The frontend derives `ext` from the clipboard
/// `File`'s MIME type / name; this allowlist is the authoritative gate so an
/// arbitrary extension can never reach the filesystem. The matching set is
/// mirrored by the protocol handler's content-type map in `app-main`.
const ALLOWED_IMAGE_EXTS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// Persist a pasted/dropped note image to the meeting's `assets/` directory and
/// return its **portable** reference (the bare `<contenthash>.<ext>` filename)
/// for the frontend to store into `notes.json`.
///
/// Routes **directly** to `persistence::save_note_asset` against
/// `IpcState::meetings_dir` — note assets are independent of the live recording
/// pipeline (see `architecture/components.md`, `persistence` "Note image
/// assets"), so the orchestrator is not involved. The blocking filesystem write
/// runs on `spawn_blocking` per the threading model.
///
/// `ext` is validated against [`ALLOWED_IMAGE_EXTS`] (case-insensitively); a
/// non-image extension is rejected as `AppError::InvalidInput`. The returned
/// filename is portable: it names only the file, not a path or a URL, so the
/// meeting folder (with `assets/`) can be copied to another machine and the
/// notes still resolve. The webview turns the filename into a working
/// `meetingasset:` URL at render time via `convertFileSrc`.
#[tauri::command]
#[specta::specta]
pub async fn save_note_image(
    meeting_id: MeetingId,
    bytes: Vec<u8>,
    ext: String,
    state: State<'_, IpcState>,
) -> Result<String, IpcError> {
    let ext = normalise_image_ext(&ext)?;
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        persistence::save_note_asset(&meetings_dir, meeting_id, &bytes, &ext)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("save_note_image task join failed: {e}"),
    })?
    .map_err(IpcError::from)
}

/// Validate + normalise a note-image extension to a lower-cased, dot-less form
/// drawn from the [`ALLOWED_IMAGE_EXTS`] allowlist.
///
/// Pure + unit-tested so the allowlist gate is verified without a Tauri runtime.
/// Strips a single leading dot, lower-cases, and rejects anything not on the
/// list as `AppError::InvalidInput`.
fn normalise_image_ext(ext: &str) -> Result<String, AppError> {
    let cleaned = ext.trim().trim_start_matches('.').to_ascii_lowercase();
    if ALLOWED_IMAGE_EXTS.contains(&cleaned.as_str()) {
        Ok(cleaned)
    } else {
        Err(AppError::InvalidInput {
            context: format!(
                "unsupported note image extension {ext:?}; allowed: {ALLOWED_IMAGE_EXTS:?}"
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Meeting list + open + actions (Phase 4)
// ---------------------------------------------------------------------------

/// List all meetings for the meeting-list view (FR-33), most-recent first.
///
/// Reads straight from the libsql `index.db` ([`MeetingIndex::list_meetings`])
/// — a cheap projection that never loads a meeting's full transcript. The index
/// is async (libsql/tokio); the future is awaited here, never `block_on`'d.
#[tauri::command]
#[specta::specta]
pub async fn list_meetings(state: State<'_, IpcState>) -> Result<Vec<MeetingListEntry>, IpcError> {
    // Self-heal: index any meeting present on disk but missing from the cache
    // (e.g. the process killed between finalise and the stop-time upsert) so it
    // can never stay hidden until the next startup `rebuild_from_disk`. Cheap
    // (a readdir + set-diff; only NEW folders are read). Best-effort — a
    // reconcile failure must not break listing, so log and serve the cache.
    if let Err(e) = state.index.reconcile_orphans(&state.meetings_dir).await {
        tracing::warn!(
            target: "ipc-bridge",
            "meeting-list self-heal reconcile failed: {e}; listing cached rows as-is"
        );
    }
    state.index.list_meetings().await.map_err(IpcError::from)
}

/// Open a meeting, returning its full restorable [`MeetingState`]
/// (metadata + transcript + optional notes).
///
/// Resolves the per-meeting folder under `IpcState::meetings_dir` and assembles
/// the state via `persistence::read_meeting_state`. The blocking `std::fs` reads
/// run on `spawn_blocking` per the threading model — the index is not consulted
/// (the folder is authoritative for the full state).
#[tauri::command]
#[specta::specta]
pub async fn open_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<MeetingState, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || open_meeting_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("open_meeting task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

/// Rename a meeting: updates `metadata.json` (authoritative) then the index row.
///
/// Routes to `persistence::meeting_ops::rename_meeting`, which keeps the on-disk
/// folder and the index consistent (see `architecture/components.md`,
/// `persistence` "Phase 4 surface growth — meeting ops").
#[tauri::command]
#[specta::specta]
pub async fn rename_meeting(
    meeting_id: MeetingId,
    title: String,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    meeting_ops::rename_meeting(&state.meetings_dir, &state.index, meeting_id, &title)
        .await
        .map_err(IpcError::from)
}

/// Set a speaker's display name on a saved meeting.
///
/// Maps a diarizer label (e.g. `"A"`) to a display name in `metadata.json`'s
/// `speaker_names`; an empty `name` clears the mapping. Returns the updated
/// map so the caller can re-render without re-reading the meeting. Names are
/// keyed by the diarizer label, so re-running diarization (which re-letters
/// speakers) resets them. Routes to `persistence::meeting_ops::set_speaker_name`.
///
/// The label and name are each capped at `MAX_SPEAKER_NAME_LEN` characters so
/// the UI cannot persist an unbounded value (mirrors the `set_speaker_name`
/// chat tool's bound).
#[tauri::command]
#[specta::specta]
pub async fn set_speaker_name(
    meeting_id: MeetingId,
    label: String,
    name: String,
    state: State<'_, IpcState>,
) -> Result<std::collections::BTreeMap<String, String>, IpcError> {
    const MAX_SPEAKER_NAME_LEN: usize = 512;
    let name = name.trim().to_string();
    if label.chars().count() > MAX_SPEAKER_NAME_LEN
        || name.chars().count() > MAX_SPEAKER_NAME_LEN
    {
        return Err(IpcError::from(minutist_common::AppError::InvalidInput {
            context: format!("speaker label/name too long (max {MAX_SPEAKER_NAME_LEN} characters)"),
        }));
    }
    meeting_ops::set_speaker_name(&state.meetings_dir, meeting_id, &label, &name)
        .await
        .map_err(IpcError::from)
}

/// Delete a meeting: removes the folder then the index row.
///
/// Routes to `persistence::meeting_ops::delete_meeting`.
#[tauri::command]
#[specta::specta]
pub async fn delete_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    meeting_ops::delete_meeting(&state.meetings_dir, &state.index, meeting_id)
        .await
        .map_err(IpcError::from)
}

/// Re-run transcription for a meeting offline (FR-33 action).
///
/// Routes to `Orchestrator::re_transcribe`, which decodes the meeting audio and
/// drives the same batched-VAD + ASR pipeline the live recorder uses, rewrites
/// `transcript.json`, refreshes the index row, and emits
/// `AppEvent::TranscriptSegment` events. Refused while a recording is in
/// progress (the orchestrator returns `AppError::InvalidInput`). The index
/// handle is shared from `IpcState` so the orchestrator need not own one.
#[tauri::command]
#[specta::specta]
pub async fn re_transcribe(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    state
        .orchestrator
        .re_transcribe(&state.index, meeting_id)
        .await
        .map_err(IpcError::from)
}

/// Re-run speaker diarization for a meeting offline (Phase 6, FR-11 action).
///
/// Routes to `Orchestrator::rediarize`, which decodes the meeting's
/// pause-INCLUDING PCM, runs the bundled `SherpaDiarizer` over the stored
/// transcript segments (resolving both diarize model directories via
/// `model-registry`), rewrites `transcript.json` with the overlaid
/// `speaker_id`s, updates `metadata.json`'s `{ speaker_count, diarizer }`,
/// refreshes the index row's `speaker_count`, and emits
/// `AppEvent::DiarizationComplete { meeting_id, speaker_count }` — the event is
/// emitted by the **orchestrator** (not here), on the shared bus the forwarder
/// subscribes to. Refused while a recording is in progress (the orchestrator
/// returns `AppError::InvalidInput`).
///
/// The `model-registry` edge stays inside the orchestrator (the diarizer is
/// built there) — there is **no** `ipc-bridge → diarizer` Cargo edge;
/// `ipc-bridge` routes via the orchestrator. The shared `IpcState::index` handle
/// is passed into the call so the orchestrator refreshes the index row without
/// owning an index of its own.
#[tauri::command]
#[specta::specta]
pub async fn rediarize_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    state
        .orchestrator
        .rediarize(&state.index, meeting_id)
        .await
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Summary (Phase 5)
// ---------------------------------------------------------------------------

/// Run summarisation for a meeting (FR-30): produce `summary.md` and emit
/// `AppEvent::SummaryReady`.
///
/// Pipeline:
/// 1. Resolve the LLM model id — `settings.llm_model_id` if set, else the
///    bundled default [`DEFAULT_LLM_MODEL_ID`].
/// 2. Resolve the model **directory** via `Orchestrator::ensure_model_path`
///    (downloads + verifies when absent). This keeps the `model-registry` edge
///    in the orchestrator — there is **no** `orchestrator → summariser` edge;
///    the summariser is loaded here, in `ipc-bridge` (the granted
///    `ipc-bridge → summariser` edge).
/// 3. Open a [`LlamaSummariser`] over the single `.gguf` in that directory
///    (skipping any `mmproj-*`), read the transcript + notes markdown, run the
///    summary, and write `summary.md` — **all on `spawn_blocking`** because the
///    summariser's `open` + `summarise` are heavy and synchronous (the
///    threading-model rule: inference on `spawn_blocking`, never in a handler).
/// 4. Emit `AppEvent::SummaryReady { meeting_id }` on the shared `event_tx` so
///    the summary view re-reads the persisted markdown.
///
/// The model resolution (`ensure_model_path`) is awaited on the async worker
/// (it may download); only the synchronous summariser work runs on
/// `spawn_blocking`.
#[tauri::command]
#[specta::specta]
pub async fn summarise_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    // The whole summarise body (held-model resolve → summarise-with-progress →
    // index refresh → `SummaryReady`) is extracted into [`run_held_summarise`] so
    // BOTH this user-triggered command and the post-stop auto-summarise chain
    // (#68, see [`stop_recording`]) drive the SAME path. The command surfaces an
    // error to the webview; the chain treats it as best-effort.
    run_held_summarise(&state.chat_handles(), meeting_id).await?;
    Ok(())
}

/// Resolve the held summariser, summarise a meeting (streaming determinate
/// `OperationProgress`), refresh the meeting-list index excerpt, and emit
/// `AppEvent::SummaryReady`.
///
/// Extracted from the `summarise_meeting` command (#68) so it can be invoked
/// from the post-stop background chain ([`run_post_stop_passes`]) as well as from
/// the command. Takes a [`ChatHandles`] — the same bundle the chat path uses —
/// so it shares the SAME lazily-loaded held model `Arc` (no GGUF reload).
///
/// Returns the summary markdown on success. The heavy `summarise` runs on
/// `spawn_blocking`, per the threading-model rule.
async fn run_held_summarise(
    handles: &ChatHandles,
    meeting_id: MeetingId,
) -> Result<String, IpcError> {
    let current = handles.settings.current();
    let event_tx = &handles.event_tx;

    // #69: surface the model LOAD as an indeterminate phase BEFORE it starts.
    // `ensure_summariser` mmaps + warms the multi-GB GGUF the FIRST time it is
    // called (and the first summarise of a session — including the post-stop
    // auto-summarise — pays it). On a warm load this flashes by; on a cold load
    // it is the bulk of the wait the user otherwise saw as a stuck 0%.
    emit_summarise_op(
        event_tx,
        meeting_id,
        None,
        "Loading the summarisation model…",
    );

    // Phase 9 (C2): use the HELD summariser substrate — loaded once on first
    // chat/summarise use and shared with the chat agent — instead of opening a
    // fresh `LlamaSummariser` per call (the old per-call GGUF reload killed
    // latency). The load resolves the model id + directory via the orchestrator
    // (keeping the `model-registry` edge there) and opens the GGUF on
    // `spawn_blocking` the FIRST time only; thereafter this is a cheap clone.
    let summariser = handles.ensure_summariser().await?;

    let meetings_dir_owned = handles.meetings_dir.clone();
    // Resolve the preset-aware effective prompt (Phase 9 — D4): the user's
    // custom `summary_system_prompt` override when non-empty, else the built-in
    // prompt for the selected `summary_preset`. The output-language instruction
    // is appended last so it wins over any conflicting text in a custom prompt.
    let system_prompt =
        apply_output_language(&current.effective_summary_prompt(), &current.output_language);

    // Heavy, synchronous summarise work on a blocking thread (the held model's
    // `summarise` builds a fresh `LlamaContext` and decodes). The held handle is
    // cloned into the closure; no GGUF reload. The held substrate is the concrete
    // `LlamaSummariser`, so we drive `summarise_with_progress` directly (the
    // `common::Summariser` trait is unchanged — see `summariser`).
    let event_tx_for_blocking = event_tx.clone();
    let summary_md = tokio::task::spawn_blocking(move || {
        summarise_meeting_with_progress(
            &meetings_dir_owned,
            meeting_id,
            summariser.as_ref(),
            &system_prompt,
            &event_tx_for_blocking,
        )
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("summarise_meeting task join failed: {e}"),
    })?
    .map_err(IpcError::from)?;

    // Live-test UX T6: refresh the meeting-list index row so its excerpt becomes
    // the summary blurb now that `summary.md` exists (the persistence excerpt
    // derivation prefers the summary blurb over the first transcript segment). A
    // failure is logged and swallowed — the index is a derived cache the next
    // startup rebuild reconciles, so a failed refresh must not fail the summary.
    let meetings_dir_for_entry = handles.meetings_dir.clone();
    match tokio::task::spawn_blocking(move || {
        meeting_list_entry_for_meta_with_summary(&meetings_dir_for_entry, meeting_id)
    })
    .await
    {
        Ok(Ok(Some(entry))) => {
            if let Err(e) = handles.index.upsert(&entry).await {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "index excerpt refresh after summarise failed: {e}; \
                     reconciled on next startup"
                );
            }
        }
        Ok(Ok(None)) => {
            // Metadata unreadable (the meeting folder was deleted mid-summarise);
            // nothing to upsert.
        }
        Ok(Err(e)) => tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "building index entry after summarise failed: {e}; reconciled on next startup"
        ),
        Err(join_err) => tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "index entry build after summarise join failed: {join_err}; \
             reconciled on next startup"
        ),
    }

    // Emit SummaryReady so the webview re-reads `summary.md` (this also clears
    // the per-row progress indicator and refreshes the list excerpt).
    emit_summary_ready(event_tx, meeting_id);

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        summary_len = summary_md.len(),
        "summary generated and persisted"
    );

    Ok(summary_md)
}

/// Build the [`MeetingListEntry`] for a meeting from its metadata, deriving the
/// excerpt from `summary.md` (the T6 blurb) when one exists, else the first
/// transcript segment — via `persistence::summary_blurb` + `read_transcript`.
///
/// Returns `Ok(None)` when the metadata cannot be read (the folder is gone).
/// Blocking `std::fs` reads; the caller drives it on `spawn_blocking`.
fn meeting_list_entry_for_meta_with_summary(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Result<Option<MeetingListEntry>, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    let meta = match persistence::read_metadata(&meeting_dir) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };

    // Prefer the summary blurb; fall back to the first transcript segment.
    let excerpt = persistence::read_summary(&meeting_dir)
        .ok()
        .flatten()
        .and_then(|md| persistence::summary_blurb(&md))
        .or_else(|| {
            persistence::read_transcript(&meeting_dir)
                .ok()
                .and_then(|segs| segs.first().map(|s| s.text.clone()))
        });

    Ok(Some(MeetingListEntry {
        id: meta.uuid,
        title: meta.title,
        started_at: meta.started_at,
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
    }))
}

/// Production summarise body that streams DETERMINATE `OperationProgress` events
/// (live-test UX T4(b)).
///
/// Mirrors [`summarise_meeting_inner`] (read transcript + notes → summarise →
/// write `summary.md`) but drives the concrete [`LlamaSummariser`]'s
/// `summarise_with_progress` so the generation loop reports `(tokens_generated,
/// max_tokens)`; the callback emits a throttled (~5 Hz) determinate
/// `AppEvent::OperationProgress`. Synchronous — the caller drives it on
/// `spawn_blocking`. (`summarise_meeting_inner` keeps the trait-based
/// `&dyn Summariser` seam for the stub unit tests, which assert the read/write
/// wiring without progress.)
fn summarise_meeting_with_progress(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    summariser: &LlamaSummariser,
    system_prompt: &str,
    event_tx: &broadcast::Sender<AppEvent>,
) -> Result<String, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());

    let mut transcript = persistence::read_transcript(&meeting_dir)?;
    // Overlay user-set speaker names onto a copy before summarising (same as the
    // non-progress path); the on-disk transcript keeps the raw labels.
    let meta = persistence::read_metadata(&meeting_dir)?;
    minutist_common::apply_speaker_overlay(&mut transcript, &meta.speaker_names);
    // #70: load the note paragraphs WITH their recording-clock anchors so the
    // summariser can weave them into the transcript at the time they were
    // written (rather than appending a flat markdown block).
    let notes = persistence::read_note_blocks(&meeting_dir)?;

    // #69: the model is loaded; the next opaque wait is building the
    // `LlamaContext` (cold GPU shader compile, tens of seconds on first use)
    // before the first prefill tick. Show an indeterminate "Preparing…" until
    // the prefill phase starts reporting.
    emit_summarise_op(event_tx, meeting_id, None, "Preparing the model…");

    // Throttle the callback to ~5 Hz so a fast GPU run does not flood the
    // broadcast bus (the meter already runs at ~30 Hz on the same channel), but
    // ALWAYS emit on a phase change (prefill→generate) and on completion so the
    // label flips promptly and the bar reaches 100%.
    let mut last_emit = std::time::Instant::now();
    let mut last_phase: u8 = 0;
    let summary_md = summariser.summarise_with_progress(
        &transcript,
        &notes,
        system_prompt,
        |progress| {
            let (phase, fraction, label): (u8, f32, &str) = match progress {
                SummariseProgress::Prefill { done, total } => (
                    1,
                    if total == 0 { 1.0 } else { done as f32 / total as f32 },
                    "Reading the meeting…",
                ),
                SummariseProgress::Generate { done, max } => (
                    2,
                    if max == 0 { 1.0 } else { done as f32 / max as f32 },
                    "Writing the summary…",
                ),
            };
            let now = std::time::Instant::now();
            let phase_changed = phase != last_phase;
            let complete = fraction >= 1.0;
            if phase_changed
                || complete
                || now.duration_since(last_emit) >= std::time::Duration::from_millis(200)
            {
                last_emit = now;
                last_phase = phase;
                emit_summarise_op(event_tx, meeting_id, Some(fraction), label);
            }
        },
    )?;

    persistence::write_summary(&meeting_dir, &summary_md)?;

    Ok(summary_md)
}

/// Emit an `AppEvent::OperationProgress` for the summarise op (#69).
///
/// `fraction` is `Some` for a determinate bar (clamped to `0..=1`) or `None`
/// for an indeterminate spinner (the model-load / context-prepare phases that
/// have no progress callback). `label` names the phase so the bar explains the
/// wait rather than sitting at a silent 0%.
fn emit_summarise_op(
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    fraction: Option<f32>,
    label: &str,
) {
    let _ = event_tx.send(AppEvent::OperationProgress {
        meeting_id,
        op: OperationKind::Summarise,
        fraction: fraction.map(|f| f.clamp(0.0, 1.0)),
        label: label.to_string(),
    });
}

/// Read a meeting's persisted summary markdown, or `None` when none exists.
///
/// Routes directly to `persistence::read_summary` (a blocking `std::fs` read on
/// `spawn_blocking`).
#[tauri::command]
#[specta::specta]
pub async fn get_summary(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<Option<String>, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || get_summary_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("get_summary task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

/// Persist an edited summary back to `summary.md` (FR-30).
///
/// Routes directly to `persistence::write_summary` (atomic tmp+rename) on
/// `spawn_blocking`.
#[tauri::command]
#[specta::specta]
pub async fn save_summary(
    meeting_id: MeetingId,
    summary_markdown: String,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        save_summary_inner(&meetings_dir, meeting_id, &summary_markdown)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("save_summary task join failed: {e}"),
    })?
    .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Summary command bodies — extracted so they can be unit-tested without a
// Tauri runtime, a real model, or the orchestrator. The summarise inner takes
// a `&dyn Summariser` so a `StubSummariser` exercises the read → summarise →
// write → event wiring (mirroring the orchestrator's re_transcribe stub-backend
// seam).
// ---------------------------------------------------------------------------

/// Inner body of [`summarise_meeting`]: read the meeting's transcript + notes
/// markdown, run `summariser`, and write `summary.md`. Returns the summary
/// markdown so the caller can log its length and the test can assert on it.
///
/// Synchronous (blocking `std::fs` + sync `Summariser::summarise`) — the caller
/// drives it on `spawn_blocking`. Takes `&dyn Summariser` so a stub can be
/// injected in tests.
///
/// `#[cfg(test)]`: the production command path now drives the concrete
/// [`summarise_meeting_with_progress`] (which streams `OperationProgress`, T4(b));
/// this trait-based seam is retained only for the stub unit tests that assert the
/// read → summarise → write wiring without a model or progress events.
#[cfg(test)]
fn summarise_meeting_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    summariser: &dyn Summariser,
    system_prompt: &str,
) -> Result<String, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());

    let mut transcript = persistence::read_transcript(&meeting_dir)?;
    // Summarise with the user-set speaker names ("Alice") rather than the raw
    // diarizer labels ("A"), matching every agent read tool. Overlay a copy —
    // the on-disk transcript keeps the raw labels. (A summary already on disk
    // keeps whatever labels it was generated with until it is regenerated.)
    let meta = persistence::read_metadata(&meeting_dir)?;
    minutist_common::apply_speaker_overlay(&mut transcript, &meta.speaker_names);
    // Note paragraphs with recording-clock anchors (#70); empty when the meeting
    // has none.
    let notes = persistence::read_note_blocks(&meeting_dir)?;

    let summary_md = summariser.summarise(&transcript, &notes, system_prompt)?;

    persistence::write_summary(&meeting_dir, &summary_md)?;

    Ok(summary_md)
}

/// Inner body of [`get_summary`]: read `summary.md` via `persistence`.
fn get_summary_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Result<Option<String>, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    persistence::read_summary(&meeting_dir)
}

/// Inner body of [`save_summary`]: write `summary.md` via `persistence`.
fn save_summary_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    summary_markdown: &str,
) -> Result<(), AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    persistence::write_summary(&meeting_dir, summary_markdown)
}

/// Resolve the summariser `n_gpu_layers` from the runtime `gpu_acceleration`
/// setting.
///
/// GPU offload happens ONLY when BOTH (a) the build was compiled with a GPU
/// feature AND (b) the setting is on. `enabled == true` → the compile-time
/// ceiling [`summariser::gpu_layers`] (already `0` in a default CPU-only build,
/// so a CPU build is unaffected by the flag); `enabled == false` → `0` (force
/// CPU even in a GPU build). Pure + unit-tested so the wiring is verified
/// without a model. See `architecture/cross-cutting.md` — "GPU portability".
pub(crate) fn resolve_summariser_gpu_layers(enabled: bool) -> u32 {
    if enabled {
        summariser::gpu_layers()
    } else {
        0
    }
}

/// Open a [`LlamaSummariser`] over the single `.gguf` weights file in
/// `model_dir`, skipping any `mmproj-*` multimodal projector.
///
/// `n_gpu_layers` is the runtime-resolved GPU-offload count (see
/// [`resolve_summariser_gpu_layers`]); it is set on the `SummariserConfig` so
/// the summariser honours the `gpu_acceleration` setting.
///
/// The LLM is text-only (the bundled Gemma 4 GGUF ships without a projector),
/// but the helper defends against a directory that also contains an `mmproj-*`
/// file so the wrong file is never loaded. A missing or ambiguous weights file
/// is an `AppError::ModelLoad`.
pub(crate) fn open_summariser_in_dir(
    model_dir: &Path,
    n_gpu_layers: u32,
) -> Result<LlamaSummariser, AppError> {
    let gguf_path = find_gguf_weights(model_dir)?;
    let config = SummariserConfig {
        n_gpu_layers,
        ..SummariserConfig::default()
    };
    LlamaSummariser::open(gguf_path, config)
}

/// Locate the single non-`mmproj` `.gguf` file in `model_dir`.
fn find_gguf_weights(model_dir: &Path) -> Result<std::path::PathBuf, AppError> {
    let read_dir = std::fs::read_dir(model_dir).map_err(|e| AppError::ModelLoad {
        model_id: model_dir.display().to_string(),
        context: format!("cannot read model directory: {e}"),
    })?;

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        let is_gguf = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        // Skip the multimodal projector — the summariser uses the text weights.
        if is_gguf && !name.to_ascii_lowercase().starts_with("mmproj") {
            candidates.push(path);
        }
    }

    match candidates.len() {
        1 => Ok(candidates.pop().expect("len == 1")),
        0 => Err(AppError::ModelLoad {
            model_id: model_dir.display().to_string(),
            context: "no .gguf weights file found in model directory".into(),
        }),
        n => Err(AppError::ModelLoad {
            model_id: model_dir.display().to_string(),
            context: format!("expected one .gguf weights file, found {n}"),
        }),
    }
}

/// Emit `AppEvent::SummaryReady { meeting_id }` on the shared broadcast sender.
///
/// A send with no live subscribers is not an error (broadcast semantics); it is
/// logged at trace, mirroring `Orchestrator::emit`.
fn emit_summary_ready(event_tx: &broadcast::Sender<AppEvent>, meeting_id: MeetingId) {
    match event_tx.send(AppEvent::SummaryReady { meeting_id }) {
        Ok(n) => tracing::trace!(
            target: "ipc-bridge",
            receivers = n,
            "SummaryReady broadcast"
        ),
        Err(_) => tracing::trace!(
            target: "ipc-bridge",
            "SummaryReady dropped (no subscribers)"
        ),
    }
}

// ---------------------------------------------------------------------------
// Chat (Phase 9)
// ---------------------------------------------------------------------------

/// Read a meeting's title for the chat scope line (best-effort; `None` when its
/// metadata can't be read). Runs the blocking `std::fs` read on `spawn_blocking`.
pub(crate) async fn read_meeting_title(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Option<String> {
    let dir = meetings_dir.join(meeting_id.0.to_string());
    tokio::task::spawn_blocking(move || persistence::read_metadata(&dir).ok().map(|m| m.title))
        .await
        .ok()
        .flatten()
}

/// Scope the chat system prompt to the open meeting.
///
/// When the chat is meeting-scoped, the agent must GROUND its answers in that
/// meeting and never ask the user for a meeting id — every meeting tool defaults
/// to it via [`ToolContext::default_meeting`]. The base prompt says "this
/// meeting" but never names which one, so without this the model has no meeting
/// identity and asks the user instead of calling a tool. With no meeting in scope
/// (a meeting-less chat) the base prompt is returned unchanged — the agent then
/// locates a meeting via `search_meetings` / an explicit id, as before.
pub(crate) fn chat_system_prompt_for_meeting(
    base: &str,
    meeting_id: Option<MeetingId>,
    title: Option<&str>,
) -> String {
    let Some(mid) = meeting_id else {
        return base.to_string();
    };
    let titled = match title {
        Some(t) if !t.trim().is_empty() => format!(" titled \"{}\"", t.trim()),
        _ => String::new(),
    };
    format!(
        "{base}\n\n# Current meeting\n\
         You are assisting with the meeting the user currently has open \
         (id: {id}{titled}). Every meeting tool defaults to THIS meeting, so NEVER \
         ask the user which meeting or for a meeting id — call the tools directly \
         (get_meeting, get_transcript, get_summary, get_notes, and the re-listen / \
         re-summarise / search / set-speaker tools) to ground your answers in it.",
        id = mid.0,
    )
}

/// Send a user message to the chat agent for a meeting, streaming the reply.
///
/// Creates or loads the chat [`ChatSession`], appends the user message, and
/// **spawns the turn on a background task**, returning the session id
/// immediately. The turn streams to the webview via the chat `AppEvent`s
/// (`ChatToken` / `ChatToolCall` / `ChatToolResult` / `ChatTurnComplete` /
/// `ChatError`) on the shared bus; the updated session is persisted via
/// [`ChatStore`] at turn end. A second `send_chat_message` for a session whose
/// turn is still running is rejected with `InvalidInput { "session busy" }`
/// (§6 — single in-flight turn per session).
///
/// The engine work runs on `spawn_blocking` (the LLM is FFI-bound); tool
/// dispatch re-enters async via a captured `Handle::block_on` for the dispatch
/// step only (§4.5 — the one place async/sync cross).
#[tauri::command]
#[specta::specta]
pub async fn send_chat_message(
    meeting_id: Option<MeetingId>,
    session_id: Option<ChatSessionId>,
    message: String,
    state: State<'_, IpcState>,
) -> Result<ChatSessionId, IpcError> {
    if message.trim().is_empty() {
        return Err(AppError::InvalidInput {
            context: "chat message must not be empty".into(),
        }
        .into());
    }

    let meetings_dir = state.meetings_dir.clone();

    // Load the existing session (when a session id + meeting id are given) or
    // start a fresh one. Persistence reads run on a blocking thread.
    let mut session = load_or_new_session(&meetings_dir, meeting_id, session_id).await?;
    let sid = session.id;

    // Single in-flight turn per session.
    {
        let mut in_flight = state
            .chat_in_flight
            .lock()
            .expect("chat_in_flight poisoned");
        if !in_flight.insert(sid) {
            return Err(AppError::InvalidInput {
                context: "session busy: a turn is already running".into(),
            }
            .into());
        }
    }

    // The per-session monotonic turn id: one past the max already recorded.
    let turn_id = session
        .messages
        .iter()
        .map(|m| m.turn_id)
        .max()
        .map_or(0, |t| t + 1);

    // Append the user message to the persisted session up front so it is durable
    // even if the turn errors mid-flight.
    session.messages.push(ChatMessage {
        role: ChatRole::User,
        content: message.clone(),
        tool_name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
        turn_id,
    });

    // Ensure the held model is loaded (downloads on first use) BEFORE spawning,
    // so a load failure surfaces synchronously to the caller rather than only as
    // a ChatError event. Cheap after the first call.
    let summariser = match state.ensure_summariser().await {
        Ok(s) => s,
        Err(e) => {
            state
                .chat_in_flight
                .lock()
                .expect("chat_in_flight poisoned")
                .remove(&sid);
            return Err(e);
        }
    };

    // Build the tool context for this session (default_meeting scopes meeting_id
    // omission for the internal UI).
    let ctx = ToolContext::new(
        Arc::clone(&state.orchestrator),
        Arc::clone(&state.index),
        meetings_dir.clone(),
        summariser.clone() as Arc<dyn Summariser>,
        state.event_tx.clone(),
        meeting_id,
    );

    // Scope the prompt to the open meeting so the agent uses the tools (which
    // default to this meeting) instead of asking the user for a meeting id.
    // The output-language instruction is appended last so it wins over any
    // conflicting text in a custom chat_system_prompt.
    let title = match meeting_id {
        Some(mid) => read_meeting_title(&meetings_dir, mid).await,
        None => None,
    };
    let current_settings = state.settings.current();
    let system_prompt = apply_output_language(
        &chat_system_prompt_for_meeting(
            &current_settings.chat_system_prompt,
            meeting_id,
            title.as_deref(),
        ),
        &current_settings.output_language,
    );
    let registry = Arc::clone(&state.tool_registry);
    let event_tx = state.event_tx.clone();
    let in_flight = Arc::clone(&state.chat_in_flight);
    let cancel_map = Arc::clone(&state.chat_cancel);
    let handle = tokio::runtime::Handle::current();

    // Register a fresh per-session cancel flag (P1) before spawning, so a
    // `cancel_chat_turn` arriving any time after this returns can raise it. The
    // decode loop checks it between tokens; the driver clears the entry at end.
    let cancel = chat_agent::CancelFlag::new();
    cancel_map
        .lock()
        .expect("chat_cancel poisoned")
        .insert(sid, cancel.clone());

    // Spawn the driver; the turn streams via events. The session id is returned
    // to the caller now. The turn task OWNS `session` (already carrying the user
    // message); at the end it appends the turn's produced messages and SAVES the
    // whole in-memory session. The single-in-flight-turn guard makes this turn the
    // sole writer, so we save the in-memory copy directly rather than
    // reload-and-append — that guarantees the user message is persisted, even when
    // the turn errors mid-flight.
    tokio::spawn(async move {
        let join = tokio::task::spawn_blocking(move || {
            let produced = run_chat_turn_on_held_model(
                &summariser,
                &registry,
                &ctx,
                &handle,
                sid,
                turn_id,
                &system_prompt,
                &session,
                &event_tx,
                &cancel,
                // The internal UI chat keeps the full tool set (no MCP gate).
                None,
            );
            (session, produced)
        })
        .await;

        match join {
            Ok((mut session, Ok(produced))) => {
                session.messages.extend(produced);
                persist_session(&meetings_dir, meeting_id, session).await;
            }
            Ok((session, Err(_))) => {
                // The driver already emitted ChatError; still persist the session
                // so the user's message (and any earlier turns) are not lost.
                persist_session(&meetings_dir, meeting_id, session).await;
            }
            Err(join_err) => {
                tracing::warn!(target: "ipc-bridge", "chat turn task join failed: {join_err}");
            }
        }
        in_flight
            .lock()
            .expect("chat_in_flight poisoned")
            .remove(&sid);
        cancel_map
            .lock()
            .expect("chat_cancel poisoned")
            .remove(&sid);
    });

    Ok(sid)
}

/// Cancel the in-flight chat turn for a session (P1).
///
/// Raises the per-session [`chat_agent::CancelFlag`] registered by
/// `send_chat_message`; the engine's decode loop observes it between tokens,
/// stops, and the driver emits the terminal `ChatTurnComplete` with the partial
/// text (cancellation is a user action, not a `ChatError`) and clears the
/// in-flight guard. Idempotent: a session with no running turn (no registered
/// flag) is a no-op success — the UI can call this freely to clear a stuck
/// "Sending…" state.
#[tauri::command]
#[specta::specta]
pub async fn cancel_chat_turn(
    session_id: ChatSessionId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    if let Some(flag) = state
        .chat_cancel
        .lock()
        .expect("chat_cancel poisoned")
        .get(&session_id)
    {
        flag.cancel();
    }
    Ok(())
}

/// The live MCP server endpoint (URL + bearer token) for the Settings → MCP
/// pane (Phase 10). `None` when the MCP server is disabled or not yet listening.
///
/// The bearer token is sensitive and crosses the IPC boundary ONLY here, on this
/// explicit read — it is never on the event bus, never logged, and not baked
/// into the bindings. The pane reveals it on user request.
///
/// v1 has no live token-rotation command: the token is generated once and
/// persisted to `{app-data}/mcp_token`, and the listener is spawned once at
/// startup. Rotating the token (delete the file → restart) is therefore
/// restart-required, consistent with the rest of the MCP lifecycle (enable /
/// port / write-tools changes are also restart-required for v1). The pane copy
/// states this; it does NOT offer a live regenerate control (C2).
#[tauri::command]
#[specta::specta]
pub async fn get_mcp_server_info(
    state: State<'_, IpcState>,
) -> Result<Option<crate::McpServerInfo>, IpcError> {
    Ok(state.mcp_info.lock().expect("mcp_info poisoned").clone())
}

// ---------------------------------------------------------------------------
// Translation (translated transcript as derived, regenerable view)
// ---------------------------------------------------------------------------

/// The set of supported target-language names for `translate_meeting`.
///
/// Matches the values in `output_language::SUBTAG_TO_LANGUAGE`. `translate_meeting`
/// validates its `target_language` argument against this set (case-sensitive full
/// English language names) so the LLM prompt is never built for a language the
/// picker does not expose. Values sorted alphabetically.
const SUPPORTED_TRANSLATION_LANGUAGES: &[&str] = &[
    "Arabic",
    "Chinese",
    "Dutch",
    "English",
    "French",
    "German",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Turkish",
];

/// Translate every segment of a meeting's transcript into `target_language` and
/// persist the results in `translations.json` alongside the meeting folder.
///
/// The translation is a **derived** view: the verbatim transcript remains
/// authoritative; `translations.json` is regenerable from it at any time. A
/// new full `write_transcript` (re-transcribe) automatically clears the sidecar
/// so stale translations never linger — see `persistence::write_transcript`.
///
/// # Concurrency
///
/// A second call for the same `(meeting_id, target_language)` pair while one is
/// already in flight is rejected with `AppError::InvalidInput` (mirrors the
/// `chat_in_flight` guard on chat sessions).
///
/// # Progress
///
/// Emits `AppEvent::OperationProgress { op: OperationKind::Translate }` (fraction
/// = segments_done / total_segments) throttled to ~5 Hz. Emits
/// `AppEvent::TranslationReady { meeting_id, language }` on every exit path
/// (success AND error) so the webview's operation-progress indicator is always
/// cleared. On a partial-failure exit the event fires before the error is
/// returned to the caller; any completed segments remain on disk.
///
/// # Errors
///
/// - `AppError::InvalidInput` when `target_language` is not in
///   [`SUPPORTED_TRANSLATION_LANGUAGES`] or the meeting has no transcript.
/// - `AppError::InvalidInput` when translation is already in-flight for this
///   `(meeting_id, target_language)` pair.
#[tauri::command]
#[specta::specta]
pub async fn translate_meeting(
    meeting_id: MeetingId,
    target_language: String,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    // Validate target_language against the supported set.
    if !SUPPORTED_TRANSLATION_LANGUAGES.contains(&target_language.as_str()) {
        return Err(AppError::InvalidInput {
            context: format!(
                "unsupported translation target language: {target_language:?}; \
                 supported: {SUPPORTED_TRANSLATION_LANGUAGES:?}"
            ),
        }
        .into());
    }

    // Reject a concurrent translate for the same (meeting_id, language) pair.
    let key = (meeting_id, target_language.clone());
    {
        let mut in_flight = state
            .translate_in_flight
            .lock()
            .expect("translate_in_flight poisoned");
        if !in_flight.insert(key.clone()) {
            return Err(AppError::InvalidInput {
                context: format!(
                    "translation already in-flight for meeting {} language {target_language:?}",
                    meeting_id.0
                ),
            }
            .into());
        }
    }

    // Emit indeterminate progress while the model loads.
    let _ = state.event_tx.send(AppEvent::OperationProgress {
        meeting_id,
        op: OperationKind::Translate,
        fraction: None,
        label: "Loading the translation model…".to_string(),
    });

    let load_result = state.ensure_summariser().await;

    // Drive the blocking work, then release the in-flight guard whether or
    // not the work succeeded (mirrors the chat_in_flight release pattern).
    let work_result = match load_result {
        Ok(summariser) => {
            let meetings_dir = state.meetings_dir.clone();
            let event_tx = state.event_tx.clone();
            let language_for_blocking = target_language.clone();
            tokio::task::spawn_blocking(move || {
                translate_meeting_blocking(
                    &meetings_dir,
                    meeting_id,
                    &language_for_blocking,
                    &summariser,
                    &event_tx,
                )
            })
            .await
            .map_err(|e| IpcError::from(AppError::Internal {
                context: format!("translate_meeting task join failed: {e}"),
            }))
            .and_then(|r| r.map_err(IpcError::from))
        }
        Err(e) => Err(e),
    };

    state
        .translate_in_flight
        .lock()
        .expect("translate_in_flight poisoned")
        .remove(&key);

    // Emit TranslationReady on every exit path so the operation-progress
    // indicator is cleared even when the pass fails mid-segment. The UI's
    // `handleEvent` only refetches translations when the meeting and language
    // match the active view; a refetch on a partial result is harmless — it
    // surfaces whatever segments completed before the error.
    let _ = state.event_tx.send(AppEvent::TranslationReady {
        meeting_id,
        language: target_language,
    });

    work_result?;

    Ok(())
}

/// Blocking body for `translate_meeting`.
///
/// Reads the transcript, translates each segment via
/// [`LlamaSummariser::translate_segment`], and merges batched results into
/// `translations.json` on the same ~200 ms cadence used for progress emission
/// (plus unconditionally on loop exit — both the normal completion and any
/// early-error exit — so partial progress always survives an interruption).
/// Batching replaces the previous per-segment flush (a full sidecar
/// read+rewrite+fsync per segment) to avoid O(n²) I/O on long meetings while
/// preserving the durability guarantee.
///
/// Emits `OperationProgress` throttled to ~5 Hz. Synchronous — the caller
/// drives this on `spawn_blocking`.
fn translate_meeting_blocking(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    target_language: &str,
    summariser: &LlamaSummariser,
    event_tx: &tokio::sync::broadcast::Sender<AppEvent>,
) -> Result<(), AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    let segments = persistence::read_transcript(&meeting_dir)?;

    let total = segments.len();
    if total == 0 {
        return Err(AppError::InvalidInput {
            context: format!(
                "meeting {} has an empty transcript; nothing to translate",
                meeting_id.0
            ),
        });
    }

    let mut pending: HashMap<usize, String> = HashMap::new();
    let mut last_flush = std::time::Instant::now();
    let mut result = Ok(());

    for (idx, segment) in segments.iter().enumerate() {
        let translated = match summariser
            .translate_segment(&segment.text, target_language)
            .map_err(|e| AppError::Internal {
                context: format!(
                    "translate_segment failed for segment {idx} of meeting {}: {e}",
                    meeting_id.0
                ),
            }) {
            Ok(t) => t,
            Err(e) => {
                result = Err(e);
                break;
            }
        };

        pending.insert(idx, translated);

        // Flush pending translations and emit progress on the same ~200 ms
        // cadence; always flush+emit on the last segment.
        let fraction = (idx + 1) as f32 / total as f32;
        let now = std::time::Instant::now();
        let is_last = idx + 1 == total;
        if is_last || now.duration_since(last_flush) >= std::time::Duration::from_millis(200) {
            last_flush = now;
            persistence::merge_translations(&meeting_dir, target_language, &pending)?;
            pending.clear();
            let _ = event_tx.send(AppEvent::OperationProgress {
                meeting_id,
                op: OperationKind::Translate,
                fraction: Some(fraction.clamp(0.0, 1.0)),
                label: format!("Translating… ({}/{total})", idx + 1),
            });
        }
    }

    // Flush any segments accumulated since the last throttled flush (covers the
    // error-exit path: partial progress on completed segments must survive).
    if !pending.is_empty() {
        persistence::merge_translations(&meeting_dir, target_language, &pending)?;
    }

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        language = target_language,
        segments = total,
        "translation loop complete"
    );

    result
}

/// Read the translations for a single meeting + language combination.
///
/// Returns a `HashMap<usize, String>` mapping segment index to translated text
/// for `target_language`, or an empty map when no translations exist yet for
/// that language (or when `translations.json` is absent). The webview calls this
/// on meeting open and on `TranslationReady` to populate the translated-view
/// overlay.
#[tauri::command]
#[specta::specta]
pub async fn get_translations(
    meeting_id: MeetingId,
    target_language: String,
    state: State<'_, IpcState>,
) -> Result<HashMap<usize, String>, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
        let mut all = persistence::read_translations(&meeting_dir)?;
        Ok::<_, AppError>(all.remove(&target_language).unwrap_or_default())
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("get_translations task join failed: {e}"),
    })?
    .map_err(IpcError::from)
}

/// Get one chat session for a meeting, or `None` when it does not exist.
#[tauri::command]
#[specta::specta]
pub async fn get_chat_session(
    meeting_id: MeetingId,
    session_id: ChatSessionId,
    state: State<'_, IpcState>,
) -> Result<Option<ChatSession>, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || ChatStore::load(&meetings_dir, meeting_id, session_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("get_chat_session task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

/// List all chat sessions for a meeting, most-recently-updated first.
#[tauri::command]
#[specta::specta]
pub async fn list_chat_sessions(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<Vec<ChatSession>, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || ChatStore::list(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("list_chat_sessions task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

/// Delete one chat session for a meeting (idempotent).
#[tauri::command]
#[specta::specta]
pub async fn delete_chat_session(
    meeting_id: MeetingId,
    session_id: ChatSessionId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || ChatStore::delete(&meetings_dir, meeting_id, session_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("delete_chat_session task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Chat command bodies — extracted so the State-free turn loop is unit-tested in
// `crate::chat` with a stub engine + stub tools (no model, no Tauri runtime).
// ---------------------------------------------------------------------------

/// Load the session named by `session_id` for `meeting_id`, or build a fresh one.
///
/// A given `(meeting_id, session_id)` that exists is loaded; otherwise a new
/// session with a fresh id is returned (and `session_id`, if it was supplied but
/// not found, is honoured so the webview's chosen id is kept). Blocking reads on
/// `spawn_blocking`.
pub(crate) async fn load_or_new_session(
    meetings_dir: &std::path::Path,
    meeting_id: Option<MeetingId>,
    session_id: Option<ChatSessionId>,
) -> Result<ChatSession, IpcError> {
    let now = chrono::Utc::now().to_rfc3339();

    if let (Some(mid), Some(sid)) = (meeting_id, session_id) {
        let dir = meetings_dir.to_path_buf();
        let existing = tokio::task::spawn_blocking(move || ChatStore::load(&dir, mid, sid))
            .await
            .map_err(|e| AppError::Internal {
                context: format!("load_or_new_session task join failed: {e}"),
            })?
            .map_err(IpcError::from)?;
        if let Some(session) = existing {
            return Ok(session);
        }
    }

    Ok(ChatSession {
        id: session_id.unwrap_or_default(),
        meeting_id,
        title: None,
        messages: Vec::new(),
        created_at: now.clone(),
        updated_at: now,
    })
}

/// Persist the full in-memory chat `session` (prior history + the user message +
/// the turn's produced messages) via [`ChatStore`].
///
/// The single-in-flight-turn guard (`chat_in_flight`) makes the running turn the
/// sole writer of this session, so we save the in-memory copy DIRECTLY rather
/// than reload-and-append. The earlier reload-and-append dropped the user message
/// entirely (it lived only in the in-memory `session`, which was never on disk).
/// A meeting-less session is not persisted (no folder to write into); the streamed
/// events already delivered the reply to the webview.
pub(crate) async fn persist_session(
    meetings_dir: &std::path::Path,
    meeting_id: Option<MeetingId>,
    mut session: ChatSession,
) {
    let Some(mid) = meeting_id else {
        return;
    };
    session.updated_at = chrono::Utc::now().to_rfc3339();
    let session_id = session.id;
    let dir = meetings_dir.to_path_buf();
    let result = tokio::task::spawn_blocking(move || ChatStore::save(&dir, mid, &session)).await;

    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(
            target: "ipc-bridge",
            session_id = %session_id.0,
            "persisting chat session failed: {e}"
        ),
        Err(join_err) => tracing::warn!(
            target: "ipc-bridge",
            "persist_session task join failed: {join_err}"
        ),
    }
}

/// Drive ONE chat turn on the held model (the `spawn_blocking` body).
///
/// Builds the real [`TurnEngine`] over a [`LlamaTurnBackend`] from the borrowed
/// held model, runs the State-free [`run_chat_turn`] loop, and dispatches each
/// tool call by re-entering async via `handle.block_on(registry.dispatch(...))`
/// — the only async/sync crossing (§4.5). Tokens + tool + completion events are
/// emitted on `event_tx` through the emit closure, which ALSO records the wire
/// messages this turn produced (the assistant final + each tool result) so the
/// caller can persist them. Returns those wire messages.
///
/// `mcp_gate` (S1) bounds the tool surface the turn may use to the MCP-allowed
/// set, REUSING the single policy in `agent-tools`
/// (`mcp_tool_descriptors_gated` / `mcp_call_allowed`):
/// - `None` — the internal UI chat: the full registry tool set, no gate.
/// - `Some(allow_writes)` — the Phase-10 inter-agent bridge: the model sees ONLY
///   the gated descriptors (so destructive ops like `retranscribe_meeting` /
///   `rediarize_meeting` are never offered), AND a non-allowed tool requested
///   anyway is rejected before dispatch as defence in depth — mirroring the
///   direct MCP `tools/call` path so a bridged external caller gets NO broader a
///   write surface than a direct MCP call under the same `mcp_write_tools`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_chat_turn_on_held_model(
    summariser: &LlamaSummariser,
    registry: &agent_tools::ToolRegistry,
    ctx: &ToolContext,
    handle: &tokio::runtime::Handle,
    session_id: ChatSessionId,
    turn_id: u64,
    system_prompt: &str,
    session: &ChatSession,
    event_tx: &broadcast::Sender<AppEvent>,
    cancel: &chat_agent::CancelFlag,
    mcp_gate: Option<bool>,
) -> Result<Vec<ChatMessage>, AppError> {
    let backend = LlamaTurnBackend::new(summariser.model(), LlamaTurnConfig::default());
    let engine = TurnEngine::new(backend);
    // The tool surface offered to the model: the full set for the UI path, or
    // the MCP-gated set for the inter-agent bridge (S1). The gating policy lives
    // in `agent-tools`; this only selects which projection to feed the engine.
    let mut descriptors = match mcp_gate {
        Some(allow_writes) => registry.mcp_tool_descriptors_gated(allow_writes),
        None => registry.descriptors(),
    };
    // Meeting-scoped chat: the context fills an omitted `meeting_id`
    // (`ToolContext::resolve_meeting`), so relax the schema's requiredness — else
    // a schema-respecting model treats `meeting_id` as a required field it lacks
    // and asks the user for it. Pairs with the prompt's "# Current meeting" scope.
    if ctx.default_meeting.is_some() {
        agent_tools::relax_meeting_id_requirement(&mut descriptors);
    }
    let cfg = chat_sampler_config();

    // Rebuild the engine-internal history: pinned system prompt + the prior
    // persisted messages (which include the just-appended user message).
    let mut history = initial_history(system_prompt);
    history.extend(session.messages.iter().map(engine_message_from_wire));
    // Everything the driver appends to `history` past this point is THIS turn's
    // output (the assistant final + each tool result).
    let prefix_len = history.len();

    // The emit closure just forwards each event to the bus. The persisted turn
    // messages are derived from the engine-history DELTA below (not from events),
    // so a `Tool` message persists the FULL machine payload (the engine's
    // `content`) rather than the one-line human `ChatToolResult.summary` — a
    // reloaded multi-turn session then feeds the model the same tool data it saw
    // live.
    let emit = |event: AppEvent| emit_chat_event(event_tx, event);

    // The dispatch closure: re-enter async for the registry dispatch only. On
    // the gated (bridge) path, REJECT a tool the MCP gate does not allow before
    // dispatching — defence in depth mirroring the direct MCP `tools/call` path
    // (`McpToolHandler::call_tool`), so a bridged caller cannot reach a tool
    // absent from its gated descriptor list even if the (stub or real) model
    // requests it by name (S1). Reuses `agent-tools`' `mcp_call_allowed`.
    let dispatch = |call: &chat_agent::ToolCall| -> AppResult<ToolOutput> {
        mcp_gate_check(registry, mcp_gate, &call.name)?;
        let args: serde_json::Value =
            serde_json::from_str(&call.arguments_json).map_err(|e| AppError::InvalidInput {
                context: format!("tool {} arguments are not valid JSON: {e}", call.name),
            })?;
        // Thread-occupancy cost (documented): this whole turn already runs on a
        // `spawn_blocking` thread (the engine decode is sync), and this nested
        // `block_on` parks THAT blocking thread for the full duration of the
        // async tool dispatch — including any in-tool `spawn_blocking` (e.g. a
        // `relisten`/`resummarise` inference). So one chat tool call holds a
        // blocking-pool thread end-to-end; concurrent tool-calling turns scale
        // with the blocking-pool size, not the worker count. Acceptable for v1
        // (single in-flight turn per session); revisit if the tool surface grows
        // long-running fan-out.
        handle.block_on(registry.dispatch(ctx, &call.name, args))
    };

    let outcome = run_chat_turn(
        &engine,
        session_id,
        turn_id,
        &mut history,
        &descriptors,
        &cfg,
        CHAT_N_CTX,
        cancel,
        dispatch,
        emit,
    );

    // The loop already emitted ChatError on failure; surface it for the caller's
    // log (the caller still persists the user message).
    outcome?;

    // Derive the turn's produced wire messages from the engine-history DELTA: the
    // assistant final + each tool result the driver appended. Tool messages carry
    // the FULL machine payload (the engine `content`) + the tool name, so a
    // reloaded session is faithful to what the model saw in-turn.
    Ok(wire_produced_from_delta(&history[prefix_len..], turn_id))
}

/// Apply the MCP write gate to one requested tool name before dispatch (S1).
///
/// `None` — the internal UI chat: no gate, every tool is allowed.
/// `Some(allow_writes)` — the inter-agent bridge: reject any tool the active gate
/// does not allow, REUSING the single policy in `agent-tools`
/// (`ToolRegistry::mcp_call_allowed`), exactly as the direct MCP `tools/call`
/// path does. Extracted so the bridge gate is unit-testable without a held model
/// (the S1 regression test in `crate::chat`).
pub(crate) fn mcp_gate_check(
    registry: &agent_tools::ToolRegistry,
    mcp_gate: Option<bool>,
    name: &str,
) -> Result<(), AppError> {
    if let Some(allow_writes) = mcp_gate {
        if !registry.mcp_call_allowed(name, allow_writes) {
            return Err(AppError::InvalidInput {
                context: format!("tool `{name}` is not exposed over MCP"),
            });
        }
    }
    Ok(())
}

/// Map the engine-history delta a turn produced (the assistant-tool_calls
/// message + each tool result + the assistant final, in order) into
/// persisted/wire [`ChatMessage`]s. Pure + unit-tested.
///
/// CQ1: the assistant-tool_calls message's `tool_calls` and each tool result's
/// `tool_call_id` are carried onto the wire shape so a reloaded multi-tool turn
/// reconstructs the valid `assistant(tool_calls) → tool(result)` sequence.
fn wire_produced_from_delta(
    new_engine_messages: &[chat_agent::ChatMessage],
    turn_id: u64,
) -> Vec<ChatMessage> {
    new_engine_messages
        .iter()
        .map(|m| ChatMessage {
            role: wire_role(m.role),
            content: m.content.clone(),
            tool_name: m.name.clone(),
            tool_call_id: m.tool_call_id.clone(),
            tool_calls: m
                .tool_calls
                .iter()
                .map(|c| minutist_common::ToolCallRecord {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    arguments_json: c.arguments_json.clone(),
                })
                .collect(),
            turn_id,
        })
        .collect()
}

/// The default chat sampler config (§6.4): a small-temperature sampling chain.
/// The driver injects a per-turn non-zero seed before each `run_turn`; the base
/// config's fixed `seed = 0` is never used on a non-greedy turn.
fn chat_sampler_config() -> chat_agent::SamplerConfig {
    chat_agent::SamplerConfig::default()
}

/// Emit one chat `AppEvent` on the shared broadcast sender (mirror of
/// [`emit_summary_ready`]). A send with no live subscribers is not an error.
fn emit_chat_event(event_tx: &broadcast::Sender<AppEvent>, event: AppEvent) {
    if event_tx.send(event).is_err() {
        tracing::trace!(target: "ipc-bridge", "chat event dropped (no subscribers)");
    }
}

/// Inner body of [`open_meeting`]: assemble the [`MeetingState`] from the
/// meeting folder under `meetings_dir`. Extracted so it can be unit-tested
/// without a Tauri runtime.
fn open_meeting_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Result<MeetingState, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    persistence::read_meeting_state(&meeting_dir)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::{MeetingId, NoteBlock};
    use persistence::{MeetingFolder, MeetingIndex};
    use tempfile::TempDir;

    /// `save_notes` → `load_notes` round-trip through a tempdir `meetings_dir`,
    /// exercising the command bodies directly (no Tauri runtime needed).
    #[test]
    fn save_then_load_round_trips_through_meetings_dir() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = MeetingId::new();
        // NotesStore writes into an *existing* meeting folder; create it via
        // the owning type so the layout matches production exactly.
        MeetingFolder::create(root, meeting_id).expect("create meeting folder");

        let notes_json = r#"{"type":"doc","content":[{"type":"paragraph","attrs":{"data-anchor-ms":1234},"content":[{"type":"text","text":"hello"}]}]}"#;
        let notes_markdown = "# Notes\n\nhello\n";

        save_notes_inner(root, meeting_id, notes_json, notes_markdown).expect("save");

        let loaded = load_notes_inner(root, meeting_id)
            .expect("load")
            .expect("notes present after save");

        // The markdown round-trips verbatim.
        assert_eq!(loaded.notes_markdown, notes_markdown);
        // The JSON round-trips structurally (re-serialised string may differ in
        // whitespace, so compare parsed values).
        let expected: serde_json::Value = serde_json::from_str(notes_json).unwrap();
        let actual: serde_json::Value = serde_json::from_str(&loaded.notes_json).unwrap();
        assert_eq!(actual, expected, "notes_json must round-trip structurally");
    }

    /// `load_notes` returns `None` when no notes have been saved for a meeting.
    #[test]
    fn load_returns_none_when_no_notes_saved() {
        let tempdir = TempDir::new().expect("tempdir");
        let meeting_id = MeetingId::new();
        let loaded = load_notes_inner(tempdir.path(), meeting_id).expect("load");
        assert!(loaded.is_none(), "absent notes must yield None");
    }

    /// Invalid `notes_json` is rejected as `AppError::InvalidInput`, not written.
    #[test]
    fn save_rejects_invalid_json() {
        let tempdir = TempDir::new().expect("tempdir");
        let meeting_id = MeetingId::new();
        MeetingFolder::create(tempdir.path(), meeting_id).expect("folder");
        let err = save_notes_inner(tempdir.path(), meeting_id, "not json", "")
            .expect_err("invalid JSON must error");
        assert!(matches!(err, AppError::InvalidInput { .. }));
    }

    /// `save_note_image`'s extension allowlist gate (`normalise_image_ext`):
    /// accepts the image set (case-/dot-insensitively), rejects everything else.
    #[test]
    fn normalise_image_ext_accepts_allowlist_rejects_others() {
        // Accepted, normalised to lower-cased, dot-less.
        for (input, expected) in [
            ("png", "png"),
            ("PNG", "png"),
            (".jpg", "jpg"),
            ("  .JPEG  ", "jpeg"),
            ("GIF", "gif"),
            ("webp", "webp"),
        ] {
            assert_eq!(
                normalise_image_ext(input).expect("allowed ext"),
                expected,
                "ext {input:?} should normalise to {expected:?}"
            );
        }
        // Rejected as InvalidInput — non-image / executable / path-y extensions.
        for evil in ["svg", "exe", "txt", "", "png.exe", "../png", "bmp"] {
            assert!(
                matches!(normalise_image_ext(evil), Err(AppError::InvalidInput { .. })),
                "ext {evil:?} must be rejected"
            );
        }
    }

    /// `save_note_asset` (the persistence body the command calls) round-trips an
    /// image and returns a portable bare-filename reference.
    #[test]
    fn save_note_asset_round_trips_via_meetings_dir() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = MeetingId::new();
        MeetingFolder::create(root, meeting_id).expect("folder");

        let bytes = b"\x89PNG\r\n\x1a\n-image-bytes".to_vec();
        let filename = persistence::save_note_asset(root, meeting_id, &bytes, "png").expect("save");
        // Portable ref: a bare filename, no separators.
        assert!(!filename.contains('/') && !filename.contains('\\'));
        assert!(filename.ends_with(".png"));

        let read = persistence::read_note_asset(root, meeting_id, &filename).expect("read");
        assert_eq!(read, bytes);
    }

    // -----------------------------------------------------------------------
    // Phase 4 meeting list/open/rename/delete round-trips (no Tauri runtime,
    // no model — a synthetic meeting folder + in-memory libsql index).
    // -----------------------------------------------------------------------

    use minutist_common::{AudioFormat, MeetingMeta, Segment};

    /// Write a synthetic meeting folder (`metadata.json` + optional
    /// `transcript.json`) under `root` and return its `MeetingId`. Mirrors the
    /// on-disk layout `persistence` produces so the readers + index agree.
    fn write_synthetic_meeting(
        root: &Path,
        title: &str,
        started_at: &str,
        first_segment_text: Option<&str>,
    ) -> MeetingId {
        let meeting_id = MeetingId::new();
        let folder = MeetingFolder::create(root, meeting_id).expect("create meeting folder");

        let meta = MeetingMeta {
            uuid: meeting_id,
            title: title.to_string(),
            started_at: started_at.to_string(),
            ended_at: Some(started_at.to_string()),
            duration_ms: 60_000,
            speaker_count: 1,
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
            app_version: "0.0.0".into(),
        };
        let meta_json = serde_json::to_vec_pretty(&meta).expect("serialise metadata");
        std::fs::write(folder.metadata_path(), meta_json).expect("write metadata.json");

        if let Some(text) = first_segment_text {
            let segments = vec![Segment {
                start_ms: 0,
                end_ms: 1_000,
                text: text.to_string(),
                speaker_id: None,
                confidence: None,
                words: Vec::new(),
                shared_speakers: Vec::new(),
            }];
            let seg_json = serde_json::to_vec_pretty(&segments).expect("serialise transcript");
            std::fs::write(folder.transcript_path(), seg_json).expect("write transcript.json");
        }

        meeting_id
    }

    /// Open an in-memory index seeded by rebuilding from the meeting folders.
    async fn seeded_index(meetings_root: &Path) -> MeetingIndex {
        let index = MeetingIndex::open(":memory:")
            .await
            .expect("open in-memory index");
        index
            .rebuild_from_disk(meetings_root)
            .await
            .expect("rebuild index from disk");
        index
    }

    /// `list_meetings` returns every indexed meeting, most-recent first, with the
    /// first transcript segment as the excerpt.
    #[tokio::test]
    async fn list_meetings_returns_indexed_rows_most_recent_first() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();

        let _older = write_synthetic_meeting(
            root,
            "Older meeting",
            "2026-06-01T09:00:00Z",
            Some("older excerpt"),
        );
        let _newer = write_synthetic_meeting(
            root,
            "Newer meeting",
            "2026-06-02T09:00:00Z",
            Some("newer excerpt"),
        );

        let index = seeded_index(root).await;
        let rows = index.list_meetings().await.expect("list_meetings");

        assert_eq!(rows.len(), 2, "both meetings must be indexed");
        assert_eq!(rows[0].title, "Newer meeting", "most-recent first");
        assert_eq!(rows[0].excerpt.as_deref(), Some("newer excerpt"));
        assert_eq!(rows[1].title, "Older meeting");
    }

    /// `open_meeting_inner` assembles a `MeetingState` matching what was written
    /// to the synthetic folder (metadata + transcript; no notes saved → None).
    #[test]
    fn open_meeting_returns_meeting_state_matching_disk() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = write_synthetic_meeting(
            root,
            "Launch sync",
            "2026-06-02T10:00:00Z",
            Some("hello world"),
        );

        let state = open_meeting_inner(root, meeting_id).expect("open_meeting");

        assert_eq!(state.meta.uuid, meeting_id);
        assert_eq!(state.meta.title, "Launch sync");
        assert_eq!(state.transcript.len(), 1);
        assert_eq!(state.transcript[0].text, "hello world");
        assert!(state.notes.is_none(), "no notes saved → None");
    }

    /// `open_meeting_inner` errors for a meeting folder that does not exist.
    #[test]
    fn open_meeting_errors_for_missing_meeting() {
        let tempdir = TempDir::new().expect("tempdir");
        let missing = MeetingId::new();
        let err =
            open_meeting_inner(tempdir.path(), missing).expect_err("missing meeting must error");
        assert!(matches!(err, AppError::Io { .. }));
    }

    /// `rename_meeting` rewrites `metadata.json` and refreshes the index row so a
    /// subsequent `list_meetings` shows the new title.
    #[tokio::test]
    async fn rename_meeting_updates_disk_and_index() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Old title", "2026-06-02T11:00:00Z", Some("excerpt"));

        let index = seeded_index(root).await;
        meeting_ops::rename_meeting(root, &index, meeting_id, "New title")
            .await
            .expect("rename");

        // On-disk metadata reflects the new title.
        let meeting_dir = root.join(meeting_id.0.to_string());
        let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
        assert_eq!(meta.title, "New title");

        // Index row reflects the new title.
        let rows = index.list_meetings().await.expect("list");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].title, "New title");
    }

    /// `set_speaker_name` writes the label→name mapping into `metadata.json` and
    /// returns the updated map; clearing with an empty name removes the entry.
    /// The index is not touched (speaker names live only in `metadata.json`).
    #[tokio::test]
    async fn set_speaker_name_persists_and_clears() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = write_synthetic_meeting(root, "Speakers", "2026-06-14T10:00:00Z", None);

        // Upsert label "A" → "Alice".
        let names = meeting_ops::set_speaker_name(root, meeting_id, "A", "Alice")
            .await
            .expect("set A");
        assert_eq!(names.get("A").map(String::as_str), Some("Alice"));

        // Confirm on-disk metadata reflects the name.
        let folder = root.join(meeting_id.0.to_string());
        let meta = persistence::read_metadata(&folder).expect("read metadata");
        assert_eq!(meta.speaker_names.get("A").map(String::as_str), Some("Alice"));

        // Add a second speaker name without clobbering the first.
        let names2 = meeting_ops::set_speaker_name(root, meeting_id, "B", "Bob")
            .await
            .expect("set B");
        assert_eq!(names2.get("A").map(String::as_str), Some("Alice"));
        assert_eq!(names2.get("B").map(String::as_str), Some("Bob"));

        // Clear "A" with an empty name.
        let names3 = meeting_ops::set_speaker_name(root, meeting_id, "A", "")
            .await
            .expect("clear A");
        assert!(!names3.contains_key("A"));
        assert_eq!(names3.get("B").map(String::as_str), Some("Bob"));
    }

    /// Stop-upsert (FR-33): after a meeting folder is written and its index row
    /// is upserted (the stop-equivalent the `stop_recording` command performs),
    /// `list_meetings` returns it **in the same session** — without a
    /// `rebuild_from_disk`. This is the in-session visibility guarantee that
    /// `Orchestrator::stop` alone does not provide (it finalises the folder but
    /// never touches the index).
    #[tokio::test]
    async fn stop_upsert_makes_meeting_visible_in_session() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();

        // A fresh, EMPTY in-memory index — no rebuild_from_disk. This models a
        // running session where the index was opened at startup and the meeting
        // recorded afterwards.
        let index = MeetingIndex::open(":memory:")
            .await
            .expect("open in-memory index");
        assert!(
            index.list_meetings().await.expect("list").is_empty(),
            "index must start empty (no rebuild)"
        );

        // Write a meeting folder + transcript, exactly as a finished recording
        // leaves on disk.
        let meeting_id = write_synthetic_meeting(
            root,
            "In-session meeting",
            "2026-06-02T13:00:00Z",
            Some("first words of the meeting"),
        );

        // The stop-equivalent: build the list entry from metadata + first
        // transcript segment, then upsert into the live index — exactly what the
        // `stop_recording` command does after `orchestrator.stop()`.
        let meta = persistence::read_metadata(&root.join(meeting_id.0.to_string()))
            .expect("read metadata");
        let entry = meeting_list_entry_for_meta(root, &meta);
        index.upsert(&entry).await.expect("upsert after stop");

        // list_meetings now returns the meeting in the SAME session.
        let rows = index.list_meetings().await.expect("list after upsert");
        assert_eq!(
            rows.len(),
            1,
            "stopped meeting must be visible without a rebuild"
        );
        assert_eq!(rows[0].id, meeting_id);
        assert_eq!(rows[0].title, "In-session meeting");
        assert_eq!(
            rows[0].excerpt.as_deref(),
            Some("first words of the meeting"),
            "excerpt must be the first transcript segment"
        );
    }

    /// `meeting_list_entry_for_meta` yields `excerpt: None` when the meeting has
    /// no transcript (a zero-segment meeting writes no `transcript.json`).
    #[test]
    fn stop_upsert_entry_has_no_excerpt_without_transcript() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Silent meeting", "2026-06-02T14:00:00Z", None);

        let meta = persistence::read_metadata(&root.join(meeting_id.0.to_string()))
            .expect("read metadata");
        let entry = meeting_list_entry_for_meta(root, &meta);

        assert_eq!(entry.id, meeting_id);
        assert_eq!(entry.title, "Silent meeting");
        assert!(entry.excerpt.is_none(), "no transcript → excerpt None");
    }

    /// `delete_meeting` removes the on-disk folder and the index row.
    #[tokio::test]
    async fn delete_meeting_removes_folder_and_index_row() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Doomed", "2026-06-02T12:00:00Z", Some("excerpt"));

        let index = seeded_index(root).await;
        assert_eq!(index.list_meetings().await.expect("list").len(), 1);

        meeting_ops::delete_meeting(root, &index, meeting_id)
            .await
            .expect("delete");

        let meeting_dir = root.join(meeting_id.0.to_string());
        assert!(!meeting_dir.exists(), "folder must be removed");
        assert!(
            index.list_meetings().await.expect("list").is_empty(),
            "index row must be removed"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 5 summary wiring (no model, no Tauri runtime). The summarise inner
    // path is driven by a `StubSummariser` so the read → summarise → write →
    // event wiring is exercised in CI without a ~3 GB GGUF — mirroring the
    // orchestrator's re_transcribe stub-backend seam.
    // -----------------------------------------------------------------------
    // `Summariser` is brought in via `use super::*` (the module-level
    // `#[cfg(test)]` import); no separate `use` needed here.

    /// A `common::Summariser` that returns a fixed markdown, recording the
    /// transcript length, notes markdown, and system prompt it was handed so the
    /// test can assert the inner path forwarded them.
    struct StubSummariser {
        fixed_markdown: String,
        seen_transcript_len: std::sync::Mutex<Option<usize>>,
        seen_speaker_ids: std::sync::Mutex<Option<Vec<Option<String>>>>,
        seen_notes: std::sync::Mutex<Option<String>>,
        seen_prompt: std::sync::Mutex<Option<String>>,
    }

    impl StubSummariser {
        fn new(markdown: &str) -> Self {
            Self {
                fixed_markdown: markdown.to_string(),
                seen_transcript_len: std::sync::Mutex::new(None),
                seen_speaker_ids: std::sync::Mutex::new(None),
                seen_notes: std::sync::Mutex::new(None),
                seen_prompt: std::sync::Mutex::new(None),
            }
        }
    }

    impl Summariser for StubSummariser {
        fn summarise(
            &self,
            transcript: &[Segment],
            notes: &[NoteBlock],
            system_prompt: &str,
        ) -> Result<String, AppError> {
            *self.seen_transcript_len.lock().unwrap() = Some(transcript.len());
            // Capture the per-segment speaker labels the inner path handed us so
            // a test can assert the speaker-name overlay was applied.
            *self.seen_speaker_ids.lock().unwrap() =
                Some(transcript.iter().map(|s| s.speaker_id.clone()).collect());
            // Capture the note text the inner path read (joined in document
            // order) — empty string when no notes were taken.
            let joined = notes
                .iter()
                .map(|n| n.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            *self.seen_notes.lock().unwrap() = Some(joined);
            *self.seen_prompt.lock().unwrap() = Some(system_prompt.to_string());
            Ok(self.fixed_markdown.clone())
        }
    }

    /// Save notes for a synthetic meeting as a Tiptap document with ONE paragraph
    /// per line of `text`, so [`persistence::read_note_blocks`] (which projects
    /// the `notes.json` paragraphs, #70) yields those lines as un-anchored
    /// [`NoteBlock`]s. Uses the same `NotesStore` path the `save_notes` command
    /// uses.
    fn write_synthetic_notes(root: &Path, meeting_id: MeetingId, text: &str) {
        let content: Vec<serde_json::Value> = text
            .lines()
            .map(|line| {
                serde_json::json!({
                    "type": "paragraph",
                    "content": [{ "type": "text", "text": line }],
                })
            })
            .collect();
        let value = serde_json::json!({ "type": "doc", "content": content });
        NotesStore::save(root, meeting_id, &value, text).expect("save notes");
    }

    /// `summarise_meeting_inner` reads the meeting's transcript + notes markdown,
    /// runs the stub summariser, writes `summary.md`, and (separately) the
    /// command emits `SummaryReady`. Here we assert the inner write + the event
    /// emission, since the inner fn and the emit helper compose what the command
    /// does without needing a model or Tauri runtime.
    #[test]
    fn summarise_inner_reads_writes_and_returns_markdown() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = write_synthetic_meeting(
            root,
            "Planning sync",
            "2026-06-02T15:00:00Z",
            Some("first agenda item"),
        );
        write_synthetic_notes(root, meeting_id, "- own the resume bug");

        let stub = StubSummariser::new("## Summary\n\nWe planned the sprint.\n");
        let prompt = "You are a meeting-notes assistant.";

        let returned = summarise_meeting_inner(root, meeting_id, &stub, prompt)
            .expect("summarise inner must succeed");

        // The returned markdown is the stub's fixed output.
        assert_eq!(returned, "## Summary\n\nWe planned the sprint.\n");

        // The stub saw the transcript + notes + prompt the inner path read.
        assert_eq!(*stub.seen_transcript_len.lock().unwrap(), Some(1));
        assert_eq!(
            stub.seen_notes.lock().unwrap().as_deref(),
            Some("- own the resume bug")
        );
        assert_eq!(stub.seen_prompt.lock().unwrap().as_deref(), Some(prompt));

        // `summary.md` is persisted and readable via the get-summary inner path.
        let loaded = get_summary_inner(root, meeting_id).expect("read summary");
        assert_eq!(
            loaded.as_deref(),
            Some("## Summary\n\nWe planned the sprint.\n"),
            "summary.md must be written by the inner path"
        );
    }

    /// The summariser must see user-set speaker names, not the raw diarizer
    /// labels: `summarise_meeting_inner` overlays `metadata.speaker_names` onto a
    /// transcript copy before handing it to the summariser, while the on-disk
    /// transcript keeps its raw labels.
    #[test]
    fn summarise_inner_overlays_speaker_names() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Sync", "2026-06-12T15:00:00Z", Some("hello"));
        let dir = root.join(meeting_id.0.to_string());

        // Label the single segment "A" and map "A" -> "Alice".
        let mut transcript = persistence::read_transcript(&dir).expect("read transcript");
        transcript[0].speaker_id = Some("A".to_string());
        persistence::write_transcript(&dir, &transcript).expect("write transcript");
        let mut meta = persistence::read_metadata(&dir).expect("read metadata");
        meta.speaker_names
            .insert("A".to_string(), "Alice".to_string());
        persistence::write_metadata(&dir, &meta).expect("write metadata");

        let stub = StubSummariser::new("ok");
        summarise_meeting_inner(root, meeting_id, &stub, "p").expect("summarise");

        // The stub saw the overlaid name, not the raw label.
        assert_eq!(
            stub.seen_speaker_ids.lock().unwrap().clone(),
            Some(vec![Some("Alice".to_string())]),
            "the overlay must rewrite the segment label to the display name"
        );
        // The on-disk transcript is untouched — still the raw label.
        let on_disk = persistence::read_transcript(&dir).expect("re-read transcript");
        assert_eq!(
            on_disk[0].speaker_id.as_deref(),
            Some("A"),
            "the overlay must not mutate the stored transcript"
        );
    }

    /// The full `summarise_meeting` wiring sans Tauri: inner write + the same
    /// `SummaryReady` event the command emits, observed on a broadcast
    /// subscriber — proving the event carries the right `meeting_id`.
    #[tokio::test]
    async fn summarise_emits_summary_ready_for_meeting() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = write_synthetic_meeting(
            root,
            "Standup",
            "2026-06-02T16:00:00Z",
            Some("status update"),
        );

        let (event_tx, mut event_rx) = broadcast::channel::<AppEvent>(8);
        let stub = StubSummariser::new("## Summary\n\nStandup notes.\n");

        let returned =
            summarise_meeting_inner(root, meeting_id, &stub, "prompt").expect("summarise");
        assert_eq!(returned, "## Summary\n\nStandup notes.\n");

        emit_summary_ready(&event_tx, meeting_id);

        let event = event_rx.recv().await.expect("an event must be broadcast");
        match event {
            AppEvent::SummaryReady { meeting_id: got } => assert_eq!(got, meeting_id),
            other => panic!("expected SummaryReady, got {other:?}"),
        }
    }

    /// Notes-free meeting: the inner path passes an empty notes markdown rather
    /// than erroring (FR-30 — a meeting with no notes still summarises).
    #[test]
    fn summarise_inner_handles_meeting_without_notes() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Quiet", "2026-06-02T17:00:00Z", Some("only words"));
        // No notes saved.

        let stub = StubSummariser::new("## Summary\n");
        summarise_meeting_inner(root, meeting_id, &stub, "prompt").expect("summarise");

        assert_eq!(
            stub.seen_notes.lock().unwrap().as_deref(),
            Some(""),
            "absent notes must pass an empty markdown string"
        );
    }

    /// `save_summary` → `get_summary` round-trip over a tempdir, exercising the
    /// command bodies directly (no Tauri runtime).
    #[test]
    fn save_then_get_summary_round_trips() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Edited", "2026-06-02T18:00:00Z", Some("words"));

        // No summary yet → None.
        assert!(
            get_summary_inner(root, meeting_id).expect("read").is_none(),
            "absent summary must read as None"
        );

        let edited = "## Summary\n\nUser-edited summary.\n";
        save_summary_inner(root, meeting_id, edited).expect("save summary");

        let loaded = get_summary_inner(root, meeting_id).expect("read summary");
        assert_eq!(loaded.as_deref(), Some(edited), "summary must round-trip");
    }

    /// `find_gguf_weights` picks the single text-weights `.gguf`, skipping an
    /// `mmproj-*` projector that may sit alongside it.
    #[test]
    fn find_gguf_weights_skips_mmproj() {
        let tempdir = TempDir::new().expect("tempdir");
        let dir = tempdir.path();
        std::fs::write(dir.join("gemma-4-E4B-it-Q4_K_M.gguf"), b"weights").expect("write weights");
        std::fs::write(dir.join("mmproj-gemma-4-E4B-it.gguf"), b"proj").expect("write proj");
        std::fs::write(dir.join("README.md"), b"notes").expect("write readme");

        let found = find_gguf_weights(dir).expect("must find the text weights");
        assert_eq!(
            found.file_name().and_then(|n| n.to_str()),
            Some("gemma-4-E4B-it-Q4_K_M.gguf")
        );
    }

    /// `find_gguf_weights` errors (rather than panicking) when no weights file
    /// is present.
    #[test]
    fn find_gguf_weights_errors_when_absent() {
        let tempdir = TempDir::new().expect("tempdir");
        let dir = tempdir.path();
        std::fs::write(dir.join("mmproj-only.gguf"), b"proj").expect("write proj");

        let err = find_gguf_weights(dir).expect_err("no text weights → error");
        assert!(matches!(err, AppError::ModelLoad { .. }));
    }

    // -----------------------------------------------------------------------
    // LLM model-id resolution (Phase 5) — the settings-override / bundled
    // default decision, unit-tested without a Tauri runtime or orchestrator.
    // -----------------------------------------------------------------------

    /// A set `settings.llm_model_id` resolves to that id (the user override
    /// wins over the bundled default).
    #[test]
    fn resolve_llm_model_id_honours_settings_override() {
        let settings = Settings {
            llm_model_id: Some(ModelId::from("granite-4.1-3b-q4_k_m")),
            ..Settings::default()
        };
        assert_eq!(
            resolve_llm_model_id(&settings),
            ModelId::from("granite-4.1-3b-q4_k_m")
        );
    }

    /// GPU off (`gpu_acceleration = false`) MUST force CPU (`0`); GPU on MUST
    /// resolve to the compile-time ceiling — itself `0` in the default CPU-only
    /// build, so a CPU build is unaffected by the flag. Pure, no model.
    #[test]
    fn resolve_summariser_gpu_layers_off_forces_cpu() {
        assert_eq!(
            resolve_summariser_gpu_layers(false),
            0,
            "GPU off must force CPU"
        );

        let on = resolve_summariser_gpu_layers(true);
        assert_eq!(
            on,
            summariser::gpu_layers(),
            "GPU on must use the compile-time ceiling"
        );
        if cfg!(any(
            feature = "vulkan",
            feature = "metal",
            feature = "cuda",
            feature = "rocm"
        )) {
            assert_eq!(
                on,
                u32::MAX,
                "a GPU-feature build offloads all layers when on"
            );
        } else {
            assert_eq!(on, 0, "a default CPU-only build stays on CPU even when on");
        }
    }

    /// An unset `settings.llm_model_id` falls back to the bundled default.
    #[test]
    fn resolve_llm_model_id_falls_back_to_default() {
        let settings = Settings {
            llm_model_id: None,
            ..Settings::default()
        };
        assert_eq!(
            resolve_llm_model_id(&settings),
            ModelId::from(DEFAULT_LLM_MODEL_ID)
        );
    }

    // -----------------------------------------------------------------------
    // Gated real-model test — skips when MINUTIST_LLM_MODEL_PATH is unset.
    //
    // To run:
    //   MINUTIST_LLM_MODEL_PATH=/path/to/gemma-4-E4B-it-Q4_K_M.gguf \
    //   cargo test -p ipc-bridge -- --include-ignored
    // -----------------------------------------------------------------------

    /// End-to-end summarise over a synthetic meeting folder using the **real**
    /// Gemma-4 GGUF pointed to by `MINUTIST_LLM_MODEL_PATH`: open the model,
    /// run `summarise_meeting_inner`, assert a non-empty markdown summary is
    /// written, and record latency. No-op skip when the env var is unset.
    #[test]
    #[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
    fn summarise_real_model_writes_non_empty_summary() {
        let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => return, // no-op skip path
        };

        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = write_synthetic_meeting(
            root,
            "Gated meeting",
            "2026-06-02T19:00:00Z",
            Some("Let's review the quarterly plan and assign action items."),
        );
        write_synthetic_notes(root, meeting_id, "- Decision: ship Phase 5");

        let summariser = LlamaSummariser::open(
            std::path::PathBuf::from(&model_path),
            SummariserConfig::default(),
        )
        .expect("model load must succeed with a valid path");

        let prompt =
            "You are a meeting-notes assistant. Produce a concise markdown summary with headings.";

        let start = std::time::Instant::now();
        let summary = summarise_meeting_inner(root, meeting_id, &summariser, prompt)
            .expect("summarise must succeed");
        let elapsed = start.elapsed();

        tracing::info!(
            target: "ipc-bridge",
            elapsed_ms = elapsed.as_millis() as u64,
            summary_len = summary.len(),
            "gated summarise_meeting complete"
        );

        assert!(!summary.trim().is_empty(), "summary must be non-empty");

        // `summary.md` must be on disk and match what was returned.
        let loaded = get_summary_inner(root, meeting_id)
            .expect("read summary")
            .expect("summary.md must exist after summarise");
        assert_eq!(loaded, summary, "persisted summary must match returned");
    }

    /// End-to-end translation pass over a 3-segment meeting using the real
    /// `MINUTIST_LLM_MODEL_PATH` model: verifies that all three translations are
    /// written to `translations.json` after `translate_meeting_blocking` returns,
    /// exercising the batched-flush path. Skipped in CI when the env var is unset.
    #[test]
    #[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
    fn translate_meeting_blocking_writes_all_segments_to_sidecar() {
        let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => return,
        };

        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = MeetingId::new();
        let folder = persistence::MeetingFolder::create(root, meeting_id).expect("folder");

        // Write a 3-segment transcript.
        let segments = vec![
            Segment { start_ms: 0,    end_ms: 1000, text: "Hello world.".into(),        speaker_id: None, confidence: None, words: vec![], shared_speakers: vec![] },
            Segment { start_ms: 1000, end_ms: 2000, text: "This is a test.".into(),     speaker_id: None, confidence: None, words: vec![], shared_speakers: vec![] },
            Segment { start_ms: 2000, end_ms: 3000, text: "Goodbye for now.".into(),    speaker_id: None, confidence: None, words: vec![], shared_speakers: vec![] },
        ];
        let seg_json = serde_json::to_vec_pretty(&segments).expect("serialise");
        std::fs::write(folder.transcript_path(), seg_json).expect("write transcript");

        let summariser = LlamaSummariser::open(
            std::path::PathBuf::from(&model_path),
            SummariserConfig::default(),
        )
        .expect("model load");

        let (event_tx, _event_rx) = broadcast::channel::<AppEvent>(8);
        let meetings_dir = root.to_path_buf();

        translate_meeting_blocking(
            &meetings_dir,
            meeting_id,
            "Spanish",
            &summariser,
            &event_tx,
        )
        .expect("translation must succeed");

        // All three segments must be in the sidecar.
        let meeting_dir = root.join(meeting_id.0.to_string());
        let all = persistence::read_translations(&meeting_dir).expect("read translations");
        let spanish = all.get("Spanish").expect("Spanish key present");
        assert_eq!(spanish.len(), 3, "all 3 segments must be persisted");
        assert!(!spanish[&0].trim().is_empty(), "segment 0 must be non-empty");
        assert!(!spanish[&1].trim().is_empty(), "segment 1 must be non-empty");
        assert!(!spanish[&2].trim().is_empty(), "segment 2 must be non-empty");
    }

    // -----------------------------------------------------------------------
    // Post-stop background orchestration (review Step 5): gating + ordering +
    // per-pass error tolerance, verified WITHOUT a Tauri runtime or a real
    // orchestrator. `post_stop_passes` is pure; `run_post_stop_passes` is driven
    // by a recording closure that injects per-pass results.
    // -----------------------------------------------------------------------

    /// Gating + ordering: no flags → no passes; each flag adds its pass; all →
    /// re-transcribe BEFORE diarize BEFORE summarise (so diarize labels the
    /// repaired transcript and summarise sees the final, #68).
    #[test]
    fn post_stop_passes_gates_and_orders() {
        assert_eq!(post_stop_passes(false, false, false), vec![]);
        assert_eq!(
            post_stop_passes(true, false, false),
            vec![PostStopPass::ReTranscribe]
        );
        assert_eq!(
            post_stop_passes(false, true, false),
            vec![PostStopPass::Rediarize]
        );
        assert_eq!(
            post_stop_passes(false, false, true),
            vec![PostStopPass::Summarise]
        );
        assert_eq!(
            post_stop_passes(true, true, true),
            vec![
                PostStopPass::ReTranscribe,
                PostStopPass::Rediarize,
                PostStopPass::Summarise
            ],
            "re-transcribe → diarize → summarise (summarise sees the final transcript)"
        );
    }

    /// #68 — when auto-summarise is on, the plan ends with `Summarise` (after any
    /// re-transcribe / diarize), and when off it is absent.
    #[test]
    fn post_stop_passes_appends_summarise_when_enabled() {
        // Auto-summarise alone (no re-transcribe, no diarize).
        assert_eq!(
            post_stop_passes(false, false, true),
            vec![PostStopPass::Summarise],
            "auto-summarise on must add the summarise pass"
        );
        // Summarise is always LAST.
        assert_eq!(
            *post_stop_passes(true, false, true).last().unwrap(),
            PostStopPass::Summarise
        );
        // Off → never planned.
        assert!(
            !post_stop_passes(true, true, false).contains(&PostStopPass::Summarise),
            "auto-summarise off must omit the summarise pass"
        );
    }

    /// An empty plan invokes `run_pass` zero times (no background work).
    #[tokio::test]
    async fn run_post_stop_passes_noop_when_empty() {
        let mut calls: Vec<PostStopPass> = Vec::new();
        run_post_stop_passes(&[], MeetingId::new(), |pass| {
            calls.push(pass);
            async { Ok(()) }
        })
        .await;
        assert!(calls.is_empty(), "no passes → run_pass never called");
    }

    /// All planned passes run, in order, when each succeeds — including the #68
    /// auto-summarise pass running LAST.
    #[tokio::test]
    async fn run_post_stop_passes_runs_all_in_order() {
        let passes = post_stop_passes(true, true, true);
        let mut calls: Vec<PostStopPass> = Vec::new();
        run_post_stop_passes(&passes, MeetingId::new(), |pass| {
            calls.push(pass);
            async { Ok(()) }
        })
        .await;
        assert_eq!(
            calls,
            vec![
                PostStopPass::ReTranscribe,
                PostStopPass::Rediarize,
                PostStopPass::Summarise
            ]
        );
    }

    /// #68 — the post-stop chain AUTO-SUMMARISES when `auto_summarise_on_stop` is
    /// on: the planned passes include `Summarise`, and the `run_pass` closure
    /// (here a stub summariser writing `summary.md` + emitting `SummaryReady`)
    /// runs it. Verified WITHOUT a Tauri runtime, a real model, or a real
    /// orchestrator — the held-summarise side effects are exercised via the
    /// trait-based [`summarise_meeting_inner`] + [`emit_summary_ready`] seam.
    #[tokio::test]
    async fn post_stop_chain_auto_summarises_when_enabled() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = write_synthetic_meeting(
            root,
            "Auto-summarised",
            "2026-06-10T09:00:00Z",
            Some("the agenda item"),
        );

        // Gating: auto-summarise on, nothing else → exactly the summarise pass.
        let passes = post_stop_passes(false, false, /* auto_summarise */ true);
        assert_eq!(passes, vec![PostStopPass::Summarise]);

        let (event_tx, mut event_rx) = broadcast::channel::<AppEvent>(16);
        let stub = StubSummariser::new("## Summary\n\nAuto-generated on stop.\n");

        // Drive the chain. The `Summarise` arm runs the SAME read → summarise →
        // write → `SummaryReady` work `run_held_summarise` performs, but through
        // the model-free stub seam so CI needs no GGUF.
        run_post_stop_passes(&passes, meeting_id, |pass| {
            let event_tx = event_tx.clone();
            let stub = &stub;
            async move {
                match pass {
                    PostStopPass::Summarise => {
                        summarise_meeting_inner(root, meeting_id, stub, "prompt")?;
                        emit_summary_ready(&event_tx, meeting_id);
                        Ok(())
                    }
                    other => panic!("unexpected pass {other:?}"),
                }
            }
        })
        .await;

        // `summary.md` was written by the auto-summarise pass.
        let written = get_summary_inner(root, meeting_id).expect("read summary");
        assert_eq!(
            written.as_deref(),
            Some("## Summary\n\nAuto-generated on stop.\n"),
            "auto-summarise must write summary.md after stop"
        );
        // And `SummaryReady` was emitted for the meeting (clears the UI indicator).
        let event = event_rx.recv().await.expect("a SummaryReady event");
        match event {
            AppEvent::SummaryReady { meeting_id: got } => assert_eq!(got, meeting_id),
            other => panic!("expected SummaryReady, got {other:?}"),
        }
    }

    /// #68 — when `auto_summarise_on_stop` is off, the plan omits `Summarise`, so
    /// the chain never invokes the summarise pass (no `summary.md`, no
    /// `SummaryReady`).
    #[tokio::test]
    async fn post_stop_chain_skips_summarise_when_disabled() {
        let passes = post_stop_passes(false, false, /* auto_summarise */ false);
        assert!(passes.is_empty(), "auto-summarise off + nothing else → no passes");

        let mut summarise_calls = 0u32;
        run_post_stop_passes(&passes, MeetingId::new(), |pass| {
            if pass == PostStopPass::Summarise {
                summarise_calls += 1;
            }
            async { Ok(()) }
        })
        .await;
        assert_eq!(summarise_calls, 0, "summarise pass must not run when disabled");
    }

    /// A failed re-transcribe (InvalidInput = busy, OR any other error) is
    /// tolerated and does NOT prevent the diarize pass from being attempted —
    /// the recording is already safely persisted.
    #[tokio::test]
    async fn run_post_stop_passes_failure_does_not_abort_later_passes() {
        for first_err in [
            AppError::InvalidInput {
                context: "busy".into(),
            },
            AppError::Internal {
                context: "boom".into(),
            },
        ] {
            let passes = post_stop_passes(true, true, false);
            let mut calls: Vec<PostStopPass> = Vec::new();
            // Move the error into the closure via an Option taken on first call.
            let mut first_err = Some(first_err);
            run_post_stop_passes(&passes, MeetingId::new(), |pass| {
                calls.push(pass);
                let result = match pass {
                    PostStopPass::ReTranscribe => {
                        Err(first_err.take().expect("one re-transcribe call"))
                    }
                    PostStopPass::Rediarize | PostStopPass::Summarise => Ok(()),
                };
                async move { result }
            })
            .await;
            assert_eq!(
                calls,
                vec![PostStopPass::ReTranscribe, PostStopPass::Rediarize],
                "diarize must still be attempted after a failed re-transcribe"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Chat-turn persistence (regression guards for the IMPL-4a review CRITICAL +
    // WARNING-2): the user message must be persisted, and a Tool message must
    // persist the FULL machine payload (not the one-line event summary).
    // -----------------------------------------------------------------------

    /// `wire_produced_from_delta` maps the engine-history delta to wire messages
    /// carrying the FULL tool payload + the tool name (not the lossy summary),
    /// AND (CQ1) the assistant-tool_calls carrier + the tool result's
    /// tool_call_id so a reloaded multi-tool turn is a valid OpenAI sequence.
    #[test]
    fn wire_produced_from_delta_keeps_full_tool_payload() {
        let call = chat_agent::ToolCall {
            id: "call_1".to_string(),
            name: "get_summary".to_string(),
            arguments_json: "{}".to_string(),
        };
        let delta = vec![
            chat_agent::ChatMessage::assistant_tool_calls("", vec![call]),
            chat_agent::ChatMessage::tool_result(
                "call_1",
                "get_summary",
                r#"{"decisions":["ship Phase 9"]}"#,
            ),
            chat_agent::ChatMessage::assistant("We decided to ship Phase 9."),
        ];
        let wire = wire_produced_from_delta(&delta, 3);
        assert_eq!(wire.len(), 3);
        // CQ1: the assistant-tool_calls message carries the OpenAI tool-call
        // carrier (id/name/arguments), preceding the tool result.
        assert_eq!(wire[0].role, ChatRole::Assistant);
        assert_eq!(wire[0].tool_calls.len(), 1);
        assert_eq!(wire[0].tool_calls[0].id, "call_1");
        assert_eq!(wire[0].tool_calls[0].name, "get_summary");
        // The tool result carries the matching tool_call_id + full payload.
        assert_eq!(wire[1].role, ChatRole::Tool);
        assert_eq!(wire[1].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(wire[1].tool_name.as_deref(), Some("get_summary"));
        assert!(
            wire[1].content.contains("decisions"),
            "the full machine payload must be persisted, not a one-line summary"
        );
        assert_eq!(wire[1].turn_id, 3);
        assert_eq!(wire[2].role, ChatRole::Assistant);
        assert_eq!(wire[2].content, "We decided to ship Phase 9.");
        assert!(wire[2].tool_name.is_none());
        assert!(wire[2].tool_calls.is_empty());
    }

    /// `persist_session` saves the WHOLE in-memory session — including the user
    /// message (the IMPL-4a CRITICAL: it was previously dropped) and the
    /// full-payload tool message — without a reload-and-append.
    #[tokio::test]
    async fn persist_session_round_trips_user_and_full_tool_payload() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = MeetingId::new();
        MeetingFolder::create(root, meeting_id).expect("meeting folder");
        let sid = ChatSessionId::new();

        let session = ChatSession {
            id: sid,
            meeting_id: Some(meeting_id),
            title: None,
            messages: vec![
                ChatMessage {
                    role: ChatRole::User,
                    content: "what was decided?".into(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    turn_id: 0,
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    content: r#"{"decisions":["ship"]}"#.into(),
                    tool_name: Some("get_summary".into()),
                    tool_call_id: Some("call_1".into()),
                    tool_calls: Vec::new(),
                    turn_id: 0,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "We decided to ship.".into(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    turn_id: 0,
                },
            ],
            created_at: "2026-06-10T00:00:00Z".into(),
            updated_at: "2026-06-10T00:00:00Z".into(),
        };

        persist_session(root, Some(meeting_id), session).await;

        let loaded = ChatStore::load(root, meeting_id, sid)
            .expect("load")
            .expect("session must be persisted");
        // CRITICAL regression guard: the user message survives.
        assert!(
            loaded
                .messages
                .iter()
                .any(|m| m.role == ChatRole::User && m.content == "what was decided?"),
            "the user message must be persisted"
        );
        // WARNING-2 guard: the Tool message keeps the full payload + tool name.
        let tool = loaded
            .messages
            .iter()
            .find(|m| m.role == ChatRole::Tool)
            .expect("a tool message must be persisted");
        assert!(
            tool.content.contains("decisions"),
            "full tool payload persisted"
        );
        assert_eq!(tool.tool_name.as_deref(), Some("get_summary"));
        assert_eq!(loaded.messages.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Chat prompt scoping — the agent must not ask for a meeting id when a
    // meeting is open (it has `default_meeting` in scope).
    // -----------------------------------------------------------------------

    #[test]
    fn chat_prompt_scopes_to_the_open_meeting() {
        let mid = MeetingId::new();
        let p = chat_system_prompt_for_meeting("BASE", Some(mid), Some("Standup"));
        assert!(p.starts_with("BASE"), "base prompt is preserved");
        assert!(p.contains("# Current meeting"));
        assert!(p.contains(&mid.0.to_string()), "the meeting id is named");
        assert!(p.contains("titled \"Standup\""));
        assert!(
            p.contains("NEVER ask the user"),
            "must instruct the agent not to ask for a meeting id"
        );
    }

    #[test]
    fn chat_prompt_without_title_omits_the_titled_clause() {
        let mid = MeetingId::new();
        let p = chat_system_prompt_for_meeting("BASE", Some(mid), None);
        assert!(p.contains(&mid.0.to_string()));
        assert!(!p.contains("titled"));
        // A blank/whitespace title is treated as no title.
        let blank = chat_system_prompt_for_meeting("BASE", Some(mid), Some("   "));
        assert!(!blank.contains("titled"));
    }

    #[test]
    fn chat_prompt_meeting_less_is_unchanged() {
        assert_eq!(
            chat_system_prompt_for_meeting("BASE", None, Some("ignored")),
            "BASE"
        );
    }

    // -----------------------------------------------------------------------
    // apply_output_language — the prompt-injection helper
    // -----------------------------------------------------------------------

    #[test]
    fn apply_output_language_appends_instruction_for_known_name() {
        let result = apply_output_language("Do the thing.", "French");
        assert_eq!(result, "Do the thing.\n\nRespond entirely in French.");
    }

    #[test]
    fn apply_output_language_no_op_for_auto_with_unmappable_locale() {
        // When "auto" cannot resolve (sys_locale not available in the test
        // sandbox, or the locale maps to a known language), the helper either
        // appends a language or returns the prompt unchanged. We test the
        // explicit-name path separately; for "auto" we just verify no panic.
        let result = apply_output_language("Base prompt.", "auto");
        // Result is either the base prompt (auto → None) or extended. Either
        // is valid — we only assert the base is preserved.
        assert!(result.starts_with("Base prompt."));
    }

    #[test]
    fn apply_output_language_no_op_for_empty_setting() {
        let result = apply_output_language("Base prompt.", "");
        assert_eq!(result, "Base prompt.");
    }

    #[test]
    fn apply_output_language_explicit_name_appended_after_custom_prompt() {
        // The instruction is appended LAST — even if a custom prompt already
        // says something about language, the explicit setting wins.
        let result = apply_output_language("Respond in English only.", "German");
        assert!(
            result.ends_with("\n\nRespond entirely in German."),
            "output-language instruction must be appended after the full prompt"
        );
    }
}
