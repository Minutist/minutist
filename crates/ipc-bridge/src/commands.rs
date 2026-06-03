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

use std::path::Path;

use meeting_app_common::{
    AppError, AppEvent, AudioDevice, MeetingId, MeetingListEntry, MeetingMeta, MeetingState,
    ModelId, ModelStatus, NotesDocument, RecordingState, Summariser,
};
use persistence::{meeting_ops, NotesStore};
use settings::Settings;
use summariser::{LlamaSummariser, SummariserConfig};
use tauri::State;
use tokio::sync::broadcast;

use crate::{error::IpcError, IpcState};

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
fn resolve_llm_model_id(settings: &Settings) -> ModelId {
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

    Ok(meta)
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
pub async fn get_recording_state(
    state: State<'_, IpcState>,
) -> Result<RecordingState, IpcError> {
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
pub async fn ensure_model(
    model_id: ModelId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
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
    let settings = state.settings.current();
    let model_id = resolve_llm_model_id(&settings);

    // Resolve (download if needed) the model directory on the async worker.
    let model_dir = state
        .orchestrator
        .ensure_model_path(&model_id)
        .await
        .map_err(IpcError::from)?;

    let meetings_dir = state.meetings_dir.clone();
    let system_prompt = settings.summary_system_prompt.clone();

    // Heavy, synchronous summariser work on a blocking thread. The summariser
    // is constructed inside the closure so the GGUF load happens off the async
    // worker threads.
    let summary_md = tokio::task::spawn_blocking(move || {
        let summariser = open_summariser_in_dir(&model_dir)?;
        summarise_meeting_inner(&meetings_dir, meeting_id, &summariser, &system_prompt)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("summarise_meeting task join failed: {e}"),
    })?
    .map_err(IpcError::from)?;

    // Emit SummaryReady so the webview re-reads `summary.md`.
    emit_summary_ready(&state.event_tx, meeting_id);

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        summary_len = summary_md.len(),
        "summary generated and persisted"
    );

    Ok(())
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
fn summarise_meeting_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    summariser: &dyn Summariser,
    system_prompt: &str,
) -> Result<String, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());

    let transcript = persistence::read_transcript(&meeting_dir)?;
    // The notes markdown comes from the assembled meeting state (notes are
    // optional — an empty string when the meeting has none).
    let notes_markdown = persistence::read_meeting_state(&meeting_dir)?
        .notes
        .map(|n| n.notes_markdown)
        .unwrap_or_default();

    let summary_md = summariser.summarise(&transcript, &notes_markdown, system_prompt)?;

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

/// Open a [`LlamaSummariser`] over the single `.gguf` weights file in
/// `model_dir`, skipping any `mmproj-*` multimodal projector.
///
/// The LLM is text-only (the bundled Gemma 4 GGUF ships without a projector),
/// but the helper defends against a directory that also contains an `mmproj-*`
/// file so the wrong file is never loaded. A missing or ambiguous weights file
/// is an `AppError::ModelLoad`.
fn open_summariser_in_dir(model_dir: &Path) -> Result<LlamaSummariser, AppError> {
    let gguf_path = find_gguf_weights(model_dir)?;
    LlamaSummariser::open(gguf_path, SummariserConfig::default())
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
    use meeting_app_common::MeetingId;
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

    // -----------------------------------------------------------------------
    // Phase 4 meeting list/open/rename/delete round-trips (no Tauri runtime,
    // no model — a synthetic meeting folder + in-memory libsql index).
    // -----------------------------------------------------------------------

    use meeting_app_common::{AudioFormat, MeetingMeta, Segment};

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
        let meeting_id =
            write_synthetic_meeting(root, "Launch sync", "2026-06-02T10:00:00Z", Some("hello world"));

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
        let err = open_meeting_inner(tempdir.path(), missing).expect_err("missing meeting must error");
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
        assert_eq!(rows.len(), 1, "stopped meeting must be visible without a rebuild");
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

    use meeting_app_common::Summariser;

    /// A `common::Summariser` that returns a fixed markdown, recording the
    /// transcript length, notes markdown, and system prompt it was handed so the
    /// test can assert the inner path forwarded them.
    struct StubSummariser {
        fixed_markdown: String,
        seen_transcript_len: std::sync::Mutex<Option<usize>>,
        seen_notes: std::sync::Mutex<Option<String>>,
        seen_prompt: std::sync::Mutex<Option<String>>,
    }

    impl StubSummariser {
        fn new(markdown: &str) -> Self {
            Self {
                fixed_markdown: markdown.to_string(),
                seen_transcript_len: std::sync::Mutex::new(None),
                seen_notes: std::sync::Mutex::new(None),
                seen_prompt: std::sync::Mutex::new(None),
            }
        }
    }

    impl Summariser for StubSummariser {
        fn summarise(
            &self,
            transcript: &[Segment],
            notes_markdown: &str,
            system_prompt: &str,
        ) -> Result<String, AppError> {
            *self.seen_transcript_len.lock().unwrap() = Some(transcript.len());
            *self.seen_notes.lock().unwrap() = Some(notes_markdown.to_string());
            *self.seen_prompt.lock().unwrap() = Some(system_prompt.to_string());
            Ok(self.fixed_markdown.clone())
        }
    }

    /// Save notes for a synthetic meeting via the same `NotesStore` path the
    /// `save_notes` command uses, so `read_meeting_state(..).notes` is populated.
    fn write_synthetic_notes(root: &Path, meeting_id: MeetingId, markdown: &str) {
        let value: serde_json::Value = serde_json::json!({ "type": "doc", "content": [] });
        NotesStore::save(root, meeting_id, &value, markdown).expect("save notes");
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

    /// The full `summarise_meeting` wiring sans Tauri: inner write + the same
    /// `SummaryReady` event the command emits, observed on a broadcast
    /// subscriber — proving the event carries the right `meeting_id`.
    #[tokio::test]
    async fn summarise_emits_summary_ready_for_meeting() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Standup", "2026-06-02T16:00:00Z", Some("status update"));

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
    // Gated real-model test — skips when MEETING_APP_LLM_MODEL_PATH is unset.
    //
    // To run:
    //   MEETING_APP_LLM_MODEL_PATH=/path/to/gemma-4-E4B-it-Q4_K_M.gguf \
    //   cargo test -p ipc-bridge -- --include-ignored
    // -----------------------------------------------------------------------

    /// End-to-end summarise over a synthetic meeting folder using the **real**
    /// Gemma-4 GGUF pointed to by `MEETING_APP_LLM_MODEL_PATH`: open the model,
    /// run `summarise_meeting_inner`, assert a non-empty markdown summary is
    /// written, and record latency. No-op skip when the env var is unset.
    #[test]
    #[ignore = "requires MEETING_APP_LLM_MODEL_PATH"]
    fn summarise_real_model_writes_non_empty_summary() {
        let model_path = match std::env::var("MEETING_APP_LLM_MODEL_PATH") {
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
}
