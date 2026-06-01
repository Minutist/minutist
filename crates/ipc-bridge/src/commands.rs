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

use meeting_app_common::{
    AppError, AudioDevice, MeetingId, MeetingMeta, ModelId, ModelStatus, RecordingState,
};
use persistence::NotesStore;
use serde::{Deserialize, Serialize};
use settings::Settings;
use specta::Type;
use tauri::State;

use crate::{error::IpcError, IpcState};

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
#[tauri::command]
#[specta::specta]
pub async fn stop_recording(state: State<'_, IpcState>) -> Result<MeetingMeta, IpcError> {
    state.orchestrator.stop().await.map_err(IpcError::from)
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

/// A persisted notes document returned by [`load_notes`].
///
/// `notes_json` carries the Tiptap/ProseMirror document **as a `String`**, not
/// a `serde_json::Value`: a bare `serde_json::Value` does not derive
/// `specta::Type`, so it cannot cross the tauri-specta boundary directly. The
/// webview owns the (de)serialisation of this opaque document; `persistence`
/// stores it verbatim (the Phase-4 transcript-chip opacity guarantee). The
/// `String`-over-the-wire choice keeps the IPC contract typed without forcing a
/// Rust-side Tiptap model.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct NotesDoc {
    pub notes_json: String,
    pub notes_markdown: String,
}

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
/// [`NotesDoc`]).
#[tauri::command]
#[specta::specta]
pub async fn load_notes(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<Option<NotesDoc>, IpcError> {
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
) -> Result<Option<NotesDoc>, AppError> {
    let loaded = NotesStore::load(meetings_dir, meeting_id)?;
    match loaded {
        None => Ok(None),
        Some(data) => {
            let notes_json = serde_json::to_string(&data.json).map_err(|e| AppError::Internal {
                context: format!("failed to re-serialise loaded notes.json: {e}"),
            })?;
            Ok(Some(NotesDoc {
                notes_json,
                notes_markdown: data.markdown,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use meeting_app_common::MeetingId;
    use persistence::MeetingFolder;
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
}
