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

use meeting_app_common::{AudioDevice, MeetingId, MeetingMeta, RecordingState};
use settings::Settings;
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
