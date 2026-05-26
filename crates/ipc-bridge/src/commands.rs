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
//! ## Type mirror note
//!
//! Command signatures use specta-typed mirror types from `specta_types`
//! (e.g. `AudioDeviceType`, `MeetingIdType`) instead of the `common` /
//! `settings` originals, because those originals do not implement
//! `specta::Type`.  The conversion is transparent — same JSON wire shape.
//! See `specta_types.rs` for details.
//!
//! ## Tauri State
//!
//! Each command that needs the orchestrator or settings receives its handles
//! as `tauri::State<'_, IpcState>`.

use tauri::State;

use crate::{
    error::IpcError,
    specta_types::{
        AudioDeviceType, MeetingIdType, MeetingMetaType, RecordingStateType, SettingsType,
    },
    IpcState,
};

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
pub async fn list_devices(state: State<'_, IpcState>) -> Result<Vec<AudioDeviceType>, IpcError> {
    let devices = state
        .orchestrator
        .list_devices()
        .await
        .map_err(IpcError::from)?;
    Ok(devices.into_iter().map(AudioDeviceType::from).collect())
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
) -> Result<MeetingIdType, IpcError> {
    let meeting_id = state
        .orchestrator
        .start(device_id)
        .await
        .map_err(IpcError::from)?;
    Ok(meeting_id.into())
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
pub async fn stop_recording(state: State<'_, IpcState>) -> Result<MeetingMetaType, IpcError> {
    let meta = state.orchestrator.stop().await.map_err(IpcError::from)?;
    Ok(meta.into())
}

/// Return a snapshot of the current recording state.
#[tauri::command]
#[specta::specta]
pub async fn get_recording_state(
    state: State<'_, IpcState>,
) -> Result<RecordingStateType, IpcError> {
    Ok(state.orchestrator.state().await.into())
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Return the current application settings.
#[tauri::command]
#[specta::specta]
pub async fn get_settings(state: State<'_, IpcState>) -> Result<SettingsType, IpcError> {
    Ok(state.settings.current().into())
}

/// Persist updated application settings.
///
/// Broadcasts the change to all `SettingsHandle` subscribers.
#[tauri::command]
#[specta::specta]
pub async fn update_settings(
    settings: SettingsType,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    let new_settings: settings::Settings = settings.into();
    state
        .settings
        .update(|s| *s = new_settings)
        .await
        .map_err(IpcError::from)
}
