//! Audio-input device enumeration.
use super::*;


/// List all available audio-input devices.
///
/// Routes through `Orchestrator::list_devices`, which wraps the FFI-bound
/// cpal enumeration in `spawn_blocking`. This keeps `ipc-bridge`'s
/// dependency table honest: it depends on `orchestrator`, not directly on
/// `audio-capture`.
#[tauri::command]
#[specta::specta]
pub async fn list_devices(state: State<'_, IpcState>) -> AppResult<Vec<AudioDevice>> {
    state
        .orchestrator
        .list_devices()
        .await
}

