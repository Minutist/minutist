//! Settings read/write commands.
use super::*;


/// Return the current application settings.
#[tauri::command]
#[specta::specta]
pub async fn get_settings(state: State<'_, IpcState>) -> AppResult<Settings> {
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
) -> AppResult<()> {
    state
        .settings
        .update(|s| *s = settings)
        .await
}

