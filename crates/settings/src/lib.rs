//! Application settings — Phase 1 fields.
//!
//! This crate is the single source of truth for runtime configuration.
//! Other crates read settings via [`SettingsHandle`]; nobody parses the
//! backing JSON file directly.
//!
//! ## Architecture constraints
//!
//! - **No `tauri::*` imports.** Tauri glue lives only in `ipc-bridge` and
//!   `app-main`. This crate receives a `PathBuf` at construction time and
//!   reads/writes JSON via `serde_json` + `std::fs`.
//! - Settings changes broadcast directly from this crate via
//!   [`tokio::sync::watch`], not through the orchestrator.
//! - Per-crate [`Error`] via `thiserror`; `From<Error> for AppError` is
//!   implemented in [`error`].

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub mod error;
pub mod handle;
pub mod store;

pub use error::Error;
pub use handle::SettingsHandle;
pub use store::{JsonFileStore, SettingsStore};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// UI colour-scheme preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Theme {
    Light,
    Dark,
    /// Follow the OS preference (default).
    #[default]
    System,
}

/// Application settings — Phase 1 fields only.
///
/// Fields added in later phases live in their respective phase plans.
/// Do **not** add ASR model selection, summary system-prompt, autosave
/// interval, or telemetry fields here — those are Phase 2+ concerns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Settings {
    /// The preferred audio-input device, identified by the opaque id
    /// returned by `audio-capture::AudioCaptureManager::list_devices`.
    /// `None` means "use the OS default".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_device_id: Option<String>,

    /// UI colour-scheme preference.
    #[serde(default)]
    pub theme: Theme,

    /// Root directory for meeting data.  `None` means "use the platform
    /// default app-data directory" (resolved by `app-main`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_directory: Option<PathBuf>,

    /// If `true`, the main window starts hidden; accessible via the tray icon.
    #[serde(default)]
    pub start_hidden: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::SettingsStore;

    // -----------------------------------------------------------------------
    // 1. JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn settings_default_round_trips_json() {
        let original = Settings::default();
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(original, restored);
    }

    #[test]
    fn settings_with_all_fields_round_trips_json() {
        let original = Settings {
            input_device_id: Some("hw:1,0".to_string()),
            theme: Theme::Dark,
            data_directory: Some(PathBuf::from("/tmp/meeting-data")),
            start_hidden: true,
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(original, restored);
    }

    // -----------------------------------------------------------------------
    // 2. Default construction — no file → returns defaults
    // -----------------------------------------------------------------------

    #[test]
    fn json_file_store_missing_file_returns_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        // Path does not exist yet.
        let store = JsonFileStore::new(path);
        let loaded = store.load().expect("load");
        assert_eq!(loaded, Settings::default());
    }

    // -----------------------------------------------------------------------
    // 3. Watch emits on update
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn watch_receiver_emits_after_update() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        let store = JsonFileStore::new(path);
        let handle = SettingsHandle::new(store).expect("handle");

        let mut rx = handle.subscribe();
        // The receiver starts with the initial value; mark it as seen.
        rx.borrow_and_update();

        handle
            .update(|s| s.theme = Theme::Light)
            .await
            .expect("update");

        // `changed()` resolves immediately because `update` already sent.
        rx.changed().await.expect("changed");
        let new_settings = rx.borrow().clone();
        assert_eq!(new_settings.theme, Theme::Light);
    }

    // -----------------------------------------------------------------------
    // 4. Corruption recovery — garbage file → defaults + no panic
    // -----------------------------------------------------------------------

    #[test]
    fn json_file_store_corrupt_file_returns_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        std::fs::write(&path, b"{{{{not valid json}}}").expect("write garbage");

        let store = JsonFileStore::new(path);
        // `load` returns an error for corrupt JSON — SettingsHandle::new
        // catches this specific error and falls back to defaults.
        let result = store.load();
        assert!(
            result.is_err(),
            "corrupt file should return an error from the raw store"
        );
    }

    #[tokio::test]
    async fn handle_new_corrupt_file_falls_back_to_defaults() {
        // Install a no-op tracing subscriber so warn! doesn't panic in tests.
        let _ = tracing_subscriber::fmt::try_init();

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        std::fs::write(&path, b"not json at all").expect("write garbage");

        let store = JsonFileStore::new(path);
        let handle = SettingsHandle::new(store).expect("handle despite corrupt file");
        assert_eq!(handle.current(), Settings::default());
    }

    // -----------------------------------------------------------------------
    // 5. Persistence — update writes to disk, reloaded store reflects it
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn update_persists_to_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");

        let handle = SettingsHandle::new(JsonFileStore::new(path.clone())).expect("handle");
        handle
            .update(|s| {
                s.theme = Theme::Dark;
                s.start_hidden = true;
            })
            .await
            .expect("update");

        // Open a fresh store at the same path and verify the value is there.
        let store2 = JsonFileStore::new(path);
        let loaded = store2.load().expect("reload");
        assert_eq!(loaded.theme, Theme::Dark);
        assert!(loaded.start_hidden);
    }
}
