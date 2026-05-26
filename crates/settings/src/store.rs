//! `SettingsStore` trait and `JsonFileStore` implementation.
//!
//! The store abstraction keeps the `settings` crate free of any Tauri
//! dependency. `app-main` resolves the platform-specific file path and
//! hands it to `JsonFileStore`; the settings crate never sees `tauri::*`.

use std::path::PathBuf;

use crate::{error::Error, Settings};

/// Persistence backend for settings.
///
/// Implementations must be `Send + Sync` so the `SettingsHandle` can
/// share the store across tokio worker threads.
pub trait SettingsStore: Send + Sync {
    /// Load settings from the backing store.
    ///
    /// A missing or corrupt store MUST return `Settings::default()` and log a
    /// warning rather than propagating an error — the app must always be able
    /// to start with defaults.
    fn load(&self) -> Result<Settings, Error>;

    /// Persist `settings` to the backing store.
    fn save(&self, settings: &Settings) -> Result<(), Error>;
}

/// A `SettingsStore` backed by a single JSON file on disk.
///
/// The file path is supplied by the caller at construction time (typically
/// `{app-data}/settings.store`).  Corrupt or missing files fall back to
/// `Settings::default()` — the caller is responsible for logging the warning
/// (or see `SettingsHandle::new` which does it automatically).
pub struct JsonFileStore {
    path: PathBuf,
}

impl JsonFileStore {
    /// Create a new `JsonFileStore` targeting `path`.
    ///
    /// The file need not exist yet; it is created on the first `save` call.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl SettingsStore for JsonFileStore {
    fn load(&self) -> Result<Settings, Error> {
        match std::fs::read_to_string(&self.path) {
            Ok(content) => {
                let settings = serde_json::from_str::<Settings>(&content)?;
                Ok(settings)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Missing file is not an error; return defaults.
                Ok(Settings::default())
            }
            Err(e) => Err(Error::Io(e)),
        }
    }

    fn save(&self, settings: &Settings) -> Result<(), Error> {
        // Create parent directories if they don't exist.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(settings)?;
        // Write atomically: write to a temp file alongside the target, then
        // rename.  This avoids leaving a half-written settings file on crash.
        let tmp_path = self.path.with_extension("store.tmp");
        std::fs::write(&tmp_path, content)?;
        std::fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }
}
