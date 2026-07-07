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
    /// A missing store returns `Settings::default()` (see [`JsonFileStore::load`]).
    /// A corrupt store propagates `Err`; the default-on-corrupt-with-warning
    /// contract the app needs is provided one layer up, by
    /// `SettingsHandle::new` (`crates/settings/src/handle.rs`), which is the
    /// sole production caller of this method.
    fn load(&self) -> Result<Settings, Error>;

    /// Persist `settings` to the backing store.
    fn save(&self, settings: &Settings) -> Result<(), Error>;
}

/// A `SettingsStore` backed by a single JSON file on disk.
///
/// The file path is supplied by the caller at construction time (typically
/// `{app-data}/settings.store`). A missing file yields `Settings::default()`
/// directly from [`Self::load`]; a corrupt file's `Err` propagates to the
/// caller. `SettingsHandle::new` is the sole production caller, and applies
/// the default-on-corrupt-with-warning fallback the app needs on top of this
/// store.
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
        use std::io::Write;

        // Create parent directories if they don't exist.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(settings)?;
        // Write atomically: write to a temp file alongside the target, fsync
        // it so the bytes are durable before the rename links them in under
        // the real name, then rename. Without the fsync, the OS is free to
        // reorder or delay the tmp file's writeback past a crash, so the
        // rename could hand the real name to a file the crash never actually
        // persisted; the fsync closes that gap.
        let tmp_path = self.path.with_extension("store.tmp");
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }
}
