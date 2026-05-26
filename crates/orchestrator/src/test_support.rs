//! Test utilities for `orchestrator` tests.
//!
//! Gated behind `#[cfg(any(test, feature = "test-source"))]` so this code
//! never ships in production builds.
//!
//! Provides a helper to build a test `Orchestrator` with a tempdir-backed
//! persistence root and a default `SettingsHandle`.

use std::path::PathBuf;

use settings::{JsonFileStore, SettingsHandle};

use crate::Orchestrator;

/// Build an `Orchestrator` suitable for unit tests.
///
/// Uses a caller-supplied `persistence_root` so tests can use `tempfile`.
pub fn test_orchestrator(persistence_root: PathBuf) -> Orchestrator {
    // Create a settings store that has no file (uses defaults).
    let settings_path = persistence_root.join(".test_settings.json");
    let store = JsonFileStore::new(settings_path);
    let handle =
        SettingsHandle::new(store).expect("test SettingsHandle construction should not fail");
    Orchestrator::new(handle, persistence_root)
}
