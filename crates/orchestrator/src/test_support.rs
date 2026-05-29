//! Test utilities for `orchestrator` tests.
//!
//! Gated behind `#[cfg(any(test, feature = "test-source"))]` so this code
//! never ships in production builds.
//!
//! Provides a helper to build a test `Orchestrator` with a tempdir-backed
//! persistence root, a default `SettingsHandle`, and a minimal `ModelRegistry`
//! backed by an empty manifest (no real models; ASR is skipped during tests).

use std::path::PathBuf;
use std::sync::Arc;

use model_registry::ModelRegistry;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::broadcast;

use crate::Orchestrator;

/// Build an `Orchestrator` suitable for unit tests.
///
/// Uses a caller-supplied `persistence_root` so tests can use `tempfile`.
/// The `ModelRegistry` is initialised with an empty manifest and a tempdir
/// cache root — no models are present, so ASR is skipped during test runs.
pub fn test_orchestrator(persistence_root: PathBuf) -> Orchestrator {
    // Create a settings store that has no file (uses defaults).
    let settings_path = persistence_root.join(".test_settings.json");
    let store = JsonFileStore::new(settings_path);
    let handle =
        SettingsHandle::new(store).expect("test SettingsHandle construction should not fail");

    // Build a ModelRegistry with an empty manifest. The empty manifest means
    // list_models() returns [] and ensure() will return ModelNotFound, causing
    // the ASR worker to skip transcription. This is the correct test behaviour:
    // no real model present → no transcript segments → recording still finalises.
    let model_cache = persistence_root.join(".test_model_cache");
    let (event_tx, _) = broadcast::channel::<meeting_app_common::AppEvent>(256);
    let registry = ModelRegistry::new(model_cache, Vec::new(), event_tx)
        .expect("test ModelRegistry construction should not fail");
    let registry = Arc::new(registry);

    Orchestrator::new(handle, persistence_root, registry)
}
