//! Model-registry commands: list known models and their runtime status, and trigger an ensure/download.
use super::*;


/// List all known models with their current runtime status.
///
/// Routes through `Orchestrator::list_models`, which wraps `ModelRegistry::list_models`
/// so that `ipc-bridge` does not need a direct `model-registry` dependency.
#[tauri::command]
#[specta::specta]
pub async fn list_models(state: State<'_, IpcState>) -> AppResult<Vec<ModelStatus>> {
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
pub async fn ensure_model(model_id: ModelId, state: State<'_, IpcState>) -> AppResult<()> {
    state
        .orchestrator
        .ensure_model(&model_id)
        .await
}

