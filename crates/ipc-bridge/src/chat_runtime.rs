//! Shared chat-runtime handles (Phase 10).
//!
//! [`ChatHandles`] bundles the held-model + persistence handles a chat turn
//! needs. Both [`crate::IpcState`] (the UI `send_chat_message` path) and the
//! Phase-10 inter-agent driver ([`crate::inter_agent`]) hold one, so the
//! held-LLM load logic (`ensure_summariser`) lives in exactly one place and both
//! paths share the SAME lazily-loaded model instance.

use std::path::PathBuf;
use std::sync::Arc;

use meeting_app_common::AppEvent;
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use settings::SettingsHandle;
use summariser::LlamaSummariser;
use tokio::sync::{broadcast, OnceCell};

use crate::commands;
use crate::error::IpcError;

/// The handles a chat turn needs, cloneable (all `Arc` / `PathBuf` /
/// `broadcast::Sender` / `SettingsHandle`). The `summariser` `OnceCell` is the
/// SAME `Arc` both consumers share, so the held model is loaded once.
#[derive(Clone)]
pub struct ChatHandles {
    pub orchestrator: Arc<Orchestrator>,
    pub index: Arc<MeetingIndex>,
    pub meetings_dir: PathBuf,
    pub event_tx: broadcast::Sender<AppEvent>,
    pub settings: SettingsHandle,
    /// The lazily-loaded held LLM substrate (shared with `IpcState`).
    pub summariser: Arc<OnceCell<Arc<LlamaSummariser>>>,
}

impl ChatHandles {
    /// Resolve the held summariser, loading the GGUF once on first use.
    ///
    /// The single load implementation (the body that `IpcState::ensure_summariser`
    /// also calls): the model id is the user-selected `settings.llm_model_id` or
    /// the bundled default; the directory is resolved (downloaded + verified when
    /// absent) via `Orchestrator::ensure_model_path`; the GGUF is opened on
    /// `spawn_blocking`. Subsequent calls return the cached `Arc`.
    pub async fn ensure_summariser(&self) -> Result<Arc<LlamaSummariser>, IpcError> {
        let handle = self
            .summariser
            .get_or_try_init(|| async {
                let settings = self.settings.current();
                let model_id = commands::resolve_llm_model_id(&settings);
                let model_dir = self.orchestrator.ensure_model_path(&model_id).await?;
                let n_gpu_layers =
                    commands::resolve_summariser_gpu_layers(settings.gpu_acceleration);
                let summariser = tokio::task::spawn_blocking(move || {
                    commands::open_summariser_in_dir(&model_dir, n_gpu_layers)
                })
                .await
                .map_err(|e| meeting_app_common::AppError::Internal {
                    context: format!("summariser load task join failed: {e}"),
                })??;
                tracing::info!(
                    target: "ipc-bridge",
                    "held LLM summariser loaded (shared by summarise + chat + inter-agent)"
                );
                Ok::<_, meeting_app_common::AppError>(Arc::new(summariser))
            })
            .await
            .map_err(IpcError::from)?;
        Ok(Arc::clone(handle))
    }
}
