//! Shared chat-runtime handles (Phase 10).
//!
//! [`ChatHandles`] bundles the held-model + persistence handles a chat turn
//! needs. Both [`crate::IpcState`] (the UI `send_chat_message` path) and the
//! Phase-10 inter-agent driver ([`crate::inter_agent`]) hold one, so the
//! held-LLM load logic (`ensure_summariser`) lives in exactly one place and both
//! paths share the SAME lazily-loaded model instance.

use std::path::PathBuf;
use std::sync::Arc;

use embedder::Bgem3Embedder;
use minutist_common::{AppEvent, Embedder};
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
    /// The lazily-loaded held BGE-M3 embedder (shared with `IpcState`), driving the
    /// RAG write path and the `retrieve_chunks` tool.
    pub embedder: Arc<OnceCell<Arc<dyn Embedder>>>,
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
                // VRAM-aware plan: the summariser's GPU decision is `plan.summariser_gpu`
                // (the summariser is budgeted FIRST). See `architecture/cross-cutting.md`
                // — "GPU portability".
                let plan = minutist_common::resolve_gpu_plan(
                    minutist_common::probe_primary_gpu().as_ref(),
                    settings.gpu_acceleration,
                    true, // always request the large tier; the VRAM clamp in resolve_gpu_plan decides
                );
                let n_gpu_layers = commands::resolve_summariser_gpu_layers(plan.summariser_gpu);
                let summariser = tokio::task::spawn_blocking(move || {
                    commands::open_summariser_in_dir(&model_dir, n_gpu_layers)
                })
                .await
                .map_err(|e| minutist_common::AppError::Internal {
                    context: format!("summariser load task join failed: {e}"),
                })??;
                tracing::info!(
                    target: "ipc-bridge",
                    "held LLM summariser loaded (shared by summarise + chat + inter-agent)"
                );
                Ok::<_, minutist_common::AppError>(Arc::new(summariser))
            })
            .await
            .map_err(IpcError::from)?;
        Ok(Arc::clone(handle))
    }

    /// Resolve the held BGE-M3 embedder, loading the GGUF once on first use.
    ///
    /// Mirrors [`Self::ensure_summariser`]: the embedder model dir is resolved
    /// (downloaded + verified when absent) via `Orchestrator::ensure_model_path`,
    /// then opened on `spawn_blocking`. Shared with `IpcState` so the model loads
    /// once and serves both the RAG write path and the `retrieve_chunks` tool.
    pub async fn ensure_embedder(&self) -> Result<Arc<dyn Embedder>, IpcError> {
        let handle = self
            .embedder
            .get_or_try_init(|| async {
                let settings = self.settings.current();
                let model_id =
                    minutist_common::ModelId::from(commands::DEFAULT_EMBED_MODEL_ID);
                let model_dir = self.orchestrator.ensure_model_path(&model_id).await?;
                let gguf = commands::find_gguf_weights(&model_dir)?;
                // The embedder is small (~600 MB); offload it whenever GPU
                // acceleration is enabled (Off forces CPU). It is not in the
                // VRAM plan (which budgets the summariser first).
                let enabled =
                    settings.gpu_acceleration != minutist_common::GpuAcceleration::Off;
                let n_gpu_layers = commands::resolve_summariser_gpu_layers(enabled);
                let embedder = tokio::task::spawn_blocking(move || {
                    Bgem3Embedder::open(&gguf, n_gpu_layers)
                })
                .await
                .map_err(|e| minutist_common::AppError::Internal {
                    context: format!("embedder load task join failed: {e}"),
                })??;
                tracing::info!(
                    target: "ipc-bridge",
                    "held BGE-M3 embedder loaded (RAG write path + retrieve_chunks)"
                );
                Ok::<_, minutist_common::AppError>(Arc::new(embedder) as Arc<dyn Embedder>)
            })
            .await
            .map_err(IpcError::from)?;
        Ok(Arc::clone(handle))
    }

    /// Preload the held summariser at startup when the user has opted in
    /// (`settings.preload_summariser`, the default) AND the LLM is already
    /// downloaded — so the first Summarise / chat is instant.
    ///
    /// Best-effort and **never downloads**: it consults the registry's local
    /// availability (`Orchestrator::list_models`) and only warms an
    /// already-present model; a not-yet-fetched model is left for the first
    /// on-demand use (which downloads). All outcomes are logged, none surfaced —
    /// a failed preload must never block startup, and the model still loads
    /// lazily on first use. Driven from `app-main` on a background startup task,
    /// mirroring `Orchestrator::prewarm_asr`.
    pub async fn maybe_preload_summariser(&self) {
        if !self.settings.current().preload_summariser {
            tracing::debug!(target: "ipc-bridge", "summariser preload disabled by setting");
            return;
        }
        let model_id = commands::resolve_llm_model_id(&self.settings.current());
        let available = self.orchestrator.list_models().into_iter().any(|m| {
            m.id == model_id
                && matches!(
                    m.status,
                    minutist_common::ModelStatusState::Available { .. }
                )
        });
        if !available {
            tracing::info!(
                target: "ipc-bridge",
                model_id = %model_id.0,
                "summariser preload skipped: LLM not downloaded yet (loads on first use)"
            );
            return;
        }
        match self.ensure_summariser().await {
            Ok(_) => tracing::info!(target: "ipc-bridge", "summariser preloaded at startup"),
            Err(e) => tracing::warn!(
                target: "ipc-bridge",
                "summariser preload failed: {e}; will retry on first use"
            ),
        }
    }
}
