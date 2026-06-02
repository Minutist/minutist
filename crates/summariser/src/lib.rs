//! `summariser` — local-LLM text summarisation.
//!
//! Implements [`meeting_app_common::Summariser`] by driving a GGUF text model
//! through `llama-cpp-2`. The model is resolved + selected by the caller
//! (settings `llm_model_id` → `model-registry`), so this crate is
//! **model-agnostic**: it reads the GGUF's baked-in chat template rather than
//! assuming a specific family (Gemma 4 / Qwen / Granite). See
//! `architecture/components.md` — `summariser`, and `cross-cutting.md`
//! ("llama.cpp prefill batching" — chunked-prefill is mandatory).
//!
//! **Skeleton (Phase 5 Gate A).** The public surface — the [`LlamaSummariser`]
//! type and its `Summariser` impl — is frozen here so the `ipc-bridge`
//! summarise wiring and the orchestrator model-path helper can compile against
//! it while Stream S1 fills the inference body. The `summarise` body is a stub
//! that returns an `AppError` (never panics) until S1 lands the real
//! implementation (prompt build from the baked-in chat template with thinking
//! disabled, chunked-prefill, greedy generation + `is_eog_token` stop,
//! incremental detokenisation).

use std::path::{Path, PathBuf};

use meeting_app_common::{AppResult, Segment, Summariser};

mod error;
pub use error::Error;

/// Runtime knobs for the summariser. Defaults follow the cross-cutting
/// chunked-prefill rule (`n_batch` is the per-decode chunk size).
#[derive(Debug, Clone)]
pub struct SummariserConfig {
    /// Context window to allocate, in tokens.
    pub n_ctx: u32,
    /// Per-decode batch size — the chunked-prefill chunk size.
    pub n_batch: u32,
    /// Hard cap on generated tokens for one summary.
    pub max_tokens: usize,
}

impl Default for SummariserConfig {
    fn default() -> Self {
        Self {
            // Gemma 4 E4B is 128K-capable; 32K comfortably holds a 30-min
            // transcript + notes + prompt and bounds KV memory on-device.
            n_ctx: 32_768,
            n_batch: 512,
            max_tokens: 2_048,
        }
    }
}

/// A local-LLM summariser backed by a GGUF text model via `llama-cpp-2`.
///
/// Construct with [`LlamaSummariser::open`]. The model load and the
/// `summarise` inference are implemented by Stream S1 (this is the frozen
/// skeleton).
pub struct LlamaSummariser {
    model_path: PathBuf,
    config: SummariserConfig,
}

impl LlamaSummariser {
    /// Open a summariser over the GGUF at `model_path`.
    ///
    /// Skeleton: stores the path + config. Stream S1 adds the process-wide
    /// `LlamaBackend` `OnceLock` singleton + `LlamaModel` load here (mirroring
    /// `asr-runtime`).
    pub fn open(model_path: PathBuf, config: SummariserConfig) -> AppResult<Self> {
        Ok(Self { model_path, config })
    }

    /// The GGUF path this summariser was opened over.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// The active configuration.
    pub fn config(&self) -> &SummariserConfig {
        &self.config
    }
}

impl Summariser for LlamaSummariser {
    fn summarise(
        &self,
        _transcript: &[Segment],
        _notes_markdown: &str,
        _system_prompt: &str,
    ) -> AppResult<String> {
        // Stream S1 replaces this body. Returns an error (not a panic) so the
        // skeleton is safe to wire end-to-end before the implementation lands.
        tracing::warn!(
            target: "summariser",
            "summarise() called on the Phase-5 skeleton stub; inference not yet implemented"
        );
        Err(Error::Inference(
            "summariser inference not yet implemented (Phase 5 Gate-A skeleton)".to_string(),
        )
        .into())
    }
}
