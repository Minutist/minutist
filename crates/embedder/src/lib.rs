//! `embedder` — the text-embedding model crate.
//!
//! [`Bgem3Embedder`] implements [`minutist_common::Embedder`] over a held
//! llama.cpp model (BGE-M3 by default): CLS-pooled, 1024-dim, **L2-normalised**
//! output, so the retrieval cosine reduces to a dot product. It is the embedding
//! peer of `summariser` / `asr-runtime` / `diarizer` — a model-loading leaf crate
//! that owns the `llama-cpp-2` FFI for embeddings, depending only on `common`
//! ([`Embedder`], `AppError`, `shared_llama_backend`,
//! `voiceprint_math::unit_normalise`) plus `llama-cpp-2` itself — so `ipc-bridge`
//! (a Tauri crate) carries no direct llama edge of its own for embeddings.
//! GPU features (`vulkan` / `cuda` / `metal` / `rocm`) forward straight through
//! to `llama-cpp-2`, enabled transitively by `ipc-bridge`.
//!
//! [`Bgem3Embedder`] is constructed and held lazily by `ipc-bridge`
//! (`ensure_embedder`) and consumed by the RAG attach/transcript write path and
//! the `retrieve_chunks` tool, both in `rag-retrieval`/`agent-tools` territory —
//! this crate supplies only the model, never the retrieval logic that consumes
//! its output.
//!
//! # Why a separate context per call
//!
//! BGE-M3 is a bidirectional encoder: the whole token sequence is `encode`d in one
//! pass and the pooled vector read back. The `LlamaContext` is `!Sync` and cheap
//! relative to the encode, so it is built fresh per text and never stored — which
//! keeps [`Bgem3Embedder`] `Send + Sync` (it holds only the `LlamaModel`, which is
//! `unsafe impl Send + Sync` in `llama-cpp-2`, plus `Copy` scalars). The shared
//! process-wide [`shared_llama_backend`] is reused (a private `LlamaBackend::init`
//! would return `BackendAlreadyInitialized` and break ASR).
//!
//! # Batch shape
//!
//! [`Bgem3Embedder::embed_batch`] embeds each input with its own fresh context (the
//! spike-proven path). True multi-sequence packing into one batch is a possible
//! future optimisation but is unverified and not shipped here.
//!
//! # Test-only dev-dependencies
//!
//! The `#[ignore]`d real-model retrieval-quality eval
//! (`tests/retrieval_quality_eval.rs`, gated on `MINUTIST_BGE_M3_PATH`) drives
//! the full retrieval path end-to-end: it embeds a planted-fact corpus with the
//! real BGE-M3, indexes the vectors via `persistence::RagStore`, and fuses the
//! dense and lexical legs with `rag_retrieval::rrf_fuse`, so it catches a
//! degraded or mis-quantised embedder (near-zero or scrambled vectors) that
//! stub-embedder unit coverage cannot. This adds test-only dev-dependency edges
//! `embedder → persistence` and `embedder → rag-retrieval` (both of which
//! depend only on `common`, so no cycle results); these are **not** runtime
//! edges and do not appear in the crate's production dependency shape
//! (mirrors `diarizer`'s test-only `persistence` dev-dependency).

use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use minutist_common::llama_backend::shared_llama_backend;
use minutist_common::voiceprint_math::unit_normalise;
use minutist_common::{AppError, AppResult, Embedder};

/// BGE-M3 text embedder backed by a held llama.cpp model.
pub struct Bgem3Embedder {
    model: LlamaModel,
    /// The manifest id this model was loaded under (e.g. `"bge-m3-q8_0"`), reported
    /// via [`Embedder::model_id`] so retrieval scopes to vectors from the same model.
    model_id: String,
    n_embd: usize,
    n_threads: i32,
    /// Hard cap on tokens fed to a single encode (BGE-M3's context is 8192). Real
    /// chunks are ~256 tokens, so this only guards a pathologically long input.
    max_tokens: usize,
}

impl Bgem3Embedder {
    /// Load the embedder GGUF at `gguf_path`.
    ///
    /// `model_id` is the manifest id (e.g. `"bge-m3-q8_0"`) reported via
    /// [`Embedder::model_id`] and persisted with each vector so retrieval only
    /// scores against the same model. `n_gpu_layers` follows the `gpu_acceleration`
    /// setting (`0` = CPU even in a GPU build; `u32::MAX` = offload every layer),
    /// matching `LlamaSummariser`. Returns [`AppError::ModelLoad`] if the file is
    /// absent, fails to load, or reports a non-positive embedding dimension.
    pub fn open(
        gguf_path: impl AsRef<Path>,
        model_id: impl Into<String>,
        n_gpu_layers: u32,
    ) -> AppResult<Self> {
        let model_id = model_id.into();
        let path = gguf_path.as_ref();
        let backend = shared_llama_backend()?;
        if !path.exists() {
            return Err(AppError::ModelLoad {
                model_id: model_id.clone(),
                context: format!("embedder GGUF not found at {}", path.display()),
            });
        }
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &model_params).map_err(|e| {
            AppError::ModelLoad {
                model_id: model_id.clone(),
                context: e.to_string(),
            }
        })?;
        // Fail loudly on a degenerate model rather than coercing to a 0-dim embedder
        // that would silently produce empty vectors and disable retrieval.
        let raw_n_embd = model.n_embd();
        let n_embd = usize::try_from(raw_n_embd)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| AppError::ModelLoad {
                model_id: model_id.clone(),
                context: format!("model reports a non-positive embedding dimension: {raw_n_embd}"),
            })?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);
        tracing::info!(
            target: "embedder",
            model = %path.display(),
            model_id = %model_id,
            n_embd,
            "BGE-M3 embedder loaded"
        );
        Ok(Self {
            model,
            model_id,
            n_embd,
            n_threads,
            max_tokens: 8192,
        })
    }

    /// Embed one text into an L2-normalised vector via a fresh embeddings context.
    fn embed_one(&self, text: &str) -> AppResult<Vec<f32>> {
        let backend = shared_llama_backend()?;
        let mut tokens = self.model.str_to_token(text, AddBos::Always).map_err(|e| {
            AppError::Inference {
                backend: "bge-m3".into(),
                context: format!("tokenise failed: {e:?}"),
            }
        })?;
        if tokens.is_empty() {
            return Err(AppError::Inference {
                backend: "bge-m3".into(),
                context: "text produced no tokens".into(),
            });
        }
        tokens.truncate(self.max_tokens);

        // The encoder processes the whole sequence in one batch, so n_ctx / n_batch
        // must cover it. A small floor keeps very short inputs valid.
        let span = (tokens.len() as u32).max(8);
        let n_ctx = NonZeroU32::new(span).expect("span >= 8 is non-zero");
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(span)
            .with_n_ubatch(span)
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Cls);
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| AppError::Inference {
                backend: "bge-m3".into(),
                context: format!("embeddings context init failed: {e:?}"),
            })?;

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch
            .add_sequence(&tokens, 0, true)
            .map_err(|e| AppError::Inference {
                backend: "bge-m3".into(),
                context: format!("batch fill failed: {e:?}"),
            })?;
        ctx.encode(&mut batch).map_err(|e| AppError::Inference {
            backend: "bge-m3".into(),
            context: format!("encode failed: {e:?}"),
        })?;

        // CLS-pooled vector for the single sequence (seq id 0).
        let raw = ctx.embeddings_seq_ith(0).map_err(|e| AppError::Inference {
            backend: "bge-m3".into(),
            context: format!("read pooled embedding failed: {e:?}"),
        })?;
        let mut v = raw.to_vec();
        unit_normalise(&mut v);
        Ok(v)
    }
}

impl Embedder for Bgem3Embedder {
    fn embed_batch(&self, texts: &[&str]) -> AppResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_one(t)).collect()
    }

    fn dim(&self) -> usize {
        self.n_embd
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}
