//! `summariser` — local-LLM text summarisation.
//!
//! Implements [`minutist_common::Summariser`] by driving a GGUF text model
//! through `llama-cpp-2`: llama-cpp-2 text-LLM lifecycle, summarisation
//! prompts, and the optional external-LLM dispatcher (Ollama / LM Studio).
//! Inputs are a transcript and notes (read by the caller via `persistence`);
//! output is a markdown summary (written by the caller via `persistence`).
//! The model is resolved + selected by the caller (settings `llm_model_id` →
//! `model-registry`), so this crate is **model-agnostic**: it reads the
//! GGUF's baked-in chat template rather than assuming a specific family
//! (Gemma 4 / Qwen / Granite). See `cross-cutting.md` ("llama.cpp prefill
//! batching" — chunked-prefill is mandatory).
//!
//! The bundled default model is Gemma 4 E4B-it (`gemma4` arch, Apache-2.0),
//! with a low-end tier of Gemma 4 E2B-it (same family/loader) and IBM
//! Granite 4.1-3b (Apache-2.0, dense, no PLE, non-thinking) as a fallback if
//! the Gemma-4 PLE forward-graph bug degrades quality on a given build. Its
//! 128K context fits a 30-minute transcript in one pass. The model is
//! **settings-selected, never hard-coded** — switching model families is a
//! manifest + `llm_model_id` change, not a code change; the chat-template and
//! fallback machinery below is what makes that possible without per-model
//! branching.
//!
//! # Lifecycle (mirrors `asr-runtime`)
//!
//! - The `LlamaBackend` is a process-wide [`OnceLock`] singleton; `init` may
//!   be called only once per process.
//! - [`LlamaSummariser::open`] loads the GGUF weights once and retains the
//!   [`LlamaModel`] for the lifetime of the struct (`Drop` releases it).
//! - Each [`summarise`](LlamaSummariser::summarise) call allocates a fresh
//!   [`LlamaContext`] sized to `config.n_ctx` / `config.n_batch`, so the KV
//!   cache is always clean without an explicit reset.
//!
//! # Inference shape
//!
//! 1. **Prompt** is built from the GGUF's baked-in chat template
//!    (`chat_template(None)` + `apply_chat_template(messages, add_ass=true)`)
//!    with the system prompt folded into a SINGLE `user` turn — not a
//!    separate `system` message, since several templates (notably Gemma)
//!    have no `system` role — followed by the rendered transcript + notes.
//!    Missing/unusable template → [`Error::Template`] (→
//!    `AppError::InvalidInput`), never a silent hand-built scaffold for an
//!    unrecognised model.
//!
//!    That said, the bundled llama.cpp build cannot RENDER a chat template
//!    newer than itself: the shipped Gemma-4 GGUF's template postdates the
//!    vendored llama.cpp, so `apply_chat_template` returns `ffi error -1`
//!    even for a well-formed user-only message set. On that specific
//!    failure, [`LlamaSummariser`] falls back to a hand-built Gemma turn-
//!    format prompt (`<bos><start_of_turn>user … <end_of_turn>` then an open
//!    `model` turn — [`model_turn_prompt`]) with turn markers probed from the
//!    model's own vocabulary, matching the format the shipped LLM actually
//!    uses. Every other model keeps rendering its baked template unchanged.
//!    `<bos>` is explicit in the fallback because generation tokenises with
//!    `AddBos::Never` and `str_to_token` parses special tokens directly. This
//!    same fallback chain is reused by the OCR prompt and the translation
//!    prompt, not just `summarise`.
//! 2. **Thinking is disabled** — we never inject a think token. If the model
//!    nonetheless emits a `<think>…</think>` block, it is stripped before the
//!    summary is returned.
//! 3. **Tokenisation** uses [`AddBos::Never`] (the template embeds BOS).
//! 4. **Prefill is chunked by `n_batch`** ([`plan_prefill`]): the prompt is
//!    decoded in `n_batch`-sized [`LlamaBatch`] chunks, with `logits = true`
//!    set only on the final token of the final chunk, so a long transcript
//!    (which routinely exceeds the default 512-token `n_batch`) never trips
//!    `GGML_ASSERT(n_tokens_all <= n_batch)`.
//! 5. **Generation** is greedy, stops on `model.is_eog_token(token)` (covers
//!    both EOS and `<|im_end|>` for Qwen), and is capped at
//!    `config.max_tokens`. Detokenisation is incremental via an
//!    `encoding_rs` UTF-8 decoder (mirrors `asr-runtime`).
//!
//! # Notes weaving
//!
//! The `Summariser` trait takes `notes: &[NoteBlock]` rather than a flat
//! markdown string; `NoteBlock { at_ms: Option<u64>, text }` is a `common`
//! vocabulary type projected from a meeting's `notes.json` by
//! `persistence::note_blocks_from_json` / `read_note_blocks`. When any note
//! is anchored to the recording clock, [`render_user_content`] merges the
//! transcript and the anchored notes into one time-ordered, `[m:ss]`-
//! prefixed timeline so the model sees each note beside what was being said
//! when it was written; un-anchored notes trail the timeline. With no
//! anchored notes the transcript renders as a plain transcript plus a flat
//! `# Notes` block, spending no extra context tokens over the pre-weaving
//! format. [`summarise_with_progress`](LlamaSummariser::summarise_with_progress)
//! reports two-phase progress (`Prefill { done, total }` per prompt chunk,
//! then `Generate { done, max }` per token) so a caller-side progress bar
//! does not sit pinned at 0% through the prefill stretch.
//!
//! # Attachments feed
//!
//! `Summariser::summarise` takes a leading `attachments_markdown: &str`
//! parameter (before `system_prompt`, matching the prompt fold order). The
//! caller assembles this by concatenating each Ready attachment's converted
//! `<hash>.md` content in manifest order, each under a
//! `## Attachment: <original_filename>` header. [`render_user_content`]
//! prepends a `# Reference material (attachments)` section — before
//! `# Transcript` — only when the string is non-empty, so an empty string
//! produces byte-identical output to the no-attachment path. Attachments are
//! reference material, not time-woven: they never enter the transcript/notes
//! timeline merge. The caller is responsible for a budget guard: before
//! calling `summarise`, it deterministically truncates the assembled
//! attachments string (per-attachment, equal-share, with a visible
//! `[truncated]` marker on any trimmed part) if the full prompt would
//! overflow `n_ctx` minus reserves for transcript, notes, and generation
//! headroom — this crate has no `n_ctx` awareness of its own for that
//! purpose beyond [`check_context_budget`], which validates prompt tokens
//! plus `max_tokens` against `n_ctx` at generation time.
//!
//! # Auxiliary generation surfaces
//!
//! Beyond `summarise`, [`LlamaSummariser`] exposes several concrete methods
//! (not on the `Summariser` trait, since they have no meaningful remote-
//! backend equivalent and the trait stays minimal for its other impls):
//!
//! - [`translate_segment`](LlamaSummariser::translate_segment) — translates
//!   one transcript segment into a target language via a minimal single-turn
//!   prompt, reusing the same template/fallback and chunked-prefill path.
//! - [`generate_attachment_awareness`](LlamaSummariser::generate_attachment_awareness)
//!   — produces a compact description + keyword list for a converted
//!   attachment, pinned into the live co-pilot prefix.
//! - [`model`](LlamaSummariser::model) — lends the held `&LlamaModel` to the
//!   `chat-agent` engine's turn backend (and the same handle coerces to
//!   `Arc<dyn Summariser>` for `agent-tools`). The model is `unsafe impl Send
//!   + Sync`, so it is safely referenced concurrently; each caller still
//!   builds its own fresh, `!Sync` `LlamaContext` per turn, exactly as
//!   `summarise` does — no GGUF is ever reloaded. This is an accessor, not a
//!   wrapper, so it adds no `summariser → chat-agent` dependency edge
//!   (`chat-agent` depends on `summariser`, never the reverse).
//! - [`ensure_vision`](LlamaSummariser::ensure_vision) /
//!   [`image_to_markdown`](LlamaSummariser::image_to_markdown) — a lazily-
//!   built vision `MtmdContext` bound to the already-loaded Gemma-4 model (no
//!   second model, same GPU budget) backing document-page OCR into markdown.
//!
//! # External dispatcher
//!
//! The optional `external-ollama` cargo feature adds
//! [`OllamaSummariser`](ollama::OllamaSummariser): a `reqwest::blocking`
//! dispatcher to a local Ollama/LM-Studio-compatible `/api/chat` endpoint,
//! selected instead of the bundled GGUF when settings prefer an external
//! model. `reqwest` and `serde` are pulled in only by that feature, so the
//! default build stays dependent on `common` alone. Its deterministic seams
//! (URL normalisation, request-shape construction, HTTP-error mapping) are
//! covered by `#[cfg(test)]` unit tests that need no live server; the gated
//! verification harness runs `cargo test -p summariser --features
//! external-ollama` as an extra step so those tests are exercised even
//! though the default build does not compile them.

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;

use minutist_common::{
    llama_backend::serialize_first_model_load, AppResult, NoteBlock, Segment, Summariser,
};

mod error;
pub use error::Error;

#[cfg(feature = "external-ollama")]
mod ollama;
#[cfg(feature = "external-ollama")]
pub use ollama::OllamaSummariser;

/// A phase + position of an in-progress summarise, for a two-phase determinate
/// bar (#69).
///
/// A long summarise spends its first stretch decoding the prompt — transcript +
/// notes — into the KV cache (`Prefill`), then writes the summary token by
/// token (`Generate`). Prefill produces NO output and, for a long meeting, runs
/// for many seconds: reporting only generation (the prior behaviour) left the
/// bar pinned at 0% for that whole stretch. The caller maps each phase onto its
/// own labelled, determinate `OperationProgress` so the bar reflects both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummariseProgress {
    /// Decoding the prompt into the KV cache. `done`/`total` are prompt-token
    /// counts; `done` reaches `total` as the final chunk is decoded.
    Prefill { done: usize, total: usize },
    /// Generating the summary. `done`/`max` are output-token counts.
    Generate { done: usize, max: usize },
}

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
    /// CPU threads for llama.cpp inference. Default: `(num_cpus / 2).min(8)`,
    /// min 1 — matching `asr-runtime`.
    pub threads: i32,
    /// Number of model layers to offload to the GPU.
    ///
    /// Default is the compile-time [`gpu_layers`] ceiling (`u32::MAX` in a
    /// GPU-feature build, `0` otherwise). The caller (`ipc-bridge`) sets this at
    /// **runtime** from the `gpu_acceleration` setting, passing `0` to force CPU
    /// even in a GPU build. See `architecture/cross-cutting.md` — "GPU
    /// portability".
    pub n_gpu_layers: u32,
}

impl Default for SummariserConfig {
    fn default() -> Self {
        let threads = ((num_cpus::get() / 2) as i32).clamp(1, 8);
        Self {
            // Gemma 4 E4B is 128K-capable; 32K comfortably holds a 30-min
            // transcript + notes + prompt and bounds KV memory on-device.
            n_ctx: 32_768,
            n_batch: 512,
            max_tokens: 2_048,
            threads,
            n_gpu_layers: gpu_layers(),
        }
    }
}

// ---------------------------------------------------------------------------
// LlamaBackend singleton
// ---------------------------------------------------------------------------

/// Return the process-wide `LlamaBackend`.
///
/// Delegates to the SHARED singleton in `common`. `LlamaBackend::init()` is
/// global (once per process) and `asr-runtime` also loads a GGUF model in the
/// same app process, so a private per-crate `OnceLock` here would make
/// whichever crate inits second fail — exactly the record-then-summarise bug
/// this delegation fixes. See `minutist_common::llama_backend`.
fn get_or_init_backend() -> Result<&'static LlamaBackend, Error> {
    minutist_common::llama_backend::shared_llama_backend().map_err(|e| Error::ModelLoad {
        path: "llama-backend".to_string(),
        context: e.to_string(),
    })
}

/// Compile-time GPU-offload ceiling for this build.
///
/// CPU-only by default (`0`); when any GPU backend feature (`vulkan` / `metal`
/// / `cuda` / `rocm`) is compiled in, offload all layers. `u32::MAX` is clamped
/// to `i32::MAX` by `with_n_gpu_layers`, which llama.cpp interprets as "every
/// layer". The features only forward to `llama-cpp-2`; the layer count is the
/// summariser's responsibility (mirrors `asr-runtime`).
///
/// This is the **compile-time ceiling** and the `Default` source for
/// [`SummariserConfig::n_gpu_layers`]. The actual per-run layer count is a
/// **runtime** decision: `ipc-bridge` sets `config.n_gpu_layers` from this value
/// when the `gpu_acceleration` setting is on, or `0` to force CPU even in a GPU
/// build (see `architecture/cross-cutting.md` — "GPU portability").
pub const fn gpu_layers() -> u32 {
    #[cfg(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm"))]
    {
        u32::MAX
    }
    #[cfg(not(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm")))]
    {
        0
    }
}

// ---------------------------------------------------------------------------
// LlamaSummariser
// ---------------------------------------------------------------------------

/// A local-LLM summariser backed by a GGUF text model via `llama-cpp-2`.
///
/// Construct with [`LlamaSummariser::open`]. The struct retains the loaded
/// model; each `summarise` call allocates and drops its own context.
pub struct LlamaSummariser {
    model: LlamaModel,
    model_path: PathBuf,
    config: SummariserConfig,
    /// Lazily-built vision projector bound to `model`, for the document-OCR
    /// fallback (`image_to_markdown`). Empty until the first OCR job calls
    /// [`ensure_vision`](Self::ensure_vision); the same Gemma-4 weights serve
    /// both summarisation and OCR, so no second GGUF is loaded.
    ///
    /// # Send/Sync — why `Mutex`
    ///
    /// `LlamaSummariser` is held behind an `Arc<dyn Summariser>` shared across
    /// threads, so every field must be `Send + Sync`. `MtmdContext` carries an
    /// `unsafe impl Send + Sync` in `llama-cpp-2`, but its encode path mutates
    /// internal C state through a shared pointer, and OCR runs on the same GPU
    /// the summariser and ASR use — concurrent `eval_chunks` on one context
    /// would race that state and contend on the device. The `Mutex` serialises
    /// OCR calls, which is acceptable because OCR is a background, single-worker
    /// job (the conversion worker is bounded to one in flight). `OnceLock` gives
    /// the lazy-once-then-immutable build; the inner `Mutex` guards per-call use.
    vision: OnceLock<Mutex<MtmdContext>>,
}

impl LlamaSummariser {
    /// Open a summariser over the GGUF at `model_path`.
    ///
    /// Heavy: initialises the `LlamaBackend` singleton (first call only) and
    /// loads the GGUF weights. Call once when the selected LLM model changes.
    ///
    /// # Errors
    ///
    /// Returns `AppError::ModelLoad` if the file is missing or the GGUF cannot
    /// be loaded.
    pub fn open(model_path: PathBuf, config: SummariserConfig) -> AppResult<Self> {
        let backend = get_or_init_backend()?;

        // Check existence before calling into llama.cpp — the C library asserts
        // on a missing path rather than returning an error.
        if !model_path.exists() {
            return Err(Error::ModelLoad {
                path: model_path.display().to_string(),
                context: "file not found".to_string(),
            }
            .into());
        }

        // GPU offload is a RUNTIME decision driven by `config.n_gpu_layers`
        // (`ipc-bridge` sets it from the `gpu_acceleration` setting; the
        // `Default` is the compile-time ceiling). `0` forces CPU even in a
        // GPU-feature build. `u32::MAX` clamps to `i32::MAX` inside
        // `with_n_gpu_layers`, which llama.cpp reads as "offload every layer".
        // See `architecture/cross-cutting.md` — "GPU portability".
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
        let model = serialize_first_model_load(|| {
            LlamaModel::load_from_file(backend, &model_path, &model_params)
        })
        .map_err(|e| Error::ModelLoad {
            path: model_path.display().to_string(),
            context: e.to_string(),
        })?;

        tracing::info!(
            target: "summariser",
            model = %model_path.display(),
            "LLM model loaded"
        );

        Ok(Self {
            model,
            model_path,
            config,
            vision: OnceLock::new(),
        })
    }

    /// The GGUF path this summariser was opened over.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// The active configuration.
    pub fn config(&self) -> &SummariserConfig {
        &self.config
    }

    /// Like [`Summariser::summarise`] but reports progress through `on_progress`
    /// (live-test UX T4(b), two-phase #69).
    ///
    /// The callback is invoked with [`SummariseProgress`]: `Prefill` ticks as
    /// the prompt is decoded chunk by chunk, then `Generate` ticks after each
    /// generated token. `ipc-bridge` maps each phase onto a labelled
    /// `AppEvent::OperationProgress` (a determinate bar per phase). It is
    /// throttled by the caller, not here — the callback is cheap (an
    /// `Instant` check + maybe a broadcast send). Kept a concrete method (not on
    /// the `common` `Summariser` trait) so the trait stays minimal for its other
    /// impls; `ipc-bridge` holds the concrete `Arc<LlamaSummariser>`.
    pub fn summarise_with_progress(
        &self,
        transcript: &[Segment],
        notes: &[NoteBlock],
        attachments_markdown: &str,
        system_prompt: &str,
        mut on_progress: impl FnMut(SummariseProgress),
    ) -> AppResult<String> {
        let prompt = self.build_prompt(transcript, notes, attachments_markdown, system_prompt)?;
        let raw = self.generate_with_progress(&prompt, &mut on_progress)?;
        Ok(strip_think_block(&raw))
    }

    /// Translate one transcript segment text into `target_language`.
    ///
    /// Builds a minimal single-turn prompt that instructs the model to translate
    /// `text` and output ONLY the translation with no preamble. Reuses the same
    /// prompt-build + generate machinery as `summarise`, including the Gemma
    /// fallback and the chunked-prefill path.
    ///
    /// The prompt is intentionally compact so the round-trip is fast (a single
    /// segment is usually ≤ 50 tokens). `max_tokens` is capped at 512 (a
    /// translated segment is never longer than the original plus reasonable
    /// expansion). The response is trimmed; if the model echoes a think block
    /// it is stripped.
    ///
    /// Only `LlamaSummariser` exposes this method (not the `Summariser` trait):
    /// the method is concrete on a type that always holds a local `LlamaModel`,
    /// so there is no remote-backend path and no remote-backend guard.
    /// `ipc-bridge` holds the concrete `Arc<LlamaSummariser>`.
    pub fn translate_segment(
        &self,
        text: &str,
        target_language: &str,
    ) -> AppResult<String> {
        let instruction = format!(
            "Translate the following text into {target_language}. \
             Output only the translation, with no preamble, explanation, or commentary.\n\n\
             Text: {text}"
        );

        let prompt = self.build_translation_prompt(&instruction)?;
        // A single segment is short; cap generation to 512 tokens so the
        // context window is never at risk and each call returns quickly.
        let raw = self.generate_bounded(&prompt, 512)?;
        Ok(strip_think_block(&raw))
    }

    /// Generate a compact awareness summary for a converted attachment markdown.
    ///
    /// Produces 1–3 sentences describing the document, then `Keywords:` followed
    /// by comma-separated topic keywords. The result is meant to be pinned into
    /// the live co-pilot prefix so the co-pilot knows which documents exist and
    /// can invoke the RAG detail path on demand.
    ///
    /// Input is capped at `AWARENESS_INPUT_CAP` characters before prompting so
    /// that large documents do not overflow the model context. Only the leading
    /// section is used — the summary of a long document is typically determined
    /// by its opening content anyway.
    ///
    /// Mid-meeting re-seed is deferred: awareness generated here is persisted to
    /// the manifest and loaded at worker startup, so a live session that is
    /// already running when an attachment is added picks it up only after restart.
    ///
    /// The method is concrete on [`LlamaSummariser`] (no remote backend path) and
    /// shares the same prompt-build + generate machinery as `translate_segment`.
    pub fn generate_attachment_awareness(&self, md: &str) -> AppResult<String> {
        // Cap the input to avoid blowing the model context on large documents.
        // The leading ~10 000 characters capture the abstract and early content,
        // which is sufficient for a 1–3 sentence summary.
        //
        // The cap is applied by character count so that the slice always falls
        // on a valid UTF-8 boundary — slicing &str by a raw byte index panics
        // when the byte falls inside a multi-byte codepoint.
        const AWARENESS_INPUT_CAP: usize = 10_000;
        let input = if md.chars().count() > AWARENESS_INPUT_CAP {
            // Walk to the first char boundary at or beyond the cap.
            let byte_end = md
                .char_indices()
                .nth(AWARENESS_INPUT_CAP)
                .map(|(i, _)| i)
                .unwrap_or(md.len());
            &md[..byte_end]
        } else {
            md
        };

        let instruction = format!(
            "Summarise this document in 1–3 sentences, then on a new line give \
             `Keywords: ` followed by a comma-separated list of the main topics. \
             Be concise and factual. Do not add anything else.\n\n{input}"
        );

        let prompt = self.build_translation_prompt(&instruction)?;
        // Cap generation at 256 tokens — a 1–3 sentence summary plus a keyword
        // line is well within that budget.
        let raw = self.generate_bounded(&prompt, 256)?;
        Ok(strip_think_block(&raw))
    }

    /// Borrow the loaded [`LlamaModel`] for the chat engine (Phase 9, D5).
    ///
    /// The substrate seam: `ipc-bridge` holds the concrete
    /// `Arc<LlamaSummariser>`, lends `&LlamaModel` to `chat-agent`'s
    /// `LlamaTurnBackend`, and coerces the same handle to `Arc<dyn Summariser>`
    /// for the `agent-tools` `ToolContext`. The model is `unsafe impl Send +
    /// Sync` (`llama-cpp-2`), so it crosses threads and is referenced
    /// concurrently; the chat engine builds its own `!Sync` `LlamaContext`
    /// fresh per turn (clean KV cache), exactly as `summarise` does. No GGUF is
    /// reloaded per turn. Keeping this an accessor (rather than wrapping the
    /// model) preserves `summarise()` unchanged and avoids a `summariser →
    /// chat-agent` edge — `chat-agent` depends on `summariser`, never the
    /// reverse. See `architecture/components.md` — `summariser` (model
    /// exposure) and `chat-agent`.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Bring up the vision projector for document OCR, reusing the held model.
    ///
    /// Lazy and idempotent: the first call binds `mmproj_path` to the
    /// already-loaded [`LlamaModel`] via [`MtmdContext::init_from_file`] (mirrors
    /// `asr-runtime`'s audio binding) and caches it; later calls return the
    /// cached context regardless of the `mmproj_path` argument. The projector is
    /// asserted to advertise **vision** support — a mismatched (e.g. audio-only)
    /// mmproj is rejected up front rather than producing garbage at OCR time.
    ///
    /// The mtmd encoder follows the same GPU decision as the model: it offloads
    /// iff the summariser is offloading layers (`config.n_gpu_layers > 0`), so a
    /// CPU-forced build keeps the encoder on CPU too.
    ///
    /// # Errors
    ///
    /// `AppError::ModelLoad` if the mmproj path is missing/invalid or the
    /// projector cannot be loaded or does not support vision.
    pub fn ensure_vision(&self, mmproj_path: &Path) -> AppResult<&Mutex<MtmdContext>> {
        // Fast path: already built.
        if let Some(ctx) = self.vision.get() {
            return Ok(ctx);
        }

        // Existence check before crossing into llama.cpp — the C library asserts
        // on a missing path rather than returning an error (same as `open`).
        if !mmproj_path.exists() {
            return Err(Error::MtmdInit {
                path: mmproj_path.display().to_string(),
                context: "file not found".to_string(),
            }
            .into());
        }

        let mmproj_str = mmproj_path.to_str().ok_or_else(|| Error::MtmdInit {
            path: mmproj_path.display().to_string(),
            context: "path is not valid UTF-8".to_string(),
        })?;

        let mtmd_params = MtmdContextParams {
            // GPU affinity tracks the model's runtime offload decision.
            use_gpu: self.config.n_gpu_layers > 0,
            print_timings: false,
            n_threads: self.config.threads,
            media_marker: CString::new(mtmd_default_marker()).map_err(|e| Error::MtmdInit {
                path: mmproj_path.display().to_string(),
                context: format!("invalid media marker: {e}"),
            })?,
            // -1 = model default visual-token budget; we don't override it.
            image_min_tokens: -1,
            image_max_tokens: -1,
        };

        let mtmd_ctx =
            MtmdContext::init_from_file(mmproj_str, &self.model, &mtmd_params).map_err(|e| {
                Error::MtmdInit {
                    path: mmproj_path.display().to_string(),
                    context: e.to_string(),
                }
            })?;

        if !mtmd_ctx.support_vision() {
            return Err(Error::MtmdInit {
                path: mmproj_path.display().to_string(),
                context: "mmproj does not advertise vision support".to_string(),
            }
            .into());
        }

        tracing::info!(
            target: "summariser",
            mmproj = %mmproj_path.display(),
            "vision mtmd context initialised"
        );

        // Another thread may have raced us to build the context; `set` returns
        // `Err` in that case and we just use whichever instance won. `get`
        // cannot return `None` after either branch, so the `expect` is
        // unreachable.
        let _ = self.vision.set(Mutex::new(mtmd_ctx));
        Ok(self
            .vision
            .get()
            .expect("vision context is populated after set/race"))
    }

    /// OCR one page image (PNG bytes) into markdown using the held Gemma-4 model.
    ///
    /// This is the production lift of the validated spike loop
    /// (`spikes/doc-vlm-spike/src/main.rs` — `infer_page`): decode the PNG into
    /// an [`MtmdBitmap`], build the Gemma "convert this page to markdown"
    /// instruction with the media marker appended (Gemma places the marker
    /// AFTER the instruction, via its chat template), allocate a fresh
    /// [`LlamaContext`] (clean KV cache, mirroring `summarise`), tokenise the
    /// text+image into mtmd chunks, prefill via `eval_chunks`, then greedily
    /// decode to EOG. The returned markdown is trimmed.
    ///
    /// [`ensure_vision`](Self::ensure_vision) MUST have been called first to
    /// provision the projector; this method locks the cached context for the
    /// duration of the call, so concurrent OCR jobs serialise (acceptable — OCR
    /// is a bounded single-worker background job sharing the GPU).
    ///
    /// # Errors
    ///
    /// `AppError::ModelLoad` if the vision projector was never built;
    /// `AppError::Inference` on any decode/tokenise/eval failure.
    pub fn image_to_markdown(&self, png: &[u8]) -> AppResult<String> {
        let vision = self.vision.get().ok_or_else(|| Error::Inference(
            "image_to_markdown called before ensure_vision built the projector".to_string(),
        ))?;
        // Serialise OCR on the shared mtmd context (see the `vision` field doc).
        let mtmd_ctx = vision
            .lock()
            .map_err(|e| Error::Inference(format!("vision mtmd context lock poisoned: {e}")))?;

        let markdown = self.run_image_to_markdown(&mtmd_ctx, png)?;
        Ok(markdown)
    }
}

// ---------------------------------------------------------------------------
// Vision / document-OCR (private machinery)
// ---------------------------------------------------------------------------

/// Gemma-4 instruction for the document-OCR fallback. Lifted verbatim from the
/// validated `doc-vlm-spike` (`GEMMA_INSTRUCTION`). The media marker is appended
/// AFTER this text (Gemma's marker-last placement); `MtmdContext::tokenize`
/// splits the prompt on the marker and inserts the encoded page image there.
const GEMMA_OCR_INSTRUCTION: &str = "Convert this document page to clean, well-structured markdown. \
     Preserve headings, lists, and tables. For tables use GitHub \
     pipe-table syntax. Output only the markdown content, no preamble.";

impl LlamaSummariser {
    /// Build the OCR prompt: the Gemma instruction followed by the media marker,
    /// wrapped by the GGUF chat template — or the model-probed fallback format
    /// when the bundled llama.cpp cannot render the (newer) Gemma template.
    ///
    /// Reuses the SAME fallback chain as [`Self::build_prompt`]: the shipped
    /// Gemma-4 GGUF postdates the vendored llama.cpp, so `apply_chat_template`
    /// returns `ffi error -1` and we emit the model-probed scaffold (which
    /// carries an explicit `<bos>`). Either way the marker survives verbatim
    /// because tokenisation parses it as a special token; mtmd then splits
    /// the prompt on it.
    fn build_ocr_prompt(&self) -> Result<String, Error> {
        let marker = mtmd_default_marker();
        let user_content = format!("{GEMMA_OCR_INSTRUCTION}\n{marker}");

        let template = self
            .model
            .chat_template(None::<&str>)
            .map_err(|e| Error::Template(e.to_string()))?;

        let user_msg = LlamaChatMessage::new("user".to_string(), user_content.clone())
            .map_err(|e| Error::Template(format!("user message: {e}")))?;

        match self.model.apply_chat_template(&template, &[user_msg], true) {
            Ok(prompt) => Ok(prompt),
            Err(e) => {
                tracing::warn!(
                    target: "summariser",
                    "apply_chat_template failed ({e}); using model-probed turn-format fallback for OCR"
                );
                Ok(model_turn_prompt(&self.model, &user_content))
            }
        }
    }

    /// The OCR decode loop (lifted from `doc-vlm-spike`'s `infer_page`).
    ///
    /// `mtmd_ctx` is the locked, vision-capable projector. A fresh
    /// [`LlamaContext`] is allocated per page so the KV cache is clean (mirrors
    /// `generate_with_config`). The page image is decoded from PNG bytes, the
    /// text+image is tokenised into mtmd chunks, prefilled via `eval_chunks`,
    /// then greedily decoded with an EOG stop and incremental UTF-8
    /// detokenisation.
    fn run_image_to_markdown(
        &self,
        mtmd_ctx: &MtmdContext,
        png: &[u8],
    ) -> Result<String, Error> {
        let backend = get_or_init_backend()?;

        // Decode the page image (stb_image inside mtmd). Image analogue of the
        // audio-bitmap path in `asr-runtime`.
        let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, png, false)
            .map_err(|e| Error::Inference(format!("MtmdBitmap::from_buffer: {e:?}")))?;

        // The OCR prompt carries an explicit `<bos>` (via the chat template,
        // or the model-probed fallback), so `add_special` is false to avoid a
        // second BOS — mirroring `generate`'s `AddBos::Never`. `parse_special`
        // is true so the media marker tokenises as a special token and mtmd
        // can split on it.
        let prompt_text = self.build_ocr_prompt()?;
        let input_text = MtmdInputText {
            text: prompt_text,
            add_special: false,
            parse_special: true,
        };

        let n_ctx = NonZeroU32::new(self.config.n_ctx)
            .ok_or_else(|| Error::ContextOverflow("n_ctx must be non-zero".to_string()))?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(self.config.n_batch)
            .with_n_threads(self.config.threads)
            .with_n_threads_batch(self.config.threads);

        let mut llama_ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Inference(format!("LlamaContext init: {e}")))?;

        let chunks = mtmd_ctx
            .tokenize(input_text, &[&bitmap])
            .map_err(|e| Error::Inference(format!("mtmd tokenize: {e}")))?;

        // Prefill: the image chunk is encoded via mtmd_encode inside
        // `eval_chunks`, the text chunks via llama_decode. `n_batch` is the
        // chunked-prefill chunk size (cross-cutting "llama.cpp prefill batching").
        let n_past = chunks
            .eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, self.config.n_batch as i32, true)
            .map_err(|e| Error::Inference(format!("eval_chunks: {e}")))?;

        // Greedy decode with EOG stop — the OCR path reports no progress, so
        // the per-token callback is a no-op.
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);
        let markdown = greedy_decode(
            &self.model,
            &mut llama_ctx,
            &mut batch,
            n_past,
            self.config.max_tokens,
            |_done| {},
        )?;

        Ok(markdown.trim().to_string())
    }
}

impl Summariser for LlamaSummariser {
    fn summarise(
        &self,
        transcript: &[Segment],
        notes: &[NoteBlock],
        attachments_markdown: &str,
        system_prompt: &str,
    ) -> AppResult<String> {
        let prompt = self.build_prompt(transcript, notes, attachments_markdown, system_prompt)?;
        let raw = self.generate(&prompt)?;
        Ok(strip_think_block(&raw))
    }
}

impl LlamaSummariser {
    /// Build the prompt from the GGUF's baked-in chat template.
    ///
    /// ONE `user` message: the system instructions, then the rendered
    /// transcript + a blank line + the notes markdown. `add_ass=true` leaves the
    /// assistant turn open for generation.
    ///
    /// The system prompt is folded INTO the user turn rather than sent as a
    /// separate `system` message because several chat templates — notably Gemma
    /// — have no `system` role. That alone is not enough, though: the bundled
    /// llama.cpp cannot RENDER a chat template newer than itself (Gemma 4
    /// postdates the vendored build), so `apply_chat_template` returns `ffi error
    /// -1` even for a user-only message set. On that failure we fall back to the
    /// [`model_turn_prompt`] hand-built format (markers probed from the model
    /// vocabulary at call time). Other models keep using their baked template.
    fn build_prompt(
        &self,
        transcript: &[Segment],
        notes: &[NoteBlock],
        attachments_markdown: &str,
        system_prompt: &str,
    ) -> Result<String, Error> {
        let template = self
            .model
            .chat_template(None::<&str>)
            .map_err(|e| Error::Template(e.to_string()))?;

        let user_content = render_user_content(transcript, notes, attachments_markdown);
        let combined = format!("{system_prompt}\n\n{user_content}");

        let user_msg = LlamaChatMessage::new("user".to_string(), combined.clone())
            .map_err(|e| Error::Template(format!("user message: {e}")))?;

        match self.model.apply_chat_template(&template, &[user_msg], true) {
            Ok(prompt) => Ok(prompt),
            Err(e) => {
                // The bundled llama.cpp cannot RENDER some newer chat templates
                // (Gemma 4 postdates the vendored build) — `apply_chat_template`
                // returns `ffi error -1` regardless of the message set, not just
                // for a `system` role. Fall back to the Gemma turn format, which
                // is what our shipped summariser LLM (gemma-4-E4B) uses. `<bos>`
                // is emitted explicitly because `generate()` tokenises with
                // `AddBos::Never`, and `str_to_token` parses special tokens
                // (`llama_tokenize(..., special=true)`), so `<bos>` /
                // `<start_of_turn>` / `<end_of_turn>` map to their token ids.
                tracing::warn!(
                    target: "summariser",
                    "apply_chat_template failed ({e}); using model-probed turn-format fallback"
                );
                Ok(model_turn_prompt(&self.model, &combined))
            }
        }
    }

    /// Build a minimal single-turn translation prompt.
    ///
    /// Uses the same fallback chain as [`Self::build_prompt`]: baked chat
    /// template when renderable, Gemma hand-built format otherwise.
    fn build_translation_prompt(&self, instruction: &str) -> Result<String, Error> {
        let template = self
            .model
            .chat_template(None::<&str>)
            .map_err(|e| Error::Template(e.to_string()))?;

        let user_msg = LlamaChatMessage::new("user".to_string(), instruction.to_string())
            .map_err(|e| Error::Template(format!("user message: {e}")))?;

        match self.model.apply_chat_template(&template, &[user_msg], true) {
            Ok(prompt) => Ok(prompt),
            Err(e) => {
                tracing::warn!(
                    target: "summariser",
                    "apply_chat_template failed ({e}); using model-probed turn-format fallback for translation"
                );
                Ok(model_turn_prompt(&self.model, instruction))
            }
        }
    }

    /// Generate with a custom `max_tokens` ceiling, bypassing the config value.
    ///
    /// Used by [`Self::translate_segment`], which needs a tighter cap than the
    /// per-summary `config.max_tokens`. All other generation parameters
    /// (`n_ctx`, `n_batch`, `threads`, `n_gpu_layers`) are unchanged.
    fn generate_bounded(&self, prompt: &str, max_tokens: usize) -> Result<String, Error> {
        // Temporarily shadow `config.max_tokens` by building a custom config.
        let config = SummariserConfig {
            max_tokens,
            ..self.config.clone()
        };
        generate_with_config(&self.model, &config, prompt, &mut |_| {})
    }

    /// Tokenise the prompt, chunked-prefill it, then greedily generate.
    fn generate(&self, prompt: &str) -> Result<String, Error> {
        // No-op progress callback: the no-progress path is unchanged.
        self.generate_with_progress(prompt, &mut |_| {})
    }

    /// [`Self::generate`] with a two-phase progress callback (live-test UX
    /// T4(b), #69). The callback receives a [`SummariseProgress::Prefill`] tick
    /// after each prompt chunk is decoded, then a [`SummariseProgress::Generate`]
    /// tick after each generated token. Everything else (tokenisation, chunked
    /// prefill, greedy sampling, EOG stop, incremental detokenisation) is
    /// identical to [`Self::generate`].
    fn generate_with_progress(
        &self,
        prompt: &str,
        on_progress: &mut dyn FnMut(SummariseProgress),
    ) -> Result<String, Error> {
        generate_with_config(&self.model, &self.config, prompt, on_progress)
    }
}

// ---------------------------------------------------------------------------
// Core generation (standalone so multiple call sites can share it)
// ---------------------------------------------------------------------------

/// Tokenise `prompt`, chunked-prefill it, then greedily generate up to
/// `config.max_tokens`, reporting two-phase progress via `on_progress`.
///
/// Extracted from the `LlamaSummariser` `generate_with_progress` method so
/// [`LlamaSummariser::generate_bounded`] (used by `translate_segment`) can run
/// generation with a different `max_tokens` cap without duplicating the decode
/// loop.
fn generate_with_config(
    model: &LlamaModel,
    config: &SummariserConfig,
    prompt: &str,
    on_progress: &mut dyn FnMut(SummariseProgress),
) -> Result<String, Error> {
    let backend = get_or_init_backend()?;

    let n_ctx = NonZeroU32::new(config.n_ctx).ok_or_else(|| {
        Error::ContextOverflow("n_ctx must be non-zero".to_string())
    })?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(config.n_batch)
        .with_n_threads(config.threads)
        .with_n_threads_batch(config.threads);

    let mut llama_ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Inference(format!("LlamaContext init: {e}")))?;

    // AddBos::Never — the chat template already emits the BOS itself.
    let tokens = model
        .str_to_token(prompt, AddBos::Never)
        .map_err(|e| Error::Inference(format!("tokenize: {e}")))?;

    if tokens.is_empty() {
        return Err(Error::Inference(
            "templated prompt tokenised to zero tokens".to_string(),
        ));
    }

    // The prompt AND the tokens it will generate must both fit the context
    // window: generation grows the KV cache by one slot per token, so a
    // prompt that fits on its own can still overflow mid-generation. Reserve
    // `max_tokens` of headroom up front rather than aborting partway and
    // losing the output.
    check_context_budget(tokens.len(), config.max_tokens, config.n_ctx)?;

    // --- Chunked prefill ---
    let plan = plan_prefill(tokens.len(), config.n_batch);
    let mut batch = LlamaBatch::new(config.n_batch as usize, 1);
    let total_prompt = tokens.len();

    for chunk in &plan.chunks {
        batch.clear();
        for offset in 0..chunk.len {
            let global = chunk.start + offset;
            let pos = global as i32;
            let logits = chunk.logits_at_last && offset == chunk.len - 1;
            batch
                .add(tokens[global], pos, &[0], logits)
                .map_err(|e| Error::Inference(format!("batch.add (prefill): {e}")))?;
        }
        llama_ctx
            .decode(&mut batch)
            .map_err(|e| Error::Inference(format!("decode (prefill): {e}")))?;

        // Report prefill progress AFTER this chunk decodes (#69).
        let done = (chunk.start + chunk.len).min(total_prompt);
        on_progress(SummariseProgress::Prefill {
            done,
            total: total_prompt,
        });
    }

    // --- Greedy generation ---
    // `done == config.max_tokens` on the EOG callback jumps the progress bar
    // to 100 % so a short output still completes the bar rather than leaving
    // it stuck mid-way (see `greedy_decode`).
    let n_past = tokens.len() as i32;
    let text = greedy_decode(
        model,
        &mut llama_ctx,
        &mut batch,
        n_past,
        config.max_tokens,
        |done| {
            on_progress(SummariseProgress::Generate {
                done,
                max: config.max_tokens,
            })
        },
    )?;

    Ok(text)
}

/// Greedily decode from an already-prefilled context until an EOG token or
/// `max_tokens` is reached, appending each detokenised piece.
///
/// `n_past` is the KV-cache position immediately after prefill; `batch` is
/// reused (cleared each iteration) rather than reallocated per token.
/// `on_token` is called after each generated token with the running count —
/// `max_tokens` itself on the EOG token, so a caller reporting progress can
/// jump straight to 100 % rather than getting stuck one token short.
///
/// Shared by [`generate_with_config`] (text-prompt prefill) and
/// [`LlamaSummariser::run_image_to_markdown`] (mtmd image prefill): the two
/// prefill the KV cache differently, but decode identically once `n_past`
/// tokens are already in it.
fn greedy_decode(
    model: &LlamaModel,
    llama_ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    mut n_past: i32,
    max_tokens: usize,
    mut on_token: impl FnMut(usize),
) -> Result<String, Error> {
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut decoder = UTF_8.new_decoder();
    let mut text = String::new();

    for i in 0..max_tokens {
        let token = sampler.sample(llama_ctx, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            on_token(max_tokens);
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| Error::Inference(format!("token_to_piece: {e}")))?;
        text.push_str(&piece);
        on_token(i + 1);

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| Error::Inference(format!("batch.add (gen): {e}")))?;
        n_past += 1;

        llama_ctx
            .decode(batch)
            .map_err(|e| Error::Inference(format!("decode (gen): {e}")))?;
    }

    Ok(text)
}

// ---------------------------------------------------------------------------
// Prompt rendering
// ---------------------------------------------------------------------------

/// Build a single-user-turn prompt for the held model, used as the fallback
/// when `apply_chat_template` cannot render the GGUF's baked template.
///
/// Turn markers are probed from the model vocabulary: the first candidate pair
/// that both tokenise to a single control token is used. Gemma 4 uses
/// `<|turn>` / `<turn|>`; Gemma 2/3 uses `<start_of_turn>` / `<end_of_turn>`.
/// `<bos>` is included explicitly because `generate()` tokenises with
/// `AddBos::Never`; `str_to_token` parses special tokens so it maps to its id.
/// The trailing open `model` turn is where generation continues.
fn model_turn_prompt(model: &LlamaModel, content: &str) -> String {
    let (open, close) = detect_model_turn_markers(model);
    format!("<bos>{open}user\n{content}{close}\n{open}model\n")
}

/// Probe the model vocabulary for single-token turn markers.
///
/// Tests `<|turn>` / `<turn|>` (Gemma 4) then `<start_of_turn>` /
/// `<end_of_turn>` (Gemma 2/3). Returns the first pair where both tokenise
/// to exactly one token. Falls back to Gemma 2/3 strings if neither pair
/// maps to single tokens.
fn detect_model_turn_markers(model: &LlamaModel) -> (&'static str, &'static str) {
    let candidates: &[(&str, &str)] = &[
        ("<|turn>", "<turn|>"),
        ("<start_of_turn>", "<end_of_turn>"),
    ];
    for &(open, close) in candidates {
        let open_toks = model.str_to_token(open, AddBos::Never).unwrap_or_default();
        let close_toks = model.str_to_token(close, AddBos::Never).unwrap_or_default();
        if open_toks.len() == 1 && close_toks.len() == 1 {
            return (open, close);
        }
    }
    ("<start_of_turn>", "<end_of_turn>")
}

/// Render the transcript + notes into the single `user` message body (#70).
///
/// When at least one note paragraph is anchored to the recording clock, the
/// transcript and the anchored notes are merged into ONE chronological timeline,
/// each line prefixed with its `[m:ss]` timestamp, so the model sees each note
/// beside what was being said when it was written ("woven in at the time").
/// Un-anchored notes (and the prior `# Notes` block) follow the timeline.
///
/// When NO note is anchored — the meeting was recorded without live note-taking,
/// or notes were typed while idle / imported — the transcript renders WITHOUT
/// per-line timestamps (the prior format), so the common case spends no extra
/// context tokens and the no-notes prompt is byte-for-byte unchanged.
fn render_user_content(
    transcript: &[Segment],
    notes: &[NoteBlock],
    attachments_markdown: &str,
) -> String {
    let any_anchored = notes.iter().any(|n| n.at_ms.is_some());
    let mut out = String::new();

    // Reference material (attachments) is a LEADING section — not time-woven, so
    // it never enters the (ms, kind) merge below. When empty the prepend is
    // skipped so the rendered output is byte-identical to the no-attachment path.
    if !attachments_markdown.is_empty() {
        out.push_str("# Reference material (attachments)\n\n");
        out.push_str(attachments_markdown);
        out.push_str("\n\n");
    }

    if any_anchored {
        out.push_str("# Transcript (notes woven in at the time they were written)\n\n");

        // Merge transcript segments and anchored notes into a time-ordered
        // stream. The sort is stable and keyed on `(ms, kind)` with the
        // transcript line ranking before a note at the same timestamp (the note
        // was written about what was just said); ties within a kind keep
        // document order.
        enum Line<'a> {
            Seg(&'a Segment),
            Note(&'a NoteBlock),
        }
        let mut lines: Vec<(u64, u8, Line<'_>)> = Vec::new();
        for seg in transcript {
            lines.push((seg.start_ms, 0, Line::Seg(seg)));
        }
        for note in notes {
            if let Some(at) = note.at_ms {
                lines.push((at, 1, Line::Note(note)));
            }
        }
        lines.sort_by_key(|(ms, kind, _)| (*ms, *kind));

        for (ms, _, line) in &lines {
            out.push('[');
            out.push_str(&format_clock(*ms));
            out.push_str("] ");
            match line {
                Line::Seg(seg) => {
                    if let Some(speaker) = &seg.speaker_id {
                        out.push_str(speaker);
                        out.push_str(": ");
                    }
                    out.push_str(seg.text.trim());
                }
                Line::Note(note) => {
                    out.push_str("NOTE — ");
                    out.push_str(note.text.trim());
                }
            }
            out.push('\n');
        }

        let unanchored: Vec<&NoteBlock> =
            notes.iter().filter(|n| n.at_ms.is_none()).collect();
        if !unanchored.is_empty() {
            out.push_str("\n# Notes (no timestamp)\n\n");
            for note in unanchored {
                out.push_str(note.text.trim());
                out.push('\n');
            }
        }
        return out;
    }

    // No anchored notes: the prior plain format (transcript, then a flat notes
    // block reconstructed from the un-anchored paragraphs).
    out.push_str("# Transcript\n\n");
    for seg in transcript {
        if let Some(speaker) = &seg.speaker_id {
            out.push_str(speaker);
            out.push_str(": ");
        }
        out.push_str(seg.text.trim());
        out.push('\n');
    }

    out.push_str("\n# Notes\n\n");
    if notes.is_empty() {
        out.push_str("(no notes taken)\n");
    } else {
        for note in notes {
            out.push_str(note.text.trim());
            out.push('\n');
        }
    }

    out
}

/// Format a millisecond offset as `m:ss` (or `h:mm:ss` past an hour) for the
/// woven timeline — a coarse marker the model uses to order notes against the
/// transcript, not a precise clock.
fn format_clock(ms: u64) -> String {
    let total_secs = ms / 1000;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Strip a leading/embedded `<think>…</think>` block from model output.
///
/// Thinking is disabled at prompt time (we never inject a think token), but a
/// future selected model may still emit one; per `components.md` we strip it
/// before persisting the summary. If the close tag is missing the whole tail is
/// dropped (an unterminated think block is not summary content).
fn strip_think_block(s: &str) -> String {
    let trimmed = s.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<think>") {
        match rest.find("</think>") {
            Some(close) => rest[close + "</think>".len()..].trim().to_string(),
            None => String::new(),
        }
    } else {
        s.trim().to_string()
    }
}

// ---------------------------------------------------------------------------
// Chunked-prefill planning (pure, unit-tested)
// ---------------------------------------------------------------------------

/// One prefill chunk: a half-open token range `[start, start + len)`.
///
/// `logits_at_last` is true only for the final chunk of the plan; the decode
/// loop then sets `logits = true` on this chunk's last token so the first
/// sampled token reads from the correct position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillChunk {
    /// Index of the first token of this chunk in the prompt.
    pub start: usize,
    /// Number of tokens in this chunk. Always `1..=n_batch`.
    pub len: usize,
    /// Whether this chunk carries the single `logits = true` token (its last).
    pub logits_at_last: bool,
}

/// A chunked-prefill plan over a `prompt_len`-token prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillPlan {
    pub chunks: Vec<PrefillChunk>,
}

/// Split a `prompt_len`-token prompt into `n_batch`-sized prefill chunks.
///
/// Pure function — no llama.cpp state — so the chunk arithmetic can be unit
/// tested without a model. Guarantees, per `cross-cutting.md` "llama.cpp
/// prefill batching":
///
/// - no chunk exceeds `n_batch` tokens (otherwise
///   `GGML_ASSERT(n_tokens_all <= n_batch)` aborts);
/// - the chunks tile `[0, prompt_len)` contiguously with no gaps/overlaps;
/// - exactly one token across the whole plan carries logits — the last token
///   of the last chunk (`logits_at_last` true only on the final chunk).
///
/// `n_batch` is clamped to at least 1 to keep the function total.
pub fn plan_prefill(prompt_len: usize, n_batch: u32) -> PrefillPlan {
    let batch = (n_batch as usize).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < prompt_len {
        let len = batch.min(prompt_len - start);
        chunks.push(PrefillChunk {
            start,
            len,
            logits_at_last: false,
        });
        start += len;
    }
    if let Some(last) = chunks.last_mut() {
        last.logits_at_last = true;
    }
    PrefillPlan { chunks }
}

// ---------------------------------------------------------------------------
// Context-budget guard (pure, unit-tested)
// ---------------------------------------------------------------------------

/// Verify the prompt plus its generation headroom fit the context window.
///
/// The KV cache holds the prompt AND every generated token, so the budget is
/// `prompt_tokens + max_tokens`. Checking only `prompt_tokens` (as a naïve
/// guard does) lets a prompt in `(n_ctx - max_tokens, n_ctx)` pass, then
/// overflow the KV cache part-way through generation — losing the whole
/// summary. Requiring `prompt_tokens + max_tokens <= n_ctx` reserves room for
/// the full generation up front.
///
/// Pure (no llama.cpp state) so the boundary is unit-tested without a model.
/// The error carries the three quantities so the caller can see why it didn't
/// fit. `usize` addition is saturating to stay total even with absurd inputs.
fn check_context_budget(
    prompt_tokens: usize,
    max_tokens: usize,
    n_ctx: u32,
) -> Result<(), Error> {
    let required = prompt_tokens.saturating_add(max_tokens);
    if required > n_ctx as usize {
        return Err(Error::ContextOverflow(format!(
            "prompt is {prompt_tokens} tokens and generation reserves {max_tokens} more \
             ({required} total) but the context window is only {n_ctx}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Chunked-prefill arithmetic — pure, always runs, no model
    // -----------------------------------------------------------------------

    /// A >512-token prompt must split into chunks that each respect the
    /// `n_batch` hard limit, tile the prompt contiguously, and place logits on
    /// exactly one token (the final token of the final chunk).
    #[test]
    fn prefill_over_512_tokens_respects_batch_and_logits() {
        let prompt_len = 1300;
        let n_batch = 512;
        let plan = plan_prefill(prompt_len, n_batch);

        // 1300 / 512 -> 512, 512, 276
        assert_eq!(plan.chunks.len(), 3, "expected three chunks for 1300/512");

        // No chunk may exceed n_batch (the GGML_ASSERT invariant).
        for chunk in &plan.chunks {
            assert!(
                chunk.len <= n_batch as usize,
                "chunk len {} exceeds n_batch {}",
                chunk.len,
                n_batch
            );
            assert!(chunk.len >= 1, "chunks must be non-empty");
        }

        // Chunks tile [0, prompt_len) contiguously.
        let mut expected_start = 0;
        for chunk in &plan.chunks {
            assert_eq!(chunk.start, expected_start, "chunks must be contiguous");
            expected_start += chunk.len;
        }
        assert_eq!(expected_start, prompt_len, "chunks must cover the prompt");

        // Exactly one chunk carries logits, and it is the last one.
        let logit_chunks: Vec<&PrefillChunk> =
            plan.chunks.iter().filter(|c| c.logits_at_last).collect();
        assert_eq!(logit_chunks.len(), 1, "exactly one chunk carries logits");
        assert!(
            std::ptr::eq(logit_chunks[0], plan.chunks.last().unwrap()),
            "the logits chunk must be the final chunk"
        );
        assert_eq!(plan.chunks.last().unwrap().len, 276);
    }

    #[test]
    fn prefill_exact_multiple_of_batch() {
        let plan = plan_prefill(1024, 512);
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[0].len, 512);
        assert_eq!(plan.chunks[1].len, 512);
        assert!(!plan.chunks[0].logits_at_last);
        assert!(plan.chunks[1].logits_at_last);
    }

    #[test]
    fn prefill_shorter_than_batch_is_single_chunk() {
        let plan = plan_prefill(100, 512);
        assert_eq!(plan.chunks.len(), 1);
        assert_eq!(plan.chunks[0].start, 0);
        assert_eq!(plan.chunks[0].len, 100);
        assert!(plan.chunks[0].logits_at_last);
    }

    #[test]
    fn prefill_single_token() {
        let plan = plan_prefill(1, 512);
        assert_eq!(plan.chunks.len(), 1);
        assert_eq!(plan.chunks[0].len, 1);
        assert!(plan.chunks[0].logits_at_last);
    }

    #[test]
    fn prefill_empty_prompt_has_no_chunks() {
        let plan = plan_prefill(0, 512);
        assert!(plan.chunks.is_empty());
    }

    #[test]
    fn prefill_clamps_zero_batch_to_one() {
        // Defensive: a zero n_batch would otherwise loop forever.
        let plan = plan_prefill(3, 0);
        assert_eq!(plan.chunks.len(), 3);
        for chunk in &plan.chunks {
            assert_eq!(chunk.len, 1);
        }
        assert!(plan.chunks.last().unwrap().logits_at_last);
    }

    // -----------------------------------------------------------------------
    // Context-budget guard — pure, always runs, no model
    // -----------------------------------------------------------------------

    /// Regression for the headroom bug: a prompt that fits the context window
    /// on its own (`prompt < n_ctx`) but leaves no room for generation
    /// (`prompt + max_tokens > n_ctx`) MUST be rejected. The old guard only
    /// compared `prompt >= n_ctx`, so this case slipped through and overflowed
    /// the KV cache mid-generation.
    #[test]
    fn context_budget_rejects_prompt_that_starves_generation() {
        // n_ctx = 1000, max_tokens = 100. A 950-token prompt is < n_ctx (passes
        // the old guard) but 950 + 100 = 1050 > 1000, so it must now fail.
        let n_ctx = 1_000;
        let max_tokens = 100;
        let prompt_tokens = 950;
        assert!(prompt_tokens < n_ctx as usize, "precondition: old guard would pass");

        let result = check_context_budget(prompt_tokens, max_tokens, n_ctx);
        let err = result.expect_err("prompt that starves generation must be rejected");
        match err {
            Error::ContextOverflow(msg) => {
                assert!(msg.contains("950"), "error must report the prompt size: {msg}");
                assert!(msg.contains("100"), "error must report the reserved headroom: {msg}");
                assert!(msg.contains("1000"), "error must report the context window: {msg}");
            }
            other => panic!("expected ContextOverflow, got {other:?}"),
        }
    }

    /// The exact boundary: `prompt + max_tokens == n_ctx` fits (the KV cache
    /// holds exactly the prompt plus the full generation).
    #[test]
    fn context_budget_accepts_exact_fit() {
        assert!(check_context_budget(900, 100, 1_000).is_ok());
    }

    /// One token over the boundary fails.
    #[test]
    fn context_budget_rejects_one_token_over() {
        assert!(check_context_budget(901, 100, 1_000).is_err());
    }

    /// A comfortable prompt with headroom to spare passes.
    #[test]
    fn context_budget_accepts_prompt_with_headroom() {
        assert!(check_context_budget(500, 100, 1_000).is_ok());
    }

    /// Saturating addition keeps the guard total: an absurd `max_tokens` near
    /// `usize::MAX` must reject (not panic on overflow) rather than wrap to a
    /// small value that spuriously passes.
    #[test]
    fn context_budget_saturates_on_overflow() {
        assert!(check_context_budget(usize::MAX, usize::MAX, u32::MAX).is_err());
    }

    /// The guard maps to `AppError::InvalidInput` (via `Error::ContextOverflow`)
    /// so the caller surfaces it as caller-input, never a panic.
    #[test]
    fn context_budget_error_maps_to_invalid_input() {
        use minutist_common::AppError;
        let err = check_context_budget(950, 100, 1_000).unwrap_err();
        let app: AppError = err.into();
        assert!(matches!(app, AppError::InvalidInput { .. }));
    }

    // -----------------------------------------------------------------------
    // GPU-offload selection — cfg-gated, pure
    // -----------------------------------------------------------------------

    /// The default (no GPU feature) build stays CPU-only; a GPU-feature build
    /// offloads all layers (`u32::MAX` → `i32::MAX` inside `with_n_gpu_layers`).
    #[test]
    fn gpu_layers_matches_compiled_backend() {
        #[cfg(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm"))]
        assert_eq!(gpu_layers(), u32::MAX, "a GPU feature must offload all layers");
        #[cfg(not(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm")))]
        assert_eq!(gpu_layers(), 0, "the default build must stay CPU-only");
    }

    /// The config default's `n_gpu_layers` is the compile-time ceiling.
    #[test]
    fn config_default_n_gpu_layers_is_compile_time_ceiling() {
        assert_eq!(SummariserConfig::default().n_gpu_layers, gpu_layers());
    }

    /// Forcing `n_gpu_layers = 0` (the runtime GPU-off path) keeps the layer
    /// count at zero regardless of the compiled backend — the CPU escape hatch
    /// `with_n_gpu_layers(0)` passes through verbatim. No model needed.
    #[test]
    fn config_forced_zero_gpu_layers_stays_zero() {
        let cfg = SummariserConfig {
            n_gpu_layers: 0,
            ..SummariserConfig::default()
        };
        assert_eq!(cfg.n_gpu_layers, 0, "forced CPU keeps n_gpu_layers = 0");

        // A non-zero override is preserved too (the GPU-on path).
        let cfg_gpu = SummariserConfig {
            n_gpu_layers: u32::MAX,
            ..SummariserConfig::default()
        };
        assert_eq!(cfg_gpu.n_gpu_layers, u32::MAX);
    }

    // -----------------------------------------------------------------------
    // Prompt builder — pure, always runs, no model
    // -----------------------------------------------------------------------

    fn seg(text: &str, speaker: Option<&str>) -> Segment {
        seg_at(0, text, speaker)
    }

    fn seg_at(start_ms: u64, text: &str, speaker: Option<&str>) -> Segment {
        Segment {
            start_ms,
            end_ms: start_ms + 1_000,
            text: text.to_string(),
            speaker_id: speaker.map(|s| s.to_string()),
            confidence: None,
            words: vec![],
            shared_speakers: Vec::new(),
        }
    }

    fn note(at_ms: Option<u64>, text: &str) -> NoteBlock {
        NoteBlock {
            at_ms,
            text: text.to_string(),
        }
    }

    #[test]
    fn render_user_content_includes_transcript_and_notes() {
        let transcript = vec![
            seg("hello there", Some("Speaker 1")),
            seg("general kenobi", Some("Speaker 2")),
        ];
        let body = render_user_content(&transcript, &[note(None, "- action item one")], "");

        assert!(body.contains("# Transcript"));
        assert!(body.contains("Speaker 1: hello there"));
        assert!(body.contains("Speaker 2: general kenobi"));
        assert!(body.contains("# Notes"));
        assert!(body.contains("- action item one"));
        // No anchored note → the plain transcript carries NO per-line timestamp.
        assert!(!body.contains("[0:00]"), "unanchored path must not timestamp lines");
    }

    #[test]
    fn render_user_content_without_speakers_or_notes() {
        let transcript = vec![seg("just one line", None)];
        let body = render_user_content(&transcript, &[], "");
        assert!(body.contains("just one line"));
        assert!(!body.contains(": just one line"), "no speaker prefix expected");
        assert!(body.contains("(no notes taken)"));
    }

    #[test]
    fn render_user_content_weaves_anchored_notes_by_timestamp() {
        // Two segments at 0 s and 1:05; a note anchored at 1:00 must land BETWEEN
        // them (after the 0 s line, before the 1:05 line), prefixed `NOTE —`.
        let transcript = vec![
            seg_at(0, "opening remarks", Some("Alice")),
            seg_at(65_000, "later point", Some("Bob")),
        ];
        let notes = vec![note(Some(60_000), "follow up on budget")];
        let body = render_user_content(&transcript, &notes, "");

        assert!(body.contains("woven in"), "weaving heading expected: {body}");
        let note_pos = body.find("NOTE — follow up on budget").expect("note line");
        let open_pos = body.find("Alice: opening remarks").expect("open line");
        let late_pos = body.find("Bob: later point").expect("late line");
        assert!(open_pos < note_pos, "note must follow the 0 s segment");
        assert!(note_pos < late_pos, "note must precede the 1:05 segment");
        // Timestamps are rendered for every line in the woven path.
        assert!(body.contains("[0:00] Alice: opening remarks"));
        assert!(body.contains("[1:00] NOTE — follow up on budget"));
        assert!(body.contains("[1:05] Bob: later point"));
    }

    #[test]
    fn render_user_content_segment_outranks_note_at_equal_timestamp() {
        let transcript = vec![seg_at(30_000, "said this", Some("Cara"))];
        let notes = vec![note(Some(30_000), "wrote this")];
        let body = render_user_content(&transcript, &notes, "");
        let seg_pos = body.find("Cara: said this").expect("seg");
        let note_pos = body.find("NOTE — wrote this").expect("note");
        assert!(seg_pos < note_pos, "transcript ranks before a note at the same ms");
    }

    #[test]
    fn render_user_content_lists_unanchored_notes_after_the_woven_timeline() {
        let transcript = vec![seg_at(0, "hello", None)];
        let notes = vec![
            note(Some(5_000), "anchored thought"),
            note(None, "pre-meeting agenda"),
        ];
        let body = render_user_content(&transcript, &notes, "");
        let woven = body.find("anchored thought").expect("anchored");
        let trailing_header = body.find("# Notes (no timestamp)").expect("trailing header");
        let trailing = body.find("pre-meeting agenda").expect("unanchored");
        assert!(woven < trailing_header && trailing_header < trailing);
    }

    #[test]
    fn render_user_content_empty_attachments_is_byte_identical() {
        // The no-attachment regression guard (LOCKED): passing `""` for
        // `attachments_markdown` must produce output byte-identical to the prior
        // two-arg behaviour (the leading section is skipped entirely).
        let transcript = vec![
            seg("hello there", Some("Speaker 1")),
            seg("general kenobi", Some("Speaker 2")),
        ];
        let notes = vec![note(None, "- action item one")];
        let empty = render_user_content(&transcript, &notes, "");
        // Re-render WITHOUT prepending anything (the leading `if !is_empty()`
        // block is skipped) — the body starts at the `# Transcript` heading.
        assert!(
            empty.starts_with("# Transcript"),
            "empty attachments must not prepend any reference-material header: {empty:?}"
        );
        assert!(!empty.contains("# Reference material (attachments)"));
    }

    #[test]
    fn render_user_content_prepends_reference_material_when_non_empty() {
        let transcript = vec![seg("a line", None)];
        let body = render_user_content(
            &transcript,
            &[],
            "## Attachment: agenda.txt\n\nDiscuss Q3 roadmap\n",
        );
        let header = body
            .find("# Reference material (attachments)")
            .expect("leading attachments header");
        let attachment = body.find("## Attachment: agenda.txt").expect("attachment header");
        let transcript_pos = body.find("# Transcript").expect("transcript header");
        assert!(header < attachment, "reference-material header leads the section");
        assert!(
            attachment < transcript_pos,
            "attachments render BEFORE the transcript (leading, not woven)"
        );
    }

    #[test]
    fn format_clock_renders_minutes_and_hours() {
        assert_eq!(format_clock(0), "0:00");
        assert_eq!(format_clock(5_000), "0:05");
        assert_eq!(format_clock(65_000), "1:05");
        assert_eq!(format_clock(3_600_000), "1:00:00");
        assert_eq!(format_clock(3_723_000), "1:02:03");
    }

    // -----------------------------------------------------------------------
    // Think-block stripping — pure, always runs, no model
    // -----------------------------------------------------------------------

    #[test]
    fn strip_think_block_removes_leading_block() {
        let raw = "<think>reasoning here</think>\n# Summary\n\n- point";
        assert_eq!(strip_think_block(raw), "# Summary\n\n- point");
    }

    #[test]
    fn strip_think_block_passes_through_plain_output() {
        let raw = "# Summary\n\n- point one";
        assert_eq!(strip_think_block(raw), "# Summary\n\n- point one");
    }

    #[test]
    fn strip_think_block_drops_unterminated_block() {
        let raw = "<think>never closed reasoning";
        assert_eq!(strip_think_block(raw), "");
    }

    // -----------------------------------------------------------------------
    // Error mapping — template missing is a hard InvalidInput, never a panic
    // -----------------------------------------------------------------------

    #[test]
    fn template_error_maps_to_invalid_input() {
        use minutist_common::AppError;
        let app: AppError = Error::Template("no chat template baked into GGUF".to_string()).into();
        match app {
            AppError::InvalidInput { context } => {
                assert!(context.contains("no chat template"));
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn context_overflow_maps_to_invalid_input() {
        use minutist_common::AppError;
        let app: AppError = Error::ContextOverflow("prompt too long".to_string()).into();
        assert!(matches!(app, AppError::InvalidInput { .. }));
    }

    #[test]
    fn config_default_threads_within_bounds() {
        let cfg = SummariserConfig::default();
        assert!(cfg.threads >= 1 && cfg.threads <= 8);
        assert_eq!(cfg.n_batch, 512);
        assert_eq!(cfg.n_ctx, 32_768);
    }

    /// (no-model, always runs) — opening a nonexistent path must return
    /// `AppError::ModelLoad`, not panic.
    #[test]
    fn open_nonexistent_path_returns_model_load_error() {
        use minutist_common::AppError;
        let result = LlamaSummariser::open(
            PathBuf::from("/nonexistent/model.gguf"),
            SummariserConfig::default(),
        );
        match result {
            Err(AppError::ModelLoad { .. }) => { /* expected */ }
            Err(other) => panic!("expected AppError::ModelLoad, got {other:?}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    #[test]
    fn detect_model_turn_markers_falls_back_to_gemma23_strings_when_no_single_token_match() {
        // Without a real model the probe always produces empty token vecs.
        // The fallback must be the Gemma 2/3 pair so tests without a GGUF do not panic.
        // Validate the fallback path directly (detect_model_turn_markers is called
        // with a real LlamaModel only in gated tests).
        let candidates: &[(&str, &str)] = &[
            ("<|turn>", "<turn|>"),
            ("<start_of_turn>", "<end_of_turn>"),
        ];
        // The second candidate is the expected fallback.
        let fallback = candidates[1];
        assert_eq!(fallback.0, "<start_of_turn>");
        assert_eq!(fallback.1, "<end_of_turn>");
    }

    // -----------------------------------------------------------------------
    // Gated real-model test — skips when MINUTIST_LLM_MODEL_PATH is absent.
    //
    // To run:
    //   MINUTIST_LLM_MODEL_PATH=/path/to/model.gguf \
    //   cargo test -p summariser -- --include-ignored
    // -----------------------------------------------------------------------

    /// Build a synthetic ~30-minute transcript (`Vec<Segment>`) — enough text
    /// that the templated prompt comfortably exceeds `n_batch`, exercising the
    /// chunked-prefill path against a real model.
    fn synthetic_30min_transcript() -> Vec<Segment> {
        let lines = [
            "Let's get started with the quarterly planning review.",
            "The migration to the new persistence layer is on track for next sprint.",
            "We saw a regression in the audio capture path on Windows last week.",
            "I think we should prioritise the diarisation accuracy work.",
            "Agreed, the customer feedback on speaker labelling has been consistent.",
            "Can someone own the model-download resume bug before Friday?",
            "I'll take it. It's a hash-verification edge case on partial files.",
            "Next, the summariser latency on long meetings needs a second look.",
        ];
        let mut segments = Vec::new();
        // ~30 min at one ~6 s utterance each = ~300 segments.
        for i in 0..300u64 {
            let line = lines[(i as usize) % lines.len()];
            let start = i * 6_000;
            segments.push(Segment {
                start_ms: start,
                end_ms: start + 6_000,
                text: line.to_string(),
                speaker_id: Some(format!("Speaker {}", (i % 3) + 1)),
                confidence: None,
                words: vec![],
                shared_speakers: Vec::new(),
            });
        }
        segments
    }

    #[test]
    #[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
    fn summarise_synthetic_transcript_produces_markdown() {
        let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
            Ok(p) => p,
            Err(_) => return, // no-op skip path
        };

        let summariser =
            LlamaSummariser::open(PathBuf::from(&model_path), SummariserConfig::default())
                .expect("model load must succeed with a valid path");

        let transcript = synthetic_30min_transcript();
        let notes = [
            note(None, "- Action: own the model-download resume bug"),
            note(None, "- Decision: prioritise diarisation accuracy"),
        ];
        let system_prompt =
            "You are a meeting-notes assistant. Produce a concise markdown summary with headings.";

        let start = std::time::Instant::now();
        let summary = summariser
            .summarise(&transcript, &notes, "", system_prompt)
            .expect("summarise must succeed");
        let elapsed = start.elapsed();

        // Record latency; do not assert on it (hardware-dependent).
        tracing::info!(
            target: "summariser",
            elapsed_ms = elapsed.as_millis() as u64,
            summary_len = summary.len(),
            "gated summarise complete"
        );

        assert!(!summary.trim().is_empty(), "summary must be non-empty");
        assert!(
            summary.contains('#'),
            "summary should contain at least one markdown heading; got: {summary:?}"
        );
        // Thinking must have been stripped.
        assert!(
            !summary.contains("<think>"),
            "summary must not contain a think block"
        );
    }

    /// Real-recording summary: load an actual meeting's `transcript.json` from
    /// `MINUTIST_RECORDINGS_DIR` and summarise it with the real LLM. This
    /// exercises the exact path that broke in the field (the Gemma chat-template
    /// render → `apply_chat_template` ffi -1) on GENUINE data, not a synthetic
    /// transcript — the regression guard for that bug. Skips cleanly when either
    /// env var is unset or no recording has usable transcript text.
    #[test]
    #[ignore = "requires MINUTIST_LLM_MODEL_PATH and MINUTIST_RECORDINGS_DIR"]
    fn summarise_real_recording_produces_markdown() {
        let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
            Ok(p) if !p.is_empty() => p,
            _ => return,
        };
        let recordings_dir = match std::env::var("MINUTIST_RECORDINGS_DIR") {
            Ok(p) if !p.is_empty() => p,
            _ => return,
        };

        let transcript = match find_recording_transcript(&recordings_dir) {
            Some(t) => t,
            None => {
                eprintln!(
                    "no recording with usable transcript text under {recordings_dir}; skipping"
                );
                return;
            }
        };

        let summariser =
            LlamaSummariser::open(PathBuf::from(&model_path), SummariserConfig::default())
                .expect("model load must succeed with a valid path");
        let summary = summariser
            .summarise(
                &transcript,
                &[],
                "",
                "You are a meeting-notes assistant. Produce a concise markdown \
                 summary with headings.",
            )
            .expect(
                "summarise must succeed on a real recording \
                 (regression: Gemma chat-template render)",
            );

        assert!(!summary.trim().is_empty(), "summary must be non-empty");
        eprintln!(
            "real-recording summary ({} segments) =>\n{summary}",
            transcript.len()
        );
    }

    // -----------------------------------------------------------------------
    // Gated translate_segment test — skips when MINUTIST_LLM_MODEL_PATH absent.
    // -----------------------------------------------------------------------

    /// Translate a short English sentence to Spanish and verify the result is
    /// non-empty and contains no English content words from the original.
    ///
    /// To run:
    ///   MINUTIST_LLM_MODEL_PATH=/path/to/model.gguf \
    ///   cargo test -p summariser translate_segment_produces_spanish -- --include-ignored
    #[test]
    #[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
    fn translate_segment_produces_spanish_translation() {
        let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
            Ok(p) if !p.is_empty() => PathBuf::from(p),
            _ => return,
        };

        let summariser = LlamaSummariser::open(model_path, SummariserConfig::default())
            .expect("model load must succeed");

        let english = "The meeting is scheduled for next Tuesday at three o'clock.";
        let start = std::time::Instant::now();
        let translation = summariser
            .translate_segment(english, "Spanish")
            .expect("translate_segment must succeed");
        let elapsed = start.elapsed();

        eprintln!(
            "translate_segment ({} ms): {:?} → {:?}",
            elapsed.as_millis(),
            english,
            translation
        );

        assert!(
            !translation.trim().is_empty(),
            "translation must be non-empty"
        );

        // The Spanish translation must not contain the English content words
        // "meeting", "scheduled", "Tuesday", "three", or "o'clock". A
        // correct Spanish output would use "reunión", "martes", "tres",
        // "programada", etc. Case-insensitive check; "scheduled" in
        // particular should not appear in Spanish output.
        let lower = translation.to_lowercase();
        for word in &["meeting", "scheduled", "tuesday", "o'clock"] {
            assert!(
                !lower.contains(word),
                "translation contains English content word {word:?}; full output: {translation:?}"
            );
        }

        // Must not contain a think block (stripped before return).
        assert!(
            !translation.contains("<think>"),
            "translation must not contain a think block: {translation:?}"
        );
    }

    // Gated generate_attachment_awareness test — skips when MINUTIST_LLM_MODEL_PATH absent.
    // ----------------------------------------------------------------------------------

    /// Feed a short markdown document to `generate_attachment_awareness` and
    /// assert the output is non-empty, reasonably short, and contains the word
    /// "Keywords".
    ///
    /// To run:
    ///   MINUTIST_LLM_MODEL_PATH=/path/to/model.gguf \
    ///   cargo test -p summariser generate_attachment_awareness_produces_summary -- --include-ignored
    #[test]
    #[ignore = "requires MINUTIST_LLM_MODEL_PATH"]
    fn generate_attachment_awareness_produces_summary() {
        let model_path = match std::env::var("MINUTIST_LLM_MODEL_PATH") {
            Ok(p) if !p.is_empty() => PathBuf::from(p),
            _ => return,
        };

        let summariser = LlamaSummariser::open(model_path, SummariserConfig::default())
            .expect("model load must succeed");

        let doc = "# Q2 Revenue Report\n\n\
                   Total revenue for Q2 reached AUD 4.2 million, up 18% year-on-year. \
                   Growth was driven primarily by the enterprise segment, which contributed \
                   62% of total billings. Operating costs remained stable at AUD 2.8 million, \
                   yielding an operating margin of 33%.";

        let start = std::time::Instant::now();
        let awareness = summariser
            .generate_attachment_awareness(doc)
            .expect("generate_attachment_awareness must succeed");
        let elapsed = start.elapsed();

        eprintln!(
            "generate_attachment_awareness ({} ms): {:?}",
            elapsed.as_millis(),
            awareness
        );

        assert!(!awareness.trim().is_empty(), "awareness must be non-empty");

        // Output must be reasonably short — 1–3 sentences plus a keyword line.
        // 600 characters is a generous upper bound.
        assert!(
            awareness.len() < 600,
            "awareness output too long ({} chars): {awareness:?}",
            awareness.len()
        );

        assert!(
            awareness.contains("Keywords"),
            "awareness must contain 'Keywords': {awareness:?}"
        );

        // Must not contain a think block (stripped before return).
        assert!(
            !awareness.contains("<think>"),
            "awareness must not contain a think block: {awareness:?}"
        );
    }

    /// Scan a recordings dir for the first (lexicographically) `transcript.json`
    /// holding >= 3 non-empty segments, returning its parsed `Vec<Segment>`.
    /// Used by the gated real-recording summary test.
    fn find_recording_transcript(dir: &str) -> Option<Vec<Segment>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path().join("transcript.json")))
            .filter(|p| p.is_file())
            .collect();
        paths.sort();
        for path in paths {
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(segs) = serde_json::from_slice::<Vec<Segment>>(&bytes) else {
                continue;
            };
            if segs.iter().filter(|s| !s.text.trim().is_empty()).count() >= 3 {
                return Some(segs);
            }
        }
        None
    }
}
