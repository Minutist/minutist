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
//!    with a `system` message (the user-configured prompt) and a `user`
//!    message (the rendered transcript + the notes markdown). Missing/unusable
//!    template → [`Error::Template`] (→ `AppError::InvalidInput`), never a
//!    hand-built scaffold.
//! 2. **Thinking is disabled** — we never inject a think token. If the model
//!    nonetheless emits a `<think>…</think>` block, it is stripped before the
//!    summary is returned.
//! 3. **Tokenisation** uses [`AddBos::Never`] (the template embeds BOS).
//! 4. **Prefill is chunked by `n_batch`** ([`plan_prefill`]): the prompt is
//!    decoded in `n_batch`-sized [`LlamaBatch`] chunks, with `logits = true`
//!    set only on the final token of the final chunk, so a long transcript
//!    never trips `GGML_ASSERT(n_tokens_all <= n_batch)`.
//! 5. **Generation** is greedy, stops on `model.is_eog_token(token)`, and is
//!    capped at `config.max_tokens`. Detokenisation is incremental via an
//!    `encoding_rs` UTF-8 decoder (mirrors `asr-runtime`).

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use meeting_app_common::{AppResult, NoteBlock, Segment, Summariser};

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
/// this delegation fixes. See `meeting_app_common::llama_backend`.
fn get_or_init_backend() -> Result<&'static LlamaBackend, Error> {
    meeting_app_common::llama_backend::shared_llama_backend().map_err(|e| Error::ModelLoad {
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
        let model =
            LlamaModel::load_from_file(backend, &model_path, &model_params).map_err(|e| {
                Error::ModelLoad {
                    path: model_path.display().to_string(),
                    context: e.to_string(),
                }
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
        system_prompt: &str,
        mut on_progress: impl FnMut(SummariseProgress),
    ) -> AppResult<String> {
        let prompt = self.build_prompt(transcript, notes, system_prompt)?;
        let raw = self.generate_with_progress(&prompt, &mut on_progress)?;
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
}

impl Summariser for LlamaSummariser {
    fn summarise(
        &self,
        transcript: &[Segment],
        notes: &[NoteBlock],
        system_prompt: &str,
    ) -> AppResult<String> {
        let prompt = self.build_prompt(transcript, notes, system_prompt)?;
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
    /// [`gemma_turn_prompt`] hand-built format — the format our shipped
    /// summariser LLM uses. Other models keep using their baked template.
    fn build_prompt(
        &self,
        transcript: &[Segment],
        notes: &[NoteBlock],
        system_prompt: &str,
    ) -> Result<String, Error> {
        let template = self
            .model
            .chat_template(None::<&str>)
            .map_err(|e| Error::Template(e.to_string()))?;

        let user_content = render_user_content(transcript, notes);
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
                    "apply_chat_template failed ({e}); using the Gemma turn-format fallback"
                );
                Ok(gemma_turn_prompt(&combined))
            }
        }
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
        let backend = get_or_init_backend()?;

        let n_ctx = NonZeroU32::new(self.config.n_ctx).ok_or_else(|| {
            Error::ContextOverflow("n_ctx must be non-zero".to_string())
        })?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(self.config.n_batch)
            .with_n_threads(self.config.threads)
            .with_n_threads_batch(self.config.threads);

        let mut llama_ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Inference(format!("LlamaContext init: {e}")))?;

        // AddBos::Never — the chat template already emits the BOS itself.
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Never)
            .map_err(|e| Error::Inference(format!("tokenize: {e}")))?;

        if tokens.is_empty() {
            return Err(Error::Inference("templated prompt tokenised to zero tokens".to_string()));
        }

        // The prompt AND the tokens it will generate must both fit the context
        // window: generation grows the KV cache by one slot per token, so a
        // prompt that fits on its own can still overflow mid-generation. Reserve
        // `max_tokens` of headroom up front rather than aborting partway and
        // losing the summary.
        check_context_budget(tokens.len(), self.config.max_tokens, self.config.n_ctx)?;

        // --- Chunked prefill ---
        let plan = plan_prefill(tokens.len(), self.config.n_batch);
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);
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

            // Report prefill progress AFTER this chunk decodes (#69): `done` is
            // the count of prompt tokens whose KV cache is now populated.
            let done = (chunk.start + chunk.len).min(total_prompt);
            on_progress(SummariseProgress::Prefill {
                done,
                total: total_prompt,
            });
        }

        // --- Greedy generation ---
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut decoder = UTF_8.new_decoder();
        let mut text = String::new();
        let mut n_past = tokens.len() as i32;

        for i in 0..self.config.max_tokens {
            let token = sampler.sample(&llama_ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                // EOG: jump the bar to 100 % so a short summary still completes
                // the bar rather than leaving it stuck mid-way.
                on_progress(SummariseProgress::Generate {
                    done: self.config.max_tokens,
                    max: self.config.max_tokens,
                });
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| Error::Inference(format!("token_to_piece: {e}")))?;
            text.push_str(&piece);

            // Report progress AFTER appending this token (`i + 1` generated).
            on_progress(SummariseProgress::Generate {
                done: i + 1,
                max: self.config.max_tokens,
            });

            batch.clear();
            batch
                .add(token, n_past, &[0], true)
                .map_err(|e| Error::Inference(format!("batch.add (gen): {e}")))?;
            n_past += 1;

            llama_ctx
                .decode(&mut batch)
                .map_err(|e| Error::Inference(format!("decode (gen): {e}")))?;
        }

        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// Prompt rendering
// ---------------------------------------------------------------------------

/// Hand-built Gemma single-user-turn prompt, used as the fallback when
/// `apply_chat_template` cannot render the GGUF's baked template (Gemma 4
/// postdates the bundled llama.cpp). `<bos>` is included explicitly because
/// `generate()` tokenises with `AddBos::Never`; `str_to_token` parses special
/// tokens, so `<bos>` / `<start_of_turn>` / `<end_of_turn>` map to their ids.
/// The trailing open `model` turn is where generation continues.
fn gemma_turn_prompt(content: &str) -> String {
    format!("<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n")
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
fn render_user_content(transcript: &[Segment], notes: &[NoteBlock]) -> String {
    let any_anchored = notes.iter().any(|n| n.at_ms.is_some());
    let mut out = String::new();

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
        use meeting_app_common::AppError;
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
        let body = render_user_content(&transcript, &[note(None, "- action item one")]);

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
        let body = render_user_content(&transcript, &[]);
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
        let body = render_user_content(&transcript, &notes);

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
        let body = render_user_content(&transcript, &notes);
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
        let body = render_user_content(&transcript, &notes);
        let woven = body.find("anchored thought").expect("anchored");
        let trailing_header = body.find("# Notes (no timestamp)").expect("trailing header");
        let trailing = body.find("pre-meeting agenda").expect("unanchored");
        assert!(woven < trailing_header && trailing_header < trailing);
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
        use meeting_app_common::AppError;
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
        use meeting_app_common::AppError;
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
        use meeting_app_common::AppError;
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
    fn gemma_turn_prompt_wraps_content_with_bos_and_open_model_turn() {
        let p = gemma_turn_prompt("INSTRUCTIONS\n\nbody");
        assert_eq!(
            p,
            "<bos><start_of_turn>user\nINSTRUCTIONS\n\nbody<end_of_turn>\n<start_of_turn>model\n"
        );
        // BOS is explicit (generate() uses AddBos::Never) and the assistant turn
        // is left open for generation (ends after the `model` turn marker).
        assert!(p.starts_with("<bos>"));
        assert!(p.ends_with("<start_of_turn>model\n"));
    }

    // -----------------------------------------------------------------------
    // Gated real-model test — skips when MEETING_APP_LLM_MODEL_PATH is absent.
    //
    // To run:
    //   MEETING_APP_LLM_MODEL_PATH=/path/to/model.gguf \
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
            });
        }
        segments
    }

    #[test]
    #[ignore = "requires MEETING_APP_LLM_MODEL_PATH"]
    fn summarise_synthetic_transcript_produces_markdown() {
        let model_path = match std::env::var("MEETING_APP_LLM_MODEL_PATH") {
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
            .summarise(&transcript, &notes, system_prompt)
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
    /// `MEETING_APP_RECORDINGS_DIR` and summarise it with the real LLM. This
    /// exercises the exact path that broke in the field (the Gemma chat-template
    /// render → `apply_chat_template` ffi -1) on GENUINE data, not a synthetic
    /// transcript — the regression guard for that bug. Skips cleanly when either
    /// env var is unset or no recording has usable transcript text.
    #[test]
    #[ignore = "requires MEETING_APP_LLM_MODEL_PATH and MEETING_APP_RECORDINGS_DIR"]
    fn summarise_real_recording_produces_markdown() {
        let model_path = match std::env::var("MEETING_APP_LLM_MODEL_PATH") {
            Ok(p) if !p.is_empty() => p,
            _ => return,
        };
        let recordings_dir = match std::env::var("MEETING_APP_RECORDINGS_DIR") {
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
