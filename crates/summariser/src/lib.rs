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
use std::sync::OnceLock;

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use meeting_app_common::{AppResult, Segment, Summariser};

mod error;
pub use error::Error;

#[cfg(feature = "external-ollama")]
mod ollama;
#[cfg(feature = "external-ollama")]
pub use ollama::OllamaSummariser;

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
        }
    }
}

// ---------------------------------------------------------------------------
// LlamaBackend singleton
// ---------------------------------------------------------------------------

/// Process-wide singleton. `LlamaBackend::init` may only be called once per
/// process; subsequent calls return an error. `asr-runtime` owns its own copy
/// of this pattern — both crates share the single underlying backend because
/// only one of them can win the `init()` race and the loser falls back to the
/// populated `OnceLock`.
static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn get_or_init_backend() -> Result<&'static LlamaBackend, Error> {
    if let Some(b) = LLAMA_BACKEND.get() {
        return Ok(b);
    }
    // We may lose a race here between two concurrent callers both seeing
    // `None`. `OnceLock::get_or_try_init` would handle this atomically, but it
    // is nightly-only. Instead: the first caller wins, the second caller's
    // `init()` call fails, and we fall back to a second `get()` which by that
    // point is populated.
    match LlamaBackend::init() {
        Ok(b) => {
            let _ = LLAMA_BACKEND.set(b);
        }
        Err(e) => {
            if LLAMA_BACKEND.get().is_none() {
                return Err(Error::ModelLoad {
                    path: "llama-backend".to_string(),
                    context: e.to_string(),
                });
            }
        }
    }
    LLAMA_BACKEND.get().ok_or_else(|| Error::ModelLoad {
        path: "llama-backend".to_string(),
        context: "OnceLock unexpectedly empty".to_string(),
    })
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

        // Text-only model: no GPU layers offloaded by default (CPU-first, as
        // with `asr-runtime`).
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
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
}

impl Summariser for LlamaSummariser {
    fn summarise(
        &self,
        transcript: &[Segment],
        notes_markdown: &str,
        system_prompt: &str,
    ) -> AppResult<String> {
        let prompt = self.build_prompt(transcript, notes_markdown, system_prompt)?;
        let raw = self.generate(&prompt)?;
        Ok(strip_think_block(&raw))
    }
}

impl LlamaSummariser {
    /// Build the prompt from the GGUF's baked-in chat template.
    ///
    /// Two messages: `system` = the user-configured prompt, `user` = the
    /// rendered transcript + a blank line + the notes markdown. `add_ass=true`
    /// leaves the assistant turn open for generation. Model-agnostic: a missing
    /// or unusable template is a hard [`Error::Template`].
    fn build_prompt(
        &self,
        transcript: &[Segment],
        notes_markdown: &str,
        system_prompt: &str,
    ) -> Result<String, Error> {
        let template = self
            .model
            .chat_template(None::<&str>)
            .map_err(|e| Error::Template(e.to_string()))?;

        let user_content = render_user_content(transcript, notes_markdown);

        let system_msg = LlamaChatMessage::new("system".to_string(), system_prompt.to_string())
            .map_err(|e| Error::Template(format!("system message: {e}")))?;
        let user_msg = LlamaChatMessage::new("user".to_string(), user_content)
            .map_err(|e| Error::Template(format!("user message: {e}")))?;

        self.model
            .apply_chat_template(&template, &[system_msg, user_msg], true)
            .map_err(|e| Error::Template(format!("apply_chat_template: {e}")))
    }

    /// Tokenise the prompt, chunked-prefill it, then greedily generate.
    fn generate(&self, prompt: &str) -> Result<String, Error> {
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

        // The prompt plus at least one generated token must fit the context.
        if tokens.len() >= self.config.n_ctx as usize {
            return Err(Error::ContextOverflow(format!(
                "prompt is {} tokens but context window is {}",
                tokens.len(),
                self.config.n_ctx
            )));
        }

        // --- Chunked prefill ---
        let plan = plan_prefill(tokens.len(), self.config.n_batch);
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);

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
        }

        // --- Greedy generation ---
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut decoder = UTF_8.new_decoder();
        let mut text = String::new();
        let mut n_past = tokens.len() as i32;

        for _ in 0..self.config.max_tokens {
            let token = sampler.sample(&llama_ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| Error::Inference(format!("token_to_piece: {e}")))?;
            text.push_str(&piece);

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

/// Render the transcript + notes into the single `user` message body.
///
/// Speaker-attributed lines when `speaker_id` is present, plain text
/// otherwise. The notes markdown follows after a blank line under a heading so
/// the model can distinguish "what was said" from "what the user wrote".
fn render_user_content(transcript: &[Segment], notes_markdown: &str) -> String {
    let mut out = String::new();
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
    if notes_markdown.trim().is_empty() {
        out.push_str("(no notes taken)\n");
    } else {
        out.push_str(notes_markdown.trim());
        out.push('\n');
    }

    out
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
    // Prompt builder — pure, always runs, no model
    // -----------------------------------------------------------------------

    fn seg(text: &str, speaker: Option<&str>) -> Segment {
        Segment {
            start_ms: 0,
            end_ms: 1_000,
            text: text.to_string(),
            speaker_id: speaker.map(|s| s.to_string()),
            confidence: None,
            words: vec![],
        }
    }

    #[test]
    fn render_user_content_includes_transcript_and_notes() {
        let transcript = vec![
            seg("hello there", Some("Speaker 1")),
            seg("general kenobi", Some("Speaker 2")),
        ];
        let body = render_user_content(&transcript, "- action item one");

        assert!(body.contains("# Transcript"));
        assert!(body.contains("Speaker 1: hello there"));
        assert!(body.contains("Speaker 2: general kenobi"));
        assert!(body.contains("# Notes"));
        assert!(body.contains("- action item one"));
    }

    #[test]
    fn render_user_content_without_speakers_or_notes() {
        let transcript = vec![seg("just one line", None)];
        let body = render_user_content(&transcript, "   ");
        assert!(body.contains("just one line"));
        assert!(!body.contains(": just one line"), "no speaker prefix expected");
        assert!(body.contains("(no notes taken)"));
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
        let notes = "- Action: own the model-download resume bug\n- Decision: prioritise diarisation accuracy";
        let system_prompt =
            "You are a meeting-notes assistant. Produce a concise markdown summary with headings.";

        let start = std::time::Instant::now();
        let summary = summariser
            .summarise(&transcript, notes, system_prompt)
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
}
