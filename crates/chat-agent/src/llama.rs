//! The real [`TurnBackend`] over `llama-cpp-2`'s native OpenAI-compatible tool
//! calling (§0a / §6).
//!
//! For ONE turn:
//! 1. Render the prompt from the OpenAI-format `messages_json` + `tools_json`
//!    using the GGUF's own tool template
//!    ([`LlamaModel::apply_chat_template_oaicompat`]). This returns the prompt,
//!    the chat format + (optional) PEG parser the streaming oaicompat parser
//!    needs, and — when tools are offered — a lazy GBNF grammar over the tool
//!    schemas.
//! 2. Allocate a FRESH [`LlamaContext`] (clean KV cache, like
//!    `summariser::generate`). Tokenise `AddBos::Never` (the template embeds
//!    BOS). Chunked-prefill by `n_batch` (reuse `summariser::plan_prefill`).
//! 3. Build the sampler chain (§6.4): greedy when `temperature == 0.0`, else
//!    `penalties → top_k → top_p → min_p → temp → dist(seed)`; the lazy GBNF
//!    grammar is prepended as the reliability backstop when
//!    `cfg.grammar_backstop` is set AND the template produced one.
//! 4. Decode token-by-token; feed each detokenised piece to the streaming
//!    oaicompat parser ([`ChatParseStateOaicompat`]); stream ONLY `content`
//!    deltas through the token callback (never tool-call JSON).
//! 5. Do a final authoritative non-partial parse to extract the assistant
//!    `content` + any `tool_calls` into a [`RawTurn`].
//!
//! The model is borrowed (`&LlamaModel`) from the held `summariser` substrate
//! (D5) — no second model load, no `model-registry` edge.

use std::num::NonZeroU32;

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::model::ChatTemplateResult;
use llama_cpp_2::sampling::LlamaSampler;

use summariser::plan_prefill;

use crate::backend::{RawTurn, TurnBackend};
use crate::error::Error;
use crate::types::{SamplerConfig, ToolCall};

/// Runtime knobs for the real turn backend. Mirrors the relevant
/// `summariser::SummariserConfig` fields so the chat context is sized like the
/// summary context.
#[derive(Debug, Clone)]
pub struct LlamaTurnConfig {
    /// Context window to allocate, in tokens.
    pub n_ctx: u32,
    /// Per-decode batch size — the chunked-prefill chunk size.
    pub n_batch: u32,
    /// CPU threads for llama.cpp inference.
    pub threads: i32,
    /// Number of model layers to offload to the GPU (runtime GPU toggle; `0`
    /// forces CPU even in a GPU build, like `summariser`).
    pub n_gpu_layers: u32,
}

impl Default for LlamaTurnConfig {
    fn default() -> Self {
        let threads = ((num_cpus_get() / 2) as i32).clamp(1, 8);
        Self {
            n_ctx: 32_768,
            n_batch: 512,
            threads,
            n_gpu_layers: summariser::gpu_layers(),
        }
    }
}

/// `num_cpus` is not a direct dep here; the value only seeds a default the
/// driver overrides from settings, so a conservative fixed fallback is fine.
fn num_cpus_get() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// The real turn backend. Borrows the loaded model from the substrate.
pub struct LlamaTurnBackend<'m> {
    model: &'m LlamaModel,
    config: LlamaTurnConfig,
}

// The borrowed model is `Send + Sync` (`unsafe impl` in `llama-cpp-2`); the
// fresh `LlamaContext` is built per call and never stored, so the backend is
// `Send + Sync` for the engine's `Arc<dyn ChatEngine>`.
impl<'m> LlamaTurnBackend<'m> {
    /// Build a turn backend over the substrate model (lent by `ipc-bridge` from
    /// the held `Arc<LlamaSummariser>` via `LlamaSummariser::model()`).
    pub fn new(model: &'m LlamaModel, config: LlamaTurnConfig) -> Self {
        Self { model, config }
    }

    fn backend() -> Result<&'static LlamaBackend, Error> {
        meeting_app_common::llama_backend::shared_llama_backend()
            .map_err(|e| Error::Inference(format!("llama backend init: {e}")))
    }

    /// Render the prompt + tool grammar/parser via the oaicompat template.
    fn render(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
    ) -> Result<ChatTemplateResult, Error> {
        let template = self
            .model
            .chat_template(None::<&str>)
            .map_err(|e| Error::Template(format!("read GGUF chat template: {e}")))?;

        let params = OpenAIChatTemplateParams {
            messages_json,
            tools_json,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            // Ask the template machinery to wire up tool-call parsing when tools
            // are offered, so the streaming/final parser separates content from
            // tool calls.
            parse_tool_calls: tools_json.is_some(),
        };

        self.model
            .apply_chat_template_oaicompat(&template, &params)
            .map_err(|e| Error::Template(format!("oaicompat render: {e}")))
    }

    /// Build the sampler chain for one turn (§6.4).
    fn sampler(
        &self,
        cfg: &SamplerConfig,
        rendered: &ChatTemplateResult,
    ) -> Result<LlamaSampler, Error> {
        // Optional lazy-GBNF backstop: the template-emitted grammar (lazy, with
        // its own triggers) when the template produced one, else a grammar
        // compiled from the rendered tools (already part of the prompt) — only
        // when the driver armed the flag.
        let grammar = if cfg.grammar_backstop {
            self.lazy_grammar(rendered)?
        } else {
            None
        };

        if cfg.is_greedy() {
            // Deterministic test/precise mode: (grammar →) greedy.
            let mut chain = Vec::new();
            if let Some(g) = grammar {
                chain.push(g);
            }
            chain.push(LlamaSampler::greedy());
            return Ok(LlamaSampler::chain_simple(chain));
        }

        // Default chat chain: (grammar →) penalties → top_k → top_p → min_p →
        // temp → dist(seed).
        let mut chain = Vec::new();
        if let Some(g) = grammar {
            chain.push(g);
        }
        chain.push(LlamaSampler::penalties(64, 1.1, 0.0, 0.0));
        chain.push(LlamaSampler::top_k(64));
        chain.push(LlamaSampler::top_p(cfg.top_p, 1));
        chain.push(LlamaSampler::min_p(0.05, 1));
        chain.push(LlamaSampler::temp(cfg.temperature));
        chain.push(LlamaSampler::dist(cfg.seed));
        Ok(LlamaSampler::chain_simple(chain))
    }

    /// The lazy GBNF grammar backstop. Prefers the template-emitted lazy grammar
    /// (which carries its own `<tool_call>`-style triggers); falls back to
    /// compiling the rendered tools' grammar lazily on a generic trigger when
    /// the template emitted none.
    fn lazy_grammar(
        &self,
        rendered: &ChatTemplateResult,
    ) -> Result<Option<LlamaSampler>, Error> {
        // Template-emitted grammar: snap it lazily after the template's triggers
        // (so free-text turns are unconstrained, tool-call turns are forced to
        // valid JSON). Word triggers only — token triggers carry ids we honour.
        if let Some(grammar) = rendered.grammar.as_deref() {
            if grammar.trim().is_empty() {
                return Ok(None);
            }
            let trigger_words: Vec<String> = rendered
                .grammar_triggers
                .iter()
                .map(|t| t.value.clone())
                .collect();
            let trigger_tokens: Vec<_> = rendered
                .grammar_triggers
                .iter()
                .filter_map(|t| t.token)
                .collect();

            let sampler = if rendered.grammar_lazy
                && (!trigger_words.is_empty() || !trigger_tokens.is_empty())
            {
                LlamaSampler::grammar_lazy(
                    self.model,
                    grammar,
                    "root",
                    trigger_words,
                    &trigger_tokens,
                )
            } else {
                LlamaSampler::grammar(self.model, grammar, "root")
            }
            .map_err(|e| Error::Grammar(format!("template grammar: {e}")))?;
            return Ok(Some(sampler));
        }
        Ok(None)
    }

    /// Compile a GBNF grammar from a tool's JSON schema (exposed for the
    /// CI-gating "every registry schema compiles" unit test in the engine).
    ///
    /// The production turn uses the template-emitted lazy grammar
    /// ([`Self::lazy_grammar`]), not this helper, so it is test-only — the
    /// CI-gate test compiles every v1 registry schema through it.
    #[cfg(test)]
    pub(crate) fn schema_grammar(schema_json: &str) -> Result<String, Error> {
        llama_cpp_2::json_schema_to_grammar(schema_json)
            .map_err(|e| Error::Grammar(format!("schema → grammar: {e}")))
    }
}

impl TurnBackend for LlamaTurnBackend<'_> {
    fn run(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        cfg: &SamplerConfig,
        token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, Error> {
        let backend = Self::backend()?;
        let rendered = self.render(messages_json, tools_json)?;

        let n_ctx = NonZeroU32::new(self.config.n_ctx)
            .ok_or_else(|| Error::Inference("n_ctx must be non-zero".to_string()))?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(self.config.n_batch)
            .with_n_threads(self.config.threads)
            .with_n_threads_batch(self.config.threads);
        let mut llama_ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| Error::Inference(format!("LlamaContext init: {e}")))?;

        let tokens = self
            .model
            .str_to_token(&rendered.prompt, AddBos::Never)
            .map_err(|e| Error::Inference(format!("tokenize: {e}")))?;
        if tokens.is_empty() {
            return Err(Error::Template(
                "rendered prompt tokenised to zero tokens".to_string(),
            ));
        }
        // The prompt + the generation it reserves must both fit n_ctx; the
        // driver's sliding window already trimmed, but a single over-budget turn
        // is caught here as the hard floor (§6.2) rather than overflowing mid
        // generation.
        if tokens.len().saturating_add(cfg.max_tokens) > self.config.n_ctx as usize {
            return Err(Error::ContextOverflow(format!(
                "prompt is {} tokens and generation reserves {} more but the context window is {}",
                tokens.len(),
                cfg.max_tokens,
                self.config.n_ctx
            )));
        }

        // --- Chunked prefill (reuse the summariser's pure planner) ---
        let plan = plan_prefill(tokens.len(), self.config.n_batch);
        let mut batch = LlamaBatch::new(self.config.n_batch as usize, 1);
        for chunk in &plan.chunks {
            batch.clear();
            for offset in 0..chunk.len {
                let global = chunk.start + offset;
                let logits = chunk.logits_at_last && offset == chunk.len - 1;
                batch
                    .add(tokens[global], global as i32, &[0], logits)
                    .map_err(|e| Error::Inference(format!("batch.add (prefill): {e}")))?;
            }
            llama_ctx
                .decode(&mut batch)
                .map_err(|e| Error::Inference(format!("decode (prefill): {e}")))?;
        }

        // --- Generation with streaming oaicompat parse ---
        let mut sampler = self.sampler(cfg, &rendered)?;
        let mut parser = rendered
            .streaming_state_oaicompat()
            .map_err(|e| Error::Template(format!("init oaicompat parser: {e}")))?;
        let mut decoder = UTF_8.new_decoder();
        let mut raw_text = String::new();
        let mut n_past = tokens.len() as i32;

        for _ in 0..cfg.max_tokens {
            let token = sampler.sample(&llama_ctx, -1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| Error::Inference(format!("token_to_piece: {e}")))?;
            raw_text.push_str(&piece);

            // Stream only user-visible content deltas; the parser separates
            // tool-call deltas, which we never surface as text.
            if let Ok(deltas) = parser.update(&piece, true) {
                for delta in deltas {
                    if let Some(content) = content_delta(&delta) {
                        if !content.is_empty() {
                            token_cb(&content);
                        }
                    }
                }
            }

            batch.clear();
            batch
                .add(token, n_past, &[0], true)
                .map_err(|e| Error::Inference(format!("batch.add (gen): {e}")))?;
            n_past += 1;
            llama_ctx
                .decode(&mut batch)
                .map_err(|e| Error::Inference(format!("decode (gen): {e}")))?;
        }

        // --- Authoritative final parse ---
        let final_json = rendered
            .parse_response_oaicompat(&raw_text, false)
            .map_err(|e| Error::MalformedOutput(format!("final oaicompat parse: {e}")))?;
        Ok(parse_final_message(&final_json))
    }
}

/// Extract the `content` string from an OpenAI streaming-delta JSON, if any.
/// The delta shape is `{"choices":[{"delta":{"content":"…"}}]}` (or a flat
/// `{"content":"…"}` depending on the diff form); tolerate both.
fn content_delta(delta_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(delta_json).ok()?;
    if let Some(c) = v
        .pointer("/choices/0/delta/content")
        .and_then(|c| c.as_str())
    {
        return Some(c.to_string());
    }
    v.get("content")
        .and_then(|c| c.as_str())
        .map(str::to_string)
}

/// Map the authoritative final OpenAI message JSON into a [`RawTurn`].
///
/// Shape (the oaicompat final parse): a message object carrying `content`
/// and/or `tool_calls: [{ id?, function: { name, arguments } }]`. Tolerate the
/// `{choices:[{message:…}]}` wrapper too.
fn parse_final_message(final_json: &str) -> RawTurn {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(final_json) else {
        return RawTurn::default();
    };
    let msg = v
        .pointer("/choices/0/message")
        .filter(|m| m.is_object())
        .unwrap_or(&v);

    let text = msg
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string();

    let mut tool_calls = Vec::new();
    if let Some(calls) = msg.get("tool_calls").and_then(|c| c.as_array()) {
        for (i, call) in calls.iter().enumerate() {
            let func = call.get("function").unwrap_or(call);
            let Some(name) = func.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            // `arguments` is a JSON string in OpenAI; tolerate an object too.
            let arguments_json = match func.get("arguments") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => "{}".to_string(),
            };
            let id = call
                .get("id")
                .and_then(|i| i.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("call_{i}"));
            tool_calls.push(ToolCall {
                id,
                name: name.to_string(),
                arguments_json,
            });
        }
    }
    RawTurn { text, tool_calls }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_final_message_extracts_final_text() {
        let j = json!({ "role": "assistant", "content": "the answer" }).to_string();
        let turn = parse_final_message(&j);
        assert_eq!(turn.text, "the answer");
        assert!(turn.tool_calls.is_empty());
    }

    #[test]
    fn parse_final_message_extracts_tool_calls() {
        let j = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [
                { "id": "abc", "function": { "name": "get_transcript", "arguments": "{\"meeting_id\":\"x\"}" } }
            ]
        })
        .to_string();
        let turn = parse_final_message(&j);
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].id, "abc");
        assert_eq!(turn.tool_calls[0].name, "get_transcript");
        assert_eq!(turn.tool_calls[0].arguments_json, "{\"meeting_id\":\"x\"}");
    }

    #[test]
    fn parse_final_message_tolerates_choices_wrapper_and_synthesises_id() {
        let j = json!({
            "choices": [{ "message": {
                "tool_calls": [
                    { "function": { "name": "list_meetings", "arguments": {} } }
                ]
            }}]
        })
        .to_string();
        let turn = parse_final_message(&j);
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "list_meetings");
        assert_eq!(turn.tool_calls[0].id, "call_0", "missing id is synthesised");
        // An object `arguments` is stringified.
        assert_eq!(turn.tool_calls[0].arguments_json, "{}");
    }

    #[test]
    fn parse_final_message_garbage_is_empty_turn() {
        assert_eq!(parse_final_message("not json"), RawTurn::default());
    }

    #[test]
    fn content_delta_reads_both_delta_shapes() {
        assert_eq!(
            content_delta(&json!({ "choices": [{ "delta": { "content": "hi" } }] }).to_string()),
            Some("hi".to_string())
        );
        assert_eq!(
            content_delta(&json!({ "content": "yo" }).to_string()),
            Some("yo".to_string())
        );
        assert_eq!(content_delta(&json!({ "role": "assistant" }).to_string()), None);
    }
}
