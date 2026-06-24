//! `chat-agent` — the stateless, OpenAI-compatible, tool-calling chat TURN
//! engine (Phase 9).
//!
//! One assistant turn per call over the bundled local LLM, reusing the
//! `summariser` substrate (the loaded `LlamaModel`, D5) and the `agent-tools`
//! tool descriptors. The crate sits ABOVE both `summariser` and `agent-tools`;
//! folding the loop into `summariser` would force a backwards `summariser →
//! agent-tools` edge.
//!
//! # The boundary (binding — §1.2 / §1.3)
//!
//! The engine is **stateless per call** and **does not dispatch tools**. The
//! DRIVER (`ipc-bridge`, a later phase) owns the conversation history, the
//! sliding window, the turn loop, the max-iteration cap, and tool dispatch via
//! `agent_tools::ToolRegistry::dispatch`. One [`ChatEngine::run_turn`] call:
//!
//! 1. renders the prompt from the history + the offered tool descriptors (the
//!    GGUF's own OpenAI tool template),
//! 2. generates with streaming (the engine calls a `token_cb` per detokenised
//!    user-visible piece; the driver turns those into `ChatToken` `AppEvent`s
//!    — the engine itself emits NO events and holds no broadcast handle),
//! 3. parses the output and returns a [`TurnOutcome`]: either a final assistant
//!    text OR a list of tool calls for the driver to execute.
//!
//! The driver then runs the calls, appends a [`ChatMessage::tool_result`] per
//! call, and invokes the engine again — until the model answers without a tool
//! call, the iteration cap is hit, or the context fills. That loop and the cap
//! live in the driver, NOT here.
//!
//! # The substrate seam (D5)
//!
//! The real turn needs the loaded `LlamaModel`. `summariser` exposes it via
//! `LlamaSummariser::model()`. `ipc-bridge` holds the concrete
//! `Arc<LlamaSummariser>`, lends `&LlamaModel` to [`LlamaTurnBackend`], and
//! coerces the same handle to `Arc<dyn Summariser>` for the `agent-tools`
//! `ToolContext`. The model is `Send + Sync`; the chat `LlamaContext` is built
//! fresh per turn (clean KV cache). No GGUF is reloaded per turn.
//!
//! # Testability (the [`TurnBackend`] seam)
//!
//! The LLM call is behind [`TurnBackend`]: the real [`LlamaTurnBackend`] uses
//! the oaicompat APIs; a test stub returns canned text/tool-calls. The engine's
//! turn logic (prompt assembly, outcome parsing, tool-call extraction, error
//! mapping) is unit-tested with the stub (no model). `LlamaTurnBackend` is
//! covered by a gated test (skip-on-unset `MINUTIST_LLM_MODEL_PATH`).

mod backend;
mod error;
pub mod live;
mod llama;
mod types;
mod window;

pub use backend::{messages_json, tools_json, RawTurn, TurnBackend};
pub use error::Error;
pub use live::{LiveSession, LiveSessionBackend, LlamaLiveBackend, LlamaLiveConfig};
pub use llama::{LlamaTurnBackend, LlamaTurnConfig};
pub use types::{CancelFlag, ChatMessage, Role, SamplerConfig, ToolCall, TurnOutcome};
pub use window::{fits_budget, trim_to_budget, TrimOutcome, HARD_FLOOR_REJECT};

use agent_tools::ToolDescriptor;
use minutist_common::AppResult;

/// The stateless chat turn engine.
///
/// Generates ONE assistant turn over the full conversation `history`. Does NOT
/// own history, run tools, or emit events. See the crate docs for the
/// engine/driver boundary.
pub trait ChatEngine: Send + Sync {
    /// Run one assistant turn.
    ///
    /// - `history` — the windowed conversation (the driver applied
    ///   [`trim_to_budget`] already). Borrowed, never mutated.
    /// - `tool_descriptors` — from `agent_tools::ToolRegistry::descriptors()`.
    ///   An empty slice forces a tool-less, final-answer turn (the driver's
    ///   max-iteration escape).
    /// - `cfg` — per-turn sampling knobs.
    /// - `cancel` — the per-turn cancellation signal (P1). The decode loop
    ///   checks it BETWEEN tokens; when raised mid-generation the call returns
    ///   [`TurnOutcome::Cancelled`] carrying the partial text.
    /// - `token_cb` — called per detokenised user-visible piece (the driver
    ///   maps these to `ChatToken` events). Tool-call JSON is NEVER streamed
    ///   through it.
    fn run_turn(
        &self,
        history: &[ChatMessage],
        tool_descriptors: &[ToolDescriptor],
        cfg: &SamplerConfig,
        cancel: &CancelFlag,
        token_cb: &mut dyn FnMut(&str),
    ) -> AppResult<TurnOutcome>;
}

/// The production engine: assembles the oaicompat prompt, drives a
/// [`TurnBackend`], and maps the raw turn to a [`TurnOutcome`].
///
/// Generic over the backend so the unit tests inject a stub and the real path
/// injects [`LlamaTurnBackend`]. `ipc-bridge` holds `Arc<dyn ChatEngine>`.
pub struct TurnEngine<B: TurnBackend> {
    backend: B,
}

impl<B: TurnBackend> TurnEngine<B> {
    /// Build the engine over a turn backend.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: TurnBackend> ChatEngine for TurnEngine<B> {
    fn run_turn(
        &self,
        history: &[ChatMessage],
        tool_descriptors: &[ToolDescriptor],
        cfg: &SamplerConfig,
        cancel: &CancelFlag,
        token_cb: &mut dyn FnMut(&str),
    ) -> AppResult<TurnOutcome> {
        let messages = messages_json(history)?;
        let tools = tools_json(tool_descriptors)?;

        let raw = self
            .backend
            .run(&messages, tools.as_deref(), cfg, cancel, token_cb)?;

        // A cancel observed mid-decode short-circuits the outcome mapping: the
        // turn was interrupted, not a final answer or a tool request (P1).
        if raw.cancelled {
            return Ok(TurnOutcome::Cancelled { partial: raw.text });
        }

        // Outcome mapping (§1.2): tool calls take precedence; else final text;
        // an empty turn (no calls, no text) is malformed.
        if !raw.tool_calls.is_empty() {
            return Ok(TurnOutcome::ToolCalls(raw.tool_calls));
        }
        let text = raw.text.trim();
        if text.is_empty() {
            return Err(Error::MalformedOutput(
                "assistant produced neither text nor a tool call".to_string(),
            )
            .into());
        }
        Ok(TurnOutcome::Final(text.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::AppError;
    use serde_json::json;
    use std::sync::Mutex;

    // ----- A stub backend driving the engine without a model ----------------

    /// A canned-response backend. Records the `messages_json` / `tools_json` it
    /// was handed (so the engine's prompt assembly is observable) and streams
    /// any `stream_chunks` through the callback before returning `result`. The
    /// `seen` cell is a `Mutex` (not `RefCell`) because `TurnBackend: Send +
    /// Sync`.
    struct StubBackend {
        result: RawTurn,
        stream_chunks: Vec<String>,
        seen: Mutex<Option<(String, Option<String>)>>,
    }

    impl StubBackend {
        fn final_text(text: &str) -> Self {
            Self {
                result: RawTurn {
                    text: text.to_string(),
                    tool_calls: vec![],
                    cancelled: false,
                },
                stream_chunks: text.split_inclusive(' ').map(str::to_string).collect(),
                seen: Mutex::new(None),
            }
        }
        fn tool_call(name: &str, args: &str) -> Self {
            Self {
                result: RawTurn {
                    text: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "call_0".into(),
                        name: name.into(),
                        arguments_json: args.into(),
                    }],
                    cancelled: false,
                },
                stream_chunks: vec![],
                seen: Mutex::new(None),
            }
        }
        fn empty() -> Self {
            Self {
                result: RawTurn::default(),
                stream_chunks: vec![],
                seen: Mutex::new(None),
            }
        }
    }

    impl TurnBackend for StubBackend {
        fn run(
            &self,
            messages_json: &str,
            tools_json: Option<&str>,
            _cfg: &SamplerConfig,
            _cancel: &CancelFlag,
            token_cb: &mut dyn FnMut(&str),
        ) -> Result<RawTurn, Error> {
            *self.seen.lock().unwrap() =
                Some((messages_json.to_string(), tools_json.map(str::to_string)));
            for chunk in &self.stream_chunks {
                token_cb(chunk);
            }
            Ok(self.result.clone())
        }
    }

    fn descriptor(name: &'static str) -> ToolDescriptor {
        ToolDescriptor {
            name,
            title: "Test tool",
            description: "d",
            input_schema: json!({
                "type": "object",
                "properties": { "meeting_id": { "type": "string" } },
                "required": ["meeting_id"],
                "additionalProperties": false,
            }),
        }
    }

    fn history() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user("what was said?"),
        ]
    }

    #[test]
    fn final_text_turn_streams_and_returns_final() {
        let engine = TurnEngine::new(StubBackend::final_text("hello world"));
        let mut streamed = String::new();
        let outcome = engine
            .run_turn(
                &history(),
                &[descriptor("get_transcript")],
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |t| streamed.push_str(t),
            )
            .unwrap();
        assert_eq!(outcome, TurnOutcome::Final("hello world".to_string()));
        assert_eq!(
            streamed, "hello world",
            "tokens stream through the callback"
        );
    }

    #[test]
    fn tool_call_turn_parses_into_tool_calls() {
        let engine = TurnEngine::new(StubBackend::tool_call(
            "get_transcript",
            "{\"meeting_id\":\"abc\"}",
        ));
        let outcome = engine
            .run_turn(
                &history(),
                &[descriptor("get_transcript")],
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
        match outcome {
            TurnOutcome::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_transcript");
                assert_eq!(calls[0].arguments_json, "{\"meeting_id\":\"abc\"}");
            }
            other => panic!("expected ToolCalls, got {other:?}"),
        }
    }

    #[test]
    fn malformed_empty_turn_maps_to_invalid_input() {
        let engine = TurnEngine::new(StubBackend::empty());
        let err = engine
            .run_turn(
                &history(),
                &[],
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap_err();
        assert!(
            matches!(err, AppError::InvalidInput { .. }),
            "malformed output surfaces as InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn backend_error_maps_to_app_error() {
        struct Failing;
        impl TurnBackend for Failing {
            fn run(
                &self,
                _: &str,
                _: Option<&str>,
                _: &SamplerConfig,
                _: &CancelFlag,
                _: &mut dyn FnMut(&str),
            ) -> Result<RawTurn, Error> {
                Err(Error::Inference("decode blew up".into()))
            }
        }
        let engine = TurnEngine::new(Failing);
        let err = engine
            .run_turn(
                &history(),
                &[],
                &SamplerConfig::default(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap_err();
        assert!(matches!(err, AppError::Inference { .. }));
    }

    #[test]
    fn cancelled_raw_turn_maps_to_cancelled_outcome() {
        // P1: a backend that reports `cancelled` (the decode loop saw the flag
        // raised) maps to `TurnOutcome::Cancelled` carrying the partial text,
        // never a `Final` or a malformed-output error.
        struct CancelledBackend;
        impl TurnBackend for CancelledBackend {
            fn run(
                &self,
                _: &str,
                _: Option<&str>,
                _: &SamplerConfig,
                _: &CancelFlag,
                token_cb: &mut dyn FnMut(&str),
            ) -> Result<RawTurn, Error> {
                token_cb("partial ");
                Ok(RawTurn {
                    text: "partial ".to_string(),
                    tool_calls: vec![],
                    cancelled: true,
                })
            }
        }
        let engine = TurnEngine::new(CancelledBackend);
        let cancel = CancelFlag::new();
        cancel.cancel();
        let outcome = engine
            .run_turn(
                &history(),
                &[descriptor("get_transcript")],
                &SamplerConfig::deterministic(),
                &cancel,
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            outcome,
            TurnOutcome::Cancelled {
                partial: "partial ".to_string()
            }
        );
    }

    #[test]
    fn engine_assembles_oaicompat_prompt_from_history_and_descriptors() {
        let backend = StubBackend::final_text("ok");
        let engine = TurnEngine::new(backend);
        engine
            .run_turn(
                &history(),
                &[descriptor("get_transcript"), descriptor("get_summary")],
                &SamplerConfig::deterministic(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
        let seen = engine.backend.seen.lock().unwrap();
        let (messages, tools) = seen.as_ref().expect("backend saw a call");
        // messages_json carries the OpenAI roles.
        assert!(messages.contains("\"role\":\"system\""));
        assert!(messages.contains("\"role\":\"user\""));
        // tools_json wraps both descriptors in the OpenAI function shape.
        let tools = tools.as_ref().expect("tools were offered");
        assert!(tools.contains("\"type\":\"function\""));
        assert!(tools.contains("get_transcript"));
        assert!(tools.contains("get_summary"));
    }

    #[test]
    fn empty_descriptors_force_a_tool_less_turn() {
        let backend = StubBackend::final_text("done");
        let engine = TurnEngine::new(backend);
        engine
            .run_turn(
                &history(),
                &[],
                &SamplerConfig::default(),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
        let seen = engine.backend.seen.lock().unwrap();
        let (_, tools) = seen.as_ref().unwrap();
        assert!(
            tools.is_none(),
            "no descriptors ⇒ tools_json None (tool-less turn)"
        );
    }

    // ----- Deterministic sampler selection ----------------------------------

    #[test]
    fn sampler_config_zero_temperature_is_greedy() {
        assert!(SamplerConfig::deterministic().is_greedy());
        assert!(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        }
        .is_greedy());
        assert!(!SamplerConfig::default().is_greedy());
        assert!(!SamplerConfig {
            temperature: 0.7,
            ..SamplerConfig::default()
        }
        .is_greedy());
    }

    // ----- Every registry schema must compile to a GBNF grammar (CI gate) ---

    /// The `agent-tools` contract forbids regex `pattern` in schemas because the
    /// vendored llama.cpp schema→GBNF converter rejects PCRE shorthands. Compile
    /// every v1 registry schema through `json_schema_to_grammar`; a future
    /// incompatible schema fails CI here (a Phase 9 guard).
    #[test]
    fn every_registry_schema_compiles_to_grammar() {
        let registry = agent_tools::ToolRegistry::v1(false);
        for desc in registry.descriptors() {
            let schema = serde_json::to_string(&desc.input_schema).unwrap();
            let grammar = LlamaTurnBackend::schema_grammar(&schema).unwrap_or_else(|e| {
                panic!(
                    "tool {} schema must compile to a GBNF grammar: {e:?}",
                    desc.name
                )
            });
            assert!(
                grammar.contains("root ::="),
                "tool {} grammar must define a root rule",
                desc.name
            );
        }
    }

    // ----- The sliding-window trim is exercised here too (engine-facing) ----

    #[test]
    fn sliding_window_trim_keeps_pinned_head_and_recent() {
        // head + four turns; force eviction; the head and most-recent survive.
        let lens = [20usize, 40, 40, 40, 40];
        match trim_to_budget(&lens, 50, 10, 200) {
            TrimOutcome::Fits { drop_after_head } => {
                assert!(drop_after_head >= 1, "eviction happened");
                let surviving: usize = lens[0] + lens[1 + drop_after_head..].iter().sum::<usize>();
                assert!(fits_budget(surviving, 50, 10, 200));
            }
            other => panic!("expected Fits, got {other:?}"),
        }
        // A single over-budget turn is the hard floor.
        assert_eq!(
            trim_to_budget(&[50, 200], 50, 10, 200),
            TrimOutcome::HardFloor
        );
    }
}
