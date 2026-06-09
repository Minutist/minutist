//! The chat driver — the State-free turn loop, the held-model seam, and the
//! chat command bodies (Phase 9 §6).
//!
//! `chat-agent`'s [`ChatEngine`] is **stateless per call**: it renders ONE
//! assistant turn and never owns history, runs tools, or emits events. This
//! module is the DRIVER: it owns the conversation history, applies the sliding
//! window, runs the tool loop with a max-iteration cap, injects a per-turn seed,
//! and maps the engine's streamed tokens + tool calls onto the chat
//! `AppEvent`s.
//!
//! # The driver-loop seam ([`run_chat_turn`])
//!
//! The loop is generic over the engine and a tool-dispatch closure, and takes an
//! emit closure, so the default test suite drives a full turn with a STUB engine
//! and STUB tools — no Tauri runtime, no model, no tokio reactor. Production
//! injects the real [`chat_agent::TurnEngine`] over [`chat_agent::LlamaTurnBackend`]
//! (built from the held model) and a dispatch closure that re-enters async via a
//! captured `Handle::block_on(registry.dispatch(...))` (§4.5: the ONE place
//! async/sync cross). The whole call runs on `spawn_blocking` because the engine
//! is FFI-bound.
//!
//! # Per-turn seed (binding)
//!
//! `chat_agent::SamplerConfig`'s default `seed` is `0` — FIXED/reproducible, so
//! every non-greedy reply would be verbatim-identical. The driver therefore
//! injects a per-turn seed via [`per_turn_seed`] (never `0`) before each
//! `run_turn`, EXCEPT in the deterministic (greedy) profile where the seed is
//! ignored anyway.

use std::sync::atomic::{AtomicU64, Ordering};

use chat_agent::{ChatEngine, ChatMessage, Role, SamplerConfig, ToolCall, TurnOutcome};
use chat_agent::{fits_budget, trim_to_budget, TrimOutcome, HARD_FLOOR_REJECT};
use agent_tools::{ToolDescriptor, ToolOutput};
use meeting_app_common::{AppError, AppEvent, AppResult, ChatSessionId};

/// The max number of tool-call iterations one turn may take before the driver
/// forces a final answer (§6.1). A model that keeps requesting tools past this
/// is looping; the driver re-prompts ONCE with no tools to extract a final
/// reply, and emits [`AppEvent::ChatError`] if even that fails.
pub const MAX_TOOL_ITERATIONS: u32 = 8;

/// A fixed reserve (tokens) for template markers the per-message length estimate
/// may miss, mirroring `chat_agent::window`'s contract. Conservative.
const CONTEXT_RESERVE_TOKENS: usize = 256;

/// Outcome of one driver turn — the assistant's final text, plus the number of
/// tool iterations it took (for logging / tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnResult {
    /// The assistant's final reply text.
    pub final_text: String,
    /// How many tool-call iterations the turn ran.
    pub tool_iterations: u32,
}

/// Run ONE chat turn end-to-end: the engine/tool loop with streaming, tool
/// dispatch, the max-iteration cap, and the chat-event emissions.
///
/// State-free and generic so the default test suite drives it without a model
/// or a Tauri runtime:
///
/// - `engine` — any [`ChatEngine`] (the real `TurnEngine<LlamaTurnBackend>` in
///   production; a stub in tests).
/// - `history` — the FULL conversation so far (the driver owns it; index 0 is
///   the pinned system prompt). Mutated in place: the loop appends the tool
///   messages it produces, and the caller reads the appended assistant text from
///   the returned [`TurnResult`].
/// - `descriptors` — the offered tools (`registry.descriptors()`); an empty
///   slice forces a tool-less final-answer turn (the max-iteration escape).
/// - `cfg` — the base sampler config; the driver injects a per-turn seed into a
///   clone before each `run_turn` (greedy turns ignore the seed).
/// - `dispatch` — runs ONE tool call to a [`ToolOutput`] (production: a closure
///   that does `Handle::block_on(registry.dispatch(ctx, name, args))`).
/// - `emit` — receives each chat `AppEvent` (production: forwards to the
///   broadcast bus; tests: records into a Vec).
///
/// Returns the assistant's final text. Streaming `ChatToken`s, `ChatToolCall` /
/// `ChatToolResult` per dispatch, and the terminal `ChatTurnComplete` are all
/// emitted through `emit`; a hard-floor context overflow or a backend error
/// surfaces both as an `Err(AppError)` AND an emitted `ChatError`.
#[allow(clippy::too_many_arguments)]
pub fn run_chat_turn<E, D, M>(
    engine: &E,
    session_id: ChatSessionId,
    turn_id: u64,
    history: &mut Vec<ChatMessage>,
    descriptors: &[ToolDescriptor],
    cfg: &SamplerConfig,
    n_ctx: usize,
    mut dispatch: D,
    mut emit: M,
) -> AppResult<TurnResult>
where
    E: ChatEngine + ?Sized,
    D: FnMut(&ToolCall) -> AppResult<ToolOutput>,
    M: FnMut(AppEvent),
{
    let mut iteration: u32 = 0;
    loop {
        // Apply the sliding-window trim before each engine call (§6.2). A
        // hard-floor overflow is a genuinely-too-large turn → reject + ChatError.
        apply_trim(history, cfg, n_ctx).inspect_err(|e| {
            emit(AppEvent::ChatError {
                session_id,
                message: e.to_string(),
            });
        })?;

        // The max-iteration escape: once the cap is hit, offer NO tools so the
        // model must answer in free text (§6.1 step d).
        let offered: &[ToolDescriptor] = if iteration >= MAX_TOOL_ITERATIONS {
            &[]
        } else {
            descriptors
        };

        let turn_cfg = with_per_turn_seed(cfg, turn_id, iteration);

        // Stream tokens straight through as ChatToken events.
        let mut token_cb = |token: &str| {
            emit(AppEvent::ChatToken {
                session_id,
                turn_id,
                token: token.to_string(),
            });
        };

        let outcome = engine
            .run_turn(history, offered, &turn_cfg, &mut token_cb)
            .inspect_err(|e| {
                emit(AppEvent::ChatError {
                    session_id,
                    message: e.to_string(),
                });
            })?;

        match outcome {
            TurnOutcome::Final(text) => {
                history.push(ChatMessage::assistant(text.clone()));
                emit(AppEvent::ChatTurnComplete {
                    session_id,
                    turn_id,
                    final_text: text.clone(),
                });
                return Ok(TurnResult {
                    final_text: text,
                    tool_iterations: iteration,
                });
            }
            TurnOutcome::ToolCalls(_calls) if offered.is_empty() => {
                // The escape turn (no tools offered) still requested a tool: the
                // model is misbehaving. Treat the loop as exhausted with no final
                // answer (§6.1 — the cap is the backstop).
                let message = format!(
                    "chat turn exceeded {MAX_TOOL_ITERATIONS} tool iterations without a final answer"
                );
                emit(AppEvent::ChatError {
                    session_id,
                    message: message.clone(),
                });
                return Err(AppError::Internal { context: message });
            }
            TurnOutcome::ToolCalls(calls) => {
                for call in &calls {
                    emit(AppEvent::ChatToolCall {
                        session_id,
                        turn_id,
                        tool: call.name.clone(),
                        args_json: call.arguments_json.clone(),
                    });

                    let (ok, summary, content) = match dispatch(call) {
                        Ok(output) => {
                            let summary = output
                                .summary
                                .clone()
                                .unwrap_or_else(|| format!("{} ok", call.name));
                            // The machine payload is fed back to the model as the
                            // tool result content (a JSON string).
                            let content = serde_json::to_string(&output.data)
                                .unwrap_or_else(|_| "{}".to_string());
                            (true, summary, content)
                        }
                        Err(e) => {
                            let message = e.to_string();
                            // Feed the error back so the model can recover on the
                            // next iteration rather than crashing the turn.
                            let content =
                                serde_json::json!({ "error": message }).to_string();
                            (false, message, content)
                        }
                    };

                    emit(AppEvent::ChatToolResult {
                        session_id,
                        turn_id,
                        tool: call.name.clone(),
                        ok,
                        summary,
                    });

                    history.push(ChatMessage::tool_result(
                        call.id.clone(),
                        call.name.clone(),
                        content,
                    ));
                }
                iteration += 1;
                // Loop: re-invoke the engine with the appended tool results.
            }
        }
    }
}

/// Apply the sliding-window trim to `history` in place (§6.2).
///
/// Estimates each message's token length (a cheap chars/4 heuristic — the engine
/// re-tokenises authoritatively, and a fresh context per turn makes eviction
/// free), then drops the oldest non-pinned messages per
/// [`chat_agent::trim_to_budget`]. A hard floor (a single turn too large even
/// after dropping all evictable history) is rejected as
/// `AppError::InvalidInput { context: HARD_FLOOR_REJECT }`.
fn apply_trim(history: &mut Vec<ChatMessage>, cfg: &SamplerConfig, n_ctx: usize) -> AppResult<()> {
    if history.len() <= 1 {
        // Pinned head only (or empty): nothing evictable. Still hard-floor-check.
        let total: usize = history.iter().map(estimate_tokens).sum();
        if !fits_budget(total, cfg.max_tokens, CONTEXT_RESERVE_TOKENS, n_ctx) {
            return Err(AppError::InvalidInput {
                context: HARD_FLOOR_REJECT.to_string(),
            });
        }
        return Ok(());
    }

    let lens: Vec<usize> = history.iter().map(estimate_tokens).collect();
    match trim_to_budget(&lens, cfg.max_tokens, CONTEXT_RESERVE_TOKENS, n_ctx) {
        TrimOutcome::Fits { drop_after_head: 0 } => Ok(()),
        TrimOutcome::Fits { drop_after_head } => {
            // Drop `drop_after_head` messages immediately after the pinned head.
            history.drain(1..1 + drop_after_head);
            Ok(())
        }
        TrimOutcome::HardFloor => Err(AppError::InvalidInput {
            context: HARD_FLOOR_REJECT.to_string(),
        }),
    }
}

/// Cheap per-message token estimate: ~4 chars per token, min 1. The engine's
/// real tokeniser is authoritative; this only drives the driver's eviction
/// decision, and a fresh context per turn means an imperfect estimate costs at
/// most one extra eviction, never a correctness bug.
fn estimate_tokens(m: &ChatMessage) -> usize {
    (m.content.chars().count() / 4).max(1)
}

/// Build a per-turn [`SamplerConfig`] with an injected non-zero seed.
///
/// Greedy (deterministic) turns leave the config untouched — the seed is ignored
/// on the greedy path, and the test suite relies on greedy reproducibility.
/// Non-greedy turns get [`per_turn_seed`] so each reply is independently sampled
/// rather than verbatim-identical (the default `seed = 0` trap).
fn with_per_turn_seed(cfg: &SamplerConfig, turn_id: u64, iteration: u32) -> SamplerConfig {
    if cfg.is_greedy() {
        return cfg.clone();
    }
    SamplerConfig {
        seed: per_turn_seed(turn_id, iteration),
        ..cfg.clone()
    }
}

/// A monotonic process-wide counter that perturbs the seed so two turns started
/// in the same nanosecond still differ.
static SEED_NONCE: AtomicU64 = AtomicU64::new(0);

/// Derive a non-zero per-turn RNG seed from wall-clock nanos, the turn id, the
/// tool iteration, and a process-wide nonce.
///
/// Guaranteed non-zero (the `| 1` floor) so a non-greedy turn never falls back
/// to the fixed `seed = 0` reproducible path. `pub` so the seed strategy is
/// unit-testable (the test asserts it is never zero and varies).
pub fn per_turn_seed(turn_id: u64, iteration: u32) -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let nonce = SEED_NONCE.fetch_add(1, Ordering::Relaxed);
    let mixed = nanos
        ^ turn_id.rotate_left(17)
        ^ (iteration as u64).rotate_left(29)
        ^ nonce.rotate_left(41);
    // Fold to u32 and force non-zero (seed = 0 is the reproducible trap).
    let folded = (mixed ^ (mixed >> 32)) as u32;
    folded | 1
}

/// The maximum chat context to size the engine + the driver's budget guard to.
///
/// Mirrors `chat_agent::LlamaTurnConfig` / `summariser::SummariserConfig`
/// (32 768) so the driver's sliding-window arithmetic matches the context the
/// real backend allocates.
pub const CHAT_N_CTX: usize = 32_768;

/// Build the engine-internal history for a fresh session: the pinned system
/// prompt as message 0.
pub fn initial_history(system_prompt: &str) -> Vec<ChatMessage> {
    vec![ChatMessage::system(system_prompt)]
}

/// Map a `common::ChatMessage` (the persisted/wire shape) into the engine's
/// internal [`ChatMessage`]. The driver rebuilds the engine history from a loaded
/// session this way.
pub fn engine_message_from_wire(m: &meeting_app_common::ChatMessage) -> ChatMessage {
    use meeting_app_common::ChatRole;
    match m.role {
        ChatRole::System => ChatMessage::system(m.content.clone()),
        ChatRole::User => ChatMessage::user(m.content.clone()),
        ChatRole::Assistant => ChatMessage::assistant(m.content.clone()),
        ChatRole::Tool => ChatMessage::tool_result(
            // The wire shape does not carry the OpenAI tool_call_id; synthesise a
            // stable id from the turn + tool name so the template can still
            // attribute the result. The model only needs the name + content.
            format!("call_{}", m.turn_id),
            m.tool_name.clone().unwrap_or_default(),
            m.content.clone(),
        ),
    }
}

/// Map the engine's role to the persisted/wire `common::ChatRole`.
pub fn wire_role(role: Role) -> meeting_app_common::ChatRole {
    use meeting_app_common::ChatRole;
    match role {
        Role::System => ChatRole::System,
        Role::User => ChatRole::User,
        Role::Assistant => ChatRole::Assistant,
        Role::Tool => ChatRole::Tool,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_tools::ToolOutput;
    use chat_agent::ToolCall as EngineToolCall;
    use std::cell::RefCell;
    use std::sync::Mutex;

    // ----- A stub ChatEngine driving the loop without a model ---------------

    /// A scripted engine: each `run_turn` returns the next queued outcome,
    /// streaming the queued chunks for that step through the token callback.
    struct ScriptedEngine {
        /// One entry per `run_turn` call: (streamed chunks, outcome).
        steps: Mutex<std::collections::VecDeque<(Vec<String>, TurnOutcome)>>,
    }

    impl ScriptedEngine {
        fn new(steps: Vec<(Vec<String>, TurnOutcome)>) -> Self {
            Self {
                steps: Mutex::new(steps.into_iter().collect()),
            }
        }
        fn final_only(text: &str) -> Self {
            Self::new(vec![(
                text.split_inclusive(' ').map(str::to_string).collect(),
                TurnOutcome::Final(text.to_string()),
            )])
        }
    }

    impl ChatEngine for ScriptedEngine {
        fn run_turn(
            &self,
            _history: &[ChatMessage],
            _descriptors: &[ToolDescriptor],
            _cfg: &SamplerConfig,
            token_cb: &mut dyn FnMut(&str),
        ) -> AppResult<TurnOutcome> {
            let (chunks, outcome) = self
                .steps
                .lock()
                .unwrap()
                .pop_front()
                .expect("ScriptedEngine ran out of scripted steps");
            for c in &chunks {
                token_cb(c);
            }
            Ok(outcome)
        }
    }

    fn tool_call(name: &str, args: &str) -> EngineToolCall {
        EngineToolCall {
            id: format!("call_{name}"),
            name: name.to_string(),
            arguments_json: args.to_string(),
        }
    }

    fn descriptor(name: &'static str) -> ToolDescriptor {
        ToolDescriptor {
            name,
            description: "d",
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        }
    }

    fn base_history() -> Vec<ChatMessage> {
        vec![ChatMessage::system("sys"), ChatMessage::user("hi")]
    }

    /// Collect emitted events into a Vec for assertions.
    fn collector() -> (std::rc::Rc<RefCell<Vec<AppEvent>>>, impl FnMut(AppEvent)) {
        let events = std::rc::Rc::new(RefCell::new(Vec::new()));
        let sink = events.clone();
        (events, move |e: AppEvent| sink.borrow_mut().push(e))
    }

    fn no_dispatch(_: &EngineToolCall) -> AppResult<ToolOutput> {
        panic!("dispatch must not be called for a final-only turn")
    }

    #[test]
    fn final_only_turn_emits_turn_complete() {
        let engine = ScriptedEngine::final_only("hello world");
        let mut history = base_history();
        let (events, emit) = collector();
        let sid = ChatSessionId::new();

        let result = run_chat_turn(
            &engine,
            sid,
            1,
            &mut history,
            &[descriptor("get_transcript")],
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            no_dispatch,
            emit,
        )
        .expect("final-only turn must succeed");

        assert_eq!(result.final_text, "hello world");
        assert_eq!(result.tool_iterations, 0);

        let ev = events.borrow();
        // Tokens streamed, then exactly one ChatTurnComplete, no tool events.
        assert!(ev.iter().any(|e| matches!(e, AppEvent::ChatToken { .. })));
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, AppEvent::ChatTurnComplete { .. }))
                .count(),
            1
        );
        assert!(!ev.iter().any(|e| matches!(e, AppEvent::ChatToolCall { .. })));
        // The assistant message was appended to the history.
        assert_eq!(history.last().unwrap().role, Role::Assistant);
    }

    #[test]
    fn tool_call_turn_dispatches_then_loops_to_final() {
        let engine = ScriptedEngine::new(vec![
            (vec![], TurnOutcome::ToolCalls(vec![tool_call("get_transcript", "{}")])),
            (
                vec!["the ".into(), "answer".into()],
                TurnOutcome::Final("the answer".to_string()),
            ),
        ]);
        let mut history = base_history();
        let (events, emit) = collector();
        let sid = ChatSessionId::new();

        let dispatched = std::cell::Cell::new(0u32);
        let dispatch = |call: &EngineToolCall| {
            assert_eq!(call.name, "get_transcript");
            dispatched.set(dispatched.get() + 1);
            Ok(ToolOutput::new(
                serde_json::json!({ "segments": [] }),
                "0 segments",
            ))
        };

        let result = run_chat_turn(
            &engine,
            sid,
            2,
            &mut history,
            &[descriptor("get_transcript")],
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            dispatch,
            emit,
        )
        .expect("tool-call turn must reach a final answer");

        assert_eq!(result.final_text, "the answer");
        assert_eq!(result.tool_iterations, 1);
        assert_eq!(dispatched.get(), 1, "the tool must have been dispatched once");

        let ev = events.borrow();
        assert_eq!(
            ev.iter().filter(|e| matches!(e, AppEvent::ChatToolCall { .. })).count(),
            1
        );
        let result_ok = ev.iter().find_map(|e| match e {
            AppEvent::ChatToolResult { ok, summary, .. } => Some((*ok, summary.clone())),
            _ => None,
        });
        assert_eq!(result_ok, Some((true, "0 segments".to_string())));
        assert_eq!(
            ev.iter().filter(|e| matches!(e, AppEvent::ChatTurnComplete { .. })).count(),
            1
        );
        // A Tool message was appended carrying the dispatch payload.
        assert!(history.iter().any(|m| m.role == Role::Tool
            && m.content.contains("segments")));
    }

    #[test]
    fn tool_error_is_fed_back_and_turn_can_still_finish() {
        let engine = ScriptedEngine::new(vec![
            (vec![], TurnOutcome::ToolCalls(vec![tool_call("get_transcript", "{}")])),
            (vec![], TurnOutcome::Final("recovered".to_string())),
        ]);
        let mut history = base_history();
        let (events, emit) = collector();

        let dispatch = |_: &EngineToolCall| -> AppResult<ToolOutput> {
            Err(AppError::InvalidInput {
                context: "no such meeting".into(),
            })
        };

        let result = run_chat_turn(
            &engine,
            ChatSessionId::new(),
            3,
            &mut history,
            &[descriptor("get_transcript")],
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            dispatch,
            emit,
        )
        .expect("a tool error must be fed back, not crash the turn");

        assert_eq!(result.final_text, "recovered");
        let ev = events.borrow();
        let errored = ev.iter().any(|e| matches!(e, AppEvent::ChatToolResult { ok: false, .. }));
        assert!(errored, "the failed dispatch must emit ChatToolResult ok=false");
        // The error content was fed back as a Tool message.
        assert!(history.iter().any(|m| m.role == Role::Tool && m.content.contains("error")));
    }

    #[test]
    fn max_iteration_cap_emits_chat_error() {
        // An engine that ALWAYS asks for a tool, even when offered none.
        struct AlwaysTool;
        impl ChatEngine for AlwaysTool {
            fn run_turn(
                &self,
                _history: &[ChatMessage],
                _descriptors: &[ToolDescriptor],
                _cfg: &SamplerConfig,
                _token_cb: &mut dyn FnMut(&str),
            ) -> AppResult<TurnOutcome> {
                Ok(TurnOutcome::ToolCalls(vec![tool_call("get_transcript", "{}")]))
            }
        }

        let mut history = base_history();
        let (events, emit) = collector();
        let dispatch = |_: &EngineToolCall| Ok(ToolOutput::new(serde_json::json!({}), "ok"));

        let err = run_chat_turn(
            &AlwaysTool,
            ChatSessionId::new(),
            4,
            &mut history,
            &[descriptor("get_transcript")],
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            dispatch,
            emit,
        )
        .expect_err("a runaway tool loop must surface an error");

        assert!(matches!(err, AppError::Internal { .. }));
        let ev = events.borrow();
        assert!(
            ev.iter().any(|e| matches!(e, AppEvent::ChatError { .. })),
            "the cap must emit ChatError"
        );
    }

    #[test]
    fn hard_floor_context_overflow_rejects_and_emits_error() {
        let engine = ScriptedEngine::final_only("unused");
        // A single user message larger than a tiny context window → hard floor.
        let mut history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("x".repeat(10_000)),
        ];
        let (events, emit) = collector();

        let err = run_chat_turn(
            &engine,
            ChatSessionId::new(),
            5,
            &mut history,
            &[],
            &SamplerConfig::deterministic(),
            64, // tiny n_ctx forces the hard floor
            no_dispatch,
            emit,
        )
        .expect_err("an over-large single turn must be rejected");

        match err {
            AppError::InvalidInput { context } => assert_eq!(context, HARD_FLOOR_REJECT),
            other => panic!("expected InvalidInput hard floor, got {other:?}"),
        }
        let ev = events.borrow();
        assert!(ev.iter().any(|e| matches!(e, AppEvent::ChatError { .. })));
    }

    #[test]
    fn per_turn_seed_is_never_zero_and_varies() {
        // Non-zero invariant — a non-greedy turn must never fall back to seed=0.
        for turn in 0..16u64 {
            for iter in 0..4u32 {
                assert_ne!(per_turn_seed(turn, iter), 0, "seed must never be zero");
            }
        }
        // And consecutive seeds differ (the nonce + nanos guarantee variation).
        let a = per_turn_seed(1, 0);
        let b = per_turn_seed(1, 0);
        assert_ne!(a, b, "two seeds for the same turn must differ");
    }

    #[test]
    fn greedy_config_keeps_fixed_seed_non_greedy_injects() {
        // Greedy: seed untouched (deterministic test reproducibility).
        let greedy = SamplerConfig::deterministic();
        assert_eq!(with_per_turn_seed(&greedy, 1, 0).seed, greedy.seed);
        // Non-greedy: a non-zero seed is injected (not the default 0).
        let sampling = SamplerConfig::default();
        assert_eq!(sampling.seed, 0, "precondition: default seed is the fixed-0 trap");
        let injected = with_per_turn_seed(&sampling, 1, 0);
        assert_ne!(injected.seed, 0, "non-greedy turns must get a non-zero seed");
    }
}
