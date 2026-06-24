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

use agent_tools::{ToolDescriptor, ToolOutput};
use chat_agent::{fits_budget, trim_to_budget, TrimOutcome, HARD_FLOOR_REJECT};
use chat_agent::{CancelFlag, ChatEngine, ChatMessage, Role, SamplerConfig, ToolCall, TurnOutcome};
use minutist_common::{AppError, AppEvent, AppResult, ChatSessionId};

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
/// - `cancel` — the per-turn cancellation signal (P1). Passed into each engine
///   call; the decode loop checks it between tokens and returns
///   [`TurnOutcome::Cancelled`], which the driver turns into a terminal
///   `ChatTurnComplete` carrying the partial text (the session is not an error).
/// - `dispatch` — runs ONE tool call to a [`ToolOutput`] (production: a closure
///   that does `Handle::block_on(registry.dispatch(ctx, name, args))`).
/// - `emit` — receives each chat `AppEvent` (production: forwards to the
///   broadcast bus; tests: records into a Vec).
///
/// Returns the assistant's final text. Streaming `ChatToken`s, `ChatToolCall` /
/// `ChatToolResult` per dispatch, a `ChatContextTrimmed` when the sliding window
/// evicts history (P2), and the terminal `ChatTurnComplete` are all emitted
/// through `emit`; a hard-floor context overflow or a backend error surfaces
/// both as an `Err(AppError)` AND an emitted `ChatError`.
#[allow(clippy::too_many_arguments)]
pub fn run_chat_turn<E, D, M>(
    engine: &E,
    session_id: ChatSessionId,
    turn_id: u64,
    history: &mut Vec<ChatMessage>,
    descriptors: &[ToolDescriptor],
    cfg: &SamplerConfig,
    n_ctx: usize,
    cancel: &CancelFlag,
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
        // On eviction, emit ChatContextTrimmed with the dropped count (P2).
        let dropped = apply_trim(history, cfg, n_ctx).inspect_err(|e| {
            emit(AppEvent::ChatError {
                session_id,
                message: e.to_string(),
            });
        })?;
        if dropped > 0 {
            emit(AppEvent::ChatContextTrimmed {
                session_id,
                dropped_turns: dropped as u32,
            });
        }

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
            .run_turn(history, offered, &turn_cfg, cancel, &mut token_cb)
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
            TurnOutcome::Cancelled { partial } => {
                // The user cancelled mid-stream (P1). End the turn cleanly: keep
                // the partial text as the assistant reply (so the session stays a
                // valid alternation) and emit the terminal ChatTurnComplete with
                // it. NOT a ChatError — cancellation is a user action, not a
                // failure.
                history.push(ChatMessage::assistant(partial.clone()));
                emit(AppEvent::ChatTurnComplete {
                    session_id,
                    turn_id,
                    final_text: partial.clone(),
                });
                return Ok(TurnResult {
                    final_text: partial,
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
                // CQ1: the OpenAI tool protocol is `assistant(tool_calls) →
                // tool(result)*`. Append the ASSISTANT message bearing the
                // requested tool_calls BEFORE the per-call tool results, so the
                // next engine render is a valid sequence (a bare `tool` message
                // with no preceding assistant-tool_calls is malformed and the
                // GGUF tool template hard-errors or silently degrades on it).
                history.push(ChatMessage::assistant_tool_calls(
                    String::new(),
                    calls.clone(),
                ));

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
                            let content = serde_json::json!({ "error": message }).to_string();
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

/// Apply the sliding-window trim to `history` in place (§6.2). Returns the
/// number of messages evicted (`0` when nothing was dropped) so the driver can
/// emit a `ChatContextTrimmed` event (P2).
///
/// Estimates each message's token length (a cheap chars/4 heuristic — the engine
/// re-tokenises authoritatively, and a fresh context per turn makes eviction
/// free), then drops the oldest non-pinned messages per
/// [`chat_agent::trim_to_budget`]. The pure planner returns the MINIMUM count to
/// drop; the driver (which owns the message roles) then SNAPS that count FORWARD
/// to the next user-message boundary so the surviving window after the pinned
/// head starts on a `User` turn (CQ2) — leaving an orphan `Assistant`/`Tool` at
/// `history[1]` is a malformed OpenAI sequence and (with CQ1) breaks the tool
/// template. A hard floor (a single turn too large even after dropping all
/// evictable history) is rejected as
/// `AppError::InvalidInput { context: HARD_FLOOR_REJECT }`.
fn apply_trim(
    history: &mut Vec<ChatMessage>,
    cfg: &SamplerConfig,
    n_ctx: usize,
) -> AppResult<usize> {
    if history.len() <= 1 {
        // Pinned head only (or empty): nothing evictable. Still hard-floor-check.
        let total: usize = history.iter().map(estimate_tokens).sum();
        if !fits_budget(total, cfg.max_tokens, CONTEXT_RESERVE_TOKENS, n_ctx) {
            return Err(AppError::InvalidInput {
                context: HARD_FLOOR_REJECT.to_string(),
            });
        }
        return Ok(0);
    }

    let lens: Vec<usize> = history.iter().map(estimate_tokens).collect();
    match trim_to_budget(&lens, cfg.max_tokens, CONTEXT_RESERVE_TOKENS, n_ctx) {
        TrimOutcome::Fits { drop_after_head: 0 } => Ok(0),
        TrimOutcome::Fits { drop_after_head } => {
            // Snap the minimum drop count FORWARD to a user-message group
            // boundary so the survivor at history[1] is a `User` turn (CQ2).
            let drop = snap_to_group_boundary(history, drop_after_head);
            history.drain(1..1 + drop);
            Ok(drop)
        }
        TrimOutcome::HardFloor => Err(AppError::InvalidInput {
            context: HARD_FLOOR_REJECT.to_string(),
        }),
    }
}

/// Snap `drop_after_head` (the pure planner's minimum) FORWARD to the next
/// user-message group boundary (CQ2).
///
/// The OpenAI/template alternation is `user → assistant[(tool_calls) → tool*]`
/// groups after the pinned system head. Dropping a non-group-aligned prefix can
/// leave an orphan `Assistant`/`Tool` as the first survivor, which is malformed.
/// We therefore extend the drop forward to the next index `i >= 1 +
/// drop_after_head` whose message is a `User` (a group start), so the survivor
/// window begins on a clean turn boundary. We never advance past the last
/// message: the planner guarantees `drop_after_head <= history.len() - 2`, and
/// if no later `User` boundary exists we keep the planner's count rather than
/// evict the most-recent turn.
fn snap_to_group_boundary(history: &[ChatMessage], drop_after_head: usize) -> usize {
    let start = 1 + drop_after_head;
    // Search for the next group start (a User message) at or after `start`,
    // without ever reaching the final message (always retained).
    for (idx, msg) in history
        .iter()
        .enumerate()
        .take(history.len() - 1)
        .skip(start)
    {
        if msg.role == Role::User {
            return idx - 1;
        }
    }
    // No later user boundary before the last message: fall back to the planner's
    // count (better than dropping the whole tail). With CQ1 the surviving lead
    // is at worst an assistant message, still preferable to evicting the
    // most-recent turn.
    drop_after_head
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
///
/// CQ1: an `Assistant` message that carries `tool_calls` reconstructs the
/// engine's assistant-tool_calls message (so a reloaded multi-tool turn renders
/// the valid `assistant(tool_calls) → tool(result)` sequence), and a `Tool`
/// message uses its persisted `tool_call_id` so it re-links to the matching
/// call rather than a synthesised id.
pub fn engine_message_from_wire(m: &minutist_common::ChatMessage) -> ChatMessage {
    use minutist_common::ChatRole;
    match m.role {
        ChatRole::System => ChatMessage::system(m.content.clone()),
        ChatRole::User => ChatMessage::user(m.content.clone()),
        ChatRole::Assistant if !m.tool_calls.is_empty() => ChatMessage::assistant_tool_calls(
            m.content.clone(),
            m.tool_calls
                .iter()
                .map(|c| ToolCall {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    arguments_json: c.arguments_json.clone(),
                })
                .collect(),
        ),
        ChatRole::Assistant => ChatMessage::assistant(m.content.clone()),
        ChatRole::Tool => ChatMessage::tool_result(
            // Prefer the persisted tool_call_id (links to the assistant's
            // tool_calls); fall back to a stable per-turn synthetic id for an
            // older on-disk session written before the field existed.
            m.tool_call_id
                .clone()
                .unwrap_or_else(|| format!("call_{}", m.turn_id)),
            m.tool_name.clone().unwrap_or_default(),
            m.content.clone(),
        ),
    }
}

/// Map the engine's role to the persisted/wire `common::ChatRole`.
pub fn wire_role(role: Role) -> minutist_common::ChatRole {
    use minutist_common::ChatRole;
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
    /// streaming the queued chunks for that step through the token callback. It
    /// also CAPTURES the history it received on each call so a multi-iteration
    /// test can assert the exact sequence the engine saw (CQ1).
    struct ScriptedEngine {
        /// One entry per `run_turn` call: (streamed chunks, outcome).
        steps: Mutex<std::collections::VecDeque<(Vec<String>, TurnOutcome)>>,
        /// The history (a clone) seen on each `run_turn` call, in order.
        seen_histories: Mutex<Vec<Vec<ChatMessage>>>,
    }

    impl ScriptedEngine {
        fn new(steps: Vec<(Vec<String>, TurnOutcome)>) -> Self {
            Self {
                steps: Mutex::new(steps.into_iter().collect()),
                seen_histories: Mutex::new(Vec::new()),
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
            history: &[ChatMessage],
            _descriptors: &[ToolDescriptor],
            _cfg: &SamplerConfig,
            _cancel: &CancelFlag,
            token_cb: &mut dyn FnMut(&str),
        ) -> AppResult<TurnOutcome> {
            self.seen_histories.lock().unwrap().push(history.to_vec());
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
            title: "Test tool",
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
            &CancelFlag::new(),
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
        assert!(!ev
            .iter()
            .any(|e| matches!(e, AppEvent::ChatToolCall { .. })));
        // The assistant message was appended to the history.
        assert_eq!(history.last().unwrap().role, Role::Assistant);
    }

    #[test]
    fn tool_call_turn_dispatches_then_loops_to_final() {
        let engine = ScriptedEngine::new(vec![
            (
                vec![],
                TurnOutcome::ToolCalls(vec![tool_call("get_transcript", "{}")]),
            ),
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
            &CancelFlag::new(),
            dispatch,
            emit,
        )
        .expect("tool-call turn must reach a final answer");

        assert_eq!(result.final_text, "the answer");
        assert_eq!(result.tool_iterations, 1);
        assert_eq!(
            dispatched.get(),
            1,
            "the tool must have been dispatched once"
        );

        let ev = events.borrow();
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, AppEvent::ChatToolCall { .. }))
                .count(),
            1
        );
        let result_ok = ev.iter().find_map(|e| match e {
            AppEvent::ChatToolResult { ok, summary, .. } => Some((*ok, summary.clone())),
            _ => None,
        });
        assert_eq!(result_ok, Some((true, "0 segments".to_string())));
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, AppEvent::ChatTurnComplete { .. }))
                .count(),
            1
        );
        // A Tool message was appended carrying the dispatch payload.
        assert!(history
            .iter()
            .any(|m| m.role == Role::Tool && m.content.contains("segments")));

        // CQ1: the history the engine saw on the 2nd run_turn is
        // [system, user, assistant(tool_calls), tool] — the assistant message
        // bearing the tool_calls precedes the tool result (a valid OpenAI
        // sequence), not [system, user, tool, …].
        let seen = engine.seen_histories.lock().unwrap();
        assert_eq!(seen.len(), 2, "engine ran twice (tool turn + final turn)");
        let second = &seen[1];
        assert_eq!(second.len(), 4);
        assert_eq!(second[0].role, Role::System);
        assert_eq!(second[1].role, Role::User);
        assert_eq!(second[2].role, Role::Assistant);
        assert_eq!(
            second[2].tool_calls.len(),
            1,
            "the assistant message carries the requested tool_calls"
        );
        assert_eq!(second[2].tool_calls[0].name, "get_transcript");
        assert_eq!(second[3].role, Role::Tool);
        assert_eq!(
            second[3].tool_call_id.as_deref(),
            Some("call_get_transcript"),
            "the tool result links to the assistant tool_call id"
        );
    }

    #[test]
    fn tool_error_is_fed_back_and_turn_can_still_finish() {
        let engine = ScriptedEngine::new(vec![
            (
                vec![],
                TurnOutcome::ToolCalls(vec![tool_call("get_transcript", "{}")]),
            ),
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
            &CancelFlag::new(),
            dispatch,
            emit,
        )
        .expect("a tool error must be fed back, not crash the turn");

        assert_eq!(result.final_text, "recovered");
        let ev = events.borrow();
        let errored = ev
            .iter()
            .any(|e| matches!(e, AppEvent::ChatToolResult { ok: false, .. }));
        assert!(
            errored,
            "the failed dispatch must emit ChatToolResult ok=false"
        );
        // The error content was fed back as a Tool message.
        assert!(history
            .iter()
            .any(|m| m.role == Role::Tool && m.content.contains("error")));
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
                _cancel: &CancelFlag,
                _token_cb: &mut dyn FnMut(&str),
            ) -> AppResult<TurnOutcome> {
                Ok(TurnOutcome::ToolCalls(vec![tool_call(
                    "get_transcript",
                    "{}",
                )]))
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
            &CancelFlag::new(),
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
            &CancelFlag::new(),
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
        assert_eq!(
            sampling.seed, 0,
            "precondition: default seed is the fixed-0 trap"
        );
        let injected = with_per_turn_seed(&sampling, 1, 0);
        assert_ne!(
            injected.seed, 0,
            "non-greedy turns must get a non-zero seed"
        );
    }

    #[test]
    fn multi_tool_turn_history_is_valid_openai_sequence() {
        // CQ1: a turn that requests TWO tools in one iteration must produce ONE
        // assistant(tool_calls) message bearing both calls, followed by the two
        // tool results — the history the engine sees on the next run is
        // [system, user, assistant(2 tool_calls), tool, tool].
        let engine = ScriptedEngine::new(vec![
            (
                vec![],
                TurnOutcome::ToolCalls(vec![
                    tool_call("get_transcript", "{}"),
                    tool_call("get_summary", "{}"),
                ]),
            ),
            (vec![], TurnOutcome::Final("done".to_string())),
        ]);
        let mut history = base_history();
        let (_events, emit) = collector();
        let dispatch = |_: &EngineToolCall| Ok(ToolOutput::new(serde_json::json!({}), "ok"));

        run_chat_turn(
            &engine,
            ChatSessionId::new(),
            1,
            &mut history,
            &[descriptor("get_transcript"), descriptor("get_summary")],
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            &CancelFlag::new(),
            dispatch,
            emit,
        )
        .expect("multi-tool turn reaches a final answer");

        let seen = engine.seen_histories.lock().unwrap();
        let second = &seen[1];
        assert_eq!(
            second.len(),
            5,
            "[system, user, assistant(calls), tool, tool]"
        );
        assert_eq!(second[2].role, Role::Assistant);
        assert_eq!(
            second[2].tool_calls.len(),
            2,
            "ONE assistant message carries BOTH requested calls"
        );
        assert_eq!(second[3].role, Role::Tool);
        assert_eq!(second[4].role, Role::Tool);
        // No bare tool message precedes the assistant-tool_calls message.
        assert_eq!(second[1].role, Role::User);
    }

    #[test]
    fn cancel_flag_stops_turn_with_terminal_complete() {
        // P1: an engine that reports a Cancelled outcome ends the turn with a
        // terminal ChatTurnComplete carrying the partial text (NOT a ChatError),
        // and the partial is appended as the assistant reply.
        let engine = ScriptedEngine::new(vec![(
            vec!["par".into(), "tial".into()],
            TurnOutcome::Cancelled {
                partial: "partial".to_string(),
            },
        )]);
        let mut history = base_history();
        let (events, emit) = collector();
        let cancel = CancelFlag::new();
        cancel.cancel();

        let result = run_chat_turn(
            &engine,
            ChatSessionId::new(),
            1,
            &mut history,
            &[descriptor("get_transcript")],
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            &cancel,
            no_dispatch,
            emit,
        )
        .expect("a cancelled turn ends cleanly, not as an error");

        assert_eq!(result.final_text, "partial");
        let ev = events.borrow();
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, AppEvent::ChatTurnComplete { .. }))
                .count(),
            1,
            "cancellation emits a terminal ChatTurnComplete"
        );
        assert!(
            !ev.iter().any(|e| matches!(e, AppEvent::ChatError { .. })),
            "cancellation is NOT a ChatError"
        );
        assert_eq!(history.last().unwrap().role, Role::Assistant);
        assert_eq!(history.last().unwrap().content, "partial");
    }

    #[test]
    fn cancel_flag_set_makes_engine_return_cancelled_outcome() {
        // P1 (engine seam): the real engine threads the flag to the decode loop;
        // here we assert the driver passes the SAME flag through to the engine,
        // by observing the engine's behaviour change when the flag is raised.
        struct FlagWatchingEngine;
        impl ChatEngine for FlagWatchingEngine {
            fn run_turn(
                &self,
                _history: &[ChatMessage],
                _descriptors: &[ToolDescriptor],
                _cfg: &SamplerConfig,
                cancel: &CancelFlag,
                token_cb: &mut dyn FnMut(&str),
            ) -> AppResult<TurnOutcome> {
                token_cb("x");
                if cancel.is_cancelled() {
                    Ok(TurnOutcome::Cancelled {
                        partial: "x".to_string(),
                    })
                } else {
                    Ok(TurnOutcome::Final("full answer".to_string()))
                }
            }
        }
        let mut history = base_history();
        let (_events, emit) = collector();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let result = run_chat_turn(
            &FlagWatchingEngine,
            ChatSessionId::new(),
            1,
            &mut history,
            &[],
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            &cancel,
            no_dispatch,
            emit,
        )
        .expect("cancelled turn ends cleanly");
        assert_eq!(
            result.final_text, "x",
            "the raised flag reached the engine and it returned the partial"
        );
    }

    #[test]
    fn apply_trim_snaps_eviction_to_user_group_boundary() {
        // CQ2: eviction must not leave an orphan assistant/tool at history[1].
        // Build [system, user1, assistant(calls), tool, user2, assistant2] with
        // big messages and a tiny budget so the planner wants to drop a few; the
        // driver snaps the drop FORWARD so the survivor after the head is user2.
        let big = "x".repeat(4_000); // ~1000 token estimate each
        let mut history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user(big.clone()),
            ChatMessage::assistant_tool_calls(
                "",
                vec![ToolCall {
                    id: "call_1".into(),
                    name: "get_transcript".into(),
                    arguments_json: "{}".into(),
                }],
            ),
            ChatMessage::tool_result("call_1", "get_transcript", big.clone()),
            ChatMessage::user(big.clone()),
            ChatMessage::assistant(big.clone()),
        ];
        let cfg = SamplerConfig {
            max_tokens: 256,
            ..SamplerConfig::deterministic()
        };
        // n_ctx small enough that the first user-group (user1+assistant+tool) must
        // be evicted but the second group (user2+assistant2) survives.
        let dropped = apply_trim(&mut history, &cfg, 4_096).expect("fits after trim");
        assert!(dropped > 0, "eviction happened");
        // The pinned head is retained, and the first survivor after it is a User
        // (a clean group boundary) — never an orphan assistant/tool.
        assert_eq!(history[0].role, Role::System);
        assert_eq!(
            history[1].role,
            Role::User,
            "survivor after the pinned head must be a user-group start (CQ2)"
        );
        // The whole first group was dropped (no orphan tool/assistant lead).
        assert!(
            !history[1..]
                .iter()
                .take_while(|m| m.role != Role::User)
                .any(|m| matches!(m.role, Role::Tool)),
            "no orphan tool message leads the surviving window"
        );
    }

    #[test]
    fn eviction_emits_chat_context_trimmed() {
        // P2: when the sliding window evicts history, the driver emits
        // ChatContextTrimmed with the dropped count.
        let big = "x".repeat(4_000);
        let engine = ScriptedEngine::final_only("ok");
        let mut history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user(big.clone()),
            ChatMessage::assistant(big.clone()),
            ChatMessage::user(big.clone()),
            ChatMessage::assistant(big.clone()),
            ChatMessage::user("recent".to_string()),
        ];
        let (events, emit) = collector();
        // A small n_ctx forces eviction of the older groups.
        run_chat_turn(
            &engine,
            ChatSessionId::new(),
            2,
            &mut history,
            &[],
            &SamplerConfig {
                max_tokens: 256,
                ..SamplerConfig::deterministic()
            },
            4_096,
            &CancelFlag::new(),
            no_dispatch,
            emit,
        )
        .expect("turn succeeds after trim");
        let ev = events.borrow();
        let trimmed = ev.iter().find_map(|e| match e {
            AppEvent::ChatContextTrimmed { dropped_turns, .. } => Some(*dropped_turns),
            _ => None,
        });
        assert!(
            matches!(trimmed, Some(n) if n > 0),
            "eviction must emit ChatContextTrimmed with a positive dropped_turns"
        );
    }

    // ----- S1: the inter-agent bridge applies the MCP write gate -------------

    /// The bridge's MCP-gated descriptor set never offers the destructive ops,
    /// for EITHER `mcp_write_tools` value. This is the FIRST line of the gate
    /// (the model never sees them), mirroring the direct MCP `tools/list`.
    #[test]
    fn bridge_gated_descriptors_never_offer_reprocess() {
        // The bridge drives an INTERNAL registry (`v1(false)`), exactly as the
        // production driver does.
        let registry = agent_tools::ToolRegistry::v1(false);
        for allow_writes in [false, true] {
            let names: Vec<&str> = registry
                .mcp_tool_descriptors_gated(allow_writes)
                .into_iter()
                .map(|d| d.name)
                .collect();
            assert!(
                !names.contains(&"reprocess_meeting"),
                "reprocess_meeting must never be offered over the bridge (allow_writes={allow_writes})"
            );
            // A read is still offered (the bridge is otherwise functional).
            assert!(names.contains(&"get_transcript"));
        }
    }

    /// End-to-end through the SAME `run_chat_turn` loop the bridge runs: even
    /// when the (stub) model REQUESTS `reprocess_meeting` by name, the gated
    /// dispatch (the production `commands::mcp_gate_check` → `mcp_call_allowed`
    /// path) rejects it WITHOUT dispatching — for either `mcp_write_tools` value
    /// (S1). This is the defence-in-depth second line of the gate.
    #[test]
    fn bridge_turn_cannot_dispatch_destructive_tools_even_when_model_requests_them() {
        // The destructive op the direct MCP path keeps unreachable, asked for
        // one per turn, under both `mcp_write_tools` values.
        for blocked in ["reprocess_meeting"] {
            for allow_writes in [false, true] {
                let registry = agent_tools::ToolRegistry::v1(false);
                // Step 1: the model asks for the destructive op. The gate rejects
                // it WITHOUT dispatching; the loop feeds the rejection back as a
                // tool-result error. Step 2: the model answers in free text.
                let engine = ScriptedEngine::new(vec![
                    (
                        vec![],
                        TurnOutcome::ToolCalls(vec![tool_call(
                            blocked,
                            "{\"meeting_id\":\"00000000-0000-4000-8000-000000000001\"}",
                        )]),
                    ),
                    (
                        vec!["no".into()],
                        TurnOutcome::Final("cannot do that".to_string()),
                    ),
                ]);
                let mut history = base_history();
                let (events, emit) = collector();
                let sid = ChatSessionId::new();

                // Record every tool name that reaches REAL dispatch. The dispatch
                // closure is built EXACTLY as `run_chat_turn_on_held_model` builds
                // it on the bridge path: `mcp_gate_check(registry,
                // Some(allow_writes), name)` first, then (only if allowed) the
                // real dispatch — here a recording stub stands in for
                // `registry.dispatch`, which the gate must prevent reaching for a
                // blocked tool.
                let dispatched = std::cell::RefCell::new(Vec::<String>::new());
                let dispatch = |call: &EngineToolCall| -> AppResult<ToolOutput> {
                    crate::commands::mcp_gate_check(&registry, Some(allow_writes), &call.name)?;
                    dispatched.borrow_mut().push(call.name.clone());
                    Ok(ToolOutput::new(serde_json::json!({}), "ok"))
                };

                // The gated descriptor set offered to the model never lists the
                // blocked op in the first place (the first line of the gate).
                let descriptors = registry.mcp_tool_descriptors_gated(allow_writes);
                assert!(
                    !descriptors.iter().any(|d| d.name == blocked),
                    "{blocked} must not be offered over the bridge (allow_writes={allow_writes})"
                );

                run_chat_turn(
                    &engine,
                    sid,
                    1,
                    &mut history,
                    &descriptors,
                    &SamplerConfig::deterministic(),
                    CHAT_N_CTX,
                    &CancelFlag::new(),
                    dispatch,
                    emit,
                )
                .expect("the turn recovers after the gate rejects the blocked tool");

                // THE load-bearing assertion: the destructive op was NEVER
                // dispatched, for either gate value.
                assert!(
                    dispatched.borrow().is_empty(),
                    "{blocked} must never be dispatched over the bridge (allow_writes={allow_writes}); got {:?}",
                    dispatched.borrow()
                );
                // The model saw a FAILED tool result for the blocked op (fed back,
                // not executed).
                let ev = events.borrow();
                assert!(
                    ev.iter().any(|e| matches!(
                        e,
                        AppEvent::ChatToolResult { ok: false, tool, .. } if tool == blocked
                    )),
                    "the blocked tool must surface a failed ChatToolResult (allow_writes={allow_writes})"
                );
            }
        }
    }

    /// Symmetry check: the gate does NOT block an MCP-allowed read on the bridge
    /// path (so the fix is a write-surface bound, not a functionality regression).
    #[test]
    fn bridge_turn_still_dispatches_an_allowed_read() {
        let registry = agent_tools::ToolRegistry::v1(false);
        let engine = ScriptedEngine::new(vec![
            (
                vec![],
                TurnOutcome::ToolCalls(vec![tool_call("get_transcript", "{}")]),
            ),
            (vec!["ok".into()], TurnOutcome::Final("done".to_string())),
        ]);
        let mut history = base_history();
        let (_events, emit) = collector();

        let dispatched = std::cell::RefCell::new(Vec::<String>::new());
        let dispatch = |call: &EngineToolCall| -> AppResult<ToolOutput> {
            crate::commands::mcp_gate_check(&registry, Some(false), &call.name)?;
            dispatched.borrow_mut().push(call.name.clone());
            Ok(ToolOutput::new(serde_json::json!({}), "ok"))
        };
        let descriptors = registry.mcp_tool_descriptors_gated(false);
        run_chat_turn(
            &engine,
            ChatSessionId::new(),
            1,
            &mut history,
            &descriptors,
            &SamplerConfig::deterministic(),
            CHAT_N_CTX,
            &CancelFlag::new(),
            dispatch,
            emit,
        )
        .expect("an allowed read must dispatch and the turn must complete");
        assert_eq!(dispatched.borrow().as_slice(), ["get_transcript"]);
    }
}
