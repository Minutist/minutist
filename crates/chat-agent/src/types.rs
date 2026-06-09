//! The engine's message/history + outcome value types.
//!
//! These are the `chat-agent`-owned shapes the engine consumes and produces.
//! They are deliberately NOT in `common`: no `common`-level signature names
//! them, and only `chat-agent` (the engine) + `ipc-bridge` (the driver, a later
//! phase) use them. The persisted/wire chat types (`common::ChatTurn`,
//! `common::ChatRole`, the chat `AppEvent`s) live in `common`; the driver maps
//! between the two at its boundary. Keeping the engine types here keeps
//! `chat-agent` off the `common` precursor.

use serde::{Deserialize, Serialize};

/// The role of one message in the conversation history.
///
/// Maps 1:1 onto the OpenAI chat-message `role` field (`serde` snake_case), so
/// [`crate::backend::messages_json`] can serialise a `&[ChatMessage]` straight
/// into the `messages_json` the GGUF's tool template renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The system prompt (turn 0): persona, meeting context, the tool list.
    System,
    /// A user message.
    User,
    /// A prior assistant turn (free text, or one that requested tool calls).
    Assistant,
    /// A tool result fed back after the driver ran a tool call.
    Tool,
}

/// One message in the conversation history.
///
/// The driver owns the `Vec<ChatMessage>` and the sliding window; the engine
/// only borrows a `&[ChatMessage]` per [`crate::ChatEngine::run_turn`] call and
/// never mutates it (stateless per call, §1.2).
///
/// `tool_call_id` / `name` carry the OpenAI tool-message linkage so a `Tool`
/// message can be tied back to the assistant tool call it answers; both are
/// omitted from the wire shape when absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// For a `Tool` message: the id of the assistant tool call it answers
    /// (echoes [`ToolCall::id`]). `None` for system/user/assistant messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For a `Tool` message: the tool name (helps the model attribute the
    /// result). `None` for system/user/assistant messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// A plain text message (system / user / assistant free text).
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            name: None,
        }
    }

    /// A system message (turn 0).
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    /// An assistant free-text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }

    /// A tool-result message answering the call `tool_call_id` from tool
    /// `name`. The driver builds this after running [`ToolCall`] through the
    /// `agent-tools` `ToolRegistry::dispatch`, then calls the engine again.
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}

/// One tool call the assistant requested in a turn.
///
/// `arguments_json` is the raw JSON-object string the model emitted (the repo's
/// "arguments cross as a String, not a `Value`" rule, mirroring
/// `common::ToolCall`). The driver passes `name` + the parsed `arguments_json`
/// to `agent_tools::ToolRegistry::dispatch`. `id` is the OpenAI tool-call id
/// (the engine synthesises one when the template omits it) so the driver can
/// tie the eventual [`ChatMessage::tool_result`] back to this call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

/// The result of running ONE assistant turn (§1.2).
///
/// Either the assistant produced a final answer (no tool call) and the driver
/// stops the loop, OR it requested one or more tool calls and the driver
/// executes them via `agent-tools`, appends a `Tool` message per call, and
/// calls the engine again. The loop + the max-iteration cap live in the driver,
/// NOT here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnOutcome {
    /// A final assistant reply; the driver ends the turn.
    Final(String),
    /// One or more tool calls to execute; the driver dispatches, appends tool
    /// results, and re-invokes the engine.
    ToolCalls(Vec<ToolCall>),
}

/// Sampling knobs for one turn (§1.4 / §6.4).
///
/// The driver builds this from `settings` (or a constant default) per call. The
/// default chat sampler is a `temp/top_p/dist(seed)` chain; **`temperature ==
/// 0.0` selects greedy**, the deterministic test mode. `grammar_backstop`
/// arms a lazy GBNF grammar over the selected-tool schemas as the reliability
/// backstop for the 4B model — wired but behind this flag.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    /// Sampling temperature. `0.0` ⇒ greedy (deterministic).
    pub temperature: f32,
    /// Nucleus-sampling cutoff (ignored in the greedy path).
    pub top_p: f32,
    /// RNG seed for the terminal `dist(seed)` (ignored in the greedy path), so
    /// a fixed seed makes a non-greedy turn reproducible.
    pub seed: u32,
    /// Hard cap on tokens generated in one turn.
    pub max_tokens: usize,
    /// Arm the lazy-GBNF grammar backstop (constrain tool-call JSON to the
    /// offered tools' schemas). Behind a flag — off by default; the native
    /// oaicompat template already drives tool calling, and the grammar is the
    /// reliability backstop, not the primary mechanism.
    pub grammar_backstop: bool,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            // A small positive temperature avoids greedy dialogue loops (§1.4)
            // while staying close to the prompt.
            temperature: 0.7,
            top_p: 0.95,
            seed: 0,
            max_tokens: 1_024,
            grammar_backstop: false,
        }
    }
}

impl SamplerConfig {
    /// The deterministic profile used by the default test suite + any
    /// reproducible run: greedy decode, fixed everything.
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// Whether this config selects the greedy (deterministic) decode path.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}
