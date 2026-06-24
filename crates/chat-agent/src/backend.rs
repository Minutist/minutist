//! The testable LLM seam.
//!
//! [`TurnBackend`] is the one place the engine touches the model. The real impl
//! ([`crate::llama::LlamaTurnBackend`]) drives the `llama-cpp-2` oaicompat APIs
//! over a fresh `LlamaContext`, the sampler chain, and the streaming oaicompat
//! parse. A test stub returns canned text/tool-calls, so the engine's turn
//! logic (prompt assembly, outcome parsing, tool-call extraction, error
//! mapping) is unit-tested in the default suite WITHOUT an FFI model.

use agent_tools::ToolDescriptor;
use serde_json::json;

use crate::error::Error;
use crate::types::{CancelFlag, SamplerConfig, ToolCall};

/// What ONE backend run produced, before the engine maps it to a
/// [`crate::TurnOutcome`].
///
/// `tool_calls` is the parsed set of calls the assistant requested (empty when
/// it produced a free-text answer). `text` is the accumulated assistant content
/// (the final answer when `tool_calls` is empty; any leading prose otherwise).
/// The engine decides the outcome: `cancelled` ⇒ `Cancelled`; else non-empty
/// `tool_calls` ⇒ `ToolCalls`; else non-empty `text` ⇒ `Final`; else malformed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RawTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    /// The decode loop observed a raised [`CancelFlag`] between tokens and
    /// stopped early (P1). `text` then carries the partial content so far.
    pub cancelled: bool,
}

/// The LLM seam: render a prompt from OpenAI-format messages + tools, generate
/// with streaming, parse the output into a [`RawTurn`].
///
/// The engine owns prompt assembly (it converts `&[ChatMessage]` →
/// `messages_json` and `&[ToolDescriptor]` → `tools_json`) and outcome mapping;
/// the backend owns ONLY the model interaction. This keeps the FFI-bound work
/// behind one trait so everything above it is unit-testable with a stub.
///
/// Threading: `run` is synchronous (the model is FFI-bound, like
/// `summariser::generate`); the driver calls it inside `spawn_blocking`. It is
/// `Send + Sync` so the engine (and the driver's `Arc<dyn ChatEngine>`) is
/// `Send + Sync`.
pub trait TurnBackend: Send + Sync {
    /// Run one assistant turn.
    ///
    /// - `messages_json` — the conversation history as an OpenAI-compatible
    ///   messages JSON array (built by [`messages_json`]).
    /// - `tools_json` — the offered tools as an OpenAI-compatible tools JSON
    ///   array, or `None` when the turn forces a final answer (the driver's
    ///   max-iteration escape re-invokes with no tools).
    /// - `cfg` — the per-turn sampler config.
    /// - `cancel` — the per-turn cancellation signal (P1). The decode loop
    ///   checks [`CancelFlag::is_cancelled`] BETWEEN tokens and, when raised,
    ///   stops and returns a [`RawTurn`] with `cancelled: true` carrying the
    ///   partial text.
    /// - `token_cb` — called with each detokenised user-visible piece as it is
    ///   produced (the driver turns these into `ChatToken` `AppEvent`s). The
    ///   backend MUST NOT stream tool-call JSON through this callback.
    fn run(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        cfg: &SamplerConfig,
        cancel: &CancelFlag,
        token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, Error>;
}

/// Serialise the engine's history into an OpenAI-compatible `messages` JSON
/// array string for the oaicompat template.
///
/// Each [`crate::ChatMessage`] maps to `{role, content[, tool_call_id][,
/// name][, tool_calls]}`. A `Tool` message carries `tool_call_id` + `name` so
/// the template can attribute the result to the call it answers. An `Assistant`
/// message that requested tools carries a `tool_calls` array
/// (`{id, type:"function", function:{name, arguments}}` per call) — REQUIRED to
/// precede the matching `tool` result messages, else the GGUF tool template
/// renders a malformed `[…, tool, …]` sequence (CQ1). The OpenAI shape pairs
/// the assistant message's `content: null` with a non-empty `tool_calls`; we
/// emit an explicit JSON `null` content for that case.
pub fn messages_json(history: &[crate::ChatMessage]) -> Result<String, Error> {
    let arr: Vec<serde_json::Value> = history
        .iter()
        .map(|m| {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "role".to_string(),
                json!(serde_json::to_value(m.role)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_default()),
            );
            // An assistant tool-call turn has `content: null` (OpenAI shape)
            // when it emitted no leading prose; otherwise the content string.
            if m.tool_calls.is_empty() || !m.content.is_empty() {
                obj.insert("content".to_string(), json!(m.content));
            } else {
                obj.insert("content".to_string(), serde_json::Value::Null);
            }
            if let Some(id) = &m.tool_call_id {
                obj.insert("tool_call_id".to_string(), json!(id));
            }
            if let Some(name) = &m.name {
                obj.insert("name".to_string(), json!(name));
            }
            if !m.tool_calls.is_empty() {
                let calls: Vec<serde_json::Value> = m
                    .tool_calls
                    .iter()
                    .map(|c| {
                        json!({
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                // OpenAI carries `arguments` as a JSON STRING.
                                "arguments": c.arguments_json,
                            }
                        })
                    })
                    .collect();
                obj.insert("tool_calls".to_string(), serde_json::Value::Array(calls));
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    serde_json::to_string(&arr)
        .map_err(|e| Error::Template(format!("serialise messages_json: {e}")))
}

/// Convert the `agent-tools` registry descriptors into an OpenAI-compatible
/// `tools` JSON array string for the oaicompat template (and the GBNF grammar).
///
/// OpenAI tool shape:
/// `{"type":"function","function":{"name","description","parameters":<schema>}}`
/// — `parameters` is the tool's `input_schema` verbatim (the schemas are
/// regex-`pattern`-free by the `agent-tools` contract, so they compile through
/// `json_schema_to_grammar`). Returns `None` for an empty descriptor list so
/// the caller passes `tools_json: None` (a tool-less turn) rather than an empty
/// `[]`.
pub fn tools_json(descriptors: &[ToolDescriptor]) -> Result<Option<String>, Error> {
    if descriptors.is_empty() {
        return Ok(None);
    }
    let arr: Vec<serde_json::Value> = descriptors
        .iter()
        .map(|d| {
            json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.input_schema,
                }
            })
        })
        .collect();
    serde_json::to_string(&arr)
        .map(Some)
        .map_err(|e| Error::Template(format!("serialise tools_json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Role, ToolCall};
    use serde_json::Value;

    fn descriptor(name: &'static str) -> ToolDescriptor {
        ToolDescriptor {
            name,
            title: "Test tool",
            description: "desc",
            input_schema: json!({
                "type": "object",
                "properties": { "meeting_id": { "type": "string" } },
                "required": ["meeting_id"],
                "additionalProperties": false,
            }),
        }
    }

    #[test]
    fn messages_json_emits_openai_roles_and_tool_linkage() {
        let history = vec![
            ChatMessage::system("you are helpful"),
            ChatMessage::user("what was said?"),
            ChatMessage::assistant("let me check"),
            ChatMessage::tool_result("call_1", "get_transcript", "{\"segments\":[]}"),
        ];
        let s = messages_json(&history).unwrap();
        let arr: Vec<Value> = serde_json::from_str(&s).unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0]["role"], json!("system"));
        assert_eq!(arr[1]["role"], json!("user"));
        assert_eq!(arr[2]["role"], json!("assistant"));
        assert_eq!(arr[3]["role"], json!("tool"));
        // Tool linkage round-trips; non-tool messages omit it.
        assert_eq!(arr[3]["tool_call_id"], json!("call_1"));
        assert_eq!(arr[3]["name"], json!("get_transcript"));
        assert!(arr[0].get("tool_call_id").is_none());
        assert!(arr[0].get("name").is_none());
        // A plain assistant message carries NO tool_calls key.
        assert!(arr[2].get("tool_calls").is_none());
    }

    #[test]
    fn messages_json_renders_assistant_tool_calls_before_tool_result() {
        // CQ1: the assistant-tool_calls message must precede its tool result and
        // serialise the OpenAI `tool_calls` array, so the rendered sequence is a
        // valid `assistant(tool_calls) → tool(result)` exchange.
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "get_transcript".to_string(),
            arguments_json: "{\"meeting_id\":\"m1\"}".to_string(),
        };
        let history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("what was said?"),
            ChatMessage::assistant_tool_calls("", vec![call]),
            ChatMessage::tool_result("call_1", "get_transcript", "{\"segments\":[]}"),
        ];
        let s = messages_json(&history).unwrap();
        let arr: Vec<Value> = serde_json::from_str(&s).unwrap();
        assert_eq!(arr.len(), 4);
        // The assistant message (index 2) carries the OpenAI tool_calls array and
        // a null content (no leading prose).
        assert_eq!(arr[2]["role"], json!("assistant"));
        assert_eq!(arr[2]["content"], Value::Null);
        let calls = arr[2]["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], json!("call_1"));
        assert_eq!(calls[0]["type"], json!("function"));
        assert_eq!(calls[0]["function"]["name"], json!("get_transcript"));
        // `arguments` is the raw JSON string (OpenAI shape).
        assert_eq!(
            calls[0]["function"]["arguments"],
            json!("{\"meeting_id\":\"m1\"}")
        );
        // The matching tool result follows, attributed to the same call id.
        assert_eq!(arr[3]["role"], json!("tool"));
        assert_eq!(arr[3]["tool_call_id"], json!("call_1"));
    }

    #[test]
    fn role_serialises_snake_case() {
        assert_eq!(
            serde_json::to_value(Role::Assistant).unwrap(),
            json!("assistant")
        );
        assert_eq!(serde_json::to_value(Role::Tool).unwrap(), json!("tool"));
    }

    #[test]
    fn tools_json_wraps_descriptors_in_openai_function_shape() {
        let descs = vec![descriptor("get_transcript"), descriptor("get_summary")];
        let s = tools_json(&descs)
            .unwrap()
            .expect("non-empty descriptors yield Some");
        let arr: Vec<Value> = serde_json::from_str(&s).unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], json!("function"));
        assert_eq!(arr[0]["function"]["name"], json!("get_transcript"));
        assert_eq!(arr[0]["function"]["description"], json!("desc"));
        // The input_schema is forwarded verbatim as `parameters`.
        assert_eq!(arr[0]["function"]["parameters"]["type"], json!("object"));
        assert_eq!(
            arr[0]["function"]["parameters"]["required"],
            json!(["meeting_id"])
        );
        assert_eq!(arr[1]["function"]["name"], json!("get_summary"));
    }

    #[test]
    fn tools_json_empty_descriptors_is_none() {
        assert_eq!(tools_json(&[]).unwrap(), None);
    }
}
