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
use crate::types::{SamplerConfig, ToolCall};

/// What ONE backend run produced, before the engine maps it to a
/// [`crate::TurnOutcome`].
///
/// `tool_calls` is the parsed set of calls the assistant requested (empty when
/// it produced a free-text answer). `text` is the accumulated assistant content
/// (the final answer when `tool_calls` is empty; any leading prose otherwise).
/// The engine decides the outcome: non-empty `tool_calls` ⇒ `ToolCalls`; else
/// non-empty `text` ⇒ `Final`; else malformed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RawTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
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
    /// - `token_cb` — called with each detokenised user-visible piece as it is
    ///   produced (the driver turns these into `ChatToken` `AppEvent`s). The
    ///   backend MUST NOT stream tool-call JSON through this callback.
    fn run(
        &self,
        messages_json: &str,
        tools_json: Option<&str>,
        cfg: &SamplerConfig,
        token_cb: &mut dyn FnMut(&str),
    ) -> Result<RawTurn, Error>;
}

/// Serialise the engine's history into an OpenAI-compatible `messages` JSON
/// array string for the oaicompat template.
///
/// Each [`crate::ChatMessage`] maps to `{role, content[, tool_call_id][,
/// name]}`. A `Tool` message carries `tool_call_id` + `name` so the template
/// can attribute the result to the call it answers.
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
            obj.insert("content".to_string(), json!(m.content));
            if let Some(id) = &m.tool_call_id {
                obj.insert("tool_call_id".to_string(), json!(id));
            }
            if let Some(name) = &m.name {
                obj.insert("name".to_string(), json!(name));
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
    use crate::types::{ChatMessage, Role};
    use serde_json::Value;

    fn descriptor(name: &'static str) -> ToolDescriptor {
        ToolDescriptor {
            name,
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
    }

    #[test]
    fn role_serialises_snake_case() {
        assert_eq!(serde_json::to_value(Role::Assistant).unwrap(), json!("assistant"));
        assert_eq!(serde_json::to_value(Role::Tool).unwrap(), json!("tool"));
    }

    #[test]
    fn tools_json_wraps_descriptors_in_openai_function_shape() {
        let descs = vec![descriptor("get_transcript"), descriptor("get_summary")];
        let s = tools_json(&descs).unwrap().expect("non-empty descriptors yield Some");
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
