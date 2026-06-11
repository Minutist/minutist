//! The dynamic rmcp [`ServerHandler`] (Phase 10 §2.2).
//!
//! The tool set is data-driven (the `agent-tools` registry), not statically
//! known, so the `#[tool_router]` macro's static path does not apply — this is a
//! hand-written `ServerHandler` with dynamic dispatch:
//!
//! - **`tools/list`** projects [`ToolRegistry::mcp_tool_descriptors_gated`]
//!   (which composes [`agent_tools::Tool::expose_over_mcp`] with the
//!   `mcp_write_tools` gate) onto rmcp [`Tool`]s, copying each tool's
//!   `input_schema` verbatim and setting `readOnlyHint` from `!is_write`.
//! - **`tools/call`** re-checks the gate ([`ToolRegistry::mcp_call_allowed`] —
//!   defence in depth so a client cannot call a write tool absent from its
//!   gated list) then runs [`ToolRegistry::dispatch`] — the SAME path the chat
//!   loop uses.
//!
//! `agent-tools` returns [`minutist_common::AppError`]; the
//! [`crate::error::app_error_to_mcp`] map turns it into an rmcp error here (the
//! one place rmcp error types are constructed).

use std::sync::Arc;

use agent_tools::{ToolContext, ToolRegistry};
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, Implementation, InitializeResult,
    ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::ErrorData as McpError;

use crate::error::app_error_to_mcp;

/// The MCP server handler: a projection of the shared [`ToolRegistry`] onto the
/// MCP `tools/*` methods. Cloned per connection by rmcp's session manager
/// (cheap — all `Arc` + a `bool`).
#[derive(Clone)]
pub struct McpToolHandler {
    /// The shared registry (`v1(true)` — includes `send_to_internal_agent`),
    /// built by `app-main` and shared with this handler.
    registry: Arc<ToolRegistry>,
    /// The tool context the dispatch runs against — the SAME handles the
    /// internal agent uses, carrying the inter-agent bridge sender.
    ctx: Arc<ToolContext>,
    /// `settings.mcp_write_tools` (D3): when `false`, write tools are omitted
    /// from `tools/list` and rejected on `tools/call`.
    allow_writes: bool,
}

impl McpToolHandler {
    /// Build the handler from the injected registry + context + write gate.
    pub fn new(registry: Arc<ToolRegistry>, ctx: Arc<ToolContext>, allow_writes: bool) -> Self {
        Self {
            registry,
            ctx,
            allow_writes,
        }
    }

    /// Project the gated registry descriptors onto rmcp [`Tool`]s. Pure (no I/O),
    /// so it is unit-testable directly.
    pub(crate) fn list_tools_projection(&self) -> Vec<Tool> {
        self.registry
            .mcp_tool_descriptors_gated(self.allow_writes)
            .into_iter()
            .map(|d| {
                // The agent-tools input_schema is a JSON object root; rmcp wants
                // an `Arc<serde_json::Map>`. A non-object schema is a tool bug —
                // fall back to an empty object so listing never panics.
                let schema = match d.input_schema {
                    serde_json::Value::Object(map) => map,
                    _ => serde_json::Map::new(),
                };
                let mut tool = Tool::new(d.name, d.description, Arc::new(schema));
                // readOnlyHint advertises whether the tool mutates state. It is a
                // HINT only (a headless client may ignore it); the binding
                // control is the server-side gate above. `is_write` is the
                // authoritative source.
                let is_write = self
                    .registry
                    .get(d.name)
                    .map(|t| t.is_write())
                    .unwrap_or(false);
                let annotations = rmcp::model::ToolAnnotations::new()
                    .read_only(!is_write)
                    // The two MCP-exposed writes (set_speaker_name /
                    // rename_meeting) are reversible metadata edits, not
                    // destructive; advertise that. Reads are non-destructive too.
                    .destructive(false)
                    // None of the exposed tools are idempotent in the strict
                    // sense (re-running set_speaker_name with a new name changes
                    // state), so leave idempotentHint at its default.
                    .open_world(false);
                tool = tool.with_annotations(annotations);
                tool
            })
            .collect()
    }
}

impl ServerHandler for McpToolHandler {
    fn get_info(&self) -> InitializeResult {
        let server_info = Implementation::new("minutist", env!("CARGO_PKG_VERSION"))
            .with_title("Minutist (local meeting-notes app)")
            .with_description(
                "Local meeting-notes app: read transcripts, summaries, notes and \
                 metadata, and message the internal chat agent.",
            );
        // The v1 tool set is fixed at construction; no live tools/listChanged.
        let capabilities = ServerCapabilities::builder().enable_tools().build();
        InitializeResult::new(capabilities)
            // Target the Streamable HTTP transport revision (2025-11-25).
            .with_protocol_version(ProtocolVersion::V_2025_11_25)
            .with_server_info(server_info)
            .with_instructions(
                "Use the tools to read this app's meetings (transcripts, summaries, notes, \
                 metadata, speaker talk-time) and, when enabled, set speaker names or rename \
                 meetings. `send_to_internal_agent` forwards a message to the app's own chat \
                 agent and returns its reply.",
            )
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        let tools = self.list_tools_projection();
        std::future::ready(Ok(ListToolsResult::with_all_items(tools)))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, McpError>> + Send + '_ {
        let registry = Arc::clone(&self.registry);
        let ctx = Arc::clone(&self.ctx);
        let allow_writes = self.allow_writes;
        async move {
            let name = request.name.to_string();

            // Defence in depth: a tool absent from the gated `tools/list` (a
            // write tool while `mcp_write_tools` is off, or any non-exposed
            // tool) must not be callable, even if a client guesses the name.
            if !registry.mcp_call_allowed(&name, allow_writes) {
                return Err(McpError::invalid_params(
                    format!("tool `{name}` is not exposed over MCP"),
                    None,
                ));
            }

            // rmcp gives arguments as an `Option<Map>`; the registry validates +
            // dispatches over a `serde_json::Value` object.
            let args = serde_json::Value::Object(request.arguments.unwrap_or_default());

            match registry.dispatch(&ctx, &name, args).await {
                Ok(output) => {
                    // Text content block: the one-line summary, or a compact JSON
                    // render of the data when there is no summary. ALSO attach the
                    // machine payload as structured content (`CallToolResult` is
                    // `#[non_exhaustive]`, so build via `success(..)` then set the
                    // public `structured_content` field).
                    let text = output
                        .summary
                        .clone()
                        .unwrap_or_else(|| output.data.to_string());
                    let mut result = CallToolResult::success(vec![Content::text(text)]);
                    result.structured_content = Some(output.data);
                    Ok(result)
                }
                Err(e) => Err(app_error_to_mcp(&e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A real `v1(true)` registry is used for the projection tests — the
    // projection only reads tool metadata (names / schemas / is_write /
    // expose_over_mcp), which needs no live backend. (Dispatch is exercised in
    // the agent-tools crate's own tests with stub backends.)

    /// The gated tool-name set for a registry. The handler's `tools/list`
    /// projection is exactly `mcp_tool_descriptors_gated`, so testing the
    /// registry projection directly (without a full `ToolContext`, which the
    /// projection never touches) covers the write-gate logic 1:1.
    fn projection_names(allow_writes: bool, inter_agent: bool) -> Vec<String> {
        let registry = Arc::new(ToolRegistry::v1(inter_agent));
        registry
            .mcp_tool_descriptors_gated(allow_writes)
            .into_iter()
            .map(|d| d.name.to_string())
            .collect()
    }

    #[test]
    fn write_gate_off_omits_writes_keeps_reads_and_bridge() {
        let names = projection_names(false, true);
        // Reads present.
        assert!(names.contains(&"list_meetings".to_string()));
        assert!(names.contains(&"get_transcript".to_string()));
        assert!(names.contains(&"speaker_talk_time".to_string()));
        // The inter-agent tool is present (it is not a write).
        assert!(names.contains(&"send_to_internal_agent".to_string()));
        // Reversible writes ABSENT when the gate is off.
        assert!(!names.contains(&"set_speaker_name".to_string()));
        assert!(!names.contains(&"rename_meeting".to_string()));
        // Heavy/destructive writes never exposed.
        assert!(!names.contains(&"retranscribe_meeting".to_string()));
        assert!(!names.contains(&"rediarize_meeting".to_string()));
    }

    #[test]
    fn write_gate_on_adds_reversible_writes_only() {
        let names = projection_names(true, true);
        // Reversible writes now present.
        assert!(names.contains(&"set_speaker_name".to_string()));
        assert!(names.contains(&"rename_meeting".to_string()));
        // Heavy ops STILL never exposed, regardless of the gate.
        assert!(!names.contains(&"retranscribe_meeting".to_string()));
        assert!(!names.contains(&"rediarize_meeting".to_string()));
    }

    #[test]
    fn inter_agent_tool_absent_without_v1_true() {
        let names = projection_names(true, false);
        assert!(
            !names.contains(&"send_to_internal_agent".to_string()),
            "the inter-agent tool must only appear in a v1(true) registry"
        );
    }

    #[test]
    fn projection_descriptors_match_registry_one_to_one() {
        // The MCP tools/list set is exactly mcp_tool_descriptors_gated — same
        // names, same count.
        let registry = ToolRegistry::v1(true);
        let gated = registry.mcp_tool_descriptors_gated(true);
        let names: Vec<&str> = gated.iter().map(|d| d.name).collect();
        // Each descriptor schema is an object root (so the rmcp projection keeps
        // it verbatim, never falling back to empty).
        for d in &gated {
            assert!(
                d.input_schema.is_object(),
                "tool {} schema must be an object root",
                d.name
            );
        }
        // No duplicates.
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "tool names must be unique");
    }

    #[test]
    fn mcp_call_allowed_matches_listing() {
        let registry = ToolRegistry::v1(true);
        // With writes off: a read is callable, a write is not, an unknown is not.
        assert!(registry.mcp_call_allowed("get_transcript", false));
        assert!(!registry.mcp_call_allowed("set_speaker_name", false));
        assert!(!registry.mcp_call_allowed("retranscribe_meeting", false));
        assert!(!registry.mcp_call_allowed("no_such_tool", false));
        assert!(registry.mcp_call_allowed("send_to_internal_agent", false));
        // With writes on: the reversible write becomes callable; the heavy one
        // never does.
        assert!(registry.mcp_call_allowed("set_speaker_name", true));
        assert!(registry.mcp_call_allowed("rename_meeting", true));
        assert!(!registry.mcp_call_allowed("retranscribe_meeting", true));
        assert!(!registry.mcp_call_allowed("rediarize_meeting", true));
    }
}
