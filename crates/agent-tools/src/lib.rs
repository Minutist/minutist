//! `agent-tools` — the shared tool layer (Phase 9).
//!
//! One [`Tool`] trait + one [`ToolRegistry`]: the single place a tool is
//! defined. Both consumers drive the same registry — the internal chat agent
//! (Phase 9) and the MCP server (Phase 10) — so the hard constraint "the
//! internal agent and an external MCP client use the same tools" is satisfied
//! by there being exactly one definition site per tool.
//!
//! # Boundaries (binding — see `architecture/components.md`)
//!
//! Edges: `common`, `persistence`, `orchestrator`. Deliberately **no**
//! `summariser` edge — the one LLM-using tool (`resummarise`) drives an
//! `Arc<dyn minutist_common::Summariser>` held in [`ToolContext`],
//! constructed by `ipc-bridge`/`app-main` (which own the `summariser` edge).
//! Deliberately **no** `model-registry` edge — `relisten_section` resolves and
//! builds its ASR backend through [`orchestrator::Orchestrator::transcribe_pcm_window`],
//! never by calling `model-registry`.
//!
//! `agent-tools` has no `tauri`/`specta` concern: `serde_json::Value` results
//! cross the IPC boundary as a `String` in `ipc-bridge`'s event envelope, not
//! here. The `AppError → McpError` mapping is Phase 10's concern and lives in
//! `mcp-server` (keeps `rmcp` out of this crate's deps).
//!
//! # Threading
//!
//! [`Tool::execute`] is `async` because the backing operations are async
//! (`Orchestrator::reprocess`/`transcribe_pcm_window` are
//! `async fn`; `MeetingIndex::list_meetings`/`search` are async libsql). Tool
//! *bodies* still push CPU/fs/inference work onto `tokio::task::spawn_blocking`
//! — the trait being async does not relax the cross-cutting rule.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use minutist_common::{
    AppError, AppEvent, AppResult, InterAgentReply, InterAgentRequest, MeetingId, Summariser,
};
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use tokio::sync::{broadcast, mpsc, oneshot};

mod tools;

pub use tools::*;

// ---------------------------------------------------------------------------
// Inter-agent bridge (Phase 10) — the channel ends only; the chat-engine
// driver lives in `ipc-bridge`.
// ---------------------------------------------------------------------------

/// One inter-agent message: an [`InterAgentRequest`] plus the oneshot the driver
/// replies on. The bridge tool `send_to_internal_agent` builds this and `try_send`s
/// it onto the [`InterAgentBridge`] channel; the `ipc-bridge` driver task owns the
/// receiver, runs ONE chat turn on the held model via the SAME `run_chat_turn`
/// driver the UI uses, and sends the [`InterAgentReply`] back on the oneshot.
///
/// Keeping the *channel ends* (not a chat-engine handle) on the
/// [`ToolContext`] is what lets `agent-tools` stay free of a `chat-agent` edge:
/// the tool only knows `common` types + tokio channels (both already allowed
/// here), and the single place a chat turn runs stays in `ipc-bridge`.
pub type InterAgentEnvelope = (
    InterAgentRequest,
    oneshot::Sender<AppResult<InterAgentReply>>,
);

/// The bounded sender end the bridge tool holds (cross-cutting forbids unbounded
/// channels). A full queue → `InvalidInput { "internal agent busy" }` rather than
/// blocking the HTTP request thread.
pub type InterAgentBridge = mpsc::Sender<InterAgentEnvelope>;

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

/// One capability the agent (or an MCP client) can invoke.
///
/// Every tool is one `impl Tool`, registered once in [`ToolRegistry::v1`]; it
/// then appears in the LLM tool list, the GBNF grammar, and (Phase 10) the MCP
/// `tools/list` with no further edits.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Stable snake_case wire name. **Never change once shipped.**
    fn name(&self) -> &'static str;

    /// Short human-readable label for directory listings and tool-picker UIs.
    ///
    /// Required on every tool — a missing impl is a compile error. The title
    /// must be distinct from the snake_case `name()` (human-facing, not
    /// machine-facing). It is projected onto the MCP `tools/list` `title`
    /// field via [`Tool::with_title`] in `mcp-server`.
    fn title(&self) -> &'static str;

    /// One-line description: the MCP description + injected into the LLM tool
    /// list.
    fn description(&self) -> &'static str;

    /// JSON Schema (2020-12, object root) describing the tool's arguments.
    ///
    /// **Must avoid regex `pattern`** (plain types/enums/strings only): the
    /// vendored llama.cpp schema→GBNF converter fails on PCRE shorthands.
    fn input_schema(&self) -> serde_json::Value;

    /// Whether the tool mutates disk / index / state. Drives the
    /// write-serialization guard and the default MCP exposure.
    fn is_write(&self) -> bool {
        false
    }

    /// Whether the tool is exposed over MCP (Phase 10). Default-safe: reads and
    /// compute are exposed, writes are not (opt-in per tool via the allowlist
    /// in [`ToolRegistry::v1`]). See `architecture/cross-cutting.md`
    /// "Agent chat loop + tool layer".
    fn expose_over_mcp(&self) -> bool {
        !self.is_write()
    }

    /// Run the tool. `args` has already been validated against
    /// [`Tool::input_schema`] by [`ToolRegistry::dispatch`].
    async fn execute(&self, ctx: &ToolContext, args: serde_json::Value) -> AppResult<ToolOutput>;
}

// ---------------------------------------------------------------------------
// ToolContext + ToolOutput
// ---------------------------------------------------------------------------

/// The handles a tool needs, constructed once by `ipc-bridge`/`app-main` and
/// cloned per dispatch. `Clone` is cheap (all `Arc` / `PathBuf` /
/// `broadcast::Sender`).
#[derive(Clone)]
pub struct ToolContext {
    /// The recording orchestrator. Backs `relisten_section`, `reprocess_meeting`,
    /// and `get_recording_state`. Owns the `model-registry` edge so `agent-tools`
    /// need not.
    pub orchestrator: Arc<Orchestrator>,
    /// The libsql meeting index. Backs `list_meetings` + `search_meetings`, and
    /// is passed to the orchestrator's offline ops which refresh its rows.
    pub index: Arc<MeetingIndex>,
    /// `{app-data}/meetings/` — the per-meeting folder root. Tools derive
    /// `{meetings_dir}/{meeting_id}/` to drive the `persistence` readers.
    pub meetings_dir: PathBuf,
    /// The held summariser substrate (`Send + Sync`; loaded once, shared by the
    /// one-shot summary path + the chat agent). Backs `resummarise`. Held as
    /// `Arc<dyn Summariser>` directly (SP0: the bundled impl is `Send + Sync`).
    pub summariser: Arc<dyn Summariser>,
    /// The shared `AppEvent` broadcast bus. Currently UNREAD by any tool — the
    /// offline write op (`reprocess`) emits
    /// `TranscriptReady`/`DiarizationComplete` via the orchestrator's own bus,
    /// and the chat driver (Phase 9 ipc-bridge) emits the chat events itself.
    /// Held on the context so a future tool that needs to emit progress can,
    /// without a signature change. Kept symmetric with the orchestrator's bus.
    pub event_tx: broadcast::Sender<AppEvent>,
    /// The internal-UI default meeting, when the chat session is meeting-scoped.
    /// MCP leaves this `None`, so an MCP caller must pass `meeting_id`
    /// explicitly. A tool resolves the effective meeting via
    /// [`ToolContext::resolve_meeting`].
    pub default_meeting: Option<MeetingId>,
    /// The inter-agent bridge sender (Phase 10), present ONLY for the MCP
    /// registry instance (`v1(true)`). `send_to_internal_agent` reads it here;
    /// `None` for the internal chat agent's context (it must not message
    /// itself). Held here so the bridge tool needs no `chat-agent` edge — the
    /// receiver + the chat turn live in `ipc-bridge`.
    inter_agent: Option<InterAgentBridge>,
}

impl ToolContext {
    /// Construct a context.
    pub fn new(
        orchestrator: Arc<Orchestrator>,
        index: Arc<MeetingIndex>,
        meetings_dir: PathBuf,
        summariser: Arc<dyn Summariser>,
        event_tx: broadcast::Sender<AppEvent>,
        default_meeting: Option<MeetingId>,
    ) -> Self {
        Self {
            orchestrator,
            index,
            meetings_dir,
            summariser,
            event_tx,
            default_meeting,
            inter_agent: None,
        }
    }

    /// Attach the Phase-10 inter-agent bridge sender (the MCP-registry context
    /// only). The internal chat agent's context leaves this unset, so the
    /// internal agent cannot message itself (a trivial infinite-loop foot-gun).
    pub fn with_inter_agent_bridge(mut self, bridge: InterAgentBridge) -> Self {
        self.inter_agent = Some(bridge);
        self
    }

    /// The inter-agent bridge sender, when this context was built for the MCP
    /// registry. `send_to_internal_agent` reads it; `None` raises
    /// `InvalidInput` (the tool is unavailable in this scope).
    pub(crate) fn inter_agent_bridge(&self) -> Option<&InterAgentBridge> {
        self.inter_agent.as_ref()
    }

    /// The `{meetings_dir}/{meeting_id}/` folder path for `id`.
    pub(crate) fn meeting_dir(&self, id: MeetingId) -> PathBuf {
        self.meetings_dir.join(id.0.to_string())
    }
}

/// A tool's structured result.
///
/// `data` is the machine payload fed back to the LLM (as a tool result) and to
/// MCP `tools/call` structured content. `summary` is the optional one-line
/// human/LLM-facing render the UI tool-card uses (and the `ChatToolResult`
/// event carries).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolOutput {
    pub data: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl ToolOutput {
    /// Build an output with both a machine payload and a one-line summary.
    pub fn new(data: serde_json::Value, summary: impl Into<String>) -> Self {
        Self {
            data,
            summary: Some(summary.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// MCP descriptor (projection; consumed by Phase 10 tools/list)
// ---------------------------------------------------------------------------

/// A name/description/schema projection of one tool, for the LLM tool list and
/// (Phase 10) the MCP `tools/list`. Pure projection — single source of truth is
/// the [`Tool`] impl.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolDescriptor {
    pub name: &'static str,
    /// Human-readable label for directory listings (see [`Tool::title`]).
    pub title: &'static str,
    pub description: &'static str,
    pub input_schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// The single dispatch surface both consumers drive.
pub struct ToolRegistry {
    /// Insertion-ordered list of tools (stable order for the LLM tool list).
    tools: Vec<Arc<dyn Tool>>,
    /// Name → index into `tools`, for O(1) lookup in [`Self::dispatch`].
    by_name: BTreeMap<&'static str, usize>,
}

impl ToolRegistry {
    /// Build the v1 registry.
    ///
    /// `include_inter_agent_bridge` is the Phase-10 `send_to_internal_agent`
    /// flag — `ipc-bridge` passes `false` (the internal agent must not be able
    /// to message itself); `app-main` passes `true` for the registry instance
    /// handed to `mcp-server`. The tool's *body* reaches the chat engine through
    /// the bridge channel on [`ToolContext`] (see
    /// [`ToolContext::with_inter_agent_bridge`]), so this crate keeps no
    /// `chat-agent` edge.
    pub fn v1(include_inter_agent_bridge: bool) -> Self {
        let mut tools: Vec<Arc<dyn Tool>> = vec![
            // Read / compute.
            Arc::new(tools::ListMeetings),
            Arc::new(tools::SearchMeetings),
            Arc::new(tools::GetMeeting),
            Arc::new(tools::GetTranscript),
            Arc::new(tools::GetTranscriptSlice),
            Arc::new(tools::GetSummary),
            Arc::new(tools::GetNotes),
            Arc::new(tools::GetMetadata),
            Arc::new(tools::GetRecordingState),
            Arc::new(tools::SearchWithinTranscript),
            Arc::new(tools::RelistenSection),
            Arc::new(tools::Resummarise),
            Arc::new(tools::SpeakerTalkTime),
            Arc::new(tools::ListAttachments),
            Arc::new(tools::GetAttachmentMarkdown),
            // Writes (MCP exposure per the allowlist on each impl).
            Arc::new(tools::SetSpeakerName),
            Arc::new(tools::RenameMeeting),
            Arc::new(tools::ReprocessMeeting),
            // Record-control writes (#62). All `is_write` with the default
            // `expose_over_mcp` (`!is_write` → `false`), so they are write-gated
            // OFF over MCP regardless of `mcp_write_tools` — the recording
            // lifecycle is never driven over MCP in v1. The internal UI chat
            // (no MCP gate) can dispatch them.
            Arc::new(tools::StartRecording),
            Arc::new(tools::StopRecording),
            Arc::new(tools::PauseRecording),
            Arc::new(tools::ResumeRecording),
        ];

        if include_inter_agent_bridge {
            // Phase 10: the MCP-only inter-agent tool. Appended last so the
            // read/write ordering above is unchanged.
            tools.push(Arc::new(tools::SendToInternalAgent));
        }

        let by_name = tools
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name(), i))
            .collect();

        Self { tools, by_name }
    }

    /// All tool descriptors (name/description/schema), insertion order. For the
    /// LLM system prompt + GBNF grammar.
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        self.tools
            .iter()
            .map(|t| descriptor_of(t.as_ref()))
            .collect()
    }

    /// The descriptors a Phase-10 MCP `tools/list` exposes (honours
    /// [`Tool::expose_over_mcp`] — reads exposed, writes per the allowlist).
    pub fn mcp_tool_descriptors(&self) -> Vec<ToolDescriptor> {
        self.tools
            .iter()
            .filter(|t| t.expose_over_mcp())
            .map(|t| descriptor_of(t.as_ref()))
            .collect()
    }

    /// The Phase-10 MCP `tools/list` projection, with the `mcp_write_tools`
    /// setting (D3) applied ON TOP of [`Tool::expose_over_mcp`].
    ///
    /// `allow_writes` is `settings.mcp_write_tools`. The two layers compose:
    /// - a tool is a candidate only if `expose_over_mcp()` is `true` (reads +
    ///   compute + the two reversible writes + the inter-agent tool; never
    ///   `reprocess`/any destructive tool — those return `false`
    ///   on the trait and so are unreachable regardless of `allow_writes`);
    /// - when `allow_writes` is `false`, write tools are additionally dropped,
    ///   so the surface is read/compute + the inter-agent tool only.
    ///
    /// The single source of truth for what a tool *is* (its `is_write` /
    /// `expose_over_mcp`) stays on the [`Tool`] impl; this method only composes
    /// the user-toggleable gate with it, so the policy lives in one place.
    pub fn mcp_tool_descriptors_gated(&self, allow_writes: bool) -> Vec<ToolDescriptor> {
        self.tools
            .iter()
            .filter(|t| t.expose_over_mcp())
            .filter(|t| allow_writes || !t.is_write())
            .map(|t| descriptor_of(t.as_ref()))
            .collect()
    }

    /// Whether a tool of the given wire name may be *called* over MCP under the
    /// `allow_writes` gate. The server consults this in `tools/call` so a client
    /// cannot invoke a write tool that is absent from its gated `tools/list`
    /// (defence in depth — the gate is enforced on call, not only on listing).
    /// Returns `false` for an unknown name, a non-`expose_over_mcp` tool, or a
    /// write tool when `allow_writes` is `false`.
    pub fn mcp_call_allowed(&self, name: &str, allow_writes: bool) -> bool {
        match self.get(name) {
            Some(tool) => tool.expose_over_mcp() && (allow_writes || !tool.is_write()),
            None => false,
        }
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty (it never is for `v1`).
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Look up a tool by its wire name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.by_name.get(name).map(|&i| &self.tools[i])
    }

    /// THE single dispatch path for both consumers.
    ///
    /// Looks up the tool (`InvalidInput` on unknown name), validates `args`
    /// shape against the tool's schema (`InvalidInput` on mismatch), then calls
    /// `execute`. The per-tool write serialization (the metadata mutex) lives in
    /// the write tools themselves (§4.2), which hold a [`ToolContext`]-owned
    /// per-meeting lock; the offline-claim write ops inherit the orchestrator's
    /// claim for free.
    pub async fn dispatch(
        &self,
        ctx: &ToolContext,
        name: &str,
        args: serde_json::Value,
    ) -> AppResult<ToolOutput> {
        let tool = self.get(name).ok_or_else(|| AppError::InvalidInput {
            context: format!("unknown tool: {name}"),
        })?;

        validate_args(name, &tool.input_schema(), &args)?;

        tracing::debug!(target: "agent-tools", tool = name, is_write = tool.is_write(), "dispatching tool");
        tool.execute(ctx, args).await
    }
}

/// Project a `&dyn Tool` to a [`ToolDescriptor`].
fn descriptor_of(t: &dyn Tool) -> ToolDescriptor {
    ToolDescriptor {
        name: t.name(),
        title: t.title(),
        description: t.description(),
        input_schema: t.input_schema(),
    }
}

/// Drop `"meeting_id"` from each descriptor's JSON-schema `required` array.
///
/// Used when the chat is meeting-scoped (`ToolContext::default_meeting` is set):
/// a tool resolves an omitted `meeting_id` from the context via
/// [`ToolContext::resolve_meeting`], so the MODEL must be free to omit it — left
/// marked `required`, a schema-respecting model asks the user for the field
/// instead of calling the tool (the reported "it asked for a meeting id" bug).
/// The `meeting_id` PROPERTY stays in `properties` (still documented + accepted);
/// only its requiredness is relaxed. This pairs with the system-prompt scope line
/// (`ipc-bridge::chat_system_prompt_for_meeting`): the prompt tells the model
/// which meeting it is in, this lets it actually omit the id.
pub fn relax_meeting_id_requirement(descriptors: &mut [ToolDescriptor]) {
    for d in descriptors.iter_mut() {
        if let Some(required) = d
            .input_schema
            .get_mut("required")
            .and_then(|r| r.as_array_mut())
        {
            required.retain(|f| f.as_str() != Some("meeting_id"));
        }
    }
}

/// Lightweight argument validation against the tool's JSON Schema.
///
/// This is intentionally a shallow structural check, not a full JSON-Schema
/// validator, because `agent-tools`'s allowed deps include no validator crate.
/// It enforces that `args` is an object and that every property the schema marks
/// `required` is present and non-null. The real guard is **per-tool typed
/// extraction** in each tool body (`require_str` / `require_u64` /
/// `resolve_meeting` / the typed deserialise), which turns a shape or type
/// mismatch into `InvalidInput`. The GBNF grammar backstop in `chat-agent` is a
/// reliability aid that is OFF by default in the shipped chat path, so it is NOT
/// relied on here — the typed extraction holds whether or not it is armed.
fn validate_args(
    name: &str,
    schema: &serde_json::Value,
    args: &serde_json::Value,
) -> AppResult<()> {
    let obj = args.as_object().ok_or_else(|| AppError::InvalidInput {
        context: format!("tool {name}: arguments must be a JSON object"),
    })?;

    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for req in required {
            let Some(field) = req.as_str() else { continue };
            // `meeting_id` is the one required field that may be omitted at the
            // wire level: the internal-UI session can supply it from
            // `ToolContext::default_meeting` via `resolve_meeting`, which raises
            // a clear `InvalidInput` if neither is present. Validating it here
            // would make that fallback dead code. Every other required field is
            // enforced.
            if field == "meeting_id" {
                continue;
            }
            match obj.get(field) {
                Some(serde_json::Value::Null) | None => {
                    return Err(AppError::InvalidInput {
                        context: format!("tool {name}: missing required argument `{field}`"),
                    });
                }
                _ => {}
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared argument-extraction helpers (used by the tool bodies)
// ---------------------------------------------------------------------------

/// Parse a `MeetingId` (a hyphenated-UUID string, the `#[serde(transparent)]`
/// wire form) from a JSON string. `InvalidInput` on a malformed id.
pub(crate) fn parse_meeting_id(s: &str) -> AppResult<MeetingId> {
    serde_json::from_str::<MeetingId>(&format!("\"{s}\"")).map_err(|_| AppError::InvalidInput {
        context: format!("invalid meeting_id: {s}"),
    })
}

/// Resolve the effective meeting id for a tool call: the explicit `meeting_id`
/// argument when present, else the context's `default_meeting` (the internal-UI
/// session scope). `InvalidInput` when neither is available.
pub(crate) fn resolve_meeting(ctx: &ToolContext, args: &serde_json::Value) -> AppResult<MeetingId> {
    if let Some(v) = args.get("meeting_id") {
        let s = v.as_str().ok_or_else(|| AppError::InvalidInput {
            context: "meeting_id must be a string".into(),
        })?;
        return parse_meeting_id(s);
    }
    ctx.default_meeting.ok_or_else(|| AppError::InvalidInput {
        context: "no meeting_id given and no default meeting in scope".into(),
    })
}

/// Read a required string argument. `InvalidInput` when absent or not a string.
pub(crate) fn require_str<'a>(args: &'a serde_json::Value, field: &str) -> AppResult<&'a str> {
    args.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::InvalidInput {
            context: format!("missing or non-string argument `{field}`"),
        })
}

/// The dispatch-time length cap for user-supplied free-text identifiers written
/// to metadata (S4): speaker labels, display names, meeting titles. A name/title
/// is a short human label; a value past this is rejected rather than persisted,
/// bounding metadata bloat + hostile-string render. 512 chars is comfortably
/// above any legitimate name/title.
pub(crate) const MAX_NAME_LEN: usize = 512;

/// Read a required string argument and reject it if longer than [`MAX_NAME_LEN`]
/// chars (S4). Used by the two metadata-write tools (`set_speaker_name`,
/// `rename_meeting`) so an MCP/bridged caller cannot persist an unbounded value.
/// The length is measured in Unicode scalar values (`chars`), not bytes.
pub(crate) fn require_bounded_str<'a>(
    args: &'a serde_json::Value,
    field: &str,
) -> AppResult<&'a str> {
    let s = require_str(args, field)?;
    if s.chars().count() > MAX_NAME_LEN {
        return Err(AppError::InvalidInput {
            context: format!("argument `{field}` is too long (max {MAX_NAME_LEN} characters)"),
        });
    }
    Ok(s)
}

/// Read a required unsigned-integer argument. `InvalidInput` when absent or not
/// a non-negative integer.
pub(crate) fn require_u64(args: &serde_json::Value, field: &str) -> AppResult<u64> {
    args.get(field)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| AppError::InvalidInput {
            context: format!("missing or non-integer argument `{field}`"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn relax_meeting_id_requirement_drops_only_meeting_id() {
        let mut descriptors = vec![ToolDescriptor {
            name: "get_transcript",
            title: "Get transcript",
            description: "…",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "meeting_id": { "type": "string" },
                    "max_chars": { "type": "integer" }
                },
                "required": ["meeting_id", "max_chars"]
            }),
        }];
        relax_meeting_id_requirement(&mut descriptors);
        let req = descriptors[0].input_schema["required"].as_array().unwrap();
        assert!(
            !req.iter().any(|f| f.as_str() == Some("meeting_id")),
            "meeting_id is no longer required"
        );
        assert!(
            req.iter().any(|f| f.as_str() == Some("max_chars")),
            "other required fields are kept"
        );
        // The property itself stays documented + accepted.
        assert!(descriptors[0].input_schema["properties"]["meeting_id"].is_object());
    }

    #[test]
    fn relax_meeting_id_requirement_tolerates_a_missing_required_array() {
        let mut descriptors = vec![ToolDescriptor {
            name: "x",
            title: "X",
            description: "",
            input_schema: json!({ "type": "object", "properties": {} }),
        }];
        relax_meeting_id_requirement(&mut descriptors); // no `required` key → no-op, no panic
        assert!(descriptors[0].input_schema.get("required").is_none());
    }

    #[test]
    fn real_descriptors_have_meeting_id_relaxed_when_scoped() {
        // The production registry's read tools mark `meeting_id` required; after
        // relaxation (meeting-scoped chat) none of them do.
        let mut descriptors = ToolRegistry::v1(false).descriptors();
        relax_meeting_id_requirement(&mut descriptors);
        for d in &descriptors {
            if let Some(req) = d.input_schema.get("required").and_then(|r| r.as_array()) {
                assert!(
                    !req.iter().any(|f| f.as_str() == Some("meeting_id")),
                    "tool {} still requires meeting_id after relaxation",
                    d.name
                );
            }
        }
    }
}
