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
//! `Arc<dyn meeting_app_common::Summariser>` held in [`ToolContext`],
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
//! (`Orchestrator::re_transcribe`/`rediarize`/`transcribe_pcm_window` are
//! `async fn`; `MeetingIndex::list_meetings`/`search` are async libsql). Tool
//! *bodies* still push CPU/fs/inference work onto `tokio::task::spawn_blocking`
//! — the trait being async does not relax the cross-cutting rule.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use meeting_app_common::{AppError, AppResult, AppEvent, MeetingId, Summariser};
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use tokio::sync::{broadcast, Mutex};

mod tools;

pub use tools::*;

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
    /// The recording orchestrator. Backs `relisten_section`, `retranscribe_meeting`,
    /// `rediarize_meeting`, and `get_recording_state`. Owns the `model-registry`
    /// edge so `agent-tools` need not.
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
    /// offline write ops (`retranscribe`/`rediarize`) emit
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
    /// Per-meeting metadata write serialization (§4.2). The two tool-layer
    /// writers that bypass the orchestrator's offline claim
    /// (`set_speaker_name`, `rename_meeting`) take the same per-meeting async
    /// mutex for their read-modify-write of `metadata.json`, so two concurrent
    /// metadata writes cannot drop a write (last-writer-wins).
    metadata_locks: Arc<Mutex<HashMap<MeetingId, Arc<Mutex<()>>>>>,
}

impl ToolContext {
    /// Construct a context. `metadata_locks` is initialised empty.
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
            metadata_locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The `{meetings_dir}/{meeting_id}/` folder path for `id`.
    pub(crate) fn meeting_dir(&self, id: MeetingId) -> PathBuf {
        self.meetings_dir.join(id.0.to_string())
    }

    /// Resolve a per-meeting metadata mutex for `id`, creating it on first use.
    /// The returned guard must be held across the whole read-modify-write of
    /// `metadata.json` (§4.2 class 2/3).
    pub(crate) async fn metadata_lock(&self, id: MeetingId) -> Arc<Mutex<()>> {
        let mut map = self.metadata_locks.lock().await;
        Arc::clone(map.entry(id).or_insert_with(|| Arc::new(Mutex::new(()))))
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
    /// flag — Phase 9 passes `false` (the inter-agent tool body is Phase 10), so
    /// the flag is currently a no-op reserved for the Phase-10 add.
    pub fn v1(include_inter_agent_bridge: bool) -> Self {
        let tools: Vec<Arc<dyn Tool>> = vec![
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
            // Writes (MCP exposure per the allowlist on each impl).
            Arc::new(tools::SetSpeakerName),
            Arc::new(tools::RenameMeeting),
            Arc::new(tools::RetranscribeMeeting),
            Arc::new(tools::RediarizeMeeting),
        ];

        if include_inter_agent_bridge {
            // Phase 10 registers `send_to_internal_agent` here. Phase 9 ships
            // nothing for the flag; documented so the Phase-10 add is a pure
            // append with no signature change.
            tracing::debug!(
                target: "agent-tools",
                "include_inter_agent_bridge=true requested, but the inter-agent \
                 tool is a Phase-10 addition; ignoring in v1"
            );
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
        self.tools.iter().map(|t| descriptor_of(t.as_ref())).collect()
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
        description: t.description(),
        input_schema: t.input_schema(),
    }
}

/// Lightweight argument validation against the tool's JSON Schema.
///
/// This is intentionally a shallow structural check, not a full JSON-Schema
/// validator, because `agent-tools`'s allowed deps include no validator crate.
/// It enforces that `args` is an object and that every property the schema marks
/// `required` is present and non-null. Per-field type coercion and range checks
/// happen in each tool body, where the typed deserialise turns a shape mismatch
/// into `InvalidInput`. The grammar-constrained decode in `chat-agent` is the
/// upstream guard that the model emits schema-valid args in the first place.
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
pub(crate) fn resolve_meeting(
    ctx: &ToolContext,
    args: &serde_json::Value,
) -> AppResult<MeetingId> {
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
pub(crate) fn require_str<'a>(
    args: &'a serde_json::Value,
    field: &str,
) -> AppResult<&'a str> {
    args.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::InvalidInput {
            context: format!("missing or non-string argument `{field}`"),
        })
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
