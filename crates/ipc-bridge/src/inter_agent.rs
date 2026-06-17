//! The inter-agent bridge driver (Phase 10 §3).
//!
//! The Phase-10 MCP tool `send_to_internal_agent` (defined in `agent-tools`,
//! registered on the MCP registry `v1(true)`) sends an
//! [`InterAgentRequest`] over a bounded channel; THIS task owns the receiver,
//! runs ONE chat turn on the held model via the SAME [`run_chat_turn`] driver
//! the UI uses, and replies with the assistant's final text.
//!
//! [`run_chat_turn`]: crate::chat::run_chat_turn
//!
//! Keeping the receiver + the chat turn here (not in `mcp-server`) is what lets
//! `mcp-server` stay free of a `chat-agent` edge: `mcp-server` holds only the
//! channel SENDER (via the `agent-tools` `ToolContext`), and the single place a
//! chat turn runs stays in `ipc-bridge`.
//!
//! # The internal registry (no self-messaging)
//!
//! The driver dispatches tools through an INTERNAL registry built with
//! `ToolRegistry::v1(false)` — so the internal agent it drives does NOT itself
//! get `send_to_internal_agent`, closing the trivial infinite-loop foot-gun.
//!
//! # The MCP write gate on the bridge (S1)
//!
//! An external MCP client reaching the internal agent through this bridge must
//! get NO broader a write surface than a DIRECT MCP call under the active
//! `settings.mcp_write_tools`. `ToolRegistry::v1(false)` includes the destructive
//! `reprocess_meeting` op the direct MCP path keeps
//! unreachable (`expose_over_mcp() == false`), so the bridge applies the SAME
//! gate the direct path uses (the single policy in `agent-tools`):
//! - the engine is offered only the MCP-allowed descriptors
//!   (`mcp_tool_descriptors_gated(allow_writes)`), so the model never sees
//!   `reprocess`; and
//! - the per-call dispatch REJECTS a non-allowed tool (`mcp_call_allowed`) as
//!   defence in depth, even if the model requests it by name.
//!
//! The `allow_writes` bool is threaded from `settings.mcp_write_tools` into the
//! driver.
//!
//! # Concurrency (W4)
//!
//! - The bridge channel is bounded (the tool `try_send`s — a full queue surfaces
//!   to the external caller as "internal agent busy").
//! - Single in-flight turn per session: a request for a session whose turn is
//!   already running (e.g. a human mid-conversation in the UI) is rejected with
//!   `InvalidInput { "session busy" }` rather than interleaving.
//! - The `None`-session bridge maps to ONE dedicated persisted session, so N
//!   external agents serialise onto it (most get "session busy") — documented
//!   single-threaded v1 scope.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use agent_tools::{InterAgentEnvelope, ToolContext, ToolRegistry};
use minutist_common::{
    AppError, AppResult, ChatMessage, ChatRole, ChatSession, ChatSessionId, InterAgentReply,
    MeetingId, Summariser,
};
use tokio::sync::{mpsc, watch};

use crate::chat_runtime::ChatHandles;
use crate::commands::{load_or_new_session, persist_session, run_chat_turn_on_held_model};

/// The bounded depth of the inter-agent bridge channel (§3 — bounded; the tool
/// `try_send`s, so a full queue is "busy", never a block).
pub const INTER_AGENT_QUEUE_DEPTH: usize = 16;

/// A stable session id for the `None`-meeting MCP bridge session (the single
/// dedicated conversation an external agent talks to when it names no meeting).
/// Derived from a fixed namespace UUID so it round-trips across runs and N
/// external agents converge on the SAME session (and thus serialise — §3 / W4).
fn bridge_session_id() -> ChatSessionId {
    // A fixed v4-shaped UUID literal (constant). Not security-sensitive — it is
    // a routing key for the un-scoped bridge session.
    serde_json::from_str::<ChatSessionId>("\"00000000-0000-4000-8000-00000000c0de\"")
        .expect("the fixed bridge session UUID is valid")
}

/// Spawn the inter-agent driver task. Returns the channel SENDER for `app-main`
/// to hand to `mcp-server`'s `ToolContext` (via
/// `ToolContext::with_inter_agent_bridge`).
///
/// The driver exits on either of two conditions (whichever fires first):
/// - All clones of the returned sender are dropped (primary drop-based exit;
///   unchanged from the original design).
/// - `shutdown` flips to `true` (deterministic disable path: `app-main` fires
///   this when `mcp_enabled` toggles off, tying the driver's lifetime to its
///   per-instance MCP server lifecycle rather than the full app lifetime).
///
/// `handles` carries the same held-model + persistence handles the chat command
/// uses; `chat_in_flight` is the SHARED single-in-flight guard (the same set the
/// UI's `send_chat_message` uses), so an external turn and a human turn cannot
/// run on one session at once.
///
/// `allow_writes` is `settings.mcp_write_tools` (S1): it bounds the bridged
/// turn's tool surface to the SAME MCP gate a direct `tools/call` is under, so a
/// bridged external caller cannot reach a tool a direct MCP call could not.
pub fn spawn_inter_agent_driver(
    handles: ChatHandles,
    chat_in_flight: Arc<Mutex<HashSet<ChatSessionId>>>,
    allow_writes: bool,
    mut shutdown: watch::Receiver<bool>,
) -> mpsc::Sender<InterAgentEnvelope> {
    let (tx, mut rx) = mpsc::channel::<InterAgentEnvelope>(INTER_AGENT_QUEUE_DEPTH);

    // The INTERNAL registry: no inter-agent tool, so the agent cannot message
    // itself. Built once and shared across turns.
    let registry = Arc::new(ToolRegistry::v1(false));

    tauri::async_runtime::spawn(async move {
        tracing::info!(target: "ipc-bridge", "inter-agent bridge driver started");
        loop {
            tokio::select! {
                // Deterministic shutdown signal (per-instance MCP server lifecycle).
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        tracing::info!(
                            target: "ipc-bridge",
                            "inter-agent bridge driver stopped (shutdown signal)"
                        );
                        break;
                    }
                }
                // Next request from the MCP tool; None = all senders dropped.
                item = rx.recv() => {
                    match item {
                        None => {
                            tracing::info!(
                                target: "ipc-bridge",
                                "inter-agent bridge driver stopped (sender dropped)"
                            );
                            break;
                        }
                        Some((request, reply_tx)) => {
                            let result = handle_request(
                                &handles,
                                &registry,
                                &chat_in_flight,
                                allow_writes,
                                request,
                            )
                            .await;
                            // The external caller may have given up (timeout)
                            // and dropped the receiver; that is fine.
                            let _ = reply_tx.send(result);
                        }
                    }
                }
            }
        }
    });

    tx
}

/// Handle one inter-agent request: resolve the session, guard single-in-flight,
/// run ONE chat turn on the held model, persist, and return the reply text.
async fn handle_request(
    handles: &ChatHandles,
    registry: &Arc<ToolRegistry>,
    chat_in_flight: &Arc<Mutex<HashSet<ChatSessionId>>>,
    allow_writes: bool,
    request: minutist_common::InterAgentRequest,
) -> AppResult<InterAgentReply> {
    let meetings_dir = handles.meetings_dir.clone();
    let meeting_id: Option<MeetingId> = request.meeting_id;
    // A None-meeting request routes to the single dedicated bridge session;
    // otherwise the named session (or a fresh one) for the named meeting.
    let session_id = request.session_id.unwrap_or_else(bridge_session_id);

    let mut session = load_or_new_session(&meetings_dir, meeting_id, Some(session_id))
        .await
        .map_err(AppError::from)?;
    let sid = session.id;

    // Single in-flight turn per session (shared guard with the UI path).
    {
        let mut in_flight = chat_in_flight.lock().expect("chat_in_flight poisoned");
        if !in_flight.insert(sid) {
            return Err(AppError::InvalidInput {
                context: "session busy".into(),
            });
        }
    }

    // Always release the guard, even on an early error.
    let outcome = run_one_turn(
        handles,
        registry,
        &mut session,
        meeting_id,
        allow_writes,
        request.message,
    )
    .await;
    chat_in_flight
        .lock()
        .expect("chat_in_flight poisoned")
        .remove(&sid);

    let reply_text = outcome?;
    Ok(InterAgentReply {
        session_id: sid,
        reply: reply_text,
    })
}

/// Build the per-turn context, run the turn on `spawn_blocking`, persist the
/// session, and return the assistant's final text. Mirrors the UI's
/// `send_chat_message` turn body, minus the streaming-to-webview detail (the
/// chat `AppEvent`s still fire on the shared bus, so an open UI session sees the
/// turn live when `meeting_id` matches).
async fn run_one_turn(
    handles: &ChatHandles,
    registry: &Arc<ToolRegistry>,
    session: &mut ChatSession,
    meeting_id: Option<MeetingId>,
    allow_writes: bool,
    message: String,
) -> AppResult<String> {
    if message.trim().is_empty() {
        return Err(AppError::InvalidInput {
            context: "inter-agent message must not be empty".into(),
        });
    }

    let sid = session.id;
    let turn_id = session
        .messages
        .iter()
        .map(|m| m.turn_id)
        .max()
        .map_or(0, |t| t + 1);

    // Append the user message up front so it is durable even if the turn errors.
    session.messages.push(ChatMessage {
        role: ChatRole::User,
        content: message,
        tool_name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
        turn_id,
    });

    // Ensure the held model is loaded (downloads on first use).
    let summariser = handles.ensure_summariser().await?;

    // The internal-agent context drives the INTERNAL registry; default_meeting
    // scopes meeting_id omission to the addressed meeting.
    let ctx = ToolContext::new(
        Arc::clone(&handles.orchestrator),
        Arc::clone(&handles.index),
        handles.meetings_dir.clone(),
        summariser.clone() as Arc<dyn Summariser>,
        handles.event_tx.clone(),
        meeting_id,
    );

    // Scope the prompt to the addressed meeting (the inter-agent bridge can be
    // meeting-scoped too), so the internal agent grounds its answer in that
    // meeting rather than asking for a meeting id. The output-language
    // instruction is appended last so it wins over any conflicting text in
    // a custom chat_system_prompt.
    let title = match meeting_id {
        Some(mid) => crate::commands::read_meeting_title(&handles.meetings_dir, mid).await,
        None => None,
    };
    let current_settings = handles.settings.current();
    let system_prompt = crate::commands::apply_output_language(
        &crate::commands::chat_system_prompt_for_meeting(
            &current_settings.chat_system_prompt,
            meeting_id,
            title.as_deref(),
        ),
        &current_settings.output_language,
    );
    let registry = Arc::clone(registry);
    let event_tx = handles.event_tx.clone();
    let handle = tokio::runtime::Handle::current();

    // Run the turn synchronously (the external caller's tools/call blocks on it,
    // capped by the tool's timeout). The turn task OWNS `session` while running.
    let session_for_turn = session.clone();
    // The inter-agent (MCP) path has no user-facing cancel surface; drive the
    // turn with a fresh never-raised flag (P1).
    let cancel = chat_agent::CancelFlag::new();
    let join = tokio::task::spawn_blocking(move || {
        let produced = run_chat_turn_on_held_model(
            &summariser,
            &registry,
            &ctx,
            &handle,
            sid,
            turn_id,
            &system_prompt,
            &session_for_turn,
            &event_tx,
            &cancel,
            // S1: the bridge applies the SAME MCP write gate the direct MCP path
            // uses, so a bridged external caller cannot reach a tool (e.g.
            // retranscribe/rediarize) a direct `tools/call` could not.
            Some(allow_writes),
        );
        (session_for_turn, produced)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("inter-agent turn task join failed: {e}"),
    })?;

    let (mut completed, produced) = join;
    let produced = produced?;
    // Extract the assistant's final text (the last Assistant message produced).
    let final_text = produced
        .iter()
        .rev()
        .find(|m| m.role == ChatRole::Assistant)
        .map(|m| m.content.clone())
        .unwrap_or_default();

    completed.messages.extend(produced);
    // Persist the whole session (only when meeting-scoped; the None-meeting
    // bridge session is in-memory only, matching the UI path).
    persist_session(&handles.meetings_dir, meeting_id, completed).await;

    Ok(final_text)
}

/// Convenience re-export so app-main need not name the `agent_tools` envelope
/// type when wiring the bridge.
pub use agent_tools::InterAgentBridge;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_runtime::ChatHandles;
    use minutist_common::AppEvent;

    /// Verify that the driver task exits promptly when the shutdown watch fires,
    /// independently of whether any sender is still alive.
    ///
    /// The test creates a driver with a real (but idle) channel, sends the
    /// shutdown signal, and awaits the sender-side channel close: the driver
    /// exits its loop, which causes it to drop `rx`, which closes the channel,
    /// which makes the sender return `SendError` on the next send — a
    /// detectable proxy for task exit without needing to `join` the spawned
    /// task.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn driver_stops_on_shutdown_signal() {
        use orchestrator::test_support::test_orchestrator;
        use persistence::MeetingIndex;
        use std::sync::Arc;
        use tokio::sync::{broadcast, watch};

        let tempdir = tempfile::tempdir().expect("tempdir");
        let meetings_dir = tempdir.path().join("meetings");
        std::fs::create_dir_all(&meetings_dir).expect("meetings dir");

        let index = Arc::new(
            MeetingIndex::open(":memory:")
                .await
                .expect("in-memory index"),
        );
        let orchestrator = Arc::new(test_orchestrator(meetings_dir.clone()));
        let (event_tx, _rx) = broadcast::channel::<AppEvent>(16);
        let summariser_cell = Arc::new(tokio::sync::OnceCell::new());
        let settings_dir = tempfile::tempdir().expect("settings tempdir");
        let settings = settings::SettingsHandle::new(settings::JsonFileStore::new(
            settings_dir.path().join("settings.store"),
        ))
        .expect("settings");

        let handles = ChatHandles {
            orchestrator,
            index,
            meetings_dir,
            event_tx,
            settings,
            summariser: summariser_cell,
        };
        let chat_in_flight = Arc::new(Mutex::new(HashSet::new()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let tx = spawn_inter_agent_driver(handles, chat_in_flight, false, shutdown_rx);

        // The driver is running; the channel is open.
        assert!(!tx.is_closed(), "driver channel must be open before shutdown");

        // Fire the shutdown signal.
        shutdown_tx.send(true).expect("send shutdown");

        // Poll until the driver exits (receiver dropped → channel closed).
        // Bounded at 2 s; the driver exits its select loop on the next tick.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if tx.is_closed() {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("inter-agent bridge driver did not stop within 2 s after shutdown signal");
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}
