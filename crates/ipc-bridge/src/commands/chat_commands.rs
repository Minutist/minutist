//! Chat commands (Phase 9): send/cancel a chat turn, session list/get/delete, MCP server info.
//!
//! Command bodies are factored out so the State-free turn loop is unit-tested in
//! `crate::chat` with a stub engine + stub tools (no model, no Tauri runtime).
use super::*;


/// Read a meeting's title for the chat scope line (best-effort; `None` when its
/// metadata can't be read). Runs the blocking `std::fs` read on `spawn_blocking`.
pub(crate) async fn read_meeting_title(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Option<String> {
    let dir = meetings_dir.join(meeting_id.0.to_string());
    tokio::task::spawn_blocking(move || persistence::read_metadata(&dir).ok().map(|m| m.title))
        .await
        .ok()
        .flatten()
}

/// Choose the persona base for a non-live chat turn (B2).
///
/// A session that is the meeting's live co-pilot session (`session_is_live`)
/// keeps the co-pilot's own persona even when the turn runs on the non-live
/// (`run_chat_turn_on_held_model`) path — e.g. post-Stop, once the live worker
/// has shut down and no held-context session remains. Every other session
/// (a fresh or ordinary chat) uses the standard chat persona. Both keep the
/// same tool registry and the "# Current meeting" scoping applied afterwards
/// by [`chat_system_prompt_for_meeting`] — only the base voice differs.
pub(crate) fn chat_turn_base_prompt<'a>(
    session_is_live: bool,
    chat_system_prompt: &'a str,
    live_agent_system_prompt: &'a str,
) -> &'a str {
    if session_is_live {
        live_agent_system_prompt
    } else {
        chat_system_prompt
    }
}

/// Scope the chat system prompt to the open meeting.
///
/// When the chat is meeting-scoped, the agent must GROUND its answers in that
/// meeting and never ask the user for a meeting id — every meeting tool defaults
/// to it via [`ToolContext::default_meeting`]. The base prompt says "this
/// meeting" but never names which one, so without this the model has no meeting
/// identity and asks the user instead of calling a tool. With no meeting in scope
/// (a meeting-less chat) the base prompt is returned unchanged — the agent then
/// locates a meeting via `search_meetings` / an explicit id, as before.
pub(crate) fn chat_system_prompt_for_meeting(
    base: &str,
    meeting_id: Option<MeetingId>,
    title: Option<&str>,
) -> String {
    let Some(mid) = meeting_id else {
        return base.to_string();
    };
    let titled = match title {
        Some(t) if !t.trim().is_empty() => format!(" titled \"{}\"", t.trim()),
        _ => String::new(),
    };
    format!(
        "{base}\n\n# Current meeting\n\
         You are assisting with the meeting the user currently has open \
         (id: {id}{titled}). Every meeting tool defaults to THIS meeting, so NEVER \
         ask the user which meeting or for a meeting id — call the tools directly \
         (get_meeting, get_transcript, get_summary, get_notes, and the re-listen / \
         re-summarise / search / set-speaker tools) to ground your answers in it.",
        id = mid.0,
    )
}

/// Inner implementation of the live co-pilot chat routing path (A3).
///
/// Routes a user message into the live co-pilot session for `mid` via `user_tx`,
/// resolves the live [`ChatSessionId`], registers a cancel flag, spawns the
/// reply-drain task, and returns the session id immediately.
///
/// Extracted from `send_chat_message` so the live-routing logic (cancel
/// registration, drain lifecycle, guard release) can be tested without
/// constructing a full `tauri::State`.
///
/// Returns `Ok(live_sid)` on success or an error when the session lookup fails.
/// If the worker is gone at send time, `Ok(live_sid)` is returned but a
/// `ChatError` event is emitted (matching the outer command's contract).
pub(crate) async fn route_live_chat_message(
    meetings_dir: &Path,
    mid: MeetingId,
    user_tx: tokio::sync::mpsc::Sender<UserChatRequest>,
    message: String,
    event_tx: broadcast::Sender<AppEvent>,
    chat_in_flight: Arc<std::sync::Mutex<std::collections::HashSet<ChatSessionId>>>,
    chat_cancel: Arc<
        std::sync::Mutex<
            std::collections::HashMap<ChatSessionId, chat_agent::CancelFlag>,
        >,
    >,
) -> Result<ChatSessionId, AppError> {
    // Resolve the live ChatSessionId (read-only; spawn_blocking for the
    // filesystem scan under ChatStore::find_live / ChatStore::load_or_create_live).
    let md = meetings_dir.to_path_buf();
    let live_session = tokio::task::spawn_blocking(move || {
        let now = chrono::Utc::now().to_rfc3339();
        ChatStore::load_or_create_live(&md, mid, &now)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("live session lookup task join failed: {e}"),
    })??;
    let live_sid = live_session.id;

    // Single in-flight turn per session. Use the live session id as the
    // guard key so the per-session busy check applies to the live log.
    {
        let mut in_flight = chat_in_flight.lock().expect("chat_in_flight poisoned");
        if !in_flight.insert(live_sid) {
            return Err(AppError::InvalidInput {
                context: "session busy: a turn is already running".into(),
            });
        }
    }

    // Compute the turn_id from the live session's existing messages.
    let turn_id = live_session
        .messages
        .iter()
        .map(|m| m.turn_id)
        .max()
        .map_or(0, |t| t + 1);

    // Bounded reply channel from the live worker back to this drain task.
    // Depth 32 — tokens are best-effort (try_send); Done/Err block.
    const LIVE_REPLY_DEPTH: usize = 32;
    let (reply_tx, mut reply_rx) =
        tokio::sync::mpsc::channel::<UserReplyChunk>(LIVE_REPLY_DEPTH);

    // Register the cancel flag before handing off, so that
    // `cancel_chat_turn(live_sid)` can raise it at any time after this point.
    // The flag is forwarded into the live worker via `UserChatRequest::cancel`;
    // the worker's decode loop observes it between tokens. The drain task
    // removes the entry on completion.
    let cancel = chat_agent::CancelFlag::new();
    {
        chat_cancel
            .lock()
            .expect("chat_cancel poisoned")
            .insert(live_sid, cancel.clone());
    }

    // Send the user turn into the live worker. If the worker is already
    // gone (worker thread exited between the handle snapshot and now),
    // emit ChatError and clear both guards.
    if user_tx
        .send(UserChatRequest {
            message,
            reply_tx,
            cancel: cancel.clone(),
        })
        .await
        .is_err()
    {
        chat_in_flight
            .lock()
            .expect("chat_in_flight poisoned")
            .remove(&live_sid);
        chat_cancel
            .lock()
            .expect("chat_cancel poisoned")
            .remove(&live_sid);
        let _ = event_tx.send(AppEvent::ChatError {
            session_id: live_sid,
            message: "Live co-pilot stopped before the message could be delivered.".to_string(),
        });
        return Ok(live_sid);
    }

    // Spawn the drain task: converts reply chunks to broadcast events.
    // Clears both the in-flight guard and cancel entry on completion.
    tokio::spawn(async move {
        let mut saw_terminal = false;
        let mut streamed = String::new();
        while let Some(chunk) = reply_rx.recv().await {
            match chunk {
                UserReplyChunk::Token(token) => {
                    streamed.push_str(&token);
                    let _ = event_tx.send(AppEvent::ChatToken {
                        session_id: live_sid,
                        turn_id,
                        token,
                    });
                }
                UserReplyChunk::Done(final_text) => {
                    let _ = event_tx.send(AppEvent::ChatTurnComplete {
                        session_id: live_sid,
                        turn_id,
                        final_text,
                    });
                    saw_terminal = true;
                    break;
                }
                UserReplyChunk::Err(msg) => {
                    let _ = event_tx.send(AppEvent::ChatError {
                        session_id: live_sid,
                        message: msg,
                    });
                    saw_terminal = true;
                    break;
                }
            }
        }
        // Channel closed without a terminal chunk. If tokens were streamed, the
        // worker's try_send(Done) was dropped on a full buffer — treat the
        // streamed text as the final so the turn still completes. With no tokens
        // at all, the worker died before replying, so surface an error to unblock
        // the UI.
        if !saw_terminal {
            if streamed.is_empty() {
                let _ = event_tx.send(AppEvent::ChatError {
                    session_id: live_sid,
                    message: "Live co-pilot reply channel closed unexpectedly.".to_string(),
                });
            } else {
                let _ = event_tx.send(AppEvent::ChatTurnComplete {
                    session_id: live_sid,
                    turn_id,
                    final_text: streamed,
                });
            }
        }
        chat_in_flight
            .lock()
            .expect("chat_in_flight poisoned")
            .remove(&live_sid);
        chat_cancel
            .lock()
            .expect("chat_cancel poisoned")
            .remove(&live_sid);
    });

    Ok(live_sid)
}

/// Send a user message to the chat agent for a meeting, streaming the reply.
///
/// **Live path (A3):** when the target meeting is currently recording (a
/// [`crate::live_agent::LiveCopilotHandle`] exists for it), the message is
/// routed into the live co-pilot's held-context session. The live session id is
/// resolved via [`ChatStore::load_or_create_live`] and returned immediately; a
/// drain task converts the per-request reply channel into the same
/// `ChatToken` / `ChatTurnComplete` / `ChatError` events the non-live path
/// emits. Persistence is handled by [`crate::live_agent::process_request`].
///
/// **Non-live path:** creates or loads a standard [`ChatSession`], appends the
/// user message, and spawns a `LlamaTurnBackend` turn on `spawn_blocking`,
/// returning the session id immediately. The turn streams via the same chat
/// `AppEvent`s; tool dispatch re-enters async via a captured
/// `Handle::block_on`. A second `send_chat_message` for a session whose turn
/// is still running is rejected with `InvalidInput { "session busy" }`
/// (§6 — single in-flight turn per session).
#[tauri::command]
#[specta::specta]
pub async fn send_chat_message(
    meeting_id: Option<MeetingId>,
    session_id: Option<ChatSessionId>,
    message: String,
    state: State<'_, IpcState>,
) -> AppResult<ChatSessionId> {
    if message.trim().is_empty() {
        return Err(AppError::InvalidInput {
            context: "chat message must not be empty".into(),
        }
        .into());
    }

    let meetings_dir = state.meetings_dir.clone();

    // --- Live co-pilot routing (A3) ---
    //
    // When the target meeting is currently recording (a LiveCopilotHandle exists
    // for it), route the message into the live co-pilot's held-context session
    // rather than spinning up a fresh LlamaTurnBackend. The implementation lives
    // in `route_live_chat_message` so the logic is testable without a full
    // Tauri runtime.
    if let Some(mid) = meeting_id {
        // Snapshot the handle while holding the lock for the minimum time.
        let live_handle = state
            .live_copilot_handles
            .lock()
            .expect("live_copilot_handles poisoned")
            .get(&mid)
            .map(|h| h.user_tx.clone());

        if let Some(user_tx) = live_handle {
            return route_live_chat_message(
                &meetings_dir,
                mid,
                user_tx,
                message,
                state.event_tx.clone(),
                Arc::clone(&state.chat_runtime.chat_in_flight),
                Arc::clone(&state.chat_runtime.chat_cancel),
            )
            .await;
        }
    }
    // --- End live routing ---

    // Non-live path: load or create a standard (post-meeting / off-meeting) chat
    // session and run a fresh LlamaTurnBackend turn.
    //
    // Load the existing session (when a session id + meeting id are given) or
    // start a fresh one. Persistence reads run on a blocking thread.
    let mut session = load_or_new_session(&meetings_dir, meeting_id, session_id).await?;
    let sid = session.id;

    // Single in-flight turn per session.
    {
        let mut in_flight = state
            .chat_runtime
            .chat_in_flight
            .lock()
            .expect("chat_in_flight poisoned");
        if !in_flight.insert(sid) {
            return Err(AppError::InvalidInput {
                context: "session busy: a turn is already running".into(),
            }
            .into());
        }
    }

    // The per-session monotonic turn id: one past the max already recorded.
    let turn_id = session
        .messages
        .iter()
        .map(|m| m.turn_id)
        .max()
        .map_or(0, |t| t + 1);

    // Append the user message to the persisted session up front so it is durable
    // even if the turn errors mid-flight.
    session.messages.push(ChatMessage {
        role: ChatRole::User,
        content: message.clone(),
        tool_name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
        turn_id,
    });

    // Register the per-session cancel flag (P1) BEFORE `ensure_summariser`,
    // which can block for a long time on the first call (it loads and warms
    // the multi-GB GGUF). `cancel_chat_turn` finds a flag to raise only via
    // this map; registering it after the load would leave a `cancel_chat_turn`
    // that arrives during a slow first load silently dropped.
    let cancel_map = Arc::clone(&state.chat_runtime.chat_cancel);
    let cancel = chat_agent::CancelFlag::new();
    cancel_map
        .lock()
        .expect("chat_cancel poisoned")
        .insert(sid, cancel.clone());

    // Ensure the held model is loaded (downloads on first use) BEFORE spawning,
    // so a load failure surfaces synchronously to the caller rather than only as
    // a ChatError event. Cheap after the first call.
    let summariser = match state.ensure_summariser().await {
        Ok(s) => s,
        Err(e) => {
            state
                .chat_runtime
                .chat_in_flight
                .lock()
                .expect("chat_in_flight poisoned")
                .remove(&sid);
            cancel_map
                .lock()
                .expect("chat_cancel poisoned")
                .remove(&sid);
            return Err(e);
        }
    };

    // Build the tool context for this session (default_meeting scopes meeting_id
    // omission for the internal UI).
    // Peek the held embedder WITHOUT loading it: retrieval is a bonus tool, so a
    // chat turn that never calls retrieve_chunks must not pay the model-load /
    // download cost (the write path loads it; the tool errors gracefully when None).
    let embedder = state.embedder_if_loaded();
    let ctx = ToolContext::new(
        Arc::new(crate::OrchestratorRecordingControl(Arc::clone(
            &state.orchestrator,
        ))),
        Arc::clone(&state.index),
        meetings_dir.clone(),
        summariser.clone() as Arc<dyn Summariser>,
        embedder,
        state.event_tx.clone(),
        meeting_id,
    );

    // Scope the prompt to the open meeting so the agent uses the tools (which
    // default to this meeting) instead of asking the user for a meeting id.
    // The output-language instruction is appended last so it wins over any
    // conflicting text in a custom chat_system_prompt.
    //
    // A session that is the meeting's live co-pilot session (`is_live`) keeps
    // the co-pilot's persona (`live_agent_system_prompt`) even though it now
    // runs on this non-live turn path post-Stop — otherwise the voice would
    // shift mid-conversation from co-pilot to generic chat assistant. It still
    // gets the full tool registry and the "# Current meeting" scoping below,
    // so the co-pilot can look things up after recording ends. Ordinary
    // (non-live) sessions are unaffected.
    let title = match meeting_id {
        Some(mid) => read_meeting_title(&meetings_dir, mid).await,
        None => None,
    };
    let current_settings = state.settings.current();
    let base_prompt = chat_turn_base_prompt(
        session.is_live,
        &current_settings.chat_system_prompt,
        &current_settings.live_agent_system_prompt,
    );
    let system_prompt = apply_output_language(
        &chat_system_prompt_for_meeting(base_prompt, meeting_id, title.as_deref()),
        &current_settings.output_language,
    );
    let registry = Arc::clone(&state.chat_runtime.tool_registry);
    let event_tx = state.event_tx.clone();
    let in_flight = Arc::clone(&state.chat_runtime.chat_in_flight);
    let handle = tokio::runtime::Handle::current();
    // `cancel_map` and `cancel` are already registered above, before
    // `ensure_summariser`, and are reused as-is here — re-registering a fresh
    // flag at this point would silently discard a cancel raised during the
    // model load that just completed.

    // Spawn the driver; the turn streams via events. The session id is returned
    // to the caller now. The turn task OWNS `session` (already carrying the user
    // message); at the end it appends the turn's produced messages and SAVES the
    // whole in-memory session. The single-in-flight-turn guard makes this turn the
    // sole writer, so we save the in-memory copy directly rather than
    // reload-and-append — that guarantees the user message is persisted, even when
    // the turn errors mid-flight.
    tokio::spawn(async move {
        let join = tokio::task::spawn_blocking(move || {
            let produced = run_chat_turn_on_held_model(
                &summariser,
                &registry,
                &ctx,
                &handle,
                sid,
                turn_id,
                &system_prompt,
                &session,
                &event_tx,
                &cancel,
                // The internal UI chat keeps the full tool set (no MCP gate).
                None,
            );
            (session, produced)
        })
        .await;

        match join {
            Ok((mut session, Ok(produced))) => {
                session.messages.extend(produced);
                persist_session(&meetings_dir, meeting_id, session).await;
            }
            Ok((session, Err(_))) => {
                // The driver already emitted ChatError; still persist the session
                // so the user's message (and any earlier turns) are not lost.
                persist_session(&meetings_dir, meeting_id, session).await;
            }
            Err(join_err) => {
                tracing::warn!(target: "ipc-bridge", "chat turn task join failed: {join_err}");
            }
        }
        in_flight
            .lock()
            .expect("chat_in_flight poisoned")
            .remove(&sid);
        cancel_map
            .lock()
            .expect("chat_cancel poisoned")
            .remove(&sid);
    });

    Ok(sid)
}

/// Cancel the in-flight chat turn for a session (P1).
///
/// Raises the per-session [`chat_agent::CancelFlag`] registered by
/// `send_chat_message`; the engine's decode loop observes it between tokens,
/// stops, and the driver emits the terminal `ChatTurnComplete` with the partial
/// text (cancellation is a user action, not a `ChatError`) and clears the
/// in-flight guard. Idempotent: a session with no running turn (no registered
/// flag) is a no-op success — the UI can call this freely to clear a stuck
/// "Sending…" state.
#[tauri::command]
#[specta::specta]
pub async fn cancel_chat_turn(
    session_id: ChatSessionId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    if let Some(flag) = state
        .chat_runtime
        .chat_cancel
        .lock()
        .expect("chat_cancel poisoned")
        .get(&session_id)
    {
        flag.cancel();
    }
    Ok(())
}

/// The live MCP server endpoint (URL + bearer token) for the Settings → MCP
/// pane (Phase 10). `None` when the MCP server is disabled or not yet listening.
///
/// The bearer token is sensitive and crosses the IPC boundary ONLY here, on this
/// explicit read — it is never on the event bus, never logged, and not baked
/// into the bindings. The pane reveals it on user request.
///
/// v1 has no live token-rotation command: the token is generated once and
/// persisted to `{app-data}/mcp_token`, and the listener is spawned once at
/// startup. Rotating the token (delete the file → restart) is therefore
/// restart-required, consistent with the rest of the MCP lifecycle (enable /
/// port / write-tools changes are also restart-required for v1). The pane copy
/// states this; it does NOT offer a live regenerate control (C2).
#[tauri::command]
#[specta::specta]
pub async fn get_mcp_server_info(
    state: State<'_, IpcState>,
) -> AppResult<Option<crate::McpServerInfo>> {
    Ok(state.connected.mcp_info.lock().expect("mcp_info poisoned").clone())
}


/// Load the session named by `session_id` for `meeting_id`, or build a fresh one.
///
/// A given `(meeting_id, session_id)` that exists is loaded; otherwise, when no
/// `session_id` was supplied, the meeting's live co-pilot session (if any) is
/// continued — `send_chat_message` from the webview after Stop omits
/// `session_id` on the meeting's first post-recording open, and the
/// conversation the co-pilot held during recording must carry on rather than
/// be orphaned behind a fresh, unrelated session. Only when neither an exact
/// match nor a live session exists is a new session (with a fresh id, or the
/// caller-supplied `session_id` if one was given but not found) returned.
/// Blocking reads on `spawn_blocking`.
pub(crate) async fn load_or_new_session(
    meetings_dir: &std::path::Path,
    meeting_id: Option<MeetingId>,
    session_id: Option<ChatSessionId>,
) -> AppResult<ChatSession> {
    let now = chrono::Utc::now().to_rfc3339();

    if let (Some(mid), Some(sid)) = (meeting_id, session_id) {
        let dir = meetings_dir.to_path_buf();
        let existing = tokio::task::spawn_blocking(move || ChatStore::load(&dir, mid, sid))
            .await
            .map_err(|e| AppError::Internal {
                context: format!("load_or_new_session task join failed: {e}"),
            })??;
        if let Some(session) = existing {
            return Ok(session);
        }
    }

    if session_id.is_none() {
        if let Some(mid) = meeting_id {
            let dir = meetings_dir.to_path_buf();
            let live = tokio::task::spawn_blocking(move || ChatStore::find_live(&dir, mid))
                .await
                .map_err(|e| AppError::Internal {
                    context: format!("load_or_new_session live lookup task join failed: {e}"),
                })??;
            if let Some(session) = live {
                return Ok(session);
            }
        }
    }

    Ok(ChatSession {
        id: session_id.unwrap_or_default(),
        meeting_id,
        title: None,
        messages: Vec::new(),
        created_at: now.clone(),
        updated_at: now,
        is_live: false,
    })
}

/// Persist the full in-memory chat `session` (prior history + the user message +
/// the turn's produced messages) via [`ChatStore`].
///
/// The single-in-flight-turn guard (`chat_in_flight`) makes the running turn the
/// sole writer of this session, so we save the in-memory copy DIRECTLY rather
/// than reload-and-append. The earlier reload-and-append dropped the user message
/// entirely (it lived only in the in-memory `session`, which was never on disk).
/// A meeting-less session is not persisted (no folder to write into); the streamed
/// events already delivered the reply to the webview.
pub(crate) async fn persist_session(
    meetings_dir: &std::path::Path,
    meeting_id: Option<MeetingId>,
    mut session: ChatSession,
) {
    let Some(mid) = meeting_id else {
        return;
    };
    session.updated_at = chrono::Utc::now().to_rfc3339();
    let session_id = session.id;
    let dir = meetings_dir.to_path_buf();
    let result = tokio::task::spawn_blocking(move || ChatStore::save(&dir, mid, &session)).await;

    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(
            target: "ipc-bridge",
            session_id = %session_id.0,
            "persisting chat session failed: {e}"
        ),
        Err(join_err) => tracing::warn!(
            target: "ipc-bridge",
            "persist_session task join failed: {join_err}"
        ),
    }
}

/// Drive ONE chat turn on the held model (the `spawn_blocking` body).
///
/// Builds the real [`TurnEngine`] over a [`LlamaTurnBackend`] from the borrowed
/// held model, runs the State-free [`run_chat_turn`] loop, and dispatches each
/// tool call by re-entering async via `handle.block_on(registry.dispatch(...))`
/// — the only async/sync crossing (§4.5). Tokens + tool + completion events are
/// emitted on `event_tx` through the emit closure, which ALSO records the wire
/// messages this turn produced (the assistant final + each tool result) so the
/// caller can persist them. Returns those wire messages.
///
/// `mcp_gate` (S1) bounds the tool surface the turn may use to the MCP-allowed
/// set, REUSING the single policy in `agent-tools`
/// (`mcp_tool_descriptors_gated` / `mcp_call_allowed`):
/// - `None` — the internal UI chat: the full registry tool set, no gate.
/// - `Some(allow_writes)` — the Phase-10 inter-agent bridge: the model sees ONLY
///   the gated descriptors (so destructive ops like `reprocess_meeting` are never
///   offered), AND a non-allowed tool requested
///   anyway is rejected before dispatch as defence in depth — mirroring the
///   direct MCP `tools/call` path so a bridged external caller gets NO broader a
///   write surface than a direct MCP call under the same `mcp_write_tools`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_chat_turn_on_held_model(
    summariser: &LlamaSummariser,
    registry: &agent_tools::ToolRegistry,
    ctx: &ToolContext,
    handle: &tokio::runtime::Handle,
    session_id: ChatSessionId,
    turn_id: u64,
    system_prompt: &str,
    session: &ChatSession,
    event_tx: &broadcast::Sender<AppEvent>,
    cancel: &chat_agent::CancelFlag,
    mcp_gate: Option<bool>,
) -> Result<Vec<ChatMessage>, AppError> {
    let backend = LlamaTurnBackend::new(summariser.model(), LlamaTurnConfig::default());
    let engine = TurnEngine::new(backend);
    // The tool surface offered to the model: the full set for the UI path, or
    // the MCP-gated set for the inter-agent bridge (S1). The gating policy lives
    // in `agent-tools`; this only selects which projection to feed the engine.
    let mut descriptors = match mcp_gate {
        Some(allow_writes) => registry.mcp_tool_descriptors_gated(allow_writes),
        None => registry.descriptors(),
    };
    // Meeting-scoped chat: the context fills an omitted `meeting_id`
    // (`ToolContext::resolve_meeting`), so relax the schema's requiredness — else
    // a schema-respecting model treats `meeting_id` as a required field it lacks
    // and asks the user for it. Pairs with the prompt's "# Current meeting" scope.
    if ctx.default_meeting.is_some() {
        agent_tools::relax_meeting_id_requirement(&mut descriptors);
    }
    let cfg = chat_sampler_config();

    // Rebuild the engine-internal history: pinned system prompt + the prior
    // persisted messages (which include the just-appended user message).
    let mut history = initial_history(system_prompt);
    history.extend(session.messages.iter().filter_map(engine_message_from_wire));
    // Everything the driver appends to `history` past this point is THIS turn's
    // output (the assistant final + each tool result).
    let prefix_len = history.len();

    // The emit closure just forwards each event to the bus. The persisted turn
    // messages are derived from the engine-history DELTA below (not from events),
    // so a `Tool` message persists the FULL machine payload (the engine's
    // `content`) rather than the one-line human `ChatToolResult.summary` — a
    // reloaded multi-turn session then feeds the model the same tool data it saw
    // live.
    let emit = |event: AppEvent| emit_chat_event(event_tx, event);

    // The dispatch closure: re-enter async for the registry dispatch only. On
    // the gated (bridge) path, REJECT a tool the MCP gate does not allow before
    // dispatching — defence in depth mirroring the direct MCP `tools/call` path
    // (`McpToolHandler::call_tool`), so a bridged caller cannot reach a tool
    // absent from its gated descriptor list even if the (stub or real) model
    // requests it by name (S1). Reuses `agent-tools`' `mcp_call_allowed`.
    let dispatch = |call: &chat_agent::ToolCall| -> AppResult<ToolOutput> {
        mcp_gate_check(registry, mcp_gate, &call.name)?;
        let args: serde_json::Value =
            serde_json::from_str(&call.arguments_json).map_err(|e| AppError::InvalidInput {
                context: format!("tool {} arguments are not valid JSON: {e}", call.name),
            })?;
        // Thread-occupancy cost (documented): this whole turn already runs on a
        // `spawn_blocking` thread (the engine decode is sync), and this nested
        // `block_on` parks THAT blocking thread for the full duration of the
        // async tool dispatch — including any in-tool `spawn_blocking` (e.g. a
        // `relisten`/`resummarise` inference). So one chat tool call holds a
        // blocking-pool thread end-to-end; concurrent tool-calling turns scale
        // with the blocking-pool size, not the worker count. Acceptable for v1
        // (single in-flight turn per session); revisit if the tool surface grows
        // long-running fan-out.
        handle.block_on(registry.dispatch(ctx, &call.name, args))
    };

    let outcome = run_chat_turn(
        &engine,
        session_id,
        turn_id,
        &mut history,
        &descriptors,
        &cfg,
        CHAT_N_CTX,
        cancel,
        dispatch,
        emit,
    );

    // The loop already emitted ChatError on failure; surface it for the caller's
    // log (the caller still persists the user message).
    outcome?;

    // Derive the turn's produced wire messages from the engine-history DELTA: the
    // assistant final + each tool result the driver appended. Tool messages carry
    // the FULL machine payload (the engine `content`) + the tool name, so a
    // reloaded session is faithful to what the model saw in-turn.
    Ok(wire_produced_from_delta(&history[prefix_len..], turn_id))
}

/// Apply the MCP write gate to one requested tool name before dispatch (S1).
///
/// `None` — the internal UI chat: no gate, every tool is allowed.
/// `Some(allow_writes)` — the inter-agent bridge: reject any tool the active gate
/// does not allow, REUSING the single policy in `agent-tools`
/// (`ToolRegistry::mcp_call_allowed`), exactly as the direct MCP `tools/call`
/// path does. Extracted so the bridge gate is unit-testable without a held model
/// (the S1 regression test in `crate::chat`).
pub(crate) fn mcp_gate_check(
    registry: &agent_tools::ToolRegistry,
    mcp_gate: Option<bool>,
    name: &str,
) -> Result<(), AppError> {
    if let Some(allow_writes) = mcp_gate {
        if !registry.mcp_call_allowed(name, allow_writes) {
            return Err(AppError::InvalidInput {
                context: format!("tool `{name}` is not exposed over MCP"),
            });
        }
    }
    Ok(())
}

/// Map the engine-history delta a turn produced (the assistant-tool_calls
/// message + each tool result + the assistant final, in order) into
/// persisted/wire [`ChatMessage`]s. Pure + unit-tested.
///
/// CQ1: the assistant-tool_calls message's `tool_calls` and each tool result's
/// `tool_call_id` are carried onto the wire shape so a reloaded multi-tool turn
/// reconstructs the valid `assistant(tool_calls) → tool(result)` sequence.
pub(crate) fn wire_produced_from_delta(
    new_engine_messages: &[chat_agent::ChatMessage],
    turn_id: u64,
) -> Vec<ChatMessage> {
    new_engine_messages
        .iter()
        .map(|m| ChatMessage {
            role: wire_role(m.role),
            content: m.content.clone(),
            tool_name: m.name.clone(),
            tool_call_id: m.tool_call_id.clone(),
            tool_calls: m
                .tool_calls
                .iter()
                .map(|c| minutist_common::ToolCallRecord {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    arguments_json: c.arguments_json.clone(),
                })
                .collect(),
            turn_id,
        })
        .collect()
}

/// The default chat sampler config (§6.4): a small-temperature sampling chain.
/// The driver injects a per-turn non-zero seed before each `run_turn`; the base
/// config's fixed `seed = 0` is never used on a non-greedy turn.
fn chat_sampler_config() -> chat_agent::SamplerConfig {
    chat_agent::SamplerConfig::default()
}

/// Emit one chat `AppEvent` on the shared broadcast sender (mirror of
/// [`emit_summary_ready`]). A send with no live subscribers is not an error.
fn emit_chat_event(event_tx: &broadcast::Sender<AppEvent>, event: AppEvent) {
    if event_tx.send(event).is_err() {
        tracing::trace!(target: "ipc-bridge", "chat event dropped (no subscribers)");
    }
}

/// Inner body of [`open_meeting`]: assemble the [`MeetingState`] from the
/// meeting folder under `meetings_dir`. Extracted so it can be unit-tested
/// without a Tauri runtime.
pub(crate) fn open_meeting_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Result<MeetingState, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    persistence::read_meeting_state(&meeting_dir)
}

