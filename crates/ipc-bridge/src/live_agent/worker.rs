//! The dedicated `!Send` worker thread: owns the [`LiveSession`] for the
//! session lifetime, seeds the prefix once, and runs each turn (transcript or
//! user-chat) to completion, including the KV-eviction retry path.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chat_agent::{
    detect_turn_markers, CancelFlag, ConversationalTurn, Error as ChatAgentError, LiveSession,
    LiveSessionBackend, LlamaLiveBackend, LlamaLiveConfig, TurnMarkers,
};
use minutist_common::{ChatRole, Embedder, MeetingId};
use orchestrator::Orchestrator;
use persistence::{meeting_db_path, ChatStore, RagStore};
use settings::SettingsHandle;
use summariser::LlamaSummariser;
use tokio::sync::{mpsc, OnceCell};

use super::{
    context::{self, LiveRetrieval},
    driver, sanitise_untrusted, CopilotTurnRequest, TurnKind, UserReplyChunk, WorkerResult,
    COPILOT_NOOP_SENTINEL, FRAMING_TOKEN_MARGIN, LIVE_WINDOW_BUDGET_CHARS,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_worker_thread(
    meeting_id: MeetingId,
    // HIGH-priority user-chat requests (drained first by the biased select).
    user_req_rx: mpsc::Receiver<CopilotTurnRequest>,
    // LOW-priority transcript requests (drained only when no user turn is pending).
    transcript_req_rx: mpsc::Receiver<CopilotTurnRequest>,
    res_tx: mpsc::Sender<WorkerResult>,
    summariser_cell: Arc<OnceCell<Arc<LlamaSummariser>>>,
    embedder_cell: Arc<OnceCell<Arc<dyn Embedder>>>,
    orchestrator: Arc<Orchestrator>,
    settings: SettingsHandle,
    meetings_dir: PathBuf,
    // C2/M5: raised by the driver on shutdown so the startup prefix seed
    // aborts promptly, unblocking the join (avoids a ~40 s zombie).
    startup_cancel: CancelFlag,
) {
    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        "live-agent worker thread started"
    );

    // If the worker fails to START it must send a terminal error so the driver
    // surfaces a `LiveDigestError` rather than going silent. A `blocking_send`
    // is a no-op once the driver has torn down (a clean Stop during startup),
    // so it raises no spurious error.
    let fail = |msg: String| {
        let _ = res_tx.blocking_send(WorkerResult::Err(msg));
    };

    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker: failed to build tokio runtime: {e}"
            );
            fail(format!("live agent failed to start (runtime): {e}"));
            return;
        }
    };

    // Resolve the held summariser. This calls ensure_summariser which loads the
    // GGUF if not yet loaded. Runs at thread start (before the first refresh),
    // so the load cost is paid once at session spawn, not mid-recording.
    let summariser_arc = match rt.block_on(crate::chat_runtime::ensure_summariser(
        &summariser_cell,
        &orchestrator,
        &settings,
    )) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker: failed to load summariser model: {e}"
            );
            fail(format!("live agent failed to start (model load): {e}"));
            return;
        }
    };

    // Load the held embedder in the BACKGROUND now that the summariser is in.
    // It populates the shared cell (the same `Arc<OnceCell>` the chat / RAG write
    // paths use), so the retrieval loop peeks it each refresh. Progress is
    // cooperative: the spawned task advances only at the worker loop's `.await`
    // points — the model open itself runs on a `spawn_blocking` thread so it does
    // proceed in parallel, but on a busy meeting "ready" can lag the first refresh.
    // The GPU offload races the summariser warm-up with no VRAM admission control;
    // an allocation failure is caught and retrieval is disabled for the session.
    // All best-effort — on any failure the agent runs without injected context.
    {
        let bg_cell = embedder_cell.clone();
        let bg_orchestrator = orchestrator.clone();
        let bg_settings = settings.clone();
        rt.spawn(async move {
            if let Err(e) =
                crate::chat_runtime::ensure_embedder(&bg_cell, &bg_orchestrator, &bg_settings).await
            {
                tracing::warn!(
                    target: "ipc-bridge",
                    error = %e,
                    "live-agent: background embedder load failed (retrieval disabled this session)"
                );
            }
        });
    }

    // Construct the LlamaLiveBackend on this thread. LlamaLiveBackend<'m>
    // borrows &'m LlamaModel; `_keep` (the Arc<LlamaSummariser>) is declared
    // BEFORE `session` so Rust's reverse-declaration drop order guarantees
    // `session` — and the embedded LlamaLiveBackend holding the &LlamaModel
    // borrow — is dropped BEFORE `_keep`. The raw pointer widens the lifetime
    // past the borrow checker's view of the Arc (which cannot see the
    // stack-order guarantee). LlamaModel is `unsafe impl Send + Sync`
    // (architecture/cross-cutting.md); the borrow is shared/read-only.
    let _keep = summariser_arc;
    let model_ptr = std::ptr::from_ref(_keep.model());
    // SAFETY: `_keep` outlives `session` by the declaration-order drop
    // guarantee above; the borrow is read-only; LlamaModel is Send + Sync.
    let model_ref = unsafe { &*model_ptr };

    // Detect the actual turn markers from the loaded model vocabulary BEFORE
    // building the prefix — the prefix must use the same markers the model
    // actually understands as control tokens (Gemma 4: <|turn>/<turn|>;
    // Gemma 2/3: <start_of_turn>/<end_of_turn>).
    let markers = detect_turn_markers(model_ref);
    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        turn_open = %markers.turn_open,
        turn_close = %markers.turn_close,
        "live-agent: detected turn markers from model vocab"
    );

    // Build the awareness block from any Ready attachments that have a summary.
    // Manifest read failure or no attachments with awareness → empty block, and
    // the co-pilot runs as before. An attachment added DURING a live session is
    // not reflected here; mid-session re-seed is deferred (A1 dirty-prefix /
    // eviction-rebuild mechanism, a later U2 item).
    // Per-line char budget for each awareness entry. The awareness text is
    // generated with a 256-token cap, so this is a backstop against abnormally
    // long lines after sanitisation.
    const AWARENESS_LINE_CAP: usize = 512;
    // Overall budget for the assembled awareness block. The block sits inside
    // the pinned prefix (between the system prompt and the close marker); keeping
    // it well under ~4 000 characters ensures the framing tokens are never
    // reached by prefill_prefix's tail-truncation, preserving the closed-turn
    // contract that append_turn depends on.
    const AWARENESS_BLOCK_CAP: usize = 4_096;

    let awareness_block: String = match persistence::read_manifest(&meetings_dir, meeting_id) {
        Ok(entries) => {
            let mut block = String::new();
            for e in entries {
                if block.len() >= AWARENESS_BLOCK_CAP {
                    tracing::debug!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        "live-agent: awareness block budget reached; \
                         remaining attachments omitted from prefix"
                    );
                    break;
                }
                if !matches!(e.conversion, minutist_common::ConversionState::Ready) {
                    continue;
                }
                let Some(a) = e.awareness else { continue };
                // Both the filename and the awareness text share the same
                // untrusted trust boundary (the filename is user-supplied;
                // the awareness is model-generated from untrusted document
                // content). Sanitise both to neutralise any turn-marker
                // sequences, then collapse newlines to a space so a crafted
                // filename or summary cannot fabricate extra bullet lines in
                // the pinned "Attached documents" list (each attachment must
                // occupy exactly one bullet).
                let safe_name = sanitise_untrusted(&e.original_filename, &markers).replace(['\n', '\r'], " ");
                let safe_text = sanitise_untrusted(&a, &markers).replace(['\n', '\r'], " ");
                let line = format!("- {}: {}\n", safe_name, safe_text);
                // Truncate individual lines to avoid one attachment consuming
                // the whole block budget.
                let line = if line.len() > AWARENESS_LINE_CAP {
                    let cap = line
                        .char_indices()
                        .take_while(|(i, _)| *i < AWARENESS_LINE_CAP)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(AWARENESS_LINE_CAP);
                    format!("{}\n", &line[..cap].trim_end())
                } else {
                    line
                };
                block.push_str(&line);
            }
            block
        }
        Err(e) => {
            tracing::debug!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                error = %e,
                "live-agent: manifest read failed; starting without attachment awareness"
            );
            String::new()
        }
    };

    // The prefix carries the system prompt and, when attachments with awareness
    // exist, a compact per-attachment list. Attachment detail is retrieved on
    // demand via the RAG path (the detail tier); this awareness tier only tells
    // the co-pilot WHAT documents are attached so it can decide to surface them.
    let prefix = context::build_prefix(&settings.current(), &markers, &awareness_block);
    tracing::debug!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        prefix_chars = prefix.len(),
        "live-agent: prefix built on worker thread"
    );

    let backend = match LlamaLiveBackend::new(model_ref, LlamaLiveConfig::default()) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker: failed to construct LlamaLiveBackend: {e}"
            );
            fail(format!("live agent failed to start (backend): {e}"));
            return;
        }
    };

    let mut session = LiveSession::new(backend);

    // C2/M5: seed the prefix ONCE at session start, before the cadence loop.
    // The driver-provided startup_cancel is checked between chunks so a Stop
    // during the ~40 s prefill aborts promptly and unblocks the driver's join.
    match session.seed_prefix_typed(&prefix, &startup_cancel) {
        Ok(n) => {
            tracing::info!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                prefix_tokens = n,
                "live-agent: prefix seeded at session start"
            );
        }
        Err(e) => {
            tracing::error!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker: prefix seed failed: {e}; aborting session"
            );
            fail(format!("live agent failed to start (prefix seed): {e}"));
            return;
        }
    }

    // B3: initialise the tool machinery once after the prefix is seeded.
    // Pass None for v1 — retrieval is auto-injected; the tool-dispatch loop
    // is built and structured for a future tool set but currently dormant
    // (no tools are offered). DEFERRED: wire real tools in a later phase.
    if let Err(e) = session.init_tool_machinery(None) {
        tracing::error!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "live-agent worker: init_tool_machinery failed: {e}; aborting session"
        );
        fail(format!("live agent failed to start (tool machinery): {e}"));
        return;
    }

    // Open the per-meeting RAG cache (created if absent — an empty cache until
    // attachments / transcript are indexed). On failure, retrieval is disabled
    // for the session (the agent still runs, just without injected context).
    // The tier-scaled `k` is fixed for the session: the GPU tier does not
    // change mid-meeting.
    let s = settings.current();
    let is_integrated = minutist_common::probe_primary_gpu()
        .map(|p| p.is_integrated)
        // No probe → assume the tight (integrated) tier so an unknown GPU never
        // gets the generous per-turn prefill budget.
        .unwrap_or(true);
    let retrieval = match rt.block_on(RagStore::open(meeting_db_path(&meetings_dir, meeting_id))) {
        Ok(store) => Some(LiveRetrieval {
            embedder_cell,
            store,
            meetings_dir: meetings_dir.clone(),
            k: context::tier_scaled_k(s.live_agent_retrieval_k, is_integrated),
            char_budget: s.live_agent_retrieval_budget_chars,
        }),
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                error = %e,
                "live-agent: opening meeting.db failed; retrieval disabled this session"
            );
            None
        }
    };

    // Seed the turn_id counter from any messages already persisted to this
    // meeting's live ChatSession. A Pause→Resume cycle re-spawns the worker
    // for the SAME meeting; without seeding, the counter would restart at 0
    // and collide with turn_ids already in the durable log.
    let initial_turn_id: u64 = {
        let now = chrono::Utc::now().to_rfc3339();
        match ChatStore::load_or_create_live(&meetings_dir, meeting_id, &now) {
            Ok(session) => session
                .messages
                .iter()
                .map(|m| m.turn_id)
                .max()
                .map_or(0, |max| max + 1),
            Err(e) => {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "live-agent: could not read existing live session to seed turn_id: {e}"
                );
                0
            }
        }
    };

    rt.block_on(run_worker_loop(
        meeting_id,
        user_req_rx,
        transcript_req_rx,
        res_tx,
        &mut session,
        retrieval.as_ref(),
        &markers,
        &meetings_dir,
        initial_turn_id,
    ));

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        "live-agent worker thread exited"
    );
}

/// Load the held summariser using the shared OnceCell, mirroring the logic in
/// `ChatHandles::ensure_summariser`. Called once at worker thread start.
pub(crate) async fn run_worker_loop<B: LiveSessionBackend + ConversationalTurn>(
    meeting_id: MeetingId,
    // HIGH-priority user-chat channel; drained before transcript on each iteration.
    mut user_req_rx: mpsc::Receiver<CopilotTurnRequest>,
    // LOW-priority transcript channel; consumed only when no user turn is pending.
    mut transcript_req_rx: mpsc::Receiver<CopilotTurnRequest>,
    res_tx: mpsc::Sender<WorkerResult>,
    session: &mut LiveSession<B>,
    retrieval: Option<&LiveRetrieval>,
    markers: &TurnMarkers,
    meetings_dir: &Path,
    // Seeded from the existing live ChatSession so turn_ids remain monotonic
    // across a Pause→Resume worker respawn.
    initial_turn_id: u64,
) {
    // Monotonic counter for turn persistence. Seeded from any messages already
    // in the durable log so a worker respawn (Pause→Resume) does not produce
    // duplicate or non-monotonic turn_ids.
    let mut turn_id: u64 = initial_turn_id;

    loop {
        // Biased select: drain a pending user turn before a pending transcript
        // turn. Both channels are depth-1 so at most one request is waiting per
        // lane; the driver only sends on a lane after receiving the previous result.
        let req = tokio::select! {
            biased;
            msg = user_req_rx.recv() => match msg {
                Some(r) => r,
                None => break,
            },
            msg = transcript_req_rx.recv() => match msg {
                Some(r) => r,
                None => break,
            },
        };
        // Retrieve attachment / earlier-transcript context relevant to this
        // turn's content. `None` until the embedder has loaded or the cache is
        // empty; the agent still produces turns without injected context.
        let injected = match retrieval {
            Some(rc) => context::build_retrieval_block(rc, &req.content).await,
            None => None,
        };
        let result = process_request(
            meeting_id,
            session,
            req,
            injected.as_deref(),
            markers,
            meetings_dir,
            &mut turn_id,
        );
        // Both CapacityExhausted and Err are terminal: the held context is
        // untrustworthy after either condition. Stop after sending.
        let is_terminal = matches!(
            result,
            WorkerResult::CapacityExhausted(_) | WorkerResult::Err(_)
        );
        if res_tx.send(result).await.is_err() {
            tracing::debug!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker: result receiver dropped; exiting"
            );
            return;
        }
        // Stop processing requests after a terminal result.
        if is_terminal {
            tracing::info!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker: stopping after terminal result"
            );
            return;
        }

        // Incrementally index newly-sealed transcript turns after the reply is
        // sent, so later turns can retrieve earlier discussion. Best-effort and
        // off the critical path (the result is already sent).
        if let Some(rc) = retrieval {
            if let Some(embedder) = rc.embedder_cell.get().cloned() {
                match crate::rag_index::index_transcript_incremental(
                    &rc.store,
                    &rc.meetings_dir,
                    meeting_id,
                    &embedder,
                )
                .await
                {
                    Ok(n) if n > 0 => tracing::debug!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        chunks = n,
                        "live-agent: incrementally indexed transcript turns"
                    ),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        error = %e,
                        "live-agent: incremental transcript index failed (best-effort)"
                    ),
                }
            }
        }
    }
}

/// Build the framed turn content for one [`CopilotTurnRequest`].
///
/// Both transcript and user-chat inputs are delivered as `"user"` role to the
/// model. The framing differs by [`TurnKind`]:
///
/// - `Transcript`: the retrieved context block (if any) + a sentinel-instruction
///   suffix. The model must reply with [`COPILOT_NOOP_SENTINEL`] when there is
///   nothing worth surfacing.
/// - `UserChat`: the user's message verbatim (must-reply, no sentinel instruction).
///
/// All untrusted text (transcript content, retrieved chunks) is sanitised with
/// `sanitise_untrusted` so chat-control tokens in the content cannot break the
/// model's turn framing.
fn build_turn_content(req: &CopilotTurnRequest, markers: &TurnMarkers) -> String {
    let mut content = String::new();

    // Auto-inject the retrieved context block first, sanitised.
    if let Some(ctx) = &req.retrieved {
        content.push_str(&sanitise_untrusted(ctx, markers));
        content.push('\n');
    }

    match req.kind {
        TurnKind::Transcript => {
            let window = sanitise_untrusted(
                context::tail_chars(&req.content, LIVE_WINDOW_BUDGET_CHARS),
                markers,
            );
            content.push_str("New meeting transcript:\n");
            content.push_str(&window);
            content.push_str(
                "\n\nIf (and only if) something here is worth surfacing to the user \
                (a decision, an action item, an answer to a standing request, an \
                unresolved reference), reply with a short note. If there is nothing \
                worth surfacing, reply with EXACTLY `<<NOOP>>` and nothing else.",
            );
        }
        TurnKind::UserChat => {
            // User content is verbatim; sanitise against chat-control tokens.
            content.push_str(&sanitise_untrusted(&req.content, markers));
        }
    }

    content
}

/// Map a typed [`chat_agent::Error`] from `LiveSession::converse_typed` to a
/// terminal [`WorkerResult`].
///
/// Only `Error::ContextOverflow` maps to `CapacityExhausted`; all other
/// variants (including `MalformedOutput` and `Template`) map to `Err` with an
/// accurate description. Classifying on the typed error rather than on the
/// lossy `AppError::InvalidInput` boundary preserves this distinction:
/// `AppError::From<Error>` collapses `Template`, `Grammar`, `ContextOverflow`,
/// and `MalformedOutput` into the same `InvalidInput` variant, so classifying
/// after that conversion would mislabel a transient parse glitch as "context
/// window filled".
pub(crate) fn classify_converse_error(
    meeting_id: MeetingId,
    e: ChatAgentError,
    phase: &str,
) -> WorkerResult {
    match e {
        ChatAgentError::ContextOverflow(ref msg) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "{phase}: context overflow: {msg}"
            );
            WorkerResult::CapacityExhausted(format!("context overflow ({phase}): {msg}"))
        }
        other => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "{phase} failed: {other}"
            );
            WorkerResult::Err(format!("{phase} failed: {other}"))
        }
    }
}

/// Perform a KV eviction: reset the session to its pinned prefix and prepend a
/// verbatim recap of recent turns to `model_prompt`.
///
/// Returns `Some(WorkerResult::CapacityExhausted(...))` if the reset fails
/// (leaving the KV in an unknown state), `None` on success.
///
/// On success, `model_prompt` is mutated in place: a sanitised recap header is
/// prepended when recent turns are available in the persisted log.
fn do_evict<B: LiveSessionBackend + ConversationalTurn>(
    session: &mut LiveSession<B>,
    meeting_id: MeetingId,
    meetings_dir: &Path,
    markers: &TurnMarkers,
    model_prompt: &mut String,
) -> Option<WorkerResult> {
    let recap = context::load_eviction_recap(meetings_dir, meeting_id);
    let n_past_before = session.n_past();
    if let Err(e) = session.reset_to_prefix() {
        tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "live-agent: reset_to_prefix failed during eviction: {e}; \
             context state may be inconsistent"
        );
        // A failed reset leaves the KV in an unknown state; treat as terminal.
        return Some(WorkerResult::CapacityExhausted(format!(
            "context eviction failed (reset_to_prefix): {e}"
        )));
    }
    let recap_turns = recap.as_deref().map(|r| r.lines().count()).unwrap_or(0);
    let recap_chars = recap.as_deref().map(str::len).unwrap_or(0);
    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        n_past_before,
        recap_turns,
        recap_chars,
        "live-agent: context evicted and reset to prefix; recap prepended"
    );
    if let Some(r) = recap {
        let header = format!(
            "Earlier in this conversation (older context was condensed):\n{}\n\n",
            r
        );
        let safe_header = sanitise_untrusted(&header, markers);
        *model_prompt = safe_header + model_prompt.as_str();
    }
    None
}

pub(crate) fn process_request<B: LiveSessionBackend + ConversationalTurn>(
    meeting_id: MeetingId,
    session: &mut LiveSession<B>,
    req: CopilotTurnRequest,
    retrieved: Option<&str>,
    markers: &TurnMarkers,
    meetings_dir: &Path,
    turn_id: &mut u64,
) -> WorkerResult {
    let kind = req.kind;
    // The raw (unframed) content is what we persist to the conversation log so
    // the U4 chat view shows clean transcript text or the user's literal
    // message — not the scaffolding (NOOP instructions, RAG blocks, or
    // sanitised copies) that is only for the model's consumption.
    let log_content = req.content.clone();

    // Extract the reply channel BEFORE moving req into req_with_retrieved, so
    // it is available for the token callback and the terminal send below.
    let reply_tx = req.reply_tx.as_ref().cloned();

    // Build the full framed prompt: retrieved context block + kind-specific
    // instructions. This is passed to the model only, never persisted.
    let req_with_retrieved = CopilotTurnRequest {
        retrieved: retrieved.map(str::to_string),
        ..req
    };
    let mut model_prompt = build_turn_content(&req_with_retrieved, markers);

    // U2 eviction: if the framed prompt + generation budget would not fit in
    // the remaining context, reset the KV to the pinned prefix and prepend a
    // verbatim recap of the last EVICT_RECAP_TURNS User/Assistant turns so the
    // model retains recent context. Older turns remain in the persisted log and
    // the RAG index (recoverable on demand).
    //
    // The estimate uses chars/3 (conservative — CJK/punctuation-dense text can
    // tokenise at 1 token per char) plus FRAMING_TOKEN_MARGIN to cover the turn
    // markers that `append_turn` prepends. An over-estimate causes a harmless
    // early eviction; an under-estimate can let the turn slip past this gate
    // and hit the hard ContextOverflow guard in `append_turn`.
    //
    // v2 (rolling-summary layered budget) is deferred — see EVICT_RECAP_CHARS.
    let max_tokens = req_with_retrieved.sampler.max_tokens;
    let estimated_tokens = model_prompt.len() / 3 + FRAMING_TOKEN_MARGIN;
    let mut already_evicted = false;
    if !session.has_room_for(estimated_tokens, max_tokens) {
        if let Some(msg) = do_evict(session, meeting_id, meetings_dir, markers, &mut model_prompt) {
            if let Some(ref tx) = reply_tx {
                if let WorkerResult::Err(ref e) | WorkerResult::CapacityExhausted(ref e) = msg {
                    let _ = tx.try_send(UserReplyChunk::Err(e.clone()));
                }
            }
            return msg;
        }
        already_evicted = true;
    }

    // Persist the input turn using the clean content (always, even when the
    // reply is suppressed, so the conversation log is complete).
    let input_role = match kind {
        TurnKind::Transcript => ChatRole::Digest,
        TurnKind::UserChat => ChatRole::User,
    };
    driver::persist_turn(meetings_dir, meeting_id, input_role, &log_content, turn_id);

    let mut generated = String::new();
    // converse_typed preserves the typed chat_agent::Error so ContextOverflow
    // is matched structurally in classify_converse_error, not confused with a
    // transient MalformedOutput or Template failure.
    //
    // For UserChat turns, each decoded piece is forwarded via reply_tx so the
    // command task can emit ChatToken events in real time. try_send is used —
    // tokens are best-effort hints; a full buffer drops the piece and decoding
    // continues unblocked. ChatTurnComplete (sent at the end with the full text)
    // is authoritative and reconciles any dropped tokens.
    //
    // For Transcript turns, reply_tx is None and the callback only accumulates
    // the generated text; LiveCopilotMessage is emitted by the driver on
    // WorkerResult::Message.

    // Inline helper: sends an Err chunk on reply_tx when a terminal WorkerResult is
    // about to be returned early, so the command task's drain loop is unblocked.
    // try_send is used — a send failure means the receiver already exited.
    macro_rules! send_err_chunk {
        ($result:expr) => {
            if let Some(ref tx) = reply_tx {
                if let WorkerResult::Err(ref e) | WorkerResult::CapacityExhausted(ref e) =
                    $result
                {
                    let _ = tx.try_send(UserReplyChunk::Err(e.clone()));
                }
            }
        };
    }

    // If converse returns ContextOverflow and eviction was not already
    // triggered (the estimate heuristic under-counted), attempt one
    // evict-and-retry before falling through to CapacityExhausted. This
    // covers the boundary case where the real token count exceeds the
    // estimate but the session would fit a fresh post-eviction context.
    //
    // Clone the sender for the decode callbacks: mpsc::Sender is cheaply
    // cloneable (Arc-backed). The clones are only live for the duration of
    // their enclosing converse_typed call; `reply_tx` is retained for the
    // terminal Done/Err sends below.
    let raw = {
        let cb_tx = reply_tx.clone();
        match session.converse_typed(
            "user",
            &model_prompt,
            &req_with_retrieved.sampler,
            &req_with_retrieved.cancel,
            &mut |piece: &str| {
                generated.push_str(piece);
                if let Some(ref tx) = cb_tx {
                    // try_send: drop the token on a full buffer. The authoritative
                    // final text is carried on UserReplyChunk::Done.
                    let _ = tx.try_send(UserReplyChunk::Token(piece.to_string()));
                }
            },
        ) {
            Ok(r) => r,
            Err(ChatAgentError::ContextOverflow(_)) if !already_evicted => {
                // Estimate was too optimistic — evict now and retry once.
                tracing::info!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "live-agent: ContextOverflow after estimate passed; \
                     evicting and retrying"
                );
                // Reset model_prompt to the original (unrecapped) framing before
                // re-evicting, so the recap is added exactly once.
                model_prompt = build_turn_content(&req_with_retrieved, markers);
                if let Some(msg) =
                    do_evict(session, meeting_id, meetings_dir, markers, &mut model_prompt)
                {
                    send_err_chunk!(msg);
                    return msg;
                }
                generated.clear();
                let cb_tx2 = reply_tx.clone();
                match session.converse_typed(
                    "user",
                    &model_prompt,
                    &req_with_retrieved.sampler,
                    &req_with_retrieved.cancel,
                    &mut |piece: &str| {
                        generated.push_str(piece);
                        if let Some(ref tx) = cb_tx2 {
                            let _ = tx.try_send(UserReplyChunk::Token(piece.to_string()));
                        }
                    },
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        let result =
                            classify_converse_error(meeting_id, e, "converse-after-evict");
                        send_err_chunk!(result);
                        return result;
                    }
                }
            }
            Err(e) => {
                let result = classify_converse_error(meeting_id, e, "converse");
                send_err_chunk!(result);
                return result;
            }
        }
    };

    // No tools are offered to the live-agent model, so `raw.tool_calls` is
    // always empty here (contrast the non-live chat path, which does dispatch
    // through `agent-tools`); there is no dispatch loop to run.

    // Use raw.text when the callback-accumulated buffer is empty (happens when
    // the model returns the full text via raw.text rather than incremental pieces).
    if generated.is_empty() {
        generated = raw.text;
    }
    // `generated` now holds the complete reply text.
    let reply_text = generated;

    // Response policy (B3 §4).
    match kind {
        TurnKind::Transcript => {
            let trimmed = reply_text.trim();
            if trimmed.is_empty() || trimmed == COPILOT_NOOP_SENTINEL {
                WorkerResult::Suppressed
            } else {
                driver::persist_turn(
                    meetings_dir,
                    meeting_id,
                    ChatRole::Assistant,
                    &reply_text,
                    turn_id,
                );
                WorkerResult::Message {
                    role_is_user_reply: false,
                    content: reply_text,
                }
            }
        }
        TurnKind::UserChat => {
            // Always surface a reply for user turns; even an empty reply is
            // sent so the caller gets an acknowledgement.
            driver::persist_turn(
                meetings_dir,
                meeting_id,
                ChatRole::Assistant,
                &reply_text,
                turn_id,
            );
            // Send the authoritative final text on reply_tx so the command task
            // can emit ChatTurnComplete. This MUST be try_send, never
            // blocking_send: `process_request` runs inside the worker's
            // `rt.block_on(run_worker_loop)`, i.e. within a Tokio runtime
            // context, where blocking_send panics ("Cannot block the current
            // thread from within a runtime"). If the bounded buffer is full (the
            // command-side drain has fallen behind), Done is dropped — but the
            // drain reconstructs the final turn from the streamed Token chunks,
            // so the turn still completes. A send failure equally covers the
            // command task having already exited (user navigated away); the
            // persisted turn is durable either way.
            if let Some(ref tx) = reply_tx {
                if tx
                    .try_send(UserReplyChunk::Done(reply_text.clone()))
                    .is_err()
                {
                    tracing::debug!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        "live-agent: reply_tx Done not delivered (buffer full or \
                         command task exited); drain reconstructs from streamed tokens"
                    );
                }
            }
            WorkerResult::Message {
                role_is_user_reply: true,
                content: reply_text,
            }
        }
    }
}
