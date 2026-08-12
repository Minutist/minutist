//! The async driver task: owns the event loop, the transcript cadence gate,
//! and turn persistence.
//!
//! Subscribes to the orchestrator's `AppEvent` bus, accumulates a rolling
//! transcript tail, and arbitrates between user-typed messages (HIGH
//! priority) and cadence-gated transcript windows (LOW priority), forwarding
//! each as a [`super::CopilotTurnRequest`] to the worker thread. See the
//! `live_agent` module doc for the full session lifecycle.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chat_agent::{CancelFlag, SamplerConfig};
use minutist_common::{AppEvent, ChatMessage, ChatRole, MeetingId};
use orchestrator::Orchestrator;
use persistence::ChatStore;
use settings::SettingsHandle;
use tokio::sync::{broadcast, mpsc, watch};

use super::{
    context, CopilotTurnRequest, TurnKind, UserChatRequest, UserReplyChunk, WorkerResult,
    LIVE_WINDOW_BUDGET_CHARS,
};

/// Return `true` when a digest refresh should fire.
///
/// All conditions must hold simultaneously:
/// - `new_segments >= min_segments`
/// - `elapsed_secs >= f64::from(min_seconds)`
/// - `!in_flight`
///
/// Pure: no side effects, no external state.
pub fn should_refresh(
    new_segments: u32,
    elapsed_secs: f64,
    in_flight: bool,
    min_segments: u32,
    min_seconds: u32,
) -> bool {
    !in_flight && new_segments >= min_segments && elapsed_secs >= f64::from(min_seconds)
}

/// How long the transcript cadence backs off after the ASR pipeline reports
/// its flush queue is full. The co-pilot's own turns compete for the same
/// worker cycles as ASR flush processing; when ASR is already dropping work
/// under backpressure, dispatching another automatic transcript turn would
/// compound the problem, so the cadence yields for this cooldown window
/// (deferring the turn — the accumulated tail is not discarded, so a later
/// tick picks it up once the cooldown elapses).
const ASR_BACKPRESSURE_COOLDOWN: Duration = Duration::from_secs(8);

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_driver_task(
    meeting_id: MeetingId,
    orchestrator: Arc<Orchestrator>,
    event_tx: broadcast::Sender<AppEvent>,
    settings: SettingsHandle,
    // LOW-priority transcript-turn sender.
    transcript_req_tx: mpsc::Sender<CopilotTurnRequest>,
    // HIGH-priority user-turn sender (wraps raw user messages from user_msg_rx).
    user_req_tx: mpsc::Sender<CopilotTurnRequest>,
    // User-chat requests arriving from the registry handle (via send_chat_message).
    mut user_msg_rx: mpsc::Receiver<UserChatRequest>,
    mut res_rx: mpsc::Receiver<WorkerResult>,
    shutdown: &mut watch::Receiver<bool>,
    // C2/M5: raised on shutdown to abort the worker thread's startup prefix
    // seed if it is still in progress (the ~40 s prefill).
    startup_cancel: CancelFlag,
    _meetings_dir: std::path::PathBuf,
) {
    let mut events = orchestrator.subscribe_events();

    let mut tail = String::new();
    let mut new_segments: u32 = 0;
    let mut last_refresh = Instant::now();
    let mut in_flight = false;
    let mut active_cancel: Option<CancelFlag> = None;
    // Once the held context is exhausted OR a terminal decode error occurs,
    // stop dispatching further turns.
    let mut terminal = false;
    // Set to a future instant on ASR flush-queue backpressure; the transcript
    // cadence gate (not the user-chat lane) yields until this elapses.
    // Initialised to "now" so the gate is open from the first iteration.
    let mut asr_pressure_until = Instant::now();

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        "live-agent driver started"
    );

    loop {
        // === User-priority arbitration ===
        //
        // A pending user message always beats a not-yet-started transcript
        // turn (spec B2 / §8). Drain the user lane with `try_recv` BEFORE
        // evaluating the cadence gate so a queued user message is dispatched
        // immediately on this iteration and the cadence gate is skipped.
        // This replaces the earlier approach where the cadence gate ran first
        // (causing transcript turns to preempt user turns — a priority
        // inversion), and makes the `biased` select in the worker meaningful:
        // the worker sees both lanes ready only when the driver sends on both
        // in rapid succession, which is now impossible because the driver
        // drains the user lane first.
        if !in_flight && !terminal {
            if let Ok(user_req) = user_msg_rx.try_recv() {
                // Use the cancel flag from the request so that `cancel_chat_turn`
                // registered by the command task can raise it to stop the decode loop.
                let cancel = user_req.cancel.clone();
                active_cancel = Some(cancel.clone());
                in_flight = true;
                // Flush any transcript accumulated since the last cadence turn
                // into THIS user turn, so the co-pilot answers with the meeting
                // current to now — not just to the last batch. Bounded to the
                // recent window; older transcript is already in the held context
                // (prior transcript turns) and reachable via RAG retrieval.
                let pending = std::mem::take(&mut tail);
                new_segments = 0;
                let content = compose_user_turn_content(
                    &user_req.message,
                    context::tail_chars(&pending, LIVE_WINDOW_BUDGET_CHARS),
                );
                tracing::debug!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    pending_chars = pending.len(),
                    "live-agent user turn dispatched (preempts pending transcript; \
                     flushed pending transcript into the turn)"
                );
                let req = CopilotTurnRequest {
                    kind: TurnKind::UserChat,
                    content,
                    retrieved: None,
                    sampler: SamplerConfig {
                        max_tokens: 1024,
                        ..SamplerConfig::deterministic()
                    },
                    cancel,
                    reply_tx: Some(user_req.reply_tx),
                };
                if user_req_tx.send(req).await.is_err() {
                    tracing::warn!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        "live-agent worker disappeared while forwarding user message"
                    );
                }
            }
        }

        // === Terminal-state rejection ===
        //
        // When the held context is terminal (capacity exhausted or decode
        // error), new user requests cannot be dispatched via the normal path
        // (both consumption sites above are gated `!terminal`). Drain any
        // pending requests here and reply immediately with `UserReplyChunk::Err`
        // so the drain task in `route_live_chat_message` terminates promptly,
        // clearing `chat_in_flight` and unblocking the chat UI. Without this
        // drain a queued `UserChatRequest` sits in `user_msg_rx` indefinitely
        // and `reply_rx.recv()` in the drain task never returns.
        //
        // `try_recv` is used deliberately: this path must not block the loop
        // iteration waiting for a message that may never arrive. Any request
        // already queued is drained on this pass; a request that arrives later
        // (after the `select!` below yields) is caught on the next loop iteration
        // or — if the channel is still open — by the `select!` arm below which
        // is enabled whenever `terminal && !in_flight` to make the loop
        // responsive without busy-spinning on an empty channel.
        if terminal {
            while let Ok(user_req) = user_msg_rx.try_recv() {
                tracing::debug!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "live-agent: rejecting user message after terminal state"
                );
                let _ = user_req.reply_tx.try_send(UserReplyChunk::Err(
                    "Live co-pilot paused: context window filled for this session."
                        .to_string(),
                ));
            }
        }

        // === Transcript cadence gate ===
        //
        // Only fires when no request is already in flight (including a user
        // turn that was just dispatched above).
        if !in_flight && !terminal {
            let s = settings.current();
            let elapsed = last_refresh.elapsed().as_secs_f64();
            let cadence_due = should_refresh(
                new_segments,
                elapsed,
                in_flight,
                s.live_agent_min_segments,
                s.live_agent_min_seconds,
            );
            if cadence_due && Instant::now() < asr_pressure_until {
                // ASR is behind (flush queue full); defer this turn rather than
                // compound the backpressure. `tail`/`new_segments` are left
                // accumulated so the next cadence check (once the cooldown
                // elapses) picks up everything gathered in the meantime.
                tracing::debug!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "live-agent: transcript turn skipped; ASR backpressure cooldown active"
                );
            } else if cadence_due {
                let cancel = CancelFlag::new();
                active_cancel = Some(cancel.clone());
                in_flight = true;
                new_segments = 0;
                // Consume the accumulated tail. On a terminal error the taken
                // tail is not restored — the session ends and no retry is possible.
                let tail_snapshot = std::mem::take(&mut tail);

                match transcript_req_tx
                    .send(CopilotTurnRequest {
                        kind: TurnKind::Transcript,
                        content: tail_snapshot,
                        retrieved: None, // retrieval is built inside the worker
                        sampler: SamplerConfig {
                            max_tokens: 1024,
                            ..SamplerConfig::deterministic()
                        },
                        cancel,
                        reply_tx: None,
                    })
                    .await
                {
                    Ok(()) => {
                        tracing::debug!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent transcript turn dispatched"
                        );
                    }
                    Err(_) => {
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent worker thread disappeared; stopping driver"
                        );
                        let _ = event_tx.send(AppEvent::LiveDigestError {
                            meeting_id,
                            message: "Live co-pilot stopped: the agent could not start."
                                .to_string(),
                        });
                        return;
                    }
                }
            }
        }

        tokio::select! {
            biased;

            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    tracing::info!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        "live-agent driver received shutdown signal"
                    );
                    // C2/M5: raise startup_cancel so the worker's prefix seed
                    // (if still running) aborts promptly, unblocking the join.
                    startup_cancel.cancel();
                    if let Some(c) = active_cancel.take() {
                        c.cancel();
                    }
                    return;
                }
            }

            // A raw user message arrived from the registry handle but was not
            // caught by the try_recv above (e.g. arrived after the try_recv but
            // before the select, or while in_flight was true).
            //
            // When the driver is not terminal: wrap the request into a
            // CopilotTurnRequest and forward on the HIGH-priority channel.
            //
            // When the driver is terminal (context exhausted or decode error):
            // reject the request immediately via its embedded reply_tx so the
            // drain task in `route_live_chat_message` terminates and clears
            // both `chat_in_flight` and the chat UI. The guard includes
            // `terminal` (enabling the arm when terminal && !in_flight) so the
            // select wakes up promptly for newly queued requests even after the
            // context is exhausted — otherwise the select would sleep until
            // another arm fires, delaying the rejection.
            user_msg = user_msg_rx.recv(), if !in_flight => {
                match user_msg {
                    Some(user_req) if terminal => {
                        // Terminal: reject the request and let the drain task
                        // clear chat_in_flight / chat_cancel.
                        tracing::debug!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent: rejecting user message after terminal state"
                        );
                        let _ = user_req.reply_tx.try_send(UserReplyChunk::Err(
                            "Live co-pilot paused: context window filled for this session."
                                .to_string(),
                        ));
                    }
                    Some(user_req) => {
                        // Use the cancel flag from the request so that `cancel_chat_turn`
                        // registered by the command task can reach the decode loop.
                        let cancel = user_req.cancel.clone();
                        active_cancel = Some(cancel.clone());
                        in_flight = true;
                        let req = CopilotTurnRequest {
                            kind: TurnKind::UserChat,
                            content: user_req.message,
                            retrieved: None,
                            sampler: SamplerConfig {
                                max_tokens: 1024,
                                ..SamplerConfig::deterministic()
                            },
                            cancel,
                            reply_tx: Some(user_req.reply_tx),
                        };
                        if user_req_tx.send(req).await.is_err() {
                            tracing::warn!(
                                target: "ipc-bridge",
                                meeting_id = %meeting_id.0,
                                "live-agent worker disappeared while forwarding user message"
                            );
                        }
                    }
                    None => {
                        // The registry handle was dropped (session ended).
                        tracing::debug!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent user message channel closed"
                        );
                    }
                }
            }

            result = res_rx.recv() => {
                in_flight = false;
                active_cancel = None;
                last_refresh = Instant::now();
                match result {
                    Some(WorkerResult::Message { role_is_user_reply, content }) => {
                        // Transcript-triggered replies surface via the co-pilot feed
                        // (LiveCopilotMessage → LiveDigestPanel). User-chat replies
                        // stream back on reply_tx (ChatToken/ChatTurnComplete events)
                        // and render in the chat panel — emitting LiveCopilotMessage for
                        // those would double-render the reply in the feed.
                        if role_is_user_reply {
                            tracing::debug!(
                                target: "ipc-bridge",
                                meeting_id = %meeting_id.0,
                                "live-agent: user-chat reply streamed via reply_tx (no co-pilot feed event)"
                            );
                        } else {
                            tracing::debug!(
                                target: "ipc-bridge",
                                meeting_id = %meeting_id.0,
                                "live-agent: surfacing transcript-triggered co-pilot message"
                            );
                            let _ = event_tx.send(AppEvent::LiveCopilotMessage {
                                meeting_id,
                                turn_id: 0,
                                role: ChatRole::Assistant,
                                content,
                            });
                        }
                    }
                    Some(WorkerResult::Suppressed) => {
                        // Transcript turn produced the NOOP sentinel — nothing
                        // worth surfacing; no event emitted.
                        tracing::debug!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent: transcript turn suppressed (NOOP)"
                        );
                    }
                    Some(WorkerResult::Err(e)) => {
                        // A decode error leaves the held context untrustworthy
                        // (M1/M2 in live.rs). Terminal: emit one error event and
                        // stop all further turns.
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent turn error (terminal): {e}"
                        );
                        terminal = true;
                        let _ = event_tx.send(AppEvent::LiveDigestError {
                            meeting_id,
                            message: format!(
                                "Live co-pilot paused: inference error. Error: {e}"
                            ),
                        });
                    }
                    Some(WorkerResult::CapacityExhausted(e)) => {
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent context capacity exhausted: {e}; \
                             no further turns for this session"
                        );
                        terminal = true;
                        let _ = event_tx.send(AppEvent::LiveDigestError {
                            meeting_id,
                            message: "Live co-pilot paused: context window filled for this \
                                 session."
                                .to_string(),
                        });
                    }
                    None => {
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent worker result channel closed unexpectedly"
                        );
                        return;
                    }
                }
            }

            event = events.recv() => {
                match event {
                    Ok(AppEvent::TranscriptSegment { meeting_id: mid, segment })
                        if mid == meeting_id =>
                    {
                        tail.push_str(&segment.text);
                        tail.push('\n');
                        new_segments += 1;
                    }
                    Ok(AppEvent::StateChanged { state }) => {
                        use minutist_common::RecordingState;
                        match state {
                            RecordingState::Recording { meeting_id: mid, .. }
                            | RecordingState::Paused { meeting_id: mid, .. }
                                if mid == meeting_id => {}
                            _ => {
                                tracing::info!(
                                    target: "ipc-bridge",
                                    meeting_id = %meeting_id.0,
                                    "live-agent: recording left active state; stopping"
                                );
                                startup_cancel.cancel();
                                if let Some(c) = active_cancel.take() {
                                    c.cancel();
                                }
                                return;
                            }
                        }
                    }
                    Ok(AppEvent::AsrBackpressure { meeting_id: mid }) if mid == meeting_id => {
                        asr_pressure_until = Instant::now() + ASR_BACKPRESSURE_COOLDOWN;
                        tracing::debug!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            cooldown_secs = ASR_BACKPRESSURE_COOLDOWN.as_secs(),
                            "live-agent: ASR flush-queue backpressure observed; \
                             pausing the transcript cadence"
                        );
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            dropped = n,
                            "live-agent subscriber lagged; some TranscriptSegment events dropped"
                        );
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::info!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent event channel closed; stopping driver"
                        );
                        startup_cancel.cancel();
                        return;
                    }
                }
            }
        }
    }
}

/// Persist one conversational turn to the meeting's live `ChatSession`.
///
/// `turn_id` is the caller's monotonic counter — pass `&mut turn_id` and the
/// helper increments it after each successful append so the caller's counter
/// stays in sync with the stored message sequence.
///
/// Best-effort: any persistence error is logged and swallowed so a storage
/// hiccup never breaks the live co-pilot stream.
pub(crate) fn persist_turn(
    meetings_dir: &Path,
    meeting_id: MeetingId,
    role: ChatRole,
    content: &str,
    turn_id: &mut u64,
) {
    let now = chrono::Utc::now().to_rfc3339();
    let id = *turn_id;
    let persisted = (|| -> minutist_common::AppResult<()> {
        let mut session = ChatStore::load_or_create_live(meetings_dir, meeting_id, &now)?;
        session.messages.push(ChatMessage {
            role,
            content: content.to_string(),
            tool_name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            turn_id: id,
        });
        session.updated_at = now.clone();
        ChatStore::save(meetings_dir, meeting_id, &session)
    })();
    match persisted {
        Ok(()) => {
            *turn_id += 1;
        }
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                turn_id = id,
                "live-agent: persist turn failed: {e}"
            );
        }
    }
}

/// Compose a user-chat turn's content, prepending any transcript that has
/// accumulated since the last cadence turn (`pending`) so the co-pilot answers
/// with the meeting current to now, not only to the last batch. `pending` is
/// already bounded by the caller; empty pending yields the message unchanged.
/// (`build_turn_content` sanitises the whole result — the transcript is
/// untrusted content.)
pub(crate) fn compose_user_turn_content(message: &str, pending: &str) -> String {
    if pending.trim().is_empty() {
        message.to_string()
    } else {
        format!(
            "Most recent meeting transcript, not yet processed:\n{pending}\n\n\
             User message: {message}"
        )
    }
}
