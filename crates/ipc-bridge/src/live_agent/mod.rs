//! Live in-meeting co-pilot driver (Phase 9 / U2 merged-input loop).
//!
//! [`spawn_live_agent`] is called by the recording-start path when
//! `live_agent_should_run(mode, gpu_probe, gpu_acceleration)` returns `true`.
//! It drives a **single keep-alive [`LiveSession`]** for the entire meeting,
//! feeding both transcript windows and user-typed messages into one held KV
//! context via `session.converse` — no prune-to-prefix between turns.
//!
//! # Session lifecycle
//!
//! 1. Subscribe to [`AppEvent::TranscriptSegment`] for the recording's meeting id.
//! 2. Accumulate a rolling transcript tail in a text buffer.
//! 3. Gate transcript turns on the settings-backed cadence gate ([`should_refresh`]).
//! 4. On a cadence fire, send a [`CopilotTurnRequest`] (kind [`TurnKind::Transcript`])
//!    on the LOW-priority transcript channel to the dedicated `std::thread` that owns
//!    the `!Send` [`LiveSession`].
//! 5. User-typed messages arrive via [`LiveCopilotHandle::user_tx`] on the
//!    HIGH-priority channel; the worker's `tokio::select! { biased; }` drains user
//!    turns before transcript turns.
//! 6. The worker calls `session.converse("user", content, ...)`, applies the
//!    NOOP-sentinel response policy, and replies with a [`WorkerResult`].
//! 7. A non-suppressed [`WorkerResult::Message`] from a Transcript turn emits
//!    [`AppEvent::LiveCopilotMessage`] and is persisted to the live `ChatSession`.
//!    A non-suppressed message from a UserChat turn streams back on the
//!    request's `reply_tx` ([`UserReplyChunk`]) instead; the `send_chat_message`
//!    command task drains those chunks and emits `ChatToken`/`ChatTurnComplete`
//!    events on the broadcast bus so the chat panel renders the reply.
//!
//! # Threading
//!
//! The Tauri async task (spawned by [`spawn_live_agent`]) owns the event loop
//! and the tail buffer. A dedicated `std::thread` owns the `!Send`
//! [`chat_agent::LlamaLiveBackend`] / [`chat_agent::LiveSession`] for the
//! session lifetime. The user (HIGH) and transcript (LOW) channels are each
//! bounded depth-1 `tokio::sync::mpsc` channels; the driver fires a new request
//! only after receiving the previous result, enforcing single-in-flight without
//! a mutex.
//!
//! # Registry and chat routing
//!
//! `spawn_live_agent` inserts a [`LiveCopilotHandle`] into the caller-supplied
//! registry (`Arc<Mutex<HashMap<MeetingId, LiveCopilotHandle>>>`) on start and
//! removes it on teardown. The registry is stored on [`crate::IpcState`] so the
//! `send_chat_message` command can route user messages to the live co-pilot when
//! the target meeting is currently recording. The command resolves the live
//! [`minutist_common::ChatSessionId`] via `ChatStore::find_live`, sends a
//! [`UserChatRequest`] on [`LiveCopilotHandle::user_tx`], and spawns a task that
//! drains the reply channel into `ChatToken` / `ChatTurnComplete` / `ChatError`
//! broadcast events — the same events the post-meeting `LlamaTurnBackend` path
//! emits, so the chat UI renders the reply without any per-path branching.
//!
//! # Prefix and retrieval
//!
//! The prefix (`build_prefix`) carries the system prompt and, when attachments
//! with awareness are present, a compact per-attachment list
//! ("## Attached documents (retrieve details on demand)"). Awareness summaries
//! are generated at attach time and loaded from the manifest at worker startup.
//! Attachment detail and earlier-transcript context is NOT pinned: each turn
//! retrieves the chunks relevant to the current discussion (dense + lexical
//! over the meeting's `meeting.db`, fused by RRF) and injects them as a leading
//! block in the turn content. A tier-scaled `k` keeps the per-turn prefill small
//! on an integrated GPU and generous on a discrete one. The held embedder loads
//! in the background at worker start; until it is ready (or while `meeting.db`
//! is empty) the agent degrades to transcript-only context.
//!
//! # Cadence gate
//!
//! [`should_refresh`] is a **pure** function (no side effects, fully unit-tested):
//! returns `true` when:
//! - `new_segments >= min_segments`, AND
//! - `elapsed_secs >= min_seconds as f64`, AND
//! - `!in_flight`.
//!
//! The AND gate (not OR) prevents premature turns during sparse meetings.
//!
//! # Context capacity policy
//!
//! The worker tracks whether the held context has reached capacity. On a
//! [`chat_agent::Error::ContextOverflow`] the session emits one
//! `LiveDigestError` noting capacity is exhausted and sets a permanent
//! `terminal` flag that stops all further turns for the session.
//! This is the v1 policy: no re-seed mid-recording (re-seeding costs another
//! ~40 s prefill and would starve ASR inference).
//!
//! After the terminal flag is set, the driver's select arm (gated
//! `if !in_flight`) remains enabled for `user_msg_rx` and immediately rejects
//! any incoming [`UserChatRequest`] with a [`UserReplyChunk::Err`], so the
//! drain task in `send_chat_message` terminates promptly and clears both
//! `chat_in_flight` and `chat_cancel` rather than hanging indefinitely.
//!
//! # Module layout
//!
//! - This module: the crate-facing surface (wire types, [`LiveAgentHandles`],
//!   [`spawn_live_agent`]) and the small shared constants/framing helpers.
//! - [`driver`]: the async driver task that owns the event loop, the cadence
//!   gate, and turn persistence.
//! - [`worker`]: the dedicated `!Send` worker thread that owns the
//!   [`LiveSession`] and runs each turn to completion.
//! - [`context`]: prefix construction, RAG retrieval, and the eviction recap
//!   loader — the context-assembly helpers the worker calls into.
//! - [`test_support`] / `tests`: stub backends and the unit test suite.

mod context;
mod driver;
mod worker;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests;

// Blanket re-export so cross-submodule call sites, and (under `#[cfg(test)]`)
// `tests.rs`'s `use super::*;`, reach every `pub(crate)` item by its original
// flat name regardless of which submodule now holds it. rustc's unused_imports
// lint cannot see the only-under-cfg(test) consumer of this glob (removing it
// breaks `cargo test -p ipc-bridge`), hence the explicit allow.
#[allow(unused_imports)]
pub(crate) use context::*;
#[allow(unused_imports)]
pub(crate) use driver::*;
#[allow(unused_imports)]
pub(crate) use worker::*;

pub use driver::should_refresh;

use std::path::PathBuf;
use std::sync::Arc;

use chat_agent::CancelFlag;
use minutist_common::{AppEvent, Embedder, MeetingId};
use orchestrator::Orchestrator;
use settings::SettingsHandle;
use summariser::LlamaSummariser;
use tokio::sync::{broadcast, mpsc, watch, OnceCell};

/// Depth of the request (user + transcript) and result channels. Depth 1
/// enforces single-in-flight: the driver never sends a second request before
/// receiving the previous result.
const WORKER_CHANNEL_DEPTH: usize = 1;

/// One chunk in the streaming reply from the live co-pilot to the
/// `send_chat_message` command. The command spawns a drain task that converts
/// these into [`minutist_common::AppEvent`] chat events on the broadcast bus.
///
/// `Token` chunks are best-effort hints; `Done` is authoritative (its payload
/// is the full reconciled text). A dropped `Token` is harmless because the
/// `ChatTurnComplete` event carries `final_text`.
#[derive(Debug)]
pub enum UserReplyChunk {
    /// One streamed token (or token fragment) from the decode loop.
    Token(String),
    /// The turn completed. `0` is the full reply text.
    Done(String),
    /// The turn failed. `0` is a human-readable error description.
    Err(String),
}

/// A user-chat turn request sent from `send_chat_message` into the live worker
/// via [`LiveCopilotHandle::user_tx`].
pub struct UserChatRequest {
    /// The verbatim message the user typed.
    pub message: String,
    /// Bounded channel the worker uses to stream the reply back. The command
    /// task drains this and emits `ChatToken` / `ChatTurnComplete` / `ChatError`
    /// on the broadcast bus. Depth 32 — each send is `try_send` (tokens are
    /// dropped on full; `Done`/`Err` block because they are authoritative).
    pub reply_tx: mpsc::Sender<UserReplyChunk>,
    /// Per-turn cancel flag registered in `IpcState::chat_cancel` by the
    /// command task. The driver forwards it onto `CopilotTurnRequest::cancel`
    /// so that `cancel_chat_turn` reaches the worker's decode loop.
    pub cancel: CancelFlag,
}

/// A lightweight handle to the live co-pilot for one meeting, stored in the
/// per-session registry on [`crate::IpcState`].
///
/// `spawn_live_agent` inserts this into the registry on start and removes it
/// on teardown; the `send_chat_message` command checks the registry to
/// determine whether to route the user message to the live co-pilot (when the
/// target meeting is currently recording) or to the post-meeting
/// `LlamaTurnBackend` path.
pub struct LiveCopilotHandle {
    /// The HIGH-priority user-chat input channel into the live worker. Send a
    /// [`UserChatRequest`] here to inject a user turn; the worker drains this
    /// channel before the LOW-priority transcript channel on each `biased`
    /// select and streams the reply back on the request's `reply_tx`.
    pub user_tx: mpsc::Sender<UserChatRequest>,
}

// ---------------------------------------------------------------------------
// Chat-template framing (#0022)
//
// The held LLM is instruction-tuned Gemma. `llama-cpp-2` cannot render Gemma's
// baked template via `apply_chat_template` for the held-context split, so the
// driver hand-assembles the turn markers. The prefix is a complete, closed user
// turn (`<bos>{open}user\n{system}{close}\n`); `append_turn` (via `converse_typed`)
// then appends each subsequent turn on top of the growing KV.
//
// Turn markers are NOT hardcoded — they are derived from the model vocabulary
// via `chat_agent::detect_turn_markers` at worker-thread start. Gemma 4 uses
// `<|turn>` / `<turn|>` (single control tokens 105/106); Gemma 2/3 uses
// `<start_of_turn>` / `<end_of_turn>`. The `TurnMarkers` value is threaded
// into `build_prefix` and `sanitise_untrusted` so
// neither path ever bakes a model-specific string at compile time.
// ---------------------------------------------------------------------------

/// Cap on the recent-transcript window fed per turn, in characters
/// (≈ `chars / 4` tokens). Bounds the transcript text included in each
/// `TurnKind::Transcript` turn. Older transcript that scrolls past this cap is
/// recovered on demand by the RAG retrieval layer, which injects relevant earlier
/// turns as a leading block in the turn content (see `build_retrieval_block`).
pub(crate) const LIVE_WINDOW_BUDGET_CHARS: usize = 8_000;

/// Number of most-recent User + Assistant turns to include in the recap block
/// prepended after a KV eviction. Older turns are dropped from the recap (but
/// remain in the persisted log and the RAG index, recoverable on demand).
pub(crate) const EVICT_RECAP_TURNS: usize = 8;

/// Approximate character budget for the whole recap block. The recap is trimmed
/// by dropping older turns first (most-recent-first ordering ensures the
/// most-relevant context survives the cap). Chosen to stay well within the
/// prefix token headroom after eviction.
///
/// v2 (rolling-summary layered budget) is deferred: rather than only keeping
/// last-K verbatim, v2 would summarise the evicted middle and prepend that
/// summary. The infrastructure (`reset_to_prefix`, the recap header path) is in
/// place; the summarisation call is the missing piece.
pub(crate) const EVICT_RECAP_CHARS: usize = 4_000;

/// Per-line character cap applied to each recap entry before the whole-block cap.
/// Prevents a single very long turn from consuming the entire recap budget.
pub(crate) const EVICT_RECAP_LINE_CAP: usize = 500;

/// Token headroom added to every token-count estimate in the eviction trigger.
///
/// `append_turn` tokenises the full framing (turn markers + newlines + content);
/// the framing markers alone add ~6–16 tokens depending on the model. This
/// margin absorbs marker overhead and the inherent ±error in the chars/3
/// heuristic, ensuring the trigger fires conservatively early rather than
/// letting a boundary turn slip past `has_room_for` and hit the hard
/// `ContextOverflow` guard in `append_turn`.
pub(crate) const FRAMING_TOKEN_MARGIN: usize = 32;

/// Neutralise chat-control token strings in untrusted content so they tokenise
/// as ordinary text, not special tokens. Inserts a space after the `<` of each
/// marker — enough to break the exact-string special-token match while staying
/// human-readable. A no-op (no allocation) when no marker is present.
///
/// The set of tokens to neutralise is derived from `markers` (the model's
/// actual turn boundaries) plus the universal BOS/EOS strings.
pub(crate) fn sanitise_untrusted(s: &str, markers: &chat_agent::TurnMarkers) -> String {
    // Always neutralise BOS/EOS regardless of model.
    let control: &[&str] = &["<bos>", "<eos>"];
    let model_markers = [markers.turn_open.as_str(), markers.turn_close.as_str()];
    let all: Vec<&str> = control.iter().copied().chain(model_markers).collect();
    if all.iter().any(|t| s.contains(t)) {
        let mut out = s.to_string();
        for tok in &all {
            out = out.replace(tok, &tok.replacen('<', "< ", 1));
        }
        out
    } else {
        s.to_string()
    }
}

/// Distinguishes the two input lanes into the worker.
///
/// Both kinds are delivered as `"user"` role to the model, but the framing
/// and response policy differ: transcript turns apply the NOOP-sentinel policy;
/// user-chat turns always produce a visible reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnKind {
    /// A user-typed message. Always yields a [`WorkerResult::Message`].
    UserChat,
    /// An auto-injected transcript window. Yields [`WorkerResult::Suppressed`]
    /// when the model replies with [`COPILOT_NOOP_SENTINEL`] or empty text.
    Transcript,
}

/// A turn request sent from the driver to the worker.
pub(crate) struct CopilotTurnRequest {
    /// Whether this is a transcript auto-injection or a user message.
    pub(crate) kind: TurnKind,
    /// Raw content: the new transcript window text, or the user's message verbatim.
    pub(crate) content: String,
    /// Pre-built RAG context block (already sanitised), or `None` when the
    /// embedder is not yet ready or retrieval returned nothing.
    pub(crate) retrieved: Option<String>,
    pub(crate) sampler: chat_agent::SamplerConfig,
    pub(crate) cancel: CancelFlag,
    /// For `TurnKind::UserChat` turns only: the channel the worker uses to
    /// stream the reply back to the `send_chat_message` command. `None` for
    /// transcript turns (those surface via `AppEvent::LiveCopilotMessage`).
    pub(crate) reply_tx: Option<mpsc::Sender<UserReplyChunk>>,
}

/// The result of one worker turn.
#[derive(Debug)]
pub(crate) enum WorkerResult {
    /// A visible assistant reply to surface to the user.
    Message {
        /// `false` for transcript-triggered assistant replies; `true` for
        /// replies that echo a user-chat turn (role distinction for the UI).
        role_is_user_reply: bool,
        content: String,
    },
    /// The transcript turn produced the NOOP sentinel or empty text — nothing
    /// worth surfacing.
    Suppressed,
    /// The held context reached capacity. No further turns are possible for
    /// this session.
    CapacityExhausted(String),
    /// A terminal decode error. The held context is untrustworthy.
    Err(String),
}

/// The sentinel string the model returns when a transcript window contains
/// nothing worth surfacing to the user.
pub(crate) const COPILOT_NOOP_SENTINEL: &str = "<<NOOP>>";

/// Handles required by the live-agent driver.
pub struct LiveAgentHandles {
    pub orchestrator: Arc<Orchestrator>,
    pub meetings_dir: PathBuf,
    pub event_tx: broadcast::Sender<AppEvent>,
    pub settings: SettingsHandle,
    /// The lazily-loaded held LLM substrate (shared with the chat and
    /// summarise paths). The worker thread calls `ensure_summariser` on this
    /// to obtain the `Arc<LlamaSummariser>` it borrows `&LlamaModel` from.
    pub summariser: Arc<OnceCell<Arc<LlamaSummariser>>>,
    /// The lazily-loaded held BGE-M3 embedder (the SAME `Arc<OnceCell>` the chat
    /// and RAG write paths share). The worker loads it in the background at start
    /// and peeks it each refresh to embed the retrieval query; `None` until ready.
    pub embedder: Arc<OnceCell<Arc<dyn Embedder>>>,
}

/// Spawn the live-agent auto-driver task for an active recording.
///
/// The task exits when:
/// - `shutdown` flips to `true`, OR
/// - the orchestrator event channel closes, OR
/// - the worker thread disappears.
///
/// The caller raises `shutdown` when the recording leaves Recording/Paused.
///
/// On success the task inserts a [`LiveCopilotHandle`] into `registry` (keyed
/// by `meeting_id`) and removes it when the driver exits. The handle exposes
/// the HIGH-priority user-chat channel so the `send_chat_message` command can
/// route user turns directly to the live co-pilot during an active recording.
pub fn spawn_live_agent(
    handles: LiveAgentHandles,
    meeting_id: MeetingId,
    mut shutdown: watch::Receiver<bool>,
    registry: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<MeetingId, LiveCopilotHandle>>>,
) {
    let LiveAgentHandles {
        orchestrator,
        meetings_dir,
        event_tx,
        settings,
        summariser,
        embedder,
    } = handles;

    // Two bounded channels into the worker: HIGH = user chat, LOW = transcript.
    // Both depth-1 so the driver never queues more than one pending request per
    // lane; the biased select in the worker ensures a user turn is consumed
    // before any pending transcript turn.
    let (user_req_tx, user_req_rx) = mpsc::channel::<CopilotTurnRequest>(WORKER_CHANNEL_DEPTH);
    let (transcript_req_tx, transcript_req_rx) = mpsc::channel::<CopilotTurnRequest>(WORKER_CHANNEL_DEPTH);
    let (res_tx, res_rx) = mpsc::channel::<WorkerResult>(WORKER_CHANNEL_DEPTH);

    // The user-lane sender is exposed through the handle inserted into the registry.
    // The driver task wraps UserChatRequests from the registry handle into full
    // CopilotTurnRequests (carrying the reply_tx) before forwarding on user_req_tx.
    let (user_msg_tx, user_msg_rx) = mpsc::channel::<UserChatRequest>(WORKER_CHANNEL_DEPTH);

    // Clone the fields needed for model loading and prefix building inside the
    // worker thread.
    let worker_orchestrator = orchestrator.clone();
    let worker_settings = settings.clone();
    let worker_meetings_dir = meetings_dir.clone();
    // U1: the driver persists each digest into the meeting's live co-pilot
    // session (ChatStore root = the meetings dir), so it needs its own handle.
    let driver_meetings_dir = meetings_dir.clone();

    // C2/M5: the startup cancel flag is shared between the driver (which raises
    // it on shutdown) and the worker thread (which uses it as the cancel token
    // for the ~40 s prefix seed). A Stop during the seed therefore aborts it
    // promptly instead of blocking the join for up to ~40 s.
    let startup_cancel = CancelFlag::new();
    let driver_startup_cancel = startup_cancel.clone();

    let join_handle = match std::thread::Builder::new()
        .name(format!("live-agent-{}", meeting_id.0))
        .spawn(move || {
            worker::run_worker_thread(
                meeting_id,
                user_req_rx,
                transcript_req_rx,
                res_tx,
                summariser,
                embedder,
                worker_orchestrator,
                worker_settings,
                worker_meetings_dir,
                startup_cancel,
            )
        }) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "failed to spawn live-agent worker thread: {e}"
            );
            return;
        }
    };

    // Insert the handle into the registry before the driver task starts so
    // the chat-command path can route user messages as soon as the recording
    // becomes live. The driver task removes it on exit.
    {
        let mut reg = registry.lock().expect("live_copilot_handles poisoned");
        reg.insert(
            meeting_id,
            LiveCopilotHandle {
                user_tx: user_msg_tx,
            },
        );
    }

    tauri::async_runtime::spawn(async move {
        driver::run_driver_task(
            meeting_id,
            orchestrator,
            event_tx,
            settings,
            transcript_req_tx,
            user_req_tx,
            user_msg_rx,
            res_rx,
            &mut shutdown,
            driver_startup_cancel,
            driver_meetings_dir,
        )
        .await;
        tracing::info!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "live-agent driver task exited; joining worker thread"
        );
        // Remove the handle from the registry so no new messages are queued
        // after the driver exits.
        {
            let mut reg = registry.lock().expect("live_copilot_handles poisoned");
            reg.remove(&meeting_id);
        }
        // M6: join the worker thread so it is reaped, not leaked. The driver
        // has already signalled the cancel flag (or the worker's req channel
        // dropped naturally) before this point; the join simply waits for the
        // thread to observe the cancel and return.
        if let Err(e) = join_handle.join() {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker thread panicked: {e:?}"
            );
        }
    });
}
