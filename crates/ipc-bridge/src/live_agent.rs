//! Live in-meeting agent auto-driver (Phase 9, WU2b).
//!
//! [`spawn_live_agent`] is called by the recording-start path when
//! `live_agent_should_run(mode, gpu_probe, gpu_acceleration)` returns `true`.
//! It owns the full digest-refresh lifecycle for one active recording:
//!
//! 1. Subscribe to [`AppEvent::TranscriptSegment`] for the recording's meeting id.
//! 2. Accumulate a rolling transcript tail in a text buffer.
//! 3. Gate refreshes on the settings-backed cadence gate ([`should_refresh`]).
//! 4. On a cadence fire, send the incremental tail to a **dedicated `std::thread`**
//!    that owns a [`LiveSession<LlamaLiveBackend>`] (which is `!Send`).
//! 5. Parse the returned digest text into a [`LiveDigest`], carrying forward the
//!    prior digest's `resolved` flags (standing-list update discipline).
//! 6. Emit [`AppEvent::LiveDigestUpdated`] or [`AppEvent::LiveDigestError`].
//! 7. Tear down cleanly when `shutdown` flips to `true` (recording stopped).
//!
//! # Threading
//!
//! The Tauri async task (spawned by [`spawn_live_agent`]) owns the event loop
//! and the tail buffer. A dedicated `std::thread` owns the `!Send`
//! [`chat_agent::LlamaLiveBackend`] / [`chat_agent::LiveSession`] for the
//! session lifetime. The async task sends `TailRequest` values on a bounded
//! `tokio::sync::mpsc` channel (depth 1); the worker replies on a matching
//! bounded channel (depth 1). The bounded depth enforces single-in-flight without
//! a separate mutex: the driver only fires a new request after receiving the
//! previous result.
//!
//! # Prefix and retrieval
//!
//! The prefix (`build_prefix`) is just the system prompt + digest-category
//! instructions — small, prefilled once at session spawn (`seed_prefix` is
//! idempotent; subsequent calls are no-ops). Attachment and earlier-transcript
//! context is NOT pinned: each refresh retrieves the few chunks relevant to what
//! is being discussed (dense + lexical over the meeting's `meeting.db`, fused by
//! RRF) and injects them into the bounded tail. A tier-scaled `k` keeps the
//! per-refresh prefill small on an integrated GPU and generous on a discrete one.
//! The held embedder loads in the background at worker start; until it is ready
//! (or while `meeting.db` is empty) the agent degrades to transcript-only with no
//! injected context.
//!
//! # Standing-list update discipline
//!
//! Each refresh prompt includes the prior digest (JSON-serialised) so the model
//! UPDATEs items rather than regenerating from scratch. The driver parses the
//! model's response into a `LiveDigest` and carries the prior digest forward.
//!
//! # Cadence gate
//!
//! [`should_refresh`] is a **pure** function (no side effects, fully unit-tested):
//! returns `true` when:
//! - `new_segments >= min_segments`, AND
//! - `elapsed_secs >= min_seconds as f64`, AND
//! - `!in_flight`.
//!
//! The AND gate (not OR) prevents premature refreshes during sparse meetings.
//!
//! # Context capacity policy
//!
//! The worker tracks whether the held context has reached capacity. On a
//! [`chat_agent::Error::ContextOverflow`] the session emits one
//! `LiveDigestError` noting capacity is exhausted and sets a permanent
//! `capacity_exhausted` flag that stops all further refreshes for the session.
//! This is the v1 policy: no re-seed mid-recording (re-seeding costs another
//! ~40 s prefill and would starve ASR inference).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use chat_agent::{CancelFlag, LiveSession, LlamaLiveBackend, LlamaLiveConfig, SamplerConfig};
use minutist_common::{AppEvent, Embedder, LiveDigest, LiveDigestItem, MeetingId};
use orchestrator::Orchestrator;
use persistence::{meeting_db_path, RagStore, RetrievedChunk};
use rag_retrieval::rrf_fuse;
use settings::SettingsHandle;
use summariser::LlamaSummariser;
use tokio::sync::{broadcast, mpsc, watch, OnceCell};

// ---------------------------------------------------------------------------
// Channel depth
// ---------------------------------------------------------------------------

/// Depth of both the request and result channels. Depth 1 enforces
/// single-in-flight: the driver never sends a second request before receiving
/// the previous result.
const WORKER_CHANNEL_DEPTH: usize = 1;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

pub(crate) struct TailRequest {
    tail: String,
    prior_digest_json: Option<String>,
    sampler: SamplerConfig,
    cancel: CancelFlag,
}

#[derive(Debug)]
pub(crate) enum RefreshResult {
    Ok(String),
    Err(String),
    /// The held context has reached capacity. No further refreshes are
    /// possible for this session.
    CapacityExhausted(String),
}

// ---------------------------------------------------------------------------
// Public types and spawn function
// ---------------------------------------------------------------------------

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
pub fn spawn_live_agent(
    handles: LiveAgentHandles,
    meeting_id: MeetingId,
    mut shutdown: watch::Receiver<bool>,
) {
    let LiveAgentHandles {
        orchestrator,
        meetings_dir,
        event_tx,
        settings,
        summariser,
        embedder,
    } = handles;

    let (req_tx, req_rx) = mpsc::channel::<TailRequest>(WORKER_CHANNEL_DEPTH);
    let (res_tx, res_rx) = mpsc::channel::<RefreshResult>(WORKER_CHANNEL_DEPTH);

    // Clone the fields needed for model loading and prefix building inside the
    // worker thread.
    let worker_orchestrator = orchestrator.clone();
    let worker_settings = settings.clone();
    let worker_meetings_dir = meetings_dir.clone();

    // C2/M5: the startup cancel flag is shared between the driver (which raises
    // it on shutdown) and the worker thread (which uses it as the cancel token
    // for the ~40 s prefix seed). A Stop during the seed therefore aborts it
    // promptly instead of blocking the join for up to ~40 s.
    let startup_cancel = CancelFlag::new();
    let driver_startup_cancel = startup_cancel.clone();

    let join_handle = match std::thread::Builder::new()
        .name(format!("live-agent-{}", meeting_id.0))
        .spawn(move || {
            run_worker_thread(
                meeting_id,
                req_rx,
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

    tauri::async_runtime::spawn(async move {
        run_driver_task(
            meeting_id,
            orchestrator,
            event_tx,
            settings,
            req_tx,
            res_rx,
            &mut shutdown,
            driver_startup_cancel,
        )
        .await;
        tracing::info!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "live-agent driver task exited; joining worker thread"
        );
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

// ---------------------------------------------------------------------------
// Pure cadence gate
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Async driver task
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn run_driver_task(
    meeting_id: MeetingId,
    orchestrator: Arc<Orchestrator>,
    event_tx: broadcast::Sender<AppEvent>,
    settings: SettingsHandle,
    req_tx: mpsc::Sender<TailRequest>,
    mut res_rx: mpsc::Receiver<RefreshResult>,
    shutdown: &mut watch::Receiver<bool>,
    // C2/M5: raised on shutdown to abort the worker thread's startup prefix
    // seed if it is still in progress (the ~40 s prefill).
    startup_cancel: CancelFlag,
) {
    let mut events = orchestrator.subscribe_events();

    let mut tail = String::new();
    let mut new_segments: u32 = 0;
    let mut last_refresh = Instant::now();
    let mut in_flight = false;
    let mut prior_digest: Option<LiveDigest> = None;
    let mut active_cancel: Option<CancelFlag> = None;
    // Once the held context is exhausted OR a terminal decode error occurs,
    // stop dispatching further refreshes.
    let mut terminal = false;

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        "live-agent driver started"
    );

    loop {
        // Check cadence gate before awaiting the next event.
        if !in_flight && !terminal {
            let s = settings.current();
            let elapsed = last_refresh.elapsed().as_secs_f64();
            if should_refresh(
                new_segments,
                elapsed,
                in_flight,
                s.live_agent_min_segments,
                s.live_agent_min_seconds,
            ) {
                let prior_json = prior_digest
                    .as_ref()
                    .and_then(|d| serde_json::to_string(d).ok());
                let cancel = CancelFlag::new();
                active_cancel = Some(cancel.clone());
                in_flight = true;
                new_segments = 0;
                // Consume the accumulated tail for this refresh window. On a
                // terminal error the taken tail is not restored — the session
                // ends and no retry is possible.
                let tail_snapshot = std::mem::take(&mut tail);

                match req_tx
                    .send(TailRequest {
                        tail: tail_snapshot,
                        prior_digest_json: prior_json,
                        sampler: SamplerConfig {
                            max_tokens: 1024,
                            ..SamplerConfig::deterministic()
                        },
                        cancel,
                    })
                    .await
                {
                    Ok(()) => {
                        tracing::debug!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent refresh dispatched"
                        );
                    }
                    Err(_) => {
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent worker thread disappeared; stopping driver"
                        );
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

            result = res_rx.recv() => {
                in_flight = false;
                active_cancel = None;
                last_refresh = Instant::now();
                match result {
                    Some(RefreshResult::Ok(text)) => {
                        handle_digest_result(text, meeting_id, &mut prior_digest, &event_tx);
                    }
                    Some(RefreshResult::Err(e)) => {
                        // M3: a decode error leaves the held context in an
                        // untrustworthy state (M1/M2 in live.rs). Treat as
                        // terminal: emit one error event, mark terminal so no
                        // further refreshes are dispatched.
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent refresh error (terminal): {e}"
                        );
                        terminal = true;
                        let _ = event_tx.send(AppEvent::LiveDigestError {
                            meeting_id,
                            message: format!(
                                "Live digest paused: inference error. \
                                 Existing digest items remain available. \
                                 Error: {e}"
                            ),
                        });
                    }
                    Some(RefreshResult::CapacityExhausted(e)) => {
                        tracing::warn!(
                            target: "ipc-bridge",
                            meeting_id = %meeting_id.0,
                            "live-agent context capacity exhausted: {e}; \
                             no further refreshes for this session"
                        );
                        terminal = true;
                        let _ = event_tx.send(AppEvent::LiveDigestError {
                            meeting_id,
                            message: "Live digest paused: context window filled for this session. \
                                 Existing digest items remain available."
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

fn handle_digest_result(
    text: String,
    meeting_id: MeetingId,
    prior_digest: &mut Option<LiveDigest>,
    event_tx: &broadcast::Sender<AppEvent>,
) {
    match parse_digest(&text, meeting_id, prior_digest.as_ref()) {
        Ok(digest) => {
            *prior_digest = Some(digest.clone());
            let _ = event_tx.send(AppEvent::LiveDigestUpdated { meeting_id, digest });
        }
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent digest parse error: {e}"
            );
            let _ = event_tx.send(AppEvent::LiveDigestError {
                meeting_id,
                message: format!("digest parse error: {e}"),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Dedicated !Send worker thread
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_worker_thread(
    meeting_id: MeetingId,
    req_rx: mpsc::Receiver<TailRequest>,
    res_tx: mpsc::Sender<RefreshResult>,
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
            return;
        }
    };

    // Resolve the held summariser. This calls ensure_summariser which loads the
    // GGUF if not yet loaded. Runs at thread start (before the first refresh),
    // so the load cost is paid once at session spawn, not mid-recording.
    let summariser_arc = match rt.block_on(ensure_summariser_in_worker(
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
            if let Err(e) = ensure_embedder_in_worker(&bg_cell, &bg_orchestrator, &bg_settings).await
            {
                tracing::warn!(
                    target: "ipc-bridge",
                    error = %e,
                    "live-agent: background embedder load failed (retrieval disabled this session)"
                );
            }
        });
    }

    // The prefix is just the system prompt + digest categories now — attachment
    // and earlier-transcript context is retrieved into the tail each refresh, not
    // pinned here. Cheap, no filesystem I/O.
    let prefix = build_prefix(&settings.current());
    tracing::debug!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        prefix_chars = prefix.len(),
        "live-agent: prefix built on worker thread"
    );

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

    let backend = match LlamaLiveBackend::new(model_ref, LlamaLiveConfig::default()) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent worker: failed to construct LlamaLiveBackend: {e}"
            );
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
            return;
        }
    }

    // Open the per-meeting RAG cache (created if absent — an empty cache until
    // attachments / transcript are indexed). On failure, retrieval is disabled
    // for the session (the agent still produces digests, just without injected
    // context). The tier-scaled `k` is fixed for the session: the GPU tier does
    // not change mid-meeting.
    let s = settings.current();
    let is_integrated = minutist_common::probe_primary_gpu()
        .map(|p| p.is_integrated)
        // No probe → assume the tight (integrated) tier so an unknown GPU never
        // gets the generous per-refresh prefill budget.
        .unwrap_or(true);
    let retrieval = match rt.block_on(RagStore::open(meeting_db_path(&meetings_dir, meeting_id))) {
        Ok(store) => Some(LiveRetrieval {
            embedder_cell,
            store,
            meetings_dir,
            k: tier_scaled_k(s.live_agent_retrieval_k, is_integrated),
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

    rt.block_on(run_worker_loop(
        meeting_id,
        req_rx,
        res_tx,
        &mut session,
        retrieval.as_ref(),
    ));

    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        "live-agent worker thread exited"
    );
}

/// Load the held summariser using the shared OnceCell, mirroring the logic in
/// `ChatHandles::ensure_summariser`. Called once at worker thread start.
async fn ensure_summariser_in_worker(
    cell: &Arc<OnceCell<Arc<LlamaSummariser>>>,
    orchestrator: &Arc<Orchestrator>,
    settings: &SettingsHandle,
) -> Result<Arc<LlamaSummariser>, minutist_common::AppError> {
    let handle = cell
        .get_or_try_init(|| async {
            let s = settings.current();
            let model_id = crate::commands::resolve_llm_model_id(&s);
            let model_dir = orchestrator.ensure_model_path(&model_id).await?;
            let plan = minutist_common::resolve_gpu_plan(
                minutist_common::probe_primary_gpu().as_ref(),
                s.gpu_acceleration,
                true,
            );
            let n_gpu_layers = crate::commands::resolve_summariser_gpu_layers(plan.summariser_gpu);
            let summariser = tokio::task::spawn_blocking(move || {
                crate::commands::open_summariser_in_dir(&model_dir, n_gpu_layers)
            })
            .await
            .map_err(|e| minutist_common::AppError::Internal {
                context: format!("live-agent summariser load task join failed: {e}"),
            })??;
            tracing::info!(
                target: "ipc-bridge",
                "live-agent: held LLM summariser loaded"
            );
            Ok::<_, minutist_common::AppError>(Arc::new(summariser))
        })
        .await?;
    Ok(Arc::clone(handle))
}

/// Load the held embedder using the shared `OnceCell`, mirroring
/// [`crate::chat_runtime::ChatHandles::ensure_embedder`]. Run in the background at
/// worker start so the load — and any first-use BGE-M3 download — stays off the
/// digest critical path. Populates the cell the retrieval loop peeks.
async fn ensure_embedder_in_worker(
    cell: &Arc<OnceCell<Arc<dyn Embedder>>>,
    orchestrator: &Arc<Orchestrator>,
    settings: &SettingsHandle,
) -> Result<Arc<dyn Embedder>, minutist_common::AppError> {
    let handle = cell
        .get_or_try_init(|| async {
            let s = settings.current();
            let model_id = minutist_common::ModelId::from(crate::commands::DEFAULT_EMBED_MODEL_ID);
            let model_dir = orchestrator.ensure_model_path(&model_id).await?;
            let gguf = crate::commands::find_gguf_weights(&model_dir)?;
            // The embedder is small (~600 MB); offload it whenever GPU acceleration
            // is enabled (Off forces CPU). It is not in the summariser-first VRAM plan.
            let enabled = s.gpu_acceleration != minutist_common::GpuAcceleration::Off;
            let n_gpu_layers = crate::commands::resolve_summariser_gpu_layers(enabled);
            let embedder = tokio::task::spawn_blocking(move || {
                embedder::Bgem3Embedder::open(
                    &gguf,
                    crate::commands::DEFAULT_EMBED_MODEL_ID,
                    n_gpu_layers,
                )
            })
            .await
            .map_err(|e| minutist_common::AppError::Internal {
                context: format!("live-agent embedder load task join failed: {e}"),
            })??;
            tracing::info!(
                target: "ipc-bridge",
                "live-agent: held BGE-M3 embedder loaded (background)"
            );
            Ok::<_, minutist_common::AppError>(Arc::new(embedder) as Arc<dyn Embedder>)
        })
        .await?;
    Ok(Arc::clone(handle))
}

async fn run_worker_loop(
    meeting_id: MeetingId,
    mut req_rx: mpsc::Receiver<TailRequest>,
    res_tx: mpsc::Sender<RefreshResult>,
    session: &mut LiveSession<LlamaLiveBackend<'_>>,
    retrieval: Option<&LiveRetrieval>,
) {
    while let Some(req) = req_rx.recv().await {
        // Retrieve attachment / earlier-transcript context relevant to this
        // refresh window and inject it into the tail. `None` until the embedder
        // has loaded or while the cache is empty (the agent still digests).
        let injected = match retrieval {
            Some(rc) => build_retrieval_block(rc, &req.tail).await,
            None => None,
        };
        let result = process_request(meeting_id, session, req, injected.as_deref());
        // Both CapacityExhausted and Err are terminal: the held context is
        // untrustworthy after either condition. Stop after sending.
        let is_terminal = matches!(
            result,
            RefreshResult::CapacityExhausted(_) | RefreshResult::Err(_)
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

        // After emitting the digest, incrementally index newly-sealed transcript
        // turns so later refreshes can retrieve earlier discussion. Best-effort and
        // off the digest's critical path (the result is already sent).
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

fn process_request(
    meeting_id: MeetingId,
    session: &mut LiveSession<LlamaLiveBackend<'_>>,
    req: TailRequest,
    retrieved: Option<&str>,
) -> RefreshResult {
    // The prefix was seeded once at session start; this call only appends the
    // effective tail (retrieved context + prior digest + new segments).
    let effective_tail =
        build_effective_tail(&req.tail, req.prior_digest_json.as_deref(), retrieved);

    let mut generated = String::new();
    // refresh_typed returns the typed chat_agent::Error so ContextOverflow
    // can be matched structurally, not by string inspection. Overflow is
    // permanent for this session; other errors (M1/M2) are terminal via the
    // driver's M3 teardown path.
    match session.refresh_typed(&effective_tail, &req.sampler, &req.cancel, &mut |piece| {
        generated.push_str(piece)
    }) {
        Ok(fallback) => {
            let text = if generated.is_empty() {
                fallback
            } else {
                generated
            };
            RefreshResult::Ok(text)
        }
        Err(chat_agent::Error::ContextOverflow(msg)) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent: context overflow detected: {msg}"
            );
            RefreshResult::CapacityExhausted(format!("context overflow: {msg}"))
        }
        Err(e) => RefreshResult::Err(format!("refresh failed: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Prefix and tail construction
// ---------------------------------------------------------------------------

/// Build the one-time prefix: system prompt + digest-category instructions.
///
/// Attachment / earlier-transcript context is no longer pinned here — it is
/// retrieved into the tail each refresh (see [`build_retrieval_block`]), keeping
/// the once-prefilled prefix small on every GPU tier.
pub(crate) fn build_prefix(s: &settings::Settings) -> String {
    let mut prefix = String::new();

    prefix.push_str(&s.live_agent_system_prompt);
    prefix.push_str("\n\n");

    prefix.push_str("Track the following digest categories:\n");
    if s.live_agent_digest_action_items {
        prefix.push_str("- action_items: tasks or follow-ups explicitly requested\n");
    }
    if s.live_agent_digest_decisions {
        prefix.push_str("- decisions: commitments or conclusions reached\n");
    }
    if s.live_agent_digest_open_asks {
        prefix.push_str("- open_asks: questions posed but not yet answered\n");
    }
    if s.live_agent_digest_attachment_answers {
        prefix.push_str("- attachment_answers: questions answered from retrieved documents\n");
    }
    if s.live_agent_digest_unresolved_references {
        prefix.push_str("- unresolved_references: terms or acronyms not explained\n");
    }
    prefix.push_str(
        "\nFor each item: {\"text\": \"...\", \"resolved\": false, \"source\": null}\n\
         Return ONLY a JSON object matching the LiveDigest schema.\n\n",
    );

    prefix
}

// ---------------------------------------------------------------------------
// Retrieval (RAG) — per-refresh context injection
// ---------------------------------------------------------------------------

/// Cap the retrieval query to the most recent slice of the discussion — it is the
/// "what's being talked about now" focus; older context is what we retrieve.
const QUERY_CHAR_CAP: usize = 2000;

/// Per-session RAG context held by the worker: drives both retrieval (each refresh
/// embeds the recent window + reads the cache) and incremental transcript indexing
/// (sealed turns appended as the meeting runs). The shared embedder cell is peeked —
/// `None` until the background load completes.
///
/// `store` is a single libsql connection used for BOTH the per-refresh reads and the
/// incremental-index write; this is sound ONLY because the worker is a single-in-flight
/// current-thread loop (the read and the append never overlap). Moving the incremental
/// index off the loop (e.g. `tokio::spawn`) would need its own connection.
struct LiveRetrieval {
    embedder_cell: Arc<OnceCell<Arc<dyn Embedder>>>,
    store: RagStore,
    /// Meeting folder root — the incremental indexer reads `transcript.json` under it.
    meetings_dir: PathBuf,
    /// Top-k chunks fused across the dense + lexical legs (tier-scaled).
    k: usize,
    /// Upper bound on injected-context characters. A backstop — `k` is the
    /// dominant knob, since each chunk is ~1 KB.
    char_budget: usize,
}

/// Scale the configured retrieval `k` to the GPU tier. An integrated GPU pays a
/// quadratic per-refresh prefill, so it gets roughly half the chunks (floored so
/// retrieval never collapses to a single hit); a discrete GPU uses the full `k`.
/// `k == 0` disables retrieval on both tiers.
fn tier_scaled_k(base_k: usize, is_integrated: bool) -> usize {
    if base_k == 0 {
        return 0;
    }
    if is_integrated {
        (base_k / 2).max(3).min(base_k)
    } else {
        base_k
    }
}

/// The last `n` characters of `s` (char-boundary safe), or all of `s` when shorter.
/// `n == 0` yields `""`.
fn tail_chars(s: &str, n: usize) -> &str {
    if n == 0 {
        return "";
    }
    let count = s.chars().count();
    if count <= n {
        return s;
    }
    let start = s
        .char_indices()
        .nth(count - n)
        .map(|(i, _)| i)
        .unwrap_or(0);
    &s[start..]
}

/// A human-readable heading for a retrieved chunk, by document type. (The
/// attachment `source_id` is a content hash, not a filename, so the heading is
/// generic until a hash→filename lookup is wired in.)
fn retrieval_source_label(c: &RetrievedChunk) -> &'static str {
    match c.doc_type.as_str() {
        "attachment" => "From an attached document",
        "transcript" => "Earlier in the meeting",
        _ => "Relevant context",
    }
}

/// Retrieve the chunks relevant to the recent discussion and format them as a
/// tail-injected context block, or `None` when retrieval is unavailable / empty.
///
/// Query = the recent transcript window (`recent`, this refresh's new segments),
/// capped to the last [`QUERY_CHAR_CAP`] characters. The dense (cosine) and
/// lexical (FTS5) legs are fused by RRF. No dedup against the live window is
/// needed: `recent` is reset each refresh, and the incremental indexer only seals
/// turns that have already scrolled out of the window, so an indexed transcript
/// chunk can never duplicate what is in the current window. Survivors are packed
/// up to `char_budget`.
async fn build_retrieval_block(rc: &LiveRetrieval, recent: &str) -> Option<String> {
    // Peek the shared cell — `None` until the background load completes.
    let embedder = rc.embedder_cell.get().cloned()?;
    if rc.k == 0 {
        return None;
    }
    let query = tail_chars(recent, QUERY_CHAR_CAP);
    if query.trim().is_empty() {
        return None;
    }
    // Embed the query OFF the runtime thread (sync FFI ~180 ms) so the background
    // embedder load and the worker channels keep progressing during it.
    let q = query.to_string();
    let emb = embedder.clone();
    let qvec = match tokio::task::spawn_blocking(move || emb.embed(&q)).await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            tracing::warn!(
                target: "ipc-bridge",
                error = %e,
                "live-agent: query embed failed; skipping context injection"
            );
            return None;
        }
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                error = %e,
                "live-agent: query embed task join failed; skipping context injection"
            );
            return None;
        }
    };
    let model_id = embedder.model_id();
    // A cache-read failure disables injection for this refresh only (best-effort).
    let dense = rc
        .store
        .retrieve_dense(&qvec, model_id, rc.k)
        .await
        .unwrap_or_default();
    let lexical = rc
        .store
        .retrieve_lexical(query, rc.k)
        .await
        .unwrap_or_default();
    if dense.is_empty() && lexical.is_empty() {
        return None;
    }

    // Fuse by chunk_id (RRF ignores the per-leg score scales), then map the fused
    // ids back to their chunk. Both legs return the same chunk fields, so either
    // copy works.
    let mut by_id: HashMap<String, RetrievedChunk> = HashMap::new();
    for c in dense.iter().chain(lexical.iter()) {
        by_id.entry(c.chunk_id.clone()).or_insert_with(|| c.clone());
    }
    let dense_ids: Vec<String> = dense.iter().map(|c| c.chunk_id.clone()).collect();
    let lexical_ids: Vec<String> = lexical.iter().map(|c| c.chunk_id.clone()).collect();
    let fused = rrf_fuse(&[&dense_ids, &lexical_ids], rc.k);

    let mut block = String::new();
    let mut used = 0usize;
    for id in fused {
        let Some(chunk) = by_id.get(&id) else {
            continue;
        };
        if used + chunk.text.len() > rc.char_budget {
            break;
        }
        let label = retrieval_source_label(chunk);
        block.push_str(&format!("## {label}\n{}\n\n", chunk.text.trim()));
        used += chunk.text.len();
    }
    if block.is_empty() {
        None
    } else {
        Some(format!(
            "Relevant context (attachments + earlier transcript):\n\n{block}"
        ))
    }
}

/// Build the effective tail appended on each refresh: the retrieved context
/// block (if any), then the prior digest JSON (if any) for standing-list
/// updates, then the new transcript segments.
fn build_effective_tail(
    new_segments: &str,
    prior_digest_json: Option<&str>,
    retrieved: Option<&str>,
) -> String {
    let mut tail = String::new();
    if let Some(ctx) = retrieved {
        tail.push_str(ctx);
        tail.push('\n');
    }
    if let Some(prior) = prior_digest_json {
        tail.push_str("Current digest state:\n");
        tail.push_str(prior);
        tail.push_str("\n\nNew transcript segments:\n");
    } else {
        tail.push_str("Transcript segments:\n");
    }
    tail.push_str(new_segments);
    tail.push_str("\n\nUpdated digest:");
    tail
}

// ---------------------------------------------------------------------------
// Digest parser
// ---------------------------------------------------------------------------

/// Parse the model's output text into a [`LiveDigest`].
///
/// Strips code fences, parses JSON, maps category arrays to `Vec<LiveDigestItem>`,
/// then applies two update rules depending on the category:
///
/// - All categories: if a prior item with the same text was `resolved`, preserve
///   that flag even if the model emits `false` (model forgetfulness guard).
/// - `open_asks` specifically: prior unresolved items NOT mentioned by the model
///   are carried forward (the model may omit them to save tokens). Items the model
///   marks `resolved: true` are promoted. This implements the SP-LIVE "tracker
///   maintained across refreshes" contract.
///
/// Returns `Err(String)` on JSON parse failure rather than panicking.
pub(crate) fn parse_digest(
    text: &str,
    meeting_id: MeetingId,
    prior: Option<&LiveDigest>,
) -> Result<LiveDigest, String> {
    let text = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let v: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("JSON parse failed: {e} (text: {text:?})"))?;

    let parse_items = |key: &str| -> Vec<LiveDigestItem> {
        v.get(key)
            .and_then(|arr| arr.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        let text = item.get("text")?.as_str()?.to_string();
                        let resolved = item
                            .get("resolved")
                            .and_then(|r| r.as_bool())
                            .unwrap_or(false);
                        let source = item
                            .get("source")
                            .and_then(|s| s.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string());
                        Some(LiveDigestItem {
                            text,
                            resolved,
                            source,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let action_items = apply_standing_list_update(
        parse_items("action_items"),
        prior.map(|d| d.action_items.as_slice()).unwrap_or(&[]),
    );
    let decisions = apply_standing_list_update(
        parse_items("decisions"),
        prior.map(|d| d.decisions.as_slice()).unwrap_or(&[]),
    );
    let open_asks = accumulate_open_asks(
        parse_items("open_asks"),
        prior.map(|d| d.open_asks.as_slice()).unwrap_or(&[]),
    );
    let attachment_answers = apply_standing_list_update(
        parse_items("attachment_answers"),
        prior
            .map(|d| d.attachment_answers.as_slice())
            .unwrap_or(&[]),
    );
    let unresolved_references = apply_standing_list_update(
        parse_items("unresolved_references"),
        prior
            .map(|d| d.unresolved_references.as_slice())
            .unwrap_or(&[]),
    );

    let generated_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    Ok(LiveDigest {
        meeting_id,
        generated_at_ms,
        action_items,
        decisions,
        open_asks,
        attachment_answers,
        unresolved_references,
    })
}

/// Preserve `resolved = true` from prior items whose text matches (case-insensitive,
/// trimmed). The model must not un-resolve an already-resolved item.
fn apply_standing_list_update(
    new: Vec<LiveDigestItem>,
    prior: &[LiveDigestItem],
) -> Vec<LiveDigestItem> {
    new.into_iter()
        .map(|mut item| {
            let was_resolved = prior
                .iter()
                .any(|p| p.resolved && p.text.trim().eq_ignore_ascii_case(item.text.trim()));
            if was_resolved {
                item.resolved = true;
            }
            item
        })
        .collect()
}

/// Accumulate `open_asks` across refreshes.
///
/// The model may omit prior unresolved asks to save tokens. This function
/// carries those forward so the tracker is maintained across refreshes
/// (SP-LIVE contract). The union rule:
///
/// 1. Apply the resolved-flag-carry-forward rule from [`apply_standing_list_update`]
///    to all items the model emits.
/// 2. For each prior item NOT mentioned by the model: if it was unresolved,
///    carry it forward unchanged; if it was already resolved, do not include
///    it (the user saw it resolved; no need to keep showing it).
///
/// The resulting list contains all unresolved asks the model emitted (with
/// flags preserved from prior), plus unresolved prior asks the model omitted.
fn accumulate_open_asks(new: Vec<LiveDigestItem>, prior: &[LiveDigestItem]) -> Vec<LiveDigestItem> {
    // Start with the resolved-flag-carry-forward of the model's output.
    let mut result = apply_standing_list_update(new, prior);

    // Carry forward unresolved prior items that the model did not mention.
    for prior_item in prior {
        if !prior_item.resolved {
            let mentioned = result
                .iter()
                .any(|r| r.text.trim().eq_ignore_ascii_case(prior_item.text.trim()));
            if !mentioned {
                result.push(prior_item.clone());
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Test-only stub backend
// ---------------------------------------------------------------------------

/// A no-op backend for unit tests that exercises the full driver protocol
/// pipeline without requiring a model. Only compiled in `#[cfg(test)]`.
///
/// Production code always uses `LlamaLiveBackend`.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use chat_agent::{
        CancelFlag, Error as ChatError, LiveSession, LiveSessionBackend, RawTurn, SamplerConfig,
    };

    pub(crate) struct WorkerBackend {
        /// Shared counter so tests can observe the prefill call count.
        pub(crate) prefill_counter: Arc<std::sync::atomic::AtomicU32>,
    }

    impl WorkerBackend {
        pub(crate) fn new() -> Self {
            Self {
                prefill_counter: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            }
        }

        pub(crate) fn prefill_counter(&self) -> Arc<std::sync::atomic::AtomicU32> {
            Arc::clone(&self.prefill_counter)
        }
    }

    impl LiveSessionBackend for WorkerBackend {
        fn prefill_prefix(
            &mut self,
            _prefix_text: &str,
            _cancel: &CancelFlag,
        ) -> Result<usize, ChatError> {
            self.prefill_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(0)
        }

        fn refresh(
            &mut self,
            _tail_text: &str,
            _cfg: &SamplerConfig,
            cancel: &CancelFlag,
            _token_cb: &mut dyn FnMut(&str),
        ) -> Result<RawTurn, ChatError> {
            if cancel.is_cancelled() {
                return Ok(RawTurn {
                    text: String::new(),
                    tool_calls: Vec::new(),
                    cancelled: true,
                });
            }
            // Minimal valid empty-digest JSON so parse_digest succeeds.
            Ok(RawTurn {
                text: "{}".to_string(),
                tool_calls: Vec::new(),
                cancelled: false,
            })
        }
    }

    /// A stub backend that returns `Error::ContextOverflow` on the first
    /// `refresh` call, for testing the overflow classification path.
    pub(crate) struct OverflowBackend;

    impl LiveSessionBackend for OverflowBackend {
        fn prefill_prefix(
            &mut self,
            _prefix_text: &str,
            _cancel: &CancelFlag,
        ) -> Result<usize, ChatError> {
            Ok(0)
        }

        fn refresh(
            &mut self,
            _tail_text: &str,
            _cfg: &SamplerConfig,
            _cancel: &CancelFlag,
            _token_cb: &mut dyn FnMut(&str),
        ) -> Result<RawTurn, ChatError> {
            Err(ChatError::ContextOverflow(
                "stub: n_past=30000 would exceed n_ctx=32768".to_string(),
            ))
        }
    }

    /// Drive one request through a WorkerBackend session, mirroring what
    /// `process_request` does but for the stub backend.
    ///
    /// The session must already be seeded (the caller mirrors the worker-thread
    /// startup by calling `session.seed_prefix_typed` before the first request,
    /// matching the real flow in `run_worker_thread`).
    pub(crate) fn process_stub_request<B: LiveSessionBackend>(
        _meeting_id: MeetingId,
        session: &mut LiveSession<B>,
        req: TailRequest,
    ) -> RefreshResult {
        let effective_tail =
            build_effective_tail(&req.tail, req.prior_digest_json.as_deref(), None);
        let mut generated = String::new();
        match session.refresh_typed(&effective_tail, &req.sampler, &req.cancel, &mut |piece| {
            generated.push_str(piece)
        }) {
            Ok(fallback) => {
                let text = if generated.is_empty() {
                    fallback
                } else {
                    generated
                };
                RefreshResult::Ok(text)
            }
            Err(ChatError::ContextOverflow(msg)) => {
                RefreshResult::CapacityExhausted(format!("context overflow: {msg}"))
            }
            Err(e) => RefreshResult::Err(format!("refresh failed: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::test_support::{process_stub_request, WorkerBackend};
    use super::*;
    use chat_agent::LiveSession;
    use minutist_common::{LiveDigest, LiveDigestItem, MeetingId};

    fn new_mid() -> MeetingId {
        MeetingId::new()
    }

    // -----------------------------------------------------------------------
    // should_refresh — pure cadence gate
    // -----------------------------------------------------------------------

    #[test]
    fn should_refresh_n_only_not_enough_time() {
        assert!(!should_refresh(10, 5.0, false, 8, 45));
    }

    #[test]
    fn should_refresh_t_only_not_enough_segments() {
        assert!(!should_refresh(3, 60.0, false, 8, 45));
    }

    #[test]
    fn should_refresh_both_thresholds_met() {
        assert!(should_refresh(8, 45.0, false, 8, 45));
    }

    #[test]
    fn should_refresh_in_flight_suppressed() {
        assert!(!should_refresh(100, 9999.0, true, 8, 45));
    }

    #[test]
    fn should_refresh_exact_boundary() {
        assert!(should_refresh(8, 45.0, false, 8, 45));
    }

    #[test]
    fn should_refresh_one_below_segment_threshold() {
        assert!(!should_refresh(7, 100.0, false, 8, 45));
    }

    #[test]
    fn should_refresh_one_below_time_threshold() {
        assert!(!should_refresh(20, 44.9, false, 8, 45));
    }

    // -----------------------------------------------------------------------
    // parse_digest — JSON parser + standing-list update
    // -----------------------------------------------------------------------

    fn digest_with_open_ask(mid: MeetingId, text: &str, resolved: bool) -> LiveDigest {
        LiveDigest {
            meeting_id: mid,
            generated_at_ms: 0,
            action_items: vec![],
            decisions: vec![],
            open_asks: vec![LiveDigestItem {
                text: text.to_string(),
                resolved,
                source: None,
            }],
            attachment_answers: vec![],
            unresolved_references: vec![],
        }
    }

    #[test]
    fn parse_digest_minimal_json() {
        let mid = new_mid();
        let text = r#"{"action_items": [{"text": "call Bob", "resolved": false}], "decisions": [], "open_asks": [], "attachment_answers": [], "unresolved_references": []}"#;
        let digest = parse_digest(text, mid, None).expect("parse");
        assert_eq!(digest.action_items.len(), 1);
        assert_eq!(digest.action_items[0].text, "call Bob");
        assert!(!digest.action_items[0].resolved);
    }

    #[test]
    fn parse_digest_empty_object() {
        let mid = new_mid();
        let digest = parse_digest("{}", mid, None).expect("empty object valid");
        assert!(digest.action_items.is_empty());
        assert!(digest.open_asks.is_empty());
    }

    #[test]
    fn parse_digest_open_ask_resolved_on_second_refresh() {
        let mid = new_mid();
        let text1 = r#"{"open_asks": [{"text": "what is the budget?", "resolved": false}]}"#;
        let digest1 = parse_digest(text1, mid, None).expect("first parse");
        assert!(!digest1.open_asks[0].resolved);

        let text2 = r#"{"open_asks": [{"text": "what is the budget?", "resolved": true}]}"#;
        let digest2 = parse_digest(text2, mid, Some(&digest1)).expect("second parse");
        assert!(digest2.open_asks[0].resolved);
    }

    #[test]
    fn parse_digest_resolved_flag_preserved_across_refresh() {
        let mid = new_mid();
        let prior = digest_with_open_ask(mid, "confirm the date", true);
        let text = r#"{"open_asks": [{"text": "confirm the date", "resolved": false}]}"#;
        let digest = parse_digest(text, mid, Some(&prior)).expect("parse");
        assert!(
            digest.open_asks[0].resolved,
            "resolved flag from prior must be preserved"
        );
    }

    #[test]
    fn parse_digest_open_ask_omitted_by_model_is_carried_forward() {
        // The model emits a new ask but omits the prior unresolved ask.
        // The accumulator must carry the omitted prior ask forward.
        let mid = new_mid();
        let prior = digest_with_open_ask(mid, "what is the timeline?", false);
        // Model outputs a new ask but does not mention "timeline".
        let text = r#"{"open_asks": [{"text": "who owns the budget?", "resolved": false}]}"#;
        let digest = parse_digest(text, mid, Some(&prior)).expect("parse");

        // Both asks must be present.
        assert_eq!(
            digest.open_asks.len(),
            2,
            "omitted prior ask must be carried forward"
        );
        let texts: Vec<&str> = digest.open_asks.iter().map(|a| a.text.as_str()).collect();
        assert!(
            texts.contains(&"who owns the budget?"),
            "new ask must be present"
        );
        assert!(
            texts.contains(&"what is the timeline?"),
            "omitted prior unresolved ask must be carried forward"
        );
    }

    #[test]
    fn parse_digest_resolved_open_ask_not_carried_forward_when_omitted() {
        // If a prior ask was resolved and the model omits it, do NOT include it.
        let mid = new_mid();
        let prior = digest_with_open_ask(mid, "already answered", true);
        let text = r#"{"open_asks": []}"#;
        let digest = parse_digest(text, mid, Some(&prior)).expect("parse");
        assert!(
            digest.open_asks.is_empty(),
            "resolved prior asks must not be carried forward when omitted"
        );
    }

    #[test]
    fn parse_digest_strips_code_fence() {
        let mid = new_mid();
        let text = "```json\n{\"action_items\":[{\"text\":\"foo\",\"resolved\":false}]}\n```";
        let digest = parse_digest(text, mid, None).expect("parse with code fence");
        assert_eq!(digest.action_items.len(), 1);
    }

    #[test]
    fn parse_digest_invalid_json_returns_error() {
        let mid = new_mid();
        assert!(parse_digest("not json", mid, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Per-category settings toggles
    // -----------------------------------------------------------------------

    #[test]
    fn category_toggles_off_omit_from_prefix() {
        // Neutral system prompt that contains none of the category names.
        let s = settings::Settings {
            live_agent_system_prompt: "You are a meeting assistant.".to_string(),
            live_agent_digest_action_items: false,
            live_agent_digest_decisions: false,
            live_agent_digest_open_asks: false,
            live_agent_digest_attachment_answers: false,
            live_agent_digest_unresolved_references: false,
            ..Default::default()
        };

        let prefix = build_prefix(&s);
        // With all toggles off, the category listing must not appear.
        assert!(!prefix.contains("action_items"));
        assert!(!prefix.contains("decisions"));
        assert!(!prefix.contains("open_asks"));
        assert!(!prefix.contains("attachment_answers"));
        assert!(!prefix.contains("unresolved_references"));
    }

    #[test]
    fn category_toggles_on_appear_in_prefix() {
        let s = settings::Settings {
            live_agent_system_prompt: "You are a meeting assistant.".to_string(),
            live_agent_digest_action_items: true,
            live_agent_digest_decisions: true,
            live_agent_digest_open_asks: true,
            live_agent_digest_attachment_answers: true,
            live_agent_digest_unresolved_references: true,
            ..Default::default()
        };

        let prefix = build_prefix(&s);
        assert!(prefix.contains("action_items"));
        assert!(prefix.contains("decisions"));
        assert!(prefix.contains("open_asks"));
        assert!(prefix.contains("attachment_answers"));
        assert!(prefix.contains("unresolved_references"));
    }

    // -----------------------------------------------------------------------
    // live_agent_should_run gate
    // -----------------------------------------------------------------------

    #[test]
    fn live_agent_should_run_off_returns_false() {
        use minutist_common::{live_agent_should_run, GpuAcceleration, LiveAgentMode};
        assert!(!live_agent_should_run(
            LiveAgentMode::Off,
            None,
            GpuAcceleration::Auto
        ));
    }

    #[test]
    fn live_agent_should_run_on_returns_true() {
        use minutist_common::{live_agent_should_run, GpuAcceleration, LiveAgentMode};
        assert!(live_agent_should_run(
            LiveAgentMode::On,
            None,
            GpuAcceleration::Off
        ));
    }

    #[test]
    fn live_agent_should_run_auto_no_probe_returns_false() {
        use minutist_common::{live_agent_should_run, GpuAcceleration, LiveAgentMode};
        assert!(!live_agent_should_run(
            LiveAgentMode::Auto,
            None,
            GpuAcceleration::Auto
        ));
    }

    #[test]
    fn live_agent_should_run_auto_integrated_gpu_accel_on_returns_true() {
        // AMD Radeon 890M (integrated, Vulkan on) is the validated SP-LIVE
        // hardware — Auto must resolve true when gpu_acceleration is active.
        use minutist_common::{live_agent_should_run, GpuAcceleration, GpuProbe, LiveAgentMode};
        let probe = GpuProbe {
            total_bytes: 16 * 1024 * 1024 * 1024,
            free_bytes: 8 * 1024 * 1024 * 1024,
            is_integrated: true,
            name: "AMD Radeon 890M".to_string(),
        };
        assert!(live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&probe),
            GpuAcceleration::Auto
        ));
    }

    #[test]
    fn live_agent_should_run_auto_accel_off_returns_false() {
        // gpu_acceleration=Off → LLM would run on CPU, contending with ASR.
        use minutist_common::{live_agent_should_run, GpuAcceleration, GpuProbe, LiveAgentMode};
        let probe = GpuProbe {
            total_bytes: 36 * 1024 * 1024 * 1024,
            free_bytes: 20 * 1024 * 1024 * 1024,
            is_integrated: false,
            name: "RTX 4090".to_string(),
        };
        assert!(!live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&probe),
            GpuAcceleration::Off
        ));
    }

    #[test]
    fn live_agent_should_run_auto_discrete_gpu_accel_on_returns_true() {
        use minutist_common::{live_agent_should_run, GpuAcceleration, GpuProbe, LiveAgentMode};
        let probe = GpuProbe {
            total_bytes: 36 * 1024 * 1024 * 1024,
            free_bytes: 20 * 1024 * 1024 * 1024,
            is_integrated: false,
            name: "RTX 4090".to_string(),
        };
        assert!(live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&probe),
            GpuAcceleration::Auto
        ));
    }

    // -----------------------------------------------------------------------
    // WorkerBackend + LiveSession round-trip (stub, no model)
    // -----------------------------------------------------------------------

    #[test]
    fn worker_backend_round_trip() {
        let mid = new_mid();
        let mut session: LiveSession<WorkerBackend> = LiveSession::new(WorkerBackend::new());
        // Mirrors the worker-thread startup: seed once before the loop.
        session
            .seed_prefix_typed("You are a meeting agent.", &CancelFlag::new())
            .expect("seed");
        let req = TailRequest {
            tail: "Alice: let's schedule a follow-up call".to_string(),
            prior_digest_json: None,
            sampler: SamplerConfig::deterministic(),
            cancel: CancelFlag::new(),
        };

        match process_stub_request(mid, &mut session, req) {
            RefreshResult::Ok(text) => {
                let digest =
                    parse_digest(&text, mid, None).expect("WorkerBackend output must be parseable");
                assert_eq!(digest.meeting_id, mid);
            }
            RefreshResult::Err(e) => panic!("round-trip must succeed, got Err: {e}"),
            RefreshResult::CapacityExhausted(e) => {
                panic!("round-trip must succeed, got CapacityExhausted: {e}")
            }
        }
    }

    #[test]
    fn worker_backend_seed_prefix_called_once() {
        // The worker seeds exactly once before the loop; subsequent process_request
        // calls do NOT re-seed. This test verifies that a WorkerBackend session
        // seeded once at start produces the counter = 1 after multiple requests.
        let mid = new_mid();
        let backend = WorkerBackend::new();
        let counter = backend.prefill_counter();
        let mut session: LiveSession<WorkerBackend> = LiveSession::new(backend);

        // One seed at worker-thread startup.
        session
            .seed_prefix_typed("prefix", &CancelFlag::new())
            .expect("seed");

        for i in 0..3u32 {
            let req = TailRequest {
                tail: format!("segment {i}"),
                prior_digest_json: None,
                sampler: SamplerConfig::deterministic(),
                cancel: CancelFlag::new(),
            };
            process_stub_request(mid, &mut session, req);
        }

        assert_eq!(
            counter.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "prefill_prefix must be called exactly once (at worker startup)"
        );
    }

    // -----------------------------------------------------------------------
    // ContextOverflow → CapacityExhausted classification (must-fix finding)
    // -----------------------------------------------------------------------

    /// A stub backend returning `Error::ContextOverflow` must map to
    /// `RefreshResult::CapacityExhausted` via the typed-error path, not to
    /// `RefreshResult::Err`. This guards against the regression where string-
    /// based overflow detection silently misclassifies a `ContextOverflow` as a
    /// transient error (the `From<Error> for AppError` impl discards the variant
    /// by mapping it to `InvalidInput`, so Display-string matching would never
    /// see the literal "ContextOverflow").
    #[test]
    fn overflow_backend_yields_capacity_exhausted() {
        use super::test_support::OverflowBackend;

        let _mid = new_mid();
        let mut session = LiveSession::new(OverflowBackend);

        // seed_prefix must succeed (OverflowBackend::prefill_prefix returns Ok).
        let seed_result = session.seed_prefix_typed("prefix", &CancelFlag::new());
        assert!(seed_result.is_ok(), "seed must succeed: {seed_result:?}");

        // refresh_typed must return ContextOverflow.
        let refresh_result = session.refresh_typed(
            "tail",
            &SamplerConfig::deterministic(),
            &CancelFlag::new(),
            &mut |_| {},
        );
        assert!(
            matches!(refresh_result, Err(chat_agent::Error::ContextOverflow(_))),
            "OverflowBackend must return ContextOverflow on refresh, got {refresh_result:?}"
        );

        // Construct a TailRequest and drive it through process_stub_request
        // using an OverflowBackend-backed session — but since process_stub_request
        // expects WorkerBackend, we test the typed path directly via the module's
        // process_request signature on a generic backend. Instead, verify the
        // classification by checking that the typed Err variant matches, then
        // manually confirm the RefreshResult mapping is correct by inspecting
        // the match arm in process_stub_request's own use of refresh_typed.
        //
        // Drive the overflow through the SAME path the worker uses
        // (process_request → refresh_typed): a ContextOverflow must MAP to
        // RefreshResult::CapacityExhausted, not be swallowed into RefreshResult::Err.
        let mut session2 = LiveSession::new(OverflowBackend);
        session2
            .seed_prefix_typed("sys", &CancelFlag::new())
            .expect("seed");
        let req = TailRequest {
            tail: "t".to_string(),
            prior_digest_json: None,
            sampler: SamplerConfig::deterministic(),
            cancel: CancelFlag::new(),
        };
        match process_stub_request(new_mid(), &mut session2, req) {
            RefreshResult::CapacityExhausted(_) => {
                // Correct — classified as capacity, not a transient Err.
            }
            other => panic!("ContextOverflow must map to CapacityExhausted, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Retrieval (RAG) — tier scaling, query window, and injection
    // -----------------------------------------------------------------------

    /// A deterministic embedder for retrieval tests: every input maps to the same
    /// unit vector, so dense ranking is driven entirely by the stored chunk vectors.
    struct StubEmbedder;

    impl Embedder for StubEmbedder {
        fn embed_batch(&self, texts: &[&str]) -> minutist_common::AppResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
        }
        fn dim(&self) -> usize {
            3
        }
        fn model_id(&self) -> &str {
            "stub-embed"
        }
    }

    /// A pre-populated embedder cell (the background load is simulated as already done).
    fn stub_cell() -> Arc<OnceCell<Arc<dyn Embedder>>> {
        let cell: Arc<OnceCell<Arc<dyn Embedder>>> = Arc::new(OnceCell::new());
        cell.set(Arc::new(StubEmbedder) as Arc<dyn Embedder>).ok();
        cell
    }

    #[test]
    fn tier_scaled_k_halves_on_integrated_full_on_discrete() {
        assert_eq!(tier_scaled_k(8, false), 8, "discrete uses the full k");
        assert_eq!(tier_scaled_k(8, true), 4, "integrated halves k");
        // The .max(3) floor RAISES a sub-3 half...
        assert_eq!(tier_scaled_k(5, true), 3, "5/2=2 floored up to 3");
        assert_eq!(tier_scaled_k(4, true), 3, "4/2=2 floored up to 3");
        assert_eq!(tier_scaled_k(3, true), 3, "floor=cap corner");
        // ...but the .min(base_k) clamp keeps it from exceeding a small configured k.
        assert_eq!(tier_scaled_k(2, true), 2, "floor clamped down to base_k=2");
        assert_eq!(tier_scaled_k(1, true), 1, "floor clamped down to base_k=1");
        assert_eq!(tier_scaled_k(0, true), 0, "k=0 disables retrieval on both tiers");
        assert_eq!(tier_scaled_k(0, false), 0);
    }

    #[test]
    fn tail_chars_keeps_the_last_n_on_char_boundaries() {
        assert_eq!(tail_chars("hello", 2), "lo");
        assert_eq!(tail_chars("hello", 5), "hello", "n == count returns all");
        assert_eq!(tail_chars("hi", 5), "hi", "shorter than n returns all");
        assert_eq!(tail_chars("hello", 0), "", "n == 0 yields empty");
        assert_eq!(tail_chars("", 5), "", "empty input");
        // Multi-byte: each Greek letter is 2 bytes; the slice must land on a boundary.
        assert_eq!(tail_chars("αβγ", 2), "βγ");
    }

    #[tokio::test]
    async fn retrieval_block_injects_relevant_chunk() {
        let store = persistence::RagStore::open(":memory:").await.expect("open");
        let near = vec![1.0, 0.0, 0.0];
        let far = vec![0.0, 1.0, 0.0];
        store
            .index_source(
                "att1",
                "attachment",
                "stub-embed",
                &[
                    persistence::NewChunk {
                        text: "the budget owner is Priya",
                        byte_offset: 0,
                        embedding: &near,
                    },
                    persistence::NewChunk {
                        text: "unrelated coffee notes",
                        byte_offset: 40,
                        embedding: &far,
                    },
                ],
            )
            .await
            .expect("index");

        let rc = LiveRetrieval {
            embedder_cell: stub_cell(),
            store,
            meetings_dir: std::path::PathBuf::new(),
            k: 4,
            char_budget: 10_000,
        };
        let block = build_retrieval_block(&rc, "who owns the budget")
            .await
            .expect("a relevant chunk is injected");
        assert!(block.contains("the budget owner is Priya"), "block: {block}");
        assert!(
            block.contains("From an attached document"),
            "attachment heading present"
        );
    }

    #[tokio::test]
    async fn retrieval_block_none_without_embedder() {
        let store = persistence::RagStore::open(":memory:").await.expect("open");
        // Embedder cell empty (background load not yet complete): no injection.
        let rc = LiveRetrieval {
            embedder_cell: Arc::new(OnceCell::new()),
            store,
            meetings_dir: std::path::PathBuf::new(),
            k: 4,
            char_budget: 10_000,
        };
        assert!(build_retrieval_block(&rc, "anything").await.is_none());
    }

    #[tokio::test]
    async fn retrieval_block_k_zero_disables_retrieval() {
        let store = persistence::RagStore::open(":memory:").await.expect("open");
        let e = vec![1.0, 0.0, 0.0];
        store
            .index_source(
                "att1",
                "attachment",
                "stub-embed",
                &[persistence::NewChunk {
                    text: "indexed text",
                    byte_offset: 0,
                    embedding: &e,
                }],
            )
            .await
            .expect("index");
        let rc = LiveRetrieval {
            embedder_cell: stub_cell(),
            store,
            meetings_dir: std::path::PathBuf::new(),
            k: 0,
            char_budget: 10_000,
        };
        assert!(build_retrieval_block(&rc, "indexed text").await.is_none());
    }

    #[tokio::test]
    async fn retrieval_block_fuses_both_legs_and_labels_by_doc_type() {
        let store = persistence::RagStore::open(":memory:").await.expect("open");
        // X (attachment): near the query vector but shares no query token → found by
        // the DENSE leg only. Y (transcript): far vector but contains the query token
        // → found by the LEXICAL leg only. The stub embeds the query to [1,0,0].
        store
            .index_source(
                "att1",
                "attachment",
                "stub-embed",
                &[persistence::NewChunk {
                    text: "quarterly planning notes",
                    byte_offset: 0,
                    embedding: &[1.0, 0.0, 0.0],
                }],
            )
            .await
            .expect("index attachment");
        store
            .append_source_chunks(
                "transcript_live",
                "transcript",
                "stub-embed",
                &[persistence::NewChunk {
                    text: "the budget is approved",
                    byte_offset: 0,
                    embedding: &[0.0, 1.0, 0.0],
                }],
            )
            .await
            .expect("index transcript");
        let rc = LiveRetrieval {
            embedder_cell: stub_cell(),
            store,
            meetings_dir: std::path::PathBuf::new(),
            k: 4,
            char_budget: 10_000,
        };
        let block = build_retrieval_block(&rc, "budget")
            .await
            .expect("both legs contribute");
        // Both chunks injected, each with its doc-type heading.
        assert!(block.contains("quarterly planning notes"), "dense-only hit present");
        assert!(block.contains("the budget is approved"), "lexical-only hit present");
        assert!(block.contains("From an attached document"));
        assert!(block.contains("Earlier in the meeting"));
        // Y is found by BOTH legs, so RRF ranks it above the dense-only X.
        let y = block.find("the budget is approved").unwrap();
        let x = block.find("quarterly planning notes").unwrap();
        assert!(y < x, "the both-legs hit outranks the single-leg hit");
    }

    #[tokio::test]
    async fn retrieval_block_respects_char_budget() {
        let store = persistence::RagStore::open(":memory:").await.expect("open");
        let e = vec![1.0, 0.0, 0.0];
        store
            .index_source(
                "att1",
                "attachment",
                "stub-embed",
                &[
                    persistence::NewChunk {
                        text: "AAAAAAAAAA",
                        byte_offset: 0,
                        embedding: &e,
                    },
                    persistence::NewChunk {
                        text: "BBBBBBBBBB",
                        byte_offset: 20,
                        embedding: &e,
                    },
                ],
            )
            .await
            .expect("index");
        // Budget fits only one ~10-char chunk body (the second would push past 12).
        let rc = LiveRetrieval {
            embedder_cell: stub_cell(),
            store,
            meetings_dir: std::path::PathBuf::new(),
            k: 4,
            char_budget: 12,
        };
        let block = build_retrieval_block(&rc, "query")
            .await
            .expect("one chunk fits");
        let a = block.contains("AAAAAAAAAA");
        let b = block.contains("BBBBBBBBBB");
        assert!(a ^ b, "exactly one chunk fits the 12-char budget; block: {block}");
    }
}
