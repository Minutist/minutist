//! [`IpcState`] — the Tauri managed state shared across every command handler.
//!
//! `app-main` constructs one [`IpcState`] and passes it to
//! `tauri::Builder::manage`. Fields used by nearly every command handler
//! (the orchestrator, settings, meeting-index, and event-bus handles) stay
//! flat on [`IpcState`] itself; fields that belong to one cohesive subsystem
//! are grouped into a sub-bundle: [`ChatRuntimeState`] (the chat/summarise
//! substrate), [`ConnectedState`] (the connected-tier control surfaces), and
//! [`DiagnosticsInfo`] (the `get_diagnostic_report` labelling fields).

use std::path::PathBuf;
use std::sync::Arc;

use agent_tools::{RecordingControl, ToolRegistry};
use async_trait::async_trait;
use minutist_common::{AppEvent, AppResult};
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use settings::SettingsHandle;
use summariser::LlamaSummariser;
use tokio::sync::{broadcast, mpsc, OnceCell};

use crate::attachments::ConvertJob;
use crate::chat_runtime::ChatHandles;
use crate::live_agent::LiveCopilotHandle;
use crate::sync::SyncControl;
use crate::tunnel::TunnelControl;

/// Adapts `orchestrator::Orchestrator` to `agent_tools::RecordingControl`.
///
/// `agent-tools` has no dependency on the concrete `orchestrator` crate (see
/// its "Boundaries" doc); Rust's orphan rule means the impl for a foreign
/// trait + a foreign type must live behind a local newtype, so `ipc-bridge`
/// (which owns both edges) provides this one. Every `ToolContext::new` call
/// site in this crate (and in `app-main`, which depends on `ipc-bridge`) wraps
/// its `Arc<Orchestrator>` in this before constructing the context — the same
/// pattern [`crate::tunnel::DisabledTunnel`] / the `connected`-tier
/// `ConnectedTunnel` use for `TunnelControl`.
pub struct OrchestratorRecordingControl(pub Arc<Orchestrator>);

#[async_trait]
impl RecordingControl for OrchestratorRecordingControl {
    async fn state(&self) -> minutist_common::RecordingState {
        self.0.state().await
    }
    async fn start(
        &self,
        meeting_id: minutist_common::MeetingId,
        device_id: Option<String>,
    ) -> AppResult<minutist_common::MeetingId> {
        self.0.start(meeting_id, device_id).await
    }
    async fn stop(&self) -> AppResult<minutist_common::MeetingMeta> {
        self.0.stop().await
    }
    async fn pause(&self) -> AppResult<()> {
        self.0.pause().await
    }
    async fn resume(&self) -> AppResult<()> {
        self.0.resume().await
    }
    async fn reprocess(
        &self,
        index: &MeetingIndex,
        meeting_id: minutist_common::MeetingId,
    ) -> AppResult<()> {
        self.0.reprocess(index, meeting_id).await
    }
    async fn transcribe_pcm_window(
        &self,
        meeting_id: minutist_common::MeetingId,
        start_ms: u64,
        end_ms: u64,
        language: Option<String>,
    ) -> AppResult<Vec<minutist_common::Segment>> {
        self.0
            .transcribe_pcm_window(meeting_id, start_ms, end_ms, language)
            .await
    }
}

/// The lazily-loaded LLM/embedder substrate + chat-driver bookkeeping shared
/// by the chat, summarise, and live-agent paths.
pub struct ChatRuntimeState {
    /// The lazily-loaded, **held** LLM summariser substrate (Phase 9, C2).
    ///
    /// The GGUF is loaded **once** on first chat/summarise use (via
    /// [`IpcState::ensure_summariser`]) and retained for the process
    /// lifetime, rather than reloaded per call. Both the one-shot
    /// `summarise_meeting` path and the chat driver share this handle: the
    /// chat engine borrows `&LlamaModel` via `LlamaSummariser::model()`, and
    /// the `agent-tools` `ToolContext`'s `resummarise` coerces it to
    /// `Arc<dyn Summariser>`. `tokio::sync::OnceCell` gives a `Send + Sync`,
    /// awaitable single-init that the async command handlers can drive
    /// without a `block_on`. See `architecture/cross-cutting.md` — "Agent
    /// chat loop".
    pub summariser: Arc<OnceCell<Arc<LlamaSummariser>>>,
    /// The lazily-loaded held BGE-M3 embedder (RAG). The SAME `Arc<OnceCell>`
    /// [`ChatHandles`] holds, so the model loads once and serves both the RAG
    /// write path and the `retrieve_chunks` tool.
    pub embedder: Arc<OnceCell<Arc<dyn minutist_common::Embedder>>>,
    /// The chat tool registry, built once (`ToolRegistry::v1(false)` — the
    /// inter-agent bridge tool is Phase 10). Shared by the chat driver, which
    /// reads `descriptors()` for the offered tools and `dispatch(...)` to run
    /// them. Held as `Arc` so it crosses into the per-turn `spawn_blocking`
    /// dispatch closure.
    pub tool_registry: Arc<ToolRegistry>,
    /// Per-session in-flight guard for the chat driver. A session id present
    /// in this set has a turn currently running; `send_chat_message` rejects
    /// a concurrent turn for the same session (§6 — single in-flight turn). A
    /// `std::sync::Mutex<HashSet>` (not async) because every access is a
    /// brief, non-awaiting insert/remove. SHARED with the Phase-10
    /// inter-agent driver (so an external turn and a human turn cannot run on
    /// one session at once).
    pub chat_in_flight:
        Arc<std::sync::Mutex<std::collections::HashSet<minutist_common::ChatSessionId>>>,
    /// Per-session chat-turn cancellation flags (P1). A running turn
    /// registers a `chat_agent::CancelFlag` here keyed by session id;
    /// `cancel_chat_turn` raises it, and the decode loop stops at the next
    /// between-token check. The driver removes the entry when the turn ends.
    /// A `std::sync::Mutex<HashMap>` (not async) because every access is a
    /// brief, non-awaiting insert / get / remove, mirroring `chat_in_flight`.
    pub chat_cancel: Arc<
        std::sync::Mutex<
            std::collections::HashMap<minutist_common::ChatSessionId, chat_agent::CancelFlag>,
        >,
    >,
}

/// The connected-tier (WS4-A / WS4-B) control surfaces: relay tunnel, notes
/// sync, and the live MCP endpoint. Grouped together because all three are
/// injected as one unit by `app-main` — a free build gets the disabled
/// implementations for `tunnel`/`sync` and a permanently-`None` `mcp_info`;
/// a connected build wires all three to the real relay/sync/MCP-server
/// state.
pub struct ConnectedState {
    /// The connected-tier relay tunnel control surface (WS4-A S5b). `app-main`
    /// injects a `connected`-gated implementation (holding the `tunnel-client`
    /// pairing + lifecycle types); the free build (and a connected build with no
    /// relay wiring) gets [`crate::tunnel::DisabledTunnel`], which reports
    /// `Disconnected` and rejects pairing as unsupported. The tunnel IPC commands
    /// (`tunnel_begin_pairing`, `tunnel_poll_pairing`, `set_connector_enabled`,
    /// `tunnel_status`) call through this trait so `ipc-bridge` takes no
    /// `tunnel-client` dependency edge.
    pub tunnel: Arc<dyn TunnelControl>,
    /// The peer-to-peer notes-sync control surface (WS4-B S5). `app-main` injects
    /// a `connected`-gated implementation (holding the `sync` engine: iroh
    /// endpoint + pairing + notes-sync protocol); the free build (and a connected
    /// build with no sync wiring) gets [`crate::sync::DisabledSync`], which
    /// reports `Disabled` and rejects ticket / peer / sync operations as
    /// unsupported. The sync IPC commands (`sync_status`, `sync_get_my_ticket`,
    /// `sync_add_peer`, `sync_now`) call through this trait so `ipc-bridge` takes
    /// no `sync` dependency edge — the same seam as [`Self::tunnel`].
    pub sync: Arc<dyn SyncControl>,
    /// The live MCP server endpoint, set by `app-main` after `mcp_server::serve`
    /// binds (Phase 10). `None` when the MCP server is disabled or not yet
    /// listening. The `get_mcp_server_info` command reads it to reveal the URL +
    /// bearer token for the user to paste into an external MCP client's config. The
    /// token is held here (not on the event bus) and revealed only on explicit
    /// user request.
    pub mcp_info: Arc<std::sync::Mutex<Option<McpServerInfo>>>,
}

/// Static labelling fields for `get_diagnostic_report` (#0014): where the
/// logs live, and the app version + platform string to stamp the report
/// with. `ipc-bridge` only READS `logs_dir`; `app-main` owns writes to it.
pub struct DiagnosticsInfo {
    /// The logs directory (`{app-data}/logs/`), owned by `app-main` but its path
    /// is shared here so `get_diagnostic_report` (#0014) can read the rolling
    /// `minutist.log` tail and the `last-crash.txt` written by the panic hook.
    pub logs_dir: PathBuf,
    /// The application version (e.g. `"0.0.0"`), resolved by `app-main` from the
    /// Tauri package info and shared here so `get_diagnostic_report` (#0014) can
    /// label the report without a `tauri`-version dependency in this crate.
    pub app_version: String,
    /// `"{os} / {arch} / {build}"`, constructed by `app-main` (which owns the
    /// `connected` Cargo feature that distinguishes the free / connected build)
    /// and shared here for `get_diagnostic_report` (#0014). Carries no
    /// machine-identifying detail (no hostname / user).
    pub platform: String,
}

/// Tauri managed state shared across all command handlers.
///
/// `app-main` constructs this and passes it to `tauri::Builder::manage`.
pub struct IpcState {
    pub orchestrator: Arc<Orchestrator>,
    pub settings: SettingsHandle,
    /// Root of the per-meeting folders (`{app-data}/meetings/`). The same
    /// directory `orchestrator` / `persistence` use. `save_notes` /
    /// `load_notes` / `open_meeting` route directly to `persistence` against
    /// this root, bypassing the orchestrator (folder I/O is independent of the
    /// recording pipeline — see `architecture/components.md`, `persistence`
    /// "Phase 3 surface growth — notes" and "Phase 4 surface growth").
    pub meetings_dir: PathBuf,
    /// Path to the libsql `index.db` (`{app-data}/index.db`), resolved by
    /// `app-main` via `persistence::index::index_db_path`. Retained for
    /// diagnostics / rebuild; the live query handle is [`Self::index`].
    pub index_db_path: PathBuf,
    /// The shared, already-open libsql meeting index. Opened once by `app-main`
    /// (libsql is async, so `open` is awaited at startup) and shared here so the
    /// Phase-4 meeting commands (`list_meetings`, `rename_meeting`,
    /// `delete_meeting`, and the index-upsert side of `re_transcribe`) query a
    /// single connection without re-opening per command. The index methods are
    /// `async fn` and are awaited in the command handlers — never `block_on`'d.
    pub index: Arc<MeetingIndex>,
    /// The shared `AppEvent` broadcast sender — the **same** channel `app-main`
    /// constructs once and hands to both the `ModelRegistry` and the
    /// `Orchestrator` (via `with_event_tx`). The event forwarder subscribes to
    /// it via `Orchestrator::subscribe_events`, so any event emitted here
    /// reaches the webview. `summarise_meeting` (Phase 5) emits
    /// `AppEvent::SummaryReady` on this sender after `summary.md` is written, so
    /// the summary view re-reads the persisted markdown. Cloning the sender (not
    /// re-deriving it from the orchestrator) keeps the single-bus invariant from
    /// `architecture/cross-cutting.md` "Model lifecycle".
    pub event_tx: broadcast::Sender<AppEvent>,
    /// The send half of the BOUNDED attachment-conversion job queue. `app-main`
    /// constructs the `(tx, rx)` pair (bound
    /// [`crate::attachments::ATTACHMENT_CONVERT_QUEUE_BOUND`]), stores `tx` here, and
    /// spawns the single long-lived worker on `rx` via
    /// [`crate::attachments::spawn_attachment_convert_worker`]. `add_attachment` `try_send`s a
    /// [`ConvertJob`] onto this queue; a full queue surfaces back-pressure by
    /// marking the row `Failed` rather than blocking the command. Bounded per the
    /// "bounded channels only" rule (`architecture/cross-cutting.md`).
    pub attachment_convert_tx: mpsc::Sender<ConvertJob>,
    /// The chat/summarise substrate: the held LLM + embedder, tool registry,
    /// and per-session in-flight/cancellation bookkeeping. See
    /// [`ChatRuntimeState`].
    pub chat_runtime: ChatRuntimeState,
    /// Per-`(meeting_id, language)` in-flight guard for the translation driver.
    /// A pair present in this set has a `translate_meeting` call currently
    /// running for that language; a second call for the same pair is rejected
    /// (mirrors `chat_runtime.chat_in_flight`). A `std::sync::Mutex<HashSet>`
    /// (not async) because every access is a brief, non-awaiting insert/remove.
    pub translate_in_flight:
        Arc<std::sync::Mutex<std::collections::HashSet<(minutist_common::MeetingId, String)>>>,
    /// The connected-tier (relay tunnel + notes sync + MCP endpoint) control
    /// surfaces. See [`ConnectedState`].
    pub connected: ConnectedState,
    /// The voiceprint library (`voiceprints.db`), opened at startup by `app-main`
    /// via `persistence::voiceprints_db_path` against the effective data root.
    ///
    /// Shared here so `set_speaker_name` can trigger
    /// `Orchestrator::enrol_voiceprint` when `settings.voiceprint_enrolment_enabled`
    /// is `true`. Mirrors `IpcState::index` — an already-open handle, shared via
    /// `Arc` so clones are cheap. `app-main` maps an open/migration failure to
    /// enrolment-OFF by supplying an `Arc<Option<VoiceprintStore>>` that holds
    /// `None`; enrolment commands check the `Option` and skip when `None`.
    ///
    /// No new dependency edge: `ipc-bridge` already depends on `persistence`.
    pub voiceprints: Arc<Option<persistence::VoiceprintStore>>,
    /// Per-meeting live co-pilot handles (U2 B5 / U4 A3).
    ///
    /// `spawn_live_agent` inserts a [`LiveCopilotHandle`] keyed by `MeetingId`
    /// when a recording starts and removes it when the agent exits. The
    /// `send_chat_message` command checks this registry to determine whether the
    /// target meeting is currently live: if a handle exists, the user message is
    /// sent to the live co-pilot via `handle.user_tx`; the reply streams back on a
    /// per-request [`crate::live_agent::UserChatRequest::reply_tx`] channel. See
    /// `commands/chat_commands.rs` — `send_chat_message` live path (A3) and
    /// `live_agent/` — "Registry and chat routing".
    ///
    /// A `std::sync::Mutex<HashMap>` (not async): every access is a brief
    /// non-awaiting insert / lookup / remove, mirroring `chat_runtime.chat_in_flight`.
    pub live_copilot_handles: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<minutist_common::MeetingId, LiveCopilotHandle>>,
    >,
    /// Diagnostics-report labelling fields. See [`DiagnosticsInfo`].
    pub diagnostics: DiagnosticsInfo,
}

/// The live MCP endpoint surfaced to the Settings → MCP pane via
/// `get_mcp_server_info` (Phase 10). The bearer `token` is sensitive: it is
/// never logged and only crosses the IPC boundary on this explicit read.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, specta::Type)]
pub struct McpServerInfo {
    /// The full Streamable HTTP endpoint URL, e.g. `http://127.0.0.1:8765/mcp`.
    pub url: String,
    /// The bearer token the bridge must present.
    pub token: String,
}

impl IpcState {
    /// Resolve the **held** summariser, loading the GGUF once on first use.
    ///
    /// A thin delegate to [`ChatHandles::ensure_summariser`], which is the single
    /// load implementation (model-id resolution, directory ensure, the
    /// `spawn_blocking` GGUF open, and the load-time GPU placement) — see its doc
    /// for the load logic. This wrapper exists so the UI `summarise` / chat paths
    /// can call it directly off `IpcState` without first materialising a
    /// [`ChatHandles`]; both routes share the SAME lazily-loaded model `Arc`.
    pub async fn ensure_summariser(&self) -> AppResult<Arc<LlamaSummariser>> {
        self.chat_handles().ensure_summariser().await
    }

    /// Resolve the held BGE-M3 embedder (RAG), loading it once on first use.
    /// Shares the same `Arc<OnceCell>` as the write path, so it loads once.
    pub async fn ensure_embedder(&self) -> AppResult<Arc<dyn minutist_common::Embedder>> {
        self.chat_handles().ensure_embedder().await
    }

    /// The held embedder ONLY if already loaded (non-blocking peek; never loads or
    /// downloads). The write path loads it; `ToolContext` build uses this so a chat
    /// turn that never retrieves doesn't trigger a model download.
    pub fn embedder_if_loaded(&self) -> Option<Arc<dyn minutist_common::Embedder>> {
        self.chat_runtime.embedder.get().cloned()
    }

    /// Bundle the chat-runtime handles (held model + persistence + settings +
    /// event bus) into a [`ChatHandles`]. Both the UI chat path (via
    /// [`Self::ensure_summariser`]) and the Phase-10 inter-agent driver use this
    /// so the held-model load logic lives in one place and both share the SAME
    /// lazily-loaded model.
    pub fn chat_handles(&self) -> ChatHandles {
        ChatHandles {
            orchestrator: Arc::clone(&self.orchestrator),
            index: Arc::clone(&self.index),
            meetings_dir: self.meetings_dir.clone(),
            event_tx: self.event_tx.clone(),
            settings: self.settings.clone(),
            summariser: Arc::clone(&self.chat_runtime.summariser),
            embedder: Arc::clone(&self.chat_runtime.embedder),
        }
    }

    /// Log the GPU probe + the resolved plan for the current settings, for
    /// diagnosing the VRAM auto-detection on real hardware (the thresholds are
    /// estimates pending this evidence).
    ///
    /// `probe_primary_gpu` initialises the llama backend if needed, so call this
    /// off the async runtime at startup (it can block). See
    /// `architecture/cross-cutting.md` — "GPU portability".
    pub fn log_gpu_probe(&self) {
        let s = self.settings.current();
        let (probe, plan) = minutist_common::probe_and_resolve_gpu_plan(s.gpu_acceleration);
        match &probe {
            Some(p) => tracing::info!(
                target: "ipc-bridge",
                gpu = %p.name,
                total_mb = p.total_bytes / 1048576,
                free_mb = p.free_bytes / 1048576,
                integrated = p.is_integrated,
                mode = ?s.gpu_acceleration,
                summariser_gpu = plan.summariser_gpu,
                asr_gpu = plan.asr_gpu,
                effective_prefer_large = plan.effective_prefer_large,
                "GPU probe + resolved plan"
            ),
            None => tracing::info!(
                target: "ipc-bridge",
                mode = ?s.gpu_acceleration,
                "GPU probe: no usable GPU (CPU fallback)"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Meeting-index bootstrap helper
// ---------------------------------------------------------------------------

/// Open the libsql meeting index and rebuild it from disk, for `app-main` to
/// inject into [`IpcState`].
///
/// Resolves `index.db` under `app_data_root` via
/// `persistence::index::index_db_path`, opens the index (running its forward-only
/// migrations), and rebuilds it from the per-meeting folders under
/// `meetings_root` (the index is a derived cache, so a startup rebuild makes it
/// converge even after a crash between a folder write and an index update).
///
/// libsql is async; this helper drives the open + rebuild on
/// `tauri::async_runtime::block_on`. It is **startup-only** — the no-`block_on`
/// rule binds Tauri command handlers, not bootstrap. Keeping the helper here
/// (rather than in `app-main`) preserves the dependency table: `ipc-bridge`
/// owns the `persistence` edge, `app-main` does not.
///
/// A rebuild failure is logged and swallowed (the existing index is kept) so a
/// single unreadable folder never blocks startup.
///
/// `index.db` itself is a derived cache (see the module doc on
/// `persistence::index`), so a corrupt or unreadable file is never fatal: a
/// failed open quarantines the file (and its WAL/SHM sidecars, if any) under a
/// fixed `.corrupt` suffix — overwriting any previous quarantine — recreates a
/// fresh `index.db`, and rebuilds it from `meetings_root`. This never panics;
/// the absolute last resort (the fresh open itself also failing, e.g. an
/// unwritable app-data directory) falls back to an in-memory index so startup
/// still completes, degraded rather than aborted.
pub fn open_meeting_index(
    app_data_root: &std::path::Path,
    meetings_root: &std::path::Path,
) -> (PathBuf, Arc<MeetingIndex>) {
    let index_db_path = persistence::index::index_db_path(app_data_root);
    let meetings_root = meetings_root.to_path_buf();
    let db_path = index_db_path.clone();

    let index = tauri::async_runtime::block_on(async move {
        let index = match MeetingIndex::open(&db_path).await {
            Ok(index) => index,
            Err(e) => {
                tracing::warn!(
                    target: "ipc-bridge",
                    path = %db_path.display(),
                    "index.db open failed ({e}); quarantining the corrupt file and rebuilding"
                );
                quarantine_corrupt_db_file(&db_path);
                match MeetingIndex::open(&db_path).await {
                    Ok(index) => index,
                    Err(e) => {
                        tracing::error!(
                            target: "ipc-bridge",
                            path = %db_path.display(),
                            "index.db could not be recreated after quarantining ({e}); \
                             falling back to an in-memory index for this session"
                        );
                        MeetingIndex::open(":memory:")
                            .await
                            .expect("in-memory libsql database must always open")
                    }
                }
            }
        };
        match index.rebuild_from_disk(&meetings_root).await {
            Ok(n) => tracing::info!(
                target: "ipc-bridge",
                indexed = n,
                "index.db rebuilt from disk on startup"
            ),
            Err(e) => tracing::warn!(
                target: "ipc-bridge",
                "index.db rebuild on startup failed: {e}; continuing with existing index"
            ),
        }
        Arc::new(index)
    });

    (index_db_path, index)
}

/// Move a corrupt/unreadable `index.db` (plus its `-wal`/`-shm` sidecars, if
/// present) aside to a fixed `.corrupt` suffix so a fresh database can be
/// created at the original path.
///
/// The suffix is fixed rather than timestamped (no clock is threaded through
/// this helper) — a repeat corruption simply overwrites the previous
/// quarantine copy. Best-effort: a rename failure is logged and otherwise
/// ignored, since the subsequent open attempt is the real success signal.
fn quarantine_corrupt_db_file(db_path: &std::path::Path) {
    for suffix in ["", "-wal", "-shm"] {
        let src = append_to_file_name(db_path, suffix);
        if !src.exists() {
            continue;
        }
        let dest = append_to_file_name(db_path, &format!("{suffix}.corrupt"));
        if let Err(e) = std::fs::rename(&src, &dest) {
            tracing::warn!(
                target: "ipc-bridge",
                path = %src.display(),
                "failed to quarantine corrupt index.db file component ({e})"
            );
        }
    }
}

/// Append `suffix` to a path's file name, e.g. `index.db` + `-wal` ->
/// `index.db-wal`.
fn append_to_file_name(path: &std::path::Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

// ---------------------------------------------------------------------------
// Voiceprint store bootstrap helper
// ---------------------------------------------------------------------------

/// Open the `voiceprints.db` store for `app-main` to inject into [`IpcState`].
///
/// Resolves `voiceprints.db` under `app_data_root` via
/// `persistence::voiceprints_db_path`, opens the store (running the
/// forward-only migration runner), and wraps it in `Arc<Option<...>>`.
///
/// On any open or migration error the error is logged and `Arc::new(None)` is
/// returned — the caller stores `None` on `IpcState::voiceprints`, and the
/// `set_speaker_name` enrolment path skips silently when it observes `None`
/// (the corruption-degrade-to-OFF contract from §2.2).
///
/// This helper mirrors [`open_meeting_index`]: it is startup-only and drives the
/// async `open` on `tauri::async_runtime::block_on`. Keeping it here preserves
/// the dependency table — `ipc-bridge` owns the `persistence` edge;
/// `app-main` does not.
pub fn open_voiceprints(
    app_data_root: &std::path::Path,
) -> Arc<Option<persistence::VoiceprintStore>> {
    let db_path = persistence::voiceprints_db_path(app_data_root);
    tauri::async_runtime::block_on(async move {
        match persistence::VoiceprintStore::open(&db_path).await {
            Ok(store) => {
                tracing::info!(
                    target: "ipc-bridge",
                    path = %db_path.display(),
                    "voiceprints.db opened"
                );
                Arc::new(Some(store))
            }
            Err(e) => {
                tracing::warn!(
                    target: "ipc-bridge",
                    path = %db_path.display(),
                    "voiceprints.db open failed ({e}); voiceprint enrolment degraded to OFF"
                );
                Arc::new(None)
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_meeting_index_recovers_from_a_corrupt_db_file() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let app_data_root = tempdir.path();
        let meetings_root = tempdir.path().join("meetings");
        std::fs::create_dir_all(&meetings_root).expect("meetings root");

        let db_path = persistence::index::index_db_path(app_data_root);
        std::fs::write(&db_path, b"not a valid sqlite/libsql database file")
            .expect("write corrupt index.db");

        // Must not panic, and must return a working index rebuilt from disk.
        let (returned_path, index) = open_meeting_index(app_data_root, &meetings_root);
        assert_eq!(returned_path, db_path);
        let meetings = tauri::async_runtime::block_on(index.list_meetings())
            .expect("recovered index should be queryable");
        assert!(meetings.is_empty());

        // The corrupt file was quarantined (renamed aside), not silently
        // deleted or left blocking the fresh database.
        let quarantined = append_to_file_name(&db_path, ".corrupt");
        assert!(
            quarantined.exists(),
            "corrupt index.db should be quarantined at {quarantined:?}"
        );
    }
}
