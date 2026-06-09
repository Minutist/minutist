//! `ipc-bridge` — Tauri command + event surface for meeting-app.
//!
//! This is the **only** crate in the workspace that imports `tauri::*`.
//! Every other crate is free of Tauri imports, which keeps them testable
//! without a running Tauri app.
//!
//! ## Commands (26 total)
//!
//! | Command | Returns | Phase |
//! |---|---|---|
//! | `list_devices` | `Vec<AudioDevice>` | 1 |
//! | `start_recording` | `MeetingId` | 1 |
//! | `pause_recording` | `()` | 1 |
//! | `resume_recording` | `()` | 1 |
//! | `stop_recording` | `MeetingMeta` | 1 |
//! | `get_recording_state` | `RecordingState` | 1 |
//! | `get_settings` | `Settings` | 1 |
//! | `update_settings` | `()` | 1 |
//! | `list_models` | `Vec<ModelStatus>` | 2 |
//! | `ensure_model` | `()` | 2 |
//! | `save_notes` | `()` | 3 |
//! | `load_notes` | `Option<NotesDocument>` | 3 |
//! | `list_meetings` | `Vec<MeetingListEntry>` | 4 |
//! | `open_meeting` | `MeetingState` | 4 |
//! | `rename_meeting` | `()` | 4 |
//! | `delete_meeting` | `()` | 4 |
//! | `re_transcribe` | `()` | 4 |
//! | `summarise_meeting` | `()` | 5 |
//! | `get_summary` | `Option<String>` | 5 |
//! | `save_summary` | `()` | 5 |
//! | `rediarize_meeting` | `()` | 6 |
//! | `send_chat_message` | `ChatSessionId` | 9 |
//! | `get_chat_session` | `Option<ChatSession>` | 9 |
//! | `list_chat_sessions` | `Vec<ChatSession>` | 9 |
//! | `delete_chat_session` | `()` | 9 |
//! | `get_mcp_server_info` | `Option<McpServerInfo>` | 10 |
//!
//! The Phase-4 `re_summarise` stub (which returned `Unsupported`) was removed
//! in Phase 5 once `summarise_meeting` landed: the meeting-list row's Summarise
//! action points at `summarise_meeting`, so the stub had no caller.
//!
//! The Phase-9 chat commands realise the granted `ipc-bridge → agent-tools` +
//! `ipc-bridge → chat-agent` edges. `send_chat_message` creates/loads the chat
//! [`meeting_app_common::ChatSession`], appends the user message, and spawns the
//! turn on a background task; the turn streams via the chat `AppEvent`s and the
//! session is persisted via `persistence::ChatStore` at turn end. The held LLM
//! substrate ([`IpcState::summariser`], a lazily-loaded `Arc<LlamaSummariser>`)
//! is loaded once and shared by both the chat engine (which borrows
//! `&LlamaModel`) and `summarise_meeting` (Phase 9 — C2).
//!
//! All commands return `Result<T, IpcError>`.
//!
//! `save_notes` / `load_notes` route **directly** to `persistence::NotesStore`
//! against `IpcState::meetings_dir`, bypassing the orchestrator: notes I/O is
//! independent of the live recording pipeline and may run concurrently with an
//! active recording.
//!
//! The Phase-4 read/action commands route directly to `persistence`
//! (`list_meetings` / `rename_meeting` / `delete_meeting` via the shared
//! `IpcState::index`; `open_meeting` via `read_meeting_state`), except
//! `re_transcribe`, which routes to `Orchestrator::re_transcribe` (offline
//! re-run of the live ASR pipeline).
//!
//! The Phase-5 summary commands (`summarise_meeting` / `get_summary` /
//! `save_summary`) realise the granted `ipc-bridge → summariser` edge.
//! `summarise_meeting` resolves the selected LLM directory via
//! `Orchestrator::ensure_model_path` (keeping the `model-registry` edge in the
//! orchestrator — there is **no** `orchestrator → summariser` edge), opens a
//! `summariser::LlamaSummariser`, runs it over the meeting's transcript + notes
//! on `spawn_blocking`, writes `summary.md` via `persistence::write_summary`,
//! and emits `AppEvent::SummaryReady` on `IpcState::event_tx`. `get_summary` /
//! `save_summary` route directly to `persistence::{read_summary, write_summary}`.
//!
//! The Phase-6 `rediarize_meeting` command routes to `Orchestrator::rediarize`
//! (the offline re-diarize). The diarizer is built **inside the orchestrator**
//! (which holds the granted `orchestrator → diarizer` edge and resolves the
//! diarize models via `model-registry`), so there is **no**
//! `ipc-bridge → diarizer` edge — `ipc-bridge` routes via the orchestrator. The
//! `AppEvent::DiarizationComplete` event is emitted by the **orchestrator**, not
//! here.
//!
//! ## Specta types
//!
//! `common` and `settings` derive `specta::Type` directly behind their
//! optional `specta` feature, which `ipc-bridge` enables. The mirror layer
//! that Phase 1 carried in `specta_types.rs` was removed in P0a; the
//! generated TS bindings consume the `common` / `settings` types directly.
//!
//! ## Events
//!
//! `AppEventPayload` is a `#[serde(transparent)]` newtype around
//! `common::AppEvent`. The wire name is `"app-event-payload"`.
//! `spawn_event_forwarder` subscribes to the orchestrator's broadcast
//! channel and emits each event to all Tauri windows.
//!
//! ## Tracing
//!
//! All log calls use `target: "ipc-bridge"`.

pub mod chat;
pub mod chat_runtime;
pub mod commands;
pub mod error;
pub mod events;
pub mod inter_agent;

use std::path::PathBuf;
use std::sync::Arc;

use agent_tools::ToolRegistry;
use meeting_app_common::AppEvent;
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use settings::SettingsHandle;
use summariser::LlamaSummariser;
use tauri_specta::{collect_commands, collect_events, Builder};
use tokio::sync::{broadcast, OnceCell};

pub use chat_runtime::ChatHandles;
pub use error::{Error, IpcError};
pub use events::{spawn_event_forwarder, AppEventPayload};
pub use inter_agent::spawn_inter_agent_driver;

// ---------------------------------------------------------------------------
// IpcState — Tauri managed state
// ---------------------------------------------------------------------------

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
    /// The lazily-loaded, **held** LLM summariser substrate (Phase 9, C2).
    ///
    /// The GGUF is loaded **once** on first chat/summarise use (via
    /// [`Self::ensure_summariser`]) and retained for the process lifetime, rather
    /// than reloaded per call. Both the one-shot `summarise_meeting` path and the
    /// chat driver share this handle: the chat engine borrows `&LlamaModel` via
    /// `LlamaSummariser::model()`, and the `agent-tools` `ToolContext`'s
    /// `resummarise` coerces it to `Arc<dyn Summariser>`. `tokio::sync::OnceCell`
    /// gives a `Send + Sync`, awaitable single-init that the async command
    /// handlers can drive without a `block_on`. See
    /// `architecture/cross-cutting.md` — "Agent chat loop".
    pub summariser: Arc<OnceCell<Arc<LlamaSummariser>>>,
    /// The chat tool registry, built once (`ToolRegistry::v1(false)` — the
    /// inter-agent bridge tool is Phase 10). Shared by the chat driver, which
    /// reads `descriptors()` for the offered tools and `dispatch(...)` to run
    /// them. Held as `Arc` so it crosses into the per-turn `spawn_blocking`
    /// dispatch closure.
    pub tool_registry: Arc<ToolRegistry>,
    /// Per-session in-flight guard for the chat driver. A session id present in
    /// this set has a turn currently running; `send_chat_message` rejects a
    /// concurrent turn for the same session (§6 — single in-flight turn). A
    /// `std::sync::Mutex<HashSet>` (not async) because every access is a brief,
    /// non-awaiting insert/remove. SHARED with the Phase-10 inter-agent driver
    /// (so an external turn and a human turn cannot run on one session at once).
    pub chat_in_flight:
        Arc<std::sync::Mutex<std::collections::HashSet<meeting_app_common::ChatSessionId>>>,
    /// The live MCP server endpoint, set by `app-main` after `mcp_server::serve`
    /// binds (Phase 10). `None` when the MCP server is disabled or not yet
    /// listening. The `get_mcp_server_info` command reads it to reveal the URL +
    /// bearer token for the user to paste into an external MCP client's config. The
    /// token is held here (not on the event bus) and revealed only on explicit
    /// user request.
    pub mcp_info: Arc<std::sync::Mutex<Option<McpServerInfo>>>,
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
    /// The model id is the user-selected `settings.llm_model_id` or the bundled
    /// default ([`commands::DEFAULT_LLM_MODEL_ID`]); the directory is resolved
    /// (downloaded + verified when absent) via `Orchestrator::ensure_model_path`
    /// — keeping the `model-registry` edge inside the orchestrator. The GGUF is
    /// opened on `spawn_blocking` (the heavy load is synchronous) with the
    /// GPU-offload count resolved from the `gpu_acceleration` setting **at load
    /// time** (a held model is loaded once, so its GPU placement is fixed for the
    /// process — toggling the setting takes effect on the next start). Subsequent
    /// calls return the cached `Arc` without reloading.
    pub async fn ensure_summariser(&self) -> Result<Arc<LlamaSummariser>, IpcError> {
        self.chat_handles().ensure_summariser().await
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
            summariser: Arc::clone(&self.summariser),
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
pub fn open_meeting_index(
    app_data_root: &std::path::Path,
    meetings_root: &std::path::Path,
) -> (PathBuf, Arc<MeetingIndex>) {
    let index_db_path = persistence::index::index_db_path(app_data_root);
    let meetings_root = meetings_root.to_path_buf();
    let db_path = index_db_path.clone();

    let index = tauri::async_runtime::block_on(async move {
        let index = MeetingIndex::open(&db_path)
            .await
            .expect("failed to open index.db");
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

// ---------------------------------------------------------------------------
// bindings_builder — shared builder for app-main and the export helper
// ---------------------------------------------------------------------------

/// Construct a `tauri_specta::Builder` pre-loaded with all Phase 1–9 commands
/// and the `AppEventPayload` event.
///
/// Both `app-main` (to build the invoke handler) and a bindings-export helper
/// binary can call this function to get the same builder, ensuring the
/// generated TypeScript bindings are always in sync with the runtime handler.
///
/// # Usage
///
/// ```rust,ignore
/// let builder = ipc_bridge::bindings_builder();
///
/// // In app-main — wire into Tauri:
/// tauri::Builder::default()
///     .manage(ipc_state)
///     .invoke_handler(builder.invoke_handler())
///     .setup(move |app| { builder.mount_events(app); Ok(()) });
///
/// // In a bindings-export binary:
/// builder
///     .export(specta_typescript::Typescript::default(), "ui/src/ipc/bindings.ts")
///     .expect("export failed");
/// ```
pub fn bindings_builder() -> Builder<tauri::Wry> {
    Builder::<tauri::Wry>::new()
        .commands(collect_commands![
            commands::list_devices,
            commands::start_recording,
            commands::pause_recording,
            commands::resume_recording,
            commands::stop_recording,
            commands::get_recording_state,
            commands::get_settings,
            commands::update_settings,
            commands::list_models,
            commands::ensure_model,
            commands::save_notes,
            commands::load_notes,
            commands::list_meetings,
            commands::open_meeting,
            commands::rename_meeting,
            commands::delete_meeting,
            commands::re_transcribe,
            commands::summarise_meeting,
            commands::get_summary,
            commands::save_summary,
            commands::rediarize_meeting,
            commands::send_chat_message,
            commands::get_chat_session,
            commands::list_chat_sessions,
            commands::delete_chat_session,
            commands::get_mcp_server_info,
        ])
        .events(collect_events![AppEventPayload])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tauri_specta::Event;

    /// Verify that `bindings_builder()` produces a builder with the full command
    /// surface registered, by inspecting the TypeScript export.
    ///
    /// tauri-specta rc.21 does not expose the internal command list publicly.
    /// We use `export_str` to generate the TypeScript bindings string and scan
    /// it for each expected command name.  Each command appears in the TS
    /// as a string literal in the `invoke` call.
    ///
    /// Command-count ledger: P1 8 → P2 10 → P3 12 → P4 18 → P5 20 → P6 21 → P9 25
    /// → P10 26 (P5 removes `re_summarise` and adds `summarise_meeting` /
    /// `get_summary` / `save_summary`: 18 − 1 + 3 = 20; P6 adds `rediarize_meeting`:
    /// 20 + 1 = 21; P9 adds `send_chat_message` / `get_chat_session` /
    /// `list_chat_sessions` / `delete_chat_session`: 21 + 4 = 25; P10 adds
    /// `get_mcp_server_info`: 25 + 1 = 26).
    ///
    /// `BigIntExportBehavior::Number` is used to allow `u64` fields (e.g.,
    /// timestamps and byte counts) to export as TypeScript `number` rather
    /// than erroring.  This matches the Handy project's pattern per Phase 1
    #[test]
    fn bindings_builder_registers_expected_command_ledger() {
        use specta_typescript::{BigIntExportBehavior, Typescript};

        let builder = bindings_builder();
        let ts = builder
            .export_str(Typescript::default().bigint(BigIntExportBehavior::Number))
            .expect("export_str should succeed for a correctly-configured builder");

        // Each command appears as a string literal in the `invoke(...)` call.
        let expected = [
            "list_devices",
            "start_recording",
            "pause_recording",
            "resume_recording",
            "stop_recording",
            "get_recording_state",
            "get_settings",
            "update_settings",
            "list_models",
            "ensure_model",
            "save_notes",
            "load_notes",
            "list_meetings",
            "open_meeting",
            "rename_meeting",
            "delete_meeting",
            "re_transcribe",
            "summarise_meeting",
            "get_summary",
            "save_summary",
            "rediarize_meeting",
            "send_chat_message",
            "get_chat_session",
            "list_chat_sessions",
            "delete_chat_session",
            "get_mcp_server_info",
        ];

        assert_eq!(
            expected.len(),
            26,
            "command ledger must be 26 after Phase 10 (Phase 9's 25 + get_mcp_server_info)"
        );

        // `re_summarise` was removed in Phase 5 (no caller once
        // `summarise_meeting` landed); assert it is gone from the surface.
        assert!(
            !ts.contains("re_summarise"),
            "re_summarise must be removed from the command surface in Phase 5"
        );

        for name in &expected {
            assert!(
                ts.contains(name),
                "expected command '{name}' not found in generated TypeScript:\n{ts}"
            );
        }
    }

    /// Verify that `AppEventPayload` is registered in the builder's event
    /// registry, by checking its `Event::NAME` constant appears in the TS
    /// export.
    #[test]
    fn bindings_builder_registers_app_event_payload() {
        use specta_typescript::{BigIntExportBehavior, Typescript};

        let builder = bindings_builder();
        let ts = builder
            .export_str(Typescript::default().bigint(BigIntExportBehavior::Number))
            .expect("export_str should succeed");

        let event_name = events::AppEventPayload::NAME;
        assert!(
            ts.contains(event_name),
            "AppEventPayload event '{event_name}' not found in generated TypeScript:\n{ts}"
        );
    }
}
