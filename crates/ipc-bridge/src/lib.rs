//! `ipc-bridge` — Tauri command + event surface for minutist.
//!
//! This is the **only** crate in the workspace that imports `tauri::*`.
//! Every other crate is free of Tauri imports, which keeps them testable
//! without a running Tauri app.
//!
//! ## Commands (35 total)
//!
//! | Command | Returns | Phase |
//! |---|---|---|
//! | `list_devices` | `Vec<AudioDevice>` | 1 |
//! | `start_recording` | `MeetingId` | 1 |
//! | `prewarm_asr` | `()` | live-test UX |
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
//! | `apply_notes_update` | `()` | CRDT editor binding (B6 WU7) |
//! | `load_notes_ydoc` | `Option<Vec<u8>>` (v1 state) | CRDT editor binding (B6 WU7) |
//! | `save_note_image` | `String` (portable asset ref) | notes images |
//! | `list_meetings` | `Vec<MeetingListEntry>` | 4 |
//! | `open_meeting` | `MeetingState` | 4 |
//! | `rename_meeting` | `()` | 4 |
//! | `set_speaker_name` | `()` | transcript speaker rename |
//! | `delete_meeting` | `()` | 4 |
//! | `re_transcribe` | `()` | 4 |
//! | `summarise_meeting` | `()` | 5 |
//! | `get_summary` | `Option<String>` | 5 |
//! | `save_summary` | `()` | 5 |
//! | `rediarize_meeting` | `()` | 6 |
//! | `send_chat_message` | `ChatSessionId` | 9 |
//! | `cancel_chat_turn` | `()` | 9 |
//! | `get_chat_session` | `Option<ChatSession>` | 9 |
//! | `list_chat_sessions` | `Vec<ChatSession>` | 9 |
//! | `delete_chat_session` | `()` | 9 |
//! | `get_mcp_server_info` | `Option<McpServerInfo>` | 10 |
//! | `translate_meeting` | `()` | translation |
//! | `get_translations` | `HashMap<usize, String>` | translation |
//! | `get_diagnostic_report` | `DiagnosticReport` | report (#0014) |
//!
//! The Phase-4 `re_summarise` stub (which returned `Unsupported`) was removed
//! in Phase 5 once `summarise_meeting` landed: the meeting-list row's Summarise
//! action points at `summarise_meeting`, so the stub had no caller.
//!
//! The Phase-9 chat commands realise the granted `ipc-bridge → agent-tools` +
//! `ipc-bridge → chat-agent` edges. `send_chat_message` creates/loads the chat
//! [`minutist_common::ChatSession`], appends the user message, and spawns the
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
//! The CRDT editor binding (B6 WU7, `planning/DESIGN_notes-crdt.md` §8) adds two
//! more notes commands on the same direct `persistence::NotesStore` route:
//! `apply_notes_update` (merge an editor-produced lib0-v1 Yjs update onto the
//! authoritative `notes.ydoc`, then re-derive `notes.json` / `notes.md`) and
//! `load_notes_ydoc` (read the stored doc as a v1 state update for the editor to
//! `Y.applyUpdate` on open). For an open Yjs-native editor `apply_notes_update`
//! is the PRIMARY write path — it preserves CRDT history, whereas `save_notes`
//! rebuilds the doc from JSON (kept for back-compat / non-collab callers). The
//! binary update crosses the wire as `Vec<u8>` (`number[]`), matching
//! `save_note_image`. The on-disk durable blob stays v2; only the editor
//! interchange is v1 (the two must not be crossed — see `persistence::ydoc`).
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
pub mod diagnostics;
pub mod error;
pub mod events;
pub mod inter_agent;
pub mod output_language;

use std::path::PathBuf;
use std::sync::Arc;

use agent_tools::ToolRegistry;
use minutist_common::AppEvent;
use orchestrator::Orchestrator;
use settings::SettingsHandle;
use tauri_specta::{collect_commands, collect_events, Builder};
use tokio::sync::{broadcast, OnceCell};

pub use chat_runtime::ChatHandles;
pub use error::{Error, IpcError};
pub use events::{spawn_event_forwarder, AppEventPayload};
pub use inter_agent::spawn_inter_agent_driver;
/// Re-exports so `app-main` can name these types without direct deps on
/// `persistence` and `summariser`; both already appear in the public API
/// (`open_meeting_index` returns `Arc<MeetingIndex>`; `IpcState::summariser`
/// holds `Arc<OnceCell<Arc<LlamaSummariser>>>`).
pub use persistence::MeetingIndex;
pub use summariser::LlamaSummariser;

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
        Arc<std::sync::Mutex<std::collections::HashSet<minutist_common::ChatSessionId>>>,
    /// Per-session chat-turn cancellation flags (P1). A running turn registers a
    /// `chat_agent::CancelFlag` here keyed by session id; `cancel_chat_turn`
    /// raises it, and the decode loop stops at the next between-token check. The
    /// driver removes the entry when the turn ends. A `std::sync::Mutex<HashMap>`
    /// (not async) because every access is a brief, non-awaiting insert / get /
    /// remove, mirroring `chat_in_flight`.
    pub chat_cancel: Arc<
        std::sync::Mutex<
            std::collections::HashMap<minutist_common::ChatSessionId, chat_agent::CancelFlag>,
        >,
    >,
    /// Per-`(meeting_id, language)` in-flight guard for the translation driver.
    /// A pair present in this set has a `translate_meeting` call currently
    /// running for that language; a second call for the same pair is rejected
    /// (mirrors `chat_in_flight`). A `std::sync::Mutex<HashSet>` (not async)
    /// because every access is a brief, non-awaiting insert/remove.
    pub translate_in_flight:
        Arc<std::sync::Mutex<std::collections::HashSet<(minutist_common::MeetingId, String)>>>,
    /// The live MCP server endpoint, set by `app-main` after `mcp_server::serve`
    /// binds (Phase 10). `None` when the MCP server is disabled or not yet
    /// listening. The `get_mcp_server_info` command reads it to reveal the URL +
    /// bearer token for the user to paste into an external MCP client's config. The
    /// token is held here (not on the event bus) and revealed only on explicit
    /// user request.
    pub mcp_info: Arc<std::sync::Mutex<Option<McpServerInfo>>>,
    /// The logs directory (`{app-data}/logs/`), owned by `app-main` but its path
    /// is shared here so `get_diagnostic_report` (#0014) can read the rolling
    /// `minutist.log` tail and the `last-crash.txt` written by the panic hook.
    /// `ipc-bridge` only READS this directory; `app-main` owns writes to it.
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

    /// Log the GPU probe + the resolved plan for the current settings, for
    /// diagnosing the VRAM auto-detection on real hardware (the thresholds are
    /// estimates pending this evidence).
    ///
    /// `probe_primary_gpu` initialises the llama backend if needed, so call this
    /// off the async runtime at startup (it can block). See
    /// `architecture/cross-cutting.md` — "GPU portability".
    pub fn log_gpu_probe(&self) {
        let s = self.settings.current();
        let probe = minutist_common::probe_primary_gpu();
        let plan = minutist_common::resolve_gpu_plan(
            probe.as_ref(),
            s.gpu_acceleration,
            true, // always request the large tier; the VRAM clamp in resolve_gpu_plan decides
        );
        match &probe {
            Some(p) => tracing::info!(
                target: "app-main",
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
                target: "app-main",
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
// Note-image asset serving — the `meetingasset:` URI scheme resolver
// ---------------------------------------------------------------------------

/// The custom URI scheme name used to serve note image assets to the webview.
///
/// `app-main` registers a protocol handler under this name on the Tauri
/// `Builder`; the frontend turns a stored portable filename into a working URL
/// via `convertFileSrc(<meeting_id>/<filename>, MEETING_ASSET_SCHEME)`. Tauri
/// serves it at `meetingasset://localhost/<path>` (macOS/Linux) or
/// `http://meetingasset.localhost/<path>` (Windows) — both deliver the SAME
/// request path to the handler, so the platform difference is invisible here.
pub const MEETING_ASSET_SCHEME: &str = "meetingasset";

/// A resolved note-image asset: its bytes plus the MIME content type to serve.
pub struct ResolvedNoteAsset {
    /// The asset file bytes.
    pub bytes: Vec<u8>,
    /// The `Content-Type` for the response, inferred from the extension.
    pub content_type: &'static str,
}

/// Resolve a `meetingasset:` request path to the asset bytes + content type.
///
/// The request path is `/<meeting_id>/<filename>` (URL path component, as
/// produced by `convertFileSrc(<meeting_id>/<filename>, MEETING_ASSET_SCHEME)`).
/// This:
///
/// 1. Splits the path into exactly `meeting_id` + `filename` (rejecting any
///    extra segments — no nesting),
/// 2. Parses `meeting_id` as a UUID (rejecting anything else),
/// 3. Reads the bytes via `persistence::read_note_asset`, which applies its own
///    path-traversal guard on `filename` (separator / `..` rejected),
/// 4. Infers the `Content-Type` from the filename extension.
///
/// Lives here (not in `app-main`) so the `persistence` dependency edge stays
/// inside `ipc-bridge` (the dependency table grants `ipc-bridge → persistence`,
/// NOT `app-main → persistence`). `app-main` calls this from its registered
/// protocol handler and only handles the HTTP-response shaping.
///
/// Returns `AppError::InvalidInput` for a malformed path / id, and surfaces the
/// `persistence` error (traversal rejection → `InvalidInput`; missing file →
/// `Io`) otherwise. The handler maps any error to a 404/empty response, so no
/// detail leaks to the webview.
pub fn resolve_note_asset(
    meetings_dir: &std::path::Path,
    request_path: &str,
) -> Result<ResolvedNoteAsset, minutist_common::AppError> {
    use minutist_common::{AppError, MeetingId};

    // Strip the leading '/', then split into exactly two non-empty segments.
    let trimmed = request_path.trim_start_matches('/');
    let mut parts = trimmed.splitn(2, '/');
    let (id_str, filename) = match (parts.next(), parts.next()) {
        (Some(id), Some(file)) if !id.is_empty() && !file.is_empty() => (id, file),
        _ => {
            return Err(AppError::InvalidInput {
                context: format!("malformed meetingasset path: {request_path:?}"),
            })
        }
    };
    // `filename` must be a single segment — no further '/'.
    if filename.contains('/') {
        return Err(AppError::InvalidInput {
            context: format!("meetingasset path has nested segments: {request_path:?}"),
        });
    }

    let uuid = uuid::Uuid::parse_str(id_str).map_err(|_| AppError::InvalidInput {
        context: format!("meetingasset path has a non-UUID meeting id: {id_str:?}"),
    })?;
    let meeting_id = MeetingId(uuid);

    // `read_note_asset` applies the path-traversal guard on `filename`.
    let bytes = persistence::read_note_asset(meetings_dir, meeting_id, filename)?;
    let content_type = content_type_for(filename);

    Ok(ResolvedNoteAsset {
        bytes,
        content_type,
    })
}

/// Infer the image `Content-Type` from a filename extension.
///
/// Mirrors the `ipc-bridge` `save_note_image` allowlist; an unknown extension
/// falls back to `application/octet-stream` (the persistence layer only ever
/// writes allow-listed extensions, so this is a defensive default).
fn content_type_for(filename: &str) -> &'static str {
    let ext = std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
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
            commands::prewarm_asr,
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
            commands::apply_notes_update,
            commands::load_notes_ydoc,
            commands::save_note_image,
            commands::list_meetings,
            commands::open_meeting,
            commands::rename_meeting,
            commands::set_speaker_name,
            commands::delete_meeting,
            commands::re_transcribe,
            commands::summarise_meeting,
            commands::get_summary,
            commands::save_summary,
            commands::rediarize_meeting,
            commands::send_chat_message,
            commands::cancel_chat_turn,
            commands::get_chat_session,
            commands::list_chat_sessions,
            commands::delete_chat_session,
            commands::get_mcp_server_info,
            commands::translate_meeting,
            commands::get_translations,
            diagnostics::get_diagnostic_report,
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

    // -----------------------------------------------------------------------
    // Note-image asset resolver (`meetingasset:` scheme). No Tauri runtime
    // needed — drives `resolve_note_asset` against a tempdir meetings root.
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_note_asset_serves_saved_image_with_content_type() {
        use minutist_common::MeetingId;
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        persistence::MeetingFolder::create(root, id).expect("folder");

        let bytes = b"\x89PNG\r\n\x1a\nfake".to_vec();
        let filename = persistence::save_note_asset(root, id, &bytes, "png").expect("save");

        // Path as `convertFileSrc(<id>/<filename>)` yields it (leading slash).
        let path = format!("/{}/{}", id.0, filename);
        let resolved = resolve_note_asset(root, &path).expect("resolve");
        assert_eq!(resolved.bytes, bytes);
        assert_eq!(resolved.content_type, "image/png");
    }

    #[test]
    fn resolve_note_asset_rejects_malformed_and_traversal_paths() {
        use minutist_common::{AppError, MeetingId};
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        persistence::MeetingFolder::create(root, id).expect("folder");

        // Malformed (not <uuid>/<file>) → InvalidInput.
        for bad in ["", "/", "/only-one-segment", "/not-a-uuid/file.png"] {
            assert!(
                matches!(resolve_note_asset(root, bad), Err(AppError::InvalidInput { .. })),
                "path {bad:?} should be rejected as InvalidInput"
            );
        }

        // Valid UUID but traversal filename → rejected by persistence guard
        // (InvalidInput), and a nested path → rejected here.
        let traversal = format!("/{}/../../etc/passwd", id.0);
        assert!(resolve_note_asset(root, &traversal).is_err());
        let nested = format!("/{}/sub/dir.png", id.0);
        assert!(
            matches!(resolve_note_asset(root, &nested), Err(AppError::InvalidInput { .. })),
            "nested path should be rejected"
        );
    }

    #[test]
    fn content_type_for_maps_known_extensions() {
        assert_eq!(content_type_for("a.png"), "image/png");
        assert_eq!(content_type_for("a.jpg"), "image/jpeg");
        assert_eq!(content_type_for("a.jpeg"), "image/jpeg");
        assert_eq!(content_type_for("a.gif"), "image/gif");
        assert_eq!(content_type_for("a.webp"), "image/webp");
        assert_eq!(content_type_for("a.bin"), "application/octet-stream");
        assert_eq!(content_type_for("noext"), "application/octet-stream");
    }

    /// Verify that `bindings_builder()` produces a builder with the full command
    /// surface registered, by inspecting the TypeScript export.
    ///
    /// tauri-specta rc.21 does not expose the internal command list publicly.
    /// We use `export_str` to generate the TypeScript bindings string and scan
    /// it for each expected command name.  Each command appears in the TS
    /// as a string literal in the `invoke` call.
    ///
    /// Command-count ledger: P1 8 → P2 10 → P3 12 → P4 18 → P5 20 → P6 21 → P9 25
    /// → P10 26 → P9 review-fix 27 → +prewarm_asr 28 → +save_note_image 29 → P4
    /// transcript-rename adds `set_speaker_name` 30 → +translate_meeting
    /// +get_translations 32 → +get_diagnostic_report 33 (#0014) →
    /// +apply_notes_update +load_notes_ydoc 35 (B6 WU7 — CRDT editor binding) (P5 removes
    /// `re_summarise` and adds `summarise_meeting`
    /// / `get_summary` / `save_summary`: 18 − 1 + 3 = 20; P6 adds
    /// `rediarize_meeting`: 20 + 1 = 21; P9 adds `send_chat_message` /
    /// `get_chat_session` / `list_chat_sessions` / `delete_chat_session`: 21 + 4
    /// = 25; P10 adds `get_mcp_server_info`: 25 + 1 = 26; the P9 chat review-fix
    /// adds `cancel_chat_turn` (P1 — turn cancellation): 26 + 1 = 27;
    /// `prewarm_asr` (live-test UX T2): 27 + 1 = 28; `save_note_image` (note image
    /// paste/drop): 28 + 1 = 29; `set_speaker_name` (transcript speaker rename):
    /// 29 + 1 = 30; translation commands: 30 + 2 = 32; `get_diagnostic_report`
    /// (#0014 — the "Report a problem" snapshot): 32 + 1 = 33;
    /// `apply_notes_update` + `load_notes_ydoc` (B6 WU7 — CRDT editor binding):
    /// 33 + 2 = 35).
    ///
    /// `BigIntExportBehavior::Number` is used to allow `u64` fields (e.g.,
    /// timestamps and byte counts) to export as TypeScript `number` rather
    /// than erroring.  This matches the Handy project's pattern per Phase 1.
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
            "prewarm_asr",
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
            "apply_notes_update",
            "load_notes_ydoc",
            "save_note_image",
            "list_meetings",
            "open_meeting",
            "rename_meeting",
            "set_speaker_name",
            "delete_meeting",
            "re_transcribe",
            "summarise_meeting",
            "get_summary",
            "save_summary",
            "rediarize_meeting",
            "send_chat_message",
            "cancel_chat_turn",
            "get_chat_session",
            "list_chat_sessions",
            "delete_chat_session",
            "get_mcp_server_info",
            "translate_meeting",
            "get_translations",
            "get_diagnostic_report",
        ];

        assert_eq!(
            expected.len(),
            35,
            "command ledger must be 35 (33 + apply_notes_update + load_notes_ydoc, B6 WU7)"
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
