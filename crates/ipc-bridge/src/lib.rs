//! `ipc-bridge` — Tauri command + event surface for minutist.
//!
//! This is the **only** crate in the workspace that imports `tauri::*`.
//! Every other crate is free of Tauri imports, which keeps them testable
//! without a running Tauri app.
//!
//! ## Commands (61 total)
//!
//! | Command | Returns | Phase |
//! |---|---|---|
//! | `list_devices` | `Vec<AudioDevice>` | 1 |
//! | `create_meeting` | `MeetingId` | "New meeting" prep draft |
//! | `start_recording` | `MeetingId` | 1 |
//! | `prewarm_asr` | `()` | live-test UX |
//! | `pause_recording` | `()` | 1 |
//! | `resume_recording` | `()` | 1 |
//! | `set_recording_title` | `()` | live meeting title |
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
//! | `add_attachment` | `AttachmentEntry` | attachments (#0016) |
//! | `list_attachments` | `Vec<AttachmentEntry>` | attachments (#0016) |
//! | `open_attachment` | `()` | attachments (#0016) |
//! | `remove_attachment` | `()` | attachments (#0016) |
//! | `list_meetings` | `Vec<MeetingListEntry>` | 4 |
//! | `open_meeting` | `MeetingState` | 4 |
//! | `rename_meeting` | `()` | 4 |
//! | `set_speaker_name` | `()` | transcript speaker rename |
//! | `delete_meeting` | `()` | 4 |
//! | `open_meeting_folder` | `()` | themed context menus (#0034) |
//! | `list_collections` | `Vec<Collection>` | meeting folders |
//! | `create_collection` | `Collection` | meeting folders |
//! | `rename_collection` | `()` | meeting folders |
//! | `delete_collection` | `()` | meeting folders |
//! | `set_meeting_collection` | `()` | meeting folders |
//! | `reprocess` | `()` | 4/6 (re-transcribe + re-diarize, #0015) |
//! | `summarise_meeting` | `()` | 5 |
//! | `get_summary` | `Option<String>` | 5 |
//! | `save_summary` | `()` | 5 |
//! | `send_chat_message` | `ChatSessionId` | 9 |
//! | `cancel_chat_turn` | `()` | 9 |
//! | `get_chat_session` | `Option<ChatSession>` | 9 |
//! | `list_chat_sessions` | `Vec<ChatSession>` | 9 |
//! | `delete_chat_session` | `()` | 9 |
//! | `get_mcp_server_info` | `Option<McpServerInfo>` | 10 |
//! | `translate_meeting` | `()` | translation |
//! | `get_translations` | `HashMap<usize, String>` | translation |
//! | `get_diagnostic_report` | `DiagnosticReport` | report (#0014) |
//! | `tunnel_begin_pairing` | `PairingPrompt` | WS4-A S5b |
//! | `tunnel_poll_pairing` | `TunnelStatus` | WS4-A S5b |
//! | `set_connector_enabled` | `TunnelSnapshot` | WS4-A S5b |
//! | `tunnel_status` | `TunnelSnapshot` | WS4-A S5b |
//! | `sync_status` | `SyncStatus` | WS4-B S5 |
//! | `sync_get_my_ticket` | `String` | WS4-B S5 |
//! | `sync_add_peer` | `()` | WS4-B S5 |
//! | `sync_now` | `()` | WS4-B S5 |
//! | `reject_match` | `()` | #0003 WU8 |
//! | `clear_all_voiceprints` | `()` | #0003 WU8 (§4 privacy) |
//! | `list_voiceprints` | `Vec<VoiceprintIdentityInfo>` | #0003 WU8 |
//! | `merge_voiceprint_identities` | `()` | #0003 WU8 |
//! | `rename_voiceprint_identity` | `()` | #0003 WU8 |
//! | `delete_voiceprint_identity` | `()` | #0003 WU8 |
//! | `forget_meeting_voiceprints` | `()` | #0003 WU8 |
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
//! All commands return `AppResult<T>`.
//!
//! `save_notes` / `load_notes` route **directly** to `notes_crdt::NotesStore`
//! against `IpcState::meetings_dir`, bypassing the orchestrator: notes I/O is
//! independent of the live recording pipeline and may run concurrently with an
//! active recording.
//!
//! The CRDT editor binding (B6 WU7, `planning/DESIGN_notes-crdt.md` §8) adds two
//! more notes commands on the same direct `notes_crdt::NotesStore` route:
//! `apply_notes_update` (merge an editor-produced lib0-v1 Yjs update onto the
//! authoritative `notes.ydoc`, then re-derive `notes.json` / `notes.md`) and
//! `load_notes_ydoc` (read the stored doc as a v1 state update for the editor to
//! `Y.applyUpdate` on open). For an open Yjs-native editor `apply_notes_update`
//! is the PRIMARY write path — it preserves CRDT history, whereas `save_notes`
//! rebuilds the doc from JSON (kept for back-compat / non-collab callers). The
//! binary update crosses the wire as `Vec<u8>` (`number[]`), matching
//! `save_note_image`. The on-disk durable blob stays v2; only the editor
//! interchange is v1 (the two must not be crossed — see `notes_crdt::ydoc`).
//!
//! The Phase-4 read/action commands route directly to `persistence`
//! (`list_meetings` / `rename_meeting` / `delete_meeting` via the shared
//! `IpcState::index`; `open_meeting` via `read_meeting_state`), except
//! `reprocess`, which routes to `Orchestrator::reprocess` (offline re-run of the
//! live ASR pipeline followed by re-diarization, #0015).
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
//! The `reprocess` command merges the former `re_transcribe` (Phase 4) and
//! `rediarize_meeting` (Phase 6) commands (#0015): one `Orchestrator::reprocess`
//! re-transcribes then re-diarizes under a single offline claim. The diarizer
//! and the ASR backend are built **inside the orchestrator** (which holds the
//! granted `orchestrator → diarizer` edge and resolves models via
//! `model-registry`), so there is **no** `ipc-bridge → diarizer` edge —
//! `ipc-bridge` routes via the orchestrator. The `AppEvent::DiarizationComplete`
//! event is emitted by the **orchestrator**, not here.
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

pub mod attachments;
pub mod chat;
pub mod chat_runtime;
pub mod commands;
pub mod diagnostics;
pub mod error;
pub mod events;
pub mod inter_agent;
mod ipc_state;
pub mod lifecycle;
pub mod live_agent;
pub mod output_language;
pub mod rag_index;
pub mod sync;
pub mod tunnel;

use orchestrator::Orchestrator;
use tauri_specta::{collect_commands, collect_events, Builder};

pub use attachments::{
    requeue_pending, spawn_attachment_convert_worker, ConvertJob, GemmaVlm,
    ATTACHMENT_CONVERT_QUEUE_BOUND,
};
pub use chat_runtime::ChatHandles;
pub use commands::run_held_summarise;
pub use error::Error;
pub use events::{spawn_event_forwarder, AppEventPayload};
pub use inter_agent::spawn_inter_agent_driver;
pub use ipc_state::{
    open_meeting_index, open_voiceprints, ChatRuntimeState, ConnectedState, DiagnosticsInfo,
    IpcState, McpServerInfo,
};
pub use lifecycle::spawn_lifecycle_subscriber;
pub use live_agent::{spawn_live_agent, LiveAgentHandles, LiveCopilotHandle};
/// Re-exports so `app-main` can name these types without direct deps on
/// `persistence` and `summariser`; both already appear in the public API
/// (`open_meeting_index` returns `Arc<MeetingIndex>`;
/// `IpcState::chat_runtime.summariser` holds
/// `Arc<OnceCell<Arc<LlamaSummariser>>>`).
pub use persistence::MeetingIndex;
pub use summariser::LlamaSummariser;
pub use sync::{disabled_sync, DisabledSync, SyncControl};
pub use tunnel::{disabled_tunnel, DisabledTunnel, PairingPrompt, TunnelControl, TunnelSnapshot};

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
/// `convertFileSrc` percent-encodes the whole `<meeting_id>/<filename>` string
/// (the separating `/` arrives as `%2F`, one path segment) and
/// `request.uri().path()` is not pre-decoded, so this:
///
/// 1. Percent-decodes the path via the same [`percent_decode_path`] helper
///    [`parse_recording_request`] uses (a no-op on an already-decoded path, so
///    this is also correct if a caller ever passes one),
/// 2. Splits the decoded path into exactly `meeting_id` + `filename` (rejecting
///    any extra segments — no nesting),
/// 3. Parses `meeting_id` as a UUID (rejecting anything else),
/// 4. Reads the bytes via `persistence::read_note_asset`, which applies its own
///    path-traversal guard on `filename` (separator / `..` rejected),
/// 5. Infers the `Content-Type` from the filename extension.
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

    // Decode first, then strip the leading '/' and split into exactly two
    // non-empty segments — see the decode-then-split note above.
    let decoded = percent_decode_path(request_path);
    let trimmed = decoded.trim_start_matches('/');
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
// Attachment asset serving — the `attachment:` URI scheme resolver
// ---------------------------------------------------------------------------

/// The custom URI scheme name used to serve attachment original bytes to the
/// webview (an inline thumbnail/expand for an `AttachmentRef` note node, #0038).
///
/// Mirrors [`MEETING_ASSET_SCHEME`]: `app-main` registers a protocol handler
/// under this name; the frontend turns a stored portable
/// `(attachment id, filename)` ref into a working URL via
/// `convertFileSrc(<meeting_id>/<filename>, ATTACHMENT_SCHEME)`.
pub const ATTACHMENT_SCHEME: &str = "attachment";

/// A resolved attachment asset: its bytes plus the MIME content type to serve.
pub struct ResolvedAttachmentAsset {
    /// The attachment original's bytes.
    pub bytes: Vec<u8>,
    /// The `Content-Type` for the response, inferred from the extension.
    pub content_type: &'static str,
}

/// Resolve an `attachment:` request path to the attachment original's bytes +
/// content type.
///
/// The request path is `/<meeting_id>/<filename>` (URL path component, as
/// produced by `convertFileSrc(<meeting_id>/<filename>, ATTACHMENT_SCHEME)`,
/// where `filename` is the attachment's content-addressed `<hash>.<ext>` on-disk
/// name). Exactly mirrors [`resolve_note_asset`]'s parsing:
///
/// 1. Percent-decodes the path — `convertFileSrc` encodes the separating `/` as
///    `%2F` (one path segment), and `request.uri().path()` is not pre-decoded,
///    so this MUST decode before splitting (the D1 bug this mirrors: decoding
///    after splitting silently drops the meeting-id/filename separation).
/// 2. Splits into exactly `meeting_id` + `filename` (no nested segments).
/// 3. Parses `meeting_id` as a UUID.
/// 4. Reads the bytes via `persistence::read_attachment_original`, which applies
///    its own path-traversal guard on `filename` (separator / `..` rejected).
/// 5. Infers the `Content-Type` from the filename extension.
///
/// Lives here (not in `app-main`) so the `persistence` dependency edge stays
/// inside `ipc-bridge`. Returns `AppError::InvalidInput` for a malformed
/// path/id, and surfaces the `persistence` error otherwise; the handler maps
/// any error to a 404/empty response, so no detail leaks to the webview.
pub fn resolve_attachment_asset(
    meetings_dir: &std::path::Path,
    request_path: &str,
) -> Result<ResolvedAttachmentAsset, minutist_common::AppError> {
    use minutist_common::{AppError, MeetingId};

    let decoded = percent_decode_path(request_path);
    let trimmed = decoded.trim_start_matches('/');
    let mut parts = trimmed.splitn(2, '/');
    let (id_str, filename) = match (parts.next(), parts.next()) {
        (Some(id), Some(file)) if !id.is_empty() && !file.is_empty() => (id, file),
        _ => {
            return Err(AppError::InvalidInput {
                context: format!("malformed attachment path: {request_path:?}"),
            })
        }
    };
    // `filename` must be a single segment — no further '/'.
    if filename.contains('/') {
        return Err(AppError::InvalidInput {
            context: format!("attachment path has nested segments: {request_path:?}"),
        });
    }

    let uuid = uuid::Uuid::parse_str(id_str).map_err(|_| AppError::InvalidInput {
        context: format!("attachment path has a non-UUID meeting id: {id_str:?}"),
    })?;
    let meeting_id = MeetingId(uuid);

    // `read_attachment_original` applies the path-traversal guard on `filename`.
    let bytes = persistence::read_attachment_original(meetings_dir, meeting_id, filename)?;
    let content_type = content_type_for_attachment(filename);

    Ok(ResolvedAttachmentAsset {
        bytes,
        content_type,
    })
}

/// Infer the `Content-Type` for a served attachment original from its filename
/// extension.
///
/// Only the image extensions matter for correct in-editor rendering (the
/// `AttachmentRef` node's thumbnail); every other supported attachment type
/// (`pdf`, `docx`, `xlsx`, …) serves as `application/octet-stream` — the
/// non-image expand affordance does not render the bytes inline, so a precise
/// MIME type is not load-bearing there.
fn content_type_for_attachment(filename: &str) -> &'static str {
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
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

// ---------------------------------------------------------------------------
// Recording-audio re-listen serving — the `meetingrecording:` URI scheme
// ---------------------------------------------------------------------------

/// The custom URI scheme used to serve decoded recording-audio slices to the
/// webview for the transcript "play segment" affordance.
///
/// `app-main` registers an **asynchronous** protocol handler under this name
/// (the decode + window mapping run on the orchestrator's blocking pool, so the
/// handler cannot be the synchronous `meetingasset:` form). The frontend builds
/// a URL via `convertFileSrc("<meeting_id>/<start_ms>-<end_ms>",
/// MEETING_RECORDING_SCHEME)`; the handler returns a WAV of exactly that
/// transcript window (see `Orchestrator::extract_segment_wav`).
pub const MEETING_RECORDING_SCHEME: &str = "meetingrecording";

/// Resolve a `meetingrecording:` request path to a 16 kHz mono PCM16 WAV of the
/// requested transcript window.
///
/// The request path is `/<meeting_id>/<start_ms>-<end_ms>` (URL path component,
/// as produced by `convertFileSrc(..., MEETING_RECORDING_SCHEME)`; a trailing
/// `.wav` extension, if the frontend adds one, is ignored). This parses the id +
/// window and delegates the decode + pause-aware window mapping + WAV encode to
/// `Orchestrator::extract_segment_wav` (the orchestrator owns the audio-timeline
/// domain). `start_ms`/`end_ms` are transcript-clock (pause-EXCLUDING) ms.
///
/// Lives here (not in `app-main`) so the `orchestrator` dependency edge stays
/// inside `ipc-bridge`; `app-main`'s registered handler only shapes the HTTP
/// response. Returns `AppError::InvalidInput` for a malformed path / id /
/// window, and surfaces the orchestrator error (out-of-range window, decode
/// failure) otherwise; the handler maps any error to a 404 so no detail leaks.
pub async fn resolve_recording_slice(
    orchestrator: &Orchestrator,
    request_path: &str,
) -> Result<Vec<u8>, minutist_common::AppError> {
    let (meeting_id, start_ms, end_ms) = parse_recording_request(request_path)?;
    orchestrator
        .extract_segment_wav(meeting_id, start_ms, end_ms)
        .await
}

/// Parse a `meetingrecording:` request path `/<meeting_id>/<start_ms>-<end_ms>`
/// into `(MeetingId, start_ms, end_ms)`. A trailing `.<ext>` on the window is
/// ignored. Pure (no I/O) so it is unit-testable without an `Orchestrator`;
/// [`resolve_recording_slice`] wraps it.
///
/// Fails closed (`AppError::InvalidInput`) on any malformation: missing/extra
/// path segments, a non-UUID id, a window that is not exactly `<int>-<int>`, or
/// `end <= start`. (The orchestrator re-checks the span, but rejecting here keeps
/// the parse total and self-contained.) `start_ms`/`end_ms` are transcript-clock
/// (pause-EXCLUDING) milliseconds.
fn parse_recording_request(
    request_path: &str,
) -> Result<(minutist_common::MeetingId, u64, u64), minutist_common::AppError> {
    use minutist_common::{AppError, MeetingId};

    // The async URI-scheme handler receives the raw, still-percent-encoded path:
    // `convertFileSrc` runs `encodeURIComponent` over the whole "<id>/<window>"
    // string, so the separating '/' arrives as "%2F" (one path segment). Decode
    // before splitting. Decoding a path with no escapes is a no-op, so this is
    // also correct if a caller ever passes an already-decoded path.
    let decoded = percent_decode_path(request_path);
    let trimmed = decoded.trim_start_matches('/');
    let mut parts = trimmed.splitn(2, '/');
    let (id_str, window) = match (parts.next(), parts.next()) {
        (Some(id), Some(w)) if !id.is_empty() && !w.is_empty() => (id, w),
        _ => {
            return Err(AppError::InvalidInput {
                context: format!("malformed meetingrecording path: {request_path:?}"),
            })
        }
    };
    if window.contains('/') {
        return Err(AppError::InvalidInput {
            context: format!("meetingrecording path has nested segments: {request_path:?}"),
        });
    }

    let uuid = uuid::Uuid::parse_str(id_str).map_err(|_| AppError::InvalidInput {
        context: format!("meetingrecording path has a non-UUID meeting id: {id_str:?}"),
    })?;

    // window = "<start_ms>-<end_ms>" (ignore any trailing extension).
    let window = window.split('.').next().unwrap_or(window);
    let (start_str, end_str) = window
        .split_once('-')
        .ok_or_else(|| AppError::InvalidInput {
            context: format!("meetingrecording window is not '<start>-<end>': {window:?}"),
        })?;
    let start_ms: u64 = start_str.parse().map_err(|_| AppError::InvalidInput {
        context: format!("meetingrecording start_ms is not an integer: {start_str:?}"),
    })?;
    let end_ms: u64 = end_str.parse().map_err(|_| AppError::InvalidInput {
        context: format!("meetingrecording end_ms is not an integer: {end_str:?}"),
    })?;
    if end_ms <= start_ms {
        return Err(AppError::InvalidInput {
            context: format!(
                "meetingrecording window end ({end_ms}) must exceed start ({start_ms})"
            ),
        });
    }

    Ok((MeetingId(uuid), start_ms, end_ms))
}

/// Minimal percent-decoder for a URI path: turn `%XX` byte escapes back into
/// their bytes (lossy UTF-8), pass everything else through. Sufficient for the
/// `meetingrecording:` path (a UUID + `<start>-<end>` window, where the only
/// escape `convertFileSrc` introduces is `%2F` for the separating slash); kept
/// dependency-free. An incomplete trailing escape is treated literally.
fn percent_decode_path(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Hex digit (`0-9A-Fa-f`) to its 0–15 value, or `None`.
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
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
            commands::create_meeting,
            commands::start_recording,
            commands::prewarm_asr,
            commands::pause_recording,
            commands::resume_recording,
            commands::set_recording_title,
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
            commands::add_attachment,
            commands::list_attachments,
            commands::open_attachment,
            commands::remove_attachment,
            commands::list_meetings,
            commands::open_meeting,
            commands::rename_meeting,
            commands::set_speaker_name,
            commands::delete_meeting,
            commands::open_meeting_folder,
            commands::list_collections,
            commands::create_collection,
            commands::rename_collection,
            commands::delete_collection,
            commands::set_meeting_collection,
            commands::reprocess,
            commands::summarise_meeting,
            commands::get_summary,
            commands::save_summary,
            commands::send_chat_message,
            commands::cancel_chat_turn,
            commands::get_chat_session,
            commands::list_chat_sessions,
            commands::delete_chat_session,
            commands::get_mcp_server_info,
            commands::translate_meeting,
            commands::get_translations,
            commands::reject_match,
            commands::clear_all_voiceprints,
            commands::list_voiceprints,
            commands::merge_voiceprint_identities,
            commands::rename_voiceprint_identity,
            commands::delete_voiceprint_identity,
            commands::forget_meeting_voiceprints,
            diagnostics::get_diagnostic_report,
            tunnel::tunnel_begin_pairing,
            tunnel::tunnel_poll_pairing,
            tunnel::set_connector_enabled,
            tunnel::tunnel_status,
            tunnel::delete_account,
            sync::sync_status,
            sync::sync_get_my_ticket,
            sync::sync_add_peer,
            sync::sync_now,
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
        notes_crdt::MeetingFolder::create(root, id).expect("folder");

        let bytes = b"\x89PNG\r\n\x1a\nfake".to_vec();
        let filename = persistence::save_note_asset(root, id, &bytes, "png").expect("save");

        // Path as `convertFileSrc(<id>/<filename>)` yields it (leading slash).
        let path = format!("/{}/{}", id.0, filename);
        let resolved = resolve_note_asset(root, &path).expect("resolve");
        assert_eq!(resolved.bytes, bytes);
        assert_eq!(resolved.content_type, "image/png");
    }

    #[test]
    fn resolve_note_asset_decodes_percent_encoded_separator() {
        use minutist_common::MeetingId;
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        notes_crdt::MeetingFolder::create(root, id).expect("folder");

        let bytes = b"\x89PNG\r\n\x1a\nfake".to_vec();
        let filename = persistence::save_note_asset(root, id, &bytes, "png").expect("save");

        // `convertFileSrc` percent-encodes the whole "<id>/<filename>" string,
        // so the separating '/' arrives as "%2F" (one path segment) and
        // `request.uri().path()` is not pre-decoded — this is the realistic
        // shape of a real `meetingasset:` request, unlike the literal-slash
        // path used above.
        let path = format!("/{}%2F{}", id.0, filename);
        let resolved = resolve_note_asset(root, &path).expect("resolve encoded path");
        assert_eq!(resolved.bytes, bytes);
        assert_eq!(resolved.content_type, "image/png");
    }

    // -----------------------------------------------------------------------
    // Recording-slice request parsing (`meetingrecording:` scheme). Pure — no
    // Orchestrator/Tauri runtime; drives `parse_recording_request` directly.
    // -----------------------------------------------------------------------

    #[test]
    fn parse_recording_request_accepts_valid_window_and_ignores_extension() {
        let uuid = "00000000-0000-0000-0000-000000000000";
        let (id, start, end) =
            parse_recording_request(&format!("/{uuid}/1000-2000")).expect("valid window");
        assert_eq!(id.0, uuid::Uuid::nil());
        assert_eq!((start, end), (1000, 2000));

        // A trailing `.wav` (should the frontend ever append one) is ignored.
        let (_, start, end) =
            parse_recording_request(&format!("/{uuid}/1000-2000.wav")).expect("ext ignored");
        assert_eq!((start, end), (1000, 2000));

        // convertFileSrc percent-encodes the separating slash as %2F — the path
        // arrives as one segment and must be decoded before splitting.
        let (id, start, end) =
            parse_recording_request(&format!("/{uuid}%2F1000-2000")).expect("encoded slash");
        assert_eq!(id.0, uuid::Uuid::nil());
        assert_eq!((start, end), (1000, 2000));
    }

    #[test]
    fn parse_recording_request_rejects_malformed_paths() {
        use minutist_common::AppError;
        let uuid = "00000000-0000-0000-0000-000000000000";
        let bad = [
            String::new(),                  // empty
            "/".to_string(),                // just the slash
            format!("/{uuid}"),             // no window
            format!("/{uuid}/"),            // empty window
            format!("/{uuid}/-5"),          // empty start
            format!("/{uuid}/5-"),          // empty end
            format!("/{uuid}/100-50"),      // end < start
            format!("/{uuid}/5-5"),         // end == start
            format!("/{uuid}/5-10-20"),     // extra dash
            format!("/{uuid}/a-b"),         // non-integer bounds
            format!("/{uuid}/1000-2000/x"), // nested segment
            "/not-a-uuid/1-2".to_string(),  // non-UUID id
        ];
        for path in bad {
            assert!(
                matches!(
                    parse_recording_request(&path),
                    Err(AppError::InvalidInput { .. })
                ),
                "expected InvalidInput for {path:?}",
            );
        }
    }

    #[test]
    fn resolve_note_asset_rejects_malformed_and_traversal_paths() {
        use minutist_common::{AppError, MeetingId};
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        notes_crdt::MeetingFolder::create(root, id).expect("folder");

        // Malformed (not <uuid>/<file>) → InvalidInput.
        for bad in ["", "/", "/only-one-segment", "/not-a-uuid/file.png"] {
            assert!(
                matches!(
                    resolve_note_asset(root, bad),
                    Err(AppError::InvalidInput { .. })
                ),
                "path {bad:?} should be rejected as InvalidInput"
            );
        }

        // Valid UUID but traversal filename → rejected by persistence guard
        // (InvalidInput), and a nested path → rejected here.
        let traversal = format!("/{}/../../etc/passwd", id.0);
        assert!(resolve_note_asset(root, &traversal).is_err());
        let nested = format!("/{}/sub/dir.png", id.0);
        assert!(
            matches!(
                resolve_note_asset(root, &nested),
                Err(AppError::InvalidInput { .. })
            ),
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

    // -----------------------------------------------------------------------
    // Attachment asset resolver (`attachment:` scheme, #0038). No Tauri runtime
    // needed — drives `resolve_attachment_asset` against a tempdir meetings
    // root, mirroring the `resolve_note_asset` tests above.
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_attachment_asset_serves_saved_original_with_content_type() {
        use minutist_common::MeetingId;
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();

        let bytes = b"%PDF-1.4 fake".to_vec();
        let hash = persistence::save_attachment_original(root, id, &bytes, "pdf").expect("save");
        let filename = format!("{hash}.pdf");

        let path = format!("/{}/{}", id.0, filename);
        let resolved = resolve_attachment_asset(root, &path).expect("resolve");
        assert_eq!(resolved.bytes, bytes);
        assert_eq!(resolved.content_type, "application/pdf");
    }

    #[test]
    fn resolve_attachment_asset_decodes_percent_encoded_separator() {
        use minutist_common::MeetingId;
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();

        let bytes = b"\x89PNG\r\n\x1a\nfake".to_vec();
        let hash = persistence::save_attachment_original(root, id, &bytes, "png").expect("save");
        let filename = format!("{hash}.png");

        // `convertFileSrc` percent-encodes the whole "<id>/<filename>" string,
        // so the separating '/' arrives as "%2F" (one path segment) and
        // `request.uri().path()` is not pre-decoded — this is the realistic
        // shape of a real `attachment:` request (the D1 bug this mirrors: a
        // decode-after-split would silently mis-parse this).
        let path = format!("/{}%2F{}", id.0, filename);
        let resolved = resolve_attachment_asset(root, &path).expect("resolve encoded path");
        assert_eq!(resolved.bytes, bytes);
        assert_eq!(resolved.content_type, "image/png");
    }

    #[test]
    fn resolve_attachment_asset_rejects_malformed_and_traversal_paths() {
        use minutist_common::{AppError, MeetingId};
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();

        // Malformed (not <uuid>/<file>) → InvalidInput.
        for bad in ["", "/", "/only-one-segment", "/not-a-uuid/file.pdf"] {
            assert!(
                matches!(
                    resolve_attachment_asset(root, bad),
                    Err(AppError::InvalidInput { .. })
                ),
                "path {bad:?} should be rejected as InvalidInput"
            );
        }

        // Valid UUID but traversal filename → rejected by persistence guard
        // (InvalidInput), and a nested path → rejected here.
        let traversal = format!("/{}/../../etc/passwd", id.0);
        assert!(resolve_attachment_asset(root, &traversal).is_err());
        let nested = format!("/{}/sub/dir.pdf", id.0);
        assert!(
            matches!(
                resolve_attachment_asset(root, &nested),
                Err(AppError::InvalidInput { .. })
            ),
            "nested path should be rejected"
        );
    }

    /// Verify that `bindings_builder()`'s TypeScript export is byte-identical
    /// to the checked-in `ui/src/ipc/bindings.ts`.
    ///
    /// A hand-maintained command-name ledger drifts silently from the real
    /// command surface (it only ever checked the in-memory export against
    /// itself, never against the committed frontend file the app actually
    /// loads). Comparing directly against the committed file makes drift a
    /// build failure: after adding, removing, or renaming a `#[tauri::command]`
    /// or changing a type reachable from one, regenerate the bindings with
    /// `make bindings` (`cargo run -p minutist --bin generate-bindings`,
    /// which uses this same builder and export config) and commit the result.
    ///
    /// `BigIntExportBehavior::Number` is used to allow `u64` fields (e.g.,
    /// timestamps and byte counts) to export as TypeScript `number` rather
    /// than erroring — matching `generate-bindings`'s export config exactly.
    #[test]
    fn bindings_builder_output_matches_committed_bindings_ts() {
        use specta_typescript::{BigIntExportBehavior, Typescript};

        let builder = bindings_builder();
        let generated = builder
            .export_str(Typescript::default().bigint(BigIntExportBehavior::Number))
            .expect("export_str should succeed for a correctly-configured builder");

        let committed_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("ui")
            .join("src")
            .join("ipc")
            .join("bindings.ts");
        let committed = std::fs::read_to_string(&committed_path).unwrap_or_else(|e| {
            panic!(
                "failed to read committed bindings file at {}: {e}",
                committed_path.display()
            )
        });

        assert_eq!(
            generated, committed,
            "ui/src/ipc/bindings.ts is stale relative to the Rust IPC command \
             surface. Regenerate it with `make bindings` \
             (cargo run -p minutist --bin generate-bindings) and commit the result."
        );
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
