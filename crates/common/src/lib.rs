//! Shared interface types and trait definitions for meeting-app.
//!
//! This crate is the architectural contract. Every other crate depends on
//! it; nothing here may depend on another crate in this workspace.
//!
//! Changes here ripple to every downstream crate. Adding, removing, or
//! changing a public item is an **architecture-owner** decision and
//! requires an update to `architecture/components.md` in the same commit.
//!
//! The trait method signatures here are **load-bearing**: parallel
//! sub-agents implement these traits independently against these
//! signatures. Do not change a signature without coordinating the
//! downstream crates.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The process-wide shared `LlamaBackend` (feature-gated; enabled by the two
/// llama.cpp-using crates, `asr-runtime` + `summariser`, so they share one
/// global backend init). See the module docs.
#[cfg(feature = "llama-backend")]
pub mod llama_backend;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Stable identifier for a meeting on disk. UUIDv4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct MeetingId(
    // Use `#[specta(type = String)]` so the TS binding mirrors how serde
    // emits a Uuid (a hyphenated lowercase string) without needing the
    // optional `uuid` feature on the `specta` crate.
    #[cfg_attr(feature = "specta", specta(type = String))] pub Uuid,
);

impl MeetingId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MeetingId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identifier for a chat session on disk. UUIDv4. Mirrors [`MeetingId`].
///
/// A chat session is meeting-scoped; `persistence` stores its turns under
/// `{meetings_dir}/{meeting_id}/chat/{session_id}.json` (Phase 9 §7). The
/// streaming chat `AppEvent`s carry this so the webview store routes deltas to
/// the right session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct ChatSessionId(
    // Use `#[specta(type = String)]` so the TS binding mirrors how serde
    // emits a Uuid (a hyphenated lowercase string) without needing the
    // optional `uuid` feature on the `specta` crate.
    #[cfg_attr(feature = "specta", specta(type = String))] pub Uuid,
);

impl ChatSessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ChatSessionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identifier for a model in the registry.
///
/// Examples: `"qwen3-asr-1.7b-q8_0"`, `"qwen2.5-3b-instruct-q4_k_m"`,
/// `"silero-vad-v4"`, `"sherpa-pyannote-segmentation-3-0"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct ModelId(pub String);

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Audio + transcript primitives
// ---------------------------------------------------------------------------

/// A contiguous block of audio samples bounded by VAD silence detections.
///
/// Sample rate is implicit (the workspace standardises on 16 kHz mono); if
/// that changes, this struct needs to carry the rate explicitly and
/// downstream crates need to be updated.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// One transcript segment with optional speaker assignment.
///
/// Speaker is populated by the `Diarizer` impl post-hoc; ASR backends
/// leave it `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordTimestamp>,
}

/// Optional per-word timestamp data when the ASR model supports it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct WordTimestamp {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// One audio-input device exposed to the device-picker UI.
///
/// `id` is a stable, opaque string the IPC layer round-trips back to
/// `audio-capture` to select the device. Format is implementation-defined
/// (cpal device-name plus host index on the Rust side). `name` is the
/// display label; `is_default` reflects the OS's default-input choice at
/// query time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

/// One audio-meter sample emitted at ~30 Hz while recording.
///
/// `peak` is the maximum absolute sample magnitude in [0.0, 1.0] over the
/// most-recent meter window (~33 ms of audio). `rms` is the root-mean-square
/// over the same window. Consumers may render either; both are cheap to
/// compute alongside capture.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct AudioMeterFrame {
    pub peak: f32,
    pub rms: f32,
}

/// Audio-file format descriptor captured at write time. Phase 1 writes
/// Opus 16 kHz mono; downstream phases re-decode using these fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct AudioFormat {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitrate_kbps: Option<u32>,
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/// Coarse model classification — drives the per-kind cache subdirectory
/// under `{app-data}/models/{kind}/` (see `architecture/cross-cutting.md`
/// "Filesystem layout").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Asr,
    Llm,
    Diarize,
}

/// Catalogue entry describing one model the app knows about.
///
/// `model-registry` reads this from the bundled `resources/models.json`
/// at startup and surfaces it (plus runtime state) as `ModelStatus`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelManifestEntry {
    pub id: ModelId,
    pub kind: ModelKind,
    pub display_name: String,
    /// Sibling files that belong to this model, relative to the cache
    /// dir for this entry. For Qwen3-ASR this lists both the GGUF and
    /// the mmproj.
    pub files: Vec<ModelFileEntry>,
    /// Approximate download size in bytes (sum of `files[*].size`).
    pub total_size_bytes: u64,
    /// SPDX licence identifier of the underlying weights ("apache-2.0",
    /// "openrail", etc.). Surfaced in About dialog (Phase 7) and used to
    /// gate bundling decisions.
    pub license: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelFileEntry {
    pub filename: String,
    pub url: String,
    pub size: u64,
    /// Lowercase-hex SHA-256.
    pub sha256: String,
}

/// Runtime state of one model on this user's machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ModelStatusState {
    /// Files are present and hashes match. `local_dir` is the cache
    /// directory absolute path.
    Available { local_dir: String },
    /// Files are missing or partial. `bytes_present` and `bytes_total`
    /// are summed across the manifest's `files`.
    Missing {
        bytes_present: u64,
        bytes_total: u64,
    },
    /// A download is in progress. The webview tracks granular progress
    /// via `AppEvent::ModelDownloadProgress` events; this state is the
    /// snapshot at query time.
    Downloading { bytes_done: u64, bytes_total: u64 },
    /// A previous download or hash check failed. `message` is a stable
    /// human-readable string suitable for surfacing in UI.
    Failed { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelStatus {
    pub id: ModelId,
    pub kind: ModelKind,
    pub display_name: String,
    pub status: ModelStatusState,
    /// SPDX licence identifier of the underlying weights, copied verbatim
    /// from the manifest entry's `license` ("apache-2.0", "mit", etc.).
    /// Surfaced in the About dialog so the bundled-model list never drifts
    /// from `resources/models.json`.
    pub license: String,
}

// ---------------------------------------------------------------------------
// Meeting metadata
// ---------------------------------------------------------------------------

/// Per-meeting metadata persisted as `metadata.json`.
///
/// Timestamps are ISO 8601 strings to avoid pulling `chrono` into `common`.
/// Consumers parse as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct MeetingMeta {
    pub uuid: MeetingId,
    pub title: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub duration_ms: u64,
    pub speaker_count: u32,
    pub audio_format: AudioFormat,
    pub asr_model: Option<ModelDescriptor>,
    pub llm_model: Option<ModelDescriptor>,
    pub diarizer: Option<ModelDescriptor>,
    /// User-set display names for identified speakers, keyed by the diarizer's
    /// label (e.g. `"A"` → `"Alice"`). Written by the `set_speaker_name` chat
    /// tool and overlaid at read time; cleared by re-diarization (which can
    /// re-letter speakers, see `cross-cutting.md` "Agent chat loop"). Phase 9.
    ///
    /// `#[serde(default, skip_serializing_if = …)]` so existing `metadata.json`
    /// (written before the field existed) still deserialises and the wire shape
    /// only grows when the map is non-empty.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub speaker_names: std::collections::BTreeMap<String, String>,
    pub app_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ModelDescriptor {
    pub name: String,
    pub quantisation: Option<String>,
    pub version: String,
}

// ---------------------------------------------------------------------------
// Meeting-list + restore types (Phase 4)
// ---------------------------------------------------------------------------

/// A summary row for the meeting-list view (FR-33). Cheap to query from the
/// `persistence` index without loading a meeting's full transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct MeetingListEntry {
    pub id: MeetingId,
    pub title: String,
    /// RFC3339 start timestamp (wall-clock), mirroring `MeetingMeta::started_at`.
    pub started_at: String,
    pub duration_ms: u64,
    pub speaker_count: u32,
    /// Short transcript excerpt for the list preview, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,
}

/// The notes document as it crosses the IPC boundary.
///
/// `notes_json` is the Tiptap document serialised to a JSON **string** —
/// `serde_json::Value` does not derive `specta::Type`, so the opaque document
/// rides the wire as a string and the webview owns its (de)serialisation.
/// `persistence` stores it verbatim (the transcript-chip opacity guarantee).
/// This is the canonical wire-facing notes carrier; `ipc-bridge` re-uses it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct NotesDocument {
    pub notes_json: String,
    pub notes_markdown: String,
}

/// The full restorable state of a meeting, assembled by `persistence` for
/// `open_meeting`: metadata, transcript segments, and the notes document
/// (absent when the meeting has no saved notes yet).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct MeetingState {
    pub meta: MeetingMeta,
    pub transcript: Vec<Segment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<NotesDocument>,
}

// ---------------------------------------------------------------------------
// Chat session wire types (Phase 9 — chat persistence + the chat UI)
// ---------------------------------------------------------------------------

/// The role of one persisted chat message as it crosses the IPC boundary and is
/// stored on disk.
///
/// This is the **wire / persisted** role, distinct from `chat-agent`'s
/// engine-internal `Role` (which serialises the same snake_case names for the
/// oaicompat template but is a different type owned by that crate). The driver
/// (`ipc-bridge`) maps between the engine's history and these persisted/wire
/// shapes at its boundary. `serde` snake_case so the TS binding mirrors the
/// engine's role names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum ChatRole {
    /// The session's system prompt (turn 0): persona + meeting context.
    System,
    /// A message the user typed.
    User,
    /// An assistant reply (the model's final text for a turn).
    Assistant,
    /// A tool-result message appended after the driver ran a tool call.
    Tool,
}

/// One persisted chat message (the wire / on-disk shape).
///
/// Distinct from `chat-agent`'s engine-internal message: this is the durable,
/// specta-typed record the webview renders and `persistence::ChatStore`
/// serialises. `turn_id` is the per-session monotonic turn counter the streaming
/// chat `AppEvent`s also carry, so the UI can correlate a stored message with
/// the deltas it saw live. `tool_name` is present only on `Tool` messages (the
/// name of the tool whose result the message carries).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// For a `Tool` message: the tool whose result this message carries.
    /// `None` for system/user/assistant messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// The per-session monotonic turn this message belongs to. The user message
    /// and the assistant/tool messages produced answering it share one `turn_id`
    /// (the same value the `ChatToken`/`ChatTurnComplete` events carry).
    pub turn_id: u64,
}

/// A persisted chat session for one meeting.
///
/// `persistence::ChatStore` stores this under
/// `{meetings_dir}/{meeting_id}/chat/{session_id}.json` (atomic tmp+rename); the
/// chat IPC commands load/save it. `meeting_id` is optional so a session may be
/// un-scoped (an MCP-originated session that targets no specific meeting);
/// `title` is optional so an untitled session round-trips. Timestamps are RFC
/// 3339 strings to avoid pulling a time crate into `common`, mirroring
/// `MeetingMeta::started_at`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ChatSession {
    pub id: ChatSessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meeting_id: Option<MeetingId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub created_at: String,
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Inter-agent bridge (Phase 9 precursor; consumed by Phase 10 MCP)
// ---------------------------------------------------------------------------

/// A request from an external agent (the Phase 10 MCP `send_to_internal_agent`
/// tool) to the internal chat agent. Landed now so Phase 10 adds zero `common`
/// change. `session_id`/`meeting_id` scope the request; both optional so a
/// caller may start a fresh session or target an existing one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct InterAgentRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<ChatSessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meeting_id: Option<MeetingId>,
    pub message: String,
}

/// The internal chat agent's reply to an [`InterAgentRequest`]. Carries the
/// session id so the external caller can continue the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct InterAgentReply {
    pub session_id: ChatSessionId,
    pub reply: String,
}

// ---------------------------------------------------------------------------
// Recording state
// ---------------------------------------------------------------------------

/// Top-level state of the recording pipeline. Emitted to the webview on
/// transitions via `AppEvent::StateChanged`.
///
/// **Timestamp semantics:** `started_at_ms` and `paused_at_ms` are
/// **wall-clock milliseconds since the Unix epoch** (UTC), not
/// recording-clock offsets. The webview can compute live elapsed-recording
/// duration as `Date.now() - started_at_ms` (subtracting accumulated
/// pause-time client-side if needed). Phase-internal timestamps that are
/// genuinely recording-clock (e.g. `Segment::start_ms`, `AudioChunk::start_ms`)
/// remain recording-clock — those are a different namespace and carry the
/// `_ms` suffix without the `_at` infix.
///
/// **Do NOT use `Date.now() - started_at_ms` as a paragraph-anchor source.**
/// That wall-clock delta is pause-*including* and drifts from the audio
/// timeline. Notes paragraph anchors must be stamped from
/// `AppEvent::RecordingClock { clock_ms }`, which is the capture-sample,
/// pause-*excluding* clock (same origin as `Segment::start_ms`). The
/// `started_at_ms` recipe above is for elapsed-time *display* only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum RecordingState {
    Idle,
    Recording {
        meeting_id: MeetingId,
        /// Wall-clock ms since Unix epoch when this Recording started.
        started_at_ms: u64,
    },
    Paused {
        meeting_id: MeetingId,
        /// Wall-clock ms since Unix epoch when this Pause began.
        paused_at_ms: u64,
    },
    Stopping {
        meeting_id: MeetingId,
    },
    /// The recorder is busy but not capturing: either a just-stopped meeting is
    /// finalising in the background (the live ASR backlog drains and
    /// `transcript.json` / `metadata.json` / `audio.opus` are written), or an
    /// offline re-transcribe / re-diarize pass holds the recorder (the automatic
    /// post-stop repairs and the user-triggered actions both claim the slot).
    /// The UI stays responsive during this window — only starting a NEW recording
    /// waits. After a stop, `Idle` plus `AppEvent::MeetingFinalised` fire on
    /// completion; an offline pass returns to `Idle` when it finishes.
    Finalising {
        meeting_id: MeetingId,
    },
}

// ---------------------------------------------------------------------------
// IPC events
// ---------------------------------------------------------------------------

/// Events emitted from the Rust core to the webview via tauri-specta.
///
/// Adding a variant requires updating `ipc-bridge` (encoder), the webview
/// IPC client (decoder), and re-running the bindings generation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum AppEvent {
    /// Audio meter sample emitted at ~30 Hz while recording. Carries both
    /// peak and RMS so the UI can pick the rendering it wants without an
    /// extra round-trip.
    AudioMeter { frame: AudioMeterFrame },
    /// The available audio-input device list changed (hotplug or default
    /// device switch). The webview should re-query `list_devices`.
    DevicesChanged,
    /// Recording state changed.
    StateChanged { state: RecordingState },
    /// A new transcript segment was produced.
    TranscriptSegment {
        meeting_id: MeetingId,
        segment: Segment,
    },
    /// The live recording clock advanced. Emitted at a throttled rate
    /// (~5 Hz) while recording. `clock_ms` is the capture-sample,
    /// pause-*excluding* offset from the start of the recording — the same
    /// timeline as `Segment::start_ms` and `AudioChunk::start_ms`. The notes
    /// editor stamps paragraph anchors (`data-anchor-ms`) from this value so
    /// anchors line up with transcript segments; do NOT derive anchors from
    /// `Date.now() - started_at_ms` (that is pause-including wall-clock).
    RecordingClock {
        meeting_id: MeetingId,
        clock_ms: u64,
    },
    /// Diarization finished assigning speakers to a meeting's segments.
    DiarizationComplete {
        meeting_id: MeetingId,
        speaker_count: u32,
    },
    /// Summary generation finished; `summary.md` now exists for this meeting.
    SummaryReady { meeting_id: MeetingId },
    /// A stopped meeting finished finalising on disk (`transcript.json` +
    /// `metadata.json` written, `audio.opus` closed). The webview refreshes the
    /// meeting list so the just-recorded meeting appears. Distinct from
    /// `StateChanged { Idle }`: the list refresh keys on the meeting being
    /// *ready on disk*, which is exactly when this fires.
    MeetingFinalised { meeting_id: MeetingId },
    /// An offline re-transcribe finished rewriting `transcript.json`. The webview
    /// re-reads the meeting's transcript (list excerpt + any open-meeting view),
    /// mirroring `DiarizationComplete`. Emitted by both the user-triggered
    /// re-transcribe and the background post-stop repair, so a repaired
    /// transcript surfaces without a manual refresh even when diarization is off.
    TranscriptReady { meeting_id: MeetingId },
    /// Model download progress, used by the first-run flow.
    ModelDownloadProgress {
        model_id: ModelId,
        bytes_done: u64,
        bytes_total: Option<u64>,
    },
    /// A newer release is available (the updater's check found one). The
    /// webview shows an update-available prompt. Emitted by app-main's
    /// `tauri-plugin-updater` integration; see `architecture/cross-cutting.md`
    /// "Auto-update".
    UpdateAvailable {
        version: String,
        notes: Option<String>,
    },
    /// Update-download progress while applying an accepted update. `total_bytes`
    /// is `None` when the server sends no content length.
    UpdateProgress {
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    /// User-visible settings changed; subscribers should re-read.
    SettingsChanged,
    /// A recoverable error occurred during a background task. The pipeline
    /// continues; the webview shows a notification.
    ErrorOccurred { error: AppError },

    // --- Chat agent (Phase 9) --------------------------------------------
    // These ride the existing `AppEventPayload` newtype + the single
    // `collect_events![AppEventPayload]` registration in `ipc-bridge` — no new
    // event registration. `turn_id` is a per-session monotonic turn counter.
    /// One streamed token (or token fragment) of the assistant's reply for the
    /// in-flight turn. Lossy: a dropped delta is reconciled by the `final_text`
    /// carried on `ChatTurnComplete` (see `cross-cutting.md`).
    ChatToken {
        session_id: ChatSessionId,
        turn_id: u64,
        token: String,
    },
    /// The assistant requested a tool call mid-turn. `args_json` is the tool's
    /// arguments serialised as a JSON string (the repo's "Value crosses as
    /// String" rule).
    ChatToolCall {
        session_id: ChatSessionId,
        turn_id: u64,
        tool: String,
        args_json: String,
    },
    /// A tool call finished. `ok` is `false` when the tool errored; `summary` is
    /// the one-line human/LLM-facing render shown on the UI tool card.
    ChatToolResult {
        session_id: ChatSessionId,
        turn_id: u64,
        tool: String,
        ok: bool,
        summary: String,
    },
    /// The assistant turn finished. `final_text` carries the FULL reconciled
    /// reply so the store can overwrite regardless of any dropped `ChatToken`
    /// deltas (lossy-broadcast mitigation).
    ChatTurnComplete {
        session_id: ChatSessionId,
        turn_id: u64,
        final_text: String,
    },
    /// The chat turn failed. `message` is a stable human-readable string the
    /// webview surfaces in the chat pane.
    ChatError {
        session_id: ChatSessionId,
        message: String,
    },

    // --- MCP server (Phase 10) -------------------------------------------
    /// The in-process MCP server bound its loopback Streamable HTTP listener.
    /// `app-main` emits this after `mcp_server::serve` returns the bound addr so
    /// the Settings → MCP pane can show the live endpoint URL. The bearer token
    /// is deliberately NOT carried on the event bus (it is revealed only via the
    /// `get_mcp_server_info` command on explicit user request); see
    /// `architecture/cross-cutting.md` — "MCP transport".
    McpServerListening { url: String },
}

// ---------------------------------------------------------------------------
// Error type at the architectural boundary
// ---------------------------------------------------------------------------

/// The shared error type that crosses crate boundaries.
///
/// Per-crate `Error` enums (defined with `thiserror` in their owning
/// crate) provide structured `From` impls into `AppError`. The webview
/// only ever sees `AppError`. Variants have stable discriminants — the
/// TypeScript binding is generated from this enum, so renaming or
/// removing a variant is a breaking IPC change.
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum AppError {
    #[error("I/O error: {context}")]
    Io { context: String },
    #[error("model {model_id} failed to load: {context}")]
    ModelLoad { model_id: String, context: String },
    #[error("model {model_id} not found in registry")]
    ModelNotFound { model_id: String },
    #[error("model download failed: {context}")]
    ModelDownload { context: String },
    #[error("inference failed in {backend}: {context}")]
    Inference { backend: String, context: String },
    #[error("invalid input: {context}")]
    InvalidInput { context: String },
    #[error("operation cancelled")]
    Cancelled,
    #[error("operation not supported: {context}")]
    Unsupported { context: String },
    #[error("internal error: {context}")]
    Internal { context: String },
}

/// Convenience alias for `Result<T, AppError>`. Use in trait method
/// signatures and at crate boundaries; per-crate code may use its own
/// `Result<T, CrateError>` internally.
pub type AppResult<T> = Result<T, AppError>;

// ---------------------------------------------------------------------------
// Architectural traits
// ---------------------------------------------------------------------------

/// Synchronous ASR backend. Implementations live in `asr-runtime`
/// (production) and may be mocked in tests.
///
/// Threading: the trait is sync because real implementations are FFI-bound
/// (llama.cpp) and don't expose async. Callers wrap calls in
/// `tokio::task::spawn_blocking`. See `architecture/cross-cutting.md` —
/// Threading model.
///
/// Lifecycle: implementations own their loaded model. `Drop` releases it.
/// The trait does not include load / unload; the consuming crate constructs
/// the backend with a `ModelId` and the path resolved by `model-registry`,
/// and drops it on settings change.
pub trait AsrBackend: Send {
    /// Transcribe one VAD-bounded audio chunk into zero or more segments.
    ///
    /// `chunk.start_ms` is the recording-clock offset of the first sample.
    /// Returned segments carry timestamps relative to the start of the
    /// recording, not the start of the chunk.
    ///
    /// `speaker_id` is left `None`; diarization is a separate pass.
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>>;
}

/// Which ASR backend transcribes a recording (Phase 8 — hybrid ASR).
///
/// Two backends implement [`AsrBackend`]: `asr-parakeet` (sherpa-onnx Parakeet
/// TDT v3 — English + 24 EU languages, per-word timestamps) and `asr-runtime`
/// (llama-cpp-2 Qwen3-ASR — 52 languages/dialects, no timestamps; a 0.6B CPU
/// default and an optional 1.7B GPU tier). The orchestrator builds the chosen
/// one behind `Box<dyn AsrBackend>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AsrEngine {
    /// Parakeet TDT 0.6B v3 via sherpa-onnx. Primary for the languages it covers.
    ParakeetEuV3,
    /// Qwen3-ASR 0.6B via llama-cpp-2 mtmd. Broad-language CPU default / fallback.
    Qwen06B,
    /// Qwen3-ASR 1.7B via llama-cpp-2 mtmd. Opt-in GPU tier (broader + better
    /// multilingual accuracy).
    Qwen17B,
}

/// The languages NVIDIA Parakeet TDT 0.6B v3 covers (English + 24 European
/// locales), as full English names matched case-insensitively against the
/// `transcription_language` setting. Anything outside this set — Chinese,
/// Japanese, Korean, Arabic, etc. — routes to Qwen instead. Keep this in step
/// with the model card; see `architecture/cross-cutting.md` — "ASR engine
/// routing".
pub const PARAKEET_LANGUAGES: &[&str] = &[
    "Bulgarian",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hungarian",
    "Italian",
    "Latvian",
    "Lithuanian",
    "Maltese",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swedish",
    "Ukrainian",
];

/// Choose the ASR engine deterministically from the user's transcription-language
/// setting (never by inspecting the audio — the language isn't known before
/// transcription). Pure so the orchestrator and any future UI surface agree.
///
/// - language in [`PARAKEET_LANGUAGES`] → [`AsrEngine::ParakeetEuV3`] (better
///   English/EU accuracy + timestamps);
/// - the `""` / `"auto"` sentinel (auto-detect) → Qwen (broadest coverage is the
///   safe default when the language is unknown);
/// - any other named language (Chinese, Japanese, …) → Qwen.
///
/// Within the Qwen branch, `prefer_gpu_qwen` selects the 1.7B GPU tier over the
/// 0.6B CPU default.
pub fn asr_engine_for_language(transcription_language: &str, prefer_gpu_qwen: bool) -> AsrEngine {
    let lang = transcription_language.trim();
    let is_auto = lang.is_empty() || lang.eq_ignore_ascii_case("auto");
    if !is_auto
        && PARAKEET_LANGUAGES
            .iter()
            .any(|l| l.eq_ignore_ascii_case(lang))
    {
        AsrEngine::ParakeetEuV3
    } else if prefer_gpu_qwen {
        AsrEngine::Qwen17B
    } else {
        AsrEngine::Qwen06B
    }
}

/// Synchronous diarizer. Implementations live in `diarizer` (production).
///
/// Post-hoc only in v1: runs after the recording stops or as a
/// user-triggered re-diarize. Not on the live path.
///
/// Threading: sync, called from `spawn_blocking`.
pub trait Diarizer: Send {
    /// Assign `speaker_id` to each segment in place by clustering speaker
    /// embeddings extracted from `audio` over each segment's `[start_ms,
    /// end_ms]` window.
    ///
    /// `audio` is the entire buffered recording at `sample_rate` Hz. The
    /// `segments` slice is the ASR output for the same recording.
    ///
    /// Returns the number of distinct speakers found.
    fn assign_speakers(
        &self,
        audio: &[f32],
        sample_rate: u32,
        segments: &mut [Segment],
    ) -> AppResult<u32>;
}

/// Synchronous summariser. Implementations live in `summariser`
/// (production). Multiple impls may coexist (bundled llama.cpp,
/// external Ollama), selected by settings.
///
/// Threading: sync, called from `spawn_blocking`.
///
/// `Send + Sync` (Phase 9): a held `Arc<dyn Summariser>` is shared by the
/// one-shot summary path and the chat agent's `resummarise` tool, so it must
/// cross threads *and* be referenced concurrently. All impls satisfy this:
/// `LlamaSummariser` holds a `LlamaModel` (`unsafe impl Send + Sync`) plus a
/// `PathBuf` + config and builds its `!Sync` `LlamaContext` fresh per call
/// (never stored — SP0); `OllamaSummariser` holds a `reqwest::blocking::Client`
/// (Sync); the test stub holds `Mutex`-guarded fields (Sync).
pub trait Summariser: Send + Sync {
    /// Produce a markdown summary from a transcript + the user's notes.
    ///
    /// `notes_markdown` is the markdown export of the Tiptap notes (or
    /// empty string if no notes were taken). `system_prompt` is the
    /// user-configured prompt from settings.
    fn summarise(
        &self,
        transcript: &[Segment],
        notes_markdown: &str,
        system_prompt: &str,
    ) -> AppResult<String>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_routing_picker_languages_map_to_expected_engine() {
        // The languages the LanguagePicker offers today, and where each routes.
        let cpu = false;
        for lang in [
            "English",
            "Spanish",
            "French",
            "German",
            "Italian",
            "Portuguese",
            "Russian",
            "Dutch",
        ] {
            assert_eq!(
                asr_engine_for_language(lang, cpu),
                AsrEngine::ParakeetEuV3,
                "{lang} should use Parakeet"
            );
        }
        for lang in ["Chinese", "Japanese", "Korean", "Arabic"] {
            assert_eq!(
                asr_engine_for_language(lang, cpu),
                AsrEngine::Qwen06B,
                "{lang} should use Qwen"
            );
        }
    }

    #[test]
    fn asr_routing_auto_detect_and_empty_route_to_qwen() {
        for sentinel in ["", "auto", "Auto", "AUTO", "  "] {
            assert_eq!(asr_engine_for_language(sentinel, false), AsrEngine::Qwen06B);
        }
    }

    #[test]
    fn asr_routing_is_case_and_whitespace_insensitive() {
        assert_eq!(
            asr_engine_for_language("  english ", false),
            AsrEngine::ParakeetEuV3
        );
        assert_eq!(
            asr_engine_for_language("FRENCH", false),
            AsrEngine::ParakeetEuV3
        );
    }

    #[test]
    fn asr_routing_gpu_flag_only_affects_the_qwen_branch() {
        // Parakeet languages ignore the GPU-Qwen preference.
        assert_eq!(
            asr_engine_for_language("English", true),
            AsrEngine::ParakeetEuV3
        );
        // Qwen languages honour it: 1.7B when opted in, else 0.6B.
        assert_eq!(asr_engine_for_language("Chinese", true), AsrEngine::Qwen17B);
        assert_eq!(
            asr_engine_for_language("Chinese", false),
            AsrEngine::Qwen06B
        );
        // Auto-detect + GPU opt-in -> the bigger Qwen.
        assert_eq!(asr_engine_for_language("auto", true), AsrEngine::Qwen17B);
    }

    #[test]
    fn segment_round_trips_through_json() {
        let s = Segment {
            start_ms: 100,
            end_ms: 500,
            text: "hello world".to_string(),
            speaker_id: Some("A".to_string()),
            confidence: Some(0.92),
            words: vec![],
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Segment = serde_json::from_str(&json).unwrap();
        assert_eq!(s.start_ms, back.start_ms);
        assert_eq!(s.text, back.text);
        assert_eq!(s.speaker_id, back.speaker_id);
    }

    #[test]
    fn meeting_id_is_distinct_per_construction() {
        let a = MeetingId::new();
        let b = MeetingId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn app_error_display_includes_context() {
        let e = AppError::Inference {
            backend: "mtmd".into(),
            context: "decode failed".into(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("mtmd"));
        assert!(msg.contains("decode failed"));
    }

    #[test]
    fn recording_state_serialises_with_tag() {
        let s = RecordingState::Recording {
            meeting_id: MeetingId::new(),
            started_at_ms: 1234,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"kind\":\"recording\""));
    }

    #[test]
    fn audio_device_round_trips() {
        let d = AudioDevice {
            id: "hw:1,0".to_string(),
            name: "Built-in Microphone".to_string(),
            is_default: true,
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: AudioDevice = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn audio_meter_frame_round_trips() {
        let f = AudioMeterFrame {
            peak: 0.75,
            rms: 0.42,
        };
        let json = serde_json::to_string(&f).unwrap();
        let back: AudioMeterFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(f.peak, back.peak);
        assert_eq!(f.rms, back.rms);
    }

    #[test]
    fn app_event_audio_meter_uses_frame() {
        let e = AppEvent::AudioMeter {
            frame: AudioMeterFrame {
                peak: 0.5,
                rms: 0.3,
            },
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"audio_meter\""));
        assert!(json.contains("\"frame\""));
        assert!(json.contains("\"peak\":0.5"));
    }

    #[test]
    fn app_event_devices_changed_serialises_unit() {
        let e = AppEvent::DevicesChanged;
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"devices_changed\""));
    }

    #[test]
    fn app_event_recording_clock_round_trips() {
        let e = AppEvent::RecordingClock {
            meeting_id: MeetingId::new(),
            clock_ms: 42_000,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"recording_clock\""));
        assert!(json.contains("\"clock_ms\":42000"));
        match serde_json::from_str::<AppEvent>(&json).unwrap() {
            AppEvent::RecordingClock { clock_ms, .. } => assert_eq!(clock_ms, 42_000),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn model_kind_serialises_snake_case() {
        let asr = serde_json::to_string(&ModelKind::Asr).unwrap();
        let llm = serde_json::to_string(&ModelKind::Llm).unwrap();
        let diar = serde_json::to_string(&ModelKind::Diarize).unwrap();
        assert_eq!(asr, "\"asr\"");
        assert_eq!(llm, "\"llm\"");
        assert_eq!(diar, "\"diarize\"");
    }

    #[test]
    fn model_status_round_trips_through_json() {
        let s = ModelStatus {
            id: ModelId::from("qwen3-asr-0.6b-q8_0"),
            kind: ModelKind::Asr,
            display_name: "Qwen3-ASR 0.6B Q8_0".to_string(),
            status: ModelStatusState::Downloading {
                bytes_done: 1024 * 1024,
                bytes_total: 805 * 1024 * 1024,
            },
            license: "apache-2.0".to_string(),
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"state\":\"downloading\""));
        let back: ModelStatus = serde_json::from_str(&json).unwrap();
        match back.status {
            ModelStatusState::Downloading {
                bytes_done,
                bytes_total,
            } => {
                assert_eq!(bytes_done, 1024 * 1024);
                assert_eq!(bytes_total, 805 * 1024 * 1024);
            }
            other => panic!("unexpected state: {other:?}"),
        }
    }

    #[test]
    fn model_manifest_entry_round_trips() {
        let m = ModelManifestEntry {
            id: ModelId::from("qwen3-asr-0.6b-q8_0"),
            kind: ModelKind::Asr,
            display_name: "Qwen3-ASR 0.6B Q8_0".to_string(),
            files: vec![ModelFileEntry {
                filename: "Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf".to_string(),
                url: "https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B-Q8_0-ggml-org.gguf".to_string(),
                size: 805_000_000,
                sha256: "bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971".to_string(),
            }],
            total_size_bytes: 805_000_000,
            license: "apache-2.0".to_string(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: ModelManifestEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.files.len(), 1);
        assert_eq!(back.files[0].sha256.len(), 64);
        assert_eq!(back.kind, ModelKind::Asr);
    }

    #[test]
    fn meeting_meta_carries_audio_format() {
        let m = MeetingMeta {
            uuid: MeetingId::new(),
            title: "Sample".to_string(),
            started_at: "2026-05-27T10:00:00Z".to_string(),
            ended_at: None,
            duration_ms: 0,
            speaker_count: 0,
            audio_format: AudioFormat {
                codec: "opus".to_string(),
                sample_rate: 16_000,
                channels: 1,
                bitrate_kbps: Some(32),
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            speaker_names: std::collections::BTreeMap::new(),
            app_version: "0.0.0".to_string(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: MeetingMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.audio_format.codec, "opus");
        assert_eq!(back.audio_format.sample_rate, 16_000);
        assert_eq!(back.audio_format.channels, 1);
        assert_eq!(back.audio_format.bitrate_kbps, Some(32));
    }

    #[test]
    fn meeting_list_entry_round_trips_and_omits_absent_excerpt() {
        let e = MeetingListEntry {
            id: MeetingId::new(),
            title: "Launch sync".to_string(),
            started_at: "2026-06-02T09:58:00Z".to_string(),
            duration_ms: 1_800_000,
            speaker_count: 2,
            excerpt: None,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(!json.contains("excerpt"), "absent excerpt must be omitted");
        let back: MeetingListEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn meeting_state_round_trips_with_and_without_notes() {
        let meta = MeetingMeta {
            uuid: MeetingId::new(),
            title: "Sample".to_string(),
            started_at: "2026-06-02T10:00:00Z".to_string(),
            ended_at: Some("2026-06-02T10:30:00Z".to_string()),
            duration_ms: 1_800_000,
            speaker_count: 1,
            audio_format: AudioFormat {
                codec: "opus".to_string(),
                sample_rate: 16_000,
                channels: 1,
                bitrate_kbps: Some(32),
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            speaker_names: std::collections::BTreeMap::new(),
            app_version: "0.0.0".to_string(),
        };
        let segment = Segment {
            start_ms: 100,
            end_ms: 2_000,
            text: "hello world".to_string(),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
        };
        let with_notes = MeetingState {
            meta: meta.clone(),
            transcript: vec![segment.clone()],
            notes: Some(NotesDocument {
                notes_json: "{\"type\":\"doc\"}".to_string(),
                notes_markdown: "# Notes".to_string(),
            }),
        };
        let json = serde_json::to_string(&with_notes).unwrap();
        let back: MeetingState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.transcript.len(), 1);
        assert_eq!(back.notes.as_ref().unwrap().notes_markdown, "# Notes");

        let without_notes = MeetingState {
            meta,
            transcript: vec![segment],
            notes: None,
        };
        let json = serde_json::to_string(&without_notes).unwrap();
        assert!(!json.contains("notes"), "absent notes must be omitted");
        let back: MeetingState = serde_json::from_str(&json).unwrap();
        assert!(back.notes.is_none());
    }

    #[test]
    fn chat_session_id_is_distinct_per_construction() {
        let a = ChatSessionId::new();
        let b = ChatSessionId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn chat_session_id_serialises_as_bare_uuid_string() {
        let id = ChatSessionId::new();
        let json = serde_json::to_string(&id).unwrap();
        // `#[serde(transparent)]` → a bare hyphenated lowercase UUID string.
        assert_eq!(json, format!("\"{}\"", id.0));
        let back: ChatSessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn meeting_meta_speaker_names_default_omitted_and_round_trips() {
        // An older metadata.json without the field deserialises to an empty map.
        let old_json = r#"{
            "uuid": "00000000-0000-4000-8000-000000000000",
            "title": "Old meeting",
            "started_at": "2026-06-02T10:00:00Z",
            "ended_at": null,
            "duration_ms": 0,
            "speaker_count": 0,
            "audio_format": { "codec": "opus", "sample_rate": 16000, "channels": 1 },
            "asr_model": null,
            "llm_model": null,
            "diarizer": null,
            "app_version": "0.0.0"
        }"#;
        let restored: MeetingMeta =
            serde_json::from_str(old_json).expect("old metadata.json must still deserialise");
        assert!(
            restored.speaker_names.is_empty(),
            "missing speaker_names must deserialise to an empty map"
        );
        // An empty map is omitted from the wire shape.
        let json = serde_json::to_string(&restored).unwrap();
        assert!(
            !json.contains("speaker_names"),
            "an empty speaker_names map must be omitted"
        );

        // A populated map round-trips.
        let mut meta = restored;
        meta.speaker_names
            .insert("A".to_string(), "Alice".to_string());
        meta.speaker_names
            .insert("B".to_string(), "Bob".to_string());
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("speaker_names"));
        let back: MeetingMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.speaker_names.get("A").map(String::as_str),
            Some("Alice")
        );
        assert_eq!(back.speaker_names.get("B").map(String::as_str), Some("Bob"));
    }

    #[test]
    fn app_event_chat_token_serialises_with_tag() {
        let e = AppEvent::ChatToken {
            session_id: ChatSessionId::new(),
            turn_id: 3,
            token: "hello".to_string(),
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"chat_token\""));
        assert!(json.contains("\"turn_id\":3"));
        match serde_json::from_str::<AppEvent>(&json).unwrap() {
            AppEvent::ChatToken { turn_id, token, .. } => {
                assert_eq!(turn_id, 3);
                assert_eq!(token, "hello");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn app_event_chat_tool_call_and_result_serialise_with_tags() {
        let call = AppEvent::ChatToolCall {
            session_id: ChatSessionId::new(),
            turn_id: 1,
            tool: "get_transcript".to_string(),
            args_json: "{\"meeting_id\":\"x\"}".to_string(),
        };
        assert!(serde_json::to_string(&call)
            .unwrap()
            .contains("\"kind\":\"chat_tool_call\""));

        let result = AppEvent::ChatToolResult {
            session_id: ChatSessionId::new(),
            turn_id: 1,
            tool: "get_transcript".to_string(),
            ok: true,
            summary: "12 segments".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"kind\":\"chat_tool_result\""));
        assert!(json.contains("\"ok\":true"));
    }

    #[test]
    fn app_event_chat_turn_complete_and_error_serialise_with_tags() {
        let complete = AppEvent::ChatTurnComplete {
            session_id: ChatSessionId::new(),
            turn_id: 7,
            final_text: "the full reply".to_string(),
        };
        let json = serde_json::to_string(&complete).unwrap();
        assert!(json.contains("\"kind\":\"chat_turn_complete\""));
        assert!(json.contains("the full reply"));

        let err = AppEvent::ChatError {
            session_id: ChatSessionId::new(),
            message: "context full".to_string(),
        };
        assert!(serde_json::to_string(&err)
            .unwrap()
            .contains("\"kind\":\"chat_error\""));
    }

    #[test]
    fn chat_session_round_trips_and_omits_optionals() {
        let session = ChatSession {
            id: ChatSessionId::new(),
            meeting_id: Some(MeetingId::new()),
            title: Some("Action items".to_string()),
            messages: vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: "you are a meeting-notes assistant".to_string(),
                    tool_name: None,
                    turn_id: 0,
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "what were the action items?".to_string(),
                    tool_name: None,
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    content: "{\"segments\":[]}".to_string(),
                    tool_name: Some("get_transcript".to_string()),
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "the action items were …".to_string(),
                    tool_name: None,
                    turn_id: 1,
                },
            ],
            created_at: "2026-06-10T10:00:00Z".to_string(),
            updated_at: "2026-06-10T10:01:00Z".to_string(),
        };
        let json = serde_json::to_string(&session).unwrap();
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"role\":\"tool\""));
        // The tool message carries its tool_name; non-tool messages omit it.
        assert!(json.contains("\"tool_name\":\"get_transcript\""));
        let back: ChatSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back, session);
        assert_eq!(back.messages.len(), 4);
        assert!(back.messages[0].tool_name.is_none());
        assert_eq!(
            back.messages[2].tool_name.as_deref(),
            Some("get_transcript")
        );

        // Absent meeting_id / title are omitted from the wire shape.
        let untitled = ChatSession {
            id: ChatSessionId::new(),
            meeting_id: None,
            title: None,
            messages: vec![],
            created_at: "2026-06-10T10:00:00Z".to_string(),
            updated_at: "2026-06-10T10:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&untitled).unwrap();
        assert!(
            !json.contains("meeting_id"),
            "absent meeting_id must be omitted"
        );
        assert!(!json.contains("title"), "absent title must be omitted");
        let back: ChatSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back, untitled);
    }

    #[test]
    fn inter_agent_request_and_reply_round_trip() {
        let req = InterAgentRequest {
            session_id: None,
            meeting_id: Some(MeetingId::new()),
            message: "what were the action items?".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        // Absent session_id is omitted.
        assert!(!json.contains("session_id"));
        let back: InterAgentRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.message, req.message);
        assert!(back.session_id.is_none());
        assert_eq!(back.meeting_id, req.meeting_id);

        let reply = InterAgentReply {
            session_id: ChatSessionId::new(),
            reply: "the action items were …".to_string(),
        };
        let json = serde_json::to_string(&reply).unwrap();
        let back: InterAgentReply = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_id, reply.session_id);
        assert_eq!(back.reply, reply.reply);
    }
}
