//! Application settings — Phase 1 fields.
//!
//! This crate is the single source of truth for runtime configuration.
//! Other crates read settings via [`SettingsHandle`]; nobody parses the
//! backing JSON file directly.
//!
//! ## Architecture constraints
//!
//! - **No `tauri::*` imports.** Tauri glue lives only in `ipc-bridge` and
//!   `app-main`. This crate receives a `PathBuf` at construction time and
//!   reads/writes JSON via `serde_json` + `std::fs`.
//! - Settings changes broadcast directly from this crate via
//!   [`tokio::sync::watch`], not through the orchestrator.
//! - Per-crate [`Error`] via `thiserror`; `From<Error> for AppError` is
//!   implemented in [`error`].

use std::path::PathBuf;

use minutist_common::{GpuAcceleration, LiveAgentMode, ModelId};
use serde::{Deserialize, Serialize};

pub mod error;
pub mod handle;
pub mod store;

pub use error::Error;
pub use handle::SettingsHandle;
pub use store::{JsonFileStore, SettingsStore};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// UI colour-scheme preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum Theme {
    Light,
    Dark,
    /// Follow the OS preference (default).
    #[default]
    System,
}

/// Default notes-autosave interval, in seconds (FR-18/FR-35).
const fn default_autosave_interval_secs() -> u32 {
    5
}

/// Default GPU-acceleration toggle.
///
/// GPU is **on by default** (`true`): a GPU-feature build (Vulkan/Metal/etc.)
/// offloads inference to the device, and llama.cpp falls back to CPU at runtime
/// when no device is present, so defaulting to `true` is safe even on CPU-only
/// machines. An older store written before this field existed deserialises to
/// `true` via `#[serde(default = ...)]`, preserving the prior compile-time
/// behaviour (a GPU build offloaded; a CPU build did not — see below). The flag
/// is only effective in a GPU-feature build: a default CPU-only build always runs
/// on CPU regardless of this setting. See `architecture/cross-cutting.md` —
/// "GPU portability".
const fn default_gpu_acceleration() -> GpuAcceleration {
    GpuAcceleration::Auto
}

/// Deserialize `gpu_acceleration` accepting EITHER the new string enum
/// (`"auto"`/`"on"`/`"off"`) OR a legacy JSON bool, so an existing store
/// migrates without losing intent: legacy `true` → `Auto` (the historical
/// `true` was the unconsidered default, not a deliberate force-GPU, so it adopts
/// the new safe probe-and-fit behaviour), legacy `false` → `Off` (that WAS a
/// deliberate CPU choice, so it maps to the hard override). A missing field uses
/// `#[serde(default)]` (→ `Auto`).
fn deserialize_gpu_acceleration<'de, D>(deserializer: D) -> Result<GpuAcceleration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolOrEnum {
        // bool first: a JSON string falls through to the enum variant.
        Legacy(bool),
        Modern(GpuAcceleration),
    }
    Ok(match BoolOrEnum::deserialize(deserializer)? {
        BoolOrEnum::Legacy(true) => GpuAcceleration::Auto,
        BoolOrEnum::Legacy(false) => GpuAcceleration::Off,
        BoolOrEnum::Modern(mode) => mode,
    })
}

/// Default for `capture_system_audio`: ON. Capturing the call audio is the point
/// of a meeting-notes app, so it defaults on (opt-out); the UI advises turning
/// it off when the mic also hears the call from the speakers (echo).
const fn default_capture_system_audio() -> bool {
    true
}

/// Default MCP server port (Phase 10 — D1). A FIXED loopback port: only one app
/// instance/window runs, so a fixed default avoids the ephemeral-port friction
/// of a per-run URL change in an external MCP client's config. User-editable in
/// the MCP settings pane. An older store written before this field existed
/// deserialises to 8765 via `#[serde(default = ...)]`.
const fn default_mcp_port() -> u16 {
    8765
}

/// Default connector relay tunnel URL (WS4-A S5b). The hosted relay's WSS
/// rendezvous endpoint the device dials. User-overridable (for self-hosting /
/// testing); defaults to the minutist.ai endpoint. An older store deserialises
/// to this via `#[serde(default = ...)]`.
fn default_relay_url() -> String {
    "wss://mcp.minutist.ai/tunnel".to_string()
}

/// Default connector account-service API base URL (WS4-A S5b). The base the
/// device-code pairing client posts `/pair/start` + `/pair/poll` against.
/// User-overridable; defaults to the minutist.ai endpoint.
fn default_relay_api_url() -> String {
    "https://api.minutist.ai".to_string()
}

/// Default for `auto_summarise_on_stop`: ON (#68). After a meeting is stopped and
/// finalised, the post-stop background chain auto-runs summarisation (the third
/// step, after any re-transcribe / re-identify-speakers pass) so the summary is
/// ready without the user pressing Summarise. An older store written before this
/// field existed deserialises to `true` via `#[serde(default = ...)]`. See
/// `architecture/cross-cutting.md` — the Agent/stop lifecycle.
/// Default for `preload_summariser`: ON. The summary/chat model is warmed at
/// startup (when already downloaded) so the first use is instant; an older store
/// written before this field existed deserialises to `true` via
/// `#[serde(default = ...)]`.
const fn default_preload_summariser() -> bool {
    true
}

const fn default_auto_summarise_on_stop() -> bool {
    true
}

/// Default for `notes_paper_rules`: ON. The notes editor renders faint
/// horizontal "writing paper" rules behind the text by default; users disable
/// them in the Appearance settings. The oxblood *vertical* margin rule that
/// divides the timestamp gutter from the writing column is structural and
/// always shown — only the horizontal rules are governed by this flag. An older
/// store written before this field existed deserialises to `true`.
const fn default_notes_paper_rules() -> bool {
    true
}

/// Default ASR language hint. "English" forces the English assistant-turn
/// prefix, fixing the spurious-Chinese auto-detect bug for the primary user.
/// The sentinel "auto" restores auto-detect (no prefix; byte-identical to the
/// pre-feature behaviour). An older store written before this field existed
/// deserialises to "English" via #[serde(default = ...)].
fn default_transcription_language() -> String {
    "English".to_string()
}

/// Default output language for generated text (summaries and chat replies).
///
/// The sentinel "auto" instructs `ipc-bridge` to resolve the output language
/// from the host system locale at the time the text is generated. An explicit
/// full English language name (e.g. "French", "German") bypasses locale
/// resolution and uses that name directly. An older store written before this
/// field existed deserialises to "auto" via `#[serde(default = ...)]`.
fn default_output_language() -> String {
    "auto".to_string()
}

/// Default summary system prompt (FR-28).
///
/// A model-agnostic instruction asking for a structured meeting summary:
/// headings, key decisions, and action items. The summariser passes this
/// verbatim as the chat `system` message; users may override it from the
/// settings UI. An older store written before this field existed
/// deserialises to this value via `#[serde(default = ...)]`.
/// Default chat-agent system prompt (Phase 9).
///
/// A concise instruction framing the bundled LLM as a meeting-notes chat
/// assistant that uses tools to read the meeting and act on the user's behalf.
/// The chat engine passes this verbatim as the session's `system` message;
/// users may override it from the settings UI. An older store written before
/// this field existed deserialises to this value via `#[serde(default = ...)]`.
fn default_chat_system_prompt() -> String {
    "You are a meeting-notes assistant for a single meeting. Answer the user's \
     questions about this meeting using the available tools to read its \
     transcript, summary, notes and metadata, and to act on the user's behalf \
     (re-listen to a span, summarise differently, search, set speaker names). \
     Prefer calling a tool over guessing. Be concise and factual; cite \
     timestamps or speakers when relevant; do not invent information that is \
     not present in the meeting."
        .to_string()
}

/// Default live-agent enabled mode: `Off`. The in-meeting co-pilot is opt-in
/// until validated on real hardware; `Auto` would silently run an LLM during
/// every accelerated recording (and `Auto` itself never enables on a
/// shared-memory integrated GPU — see [`minutist_common::live_agent_should_run`]).
/// A store written before this field existed deserialises to `Off` via
/// `#[serde(default)]`.
const fn default_live_agent_enabled() -> LiveAgentMode {
    LiveAgentMode::Off
}

/// Default live-agent cadence: minimum transcript segments before a refresh.
/// 8 segments at the default meeting tempo (~1–2 per minute) is roughly
/// 4–8 minutes between refreshes — enough new material to update the digest
/// without over-refreshing. An older store deserialises to 8.
const fn default_live_agent_min_segments() -> u32 {
    8
}

/// Default live-agent cadence: minimum wall-clock seconds before a refresh.
/// 45 s is the target floor: the debouncer fires only when BOTH the segment
/// count AND the time thresholds are satisfied (whichever is stricter). An
/// older store deserialises to 45.
const fn default_live_agent_min_seconds() -> u32 {
    45
}

/// Default per-category digest toggles — all ON so every category is included
/// in the initial digest. Users disable individual categories in the settings
/// pane. An older store written before these fields existed deserialises to
/// `true` for each.
const fn default_digest_toggle_true() -> bool {
    true
}

/// Default live-agent injected-context char backstop (~80 000 chars). Caps the
/// total attachment / earlier-transcript text retrieved into the tail each refresh
/// so the held `LlamaContext` (n_ctx = 32 768) is not exhausted;
/// `live_agent_retrieval_k` is the dominant knob. An older store deserialises to
/// this value.
const fn default_live_agent_retrieval_budget_chars() -> usize {
    80_000
}

/// Default top-k chunks the live agent retrieves and injects per refresh (the
/// discrete-GPU tier; an integrated GPU is scaled down — see `tier_scaled_k` in
/// the live-agent worker). Each chunk is ~1 KB, so 8 stays well within the held
/// `LlamaContext`. `0` disables retrieval. An older store deserialises to 8.
const fn default_live_agent_retrieval_k() -> usize {
    8
}

/// Default live-agent system prompt: a digest-maintenance instruction that
/// emphasises updating the STANDING LIST rather than regenerating from scratch,
/// so action items and open asks accumulate and are resolved as the meeting
/// progresses rather than being discarded on each refresh.
fn default_live_agent_system_prompt() -> String {
    "You are a live meeting co-pilot. The meeting transcript is delivered to you \
     as the meeting happens, and the user may also type messages to you directly. \
     Use everything said in the meeting so far to answer the user's questions and \
     to help during the meeting. When you are shown new transcript, stay silent \
     unless there is something genuinely worth surfacing — a decision, an action \
     item, an answer to a standing request from the user, or an unresolved \
     reference (the instructions accompanying each transcript update tell you how \
     to stay silent). Always answer a direct question from the user, drawing on \
     the meeting so far. Be concise and factual; never invent information that is \
     not in the transcript or the attached documents."
        .to_string()
}

/// A distinctive phrase from the legacy digest-maintenance system prompt (the
/// pre-U2 default). Detects a persisted store still carrying that prompt so it
/// can be migrated to the co-pilot default: the U2 cutover changed the live
/// agent from a digest updater to a conversational co-pilot, but a store written
/// before then still holds the old instruction — which makes the model ask the
/// user to "provide the current digest and new transcript segments" instead of
/// answering.
const LEGACY_DIGEST_PROMPT_MARKER: &str = "maintaining a structured digest";

/// Deserialize `live_agent_system_prompt`, upgrading a persisted legacy
/// digest-maintenance prompt to the current co-pilot default. A prompt the user
/// has genuinely customised (not containing the legacy marker) is preserved.
fn deserialize_live_agent_system_prompt<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    if s.contains(LEGACY_DIGEST_PROMPT_MARKER) {
        Ok(default_live_agent_system_prompt())
    } else {
        Ok(s)
    }
}

/// Built-in summary prompt presets (Phase 9 — D4).
///
/// Each preset maps to a built-in system prompt via [`preset_prompt`]. The
/// selected preset drives the effective summary prompt UNLESS the user sets a
/// custom override in `summary_system_prompt` (see
/// [`Settings::effective_summary_prompt`]). `Default` reproduces the prior
/// `summarise_meeting` behaviour exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum SummaryPreset {
    /// Structured summary: Summary / Key Decisions / Action Items. The prior
    /// (pre-preset) default behaviour.
    #[default]
    Default,
    /// Like `Default` but explicitly omits greetings, small-talk and sign-off
    /// chit-chat at the start and end of the meeting.
    FilterChitChat,
    /// Focus on decisions, action items and their owners.
    ActionItems,
    /// A thorough, sectioned summary covering topics, discussion and outcomes.
    Detailed,
}

/// The built-in system prompt for a summary [`SummaryPreset`]. Pure; returns a
/// `'static` string so callers can use it without allocating. `Default`
/// returns the same instruction as the pre-preset `summary_system_prompt`
/// default, so an existing store with the default prompt and the default
/// preset produces byte-identical summarisation behaviour.
pub fn preset_prompt(preset: SummaryPreset) -> &'static str {
    match preset {
        SummaryPreset::Default => {
            "You are a meeting-notes assistant. Summarise the meeting transcript and \
             the user's notes into clear Markdown. Use these sections with `##` \
             headings, omitting any that have no content: Summary (a short overview), \
             Key Decisions (a bulleted list of decisions made), and Action Items (a \
             bulleted list of follow-ups, naming the owner when stated). Be concise \
             and factual; do not invent information that is not present in the \
             transcript or notes."
        }
        SummaryPreset::FilterChitChat => {
            "You are a meeting-notes assistant. Summarise the meeting transcript and \
             the user's notes into clear Markdown. Explicitly IGNORE and OMIT general \
             chit-chat: greetings and small-talk at the start, off-topic banter, and \
             sign-off pleasantries at the end — summarise only the substantive \
             discussion. Use these sections with `##` headings, omitting any that have \
             no content: Summary (a short overview), Key Decisions (a bulleted list of \
             decisions made), and Action Items (a bulleted list of follow-ups, naming \
             the owner when stated). Be concise and factual; do not invent information \
             that is not present in the transcript or notes."
        }
        SummaryPreset::ActionItems => {
            "You are a meeting-notes assistant. From the meeting transcript and the \
             user's notes, extract what was decided and what needs to happen next. Use \
             these sections with `##` headings, omitting any that have no content: Key \
             Decisions (a bulleted list of decisions made) and Action Items (a bulleted \
             list of follow-ups, each naming the owner when stated and a due date if \
             mentioned). Keep any narrative summary to one or two sentences at most. Be \
             concise and factual; do not invent owners, dates or commitments that are \
             not present in the transcript or notes."
        }
        SummaryPreset::Detailed => {
            "You are a meeting-notes assistant. Produce a thorough, well-structured \
             Markdown summary of the meeting transcript and the user's notes. Open with \
             a `## Summary` overview, then a `## Discussion` section with one `###` \
             subsection per topic covering the points raised and the reasoning, then \
             `## Key Decisions` (a bulleted list of decisions made) and `## Action \
             Items` (a bulleted list of follow-ups, naming the owner when stated). Omit \
             any section that has no content. Be factual and do not invent information \
             that is not present in the transcript or notes."
        }
    }
}

/// Application settings.
///
/// Fields added in later phases live in their respective phase plans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Settings {
    /// The preferred audio-input device, identified by the opaque id
    /// returned by `audio-capture::AudioCaptureManager::list_devices`.
    /// `None` means "use the OS default".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_device_id: Option<String>,

    /// UI colour-scheme preference.
    #[serde(default)]
    pub theme: Theme,

    /// Root directory for meeting data.  `None` means "use the platform
    /// default app-data directory".
    ///
    /// When `Some(path)`, `app-main` derives `meetings/`, `models/`, and
    /// `index.db` from this path rather than from the platform default.
    /// The value must be an absolute path; a relative or uncreatable path is
    /// logged and ignored (falls back to the platform default — startup is
    /// never aborted).
    ///
    /// **Invariants (bootstrap constraints):**
    /// - `settings.store` always lives at the platform default root — it is
    ///   how this override is found in the first place.
    /// - Logs always live at the platform default root — logging starts before
    ///   settings load.
    ///
    /// **Restart required.** Data roots are resolved once at startup.  Moving
    /// existing data after setting this field is the user's responsibility;
    /// the app does not migrate data automatically.
    ///
    /// **No UI.** The settings pane does not currently render this field; it
    /// must be set by editing `settings.store` directly and restarting.
    ///
    /// `specta` lacks a built-in `Type` impl for `PathBuf`; the explicit
    /// `#[specta(type = Option<String>)]` hint preserves the same wire
    /// shape (UTF-8 path string or `null`) the manual mirror produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "specta", specta(type = Option<String>))]
    pub data_directory: Option<PathBuf>,

    /// If `true`, the main window starts hidden; accessible via the tray icon.
    #[serde(default)]
    pub start_hidden: bool,

    /// Notes-editor autosave interval, in seconds (FR-18/FR-35).
    ///
    /// The editor debounces autosaves of `notes.json`/`notes.md` to this
    /// cadence. Defaults to 5 s; an older store written before this field
    /// existed deserialises to the default via `#[serde(default = ...)]`.
    #[serde(default = "default_autosave_interval_secs")]
    pub autosave_interval_secs: u32,

    /// Optional CUSTOM summary system prompt that OVERRIDES the selected
    /// [`SummaryPreset`] (FR-28 / D4).
    ///
    /// Empty by default (the common case): [`Settings::effective_summary_prompt`]
    /// then returns `preset_prompt(self.summary_preset)`, so the preset picker
    /// drives summarisation. A non-empty value is a user override and wins over
    /// the preset. The `summariser` forwards the effective prompt verbatim as the
    /// `system` message when generating `summary.md`.
    #[serde(default)]
    pub summary_system_prompt: String,

    /// Selected LLM model for summarisation (FR-35).
    ///
    /// `None` means "use the bundled default model" (resolved by the
    /// summariser against the model registry). The model is settings-selected,
    /// never hard-coded; switching is a manifest + `llm_model_id` change. An
    /// older store written before this field existed deserialises to `None`
    /// via `#[serde(default)]`.
    #[serde(default)]
    pub llm_model_id: Option<ModelId>,

    /// Whether the post-recording diarization pass runs (FR-11).
    ///
    /// Diarization is post-hoc and off by default: the orchestrator gates its
    /// on-stop diarizer pass (and the user-triggered re-diarize) on this flag.
    /// `#[serde(default)]` defaults to `false`; an older store written before
    /// this field existed deserialises to `false`.
    #[serde(default)]
    pub diarization_enabled: bool,

    /// Whether the voiceprint enrolment flow is active (issue #0003).
    ///
    /// Enrolment is an explicit, opt-in, per-speaker operation: when `true`,
    /// renaming a speaker in the UI triggers `Orchestrator::enrol_voiceprint`,
    /// which extracts a CAM++ centroid from clean segments for that label and
    /// stores it in `voiceprints.db`. Default `false` satisfies the
    /// collection-time consent obligation (BIPA / GDPR Art. 9 — no voiceprint
    /// is created without an explicit user opt-in). An older store written
    /// before this field existed deserialises to `false` via `#[serde(default)]`.
    #[serde(default)]
    pub voiceprint_enrolment_enabled: bool,

    /// Whether the first-run onboarding flow has been completed (Phase 7).
    ///
    /// The webview gates the main UI behind this: `false` shows the onboarding
    /// flow (welcome → model download → settings tour), which sets it `true` on
    /// completion. `#[serde(default)]` defaults to `false`; an older store
    /// written before this field existed deserialises to `false` (so existing
    /// users see onboarding once on upgrade — acceptable for a pre-release).
    #[serde(default)]
    pub onboarding_completed: bool,

    /// GPU-acceleration MODE (replaces the former `bool`).
    ///
    /// `Auto` (the default) probes GPU VRAM at each model load and offloads a
    /// model to the GPU only when it fits, else CPU; `On` forces GPU offload;
    /// `Off` forces CPU. Only effective in a GPU-feature build
    /// (`vulkan`/`metal`/`cuda`/`rocm`); a default CPU-only build always runs on
    /// CPU. Migrates a legacy bool store: `true → Auto`, `false → Off` (see
    /// [`deserialize_gpu_acceleration`]). `#[serde(default = ...)]` →  `Auto` when
    /// the field is missing. See `architecture/cross-cutting.md` — "GPU
    /// portability".
    #[serde(
        default = "default_gpu_acceleration",
        deserialize_with = "deserialize_gpu_acceleration"
    )]
    pub gpu_acceleration: GpuAcceleration,

    /// Whether to capture and mix the system/call (loopback) audio alongside
    /// the microphone, so a Teams-style call transcribes all participants.
    ///
    /// When `true`, `audio-capture` ALSO opens the default render endpoint in
    /// loopback mode, resamples it to 16 kHz mono, and SUMS it sample-wise with
    /// the mic into the single `samples` stream the orchestrator drains;
    /// downstream diarization separates the speakers. When `false`, behaviour is
    /// mic-only.
    ///
    /// `#[serde(default = ...)]` defaults to `true` — capturing the call is the
    /// point of a meeting-notes app, so it is opt-OUT; an older store written
    /// before this field existed deserialises to `true`. If the mic also picks
    /// the call audio up from the speakers, mixing the loopback in doubles it
    /// (echo), so the UI advises turning it off in that case. Loopback capture is
    /// currently Windows-only; on other platforms enabling this logs a warning
    /// and falls back to mic-only (never failing the recording). Echo
    /// cancellation using the loopback as the reference signal is future work —
    /// see `architecture/cross-cutting.md` — "Threading model".
    #[serde(default = "default_capture_system_audio")]
    pub capture_system_audio: bool,

    /// ASR language hint (Qwen3-ASR). A full English language name (e.g.
    /// "English", "Spanish", "Japanese") forces that language via the
    /// assistant-turn prefix; the sentinel "auto" disables forcing (auto-detect,
    /// the pre-feature behaviour). Defaults to "English" (fixes the spurious-
    /// Chinese bug). An older store deserialises to "English" via the default fn.
    #[serde(default = "default_transcription_language")]
    pub transcription_language: String,

    /// Whether the notes editor renders faint horizontal "writing paper" rules
    /// behind the text. Presentation-only: the webview reads this and toggles a
    /// class on the editor surface. The oxblood *vertical* margin rule that
    /// divides the timestamp gutter from the writing column is structural and is
    /// always shown regardless of this flag. `#[serde(default = ...)]` defaults
    /// to `true`; an older store written before this field existed deserialises
    /// to `true`.
    #[serde(default = "default_notes_paper_rules")]
    pub notes_paper_rules: bool,

    /// Chat-agent system prompt (Phase 9). The chat engine forwards this
    /// verbatim as the session's `system` message. `#[serde(default = ...)]`
    /// defaults to a meeting-notes-assistant instruction; an older store
    /// written before this field existed deserialises to that default. Added
    /// to the hand-written `Default` impl.
    #[serde(default = "default_chat_system_prompt")]
    pub chat_system_prompt: String,

    /// Selected summary prompt preset (Phase 9 — D4). Drives the effective
    /// summary prompt via [`preset_prompt`] UNLESS `summary_system_prompt` is a
    /// non-empty user override (see [`Settings::effective_summary_prompt`]).
    /// `#[serde(default)]` defaults to [`SummaryPreset::Default`] (the prior
    /// behaviour); an older store deserialises to `Default`.
    #[serde(default)]
    pub summary_preset: SummaryPreset,

    /// Whether the in-process MCP server is started (Phase 10). Opt-in, **off by
    /// default**, mirroring `external-ollama` / `capture_system_audio`. When
    /// `true`, `app-main` binds the loopback Streamable HTTP MCP endpoint at
    /// startup so an external MCP client can reach the shared tool layer. Toggling at runtime is a documented
    /// restart-required for v1. `#[serde(default)]` defaults to `false`; an older
    /// store written before this field existed deserialises to `false`. See
    /// `architecture/cross-cutting.md` — "MCP transport".
    #[serde(default)]
    pub mcp_enabled: bool,

    /// The loopback TCP port the MCP server binds (Phase 10 — D1). A FIXED
    /// default (8765): only one app instance runs, so a stable port keeps an
    /// external MCP client's saved URL valid across runs. User-editable.
    /// `#[serde(default = ...)]` defaults to 8765; an older store deserialises to
    /// 8765. See `architecture/cross-cutting.md` — "MCP transport".
    #[serde(default = "default_mcp_port")]
    pub mcp_port: u16,

    /// Whether write tools are exposed over MCP (Phase 10 — D3). **Off by
    /// default = read-only over MCP.** Consulted at registry-projection time by
    /// the MCP server: with it OFF, `tools/list` is read/compute tools + the
    /// inter-agent tool only; with it ON, the reversible writes
    /// (`set_speaker_name`, `rename_meeting`) join. `reprocess_meeting` /
    /// any destructive tool stay internal-only regardless
    /// (they are never `expose_over_mcp`). `#[serde(default)]` defaults to
    /// `false`; an older store deserialises to `false`. See
    /// `architecture/cross-cutting.md` — "MCP transport".
    #[serde(default)]
    pub mcp_write_tools: bool,

    /// Auto-summarise a meeting after it stops + finalises (#68).
    ///
    /// When `true` (the default), the post-stop background chain in `ipc-bridge`
    /// runs summarisation as its THIRD step (after any re-transcribe /
    /// re-identify-speakers pass) so the summary is ready without the user
    /// pressing Summarise. Best-effort: an error is logged, like the other
    /// passes. `#[serde(default = ...)]` defaults to `true`; an older store
    /// written before this field existed deserialises to `true`, preserving the
    /// new default for existing users. See `architecture/cross-cutting.md` — the
    /// Agent/stop lifecycle.
    #[serde(default = "default_auto_summarise_on_stop")]
    pub auto_summarise_on_stop: bool,

    /// Preload the summary/chat LLM at app startup and keep it resident.
    ///
    /// The summary path and the chat agent share ONE held `LlamaSummariser`
    /// (loaded once, kept for the process lifetime). When `true` (the default),
    /// `app-main` warms it in the background at startup IF the model is already
    /// downloaded — so the first Summarise / chat is instant rather than paying a
    /// multi-GB load. It NEVER triggers a download at startup (a not-yet-fetched
    /// model is left to load on first use / after the onboarding download). When
    /// `false`, the model loads on-demand on first use (the prior behaviour) — the
    /// escape hatch for keeping idle memory low. `#[serde(default = ...)]`
    /// defaults to `true`; an older store written before this field existed
    /// deserialises to `true`.
    #[serde(default = "default_preload_summariser")]
    pub preload_summariser: bool,

    /// Output language for all generated text: summaries and chat replies.
    ///
    /// The transcript itself is never affected — this controls only the language
    /// the LLM uses when generating output. The sentinel "auto" resolves to the
    /// host system locale at generation time (via `sys-locale` in `ipc-bridge`);
    /// a full English language name (e.g. "French", "German") passes through
    /// verbatim. `#[serde(default = ...)]` defaults to "auto"; an older store
    /// written before this field existed deserialises to "auto".
    #[serde(default = "default_output_language")]
    pub output_language: String,

    /// Whether the connected-tier relay connector is enabled (WS4-A S5b).
    /// **Off by default**, mirroring `mcp_enabled`. When `true` AND a device
    /// credential is stored (the device is paired), `app-main` (connected build
    /// only) starts the tunnel lifecycle: it dials the relay so an external MCP
    /// client (Claude web/Desktop, ChatGPT, Codex) can reach this app's tools
    /// over the relay. The connector channel transits meeting content to the AI
    /// vendor BY DESIGN (the user asked for it) — it is never described as
    /// private (D5). `#[serde(default)]` defaults to `false`; an older store
    /// deserialises to `false`. The free build ignores this field entirely (no
    /// `tunnel-client`). See `architecture/cross-cutting.md` — "Build variants".
    #[serde(default)]
    pub connector_enabled: bool,

    /// The relay tunnel WSS rendezvous URL (WS4-A S5b). User-overridable for
    /// self-hosting / testing; defaults to the minutist.ai endpoint. Must be
    /// `wss://` for an off-machine host (`ws://` only for a loopback host, the
    /// tunnel-client scheme check). `#[serde(default = ...)]` defaults to the
    /// minutist.ai endpoint; an older store deserialises to it.
    #[serde(default = "default_relay_url")]
    pub relay_url: String,

    /// The account-service API base URL the device-code pairing client posts
    /// `/pair/start` + `/pair/poll` against (WS4-A S5b). User-overridable;
    /// defaults to the minutist.ai endpoint. Must be `https://` for an off-machine
    /// host. `#[serde(default = ...)]` defaults to the minutist.ai endpoint.
    #[serde(default = "default_relay_api_url")]
    pub relay_api_url: String,

    // -----------------------------------------------------------------------
    // Live in-meeting agent (Phase 9 auto-driver)
    // -----------------------------------------------------------------------
    /// Whether the live in-meeting agent runs during an active recording.
    ///
    /// `Auto` (the default) enables the agent when GPU acceleration is active
    /// (a usable GPU is present AND `gpu_acceleration != Off`), as resolved by
    /// `live_agent_should_run`. `On` enables unconditionally; `Off` disables.
    /// Distinct from `gpu_acceleration`, which governs model-layer placement.
    /// `#[serde(default)]` → `Auto` (the `LiveAgentMode::default()`); an older
    /// store written before this field existed deserialises to `Auto`.
    #[serde(default = "default_live_agent_enabled")]
    pub live_agent_enabled: LiveAgentMode,

    /// Minimum number of new transcript segments that must accumulate before
    /// the live agent fires a digest refresh. The debouncer fires only when
    /// BOTH `live_agent_min_segments` AND `live_agent_min_seconds` are
    /// satisfied. Defaults to 8; an older store deserialises to 8.
    #[serde(default = "default_live_agent_min_segments")]
    pub live_agent_min_segments: u32,

    /// Minimum wall-clock seconds that must elapse before the live agent fires
    /// a digest refresh. The debouncer fires only when BOTH
    /// `live_agent_min_segments` AND `live_agent_min_seconds` are satisfied.
    /// Defaults to 45; an older store deserialises to 45.
    #[serde(default = "default_live_agent_min_seconds")]
    pub live_agent_min_seconds: u32,

    /// Whether action items are included in the live digest panel.
    /// Defaults to `true`; an older store deserialises to `true`.
    #[serde(default = "default_digest_toggle_true")]
    pub live_agent_digest_action_items: bool,

    /// Whether decisions are included in the live digest panel.
    /// Defaults to `true`; an older store deserialises to `true`.
    #[serde(default = "default_digest_toggle_true")]
    pub live_agent_digest_decisions: bool,

    /// Whether open / unanswered asks are included in the live digest panel.
    /// Defaults to `true`; an older store deserialises to `true`.
    #[serde(default = "default_digest_toggle_true")]
    pub live_agent_digest_open_asks: bool,

    /// Whether attachment-sourced answers are included in the live digest panel.
    /// Defaults to `true`; an older store deserialises to `true`.
    #[serde(default = "default_digest_toggle_true")]
    pub live_agent_digest_attachment_answers: bool,

    /// Whether unresolved references are included in the live digest panel.
    /// Defaults to `true`; an older store deserialises to `true`.
    #[serde(default = "default_digest_toggle_true")]
    pub live_agent_digest_unresolved_references: bool,

    /// Upper bound (characters) on the attachment / earlier-transcript context the
    /// live agent retrieves into its tail per refresh. A backstop —
    /// `live_agent_retrieval_k` is the dominant knob (each chunk is ~1 KB). Within
    /// the held `LlamaContext`'s n_ctx = 32 768. An older store deserialises to 80 000.
    /// (`serde(alias)` keeps the pre-rename `live_agent_attachment_budget_chars` key
    /// readable from existing on-disk settings.)
    #[serde(
        default = "default_live_agent_retrieval_budget_chars",
        alias = "live_agent_attachment_budget_chars"
    )]
    pub live_agent_retrieval_budget_chars: usize,

    /// Top-k attachment / earlier-transcript chunks the live agent retrieves and
    /// injects into its tail per refresh (the discrete-GPU tier; an integrated GPU
    /// is scaled down). `0` disables retrieval. An older store deserialises to 8.
    #[serde(default = "default_live_agent_retrieval_k")]
    pub live_agent_retrieval_k: usize,

    /// The system prompt for the live co-pilot, pinned as the prefix of the one
    /// keep-alive conversation. A persisted store still carrying the pre-U2
    /// digest-maintenance prompt is migrated to the co-pilot default on load (see
    /// [`deserialize_live_agent_system_prompt`]); `#[serde(default = ...)]`
    /// supplies it when the field is absent.
    #[serde(
        default = "default_live_agent_system_prompt",
        deserialize_with = "deserialize_live_agent_system_prompt"
    )]
    pub live_agent_system_prompt: String,
}

impl Settings {
    /// The effective summary system prompt (Phase 9 — D4).
    ///
    /// Returns the user's `summary_system_prompt` when it is a non-empty custom
    /// override; otherwise the built-in prompt for the selected
    /// [`summary_preset`](Self::summary_preset). `summarise_meeting` resolves the
    /// prompt through this so the preset picker and the custom-prompt override
    /// share one resolution point.
    pub fn effective_summary_prompt(&self) -> String {
        if self.summary_system_prompt.trim().is_empty() {
            preset_prompt(self.summary_preset).to_string()
        } else {
            self.summary_system_prompt.clone()
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_device_id: None,
            theme: Theme::default(),
            data_directory: None,
            start_hidden: false,
            autosave_interval_secs: default_autosave_interval_secs(),
            summary_system_prompt: String::new(),
            llm_model_id: None,
            diarization_enabled: false,
            onboarding_completed: false,
            gpu_acceleration: default_gpu_acceleration(),
            capture_system_audio: true,
            transcription_language: default_transcription_language(),
            notes_paper_rules: default_notes_paper_rules(),
            chat_system_prompt: default_chat_system_prompt(),
            summary_preset: SummaryPreset::default(),
            mcp_enabled: false,
            mcp_port: default_mcp_port(),
            mcp_write_tools: false,
            auto_summarise_on_stop: default_auto_summarise_on_stop(),
            preload_summariser: default_preload_summariser(),
            output_language: default_output_language(),
            connector_enabled: false,
            relay_url: default_relay_url(),
            relay_api_url: default_relay_api_url(),
            live_agent_enabled: default_live_agent_enabled(),
            live_agent_min_segments: default_live_agent_min_segments(),
            live_agent_min_seconds: default_live_agent_min_seconds(),
            live_agent_digest_action_items: default_digest_toggle_true(),
            live_agent_digest_decisions: default_digest_toggle_true(),
            live_agent_digest_open_asks: default_digest_toggle_true(),
            live_agent_digest_attachment_answers: default_digest_toggle_true(),
            live_agent_digest_unresolved_references: default_digest_toggle_true(),
            live_agent_retrieval_budget_chars: default_live_agent_retrieval_budget_chars(),
            live_agent_retrieval_k: default_live_agent_retrieval_k(),
            live_agent_system_prompt: default_live_agent_system_prompt(),
            voiceprint_enrolment_enabled: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::SettingsStore;

    // -----------------------------------------------------------------------
    // 1. JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn settings_default_round_trips_json() {
        let original = Settings::default();
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(original, restored);
    }

    #[test]
    fn settings_with_all_fields_round_trips_json() {
        let original = Settings {
            input_device_id: Some("hw:1,0".to_string()),
            theme: Theme::Dark,
            data_directory: Some(PathBuf::from("/tmp/meeting-data")),
            start_hidden: true,
            autosave_interval_secs: 17,
            summary_system_prompt: "Summarise tersely.".to_string(),
            llm_model_id: Some(ModelId::from("gemma-4-e4b-it-q4_k_m")),
            diarization_enabled: true,
            onboarding_completed: true,
            gpu_acceleration: GpuAcceleration::On,
            capture_system_audio: true,
            transcription_language: "Japanese".to_string(),
            notes_paper_rules: false,
            chat_system_prompt: "Be a terse assistant.".to_string(),
            summary_preset: SummaryPreset::ActionItems,
            mcp_enabled: true,
            mcp_port: 9999,
            mcp_write_tools: true,
            auto_summarise_on_stop: false,
            preload_summariser: false,
            output_language: "German".to_string(),
            connector_enabled: true,
            relay_url: "wss://relay.example/tunnel".to_string(),
            relay_api_url: "https://api.example".to_string(),
            live_agent_enabled: LiveAgentMode::On,
            live_agent_min_segments: 5,
            live_agent_min_seconds: 30,
            live_agent_digest_action_items: true,
            live_agent_digest_decisions: false,
            live_agent_digest_open_asks: true,
            live_agent_digest_attachment_answers: false,
            live_agent_digest_unresolved_references: true,
            live_agent_retrieval_budget_chars: 40_000,
            live_agent_retrieval_k: 6,
            live_agent_system_prompt: "Custom live prompt.".to_string(),
            voiceprint_enrolment_enabled: true,
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.transcription_language, "Japanese");
        assert_eq!(restored.output_language, "German");
        assert!(restored.connector_enabled);
        assert_eq!(restored.relay_url, "wss://relay.example/tunnel");
        assert_eq!(restored.live_agent_enabled, LiveAgentMode::On);
        assert_eq!(restored.live_agent_min_segments, 5);
        assert_eq!(restored.live_agent_min_seconds, 30);
        assert!(!restored.live_agent_digest_decisions);
        assert_eq!(restored.live_agent_retrieval_budget_chars, 40_000);
        assert_eq!(restored.live_agent_system_prompt, "Custom live prompt.");
        assert!(restored.voiceprint_enrolment_enabled);
        assert_eq!(original, restored);
    }

    // -----------------------------------------------------------------------
    // 1b. autosave_interval_secs: default value + missing-field deserialisation
    // -----------------------------------------------------------------------

    #[test]
    fn autosave_interval_defaults_to_five() {
        assert_eq!(Settings::default().autosave_interval_secs, 5);
    }

    #[test]
    fn autosave_interval_round_trips() {
        let original = Settings {
            autosave_interval_secs: 42,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.autosave_interval_secs, 42);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_autosave_field_defaults_to_five() {
        // A settings store written before `autosave_interval_secs` existed.
        let old_json = r#"{ "theme": "dark", "start_hidden": true }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert_eq!(
            restored.autosave_interval_secs, 5,
            "missing autosave_interval_secs must deserialise to the default (5)"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1c. summary_system_prompt + llm_model_id: defaults + missing-field
    //     deserialisation (FR-28 / FR-35)
    // -----------------------------------------------------------------------

    #[test]
    fn summary_prompt_defaults_to_default_preset() {
        // The custom override is EMPTY by default (D4), so the effective prompt is
        // the Default preset — which asks for the three structured sections (FR-28).
        let s = Settings::default();
        assert!(
            s.summary_system_prompt.is_empty(),
            "custom override must default empty so the preset drives"
        );
        assert_eq!(s.summary_preset, SummaryPreset::Default);
        let effective = s.effective_summary_prompt();
        assert!(effective.contains("Key Decisions"));
        assert!(effective.contains("Action Items"));
        assert_eq!(effective, preset_prompt(SummaryPreset::Default));
    }

    #[test]
    fn llm_model_id_defaults_to_none() {
        assert_eq!(Settings::default().llm_model_id, None);
    }

    #[test]
    fn summary_prompt_and_llm_model_id_round_trip() {
        let original = Settings {
            summary_system_prompt: "Be brief.".to_string(),
            llm_model_id: Some(ModelId::from("granite-4.1-3b-q4_k_m")),
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.summary_system_prompt, "Be brief.");
        assert_eq!(
            restored.llm_model_id,
            Some(ModelId::from("granite-4.1-3b-q4_k_m"))
        );
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_summary_fields_defaults() {
        // A settings store written before `summary_system_prompt` and
        // `llm_model_id` existed (it still carries the Phase-3 field).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            restored.summary_system_prompt.is_empty(),
            "missing summary_system_prompt must deserialise to the empty override (preset drives)"
        );
        assert_eq!(
            restored.effective_summary_prompt(),
            preset_prompt(SummaryPreset::Default),
            "with no override + default preset, the effective prompt is the Default preset"
        );
        assert_eq!(
            restored.llm_model_id, None,
            "missing llm_model_id must deserialise to None"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1d. diarization_enabled: default + round-trip + missing-field
    //     deserialisation (FR-11)
    // -----------------------------------------------------------------------

    #[test]
    fn diarization_enabled_defaults_to_false() {
        assert!(
            !Settings::default().diarization_enabled,
            "diarization is post-hoc and off by default (FR-11)"
        );
    }

    #[test]
    fn diarization_enabled_round_trips() {
        let original = Settings {
            diarization_enabled: true,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(restored.diarization_enabled);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_diarization_field_defaults_to_false() {
        // A settings store written before `diarization_enabled` existed (it
        // still carries the Phase-3/Phase-5 fields).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            !restored.diarization_enabled,
            "missing diarization_enabled must deserialise to false (FR-11 default)"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1e. auto_summarise_on_stop: default + round-trip + missing-field
    //     deserialisation (#68)
    // -----------------------------------------------------------------------

    #[test]
    fn auto_summarise_on_stop_defaults_to_true() {
        assert!(
            Settings::default().auto_summarise_on_stop,
            "auto-summarise-on-stop is ON by default (#68)"
        );
    }

    #[test]
    fn auto_summarise_on_stop_round_trips() {
        let original = Settings {
            auto_summarise_on_stop: false,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(!restored.auto_summarise_on_stop);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_auto_summarise_field_defaults_to_true() {
        // A settings store written before `auto_summarise_on_stop` existed must
        // adopt the new ON-by-default so existing users get the behaviour.
        let old_json = r#"{ "theme": "dark", "diarization_enabled": true }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            restored.auto_summarise_on_stop,
            "missing auto_summarise_on_stop must deserialise to true (#68 default)"
        );
    }

    // -----------------------------------------------------------------------
    // 1e'. preload_summariser: default + round-trip + missing-field
    //      deserialisation
    // -----------------------------------------------------------------------

    #[test]
    fn preload_summariser_defaults_to_true() {
        assert!(
            Settings::default().preload_summariser,
            "the summary/chat model preloads at startup by default"
        );
    }

    #[test]
    fn preload_summariser_round_trips() {
        let original = Settings {
            preload_summariser: false,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(!restored.preload_summariser);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_preload_summariser_field_defaults_to_true() {
        let old_json = r#"{ "theme": "dark" }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            restored.preload_summariser,
            "missing preload_summariser must deserialise to true"
        );
    }

    // -----------------------------------------------------------------------
    // 1e. onboarding_completed: default + round-trip + missing-field
    //     deserialisation (Phase 7)
    // -----------------------------------------------------------------------

    #[test]
    fn onboarding_completed_defaults_to_false() {
        assert!(
            !Settings::default().onboarding_completed,
            "first run shows onboarding (Phase 7)"
        );
    }

    #[test]
    fn onboarding_completed_round_trips() {
        let original = Settings {
            onboarding_completed: true,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(restored.onboarding_completed);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_onboarding_field_defaults_to_false() {
        // A store written before `onboarding_completed` existed.
        let old_json = r#"{ "theme": "dark", "diarization_enabled": true }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            !restored.onboarding_completed,
            "missing onboarding_completed must deserialise to false"
        );
        assert!(restored.diarization_enabled);
    }

    // -----------------------------------------------------------------------
    // 1f. gpu_acceleration: tri-state default + round-trip + legacy-bool
    //     migration + missing-field deserialisation (runtime GPU mode)
    // -----------------------------------------------------------------------

    #[test]
    fn gpu_acceleration_defaults_to_auto() {
        assert_eq!(
            Settings::default().gpu_acceleration,
            GpuAcceleration::Auto,
            "GPU mode defaults to Auto (probe VRAM and fit, else CPU)"
        );
    }

    #[test]
    fn gpu_acceleration_round_trips() {
        for mode in [
            GpuAcceleration::Auto,
            GpuAcceleration::On,
            GpuAcceleration::Off,
        ] {
            let original = Settings {
                gpu_acceleration: mode,
                ..Settings::default()
            };
            let json = serde_json::to_string(&original).expect("serialise");
            let restored: Settings = serde_json::from_str(&json).expect("deserialise");
            assert_eq!(restored.gpu_acceleration, mode);
            assert_eq!(original, restored);
        }
    }

    #[test]
    fn gpu_acceleration_migrates_a_legacy_bool_store() {
        // Legacy `true` was the unconsidered default → adopt the safe Auto.
        let t: Settings = serde_json::from_str(r#"{ "gpu_acceleration": true }"#).expect("true");
        assert_eq!(t.gpu_acceleration, GpuAcceleration::Auto);
        // Legacy `false` WAS a deliberate CPU choice → the hard Off override.
        let f: Settings = serde_json::from_str(r#"{ "gpu_acceleration": false }"#).expect("false");
        assert_eq!(f.gpu_acceleration, GpuAcceleration::Off);
        // The new string form round-trips as-is.
        let on: Settings = serde_json::from_str(r#"{ "gpu_acceleration": "on" }"#).expect("on");
        assert_eq!(on.gpu_acceleration, GpuAcceleration::On);
    }

    #[test]
    fn old_store_json_without_gpu_acceleration_field_defaults_to_auto() {
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert_eq!(restored.gpu_acceleration, GpuAcceleration::Auto);
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    #[test]
    fn live_agent_system_prompt_migrates_legacy_digest_store() {
        // A store persisted before the U2 cutover carries the digest-maintenance
        // prompt; it must migrate to the co-pilot default on load so the model
        // stops asking the user to "provide the digest and transcript segments".
        let legacy = r#"{ "live_agent_system_prompt": "You are a live meeting assistant maintaining a structured digest of an ongoing meeting. UPDATE the digest in place." }"#;
        let s: Settings = serde_json::from_str(legacy).expect("deserialise legacy store");
        assert_eq!(s.live_agent_system_prompt, default_live_agent_system_prompt());
        assert!(!s.live_agent_system_prompt.contains(LEGACY_DIGEST_PROMPT_MARKER));

        // A genuinely customised prompt (no legacy marker) is preserved verbatim.
        let custom = r#"{ "live_agent_system_prompt": "Be my terse meeting helper." }"#;
        let s2: Settings = serde_json::from_str(custom).expect("deserialise custom store");
        assert_eq!(s2.live_agent_system_prompt, "Be my terse meeting helper.");

        // An absent field falls back to the co-pilot default.
        let s3: Settings = serde_json::from_str("{}").expect("deserialise empty store");
        assert_eq!(s3.live_agent_system_prompt, default_live_agent_system_prompt());
    }

    // -----------------------------------------------------------------------
    // 1g. capture_system_audio: default + round-trip + missing-field
    //     deserialisation (loopback call-audio mixing)
    // -----------------------------------------------------------------------

    #[test]
    fn capture_system_audio_defaults_to_true() {
        assert!(
            Settings::default().capture_system_audio,
            "capturing the call is the point; system-audio capture is on by default (opt-out)"
        );
    }

    #[test]
    fn capture_system_audio_round_trips() {
        let original = Settings {
            capture_system_audio: true,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(restored.capture_system_audio);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_capture_system_audio_field_defaults_to_true() {
        // A settings store written before `capture_system_audio` existed must
        // deserialise to `true` (on by default, opt-out).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            restored.capture_system_audio,
            "missing capture_system_audio must deserialise to true (on by default)"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1h. transcription_language: default + round-trip + missing-field
    //     deserialisation (ASR language hint)
    // -----------------------------------------------------------------------

    #[test]
    fn transcription_language_defaults_to_english() {
        assert_eq!(
            Settings::default().transcription_language,
            "English",
            "the ASR language hint defaults to English (fixes the spurious-Chinese bug)"
        );
    }

    #[test]
    fn transcription_language_round_trips() {
        let original = Settings {
            transcription_language: "Spanish".to_string(),
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.transcription_language, "Spanish");
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_transcription_language_field_defaults_to_english() {
        // A settings store written before `transcription_language` existed must
        // deserialise to "English" (the default fn).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert_eq!(
            restored.transcription_language, "English",
            "missing transcription_language must deserialise to English (the default)"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1i. notes_paper_rules: default + round-trip + missing-field
    //     deserialisation (notes-editor writing-paper rules)
    // -----------------------------------------------------------------------

    #[test]
    fn notes_paper_rules_defaults_to_true() {
        assert!(
            Settings::default().notes_paper_rules,
            "the notes editor shows writing-paper rules by default (Appearance can disable)"
        );
    }

    #[test]
    fn notes_paper_rules_round_trips() {
        let original = Settings {
            notes_paper_rules: false,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(!restored.notes_paper_rules);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_notes_paper_rules_field_defaults_to_true() {
        // A settings store written before `notes_paper_rules` existed must
        // deserialise to `true` (rules on by default).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            restored.notes_paper_rules,
            "missing notes_paper_rules must deserialise to true (on by default)"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1j. chat_system_prompt + summary_preset: defaults + round-trip +
    //     missing-field deserialisation (Phase 9 — D4)
    // -----------------------------------------------------------------------

    #[test]
    fn chat_system_prompt_defaults_to_assistant_instruction() {
        let prompt = Settings::default().chat_system_prompt;
        assert!(!prompt.is_empty(), "default chat prompt must be non-empty");
        assert_eq!(prompt, default_chat_system_prompt());
    }

    #[test]
    fn summary_preset_defaults_to_default_variant() {
        assert_eq!(Settings::default().summary_preset, SummaryPreset::Default);
    }

    #[test]
    fn chat_prompt_and_preset_round_trip() {
        let original = Settings {
            chat_system_prompt: "Stay terse.".to_string(),
            summary_preset: SummaryPreset::Detailed,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.chat_system_prompt, "Stay terse.");
        assert_eq!(restored.summary_preset, SummaryPreset::Detailed);
        assert_eq!(original, restored);
    }

    #[test]
    fn summary_preset_serialises_snake_case() {
        assert_eq!(
            serde_json::to_string(&SummaryPreset::Default).unwrap(),
            "\"default\""
        );
        assert_eq!(
            serde_json::to_string(&SummaryPreset::FilterChitChat).unwrap(),
            "\"filter_chit_chat\""
        );
        assert_eq!(
            serde_json::to_string(&SummaryPreset::ActionItems).unwrap(),
            "\"action_items\""
        );
        assert_eq!(
            serde_json::to_string(&SummaryPreset::Detailed).unwrap(),
            "\"detailed\""
        );
    }

    #[test]
    fn old_store_json_without_chat_and_preset_fields_defaults() {
        // A store written before `chat_system_prompt` / `summary_preset` existed.
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert_eq!(
            restored.chat_system_prompt,
            default_chat_system_prompt(),
            "missing chat_system_prompt must deserialise to the default"
        );
        assert_eq!(
            restored.summary_preset,
            SummaryPreset::Default,
            "missing summary_preset must deserialise to Default"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1k. preset_prompt + effective_summary_prompt (Phase 9 — D4)
    // -----------------------------------------------------------------------

    #[test]
    fn preset_prompt_default_is_the_structured_instruction() {
        // The `Default` preset reproduces the pre-preset default behaviour: the
        // structured Summary / Key Decisions / Action Items instruction, so a
        // default install summarises byte-identically to before presets existed.
        let p = preset_prompt(SummaryPreset::Default);
        assert!(p.contains("Key Decisions"));
        assert!(p.contains("Action Items"));
        assert!(p.contains("Summarise the meeting transcript"));
    }

    #[test]
    fn preset_prompt_each_variant_is_distinct_and_nonempty() {
        let prompts = [
            preset_prompt(SummaryPreset::Default),
            preset_prompt(SummaryPreset::FilterChitChat),
            preset_prompt(SummaryPreset::ActionItems),
            preset_prompt(SummaryPreset::Detailed),
        ];
        for p in prompts {
            assert!(!p.is_empty(), "preset prompt must be non-empty");
        }
        // FilterChitChat explicitly mentions omitting chit-chat.
        assert!(preset_prompt(SummaryPreset::FilterChitChat)
            .to_lowercase()
            .contains("chit-chat"));
        // Distinctness: at least the chit-chat and detailed presets differ from
        // the default.
        assert_ne!(prompts[0], prompts[1]);
        assert_ne!(prompts[0], prompts[2]);
        assert_ne!(prompts[0], prompts[3]);
    }

    #[test]
    fn effective_summary_prompt_override_wins_when_nonempty() {
        let s = Settings {
            summary_system_prompt: "My custom prompt.".to_string(),
            summary_preset: SummaryPreset::Detailed,
            ..Settings::default()
        };
        assert_eq!(s.effective_summary_prompt(), "My custom prompt.");
    }

    #[test]
    fn effective_summary_prompt_falls_back_to_preset_when_override_blank() {
        // A blank (whitespace-only) override falls through to the selected preset.
        let s = Settings {
            summary_system_prompt: "   ".to_string(),
            summary_preset: SummaryPreset::ActionItems,
            ..Settings::default()
        };
        assert_eq!(
            s.effective_summary_prompt(),
            preset_prompt(SummaryPreset::ActionItems)
        );
    }

    // -----------------------------------------------------------------------
    // 1l. MCP fields: defaults + round-trip + missing-field deserialisation
    //     (Phase 10 — D1/D3)
    // -----------------------------------------------------------------------

    #[test]
    fn mcp_fields_default_to_off_and_fixed_port() {
        let s = Settings::default();
        assert!(!s.mcp_enabled, "the MCP server is opt-in, off by default");
        assert_eq!(s.mcp_port, 8765, "the fixed default MCP port is 8765 (D1)");
        assert!(
            !s.mcp_write_tools,
            "MCP is read-only by default; write exposure is opt-in (D3)"
        );
    }

    #[test]
    fn mcp_fields_round_trip() {
        let original = Settings {
            mcp_enabled: true,
            mcp_port: 7000,
            mcp_write_tools: true,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(restored.mcp_enabled);
        assert_eq!(restored.mcp_port, 7000);
        assert!(restored.mcp_write_tools);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_mcp_fields_defaults() {
        // A settings store written before the Phase-10 MCP fields existed must
        // deserialise to the safe defaults (server off, fixed port 8765,
        // read-only).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            !restored.mcp_enabled,
            "missing mcp_enabled must deserialise to false (off by default)"
        );
        assert_eq!(
            restored.mcp_port, 8765,
            "missing mcp_port must deserialise to the fixed default (8765)"
        );
        assert!(
            !restored.mcp_write_tools,
            "missing mcp_write_tools must deserialise to false (read-only over MCP)"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 1m. output_language: default + round-trip + missing-field
    //     deserialisation
    // -----------------------------------------------------------------------

    #[test]
    fn output_language_defaults_to_auto() {
        assert_eq!(
            Settings::default().output_language,
            "auto",
            "output_language must default to the sentinel \"auto\" (resolve from locale)"
        );
    }

    #[test]
    fn output_language_round_trips() {
        let original = Settings {
            output_language: "French".to_string(),
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.output_language, "French");
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_output_language_field_defaults_to_auto() {
        // A settings store written before `output_language` existed must
        // deserialise to "auto" (the sentinel that resolves from the host
        // system locale).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert_eq!(
            restored.output_language, "auto",
            "missing output_language must deserialise to \"auto\""
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 2. Default construction — no file → returns defaults
    // -----------------------------------------------------------------------

    #[test]
    fn json_file_store_missing_file_returns_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        // Path does not exist yet.
        let store = JsonFileStore::new(path);
        let loaded = store.load().expect("load");
        assert_eq!(loaded, Settings::default());
    }

    // -----------------------------------------------------------------------
    // 3. Watch emits on update
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn watch_receiver_emits_after_update() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        let store = JsonFileStore::new(path);
        let handle = SettingsHandle::new(store).expect("handle");

        let mut rx = handle.subscribe();
        // The receiver starts with the initial value; mark it as seen.
        rx.borrow_and_update();

        handle
            .update(|s| s.theme = Theme::Light)
            .await
            .expect("update");

        // `changed()` resolves immediately because `update` already sent.
        rx.changed().await.expect("changed");
        let new_settings = rx.borrow().clone();
        assert_eq!(new_settings.theme, Theme::Light);
    }

    #[tokio::test]
    async fn current_reflects_update_with_no_live_subscriber() {
        // Regression: `update` must publish via `send_replace`, not `send`.
        // `watch::Sender::send` is a no-op when there are no live subscribers,
        // which would leave `current()` stale until app restart. Nothing in
        // production holds a subscriber, and the orchestrator reads
        // `current().diarization_enabled` to gate the on-stop diarization pass —
        // so a stale read silently disables the feature.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        let handle = SettingsHandle::new(JsonFileStore::new(path)).expect("handle");

        // No `subscribe()` anywhere — the channel has zero live receivers.
        assert!(!handle.current().diarization_enabled, "default is off");

        handle
            .update(|s| s.diarization_enabled = true)
            .await
            .expect("update");

        assert!(
            handle.current().diarization_enabled,
            "current() must reflect the update even with no live subscriber"
        );

        // A second, unrelated update is also reflected (not just the first).
        handle
            .update(|s| s.theme = Theme::Light)
            .await
            .expect("update");
        assert_eq!(handle.current().theme, Theme::Light);
        assert!(handle.current().diarization_enabled, "prior field retained");
    }

    // -----------------------------------------------------------------------
    // 3b. Persist-before-publish — a failed save must NOT publish the change,
    //     even though the blocking save now runs on `spawn_blocking`.
    // -----------------------------------------------------------------------

    /// A store whose `save` always fails. `load` returns defaults so the
    /// handle constructs cleanly.
    struct FailingSaveStore;

    impl SettingsStore for FailingSaveStore {
        fn load(&self) -> Result<Settings, Error> {
            Ok(Settings::default())
        }

        fn save(&self, _settings: &Settings) -> Result<(), Error> {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "simulated save failure",
            )))
        }
    }

    #[tokio::test]
    async fn failed_save_does_not_publish() {
        let handle = SettingsHandle::new(FailingSaveStore).expect("handle");

        // A live subscriber lets us assert no change was broadcast.
        let mut rx = handle.subscribe();
        rx.borrow_and_update();

        let result = handle.update(|s| s.diarization_enabled = true).await;
        assert!(result.is_err(), "save failure must propagate as an error");

        // The change was NOT published: `current()` (which reads the watch
        // value, kept current by `send_replace`) still reflects the old value.
        assert!(
            !handle.current().diarization_enabled,
            "a failed save must not publish the change (persist-before-publish)"
        );

        // No broadcast fired: `has_changed` is false because the watch value
        // was never replaced.
        assert!(
            !rx.has_changed().expect("sender alive"),
            "a failed save must not notify subscribers"
        );
    }

    // -----------------------------------------------------------------------
    // 3c. Concurrent updates remain serialised / read-modify-write atomic,
    //     even with the save running on the blocking pool.
    // -----------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_updates_are_serialised_and_atomic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        let handle = SettingsHandle::new(JsonFileStore::new(path.clone())).expect("handle");

        // Two independent fields mutated concurrently: each update reads-modifies
        // -writes the whole struct, so without serialisation one would clobber
        // the other. After both complete, both mutations must be present.
        let h1 = handle.clone();
        let h2 = handle.clone();
        let t1 = tokio::spawn(async move {
            h1.update(|s| s.diarization_enabled = true)
                .await
                .expect("update1");
        });
        let t2 = tokio::spawn(async move {
            h2.update(|s| s.theme = Theme::Light)
                .await
                .expect("update2");
        });
        t1.await.expect("join1");
        t2.await.expect("join2");

        let current = handle.current();
        assert!(current.diarization_enabled, "first mutation retained");
        assert_eq!(current.theme, Theme::Light, "second mutation retained");

        // Disk reflects the final state too (last-writer-wins is well defined).
        let loaded = JsonFileStore::new(path).load().expect("reload");
        assert!(loaded.diarization_enabled);
        assert_eq!(loaded.theme, Theme::Light);
    }

    // -----------------------------------------------------------------------
    // 4. Corruption recovery — garbage file → defaults + no panic
    // -----------------------------------------------------------------------

    #[test]
    fn json_file_store_corrupt_file_returns_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        std::fs::write(&path, b"{{{{not valid json}}}").expect("write garbage");

        let store = JsonFileStore::new(path);
        // `load` returns an error for corrupt JSON — SettingsHandle::new
        // catches this specific error and falls back to defaults.
        let result = store.load();
        assert!(
            result.is_err(),
            "corrupt file should return an error from the raw store"
        );
    }

    #[tokio::test]
    async fn handle_new_corrupt_file_falls_back_to_defaults() {
        // Install a no-op tracing subscriber so warn! doesn't panic in tests.
        let _ = tracing_subscriber::fmt::try_init();

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");
        std::fs::write(&path, b"not json at all").expect("write garbage");

        let store = JsonFileStore::new(path);
        let handle = SettingsHandle::new(store).expect("handle despite corrupt file");
        assert_eq!(handle.current(), Settings::default());
    }

    // -----------------------------------------------------------------------
    // 5. Persistence — update writes to disk, reloaded store reflects it
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn update_persists_to_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.store");

        let handle = SettingsHandle::new(JsonFileStore::new(path.clone())).expect("handle");
        handle
            .update(|s| {
                s.theme = Theme::Dark;
                s.start_hidden = true;
            })
            .await
            .expect("update");

        // Open a fresh store at the same path and verify the value is there.
        let store2 = JsonFileStore::new(path);
        let loaded = store2.load().expect("reload");
        assert_eq!(loaded.theme, Theme::Dark);
        assert!(loaded.start_hidden);
    }

    // -----------------------------------------------------------------------
    // 6. Live-agent fields: defaults + round-trip + missing-field
    //    deserialisation (Phase 9 auto-driver, WU1)
    // -----------------------------------------------------------------------

    use minutist_common::{live_agent_should_run, GpuAcceleration, GpuProbe, LiveAgentMode};

    #[test]
    fn live_agent_enabled_defaults_to_off() {
        assert_eq!(
            Settings::default().live_agent_enabled,
            LiveAgentMode::Off,
            "live_agent_enabled must default to Off (the co-pilot is opt-in)"
        );
    }

    #[test]
    fn live_agent_min_segments_defaults_to_eight() {
        assert_eq!(
            Settings::default().live_agent_min_segments,
            8,
            "live_agent_min_segments must default to 8"
        );
    }

    #[test]
    fn live_agent_min_seconds_defaults_to_forty_five() {
        assert_eq!(
            Settings::default().live_agent_min_seconds,
            45,
            "live_agent_min_seconds must default to 45"
        );
    }

    #[test]
    fn live_agent_digest_toggles_default_to_true() {
        let s = Settings::default();
        assert!(s.live_agent_digest_action_items);
        assert!(s.live_agent_digest_decisions);
        assert!(s.live_agent_digest_open_asks);
        assert!(s.live_agent_digest_attachment_answers);
        assert!(s.live_agent_digest_unresolved_references);
    }

    #[test]
    fn live_agent_retrieval_budget_defaults_to_eighty_thousand() {
        assert_eq!(
            Settings::default().live_agent_retrieval_budget_chars,
            80_000,
            "live_agent_retrieval_budget_chars must default to 80_000"
        );
    }

    #[test]
    fn live_agent_retrieval_budget_reads_legacy_attachment_key() {
        // Back-compat: a store written before the rename used the
        // `live_agent_attachment_budget_chars` key; the serde alias must still read it.
        let json = r#"{"live_agent_attachment_budget_chars": 12345}"#;
        let s: Settings = serde_json::from_str(json).expect("deserialise legacy key");
        assert_eq!(
            s.live_agent_retrieval_budget_chars, 12345,
            "serde alias reads the pre-rename key"
        );
    }

    #[test]
    fn live_agent_system_prompt_defaults_to_nonempty_instruction() {
        let prompt = Settings::default().live_agent_system_prompt;
        assert!(
            !prompt.is_empty(),
            "default live-agent prompt must be non-empty"
        );
        // The instruction must emphasise updating the standing list.
        assert!(
            prompt.to_lowercase().contains("update"),
            "default prompt must mention updating the digest"
        );
    }

    #[test]
    fn live_agent_fields_round_trip() {
        let original = Settings {
            live_agent_enabled: LiveAgentMode::Off,
            live_agent_min_segments: 12,
            live_agent_min_seconds: 60,
            live_agent_digest_action_items: true,
            live_agent_digest_decisions: false,
            live_agent_digest_open_asks: true,
            live_agent_digest_attachment_answers: false,
            live_agent_digest_unresolved_references: true,
            live_agent_retrieval_budget_chars: 50_000,
            live_agent_system_prompt: "Custom prompt.".to_string(),
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.live_agent_enabled, LiveAgentMode::Off);
        assert_eq!(restored.live_agent_min_segments, 12);
        assert_eq!(restored.live_agent_min_seconds, 60);
        assert!(!restored.live_agent_digest_decisions);
        assert!(!restored.live_agent_digest_attachment_answers);
        assert_eq!(restored.live_agent_retrieval_budget_chars, 50_000);
        assert_eq!(restored.live_agent_system_prompt, "Custom prompt.");
        assert_eq!(original, restored);
    }

    #[test]
    fn live_agent_enabled_serialises_snake_case() {
        assert_eq!(
            serde_json::to_string(&LiveAgentMode::Auto).unwrap(),
            "\"auto\""
        );
        assert_eq!(serde_json::to_string(&LiveAgentMode::On).unwrap(), "\"on\"");
        assert_eq!(
            serde_json::to_string(&LiveAgentMode::Off).unwrap(),
            "\"off\""
        );
    }

    #[test]
    fn old_store_json_without_live_agent_fields_defaults() {
        // A store written before any live-agent fields existed must deserialise
        // all live-agent fields to their safe defaults.
        let old_json = r#"{ "theme": "dark", "diarization_enabled": true }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert_eq!(
            restored.live_agent_enabled,
            LiveAgentMode::Off,
            "missing live_agent_enabled must deserialise to Off (the safe default)"
        );
        assert_eq!(
            restored.live_agent_min_segments, 8,
            "missing live_agent_min_segments must deserialise to 8"
        );
        assert_eq!(
            restored.live_agent_min_seconds, 45,
            "missing live_agent_min_seconds must deserialise to 45"
        );
        assert!(
            restored.live_agent_digest_action_items,
            "missing live_agent_digest_action_items must deserialise to true"
        );
        assert!(
            restored.live_agent_digest_decisions,
            "missing live_agent_digest_decisions must deserialise to true"
        );
        assert!(
            restored.live_agent_digest_open_asks,
            "missing live_agent_digest_open_asks must deserialise to true"
        );
        assert!(
            restored.live_agent_digest_attachment_answers,
            "missing live_agent_digest_attachment_answers must deserialise to true"
        );
        assert!(
            restored.live_agent_digest_unresolved_references,
            "missing live_agent_digest_unresolved_references must deserialise to true"
        );
        assert_eq!(
            restored.live_agent_retrieval_budget_chars, 80_000,
            "missing live_agent_retrieval_budget_chars must deserialise to 80_000"
        );
        assert_eq!(
            restored.live_agent_system_prompt,
            default_live_agent_system_prompt(),
            "missing live_agent_system_prompt must deserialise to the default instruction"
        );
    }

    // -----------------------------------------------------------------------
    // 7. live_agent_should_run: unit tests for Auto/On/Off × GPU capability
    // -----------------------------------------------------------------------

    fn discrete_probe() -> GpuProbe {
        GpuProbe {
            total_bytes: 24 * 1024 * 1024 * 1024, // 24 GiB discrete
            free_bytes: 20 * 1024 * 1024 * 1024,
            is_integrated: false,
            name: "NVIDIA GeForce RTX 3090".to_string(),
        }
    }

    fn integrated_probe() -> GpuProbe {
        GpuProbe {
            total_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB iGPU (shared RAM)
            free_bytes: 8 * 1024 * 1024 * 1024,
            is_integrated: true,
            name: "AMD Radeon 890M".to_string(),
        }
    }

    #[test]
    fn live_agent_should_run_off_always_false() {
        assert!(!live_agent_should_run(
            LiveAgentMode::Off,
            None,
            GpuAcceleration::Auto
        ));
        assert!(!live_agent_should_run(
            LiveAgentMode::Off,
            Some(&discrete_probe()),
            GpuAcceleration::Auto
        ));
        assert!(!live_agent_should_run(
            LiveAgentMode::Off,
            Some(&integrated_probe()),
            GpuAcceleration::On
        ));
    }

    #[test]
    fn live_agent_should_run_on_always_true() {
        assert!(live_agent_should_run(
            LiveAgentMode::On,
            None,
            GpuAcceleration::Off
        ));
        assert!(live_agent_should_run(
            LiveAgentMode::On,
            Some(&discrete_probe()),
            GpuAcceleration::Auto
        ));
        assert!(live_agent_should_run(
            LiveAgentMode::On,
            Some(&integrated_probe()),
            GpuAcceleration::On
        ));
    }

    #[test]
    fn live_agent_should_run_auto_no_probe_is_false() {
        assert!(
            !live_agent_should_run(LiveAgentMode::Auto, None, GpuAcceleration::Auto),
            "Auto with no probe (CPU-only build) must be false"
        );
    }

    #[test]
    fn live_agent_should_run_auto_discrete_on_integrated_off() {
        // A discrete GPU with acceleration active passes Auto; a shared-memory
        // integrated GPU does NOT (the held context would contend with GPU ASR).
        assert!(
            live_agent_should_run(
                LiveAgentMode::Auto,
                Some(&discrete_probe()),
                GpuAcceleration::Auto
            ),
            "Auto with discrete GPU and accel on must be true"
        );
        assert!(
            !live_agent_should_run(
                LiveAgentMode::Auto,
                Some(&integrated_probe()),
                GpuAcceleration::Auto
            ),
            "Auto with integrated GPU must be false (shared-memory contention)"
        );
    }

    #[test]
    fn live_agent_should_run_auto_accel_off_is_false() {
        assert!(
            !live_agent_should_run(
                LiveAgentMode::Auto,
                Some(&discrete_probe()),
                GpuAcceleration::Off
            ),
            "Auto with accel=Off must be false (LLM would contend with CPU ASR)"
        );
        assert!(
            !live_agent_should_run(
                LiveAgentMode::Auto,
                Some(&integrated_probe()),
                GpuAcceleration::Off
            ),
            "Auto with integrated GPU and accel=Off must be false"
        );
    }
}
