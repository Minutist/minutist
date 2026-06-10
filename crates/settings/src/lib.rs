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

use meeting_app_common::ModelId;
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
const fn default_gpu_acceleration() -> bool {
    true
}

/// Default for `capture_system_audio`: ON. Capturing the call audio is the point
/// of a meeting-notes app, so it defaults on (opt-out); the UI advises turning
/// it off when the mic also hears the call from the speakers (echo).
const fn default_capture_system_audio() -> bool {
    true
}

/// Default for `prefer_large_asr_model`: OFF. The larger Qwen3-ASR-1.7B tier is
/// opt-in (bigger download, GPU-class footprint); the 0.6B CPU model is the
/// default. See `architecture/cross-cutting.md` — "ASR engine routing".
const fn default_prefer_large_asr_model() -> bool {
    false
}

/// Default MCP server port (Phase 10 — D1). A FIXED loopback port: only one app
/// instance/window runs, so a fixed default avoids the ephemeral-port friction
/// of a per-run URL change in an external MCP client's config. User-editable in
/// the MCP settings pane. An older store written before this field existed
/// deserialises to 8765 via `#[serde(default = ...)]`.
const fn default_mcp_port() -> u16 {
    8765
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
    /// default app-data directory" (resolved by `app-main`).
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

    /// Whether the first-run onboarding flow has been completed (Phase 7).
    ///
    /// The webview gates the main UI behind this: `false` shows the onboarding
    /// flow (welcome → model download → settings tour), which sets it `true` on
    /// completion. `#[serde(default)]` defaults to `false`; an older store
    /// written before this field existed deserialises to `false` (so existing
    /// users see onboarding once on upgrade — acceptable for a pre-release).
    #[serde(default)]
    pub onboarding_completed: bool,

    /// Whether GPU acceleration is used at runtime when the build supports it.
    ///
    /// GPU offload happens ONLY when BOTH (a) the build was compiled with a GPU
    /// feature (`vulkan`/`metal`/`cuda`/`rocm`) AND (b) this setting is `true`.
    /// When `false`, inference runs on CPU (`n_gpu_layers = 0`) even in a
    /// GPU-feature build — the runtime escape hatch for weak GPUs / driver
    /// trouble. `#[serde(default = ...)]` defaults to `true` (GPU on); an older
    /// store written before this field existed deserialises to `true`. In a
    /// default CPU-only build the flag has no effect (inference is always on
    /// CPU). See `architecture/cross-cutting.md` — "GPU portability".
    #[serde(default = "default_gpu_acceleration")]
    pub gpu_acceleration: bool,

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

    /// Opt into the larger Qwen3-ASR-1.7B tier for the Qwen branch (broader +
    /// better-multilingual accuracy) instead of the 0.6B CPU default. Only
    /// affects languages that route to Qwen (the Parakeet branch ignores it);
    /// see `common::asr_engine_for_language`. Off by default — it is a larger
    /// download with a GPU-class footprint. An older store deserialises to
    /// `false`. See `architecture/cross-cutting.md` — "ASR engine routing".
    #[serde(default = "default_prefer_large_asr_model")]
    pub prefer_large_asr_model: bool,

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
    /// (`set_speaker_name`, `rename_meeting`) join. `retranscribe_meeting` /
    /// `rediarize_meeting` / any destructive tool stay internal-only regardless
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
            prefer_large_asr_model: default_prefer_large_asr_model(),
            notes_paper_rules: default_notes_paper_rules(),
            chat_system_prompt: default_chat_system_prompt(),
            summary_preset: SummaryPreset::default(),
            mcp_enabled: false,
            mcp_port: default_mcp_port(),
            mcp_write_tools: false,
            auto_summarise_on_stop: default_auto_summarise_on_stop(),
            preload_summariser: default_preload_summariser(),
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
            gpu_acceleration: false,
            capture_system_audio: true,
            transcription_language: "Japanese".to_string(),
            prefer_large_asr_model: true,
            notes_paper_rules: false,
            chat_system_prompt: "Be a terse assistant.".to_string(),
            summary_preset: SummaryPreset::ActionItems,
            mcp_enabled: true,
            mcp_port: 9999,
            mcp_write_tools: true,
            auto_summarise_on_stop: false,
            preload_summariser: false,
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored.transcription_language, "Japanese");
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
    // 1f. gpu_acceleration: default + round-trip + missing-field
    //     deserialisation (runtime GPU toggle)
    // -----------------------------------------------------------------------

    #[test]
    fn gpu_acceleration_defaults_to_true() {
        assert!(
            Settings::default().gpu_acceleration,
            "GPU acceleration is on by default; a GPU-feature build offloads, a \
             CPU-only build ignores the flag (llama.cpp falls back to CPU)"
        );
    }

    #[test]
    fn gpu_acceleration_round_trips() {
        let original = Settings {
            gpu_acceleration: false,
            ..Settings::default()
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
        assert!(!restored.gpu_acceleration);
        assert_eq!(original, restored);
    }

    #[test]
    fn old_store_json_without_gpu_acceleration_field_defaults_to_true() {
        // A settings store written before `gpu_acceleration` existed must
        // deserialise to `true`, preserving the prior compile-time behaviour
        // (a GPU build offloaded by default).
        let old_json = r#"{ "theme": "dark", "start_hidden": true, "autosave_interval_secs": 5 }"#;
        let restored: Settings = serde_json::from_str(old_json).expect("deserialise old store");
        assert!(
            restored.gpu_acceleration,
            "missing gpu_acceleration must deserialise to true (GPU on by default)"
        );
        assert_eq!(restored.theme, Theme::Dark);
        assert!(restored.start_hidden);
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
}
