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

/// Default summary system prompt (FR-28).
///
/// A model-agnostic instruction asking for a structured meeting summary:
/// headings, key decisions, and action items. The summariser passes this
/// verbatim as the chat `system` message; users may override it from the
/// settings UI. An older store written before this field existed
/// deserialises to this value via `#[serde(default = ...)]`.
fn default_summary_system_prompt() -> String {
    "You are a meeting-notes assistant. Summarise the meeting transcript and \
     the user's notes into clear Markdown. Use these sections with `##` \
     headings, omitting any that have no content: Summary (a short overview), \
     Key Decisions (a bulleted list of decisions made), and Action Items (a \
     bulleted list of follow-ups, naming the owner when stated). Be concise \
     and factual; do not invent information that is not present in the \
     transcript or notes."
        .to_string()
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

    /// Summary system prompt passed to the summariser (FR-28).
    ///
    /// The `summariser` forwards this verbatim as the chat `system` message
    /// when generating `summary.md`. Defaults to a structured-summary
    /// instruction (headings / key decisions / action items); an older store
    /// written before this field existed deserialises to the default via
    /// `#[serde(default = ...)]`.
    #[serde(default = "default_summary_system_prompt")]
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
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_device_id: None,
            theme: Theme::default(),
            data_directory: None,
            start_hidden: false,
            autosave_interval_secs: default_autosave_interval_secs(),
            summary_system_prompt: default_summary_system_prompt(),
            llm_model_id: None,
            diarization_enabled: false,
            onboarding_completed: false,
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
        };
        let json = serde_json::to_string(&original).expect("serialise");
        let restored: Settings = serde_json::from_str(&json).expect("deserialise");
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
    fn summary_system_prompt_defaults_to_structured_instruction() {
        let prompt = Settings::default().summary_system_prompt;
        assert!(!prompt.is_empty(), "default prompt must be non-empty");
        // The default asks for the three structured sections (FR-28).
        assert!(prompt.contains("Key Decisions"));
        assert!(prompt.contains("Action Items"));
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
        assert_eq!(
            restored.summary_system_prompt,
            default_summary_system_prompt(),
            "missing summary_system_prompt must deserialise to the default"
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
        handle.update(|s| s.theme = Theme::Light).await.expect("update");
        assert_eq!(handle.current().theme, Theme::Light);
        assert!(handle.current().diarization_enabled, "prior field retained");
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
