//! specta-typed mirror types for `ipc-bridge`.
//!
//! # Why this module exists
//!
//! `crates/common` and `crates/settings` do not depend on `specta`, so their
//! shared types do not implement `specta::Type`.  Adding that derive to
//! `common` is an **architecture-owner decision** because:
//!
//! - `common` is the "leaf" crate in the dependency graph; adding `specta`
//!   would pull the whole specta machinery into every crate that already
//!   depends on `common`.
//! - The architecture-owner has not yet approved that change (see
//!   `architecture/domain-ownership.md` — rule 2: "No `common` edits without
//!   architecture review").
//!
//! Until that decision is made, this module provides specta-typed versions of
//! the types that appear in command signatures.  Each mirror type:
//!
//! - Derives `specta::Type` and the serde traits.
//! - Is `#[serde(transparent)]` or has identical field layout so the JSON
//!   wire representation is identical to the `common` type.
//! - Implements `From<CommonType>` / `From<MirrorType>` conversions.
//!
//! **Maintenance note.** If an architecture-owner commit adds `specta::Type`
//! to the `common` and `settings` crates, this entire module and all
//! conversion calls in `commands.rs` can be deleted — the command signatures
//! would switch directly to the `common` / `settings` types.
//!
//! See also `error.rs` for the same pattern applied to `AppError`.

use meeting_app_common::{
    AudioDevice, AudioFormat, AudioMeterFrame, MeetingId, MeetingMeta, ModelDescriptor,
    RecordingState, Segment, WordTimestamp,
};
use serde::{Deserialize, Serialize};
use specta::Type;

// ---------------------------------------------------------------------------
// MeetingIdType
// ---------------------------------------------------------------------------

/// specta-typed mirror of `common::MeetingId`.
///
/// Represented as a transparent `String` because `Uuid` does not implement
/// `specta::Type` in the workspace-pinned version of specta (rc.22 requires
/// the "uuid" feature which would need a workspace Cargo.toml edit).  The
/// JSON wire representation is identical — `Uuid` serialises as a
/// hyphenated lowercase UUID string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(transparent)]
pub struct MeetingIdType(pub String);

impl From<MeetingId> for MeetingIdType {
    fn from(id: MeetingId) -> Self {
        MeetingIdType(id.0.to_string())
    }
}

impl From<MeetingIdType> for MeetingId {
    fn from(t: MeetingIdType) -> Self {
        use std::str::FromStr;
        let uuid = uuid::Uuid::from_str(&t.0).unwrap_or_else(|_| uuid::Uuid::nil());
        MeetingId(uuid)
    }
}

// ---------------------------------------------------------------------------
// AudioDeviceType
// ---------------------------------------------------------------------------

/// specta-typed mirror of `common::AudioDevice`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct AudioDeviceType {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

impl From<AudioDevice> for AudioDeviceType {
    fn from(d: AudioDevice) -> Self {
        AudioDeviceType {
            id: d.id,
            name: d.name,
            is_default: d.is_default,
        }
    }
}

// ---------------------------------------------------------------------------
// RecordingStateType
// ---------------------------------------------------------------------------

/// specta-typed mirror of `common::RecordingState`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecordingStateType {
    Idle,
    Recording {
        meeting_id: MeetingIdType,
        started_at_ms: u64,
    },
    Paused {
        meeting_id: MeetingIdType,
        paused_at_ms: u64,
    },
    Stopping {
        meeting_id: MeetingIdType,
    },
}

impl From<RecordingState> for RecordingStateType {
    fn from(s: RecordingState) -> Self {
        match s {
            RecordingState::Idle => RecordingStateType::Idle,
            RecordingState::Recording {
                meeting_id,
                started_at_ms,
            } => RecordingStateType::Recording {
                meeting_id: meeting_id.into(),
                started_at_ms,
            },
            RecordingState::Paused {
                meeting_id,
                paused_at_ms,
            } => RecordingStateType::Paused {
                meeting_id: meeting_id.into(),
                paused_at_ms,
            },
            RecordingState::Stopping { meeting_id } => RecordingStateType::Stopping {
                meeting_id: meeting_id.into(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// MeetingMetaType
// ---------------------------------------------------------------------------

/// specta-typed mirror of `common::ModelDescriptor`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ModelDescriptorType {
    pub name: String,
    pub quantisation: Option<String>,
    pub version: String,
}

impl From<ModelDescriptor> for ModelDescriptorType {
    fn from(m: ModelDescriptor) -> Self {
        ModelDescriptorType {
            name: m.name,
            quantisation: m.quantisation,
            version: m.version,
        }
    }
}

/// specta-typed mirror of `common::AudioFormat`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AudioFormatType {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub bitrate_kbps: Option<u32>,
}

impl From<AudioFormat> for AudioFormatType {
    fn from(f: AudioFormat) -> Self {
        AudioFormatType {
            codec: f.codec,
            sample_rate: f.sample_rate,
            channels: f.channels,
            bitrate_kbps: f.bitrate_kbps,
        }
    }
}

/// specta-typed mirror of `common::MeetingMeta`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct MeetingMetaType {
    pub uuid: MeetingIdType,
    pub title: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub duration_ms: u64,
    pub speaker_count: u32,
    pub audio_format: AudioFormatType,
    pub asr_model: Option<ModelDescriptorType>,
    pub llm_model: Option<ModelDescriptorType>,
    pub diarizer: Option<ModelDescriptorType>,
    pub app_version: String,
}

impl From<MeetingMeta> for MeetingMetaType {
    fn from(m: MeetingMeta) -> Self {
        MeetingMetaType {
            uuid: m.uuid.into(),
            title: m.title,
            started_at: m.started_at,
            ended_at: m.ended_at,
            duration_ms: m.duration_ms,
            speaker_count: m.speaker_count,
            audio_format: m.audio_format.into(),
            asr_model: m.asr_model.map(Into::into),
            llm_model: m.llm_model.map(Into::into),
            diarizer: m.diarizer.map(Into::into),
            app_version: m.app_version,
        }
    }
}

// ---------------------------------------------------------------------------
// SettingsType
// ---------------------------------------------------------------------------

/// specta-typed mirror of `settings::Theme`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThemeType {
    Light,
    Dark,
    #[default]
    System,
}

impl From<settings::Theme> for ThemeType {
    fn from(t: settings::Theme) -> Self {
        match t {
            settings::Theme::Light => ThemeType::Light,
            settings::Theme::Dark => ThemeType::Dark,
            settings::Theme::System => ThemeType::System,
        }
    }
}

impl From<ThemeType> for settings::Theme {
    fn from(t: ThemeType) -> Self {
        match t {
            ThemeType::Light => settings::Theme::Light,
            ThemeType::Dark => settings::Theme::Dark,
            ThemeType::System => settings::Theme::System,
        }
    }
}

/// specta-typed mirror of `settings::Settings`.
///
/// `data_directory` is mapped to `Option<String>` (PathBuf has no
/// specta::Type impl).  The conversion from `Settings` serialises the path as
/// a UTF-8 string; the reverse parses it back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Default)]
pub struct SettingsType {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_device_id: Option<String>,
    #[serde(default)]
    pub theme: ThemeType,
    /// Serialised form of `Settings::data_directory` (`PathBuf` → `String`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_directory: Option<String>,
    #[serde(default)]
    pub start_hidden: bool,
}

impl From<settings::Settings> for SettingsType {
    fn from(s: settings::Settings) -> Self {
        SettingsType {
            input_device_id: s.input_device_id,
            theme: s.theme.into(),
            data_directory: s.data_directory.map(|p| p.to_string_lossy().into_owned()),
            start_hidden: s.start_hidden,
        }
    }
}

impl From<SettingsType> for settings::Settings {
    fn from(s: SettingsType) -> Self {
        settings::Settings {
            input_device_id: s.input_device_id,
            theme: s.theme.into(),
            data_directory: s.data_directory.map(std::path::PathBuf::from),
            start_hidden: s.start_hidden,
        }
    }
}

// ---------------------------------------------------------------------------
// AppEventPayload helper types
// ---------------------------------------------------------------------------

// These are required because AppEvent embeds types from common that also lack
// specta::Type.  AppEventPayload in events.rs wraps AppEvent; since AppEvent
// is not Type, the wrapper uses serde_json::Value and delegates to the Any
// DataType.  The full typed approach requires common to add specta derives.

/// specta-typed mirror of `common::AudioMeterFrame`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Type)]
pub struct AudioMeterFrameType {
    pub peak: f32,
    pub rms: f32,
}

impl From<AudioMeterFrame> for AudioMeterFrameType {
    fn from(f: AudioMeterFrame) -> Self {
        AudioMeterFrameType {
            peak: f.peak,
            rms: f.rms,
        }
    }
}

/// specta-typed mirror of `common::WordTimestamp`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct WordTimestampType {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

impl From<WordTimestamp> for WordTimestampType {
    fn from(w: WordTimestamp) -> Self {
        WordTimestampType {
            start_ms: w.start_ms,
            end_ms: w.end_ms,
            text: w.text,
        }
    }
}

/// specta-typed mirror of `common::Segment`.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SegmentType {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordTimestampType>,
}

impl From<Segment> for SegmentType {
    fn from(s: Segment) -> Self {
        SegmentType {
            start_ms: s.start_ms,
            end_ms: s.end_ms,
            text: s.text,
            speaker_id: s.speaker_id,
            confidence: s.confidence,
            words: s.words.into_iter().map(Into::into).collect(),
        }
    }
}

/// specta-typed mirror of `common::AppEvent`.
///
/// This is the fully-typed version of the event payload, using the mirror
/// types above.  The `AppEventPayload` wrapper in `events.rs` serialises
/// `AppEvent` via `serde` and then deserialises into this type's shape;
/// the JSON wire representations are identical because both are derived from
/// the same serde attributes.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AppEventType {
    AudioMeter {
        frame: AudioMeterFrameType,
    },
    DevicesChanged,
    StateChanged {
        state: RecordingStateType,
    },
    TranscriptSegment {
        meeting_id: MeetingIdType,
        segment: SegmentType,
    },
    DiarizationComplete {
        meeting_id: MeetingIdType,
        speaker_count: u32,
    },
    SummaryReady {
        meeting_id: MeetingIdType,
    },
    ModelDownloadProgress {
        model_id: String,
        bytes_done: u64,
        bytes_total: Option<u64>,
    },
    SettingsChanged,
    ErrorOccurred {
        error: IpcErrorType,
    },
}

/// specta-typed mirror of `common::AppError` for embedding in events.
///
/// Identical shape to `error::IpcError` but kept separate to avoid a
/// circular dependency in the specta type graph (IpcError appears in command
/// results; IpcErrorType appears inside the AppEventType::ErrorOccurred
/// variant).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum IpcErrorType {
    Io { context: String },
    ModelLoad { model_id: String, context: String },
    ModelNotFound { model_id: String },
    ModelDownload { context: String },
    Inference { backend: String, context: String },
    InvalidInput { context: String },
    Cancelled,
    Unsupported { context: String },
    Internal { context: String },
}
