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

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Stable identifier for a meeting on disk. UUIDv4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MeetingId(pub Uuid);

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

/// Stable identifier for a model in the registry.
///
/// Examples: `"qwen3-asr-1.7b-q8_0"`, `"qwen2.5-3b-instruct-q4_k_m"`,
/// `"silero-vad-v4"`, `"sherpa-pyannote-segmentation-3-0"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
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
pub struct AudioMeterFrame {
    pub peak: f32,
    pub rms: f32,
}

/// Audio-file format descriptor captured at write time. Phase 1 writes
/// Opus 16 kHz mono; downstream phases re-decode using these fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitrate_kbps: Option<u32>,
}

// ---------------------------------------------------------------------------
// Meeting metadata
// ---------------------------------------------------------------------------

/// Per-meeting metadata persisted as `metadata.json`.
///
/// Timestamps are ISO 8601 strings to avoid pulling `chrono` into `common`.
/// Consumers parse as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub app_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub name: String,
    pub quantisation: Option<String>,
    pub version: String,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
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
    /// Diarization finished assigning speakers to a meeting's segments.
    DiarizationComplete {
        meeting_id: MeetingId,
        speaker_count: u32,
    },
    /// Summary generation finished; `summary.md` now exists for this meeting.
    SummaryReady { meeting_id: MeetingId },
    /// Model download progress, used by the first-run flow.
    ModelDownloadProgress {
        model_id: ModelId,
        bytes_done: u64,
        bytes_total: Option<u64>,
    },
    /// User-visible settings changed; subscribers should re-read.
    SettingsChanged,
    /// A recoverable error occurred during a background task. The pipeline
    /// continues; the webview shows a notification.
    ErrorOccurred { error: AppError },
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
pub trait Summariser: Send {
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
            app_version: "0.0.0".to_string(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: MeetingMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.audio_format.codec, "opus");
        assert_eq!(back.audio_format.sample_rate, 16_000);
        assert_eq!(back.audio_format.channels, 1);
        assert_eq!(back.audio_format.bitrate_kbps, Some(32));
    }
}
