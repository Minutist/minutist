//! Shared interface types and trait definitions for minutist.
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

/// Pure, dependency-free vector math for voiceprint centroids.
///
/// Shared by `diarizer` (centroid building during extraction) and `persistence`
/// (fold/merge/recompute inside `VoiceprintStore`) without introducing a
/// `persistence → diarizer` edge. Both crates already depend on `common`.
pub mod voiceprint_math;

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

/// Stable identifier for a collection — a user-facing "folder" that groups
/// meetings. UUIDv4. Mirrors [`MeetingId`].
///
/// Distinct from `persistence::MeetingFolder`, which is a single meeting's
/// on-disk directory. A meeting belongs to at most one collection
/// ([`MeetingMeta::collection_id`]); the collection's definition (name, order)
/// lives in `{app-data}/collections.json` (owned by `persistence`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct CollectionId(#[cfg_attr(feature = "specta", specta(type = String))] pub Uuid);

impl CollectionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for CollectionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identifier for a meeting attachment on disk. UUIDv4. Mirrors
/// [`MeetingId`].
///
/// Carried on [`AttachmentEntry`] and the four attachment [`AppEvent`]
/// variants so the webview store can route adds / conversions / removes to
/// the right row without a re-list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct AttachmentId(#[cfg_attr(feature = "specta", specta(type = String))] pub Uuid);

impl AttachmentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AttachmentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identifier for a speaker identity in the voiceprint library. UUIDv4.
/// Mirrors [`MeetingId`].
///
/// A `VoiceprintIdentityId` is assigned once on first enrolment and survives
/// renames and merges — it is the stable primary key for the
/// `voiceprint_identity` table in `{app-data}/voiceprints.db` (owned by
/// `persistence`). Never placed in `Segment`; the diarizer label-to-name
/// resolution at read time uses display names, not this id.
///
/// Adding this type is a one-way-door architecture-owner change
/// (see `architecture/domain-ownership.md` — Parallel-work rules §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct VoiceprintIdentityId(
    // Use `#[specta(type = String)]` so the TS binding mirrors how serde
    // emits a Uuid (a hyphenated lowercase string) without needing the
    // optional `uuid` feature on the `specta` crate.
    #[cfg_attr(feature = "specta", specta(type = String))] pub Uuid,
);

impl VoiceprintIdentityId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VoiceprintIdentityId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identifier for one acquisition-condition centroid within a
/// voiceprint identity's gallery. UUIDv4. Mirrors [`MeetingId`].
///
/// A `VoiceprintCentroidId` is assigned once when a new condition gallery
/// entry is created and persisted in the `voiceprint_centroid` table in
/// `{app-data}/voiceprints.db` (owned by `persistence`). One identity can
/// hold several centroid entries — one per distinct recording condition
/// (e.g. in-person room mic vs VoIP). Matching runs over the flattened
/// gallery: every centroid of every identity, with an identity's score being
/// the maximum over its centroids.
///
/// Adding this type is a one-way-door architecture-owner change
/// (see `architecture/domain-ownership.md` — Parallel-work rules §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct VoiceprintCentroidId(
    // Use `#[specta(type = String)]` so the TS binding mirrors how serde
    // emits a Uuid (a hyphenated lowercase string) without needing the
    // optional `uuid` feature on the `specta` crate.
    #[cfg_attr(feature = "specta", specta(type = String))] pub Uuid,
);

impl VoiceprintCentroidId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for VoiceprintCentroidId {
    fn default() -> Self {
        Self::new()
    }
}

/// One uncertain-band voiceprint suggestion emitted on
/// [`AppEvent::VoiceprintSuggestions`] (§2.4 — `T_reject <= sim < T_accept`).
///
/// Carries everything the UI needs to show "is this \<display_name\>?" and to
/// confirm (`set_speaker_name`) or dismiss (`reject_match`) the suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct VoiceprintSuggestion {
    /// The diariser letter that received the uncertain match (e.g. `"A"`).
    pub label: String,
    /// Display name of the matched identity.
    pub display_name: String,
    /// Stable identity id — passed to `reject_match` when the user dismisses.
    pub identity_id: VoiceprintIdentityId,
    /// The embedding model id — also passed to `reject_match`.
    pub model_id: String,
    /// The cosine similarity (for optional display; not used by the matching
    /// logic on the UI side).
    pub similarity: f32,
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
    /// Additional speaker labels (beyond `speaker_id`) that also speak
    /// substantially within this segment's time span — set by the offline
    /// diarization pass (#0002) when a segment overlaps more than one surviving
    /// speaker turn above the share threshold AND the segment cannot be split
    /// (the no-word-timestamp / Qwen path). A mixed segment that carries per-word
    /// timestamps is instead split at the turn boundary (#0015) into
    /// single-speaker segments, each with empty `shared_speakers`. Each label is
    /// one that appears as a `speaker_id` elsewhere in the transcript. Empty for
    /// the common single-speaker case and for live / un-diarized / re-transcribed
    /// segments. Presentation only — a "multiple speakers" hint on an unsplit
    /// segment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shared_speakers: Vec<String>,
}

/// Optional per-word timestamp data when the ASR model supports it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct WordTimestamp {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// Apply the `speaker_names` overlay to a transcript: rewrite each segment's
/// `speaker_id` label to its configured display name where one exists. Labels
/// without a configured name are left as-is. Presentation-only — operates on a
/// caller-owned copy; the on-disk transcript is never mutated.
///
/// The single canonical overlay used by every read path that renders speaker
/// labels (the agent tools, the summariser input, and any future consumer).
pub fn apply_speaker_overlay(
    segments: &mut [Segment],
    speaker_names: &std::collections::BTreeMap<String, String>,
) {
    if speaker_names.is_empty() {
        return;
    }
    for seg in segments.iter_mut() {
        if let Some(label) = &seg.speaker_id {
            if let Some(name) = speaker_names.get(label) {
                seg.speaker_id = Some(name.clone());
            }
        }
    }
}

/// One note paragraph handed to the summariser (#70).
///
/// `at_ms` is `Some` when the paragraph was anchored to the recording clock:
/// the notes editor stamps `data-anchor-ms` on the first keystroke into a
/// paragraph while recording, on the SAME pause-excluding timeline as
/// [`Segment::start_ms`] (fed by `AppEvent::RecordingClock`). The summariser
/// weaves anchored paragraphs into the transcript at their timestamp so the
/// model sees each note beside what was being said when it was written.
/// `None` paragraphs (typed while idle, pasted, or written before recording
/// started) carry no time and render as a trailing block.
///
/// `text` is the paragraph's plain text (descendant text nodes concatenated);
/// markdown formatting from the editor is intentionally not preserved — the
/// summary prompt needs the words, not the styling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteBlock {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_ms: Option<u64>,
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
// Processing lifecycle + host election
// ---------------------------------------------------------------------------

/// Opaque key naming the host (device) that holds a meeting's processing
/// claim. An opaque string keeps `iroh` out of `common`: `sync` maps it
/// from/to its `iroh::EndpointId` at the wire boundary. Mirrors [`ModelId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[cfg_attr(feature = "specta", specta(transparent))]
pub struct HostRef(pub String);

/// A durable, syncable claim that one host owns a meeting's processing (ASR /
/// diarization / summarisation) and is the authoritative producer of its
/// derived outputs.
///
/// Distinct from the in-memory single-device offline slot in `orchestrator`
/// (`claim_offline`), which guards one device against two concurrent offline
/// ops and is never persisted or synced. This claim is persisted in
/// `metadata.json` and propagated to peers, so it can act as a cross-device
/// lock. Carried inside [`ProcessingLifecycle::Claimed`].
///
/// Timestamps are RFC 3339 UTC strings, matching `MeetingMeta::started_at`'s
/// format (no `chrono` in `common`). They drive lease/reap timing only and are
/// NOT used to resolve racing claims — cross-device clock skew makes them
/// unreliable, so the tiebreak is the lowest `HostRef` (decided in `sync`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ProcessingClaim {
    /// The host that owns this claim.
    pub host: HostRef,
    /// When the claim was taken (RFC 3339 UTC).
    pub claimed_at: String,
    /// Past this instant (RFC 3339 UTC) with no transition to
    /// [`ProcessingLifecycle::Processed`], any peer may reap the claim and
    /// re-elect — recovering a crashed host that cannot release its own slot.
    pub lease_expires_at: String,
}

/// Where a meeting sits in the capture → processing → processed pipeline,
/// persisted on [`MeetingMeta`] (`metadata.json`) and propagated to peers.
///
/// `processing` is **derived and host-authoritative**: the host that holds the
/// claim authors it; consumers never write it — they receive it over the
/// lifecycle sync exchange and self-heal their local copy. It is NOT part of
/// the user-editable metadata that folds into the notes-CRDT.
///
/// `#[serde(default)] = Local` so existing `metadata.json` written before this
/// field existed reads as a locally-recorded-and-processed meeting (today's
/// only path) — no migration. The same defaulted-field pattern `speaker_names`
/// / `notes_format` use.
///
/// One shape serves two roles without naming a device type: a phone or a
/// GPU-less desktop is the capture device (writes `PendingProcessing`); a
/// desktop or the headless GPU hub is the processing host (claims → produces
/// outputs). See `planning/DESIGN_processing-lifecycle.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "state", rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum ProcessingLifecycle {
    /// Recorded AND processed on this device — today's only path, and the
    /// back-compat default for metadata written before the field existed.
    #[default]
    Local,
    /// Captured on a device that does not process it locally, and offered for
    /// an eligible host to adopt. The capture device sets this at finalise
    /// instead of running the pipeline.
    PendingProcessing,
    /// A host has claimed the meeting and is producing its derived outputs.
    /// Peers observing this neither claim nor process it.
    Claimed { claim: ProcessingClaim },
    /// Processing finished; the derived outputs are authoritative. `at` is the
    /// RFC 3339 UTC completion time.
    Processed { processed_by: HostRef, at: String },
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
    /// Notes-document storage format for this meeting (O2 notes-CRDT
    /// groundwork). `0` = "JSON only, pre-CRDT": `notes.json` is authoritative
    /// and no `notes.ydoc` exists. `1` = "Yjs authoritative, projections
    /// derived": `notes.ydoc` is the source of truth and `notes.json` /
    /// `notes.md` are derived from it on every save (see
    /// `planning/DESIGN_notes-crdt.md` D-O2.7). The lazy on-open seed flips a
    /// `0` meeting to `1` the first time it is opened under a build that carries
    /// the CRDT groundwork.
    ///
    /// `#[serde(default)]` so existing `metadata.json` (written before the field
    /// existed) reads as `0` — the same defaulted-field pattern `speaker_names`
    /// used.
    #[serde(default)]
    pub notes_format: u8,
    /// The collection (user-facing "folder") this meeting belongs to, if any.
    /// `None` = unfiled. Authoritative here in `metadata.json`; the `index.db`
    /// `collection_id` column is a derived mirror for filtered listing, so a
    /// `rebuild_from_disk` reconstructs membership from this field.
    ///
    /// `#[serde(default, skip_serializing_if = …)]` so existing `metadata.json`
    /// (written before the field existed) reads as `None` and the wire shape only
    /// grows when set — the same defaulted-field pattern `speaker_names` /
    /// `notes_format` use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection_id: Option<CollectionId>,
    /// Where this meeting sits in the capture → processing → processed
    /// pipeline. Host-authoritative and self-healing — a consumer never
    /// authors it; the sync lifecycle exchange propagates the source host's
    /// state. `#[serde(default)] = Local` so existing `metadata.json` (written
    /// before the field existed) reads as a locally-processed meeting, the same
    /// defaulted-field pattern `notes_format` uses. See
    /// `planning/DESIGN_processing-lifecycle.md`.
    #[serde(default)]
    pub processing: ProcessingLifecycle,
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
    /// The collection (user-facing "folder") this meeting belongs to, if any —
    /// a derived mirror of [`MeetingMeta::collection_id`] for filtered listing.
    /// `None` = unfiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection_id: Option<CollectionId>,
}

/// A user-facing "folder" that groups meetings (UI label: "Folders").
///
/// A meeting belongs to at most one collection ([`MeetingMeta::collection_id`]);
/// collections are a flat list ordered by `position`. The authoritative
/// definitions live in `{app-data}/collections.json` (owned by `persistence`);
/// `index.db` carries only a derived `collection_id` column on each meeting row
/// for fast filtered listing. Distinct from `persistence::MeetingFolder`, which
/// is a single meeting's on-disk directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct Collection {
    pub id: CollectionId,
    pub name: String,
    /// Sort position in the flat folder list (ascending). Assigned at creation.
    pub position: u32,
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

/// One tool call an assistant message requested, in the persisted/wire shape.
///
/// Mirrors `chat-agent`'s engine-internal `ToolCall` (`id` / `name` /
/// `arguments_json` — the repo's "arguments cross as a String, not a `Value`"
/// rule). Carried on an `Assistant` [`ChatMessage`] so a reloaded session
/// reconstructs the `assistant(tool_calls) → tool(result)` exchange the GGUF
/// tool template requires (CQ1) instead of an orphan `tool` message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ToolCallRecord {
    /// The OpenAI tool-call id the matching `Tool` message answers.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The tool arguments as a JSON-object string.
    pub arguments_json: String,
}

/// One persisted chat message (the wire / on-disk shape).
///
/// Distinct from `chat-agent`'s engine-internal message: this is the durable,
/// specta-typed record the webview renders and `persistence::ChatStore`
/// serialises. `turn_id` is the per-session monotonic turn counter the streaming
/// chat `AppEvent`s also carry, so the UI can correlate a stored message with
/// the deltas it saw live. `tool_name` is present only on `Tool` messages (the
/// name of the tool whose result the message carries). `tool_calls` is present
/// only on an `Assistant` message that requested tools — the carrier that keeps
/// a reloaded multi-tool turn a valid OpenAI sequence (CQ1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// For a `Tool` message: the tool whose result this message carries.
    /// `None` for system/user/assistant messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// For a `Tool` message: the OpenAI tool-call id this result answers — the
    /// id of the matching entry in the preceding assistant message's
    /// `tool_calls` (CQ1). Persisted so a reloaded turn re-links each tool
    /// result to its call rather than synthesising an id that no longer matches
    /// the assistant `tool_calls`. `None` for non-tool messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For an `Assistant` message: the tool calls it requested (CQ1). Empty for
    /// every other message and for a plain assistant free-text reply. Defaulted
    /// so an older on-disk session (written before this field existed) reloads.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallRecord>,
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
// Attachments
// ---------------------------------------------------------------------------

/// The conversion state of a meeting attachment. Stored on [`AttachmentEntry`]
/// and updated by the bounded background conversion worker in `ipc-bridge`.
///
/// `Failed` carries a concise human string the attachments pane shows on the
/// row. The `state`/`reason` serde tagging mirrors `AppError`'s `code`/`context`
/// shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "reason", rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum ConversionState {
    /// The original has been stored; the background converter has not yet
    /// processed it.
    Pending,
    /// `<hash>.md` exists alongside the original; the attachment is ready for
    /// the summariser feed.
    Ready,
    /// Conversion finished with an error (best-effort; never crashes the worker).
    /// The string is a concise human-readable reason shown in the pane.
    Failed(String),
}

/// One persisted attachment row for a meeting.
///
/// `persistence::attachments` stores a `Vec<AttachmentEntry>` as
/// `attachments/attachments.json` (atomic tmp+rename); the IPC attachment
/// commands load/save it. Content-addressed originals live at
/// `attachments/<hash>.<ext>`; converted markdown siblings at
/// `attachments/<hash>.md`.
///
/// `hash` is hex-encoded SHA-256 of the original bytes — the dedup key shared
/// with the on-disk filenames. `byte_len` is the original's size in bytes
/// (matches existing `u64` wire fields). `added_at` is RFC 3339 (UTC), mirroring
/// `ChatSession::created_at`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct AttachmentEntry {
    pub id: AttachmentId,
    /// Hex-encoded SHA-256 of the original bytes. Shared with the on-disk
    /// `<hash>.<ext>` original and `<hash>.md` sibling; used for dedup-safe
    /// remove.
    pub hash: String,
    /// The user-visible original filename (e.g. `"Q2_report.xlsx"`).
    pub original_filename: String,
    /// Lower-cased, dot-less extension (e.g. `"xlsx"`, `"pdf"`).
    pub ext: String,
    /// Size of the original file in bytes.
    pub byte_len: u64,
    /// RFC 3339 timestamp (UTC) when the attachment was added, mirroring
    /// `ChatSession::created_at`.
    pub added_at: String,
    /// Whether `doc-convert` has processed this attachment.
    pub conversion: ConversionState,
    /// The filename of the converted markdown sibling (`"<hash>.md"`), present
    /// once `conversion` is [`ConversionState::Ready`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converted_md_filename: Option<String>,
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
// Live in-meeting agent (Phase 9 auto-driver)
// ---------------------------------------------------------------------------

/// One item in a live digest category (action item, decision, open ask, etc.).
///
/// `resolved` carries the standing-list state: `false` = outstanding, `true` =
/// resolved / answered. The live agent updates this flag across digest refreshes
/// rather than regenerating the list from scratch, so once an action item is
/// marked resolved it stays resolved even as new segments arrive. `source` is an
/// optional short attribution string (e.g. `"from slide deck"` for attachment-
/// sourced answers) shown in the panel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct LiveDigestItem {
    /// The item text (plain English; one sentence).
    pub text: String,
    /// `true` once the item is resolved, answered, or confirmed; `false` while
    /// it is still outstanding. Carried forward across refreshes so the
    /// standing list accumulates without full regeneration.
    pub resolved: bool,
    /// Optional short attribution (e.g. `"slide deck"`, `"Alice"`). Absent for
    /// most items; present when the source is worth surfacing in the panel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// The full live digest payload produced by the live agent on each refresh.
///
/// Each category is a `Vec<LiveDigestItem>` with a `resolved` flag that the
/// agent carries forward across refreshes (the 'asked-for-but-missed' tracker
/// pattern — the list accumulates and is marked resolved rather than being
/// regenerated wholesale). `generated_at_ms` is wall-clock epoch milliseconds
/// for display; `meeting_id` scopes the digest to one meeting.
///
/// Crosses IPC (derives `specta::Type`, serialises snake_case) and rides the
/// existing `AppEventPayload` + `collect_events![AppEventPayload]` channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct LiveDigest {
    pub meeting_id: MeetingId,
    /// Wall-clock epoch milliseconds when this digest was generated.
    pub generated_at_ms: u64,
    /// Tasks or follow-ups explicitly requested or implied during the meeting.
    pub action_items: Vec<LiveDigestItem>,
    /// Commitments, conclusions, or choices reached during the meeting.
    pub decisions: Vec<LiveDigestItem>,
    /// Questions posed during the meeting that have not yet received an answer.
    pub open_asks: Vec<LiveDigestItem>,
    /// Questions answered from pinned attachment context (documents, slides,
    /// etc. attached before the meeting started).
    pub attachment_answers: Vec<LiveDigestItem>,
    /// Terms, acronyms, or references mentioned but not explained in the
    /// transcript (potential knowledge gaps surfaced for the attendee).
    pub unresolved_references: Vec<LiveDigestItem>,
}

/// Whether the live in-meeting agent runs during an active recording.
///
/// `Auto` (the default) enables when GPU acceleration is ACTIVE: a usable GPU
/// is present (probe is `Some`) AND `gpu_acceleration != Off`. This ensures the
/// live agent's `LlamaContext` (n_ctx = 32 768) runs on the GPU and does not
/// contend with the CPU-bound ASR path. On the AMD Radeon 890M (integrated,
/// Vulkan on) with `gpu_acceleration = Auto`, this resolves `true` — the
/// validated SP-LIVE hardware.
///
/// This is **distinct from** [`GpuAcceleration`], which governs model-layer
/// placement (GPU vs CPU). `LiveAgentMode::Auto` means "run iff a GPU is active";
/// `GpuAcceleration::Auto` means "offload layers iff they fit in the VRAM budget".
///
/// Serialises as snake_case (`"auto"` / `"on"` / `"off"`) to match the
/// established `GpuAcceleration` pattern and the TypeScript binding shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum LiveAgentMode {
    /// Enable the live agent when GPU acceleration is active: a usable GPU is
    /// present (`probe.is_some()`) AND `gpu_acceleration != Off`. This is the
    /// recommended default: users with a GPU (integrated or discrete) running
    /// with acceleration on get the feature automatically; users on CPU-only
    /// builds or with acceleration forced off are not affected.
    #[default]
    Auto,
    /// Always enable the live agent regardless of GPU capability. Use when the
    /// user explicitly wants the feature even on a slow host (slower refreshes
    /// are acceptable trade-off).
    On,
    /// Permanently disable the live agent regardless of GPU capability.
    Off,
}

/// Decide whether the live agent should run given the user's mode preference,
/// the GPU probe result, and the GPU-acceleration setting.
///
/// PURE — takes all inputs as parameters so it is unit-testable without a GPU.
///
/// - `Off` → always `false`.
/// - `On` → always `true` (user override; no capability check).
/// - `Auto` → `true` iff `probe` is `Some` AND `gpu_acceleration != Off`.
///   This is a **GPU-acceleration-active proxy**: a usable GPU exists and the
///   user has not forced CPU mode. The live agent's `LlamaContext` then runs on
///   the GPU, off the CPU-bound ASR path.
///   Does NOT inspect `probe.is_integrated` — the AMD 890M (integrated, Vulkan
///   on) is the validated SP-LIVE hardware and must resolve `true`.
///   Does NOT invoke `resolve_gpu_plan` or inspect VRAM bytes. WU2b should
///   refine this to a VRAM-headroom check once the held-context cost is measured.
pub fn live_agent_should_run(
    mode: LiveAgentMode,
    probe: Option<&GpuProbe>,
    gpu_acceleration: GpuAcceleration,
) -> bool {
    match mode {
        LiveAgentMode::Off => false,
        LiveAgentMode::On => true,
        LiveAgentMode::Auto => probe.is_some() && gpu_acceleration != GpuAcceleration::Off,
    }
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

/// Which long-running operation an [`AppEvent::OperationProgress`] event reports.
///
/// Determinate (a `fraction` is available): `ReTranscribe` (samples processed /
/// total kept samples), `Summarise` (tokens generated / max tokens), and
/// `Translate` (segments translated / total segments).
/// Indeterminate (`fraction = None`, one opaque FFI compute call):
/// `Rediarize` and the `Finalise` drain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum OperationKind {
    /// Offline re-transcribe over the full `audio.opus` (determinate).
    ReTranscribe,
    /// LLM summary generation (determinate; tokens / max-tokens).
    Summarise,
    /// Speaker (re-)identification — internally "diarization" (indeterminate).
    Rediarize,
    /// The post-stop finalise drain (indeterminate).
    Finalise,
    /// Post-hoc translation of transcript segments (determinate; segments translated
    /// / total segments). Emitted by `ipc-bridge`'s `translate_meeting` command.
    Translate,
}

/// The connected-tier relay tunnel's live state, surfaced to the Settings →
/// Connection pane (WS4-A S5b). A pure status enum — it carries no credential or
/// account material (those cross only the pairing command's return / secure
/// storage, never the event bus). The connector channel transits meeting content
/// to the AI vendor by design and is never described as private (D5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum TunnelStatus {
    /// No device credential stored, or the connector is disabled. The app works
    /// fully locally; nothing is reachable over the relay.
    Disconnected,
    /// A device-code pairing is in progress (the user is approving in a browser).
    Pairing,
    /// A credential is stored and the tunnel is dialing / handshaking the relay.
    Connecting,
    /// The tunnel is established — the account is reachable over the relay.
    Online,
    /// The stored credential was rejected after having worked: the device was
    /// revoked (or rotated). The user must re-pair; the loop is not retrying.
    NeedsRepair,
}

/// The peer-to-peer notes-sync engine's live state, surfaced to the UI (WS4-B
/// S5). A pure status enum carrying no peer ticket or device-key material —
/// those cross only the sync commands' return values / secure storage, never the
/// event bus. The sync channel is end-to-end between the user's own paired
/// devices (D4); it is distinct from the connector channel ([`TunnelStatus`]),
/// which transits content to the AI vendor by design.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum SyncStatus {
    /// Sync is unavailable in this build (the free build) or disabled. Nothing
    /// is dialled and no inbound connections are accepted.
    Disabled,
    /// The sync engine is up with at least one paired peer, but no transfer is
    /// in flight — waiting for a change to push or a peer to dial.
    Idle,
    /// Dialling / handshaking a peer before a transfer.
    Connecting,
    /// A notes-sync transfer is in progress with a peer.
    Syncing,
    /// The sync engine hit a terminal error. `message` is a stable
    /// human-readable string the UI surfaces; the engine is not retrying.
    Error { message: String },
}

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
    /// A post-stop AUTOMATIC summary has been scheduled for this meeting. Emitted
    /// by `ipc-bridge`'s `stop_recording` only when `auto_summarise_on_stop` is on,
    /// the instant the meeting is finalised — BEFORE any transcript-repair /
    /// diarize pass it must wait for. The webview shows the summary pane as busy
    /// for the whole queued → summarising window instead of the manual
    /// "Summarise" affordance. A terminal `SummaryReady` (a summary was written)
    /// or `SummaryUnavailable` (deferred / failed) clears the busy state.
    SummaryQueued { meeting_id: MeetingId },
    /// A post-stop automatic summary will NOT produce a summary: it was deferred
    /// (a new recording started and claimed the model) or it failed. The webview
    /// clears the queued/busy state and falls back to the manual `Summarise`
    /// action. Distinct from `SummaryReady`, which means `summary.md` now exists.
    SummaryUnavailable { meeting_id: MeetingId },
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
    /// A long-running per-meeting operation made progress. The webview renders a
    /// per-meeting-row indicator: a determinate bar when `fraction` is `Some`
    /// (0.0..=1.0), an indeterminate spinner when `None`. Emitted throttled by
    /// the producing op; a terminal `DiarizationComplete` / `TranscriptReady` /
    /// `SummaryReady` clears the indicator. Rides the existing `AppEventPayload`
    /// tag machinery — no second event registration. See
    /// `architecture/cross-cutting.md` — "Operation progress".
    OperationProgress {
        meeting_id: MeetingId,
        op: OperationKind,
        /// `Some(f)` with `0.0 <= f <= 1.0` for a determinate bar; `None` for an
        /// indeterminate spinner (the op is one opaque FFI call with no progress
        /// callback, e.g. sherpa re-diarization).
        fraction: Option<f32>,
        /// A short human-readable label for the in-flight op (e.g.
        /// "Re-transcribing…", "Summarising…", "Identifying speakers…").
        label: String,
    },
    /// A post-hoc translation pass finished for a meeting. Emitted by
    /// `ipc-bridge`'s `translate_meeting` command once all segments have been
    /// translated and merged into `translations.json`. The webview re-reads the
    /// translations for the open meeting when this fires so the translated view
    /// reflects the new language without a manual refresh.
    TranslationReady {
        meeting_id: MeetingId,
        /// The full English language name the translation was produced for
        /// (e.g. `"Spanish"`). The webview uses this to refresh only the
        /// relevant language cache.
        language: String,
    },
    // --- Attachments ---------------------------------------------------------
    // These ride the existing `AppEventPayload` newtype + the single
    // `collect_events![AppEventPayload]` registration in `ipc-bridge` — no new
    // event registration needed.
    /// A new attachment was added to a meeting (original stored + manifest row
    /// written with `conversion: Pending`). The webview inserts the row without
    /// a re-list.
    AttachmentAdded {
        meeting_id: MeetingId,
        attachment: AttachmentEntry,
    },
    /// An attachment's background conversion finished; `<hash>.md` now exists
    /// and the manifest row is `Ready`. The webview flips the row to Ready.
    AttachmentConverted {
        meeting_id: MeetingId,
        attachment_id: AttachmentId,
    },
    /// An attachment's background conversion failed (best-effort; never crashes
    /// the worker). `reason` is a concise human string the pane shows on the row.
    AttachmentConversionFailed {
        meeting_id: MeetingId,
        attachment_id: AttachmentId,
        reason: String,
    },
    /// An attachment was removed (manifest row dropped; hash files unlinked iff
    /// no other row shares the hash). The webview drops the row.
    AttachmentRemoved {
        meeting_id: MeetingId,
        attachment_id: AttachmentId,
    },

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
    /// The driver's sliding window evicted the oldest non-pinned messages to
    /// keep the conversation within the context budget (Phase 9).
    /// `dropped_turns` is the number of history messages dropped (snapped to a
    /// user-message group boundary, CQ2). The webview shows a "history trimmed"
    /// affordance so the user knows older turns no longer inform the agent. The
    /// pinned system head (turn 0) is never evicted.
    ChatContextTrimmed {
        session_id: ChatSessionId,
        dropped_turns: u32,
    },

    // --- Live agent ----------------------------------------------------------
    // These ride the existing `AppEventPayload` newtype + the single
    // `collect_events![AppEventPayload]` registration in `ipc-bridge` — no new
    // event registration needed. The live agent refreshes the digest on a
    // debounced cadence driven by `TranscriptSegment` events; each refresh
    // emits a full replacement digest so a lagged subscriber can safely drop
    // intermediate updates (lossy-broadcast-safe, same approach as
    // `ChatTurnComplete.final_text`).
    /// The live in-meeting agent produced a refreshed digest for an active
    /// meeting. The payload is the FULL replacement digest; the webview
    /// replaces the previous digest wholesale rather than patching. Lossy-
    /// broadcast-safe: a dropped event is recovered on the next refresh.
    LiveDigestUpdated {
        meeting_id: MeetingId,
        digest: LiveDigest,
    },
    /// The live agent encountered an error producing a digest refresh. The
    /// panel shows `message` and retains the last valid digest, if any.
    LiveDigestError {
        meeting_id: MeetingId,
        /// A concise human-readable description of what went wrong.
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
    /// The in-process MCP server stopped (disabled via settings toggle).
    /// `app-main` emits this after the accept loop has fully exited so the
    /// Settings → MCP pane can clear the live endpoint display.
    McpServerStopped,
    /// The MCP server failed to start (bind error or summariser load failure).
    /// `app-main` emits this so the Settings → MCP pane can drop the
    /// "starting…" hint and surface the reason instead of spinning indefinitely.
    /// The `reason` field is a concise human-readable English string; it is not
    /// a serialised `AppError` variant (the pane shows it verbatim).
    McpServerStartFailed { reason: String },

    // --- Connected-tier relay tunnel (WS4-A S5b) -------------------------
    /// The relay tunnel's live state changed. `app-main` (connected build only)
    /// emits this from the reconnect loop's state callback and the pairing /
    /// lifecycle transitions, so the Settings → Connection pane reflects the live
    /// status without polling. Carries no credential / account material — the
    /// account label and pairing codes cross only the pairing command's return
    /// value, never the bus.
    TunnelStatusChanged { status: TunnelStatus },

    // --- Peer-to-peer notes sync (WS4-B S5) ------------------------------
    // These ride the existing `AppEventPayload` newtype + the single
    // `collect_events![AppEventPayload]` registration in `ipc-bridge` — no new
    // event registration. They carry no peer ticket / device-key material; the
    // shareable ticket crosses only the `sync_get_my_ticket` command's return.
    /// A notes-sync transfer for a meeting made progress. The UI renders a
    /// per-meeting indicator: a determinate bar when `fraction` is `Some`
    /// (0.0..=1.0), an indeterminate spinner when `None`. A terminal
    /// `SyncReady` / `SyncError` clears it.
    SyncProgress {
        meeting_id: MeetingId,
        /// A short human-readable label for the in-flight transfer (e.g.
        /// "Syncing notes…").
        label: String,
        /// `Some(f)` with `0.0 <= f <= 1.0` for a determinate bar; `None` for an
        /// indeterminate spinner.
        fraction: Option<f32>,
    },
    /// A notes-sync transfer for a meeting finished and the local copy is now
    /// merged. The UI re-reads the meeting's notes so the synced content
    /// surfaces without a manual refresh.
    SyncReady { meeting_id: MeetingId },
    /// A notes-sync operation failed. `context` is a stable human-readable
    /// string the UI surfaces in a notification; the sync continues for other
    /// meetings.
    SyncError { context: String },

    // --- Voiceprint matching (issue #0003 — WU5) --------------------------
    // Rides the existing `AppEventPayload` newtype + the single
    // `collect_events![AppEventPayload]` registration in `ipc-bridge` — no new
    // event registration.
    /// One or more diarizer labels landed in the uncertain match band
    /// (`T_reject <= sim < T_accept` — §2.4) after a diarisation pass. The
    /// webview shows an "is this \<name\>?" affordance for each suggestion so the
    /// user can confirm or dismiss. A confirmed suggestion applies the name via
    /// `set_speaker_name`; a dismissal calls `reject_match`. Unconfirmed
    /// suggestions leave the label as the bare diariser letter.
    ///
    /// Only emitted when `voiceprint_enrolment_enabled` is ON and the gallery
    /// returns at least one uncertain-band candidate.
    VoiceprintSuggestions {
        meeting_id: MeetingId,
        /// Each suggestion carries the diariser label, the suggested name, the
        /// matched identity id, the model id (needed by `reject_match`), and
        /// the cosine similarity for display.
        suggestions: Vec<VoiceprintSuggestion>,
    },
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

// ---------------------------------------------------------------------------
// GPU acceleration mode + the VRAM-aware offload plan
// ---------------------------------------------------------------------------

/// A snapshot of the GPU device a model would be offloaded to.
///
/// `total_bytes` / `free_bytes` are the ggml-reported device memory. NOTE: a
/// Vulkan device without `VK_EXT_memory_budget` reports `free == total` (the
/// heap size), so `free` is optimistic there; [`resolve_gpu_plan`] decides on
/// `total` and uses `free` only to TIGHTEN. `is_integrated` marks a
/// shared-system-RAM GPU (budgeted far more conservatively). Plain data (no
/// llama dependency) so the plan + its tests build in a CPU-only configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProbe {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub is_integrated: bool,
    /// Human-readable device name (e.g. "NVIDIA GeForce RTX 3080").
    pub name: String,
}

/// Probe the GPU device a model would be offloaded to, or `None` when there is
/// no usable GPU (CPU-only build, no driver, or the enumeration finds none).
///
/// Prefers the first discrete `Gpu`; falls back to an `IntegratedGpu` only when
/// no discrete GPU exists (flagged `is_integrated`). Multi-GPU is out of scope —
/// one host device is chosen, never accumulated. The real probe lives behind the
/// `llama-backend` feature (it queries the same ggml backend that loads the
/// GGUFs); a CPU-only build has no GPU and returns `None`.
#[cfg(feature = "llama-backend")]
pub fn probe_primary_gpu() -> Option<GpuProbe> {
    let devices = llama_backend::list_gpu_devices();
    // Discrete GPU first, else the integrated one.
    devices
        .iter()
        .find(|d| !d.is_integrated)
        .or_else(|| devices.first())
        .cloned()
}

/// CPU-only build (no `llama-backend` feature): there is never a GPU to probe.
#[cfg(not(feature = "llama-backend"))]
pub fn probe_primary_gpu() -> Option<GpuProbe> {
    None
}

/// GPU-acceleration mode (replaces the old `gpu_acceleration: bool`).
///
/// `Auto` (the default) probes GPU VRAM at each model load and offloads a model
/// to the GPU only when it fits, falling back to CPU otherwise. `On`/`Off` are
/// hard overrides that NEVER consult the probe — `On` forces full GPU offload
/// (the old `true`), `Off` forces CPU (the old `false`). Only effective in a
/// build compiled with a GPU feature (`vulkan`/`metal`/…); a CPU-only build
/// always runs on CPU regardless. See `architecture/cross-cutting.md` — "GPU
/// portability".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub enum GpuAcceleration {
    /// Probe VRAM per model load; GPU iff it fits, else CPU. The new default.
    #[default]
    Auto,
    /// Force GPU offload (no probe) — the old `gpu_acceleration = true`.
    On,
    /// Force CPU (no probe) — the old `gpu_acceleration = false`.
    Off,
}

/// The resolved per-model GPU-offload decision for one model-load moment.
///
/// Produced by [`resolve_gpu_plan`]. Binary per model (whole model on GPU or on
/// CPU): partial layer offload is slower than CPU for models this small, and the
/// existing `n_gpu_layers` resolution is already binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPlan {
    /// Offload the summariser / chat LLM to the GPU.
    pub summariser_gpu: bool,
    /// Offload the (Qwen) ASR model to the GPU. Moot for the Parakeet engine,
    /// which runs on its own ONNX provider, not llama.cpp.
    pub asr_gpu: bool,
    /// The ASR tier to actually use: the large tier only when the VRAM budget
    /// allows it alongside whatever is already placed on GPU, else downgraded to
    /// the small tier. Running the 1.7B model purely on CPU is strictly worse
    /// than the 0.6B CPU default, so the clamp applies for `Auto` and `On` alike.
    pub effective_prefer_large: bool,
}

const GPU_PLAN_GIB: u64 = 1024 * 1024 * 1024;
/// VRAM to host the summariser (Gemma-4-E4B Q4_K_M ~5.3 GB weights + KV @ 32K +
/// compute headroom). ESTIMATE — validate against the live probe log.
const SUMMARISER_VRAM_BYTES: u64 = 8 * GPU_PLAN_GIB;
/// VRAM to host the small ASR tier (Qwen3-ASR-0.6B Q8_0).
const ASR_SMALL_VRAM_BYTES: u64 = 2 * GPU_PLAN_GIB;
/// VRAM to host the large ASR tier (Qwen3-ASR-1.7B Q8_0).
const ASR_LARGE_VRAM_BYTES: u64 = 7 * GPU_PLAN_GIB / 2; // 3.5 GiB
/// Usable fraction of a DISCRETE GPU's memory (fragmentation + co-tenant slack).
const DISCRETE_HEADROOM: f64 = 0.90;
/// Usable fraction of an INTEGRATED GPU's "memory" (it is shared system RAM;
/// budget far more conservatively). A guess — flagged for live validation.
const IGPU_HEADROOM: f64 = 0.50;

/// Compute the usable VRAM budget from a probe.
///
/// Applies the discrete/iGPU headroom fraction to `total_bytes`, then tightens
/// by `free_bytes` when it is a credible smaller number (> 0 and < total).
/// Vulkan devices without `VK_EXT_memory_budget` report `free == total`, and a
/// bogus 0 is likewise ignored; in both cases the total-based budget is used.
fn probe_budget(p: &GpuProbe) -> u64 {
    let headroom = if p.is_integrated {
        IGPU_HEADROOM
    } else {
        DISCRETE_HEADROOM
    };
    let total_budget = (p.total_bytes as f64 * headroom) as u64;
    if p.free_bytes > 0 && p.free_bytes < p.total_bytes {
        total_budget.min((p.free_bytes as f64 * headroom) as u64)
    } else {
        total_budget
    }
}

/// Return `true` when the large ASR tier fits in `asr_headroom`.
///
/// `asr_headroom` is the VRAM remaining after any already-budgeted models have
/// been deducted (the caller is responsible for the deduction). Short-circuits
/// to `false` when `prefer_large` is false so the caller can pass the flag
/// directly without an outer `if`.
fn large_asr_fits(asr_headroom: u64, prefer_large: bool) -> bool {
    prefer_large && asr_headroom >= ASR_LARGE_VRAM_BYTES
}

/// Resolve the per-model GPU-offload plan from a VRAM probe + the user's mode.
///
/// PURE (the probe is an input) so it is unit-tested without a GPU. `Off`
/// short-circuits without consulting the probe. `Auto` and `On` both apply the
/// VRAM clamp: the large ASR tier is requested unconditionally (the VRAM guard
/// decides whether it fits). `Auto` budgets the summariser FIRST (it is resident
/// while an ASR model loads when `preload_summariser` is on), then budgets ASR
/// against the REMAINING headroom and downgrades the large ASR tier if it would
/// not fit alongside. `On` applies the same clamp so an explicit override cannot
/// trigger an out-of-memory load when the large model does not fit. A `None`
/// probe (no GPU / probe failed) resolves everything to CPU — the fail-safe.
///
/// VRAM decision base: `total × headroom`, NOT `free` — a Vulkan device without
/// `VK_EXT_memory_budget` reports `free == total`, so `free` is trusted only to
/// TIGHTEN the budget when it is a credible smaller number. See
/// `architecture/components.md` — "`common` VRAM-aware GPU placement".
pub fn resolve_gpu_plan(
    probe: Option<&GpuProbe>,
    mode: GpuAcceleration,
    prefer_large_asr: bool,
) -> GpuPlan {
    match mode {
        GpuAcceleration::On => {
            // Force GPU on for both models, but still apply the VRAM clamp for
            // the large ASR tier — a no-probe `On` cannot know whether the 1.7B
            // fits, so it falls back to the small tier.
            let effective_prefer_large = probe.is_some_and(|p| {
                // `On` forces the summariser on GPU; deduct its cost before
                // checking whether the large ASR tier also fits.
                let asr_headroom = probe_budget(p).saturating_sub(SUMMARISER_VRAM_BYTES);
                large_asr_fits(asr_headroom, prefer_large_asr)
            });
            GpuPlan {
                summariser_gpu: true,
                asr_gpu: true,
                effective_prefer_large,
            }
        }
        GpuAcceleration::Off => GpuPlan {
            summariser_gpu: false,
            asr_gpu: false,
            effective_prefer_large: false,
        },
        GpuAcceleration::Auto => {
            let Some(p) = probe else {
                return GpuPlan {
                    summariser_gpu: false,
                    asr_gpu: false,
                    effective_prefer_large: false,
                };
            };
            let budget = probe_budget(p);
            let summariser_gpu = budget >= SUMMARISER_VRAM_BYTES;
            // When the summariser fits on GPU it is deducted from the budget
            // before the ASR decision; when it runs on CPU the full budget is
            // available for ASR.
            let asr_headroom = budget.saturating_sub(if summariser_gpu {
                SUMMARISER_VRAM_BYTES
            } else {
                0
            });
            let effective_prefer_large = large_asr_fits(asr_headroom, prefer_large_asr);
            let asr_need = if effective_prefer_large {
                ASR_LARGE_VRAM_BYTES
            } else {
                ASR_SMALL_VRAM_BYTES
            };
            let asr_gpu = asr_headroom >= asr_need;
            GpuPlan {
                summariser_gpu,
                asr_gpu,
                effective_prefer_large,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostic report (issue #0014 — crash capture + "Report a problem")
// ---------------------------------------------------------------------------

/// A redacted diagnostic snapshot for the user-driven "Report a problem" flow.
///
/// Assembled and **redacted** by `ipc-bridge` (`get_diagnostic_report`), it
/// crosses the IPC boundary so the webview can pre-fill a GitHub issue form the
/// user reviews and submits from their own browser. There is **no telemetry**:
/// nothing leaves the machine except by the user's explicit browser action.
///
/// Privacy by construction: this carries only structured environment fields plus
/// an already-redacted log excerpt / backtrace. There is **no field for meeting
/// content** (transcripts, notes, titles, speaker names). The producer redacts
/// meeting-id paths out of `log_excerpt` / `backtrace`; this type cannot carry
/// raw meeting text because no such field exists.
///
/// snake_case fields map onto the camelCase `DiagnosticReport` TS shape in
/// `ui/src/diagnostics/issueReport.ts` (tauri-specta camelCases the binding).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct DiagnosticReport {
    /// Application version (e.g. `"0.0.0"`).
    pub app_version: String,
    /// OS / arch / build, e.g. `"Windows 11 / x86_64 / connected"`.
    pub platform: String,
    /// Resolved GPU plan (backend or CPU fallback), best-effort.
    pub gpu: String,
    /// Short error class, e.g. `"panic"` or `"diagnostic report"`.
    pub error_class: String,
    /// Recent log lines, already redacted (meeting-id paths stripped).
    pub log_excerpt: String,
    /// Backtrace from the last captured crash, already redacted; absent when no
    /// crash report is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backtrace: Option<String>,
}

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
    /// Assign `speaker_id` to each segment by clustering speaker embeddings
    /// extracted from `audio` over each segment's `[start_ms, end_ms]` window,
    /// returning the (possibly longer) segment list plus the distinct-speaker
    /// count.
    ///
    /// `audio` is the entire buffered recording at `sample_rate` Hz. `segments`
    /// is the ASR output for the same recording, consumed by value: a segment
    /// spanning a speaker change is split at the turn boundary, which GROWS the
    /// list — an in-place `&mut [Segment]` slice cannot add elements, so the
    /// owned list is taken in and returned out.
    ///
    /// Returns `(segments, speaker_count)`.
    fn assign_speakers(
        &self,
        audio: &[f32],
        sample_rate: u32,
        segments: Vec<Segment>,
    ) -> AppResult<(Vec<Segment>, u32)>;
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
    /// Produce a markdown summary from a transcript, the user's notes, and any
    /// attachment reference material.
    ///
    /// `notes` are the note paragraphs in document order (#70). Anchored
    /// paragraphs (`NoteBlock::at_ms == Some`) are woven into the transcript at
    /// their recording-clock timestamp; un-anchored paragraphs render as a
    /// trailing block. An empty slice means no notes were taken.
    ///
    /// `attachments_markdown` is a pre-assembled block containing every `Ready`
    /// attachment's converted markdown, each under a `## Attachment: <name>`
    /// heading, rendered as a leading `# Reference material (attachments)`
    /// section — NOT time-woven. Pass `""` when there are no ready attachments;
    /// an empty string produces byte-identical output to the no-attachment path.
    /// The caller (ipc-bridge) is responsible for assembling and deterministically
    /// truncating this string to fit within the context-window budget.
    ///
    /// `system_prompt` is the user-configured prompt from settings.
    fn summarise(
        &self,
        transcript: &[Segment],
        notes: &[NoteBlock],
        attachments_markdown: &str,
        system_prompt: &str,
    ) -> AppResult<String>;
}

/// Injected vision-OCR seam used by `doc-convert` to handle direct image
/// attachments (`png`/`jpg`/`jpeg`/`tiff`). The seam is deliberately generic
/// over PNG input, so a future PDF-page rasterisation path (planning issue
/// 0019) can reuse it without a trait change.
///
/// `doc-convert` is a `common`-only leaf — it carries no workspace edge to
/// `summariser` or `ipc-bridge`. Callers (currently `ipc-bridge`) construct a
/// concrete implementation backed by the held Gemma-4 `LlamaModel` + a lazily
/// initialised `MtmdContext`, then pass it into `convert_to_markdown`. The trait
/// is `Option<&dyn DocVlm>` at every call site so the pure-Rust path works
/// without any model present.
///
/// `Send + Sync`: implementations hold either an `Arc`-wrapped model (safe with
/// `unsafe impl`) or a `Mutex`-guarded stub (safe by construction). No
/// `LlamaContext` or `MtmdContext` is stored across calls — each call builds one
/// fresh, runs inference, and drops it before returning.
pub trait DocVlm: Send + Sync {
    /// Convert a single PNG image (one document page, or a direct image
    /// attachment) to clean markdown text via vision inference.
    ///
    /// `png` is a complete, valid PNG byte slice. The implementation is
    /// responsible for encoding the image into the format expected by the
    /// underlying model (e.g. llama.cpp `MtmdImage`).
    ///
    /// Returns `AppError::Unsupported` when no model is available or the
    /// underlying model does not support vision.
    fn image_to_markdown(&self, png: &[u8]) -> AppResult<String>;
}

/// Injected text-embedding seam used by `rag-retrieval` to turn chunk and query
/// text into vectors for cosine ranking, without `rag-retrieval` ever loading a
/// model. The concrete BGE-M3 / llama-backed implementation lives in `ipc-bridge`
/// (which owns the model substrate); `rag-retrieval` depends only on this trait.
/// Sits alongside [`Summariser`] / [`DocVlm`] as a model-backed inference seam.
///
/// `Send + Sync`: implementations hold an `Arc`-wrapped model (safe with
/// `unsafe impl`) and build their `!Sync` embeddings `LlamaContext` fresh per
/// call (never stored), so the seam can be shared across the retrieval threads
/// (`spawn_blocking`) that embed chunks at attach time and the query at ask time.
pub trait Embedder: Send + Sync {
    /// Embed `text` into a fixed-dimension, **L2-normalised** vector of length
    /// [`Self::dim`].
    ///
    /// Implementations MUST return unit-length vectors so the retrieval cosine
    /// reduces to a dot product. Used at attach time to embed chunks and at query
    /// time to embed the question.
    fn embed(&self, text: &str) -> AppResult<Vec<f32>>;

    /// The embedding dimensionality (e.g. 1024 for BGE-M3).
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_report_serde_shape_and_no_meeting_field() {
        // snake_case fields, and `backtrace: None` is omitted from the wire so
        // a non-crash report carries no `backtrace` key. There is no field for
        // meeting content by construction.
        let r = DiagnosticReport {
            app_version: "0.0.0".to_string(),
            platform: "Linux / x86_64 / connected".to_string(),
            gpu: "cpu".to_string(),
            error_class: "diagnostic report".to_string(),
            log_excerpt: "INFO app-main: started".to_string(),
            backtrace: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"app_version\""));
        assert!(json.contains("\"log_excerpt\""));
        assert!(json.contains("\"error_class\""));
        // Omitted when absent.
        assert!(
            !json.contains("backtrace"),
            "absent backtrace must be omitted: {json}"
        );
        // Present when set.
        let with_bt = DiagnosticReport {
            backtrace: Some("0: minutist::panic".to_string()),
            ..r.clone()
        };
        let json = serde_json::to_string(&with_bt).unwrap();
        assert!(json.contains("\"backtrace\""));
        let back: DiagnosticReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, with_bt);
    }

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

    // -----------------------------------------------------------------------
    // resolve_gpu_plan — pure, no GPU needed (probe is an input)
    // -----------------------------------------------------------------------

    fn probe(total_gib: u64, integrated: bool) -> GpuProbe {
        GpuProbe {
            total_bytes: total_gib * GPU_PLAN_GIB,
            free_bytes: total_gib * GPU_PLAN_GIB, // Vulkan-style free == total
            is_integrated: integrated,
            name: "test-gpu".to_string(),
        }
    }

    #[test]
    fn gpu_plan_on_forces_gpu_on_clamps_large_asr_without_probe() {
        // `On` with no probe: GPU is forced for both models, but the large ASR
        // tier is clamped to false (no probe → cannot confirm the 1.7B fits).
        let on = resolve_gpu_plan(None, GpuAcceleration::On, true);
        assert!(
            on.summariser_gpu && on.asr_gpu,
            "GPU forced for both models"
        );
        assert!(
            !on.effective_prefer_large,
            "no probe → large ASR clamped off"
        );
    }

    #[test]
    fn gpu_plan_on_with_large_card_promotes_large_asr() {
        // `On` with a 24 GB discrete card: the large ASR tier (3.5 GiB) fits
        // after the summariser (8 GiB) is deducted from the budget.
        let on = resolve_gpu_plan(Some(&probe(24, false)), GpuAcceleration::On, true);
        assert!(on.summariser_gpu && on.asr_gpu && on.effective_prefer_large);
    }

    #[test]
    fn gpu_plan_on_with_small_card_clamps_large_asr() {
        // `On` with a 12 GB card: 12 × 0.90 = 10.8 GiB; summariser (8) leaves
        // 2.8 — below the large ASR threshold (3.5), so clamped to small tier.
        let on = resolve_gpu_plan(Some(&probe(12, false)), GpuAcceleration::On, true);
        assert!(
            on.summariser_gpu && on.asr_gpu,
            "GPU forced for both models"
        );
        assert!(
            !on.effective_prefer_large,
            "large ASR clamped on 12 GB card"
        );
    }

    #[test]
    fn gpu_plan_off_forces_cpu_regardless_of_probe() {
        let off = resolve_gpu_plan(Some(&probe(64, false)), GpuAcceleration::Off, true);
        assert!(!off.summariser_gpu && !off.asr_gpu && !off.effective_prefer_large);
    }

    #[test]
    fn gpu_plan_auto_no_gpu_falls_back_to_cpu() {
        let p = resolve_gpu_plan(None, GpuAcceleration::Auto, true);
        assert!(!p.summariser_gpu && !p.asr_gpu && !p.effective_prefer_large);
    }

    #[test]
    fn gpu_plan_auto_large_card_runs_everything_on_gpu() {
        // 24 GB discrete: summariser (8) + large ASR (3.5) fit easily.
        let p = resolve_gpu_plan(Some(&probe(24, false)), GpuAcceleration::Auto, true);
        assert!(p.summariser_gpu && p.asr_gpu && p.effective_prefer_large);
    }

    #[test]
    fn gpu_plan_auto_eight_gb_card_runs_asr_on_gpu_but_summariser_on_cpu() {
        // 8 GB × 0.90 = 7.2 GiB budget < 8 GiB summariser need -> summariser CPU;
        // the full budget is then free for ASR, so the large tier fits.
        let p = resolve_gpu_plan(Some(&probe(8, false)), GpuAcceleration::Auto, true);
        assert!(!p.summariser_gpu, "8 GB cannot host the summariser");
        assert!(p.asr_gpu, "ASR fits when the summariser is on CPU");
        assert!(
            p.effective_prefer_large,
            "7.2 GiB headroom fits the large ASR"
        );
    }

    #[test]
    fn gpu_plan_auto_downgrades_large_asr_when_it_wont_fit_beside_summariser() {
        // 12 GB × 0.90 = 10.8 budget; summariser takes 8 -> 2.8 left: large (3.5)
        // does not fit, small (2.0) does, so downgrade but keep ASR on GPU.
        let p = resolve_gpu_plan(Some(&probe(12, false)), GpuAcceleration::Auto, true);
        assert!(p.summariser_gpu);
        assert!(p.asr_gpu);
        assert!(
            !p.effective_prefer_large,
            "large ASR downgraded to fit beside summariser"
        );
    }

    #[test]
    fn gpu_plan_auto_free_tightens_the_budget() {
        // total 24 GB but only 4 GB free (a budget-aware device under load):
        // 4 × 0.90 = 3.6 < 8 -> summariser CPU; small ASR (2.0) still fits.
        let mut p = probe(24, false);
        p.free_bytes = 4 * GPU_PLAN_GIB;
        let plan = resolve_gpu_plan(Some(&p), GpuAcceleration::Auto, false);
        assert!(
            !plan.summariser_gpu,
            "low free VRAM tightens the budget below the summariser need"
        );
        assert!(plan.asr_gpu);
    }

    #[test]
    fn gpu_plan_auto_integrated_gpu_budgets_conservatively() {
        // iGPU reports 16 GB shared RAM but the 0.50 cap = 8 GiB budget: exactly
        // the summariser need, nothing left for ASR on GPU.
        let p = resolve_gpu_plan(Some(&probe(16, true)), GpuAcceleration::Auto, true);
        assert!(p.summariser_gpu);
        assert!(!p.asr_gpu, "iGPU 0.50 cap leaves no ASR headroom");
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
            shared_speakers: Vec::new(),
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
            notes_format: 0,
            processing: Default::default(),
            collection_id: None,
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
    fn processing_lifecycle_defaults_to_local() {
        assert_eq!(ProcessingLifecycle::default(), ProcessingLifecycle::Local);
    }

    #[test]
    fn processing_lifecycle_serde_round_trips_each_variant() {
        let claim = ProcessingClaim {
            host: HostRef("endpoint-abc".to_string()),
            claimed_at: "2026-06-27T10:00:00Z".to_string(),
            lease_expires_at: "2026-06-27T10:30:00Z".to_string(),
        };
        for state in [
            ProcessingLifecycle::Local,
            ProcessingLifecycle::PendingProcessing,
            ProcessingLifecycle::Claimed {
                claim: claim.clone(),
            },
            ProcessingLifecycle::Processed {
                processed_by: HostRef("endpoint-abc".to_string()),
                at: "2026-06-27T10:30:00Z".to_string(),
            },
        ] {
            let json = serde_json::to_string(&state).unwrap();
            let back: ProcessingLifecycle = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
        }

        // Internally tagged on `state`, snake_case — mirrors `ModelStatusState`.
        let pending =
            serde_json::to_string(&ProcessingLifecycle::PendingProcessing).unwrap();
        assert_eq!(pending, r#"{"state":"pending_processing"}"#);
    }

    #[test]
    fn meeting_meta_without_processing_field_reads_as_local() {
        // A `metadata.json` written before the field existed must deserialise as
        // a locally-processed meeting — the `#[serde(default)]` no-migration
        // contract (DESIGN_processing-lifecycle.md §7 Q4).
        let legacy = r#"{
            "uuid": "00000000-0000-4000-8000-000000000000",
            "title": "Legacy meeting",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": null,
            "duration_ms": 0,
            "speaker_count": 0,
            "audio_format": {"codec":"opus","sample_rate":16000,"channels":1,"bitrate_kbps":32},
            "asr_model": null,
            "llm_model": null,
            "diarizer": null,
            "app_version": "0.1.0"
        }"#;
        let meta: MeetingMeta = serde_json::from_str(legacy).unwrap();
        assert_eq!(meta.processing, ProcessingLifecycle::Local);
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
            collection_id: None,
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
            notes_format: 0,
            processing: Default::default(),
            collection_id: None,
            app_version: "0.0.0".to_string(),
        };
        let segment = Segment {
            start_ms: 100,
            end_ms: 2_000,
            text: "hello world".to_string(),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
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
        // The `notes` field is `Option` with `skip_serializing_if`, so it is
        // omitted when `None`. Match the `"notes":` key precisely — a bare
        // substring search would false-positive on `meta.notes_format`, which
        // legitimately serialises and contains "notes".
        assert!(!json.contains("\"notes\":"), "absent notes must be omitted");
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
    fn app_event_chat_context_trimmed_serialises_with_tag() {
        let e = AppEvent::ChatContextTrimmed {
            session_id: ChatSessionId::new(),
            dropped_turns: 4,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"chat_context_trimmed\""));
        assert!(json.contains("\"dropped_turns\":4"));
        match serde_json::from_str::<AppEvent>(&json).unwrap() {
            AppEvent::ChatContextTrimmed { dropped_turns, .. } => {
                assert_eq!(dropped_turns, 4);
            }
            other => panic!("unexpected event: {other:?}"),
        }
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
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    turn_id: 0,
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "what were the action items?".to_string(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: String::new(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: vec![ToolCallRecord {
                        id: "call_1".to_string(),
                        name: "get_transcript".to_string(),
                        arguments_json: "{}".to_string(),
                    }],
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    content: "{\"segments\":[]}".to_string(),
                    tool_name: Some("get_transcript".to_string()),
                    tool_call_id: Some("call_1".to_string()),
                    tool_calls: Vec::new(),
                    turn_id: 1,
                },
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: "the action items were …".to_string(),
                    tool_name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
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
        // The assistant-tool_calls message carries the OpenAI tool-call carrier
        // (CQ1) so a reloaded multi-tool turn is a valid sequence.
        assert!(json.contains("\"tool_calls\""));
        assert!(json.contains("\"arguments_json\":\"{}\""));
        let back: ChatSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back, session);
        assert_eq!(back.messages.len(), 5);
        assert!(back.messages[0].tool_name.is_none());
        // index 2 = assistant(tool_calls), index 3 = the matching tool result.
        assert_eq!(back.messages[2].role, ChatRole::Assistant);
        assert_eq!(back.messages[2].tool_calls.len(), 1);
        assert_eq!(back.messages[2].tool_calls[0].id, "call_1");
        assert_eq!(
            back.messages[3].tool_name.as_deref(),
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

    // -----------------------------------------------------------------------
    // live_agent_should_run — Off/On/Auto × probe/accel variants
    // -----------------------------------------------------------------------

    fn make_gpu_probe(total_gib: u64, is_integrated: bool) -> GpuProbe {
        GpuProbe {
            total_bytes: total_gib * 1024 * 1024 * 1024,
            free_bytes: total_gib * 1024 * 1024 * 1024,
            is_integrated,
            name: if is_integrated {
                "Test Integrated GPU".to_string()
            } else {
                "Test Discrete GPU".to_string()
            },
        }
    }

    #[test]
    fn live_agent_off_always_returns_false() {
        assert!(!live_agent_should_run(
            LiveAgentMode::Off,
            None,
            GpuAcceleration::Auto
        ));
        assert!(!live_agent_should_run(
            LiveAgentMode::Off,
            Some(&make_gpu_probe(24, false)),
            GpuAcceleration::Auto
        ));
        assert!(!live_agent_should_run(
            LiveAgentMode::Off,
            Some(&make_gpu_probe(8, true)),
            GpuAcceleration::On
        ));
    }

    #[test]
    fn live_agent_on_always_returns_true() {
        assert!(live_agent_should_run(
            LiveAgentMode::On,
            None,
            GpuAcceleration::Off
        ));
        assert!(live_agent_should_run(
            LiveAgentMode::On,
            Some(&make_gpu_probe(4, false)),
            GpuAcceleration::Auto
        ));
        assert!(live_agent_should_run(
            LiveAgentMode::On,
            Some(&make_gpu_probe(8, true)),
            GpuAcceleration::On
        ));
    }

    #[test]
    fn live_agent_auto_no_probe_returns_false() {
        // No GPU probe → Auto gates off regardless of accel setting.
        assert!(!live_agent_should_run(
            LiveAgentMode::Auto,
            None,
            GpuAcceleration::Auto
        ));
        assert!(!live_agent_should_run(
            LiveAgentMode::Auto,
            None,
            GpuAcceleration::On
        ));
    }

    #[test]
    fn live_agent_auto_accel_off_returns_false() {
        // Probe present but gpu_acceleration=Off → Auto gates off.
        // The LLM would run on CPU, contending with ASR.
        assert!(!live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&make_gpu_probe(36, false)),
            GpuAcceleration::Off
        ));
        assert!(!live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&make_gpu_probe(8, true)),
            GpuAcceleration::Off
        ));
    }

    #[test]
    fn live_agent_auto_integrated_gpu_accel_on_returns_true() {
        // Integrated GPU with acceleration active (e.g. AMD Radeon 890M, Vulkan
        // on) → Auto enables. The is_integrated flag is NOT a gate; the GPU-
        // acceleration-active proxy is the correct discriminator (SP-LIVE E1).
        assert!(live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&make_gpu_probe(16, true)),
            GpuAcceleration::Auto
        ));
        assert!(live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&make_gpu_probe(16, true)),
            GpuAcceleration::On
        ));
    }

    #[test]
    fn live_agent_auto_discrete_gpu_accel_on_returns_true() {
        // Discrete GPU with acceleration active → Auto enables.
        assert!(live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&make_gpu_probe(4, false)),
            GpuAcceleration::Auto
        ));
        assert!(live_agent_should_run(
            LiveAgentMode::Auto,
            Some(&make_gpu_probe(24, false)),
            GpuAcceleration::On
        ));
    }

    #[test]
    fn live_agent_mode_default_is_auto() {
        assert_eq!(LiveAgentMode::default(), LiveAgentMode::Auto);
    }

    #[test]
    fn live_agent_mode_serialises_snake_case() {
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
    fn live_digest_and_item_round_trip() {
        let mid = MeetingId::new();
        let digest = LiveDigest {
            meeting_id: mid,
            generated_at_ms: 1_700_000_000_000,
            action_items: vec![LiveDigestItem {
                text: "Alice to send the report".to_string(),
                resolved: false,
                source: None,
            }],
            decisions: vec![LiveDigestItem {
                text: "Launch date moved to Q3".to_string(),
                resolved: true,
                source: Some("slide deck".to_string()),
            }],
            open_asks: vec![],
            attachment_answers: vec![],
            unresolved_references: vec![],
        };
        let json = serde_json::to_string(&digest).unwrap();
        let back: LiveDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.meeting_id, mid);
        assert_eq!(back.action_items.len(), 1);
        assert!(!back.action_items[0].resolved);
        assert!(back.action_items[0].source.is_none());
        assert_eq!(back.decisions.len(), 1);
        assert!(back.decisions[0].resolved);
        assert_eq!(back.decisions[0].source.as_deref(), Some("slide deck"));
        // Absent `source` is omitted from the wire shape.
        assert!(
            !json.contains("\"source\":null"),
            "absent source must be omitted not null"
        );
    }

    #[test]
    fn app_event_live_digest_updated_serialises_with_tag() {
        let mid = MeetingId::new();
        let e = AppEvent::LiveDigestUpdated {
            meeting_id: mid,
            digest: LiveDigest {
                meeting_id: mid,
                generated_at_ms: 0,
                action_items: vec![],
                decisions: vec![],
                open_asks: vec![],
                attachment_answers: vec![],
                unresolved_references: vec![],
            },
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"live_digest_updated\""));
    }

    #[test]
    fn app_event_live_digest_error_serialises_with_tag() {
        let e = AppEvent::LiveDigestError {
            meeting_id: MeetingId::new(),
            message: "context overflow".to_string(),
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"live_digest_error\""));
        assert!(json.contains("context overflow"));
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
