//! `diarizer` — opt-in offline speaker diarization.
//!
//! Implements [`meeting_app_common::Diarizer`] over a sherpa-onnx (via
//! `sherpa-rs`) two-model pipeline: a **segmentation** model + a
//! **speaker-embedding** model + clustering. It assigns `speaker_id` to the
//! ASR `Segment`s of a finished recording (a post-pass, never on the live
//! path). See `architecture/components.md` — `diarizer`, and the Phase-6 plan.
//!
//! Models (license-verified, settings-selected via `model-registry`):
//! - segmentation: pyannote/segmentation-3.0 (MIT)
//! - embedding: 3D-Speaker CAM++ zh-cn 16k-common (Apache-2.0, in-house corpus,
//!   NOT VoxCeleb — the clean commercial-redistribution path).
//!
//! **Skeleton (Phase 6 Gate A).** The public surface — [`SherpaDiarizer`] +
//! its `Diarizer` impl + [`DiarizerConfig`] — is frozen here so the
//! orchestrator on-stop / re-diarize join can compile against it while Stream
//! S1 fills the body (load sherpa `Diarize`; run `compute`; first-seen relabel
//! to "A"/"B"/…; max-overlap interval-join of sherpa turns onto each ASR
//! segment; conservative clustering for single-speaker cleanliness). The
//! `assign_speakers` body is a stub returning an `AppError` (never panics)
//! until S1 lands the implementation.

use std::path::{Path, PathBuf};

use meeting_app_common::{AppResult, Diarizer, Segment};

mod error;
pub use error::Error;

/// Clustering knobs for the diarizer.
///
/// Exactly one of `num_clusters` (known speaker count) or `cluster_threshold`
/// (unknown count; smaller → more speakers) drives sherpa's agglomerative
/// stage. For single-speaker cleanliness, callers set `num_clusters = Some(1)`
/// when the count is known, else a conservative threshold so one speaker is not
/// over-split.
#[derive(Debug, Clone)]
pub struct DiarizerConfig {
    /// Fixed speaker count, when known. `None` → use `cluster_threshold`.
    pub num_clusters: Option<u32>,
    /// Agglomerative-clustering distance threshold when the count is unknown.
    pub cluster_threshold: f32,
}

impl Default for DiarizerConfig {
    fn default() -> Self {
        Self {
            num_clusters: None,
            // Conservative default: avoid over-splitting a single speaker.
            cluster_threshold: 0.5,
        }
    }
}

/// A speaker diarizer backed by a sherpa-onnx segmentation + embedding pipeline.
///
/// Construct with [`SherpaDiarizer::open`]. The model load and the
/// `assign_speakers` inference are implemented by Stream S1 (this is the frozen
/// skeleton).
pub struct SherpaDiarizer {
    segmentation_path: PathBuf,
    embedding_path: PathBuf,
    config: DiarizerConfig,
}

impl SherpaDiarizer {
    /// Open a diarizer over the segmentation + embedding ONNX models.
    ///
    /// Skeleton: stores the resolved paths + config. Stream S1 constructs the
    /// sherpa `Diarize` engine here.
    pub fn open(
        segmentation_path: &Path,
        embedding_path: &Path,
        config: DiarizerConfig,
    ) -> AppResult<Self> {
        Ok(Self {
            segmentation_path: segmentation_path.to_path_buf(),
            embedding_path: embedding_path.to_path_buf(),
            config,
        })
    }

    /// The segmentation model path.
    pub fn segmentation_path(&self) -> &Path {
        &self.segmentation_path
    }

    /// The speaker-embedding model path.
    pub fn embedding_path(&self) -> &Path {
        &self.embedding_path
    }

    /// The active clustering configuration.
    pub fn config(&self) -> &DiarizerConfig {
        &self.config
    }
}

impl Diarizer for SherpaDiarizer {
    fn assign_speakers(
        &self,
        _audio: &[f32],
        _sample_rate: u32,
        _segments: &mut [Segment],
    ) -> AppResult<u32> {
        // Stream S1 replaces this body. Returns an error (not a panic) so the
        // skeleton is safe to wire end-to-end before the implementation lands.
        tracing::warn!(
            target: "diarizer",
            "assign_speakers() called on the Phase-6 skeleton stub; not yet implemented"
        );
        Err(Error::Inference(
            "diarizer inference not yet implemented (Phase 6 Gate-A skeleton)".to_string(),
        )
        .into())
    }
}
