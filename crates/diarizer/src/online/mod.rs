//! `online` — additive, live speaker labelling.
//!
//! A streaming-style speaker labeller that wraps a sherpa `EmbeddingExtractor`
//! (CAM++ embedding model) behind a `Mutex` and delegates clustering to a pure
//! [`OnlineClusterer`]. It emits sticky first-seen labels ("A", "B", …) per VAD
//! segment as the segment closes and NEVER retroactively relabels.
//!
//! This is an ADDITIVE live hint, not a replacement: the on-stop
//! [`crate::SherpaDiarizer`] / `common::Diarizer` pass remains AUTHORITATIVE for
//! the finished transcript. The two are independent code paths sharing only
//! [`alpha_label`] and the 16 kHz guard. Phase A does NOT wire into the
//! orchestrator (that is Phase B) and adds no `common`-level trait.

pub mod clusterer;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use minutist_common::AppResult;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};

use crate::{alpha_label, require_model_file, require_supported_sample_rate, Error, REQUIRED_SAMPLE_RATE};

use clusterer::{OnlineClusterer, OnlineClustererConfig};

/// Per-segment knobs for the live diarizer.
#[derive(Debug, Clone, Default)]
pub struct OnlineDiarizerConfig {
    /// Clustering knobs forwarded to the pure [`OnlineClusterer`].
    pub clusterer: OnlineClustererConfig,
}

/// The `(extractor, clusterer)` pair guarded by the diarizer's `Mutex`.
///
/// One mutex guards both because `compute_speaker_embedding` takes `&mut self`
/// AND `OnlineClusterer::assign` takes `&mut self`, and the two run in lockstep
/// per segment (extract then assign).
struct OnlineInner {
    extractor: EmbeddingExtractor,
    clusterer: OnlineClusterer,
}

/// Live, streaming-style speaker labeller. Wraps a sherpa `EmbeddingExtractor`
/// (CAM++ embedding model) behind a `Mutex` and delegates clustering to a pure
/// [`OnlineClusterer`]. Produces sticky first-seen labels ("A", "B", …); NEVER
/// retroactively relabels. The on-stop `SherpaDiarizer` pass remains
/// authoritative for the final transcript — this is an additive live hint, not
/// a replacement.
pub struct OnlineDiarizer {
    embedding_path: PathBuf,
    #[allow(dead_code)] // retained for symmetry with SherpaDiarizer and Phase-B introspection
    config: OnlineDiarizerConfig,
    // Mutex because `EmbeddingExtractor::compute_speaker_embedding` takes
    // `&mut self` and the public `assign_segment` takes `&self` (matches the
    // `SherpaDiarizer` `Mutex<Diarize>` pattern). `OnlineClusterer` is also
    // `&mut`, so one Mutex guards the (extractor, clusterer) pair together.
    inner: Mutex<OnlineInner>,
}

impl OnlineDiarizer {
    /// Open over the speaker-embedding ONNX model.
    ///
    /// Mirrors [`crate::SherpaDiarizer::open`]'s loading + error mapping: builds
    /// `ExtractorConfig { model: <path string>, provider: None, num_threads:
    /// None, debug: false }`, calls `EmbeddingExtractor::new(cfg)`, and maps the
    /// sherpa `eyre` error to `Error::ModelLoad`.
    ///
    /// NOTE: takes ONLY the embedding model path — the online path needs no
    /// segmentation model (VAD upstream already supplies segment boundaries).
    ///
    /// Pre-flight validated ([`require_model_file`]) before the sherpa FFI call.
    pub fn open(embedding_path: &Path, config: OnlineDiarizerConfig) -> AppResult<Self> {
        require_model_file(embedding_path)?;

        let extractor_config = ExtractorConfig {
            model: embedding_path.display().to_string(),
            provider: None,
            num_threads: None,
            debug: false,
        };
        let extractor = EmbeddingExtractor::new(extractor_config).map_err(|e| Error::ModelLoad {
            path: embedding_path.display().to_string(),
            context: format!("{e:?}"),
        })?;

        tracing::debug!(
            target: "diarizer",
            embedding_size = extractor.embedding_size,
            path = %embedding_path.display(),
            "opened online diarizer"
        );

        let clusterer = OnlineClusterer::new(config.clusterer.clone());

        Ok(Self {
            embedding_path: embedding_path.to_path_buf(),
            config,
            inner: Mutex::new(OnlineInner {
                extractor,
                clusterer,
            }),
        })
    }

    /// Label one VAD segment's audio.
    ///
    /// `samples` is 16 kHz mono f32 for a SINGLE VAD segment (NOT the whole
    /// recording). Returns the sticky label ("A", "B", "C", …). Steps:
    /// 1. reject any `sample_rate != 16000` (`InvalidInput`);
    /// 2. empty `samples` => `InvalidInput` (an empty VAD segment is a caller
    ///    bug — unlike the offline whole-recording path there is no meaningful
    ///    "0 speakers" answer for one segment);
    /// 3. lock the mutex (poison => `Inference`);
    /// 4. extract the speaker embedding (sherpa `eyre` err => `Inference`);
    /// 5. cluster-assign;
    /// 6. map the sticky index to its label.
    pub fn assign_segment(&self, samples: &[f32], sample_rate: u32) -> AppResult<String> {
        require_supported_sample_rate(sample_rate)?;

        if samples.is_empty() {
            return Err(Error::InvalidInput(
                "online diarizer requires a non-empty VAD segment".to_string(),
            )
            .into());
        }

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| Error::Inference("online diarizer mutex poisoned".to_string()))?;

        // sherpa takes ownership of the sample buffer; clone the borrowed slice
        // into an owned Vec for the FFI call.
        let embedding = inner
            .extractor
            .compute_speaker_embedding(samples.to_vec(), REQUIRED_SAMPLE_RATE)
            .map_err(|e| {
                Error::Inference(format!("sherpa compute_speaker_embedding failed: {e:?}"))
            })?;

        let assignment = inner.clusterer.assign(&embedding)?;
        Ok(alpha_label(assignment.index))
    }

    /// Distinct speakers minted so far. Mutex poison => `Inference`.
    pub fn speaker_count(&self) -> AppResult<u32> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| Error::Inference("online diarizer mutex poisoned".to_string()))?;
        Ok(inner.clusterer.len() as u32)
    }

    /// The speaker-embedding model path.
    pub fn embedding_path(&self) -> &Path {
        &self.embedding_path
    }
}

/// Stateless speaker-embedding extractor: wraps a sherpa `EmbeddingExtractor`
/// behind a `Mutex` and exposes a pure embedding + centroid surface for
/// voiceprint enrolment.
///
/// Unlike [`OnlineDiarizer`], this type owns NO clusterer — it is a building
/// block for the voiceprint enrolment flow (`persistence::VoiceprintStore`):
/// the caller supplies a set of audio windows, obtains raw embeddings via
/// [`VoiceprintExtractor::embed`], then builds a voiceprint centroid via
/// [`VoiceprintExtractor::centroid`]. The clustering and storage decisions live
/// entirely outside this type.
///
/// Diarizer-public, deliberately NOT in `common` (mirrors [`crate::SpeakerTurn`]).
/// `open` mirrors [`OnlineDiarizer::open`] exactly: same `ExtractorConfig` build,
/// same `EmbeddingExtractor::new`, same `Error::ModelLoad` mapping.
pub struct VoiceprintExtractor {
    embedding_path: PathBuf,
    extractor: Mutex<EmbeddingExtractor>,
}

impl VoiceprintExtractor {
    /// Open over the speaker-embedding ONNX model.
    ///
    /// Mirrors [`OnlineDiarizer::open`]'s loading + error mapping: builds
    /// `ExtractorConfig { model: <path string>, provider: None, num_threads:
    /// None, debug: false }`, calls `EmbeddingExtractor::new(cfg)`, and maps the
    /// sherpa `eyre` error to `Error::ModelLoad`.
    ///
    /// Pre-flight validated ([`require_model_file`]) before the sherpa FFI call.
    pub fn open(embedding_path: &Path) -> AppResult<Self> {
        require_model_file(embedding_path)?;

        let extractor_config = ExtractorConfig {
            model: embedding_path.display().to_string(),
            provider: None,
            num_threads: None,
            debug: false,
        };
        let extractor =
            EmbeddingExtractor::new(extractor_config).map_err(|e| Error::ModelLoad {
                path: embedding_path.display().to_string(),
                context: format!("{e:?}"),
            })?;

        tracing::debug!(
            target: "diarizer",
            embedding_size = extractor.embedding_size,
            path = %embedding_path.display(),
            "opened voiceprint extractor"
        );

        Ok(Self {
            embedding_path: embedding_path.to_path_buf(),
            extractor: Mutex::new(extractor),
        })
    }

    /// The speaker-embedding model path.
    pub fn embedding_path(&self) -> &Path {
        &self.embedding_path
    }

    /// Extract a raw (un-normalised) speaker embedding for one audio window.
    ///
    /// `samples` must be 16 kHz mono f32 (rejects other sample rates as
    /// `InvalidInput`, matching [`crate::online::OnlineDiarizer::assign_segment`]).
    /// Returns the raw 192-D embedding vector as produced by the CAM++ model;
    /// the caller is responsible for normalising when building a centroid.
    ///
    /// sherpa takes ownership of the sample buffer; the borrowed slice is cloned
    /// into an owned `Vec` for the FFI call (same pattern as `assign_segment`).
    pub fn embed(&self, samples: &[f32], sr: u32) -> AppResult<Vec<f32>> {
        require_supported_sample_rate(sr)?;

        if samples.is_empty() {
            return Err(Error::InvalidInput(
                "voiceprint extractor requires a non-empty audio window".to_string(),
            )
            .into());
        }

        let mut extractor = self
            .extractor
            .lock()
            .map_err(|_| Error::Inference("voiceprint extractor mutex poisoned".to_string()))?;

        extractor
            .compute_speaker_embedding(samples.to_vec(), REQUIRED_SAMPLE_RATE)
            .map_err(|e| {
                Error::Inference(format!("sherpa compute_speaker_embedding failed: {e:?}"))
            })
            .map_err(Into::into)
    }

    /// Build a [`crate::Voiceprint`] centroid from one or more audio windows.
    ///
    /// Each window is embedded independently; the resulting raw vectors are
    /// unit-normalised and then averaged + re-normalised via
    /// [`minutist_common::voiceprint_math`]:
    ///
    /// ```text
    /// centroid = unit_normalise(mean(unit_normalise(embed(w)) for w in windows))
    /// ```
    ///
    /// This matches the [`crate::online::clusterer::OnlineClusterer`]
    /// running-mean-of-unit-vectors rule: both the online clusterer and this
    /// centroid builder operate in the same unit-vector space, so a voiceprint
    /// produced here is directly comparable (via cosine) to a centroid the online
    /// clusterer has accumulated.
    ///
    /// Rejects `sr != 16000`, an empty `windows` slice, or any window whose
    /// embedding is degenerate (zero/non-finite norm — `unit_normalise` is a
    /// no-op on those).
    pub fn centroid(&self, windows: &[&[f32]], sr: u32) -> AppResult<crate::Voiceprint> {
        require_supported_sample_rate(sr)?;

        if windows.is_empty() {
            return Err(
                Error::InvalidInput("centroid requires at least one audio window".to_string())
                    .into(),
            );
        }

        // Embed + unit-normalise each window.
        let mut unit_vecs: Vec<Vec<f32>> = Vec::with_capacity(windows.len());
        for &window in windows {
            let mut emb = self.embed(window, sr)?;
            minutist_common::voiceprint_math::unit_normalise(&mut emb);
            unit_vecs.push(emb);
        }

        // Count-weighted merge (equal counts of 1 for independently-normalised
        // windows) via common::voiceprint_math::weighted_merge, then
        // unit-normalise the mean — exactly the OnlineClusterer discipline.
        let pairs: Vec<(&[f32], u64)> = unit_vecs.iter().map(|v| (v.as_slice(), 1u64)).collect();
        let vector = minutist_common::voiceprint_math::weighted_merge(&pairs);

        Ok(crate::Voiceprint { vector })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_diarizer_open_rejects_missing_model_without_panicking() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("embedding.onnx");
        match OnlineDiarizer::open(&missing, OnlineDiarizerConfig::default()) {
            Err(minutist_common::AppError::ModelLoad { .. }) => {}
            Err(other) => panic!("expected ModelLoad, got {other}"),
            Ok(_) => panic!("expected an error for a missing embedding model"),
        }
    }

    #[test]
    fn voiceprint_extractor_open_rejects_empty_model_without_panicking() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("embedding.onnx");
        std::fs::write(&path, []).expect("write empty file");
        match VoiceprintExtractor::open(&path) {
            Err(minutist_common::AppError::ModelLoad { .. }) => {}
            Err(other) => panic!("expected ModelLoad, got {other}"),
            Ok(_) => panic!("expected an error for an empty embedding model"),
        }
    }
}
