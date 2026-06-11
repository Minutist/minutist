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

use crate::{alpha_label, require_supported_sample_rate, Error, REQUIRED_SAMPLE_RATE};

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
    pub fn open(embedding_path: &Path, config: OnlineDiarizerConfig) -> AppResult<Self> {
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
