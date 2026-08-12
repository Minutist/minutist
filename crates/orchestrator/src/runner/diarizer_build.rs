//! Construction of the offline (on-stop / re-diarize) and live (Phase B) diarizer instances from the model registry.

use super::*;


/// Manifest id of the bundled segmentation model (pyannote/segmentation-3.0,
/// MIT). See `resources/models.json` and `architecture/components.md` —
/// `diarizer`.
pub(crate) const DIARIZE_SEG_MODEL_ID: &str = "pyannote-segmentation-3-0";
/// Manifest id of the bundled speaker-embedding model (3D-Speaker CAM++ zh-cn
/// 16k-common, Apache-2.0).
pub(crate) const DIARIZE_EMB_MODEL_ID: &str = "3dspeaker-campplus-zh-en-advanced";

/// Lazily build the production `SherpaDiarizer`, mirroring
/// [`build_asr_backend_for_retranscribe`].
///
/// Resolves both diarize model directories via `ModelRegistry::ensure`
/// (downloading + hash-verifying when absent), locates the single `.onnx` file
/// in each directory, and opens a `SherpaDiarizer` over the
/// `(segmentation, embedding)` pair with `DiarizerConfig::default()`.
///
/// Unlike the ASR runtime, the model directories are *ensured* (not merely
/// checked for `Available`): both the on-stop pass and the user-triggered
/// re-diarize are explicit operations, so a missing model is a downloadable
/// dependency rather than a best-effort skip. A resolution / load failure is an
/// error (`AppError::ModelLoad` / the registry's download error), surfaced to
/// the caller.
pub(crate) async fn build_diarizer(
    model_registry: &ModelRegistry,
) -> AppResult<diarizer::SherpaDiarizer> {
    let seg_id = ModelId::from(DIARIZE_SEG_MODEL_ID);
    let emb_id = ModelId::from(DIARIZE_EMB_MODEL_ID);

    let seg_dir = model_registry.ensure(&seg_id).await?;
    let emb_dir = model_registry.ensure(&emb_id).await?;

    let seg_onnx = find_file_in_dir(&seg_dir, |name| name.ends_with(".onnx"))?;
    let emb_onnx = find_file_in_dir(&emb_dir, |name| name.ends_with(".onnx"))?;

    let diarizer =
        diarizer::SherpaDiarizer::open(&seg_onnx, &emb_onnx, diarizer::DiarizerConfig::default())?;

    tracing::info!(
        target: "orchestrator",
        seg = %seg_onnx.display(),
        emb = %emb_onnx.display(),
        "diarizer initialised"
    );

    Ok(diarizer)
}

/// Best-effort, local-only builder for the live [`OnlineDiarizer`] (Phase B).
///
/// Unlike [`build_diarizer`] (which `ensure()`s the models — an explicit
/// operation may download), this is the LIVE path: it must NEVER download or
/// block at record start. It resolves the embedding model purely from disk via
/// the synchronous `Available`-check ([`ModelRegistry::list_models`] →
/// `compute_status_sync`, a `std::fs` size-only check — the exact non-blocking,
/// no-network precedent `init_asr_backend` uses), wraps ONLY the embedding model
/// (no segmentation model — VAD upstream supplies segment boundaries), and opens
/// `OnlineDiarizer::open(emb_onnx, OnlineDiarizerConfig::default())`.
///
/// Returns `None` on EVERY non-happy path so a failure degrades to "no live
/// label" without affecting recording/transcription:
/// - embedding model not locally `Available` → `info` log, `None` (the explicit
///   locked-constraint behaviour: no mid/at-start multi-GB download, no block);
/// - the `.onnx` cannot be located, or `OnlineDiarizer::open` fails (corrupt
///   model / sherpa load error) → `warn` log, `None`.
///
/// `list_models()` is synchronous, so this is a plain (non-async) fn; the heavy
/// `EmbeddingExtractor::new` load inside `open` is the caller's responsibility to
/// run off the async executor (the start path drives it on `spawn_blocking`,
/// mirroring the on-stop diarizer build).
pub(crate) fn build_online_diarizer(
    model_registry: &ModelRegistry,
) -> Option<Arc<OnlineDiarizer>> {
    let emb_id = ModelId::from(DIARIZE_EMB_MODEL_ID);

    // Local-only resolve: the embedding model must already be `Available`.
    let local_dir = model_registry
        .list_models()
        .into_iter()
        .find(|s| s.id == emb_id)
        .and_then(|s| match s.status {
            ModelStatusState::Available { local_dir } => Some(local_dir),
            _ => None,
        });

    let local_dir = match local_dir {
        Some(d) => d,
        None => {
            tracing::info!(
                target: "orchestrator",
                "live diarization: embedding model not downloaded; skipping (recording unaffected)"
            );
            return None;
        }
    };

    let emb_onnx = match find_file_in_dir(std::path::Path::new(&local_dir), |name| {
        name.ends_with(".onnx")
    }) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(
                target: "orchestrator",
                "live diarization: embedding .onnx not located ({e}); skipping"
            );
            return None;
        }
    };

    match OnlineDiarizer::open(&emb_onnx, OnlineDiarizerConfig::default()) {
        Ok(diarizer) => {
            tracing::info!(
                target: "orchestrator",
                emb = %emb_onnx.display(),
                "live online diarizer initialised"
            );
            Some(Arc::new(diarizer))
        }
        Err(e) => {
            tracing::warn!(
                target: "orchestrator",
                "live diarization: OnlineDiarizer::open failed ({e}); skipping"
            );
            None
        }
    }
}
