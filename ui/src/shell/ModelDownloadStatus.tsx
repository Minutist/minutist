import { useModelsStore } from "../state/models";

/**
 * First-run download flow.
 *
 * Shown when the ASR model is not yet available. Displays the model's
 * display_name, a progress bar while downloading, and a "Download Model"
 * button. Disappears once `isAsrModelReady` is true.
 */

const ASR_MODEL_ID = "qwen3-asr-0.6b-q8_0";

export function ModelDownloadStatus() {
  const models = useModelsStore((s) => s.models);
  const isAsrModelReady = useModelsStore((s) => s.isAsrModelReady);
  const downloadInProgress = useModelsStore((s) => s.downloadInProgress);
  const downloadErrors = useModelsStore((s) => s.downloadErrors);
  const ensureModel = useModelsStore((s) => s.ensureModel);

  // Only show when ASR model is not ready.
  if (isAsrModelReady) return null;

  // Find the ASR model entry (may not exist yet if refreshModels hasn't run).
  const asrModel = models.find((m) => m.id === ASR_MODEL_ID);
  const displayName = asrModel?.display_name ?? "ASR Model";
  const inProgress = downloadInProgress[ASR_MODEL_ID];
  const downloadError = downloadErrors[ASR_MODEL_ID];
  const isDownloading =
    inProgress !== undefined ||
    asrModel?.status.state === "downloading";

  // Progress fraction for the bar.
  let progressFraction = 0;
  if (inProgress && inProgress.bytes_total > 0) {
    progressFraction = inProgress.bytes_done / inProgress.bytes_total;
  } else if (asrModel?.status.state === "downloading") {
    const s = asrModel.status;
    if (s.state === "downloading" && s.bytes_total > 0) {
      progressFraction = s.bytes_done / s.bytes_total;
    }
  }
  const progressPct = `${(progressFraction * 100).toFixed(1)}%`;

  return (
    <div className="model-download-status" role="region" aria-label="Model download">
      <p className="model-download-status__name">{displayName}</p>

      {isDownloading && (
        <div
          className="model-download-status__progress-track"
          role="progressbar"
          aria-valuenow={Math.round(progressFraction * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Downloading ${displayName}`}
        >
          <div
            className="model-download-status__progress-bar"
            style={{ width: progressPct }}
          />
        </div>
      )}

      {!isDownloading && downloadError && (
        <p className="model-download-status__error" role="alert">
          {downloadError}
        </p>
      )}

      {!isDownloading && (
        <button
          className="model-download-status__download-btn"
          onClick={() => void ensureModel(ASR_MODEL_ID)}
        >
          {downloadError ? "Retry Download" : "Download Model"}
        </button>
      )}
    </div>
  );
}
