import { useModelsStore } from "../state/models";

/**
 * Per-model download surface.
 *
 * `ModelDownloadCard` renders one model's status (display name + size, a
 * progress bar while downloading, an error + retry on failure, a "ready" tick
 * when present) and a Download button otherwise. It is model-agnostic — keyed
 * entirely by `modelId` against the models store, which tracks progress and
 * errors per id. Used for both the ASR model and the summarisation LLM.
 *
 * `ModelDownloadStatus` is the ASR-only wrapper the main window uses: it
 * self-hides once the ASR model is ready (the first-run prompt disappears).
 */

export const ASR_MODEL_ID = "qwen3-asr-0.6b-q8_0";
export const LLM_MODEL_ID = "gemma-4-e4b-it-q4_k_m";

function formatBytes(n: number): string {
  if (n <= 0) return "";
  const gb = n / 1_000_000_000;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(n / 1_000_000)} MB`;
}

export function ModelDownloadCard({
  modelId,
  fallbackName,
}: {
  modelId: string;
  fallbackName?: string;
}) {
  const models = useModelsStore((s) => s.models);
  const downloadInProgress = useModelsStore((s) => s.downloadInProgress);
  const downloadErrors = useModelsStore((s) => s.downloadErrors);
  const ensureModel = useModelsStore((s) => s.ensureModel);

  const model = models.find((m) => m.id === modelId);
  const status = model?.status;
  const ready = status?.state === "available";
  const displayName = model?.display_name ?? fallbackName ?? modelId;
  const inProgress = downloadInProgress[modelId];
  const downloadError = downloadErrors[modelId];
  const isDownloading =
    inProgress !== undefined || status?.state === "downloading";

  // Progress fraction + total size for the bar / size hint.
  let progressFraction = 0;
  let totalBytes = 0;
  if (inProgress && inProgress.bytes_total > 0) {
    progressFraction = inProgress.bytes_done / inProgress.bytes_total;
    totalBytes = inProgress.bytes_total;
  } else if (status?.state === "downloading" && status.bytes_total > 0) {
    progressFraction = status.bytes_done / status.bytes_total;
    totalBytes = status.bytes_total;
  } else if (status?.state === "missing" && status.bytes_total > 0) {
    totalBytes = status.bytes_total;
  }
  const progressPct = `${(progressFraction * 100).toFixed(1)}%`;
  const sizeHint = !ready && totalBytes > 0 ? formatBytes(totalBytes) : "";

  return (
    <div
      className="model-download-status"
      role="region"
      aria-label={`Model ${displayName}`}
    >
      <p className="model-download-status__name">
        <span className="model-download-status__display-name">{displayName}</span>
        {sizeHint && (
          <span className="model-download-status__size"> · {sizeHint}</span>
        )}
        {ready && (
          <span className="model-download-status__ready"> · ready</span>
        )}
      </p>

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

      {!isDownloading && !ready && downloadError && (
        <p className="model-download-status__error" role="alert">
          {downloadError}
        </p>
      )}

      {!isDownloading && !ready && (
        <button
          className="model-download-status__download-btn"
          onClick={() => void ensureModel(modelId)}
        >
          {downloadError ? "Retry Download" : "Download"}
        </button>
      )}
    </div>
  );
}

/**
 * First-run ASR download prompt for the main window. Disappears once
 * `isAsrModelReady` is true.
 */
export function ModelDownloadStatus() {
  const isAsrModelReady = useModelsStore((s) => s.isAsrModelReady);
  if (isAsrModelReady) return null;
  return <ModelDownloadCard modelId={ASR_MODEL_ID} fallbackName="ASR Model" />;
}
