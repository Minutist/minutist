import { useModelsStore } from "../state/models";

/**
 * Per-model download surface.
 *
 * `ModelDownloadCard` renders one model's status (a role-first label such as
 * "Speech (Parakeet)", the backend's technical name + size underneath, a
 * progress bar while downloading, an error + retry on failure, a "ready" tick
 * when present) and a Download button otherwise. It is model-agnostic — keyed
 * entirely by `modelId` against the models store, which tracks progress and
 * errors per id. Used for both the ASR models and the summarisation LLM.
 *
 * `ModelDownloadStatus` is the ASR-only wrapper the main window uses: it
 * self-hides once the ASR model is ready (the first-run prompt disappears).
 */

// Primary speech model — Parakeet TDT v3 (English + 24 EU languages, with word
// timings). The default English experience routes here.
export const PARAKEET_MODEL_ID = "parakeet-tdt-0.6b-v3-int8";
// Broad-language speech model — Qwen3-ASR 0.6B (52 languages/dialects, incl.
// Chinese/Japanese/Korean/Arabic). Used for languages Parakeet doesn't cover.
export const ASR_MODEL_ID = "qwen3-asr-0.6b-q8_0";
export const LLM_MODEL_ID = "gemma-4-e4b-it-q4_k_m";

// Role-first labels for the known models, keyed by id. The backend's
// `display_name` is the raw model filename/variant (e.g. "Qwen3-ASR 0.6B
// Q8_0") — useful as a technical detail, but not what a first-run user needs
// to decide what to download. This is the label users see; it always takes
// priority over the backend name.
const MODEL_ROLE_LABELS: Record<string, string> = {
  [PARAKEET_MODEL_ID]: "Speech (Parakeet)",
  [ASR_MODEL_ID]: "Speech, other languages (Qwen)",
  [LLM_MODEL_ID]: "Summarise / Chat (Gemma)",
};

function formatBytes(n: number): string {
  if (n <= 0) return "";
  const gb = n / 1_000_000_000;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(n / 1_000_000)} MB`;
}

export function ModelDownloadCard({
  modelId,
  label,
  hint,
  badge,
}: {
  modelId: string;
  label?: string;
  hint?: string;
  badge?: string;
}) {
  const models = useModelsStore((s) => s.models);
  const downloadInProgress = useModelsStore((s) => s.downloadInProgress);
  const downloadErrors = useModelsStore((s) => s.downloadErrors);
  const ensureModel = useModelsStore((s) => s.ensureModel);

  const model = models.find((m) => m.id === modelId);
  const status = model?.status;
  const ready = status?.state === "available";
  const roleLabel = label ?? MODEL_ROLE_LABELS[modelId] ?? modelId;
  const technicalName = model?.display_name;
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
      aria-label={`Model ${roleLabel}`}
    >
      <p className="model-download-status__name">
        <span className="model-download-status__display-name">{roleLabel}</span>
        {ready && (
          <span className="model-download-status__ready"> · ready</span>
        )}
        {!ready && badge && (
          <span className="model-download-status__badge">{badge}</span>
        )}
      </p>
      {hint && <p className="model-download-status__hint">{hint}</p>}
      {(technicalName || sizeHint) && (
        <p className="model-download-status__technical">
          {technicalName && <span>{technicalName}</span>}
          {technicalName && sizeHint && " · "}
          {sizeHint && <span>{sizeHint}</span>}
        </p>
      )}

      {isDownloading && (
        <div
          className="model-download-status__progress-track"
          role="progressbar"
          aria-valuenow={Math.round(progressFraction * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Downloading ${roleLabel}`}
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
  return <ModelDownloadCard modelId={ASR_MODEL_ID} />;
}
