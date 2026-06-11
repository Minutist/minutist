/**
 * Update-available banner.
 *
 * A quiet, non-intrusive strip that appears below the chrome strip when the
 * Rust-side tauri-plugin-updater reports a newer release. It lives alongside
 * `ModelDownloadStatus` (the existing chrome strip consumer) and uses the same
 * `--accent` / `--sheet` / `--rule` / `--ink-soft` vocabulary as the rest of
 * the Editorial Ink language — no hard-coded colours.
 *
 * States:
 *   - `available`    → "Update available: v{x}" + "Install and restart" button.
 *   - `downloading`  → progress bar (determinate when totalBytes is known,
 *                      indeterminate otherwise) + "Downloading…" label.
 *   - `applying`     → "Restarting…" text (the app is about to close).
 *   - `idle`         → renders nothing.
 *
 * The `applyUpdate()` action is called on button click; it emits
 * `updater://apply` to the Rust updater (see `state/update.ts`).
 */
import { useUpdateStore } from "../state/update";
import "./UpdateBanner.css";

export function UpdateBanner() {
  const update = useUpdateStore((s) => s.update);
  const applyUpdate = useUpdateStore((s) => s.applyUpdate);

  if (update.kind === "idle") return null;

  if (update.kind === "available") {
    return (
      <div className="update-banner" role="status" aria-live="polite">
        <span className="update-banner__label">
          Update available: <strong>v{update.version}</strong>
        </span>
        <button
          type="button"
          className="update-banner__action"
          onClick={() => applyUpdate()}
        >
          Install and restart
        </button>
      </div>
    );
  }

  if (update.kind === "downloading") {
    const fraction =
      update.totalBytes !== null && update.totalBytes > 0
        ? update.downloadedBytes / update.totalBytes
        : null;
    return (
      <div className="update-banner" role="status" aria-live="polite">
        <span className="update-banner__label">Downloading update…</span>
        <div
          className="update-banner__progress"
          role="progressbar"
          aria-label="Download progress"
          aria-valuenow={fraction !== null ? Math.round(fraction * 100) : undefined}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          {fraction !== null ? (
            <div
              className="update-banner__progress-fill"
              style={{ width: `${(fraction * 100).toFixed(1)}%` }}
            />
          ) : (
            <div className="update-banner__progress-fill update-banner__progress-fill--indeterminate" />
          )}
        </div>
      </div>
    );
  }

  // applying
  return (
    <div className="update-banner" role="status" aria-live="polite">
      <span className="update-banner__label">Restarting…</span>
    </div>
  );
}
