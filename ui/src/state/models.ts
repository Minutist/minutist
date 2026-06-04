/**
 * Zustand store for model registry state.
 *
 * Tracks each model's status, active download progress, and whether
 * the ASR model is ready to use. Mutations flow through `handleEvent`
 * (called by the global event bridge) and the async action methods.
 */
import { create } from "zustand";
import { commands, unwrap, IpcCallError } from "../ipc/client";
import type { ModelStatus } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type { ModelStatus };

export type ModelsStore = {
  models: ModelStatus[];
  /**
   * Derived: true when the "asr"-kind model has status.state === "available".
   */
  isAsrModelReady: boolean;
  /**
   * Active download progress keyed by model id.
   * Present only while a download is in progress; removed when done or on
   * the next `refreshModels` that shows a non-downloading state.
   */
  downloadInProgress: Record<string, { bytes_done: number; bytes_total: number }>;
  /**
   * Last download failure per model id (human-readable). Set when `ensureModel`
   * rejects — e.g. a SHA-256 mismatch (stale manifest) or a network failure —
   * so the UI can show the reason and a retry instead of a frozen progress bar.
   * Cleared when a fresh `ensureModel` for that id is started.
   */
  downloadErrors: Record<string, string>;

  /** Refresh the model list from the backend. */
  refreshModels: () => Promise<void>;
  /** Trigger a model download/verify. */
  ensureModel: (id: string) => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

function deriveIsAsrReady(models: ModelStatus[]): boolean {
  return models.some(
    (m) => m.kind === "asr" && m.status.state === "available",
  );
}

export const useModelsStore = create<ModelsStore>((set, get) => ({
  models: [],
  isAsrModelReady: false,
  downloadInProgress: {},
  downloadErrors: {},

  refreshModels: async () => {
    try {
      const result = await commands.listModels();
      const models = unwrap(result);
      set({
        models,
        isAsrModelReady: deriveIsAsrReady(models),
      });
    } catch (_err) {
      // refreshModels failures are non-fatal; the UI retains stale state.
    }
  },

  ensureModel: async (id: string) => {
    // Optimistically enter the downloading state IMMEDIATELY, seeded at the
    // model's known partial fraction. The backend can spend several seconds
    // re-hashing an existing partial file before the first real progress event
    // fires; without this the Download button lingers and a click reads as
    // "nothing happened". Real progress events overwrite this seed.
    set((s) => {
      const errs = { ...s.downloadErrors };
      delete errs[id];
      const st = s.models.find((m) => m.id === id)?.status;
      const seed =
        st?.state === "missing"
          ? { bytes_done: st.bytes_present, bytes_total: st.bytes_total }
          : st?.state === "downloading"
            ? { bytes_done: st.bytes_done, bytes_total: st.bytes_total }
            : { bytes_done: 0, bytes_total: 0 };
      return {
        downloadErrors: errs,
        downloadInProgress: { ...s.downloadInProgress, [id]: seed },
      };
    });
    try {
      const result = await commands.ensureModel(id);
      unwrap(result);
      // Refresh to pick up the newly available model.
      await get().refreshModels();
      // Clear the optimistic entry — the terminal progress event usually does
      // this already; backstop for the no-event / already-present fast paths.
      set((s) => {
        const next = { ...s.downloadInProgress };
        delete next[id];
        return { downloadInProgress: next };
      });
    } catch (err) {
      // `ensure_model` returns the failure to this caller and does NOT emit a
      // separate error event, so the rejection must be surfaced here — a
      // swallowed error leaves the progress bar frozen with no explanation.
      const message =
        err instanceof IpcCallError || err instanceof Error
          ? err.message
          : "Model download failed";
      set((s) => {
        const next = { ...s.downloadInProgress };
        delete next[id];
        return {
          downloadInProgress: next,
          downloadErrors: { ...s.downloadErrors, [id]: message },
        };
      });
    }
  },

  handleEvent: (event) => {
    switch (event.kind) {
      case "model_download_progress": {
        const { model_id, bytes_done, bytes_total } = event;
        if (bytes_total === null) {
          // Unknown total; still track bytes_done so the UI shows activity.
          set((s) => ({
            downloadInProgress: {
              ...s.downloadInProgress,
              [model_id]: { bytes_done, bytes_total: 0 },
            },
          }));
        } else {
          const done = bytes_done >= bytes_total;
          if (done) {
            // Download complete — remove the in-progress entry and refresh.
            set((s) => {
              const next = { ...s.downloadInProgress };
              delete next[model_id];
              return { downloadInProgress: next };
            });
            void get().refreshModels();
          } else {
            set((s) => ({
              downloadInProgress: {
                ...s.downloadInProgress,
                [model_id]: { bytes_done, bytes_total },
              },
            }));
          }
        }
        break;
      }
      default:
        break;
    }
  },
}));
