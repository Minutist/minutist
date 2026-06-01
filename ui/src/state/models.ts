/**
 * Zustand store for model registry state.
 *
 * Tracks each model's status, active download progress, and whether
 * the ASR model is ready to use. Mutations flow through `handleEvent`
 * (called by the global event bridge) and the async action methods.
 */
import { create } from "zustand";
import { commands, unwrap } from "../ipc/client";
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
    try {
      const result = await commands.ensureModel(id);
      unwrap(result);
      // Refresh to pick up the newly available model.
      await get().refreshModels();
    } catch (_err) {
      // Errors surface via backend error_occurred events; no local lastError
      // needed here.
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
