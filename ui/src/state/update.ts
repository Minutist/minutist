/**
 * Update lifecycle store.
 *
 * Tracks the in-app update state driven by the Rust-side tauri-plugin-updater:
 *   - `update_available`  → store the available version + release notes and
 *                           surface an "Install and restart" affordance.
 *   - `update_progress`   → track download progress so the UI can show a bar.
 *   - `error_occurred`    → if an update apply fails the error surfaces via the
 *                           recording store's `lastError`; no special update
 *                           state is stored here.
 *
 * `applyUpdate()` emits the `updater://apply` raw Tauri event that
 * `updater.rs` listens for on the Rust side. It is a fire-and-forget: the
 * in-progress flag (`applying`) prevents duplicate clicks, and the app
 * restarts on success so no "done" state is needed.
 *
 * The store is fed by the global event bridge (`shell/event-listener.tsx`).
 */
import { create } from "zustand";
import type { AppEvent } from "../ipc/app-event";

/** Emits the accept event to the Rust updater. Injected so tests can mock it. */
export type ApplyFn = () => Promise<void>;

/** The default implementation uses the Tauri event API directly. */
async function defaultApplyFn(): Promise<void> {
  // Lazy-import to keep the Tauri API out of the test bundle.
  const { emit } = await import("@tauri-apps/api/event");
  await emit("updater://apply");
}

export type UpdateState =
  | { kind: "idle" }
  | {
      kind: "available";
      version: string;
      notes: string | null;
    }
  | {
      kind: "downloading";
      version: string;
      notes: string | null;
      downloadedBytes: number;
      totalBytes: number | null;
    }
  | { kind: "applying" };

export type UpdateStore = {
  update: UpdateState;
  /** Dispatcher called by the global event bridge. */
  handleEvent: (event: AppEvent) => void;
  /**
   * Emit `updater://apply` to the Rust updater. A no-op when already applying.
   * The `applyFn` seam is injectable for tests.
   */
  applyUpdate: (applyFn?: ApplyFn) => void;
};

export const useUpdateStore = create<UpdateStore>((set, get) => ({
  update: { kind: "idle" },

  handleEvent: (event) => {
    switch (event.kind) {
      case "update_available": {
        // Only move to `available` from `idle`; do not overwrite an in-flight
        // download (a second check could fire while a download is running).
        if (get().update.kind !== "idle") break;
        set({
          update: {
            kind: "available",
            version: event.version,
            notes: event.notes,
          },
        });
        break;
      }
      case "update_progress": {
        const current = get().update;
        const version =
          current.kind === "available" || current.kind === "downloading"
            ? current.version
            : "";
        const notes =
          current.kind === "available" || current.kind === "downloading"
            ? current.notes
            : null;
        set({
          update: {
            kind: "downloading",
            version,
            notes,
            downloadedBytes: event.downloaded_bytes,
            totalBytes: event.total_bytes,
          },
        });
        break;
      }
      default:
        break;
    }
  },

  applyUpdate: (applyFn = defaultApplyFn) => {
    const current = get().update;
    // Only apply when an update is available or in a known state; never re-
    // trigger while already applying.
    if (current.kind === "idle" || current.kind === "applying") return;
    set({ update: { kind: "applying" } });
    applyFn().catch(() => {
      // Revert to `available` so the user can retry. The error surfaces via
      // `error_occurred` on the shared event bus (caught by the recording store).
      set({ update: current });
    });
  },
}));
