/**
 * "Report a problem" store (issue #0014).
 *
 * The shared seam both report surfaces use — the About dialog row and the
 * error-pane button. Holds the in-flight flag and a short status line (so a
 * surface can tell the user the browser was opened and, when the report was too
 * large for the URL, that the full report was copied to the clipboard). The
 * actual fetch / URL build / browser-open lives in {@link reportProblem}; this
 * store only tracks UI state around it.
 *
 * No telemetry: nothing is sent — the action opens the user's browser at a
 * pre-filled issue the user reviews and submits themselves.
 */
import { create } from "zustand";
import { reportProblem } from "../diagnostics/reportProblem";

export type ReportProblemStore = {
  /** True while the report is being assembled / the browser opened. */
  reporting: boolean;
  /** A short status line after the last attempt; `null` before any attempt. */
  status: string | null;
  /**
   * The most recent unhandled webview error message, surfaced in the main-window
   * error pane (which offers the Report button) so a frontend crash feeds the
   * same report path as a backend error. `null` when none. The Rust diagnostic
   * snapshot is still the report's substance; this only names the surface.
   */
  webviewError: string | null;
  /**
   * Run the flow. `errorClass` (optional) names the surface the user clicked
   * from (e.g. the start-failed message), overriding the report's own class.
   */
  report: (errorClass?: string) => Promise<void>;
  /** Record an unhandled webview error (from a global error/rejection handler). */
  captureWebviewError: (message: string) => void;
  /** Dismiss the surfaced webview error. */
  clearWebviewError: () => void;
  /** Clear the status line (e.g. when a surface unmounts or re-opens). */
  clearStatus: () => void;
};

export const useReportProblemStore = create<ReportProblemStore>((set, get) => ({
  reporting: false,
  status: null,
  webviewError: null,

  report: async (errorClass?: string) => {
    if (get().reporting) return;
    set({ reporting: true, status: null });
    try {
      const outcome = await reportProblem(errorClass);
      set({
        reporting: false,
        status: outcome.copiedToClipboard
          ? "Opened your browser. The report was large, so the full report was copied to your clipboard — paste it into the “Diagnostic report” field at the bottom of the issue."
          : "Opened your browser with the report pre-filled.",
      });
    } catch (err) {
      set({
        reporting: false,
        status:
          "Could not open the report: " +
          (err instanceof Error ? err.message : String(err)),
      });
    }
  },

  captureWebviewError: (message: string) =>
    set({ webviewError: message.trim() || "Unhandled webview error" }),

  clearWebviewError: () => set({ webviewError: null }),

  clearStatus: () => set({ status: null }),
}));
