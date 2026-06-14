/**
 * "Report a problem" — the action that ties the redacted Rust diagnostic report
 * to the pre-filled GitHub issue (issue #0014).
 *
 * Decision O1/U6: no telemetry. This fetches a server-redacted snapshot,
 * composes a GitHub new-issue URL with the fields pre-filled (via {@link
 * buildIssueUrl}), and opens the user's browser. Nothing is sent automatically —
 * the user reviews and submits the issue themselves. When the URL had to elide
 * the diagnostics field to fit the length cap, the full report (via {@link
 * buildClipboardReport}) is copied to the clipboard and the issue body says so.
 *
 * The Rust binding's `DiagnosticReport` is snake_case (serde); the pure builder
 * in `issueReport.ts` uses the camelCase {@link DiagnosticReport} shape. The
 * mapping lives here, at the boundary.
 */
import { commands } from "../ipc/client";
import type { DiagnosticReport as BindingDiagnosticReport } from "../ipc/bindings";
import {
  buildIssueUrl,
  buildClipboardReport,
  type DiagnosticReport,
} from "./issueReport";

/** Map the snake_case Rust binding onto the camelCase issue-report shape. */
export function fromBinding(report: BindingDiagnosticReport): DiagnosticReport {
  return {
    appVersion: report.app_version,
    platform: report.platform,
    gpu: report.gpu,
    errorClass: report.error_class,
    logExcerpt: report.log_excerpt,
    backtrace: report.backtrace ?? null,
  };
}

/** The outcome surfaced to the caller (for an optional status message). */
export type ReportOutcome = {
  /** The URL opened in the browser. */
  url: string;
  /** True when the diagnostics field was elided and the full report copied. */
  copiedToClipboard: boolean;
};

/** Open the browser at `url`. Uses tauri-plugin-opener; the dynamic import keeps
 * the plugin out of the dev-shim/browser-preview path. */
async function openInBrowser(url: string): Promise<void> {
  const { openUrl } = await import("@tauri-apps/plugin-opener");
  await openUrl(url);
}

/**
 * Run the full "Report a problem" flow for an optional error class override
 * (e.g. the recording start-failed message), opening the browser on the
 * pre-filled issue.
 *
 * `errorClass`, when given, replaces the report's own error class so the issue
 * title/body name the surface the user clicked from. The diagnostic snapshot
 * (version / platform / GPU / redacted log excerpt) always comes from Rust.
 *
 * On an elided URL the full report is written to the clipboard first (so the
 * user can paste it into the issue's diagnostics field, as the body instructs),
 * then the browser is opened. A clipboard failure is non-fatal — the (elided)
 * URL still opens.
 */
export async function reportProblem(
  errorClass?: string,
): Promise<ReportOutcome> {
  const result = await commands.getDiagnosticReport();
  if (result.status === "error") {
    throw new Error(
      typeof result.error === "string"
        ? result.error
        : JSON.stringify(result.error),
    );
  }
  const report = fromBinding(result.data);
  if (errorClass && errorClass.trim().length > 0) {
    report.errorClass = errorClass.trim();
  }

  const { url, elided } = buildIssueUrl(report);

  let copiedToClipboard = false;
  if (elided) {
    // The diagnostics field was shortened to fit the URL; carry the full report
    // via the clipboard so nothing is lost. Best-effort.
    try {
      await navigator.clipboard.writeText(buildClipboardReport(report));
      copiedToClipboard = true;
    } catch {
      copiedToClipboard = false;
    }
  }

  await openInBrowser(url);
  return { url, copiedToClipboard };
}
