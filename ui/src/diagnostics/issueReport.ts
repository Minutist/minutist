/**
 * "Report a problem" — build a GitHub new-issue URL pre-filled from a local
 * diagnostic report (issue #0014).
 *
 * Decision O1/U6: no telemetry. The app never sends anything; it opens GitHub's
 * issue form in the user's browser with fields pre-populated so the user reviews
 * and submits it themselves. This module is the pure URL/clipboard builder.
 *
 * Privacy (binding): the {@link DiagnosticReport} type carries ONLY version,
 * platform, error-class, and a server-redacted log excerpt — there is no field
 * for meeting content (transcripts, notes, titles, speaker names). Redaction of
 * the log excerpt itself happens in Rust where the data lives; {@link
 * redactMeetingPaths} is a defensive boundary pass for any UUID-bearing path
 * that slips through.
 */

/** The public repository the issue form lives in. */
export const ISSUE_REPO = "Minutist/minutist";

/** GitHub issue-form template file (`.github/ISSUE_TEMPLATE/bug-report.yml`). */
export const ISSUE_TEMPLATE = "bug-report.yml";

/**
 * Practical URL length cap. Browsers and GitHub tolerate well beyond this, but
 * ~8 KB is the safe ceiling; past it we elide the diagnostics field and steer
 * the user to the clipboard fallback rather than risk a dropped/!truncated URL.
 */
export const URL_CAP = 8000;

/**
 * A redacted diagnostic snapshot. Structured fields only — by construction it
 * holds no meeting content. `logExcerpt` / `backtrace` are already redacted by
 * the Rust side that assembles them.
 */
export type DiagnosticReport = {
  appVersion: string;
  /** OS / arch / build, e.g. "Windows 11 / x86_64 / connected". */
  platform: string;
  /** Resolved GPU plan (backend or CPU fallback). */
  gpu: string;
  /** Short error class, e.g. "panic" or "model load failed". */
  errorClass: string;
  /** Recent log lines (server-redacted). */
  logExcerpt: string;
  /** Backtrace for a crash report; absent for a non-crash error. */
  backtrace?: string | null;
};

const MEETING_ID_RE =
  /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;

/**
 * Replace anything shaped like a meeting-id UUID with a placeholder. Defensive
 * boundary pass — a meeting id in a logged path is the one piece of meeting
 * identity that can leak through an otherwise content-free log line.
 */
export function redactMeetingPaths(text: string): string {
  return text.replace(MEETING_ID_RE, "<redacted-id>");
}

/** The full human-readable diagnostic block (clipboard fallback carries this). */
export function buildDiagnosticsBlock(report: DiagnosticReport): string {
  const parts = [
    `Error class: ${redactMeetingPaths(report.errorClass)}`,
    "",
    "Recent log lines:",
    redactMeetingPaths(report.logExcerpt).trimEnd(),
  ];
  if (report.backtrace) {
    parts.push("", "Backtrace:", redactMeetingPaths(report.backtrace).trimEnd());
  }
  return parts.join("\n");
}

/** The complete report a user pastes when the URL was too long for the field. */
export function buildClipboardReport(report: DiagnosticReport): string {
  return [
    `Minutist version: ${report.appVersion}`,
    `Platform: ${report.platform}`,
    `GPU: ${report.gpu}`,
    "",
    buildDiagnosticsBlock(report),
  ].join("\n");
}

const TRUNCATION_NOTE =
  "\n\n[diagnostic report truncated for the URL — the full report was copied " +
  "to your clipboard; paste it above]";

function composeUrl(params: URLSearchParams): string {
  return `https://github.com/${ISSUE_REPO}/issues/new?${params.toString()}`;
}

/**
 * Build the pre-filled new-issue URL. Returns `elided: true` when the
 * diagnostics field had to be shortened to fit {@link URL_CAP}; the caller then
 * offers the clipboard fallback (which carries the full report). Never silently
 * truncates — an elided URL says so in the field and the caller surfaces it.
 */
export function buildIssueUrl(report: DiagnosticReport): {
  url: string;
  elided: boolean;
} {
  const base = (diagnostics: string) => {
    const params = new URLSearchParams();
    params.set("template", ISSUE_TEMPLATE);
    params.set("title", `[bug] ${redactMeetingPaths(report.errorClass)}`);
    params.set("app-version", report.appVersion);
    params.set("platform", report.platform);
    params.set("gpu", report.gpu);
    params.set("diagnostics", diagnostics);
    return composeUrl(params);
  };

  const full = base(buildDiagnosticsBlock(report));
  if (full.length <= URL_CAP) {
    return { url: full, elided: false };
  }

  // Shrink the diagnostics field until the whole URL fits, keeping a head of the
  // block plus an explicit note. Binary-search-free: the non-diagnostics part is
  // small, so compute the budget directly.
  const withEmpty = base("");
  const overhead = withEmpty.length; // URL with an empty diagnostics value
  const noteEncodedLen = encodeURIComponent(TRUNCATION_NOTE).length;
  // Rough budget for the kept excerpt (encoded). Leave a margin for encoding
  // expansion of multi-byte chars by halving the raw budget.
  const rawBudget = Math.max(0, Math.floor((URL_CAP - overhead - noteEncodedLen) / 2));
  let kept = buildDiagnosticsBlock(report).slice(0, rawBudget);
  let url = base(kept + TRUNCATION_NOTE);
  // Final safety: if still over (pathological multi-byte encoding), halve the
  // kept excerpt progressively until it fits.
  while (url.length > URL_CAP && kept.length > 0) {
    kept = kept.slice(0, Math.floor(kept.length / 2));
    url = base(kept + TRUNCATION_NOTE);
  }
  return { url, elided: true };
}
