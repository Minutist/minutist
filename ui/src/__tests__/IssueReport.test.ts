/**
 * Tests for the "Report a problem" URL builder (issue #0014).
 *
 * Covers URL construction (the issue-form template + pre-filled field ids),
 * the ~8 KB length cap with explicit (never silent) elision, the clipboard
 * fallback carrying the full report, and the meeting-id redaction boundary.
 */
import { describe, it, expect } from "vitest";
import {
  buildIssueUrl,
  buildClipboardReport,
  redactMeetingPaths,
  ISSUE_REPO,
  ISSUE_TEMPLATE,
  URL_CAP,
  type DiagnosticReport,
} from "../diagnostics/issueReport";

const report: DiagnosticReport = {
  appVersion: "0.1.0",
  platform: "Windows 11 / x86_64 / connected",
  gpu: "vulkan (n_gpu_layers=999)",
  errorClass: "model load failed",
  logExcerpt: "ERROR app-main: failed to open model\nWARN registry: retrying",
  backtrace: null,
};

describe("buildIssueUrl", () => {
  it("targets the public repo issue form with the pre-filled field ids", () => {
    const { url, elided } = buildIssueUrl(report);
    expect(elided).toBe(false);
    expect(url.startsWith(`https://github.com/${ISSUE_REPO}/issues/new?`)).toBe(
      true,
    );
    const q = new URL(url).searchParams;
    expect(q.get("template")).toBe(ISSUE_TEMPLATE);
    expect(q.get("title")).toBe("[bug] model load failed");
    expect(q.get("app-version")).toBe("0.1.0");
    expect(q.get("platform")).toBe("Windows 11 / x86_64 / connected");
    expect(q.get("gpu")).toBe("vulkan (n_gpu_layers=999)");
    expect(q.get("diagnostics")).toContain("failed to open model");
  });

  it("carries no field for meeting content (only the structured fields)", () => {
    const { url } = buildIssueUrl(report);
    const keys = [...new URL(url).searchParams.keys()].sort();
    expect(keys).toEqual([
      "app-version",
      "diagnostics",
      "gpu",
      "platform",
      "template",
      "title",
    ]);
  });

  it("elides the diagnostics field and stays under the cap for a huge excerpt", () => {
    const huge: DiagnosticReport = {
      ...report,
      logExcerpt: "x".repeat(50_000),
    };
    const { url, elided } = buildIssueUrl(huge);
    expect(elided).toBe(true);
    expect(url.length).toBeLessThanOrEqual(URL_CAP);
    // Never silent: the truncation is announced and points at the clipboard.
    expect(decodeURIComponent(new URL(url).searchParams.get("diagnostics")!)).toContain(
      "truncated for the URL",
    );
  });
});

describe("buildClipboardReport", () => {
  it("carries the full report including version, platform, and the block", () => {
    const text = buildClipboardReport({
      ...report,
      backtrace: "0: minutist::panic\n1: core::ops",
    });
    expect(text).toContain("Minutist version: 0.1.0");
    expect(text).toContain("Platform: Windows 11 / x86_64 / connected");
    expect(text).toContain("Error class: model load failed");
    expect(text).toContain("failed to open model");
    expect(text).toContain("Backtrace:");
    expect(text).toContain("minutist::panic");
  });
});

describe("redactMeetingPaths", () => {
  it("replaces meeting-id UUIDs (the one identity that can leak via a path)", () => {
    const line =
      "ERROR persistence: write failed for " +
      "meetings/3f2504e0-4f89-41d3-9a0c-0305e82c3301/transcript.json";
    const out = redactMeetingPaths(line);
    expect(out).not.toContain("3f2504e0-4f89-41d3-9a0c-0305e82c3301");
    expect(out).toContain("<redacted-id>");
  });

  it("is applied to the URL diagnostics field", () => {
    const withId: DiagnosticReport = {
      ...report,
      logExcerpt: "ERROR path meetings/3f2504e0-4f89-41d3-9a0c-0305e82c3301/x",
    };
    const { url } = buildIssueUrl(withId);
    const diag = decodeURIComponent(new URL(url).searchParams.get("diagnostics")!);
    expect(diag).not.toContain("3f2504e0-4f89-41d3-9a0c-0305e82c3301");
    expect(diag).toContain("<redacted-id>");
  });
});
