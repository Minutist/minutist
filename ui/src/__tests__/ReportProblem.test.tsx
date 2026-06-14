/**
 * "Report a problem" wiring tests (issue #0014, part 2).
 *
 * Part 1's `IssueReport.test.ts` already covers the pure URL builder (field
 * construction, the length-cap + explicit elision, the clipboard fallback, and
 * redaction). This file covers what part 2 adds: the binding→camelCase mapping,
 * the `reportProblem` flow (fetch → build URL → open browser → clipboard
 * fallback on an elided URL → error-class override), the store action, and the
 * surface buttons (About row + the main-window error pane). The IPC + opener +
 * clipboard are mocked at their seams per the automated-testing policy.
 *
 * Default-suite test: no model, GPU, or microphone.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

// Hoisted mock fns so the `vi.mock` factories (hoisted to top of file) can
// reference them without a TDZ error.
const { openUrl, getDiagnosticReport } = vi.hoisted(() => ({
  openUrl: vi.fn().mockResolvedValue(undefined),
  getDiagnosticReport: vi.fn(),
}));

// The opener plugin — assert the URL we open.
vi.mock("@tauri-apps/plugin-opener", () => ({ openUrl }));

// The client seam: `reportProblem` calls `commands.getDiagnosticReport`.
vi.mock("../ipc/client", () => ({
  commands: { getDiagnosticReport },
}));

import { fromBinding, reportProblem } from "../diagnostics/reportProblem";
import { useReportProblemStore } from "../state/report-problem";
import { ISSUE_REPO } from "../diagnostics/issueReport";
import type { DiagnosticReport as BindingReport } from "../ipc/bindings";

const SAMPLE: BindingReport = {
  app_version: "0.0.0",
  platform: "Linux / x86_64 / connected",
  gpu: "cpu (mode=Auto)",
  error_class: "diagnostic report",
  log_excerpt: "INFO app-main: started\nWARN registry: retrying",
  backtrace: null,
};

function resetStore() {
  act(() => {
    useReportProblemStore.setState({
      reporting: false,
      status: null,
      webviewError: null,
    });
  });
}

describe("fromBinding", () => {
  it("maps snake_case binding fields onto the camelCase report shape", () => {
    const mapped = fromBinding({
      ...SAMPLE,
      backtrace: "0: panic",
    });
    expect(mapped).toEqual({
      appVersion: "0.0.0",
      platform: "Linux / x86_64 / connected",
      gpu: "cpu (mode=Auto)",
      errorClass: "diagnostic report",
      logExcerpt: "INFO app-main: started\nWARN registry: retrying",
      backtrace: "0: panic",
    });
  });

  it("normalises an absent backtrace to null", () => {
    expect(fromBinding(SAMPLE).backtrace).toBeNull();
  });
});

describe("reportProblem", () => {
  beforeEach(() => {
    openUrl.mockClear();
    getDiagnosticReport.mockReset();
  });

  it("fetches the report and opens the browser at the pre-filled issue", async () => {
    getDiagnosticReport.mockResolvedValue({ status: "ok", data: SAMPLE });
    const outcome = await reportProblem();
    expect(getDiagnosticReport).toHaveBeenCalledOnce();
    expect(openUrl).toHaveBeenCalledOnce();
    const url = openUrl.mock.calls[0][0] as string;
    expect(url.startsWith(`https://github.com/${ISSUE_REPO}/issues/new?`)).toBe(true);
    const q = new URL(url).searchParams;
    expect(q.get("app-version")).toBe("0.0.0");
    expect(q.get("diagnostics")).toContain("started");
    expect(outcome.copiedToClipboard).toBe(false);
  });

  it("overrides the error class with the caller's surface label", async () => {
    getDiagnosticReport.mockResolvedValue({ status: "ok", data: SAMPLE });
    await reportProblem("model load failed");
    const url = openUrl.mock.calls[0][0] as string;
    expect(new URL(url).searchParams.get("title")).toBe("[bug] model load failed");
  });

  it("copies the full report to the clipboard when the URL is elided", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    // A huge excerpt forces the URL cap to elide the diagnostics field.
    getDiagnosticReport.mockResolvedValue({
      status: "ok",
      data: { ...SAMPLE, log_excerpt: "x".repeat(50_000) },
    });
    const outcome = await reportProblem();
    expect(writeText).toHaveBeenCalledOnce();
    expect(writeText.mock.calls[0][0]).toContain("Minutist version: 0.0.0");
    expect(outcome.copiedToClipboard).toBe(true);
    // The (elided) URL still opens.
    expect(openUrl).toHaveBeenCalledOnce();
  });

  it("throws when the command returns an error", async () => {
    getDiagnosticReport.mockResolvedValue({
      status: "error",
      error: { code: "internal", context: "boom" },
    });
    await expect(reportProblem()).rejects.toThrow();
    expect(openUrl).not.toHaveBeenCalled();
  });
});

describe("useReportProblemStore", () => {
  beforeEach(() => {
    resetStore();
    openUrl.mockClear();
    getDiagnosticReport.mockReset();
  });

  it("report() sets a success status and clears reporting", async () => {
    getDiagnosticReport.mockResolvedValue({ status: "ok", data: SAMPLE });
    await act(async () => {
      await useReportProblemStore.getState().report();
    });
    const s = useReportProblemStore.getState();
    expect(s.reporting).toBe(false);
    expect(s.status).toContain("Opened your browser");
  });

  it("report() surfaces a failure status", async () => {
    getDiagnosticReport.mockResolvedValue({
      status: "error",
      error: "nope",
    });
    await act(async () => {
      await useReportProblemStore.getState().report();
    });
    expect(useReportProblemStore.getState().status).toContain("Could not open");
  });

  it("captureWebviewError + clearWebviewError track the latest webview error", () => {
    act(() => useReportProblemStore.getState().captureWebviewError("boom"));
    expect(useReportProblemStore.getState().webviewError).toBe("boom");
    act(() => useReportProblemStore.getState().clearWebviewError());
    expect(useReportProblemStore.getState().webviewError).toBeNull();
  });
});

describe("About — Report a problem row", () => {
  beforeEach(() => {
    resetStore();
    openUrl.mockClear();
    getDiagnosticReport.mockReset();
    getDiagnosticReport.mockResolvedValue({ status: "ok", data: SAMPLE });
  });

  it("renders the no-meeting-content note and opens the report on click", async () => {
    // About reads the models store; an empty list is fine for this surface.
    const { About } = await import("../shell/About");
    render(<About onClose={() => {}} />);
    // The privacy note is part of the row copy.
    expect(
      screen.getByText(/never your meeting content/i),
    ).toBeInTheDocument();
    const buttons = screen.getAllByRole("button", { name: /report a problem/i });
    await act(async () => {
      fireEvent.click(buttons[buttons.length - 1]);
    });
    await waitFor(() => expect(openUrl).toHaveBeenCalledOnce());
  });
});
