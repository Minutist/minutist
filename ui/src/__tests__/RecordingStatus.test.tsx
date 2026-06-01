/**
 * Tests for the top-bar RecordingStatus affordance (Editorial Ink).
 *
 * Covers the `formatElapsed` helper and the component's state-driven label +
 * live-dot behaviour: idle shows "Ready" with no clock and no pulsing dot;
 * recording shows the "Recording" label, the live dot, and an elapsed clock
 * derived from `started_at_ms`.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { act } from "react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

import { RecordingStatus, formatElapsed } from "../shell/RecordingStatus";
import { useRecordingStore } from "../state/recording";
import type { RecordingState } from "../ipc/bindings";

function setState(state: RecordingState) {
  act(() => {
    useRecordingStore.setState({ state });
  });
}

describe("formatElapsed", () => {
  it("formats sub-minute durations as M:SS", () => {
    expect(formatElapsed(0)).toBe("0:00");
    expect(formatElapsed(9_000)).toBe("0:09");
    expect(formatElapsed(59_000)).toBe("0:59");
  });

  it("formats minute-to-hour durations as M:SS", () => {
    expect(formatElapsed(75_000)).toBe("1:15");
    expect(formatElapsed(600_000)).toBe("10:00");
  });

  it("formats hour-plus durations as H:MM:SS", () => {
    expect(formatElapsed(3_661_000)).toBe("1:01:01");
  });

  it("clamps negative input to zero", () => {
    expect(formatElapsed(-5_000)).toBe("0:00");
  });
});

describe("RecordingStatus", () => {
  beforeEach(() => {
    setState({ kind: "idle" });
  });

  it("shows the Ready label and no elapsed clock when idle", () => {
    render(<RecordingStatus />);
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("data-state", "idle");
    expect(screen.getByText("Ready")).toBeInTheDocument();
    // The live (pulsing) dot class is only present while recording.
    expect(
      status.querySelector(".recording-status__dot--live"),
    ).toBeNull();
  });

  it("shows the Recording label, the live dot, and an elapsed clock", () => {
    const nowSpy = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    setState({
      kind: "recording",
      meeting_id: "m1",
      started_at_ms: 1_000_000 - 65_000, // 65 s ago
    });
    render(<RecordingStatus />);

    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("data-state", "recording");
    expect(screen.getByText("Recording")).toBeInTheDocument();
    expect(
      status.querySelector(".recording-status__dot--live"),
    ).not.toBeNull();
    expect(screen.getByText("1:05")).toBeInTheDocument();

    nowSpy.mockRestore();
  });
});
