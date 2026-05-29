/**
 * Unit tests for TranscriptPane.
 *
 * Verifies empty-state copy and per-row rendering with correct MM:SS.cc
 * timestamp prefixes.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { act } from "react";

// ---------------------------------------------------------------------------
// Tauri API mocks
// ---------------------------------------------------------------------------
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  Channel: vi.fn(),
}));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({
  WebviewWindow: vi.fn(),
}));

import { TranscriptPane, formatTimestamp } from "../transcript/TranscriptPane";
import { useRecordingStore } from "../state/recording";
import type { Segment } from "../ipc/bindings";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeSegment(start_ms: number, text: string): Segment {
  return { start_ms, end_ms: start_ms + 1000, text, words: [] };
}

// ---------------------------------------------------------------------------
// formatTimestamp unit tests
// ---------------------------------------------------------------------------

describe("formatTimestamp", () => {
  it("formats zero as 00:00.00", () => {
    expect(formatTimestamp(0)).toBe("00:00.00");
  });

  it("formats 1234 ms as 00:01.23", () => {
    expect(formatTimestamp(1234)).toBe("00:01.23");
  });

  it("formats 75400 ms as 01:15.40", () => {
    expect(formatTimestamp(75400)).toBe("01:15.40");
  });

  it("formats 3723456 ms as 62:03.45", () => {
    expect(formatTimestamp(3723456)).toBe("62:03.45");
  });
});

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("TranscriptPane", () => {
  beforeEach(() => {
    act(() => {
      useRecordingStore.setState({ transcript: [] });
    });
  });

  it("renders empty-state copy when there are no segments", () => {
    render(<TranscriptPane />);
    expect(
      screen.getByText("Transcript will appear here while you record."),
    ).toBeInTheDocument();
  });

  it("renders three rows in order with correct MM:SS.cc prefixes", () => {
    const segments: Segment[] = [
      makeSegment(0, "Hello world"),
      makeSegment(5000, "Second sentence"),
      makeSegment(61000, "Past the minute mark"),
    ];
    act(() => {
      useRecordingStore.setState({ transcript: segments });
    });

    render(<TranscriptPane />);

    // Timestamps
    expect(screen.getByText("00:00.00")).toBeInTheDocument();
    expect(screen.getByText("00:05.00")).toBeInTheDocument();
    expect(screen.getByText("01:01.00")).toBeInTheDocument();

    // Texts
    expect(screen.getByText("Hello world")).toBeInTheDocument();
    expect(screen.getByText("Second sentence")).toBeInTheDocument();
    expect(screen.getByText("Past the minute mark")).toBeInTheDocument();

    // Order: list items appear in DOM order.
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("00:00.00");
    expect(items[1]).toHaveTextContent("00:05.00");
    expect(items[2]).toHaveTextContent("01:01.00");
  });

  it("does not render the empty-state copy when segments exist", () => {
    act(() => {
      useRecordingStore.setState({ transcript: [makeSegment(0, "hi")] });
    });
    render(<TranscriptPane />);
    expect(
      screen.queryByText("Transcript will appear here while you record."),
    ).not.toBeInTheDocument();
  });
});
