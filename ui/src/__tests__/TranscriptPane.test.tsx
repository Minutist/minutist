/**
 * Unit tests for TranscriptPane.
 *
 * Verifies empty-state copy and per-row rendering with correct MM:SS.cc
 * timestamp prefixes, plus the Phase-4 cross-reference interactions:
 * highlight range (FR-22), click-to-scroll (FR-23), drag source (FR-24), and
 * the active-transcript saved-meeting branch.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
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
import { useCrossRefStore } from "../state/cross-ref";
import { useMeetingsStore } from "../state/meetings";
import { TRANSCRIPT_SEGMENT_MIME } from "../editor/transcript-dnd";
import type { MeetingState } from "../state/meetings";
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

// ---------------------------------------------------------------------------
// Cross-reference interactions (FR-22 / FR-23 / FR-24)
// ---------------------------------------------------------------------------

/**
 * Build a minimal stand-in `DataTransfer` for a synthetic dragstart. jsdom does
 * not construct a real `DataTransfer`, so we record `setData` calls and expose
 * the written entries for assertions on the payload shape.
 */
function makeDataTransfer(): DataTransfer & { store: Map<string, string> } {
  const store = new Map<string, string>();
  return {
    store,
    setData: (type: string, value: string) => {
      store.set(type, value);
    },
    getData: (type: string) => store.get(type) ?? "",
    effectAllowed: "none",
  } as unknown as DataTransfer & { store: Map<string, string> };
}

describe("TranscriptPane cross-reference interactions", () => {
  const SEGMENTS: Segment[] = [
    makeSegment(0, "first"),
    makeSegment(5_000, "second"),
    makeSegment(10_000, "third"),
    makeSegment(15_000, "fourth"),
  ];

  beforeEach(() => {
    act(() => {
      useRecordingStore.setState({
        state: { kind: "idle" },
        transcript: SEGMENTS,
      });
      useCrossRefStore.setState({ highlightedRange: null, scrollRequest: null });
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  afterEach(() => {
    cleanup();
    act(() => {
      useCrossRefStore.setState({ highlightedRange: null, scrollRequest: null });
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  // FR-22 --------------------------------------------------------------------

  it("highlights exactly the rows in highlightedRange [startIndex, endIndex) and no others", () => {
    act(() => {
      // Half-open [1, 3): rows at index 1 and 2 only.
      useCrossRefStore.setState({
        highlightedRange: { startIndex: 1, endIndex: 3 },
      });
    });
    render(<TranscriptPane />);

    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(4);

    // Index 0: outside the range — not highlighted.
    expect(rows[0]).not.toHaveClass("transcript-pane__row--highlighted");
    expect(rows[0]).not.toHaveAttribute("aria-current");
    // Index 1, 2: inside the range — highlighted + aria-current.
    expect(rows[1]).toHaveClass("transcript-pane__row--highlighted");
    expect(rows[1]).toHaveAttribute("aria-current", "true");
    expect(rows[2]).toHaveClass("transcript-pane__row--highlighted");
    expect(rows[2]).toHaveAttribute("aria-current", "true");
    // Index 3 == endIndex (exclusive) — not highlighted.
    expect(rows[3]).not.toHaveClass("transcript-pane__row--highlighted");
    expect(rows[3]).not.toHaveAttribute("aria-current");
  });

  it("highlights no rows when highlightedRange is null", () => {
    act(() => {
      useCrossRefStore.setState({ highlightedRange: null });
    });
    render(<TranscriptPane />);

    for (const row of screen.getAllByRole("listitem")) {
      expect(row).not.toHaveClass("transcript-pane__row--highlighted");
      expect(row).not.toHaveAttribute("aria-current");
    }
  });

  // FR-23 --------------------------------------------------------------------

  it("publishes a scroll request with the clicked segment's start_ms on row click", () => {
    render(<TranscriptPane />);

    expect(useCrossRefStore.getState().scrollRequest).toBeNull();

    // Click the third row (index 2 → start_ms 10_000).
    const rows = screen.getAllByRole("listitem");
    act(() => {
      fireEvent.click(rows[2]);
    });

    const req = useCrossRefStore.getState().scrollRequest;
    expect(req).not.toBeNull();
    expect(req?.anchorMs).toBe(10_000);

    // A second click on a different row re-triggers with a fresh nonce so the
    // editor scroll effect fires again.
    const firstNonce = req?.nonce ?? 0;
    act(() => {
      fireEvent.click(rows[0]);
    });
    const req2 = useCrossRefStore.getState().scrollRequest;
    expect(req2?.anchorMs).toBe(0);
    expect(req2?.nonce).toBe(firstNonce + 1);
  });

  // FR-24 --------------------------------------------------------------------

  it("writes the dragged segment payload onto dataTransfer on dragstart", () => {
    const speakerSegment: Segment = {
      start_ms: 5_000,
      end_ms: 6_000,
      text: "second",
      speaker_id: "spk_2",
      words: [],
    };
    act(() => {
      useRecordingStore.setState({
        transcript: [makeSegment(0, "first"), speakerSegment],
      });
    });
    render(<TranscriptPane />);

    const rows = screen.getAllByRole("listitem");
    const dataTransfer = makeDataTransfer();
    fireEvent.dragStart(rows[1], { dataTransfer });

    // The private MIME type carries the JSON payload in the documented shape.
    const raw = dataTransfer.store.get(TRANSCRIPT_SEGMENT_MIME);
    expect(raw).toBeDefined();
    const payload = JSON.parse(raw as string);
    expect(payload).toEqual({
      startMs: 5_000,
      endMs: 6_000,
      speakerId: "spk_2",
      text: "second",
    });

    // A text/plain fallback carries the bare text; effectAllowed is "copy".
    expect(dataTransfer.store.get("text/plain")).toBe("second");
    expect(dataTransfer.effectAllowed).toBe("copy");
  });

  it("writes speakerId null when the dragged segment has no speaker_id", () => {
    render(<TranscriptPane />);
    const rows = screen.getAllByRole("listitem");
    const dataTransfer = makeDataTransfer();
    fireEvent.dragStart(rows[0], { dataTransfer });

    const payload = JSON.parse(
      dataTransfer.store.get(TRANSCRIPT_SEGMENT_MIME) as string,
    );
    expect(payload.speakerId).toBeNull();
    expect(payload.text).toBe("first");
  });

  // Saved-meeting branch -----------------------------------------------------

  it("renders the restored saved-meeting transcript when a meeting is open + idle", () => {
    const restored: Segment[] = [
      makeSegment(1_000, "restored alpha"),
      makeSegment(2_000, "restored beta"),
    ];
    const restoredState = {
      transcript: restored,
    } as unknown as MeetingState;

    act(() => {
      // Live store holds DIFFERENT segments — proving the pane reads the saved
      // source, not the live one, while a meeting is open and idle.
      useRecordingStore.setState({
        state: { kind: "idle" },
        transcript: [makeSegment(99_000, "live should not show")],
      });
      useMeetingsStore.setState({
        openMeetingId: "saved-0001",
        openMeetingState: restoredState,
      });
    });

    render(<TranscriptPane />);

    expect(screen.getByText("restored alpha")).toBeInTheDocument();
    expect(screen.getByText("restored beta")).toBeInTheDocument();
    expect(
      screen.queryByText("live should not show"),
    ).not.toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });
});
