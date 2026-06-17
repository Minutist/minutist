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

vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn(),
  deleteMeeting: vi.fn(),
  reprocess: vi.fn().mockResolvedValue(undefined),
  setSpeakerName: vi.fn().mockResolvedValue({}),
}));

import { TranscriptPane, formatTimestamp } from "../transcript/TranscriptPane";
import { useRecordingStore } from "../state/recording";
import { useCrossRefStore } from "../state/cross-ref";
import { useMeetingsStore } from "../state/meetings";
import { useOperationProgressStore } from "../state/operation-progress";
import * as meetingsIpc from "../ipc/meetings";
import { TRANSCRIPT_SEGMENT_MIME } from "../editor/transcript-dnd";
import type { MeetingState } from "../state/meetings";
import type { Segment } from "../ipc/bindings";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeSegment(start_ms: number, text: string): Segment {
  return { start_ms, end_ms: start_ms + 1000, text, words: [], shared_speakers: [] };
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
      words: [], shared_speakers: [],
    };
    act(() => {
      useRecordingStore.setState({
        transcript: [makeSegment(0, "first"), speakerSegment],
      });
    });
    render(<TranscriptPane />);

    const rows = screen.getAllByRole("listitem");
    // The timestamp is the drag handle (the row text stays selectable), so the
    // dragstart fires on it, not the row.
    const handle = rows[1].querySelector(".transcript-pane__timestamp")!;
    const dataTransfer = makeDataTransfer();
    fireEvent.dragStart(handle, { dataTransfer });

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
    const handle = rows[0].querySelector(".transcript-pane__timestamp")!;
    const dataTransfer = makeDataTransfer();
    fireEvent.dragStart(handle, { dataTransfer });

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

// ---------------------------------------------------------------------------
// #67 — the Reprocess action toolbar at the top of the transcript pane (moved
// out of the meeting heading). #0015 merged the former Re-transcribe +
// Re-identify-speakers actions into one Reprocess button (re-transcribe then
// re-diarize under a single backend claim).
// ---------------------------------------------------------------------------

describe("TranscriptPane action toolbar (#67)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" }, transcript: [] });
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
      useOperationProgressStore.setState({ operations: {} });
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("hides the toolbar on the live recording (no saved meeting open)", () => {
    act(() => {
      useRecordingStore.setState({
        state: {
          kind: "recording",
          meeting_id: "live-1",
          started_at_ms: Date.now(),
        },
        transcript: [makeSegment(0, "live")],
      });
      useMeetingsStore.setState({ openMeetingId: null });
    });
    render(<TranscriptPane />);
    expect(
      screen.queryByRole("toolbar", { name: "Transcript actions" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Reprocess" }),
    ).not.toBeInTheDocument();
  });

  it("shows the toolbar with the Reprocess action when a saved meeting is open + idle", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "saved-1",
        openMeetingState: { transcript: [] } as unknown as MeetingState,
      });
    });
    render(<TranscriptPane />);
    expect(
      screen.getByRole("toolbar", { name: "Transcript actions" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Reprocess" }),
    ).toBeInTheDocument();
  });

  it("Reprocess invokes the meetings-store reprocess seam with the open meeting id", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "saved-2",
        openMeetingState: { transcript: [] } as unknown as MeetingState,
      });
    });
    render(<TranscriptPane />);

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Reprocess" }));
    });
    expect(meetingsIpc.reprocess).toHaveBeenCalledWith("saved-2");
  });

  it("disables Reprocess and shows the in-flight label while the re-transcribe phase runs", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "saved-3",
        openMeetingState: { transcript: [] } as unknown as MeetingState,
      });
      // The reprocess op surfaces progress under its re-transcribe phase.
      useOperationProgressStore.setState({
        operations: {
          "saved-3": {
            op: "re_transcribe",
            fraction: 0.4,
            label: "Re-transcribing…",
          },
        },
      });
    });
    render(<TranscriptPane />);
    const button = screen.getByRole("button", { name: "Reprocessing…" });
    expect(button).toBeDisabled();
    expect(
      screen.queryByRole("button", { name: "Reprocess" }),
    ).not.toBeInTheDocument();
  });

  it("disables Reprocess while the re-diarize phase of the same op runs", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "saved-4",
        openMeetingState: { transcript: [] } as unknown as MeetingState,
      });
      // The merged op surfaces progress under its re-diarize phase too; the
      // button must stay disabled for either phase.
      useOperationProgressStore.setState({
        operations: {
          "saved-4": {
            op: "rediarize",
            fraction: null,
            label: "Identifying speakers…",
          },
        },
      });
    });
    render(<TranscriptPane />);
    expect(
      screen.getByRole("button", { name: "Reprocessing…" }),
    ).toBeDisabled();
  });
});

// ---------------------------------------------------------------------------
// Speaker chip rename (issue #0001)
// ---------------------------------------------------------------------------

/**
 * Build a two-segment meeting state with two speakers so SpeakerChip
 * renders for both (the chip shows only at speaker-run boundaries).
 */
function makeSpeakerSegments(): Segment[] {
  return [
    {
      start_ms: 0,
      end_ms: 2_000,
      text: "Hello from A.",
      speaker_id: "A",
      words: [], shared_speakers: [],
    },
    {
      start_ms: 2_000,
      end_ms: 4_000,
      text: "Hello from B.",
      speaker_id: "B",
      words: [], shared_speakers: [],
    },
  ];
}

describe("SpeakerChip rename (issue #0001)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Return the updated map from the mock so the store can reflect it.
    (meetingsIpc.setSpeakerName as ReturnType<typeof vi.fn>).mockResolvedValue(
      { A: "Alice" },
    );
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" } });
      useMeetingsStore.setState({
        openMeetingId: "saved-chip",
        openMeetingState: {
          transcript: makeSpeakerSegments(),
          meta: { speaker_names: {} },
        } as unknown as MeetingState,
      });
    });
  });

  afterEach(() => {
    cleanup();
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  it("renders speaker chips as buttons (click-to-rename) on a saved idle meeting", () => {
    render(<TranscriptPane />);
    // Each speaker-run boundary gets a chip button; query by visible text.
    expect(screen.getByRole("button", { name: "Speaker A" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Speaker B" })).toBeInTheDocument();
    // Both chip buttons carry the rename tooltip.
    const titleEls = screen.getAllByTitle("Click to rename this speaker");
    expect(titleEls.length).toBeGreaterThanOrEqual(2);
  });

  it("shows the display name from speaker_names when one is set", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "saved-chip2",
        openMeetingState: {
          transcript: makeSpeakerSegments(),
          meta: { speaker_names: { A: "Alice" } },
        } as unknown as MeetingState,
      });
    });
    render(<TranscriptPane />);
    expect(screen.getByRole("button", { name: "Alice" })).toBeInTheDocument();
    // Raw label must not appear when a display name is set.
    expect(screen.queryByText("Speaker A")).not.toBeInTheDocument();
  });

  it("opens an inline input when the chip is clicked", () => {
    render(<TranscriptPane />);
    const chipA = screen.getByRole("button", { name: "Speaker A" });
    act(() => {
      fireEvent.click(chipA);
    });
    // The button is replaced by an inline input.
    expect(screen.getByRole("textbox")).toBeInTheDocument();
  });

  it("calls setSpeakerName and closes the input on Enter", async () => {
    render(<TranscriptPane />);
    const chipA = screen.getByRole("button", { name: "Speaker A" });
    act(() => {
      fireEvent.click(chipA);
    });
    const input = screen.getByRole("textbox");
    act(() => {
      fireEvent.change(input, { target: { value: "Alice" } });
      fireEvent.keyDown(input, { key: "Enter" });
    });
    // setSpeakerName must have been called (via the store's setSpeakerName action).
    expect(meetingsIpc.setSpeakerName).toHaveBeenCalled();
    // The input should no longer be visible after commit.
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("cancels the rename and restores the chip on Escape", () => {
    render(<TranscriptPane />);
    const chipA = screen.getByRole("button", { name: "Speaker A" });
    act(() => {
      fireEvent.click(chipA);
    });
    const input = screen.getByRole("textbox");
    act(() => {
      fireEvent.change(input, { target: { value: "Ignored name" } });
      fireEvent.keyDown(input, { key: "Escape" });
    });
    // The IPC call must NOT have been made.
    expect(meetingsIpc.setSpeakerName).not.toHaveBeenCalled();
    // The chip button is restored.
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Speaker A" })).toBeInTheDocument();
  });

  it("renders chips as read-only spans (not buttons) during a live recording", () => {
    act(() => {
      useRecordingStore.setState({
        state: {
          kind: "recording",
          meeting_id: "live",
          started_at_ms: Date.now(),
        },
        transcript: makeSpeakerSegments(),
      });
      useMeetingsStore.setState({
        openMeetingId: null,
        openMeetingState: null,
      });
    });
    render(<TranscriptPane />);
    // Speaker label renders, but as a non-interactive span — no rename button.
    expect(screen.getByText("Speaker A")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Speaker A" })).not.toBeInTheDocument();
  });
});
