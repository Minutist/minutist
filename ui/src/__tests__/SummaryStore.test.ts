/**
 * Tests for the summary store (Phase 5, FR-30).
 *
 * Asserts:
 *   - `summarise` invokes `summarise_meeting` and enters the in-progress state,
 *   - a `summary_ready` event re-reads the summary (via `get_summary`) and
 *     leaves the in-progress state,
 *   - `save` persists via `save_summary` and reflects the edit optimistically,
 *   - `read` loads the persisted summary.
 *
 * The IPC calls are mocked at the `../ipc/summary` seam (per the architecture
 * testing policy — do not fake the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../ipc/summary", () => ({
  summariseMeeting: vi.fn().mockResolvedValue(undefined),
  getSummary: vi.fn().mockResolvedValue(null),
  saveSummary: vi.fn().mockResolvedValue(undefined),
}));

import { summariseMeeting, getSummary, saveSummary } from "../ipc/summary";
import { useSummaryStore } from "../state/summary";
import type { AppEvent } from "../ipc/app-event";

const MEETING = "meeting-0001";

function resetStore() {
  useSummaryStore.setState({
    summaryMarkdown: null,
    summarising: false,
    meetingId: null,
    lastError: null,
  });
}

describe("useSummaryStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("summarise invokes summarise_meeting and enters in-progress", async () => {
    // A pending promise so the in-progress state is observable before resolve.
    let resolveCall: () => void = () => {};
    vi.mocked(summariseMeeting).mockReturnValueOnce(
      new Promise<void>((res) => {
        resolveCall = res;
      }),
    );

    const promise = useSummaryStore.getState().summarise(MEETING);

    // In-progress immediately after dispatch.
    expect(useSummaryStore.getState().summarising).toBe(true);
    expect(useSummaryStore.getState().meetingId).toBe(MEETING);
    expect(summariseMeeting).toHaveBeenCalledWith(MEETING);

    resolveCall();
    await promise;
    // Still in-progress: the `summary_ready` event clears it, not the resolve.
    expect(useSummaryStore.getState().summarising).toBe(true);
  });

  it("a summary_ready event re-reads the summary and leaves in-progress", async () => {
    vi.mocked(getSummary).mockResolvedValueOnce("## Summary\n\nDone.");
    useSummaryStore.setState({ summarising: true, meetingId: MEETING });

    const event: AppEvent = { kind: "summary_ready", meeting_id: MEETING };
    useSummaryStore.getState().handleEvent(event);

    await vi.waitFor(() => {
      expect(getSummary).toHaveBeenCalledWith(MEETING);
      expect(useSummaryStore.getState().summaryMarkdown).toBe(
        "## Summary\n\nDone.",
      );
      expect(useSummaryStore.getState().summarising).toBe(false);
    });
  });

  it("ignores a summary_ready for a different meeting", () => {
    useSummaryStore.setState({ summarising: true, meetingId: MEETING });
    useSummaryStore
      .getState()
      .handleEvent({ kind: "summary_ready", meeting_id: "other-meeting" });
    expect(getSummary).not.toHaveBeenCalled();
    expect(useSummaryStore.getState().summarising).toBe(true);
  });

  it("save persists via save_summary and reflects the edit", async () => {
    await useSummaryStore.getState().save(MEETING, "edited summary");
    expect(saveSummary).toHaveBeenCalledWith(MEETING, "edited summary");
    expect(useSummaryStore.getState().summaryMarkdown).toBe("edited summary");
  });

  it("read loads the persisted summary", async () => {
    vi.mocked(getSummary).mockResolvedValueOnce("loaded");
    await useSummaryStore.getState().read(MEETING);
    expect(getSummary).toHaveBeenCalledWith(MEETING);
    expect(useSummaryStore.getState().summaryMarkdown).toBe("loaded");
    expect(useSummaryStore.getState().meetingId).toBe(MEETING);
  });

  it("surfaces an error when summarise dispatch rejects", async () => {
    vi.mocked(summariseMeeting).mockRejectedValueOnce(new Error("boom"));
    await useSummaryStore.getState().summarise(MEETING);
    expect(useSummaryStore.getState().summarising).toBe(false);
    expect(useSummaryStore.getState().lastError).toBe("boom");
  });
});
