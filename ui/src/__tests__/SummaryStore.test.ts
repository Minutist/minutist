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
    autoPending: {},
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

  it("summary_queued marks the meeting auto-pending (busy)", () => {
    useSummaryStore
      .getState()
      .handleEvent({ kind: "summary_queued", meeting_id: MEETING });
    expect(useSummaryStore.getState().autoPending[MEETING]).toBe(true);
  });

  it("summary_unavailable clears the auto-pending marker (deferred/failed)", () => {
    useSummaryStore.setState({ autoPending: { [MEETING]: true } });
    useSummaryStore
      .getState()
      .handleEvent({ kind: "summary_unavailable", meeting_id: MEETING });
    expect(useSummaryStore.getState().autoPending[MEETING]).toBeUndefined();
  });

  it("summary_ready clears the auto-pending marker even for a non-loaded meeting", () => {
    // A backgrounded auto-summary for a meeting OTHER than the open one must
    // still clear its busy marker (it would otherwise spin forever on its pane).
    useSummaryStore.setState({
      meetingId: MEETING,
      autoPending: { "other-meeting": true },
    });
    useSummaryStore
      .getState()
      .handleEvent({ kind: "summary_ready", meeting_id: "other-meeting" });
    // The other meeting's marker is gone…
    expect(useSummaryStore.getState().autoPending["other-meeting"]).toBeUndefined();
    // …and the unrelated event did not re-read the open meeting's summary.
    expect(getSummary).not.toHaveBeenCalled();
  });

  it("save persists via save_summary and reflects the edit", async () => {
    await useSummaryStore.getState().save(MEETING, "edited summary");
    expect(saveSummary).toHaveBeenCalledWith(MEETING, "edited summary");
    expect(useSummaryStore.getState().summaryMarkdown).toBe("edited summary");
  });

  it("save rolls back the optimistic edit and sets lastError when save_summary rejects", async () => {
    // A persisted value is loaded before the failing edit.
    useSummaryStore.setState({ summaryMarkdown: "persisted", meetingId: MEETING });
    vi.mocked(saveSummary).mockRejectedValueOnce(new Error("disk full"));

    await useSummaryStore.getState().save(MEETING, "unsaved edit");

    expect(saveSummary).toHaveBeenCalledWith(MEETING, "unsaved edit");
    // The store must not retain the unsaved edit as if it persisted.
    expect(useSummaryStore.getState().summaryMarkdown).toBe("persisted");
    expect(useSummaryStore.getState().lastError).toBe("disk full");
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
