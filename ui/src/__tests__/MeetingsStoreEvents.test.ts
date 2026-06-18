/**
 * Live-test UX T6 — the meetings-list store refreshes on `summary_ready`.
 *
 * When a summary is written the backend refreshes the index row so the
 * meeting-list excerpt becomes the summary blurb. The store must re-run
 * `list_meetings` on the `summary_ready` event so the row shows the blurb
 * without a manual reload. Also covers the `meeting_finalised` path, which now
 * OPENS the just-stopped meeting (the stay-on-meeting flow) rather than bouncing
 * to the list.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn(),
  deleteMeeting: vi.fn(),
  reprocess: vi.fn(),
}));

import { listMeetings, openMeeting } from "../ipc/meetings";
import { useMeetingsStore } from "../state/meetings";
import type { AppEvent } from "../ipc/bindings";

describe("meetings store — terminal-event refreshes (T6 / T3)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useMeetingsStore.setState({
      meetings: [],
      loading: false,
      openMeetingId: null,
      openMeetingState: null,
      lastError: null,
    });
  });

  it("summary_ready refreshes the meeting list (so the excerpt becomes the summary blurb)", async () => {
    const event: AppEvent = { kind: "summary_ready", meeting_id: "m1" };
    useMeetingsStore.getState().handleEvent(event);
    // refresh() is fire-and-forget inside handleEvent; let the microtask run.
    await Promise.resolve();
    expect(listMeetings).toHaveBeenCalledOnce();
  });

  it("meeting_finalised opens the just-stopped meeting (stays on it) and refreshes the list", async () => {
    const event: AppEvent = { kind: "meeting_finalised", meeting_id: "m1" };
    useMeetingsStore.getState().handleEvent(event);
    // `openMeetingId` is set SYNCHRONOUSLY (before the async open load) so the
    // workspace never flashes the list as the recorder goes idle.
    expect(useMeetingsStore.getState().openMeetingId).toBe("m1");
    await Promise.resolve();
    expect(openMeeting).toHaveBeenCalledWith("m1");
    expect(listMeetings).toHaveBeenCalledOnce();
  });
});
