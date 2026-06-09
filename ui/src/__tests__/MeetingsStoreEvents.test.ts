/**
 * Live-test UX T6 — the meetings-list store refreshes on `summary_ready`.
 *
 * When a summary is written the backend refreshes the index row so the
 * meeting-list excerpt becomes the summary blurb. The store must re-run
 * `list_meetings` on the `summary_ready` event so the row shows the blurb
 * without a manual reload. Also re-asserts the existing `meeting_finalised`
 * refresh path so the T3 return-to-list flow is covered.
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
  reTranscribe: vi.fn(),
  rediarize: vi.fn(),
}));

import { listMeetings } from "../ipc/meetings";
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

  it("meeting_finalised refreshes the list (T3 return-to-list)", async () => {
    const event: AppEvent = { kind: "meeting_finalised", meeting_id: "m1" };
    useMeetingsStore.getState().handleEvent(event);
    await Promise.resolve();
    expect(listMeetings).toHaveBeenCalledOnce();
  });
});
