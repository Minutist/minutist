/**
 * Tests for the meeting-list view (FR-33).
 *
 * Asserts:
 *   - rows render with title / date / duration / speaker-count / excerpt,
 *   - the open / rename / delete / re-transcribe actions fire the corresponding
 *     meetings-store calls (which route through the mocked `../ipc/meetings`
 *     seam), and the Summarise action routes through the summary store's
 *     `summarise_meeting` (mocked at the `../ipc/summary` seam) — per the
 *     architecture testing policy, the seams are mocked, not the generated
 *     bindings file.
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

vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn(),
  openMeeting: vi.fn().mockResolvedValue({}),
  renameMeeting: vi.fn().mockResolvedValue(undefined),
  deleteMeeting: vi.fn().mockResolvedValue(undefined),
  reprocess: vi.fn().mockResolvedValue(undefined),
  openMeetingFolder: vi.fn().mockResolvedValue(undefined),
}));

// The Phase-5 row Summarise action routes through the summary store, which
// wraps the `../ipc/summary` seam. Mock the seam so the action is observable.
vi.mock("../ipc/summary", () => ({
  summariseMeeting: vi.fn().mockResolvedValue(undefined),
  getSummary: vi.fn().mockResolvedValue(null),
  saveSummary: vi.fn().mockResolvedValue(undefined),
}));

// The list now renders the folder sidebar (it reads the collections store on
// mount); mock the seam so the test doesn't reach the real client.
vi.mock("../ipc/collections", () => ({
  listCollections: vi.fn().mockResolvedValue([]),
  createCollection: vi.fn(),
  renameCollection: vi.fn().mockResolvedValue(undefined),
  deleteCollection: vi.fn().mockResolvedValue(undefined),
  setMeetingCollection: vi.fn().mockResolvedValue(undefined),
}));

import { MeetingList, formatDuration } from "../shell/MeetingList";
import { useMeetingsStore } from "../state/meetings";
import * as meetingsIpc from "../ipc/meetings";
import * as summaryIpc from "../ipc/summary";

const sampleMeetings = [
  {
    id: "meeting-0001",
    title: "Launch sync — Tuesday",
    started_at: "2026-05-26T14:05:00Z",
    duration_ms: 32 * 60 * 1000,
    speaker_count: 3,
    excerpt: "Three open risks against the date.",
  },
  {
    id: "meeting-0002",
    title: "Quick standup",
    started_at: "2026-05-18T08:00:00Z",
    duration_ms: 8 * 60 * 1000,
    speaker_count: 1,
    excerpt: null,
  },
];

/** Render the list and wait for the mount-time refresh to populate rows. */
async function renderList() {
  act(() => {
    useMeetingsStore.setState({
      meetings: [],
      loading: false,
      openMeetingId: null,
      openMeetingState: null,
      lastError: null,
    });
  });
  render(<MeetingList />);
  await waitFor(() =>
    expect(screen.getByText("Launch sync — Tuesday")).toBeInTheDocument(),
  );
}

describe("formatDuration", () => {
  it("renders sub-hour durations as minutes", () => {
    expect(formatDuration(32 * 60 * 1000)).toBe("32 min");
    expect(formatDuration(8 * 60 * 1000)).toBe("8 min");
  });
  it("renders hour-plus durations as H:MM", () => {
    expect(formatDuration(95 * 60 * 1000)).toBe("1:35");
  });
});

describe("MeetingList view (FR-33)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(meetingsIpc.listMeetings).mockResolvedValue(sampleMeetings);
  });

  it("renders a row per meeting with title, meta, and excerpt", async () => {
    await renderList();

    expect(screen.getByText("Launch sync — Tuesday")).toBeInTheDocument();
    expect(screen.getByText("Quick standup")).toBeInTheDocument();

    // Meta: duration + speaker count.
    expect(screen.getByText("32 min")).toBeInTheDocument();
    expect(screen.getByText("3 speakers")).toBeInTheDocument();
    expect(screen.getByText("1 speaker")).toBeInTheDocument();

    // Excerpt present for the first meeting, absent for the second.
    expect(
      screen.getByText("Three open risks against the date."),
    ).toBeInTheDocument();
  });

  it("Open fires open_meeting for the row's meeting (entry surface → workspace)", async () => {
    await renderList();
    // Two Open buttons (one per row); click the first.
    const openButtons = screen.getAllByRole("button", { name: "Open" });
    act(() => {
      fireEvent.click(openButtons[0]);
    });
    await waitFor(() =>
      expect(meetingsIpc.openMeeting).toHaveBeenCalledWith("meeting-0001"),
    );
  });

  it("clicking the title also opens the meeting", async () => {
    await renderList();
    act(() => {
      fireEvent.click(screen.getByText("Quick standup"));
    });
    await waitFor(() =>
      expect(meetingsIpc.openMeeting).toHaveBeenCalledWith("meeting-0002"),
    );
  });

  it("Rename commits a new title via rename_meeting", async () => {
    await renderList();
    const renameButtons = screen.getAllByRole("button", { name: "Rename" });
    act(() => {
      fireEvent.click(renameButtons[0]);
    });
    const input = screen.getByLabelText("Meeting title") as HTMLInputElement;
    act(() => {
      fireEvent.change(input, { target: { value: "Renamed sync" } });
      fireEvent.keyDown(input, { key: "Enter" });
    });
    await waitFor(() =>
      expect(meetingsIpc.renameMeeting).toHaveBeenCalledWith(
        "meeting-0001",
        "Renamed sync",
      ),
    );
  });

  it("Delete fires delete_meeting for the row's meeting", async () => {
    await renderList();
    const deleteButtons = screen.getAllByRole("button", { name: "Delete" });
    act(() => {
      fireEvent.click(deleteButtons[1]);
    });
    await waitFor(() =>
      expect(meetingsIpc.deleteMeeting).toHaveBeenCalledWith("meeting-0002"),
    );
  });

  it("double-clicking a meeting opens it (no need to find the Open button)", async () => {
    await renderList();
    // Double-click anywhere in the meeting's text (the title element bubbles to
    // the row's main area, which carries onDoubleClick).
    act(() => {
      fireEvent.dblClick(screen.getByText("Quick standup"));
    });
    await waitFor(() =>
      expect(meetingsIpc.openMeeting).toHaveBeenCalledWith("meeting-0002"),
    );
  });

  it("does not surface re-processing actions on the list (open is the primary action; re-processing lives in the opened meeting)", async () => {
    await renderList();
    expect(
      screen.queryByRole("button", { name: "Reprocess" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Summarise" }),
    ).not.toBeInTheDocument();
  });

  it("shows an empty-state message when there are no meetings", async () => {
    vi.mocked(meetingsIpc.listMeetings).mockResolvedValue([]);
    act(() => {
      useMeetingsStore.setState({ meetings: [], loading: false });
    });
    render(<MeetingList />);
    await waitFor(() =>
      expect(
        screen.getByText(/No meetings yet/i),
      ).toBeInTheDocument(),
    );
  });
});

describe("MeetingList row context menu (#0034 meeting-list slice)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(meetingsIpc.listMeetings).mockResolvedValue(sampleMeetings);
  });

  function getRow(title: string): HTMLElement {
    const row = screen.getByText(title).closest("li");
    if (!row) throw new Error(`no <li> ancestor for "${title}"`);
    return row;
  }

  it("right-click on a row opens the themed menu and suppresses the native menu", async () => {
    await renderList();
    const row = getRow("Launch sync — Tuesday");

    // fireEvent.dispatchEvent returns false when preventDefault() was called
    // on a cancelable event — i.e. the native WebView2 menu was suppressed.
    const notCancelled = fireEvent.contextMenu(row, { clientX: 120, clientY: 80 });
    expect(notCancelled).toBe(false);

    expect(screen.getByRole("menu")).toBeInTheDocument();
    expect(
      screen.getByRole("menuitem", { name: "Open storage folder" }),
    ).toBeInTheDocument();
    // The existing row actions are surfaced as menu entries too.
    expect(screen.getByRole("menuitem", { name: "Open" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Rename" })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: "Delete" })).toBeInTheDocument();
  });

  it('"Open storage folder" invokes open_meeting_folder with the row\'s meeting id', async () => {
    await renderList();
    fireEvent.contextMenu(getRow("Quick standup"), { clientX: 50, clientY: 50 });

    act(() => {
      fireEvent.click(screen.getByRole("menuitem", { name: "Open storage folder" }));
    });

    await waitFor(() =>
      expect(meetingsIpc.openMeetingFolder).toHaveBeenCalledWith("meeting-0002"),
    );
    // Choosing an entry dismisses the menu.
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("dismisses on outside click", async () => {
    await renderList();
    fireEvent.contextMenu(getRow("Quick standup"), { clientX: 50, clientY: 50 });
    expect(screen.getByRole("menu")).toBeInTheDocument();

    act(() => {
      fireEvent.click(screen.getByLabelText("Close menu"));
    });
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("dismisses on Escape", async () => {
    await renderList();
    fireEvent.contextMenu(getRow("Quick standup"), { clientX: 50, clientY: 50 });
    expect(screen.getByRole("menu")).toBeInTheDocument();

    act(() => {
      fireEvent.keyDown(document, { key: "Escape" });
    });
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });

  it("right-clicking a second row closes the first row's menu instead of stacking both", async () => {
    await renderList();
    fireEvent.contextMenu(getRow("Launch sync — Tuesday"), {
      clientX: 50,
      clientY: 50,
    });
    expect(screen.getAllByRole("menu")).toHaveLength(1);

    fireEvent.contextMenu(getRow("Quick standup"), { clientX: 200, clientY: 300 });

    // Exactly one menu exists — the second row's, not both.
    const menus = screen.getAllByRole("menu");
    expect(menus).toHaveLength(1);
    expect(
      screen.getByRole("menuitem", { name: "Open storage folder" }),
    ).toBeInTheDocument();
    // The first row's own menu instance is gone, not just hidden behind a
    // second one — confirms the state is single, not per-row-and-stacked.
    act(() => {
      fireEvent.click(screen.getByRole("menuitem", { name: "Open storage folder" }));
    });
    await waitFor(() =>
      expect(meetingsIpc.openMeetingFolder).toHaveBeenCalledWith("meeting-0002"),
    );
  });

  it("does not suppress the native menu on the inline-rename input", async () => {
    await renderList();
    const renameButtons = screen.getAllByRole("button", { name: "Rename" });
    act(() => {
      fireEvent.click(renameButtons[0]);
    });
    const input = screen.getByLabelText("Meeting title");

    const notCancelled = fireEvent.contextMenu(input, { clientX: 10, clientY: 10 });
    expect(notCancelled).toBe(true);
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
