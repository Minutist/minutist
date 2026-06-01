/**
 * Tests for the meeting-list view (FR-33).
 *
 * Asserts:
 *   - rows render with title / date / duration / speaker-count / excerpt,
 *   - the open / rename / delete / re-transcribe / re-summarise actions fire the
 *     corresponding meetings-store calls (which route through the mocked
 *     `../ipc/meetings` seam — per the architecture testing policy, the seam is
 *     mocked, not the generated bindings file).
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
  reTranscribe: vi.fn().mockResolvedValue(undefined),
  reSummarise: vi.fn().mockResolvedValue(undefined),
}));

import { MeetingList, formatDuration } from "../shell/MeetingList";
import { useMeetingsStore } from "../state/meetings";
import * as meetingsIpc from "../ipc/meetings";

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

  it("Re-transcribe fires re_transcribe for the row's meeting", async () => {
    await renderList();
    const buttons = screen.getAllByRole("button", { name: "Re-transcribe" });
    act(() => {
      fireEvent.click(buttons[0]);
    });
    await waitFor(() =>
      expect(meetingsIpc.reTranscribe).toHaveBeenCalledWith("meeting-0001"),
    );
  });

  it("Re-summarise fires re_summarise for the row's meeting", async () => {
    await renderList();
    const buttons = screen.getAllByRole("button", { name: "Re-summarise" });
    act(() => {
      fireEvent.click(buttons[0]);
    });
    await waitFor(() =>
      expect(meetingsIpc.reSummarise).toHaveBeenCalledWith("meeting-0001"),
    );
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
