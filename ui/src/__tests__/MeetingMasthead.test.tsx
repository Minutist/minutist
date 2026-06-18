/**
 * Tests for the meeting-screen masthead (the open meeting's headline).
 *
 * Asserts:
 *   - `isDefaultMeetingTitle` recognises the orchestrator's auto title,
 *   - a named meeting renders its title + dateline (date · duration · speakers),
 *   - an auto-titled meeting renders the muted "Untitled meeting" placeholder
 *     (not the raw `Recording <timestamp>` string), nudging a name,
 *   - clicking the title commits a rename through the mocked `../ipc/meetings`
 *     seam, and editing a still-default title starts from an EMPTY field.
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
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn().mockResolvedValue({}),
  renameMeeting: vi.fn().mockResolvedValue(undefined),
  deleteMeeting: vi.fn().mockResolvedValue(undefined),
  reprocess: vi.fn().mockResolvedValue(undefined),
}));

import {
  MeetingMasthead,
  isDefaultMeetingTitle,
} from "../shell/MeetingMasthead";
import { useMeetingsStore } from "../state/meetings";
import * as meetingsIpc from "../ipc/meetings";

const NAMED = {
  id: "meeting-0001",
  title: "Launch sync — Tuesday",
  started_at: "2026-05-26T14:05:00Z",
  duration_ms: 32 * 60 * 1000,
  speaker_count: 3,
  excerpt: "Three open risks against the date.",
};
const AUTO = {
  id: "meeting-0002",
  title: "Recording 2026-06-18T09:25:22Z",
  started_at: "2026-06-18T09:25:22Z",
  duration_ms: 8 * 60 * 1000,
  speaker_count: 1,
  excerpt: null,
};

function seed() {
  act(() => {
    useMeetingsStore.setState({
      meetings: [NAMED, AUTO],
      loading: false,
      openMeetingId: null,
      openMeetingState: null,
      lastError: null,
    });
  });
}

describe("isDefaultMeetingTitle", () => {
  it("recognises the orchestrator auto title and rejects a real name", () => {
    expect(isDefaultMeetingTitle("Recording 2026-06-18T09:25:22Z")).toBe(true);
    expect(isDefaultMeetingTitle("Launch sync — Tuesday")).toBe(false);
    expect(isDefaultMeetingTitle("Recording notes")).toBe(false);
  });
});

describe("MeetingMasthead", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    seed();
  });

  it("renders a named meeting's title and dateline", () => {
    render(<MeetingMasthead meetingId="meeting-0001" />);
    expect(screen.getByText("Launch sync — Tuesday")).toBeInTheDocument();
    // Deterministic dateline parts (the date string is locale-dependent).
    expect(screen.getByText("32 min")).toBeInTheDocument();
    expect(screen.getByText("3 speakers")).toBeInTheDocument();
  });

  it("shows an 'Untitled meeting' placeholder for an auto-titled meeting", () => {
    render(<MeetingMasthead meetingId="meeting-0002" />);
    expect(screen.getByText("Untitled meeting")).toBeInTheDocument();
    expect(
      screen.queryByText("Recording 2026-06-18T09:25:22Z"),
    ).not.toBeInTheDocument();
  });

  it("commits a rename through the meetings seam", async () => {
    render(<MeetingMasthead meetingId="meeting-0001" />);
    fireEvent.click(screen.getByRole("button", { name: "Rename this meeting" }));
    const input = screen.getByLabelText("Meeting title");
    expect(input).toHaveValue("Launch sync — Tuesday");
    fireEvent.change(input, { target: { value: "Renamed meeting" } });
    // The commit kicks off an async store update; flush it inside act so the
    // post-resolve setState is not an unwrapped update.
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter" });
    });
    await waitFor(() =>
      expect(meetingsIpc.renameMeeting).toHaveBeenCalledWith(
        "meeting-0001",
        "Renamed meeting",
      ),
    );
  });

  it("starts editing an auto-titled meeting from an empty field", () => {
    render(<MeetingMasthead meetingId="meeting-0002" />);
    fireEvent.click(screen.getByRole("button", { name: "Name this meeting" }));
    expect(screen.getByLabelText("Meeting title")).toHaveValue("");
  });
});
