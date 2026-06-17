/**
 * Tests for the show/hide + resizable workspace columns (FR-21/FR-30).
 *
 * The workspace is up to three `react-resizable-panels` columns — notes
 * (primary), transcript, and the summary reading column — that the user shows
 * or hides with a segmented pane-visibility toggle in the header. Panes are
 * included/excluded from the Group rather than collapsed to zero width, with a
 * single `Separator` between each pair of visible panes.
 *
 * These tests assert:
 * - a FINISHED opened meeting defaults to notes + summary (transcript hidden);
 * - a LIVE recording defaults to notes + transcript (no summary column);
 * - the toggle shows/hides a pane and reflects visibility via `aria-pressed`;
 * - a resize handle renders between visible panes and a drag does not throw;
 * - the last visible pane cannot be hidden.
 *
 * Numeric resize outcomes (exact pixel/percent sizes) are not asserted: they
 * depend on measured element dimensions, which jsdom reports as 0.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  act,
  waitFor,
  within,
} from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));
vi.mock("../ipc/bindings", () => ({
  commands: {
    listDevices: vi.fn().mockResolvedValue({ status: "ok", data: [] }),
    getSettings: vi.fn().mockResolvedValue({ status: "ok", data: {} }),
    updateSettings: vi.fn(),
    listModels: vi.fn().mockResolvedValue({ status: "ok", data: [] }),
    ensureModel: vi.fn(),
    startRecording: vi.fn(),
    pauseRecording: vi.fn(),
    resumeRecording: vi.fn(),
    stopRecording: vi.fn(),
    getRecordingState: vi.fn(),
  },
  events: {},
}));
vi.mock("../ipc/notes", () => ({
  saveNotes: vi.fn().mockResolvedValue(undefined),
  loadNotes: vi.fn().mockResolvedValue(null),
  applyNotesUpdate: vi.fn().mockResolvedValue(undefined),
  loadNotesYdoc: vi.fn().mockResolvedValue(null),
}));
vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn(),
  deleteMeeting: vi.fn(),
  reprocess: vi.fn(),
}));
// The summary column mounts the SummaryView, which reads through the summary
// seam; mock it so the column renders without a backend.
vi.mock("../ipc/summary", () => ({
  summariseMeeting: vi.fn().mockResolvedValue(undefined),
  getSummary: vi.fn().mockResolvedValue(null),
  saveSummary: vi.fn().mockResolvedValue(undefined),
}));

import { MainWindow } from "../shell/MainWindow";
import { useMeetingsStore } from "../state/meetings";
import { useRecordingStore } from "../state/recording";

/** The segmented pane-visibility toggle and its segment buttons. */
function viewToggle() {
  return screen.getByRole("group", { name: "Visible panes" });
}
function segment(name: string) {
  return within(viewToggle()).getByRole("button", { name });
}

/**
 * Render MainWindow viewing a FINISHED opened meeting (idle + a meeting open),
 * and flush its mount effects. The workspace only renders once a meeting is
 * open or a recording is live (the meeting-list is the default entry surface).
 */
async function renderFinishedMeeting() {
  act(() => {
    useRecordingStore.setState({ state: { kind: "idle" } });
    useMeetingsStore.setState({
      openMeetingId: "open-meeting-uuid",
      openMeetingState: null,
    });
  });
  const result = render(<MainWindow />);
  await waitFor(() => expect(screen.getByTestId("notes")).toBeInTheDocument());
  return result;
}

describe("MainWindow workspace columns", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" } });
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  it("a finished meeting defaults to notes + summary, transcript hidden", async () => {
    await renderFinishedMeeting();
    // Notes primary pane + summary reading column.
    expect(screen.getByLabelText("Notes")).toBeInTheDocument();
    expect(screen.getByTestId("notes")).toBeInTheDocument();
    expect(screen.getByTestId("summary")).toBeInTheDocument();
    // Transcript is hidden by default for a finished meeting.
    await waitFor(() =>
      expect(screen.queryByTestId("transcript")).not.toBeInTheDocument(),
    );
  });

  it("the view toggle reflects visibility (notes + summary on, transcript off)", async () => {
    await renderFinishedMeeting();
    expect(segment("Notes")).toHaveAttribute("aria-pressed", "true");
    expect(segment("Summary")).toHaveAttribute("aria-pressed", "true");
    expect(segment("Transcript")).toHaveAttribute("aria-pressed", "false");
  });

  it("toggling Transcript shows then hides the transcript column", async () => {
    await renderFinishedMeeting();

    act(() => fireEvent.click(segment("Transcript")));
    expect(screen.getByTestId("transcript")).toBeInTheDocument();
    expect(segment("Transcript")).toHaveAttribute("aria-pressed", "true");

    act(() => fireEvent.click(segment("Transcript")));
    await waitFor(() =>
      expect(screen.queryByTestId("transcript")).not.toBeInTheDocument(),
    );
    expect(segment("Transcript")).toHaveAttribute("aria-pressed", "false");
  });

  it("renders a resize handle between visible panes; a drag does not throw", async () => {
    await renderFinishedMeeting();
    // Notes + summary visible → exactly one separator between them.
    const separator = screen.getByRole("separator");
    expect(separator).toHaveAttribute("tabindex", "0");
    expect(separator).toHaveAttribute("aria-valuemin", "0");
    expect(separator).toHaveAttribute("aria-valuemax", "100");
    expect(() => {
      act(() => {
        fireEvent.pointerDown(separator, { clientX: 100, button: 0 });
        fireEvent.pointerMove(separator, { clientX: 60 });
        fireEvent.pointerUp(separator, { clientX: 60 });
      });
    }).not.toThrow();
  });

  it("the last visible pane cannot be hidden", async () => {
    await renderFinishedMeeting();
    // Hide the summary → only notes remains (transcript already hidden).
    act(() => fireEvent.click(segment("Summary")));
    await waitFor(() =>
      expect(screen.queryByTestId("summary")).not.toBeInTheDocument(),
    );
    // Notes is now the only visible pane; the toggle must refuse to hide it.
    act(() => fireEvent.click(segment("Notes")));
    expect(screen.getByTestId("notes")).toBeInTheDocument();
    expect(segment("Notes")).toHaveAttribute("aria-pressed", "true");
  });

  it("shows exactly two separators when all three columns are visible", async () => {
    await renderFinishedMeeting();
    act(() => fireEvent.click(segment("Transcript")));
    // notes | sep | transcript | sep | summary — one drag handle between each
    // adjacent pair, none stranded beside a hidden pane.
    expect(screen.getAllByRole("separator")).toHaveLength(2);
  });

  it("re-defaults to notes + summary when a recording stops and the meeting opens", async () => {
    // Start live (notes only — transcript hidden, no summary).
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "rec-1", started_at_ms: 0 },
      });
    });
    render(<MainWindow />);
    await waitFor(() => expect(screen.getByTestId("notes")).toBeInTheDocument());
    expect(screen.queryByTestId("transcript")).not.toBeInTheDocument();
    expect(screen.queryByTestId("summary")).not.toBeInTheDocument();

    // Recording stops and the just-recorded meeting is opened (idle + open id):
    // the mode flips to finished IN PLACE and the reset re-defaults the panes.
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" } });
      useMeetingsStore.setState({
        openMeetingId: "rec-1",
        openMeetingState: null,
      });
    });
    await waitFor(() =>
      expect(screen.getByTestId("summary")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("transcript")).not.toBeInTheDocument();
  });

  it("a live recording defaults to notes only — transcript hidden, no summary column", async () => {
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "rec-1", started_at_ms: 0 },
      });
    });
    render(<MainWindow />);
    await waitFor(() => expect(screen.getByTestId("notes")).toBeInTheDocument());

    // Transcript hidden by default (distracting while taking notes); no summary
    // column mid-recording.
    expect(screen.queryByTestId("transcript")).not.toBeInTheDocument();
    expect(screen.queryByTestId("summary")).not.toBeInTheDocument();
    // The Transcript segment is present (so it can be revealed) but unpressed;
    // no Summary segment while recording (nothing to summarise yet).
    expect(segment("Transcript")).toHaveAttribute("aria-pressed", "false");
    expect(
      within(viewToggle()).queryByRole("button", { name: "Summary" }),
    ).not.toBeInTheDocument();
  });

  it("the transcript can be revealed during a live recording via the toggle", async () => {
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "rec-1", started_at_ms: 0 },
      });
    });
    render(<MainWindow />);
    await waitFor(() => expect(screen.getByTestId("notes")).toBeInTheDocument());

    act(() => fireEvent.click(segment("Transcript")));
    expect(screen.getByTestId("transcript")).toBeInTheDocument();
    expect(segment("Transcript")).toHaveAttribute("aria-pressed", "true");
  });

  it("a revealed transcript survives a pause/resume re-render", async () => {
    // Guards the reset-effect dependency array: pause/resume change recordingState
    // but neither `inWorkspace` nor `showSummaryPane`, so the per-mode default
    // must NOT re-fire and snap the user-revealed transcript shut.
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "rec-1", started_at_ms: 0 },
      });
    });
    render(<MainWindow />);
    await waitFor(() => expect(screen.getByTestId("notes")).toBeInTheDocument());

    act(() => fireEvent.click(segment("Transcript")));
    expect(screen.getByTestId("transcript")).toBeInTheDocument();

    act(() =>
      useRecordingStore.setState({
        state: { kind: "paused", meeting_id: "rec-1", paused_at_ms: 1000 },
      }),
    );
    expect(screen.getByTestId("transcript")).toBeInTheDocument();

    act(() =>
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "rec-1", started_at_ms: 0 },
      }),
    );
    expect(screen.getByTestId("transcript")).toBeInTheDocument();
    expect(segment("Transcript")).toHaveAttribute("aria-pressed", "true");
  });
});
