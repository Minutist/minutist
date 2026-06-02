/**
 * Tests for the collapsible + resizable two-pane layout (FR-21).
 *
 * The notes editor is the primary pane; the transcript pane is secondary,
 * collapsible AND resizable via `react-resizable-panels`. These tests assert:
 * - both panes and a draggable resize handle render,
 * - the transcript pane is collapsible and the toggle collapses/expands it
 *   (panel `flex-grow` goes to 0 and back; toggle label + aria-pressed flip),
 * - the resize handle exposes the resize affordance (separator role, focusable,
 *   ARIA value range) and a pointer drag on it does not throw.
 *
 * Numeric resize outcomes (exact pixel/percent sizes) are not asserted: they
 * depend on measured element dimensions, which jsdom reports as 0. The collapse
 * round-trip exercises the same imperative resize path and is deterministic.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  act,
  waitFor,
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
}));
vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn(),
  deleteMeeting: vi.fn(),
  reTranscribe: vi.fn(),
}));

import { MainWindow } from "../shell/MainWindow";
import { useMeetingsStore } from "../state/meetings";

/**
 * Render MainWindow in the editor/transcript workspace and flush its mount
 * effects. The two-pane layout (FR-21) only renders once a meeting is open (the
 * meeting-list is the default entry surface, FR-33), so put the meetings store
 * into an open state first.
 */
async function renderMainWindow() {
  act(() => {
    useMeetingsStore.setState({
      openMeetingId: "open-meeting-uuid",
      openMeetingState: null,
    });
  });
  const result = render(<MainWindow />);
  // Let the mount-time async refreshes settle so state updates are wrapped.
  await waitFor(() => expect(screen.getByTestId("notes")).toBeInTheDocument());
  return result;
}

describe("MainWindow two-pane layout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  it("renders the notes editor (primary) and transcript (secondary) panes", async () => {
    await renderMainWindow();
    // Notes editor primary pane.
    expect(screen.getByLabelText("Notes")).toBeInTheDocument();
    expect(screen.getByTestId("notes")).toBeInTheDocument();
    // Transcript secondary pane.
    expect(screen.getByTestId("transcript")).toBeInTheDocument();
    expect(screen.getByLabelText("Transcript")).toBeInTheDocument();
  });

  it("renders a resize handle between the panes (resizable affordance)", async () => {
    await renderMainWindow();
    const separator = screen.getByRole("separator");
    // The separator is the drag affordance: focusable and not disabled, with a
    // 0..100 ARIA value range.
    expect(separator).toHaveAttribute("tabindex", "0");
    expect(separator).toHaveAttribute("aria-valuemin", "0");
    expect(separator).toHaveAttribute("aria-valuemax", "100");
    expect(separator).toHaveAttribute("aria-valuenow");
  });

  it("a pointer drag on the resize handle does not throw", async () => {
    await renderMainWindow();
    const separator = screen.getByRole("separator");
    expect(() => {
      act(() => {
        fireEvent.pointerDown(separator, { clientX: 100, button: 0 });
        fireEvent.pointerMove(separator, { clientX: 60 });
        fireEvent.pointerUp(separator, { clientX: 60 });
      });
    }).not.toThrow();
  });

  it("the toggle collapses the transcript pane and flips its label", async () => {
    await renderMainWindow();
    const toggle = screen.getByRole("button", { name: "Hide transcript" });
    const transcript = screen.getByTestId("transcript");

    // Visible initially (non-zero flex-grow).
    expect(transcript.style.flexGrow).not.toBe("0");
    expect(toggle).toHaveAttribute("aria-pressed", "false");

    act(() => {
      fireEvent.click(toggle);
    });

    // Collapsed: flex-grow goes to 0; label + aria-pressed flip.
    expect(transcript.style.flexGrow).toBe("0");
    const collapsedToggle = screen.getByRole("button", {
      name: "Show transcript",
    });
    expect(collapsedToggle).toHaveAttribute("aria-pressed", "true");
  });

  it("the toggle expands the transcript pane again (round-trip)", async () => {
    await renderMainWindow();

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Hide transcript" }));
    });
    const transcript = screen.getByTestId("transcript");
    expect(transcript.style.flexGrow).toBe("0");

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Show transcript" }));
    });

    // Expanded again.
    expect(transcript.style.flexGrow).not.toBe("0");
    expect(
      screen.getByRole("button", { name: "Hide transcript" }),
    ).toHaveAttribute("aria-pressed", "false");
  });
});
