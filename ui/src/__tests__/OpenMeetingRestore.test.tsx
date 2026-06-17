/**
 * Behaviour test for opening a saved meeting (U1 — open_meeting restore wiring).
 *
 * The SPEC Phase-4 acceptance: "close, reopen, load a past meeting fully
 * restores notes, transcript, audio, and cross-reference". Before the fix,
 * `useMeetingsStore.open()` filled `openMeetingState.{notes,transcript}` but
 * nothing consumed it — the editor read the LIVE recording store, the transcript
 * pane read the LIVE store, and notes restore was DEV-only — so opening a saved
 * meeting showed an empty editor + transcript.
 *
 * This test resolves a real `MeetingState` through the mocked `../ipc/meetings`
 * seam (per the architecture testing policy the seam is mocked, not the
 * generated bindings) and asserts the rendered RESULT, not merely that the IPC
 * was called:
 *   - the editor content reflects the restored notes (the heading text shows),
 *   - the transcript pane renders the restored segments (their text shows).
 *
 * Under the CRDT editor binding (B6 WU7) the notes restore flows through the
 * editor's Y.Doc, hydrated from `load_notes_ydoc` — so this test returns a real
 * v1 Yjs update (built from the restored notes JSON via the genuine binding) and
 * asserts the hydrated content renders.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  act,
  cleanup,
  waitFor,
  fireEvent,
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

import { openMeeting } from "../ipc/meetings";
import { loadNotesYdoc } from "../ipc/notes";
import { buildNotesYdocV1Update } from "./editor-test-utils";
import type { MeetingState } from "../ipc/meetings";
import type { Segment } from "../ipc/bindings";

vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn(),
  deleteMeeting: vi.fn(),
  reprocess: vi.fn(),
}));

import { MainWindow } from "../shell/MainWindow";
import { useMeetingsStore } from "../state/meetings";
import { useRecordingStore } from "../state/recording";

const RESTORED_SEGMENTS: Segment[] = [
  {
    start_ms: 4_200,
    end_ms: 9_800,
    text: "Restored first transcript segment.",
    words: [], shared_speakers: [],
  },
  {
    start_ms: 12_400,
    end_ms: 21_300,
    text: "Restored second transcript segment.",
    words: [], shared_speakers: [],
  },
];

const RESTORED_NOTES_JSON = JSON.stringify({
  type: "doc",
  content: [
    {
      type: "heading",
      attrs: { level: 1 },
      content: [{ type: "text", text: "Restored meeting notes heading" }],
    },
    {
      type: "paragraph",
      content: [{ type: "text", text: "A restored body paragraph." }],
    },
  ],
});

const RESTORED_STATE: MeetingState = {
  meta: {
    uuid: "saved-meeting-0001",
    title: "Saved meeting",
    started_at: "2026-05-26T14:05:00Z",
    ended_at: "2026-05-26T14:37:00Z",
    duration_ms: 32 * 60 * 1000,
    speaker_count: 2,
    audio_format: { codec: "opus", sample_rate: 16_000, channels: 1, bitrate_kbps: 32 },
    asr_model: null,
    llm_model: null,
    diarizer: null,
    speaker_names: {},
    app_version: "0.0.0",
  },
  transcript: RESTORED_SEGMENTS,
  notes: {
    notes_json: RESTORED_NOTES_JSON,
    notes_markdown: "# Restored meeting notes heading\n\nA restored body paragraph.\n",
  },
};

describe("opening a saved meeting restores notes + transcript (U1)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(openMeeting).mockResolvedValue(RESTORED_STATE);
    // Production restore now hydrates the editor's Y.Doc from `load_notes_ydoc`
    // (B6 WU7), not the notes.json `setContent` path. Return a real v1 Yjs
    // update encoding the restored notes so the collab binding hydrates and the
    // restored heading/body render.
    vi.mocked(loadNotesYdoc).mockResolvedValue(
      buildNotesYdocV1Update(JSON.parse(RESTORED_NOTES_JSON)),
    );
    act(() => {
      useRecordingStore.setState({
        state: { kind: "idle" },
        transcript: [],
        recordingClockMs: null,
      });
      useMeetingsStore.setState({
        meetings: [],
        loading: false,
        openMeetingId: null,
        openMeetingState: null,
        lastError: null,
      });
    });
  });

  afterEach(() => {
    cleanup();
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  it("renders the restored notes in the editor AND the restored segments in the transcript pane", async () => {
    await act(async () => {
      render(<MainWindow />);
      await Promise.resolve();
    });

    // Open the saved meeting through the store action (routes through the mocked
    // ../ipc/meetings seam, resolving a real MeetingState).
    await act(async () => {
      await useMeetingsStore.getState().open("saved-meeting-0001");
    });

    // The store consumed the restore payload.
    expect(openMeeting).toHaveBeenCalledWith("saved-meeting-0001");
    expect(useMeetingsStore.getState().openMeetingState).not.toBeNull();

    // The editor hydrated from the restored notes — the heading text renders in
    // the contenteditable sheet (NOT an empty editor).
    await waitFor(() =>
      expect(
        screen.getByText("Restored meeting notes heading"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText("A restored body paragraph.")).toBeInTheDocument();

    // A finished, opened meeting defaults to notes + summary (transcript
    // hidden); reveal the transcript column via the pane-visibility toggle.
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Transcript" }));
    });

    // The transcript pane rendered the restored segments (NOT the empty live
    // store, which has zero segments).
    expect(
      screen.getByText("Restored first transcript segment."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Restored second transcript segment."),
    ).toBeInTheDocument();

    // And the live recording store is still empty — proving the panes read the
    // restored saved-meeting source, not the live store.
    expect(useRecordingStore.getState().transcript).toHaveLength(0);
  });
});
