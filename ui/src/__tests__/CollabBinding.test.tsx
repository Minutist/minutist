/**
 * Behaviour test for the local-only Yjs collaboration binding (B6 WU7,
 * `planning/DESIGN_notes-crdt.md` §8).
 *
 * Mounts the REAL `Editor` with an active recording so it binds a per-meeting
 * `Y.Doc`. Three things are pinned end to end:
 *
 *  1. **Hydrate from `load_notes_ydoc`.** A v1 Yjs update (built via the genuine
 *     binding) is returned by the mocked `loadNotesYdoc`; the editor must render
 *     the hydrated content rather than an empty sheet.
 *  2. **Edit → debounced `apply_notes_update`.** A keystroke must, after the
 *     autosave debounce, ship a non-empty Yjs update + the rendered markdown to
 *     `applyNotesUpdate` (the collab write path), and NOT the legacy `saveNotes`
 *     (which is the single writer only when no doc is bound).
 *  3. **Custom nodes survive the binding.** The TranscriptChip atom hydrated
 *     through the Y.Doc renders, proving y-prosemirror binds at the ProseMirror
 *     schema level so custom node types round-trip.
 *
 * The JS-produced update fed into the binding (and, on the Rust side, into
 * `NotesStore::apply_update`) is the same lib0-v1 format both ends agree on — the
 * interop the design calls out. The Rust counterpart is
 * `persistence::ydoc::tests::v1_incremental_update_applies_and_re_derives`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, cleanup, fireEvent } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));
vi.mock("../ipc/bindings", () => ({
  commands: {
    listDevices: vi.fn(),
    startRecording: vi.fn(),
    pauseRecording: vi.fn(),
    resumeRecording: vi.fn(),
    stopRecording: vi.fn(),
    getRecordingState: vi.fn(),
    getSettings: vi.fn(),
    updateSettings: vi.fn(),
    listModels: vi.fn(),
    ensureModel: vi.fn(),
  },
  events: {},
}));
vi.mock("../ipc/notes", () => ({
  saveNotes: vi.fn().mockResolvedValue(undefined),
  loadNotes: vi.fn().mockResolvedValue(null),
  applyNotesUpdate: vi.fn().mockResolvedValue(undefined),
  loadNotesYdoc: vi.fn().mockResolvedValue(null),
}));

import type { Editor as TiptapEditor } from "@tiptap/core";
import { applyNotesUpdate, saveNotes, loadNotesYdoc } from "../ipc/notes";
import { Editor } from "../editor/Editor";
import { useRecordingStore } from "../state/recording";
import {
  typeText,
  placeCursorAtEnd,
  buildNotesYdocV1Update,
} from "./editor-test-utils";

/** Mount the real Editor with an active recording; return the live instance. */
async function mountActiveEditor(): Promise<{
  editor: TiptapEditor;
  content: HTMLElement;
}> {
  await act(async () => {
    render(<Editor />);
  });
  // Drain the hydrate chain (loadNotesYdoc promise → setDoc → editor re-create →
  // Collaboration initial sync) under fake timers: several rounds of microtask +
  // frame flush, all inside act so the resulting state updates are wrapped.
  // `waitFor` cannot be used for this — it polls on real time, which fake timers
  // freeze — so drain deterministically instead.
  for (let i = 0; i < 10; i += 1) {
    await act(async () => {
      vi.advanceTimersToNextFrame();
      await Promise.resolve();
    });
  }
  const content = screen.getByLabelText("Notes") as HTMLElement & {
    editor?: TiptapEditor;
  };
  const editor = content.editor;
  if (!editor) throw new Error("could not reach the Tiptap editor instance");
  return { editor, content };
}

describe("notes collaboration binding (B6 WU7)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(applyNotesUpdate).mockClear();
    vi.mocked(saveNotes).mockClear();
    vi.mocked(loadNotesYdoc).mockResolvedValue(null);
    useRecordingStore.setState({
      state: { kind: "recording", meeting_id: "meeting-collab", started_at_ms: 1_000 },
      devices: [],
      selectedDeviceId: null,
      settings: null,
      meter: { peak: 0, rms: 0 },
      lastError: null,
      transcript: [],
      recordingClockMs: null,
    });
  });
  afterEach(() => {
    act(() => {
      cleanup();
      vi.runOnlyPendingTimers();
    });
    vi.useRealTimers();
    useRecordingStore.setState({ state: { kind: "idle" } });
  });

  it("hydrates the editor from load_notes_ydoc", async () => {
    vi.mocked(loadNotesYdoc).mockResolvedValue(
      buildNotesYdocV1Update({
        type: "doc",
        content: [
          {
            type: "heading",
            attrs: { level: 1 },
            content: [{ type: "text", text: "Hydrated from ydoc" }],
          },
        ],
      }),
    );

    await mountActiveEditor();

    expect(loadNotesYdoc).toHaveBeenCalledWith("meeting-collab");
    expect(screen.getByText("Hydrated from ydoc")).toBeInTheDocument();
  });

  it("hydrates a custom TranscriptChip node through the binding", async () => {
    vi.mocked(loadNotesYdoc).mockResolvedValue(
      buildNotesYdocV1Update({
        type: "doc",
        content: [
          {
            type: "transcriptChip",
            attrs: {
              startMs: 12_400,
              endMs: 21_300,
              speakerId: "A",
              text: "a quoted segment",
            },
          },
        ],
      }),
    );

    await mountActiveEditor();

    // The chip's node view renders its captured text — proving the custom atom
    // round-tripped through the Y.Doc binding (y-prosemirror binds at the schema
    // level, so custom node types survive).
    expect(screen.getByText("a quoted segment")).toBeInTheDocument();
  });

  it("ships an edit to apply_notes_update on the debounce, not save_notes", async () => {
    const { editor } = await mountActiveEditor();
    act(() => {
      placeCursorAtEnd(editor);
      typeText(editor, "live note");
      vi.advanceTimersToNextFrame();
    });

    expect(applyNotesUpdate).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(5_000);
    });

    const calls = vi.mocked(applyNotesUpdate).mock.calls;
    expect(calls.length).toBeGreaterThanOrEqual(1);
    const [meetingId, update, markdown] = calls[calls.length - 1];
    expect(meetingId).toBe("meeting-collab");
    expect(update).toBeInstanceOf(Uint8Array);
    expect((update as Uint8Array).length).toBeGreaterThan(0);
    expect(markdown).toContain("live note");

    // The legacy JSON writer is disabled while a collab doc is bound — single
    // writer, no race on the same files.
    expect(saveNotes).not.toHaveBeenCalled();
  });

  it("flushes the pending update on blur", async () => {
    const { editor, content } = await mountActiveEditor();
    act(() => {
      placeCursorAtEnd(editor);
      typeText(editor, "blur me");
      vi.advanceTimersToNextFrame();
    });

    act(() => {
      fireEvent.blur(content);
    });

    const calls = vi.mocked(applyNotesUpdate).mock.calls;
    expect(calls.length).toBeGreaterThanOrEqual(1);
    expect(saveNotes).not.toHaveBeenCalled();
  });
});
