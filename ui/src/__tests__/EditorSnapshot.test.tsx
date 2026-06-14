/**
 * Behaviour test for the Editor's collab persistence payload (FR-18, B6 WU7).
 *
 * Mounts the REAL `Editor` component (not a stubbed snapshot) so the production
 * markdown renderer in `editor/Editor.tsx` actually runs. That renderer reads
 * the persisted markdown via an untyped reach into Tiptap storage —
 * `editor.storage["markdown"].getMarkdown()` — with a `?? ""` fallback. With an
 * active recording the editor binds a per-meeting Y.Doc, so the content flows
 * through `apply_notes_update` (the binary Yjs update + the rendered markdown),
 * not the legacy `save_notes` JSON path. A drift of the `"markdown"` storage key
 * or `getMarkdown` method would silently persist an empty `notes.md` while
 * staying green.
 *
 * This test pins that access end-to-end: with an active recording, real typed
 * content, and persistence triggered through the real path (debounced timer and
 * blur), it asserts on the payload actually handed to the `applyNotesUpdate` IPC
 * seam (mocked at `../ipc/notes`). The markdown must contain the typed heading
 * text — proving `getMarkdown()` resolved rather than the `?? ""` fallback — and
 * the update must be non-empty bytes (the editor's edit reached the Y.Doc).
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
import { applyNotesUpdate } from "../ipc/notes";
import { Editor } from "../editor/Editor";
import { useRecordingStore } from "../state/recording";
import { typeText, placeCursorAtEnd } from "./editor-test-utils";

/**
 * Mount the real Editor with an active recording and return the live Tiptap
 * editor instance plus its contenteditable DOM node.
 *
 * The contenteditable element (`aria-label="Notes"`) is the ProseMirror view's
 * DOM; Tiptap attaches the editor instance to that node as `.editor`. Driving
 * keystrokes through this real instance (via `typeText`) feeds them through the
 * same input-rule + storage path the production component uses, so the autosave
 * `getSnapshot` closure reads genuine editor state. The DOM node is returned so
 * a real `blur` can be dispatched on it (the production blur-flush path).
 */
async function mountActiveEditor(): Promise<{
  editor: TiptapEditor;
  content: HTMLElement;
}> {
  await act(async () => {
    render(<Editor />);
  });
  // `useEditor` re-renders once the editor instance is created (the
  // `useSyncExternalStore` notify lands on a microtask) and again when
  // ProseMirror's selection `requestAnimationFrame` fires. Flush both inside
  // act so those state updates are wrapped: the awaited microtask drains the
  // create-time re-render, and `advanceTimersToNextFrame` runs only the rAF
  // callbacks (never the autosave `setInterval`).
  await act(async () => {
    vi.advanceTimersToNextFrame();
    await Promise.resolve();
  });
  const content = screen.getByLabelText("Notes") as HTMLElement & {
    editor?: TiptapEditor;
  };
  const editor = content.editor;
  if (!editor) {
    throw new Error(
      "could not reach the Tiptap editor instance from the mounted Editor",
    );
  }
  return { editor, content };
}

/**
 * The arguments of the LAST `applyNotesUpdate` call: `(meetingId, update,
 * notesMarkdown)`. Fails if it was never called. The collab sync may coalesce a
 * burst into one or more whole-state sends; the last one carries the final
 * document, so assert on it rather than on a single-call count.
 */
function lastUpdateArgs() {
  const calls = vi.mocked(applyNotesUpdate).mock.calls;
  expect(calls.length).toBeGreaterThanOrEqual(1);
  return calls[calls.length - 1];
}

describe("Editor autosave snapshot payload", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(applyNotesUpdate).mockClear();
    useRecordingStore.setState({
      state: { kind: "recording", meeting_id: "meeting-xyz", started_at_ms: 1_000 },
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
    // Unmount while fake timers are still active and flush the editor's
    // deferred `scheduleDestroy` setTimeout inside act, so `useEditor`'s
    // teardown state update (`setEditor(null)`) is wrapped rather than firing
    // unwrapped once timers are torn down.
    act(() => {
      cleanup();
      vi.runOnlyPendingTimers();
    });
    vi.useRealTimers();
    useRecordingStore.setState({ state: { kind: "idle" } });
  });

  it("ships the typed update + rendered markdown via apply_notes_update on the debounce", async () => {
    const { editor } = await mountActiveEditor();
    act(() => {
      placeCursorAtEnd(editor);
      typeText(editor, "# Agenda heading");
      vi.advanceTimersToNextFrame();
    });

    expect(applyNotesUpdate).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(5_000);
    });

    const [meetingId, update, notesMarkdown] = lastUpdateArgs();
    expect(meetingId).toBe("meeting-xyz");

    // The Yjs update is non-empty bytes — the keystroke reached the bound Y.Doc.
    expect(update).toBeInstanceOf(Uint8Array);
    expect((update as Uint8Array).length).toBeGreaterThan(0);

    // `notesMarkdown` proves `editor.storage["markdown"].getMarkdown()` really
    // resolved — it is non-empty (NOT the `?? ""` fallback) and the `# ` input
    // rule produced a markdown heading carrying the typed text.
    expect(notesMarkdown).not.toBe("");
    expect(notesMarkdown).toContain("Agenda heading");
    expect(notesMarkdown).toContain("#");
  });

  it("flushes the pending update on blur", async () => {
    const { editor, content } = await mountActiveEditor();
    act(() => {
      placeCursorAtEnd(editor);
      typeText(editor, "# Blur heading");
      vi.advanceTimersToNextFrame();
    });

    // A real DOM blur on the contenteditable drives ProseMirror's `blur` emit,
    // which fires the production on-blur flush handler — flushing the collab
    // sync's pending update immediately rather than waiting for the debounce.
    act(() => {
      fireEvent.blur(content);
    });

    const [, update, notesMarkdown] = lastUpdateArgs();
    expect((update as Uint8Array).length).toBeGreaterThan(0);
    expect(notesMarkdown).not.toBe("");
    expect(notesMarkdown).toContain("Blur heading");
  });
});
