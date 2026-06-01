/**
 * Behaviour test for the Editor's autosave snapshot payload (FR-18).
 *
 * Mounts the REAL `Editor` component (not a stubbed snapshot) so the production
 * `getSnapshot` closure in `editor/Editor.tsx` actually runs. That closure reads
 * the persisted markdown via an untyped reach into Tiptap storage —
 * `editor.storage["markdown"].getMarkdown()` — with a `?? ""` fallback. The
 * existing autosave tests inject a stub snapshot and never execute this closure,
 * so a drift of the `"markdown"` storage key or `getMarkdown` method would
 * silently persist an empty `notes.md` while staying green.
 *
 * This test pins that access end-to-end: with an active recording, real typed
 * content, and autosave triggered through the real path (interval timer and
 * blur), it asserts on the payload actually handed to the `saveNotes` IPC seam
 * (mocked at `../ipc/notes`). `notesMarkdown` must contain the typed heading
 * text — proving `getMarkdown()` resolved rather than the `?? ""` fallback — and
 * `notesJson` must round-trip the typed content.
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
}));

import type { Editor as TiptapEditor } from "@tiptap/core";
import { saveNotes } from "../ipc/notes";
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

/** The single payload `saveNotes` was called with; fails if not called once. */
function soleSavePayload() {
  const calls = vi.mocked(saveNotes).mock.calls;
  expect(calls).toHaveLength(1);
  return calls[0][0];
}

describe("Editor autosave snapshot payload", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(saveNotes).mockClear();
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

  it("persists typed markdown + JSON via the real getSnapshot on the interval", async () => {
    const { editor } = await mountActiveEditor();
    act(() => {
      placeCursorAtEnd(editor);
      typeText(editor, "# Agenda heading");
      vi.advanceTimersToNextFrame();
    });

    expect(saveNotes).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(5_000);
    });

    const payload = soleSavePayload();
    expect(payload.meetingId).toBe("meeting-xyz");

    // `notesMarkdown` proves `editor.storage["markdown"].getMarkdown()` really
    // resolved — it is non-empty (NOT the `?? ""` fallback) and the `# ` input
    // rule produced a markdown heading carrying the typed text.
    expect(payload.notesMarkdown).not.toBe("");
    expect(payload.notesMarkdown).toContain("Agenda heading");
    expect(payload.notesMarkdown).toContain("#");

    // `notesJson` round-trips the typed content as a ProseMirror heading node.
    const doc = JSON.parse(payload.notesJson) as {
      type: string;
      content: Array<{ type: string; content?: Array<{ text?: string }> }>;
    };
    expect(doc.type).toBe("doc");
    const heading = doc.content.find((n) => n.type === "heading");
    expect(heading).toBeDefined();
    expect(heading?.content?.[0]?.text).toBe("Agenda heading");
  });

  it("persists the same real snapshot on blur", async () => {
    const { editor, content } = await mountActiveEditor();
    act(() => {
      placeCursorAtEnd(editor);
      typeText(editor, "# Blur heading");
      vi.advanceTimersToNextFrame();
    });

    // A real DOM blur on the contenteditable drives ProseMirror's `blur` emit,
    // which fires the production on-blur flush handler — exercising the same
    // real `getSnapshot` path the interval uses, just triggered immediately.
    act(() => {
      fireEvent.blur(content);
    });

    const payload = soleSavePayload();
    expect(payload.notesMarkdown).not.toBe("");
    expect(payload.notesMarkdown).toContain("Blur heading");
    expect(JSON.parse(payload.notesJson).type).toBe("doc");
  });
});
