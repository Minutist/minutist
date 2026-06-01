/**
 * Notes editor — the primary view (FR-15/16/17/18/19/20).
 *
 * A Tiptap v3 WYSIWYG editor with:
 *   - markdown-shortcut input rules (StarterKit + Typography) that transform
 *     while typing,
 *   - links, tables, and markdown round-tripping,
 *   - the ParagraphAnchor extension stamping `data-anchor-ms` from the store's
 *     `recordingClockMs` (the pause-excluding capture clock) on first keystroke
 *     while recording,
 *   - interval + on-blur autosave through the `save_notes` IPC seam, no-op when
 *     no meeting is active,
 *   - copy that writes an HTML clipboard payload so paste into Word retains
 *     formatting.
 */
import { useEffect, useRef } from "react";
import { useEditor, EditorContent } from "@tiptap/react";
import type { Editor as TiptapEditor } from "@tiptap/core";
import { useRecordingStore } from "../state/recording";
import { buildEditorExtensions } from "./extensions";
import { useAutosave } from "./useAutosave";
import { buildClipboardPayload } from "./clipboard";
import "./Editor.css";

/**
 * Read the autosave interval from the settings snapshot.
 *
 * `autosave_interval_secs` is a Phase 3 settings field that Stream S3 adds to
 * the backend `Settings` struct + regenerates into `bindings.ts`. Until then it
 * is read defensively from the (untyped-for-this-field) snapshot, falling back
 * to the default inside `useAutosave`.
 */
function readAutosaveInterval(settings: unknown): number | null {
  if (settings && typeof settings === "object") {
    const value = (settings as Record<string, unknown>)["autosave_interval_secs"];
    if (typeof value === "number") return value;
  }
  return null;
}

export function Editor() {
  const recordingState = useRecordingStore((s) => s.state);
  const settings = useRecordingStore((s) => s.settings);

  const editor = useEditor({
    extensions: buildEditorExtensions({
      // Read the recording clock lazily on each keystroke from the store so the
      // anchor value is always the latest pause-excluding capture clock — never
      // a wall-clock delta. `getState()` avoids re-creating the editor when the
      // clock advances.
      clockSource: () => {
        const s = useRecordingStore.getState();
        return {
          recording: s.state.kind === "recording",
          clockMs: s.recordingClockMs,
        };
      },
    }),
    editorProps: {
      attributes: {
        class: "notes-editor__content",
        "aria-label": "Notes",
      },
      // Override copy/cut to write a Word-friendly HTML payload.
      handleDOMEvents: {
        copy: (view, event) => writeClipboard(view.dom, event as ClipboardEvent),
        cut: (view, event) => writeClipboard(view.dom, event as ClipboardEvent),
      },
    },
  });

  const intervalSecs = readAutosaveInterval(settings);

  const { flush } = useAutosave({
    state: recordingState,
    intervalSecs,
    getSnapshot: () => {
      if (!editor) return null;
      const markdownStorage = (
        editor.storage as unknown as Record<string, unknown>
      )["markdown"] as { getMarkdown: () => string } | undefined;
      return {
        notesJson: JSON.stringify(editor.getJSON()),
        notesMarkdown: markdownStorage?.getMarkdown() ?? "",
      };
    },
  });

  // Flush on blur so notes are persisted the instant focus leaves the editor.
  const flushRef = useRef(flush);
  flushRef.current = flush;
  useEffect(() => {
    if (!editor) return;
    const handleBlur = () => flushRef.current();
    editor.on("blur", handleBlur);
    return () => {
      editor.off("blur", handleBlur);
    };
  }, [editor]);

  return (
    <div className="notes-editor">
      <EditorContent editor={editor} />
    </div>
  );
}

/**
 * Write the current selection (or whole doc) to the clipboard as HTML + text.
 *
 * Returns `true` to tell ProseMirror the event is handled (we set the data and
 * `preventDefault`).
 */
function writeClipboard(dom: HTMLElement, event: ClipboardEvent): boolean {
  const clipboardData = event.clipboardData;
  if (!clipboardData) return false;

  const selection = window.getSelection();
  const html =
    selection && selection.rangeCount > 0 && !selection.isCollapsed
      ? selectionToHtml(selection)
      : dom.innerHTML;
  const text = selection && !selection.isCollapsed ? selection.toString() : dom.textContent ?? "";

  const payload = buildClipboardPayload(html, text);
  clipboardData.setData("text/html", payload["text/html"]);
  clipboardData.setData("text/plain", payload["text/plain"]);
  event.preventDefault();
  return true;
}

/** Serialise a DOM selection to an HTML string. */
function selectionToHtml(selection: Selection): string {
  const container = document.createElement("div");
  for (let i = 0; i < selection.rangeCount; i += 1) {
    container.appendChild(selection.getRangeAt(i).cloneContents());
  }
  return container.innerHTML;
}

export type { TiptapEditor };
