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
import { useMeetingsStore } from "../state/meetings";
import { useCrossRefStore } from "../state/cross-ref";
import { activeTranscript } from "../state/active-transcript";
import { buildEditorExtensions } from "./extensions";
import { useAutosave } from "./useAutosave";
import { buildClipboardPayload } from "./clipboard";
import { handleSegmentDrop } from "./transcript-dnd";
import { scrollToNearestAnchor } from "./scroll-to-anchor";
import { shouldUseDevShim } from "../ipc/dev-shim-guard";
import { loadNotes } from "../ipc/notes";
import { readNotesPaperRules } from "../state/notes-paper-settings";
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
  // FR-22/23 cross-reference: hover maps the hovered paragraph's anchor span to
  // the transcript RANGE; click publishes a scroll request. The active
  // transcript (live vs. saved-meeting) is read non-reactively at hover time.
  const hoverNotesAnchor = useCrossRefStore((s) => s.hoverNotesAnchor);
  const scrollRequest = useCrossRefStore((s) => s.scrollRequest);

  // U1: the restored notes of the open saved meeting (null while recording or on
  // the live entry surface). Hydrated into the editor by a production effect.
  const openMeetingState = useMeetingsStore((s) => s.openMeetingState);

  // Holds the live editor instance for DOM-event handlers wired at construction
  // time (the `drop` handler), which cannot close over `editor` before it is
  // assigned. Populated by the effect below once `useEditor` returns.
  const editorRef = useRef<TiptapEditor | null>(null);

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
      // FR-22: a hovered paragraph's anchor span maps to the RANGE of transcript
      // segments in [anchorMs, nextAnchorMs) (on Segment.start_ms, the
      // pause-excluding clock — never Date.now()). The bridge supplies the next
      // anchored paragraph's anchor; the cross-ref store owns the range
      // computation against the active transcript (live or saved meeting).
      onHoverAnchor: (anchorMs, nextAnchorMs) =>
        hoverNotesAnchor(anchorMs, nextAnchorMs, activeTranscript()),
    }),
    editorProps: {
      attributes: {
        class: "notes-editor__content",
        "aria-label": "Notes",
      },
      // Override copy/cut to write a Word-friendly HTML payload; handle drops of
      // a transcript segment (FR-24) by inserting a transcript-chip node.
      handleDOMEvents: {
        copy: (view, event) => writeClipboard(view.dom, event as ClipboardEvent),
        cut: (view, event) => writeClipboard(view.dom, event as ClipboardEvent),
        drop: (_view, event) => {
          const current = editorRef.current;
          if (!current) return false;
          const dragEvent = event as DragEvent;
          const handled = handleSegmentDrop(current, dragEvent);
          if (handled) dragEvent.preventDefault();
          return handled;
        },
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

  // U1 (production): when a saved meeting is open, hydrate the editor from its
  // restored notes so opening a past meeting shows its notes instead of an empty
  // sheet (SPEC Phase-4 acceptance: "load a past meeting fully restores notes,
  // transcript, audio, and cross-reference"). Keyed on the open meeting's notes
  // so re-opening a different meeting re-hydrates; an open meeting with no saved
  // notes resets the editor to empty. NOT DEV-gated — this is the real restore
  // path the DEV shim previously stood in for.
  useEffect(() => {
    if (!editor) return;
    const notes = openMeetingState?.notes ?? null;
    if (!notes) {
      // Open meeting with no notes (or no meeting open): start from an empty
      // sheet rather than carrying a prior meeting's notes over.
      if (openMeetingState !== null) editor.commands.clearContent();
      return;
    }
    try {
      editor.commands.setContent(JSON.parse(notes.notes_json));
    } catch {
      /* malformed stored notes — leave the editor empty */
    }
  }, [editor, openMeetingState]);

  // DEV-only: in a plain `vite dev` browser (no Tauri backend, no open meeting)
  // seed the editor with sample notes so the themed sheet renders populated for
  // visual QA. The shim's `loadNotes` returns a heading + paragraphs incl. an
  // anchored one, so the left-margin timestamp marginalia shows. No-op in
  // production and tests (the guard is false, and tests mock `../ipc/notes`).
  // Skips when a saved meeting is open so the production restore effect above
  // owns the content.
  useEffect(() => {
    if (!editor || !import.meta.env.DEV || !shouldUseDevShim()) return;
    if (useMeetingsStore.getState().openMeetingState !== null) return;
    let cancelled = false;
    void loadNotes("dev-meeting-0001").then((doc) => {
      if (cancelled || !doc) return;
      try {
        editor.commands.setContent(JSON.parse(doc.notes_json));
      } catch {
        /* malformed seed — leave the editor empty */
      }
    });
    return () => {
      cancelled = true;
    };
  }, [editor]);

  // Keep the editor ref current for the construction-time `drop` handler.
  editorRef.current = editor;

  // FR-23: a clicked transcript segment publishes a scroll request (carrying the
  // segment's start_ms on the pause-excluding clock). Scroll the notes editor to
  // the paragraph whose anchor is nearest that value. The `nonce` makes repeated
  // clicks on the same segment re-trigger the effect.
  useEffect(() => {
    if (!editor || !scrollRequest) return;
    scrollToNearestAnchor(
      editor.view.dom as HTMLElement,
      scrollRequest.anchorMs,
    );
  }, [editor, scrollRequest]);

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

  // Writing-paper rules (on by default; `notes_paper_rules` Appearance setting).
  // Presentation-only — toggles the faint horizontal rules behind the text. The
  // oxblood margin rule that divides the timestamp gutter is structural and
  // shown regardless.
  const paperRules = readNotesPaperRules(settings);

  return (
    <div className={`notes-editor${paperRules ? " notes-editor--ruled" : ""}`}>
      {/*
        The scroll field holds a single sheet of binder paper that fills the
        pane: a narrow timestamp gutter on the left, an oxblood margin rule, then
        the writing column (optionally over faint writing-paper rules).
      */}
      <div className="notes-editor__field">
        <div className="notes-editor__sheet">
          <EditorContent editor={editor} />
        </div>
      </div>
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
