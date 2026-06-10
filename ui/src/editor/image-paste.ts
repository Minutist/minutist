/**
 * Image paste/drop handling for the notes editor.
 *
 * Detects image `File`(s) in a clipboard paste or a native file drop, persists
 * each via the `saveNoteImageFile` IPC seam, and inserts an `image` node whose
 * stored `src` is the PORTABLE filename ref (see `./note-image` for the
 * stored-vs-rendered contract).
 *
 * CRUCIAL non-interference rule: these handlers return `false` (do NOT
 * `preventDefault`) whenever the clipboard/drop carries NO image files, so a
 * text / markdown / HTML paste falls through to the existing `tiptap-markdown`
 * paste handling untouched. They only take over when there is at least one
 * image file to save — and even then, a paste that ALSO carries an image
 * (e.g. some rich-content sources) is treated as an image paste only when there
 * is no meaningful text payload, to avoid hijacking a normal rich-text paste.
 */
import type { Editor } from "@tiptap/core";
import { imageExtForFile, saveNoteImageFile } from "../ipc/note-images";

/** Resolves the editor's current meeting id, or `null` when none is open. */
export type MeetingIdSource = () => string | null;

/** Pull image `File`s out of a `DataTransfer` (clipboard or drop). */
export function imageFilesFromDataTransfer(
  data: DataTransfer | null,
): File[] {
  if (!data) return [];
  const files: File[] = [];
  // `DataTransfer.files` covers dropped files; `items` covers pasted images
  // (which often appear only under `items`, not `files`). Prefer `files` and
  // fall back to `items` so both paths are covered without double-counting.
  if (data.files && data.files.length > 0) {
    for (const file of Array.from(data.files)) {
      if (file.type.startsWith("image/")) files.push(file);
    }
  }
  if (files.length === 0 && data.items && data.items.length > 0) {
    for (const item of Array.from(data.items)) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
  }
  return files;
}

/** True when the transfer carries text/markdown/html the user likely wants. */
function hasTextPayload(data: DataTransfer | null): boolean {
  if (!data) return false;
  const text = data.getData("text/plain");
  const html = data.getData("text/html");
  return (text && text.trim().length > 0) || (html && html.trim().length > 0)
    ? true
    : false;
}

/**
 * Persist `files` and insert each as an `image` node at the current selection.
 *
 * Runs asynchronously (the IPC write + the `File.arrayBuffer()` read are async)
 * but the inserts preserve order. A per-file failure is reported to `onError`
 * and skipped — one bad image must not abort the rest or throw into the
 * ProseMirror event pipeline.
 */
async function insertImages(
  editor: Editor,
  meetingId: string,
  files: { file: File; ext: string }[],
  onError?: (err: unknown) => void,
): Promise<void> {
  for (const { file, ext } of files) {
    try {
      const src = await saveNoteImageFile(meetingId, file, ext);
      // Store the PORTABLE ref as the node's `src`; `note-image` converts it to
      // a display URL at render time, and `getJSON` keeps this portable value.
      editor.chain().focus().setImage({ src, alt: file.name }).run();
    } catch (err) {
      onError?.(err);
    }
  }
}

/**
 * Handle a clipboard paste. Returns `true` (and the caller should treat the
 * event as handled) only when image files were found and a meeting is open;
 * otherwise returns `false` so the normal paste pipeline runs.
 *
 * The save+insert is fire-and-forget (ProseMirror handlers are synchronous);
 * returning `true` prevents the default paste so the browser does not also drop
 * the image's text fallback.
 */
export function handleImagePaste(
  editor: Editor,
  event: ClipboardEvent,
  meetingIdSource: MeetingIdSource,
  onError?: (err: unknown) => void,
): boolean {
  const data = event.clipboardData;
  const imageFiles = imageFilesFromDataTransfer(data);
  if (imageFiles.length === 0) return false;
  // A paste that also carries real text/html is a rich-text paste, not an image
  // paste — let the markdown handler own it (do not hijack).
  if (hasTextPayload(data)) return false;

  const meetingId = meetingIdSource();
  if (meetingId === null) return false;

  const withExt = imageFiles
    .map((file) => ({ file, ext: imageExtForFile(file) }))
    .filter((x): x is { file: File; ext: string } => x.ext !== null);
  if (withExt.length === 0) return false;

  void insertImages(editor, meetingId, withExt, onError);
  return true;
}

/**
 * Handle a native file drop. Returns `true` only when image files were found
 * and a meeting is open; otherwise `false` so other drop handlers (the
 * transcript-segment drop, ProseMirror's own) run.
 */
export function handleImageDrop(
  editor: Editor,
  event: DragEvent,
  meetingIdSource: MeetingIdSource,
  onError?: (err: unknown) => void,
): boolean {
  const data = event.dataTransfer;
  const imageFiles = imageFilesFromDataTransfer(data);
  if (imageFiles.length === 0) return false;

  const meetingId = meetingIdSource();
  if (meetingId === null) return false;

  const withExt = imageFiles
    .map((file) => ({ file, ext: imageExtForFile(file) }))
    .filter((x): x is { file: File; ext: string } => x.ext !== null);
  if (withExt.length === 0) return false;

  // Place the cursor where the drop landed before inserting, when coordinates
  // are available, so images land at the pointer (mirrors transcript DnD).
  if (typeof event.clientX === "number" && typeof event.clientY === "number") {
    const coords = editor.view.posAtCoords({
      left: event.clientX,
      top: event.clientY,
    });
    if (coords) editor.commands.setTextSelection(coords.pos);
  }

  void insertImages(editor, meetingId, withExt, onError);
  return true;
}
