/**
 * Attachment paste/drop handling for the notes editor (#0038).
 *
 * A file dropped or pasted into the notes editor — of ANY type, not just
 * images — is registered as a normal meeting attachment (the `add_attachment`
 * pipeline: manifest entry, attachments pane, markdown conversion fed to the
 * summariser) via the attachments STORE's `add` action (so the attachments
 * pane reflects it the same way a pane-side "Add files" drop does), and an
 * `AttachmentRef` node carrying the returned entry's portable ref is inserted
 * at the drop/paste point.
 *
 * This supersedes `./image-paste` for all NEW drops/pastes (including images —
 * an image dropped into notes is now an attachment with an inline thumbnail,
 * not a standalone note-asset). `./note-image`'s `NoteImage` node is untouched
 * and keeps rendering images already embedded in existing meetings.
 *
 * CRUCIAL non-interference rule: these handlers return `false` (do NOT
 * `preventDefault`) whenever the clipboard/drop carries NO files, so a
 * text / markdown / HTML paste falls through to the existing `tiptap-markdown`
 * paste handling untouched. A paste that ALSO carries meaningful text/html is
 * treated as a rich-text paste, not a file paste, so as not to hijack it
 * (mirrors `./image-paste`'s rule).
 */
import type { Editor } from "@tiptap/core";
import { useAttachmentsStore } from "../state/attachments";
import type { MeetingId } from "../ipc/bindings";
import { ATTACHMENT_REF_NODE_NAME } from "./attachment-ref";

/** Resolves the editor's current meeting id, or `null` when none is open. */
export type MeetingIdSource = () => string | null;

/** Pull every `File` out of a `DataTransfer` (clipboard or drop), any type. */
export function filesFromDataTransfer(data: DataTransfer | null): File[] {
  if (!data) return [];
  if (data.files && data.files.length > 0) {
    return Array.from(data.files);
  }
  // Clipboard pastes sometimes carry the file only under `items`.
  if (data.items && data.items.length > 0) {
    const files: File[] = [];
    for (const item of Array.from(data.items)) {
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    return files;
  }
  return [];
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

/** Lower-cased, dot-less extension of a file's name, or `null` if none. */
function extOf(file: File): string | null {
  const dot = file.name.lastIndexOf(".");
  if (dot < 0 || dot === file.name.length - 1) return null;
  return file.name.slice(dot + 1).toLowerCase();
}

/**
 * Attach `files` and insert an `AttachmentRef` node for each at the current
 * selection. Runs asynchronously (the IPC add is async) but inserts in order.
 *
 * A per-file failure (no extension, or the store's `add` returning `null` —
 * e.g. the backend rejected an unsupported extension or the size limit) is
 * reported to `onError` and skipped, mirroring `./image-paste`'s per-file
 * error handling: one bad file must not abort the rest.
 */
async function insertAttachments(
  editor: Editor,
  meetingId: MeetingId,
  files: File[],
  onError?: (err: unknown) => void,
): Promise<void> {
  const add = useAttachmentsStore.getState().add;
  for (const file of files) {
    const ext = extOf(file);
    if (ext === null) {
      onError?.(new Error(`"${file.name}" has no file extension`));
      continue;
    }
    try {
      const entry = await add(meetingId, file, ext);
      if (!entry) {
        // The store already recorded the failure on `lastError` (surfaced by
        // the attachments pane); tell this caller too so the notes editor can
        // show its own inline notice if it chooses to.
        onError?.(new Error(`couldn't attach "${file.name}"`));
        continue;
      }
      editor
        .chain()
        .focus()
        .insertContent({
          type: ATTACHMENT_REF_NODE_NAME,
          attrs: {
            attachmentId: entry.id,
            filename: `${entry.hash}.${entry.ext}`,
            originalFilename: entry.original_filename,
            ext: entry.ext,
            byteLen: entry.byte_len,
          },
        })
        .run();
    } catch (err) {
      onError?.(err);
    }
  }
}

/**
 * Handle a clipboard paste. Returns `true` (and the caller should treat the
 * event as handled) only when files were found and a meeting is open;
 * otherwise returns `false` so the normal paste pipeline runs.
 */
export function handleAttachmentPaste(
  editor: Editor,
  event: ClipboardEvent,
  meetingIdSource: MeetingIdSource,
  onError?: (err: unknown) => void,
): boolean {
  const data = event.clipboardData;
  const files = filesFromDataTransfer(data);
  if (files.length === 0) return false;
  // A paste that also carries real text/html is a rich-text paste, not a file
  // paste — let the markdown handler own it (do not hijack).
  if (hasTextPayload(data)) return false;

  const meetingId = meetingIdSource();
  if (meetingId === null) return false;

  void insertAttachments(editor, meetingId, files, onError);
  return true;
}

/**
 * Handle a native file drop. Returns `true` only when files were found and a
 * meeting is open; otherwise `false` so other drop handlers (the
 * transcript-segment drop, ProseMirror's own) run.
 */
export function handleAttachmentDrop(
  editor: Editor,
  event: DragEvent,
  meetingIdSource: MeetingIdSource,
  onError?: (err: unknown) => void,
): boolean {
  const data = event.dataTransfer;
  const files = filesFromDataTransfer(data);
  if (files.length === 0) return false;

  const meetingId = meetingIdSource();
  if (meetingId === null) return false;

  // Place the cursor where the drop landed before inserting, when coordinates
  // are available, so the reference lands at the pointer (mirrors
  // `./image-paste` / the transcript DnD handler).
  if (typeof event.clientX === "number" && typeof event.clientY === "number") {
    const coords = editor.view.posAtCoords({
      left: event.clientX,
      top: event.clientY,
    });
    if (coords) editor.commands.setTextSelection(coords.pos);
  }

  void insertAttachments(editor, meetingId, files, onError);
  return true;
}
