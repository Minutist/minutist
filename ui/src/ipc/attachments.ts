/**
 * IPC seam for meeting reference-material attachments.
 *
 * Wraps the generated attachment commands (`commands.addAttachment` etc.) so the
 * attachments store works with a `File`/`Uint8Array` shape and so tests mock
 * *this* module (per `architecture/cross-cutting.md` — Automated-testing policy)
 * rather than the generated bindings file.
 *
 * `persistence` owns the on-disk `attachments/<hash>.<ext>` originals + `<hash>.md`
 * converted siblings + the `attachments.json` manifest; `ipc-bridge` exposes the
 * commands, routing them directly to `persistence`. Conversion runs on a bounded
 * background worker and reports through the four attachment `AppEvent`s — this
 * seam is the request side only.
 *
 * `open` does NOT return bytes or write a temp file: it asks the backend, which
 * resolves the stored original's on-disk path and hands it to the host OS
 * default application via `tauri-plugin-opener` (the user's PDF reader / Word /
 * Excel / image viewer). The path never crosses the IPC boundary; the webview
 * never navigates to the file.
 */
import { commands, unwrap } from "./client";
import type { AttachmentEntry, AttachmentId, MeetingId } from "./bindings";

/**
 * Maximum permitted attachment size in bytes. Mirrors
 * `ipc_bridge::commands::MAX_INPUT_BYTES` (50 MiB). Checked client-side to
 * surface the rejection inline before reading the full `ArrayBuffer`.
 */
export const MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024;

/**
 * Persist an attachment `File` for `meetingId`, returning the new manifest entry
 * (in the `Pending` conversion state). Rejects (via the thrown IPC error) when
 * the backend refuses the extension or the write fails.
 *
 * The caller supplies the lower-cased, dot-less `ext` (validated server-side
 * against `doc_convert::supported_exts()`) and the user-visible original
 * filename to display.
 */
export async function addAttachment(
  meetingId: MeetingId,
  file: File,
  ext: string,
): Promise<AttachmentEntry> {
  if (file.size > MAX_ATTACHMENT_BYTES) {
    throw new Error(
      `${file.name} is too large (${file.size} bytes); the 50 MiB limit is ${MAX_ATTACHMENT_BYTES} bytes.`,
    );
  }
  const buffer = await file.arrayBuffer();
  const bytes = Array.from(new Uint8Array(buffer));
  return unwrap(
    await commands.addAttachment(meetingId, bytes, ext, file.name),
  );
}

/** List a meeting's attachments in manifest order. */
export async function listAttachments(
  meetingId: MeetingId,
): Promise<AttachmentEntry[]> {
  return unwrap(await commands.listAttachments(meetingId));
}

/**
 * Open an attachment original in the host OS default application.
 *
 * The backend (which holds the `persistence` edge `app-main` lacks) resolves the
 * stored original's on-disk path and hands it to `tauri-plugin-opener`, so the
 * OS launches the user's PDF reader / Word / Excel / image viewer. No bytes
 * cross the wire, no temp file is written, and the path never reaches the
 * webview — this seam only asks the backend to open attachment `entry`.
 */
export async function openAttachment(
  meetingId: MeetingId,
  entry: AttachmentEntry,
): Promise<void> {
  unwrap(await commands.openAttachment(meetingId, entry.id));
}

/** Remove an attachment (dedup-safe unlink happens server-side). */
export async function removeAttachment(
  meetingId: MeetingId,
  attachmentId: AttachmentId,
): Promise<void> {
  unwrap(await commands.removeAttachment(meetingId, attachmentId));
}
