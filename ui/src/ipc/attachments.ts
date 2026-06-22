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
 * `open` does NOT return bytes or write a temp file: it asks the backend to
 * validate the original is reachable, then builds the per-platform webview URL
 * with `convertFileSrc(<meeting_id>/<hash>.<ext>, MEETING_DOC_SCHEME)` and hands
 * it to `tauri-plugin-opener`, which opens it in the OS default application. The
 * `app-main` `meetingdoc:` protocol handler resolves the bytes from
 * `{meetings_dir}/<meeting_id>/attachments/<filename>` (the sibling of the
 * verified `meetingasset:` scheme).
 */
import { convertFileSrc } from "@tauri-apps/api/core";
import { commands, unwrap } from "./client";
import type { AttachmentEntry, AttachmentId, MeetingId } from "./bindings";

/**
 * The custom URI scheme the `app-main` protocol handler serves attachment
 * originals on. Mirrors `ipc_bridge::MEETING_DOC_SCHEME` — the sibling of
 * `meetingasset:` (note images), pointed at the meeting's `attachments/` dir.
 */
export const MEETING_DOC_SCHEME = "meetingdoc";

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
 * Open an attachment original in the OS default application.
 *
 * Validates server-side that the original is reachable (the backend holds the
 * `persistence` edge `app-main` lacks), then converts the stored
 * `<hash>.<ext>` filename into a `meetingdoc:` webview URL and opens it via
 * `tauri-plugin-opener`. No bytes cross the wire and no temp file is written.
 */
export async function openAttachment(
  meetingId: MeetingId,
  entry: AttachmentEntry,
): Promise<void> {
  unwrap(await commands.openAttachment(meetingId, entry.id));
  const url = convertFileSrc(
    `${meetingId}/${entry.hash}.${entry.ext}`,
    MEETING_DOC_SCHEME,
  );
  const { openUrl } = await import("@tauri-apps/plugin-opener");
  await openUrl(url);
}

/** Remove an attachment (dedup-safe unlink happens server-side). */
export async function removeAttachment(
  meetingId: MeetingId,
  attachmentId: AttachmentId,
): Promise<void> {
  unwrap(await commands.removeAttachment(meetingId, attachmentId));
}
