/**
 * IPC seam for note-image persistence.
 *
 * Wraps the generated `save_note_image` command (`commands.saveNoteImage`) so
 * the editor's paste/drop handler works with a `File`/`Uint8Array` shape and so
 * tests mock *this* module (per the architecture testing policy) rather than the
 * generated bindings file.
 *
 * `persistence` owns the on-disk `assets/<filename>` files; `ipc-bridge` exposes
 * `save_note_image`, routing it directly to `persistence::save_note_asset`. The
 * returned value is the PORTABLE filename ref the editor stores into
 * `notes.json` (see `architecture/cross-cutting.md` — "Note image assets").
 */
import { commands, unwrap } from "./client";

/**
 * The custom URI scheme the `app-main` protocol handler serves note images on.
 *
 * Mirrors `ipc_bridge::MEETING_ASSET_SCHEME`. Used with
 * `convertFileSrc(path, MEETING_ASSET_SCHEME)` so the rendered URL is correct
 * per-platform (Windows `http://meetingasset.localhost/...` vs
 * `meetingasset://localhost/...` elsewhere) while the STORED ref stays a bare
 * portable filename.
 */
export const MEETING_ASSET_SCHEME = "meetingasset";

/** The image MIME types accepted on paste/drop, mapped to their stored ext. */
const MIME_TO_EXT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpeg",
  "image/gif": "gif",
  "image/webp": "webp",
};

/**
 * Map a `File`'s type (or, as a fallback, its name extension) to the storage
 * extension the backend allowlist accepts, or `null` when it is not an image we
 * handle. Centralised so the paste and drop paths agree.
 */
export function imageExtForFile(file: File): string | null {
  const byMime = MIME_TO_EXT[file.type];
  if (byMime) return byMime;
  // Fallback: some clipboards drop the MIME type; derive from the name.
  const dot = file.name.lastIndexOf(".");
  if (dot < 0) return null;
  const ext = file.name.slice(dot + 1).toLowerCase();
  const allowed = new Set(["png", "jpg", "jpeg", "gif", "webp"]);
  return allowed.has(ext) ? ext : null;
}

/**
 * Persist a pasted/dropped image `File` for `meetingId`, returning the portable
 * filename ref to store into the document. Rejects (via the thrown IPC error)
 * when the backend refuses the extension or the write fails.
 */
export async function saveNoteImageFile(
  meetingId: string,
  file: File,
  ext: string,
): Promise<string> {
  const buffer = await file.arrayBuffer();
  const bytes = Array.from(new Uint8Array(buffer));
  return unwrap(await commands.saveNoteImage(meetingId, bytes, ext));
}
