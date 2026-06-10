/**
 * Note-image Tiptap node — pasted/dropped images stored as FILES in the meeting
 * folder, referenced PORTABLY from `notes.json`.
 *
 * # What is stored vs what is rendered (the portability contract)
 *
 * The node's `src` attribute holds a **portable** reference: the bare asset
 * filename (`<contenthash>.<ext>`) the backend returned from `save_note_image`.
 * This is what `editor.getJSON()` serialises into `notes.json` — NOT a
 * machine-specific absolute path and NOT a platform-specific webview URL.
 * Because `notes.json` and the asset both live in the meeting folder
 * (`assets/<filename>`), the folder can be copied to another machine and the
 * reference still resolves.
 *
 * At DISPLAY time the node view converts that portable ref into a working
 * webview URL via Tauri's `convertFileSrc(<meeting_id>/<filename>,
 * "meetingasset")`, which yields `meetingasset://localhost/...` (macOS/Linux)
 * or `http://meetingasset.localhost/...` (Windows). The conversion is per-render
 * and per-platform; nothing platform-specific is ever written back to the
 * document. The `app-main` `meetingasset:` protocol handler serves the bytes
 * from `{meetings_dir}/<meeting_id>/assets/<filename>`.
 *
 * The meeting id is supplied by the editor (it is not part of the portable ref,
 * since the asset lives under the same meeting folder as `notes.json`); a stored
 * document never needs a meeting id baked in.
 */
import { Image } from "@tiptap/extension-image";
import { convertFileSrc } from "@tauri-apps/api/core";
import { MEETING_ASSET_SCHEME } from "../ipc/note-images";

/** Supplies the editor's current meeting id (or `null` when none is open). */
export type MeetingIdSource = () => string | null;

export type NoteImageOptions = {
  /** Reads the current meeting id so a portable ref resolves to its asset. */
  meetingIdSource: MeetingIdSource;
};

/**
 * Convert a stored portable image `src` into a webview-loadable URL.
 *
 * A bare filename (no scheme, no slash) is treated as a meeting-asset reference
 * and converted via `convertFileSrc(<meetingId>/<filename>, "meetingasset")`.
 * Anything that already looks like a URL (has a scheme or is a data: URI, e.g.
 * pasted remote/inline images) is passed through unchanged. When no meeting is
 * open a bare ref cannot be resolved, so it is returned as-is (the image simply
 * won't load — there is nowhere to have saved it).
 */
export function resolveImageSrc(
  src: string | null | undefined,
  meetingId: string | null,
): string {
  if (!src) return "";
  // Already a URL (scheme present) or a data URI — leave untouched.
  if (/^[a-z][a-z0-9+.-]*:/i.test(src) || src.startsWith("//")) {
    return src;
  }
  // A bare filename: resolve against the open meeting's assets directory.
  if (meetingId === null) return src;
  return convertFileSrc(`${meetingId}/${src}`, MEETING_ASSET_SCHEME);
}

/**
 * The note-image extension.
 *
 * Extends `@tiptap/extension-image` (which owns the `image` node schema,
 * `src`/`alt`/`title` attrs, HTML parse/serialise, and the `setImage` command)
 * and overrides `addNodeView` so the live editor renders the CONVERTED display
 * URL while `getJSON` keeps the portable `src`. We do NOT override `renderHTML`:
 * leaving it as the base (portable `src`) means clipboard/markdown HTML export
 * carries the portable ref too, so a copy/paste round-trip stays portable.
 */
export const NoteImage = Image.extend<NoteImageOptions>({
  addOptions() {
    return {
      ...this.parent?.(),
      // Inline images render in a paragraph; keep the default (block) — pasted
      // screenshots read as block figures, matching the transcript-chip style.
      inline: false,
      // A no-op default so the extension is safe to construct without wiring
      // (mirrors ParagraphAnchor's default clock source).
      meetingIdSource: () => null,
    };
  },

  addNodeView() {
    const meetingIdSource = this.options.meetingIdSource;
    return ({ node }) => {
      const img = document.createElement("img");
      img.className = "note-image";
      const stored = (node.attrs.src as string | null) ?? "";
      img.src = resolveImageSrc(stored, meetingIdSource());
      const alt = node.attrs.alt as string | null;
      if (alt) img.alt = alt;
      const title = node.attrs.title as string | null;
      if (title) img.title = title;
      return { dom: img };
    };
  },
});
