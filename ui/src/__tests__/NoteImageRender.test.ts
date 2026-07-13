/**
 * Behaviour tests for `resolveImageSrc` — the `NoteImage` node's portable ref
 * → display URL conversion.
 *
 * `NoteImage` is back-compat only (#0038 moved the notes-editor drop/paste
 * path onto `AttachmentRef`; see `AttachmentRef.test.ts`): it still renders
 * images already embedded as note-assets in existing meetings, so this
 * conversion must keep working even though nothing new creates the node.
 */
import { describe, it, expect, vi } from "vitest";

// `resolveImageSrc` calls `convertFileSrc` directly; stub it so this test does
// not need a live Tauri runtime.
vi.mock("@tauri-apps/api/core", () => ({
  convertFileSrc: (path: string, scheme?: string) =>
    `${scheme ?? "asset"}://localhost/${path}`,
}));

import { resolveImageSrc } from "../editor/note-image";

const MEETING_ID = "11111111-1111-4111-8111-111111111111";

describe("resolveImageSrc — portable ref → display URL", () => {
  it("converts a bare filename via convertFileSrc against the open meeting", () => {
    expect(resolveImageSrc("deadbeef.png", MEETING_ID)).toBe(
      `meetingasset://localhost/${MEETING_ID}/deadbeef.png`,
    );
  });

  it("passes through an existing URL or data URI unchanged", () => {
    expect(resolveImageSrc("https://x/y.png", MEETING_ID)).toBe(
      "https://x/y.png",
    );
    expect(resolveImageSrc("data:image/png;base64,AAAA", MEETING_ID)).toBe(
      "data:image/png;base64,AAAA",
    );
    expect(resolveImageSrc("meetingasset://localhost/a/b.png", MEETING_ID)).toBe(
      "meetingasset://localhost/a/b.png",
    );
  });

  it("returns a bare ref as-is when no meeting is open (cannot resolve)", () => {
    expect(resolveImageSrc("deadbeef.png", null)).toBe("deadbeef.png");
  });

  it("empty/nullish src yields empty string", () => {
    expect(resolveImageSrc("", MEETING_ID)).toBe("");
    expect(resolveImageSrc(null, MEETING_ID)).toBe("");
    expect(resolveImageSrc(undefined, MEETING_ID)).toBe("");
  });
});
