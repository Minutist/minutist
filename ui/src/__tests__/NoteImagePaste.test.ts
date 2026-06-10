/**
 * Behaviour tests for note-image paste/drop (image FILES → meeting folder).
 *
 * The IPC seam (`../ipc/note-images`) is mocked per the architecture testing
 * policy, so these tests assert the editor wiring — that an image paste calls
 * `saveNoteImageFile` and inserts an `image` node carrying the PORTABLE
 * filename ref — without a backend. A text-only paste must NOT be intercepted.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// `convertFileSrc` is exercised by the NoteImage node view's render path; stub
// it (and the invoke seam) so the editor constructs under jsdom.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  Channel: vi.fn(),
  convertFileSrc: (path: string, scheme?: string) =>
    `${scheme ?? "asset"}://localhost/${path}`,
}));

// Mock the IPC seam: record the saved file and return a deterministic portable
// filename ref the editor must store as the image node's `src`.
const saveNoteImageFile = vi.fn(
  async (_meetingId: string, _file: File, ext: string) =>
    `deadbeef.${ext}`,
);
vi.mock("../ipc/note-images", async () => {
  const actual = await vi.importActual<typeof import("../ipc/note-images")>(
    "../ipc/note-images",
  );
  return {
    ...actual,
    saveNoteImageFile: (...args: Parameters<typeof saveNoteImageFile>) =>
      saveNoteImageFile(...args),
  };
});

import { Editor } from "@tiptap/core";
import { buildEditorExtensions } from "../editor/extensions";
import {
  handleImagePaste,
  handleImageDrop,
  imageFilesFromDataTransfer,
} from "../editor/image-paste";
import { resolveImageSrc } from "../editor/note-image";

const MEETING_ID = "11111111-1111-4111-8111-111111111111";

function makeEditor(): Editor {
  return new Editor({
    extensions: buildEditorExtensions({
      clockSource: () => ({ recording: false, clockMs: null }),
      meetingIdSource: () => MEETING_ID,
    }),
    content: "<p></p>",
  });
}

/** A minimal File of a given MIME type with real bytes. */
function imageFile(name: string, type: string): File {
  return new File([new Uint8Array([1, 2, 3, 4])], name, { type });
}

/** Build a fake DataTransfer carrying files + optional text/html payloads. */
function fakeDataTransfer(opts: {
  files?: File[];
  text?: string;
  html?: string;
}): DataTransfer {
  const files = opts.files ?? [];
  return {
    files: files as unknown as FileList,
    items: files.map((f) => ({
      kind: "file",
      type: f.type,
      getAsFile: () => f,
    })) as unknown as DataTransferItemList,
    getData: (mime: string) => {
      if (mime === "text/plain") return opts.text ?? "";
      if (mime === "text/html") return opts.html ?? "";
      return "";
    },
  } as unknown as DataTransfer;
}

/** Count `image` nodes in the document. */
function imageNodeCount(editor: Editor): number {
  let n = 0;
  editor.state.doc.descendants((node) => {
    if (node.type.name === "image") n += 1;
  });
  return n;
}

/** The `src` attrs of all image nodes, in document order. */
function imageSrcs(editor: Editor): string[] {
  const srcs: string[] = [];
  editor.state.doc.descendants((node) => {
    if (node.type.name === "image") srcs.push(String(node.attrs.src ?? ""));
  });
  return srcs;
}

describe("note-image paste/drop", () => {
  let editor: Editor;

  beforeEach(() => {
    saveNoteImageFile.mockClear();
    editor = makeEditor();
  });

  it("extracts image files from a DataTransfer (files and items paths)", () => {
    const png = imageFile("shot.png", "image/png");
    expect(imageFilesFromDataTransfer(fakeDataTransfer({ files: [png] }))).toEqual([
      png,
    ]);
    // No files, only items (the common clipboard-paste shape).
    const itemsOnly = {
      files: [] as unknown as FileList,
      items: [
        { kind: "file", type: "image/png", getAsFile: () => png },
      ] as unknown as DataTransferItemList,
      getData: () => "",
    } as unknown as DataTransfer;
    expect(imageFilesFromDataTransfer(itemsOnly)).toEqual([png]);
  });

  it("an image paste calls save_note_image and inserts an image node with the portable ref", async () => {
    const event = {
      clipboardData: fakeDataTransfer({ files: [imageFile("a.png", "image/png")] }),
      preventDefault: vi.fn(),
    } as unknown as ClipboardEvent;

    const handled = handleImagePaste(editor, event, () => MEETING_ID);
    expect(handled).toBe(true);

    // The async save+insert runs after the synchronous handler returns.
    await vi.waitFor(() => expect(saveNoteImageFile).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(imageNodeCount(editor)).toBe(1));

    // The IPC seam was called with the meeting id + the derived "png" ext.
    expect(saveNoteImageFile).toHaveBeenCalledWith(
      MEETING_ID,
      expect.any(File),
      "png",
    );
    // The STORED src is the PORTABLE bare filename, not a converted URL.
    expect(imageSrcs(editor)).toEqual(["deadbeef.png"]);
  });

  it("a text-only paste is NOT intercepted (markdown handling owns it)", () => {
    const event = {
      clipboardData: fakeDataTransfer({ text: "# Heading\n\nbody" }),
      preventDefault: vi.fn(),
    } as unknown as ClipboardEvent;

    expect(handleImagePaste(editor, event, () => MEETING_ID)).toBe(false);
    expect(saveNoteImageFile).not.toHaveBeenCalled();
  });

  it("a rich paste carrying BOTH an image and text is left to the text handler", () => {
    const event = {
      clipboardData: fakeDataTransfer({
        files: [imageFile("a.png", "image/png")],
        html: "<p>rich content</p>",
      }),
      preventDefault: vi.fn(),
    } as unknown as ClipboardEvent;

    expect(handleImagePaste(editor, event, () => MEETING_ID)).toBe(false);
    expect(saveNoteImageFile).not.toHaveBeenCalled();
  });

  it("an image paste with no meeting open is not intercepted", () => {
    const event = {
      clipboardData: fakeDataTransfer({ files: [imageFile("a.png", "image/png")] }),
      preventDefault: vi.fn(),
    } as unknown as ClipboardEvent;

    expect(handleImagePaste(editor, event, () => null)).toBe(false);
    expect(saveNoteImageFile).not.toHaveBeenCalled();
  });

  it("a dropped image file is saved and inserted", async () => {
    // No client coords: jsdom lacks `elementFromPoint`, so `posAtCoords` would
    // throw; the handler falls back to inserting at the current selection. The
    // production path with coords is covered by the transcript-DnD tests, which
    // exercise the same `posAtCoords` seam.
    const event = {
      dataTransfer: fakeDataTransfer({ files: [imageFile("d.webp", "image/webp")] }),
      preventDefault: vi.fn(),
    } as unknown as DragEvent;

    const handled = handleImageDrop(editor, event, () => MEETING_ID);
    expect(handled).toBe(true);
    await vi.waitFor(() => expect(saveNoteImageFile).toHaveBeenCalledWith(
      MEETING_ID,
      expect.any(File),
      "webp",
    ));
    await vi.waitFor(() => expect(imageSrcs(editor)).toEqual(["deadbeef.webp"]));
  });

  it("a non-image drop is not intercepted", () => {
    const event = {
      dataTransfer: fakeDataTransfer({ text: "plain" }),
      preventDefault: vi.fn(),
    } as unknown as DragEvent;
    expect(handleImageDrop(editor, event, () => MEETING_ID)).toBe(false);
    expect(saveNoteImageFile).not.toHaveBeenCalled();
  });
});

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
