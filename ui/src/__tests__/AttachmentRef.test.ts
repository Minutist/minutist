/**
 * Behaviour tests for the attachment-ref node + drop/paste handling (#0038).
 *
 * The IPC seam (`../ipc/attachments`) is mocked per the architecture testing
 * policy, so these tests assert the editor wiring — that a dropped/pasted
 * file of ANY type calls `addAttachment` (via the attachments store's `add`)
 * and inserts an `attachmentRef` node carrying the returned entry's portable
 * ref — without a backend, and that the node view renders an image thumbnail
 * or a file-type card depending on extension.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// `convertFileSrc` is exercised by the AttachmentRef node view's image-thumbnail
// render path; stub it (and the invoke seam) so the editor constructs under
// jsdom.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  Channel: vi.fn(),
  convertFileSrc: (path: string, scheme?: string) =>
    `${scheme ?? "asset"}://localhost/${path}`,
}));

// Mock the IPC seam the attachments store's `add` action calls: record the
// added file and return a deterministic manifest entry.
const addAttachment = vi.fn(
  async (_meetingId: string, file: File, ext: string) => ({
    id: "att-1",
    hash: "deadbeef",
    original_filename: file.name,
    ext,
    byte_len: file.size,
    added_at: "2026-01-01T00:00:00Z",
    conversion: { state: "pending" as const },
  }),
);
vi.mock("../ipc/attachments", async () => {
  const actual = await vi.importActual<typeof import("../ipc/attachments")>(
    "../ipc/attachments",
  );
  return {
    ...actual,
    addAttachment: (...args: Parameters<typeof addAttachment>) =>
      addAttachment(...args),
  };
});

import { Editor } from "@tiptap/core";
import { buildEditorExtensions } from "../editor/extensions";
import {
  handleAttachmentDrop,
  handleAttachmentPaste,
} from "../editor/attachment-drop";
import { useAttachmentsStore } from "../state/attachments";

const MEETING_ID = "11111111-1111-4111-8111-111111111111";

function makeEditor(onOpenAttachment?: (id: string) => void): Editor {
  return new Editor({
    extensions: buildEditorExtensions({
      clockSource: () => ({ recording: false, clockMs: null }),
      meetingIdSource: () => MEETING_ID,
      onOpenAttachment,
    }),
    content: "<p></p>",
  });
}

/** A minimal `File` of a given MIME type with real bytes. */
function makeFile(name: string, type: string): File {
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

/** The `attachmentRef` node attrs in the document, in order. */
function attachmentRefAttrs(editor: Editor): Record<string, unknown>[] {
  const nodes: Record<string, unknown>[] = [];
  editor.state.doc.descendants((node) => {
    if (node.type.name === "attachmentRef") nodes.push(node.attrs);
  });
  return nodes;
}

/** Count `image` nodes (the back-compat `NoteImage` node) in the document. */
function imageNodeCount(editor: Editor): number {
  let n = 0;
  editor.state.doc.descendants((node) => {
    if (node.type.name === "image") n += 1;
  });
  return n;
}

describe("attachment drop/paste (#0038)", () => {
  let editor: Editor;

  beforeEach(() => {
    addAttachment.mockClear();
    useAttachmentsStore.setState({
      attachments: [],
      meetingId: null,
      loading: false,
      adding: 0,
      lastError: null,
    });
    editor = makeEditor();
  });

  it("a dropped non-image file calls addAttachment and inserts an attachmentRef node", async () => {
    const event = {
      dataTransfer: fakeDataTransfer({
        files: [makeFile("report.pdf", "application/pdf")],
      }),
      preventDefault: vi.fn(),
    } as unknown as DragEvent;

    const handled = handleAttachmentDrop(editor, event, () => MEETING_ID);
    expect(handled).toBe(true);

    await vi.waitFor(() => expect(addAttachment).toHaveBeenCalledTimes(1));
    expect(addAttachment).toHaveBeenCalledWith(
      MEETING_ID,
      expect.any(File),
      "pdf",
    );

    await vi.waitFor(() => expect(attachmentRefAttrs(editor)).toHaveLength(1));
    const [attrs] = attachmentRefAttrs(editor);
    expect(attrs).toMatchObject({
      attachmentId: "att-1",
      filename: "deadbeef.pdf",
      originalFilename: "report.pdf",
      ext: "pdf",
    });
  });

  it("a dropped image ALSO routes through the attachment pipeline, not NoteImage", async () => {
    const event = {
      dataTransfer: fakeDataTransfer({
        files: [makeFile("shot.png", "image/png")],
      }),
      preventDefault: vi.fn(),
    } as unknown as DragEvent;

    const handled = handleAttachmentDrop(editor, event, () => MEETING_ID);
    expect(handled).toBe(true);

    await vi.waitFor(() => expect(attachmentRefAttrs(editor)).toHaveLength(1));
    expect(addAttachment).toHaveBeenCalledWith(
      MEETING_ID,
      expect.any(File),
      "png",
    );
    // No standalone `image` (NoteImage) node was created for the new drop.
    expect(imageNodeCount(editor)).toBe(0);
  });

  it("a pasted file calls addAttachment and inserts an attachmentRef node", async () => {
    const event = {
      clipboardData: fakeDataTransfer({
        files: [makeFile("notes.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")],
      }),
      preventDefault: vi.fn(),
    } as unknown as ClipboardEvent;

    const handled = handleAttachmentPaste(editor, event, () => MEETING_ID);
    expect(handled).toBe(true);
    await vi.waitFor(() => expect(addAttachment).toHaveBeenCalledWith(
      MEETING_ID,
      expect.any(File),
      "docx",
    ));
    await vi.waitFor(() => expect(attachmentRefAttrs(editor)).toHaveLength(1));
  });

  it("a text-only paste is NOT intercepted (markdown handling owns it)", () => {
    const event = {
      clipboardData: fakeDataTransfer({ text: "# Heading\n\nbody" }),
      preventDefault: vi.fn(),
    } as unknown as ClipboardEvent;

    expect(handleAttachmentPaste(editor, event, () => MEETING_ID)).toBe(false);
    expect(addAttachment).not.toHaveBeenCalled();
  });

  it("a rich paste carrying BOTH a file and text is left to the text handler", () => {
    const event = {
      clipboardData: fakeDataTransfer({
        files: [makeFile("a.pdf", "application/pdf")],
        html: "<p>rich content</p>",
      }),
      preventDefault: vi.fn(),
    } as unknown as ClipboardEvent;

    expect(handleAttachmentPaste(editor, event, () => MEETING_ID)).toBe(false);
    expect(addAttachment).not.toHaveBeenCalled();
  });

  it("a drop with no meeting open is not intercepted", () => {
    const event = {
      dataTransfer: fakeDataTransfer({
        files: [makeFile("a.pdf", "application/pdf")],
      }),
      preventDefault: vi.fn(),
    } as unknown as DragEvent;

    expect(handleAttachmentDrop(editor, event, () => null)).toBe(false);
    expect(addAttachment).not.toHaveBeenCalled();
  });

  it("a file with no extension is skipped via onError, not thrown", async () => {
    const onError = vi.fn();
    const event = {
      dataTransfer: fakeDataTransfer({
        files: [makeFile("noext", "application/octet-stream")],
      }),
      preventDefault: vi.fn(),
    } as unknown as DragEvent;

    handleAttachmentDrop(editor, event, () => MEETING_ID, onError);
    await vi.waitFor(() => expect(onError).toHaveBeenCalledTimes(1));
    expect(addAttachment).not.toHaveBeenCalled();
  });
});

describe("AttachmentRef node view", () => {
  let editor: Editor;

  beforeEach(() => {
    editor = makeEditor();
  });

  it("renders an image extension as a thumbnail resolved via convertFileSrc", () => {
    editor.commands.insertContent({
      type: "attachmentRef",
      attrs: {
        attachmentId: "att-2",
        filename: "abc123.png",
        originalFilename: "shot.png",
        ext: "png",
        byteLen: 2048,
      },
    });
    const img = editor.view.dom.querySelector(
      ".attachment-ref__thumb",
    ) as HTMLImageElement | null;
    expect(img).not.toBeNull();
    expect(img?.src).toBe(`attachment://localhost/${MEETING_ID}/abc123.png`);
    expect(img?.alt).toBe("shot.png");
  });

  it("renders a non-image extension as a file-type card", () => {
    editor.commands.insertContent({
      type: "attachmentRef",
      attrs: {
        attachmentId: "att-3",
        filename: "abc123.pdf",
        originalFilename: "report.pdf",
        ext: "pdf",
        byteLen: 4200,
      },
    });
    const card = editor.view.dom.querySelector(".attachment-ref__card");
    expect(card).not.toBeNull();
    expect(card?.querySelector(".attachment-ref__name")?.textContent).toBe(
      "report.pdf",
    );
    expect(card?.querySelector(".attachment-ref__icon")?.textContent).toBe(
      "PDF",
    );
    expect(card?.querySelector(".attachment-ref__size")?.textContent).toBe(
      "4.2 kB",
    );
  });

  it("clicking a file card's expand affordance calls onOpenAttachment with the attachment id", () => {
    const onOpenAttachment = vi.fn();
    const withCallback = makeEditor(onOpenAttachment);
    withCallback.commands.insertContent({
      type: "attachmentRef",
      attrs: {
        attachmentId: "att-4",
        filename: "abc123.docx",
        originalFilename: "doc.docx",
        ext: "docx",
        byteLen: 100,
      },
    });
    const card = withCallback.view.dom.querySelector(
      ".attachment-ref__card",
    ) as HTMLElement | null;
    expect(card).not.toBeNull();
    card?.click();
    expect(onOpenAttachment).toHaveBeenCalledWith("att-4");
    withCallback.destroy();
  });
});
