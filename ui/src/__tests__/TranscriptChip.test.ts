/**
 * Behaviour tests for the transcript-chip node (FR-24 / FR-25).
 *
 * Covers:
 *   - chip insertion on a transcript-segment drop (FR-24),
 *   - getJSON → setContent round-trip preserving the chip with its attrs (FR-25,
 *     the notes.json opacity guarantee),
 *   - tiptap-markdown fenced-quotation export carrying start_ms/end_ms/speaker
 *     and the segment text (FR-25),
 *   - the node view rendering the chip into the DOM.
 *
 * Runs headless against a real Tiptap editor (jsdom), the same path the live
 * editor uses, so the node config + markdown hook are exercised end-to-end.
 */
import { describe, it, expect, vi, afterEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

import { Editor } from "@tiptap/core";
import { buildEditorExtensions } from "../editor/extensions";
import { CHIP_NODE_NAME } from "../editor/transcript-chip";
import {
  handleSegmentDrop,
  writeSegmentDrag,
  readSegmentDrag,
  segmentToDragPayload,
} from "../editor/transcript-dnd";
import type { Segment } from "../ipc/bindings";

/** Build a headless editor with the production extension set. */
function makeEditor(): Editor {
  return new Editor({
    extensions: buildEditorExtensions({
      clockSource: () => ({ recording: false, clockMs: null }),
    }),
    content: "<p>before</p>",
  });
}

/** A minimal `DataTransfer` stand-in for jsdom drop simulation. */
function fakeDataTransfer(initial: Record<string, string> = {}): DataTransfer {
  const store = new Map<string, string>(Object.entries(initial));
  return {
    setData: (type: string, value: string) => {
      store.set(type, value);
    },
    getData: (type: string) => store.get(type) ?? "",
    effectAllowed: "uninitialized",
  } as unknown as DataTransfer;
}

const SEGMENT: Segment = {
  start_ms: 12_400,
  end_ms: 21_300,
  text: "First item is the launch checklist.",
  speaker_id: "A",
  words: [],
};

/** Read the chip node (if any) from the editor's current JSON doc. */
function chipNodes(editor: Editor): Array<{ type: string; attrs: Record<string, unknown> }> {
  const doc = editor.getJSON() as {
    content?: Array<{ type: string; attrs?: Record<string, unknown> }>;
  };
  return (doc.content ?? [])
    .filter((n) => n.type === CHIP_NODE_NAME)
    .map((n) => ({ type: n.type, attrs: n.attrs ?? {} }));
}

describe("transcript-chip node", () => {
  let editors: Editor[] = [];
  afterEach(() => {
    editors.forEach((e) => e.destroy());
    editors = [];
  });
  function track(editor: Editor): Editor {
    editors.push(editor);
    return editor;
  }

  it("inserts a chip carrying the segment metadata on a transcript-segment drop (FR-24)", () => {
    const editor = track(makeEditor());

    const dataTransfer = fakeDataTransfer();
    writeSegmentDrag(dataTransfer, SEGMENT);
    const dropEvent = {
      dataTransfer,
      clientX: undefined,
      clientY: undefined,
      preventDefault: () => {},
    } as unknown as DragEvent;

    const handled = handleSegmentDrop(editor, dropEvent);
    expect(handled).toBe(true);

    const chips = chipNodes(editor);
    expect(chips).toHaveLength(1);
    expect(chips[0].attrs.startMs).toBe(12_400);
    expect(chips[0].attrs.endMs).toBe(21_300);
    expect(chips[0].attrs.speakerId).toBe("A");
    expect(chips[0].attrs.text).toBe("First item is the launch checklist.");
  });

  it("ignores a drop that does not carry a transcript segment", () => {
    const editor = track(makeEditor());
    const dropEvent = {
      dataTransfer: fakeDataTransfer({ "text/plain": "hello" }),
      preventDefault: () => {},
    } as unknown as DragEvent;

    expect(handleSegmentDrop(editor, dropEvent)).toBe(false);
    expect(chipNodes(editor)).toHaveLength(0);
  });

  it("preserves the chip across a getJSON → setContent round-trip (FR-25 opacity)", () => {
    const editor = track(makeEditor());
    editor.commands.insertTranscriptChip({
      startMs: SEGMENT.start_ms,
      endMs: SEGMENT.end_ms,
      speakerId: SEGMENT.speaker_id ?? null,
      text: SEGMENT.text,
    });

    // Serialise to JSON (what notes.json stores opaquely) and reload it.
    const json = editor.getJSON();
    const reloaded = track(makeEditor());
    reloaded.commands.setContent(json);

    const chips = chipNodes(reloaded);
    expect(chips).toHaveLength(1);
    expect(chips[0].attrs).toMatchObject({
      startMs: 12_400,
      endMs: 21_300,
      speakerId: "A",
      text: "First item is the launch checklist.",
    });
  });

  it("exports the chip as a fenced ```transcript quotation in markdown (FR-25)", () => {
    const editor = track(makeEditor());
    editor.commands.insertTranscriptChip({
      startMs: SEGMENT.start_ms,
      endMs: SEGMENT.end_ms,
      speakerId: SEGMENT.speaker_id ?? null,
      text: SEGMENT.text,
    });

    const markdownStorage = (
      editor.storage as unknown as Record<string, unknown>
    )["markdown"] as { getMarkdown: () => string };
    const md = markdownStorage.getMarkdown();

    // Fence opens with the `transcript` info string carrying the metadata…
    expect(md).toContain("```transcript");
    expect(md).toContain("start_ms=12400");
    expect(md).toContain("end_ms=21300");
    expect(md).toContain("speaker=A");
    // …and the body is the quoted segment text.
    expect(md).toContain("First item is the launch checklist.");
    // The fence closes.
    expect(md).toContain("```");
  });

  it("omits the speaker token in markdown when the segment has no speaker", () => {
    const editor = track(makeEditor());
    editor.commands.insertTranscriptChip({
      startMs: 1000,
      endMs: 2000,
      speakerId: null,
      text: "No speaker here.",
    });
    const markdownStorage = (
      editor.storage as unknown as Record<string, unknown>
    )["markdown"] as { getMarkdown: () => string };
    const md = markdownStorage.getMarkdown();
    expect(md).toContain("```transcript");
    expect(md).not.toContain("speaker=");
  });

  it("renders the chip via its node view into the editor DOM", () => {
    const editor = track(makeEditor());
    editor.commands.insertTranscriptChip({
      startMs: SEGMENT.start_ms,
      endMs: SEGMENT.end_ms,
      speakerId: SEGMENT.speaker_id ?? null,
      text: SEGMENT.text,
    });

    const chipEl = editor.view.dom.querySelector("[data-transcript-chip]");
    expect(chipEl).not.toBeNull();
    expect(chipEl?.querySelector(".transcript-chip__text")?.textContent).toBe(
      "First item is the launch checklist.",
    );
    // The meta line carries the speaker and a MM:SS.cc stamp for 12_400 ms.
    const meta = chipEl?.querySelector(".transcript-chip__meta")?.textContent;
    expect(meta).toContain("A");
    expect(meta).toContain("00:12.40");
  });
});

describe("transcript-dnd payload helpers", () => {
  it("round-trips a segment through write/read on a DataTransfer", () => {
    const dataTransfer = fakeDataTransfer();
    writeSegmentDrag(dataTransfer, SEGMENT);
    const parsed = readSegmentDrag(dataTransfer);
    expect(parsed).toEqual(segmentToDragPayload(SEGMENT));
    // The text/plain fallback carries the bare segment text.
    expect(dataTransfer.getData("text/plain")).toBe(SEGMENT.text);
  });

  it("returns null for a DataTransfer without the segment MIME type", () => {
    expect(readSegmentDrag(fakeDataTransfer({ "text/plain": "x" }))).toBeNull();
    expect(readSegmentDrag(null)).toBeNull();
  });
});
