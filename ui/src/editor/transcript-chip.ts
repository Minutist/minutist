/**
 * Transcript-chip Tiptap node (FR-24 / FR-25).
 *
 * A first-class block node inserted into the notes when the user drags a
 * transcript segment into the editor. The chip captures the segment it came
 * from — `start_ms` / `end_ms` (recording-clock, pause-EXCLUDING, the same
 * timeline as `Segment::start_ms`), `speaker_id`, and `text` — so the notes
 * stay tied to the transcript at segment granularity.
 *
 * Two guarantees this node has to satisfy:
 *
 *  - **Round-trip through `notes.json` (FR-25).** The chip is a real
 *    ProseMirror node with explicit `attrs`, parsed back from a
 *    `data-transcript-chip` DOM element. `persistence` stores `notes.json` as an
 *    opaque `serde_json::Value` (the Phase-3 opacity guarantee), so a `getJSON`
 *    → `setContent` cycle reconstructs the chip losslessly with its attributes
 *    intact.
 *  - **Markdown export as a fenced quotation (FR-25).** `tiptap-markdown` reads
 *    a node's `addStorage().markdown.serialize` hook (see
 *    `tiptap-markdown/src/serialize/MarkdownSerializer`); we emit a fenced
 *    ```transcript block carrying `start_ms` / `end_ms` / `speaker_id` on the
 *    info line and the segment text as the body, so a chip survives the
 *    `notes.md` export with its metadata.
 *
 * The chip is an `atom` (leaf) node: it has no editable inner content, so the
 * captured text cannot drift from the segment it quotes. It is `draggable` and
 * `selectable` so it behaves like a normal block in the document.
 */
import { Node, mergeAttributes } from "@tiptap/core";
import { formatTimestamp } from "../transcript/TranscriptPane";

/** The DOM attribute / fenced-info keys the chip round-trips through. */
export const CHIP_NODE_NAME = "transcriptChip";

/** Attributes captured from the source `Segment` when a chip is inserted. */
export type TranscriptChipAttrs = {
  startMs: number;
  endMs: number;
  speakerId: string | null;
  text: string;
};

/**
 * `MarkdownSerializerState` surface the chip's serializer uses. Declared
 * locally because `tiptap-markdown` ships no types for it; only the two methods
 * the chip needs are described.
 */
type MarkdownState = {
  write: (text: string) => void;
  closeBlock: (node: unknown) => void;
};

declare module "@tiptap/core" {
  interface Commands<ReturnType> {
    transcriptChip: {
      /** Insert a transcript chip carrying the given segment metadata. */
      insertTranscriptChip: (attrs: TranscriptChipAttrs) => ReturnType;
    };
  }
}

/** Parse a numeric DOM attribute, defaulting to 0 when absent/NaN. */
function numAttr(element: HTMLElement, name: string): number {
  const raw = element.getAttribute(name);
  if (raw === null) return 0;
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) ? 0 : parsed;
}

/**
 * Build the fenced-quotation info line, e.g.
 * `transcript start_ms=12400 end_ms=21300 speaker=A`. `speaker` is omitted when
 * the segment carries no speaker id.
 */
function fenceInfo(attrs: TranscriptChipAttrs): string {
  const parts = [
    "transcript",
    `start_ms=${attrs.startMs}`,
    `end_ms=${attrs.endMs}`,
  ];
  if (attrs.speakerId !== null && attrs.speakerId !== "") {
    parts.push(`speaker=${attrs.speakerId}`);
  }
  return parts.join(" ");
}

export const TranscriptChip = Node.create({
  name: CHIP_NODE_NAME,

  group: "block",
  atom: true,
  selectable: true,
  draggable: true,

  addAttributes() {
    return {
      startMs: {
        default: 0,
        parseHTML: (element) => numAttr(element, "data-start-ms"),
        renderHTML: (attributes) => ({
          "data-start-ms": String(attributes.startMs ?? 0),
        }),
      },
      endMs: {
        default: 0,
        parseHTML: (element) => numAttr(element, "data-end-ms"),
        renderHTML: (attributes) => ({
          "data-end-ms": String(attributes.endMs ?? 0),
        }),
      },
      speakerId: {
        default: null,
        parseHTML: (element) => element.getAttribute("data-speaker-id"),
        renderHTML: (attributes) => {
          const value = attributes.speakerId;
          if (value === null || value === undefined || value === "") return {};
          return { "data-speaker-id": String(value) };
        },
      },
      text: {
        default: "",
        parseHTML: (element) =>
          element.getAttribute("data-text") ?? element.textContent ?? "",
        renderHTML: (attributes) => ({
          "data-text": String(attributes.text ?? ""),
        }),
      },
    };
  },

  parseHTML() {
    return [{ tag: "div[data-transcript-chip]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    // A static (non-node-view) HTML rendering so the chip serialises to HTML
    // (clipboard / round-trip) even outside the live editor. The interactive
    // appearance is provided by the node view below.
    const speaker = node.attrs.speakerId
      ? `${node.attrs.speakerId} · `
      : "";
    const stamp = formatTimestamp(Number(node.attrs.startMs) || 0);
    return [
      "div",
      mergeAttributes(HTMLAttributes, {
        "data-transcript-chip": "",
        class: "transcript-chip",
      }),
      [
        "span",
        { class: "transcript-chip__meta" },
        `${speaker}${stamp}`,
      ],
      ["span", { class: "transcript-chip__text" }, String(node.attrs.text ?? "")],
    ];
  },

  // A presentation node view so the chip reads as a quiet quotation card inside
  // the sheet. It mirrors `renderHTML`; ProseMirror still owns selection/drag
  // because the node is an atom.
  addNodeView() {
    return ({ node }) => {
      const dom = document.createElement("div");
      dom.setAttribute("data-transcript-chip", "");
      dom.className = "transcript-chip";

      const meta = document.createElement("span");
      meta.className = "transcript-chip__meta tnum";
      const speaker = node.attrs.speakerId ? `${node.attrs.speakerId} · ` : "";
      meta.textContent = `${speaker}${formatTimestamp(Number(node.attrs.startMs) || 0)}`;

      const body = document.createElement("span");
      body.className = "transcript-chip__text";
      body.textContent = String(node.attrs.text ?? "");

      dom.appendChild(meta);
      dom.appendChild(body);
      return { dom };
    };
  },

  addCommands() {
    return {
      insertTranscriptChip:
        (attrs: TranscriptChipAttrs) =>
        ({ commands }) =>
          commands.insertContent({
            type: CHIP_NODE_NAME,
            attrs: {
              startMs: attrs.startMs,
              endMs: attrs.endMs,
              speakerId: attrs.speakerId,
              text: attrs.text,
            },
          }),
    };
  },

  // tiptap-markdown hook: export the chip as a fenced ```transcript quotation
  // so its metadata survives the `notes.md` export (FR-25).
  addStorage() {
    return {
      markdown: {
        serialize(state: MarkdownState, node: { attrs: Record<string, unknown> }) {
          const attrs: TranscriptChipAttrs = {
            startMs: Number(node.attrs.startMs) || 0,
            endMs: Number(node.attrs.endMs) || 0,
            speakerId:
              node.attrs.speakerId === null || node.attrs.speakerId === undefined
                ? null
                : String(node.attrs.speakerId),
            text: String(node.attrs.text ?? ""),
          };
          state.write("```" + fenceInfo(attrs) + "\n");
          state.write(attrs.text);
          state.write("\n```");
          state.closeBlock(node);
        },
        parse: {
          // Round-trip into the editor is via `notes.json` (the opaque store),
          // not the markdown export, so no markdown-it parse rule is needed.
        },
      },
    };
  },
});
