/**
 * Anchor-marginalia decoration (Editorial Ink, presentation-only).
 *
 * Renders each anchored paragraph's `data-anchor-ms` value as a quiet
 * timestamp in the LEFT margin gutter — an editorial side-note. This is purely
 * a ProseMirror *decoration*: it adds no node attributes and dispatches no
 * transactions, so it cannot interfere with the `ParagraphAnchor` stamping
 * logic (which owns the `data-anchor-ms` attribute) and never shifts the text
 * column (the marginalia is absolutely positioned into the sheet's left gutter
 * by `Editor.css`).
 *
 * Anchors live on the recording-clock timeline (same origin as
 * `Segment::start_ms`); the gutter shows a coarse `M:SS` / `H:MM:SS` form (see
 * `formatAnchorMark`) — finer than the transcript's `MM:SS.cc` would overflow
 * the narrow gutter and adds no value to a margin side-note.
 */
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import type { EditorState } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { ANCHOR_ATTR } from "./paragraph-anchor";

const anchorMarginaliaPluginKey = new PluginKey("anchorMarginalia");

/**
 * Format a recording-clock millisecond offset for the gutter side-note as
 * `M:SS` / `MM:SS`, rolling into `H:MM:SS` past an hour. Deliberately coarser
 * than the transcript pane's `MM:SS.cc`: centisecond precision is noise in a
 * margin note, and dropping it keeps even a multi-hour meeting's stamp inside
 * the narrow timestamp gutter (the unbounded `MM:SS.cc` form overflowed it).
 */
export function formatAnchorMark(ms: number): string {
  const totalSeconds = Math.floor(Math.max(0, ms) / 1000);
  const ss = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const mm = totalMinutes % 60;
  const hh = Math.floor(totalMinutes / 60);
  const pad2 = (n: number) => String(n).padStart(2, "0");
  return hh > 0 ? `${hh}:${pad2(mm)}:${pad2(ss)}` : `${mm}:${pad2(ss)}`;
}

/** Build the decoration set: one gutter widget per anchored paragraph. */
function buildDecorations(state: EditorState): DecorationSet {
  const decorations: Decoration[] = [];
  state.doc.descendants((node: ProseMirrorNode, pos: number) => {
    if (node.type.name !== "paragraph") return;
    const raw = node.attrs[ANCHOR_ATTR];
    if (raw === null || raw === undefined) return;
    const ms = typeof raw === "number" ? raw : Number.parseInt(String(raw), 10);
    if (Number.isNaN(ms)) return;

    decorations.push(
      Decoration.widget(
        pos + 1,
        () => {
          const el = document.createElement("span");
          el.className = "notes-editor__anchor-mark tnum";
          el.setAttribute("contenteditable", "false");
          el.setAttribute("aria-hidden", "true");
          el.textContent = formatAnchorMark(ms);
          return el;
        },
        // Render on the left, never selectable, and ignored by mapping so it
        // stays pinned to the paragraph start.
        { side: -1, ignoreSelection: true, key: `anchor-${pos}-${ms}` },
      ),
    );
  });
  return DecorationSet.create(state.doc, decorations);
}

/**
 * Install a decoration-only plugin that paints anchor marginalia. Recomputed
 * on every doc change; cheap for the document sizes this editor handles.
 */
export const AnchorMarginalia = Extension.create({
  name: "anchorMarginalia",

  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: anchorMarginaliaPluginKey,
        state: {
          init: (_config, state) => buildDecorations(state),
          apply: (tr, old, _oldState, newState) =>
            tr.docChanged ? buildDecorations(newState) : old,
        },
        props: {
          decorations(state) {
            return anchorMarginaliaPluginKey.getState(state) as
              | DecorationSet
              | undefined;
          },
        },
      }),
    ];
  },
});
