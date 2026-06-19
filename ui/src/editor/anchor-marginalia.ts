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
import { ANCHOR_ATTR, WALL_ATTR } from "./paragraph-anchor";

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

/**
 * Format a wall-clock epoch-ms timestamp as a local time-of-day for the gutter
 * (e.g. "1:18 PM" or "13:18", per the user's locale). This is what the gutter
 * shows; [`formatAnchorMark`] (elapsed) is only the last-resort fallback when
 * neither a stored wall-clock nor a meeting start time is available.
 */
export function formatWallClock(epochMs: number): string {
  return new Date(epochMs).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

/** Coerce a stored attribute (number or numeric string) to a number, or null. */
function attrNumber(raw: unknown): number | null {
  if (raw === null || raw === undefined) return null;
  const n = typeof raw === "number" ? raw : Number.parseInt(String(raw), 10);
  return Number.isNaN(n) ? null : n;
}

/**
 * Build the decoration set: one gutter widget per anchored paragraph, showing
 * the local time-of-day the note was written.
 *
 * `startedAtMs` is the open/recording meeting's start wall-clock (epoch ms), used
 * to derive a time-of-day for notes that predate the stored wall-clock; `null`
 * when unknown.
 */
function buildDecorations(
  state: EditorState,
  startedAtMs: number | null,
): DecorationSet {
  const decorations: Decoration[] = [];
  state.doc.descendants((node: ProseMirrorNode, pos: number) => {
    if (node.type.name !== "paragraph") return;
    const ms = attrNumber(node.attrs[ANCHOR_ATTR]);
    if (ms === null) return;

    // Prefer the wall-clock stamped at anchor time (correct across pauses); else
    // derive from the meeting start + the recording offset (older notes); else,
    // with no start time, fall back to the bare elapsed offset.
    const wall = attrNumber(node.attrs[WALL_ATTR]);
    const label =
      wall !== null
        ? formatWallClock(wall)
        : startedAtMs !== null
          ? formatWallClock(startedAtMs + ms)
          : formatAnchorMark(ms);

    decorations.push(
      Decoration.widget(
        pos + 1,
        () => {
          const el = document.createElement("span");
          el.className = "notes-editor__anchor-mark tnum";
          el.setAttribute("contenteditable", "false");
          el.setAttribute("aria-hidden", "true");
          el.textContent = label;
          return el;
        },
        // Render on the left, never selectable, and ignored by mapping so it
        // stays pinned to the paragraph start.
        { side: -1, ignoreSelection: true, key: `anchor-${pos}-${label}` },
      ),
    );
  });
  return DecorationSet.create(state.doc, decorations);
}

/**
 * Install a decoration-only plugin that paints anchor marginalia. Recomputed
 * on every doc change; cheap for the document sizes this editor handles.
 */
export type AnchorMarginaliaOptions = {
  /**
   * Supplies the open/recording meeting's start wall-clock (epoch ms) for the
   * derived-time fallback used by notes that predate the stored wall-clock.
   * Returns `null` when unknown (the gutter then shows elapsed for those notes).
   */
  startedAtMs: () => number | null;
};

export const AnchorMarginalia = Extension.create<AnchorMarginaliaOptions>({
  name: "anchorMarginalia",

  addOptions() {
    return { startedAtMs: () => null };
  },

  addProseMirrorPlugins() {
    const startedAtMs = this.options.startedAtMs;
    return [
      new Plugin({
        key: anchorMarginaliaPluginKey,
        state: {
          init: (_config, state) => buildDecorations(state, startedAtMs()),
          apply: (tr, old, _oldState, newState) =>
            tr.docChanged ? buildDecorations(newState, startedAtMs()) : old,
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
