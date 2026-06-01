/**
 * Notes paragraph hover bridge (FR-22, read side).
 *
 * A presentation-only ProseMirror plugin that reports the `data-anchor-ms` of
 * the paragraph currently under the pointer AND the anchor of the next
 * anchored paragraph after it in document order. It is the read half of the
 * cross-reference: hovering a notes paragraph publishes the paragraph's anchor
 * span (a recording-clock, pause-EXCLUDING value range on the same timeline as
 * `Segment::start_ms`) to a callback; the cross-ref store maps it to the
 * RANGE of transcript segments whose `start_ms ∈ [anchor, nextAnchor)` and the
 * transcript pane highlights all rows in that range (FR-22).
 *
 * Critically, this plugin **never mutates the document and dispatches no
 * transactions** — it only reads `mouseover` / `mouseout` DOM events and calls
 * back with anchor values (or `null`). It therefore cannot interfere with
 * `ParagraphAnchor`'s first-keystroke stamping (the A4 binding rule) the same
 * way `AnchorMarginalia` cannot: both are pure read/decoration layers.
 */
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { ANCHOR_ATTR } from "./paragraph-anchor";

/**
 * Reports the hovered paragraph's anchor (ms) and the next anchored paragraph's
 * anchor (ms), or `(null, null)` when hover leaves. `nextAnchorMs` is `null`
 * when the hovered paragraph is the last anchored one (the highlight then runs
 * through the end of the recording).
 */
export type HoverAnchorReporter = (
  anchorMs: number | null,
  nextAnchorMs: number | null,
) => void;

export type HoverBridgeOptions = {
  /** Receives the anchor ms of the hovered paragraph + the next anchor (or `null`). */
  onHoverAnchor: HoverAnchorReporter;
};

const hoverBridgePluginKey = new PluginKey("notesHoverBridge");

/** Parse an element's `data-anchor-ms`, or `null` when absent / malformed. */
function anchorOf(el: Element): number | null {
  const raw = el.getAttribute(ANCHOR_ATTR);
  if (raw === null) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

/**
 * Walk up from a DOM node to the nearest element carrying `data-anchor-ms`,
 * bounded by the editor root. Returns the anchored element or `null`.
 */
function anchoredElementFromTarget(
  target: EventTarget | null,
  root: HTMLElement,
): HTMLElement | null {
  let el = target instanceof HTMLElement ? target : null;
  while (el && el !== root) {
    if (el.getAttribute(ANCHOR_ATTR) !== null) return el;
    el = el.parentElement;
  }
  return null;
}

/**
 * The anchor of the first anchored element that appears AFTER `from` in
 * `root`'s document order, or `null` when `from` is the last anchored element.
 *
 * Used to bound the FR-22 highlight range: the hovered paragraph anchors the
 * transcript span `[anchor(from), anchor(next))`; a `null` next means the span
 * runs through the end of the recording.
 */
function nextAnchorAfter(root: HTMLElement, from: HTMLElement): number | null {
  const anchored = Array.from(
    root.querySelectorAll<HTMLElement>(`[${ANCHOR_ATTR}]`),
  );
  const fromIndex = anchored.indexOf(from);
  if (fromIndex === -1) return null;
  for (let i = fromIndex + 1; i < anchored.length; i += 1) {
    const ms = anchorOf(anchored[i]);
    if (ms !== null) return ms;
  }
  return null;
}

export const NotesHoverBridge = Extension.create<HoverBridgeOptions>({
  name: "notesHoverBridge",

  addOptions() {
    return {
      // Default no-op reporter so the extension is inert unless the Editor
      // wires a real reporter (tests wire a spy).
      onHoverAnchor: () => {},
    };
  },

  addProseMirrorPlugins() {
    const onHoverAnchor = this.options.onHoverAnchor;
    // Track the last reported anchor so we only fire on change (avoids a flood
    // of identical reports as the pointer moves within one paragraph).
    let lastReported: number | null = null;

    const report = (anchorMs: number | null, nextAnchorMs: number | null) => {
      if (anchorMs === lastReported) return;
      lastReported = anchorMs;
      onHoverAnchor(anchorMs, nextAnchorMs);
    };

    return [
      new Plugin({
        key: hoverBridgePluginKey,
        props: {
          handleDOMEvents: {
            mouseover: (view, event) => {
              const root = view.dom as HTMLElement;
              const anchored = anchoredElementFromTarget(event.target, root);
              if (anchored === null) {
                report(null, null);
              } else {
                report(anchorOf(anchored), nextAnchorAfter(root, anchored));
              }
              return false;
            },
            mouseout: (view, event) => {
              // Only clear when leaving the editor root entirely; moving between
              // child nodes within an anchored paragraph keeps the report stable
              // because `report` de-dupes.
              const related = (event as MouseEvent).relatedTarget;
              const root = view.dom as HTMLElement;
              if (!(related instanceof Node) || !root.contains(related)) {
                report(null, null);
              }
              return false;
            },
          },
        },
      }),
    ];
  },
});
