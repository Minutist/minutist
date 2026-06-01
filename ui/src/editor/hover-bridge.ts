/**
 * Notes paragraph hover bridge (FR-22, read side).
 *
 * A presentation-only ProseMirror plugin that reports the `data-anchor-ms` of
 * the paragraph currently under the pointer. It is the read half of the
 * cross-reference: hovering a notes paragraph publishes the paragraph's anchor
 * (a recording-clock, pause-EXCLUDING value on the same timeline as
 * `Segment::start_ms`) to a callback; the cross-ref store maps it to the
 * nearest transcript segment and the transcript pane highlights that segment.
 *
 * Critically, this plugin **never mutates the document and dispatches no
 * transactions** — it only reads `mouseover` / `mouseout` DOM events and calls
 * back with an anchor value (or `null`). It therefore cannot interfere with
 * `ParagraphAnchor`'s first-keystroke stamping (the A4 binding rule) the same
 * way `AnchorMarginalia` cannot: both are pure read/decoration layers.
 */
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { ANCHOR_ATTR } from "./paragraph-anchor";

/** Reports the hovered paragraph's anchor (ms) or `null` when hover leaves. */
export type HoverAnchorReporter = (anchorMs: number | null) => void;

export type HoverBridgeOptions = {
  /** Receives the anchor ms of the hovered paragraph (or `null`). */
  onHoverAnchor: HoverAnchorReporter;
};

const hoverBridgePluginKey = new PluginKey("notesHoverBridge");

/**
 * Walk up from a DOM node to the nearest element carrying `data-anchor-ms`,
 * bounded by the editor root. Returns the parsed anchor or `null`.
 */
function anchorFromTarget(
  target: EventTarget | null,
  root: HTMLElement,
): number | null {
  let el = target instanceof HTMLElement ? target : null;
  while (el && el !== root) {
    const raw = el.getAttribute(ANCHOR_ATTR);
    if (raw !== null) {
      const parsed = Number.parseInt(raw, 10);
      return Number.isNaN(parsed) ? null : parsed;
    }
    el = el.parentElement;
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

    const report = (anchorMs: number | null) => {
      if (anchorMs === lastReported) return;
      lastReported = anchorMs;
      onHoverAnchor(anchorMs);
    };

    return [
      new Plugin({
        key: hoverBridgePluginKey,
        props: {
          handleDOMEvents: {
            mouseover: (view, event) => {
              report(anchorFromTarget(event.target, view.dom as HTMLElement));
              return false;
            },
            mouseout: (view, event) => {
              // Only clear when leaving the editor root entirely; moving between
              // child nodes within an anchored paragraph keeps the report stable
              // because `report` de-dupes.
              const related = (event as MouseEvent).relatedTarget;
              const root = view.dom as HTMLElement;
              if (!(related instanceof Node) || !root.contains(related)) {
                report(null);
              }
              return false;
            },
          },
        },
      }),
    ];
  },
});
