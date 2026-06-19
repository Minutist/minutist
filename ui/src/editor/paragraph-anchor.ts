/**
 * Paragraph-anchor Tiptap/ProseMirror extension.
 *
 * Stamps a `data-anchor-ms` attribute onto a paragraph on the FIRST keystroke
 * that lands inside it, but ONLY while a recording is in progress. The stamped
 * value is the capture-sample, pause-**excluding** recording clock
 * (`recordingClockMs`, fed by `AppEvent::RecordingClock`) — NOT
 * `Date.now() - started_at_ms`. See `architecture/cross-cutting.md` —
 * "Notes paragraph-anchor clock" (binding rule, correction A4).
 *
 * Stamping is keyed on the *absence* of the attribute: once a paragraph carries
 * `data-anchor-ms`, subsequent edits to it never re-stamp. This realises the
 * "key the stamp on paragraph node identity so editing an already-anchored
 * paragraph does not re-stamp" requirement — an anchored paragraph keeps its
 * original anchor through every later edit, and a brand-new (unanchored)
 * paragraph gets the current clock on its first keystroke.
 *
 * When idle (`recording === false`) no anchor is stamped, and the attribute is
 * absent (`null`) on paragraphs created while idle.
 */
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import type { EditorState, Transaction } from "@tiptap/pm/state";
import { ReplaceStep, ReplaceAroundStep } from "@tiptap/pm/transform";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";

/** The data attribute carrying the recording-clock anchor, in milliseconds. */
export const ANCHOR_ATTR = "data-anchor-ms";

/**
 * Wall-clock time (epoch ms) at the moment a paragraph was anchored. Stamped
 * ALONGSIDE [`ANCHOR_ATTR`] for the gutter's time-of-day display (issue: users
 * read the gutter as a clock, not elapsed time). It is display-only: the
 * recording-offset `ANCHOR_ATTR` remains the cross-reference / summariser
 * timeline. Captured at stamp time so it is correct even across pauses (a naive
 * `start + offset` conversion would under-count by the pause duration).
 */
export const WALL_ATTR = "data-anchor-wall";

/**
 * Live recording-clock snapshot consulted on each keystroke.
 *
 * `recording` gates whether anchoring happens at all; `clockMs` is the value
 * stamped (the pause-excluding capture clock); `wallMs` is the wall-clock epoch
 * ms stamped into [`WALL_ATTR`] for the gutter display. Injected so the extension
 * stays decoupled from the Zustand store and can be driven by a simulated clock
 * in tests.
 */
export type AnchorClockSource = () => {
  recording: boolean;
  clockMs: number | null;
  /** Wall-clock epoch ms; omit/null to stamp only the offset (the gutter then
   *  derives the time-of-day from the meeting start). */
  wallMs?: number | null;
};

export type ParagraphAnchorOptions = {
  /** Supplies the current recording state + pause-excluding clock. */
  clockSource: AnchorClockSource;
};

const paragraphAnchorPluginKey = new PluginKey("paragraphAnchor");

/** True when `node` carries no anchor yet. */
function isUnanchored(node: ProseMirrorNode): boolean {
  const existing = node.attrs[ANCHOR_ATTR];
  return existing === null || existing === undefined;
}

/**
 * Classify the steps of a set of transactions.
 *
 * - `insertedText`: at least one step inserted inline text content (a real
 *   keystroke producing characters). This is what "first keystroke into a
 *   paragraph" means.
 * - `splitParagraph`: at least one step split a textblock (Enter), creating a
 *   new paragraph. Split-created paragraphs inherit the parent's attrs in
 *   ProseMirror, so their inherited anchor must be reset to `null` and not
 *   treated as "already anchored".
 */
function classifySteps(transactions: readonly Transaction[]): {
  insertedText: boolean;
  splitParagraph: boolean;
} {
  let insertedText = false;
  let splitParagraph = false;
  for (const tr of transactions) {
    for (const step of tr.steps) {
      if (step instanceof ReplaceStep) {
        const slice = step.slice;
        if (slice.openStart > 0 || slice.openEnd > 0) {
          // An open slice across a block boundary is a split.
          splitParagraph = true;
        } else if (slice.content.size > 0) {
          // A closed, non-empty slice inside a textblock is text insertion.
          slice.content.forEach((child) => {
            if (child.isText || child.isInline) insertedText = true;
          });
        }
      } else if (step instanceof ReplaceAroundStep) {
        // Wrapping / list-conversion input rules; not a plain keystroke.
        splitParagraph = splitParagraph || step.slice.openStart > 0;
      }
    }
  }
  return { insertedText, splitParagraph };
}

/**
 * Extend the built-in paragraph node with a nullable `data-anchor-ms`
 * attribute, and install a plugin that stamps it on first keystroke while
 * recording.
 */
export const ParagraphAnchor = Extension.create<ParagraphAnchorOptions>({
  name: "paragraphAnchor",

  addOptions() {
    return {
      // Default source reports "idle" so the extension is a no-op unless an
      // explicit clock source is supplied (the production Editor wires the
      // recording store; tests wire a simulated clock).
      clockSource: () => ({ recording: false, clockMs: null, wallMs: null }),
    };
  },

  // Register the `data-anchor-ms` attribute on the paragraph node. Rendered as
  // a real DOM attribute so it round-trips through HTML and ProseMirror JSON.
  addGlobalAttributes() {
    return [
      {
        types: ["paragraph"],
        attributes: {
          [ANCHOR_ATTR]: {
            default: null,
            parseHTML: (element: HTMLElement) => {
              const raw = element.getAttribute(ANCHOR_ATTR);
              return raw === null ? null : Number.parseInt(raw, 10);
            },
            renderHTML: (attributes: Record<string, unknown>) => {
              const value = attributes[ANCHOR_ATTR];
              if (value === null || value === undefined) return {};
              return { [ANCHOR_ATTR]: String(value) };
            },
          },
          [WALL_ATTR]: {
            default: null,
            parseHTML: (element: HTMLElement) => {
              const raw = element.getAttribute(WALL_ATTR);
              return raw === null ? null : Number.parseInt(raw, 10);
            },
            renderHTML: (attributes: Record<string, unknown>) => {
              const value = attributes[WALL_ATTR];
              if (value === null || value === undefined) return {};
              return { [WALL_ATTR]: String(value) };
            },
          },
        },
      },
    ];
  },

  addProseMirrorPlugins() {
    const clockSource = this.options.clockSource;

    return [
      new Plugin({
        key: paragraphAnchorPluginKey,
        /**
         * Inspect every applied transaction. If the document changed via a
         * text-input step while recording, stamp `data-anchor-ms` onto each
         * not-yet-anchored paragraph that the change touched.
         *
         * Using `appendTransaction` keeps the stamp atomic with the keystroke
         * (it shows up in the same editor update) and lets us read the
         * post-change document so newly created paragraphs are visible.
         */
        appendTransaction(
          transactions: readonly Transaction[],
          _oldState: EditorState,
          newState: EditorState,
        ): Transaction | null {
          const docChanged = transactions.some((tr) => tr.docChanged);
          if (!docChanged) return null;

          // The paragraph the cursor now sits in — the one a keystroke landed
          // in, or the new paragraph created by a split.
          const { $from } = newState.selection;
          let cursorParaPos: number | null = null;
          let cursorPara: ProseMirrorNode | null = null;
          for (let depth = $from.depth; depth >= 0; depth -= 1) {
            const node = $from.node(depth);
            if (node.type.name === "paragraph") {
              cursorParaPos = $from.before(depth);
              cursorPara = node;
              break;
            }
          }

          const { insertedText, splitParagraph } = classifySteps(transactions);
          const tr = newState.tr;
          let changed = false;

          // A split copies the parent paragraph's attrs onto the new (empty)
          // paragraph. Reset the cursor paragraph's inherited anchor so the
          // first real keystroke into it stamps a fresh value (and an idle
          // split never resurrects an anchor). Runs regardless of recording.
          if (
            splitParagraph &&
            cursorParaPos !== null &&
            cursorPara !== null &&
            !isUnanchored(cursorPara)
          ) {
            tr.setNodeAttribute(cursorParaPos, ANCHOR_ATTR, null);
            // Clear the paired wall-clock too so a split never resurrects a
            // stale time-of-day on the new paragraph.
            tr.setNodeAttribute(cursorParaPos, WALL_ATTR, null);
            changed = true;
          }

          // Stamp on the first text-bearing keystroke while recording.
          if (insertedText && !splitParagraph) {
            const { recording, clockMs, wallMs } = clockSource();
            if (
              recording &&
              clockMs !== null &&
              cursorParaPos !== null &&
              cursorPara !== null &&
              isUnanchored(cursorPara)
            ) {
              tr.setNodeAttribute(cursorParaPos, ANCHOR_ATTR, clockMs);
              // Pair the recording offset with the wall-clock at stamp time for
              // the gutter's time-of-day display (the offset stays authoritative
              // for cross-reference).
              if (wallMs !== null && wallMs !== undefined) {
                tr.setNodeAttribute(cursorParaPos, WALL_ATTR, wallMs);
              }
              changed = true;
            }
          }

          if (!changed) return null;
          // Mark the appended transaction and keep it out of undo history.
          tr.setMeta(paragraphAnchorPluginKey, true);
          tr.setMeta("addToHistory", false);
          return tr;
        },
      }),
    ];
  },
});
