/**
 * Cross-reference store (FR-22 / FR-23).
 *
 * Holds the transient highlight/scroll links between the notes editor and the
 * transcript pane, at SEGMENT granularity on the pause-EXCLUDING timeline:
 *
 *  - **FR-22 (notes → transcript).** When a notes paragraph is hovered, its
 *    `data-anchor-ms` is mapped to the NEAREST `Segment.start_ms`; the resulting
 *    segment index is published as `highlightedSegmentIndex` and the transcript
 *    pane highlights that row's range.
 *  - **FR-23 (transcript → notes).** When a transcript segment is clicked, the
 *    segment's `start_ms` is published as `scrollToAnchorMs`; the editor scrolls
 *    to the paragraph whose anchor is nearest that value.
 *
 * Both mappings are on the recording-clock, pause-EXCLUDING timeline (the same
 * origin as `Segment::start_ms` and the notes `data-anchor-ms`), NEVER wall
 * clock (`Date.now()`). See `architecture/cross-cutting.md` — "Notes
 * paragraph-anchor clock".
 */
import { create } from "zustand";
import type { Segment } from "../ipc/bindings";

/**
 * Index of the segment in `segments` whose `start_ms` is nearest `anchorMs`, or
 * `null` when `segments` is empty.
 *
 * "Nearest" is by absolute distance on the start_ms axis. Ties (equidistant
 * between two segments) resolve to the earlier segment. This is the FR-22
 * paragraph-anchor → segment mapping; it operates purely on `start_ms`, the
 * pause-excluding recording clock.
 */
export function nearestSegmentIndex(
  segments: Segment[],
  anchorMs: number,
): number | null {
  if (segments.length === 0) return null;
  let bestIndex = 0;
  let bestDistance = Math.abs(segments[0].start_ms - anchorMs);
  for (let i = 1; i < segments.length; i += 1) {
    const distance = Math.abs(segments[i].start_ms - anchorMs);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

export type CrossRefStore = {
  /**
   * The transcript segment index to highlight (FR-22), or `null` when no notes
   * paragraph is hovered / the hovered paragraph has no anchor.
   */
  highlightedSegmentIndex: number | null;
  /**
   * A pending request for the editor to scroll to the paragraph nearest this
   * anchor ms (FR-23). Carries a monotonically increasing `nonce` so repeated
   * clicks on the same segment still re-trigger the scroll effect. `null` until
   * the first transcript-segment click.
   */
  scrollRequest: { anchorMs: number; nonce: number } | null;

  /**
   * FR-22: a notes paragraph was hovered. Map its anchor to the nearest segment
   * and publish the index. A `null` anchor (hover left, or an unanchored
   * paragraph) clears the highlight.
   */
  hoverNotesAnchor: (anchorMs: number | null, segments: Segment[]) => void;
  /**
   * FR-23: a transcript segment was clicked. Publish a scroll request carrying
   * the segment's `start_ms` so the editor scrolls to the nearest paragraph.
   */
  clickTranscriptSegment: (segment: Segment) => void;
};

export const useCrossRefStore = create<CrossRefStore>((set, get) => ({
  highlightedSegmentIndex: null,
  scrollRequest: null,

  hoverNotesAnchor: (anchorMs, segments) => {
    if (anchorMs === null) {
      set({ highlightedSegmentIndex: null });
      return;
    }
    set({ highlightedSegmentIndex: nearestSegmentIndex(segments, anchorMs) });
  },

  clickTranscriptSegment: (segment) => {
    const prev = get().scrollRequest;
    set({
      scrollRequest: {
        anchorMs: segment.start_ms,
        nonce: (prev?.nonce ?? 0) + 1,
      },
    });
  },
}));
