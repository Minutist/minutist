/**
 * Cross-reference store (FR-22 / FR-23).
 *
 * Holds the transient highlight/scroll links between the notes editor and the
 * transcript pane, on the pause-EXCLUDING timeline:
 *
 *  - **FR-22 (notes → transcript).** When a notes paragraph is hovered, the
 *    transcript RANGE it anchors is highlighted: every segment whose
 *    `start_ms ∈ [anchor(P), anchor(nextAnchoredParagraph))` — for the last
 *    anchored paragraph, through the end of the recording. The resulting set of
 *    segment indices is published as `highlightedRange` and the transcript pane
 *    highlights all rows in that half-open range. (the highlight
 *    is a paragraph's whole span of transcript, not just the single nearest
 *    segment.)
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
 * A half-open range of transcript segment indices `[startIndex, endIndex)` to
 * highlight (FR-22). `endIndex` is exclusive. Both indices are into the
 * transcript array the highlight was computed against.
 */
export type SegmentRange = {
  startIndex: number;
  endIndex: number;
};

/**
 * Index of the segment in `segments` whose `start_ms` is nearest `anchorMs`, or
 * `null` when `segments` is empty.
 *
 * "Nearest" is by absolute distance on the start_ms axis. Ties (equidistant
 * between two segments) resolve to the earlier segment. Retained for the FR-23
 * notes-side mapping symmetry and as the building block other callers may reuse;
 * it operates purely on `start_ms`, the pause-excluding recording clock.
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

/**
 * FR-22 range mapping. Given a hovered paragraph's anchor (`anchorMs`) and the
 * NEXT anchored paragraph's anchor (`nextAnchorMs`, `null` when the hovered
 * paragraph is the last anchored one), return the half-open range of segment
 * indices whose `start_ms` falls in `[anchorMs, nextAnchorMs)` — or, when
 * `nextAnchorMs` is `null`, `[anchorMs, +∞)` (through the end of the recording).
 *
 * Returns `null` when no segment falls in the range (e.g. an anchor past every
 * segment, or an empty transcript), so the transcript pane clears its highlight.
 *
 * Mapping is purely on `Segment.start_ms`, the pause-excluding recording clock —
 * never wall-clock.
 */
export function segmentRangeForAnchors(
  segments: Segment[],
  anchorMs: number,
  nextAnchorMs: number | null,
): SegmentRange | null {
  if (segments.length === 0) return null;

  // First segment at or after the paragraph's anchor.
  let startIndex = -1;
  for (let i = 0; i < segments.length; i += 1) {
    if (segments[i].start_ms >= anchorMs) {
      startIndex = i;
      break;
    }
  }
  if (startIndex === -1) return null;

  // First segment at or after the next paragraph's anchor (exclusive upper
  // bound). With no next anchor, the range runs through the end of the
  // recording.
  let endIndex = segments.length;
  if (nextAnchorMs !== null) {
    for (let i = startIndex; i < segments.length; i += 1) {
      if (segments[i].start_ms >= nextAnchorMs) {
        endIndex = i;
        break;
      }
    }
  }

  if (endIndex <= startIndex) return null;
  return { startIndex, endIndex };
}

export type CrossRefStore = {
  /**
   * The half-open range of transcript segment indices to highlight (FR-22), or
   * `null` when no notes paragraph is hovered / the hovered paragraph has no
   * anchor / no segment falls in the paragraph's span.
   */
  highlightedRange: SegmentRange | null;
  /**
   * A pending request for the editor to scroll to the paragraph nearest this
   * anchor ms (FR-23). Carries a monotonically increasing `nonce` so repeated
   * clicks on the same segment still re-trigger the scroll effect. `null` until
   * the first transcript-segment click.
   */
  scrollRequest: { anchorMs: number; nonce: number } | null;

  /**
   * FR-22: a notes paragraph was hovered. `anchorMs` is the hovered paragraph's
   * anchor; `nextAnchorMs` is the NEXT anchored paragraph's anchor (`null` when
   * the hovered paragraph is the last anchored one — the range then runs through
   * end-of-recording). Publishes the half-open range of matching segment
   * indices. A `null` `anchorMs` (hover left, or an unanchored paragraph) clears
   * the highlight.
   */
  hoverNotesAnchor: (
    anchorMs: number | null,
    nextAnchorMs: number | null,
    segments: Segment[],
  ) => void;
  /**
   * FR-23: a transcript segment was clicked. Publish a scroll request carrying
   * the segment's `start_ms` so the editor scrolls to the nearest paragraph.
   */
  clickTranscriptSegment: (segment: Segment) => void;
};

export const useCrossRefStore = create<CrossRefStore>((set, get) => ({
  highlightedRange: null,
  scrollRequest: null,

  hoverNotesAnchor: (anchorMs, nextAnchorMs, segments) => {
    if (anchorMs === null) {
      set({ highlightedRange: null });
      return;
    }
    set({
      highlightedRange: segmentRangeForAnchors(segments, anchorMs, nextAnchorMs),
    });
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
