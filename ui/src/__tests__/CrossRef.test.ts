/**
 * Behaviour tests for the cross-reference store + helpers (FR-22 / FR-23).
 *
 * The binding invariant under test: the mapping between notes paragraphs and
 * transcript segments is on `Segment.start_ms` (the recording-clock,
 * pause-EXCLUDING timeline, same origin as the notes `data-anchor-ms`), NEVER
 * wall-clock `Date.now()`. To prove the mapping is not wall-clock derived, the
 * test pins `Date.now()` to a value far from any segment start.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

import { Editor } from "@tiptap/core";
import {
  nearestSegmentIndex,
  segmentRangeForAnchors,
  useCrossRefStore,
} from "../state/cross-ref";
import { buildEditorExtensions } from "../editor/extensions";
import { nearestAnchoredElement } from "../editor/scroll-to-anchor";
import { ANCHOR_ATTR } from "../editor/paragraph-anchor";
import type { Segment } from "../ipc/bindings";

const SEGMENTS: Segment[] = [
  { start_ms: 4_200, end_ms: 9_800, text: "one", words: [], shared_speakers: [] },
  { start_ms: 12_400, end_ms: 21_300, text: "two", words: [], shared_speakers: [] },
  { start_ms: 24_100, end_ms: 33_900, text: "three", words: [], shared_speakers: [] },
  { start_ms: 51_000, end_ms: 61_700, text: "four", words: [], shared_speakers: [] },
];

// A wall-clock value far from any segment start, so a Date.now()-based mapping
// would resolve to the wrong (last) segment for every input.
const WALL_CLOCK = 1_900_000_000_000;

describe("nearestSegmentIndex (FR-22 mapping is on Segment.start_ms)", () => {
  beforeEach(() => {
    vi.spyOn(Date, "now").mockReturnValue(WALL_CLOCK);
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("maps an anchor to the segment with the nearest start_ms", () => {
    // 13_000 is closest to segment[1] (start 12_400), not segment[2] (24_100).
    expect(nearestSegmentIndex(SEGMENTS, 13_000)).toBe(1);
    // 23_000 is closest to segment[2] (24_100).
    expect(nearestSegmentIndex(SEGMENTS, 23_000)).toBe(2);
    // An anchor before the first segment resolves to segment[0].
    expect(nearestSegmentIndex(SEGMENTS, 0)).toBe(0);
    // An anchor after the last resolves to the last.
    expect(nearestSegmentIndex(SEGMENTS, 90_000)).toBe(3);
  });

  it("uses start_ms distance, NOT wall-clock (Date.now is far away)", () => {
    // If the mapping were Date.now()-derived, every anchor would land on the
    // last segment (WALL_CLOCK is huge). It resolves on start_ms instead.
    expect(nearestSegmentIndex(SEGMENTS, 4_300)).toBe(0);
    expect(Date.now()).toBe(WALL_CLOCK); // confirm the spy is in effect
  });

  it("resolves ties to the earlier segment", () => {
    // Exactly between segment[0] (4_200) and segment[1] (12_400) → 8_300.
    expect(nearestSegmentIndex(SEGMENTS, 8_300)).toBe(0);
  });

  it("returns null for an empty transcript", () => {
    expect(nearestSegmentIndex([], 1_000)).toBeNull();
  });
});

describe("segmentRangeForAnchors (FR-22 RANGE mapping is on Segment.start_ms)", () => {
  it("highlights the half-open range [anchor, nextAnchor) of segments", () => {
    // Anchor at segment[1].start_ms (12_400), next anchor at segment[3] (51_000):
    // segments whose start_ms ∈ [12_400, 51_000) are indices 1 and 2.
    expect(segmentRangeForAnchors(SEGMENTS, 12_400, 51_000)).toEqual({
      startIndex: 1,
      endIndex: 3,
    });
  });

  it("runs through the end of the recording when there is no next anchor", () => {
    // No next anchored paragraph → [12_400, +∞): indices 1, 2, 3.
    expect(segmentRangeForAnchors(SEGMENTS, 12_400, null)).toEqual({
      startIndex: 1,
      endIndex: 4,
    });
  });

  it("starts at the first segment at or after the anchor", () => {
    // Anchor before the first segment, next at 24_100 → [0, 24_100): indices 0, 1.
    expect(segmentRangeForAnchors(SEGMENTS, 0, 24_100)).toEqual({
      startIndex: 0,
      endIndex: 2,
    });
  });

  it("returns null when no segment falls in the paragraph's span", () => {
    // Anchor past every segment.
    expect(segmentRangeForAnchors(SEGMENTS, 90_000, null)).toBeNull();
    // Empty transcript.
    expect(segmentRangeForAnchors([], 1_000, null)).toBeNull();
  });

  it("does NOT collapse to a single nearest segment (range semantics)", () => {
    // anchors [1000, 5000], segments at [1200, 3000, 6000]: hovering the
    // paragraph anchored at 1000 highlights the first TWO segments
    // (start_ms ∈ [1000, 5000) → 1200 and 3000), not just the nearest one.
    const segments: Segment[] = [
      { start_ms: 1_200, end_ms: 2_000, text: "a", words: [], shared_speakers: [] },
      { start_ms: 3_000, end_ms: 4_000, text: "b", words: [], shared_speakers: [] },
      { start_ms: 6_000, end_ms: 7_000, text: "c", words: [], shared_speakers: [] },
    ];
    expect(segmentRangeForAnchors(segments, 1_000, 5_000)).toEqual({
      startIndex: 0,
      endIndex: 2,
    });
    // Guard against a regression to single-segment semantics: the nearest
    // segment to anchor 1000 is index 0 alone, but the range covers two.
    expect(nearestSegmentIndex(segments, 1_000)).toBe(0);
  });
});

describe("useCrossRefStore (FR-22 hover, FR-23 click)", () => {
  beforeEach(() => {
    useCrossRefStore.setState({
      highlightedRange: null,
      scrollRequest: null,
    });
  });

  it("hovering a notes anchor highlights the range it spans (FR-22)", () => {
    // Anchor at 12_400 with next anchor 51_000 → indices [1, 3).
    useCrossRefStore.getState().hoverNotesAnchor(12_400, 51_000, SEGMENTS);
    expect(useCrossRefStore.getState().highlightedRange).toEqual({
      startIndex: 1,
      endIndex: 3,
    });
  });

  it("a null hover (leave / unanchored paragraph) clears the highlight", () => {
    useCrossRefStore.setState({
      highlightedRange: { startIndex: 1, endIndex: 3 },
    });
    useCrossRefStore.getState().hoverNotesAnchor(null, null, SEGMENTS);
    expect(useCrossRefStore.getState().highlightedRange).toBeNull();
  });

  it("clicking a transcript segment publishes a scroll request on its start_ms (FR-23)", () => {
    useCrossRefStore.getState().clickTranscriptSegment(SEGMENTS[2]);
    const req = useCrossRefStore.getState().scrollRequest;
    expect(req?.anchorMs).toBe(24_100);
    expect(req?.nonce).toBe(1);
  });

  it("re-clicking the same segment bumps the nonce so the scroll re-triggers", () => {
    const store = useCrossRefStore.getState();
    store.clickTranscriptSegment(SEGMENTS[1]);
    store.clickTranscriptSegment(SEGMENTS[1]);
    const req = useCrossRefStore.getState().scrollRequest;
    expect(req?.anchorMs).toBe(12_400);
    expect(req?.nonce).toBe(2);
  });
});

describe("hover-bridge → cross-ref store (FR-22 end-to-end)", () => {
  let editor: Editor;

  beforeEach(() => {
    useCrossRefStore.setState({
      highlightedRange: null,
      scrollRequest: null,
    });
    // Wire the editor's hover reporter to the cross-ref store, exactly as the
    // production Editor does — onHoverAnchor maps the hovered anchor + the next
    // anchor against the segments to a RANGE.
    editor = new Editor({
      extensions: buildEditorExtensions({
        clockSource: () => ({ recording: false, clockMs: null }),
        onHoverAnchor: (anchorMs, nextAnchorMs) =>
          useCrossRefStore
            .getState()
            .hoverNotesAnchor(anchorMs, nextAnchorMs, SEGMENTS),
      }),
      // Three anchored paragraphs at 12_400 (segment[1]), 24_100 (segment[2]),
      // and 51_000 (segment[3]) plus a leading unanchored one.
      content:
        '<p>unanchored</p>' +
        '<p data-anchor-ms="12400">first anchored</p>' +
        '<p data-anchor-ms="24100">second anchored</p>' +
        '<p data-anchor-ms="51000">third anchored</p>',
    });
  });

  afterEach(() => {
    editor.destroy();
  });

  it("hovering an anchored paragraph highlights its span up to the next anchor", () => {
    const anchored = editor.view.dom.querySelectorAll(
      `[${ANCHOR_ATTR}]`,
    )[0] as HTMLElement;
    expect(anchored.getAttribute(ANCHOR_ATTR)).toBe("12400");

    anchored.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));

    // [12_400, 24_100) → segment index 1 only (24_100 belongs to the next para).
    expect(useCrossRefStore.getState().highlightedRange).toEqual({
      startIndex: 1,
      endIndex: 2,
    });
  });

  it("hovering the LAST anchored paragraph highlights through end-of-recording", () => {
    const anchored = editor.view.dom.querySelectorAll(
      `[${ANCHOR_ATTR}]`,
    )[2] as HTMLElement;
    expect(anchored.getAttribute(ANCHOR_ATTR)).toBe("51000");

    anchored.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));

    // [51_000, +∞) → segment index 3 through the last segment.
    expect(useCrossRefStore.getState().highlightedRange).toEqual({
      startIndex: 3,
      endIndex: 4,
    });
  });

  it("leaving the editor clears the highlight", () => {
    const anchored = editor.view.dom.querySelectorAll(
      `[${ANCHOR_ATTR}]`,
    )[0] as HTMLElement;
    anchored.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    expect(useCrossRefStore.getState().highlightedRange).not.toBeNull();

    // mouseout with relatedTarget outside the editor root clears the highlight.
    editor.view.dom.dispatchEvent(
      new MouseEvent("mouseout", { bubbles: true, relatedTarget: null }),
    );
    expect(useCrossRefStore.getState().highlightedRange).toBeNull();
  });
});

describe("nearestAnchoredElement (FR-23 notes-side mapping)", () => {
  function buildDoc(anchors: number[]): HTMLElement {
    const root = document.createElement("div");
    anchors.forEach((ms, i) => {
      const p = document.createElement("p");
      p.setAttribute(ANCHOR_ATTR, String(ms));
      p.textContent = `para ${i}`;
      root.appendChild(p);
    });
    // An unanchored paragraph must be ignored by the mapping.
    const plain = document.createElement("p");
    plain.textContent = "no anchor";
    root.appendChild(plain);
    return root;
  }

  it("finds the anchored paragraph nearest a target start_ms", () => {
    const root = buildDoc([4_200, 12_400, 24_100]);
    const el = nearestAnchoredElement(root, 23_000);
    expect(el?.getAttribute(ANCHOR_ATTR)).toBe("24100");
  });

  it("returns null when there are no anchored paragraphs", () => {
    const root = document.createElement("div");
    const p = document.createElement("p");
    p.textContent = "plain";
    root.appendChild(p);
    expect(nearestAnchoredElement(root, 1_000)).toBeNull();
  });
});
