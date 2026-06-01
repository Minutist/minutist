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
  useCrossRefStore,
} from "../state/cross-ref";
import { buildEditorExtensions } from "../editor/extensions";
import { nearestAnchoredElement } from "../editor/scroll-to-anchor";
import { ANCHOR_ATTR } from "../editor/paragraph-anchor";
import type { Segment } from "../ipc/bindings";

const SEGMENTS: Segment[] = [
  { start_ms: 4_200, end_ms: 9_800, text: "one", words: [] },
  { start_ms: 12_400, end_ms: 21_300, text: "two", words: [] },
  { start_ms: 24_100, end_ms: 33_900, text: "three", words: [] },
  { start_ms: 51_000, end_ms: 61_700, text: "four", words: [] },
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

describe("useCrossRefStore (FR-22 hover, FR-23 click)", () => {
  beforeEach(() => {
    useCrossRefStore.setState({
      highlightedSegmentIndex: null,
      scrollRequest: null,
    });
  });

  it("hovering a notes anchor highlights the nearest segment (FR-22)", () => {
    useCrossRefStore.getState().hoverNotesAnchor(24_100, SEGMENTS);
    expect(useCrossRefStore.getState().highlightedSegmentIndex).toBe(2);
  });

  it("a null hover (leave / unanchored paragraph) clears the highlight", () => {
    useCrossRefStore.setState({ highlightedSegmentIndex: 2 });
    useCrossRefStore.getState().hoverNotesAnchor(null, SEGMENTS);
    expect(useCrossRefStore.getState().highlightedSegmentIndex).toBeNull();
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
      highlightedSegmentIndex: null,
      scrollRequest: null,
    });
    // Wire the editor's hover reporter to the cross-ref store, exactly as the
    // production Editor does — onHoverAnchor maps against the live segments.
    editor = new Editor({
      extensions: buildEditorExtensions({
        clockSource: () => ({ recording: false, clockMs: null }),
        onHoverAnchor: (anchorMs) =>
          useCrossRefStore.getState().hoverNotesAnchor(anchorMs, SEGMENTS),
      }),
      // Two paragraphs, the second anchored at 24_100 (segment[2]).
      content:
        '<p>unanchored</p><p data-anchor-ms="24100">anchored para</p>',
    });
  });

  afterEach(() => {
    editor.destroy();
  });

  it("hovering an anchored paragraph highlights the nearest segment (on start_ms)", () => {
    const anchored = editor.view.dom.querySelector(
      `[${ANCHOR_ATTR}]`,
    ) as HTMLElement;
    expect(anchored).not.toBeNull();

    anchored.dispatchEvent(
      new MouseEvent("mouseover", { bubbles: true }),
    );

    // 24_100 maps to segment index 2 — proving the bridge reports the anchor and
    // the store maps it on Segment.start_ms.
    expect(useCrossRefStore.getState().highlightedSegmentIndex).toBe(2);
  });

  it("leaving the editor clears the highlight", () => {
    const anchored = editor.view.dom.querySelector(
      `[${ANCHOR_ATTR}]`,
    ) as HTMLElement;
    anchored.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    expect(useCrossRefStore.getState().highlightedSegmentIndex).toBe(2);

    // mouseout with relatedTarget outside the editor root clears the highlight.
    editor.view.dom.dispatchEvent(
      new MouseEvent("mouseout", { bubbles: true, relatedTarget: null }),
    );
    expect(useCrossRefStore.getState().highlightedSegmentIndex).toBeNull();
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
