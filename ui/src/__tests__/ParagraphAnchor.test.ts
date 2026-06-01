/**
 * Behaviour tests for the paragraph-anchor extension (FR-19, correction A4).
 *
 * Binding rule under test: a paragraph's `data-anchor-ms` is stamped on the
 * FIRST keystroke into it, ONLY while recording, from the pause-**excluding**
 * capture clock (`recordingClockMs` fed by `AppEvent::RecordingClock`) — NOT
 * from `Date.now() - started_at_ms`.
 *
 * The clock is a simulated value deliberately offset from wall-clock so the
 * test can assert the anchor equals the simulated clock and is NOT a
 * `Date.now()`-derived value.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

import { Editor } from "@tiptap/core";
import { buildEditorExtensions } from "../editor/extensions";
import {
  typeText,
  placeCursorAtEnd,
  pressEnter,
  paragraphAnchors,
} from "./editor-test-utils";

/** A mutable simulated clock the test controls; mirrors the store field. */
type SimClock = { recording: boolean; clockMs: number | null };

function makeEditor(clock: SimClock): Editor {
  return new Editor({
    extensions: buildEditorExtensions({
      clockSource: () => ({ recording: clock.recording, clockMs: clock.clockMs }),
    }),
    content: "<p></p>",
  });
}

describe("paragraph-anchor extension", () => {
  // Pin a wall-clock value far from the simulated recording clock so we can
  // prove the anchor is not Date.now()-derived.
  const WALL_CLOCK = 1_900_000_000_000; // year 2030-ish epoch ms

  beforeEach(() => {
    vi.spyOn(Date, "now").mockReturnValue(WALL_CLOCK);
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stamps the anchor on first keystroke while recording, equal to clockMs", () => {
    const clock: SimClock = { recording: true, clockMs: 4200 };
    const editor = makeEditor(clock);
    placeCursorAtEnd(editor);

    typeText(editor, "h");

    expect(paragraphAnchors(editor)).toEqual([4200]);
    editor.destroy();
  });

  it("the stamped anchor equals recordingClockMs and is NOT Date.now()-derived", () => {
    const clock: SimClock = { recording: true, clockMs: 7350 };
    const editor = makeEditor(clock);
    placeCursorAtEnd(editor);

    typeText(editor, "note");

    const [anchor] = paragraphAnchors(editor);
    // Equals the pause-excluding capture clock…
    expect(anchor).toBe(7350);
    // …and is nowhere near the (mocked) wall clock — a Date.now()-based
    // implementation would have stamped ~1.9e12 instead of 7350.
    expect(anchor).not.toBe(WALL_CLOCK);
    expect(anchor).toBeLessThan(WALL_CLOCK);
    editor.destroy();
  });

  it("does not re-stamp an already-anchored paragraph on later edits", () => {
    const clock: SimClock = { recording: true, clockMs: 1000 };
    const editor = makeEditor(clock);
    placeCursorAtEnd(editor);

    typeText(editor, "first");
    expect(paragraphAnchors(editor)).toEqual([1000]);

    // Clock advances; editing the SAME paragraph must keep its original anchor.
    clock.clockMs = 9000;
    placeCursorAtEnd(editor);
    typeText(editor, " more");

    expect(paragraphAnchors(editor)).toEqual([1000]);
    editor.destroy();
  });

  it("stamps a new paragraph with the current clock, leaving the old one intact", () => {
    const clock: SimClock = { recording: true, clockMs: 1000 };
    const editor = makeEditor(clock);
    placeCursorAtEnd(editor);

    typeText(editor, "para one");
    expect(paragraphAnchors(editor)).toEqual([1000]);

    // New paragraph at a later clock value.
    clock.clockMs = 5000;
    pressEnter(editor);
    typeText(editor, "para two");

    expect(paragraphAnchors(editor)).toEqual([1000, 5000]);
    editor.destroy();
  });

  it("does not stamp when idle (not recording)", () => {
    const clock: SimClock = { recording: false, clockMs: null };
    const editor = makeEditor(clock);
    placeCursorAtEnd(editor);

    typeText(editor, "typed while idle");

    expect(paragraphAnchors(editor)).toEqual([null]);
    editor.destroy();
  });

  it("does not stamp before the first clock value arrives (clockMs null)", () => {
    // Recording has started but no RecordingClock event has been seen yet.
    const clock: SimClock = { recording: true, clockMs: null };
    const editor = makeEditor(clock);
    placeCursorAtEnd(editor);

    typeText(editor, "early");
    expect(paragraphAnchors(editor)).toEqual([null]);

    // Once the clock arrives, the next keystroke into the (still unanchored)
    // paragraph stamps it.
    clock.clockMs = 250;
    placeCursorAtEnd(editor);
    typeText(editor, " late");
    expect(paragraphAnchors(editor)).toEqual([250]);
    editor.destroy();
  });

  it("a paragraph typed while idle stays unanchored even after recording starts editing a different paragraph", () => {
    const clock: SimClock = { recording: false, clockMs: null };
    const editor = makeEditor(clock);
    placeCursorAtEnd(editor);

    typeText(editor, "idle para");
    expect(paragraphAnchors(editor)).toEqual([null]);

    // Recording starts; a NEW paragraph gets anchored, the idle one does not.
    clock.recording = true;
    clock.clockMs = 3000;
    pressEnter(editor);
    typeText(editor, "recording para");

    expect(paragraphAnchors(editor)).toEqual([null, 3000]);
    editor.destroy();
  });
});
