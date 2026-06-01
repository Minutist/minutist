/**
 * Proves the anchor clock source the production Editor wires (the recording
 * store) yields `recordingClockMs`, not a wall-clock value.
 *
 * This guards the binding correction A4 end-to-end at the store boundary: the
 * value an editor would stamp equals `useRecordingStore.getState().recordingClockMs`
 * after a `recording_clock` event, and changes only when that event fires —
 * never tracking `Date.now()`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));
vi.mock("../ipc/bindings", () => ({
  commands: {
    listDevices: vi.fn(),
    startRecording: vi.fn(),
    pauseRecording: vi.fn(),
    resumeRecording: vi.fn(),
    stopRecording: vi.fn(),
    getRecordingState: vi.fn(),
    getSettings: vi.fn(),
    updateSettings: vi.fn(),
    listModels: vi.fn(),
    ensureModel: vi.fn(),
  },
  events: {},
}));

import { Editor } from "@tiptap/core";
import { buildEditorExtensions } from "../editor/extensions";
import { useRecordingStore } from "../state/recording";
import { typeText, placeCursorAtEnd, paragraphAnchors } from "./editor-test-utils";

/** The same clock source the production Editor component uses. */
function storeClockSource() {
  const s = useRecordingStore.getState();
  return {
    recording: s.state.kind === "recording",
    clockMs: s.recordingClockMs,
  };
}

describe("anchor clock sourced from the recording store", () => {
  const WALL_CLOCK = 1_950_000_000_000;

  beforeEach(() => {
    vi.spyOn(Date, "now").mockReturnValue(WALL_CLOCK);
    useRecordingStore.setState({
      state: { kind: "idle" },
      devices: [],
      selectedDeviceId: null,
      settings: null,
      meter: { peak: 0, rms: 0 },
      lastError: null,
      transcript: [],
      recordingClockMs: null,
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stamps the store's recordingClockMs, not Date.now()", () => {
    // Enter recording, then receive a recording_clock event at 8800 ms.
    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "recording", meeting_id: "m1", started_at_ms: WALL_CLOCK },
    });
    useRecordingStore.getState().handleEvent({
      kind: "recording_clock",
      meeting_id: "m1",
      clock_ms: 8800,
    });

    const editor = new Editor({
      extensions: buildEditorExtensions({ clockSource: storeClockSource }),
      content: "<p></p>",
    });
    placeCursorAtEnd(editor);
    typeText(editor, "x");

    expect(paragraphAnchors(editor)).toEqual([8800]);
    expect(paragraphAnchors(editor)[0]).not.toBe(WALL_CLOCK);
    editor.destroy();
  });

  it("does not stamp once recording stops (clock cleared to null)", () => {
    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "recording", meeting_id: "m1", started_at_ms: WALL_CLOCK },
    });
    useRecordingStore.getState().handleEvent({
      kind: "recording_clock",
      meeting_id: "m1",
      clock_ms: 1000,
    });
    // Stop recording — store clears recordingClockMs and leaves `recording`.
    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "idle" },
    });

    const editor = new Editor({
      extensions: buildEditorExtensions({ clockSource: storeClockSource }),
      content: "<p></p>",
    });
    placeCursorAtEnd(editor);
    typeText(editor, "typed after stop");

    expect(paragraphAnchors(editor)).toEqual([null]);
    editor.destroy();
  });
});
