/**
 * Tests for the `recording_clock` event handling in the recording store.
 *
 * Verifies:
 * - a `recording_clock` event updates `recordingClockMs` to `clock_ms`.
 * - transitioning to `idle` (or `stopping`) clears `recordingClockMs` to null.
 * - transitioning to `recording` does not pre-populate the clock (it stays at
 *   its prior value until the next `recording_clock` event).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  Channel: vi.fn(),
}));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({
  WebviewWindow: vi.fn(),
}));

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

import { useRecordingStore } from "../state/recording";
import type { AppEvent } from "../ipc/app-event";

describe("recording store — recording_clock", () => {
  beforeEach(() => {
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

  it("recording_clock event sets recordingClockMs to clock_ms", () => {
    const event: AppEvent = {
      kind: "recording_clock",
      meeting_id: "m1",
      clock_ms: 12_345,
    };

    useRecordingStore.getState().handleEvent(event);

    expect(useRecordingStore.getState().recordingClockMs).toBe(12_345);
  });

  it("subsequent recording_clock events advance recordingClockMs", () => {
    const send = (clock_ms: number) =>
      useRecordingStore.getState().handleEvent({
        kind: "recording_clock",
        meeting_id: "m1",
        clock_ms,
      });

    send(1000);
    expect(useRecordingStore.getState().recordingClockMs).toBe(1000);
    send(2000);
    expect(useRecordingStore.getState().recordingClockMs).toBe(2000);
    send(3500);
    expect(useRecordingStore.getState().recordingClockMs).toBe(3500);
  });

  it("transition to idle clears recordingClockMs", () => {
    useRecordingStore.setState({ recordingClockMs: 9000 });

    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "idle" },
    });

    expect(useRecordingStore.getState().recordingClockMs).toBeNull();
  });

  it("transition to stopping clears recordingClockMs", () => {
    useRecordingStore.setState({ recordingClockMs: 9000 });

    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "stopping", meeting_id: "m1" },
    });

    expect(useRecordingStore.getState().recordingClockMs).toBeNull();
  });

  it("transition to recording does not synthesise a clock value", () => {
    // recording_clock is the sole source; entering `recording` must not
    // pre-populate the clock from any wall-clock source.
    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "recording", meeting_id: "m1", started_at_ms: Date.now() },
    });

    expect(useRecordingStore.getState().recordingClockMs).toBeNull();
  });
});
