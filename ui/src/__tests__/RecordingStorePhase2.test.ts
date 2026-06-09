/**
 * Phase 2 unit tests for the recording Zustand store.
 *
 * Verifies:
 * - `start` is a no-op that sets `lastError` when `isAsrModelReady` is false.
 * - `transcript_segment` events append segments.
 * - `state_changed → recording` clears the transcript.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Tauri API mocks
// ---------------------------------------------------------------------------
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

vi.mock("../ipc/bindings", () => {
  return {
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
  };
});

import { commands } from "../ipc/bindings";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import type { AppEvent, Segment } from "../ipc/bindings";

function makeSegment(start_ms: number, text: string): Segment {
  return { start_ms, end_ms: start_ms + 500, text, words: [] };
}

describe("recording store — Phase 2 extensions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useRecordingStore.setState({
      state: { kind: "idle" },
      devices: [],
      selectedDeviceId: null,
      settings: null,
      meter: { peak: 0, rms: 0 },
      lastError: null,
      preparing: false,
      transcript: [],
    });
    useModelsStore.setState({
      models: [],
      isAsrModelReady: false,
      downloadInProgress: {},
    });
  });

  it("start is a no-op and sets lastError when isAsrModelReady is false", async () => {
    useModelsStore.setState({ isAsrModelReady: false });

    await useRecordingStore.getState().start();

    expect(commands.startRecording).not.toHaveBeenCalled();
    expect(useRecordingStore.getState().lastError).toBe(
      "ASR model not yet downloaded",
    );
  });

  it("start calls startRecording when isAsrModelReady is true", async () => {
    useModelsStore.setState({ isAsrModelReady: true });
    vi.mocked(commands.startRecording).mockResolvedValueOnce({
      status: "ok",
      data: "meeting-uuid",
    });

    await useRecordingStore.getState().start();

    expect(commands.startRecording).toHaveBeenCalledOnce();
    expect(useRecordingStore.getState().lastError).toBeNull();
  });

  /** Put the store into a live recording so transcript_segment events for that
   * meeting append (segments are filtered by the live meeting_id). */
  function startLive(meetingId: string) {
    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "recording", meeting_id: meetingId, started_at_ms: 0 },
    });
  }

  it("handleEvent transcript_segment appends segment to transcript", () => {
    startLive("m1");
    const seg = makeSegment(1000, "first segment");
    const event: AppEvent = {
      kind: "transcript_segment",
      meeting_id: "m1",
      segment: seg,
    };

    useRecordingStore.getState().handleEvent(event);

    expect(useRecordingStore.getState().transcript).toHaveLength(1);
    expect(useRecordingStore.getState().transcript[0]).toEqual(seg);
  });

  it("handleEvent transcript_segment appends multiple segments in order", () => {
    startLive("m1");
    const segs = [
      makeSegment(0, "first"),
      makeSegment(1000, "second"),
      makeSegment(2000, "third"),
    ];

    for (const seg of segs) {
      useRecordingStore.getState().handleEvent({
        kind: "transcript_segment",
        meeting_id: "m1",
        segment: seg,
      });
    }

    const stored = useRecordingStore.getState().transcript;
    expect(stored).toHaveLength(3);
    expect(stored.map((s) => s.text)).toEqual(["first", "second", "third"]);
  });

  it("handleEvent transcript_segment ignores segments for a non-live meeting", () => {
    // A background/offline re-transcribe emits transcript_segment for a
    // previously-stopped meeting; with no matching live recording it must NOT
    // pollute the live transcript array.
    startLive("m1");
    useRecordingStore.getState().handleEvent({
      kind: "transcript_segment",
      meeting_id: "other-meeting",
      segment: makeSegment(0, "from an offline re-transcribe"),
    });
    expect(useRecordingStore.getState().transcript).toHaveLength(0);
  });

  it("handleEvent state_changed → recording clears transcript", () => {
    useRecordingStore.setState({
      transcript: [makeSegment(0, "old segment")],
    });

    const event: AppEvent = {
      kind: "state_changed",
      state: { kind: "recording", meeting_id: "new-meeting", started_at_ms: Date.now() },
    };

    useRecordingStore.getState().handleEvent(event);

    expect(useRecordingStore.getState().transcript).toHaveLength(0);
    expect(useRecordingStore.getState().state.kind).toBe("recording");
  });

  it("handleEvent state_changed → idle does not clear transcript", () => {
    useRecordingStore.setState({
      transcript: [makeSegment(0, "kept segment")],
    });

    const event: AppEvent = {
      kind: "state_changed",
      state: { kind: "idle" },
    };

    useRecordingStore.getState().handleEvent(event);

    expect(useRecordingStore.getState().transcript).toHaveLength(1);
  });

  // -------------------------------------------------------------------------
  // Live-test UX T1: double-press guard + the `preparing` optimistic transient.
  // -------------------------------------------------------------------------

  it("start sets the optimistic `preparing` flag before the recording event arrives", async () => {
    useModelsStore.setState({ isAsrModelReady: true });
    vi.mocked(commands.startRecording).mockResolvedValueOnce({
      status: "ok",
      data: "m-prep",
    });

    await useRecordingStore.getState().start();

    // The backend has not yet emitted `recording`, so the UI is still preparing.
    expect(useRecordingStore.getState().preparing).toBe(true);
    expect(commands.startRecording).toHaveBeenCalledOnce();

    // The `recording` state event resolves the preparing transient.
    useRecordingStore.getState().handleEvent({
      kind: "state_changed",
      state: { kind: "recording", meeting_id: "m-prep", started_at_ms: 0 },
    });
    expect(useRecordingStore.getState().preparing).toBe(false);
  });

  it("start is a no-op while already preparing (double-press guard)", async () => {
    useModelsStore.setState({ isAsrModelReady: true });
    // Already preparing (a start is in flight).
    useRecordingStore.setState({ preparing: true });

    await useRecordingStore.getState().start();

    expect(commands.startRecording).not.toHaveBeenCalled();
  });

  it("start is a no-op while not idle (double-press while recording)", async () => {
    useModelsStore.setState({ isAsrModelReady: true });
    useRecordingStore.setState({
      state: { kind: "recording", meeting_id: "m1", started_at_ms: 0 },
      preparing: false,
    });

    await useRecordingStore.getState().start();

    // A second press while Recording must NOT re-invoke startRecording (the
    // orchestrator would reject it with "start called when not idle").
    expect(commands.startRecording).not.toHaveBeenCalled();
  });

  it("start clears `preparing` and surfaces lastError when startRecording rejects", async () => {
    useModelsStore.setState({ isAsrModelReady: true });
    vi.mocked(commands.startRecording).mockResolvedValueOnce({
      status: "error",
      error: { code: "internal", context: "device busy" },
    });

    await useRecordingStore.getState().start();

    expect(useRecordingStore.getState().preparing).toBe(false);
    expect(useRecordingStore.getState().lastError).not.toBeNull();
  });
});
