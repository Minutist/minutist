/**
 * Behaviour tests for the active-transcript source selector (U1).
 *
 * `active-transcript` chooses the SAVED meeting's transcript only when a meeting
 * is open AND nothing is being recorded
 * (`openMeetingId !== null && recordingKind === "idle"`); otherwise it reads the
 * LIVE recording store. The "viewing a saved meeting while idle" branch is
 * covered by OpenMeetingRestore.test.tsx; this file pins the precedence rule:
 * when a meeting is open BUT a recording is in progress, the LIVE transcript
 * wins. A regression dropping the `=== "idle"` guard would return the stale
 * saved transcript and fail these tests.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { act } from "react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

import {
  activeTranscript,
  useActiveTranscript,
  isViewingSavedMeeting,
} from "../state/active-transcript";
import { useRecordingStore } from "../state/recording";
import { useMeetingsStore } from "../state/meetings";
import type { MeetingState } from "../state/meetings";
import type { Segment } from "../ipc/bindings";

const LIVE: Segment[] = [
  { start_ms: 0, end_ms: 1_000, text: "live segment", words: [] },
];
const SAVED: Segment[] = [
  { start_ms: 5_000, end_ms: 6_000, text: "saved segment one", words: [] },
  { start_ms: 7_000, end_ms: 8_000, text: "saved segment two", words: [] },
];

function setSavedMeetingOpen() {
  useMeetingsStore.setState({
    openMeetingId: "saved-0001",
    openMeetingState: { transcript: SAVED } as unknown as MeetingState,
  });
}

describe("active-transcript: recording takes precedence over an open meeting", () => {
  beforeEach(() => {
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" }, transcript: LIVE });
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  afterEach(() => {
    act(() => {
      useMeetingsStore.setState({ openMeetingId: null, openMeetingState: null });
    });
  });

  it("activeTranscript() returns the LIVE transcript when a meeting is open but recording is in progress", () => {
    act(() => {
      setSavedMeetingOpen();
      // A meeting id is still set, but a recording resumed/started.
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "m1", started_at_ms: 1 },
      });
    });

    // Live wins despite openMeetingId !== null — this is the precedence rule a
    // regression dropping `=== "idle"` would break (it would return SAVED).
    expect(activeTranscript()).toBe(LIVE);
    expect(activeTranscript()).not.toBe(SAVED);
    expect(isViewingSavedMeeting()).toBe(false);
  });

  it("useActiveTranscript() returns the LIVE transcript when a meeting is open but recording is in progress", () => {
    act(() => {
      setSavedMeetingOpen();
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "m1", started_at_ms: 1 },
      });
    });

    const { result } = renderHook(() => useActiveTranscript());
    expect(result.current).toBe(LIVE);
    expect(result.current).not.toBe(SAVED);
  });

  it("flips back to LIVE when a recording starts while a saved meeting is open", () => {
    act(() => {
      setSavedMeetingOpen();
    });

    const { result, rerender } = renderHook(() => useActiveTranscript());
    // Idle + meeting open → saved transcript is shown.
    expect(result.current).toBe(SAVED);

    // A recording begins while the meeting id is still set → live takes over.
    act(() => {
      useRecordingStore.setState({
        state: { kind: "recording", meeting_id: "m1", started_at_ms: 1 },
      });
    });
    rerender();
    expect(result.current).toBe(LIVE);
  });
});
