/**
 * Tests for notes autosave (FR-18).
 *
 * Verifies:
 * - autosave fires on the configured interval while a meeting is active,
 * - flush() saves immediately (used on blur),
 * - autosave is a no-op (interval + flush) when there is no MeetingId,
 * - the configured `autosave_interval_secs` controls the cadence.
 *
 * The `save_notes` IPC call is mocked at the `../ipc/notes` seam (per the
 * architecture testing policy — do not fake the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

vi.mock("../ipc/notes", () => ({
  saveNotes: vi.fn().mockResolvedValue(undefined),
  loadNotes: vi.fn().mockResolvedValue(null),
}));

import { saveNotes } from "../ipc/notes";
import { useAutosave, activeMeetingId } from "../editor/useAutosave";
import type { RecordingState } from "../ipc/bindings";

const recordingState: RecordingState = {
  kind: "recording",
  meeting_id: "meeting-123",
  started_at_ms: 1_000,
};
const idleState: RecordingState = { kind: "idle" };

const snapshot = { notesJson: '{"type":"doc"}', notesMarkdown: "# note" };

describe("activeMeetingId", () => {
  it("returns the meeting id for recording/paused/stopping", () => {
    expect(activeMeetingId(recordingState)).toBe("meeting-123");
    expect(
      activeMeetingId({ kind: "paused", meeting_id: "m2", paused_at_ms: 0 }),
    ).toBe("m2");
    expect(activeMeetingId({ kind: "stopping", meeting_id: "m3" })).toBe("m3");
  });

  it("returns null when idle", () => {
    expect(activeMeetingId(idleState)).toBeNull();
  });
});

describe("useAutosave", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(saveNotes).mockClear();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("fires on the configured interval with the meeting id and snapshot", () => {
    renderHook(() =>
      useAutosave({
        state: recordingState,
        intervalSecs: 5,
        getSnapshot: () => snapshot,
      }),
    );

    expect(saveNotes).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(saveNotes).toHaveBeenCalledTimes(1);
    expect(saveNotes).toHaveBeenCalledWith({
      meetingId: "meeting-123",
      notesJson: snapshot.notesJson,
      notesMarkdown: snapshot.notesMarkdown,
    });

    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(saveNotes).toHaveBeenCalledTimes(2);
  });

  it("honours a custom interval", () => {
    renderHook(() =>
      useAutosave({
        state: recordingState,
        intervalSecs: 2,
        getSnapshot: () => snapshot,
      }),
    );

    act(() => {
      vi.advanceTimersByTime(2_000);
    });
    expect(saveNotes).toHaveBeenCalledTimes(1);
  });

  it("uses the 5s default when interval is null", () => {
    renderHook(() =>
      useAutosave({
        state: recordingState,
        intervalSecs: null,
        getSnapshot: () => snapshot,
      }),
    );

    act(() => {
      vi.advanceTimersByTime(4_999);
    });
    expect(saveNotes).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(saveNotes).toHaveBeenCalledTimes(1);
  });

  it("flush() saves immediately", () => {
    const { result } = renderHook(() =>
      useAutosave({
        state: recordingState,
        intervalSecs: 5,
        getSnapshot: () => snapshot,
      }),
    );

    act(() => {
      result.current.flush();
    });

    expect(saveNotes).toHaveBeenCalledTimes(1);
    expect(saveNotes).toHaveBeenCalledWith({
      meetingId: "meeting-123",
      notesJson: snapshot.notesJson,
      notesMarkdown: snapshot.notesMarkdown,
    });
  });

  it("is a no-op on the interval when there is no MeetingId (idle)", () => {
    renderHook(() =>
      useAutosave({
        state: idleState,
        intervalSecs: 5,
        getSnapshot: () => snapshot,
      }),
    );

    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(saveNotes).not.toHaveBeenCalled();
  });

  it("flush() is a no-op when there is no MeetingId (idle)", () => {
    const { result } = renderHook(() =>
      useAutosave({
        state: idleState,
        intervalSecs: 5,
        getSnapshot: () => snapshot,
      }),
    );

    act(() => {
      result.current.flush();
    });
    expect(saveNotes).not.toHaveBeenCalled();
  });

  it("does not save when the snapshot is null", () => {
    renderHook(() =>
      useAutosave({
        state: recordingState,
        intervalSecs: 5,
        getSnapshot: () => null,
      }),
    );

    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(saveNotes).not.toHaveBeenCalled();
  });
});
