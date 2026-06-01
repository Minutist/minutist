/**
 * Notes autosave (FR-18).
 *
 * Persists the editor's notes on a fixed interval (`autosave_interval_secs`,
 * default 5 s) and on blur, via the `saveNotes` IPC seam (`../ipc/notes`).
 *
 * Hard rule: autosave is a **no-op when there is no active recording / no
 * MeetingId**. Notes are scoped to a meeting; with no meeting there is nowhere
 * to persist them. The active MeetingId is derived from the recording state
 * (`recording` / `paused` / `stopping` carry it; `idle` does not).
 */
import { useEffect, useRef } from "react";
import type { RecordingState } from "../ipc/bindings";
import { saveNotes } from "../ipc/notes";

/** Default autosave cadence when settings has no `autosave_interval_secs`. */
export const DEFAULT_AUTOSAVE_INTERVAL_SECS = 5;

/**
 * Extract the MeetingId from a recording state, or `null` when idle.
 *
 * `idle` has no meeting; `recording` / `paused` / `stopping` all reference the
 * in-flight meeting.
 */
export function activeMeetingId(state: RecordingState): string | null {
  switch (state.kind) {
    case "recording":
    case "paused":
    case "stopping":
      return state.meeting_id;
    case "idle":
      return null;
  }
}

/** A snapshot of the notes the autosave loop should persist. */
export type NotesSnapshot = {
  notesJson: string;
  notesMarkdown: string;
};

export type UseAutosaveArgs = {
  /** Current recording state — gates whether autosave runs at all. */
  state: RecordingState;
  /** Autosave cadence in seconds; falls back to the default when null. */
  intervalSecs: number | null;
  /**
   * Reads the latest notes to persist. Returns `null` when there is nothing
   * worth saving (e.g. the editor is not ready yet).
   */
  getSnapshot: () => NotesSnapshot | null;
  /** Optional error sink; receives the rejection if a save fails. */
  onError?: (err: unknown) => void;
};

/**
 * Drive interval autosave plus an imperative flush.
 *
 * Returns a `flush()` callback the caller wires to the editor's blur handler so
 * notes are persisted immediately when focus leaves the editor.
 */
export function useAutosave(args: UseAutosaveArgs): { flush: () => void } {
  const { state, intervalSecs, getSnapshot, onError } = args;

  // Keep the latest snapshot/error callbacks in refs so the interval effect
  // does not re-subscribe on every keystroke (which would reset the timer).
  const getSnapshotRef = useRef(getSnapshot);
  const onErrorRef = useRef(onError);
  getSnapshotRef.current = getSnapshot;
  onErrorRef.current = onError;

  const meetingId = activeMeetingId(state);

  const performSave = useRef((id: string) => {
    const snapshot = getSnapshotRef.current();
    if (snapshot === null) return;
    saveNotes({
      meetingId: id,
      notesJson: snapshot.notesJson,
      notesMarkdown: snapshot.notesMarkdown,
    }).catch((err: unknown) => {
      onErrorRef.current?.(err);
    });
  });

  // Interval autosave. Only active while a MeetingId exists.
  useEffect(() => {
    if (meetingId === null) return;
    const secs =
      intervalSecs && intervalSecs > 0
        ? intervalSecs
        : DEFAULT_AUTOSAVE_INTERVAL_SECS;
    const handle = setInterval(() => {
      performSave.current(meetingId);
    }, secs * 1000);
    return () => {
      clearInterval(handle);
    };
  }, [meetingId, intervalSecs]);

  // Imperative flush (wired to editor blur). No-op without a MeetingId.
  const flush = useRef(() => {
    if (meetingId === null) return;
    performSave.current(meetingId);
  });
  // Keep the closed-over meetingId fresh without changing the callback identity.
  flush.current = () => {
    if (meetingId === null) return;
    performSave.current(meetingId);
  };

  return { flush: () => flush.current() };
}
