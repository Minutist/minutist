/**
 * Notes autosave (FR-18).
 *
 * Persists the editor's notes on a fixed interval (`autosave_interval_secs`,
 * default 5 s) and on blur, via the `saveNotes` IPC seam (`../ipc/notes`).
 *
 * The document the notes belong to is the active recording while capturing,
 * otherwise the open saved meeting being viewed — the same identity rule the
 * rest of the UI uses (`activeMeetingId(state) ?? openMeetingId`). Autosave is
 * a no-op only when there is neither: the live entry surface with nothing open.
 * Without the open-meeting fallback, edits to a finished/opened meeting are
 * silently lost (idle carries no recording MeetingId).
 */
import { useEffect, useRef } from "react";
import type { RecordingState } from "../ipc/bindings";
import { saveNotes } from "../ipc/notes";

/** Default autosave cadence when settings has no `autosave_interval_secs`. */
export const DEFAULT_AUTOSAVE_INTERVAL_SECS = 5;

/**
 * Extract the MeetingId from a recording state, or `null` when idle.
 *
 * `idle` has no meeting; `recording` / `paused` / `stopping` / `finalising` all
 * reference the in-flight (or finalising) meeting.
 */
export function activeMeetingId(state: RecordingState): string | null {
  switch (state.kind) {
    case "recording":
    case "paused":
    case "stopping":
    case "finalising":
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
  /** Current recording state — supplies the MeetingId while capturing. */
  state: RecordingState;
  /**
   * The open saved meeting's id, used when no recording is active so edits to a
   * finished/opened meeting persist. Recording takes precedence; absent/`null`
   * with an idle recorder means no meeting → autosave is a no-op.
   */
  openMeetingId?: string | null;
  /** Autosave cadence in seconds; falls back to the default when null. */
  intervalSecs: number | null;
  /**
   * Whether this legacy JSON autosave path is active. Defaults to `true`. The
   * collaboration binding (B6 WU7) sets this `false` when a Y.Doc is bound, so
   * the Y.Doc → `apply_notes_update` path is the single writer and this autosave
   * does not race it on the same files.
   */
  enabled?: boolean;
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
  const { state, openMeetingId, intervalSecs, getSnapshot, onError } = args;
  const enabled = args.enabled ?? true;

  // Keep the latest snapshot/error callbacks in refs so the interval effect
  // does not re-subscribe on every keystroke (which would reset the timer).
  const getSnapshotRef = useRef(getSnapshot);
  const onErrorRef = useRef(onError);
  getSnapshotRef.current = getSnapshot;
  onErrorRef.current = onError;

  // Recording (live) takes precedence; otherwise the open saved meeting. Null
  // only on the live entry surface with nothing open → autosave no-ops. Also
  // null when this path is disabled (collab binding owns persistence).
  const meetingId = enabled
    ? activeMeetingId(state) ?? openMeetingId ?? null
    : null;

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
