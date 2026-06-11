/**
 * Per-meeting long-operation progress store (live-test UX T3 + T4).
 *
 * Tracks the in-flight long-running operation for each meeting so the
 * meeting-list renders a NON-BLOCKING per-row indicator: a determinate bar when
 * a `fraction` is available, an indeterminate spinner when it is `None`. The
 * indicator clears on the operation's terminal event so a finished pass leaves
 * no stale spinner.
 *
 * Fed by the global event bridge (`shell/event-listener.tsx`) via `handleEvent`:
 *   - `operation_progress`  → upsert the row's in-flight op (label + fraction).
 *   - `transcript_ready`     → clear (the re-transcribe pass finished).
 *   - `diarization_complete` → clear (the re-identify-speakers pass finished).
 *   - `summary_ready`        → clear (the summarise pass finished).
 *   - `meeting_finalised`    → clear (the finalise drain finished).
 *   - `translation_ready`    → clear (the translate pass finished).
 *
 * The store keeps only transient UI state keyed by `MeetingId`; it is never a
 * source of truth the backend owns.
 */
import { create } from "zustand";
import type { MeetingId, OperationKind } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

/** The in-flight operation shown on a meeting-list row. */
export type MeetingOperation = {
  /** Which operation is running (drives the label / clearing logic). */
  op: OperationKind;
  /** `null` = indeterminate spinner; a `0..=1` value = determinate bar. */
  fraction: number | null;
  /** Short human-readable label (e.g. "Re-transcribing…"). */
  label: string;
};

export type OperationProgressStore = {
  /** In-flight operation per meeting id, or absent when nothing is running. */
  operations: Record<MeetingId, MeetingOperation>;
  /** The in-flight operation for a meeting, or `null` when idle. */
  operationFor: (meetingId: MeetingId) => MeetingOperation | null;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useOperationProgressStore = create<OperationProgressStore>(
  (set, get) => ({
    operations: {},

    operationFor: (meetingId) => get().operations[meetingId] ?? null,

    handleEvent: (event) => {
      switch (event.kind) {
        case "operation_progress": {
          set((s) => ({
            operations: {
              ...s.operations,
              [event.meeting_id]: {
                op: event.op,
                fraction: event.fraction,
                label: event.label,
              },
            },
          }));
          break;
        }
        // Terminal events: clear the row's indicator. Each is the completion
        // signal for one (or more) of the operations.
        case "transcript_ready":
        case "diarization_complete":
        case "summary_ready":
        case "meeting_finalised":
        case "translation_ready": {
          const meetingId = event.meeting_id;
          set((s) => {
            if (!(meetingId in s.operations)) return s;
            const next = { ...s.operations };
            delete next[meetingId];
            return { operations: next };
          });
          break;
        }
        default:
          break;
      }
    },
  }),
);
