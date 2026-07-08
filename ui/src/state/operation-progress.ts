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
 *   - `summary_unavailable`  → clear (an auto-summarise was deferred/failed).
 *   - `meeting_finalised`    → clear (the finalise drain finished).
 *   - `translation_ready`    → clear (the translate pass finished).
 *   - `error_occurred`       → clear the `finalise` row (the post-stop
 *     handshake aborted). `AppError` carries no `meeting_id`, so this cannot
 *     target a specific row the way the other terminal events do; `finalise`
 *     is the one op guaranteed to be at most single-flight (only the
 *     just-stopped meeting can be finalising at a time), so clearing every
 *     row whose `op` is `"finalise"` is unambiguous. Other ops keep their own
 *     meeting-scoped terminal event and are unaffected.
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
        case "summary_unavailable":
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
        // `error_occurred` carries no `meeting_id` (see the module doc), so it
        // cannot be matched to a row the way the meeting-scoped terminal
        // events above are. `finalise` is the only op that can never have more
        // than one row in flight app-wide, so clearing every `finalise` row is
        // an unambiguous, safe interpretation of "the operation that just
        // errored" — it clears the spinner left behind by an aborted post-stop
        // finalise handshake without guessing at a meeting id.
        case "error_occurred": {
          set((s) => {
            const next: Record<MeetingId, MeetingOperation> = {};
            let changed = false;
            for (const [meetingId, op] of Object.entries(s.operations)) {
              if (op.op === "finalise") {
                changed = true;
                continue;
              }
              next[meetingId] = op;
            }
            return changed ? { operations: next } : s;
          });
          break;
        }
        default:
          break;
      }
    },
  }),
);
