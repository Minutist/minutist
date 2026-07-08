/**
 * Live-test UX T3/T4 — the per-meeting operation-progress store.
 *
 * Verifies:
 * - `operation_progress` upserts the in-flight op (determinate + indeterminate).
 * - the terminal events (`transcript_ready` / `diarization_complete` /
 *   `summary_ready` / `meeting_finalised`) clear the row's indicator.
 * - a terminal event for an unrelated meeting leaves other rows untouched.
 */
import { describe, it, expect, beforeEach } from "vitest";

import { useOperationProgressStore } from "../state/operation-progress";
import type { AppEvent } from "../ipc/bindings";

describe("operation-progress store", () => {
  beforeEach(() => {
    useOperationProgressStore.setState({ operations: {} });
  });

  it("operation_progress upserts a determinate op (fraction present)", () => {
    const event: AppEvent = {
      kind: "operation_progress",
      meeting_id: "m1",
      op: "re_transcribe",
      fraction: 0.42,
      label: "Re-transcribing…",
    };
    useOperationProgressStore.getState().handleEvent(event);

    const op = useOperationProgressStore.getState().operationFor("m1");
    expect(op).toEqual({
      op: "re_transcribe",
      fraction: 0.42,
      label: "Re-transcribing…",
    });
  });

  it("operation_progress upserts an indeterminate op (fraction null)", () => {
    useOperationProgressStore.getState().handleEvent({
      kind: "operation_progress",
      meeting_id: "m1",
      op: "rediarize",
      fraction: null,
      label: "Identifying speakers…",
    });

    const op = useOperationProgressStore.getState().operationFor("m1");
    expect(op?.fraction).toBeNull();
    expect(op?.op).toBe("rediarize");
  });

  it("a later progress event overwrites the prior fraction for the same meeting", () => {
    const store = useOperationProgressStore.getState();
    store.handleEvent({
      kind: "operation_progress",
      meeting_id: "m1",
      op: "summarise",
      fraction: 0.1,
      label: "Summarising…",
    });
    store.handleEvent({
      kind: "operation_progress",
      meeting_id: "m1",
      op: "summarise",
      fraction: 0.8,
      label: "Summarising…",
    });
    expect(useOperationProgressStore.getState().operationFor("m1")?.fraction).toBe(
      0.8,
    );
  });

  it.each([
    "transcript_ready",
    "diarization_complete",
    "summary_ready",
    "summary_unavailable",
    "meeting_finalised",
  ] as const)("terminal event %s clears the meeting's indicator", (kind) => {
    useOperationProgressStore.setState({
      operations: {
        m1: { op: "re_transcribe", fraction: 0.5, label: "Re-transcribing…" },
      },
    });

    // diarization_complete carries an extra field; build a tolerant event.
    const event = (
      kind === "diarization_complete"
        ? { kind, meeting_id: "m1", speaker_count: 2 }
        : { kind, meeting_id: "m1" }
    ) as AppEvent;

    useOperationProgressStore.getState().handleEvent(event);
    expect(useOperationProgressStore.getState().operationFor("m1")).toBeNull();
  });

  it("a terminal event for another meeting leaves other rows untouched", () => {
    useOperationProgressStore.setState({
      operations: {
        m1: { op: "summarise", fraction: 0.3, label: "Summarising…" },
        m2: { op: "rediarize", fraction: null, label: "Identifying speakers…" },
      },
    });

    useOperationProgressStore.getState().handleEvent({
      kind: "summary_ready",
      meeting_id: "m1",
    });

    expect(useOperationProgressStore.getState().operationFor("m1")).toBeNull();
    expect(useOperationProgressStore.getState().operationFor("m2")).not.toBeNull();
  });

  // -------------------------------------------------------------------------
  // `error_occurred` — the finalise-handshake-abort terminal signal (G3d).
  // `AppError` carries no `meeting_id`, so this cannot target one row by id;
  // it clears every `finalise` row instead (at most one can be in flight).
  // -------------------------------------------------------------------------

  it("error_occurred clears a finalise row (the post-stop handshake aborted)", () => {
    useOperationProgressStore.setState({
      operations: {
        m1: { op: "finalise", fraction: null, label: "Finishing up…" },
      },
    });

    useOperationProgressStore.getState().handleEvent({
      kind: "error_occurred",
      error: { code: "io", context: "disk full" },
    });

    expect(useOperationProgressStore.getState().operationFor("m1")).toBeNull();
  });

  it("error_occurred leaves non-finalise rows untouched", () => {
    useOperationProgressStore.setState({
      operations: {
        m1: { op: "finalise", fraction: null, label: "Finishing up…" },
        m2: { op: "re_transcribe", fraction: 0.5, label: "Re-transcribing…" },
      },
    });

    useOperationProgressStore.getState().handleEvent({
      kind: "error_occurred",
      error: { code: "internal", context: "boom" },
    });

    expect(useOperationProgressStore.getState().operationFor("m1")).toBeNull();
    expect(useOperationProgressStore.getState().operationFor("m2")).toEqual({
      op: "re_transcribe",
      fraction: 0.5,
      label: "Re-transcribing…",
    });
  });

  it("error_occurred is a no-op when nothing is finalising", () => {
    useOperationProgressStore.setState({
      operations: {
        m1: { op: "summarise", fraction: 0.2, label: "Summarising…" },
      },
    });

    useOperationProgressStore.getState().handleEvent({
      kind: "error_occurred",
      error: { code: "cancelled" },
    });

    expect(useOperationProgressStore.getState().operationFor("m1")).toEqual({
      op: "summarise",
      fraction: 0.2,
      label: "Summarising…",
    });
  });
});
