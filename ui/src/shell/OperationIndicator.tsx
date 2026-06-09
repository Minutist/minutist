/**
 * Non-blocking per-row operation indicator (live-test UX T3 + T4).
 *
 * Renders the in-flight long-operation for a meeting:
 *   - a determinate bar when a `fraction` (0..=1) is available;
 *   - an indeterminate spinner when `fraction` is `null`.
 *
 * Renders nothing when the meeting has no in-flight operation. Modest by design:
 * a label + a thin bar / a small spinner, styled from theme tokens.
 */
import type { MeetingId } from "../ipc/bindings";
import { useOperationProgressStore } from "../state/operation-progress";

export function OperationIndicator(props: { meetingId: MeetingId }) {
  const op = useOperationProgressStore((s) => s.operations[props.meetingId]);
  if (!op) return null;

  const determinate = op.fraction !== null;
  const pct = determinate
    ? Math.round(Math.min(1, Math.max(0, op.fraction as number)) * 100)
    : null;

  return (
    <div
      className="operation-indicator"
      role="status"
      aria-live="polite"
      data-op={op.op}
    >
      <span className="operation-indicator__label">
        {op.label}
        {pct !== null ? ` ${pct}%` : ""}
      </span>
      {determinate ? (
        <div
          className="operation-indicator__bar"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct ?? 0}
        >
          <div
            className="operation-indicator__bar-fill"
            style={{ width: `${pct ?? 0}%` }}
          />
        </div>
      ) : (
        <span
          className="operation-indicator__spinner"
          aria-hidden="true"
        />
      )}
    </div>
  );
}
