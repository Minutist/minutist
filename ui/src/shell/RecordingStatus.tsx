import { useEffect, useRef, useState } from "react";
import { useRecordingStore } from "../state/recording";
import type { RecordingState } from "../ipc/bindings";

/**
 * Format an elapsed-millisecond duration as H:MM:SS (or M:SS under an hour).
 * Tabular figures are applied by the caller via the `.tnum` class.
 */
export function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const ss = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const mm = totalMinutes % 60;
  const hh = Math.floor(totalMinutes / 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  return hh > 0 ? `${hh}:${pad(mm)}:${pad(ss)}` : `${mm}:${pad(ss)}`;
}

/** Wall-clock start, or `null` when there is nothing to count from. */
function startedAtMs(state: RecordingState): number | null {
  switch (state.kind) {
    case "recording":
      return state.started_at_ms;
    case "paused":
      // Freeze the clock at the moment of pause.
      return state.paused_at_ms;
    default:
      return null;
  }
}

/**
 * The focal element of the top bar: an oxblood status dot (gently pulsing only
 * while recording) plus the elapsed recording time in tabular Newsreader.
 *
 * Idle shows a quiet "Ready" label; recording/paused show the live elapsed
 * clock. The clock is display-only (`Date.now() - started_at_ms`) per the
 * `RecordingState` timestamp contract — NOT a paragraph-anchor source.
 */
export function RecordingStatus() {
  const state = useRecordingStore((s) => s.state);
  const kind = state.kind;
  const isRecording = kind === "recording";
  const isPaused = kind === "paused";
  const isStopping = kind === "stopping";

  const start = startedAtMs(state);
  const [now, setNow] = useState(() => Date.now());

  // Tick the displayed clock once a second only while actively recording.
  const startRef = useRef(start);
  startRef.current = start;
  useEffect(() => {
    if (!isRecording) return;
    setNow(Date.now());
    const handle = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(handle);
  }, [isRecording]);

  let label: string;
  let elapsed: string | null = null;
  if (isRecording && start !== null) {
    label = "Recording";
    elapsed = formatElapsed(now - start);
  } else if (isPaused) {
    // The paused state carries `paused_at_ms` (wall-clock of the pause), not
    // the recording start, so an accurate elapsed-since-start is not derivable
    // here. Show the state without a misleading clock.
    label = "Paused";
  } else if (isStopping) {
    label = "Stopping…";
  } else {
    label = "Ready";
  }

  return (
    <div
      className="recording-status"
      data-state={kind}
      role="status"
      aria-live="polite"
    >
      <span
        className={`recording-status__dot${
          isRecording ? " recording-status__dot--live" : ""
        }`}
        aria-hidden="true"
      />
      <span className="recording-status__label">{label}</span>
      {elapsed !== null && (
        <span className="recording-status__elapsed tnum">{elapsed}</span>
      )}
    </div>
  );
}
