import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import type { RecordingState } from "../ipc/bindings";

/**
 * Derive the two context-aware transport toggles from the current recording
 * state (#66 — consolidate the former four always-on buttons to two).
 *
 * There are two buttons:
 *   - RECORD: "Start" (calls `start`) when idle; "Stop" (calls `stop`) when
 *     recording OR paused. Disabled while finalising/stopping and, when idle,
 *     unless the ASR model is ready and no start is already in flight.
 *   - PAUSE: "Pause" (calls `pause`) when recording; "Resume" (calls `resume`)
 *     when paused; disabled otherwise (idle / finalising / stopping).
 *
 * `preparing` is the client-only optimistic transient (live-test UX T1): while
 * the first record lazy-loads the ASR model, the record toggle MUST stay
 * disabled so a double-press cannot re-invoke `startRecording` (which the
 * orchestrator rejects with "start called when not idle").
 *
 * Exported for use in unit tests without requiring the Zustand store.
 */
export type RecordAction = "start" | "stop";
export type PauseAction = "pause" | "resume";

export type ButtonStates = {
  /** The label/intent of the RECORD toggle. */
  recordAction: RecordAction;
  recordEnabled: boolean;
  /** The label/intent of the PAUSE toggle. */
  pauseAction: PauseAction;
  pauseEnabled: boolean;
};

export function deriveButtonStates(
  state: RecordingState,
  isAsrModelReady: boolean,
  preparing: boolean,
): ButtonStates {
  const isIdle = state.kind === "idle";
  const isRecording = state.kind === "recording";
  const isPaused = state.kind === "paused";

  // RECORD: Start from idle, Stop while recording or paused.
  const recordAction: RecordAction = isIdle ? "start" : "stop";
  const recordEnabled = isIdle
    ? // Start only from a genuinely idle recorder, with the model ready, and NOT
      // while a start is already in flight (preparing) — a double-press is then
      // impossible.
      isAsrModelReady && !preparing
    : // Stop only from recording/paused — disabled while stopping/finalising.
      isRecording || isPaused;

  // PAUSE: Pause while recording, Resume while paused, disabled otherwise.
  const pauseAction: PauseAction = isPaused ? "resume" : "pause";
  const pauseEnabled = isRecording || isPaused;

  return { recordAction, recordEnabled, pauseAction, pauseEnabled };
}

export function MeetingControls() {
  const state = useRecordingStore((s) => s.state);
  const start = useRecordingStore((s) => s.start);
  const stop = useRecordingStore((s) => s.stop);
  const pause = useRecordingStore((s) => s.pause);
  const resume = useRecordingStore((s) => s.resume);
  const preparing = useRecordingStore((s) => s.preparing);
  const isAsrModelReady = useModelsStore((s) => s.isAsrModelReady);

  const { recordAction, recordEnabled, pauseAction, pauseEnabled } =
    deriveButtonStates(state, isAsrModelReady, preparing);

  // The record toggle shows "Preparing…" only while a fresh start is loading the
  // model; once recording/paused it is the Stop control regardless.
  const recordLabel =
    recordAction === "start" ? (preparing ? "Preparing…" : "Start") : "Stop";
  const pauseLabel = pauseAction === "pause" ? "Pause" : "Resume";

  return (
    <div className="meeting-controls">
      <button
        className="meeting-controls__record"
        data-action={recordAction}
        onClick={() => void (recordAction === "start" ? start() : stop())}
        disabled={!recordEnabled}
        aria-label={recordLabel}
      >
        {recordLabel}
      </button>
      <button
        className="meeting-controls__pause"
        data-action={pauseAction}
        onClick={() => void (pauseAction === "pause" ? pause() : resume())}
        disabled={!pauseEnabled}
        aria-label={pauseLabel}
      >
        {pauseLabel}
      </button>
    </div>
  );
}
