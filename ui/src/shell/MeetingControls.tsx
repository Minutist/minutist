import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import type { RecordingState } from "../ipc/bindings";

/**
 * Returns per-button enabled flags derived from the current recording state.
 *
 * `preparing` is the client-only optimistic transient (live-test UX T1): while
 * the first record lazy-loads the ASR model, Start MUST stay disabled so a
 * double-press cannot re-invoke `startRecording` (which the orchestrator rejects
 * with "start called when not idle").
 *
 * Exported for use in unit tests without requiring the Zustand store.
 */
export function deriveButtonStates(
  state: RecordingState,
  isAsrModelReady: boolean,
  preparing: boolean,
) {
  const isIdle = state.kind === "idle";
  const isRecording = state.kind === "recording";
  const isPaused = state.kind === "paused";

  return {
    // Start only from a genuinely idle recorder, with the model ready, and NOT
    // while a start is already in flight (preparing) — a double-press is then
    // impossible.
    startEnabled: isIdle && isAsrModelReady && !preparing,
    stopEnabled: isRecording || isPaused,
    pauseEnabled: isRecording,
    resumeEnabled: isPaused,
  };
}

export function MeetingControls() {
  const state = useRecordingStore((s) => s.state);
  const start = useRecordingStore((s) => s.start);
  const stop = useRecordingStore((s) => s.stop);
  const pause = useRecordingStore((s) => s.pause);
  const resume = useRecordingStore((s) => s.resume);
  const preparing = useRecordingStore((s) => s.preparing);
  const isAsrModelReady = useModelsStore((s) => s.isAsrModelReady);

  const { startEnabled, stopEnabled, pauseEnabled, resumeEnabled } =
    deriveButtonStates(state, isAsrModelReady, preparing);

  return (
    <div className="meeting-controls">
      <button onClick={() => void start()} disabled={!startEnabled}>
        {preparing ? "Preparing…" : "Start"}
      </button>
      <button onClick={() => void pause()} disabled={!pauseEnabled}>
        Pause
      </button>
      <button onClick={() => void resume()} disabled={!resumeEnabled}>
        Resume
      </button>
      <button onClick={() => void stop()} disabled={!stopEnabled}>
        Stop
      </button>
    </div>
  );
}
