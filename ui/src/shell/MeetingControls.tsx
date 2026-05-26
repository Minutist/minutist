import { useRecordingStore } from "../state/recording";
import type { RecordingStateType } from "../ipc/bindings";

/**
 * Returns per-button enabled flags derived from the current recording state.
 *
 * Exported for use in unit tests without requiring the Zustand store.
 */
export function deriveButtonStates(state: RecordingStateType) {
  const isIdle = state.kind === "idle";
  const isRecording = state.kind === "recording";
  const isPaused = state.kind === "paused";

  return {
    startEnabled: isIdle,
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

  const { startEnabled, stopEnabled, pauseEnabled, resumeEnabled } =
    deriveButtonStates(state);

  return (
    <div className="meeting-controls">
      <button onClick={() => void start()} disabled={!startEnabled}>
        Start
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
