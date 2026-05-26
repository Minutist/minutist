import { useRecordingStore } from "../state/recording";

/**
 * Horizontal bar visualising the current audio peak level.
 *
 * The peak value from the backend is a normalised float in [0, 1].
 * Updates are driven by incoming `audio_meter` events from the backend
 * (~30 Hz); no separate requestAnimationFrame polling is added.
 */
export function AudioMeter() {
  const peak = useRecordingStore((s) => s.meter.peak);

  // Clamp to [0, 1] in case of floating-point overshoot from the backend.
  const clamped = Math.min(1, Math.max(0, peak));
  const pct = `${(clamped * 100).toFixed(1)}%`;

  return (
    <div className="audio-meter" aria-label="Audio level meter">
      <div
        className="audio-meter__bar"
        style={{ width: pct }}
        role="meter"
        aria-valuenow={Math.round(clamped * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
}
