/**
 * Live-recording masthead — name the meeting while it records.
 *
 * The in-progress meeting has no `metadata.json` and no title yet (the
 * `Recording <timestamp>` default is only synthesized at stop), so this is an
 * always-editable title field (placeholder "Name this meeting") rather than the
 * click-to-rename affordance the finished-meeting `MeetingMasthead` uses. Each
 * keystroke echoes into the recording store and is pushed to the orchestrator
 * (`setTitle` → `set_recording_title`), which holds it for the live meeting and
 * applies it at `stop()` in place of the default — so the title is captured
 * progressively and is already stored well before the user stops.
 *
 * Shown only while recording / paused (see MainWindow); once finalised+opened,
 * the finished-meeting masthead takes over and the title is renamed the usual way.
 * Reuses the `.meeting-masthead` styling for visual continuity.
 */
import { useRecordingStore } from "../state/recording";
import "./MeetingMasthead.css";

export function RecordingMasthead() {
  const pendingTitle = useRecordingStore((s) => s.pendingTitle);
  const setTitle = useRecordingStore((s) => s.setTitle);

  return (
    <div className="meeting-masthead">
      <input
        className="meeting-masthead__title-input"
        aria-label="Meeting title"
        placeholder="Name this meeting"
        value={pendingTitle}
        onChange={(e) => void setTitle(e.target.value)}
        onKeyDown={(e) => {
          // Enter just commits (the value is already pushed on change) and drops
          // focus, matching the finished-meeting rename's Enter behaviour.
          if (e.key === "Enter") {
            e.preventDefault();
            (e.target as HTMLInputElement).blur();
          }
        }}
      />
      <p className="meeting-masthead__dateline">
        <span>Name it now, or rename it after you stop.</span>
      </p>
    </div>
  );
}
