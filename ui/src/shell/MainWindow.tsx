import { useEffect } from "react";
import { useRecordingStore } from "../state/recording";
import { DevicePicker } from "./DevicePicker";
import { MeetingControls } from "./MeetingControls";
import { AudioMeter } from "./AudioMeter";
import "./MainWindow.css";

/**
 * Root shell component.
 *
 * Composes DevicePicker, MeetingControls, and AudioMeter. Fetches the
 * initial device list on mount. Error banner shows `lastError` when set.
 *
 * The global event bridge (`useAppEventBridge`) is mounted one level up in
 * `App.tsx` so it cannot be accidentally unmounted when this component
 * re-renders.
 */
export function MainWindow() {
  const refreshDevices = useRecordingStore((s) => s.refreshDevices);
  const lastError = useRecordingStore((s) => s.lastError);

  useEffect(() => {
    void refreshDevices();
  }, [refreshDevices]);

  return (
    <div className="main-window">
      <header className="main-window__header">
        <h1>meeting-app</h1>
      </header>

      <main className="main-window__content">
        <section className="main-window__controls">
          <DevicePicker />
          <MeetingControls />
        </section>

        <section className="main-window__meter">
          <label>Audio level</label>
          <AudioMeter />
        </section>

        {lastError && (
          <div className="main-window__error" role="alert">
            {lastError}
          </div>
        )}
      </main>
    </div>
  );
}
