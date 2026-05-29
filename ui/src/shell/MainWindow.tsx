import { useEffect } from "react";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { DevicePicker } from "./DevicePicker";
import { MeetingControls } from "./MeetingControls";
import { AudioMeter } from "./AudioMeter";
import { ModelDownloadStatus } from "./ModelDownloadStatus";
import { TranscriptPane } from "../transcript/TranscriptPane";
import "./MainWindow.css";

/**
 * Root shell component.
 *
 * Two-column 50/50 layout:
 *   Left  — ModelDownloadStatus (first-run) + DevicePicker + MeetingControls + AudioMeter
 *   Right — TranscriptPane
 *
 * Fetches initial settings, devices, and model list on mount. Error banner
 * shows `lastError` when set.
 *
 * The global event bridge (`useAppEventBridge`) is mounted one level up in
 * `App.tsx` so it cannot be accidentally unmounted when this component
 * re-renders.
 */
export function MainWindow() {
  const refreshDevices = useRecordingStore((s) => s.refreshDevices);
  const refreshSettings = useRecordingStore((s) => s.refreshSettings);
  const lastError = useRecordingStore((s) => s.lastError);
  const refreshModels = useModelsStore((s) => s.refreshModels);

  useEffect(() => {
    // Load persisted settings first so `selectedDeviceId` reflects the
    // user's saved choice; then enumerate devices so the picker is
    // populated. Fetch model list so ModelDownloadStatus renders correctly.
    void refreshSettings();
    void refreshDevices();
    void refreshModels();
  }, [refreshDevices, refreshSettings, refreshModels]);

  return (
    <div className="main-window">
      <header className="main-window__header">
        <h1>meeting-app</h1>
      </header>

      <main className="main-window__content main-window__content--columns">
        {/* Left column: controls */}
        <section className="main-window__left-col">
          <ModelDownloadStatus />

          <div className="main-window__controls">
            <DevicePicker />
            <MeetingControls />
          </div>

          <section className="main-window__meter">
            <label>Audio level</label>
            <AudioMeter />
          </section>

          {lastError && (
            <div className="main-window__error" role="alert">
              {lastError}
            </div>
          )}
        </section>

        {/* Right column: transcript */}
        <section className="main-window__right-col">
          <TranscriptPane />
        </section>
      </main>
    </div>
  );
}
