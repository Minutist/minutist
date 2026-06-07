/**
 * Settings drawer.
 *
 * A right-anchored slide-in panel that holds the rarely-changed capture /
 * inference configuration that previously lived inline in the top bar: input
 * device, transcription-language hint, diarize-on-stop, GPU acceleration, and
 * system-audio capture. Pulling these out of the masthead keeps the chrome a
 * quiet editorial bar (no overflow) while leaving each setting one click away.
 *
 * Presentational only: it adds no Tauri command. Each control routes through
 * the existing `useRecordingStore` seams (the same ones the inline controls
 * used), so the persistence behaviour — and its tests — are unchanged.
 *
 * Dialog affordances mirror {@link About}: focus moves to the close control on
 * open, Escape and a scrim click dismiss. Rendered in the Editorial Ink
 * language using `theme.css` tokens only.
 */
import { useEffect, useRef } from "react";
import { useRecordingStore } from "../state/recording";
import { readDiarizationEnabled } from "../state/diarization-settings";
import { readGpuAcceleration } from "../state/gpu-acceleration-settings";
import { readCaptureSystemAudio } from "../state/system-audio-settings";
import { DevicePicker } from "./DevicePicker";
import { LanguagePicker } from "./LanguagePicker";
import "./SettingsDrawer.css";

export type SettingsDrawerProps = {
  /** Whether the drawer is shown. */
  open: boolean;
  /** Called when the drawer should close (scrim click, Close button, Esc). */
  onClose: () => void;
};

export function SettingsDrawer({ open, onClose }: SettingsDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const settings = useRecordingStore((s) => s.settings);
  const setDiarizationEnabled = useRecordingStore(
    (s) => s.setDiarizationEnabled,
  );
  const setGpuAcceleration = useRecordingStore((s) => s.setGpuAcceleration);
  const setCaptureSystemAudio = useRecordingStore(
    (s) => s.setCaptureSystemAudio,
  );

  const diarizationEnabled = readDiarizationEnabled(settings);
  const gpuAcceleration = readGpuAcceleration(settings);
  const captureSystemAudio = readCaptureSystemAudio(settings);

  // Focus the close control on open, and close on Escape — the minimum dialog
  // affordances; mirrors the About dialog.
  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="settings-drawer-overlay" onClick={onClose}>
      <aside
        className="settings-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        // Keep clicks inside the drawer from bubbling to the dismiss handler.
        onClick={(e) => e.stopPropagation()}
      >
        <header className="settings-drawer__head">
          <h2 className="settings-drawer__title" id="settings-title">
            Settings
          </h2>
          <button
            ref={closeRef}
            type="button"
            className="settings-drawer__close"
            aria-label="Close settings"
            onClick={onClose}
          >
            Done
          </button>
        </header>

        <section className="settings-drawer__group" aria-label="Capture">
          <h3 className="settings-drawer__group-title">Capture</h3>
          <DevicePicker />
          <LanguagePicker />
          <label
            className="settings-drawer__toggle"
            title="Capture the call / system audio and mix it with your microphone so all participants are transcribed. Turn this off if your microphone also picks up the call from your speakers (echo)."
          >
            <input
              type="checkbox"
              checked={captureSystemAudio}
              disabled={settings === null}
              onChange={(e) => void setCaptureSystemAudio(e.target.checked)}
            />
            <span>Capture call / system audio</span>
          </label>
        </section>

        <section className="settings-drawer__group" aria-label="Processing">
          <h3 className="settings-drawer__group-title">Processing</h3>
          <label className="settings-drawer__toggle">
            <input
              type="checkbox"
              checked={diarizationEnabled}
              disabled={settings === null}
              onChange={(e) => void setDiarizationEnabled(e.target.checked)}
            />
            <span>Diarize speakers on stop</span>
          </label>
          <label className="settings-drawer__toggle">
            <input
              type="checkbox"
              checked={gpuAcceleration}
              disabled={settings === null}
              onChange={(e) => void setGpuAcceleration(e.target.checked)}
            />
            <span>GPU acceleration</span>
          </label>
        </section>
      </aside>
    </div>
  );
}
