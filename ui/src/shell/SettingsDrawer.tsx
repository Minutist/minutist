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
import { readPreferLargeAsrModel } from "../state/large-asr-model-settings";
import { readCaptureSystemAudio } from "../state/system-audio-settings";
import { readNotesPaperRules } from "../state/notes-paper-settings";
import { readTheme } from "../state/onboarding-settings";
import type { Theme } from "../ipc/bindings";
import { DevicePicker } from "./DevicePicker";
import { LanguagePicker } from "./LanguagePicker";
import { McpSettingsPane } from "./McpSettingsPane";
import "./SettingsDrawer.css";

export type SettingsDrawerProps = {
  /** Whether the drawer is shown. */
  open: boolean;
  /** Called when the drawer should close (scrim click, Close button, Esc). */
  onClose: () => void;
  /** Open the About dialog (the affordance lives in this drawer's footer). */
  onAbout: () => void;
};

export function SettingsDrawer({ open, onClose, onAbout }: SettingsDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const settings = useRecordingStore((s) => s.settings);
  const setDiarizationEnabled = useRecordingStore(
    (s) => s.setDiarizationEnabled,
  );
  const setGpuAcceleration = useRecordingStore((s) => s.setGpuAcceleration);
  const setPreferLargeAsrModel = useRecordingStore(
    (s) => s.setPreferLargeAsrModel,
  );
  const setCaptureSystemAudio = useRecordingStore(
    (s) => s.setCaptureSystemAudio,
  );
  const setTheme = useRecordingStore((s) => s.setTheme);
  const setNotesPaperRules = useRecordingStore((s) => s.setNotesPaperRules);

  const diarizationEnabled = readDiarizationEnabled(settings);
  const gpuAcceleration = readGpuAcceleration(settings);
  const preferLargeAsrModel = readPreferLargeAsrModel(settings);
  const captureSystemAudio = readCaptureSystemAudio(settings);
  const theme = readTheme(settings);
  const notesPaperRules = readNotesPaperRules(settings);

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

        <section className="settings-drawer__group" aria-label="Appearance">
          <h3 className="settings-drawer__group-title">Appearance</h3>
          <div className="settings-drawer__field">
            <label htmlFor="settings-theme">Colour theme</label>
            <select
              id="settings-theme"
              value={theme}
              disabled={settings === null}
              onChange={(e) => void setTheme(e.target.value as Theme)}
            >
              <option value="system">Match system</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
          <label
            className="settings-drawer__toggle"
            title="Show faint horizontal writing-paper rules behind the notes. The oxblood margin rule that separates the timestamp gutter from the text is always shown."
          >
            <input
              type="checkbox"
              checked={notesPaperRules}
              disabled={settings === null}
              onChange={(e) => void setNotesPaperRules(e.target.checked)}
            />
            <span>Ruled writing paper</span>
          </label>
        </section>

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
          <label
            className="settings-drawer__toggle"
            title="Use the larger Qwen3-ASR-1.7B model for languages handled by Qwen (Chinese, Japanese, Korean, Arabic, …). A larger download, best with a GPU. English and European languages use Parakeet regardless of this setting."
          >
            <input
              type="checkbox"
              checked={preferLargeAsrModel}
              disabled={settings === null}
              onChange={(e) => void setPreferLargeAsrModel(e.target.checked)}
            />
            <span>Higher-accuracy speech model (GPU)</span>
          </label>
        </section>

        <McpSettingsPane />

        <footer className="settings-drawer__footer">
          <button
            type="button"
            className="settings-drawer__about"
            aria-haspopup="dialog"
            onClick={onAbout}
          >
            About meeting-app
          </button>
        </footer>
      </aside>
    </div>
  );
}
