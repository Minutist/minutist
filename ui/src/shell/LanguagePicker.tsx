import { useRecordingStore } from "../state/recording";
import { readTranscriptionLanguage } from "../state/transcription-language-settings";

/**
 * Supported ASR languages offered by the dropdown.
 *
 * A UI-side constant list, decoupled from the wire type (the Rust field is a
 * plain String). Each value other than "auto" is a full English language name
 * sent verbatim to the backend; "auto" is the sentinel the Rust resolver maps to
 * `None` (auto-detect, no prefix). Keep "Auto-detect" first.
 */
const LANGUAGES = [
  "English",
  "Chinese",
  "Spanish",
  "French",
  "German",
  "Italian",
  "Portuguese",
  "Japanese",
  "Korean",
  "Russian",
  "Dutch",
  "Arabic",
];

/**
 * Labelled `<select>` for choosing the ASR transcription-language hint.
 *
 * Mirrors {@link DevicePicker}: it reads the current value from the settings
 * snapshot and persists changes through `setTranscriptionLanguage` (which
 * round-trips via `update_settings`). Disabled until the snapshot has loaded so
 * the round-trip never clobbers settings with a partial object.
 */
export function LanguagePicker() {
  const settings = useRecordingStore((s) => s.settings);
  const setTranscriptionLanguage = useRecordingStore(
    (s) => s.setTranscriptionLanguage,
  );

  const current = readTranscriptionLanguage(settings);

  return (
    <div className="language-picker">
      <label htmlFor="language-select">Transcription language</label>
      <select
        id="language-select"
        value={current}
        disabled={settings === null}
        onChange={(e) => void setTranscriptionLanguage(e.target.value)}
      >
        <option value="auto">Auto-detect</option>
        {LANGUAGES.map((l) => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>
    </div>
  );
}
