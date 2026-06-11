import { useRecordingStore } from "../state/recording";
import { readOutputLanguage } from "../state/output-language-settings";

/**
 * Languages the output-language picker offers.
 *
 * Each value is a full English language name sent verbatim to the backend,
 * which appends a "Respond entirely in {lang}." instruction to the LLM system
 * prompt. The list mirrors the 15 languages covered by the Rust subtag-to-name
 * mapping in `ipc-bridge::output_language`. Keep alphabetical after Auto.
 * The transcript is never affected — this controls LLM output only.
 */
export const OUTPUT_LANGUAGES = [
  "Arabic",
  "Chinese",
  "Dutch",
  "English",
  "French",
  "German",
  "Hindi",
  "Italian",
  "Japanese",
  "Korean",
  "Polish",
  "Portuguese",
  "Russian",
  "Spanish",
  "Turkish",
];

/**
 * Labelled `<select>` for choosing the LLM output language.
 *
 * Mirrors {@link LanguagePicker}: it reads the current value from the settings
 * snapshot and persists changes through `setOutputLanguage` (which round-trips
 * via `update_settings`). Disabled until the snapshot has loaded so the
 * round-trip never clobbers settings with a partial object.
 *
 * "Auto (system)" is always the first option and maps to the wire sentinel
 * `"auto"`. The transcript is never affected.
 */
export function OutputLanguagePicker() {
  const settings = useRecordingStore((s) => s.settings);
  const setOutputLanguage = useRecordingStore((s) => s.setOutputLanguage);

  const current = readOutputLanguage(settings);

  return (
    <div className="language-picker">
      <label htmlFor="output-language-select">Output language</label>
      <select
        id="output-language-select"
        value={current}
        disabled={settings === null}
        onChange={(e) => void setOutputLanguage(e.target.value)}
      >
        <option value="auto">Auto (system)</option>
        {OUTPUT_LANGUAGES.map((l) => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>
    </div>
  );
}
