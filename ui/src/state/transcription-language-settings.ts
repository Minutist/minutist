/**
 * ASR transcription-language settings helpers.
 *
 * The hint is the `settings.transcription_language` field (a full English
 * language name, e.g. "English", "Spanish", "Japanese", or the sentinel "auto").
 * A real name forces that language via the Qwen3-ASR assistant-turn prefix; the
 * sentinel "auto" disables forcing (auto-detect, the pre-feature behaviour). It
 * defaults to "English" (fixes the spurious-Chinese auto-detect bug).
 *
 * The field is owned/validated by the `settings` + `asr-runtime` crates (a
 * String, not an enum, so the supported language list lives outside the wire
 * type — see the settings design). These helpers read/write the canonical field
 * directly, cloning the `readTheme`/`withTheme` string pattern (not the boolean
 * helpers). The dropdown's option set is a UI-side constant, decoupled from the
 * wire type. See `architecture/components.md` — the `settings` and `asr-runtime`
 * sections.
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the transcription-language hint from a settings snapshot.
 *
 * Defaults to "English" (the schema default) when the field is absent (an older
 * store written before it existed) or the snapshot is `null` (settings not yet
 * loaded).
 */
export function readTranscriptionLanguage(settings: Settings | null): string {
  if (settings === null) return "English"; // schema default
  return settings.transcription_language ?? "English";
}

/**
 * Return a copy of `settings` with the transcription-language hint set,
 * preserving every other field so the `update_settings` round-trip does not
 * clobber the rest of the store.
 */
export function withTranscriptionLanguage(
  settings: Settings,
  language: string,
): Settings {
  return { ...settings, transcription_language: language };
}
