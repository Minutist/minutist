/**
 * Output-language settings helpers.
 *
 * The output language is the `settings.output_language` field (a full English
 * language name, e.g. "French", "German", or the sentinel "auto"). "auto"
 * resolves to the host system locale at generation time on the Rust side;
 * any other value forces that language for all LLM-generated text (summaries
 * and chat replies). The transcript is never affected.
 *
 * The field is owned/validated by the `settings` crate (a String, not an
 * enum, so the supported language list lives outside the wire type). These
 * helpers read/write the canonical field directly, following the
 * `readTheme`/`withTheme` string pattern. The dropdown's option set is a
 * UI-side constant, decoupled from the wire type.
 * See `architecture/components.md` — the `settings` and `ipc-bridge` sections.
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the output-language setting from a settings snapshot.
 *
 * Defaults to "auto" (the schema default) when the field is absent (an older
 * store written before it existed) or the snapshot is `null` (settings not yet
 * loaded).
 */
export function readOutputLanguage(settings: Settings | null): string {
  if (settings === null) return "auto"; // schema default
  return settings.output_language ?? "auto";
}

/**
 * Return a copy of `settings` with the output-language setting applied,
 * preserving every other field so the `update_settings` round-trip does not
 * clobber the rest of the store.
 */
export function withOutputLanguage(
  settings: Settings,
  language: string,
): Settings {
  return { ...settings, output_language: language };
}
