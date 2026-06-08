/**
 * Notes writing-paper rules settings helpers.
 *
 * `settings.notes_paper_rules` (on by default) governs whether the notes editor
 * paints faint horizontal "writing paper" rules behind the text. The oxblood
 * vertical margin rule that divides the timestamp gutter from the writing
 * column is structural and always shown — it is NOT governed by this flag. The
 * field is owned by the `settings` crate and is a first-class member of the
 * generated `Settings` type, so — exactly like the GPU-acceleration helpers —
 * these read/write the canonical field directly with no augmentation shim.
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the writing-paper-rules flag from a settings snapshot.
 *
 * Defaults to `true` (on) when the field is absent (an older store written
 * before it existed) — matching the backend `#[serde(default)]` of `true` — and
 * to `true` when the snapshot is `null` so the rules render while settings are
 * still loading rather than flashing off-then-on.
 */
export function readNotesPaperRules(settings: Settings | null): boolean {
  if (settings === null) return true;
  return settings.notes_paper_rules !== false;
}

/**
 * Return a copy of `settings` with the writing-paper-rules flag set, preserving
 * every other field so the `update_settings` round-trip does not clobber the
 * rest of the store.
 */
export function withNotesPaperRules(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, notes_paper_rules: enabled };
}
