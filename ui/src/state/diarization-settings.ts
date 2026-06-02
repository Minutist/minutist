/**
 * Diarization settings helpers (Phase 6).
 *
 * The Phase-6 diarization toggle is the `settings.diarization_enabled` field
 * (off by default; gates the orchestrator's on-stop diarization pass — see
 * `architecture/components.md`, the `diarizer` Phase-6 section). The field is
 * owned by the `settings` crate and is now a first-class member of the generated
 * `Settings` type (the Phase-6 backend JOIN regenerated `bindings.ts`), so the
 * earlier `Settings`-augmentation shim has collapsed: these helpers read/write
 * the canonical field directly.
 */
import type { Settings } from "../ipc/bindings";

/**
 * `Settings` carrying the Phase-6 diarization toggle.
 *
 * Retained as a named alias of the canonical generated `Settings` (the field is
 * now part of it) so existing call sites and tests keep importing this type with
 * no change.
 */
export type SettingsWithDiarization = Settings;

/**
 * Read the diarization-enabled flag from a settings snapshot.
 *
 * Defaults to `false` (off) when the field is absent (an older store written
 * before it existed) or the snapshot is `null`.
 */
export function readDiarizationEnabled(settings: Settings | null): boolean {
  if (settings === null) return false;
  return settings.diarization_enabled === true;
}

/**
 * Return a copy of `settings` with the diarization-enabled flag set, preserving
 * every other field so the `update_settings` round-trip does not clobber the
 * rest of the store.
 */
export function withDiarizationEnabled(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, diarization_enabled: enabled };
}
