/**
 * Diarization settings helpers.
 *
 * `settings.diarization_enabled` gates the orchestrator's live + on-stop
 * diarization passes and is owned by the `settings` crate, which defaults it
 * to `true` (see `default_diarization_enabled` in `crates/settings/src/lib.rs`).
 * These helpers read/write that field directly.
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the diarization-enabled flag from a settings snapshot.
 *
 * Defaults to `true` (on), matching the backend's `default_diarization_enabled`,
 * when the field is absent (an older store written before it existed) or the
 * snapshot is `null` (not yet loaded).
 */
export function readDiarizationEnabled(settings: Settings | null): boolean {
  if (settings === null) return true;
  return settings.diarization_enabled !== false;
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
