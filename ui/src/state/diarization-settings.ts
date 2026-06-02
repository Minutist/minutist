/**
 * Diarization settings helpers (Phase 6).
 *
 * The Phase-6 diarization toggle is a new `settings.diarization_enabled` field
 * (off by default; gates the orchestrator's on-stop diarization pass — see
 * `architecture/components.md`, the `diarizer` Phase-6 section). The field is
 * owned by the `settings` crate; once it lands there, tauri-specta regenerates
 * the `Settings` type in `bindings.ts` to carry it. Until the backend JOIN
 * (Stream S5) regenerates the bindings, the generated `Settings` type does not
 * yet declare the field, and the webview cannot hand-edit the generated bindings
 * (A9).
 *
 * This module bridges the gap WITHOUT editing `bindings.ts`: it models the field
 * as an optional augmentation of the generated `Settings` type and reads/writes
 * it as an extra JSON property. The `update_settings` round-trip carries the
 * property losslessly (serde on the backend ignores unknown fields today and
 * will deserialise the field once it exists), so the toggle persists across an
 * app restart exactly like the device selection. When the field lands in the
 * generated type, this augmentation collapses into the canonical `Settings`
 * shape with no call-site change.
 */
import type { Settings } from "../ipc/bindings";

/** `Settings` augmented with the Phase-6 diarization toggle. */
export type SettingsWithDiarization = Settings & {
  /**
   * Whether the orchestrator runs speaker diarization on stop (Phase 6).
   * Off by default; the user opts in. Re-diarize is always available
   * independently of this flag (it is an explicit user action).
   */
  diarization_enabled?: boolean | null;
};

/**
 * Read the diarization-enabled flag from a settings snapshot.
 *
 * Defaults to `false` (off) when the field is absent — both before the backend
 * field lands and for an older store written without it.
 */
export function readDiarizationEnabled(settings: Settings | null): boolean {
  if (settings === null) return false;
  return (settings as SettingsWithDiarization).diarization_enabled === true;
}

/**
 * Return a copy of `settings` with the diarization-enabled flag set, preserving
 * every other field so the `update_settings` round-trip does not clobber the
 * rest of the store.
 */
export function withDiarizationEnabled(
  settings: Settings,
  enabled: boolean,
): SettingsWithDiarization {
  return { ...settings, diarization_enabled: enabled };
}
