/**
 * Larger-ASR-model (GPU tier) settings helpers.
 *
 * `settings.prefer_large_asr_model` opts the Qwen branch into Qwen3-ASR-1.7B
 * (broader + better-multilingual accuracy) instead of the 0.6B CPU default. It
 * only affects languages that route to Qwen — the Parakeet branch (English + EU)
 * ignores it. Off by default (the 1.7B is a larger download with a GPU-class
 * footprint). First-class field on the generated `Settings` type, so — like the
 * GPU-acceleration helpers — these read/write the canonical field directly.
 * See `architecture/cross-cutting.md` — "ASR engine routing".
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the larger-ASR-model flag from a settings snapshot. Defaults to `false`
 * (the field absent, or the snapshot still loading) — matching the backend
 * `#[serde(default)]` of `false`.
 */
export function readPreferLargeAsrModel(settings: Settings | null): boolean {
  if (settings === null) return false;
  return settings.prefer_large_asr_model === true;
}

/**
 * Return a copy of `settings` with the larger-ASR-model flag set, preserving
 * every other field so the `update_settings` round-trip does not clobber the
 * rest of the store.
 */
export function withPreferLargeAsrModel(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, prefer_large_asr_model: enabled };
}
