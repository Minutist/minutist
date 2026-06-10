/**
 * Preload-summariser settings helpers.
 *
 * `settings.preload_summariser` (on by default) controls whether the shared
 * summary/chat LLM is warmed at app startup and kept resident, so the first
 * Summarise / chat is instant instead of paying a multi-GB load. When off, the
 * model loads on-demand on first use. The flag is a first-class member of the
 * generated `Settings` type, so — like the GPU-acceleration helpers — these
 * read/write the canonical field directly with no augmentation shim.
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the preload flag from a settings snapshot.
 *
 * Defaults to `true` (on) when the field is absent (an older store written
 * before it existed) — matching the backend `#[serde(default)]` of `true` — and
 * to `true` when the snapshot is `null` so the checkbox renders checked while
 * settings are still loading.
 */
export function readPreloadSummariser(settings: Settings | null): boolean {
  if (settings === null) return true;
  return settings.preload_summariser !== false;
}

/**
 * Return a copy of `settings` with the preload flag set, preserving every other
 * field so the `update_settings` round-trip does not clobber the rest of the
 * store.
 */
export function withPreloadSummariser(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, preload_summariser: enabled };
}
