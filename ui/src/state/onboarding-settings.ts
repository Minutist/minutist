/**
 * Onboarding settings helpers (Phase 7).
 *
 * The first-run onboarding gate is the `settings.onboarding_completed` field
 * (false by default; the webview shows the onboarding flow until it is true —
 * see `architecture/components.md`, the Editorial Ink "first-run / onboarding"
 * note and the `Settings` doc-comment). The field is owned by the `settings`
 * crate and is a first-class member of the generated `Settings` type, so —
 * exactly like the Phase-6 diarization helpers — these read/write the canonical
 * field directly with no augmentation shim.
 */
import type { Settings, Theme } from "../ipc/bindings";

/**
 * Read the onboarding-completed flag from a settings snapshot.
 *
 * Defaults to `false` (not completed → show onboarding) when the field is
 * absent (an older store written before it existed) or the snapshot is `null`
 * (settings not yet loaded). Treating a not-yet-loaded snapshot as "not
 * completed" would flash onboarding on every launch, so the gate distinguishes
 * the two cases via {@link isOnboardingResolved}; this helper answers only the
 * narrow "is the flag true" question.
 */
export function readOnboardingCompleted(settings: Settings | null): boolean {
  if (settings === null) return false;
  return settings.onboarding_completed === true;
}

/**
 * Whether the onboarding gate can be evaluated yet.
 *
 * `false` while the settings snapshot is still loading (`null`); the gate keeps
 * the app neutral (renders nothing app-specific) until this is `true` so it
 * never flashes onboarding to a returning user during the load round-trip.
 */
export function isOnboardingResolved(settings: Settings | null): boolean {
  return settings !== null;
}

/**
 * Return a copy of `settings` with the onboarding-completed flag set,
 * preserving every other field so the `update_settings` round-trip does not
 * clobber the rest of the store.
 */
export function withOnboardingCompleted(
  settings: Settings,
  completed: boolean,
): Settings {
  return { ...settings, onboarding_completed: completed };
}

/**
 * Read the colour-scheme preference from a settings snapshot, defaulting to
 * `"system"` (the schema default) when absent or the snapshot is `null`.
 */
export function readTheme(settings: Settings | null): Theme {
  if (settings === null) return "system";
  return settings.theme ?? "system";
}

/**
 * Return a copy of `settings` with the theme preference set, preserving every
 * other field for the `update_settings` round-trip.
 */
export function withTheme(settings: Settings, theme: Theme): Settings {
  return { ...settings, theme };
}
