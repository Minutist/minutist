/**
 * Summary-prompt settings helpers (Phase 9 — D4).
 *
 * Two related fields drive the effective summary prompt:
 *
 *   - `settings.summary_preset` — a `SummaryPreset` enum selecting a built-in
 *     prompt (the four presets). Defaults to `"default"` (the prior behaviour).
 *   - `settings.summary_system_prompt` — an OPTIONAL custom override. Empty (the
 *     common case) means "use the selected preset"; a non-empty value wins over
 *     the preset (the backend's `Settings::effective_summary_prompt`).
 *
 * Both fields are owned by the `settings` crate and are first-class members of
 * the generated `Settings` type (the Phase-9 backend JOIN regenerated
 * `bindings.ts`). These helpers read/write the canonical fields directly,
 * cloning the `withTranscriptionLanguage` string/`withDiarizationEnabled` value
 * patterns. The preset dropdown's human labels are a UI-side constant, decoupled
 * from the wire enum. See `architecture/components.md` — the `settings` section.
 */
import type { Settings, SummaryPreset } from "../ipc/bindings";

/** Human label for each summary preset, shown in the picker. UI-side only. */
export const SUMMARY_PRESET_LABELS: Record<SummaryPreset, string> = {
  default: "Structured (Summary / Decisions / Actions)",
  filter_chit_chat: "Structured, skip chit-chat",
  action_items: "Action items + owners",
  detailed: "Detailed, sectioned",
};

/** The presets in display order (mirrors the wire enum order). */
export const SUMMARY_PRESETS: SummaryPreset[] = [
  "default",
  "filter_chit_chat",
  "action_items",
  "detailed",
];

/**
 * Read the selected summary preset from a settings snapshot.
 *
 * Defaults to `"default"` (the schema default) when the field is absent (an
 * older store written before it existed) or the snapshot is `null`.
 */
export function readSummaryPreset(settings: Settings | null): SummaryPreset {
  if (settings === null) return "default";
  return settings.summary_preset ?? "default";
}

/**
 * Read the custom summary-prompt override from a settings snapshot.
 *
 * Defaults to the empty string (the common case — the preset drives) when the
 * field is absent or the snapshot is `null`.
 */
export function readSummarySystemPrompt(settings: Settings | null): string {
  if (settings === null) return "";
  return settings.summary_system_prompt ?? "";
}

/**
 * Return a copy of `settings` with the summary preset set, preserving every
 * other field so the `update_settings` round-trip does not clobber the rest of
 * the store.
 */
export function withSummaryPreset(
  settings: Settings,
  preset: SummaryPreset,
): Settings {
  return { ...settings, summary_preset: preset };
}

/**
 * Return a copy of `settings` with the custom summary-prompt override set,
 * preserving every other field. An empty string means "use the selected
 * preset".
 */
export function withSummarySystemPrompt(
  settings: Settings,
  prompt: string,
): Settings {
  return { ...settings, summary_system_prompt: prompt };
}
