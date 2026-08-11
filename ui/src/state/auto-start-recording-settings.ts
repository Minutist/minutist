/**
 * "New meeting" auto-start-recording setting helpers.
 *
 * The toggle is the `settings.auto_start_recording_on_new_meeting` field (OFF
 * by default). When off, creating a new meeting opens the "New meeting" prep
 * screen — set a title, write notes, attach resources — and recording starts
 * only when the user presses Start. When on, recording starts the instant a
 * new meeting is created (the legacy behaviour).
 *
 * The field is owned by the `settings` crate and is a first-class member of
 * the generated `Settings` type, so — exactly like the system-audio / GPU
 * helpers — this reads/writes the canonical field directly with no
 * augmentation shim. See `architecture/components.md` — the `settings` and
 * `orchestrator` sections.
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the auto-start-recording flag from a settings snapshot.
 *
 * Defaults to `false` (off — the prep screen) when the field is absent (an
 * older store written before it existed) or the snapshot is `null` —
 * matching the backend `#[serde(default = ...)]` of `false`. Only an
 * explicit `true` restores the legacy immediate-start behaviour.
 */
export function readAutoStartRecordingOnNewMeeting(
  settings: Settings | null,
): boolean {
  if (settings === null) return false;
  return settings.auto_start_recording_on_new_meeting === true;
}

/**
 * Return a copy of `settings` with the auto-start-recording flag set,
 * preserving every other field so the `update_settings` round-trip does not
 * clobber the rest of the store.
 */
export function withAutoStartRecordingOnNewMeeting(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, auto_start_recording_on_new_meeting: enabled };
}
