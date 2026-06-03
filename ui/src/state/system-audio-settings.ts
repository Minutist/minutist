/**
 * System-audio (call / loopback) capture settings helpers.
 *
 * The toggle is the `settings.capture_system_audio` field (ON by default).
 * When on, `audio-capture` ALSO captures the system/render endpoint in loopback
 * mode and SUMS it with the microphone into the single stream the orchestrator
 * transcribes, so a Teams-style call captures all participants — not just the
 * user. When off, capture is mic-only.
 *
 * It defaults on because capturing the call is the point; it is opt-OUT. Turn it
 * off if it is echo-prone: if the microphone also picks the call audio up from
 * the speakers, mixing the loopback in doubles that audio. Loopback capture is
 * currently Windows-only; on other platforms the backend logs a warning and
 * falls back to mic-only.
 *
 * The field is owned by the `settings` crate and is a first-class member of the
 * generated `Settings` type, so — exactly like the diarization / GPU helpers —
 * these read/write the canonical field directly with no augmentation shim. See
 * `architecture/components.md` — the `settings` and `audio-capture` sections.
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the system-audio-capture flag from a settings snapshot.
 *
 * Defaults to `true` (on) when the field is absent (an older store written
 * before it existed) or the snapshot is `null` — matching the backend
 * `#[serde(default = ...)]` of `true`. Only an explicit `false` is off.
 */
export function readCaptureSystemAudio(settings: Settings | null): boolean {
  if (settings === null) return true;
  return settings.capture_system_audio !== false;
}

/**
 * Return a copy of `settings` with the system-audio-capture flag set,
 * preserving every other field so the `update_settings` round-trip does not
 * clobber the rest of the store.
 */
export function withCaptureSystemAudio(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, capture_system_audio: enabled };
}
