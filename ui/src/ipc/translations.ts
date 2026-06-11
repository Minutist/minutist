/**
 * Thin IPC seam for the translation commands.
 *
 * `translate_meeting` triggers a post-hoc translation pass that calls the
 * held LLM on each segment and persists the results in `translations.json`.
 * `get_translations` reads back the per-language map for a meeting.
 *
 * Tests mock THIS module, not the generated bindings file
 * (see `architecture/cross-cutting.md` — Automated testing policy).
 */
import { commands, unwrap } from "./client";
import type { MeetingId } from "./bindings";

/**
 * Trigger a translation pass for every segment of `meetingId`'s transcript.
 *
 * The backend validates `targetLanguage` against the 15-language supported
 * set, rejects concurrent calls for the same (meeting_id, language) pair,
 * and emits `OperationProgress { op: "translate" }` while the pass runs.
 * Resolves once all segments have been translated and
 * `AppEvent::TranslationReady` has been emitted by the backend.
 */
export async function translateMeeting(
  meetingId: MeetingId,
  targetLanguage: string,
): Promise<void> {
  unwrap(await commands.translateMeeting(meetingId, targetLanguage));
}

/**
 * Read the translations for one meeting + language combination.
 *
 * Returns a map from segment index (as a number key) to the translated text.
 * An empty map means no translations exist yet for that language.
 *
 * Note: JSON serialization converts number keys to strings on the wire, so
 * the actual runtime keys are strings even though the TypeScript type says
 * `Partial<{[key in number]: string}>`. The translation store normalises
 * this to a `Map<number, string>` before use.
 */
export async function getTranslations(
  meetingId: MeetingId,
  targetLanguage: string,
): Promise<Map<number, string>> {
  const raw = unwrap(
    await commands.getTranslations(meetingId, targetLanguage),
  );
  // `raw` arrives with string keys over the wire (JSON object keys are always
  // strings) even though the TypeScript type says `number` keys. Parse each
  // key to a number and drop any that are not valid non-negative integers.
  const result = new Map<number, string>();
  for (const [k, v] of Object.entries(raw as Record<string, string>)) {
    const idx = parseInt(k, 10);
    if (!Number.isNaN(idx) && idx >= 0) result.set(idx, v);
  }
  return result;
}
