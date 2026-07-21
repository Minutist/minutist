/**
 * Thin IPC client for the meeting-list + meeting-open surface (Phase 4, FR-33).
 *
 * The backend commands (`list_meetings`, `open_meeting`, `rename_meeting`,
 * `delete_meeting`, `reprocess`) are present on the generated `commands`
 * surface (Stream C / the Phase-4 backend JOIN regenerated `bindings.ts`). This
 * module reaches them through the shim-aware `commands` from `./client` (NOT raw
 * `./bindings`) — the same single injection point `notes.ts` uses, so the DEV
 * shim and the Vitest mocks both intercept here.
 *
 * Re-summarisation now goes through the Phase-5 `../ipc/summary` seam
 * (`summarise_meeting`): the meeting-list row's Summarise action runs the real
 * summariser via the summary store, so the Phase-4 `re_summarise` stub (and its
 * `reSummarise` wrapper here) were removed.
 *
 * Tests mock THIS module (per `architecture/cross-cutting.md` — Automated
 * testing policy); they do not fake the generated bindings file.
 *
 * `MeetingListEntry` and `MeetingState` are re-exported from the generated
 * `./bindings` (the canonical `common::MeetingListEntry` / `common::MeetingState`
 * shapes) so call sites and the DEV shim keep importing them from this seam with
 * no change.
 */
import { commands, unwrap } from "./client";
import type { MeetingId } from "./bindings";
import type { MeetingListEntry, MeetingState } from "./bindings";

export type { MeetingListEntry, MeetingState };

// The Phase-4 meeting commands are exposed on the shim-aware `commands` surface
// from `./client`. This module wraps them so callers work with plain
// `MeetingListEntry[]` / `MeetingState` / `void` results and so tests mock THIS
// module rather than the generated bindings file.

/** List all meetings for the meeting-list view (FR-33). */
export async function listMeetings(): Promise<MeetingListEntry[]> {
  return unwrap(await commands.listMeetings());
}

/** Open a meeting, returning its full restorable state. */
export async function openMeeting(meetingId: MeetingId): Promise<MeetingState> {
  return unwrap(await commands.openMeeting(meetingId));
}

/** Rename a meeting. */
export async function renameMeeting(
  meetingId: MeetingId,
  title: string,
): Promise<void> {
  unwrap(await commands.renameMeeting(meetingId, title));
}

/**
 * Set a speaker's display name (maps a diarizer label such as `"A"` to a
 * name); an empty `name` clears it. Returns the updated label→name map.
 */
export async function setSpeakerName(
  meetingId: MeetingId,
  label: string,
  name: string,
): Promise<Partial<Record<string, string>>> {
  return unwrap(await commands.setSpeakerName(meetingId, label, name));
}

/** Delete a meeting and its on-disk folder. */
export async function deleteMeeting(meetingId: MeetingId): Promise<void> {
  unwrap(await commands.deleteMeeting(meetingId));
}

/**
 * Open a meeting's on-disk directory in the host OS file explorer (themed
 * context menus, #0034 — meeting-list "Open storage folder" entry). The
 * backend resolves the path server-side and hands it to the platform opener;
 * no filesystem path crosses the IPC boundary.
 */
export async function openMeetingFolder(meetingId: MeetingId): Promise<void> {
  unwrap(await commands.openMeetingFolder(meetingId));
}

/**
 * Reprocess a meeting offline (#0015): re-transcribe THEN re-diarize under one
 * offline claim. Merges the former `reTranscribe` (FR-33) + `rediarize` (FR-11)
 * seams into a single command.
 *
 * Tests mock THIS module (per `architecture/cross-cutting.md` — Automated
 * testing policy); they do not fake the generated bindings file.
 *
 * The backend (`Orchestrator::reprocess`) re-runs ASR over the complete
 * `audio.opus` (rewriting `transcript.json`), then runs the `SherpaDiarizer`
 * over the FRESH transcript and finalises once — updating `metadata.json`'s
 * `speaker_count` + `diarizer`, refreshing the index row, and emitting
 * `AppEvent::TranscriptSegment` + `AppEvent::DiarizationComplete` on the shared
 * bus. The diarize step clears `metadata.json`'s `speaker_names` (re-lettering
 * can change who each label is), so any user-assigned speaker names are reset.
 * Long-running (ASR + sherpa inference on `spawn_blocking` tasks); resolves once
 * the whole pass completes. Refuses unless the recorder is `Idle`.
 */
export async function reprocess(meetingId: MeetingId): Promise<void> {
  unwrap(await commands.reprocess(meetingId));
}

/**
 * Voiceprint correction path (#0003 §2.4): clear the uncertain-band
 * auto-assigned name for `label` on `meetingId` and remove that
 * meeting/label's embedding contribution from the identity's gallery so it
 * does not bias future identifications.
 */
export async function rejectMatch(
  meetingId: MeetingId,
  label: string,
  identityId: string,
  modelId: string,
): Promise<void> {
  unwrap(await commands.rejectMatch(meetingId, label, identityId, modelId));
}
