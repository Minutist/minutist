/**
 * Thin IPC client for the meeting-list + meeting-open surface (Phase 4, FR-33).
 *
 * The backend commands (`list_meetings`, `open_meeting`, `rename_meeting`,
 * `delete_meeting`, `re_transcribe`) are present on the generated `commands`
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
import { commands, unwrap, callPendingCommand } from "./client";
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

/** Delete a meeting and its on-disk folder. */
export async function deleteMeeting(meetingId: MeetingId): Promise<void> {
  unwrap(await commands.deleteMeeting(meetingId));
}

/** Re-run transcription for a meeting (FR-33 action). */
export async function reTranscribe(meetingId: MeetingId): Promise<void> {
  unwrap(await commands.reTranscribe(meetingId));
}

/**
 * Re-run speaker diarization for a meeting (Phase 6).
 *
 * The backend command (`rediarize_meeting`) is NOT yet on the generated
 * `commands` surface — the diarizer/orchestrator/ipc-bridge JOIN (Stream S5)
 * adds it and regenerates `bindings.ts`. Until then this routes through the
 * shim-aware `callPendingCommand` raw-`invoke` path (the same approach the
 * Phase-4 meeting commands and the Phase-5 summary commands used before their
 * bindings regenerated — A9: do NOT hand-write a bindings stub). Once
 * regenerated it folds into `commands.rediarizeMeeting` like every other
 * command.
 *
 * Tests mock THIS module (per `architecture/cross-cutting.md` — Automated
 * testing policy); they do not fake the generated bindings file.
 *
 * Assumed signature the JOIN (Stream S5) MUST match:
 *
 *   `rediarize_meeting(meeting_id: MeetingId) -> ()`
 *
 *   Snake-case command name `rediarize_meeting`; tauri-specta generates the
 *   camelCase `rediarizeMeeting` method. It decodes the meeting's
 *   pause-INCLUDING PCM (`persistence::reader::read_audio_pcm`), runs the
 *   `SherpaDiarizer` over the stored transcript segments (resolving both model
 *   dirs via `model-registry`), rewrites `transcript.json`
 *   (`persistence::write_transcript`) with the overlaid `speaker_id`s, refreshes
 *   the index row's `speaker_count`, and emits
 *   `AppEvent::DiarizationComplete { meeting_id, speaker_count }` on the shared
 *   bus. Long-running (sherpa inference on a `spawn_blocking` task); resolves
 *   once the run is dispatched/complete. Like `re_transcribe`, it refuses unless
 *   the recorder is `Idle`.
 */
export async function rediarize(meetingId: MeetingId): Promise<void> {
  unwrap(await callPendingCommand<null>("rediarize_meeting", [meetingId]));
}
