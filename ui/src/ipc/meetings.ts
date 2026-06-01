/**
 * Thin IPC client for the meeting-list + meeting-open surface (Phase 4, FR-33).
 *
 * Stream C adds the backend commands (`list_meetings`, `open_meeting`,
 * `rename_meeting`, `delete_meeting`, `re_transcribe`, `re_summarise`) and
 * regenerates `bindings.ts` BEFORE the frontend consumes them. Until that
 * regeneration lands, the generated `commands` surface does not carry these
 * methods, so this module reaches them through a typed cast over the shim-aware
 * `commands` from `./client` (NOT raw `./bindings`) — the same single injection
 * point `notes.ts` uses, so the DEV shim and the Vitest mocks both intercept
 * here.
 *
 * Tests mock THIS module (per `architecture/cross-cutting.md` — Automated
 * testing policy); they do not fake the generated bindings file.
 *
 * ---------------------------------------------------------------------------
 * Assumed backend command signatures (for Stream C to match) — all return
 * `Result<T, IpcError>` like every other `ipc-bridge` command:
 *
 *   list_meetings()                       -> MeetingListEntry[]
 *   open_meeting(meeting_id: MeetingId)   -> MeetingState
 *   rename_meeting(meeting_id, title)     -> ()      // new title string
 *   delete_meeting(meeting_id)            -> ()
 *   re_transcribe(meeting_id)             -> ()      // kicks off; emits
 *                                                    // AppEvent::TranscriptSegment
 *   re_summarise(meeting_id)              -> ()      // kicks off; emits
 *                                                    // AppEvent::SummaryReady
 *
 * Payload shapes (mirroring `crates/common/src/lib.rs`):
 *
 *   MeetingListEntry { id: MeetingId; title: string; started_at: string;
 *                      duration_ms: number; speaker_count: number;
 *                      excerpt?: string | null }
 *   MeetingState     { meta: MeetingMeta; transcript: Segment[];
 *                      notes?: NotesDoc | null }
 *
 * `MeetingListEntry` / `MeetingState` already exist in `common` with `specta`
 * derives, so Stream C only needs to expose the six commands; the generated
 * `bindings.ts` will then carry both types and these methods, and the cast in
 * this module collapses to a no-op.
 * ---------------------------------------------------------------------------
 */
import { commands, unwrap } from "./client";
import type { MeetingId, MeetingMeta, NotesDoc, Segment } from "./bindings";

/**
 * A meeting-list row (FR-33). Mirrors `common::MeetingListEntry`; restated here
 * because the generated binding for it lands with Stream C's regeneration.
 */
export type MeetingListEntry = {
  id: MeetingId;
  title: string;
  /** RFC3339 wall-clock start timestamp (mirrors `MeetingMeta.started_at`). */
  started_at: string;
  duration_ms: number;
  speaker_count: number;
  excerpt?: string | null;
};

/**
 * The full restorable state of a meeting, returned by `open_meeting`. Mirrors
 * `common::MeetingState`.
 */
export type MeetingState = {
  meta: MeetingMeta;
  transcript: Segment[];
  notes?: NotesDoc | null;
};

// The Phase-4 meeting commands are exposed on the shim-aware `commands` surface
// from `./client` (typed there as `PendingCommands` until Stream C regenerates
// `bindings.ts`). This module wraps them so callers work with plain
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

/** Re-run summarisation for a meeting (FR-33 action). */
export async function reSummarise(meetingId: MeetingId): Promise<void> {
  unwrap(await commands.reSummarise(meetingId));
}
