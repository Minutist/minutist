/**
 * Thin IPC client for the meeting-summary surface (Phase 5, FR-30).
 *
 * Wraps the shim-aware `commands` from `./client` (NOT raw `./bindings` — A9),
 * the same single injection point `notes.ts` / `meetings.ts` use, so the DEV
 * shim and the Vitest mocks both intercept here. Tests mock THIS module (per
 * `architecture/cross-cutting.md` — Automated-testing policy); they do not fake
 * the generated bindings file.
 *
 * The three backend commands are added by the Phase-5 backend JOIN (Stream S5),
 * which regenerates `bindings.ts`. The assumed command signatures (the JOIN
 * must match these):
 *
 *   - `summarise_meeting(meeting_id) -> ()` — runs the summariser on the
 *     meeting's transcript + notes, writes `summary.md`, and emits
 *     `AppEvent::SummaryReady { meeting_id }`. Long-running (LLM inference on a
 *     `spawn_blocking` task); resolves once the run is dispatched/complete.
 *   - `get_summary(meeting_id) -> Option<String>` — reads `summary.md`, or
 *     `None` when no summary exists yet (`persistence::read_summary`).
 *   - `save_summary(meeting_id, summary_markdown) -> ()` — persists an edited
 *     summary back to `summary.md` (`persistence::write_summary`).
 *
 * `summariser` produces the file via `persistence`; `ipc-bridge` exposes the
 * commands. The opaque markdown crosses the wire as a `String`.
 */
import { commands, unwrap } from "./client";
import type { MeetingId } from "./bindings";

/**
 * Run summarisation for a meeting (FR-30).
 *
 * Resolves once the backend has produced `summary.md`; the
 * `AppEvent::SummaryReady` event is the authoritative "ready" signal the store
 * listens for (the resolved promise only confirms dispatch). Rejects on an IPC
 * error.
 */
export async function summariseMeeting(meetingId: MeetingId): Promise<void> {
  unwrap(await commands.summariseMeeting(meetingId));
}

/** Read the persisted summary markdown for a meeting, or `null` if none. */
export async function getSummary(
  meetingId: MeetingId,
): Promise<string | null> {
  return unwrap(await commands.getSummary(meetingId));
}

/** Persist an edited summary back to `summary.md` (FR-30). */
export async function saveSummary(
  meetingId: MeetingId,
  summaryMarkdown: string,
): Promise<void> {
  unwrap(await commands.saveSummary(meetingId, summaryMarkdown));
}
