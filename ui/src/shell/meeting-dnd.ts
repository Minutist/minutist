/**
 * Drag-and-drop seam for filing a meeting into a folder by dragging its row from
 * the meeting list onto a sidebar folder. A dedicated MIME keeps it distinct
 * from the transcript-segment drag (`editor/transcript-dnd.ts`), so a meeting
 * drag is only ever accepted by a folder drop target.
 */
import type { MeetingId } from "../ipc/bindings";

/** MIME identifying a dragged meeting id on the DataTransfer. */
export const MEETING_DND_MIME = "application/x-minutist-meeting-id";

/** Mark a drag as carrying `meetingId` (called from a meeting row's dragstart). */
export function writeMeetingDrag(dt: DataTransfer, meetingId: MeetingId): void {
  dt.setData(MEETING_DND_MIME, meetingId);
  dt.effectAllowed = "move";
}

/**
 * Whether the in-flight drag is a meeting (so a folder can accept it). Uses
 * `types` because `getData` is blocked during `dragover` for security; the id
 * itself is only readable on `drop`.
 */
export function hasMeetingDrag(dt: DataTransfer): boolean {
  return Array.from(dt.types).includes(MEETING_DND_MIME);
}

/** Read the dragged meeting id on drop, or `null` if this isn't a meeting drag. */
export function readMeetingDrag(dt: DataTransfer): MeetingId | null {
  const id = dt.getData(MEETING_DND_MIME);
  return id ? (id as MeetingId) : null;
}
