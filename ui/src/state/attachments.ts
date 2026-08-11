/**
 * Attachments store — meeting reference-material attachments.
 *
 * Holds the attachment rows shown in the attachments pane: the manifest list for
 * the loaded meeting, a busy flag while a list/add is in flight, and the last
 * error. All mutations route through the `../ipc/attachments` seam (mocked in
 * tests); the store keeps only transient UI state — `attachments.json` on disk
 * is authoritative.
 *
 * The flow:
 *   - `read(id)` loads the persisted manifest (on open / pane mount).
 *   - `add(id, file, ext)` stores the original (`Pending`) and inserts the row;
 *     the backend's `attachment_added` event also carries the row, so the
 *     optimistic insert dedupes against it by id.
 *   - The background converter emits `attachment_converted` /
 *     `attachment_conversion_failed`; `handleEvent` flips the row's state.
 *   - `remove(id, attId)` drops the row; `attachment_removed` confirms it.
 *
 * Like `summary.ts`, event handling is gated on the loaded `meetingId` so a
 * backgrounded conversion for another meeting never clobbers the open pane.
 */
import { create } from "zustand";
import {
  addAttachment,
  listAttachments,
  openAttachment,
  removeAttachment,
} from "../ipc/attachments";
import type { AttachmentEntry, AttachmentId, MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";
import { errorMessage } from "../lib/errors";

export type AttachmentsStore = {
  /** The manifest rows for the loaded meeting, in manifest order. */
  attachments: AttachmentEntry[];
  /** The meeting whose attachments are currently loaded, or `null`. */
  meetingId: MeetingId | null;
  /** True while the initial list load is in flight (drives the loading state). */
  loading: boolean;
  /** Count of concurrent add calls in flight (drives the add affordance's busy state). */
  adding: number;
  /** Last error surfaced by an attachment IPC call. */
  lastError: string | null;

  /** Load the persisted manifest for a meeting (on open / pane mount). */
  read: (meetingId: MeetingId) => Promise<void>;
  /**
   * Add an attachment file; inserts the `Pending` row on success. Returns the
   * new manifest entry, or `null` on failure (the failure is also recorded on
   * `lastError`) — e.g. the notes editor's attachment-drop handler (#0038)
   * needs the entry's id/hash/ext to build its inline `AttachmentRef` node.
   */
  add: (meetingId: MeetingId, file: File, ext: string) => Promise<AttachmentEntry | null>;
  /** Open an attachment original in the OS default application. */
  open: (meetingId: MeetingId, entry: AttachmentEntry) => Promise<void>;
  /** Remove an attachment (optimistically drops the row). */
  remove: (meetingId: MeetingId, attachmentId: AttachmentId) => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

/** Replace the row with `id` via `patch`, leaving the rest (and order) intact. */
function patchRow(
  rows: AttachmentEntry[],
  id: AttachmentId,
  patch: (row: AttachmentEntry) => AttachmentEntry,
): AttachmentEntry[] {
  return rows.map((row) => (row.id === id ? patch(row) : row));
}

export const useAttachmentsStore = create<AttachmentsStore>((set, get) => ({
  attachments: [],
  meetingId: null,
  loading: false,
  adding: 0,
  lastError: null,

  read: async (meetingId) => {
    set({ meetingId, loading: true });
    try {
      const attachments = await listAttachments(meetingId);
      // Only commit if this is still the loaded meeting — a fast meeting switch
      // must not let a stale list overwrite the current one.
      if (get().meetingId !== meetingId) return;
      set({ attachments, loading: false, lastError: null });
    } catch (err) {
      if (get().meetingId !== meetingId) return;
      set({ loading: false, lastError: errorMessage(err) });
    }
  },

  add: async (meetingId, file, ext) => {
    set((s) => ({ adding: s.adding + 1, lastError: null }));
    try {
      const entry = await addAttachment(meetingId, file, ext);
      // Insert the new (Pending) row if it is not already present — the
      // `attachment_added` event may have raced ahead and inserted it.
      set((s) => {
        const next = s.adding - 1;
        if (s.meetingId !== meetingId) return { adding: next };
        const exists = s.attachments.some((a) => a.id === entry.id);
        return {
          adding: next,
          attachments: exists ? s.attachments : [...s.attachments, entry],
        };
      });
      return entry;
    } catch (err) {
      set((s) => ({ adding: s.adding - 1, lastError: errorMessage(err) }));
      return null;
    }
  },

  open: async (meetingId, entry) => {
    try {
      await openAttachment(meetingId, entry);
      set({ lastError: null });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  remove: async (meetingId, attachmentId) => {
    // Optimistically drop the row, capturing it first so a failed remove can
    // restore it (the store must not transiently misrepresent the manifest).
    const previous = get().attachments;
    set((s) => ({
      attachments: s.attachments.filter((a) => a.id !== attachmentId),
    }));
    try {
      await removeAttachment(meetingId, attachmentId);
      set({ lastError: null });
    } catch (err) {
      // Restore only if the meeting is still loaded and nothing else changed it.
      set((s) =>
        s.meetingId === meetingId
          ? { attachments: previous, lastError: errorMessage(err) }
          : { lastError: errorMessage(err) },
      );
    }
  },

  handleEvent: (event) => {
    switch (event.kind) {
      case "attachment_added": {
        // Insert the row for the loaded meeting if not already present (the
        // `add` action may have inserted it from the command return first).
        if (get().meetingId !== event.meeting_id) return;
        set((s) => {
          const exists = s.attachments.some((a) => a.id === event.attachment.id);
          return exists
            ? {}
            : { attachments: [...s.attachments, event.attachment] };
        });
        return;
      }
      case "attachment_converted": {
        // Flip the row to Ready. The converted markdown filename is not carried
        // on the event (it lives in the manifest), so re-read to pick it up; the
        // visible state change (Ready) is what the pane renders.
        if (get().meetingId !== event.meeting_id) return;
        set((s) => ({
          attachments: patchRow(s.attachments, event.attachment_id, (row) => ({
            ...row,
            conversion: { state: "ready" },
          })),
        }));
        void get().read(event.meeting_id);
        return;
      }
      case "attachment_conversion_failed": {
        if (get().meetingId !== event.meeting_id) return;
        set((s) => ({
          attachments: patchRow(s.attachments, event.attachment_id, (row) => ({
            ...row,
            conversion: { state: "failed", reason: event.reason },
          })),
        }));
        return;
      }
      case "attachment_removed": {
        if (get().meetingId !== event.meeting_id) return;
        set((s) => ({
          attachments: s.attachments.filter(
            (a) => a.id !== event.attachment_id,
          ),
        }));
        return;
      }
      default:
        return;
    }
  },
}));
