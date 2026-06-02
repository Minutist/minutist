/**
 * Meetings-list store (FR-33).
 *
 * Holds the meeting index shown on the entry surface (the meeting-list view)
 * plus which meeting is currently open. All mutations route through the
 * `../ipc/meetings` seam (mocked in tests); the store keeps only transient UI
 * state, never a source of truth the backend owns.
 *
 * The list view is the entry surface BEFORE a meeting is open: `openMeetingId`
 * is `null` while the list is shown, and set to the opened meeting once
 * `open()` resolves. `MainWindow` switches between the list and the editor/
 * transcript workspace on this value.
 */
import { create } from "zustand";
import {
  listMeetings,
  openMeeting,
  renameMeeting,
  deleteMeeting,
  reTranscribe,
  rediarize,
} from "../ipc/meetings";
import type { MeetingListEntry, MeetingState } from "../ipc/meetings";
import type { MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type { MeetingListEntry, MeetingState };

export type MeetingsStore = {
  /** The meeting-list rows (FR-33). */
  meetings: MeetingListEntry[];
  /** True while a list/open/action IPC call is in flight. */
  loading: boolean;
  /** The id of the currently open meeting, or `null` while the list is shown. */
  openMeetingId: MeetingId | null;
  /** The restored state of the open meeting, or `null` when none is open. */
  openMeetingState: MeetingState | null;
  /** Last error surfaced by a meetings IPC call. */
  lastError: string | null;

  /** Refresh the meeting list (FR-33). */
  refresh: () => Promise<void>;
  /** Open a meeting; loads its full state and marks it open. */
  open: (meetingId: MeetingId) => Promise<void>;
  /** Return to the meeting list (close the open meeting). */
  close: () => void;
  /** Rename a meeting, then refresh the list so the new title shows. */
  rename: (meetingId: MeetingId, title: string) => Promise<void>;
  /** Delete a meeting, then refresh the list. */
  remove: (meetingId: MeetingId) => Promise<void>;
  /** Re-run transcription for a meeting. */
  reTranscribe: (meetingId: MeetingId) => Promise<void>;
  /** Re-run speaker diarization for a meeting (Phase 6). */
  rediarize: (meetingId: MeetingId) => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export const useMeetingsStore = create<MeetingsStore>((set, get) => ({
  meetings: [],
  loading: false,
  openMeetingId: null,
  openMeetingState: null,
  lastError: null,

  refresh: async () => {
    set({ loading: true });
    try {
      const meetings = await listMeetings();
      // Coerce defensively: the backend `list_meetings` command lands with
      // Stream C, and a raw invoke before then can resolve `undefined`. Keep the
      // store invariant that `meetings` is always an array so the view's
      // `meetings.length` read never throws.
      set({
        meetings: Array.isArray(meetings) ? meetings : [],
        loading: false,
        lastError: null,
      });
    } catch (err) {
      set({ loading: false, lastError: errorMessage(err) });
    }
  },

  open: async (meetingId) => {
    set({ loading: true });
    try {
      const state = await openMeeting(meetingId);
      set({
        openMeetingId: meetingId,
        openMeetingState: state,
        loading: false,
        lastError: null,
      });
    } catch (err) {
      set({ loading: false, lastError: errorMessage(err) });
    }
  },

  close: () => {
    set({ openMeetingId: null, openMeetingState: null });
  },

  rename: async (meetingId, title) => {
    try {
      await renameMeeting(meetingId, title);
      set({ lastError: null });
      await get().refresh();
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  remove: async (meetingId) => {
    try {
      await deleteMeeting(meetingId);
      set({ lastError: null });
      await get().refresh();
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  reTranscribe: async (meetingId) => {
    try {
      await reTranscribe(meetingId);
      set({ lastError: null });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  rediarize: async (meetingId) => {
    try {
      await rediarize(meetingId);
      set({ lastError: null });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  handleEvent: (event) => {
    if (event.kind !== "diarization_complete") return;
    // Diarization finished assigning speakers to THIS meeting's segments
    // (`transcript.json` was rewritten with the overlaid `speaker_id`s). Re-read
    // that meeting's transcript via `open_meeting` SCOPED TO THE EVENT'S
    // `meeting_id` — not the live recording store — so the restored
    // `openMeetingState.transcript` (the source the transcript pane reads for a
    // saved meeting, U1) reflects the new speaker tags. Only act when the event
    // is for the meeting currently open, so an unrelated meeting's event does
    // not clobber the open-meeting view (and a re-diarize triggered from the
    // list while no meeting is open quietly refreshes only the list).
    const eventMeetingId = event.meeting_id;
    if (get().openMeetingId !== eventMeetingId) {
      // Not the open meeting: refresh the list so the row's speaker count
      // reflects the new diarization, but do not touch the open-meeting state.
      void get().refresh();
      return;
    }
    void (async () => {
      try {
        const state = await openMeeting(eventMeetingId);
        // Guard against a race: only apply if THIS meeting is still the open
        // one once the async read resolves.
        if (get().openMeetingId === eventMeetingId) {
          set({ openMeetingState: state, lastError: null });
        }
      } catch (err) {
        set({ lastError: errorMessage(err) });
      }
    })();
    // Keep the list's speaker counts current too.
    void get().refresh();
  },
}));
