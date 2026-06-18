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
  setSpeakerName,
  deleteMeeting,
  reprocess,
} from "../ipc/meetings";
import { setMeetingCollection } from "../ipc/collections";
import type { MeetingListEntry, MeetingState } from "../ipc/meetings";
import type { CollectionId, MeetingId } from "../ipc/bindings";
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
  /**
   * The meeting currently being reprocessed, set the instant `reprocess` is
   * called — BEFORE the backend's first `operation_progress` event lands (which
   * lags the click by seconds: claim + audio decode + first ASR flush). Drives
   * immediate button feedback and guards against a double-press re-claiming the
   * offline slot. `null` when no reprocess is in flight.
   */
  reprocessingId: MeetingId | null;
  /** Wall-clock ms when the in-flight reprocess started (for elapsed / ETA). */
  reprocessStartedMs: number | null;

  /** Refresh the meeting list (FR-33). */
  refresh: () => Promise<void>;
  /** Open a meeting; loads its full state and marks it open. */
  open: (meetingId: MeetingId) => Promise<void>;
  /** Return to the meeting list (close the open meeting). */
  close: () => void;
  /** Rename a meeting, then refresh the list so the new title shows. */
  rename: (meetingId: MeetingId, title: string) => Promise<void>;
  /**
   * Set (or clear, with an empty name) a speaker's display name on the open
   * meeting, updating its restored `meta.speaker_names` in place so the
   * transcript re-renders without a reload.
   */
  setSpeakerName: (label: string, name: string) => Promise<void>;
  /** Delete a meeting, then refresh the list. */
  remove: (meetingId: MeetingId) => Promise<void>;
  /**
   * File a meeting into a folder (or unfile it with `null`), then refresh the
   * list so the row's `collection_id` (and the sidebar counts derived from it)
   * update.
   */
  setCollection: (
    meetingId: MeetingId,
    collectionId: CollectionId | null,
  ) => Promise<void>;
  /**
   * Reprocess a meeting (#0015): re-transcribe THEN re-diarize under one offline
   * claim. Resets any user-assigned speaker names (the diarize step re-letters).
   */
  reprocess: (meetingId: MeetingId) => Promise<void>;
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
  reprocessingId: null,
  reprocessStartedMs: null,

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

  setSpeakerName: async (label, name) => {
    const { openMeetingId, openMeetingState } = get();
    if (openMeetingId === null || openMeetingState === null) return;
    try {
      const speaker_names = await setSpeakerName(openMeetingId, label, name);
      set({
        openMeetingState: {
          ...openMeetingState,
          meta: { ...openMeetingState.meta, speaker_names },
        },
        lastError: null,
      });
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

  setCollection: async (meetingId, collectionId) => {
    try {
      await setMeetingCollection(meetingId, collectionId);
      set({ lastError: null });
      await get().refresh();
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  reprocess: async (meetingId) => {
    // Ignore a re-entrant press: the backend allows only one offline op, and
    // the button is disabled the instant `reprocessingId` is set below, but
    // guard here too so a fast double-click can't fire two calls before React
    // re-renders.
    if (get().reprocessingId !== null) return;
    // Set the local in-flight flag NOW (before awaiting) so the button greys to
    // "Reprocessing…" immediately rather than waiting seconds for the backend's
    // first progress event.
    set({
      reprocessingId: meetingId,
      reprocessStartedMs: Date.now(),
      lastError: null,
    });
    try {
      await reprocess(meetingId);
    } catch (err) {
      set({ lastError: errorMessage(err) });
    } finally {
      // The IPC promise resolves only when the whole pass (re-transcribe →
      // diarize) finishes, so the flag spans the entire operation.
      set({ reprocessingId: null, reprocessStartedMs: null });
    }
  },

  handleEvent: (event) => {
    if (event.kind === "meeting_finalised") {
      // A just-stopped meeting finished finalising on disk (the background
      // drain/finalise completed). Refresh the list so the new row appears —
      // `list_meetings` reconciles it from disk if the stop-time upsert has not
      // landed yet. This is what makes the meeting show without a manual
      // refresh now that finalise is decoupled from the stop response.
      void get().refresh();
      return;
    }
    if (event.kind === "summary_ready") {
      // Live-test UX T6: a summary was written, so the meeting-list excerpt is
      // now the summary blurb (the backend refreshed the index row). Refresh the
      // list so the row shows the blurb without a manual reload. (The summary
      // PANE re-read is the summary store's job; this only touches the list.)
      void get().refresh();
      return;
    }
    if (event.kind !== "diarization_complete" && event.kind !== "transcript_ready")
      return;
    // A background/offline pass rewrote THIS meeting's `transcript.json`:
    // diarization overlaid `speaker_id`s (`diarization_complete`), or a
    // re-transcribe repaired the text (`transcript_ready`). Re-read that
    // meeting's transcript via `open_meeting` SCOPED TO THE EVENT'S `meeting_id`
    // — not the live recording store — so the restored `openMeetingState.transcript`
    // (the source the transcript pane reads for a saved meeting, U1) reflects the
    // new content. Only act when the event is for the meeting currently open, so
    // an unrelated meeting's event does not clobber the open-meeting view (a pass
    // triggered while no meeting is open quietly refreshes only the list).
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
