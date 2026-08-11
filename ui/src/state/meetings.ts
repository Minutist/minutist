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
  rejectMatch,
} from "../ipc/meetings";
import { setMeetingCollection } from "../ipc/collections";
import type { MeetingListEntry, MeetingState } from "../ipc/meetings";
import type {
  CollectionId,
  MeetingId,
  VoiceprintSuggestion,
} from "../ipc/bindings";
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
  /**
   * Pending uncertain-band voiceprint suggestions for the open meeting, keyed
   * by `label`. Set when `AppEvent::VoiceprintSuggestions` arrives; cleared
   * when the user confirms or dismisses each suggestion, or when the open
   * meeting changes.
   */
  voiceprintSuggestions: VoiceprintSuggestion[];
  /**
   * Dismiss a voiceprint suggestion (without accepting the name) and call
   * `reject_match` to drop that meeting/label contribution from the library.
   * Best-effort: errors are logged, not surfaced to the user.
   */
  dismissVoiceprintSuggestion: (
    suggestion: VoiceprintSuggestion,
  ) => Promise<void>;
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
  voiceprintSuggestions: [],

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
    set({ loading: true, voiceprintSuggestions: [] });
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
    set({ openMeetingId: null, openMeetingState: null, voiceprintSuggestions: [] });
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

  dismissVoiceprintSuggestion: async (suggestion) => {
    const { openMeetingId } = get();
    if (openMeetingId === null) return;
    // Drop the suggestion immediately so the UI clears without waiting for the
    // IPC round-trip.
    set((s) => ({
      voiceprintSuggestions: s.voiceprintSuggestions.filter(
        (x) => !(x.label === suggestion.label && x.identity_id === suggestion.identity_id),
      ),
    }));
    try {
      await rejectMatch(openMeetingId, suggestion.label, suggestion.identity_id, suggestion.model_id);
    } catch (err) {
      // Best-effort: log but do not surface to the user (the suggestion is
      // already gone from the UI; the backend may retry on next reprocess).
      console.error("rejectMatch failed", err);
    }
  },

  handleEvent: (event) => {
    if (event.kind === "meeting_finalised") {
      // A just-stopped meeting finished finalising on disk. Stay on it: OPEN the
      // meeting so the workspace shows the recording the user just made rather
      // than bouncing to the list. `MeetingFinalised` fires exactly once per stop
      // (background re-transcribe / diarize passes emit `transcript_ready` /
      // `diarization_complete`, never this), so it always targets the
      // just-recorded meeting. Set `openMeetingId` SYNCHRONOUSLY before the async
      // `open()` load so the workspace never flashes the list as the recorder
      // goes `Idle` (the orchestrator emits this BEFORE the `Idle` transition).
      // Background passes then update the open meeting in place (see the
      // transcript_ready / diarization_complete branch below).
      //
      // `openMeetingState` is cleared in the SAME set() — it still held the
      // PRE-recording draft snapshot (`meta.recording_started: false`) when
      // this meeting was promoted (promotion never re-reads it), and the
      // recorder goes `Idle` moments after this event. Left stale, that
      // combination reads as "an open, unstarted draft" to
      // `MeetingControls`/`MainWindow` for the duration of the async `open()`
      // below — a real press of "Start" or "Discard" in that window would
      // truncate or delete the meeting that was just recorded.
      set({ openMeetingId: event.meeting_id, openMeetingState: null });
      void get().open(event.meeting_id);
      // Refresh the list too, so the row (and its masthead entry) is present.
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
    if (event.kind === "voiceprint_suggestions") {
      // The matcher found uncertain-band hits for the open meeting. Only act if
      // the event targets the currently open meeting — an event for a background
      // meeting should not clobber the open view.
      if (get().openMeetingId === event.meeting_id) {
        set({ voiceprintSuggestions: event.suggestions });
      }
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
