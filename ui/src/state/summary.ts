/**
 * Summary store (Phase 5, FR-30).
 *
 * Holds the meeting summary shown/edited in the summary view: the persisted
 * markdown, the in-progress flag while summarisation runs, and the last error.
 * All mutations route through the `../ipc/summary` seam (mocked in tests); the
 * store keeps only transient UI state, never a source of truth the backend owns
 * (`summary.md` on disk is authoritative).
 *
 * The flow:
 *   - `summarise(id)` calls `summarise_meeting`, entering the in-progress state.
 *   - The backend emits `AppEvent::SummaryReady { meeting_id }` when `summary.md`
 *     is written; `handleEvent` re-reads it via `get_summary` and leaves the
 *     in-progress state.
 *   - `read(id)` loads the persisted summary (on open / mount).
 *   - `save(id, markdown)` persists an edited summary (`save_summary`).
 */
import { create } from "zustand";
import { summariseMeeting, getSummary, saveSummary } from "../ipc/summary";
import type { MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";
import { errorMessage } from "../lib/errors";

export type SummaryStore = {
  /** The persisted summary markdown for the active meeting, or `null` if none. */
  summaryMarkdown: string | null;
  /**
   * True while a `summarise_meeting` run is in flight — set when `summarise`
   * dispatches, cleared when the `summary_ready` event re-read completes (or on
   * error). Drives the in-progress affordance in the summary view.
   */
  summarising: boolean;
  /**
   * Meetings with a post-stop AUTOMATIC summary queued or running (keyed by id).
   * Set by the `summary_queued` event the backend emits at stop — BEFORE any
   * transcript-repair pass the summary waits for — and cleared by the terminal
   * `summary_ready` (a summary was written) or `summary_unavailable` (deferred /
   * failed). It makes the summary pane show a busy state for the whole queued →
   * summarising window, not just once the determinate `summarise` op streams
   * (which can be minutes later, after re-transcribe / re-diarize). Per-meeting,
   * since a backgrounded auto-summary may be running for a meeting other than the
   * one currently open. This is transient UI state, never a source of truth.
   */
  autoPending: Record<MeetingId, boolean>;
  /** The meeting whose summary is currently loaded, or `null`. */
  meetingId: MeetingId | null;
  /** Last error surfaced by a summary IPC call. */
  lastError: string | null;

  /**
   * In-progress raw-markdown edit, held in the store (not in the SummaryView's
   * local state) so it survives the summary pane being hidden/unmounted — the
   * pane is conditionally rendered, so a mount-local draft would be silently
   * lost when the user toggles the column off. `editing` is the edit-mode flag,
   * `editDraft` the working text, and `editMeetingId` scopes the draft to its
   * meeting (a draft for meeting A is not shown when meeting B is open).
   */
  editing: boolean;
  editDraft: string;
  editMeetingId: MeetingId | null;

  /** Load the persisted summary for a meeting (on open / view mount). */
  read: (meetingId: MeetingId) => Promise<void>;
  /** Kick off summarisation for a meeting; enters the in-progress state. */
  summarise: (meetingId: MeetingId) => Promise<void>;
  /** Persist an edited summary (FR-30). */
  save: (meetingId: MeetingId, summaryMarkdown: string) => Promise<void>;
  /** Enter edit mode for a meeting, seeding the draft (scoped to that meeting). */
  beginEdit: (meetingId: MeetingId, initial: string) => void;
  /** Update the in-progress draft text. */
  setDraft: (text: string) => void;
  /** Leave edit mode, discarding the draft (used on cancel and after save). */
  endEdit: () => void;
  /** Clear the loaded summary (e.g. when returning to the meeting list). */
  clear: () => void;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

/** Drop a meeting's entry from the auto-pending map (immutably, no-op if absent). */
function withoutPending(
  pending: Record<MeetingId, boolean>,
  meetingId: MeetingId,
): Record<MeetingId, boolean> {
  if (!(meetingId in pending)) return pending;
  const next = { ...pending };
  delete next[meetingId];
  return next;
}

export const useSummaryStore = create<SummaryStore>((set, get) => ({
  summaryMarkdown: null,
  summarising: false,
  autoPending: {},
  meetingId: null,
  lastError: null,
  editing: false,
  editDraft: "",
  editMeetingId: null,

  read: async (meetingId) => {
    set({ meetingId });
    try {
      const markdown = await getSummary(meetingId);
      set({ summaryMarkdown: markdown, lastError: null });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  summarise: async (meetingId) => {
    set({ summarising: true, meetingId, lastError: null });
    try {
      await summariseMeeting(meetingId);
    } catch (err) {
      // Dispatch failed — leave the in-progress state and surface the error.
      // (On success, `summary_ready` clears `summarising` after the re-read.)
      set({ summarising: false, lastError: errorMessage(err) });
    }
  },

  save: async (meetingId, summaryMarkdown) => {
    // Optimistically reflect the edit so the rendered view matches the editor,
    // capturing the prior value first so a failed persist can roll it back —
    // the store must not transiently misrepresent what is persisted on disk.
    const previousMarkdown = get().summaryMarkdown;
    set({ summaryMarkdown, meetingId });
    try {
      await saveSummary(meetingId, summaryMarkdown);
      set({ lastError: null });
    } catch (err) {
      // Persist failed — roll back the optimistic edit and surface the error.
      set({ summaryMarkdown: previousMarkdown, lastError: errorMessage(err) });
    }
  },

  beginEdit: (meetingId, initial) => {
    set({ editing: true, editDraft: initial, editMeetingId: meetingId });
  },

  setDraft: (text) => {
    set({ editDraft: text });
  },

  endEdit: () => {
    set({ editing: false, editDraft: "", editMeetingId: null });
  },

  clear: () => {
    set({
      summaryMarkdown: null,
      summarising: false,
      meetingId: null,
      editing: false,
      editDraft: "",
      editMeetingId: null,
    });
  },

  handleEvent: (event) => {
    switch (event.kind) {
      case "summary_queued": {
        // A post-stop auto-summary was scheduled — mark this meeting busy so the
        // pane shows progress, not the manual Summarise button, for the whole
        // (possibly minutes-long) queued → summarising window.
        const meetingId = event.meeting_id;
        set((s) => ({ autoPending: { ...s.autoPending, [meetingId]: true } }));
        return;
      }
      case "summary_unavailable": {
        // The auto-summary was deferred (a new recording started) or failed:
        // clear the busy marker so the pane falls back to the manual action.
        // Per-meeting + ungated on the loaded meeting (a backgrounded one must
        // clear even when another is open).
        set((s) => ({ autoPending: withoutPending(s.autoPending, event.meeting_id) }));
        return;
      }
      case "summary_ready": {
        // A summary was written. Clear the busy marker for THIS meeting first,
        // regardless of which meeting is open, so a backgrounded auto-summary
        // never leaves a stale spinner on its pane.
        set((s) => ({ autoPending: withoutPending(s.autoPending, event.meeting_id) }));
        // Re-read `summary.md` into the view only when the event is for the
        // loaded meeting (or none is loaded yet), so an unrelated meeting's
        // summary does not clobber the view, and leave the in-progress state.
        const current = get().meetingId;
        if (current !== null && current !== event.meeting_id) return;
        set({ meetingId: event.meeting_id });
        void (async () => {
          try {
            const markdown = await getSummary(event.meeting_id);
            set({
              summaryMarkdown: markdown,
              summarising: false,
              lastError: null,
            });
          } catch (err) {
            set({ summarising: false, lastError: errorMessage(err) });
          }
        })();
        return;
      }
      default:
        return;
    }
  },
}));
