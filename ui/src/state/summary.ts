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

export type SummaryStore = {
  /** The persisted summary markdown for the active meeting, or `null` if none. */
  summaryMarkdown: string | null;
  /**
   * True while a `summarise_meeting` run is in flight — set when `summarise`
   * dispatches, cleared when the `summary_ready` event re-read completes (or on
   * error). Drives the in-progress affordance in the summary view.
   */
  summarising: boolean;
  /** The meeting whose summary is currently loaded, or `null`. */
  meetingId: MeetingId | null;
  /** Last error surfaced by a summary IPC call. */
  lastError: string | null;

  /** Load the persisted summary for a meeting (on open / view mount). */
  read: (meetingId: MeetingId) => Promise<void>;
  /** Kick off summarisation for a meeting; enters the in-progress state. */
  summarise: (meetingId: MeetingId) => Promise<void>;
  /** Persist an edited summary (FR-30). */
  save: (meetingId: MeetingId, summaryMarkdown: string) => Promise<void>;
  /** Clear the loaded summary (e.g. when returning to the meeting list). */
  clear: () => void;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export const useSummaryStore = create<SummaryStore>((set, get) => ({
  summaryMarkdown: null,
  summarising: false,
  meetingId: null,
  lastError: null,

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

  clear: () => {
    set({ summaryMarkdown: null, summarising: false, meetingId: null });
  },

  handleEvent: (event) => {
    if (event.kind !== "summary_ready") return;
    // A summary was produced for this meeting. Re-read `summary.md` so the
    // view shows the fresh content, and leave the in-progress state. Only act
    // when the event is for the meeting currently loaded (or none is loaded
    // yet) so an unrelated meeting's summary does not clobber the view.
    const current = get().meetingId;
    if (current !== null && current !== event.meeting_id) return;
    set({ meetingId: event.meeting_id });
    void (async () => {
      try {
        const markdown = await getSummary(event.meeting_id);
        set({ summaryMarkdown: markdown, summarising: false, lastError: null });
      } catch (err) {
        set({ summarising: false, lastError: errorMessage(err) });
      }
    })();
  },
}));
