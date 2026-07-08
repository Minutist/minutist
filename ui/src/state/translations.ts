/**
 * Per-meeting translation store.
 *
 * Holds the active translated-view state: which language (if any) the user
 * has chosen to view, the fetched segment translations for the open meeting
 * + language, and the in-flight flag that disables the Translate button while
 * the backend pass runs.
 *
 * All mutations route through `../ipc/translations` (mocked in tests).
 * The store keeps only transient UI state — `translations.json` is the
 * persistent truth owned by the backend.
 *
 * The translated view is opt-in and per-session: it does not persist across
 * meetings or app restarts.
 */
import { create } from "zustand";
import { translateMeeting, getTranslations } from "../ipc/translations";
import type { MeetingId, Segment } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";
import { activeTranscript } from "./active-transcript";

export type TranslationsStore = {
  /**
   * The language the user has selected to translate into, or `null` for the
   * verbatim view. Drives whether the translated overlay is shown.
   */
  selectedLanguage: string | null;
  /**
   * Cached translations for the currently open meeting + `selectedLanguage`,
   * keyed by segment `start_ms` rather than array position. The backend's
   * `get_translations` returns a segment-INDEX-keyed map (see
   * `../ipc/translations`, matching how `translations.json` is written);
   * `translate` / `loadTranslations` convert it against the segment list
   * supplied by the caller before storing it here. Keying by `start_ms`
   * (stable across a re-diarize's speaker-turn split/merge, #0015) rather
   * than index means a segment array that has shifted under the cached
   * translations simply fails to match — showing the verbatim fallback —
   * instead of overlaying a translation onto the wrong row.
   */
  translations: Map<number, string>;
  /**
   * True while a `translate_meeting` call is in flight (used to disable the
   * Translate button and show the operation-progress indicator instead).
   */
  translateInFlight: boolean;
  /** Last error from a translation IPC call. */
  lastError: string | null;
  /**
   * The open meeting id, kept here so `handleEvent` can reload translations
   * when `TranslationReady` fires for the active meeting. Set by the
   * `TranscriptPane` via `setOpenMeeting` when a meeting opens/closes.
   */
  openMeetingId: MeetingId | null;

  /** Track which meeting is currently open (called from TranscriptPane). */
  setOpenMeeting: (meetingId: MeetingId | null) => void;

  /**
   * Switch to the verbatim view by clearing the selected language (and
   * dropping the cached translations from memory, since they are no longer
   * displayed).
   */
  showVerbatim: () => void;

  /**
   * Fetch the existing translations for `meetingId` + `language` from the
   * backend and switch to the translated view.
   *
   * `segments` is the transcript currently in view for `meetingId` — used to
   * convert the backend's index-keyed result into the `start_ms`-keyed map
   * this store holds; it MUST be the segment array the indices were computed
   * against (the active transcript at call time), or the mapping is wrong.
   *
   * Does not trigger a new translation pass — call `translate` for that.
   * Useful when opening a meeting that already has translations on disk, or
   * after a `TranslationReady` event refreshes the cache.
   */
  loadTranslations: (
    meetingId: MeetingId,
    language: string,
    segments: Segment[],
  ) => Promise<void>;

  /**
   * Trigger a new translation pass, then load the results.
   *
   * Calls `translate_meeting` (which runs the LLM over every segment and
   * writes incremental progress into `translations.json`), then calls
   * `get_translations` once the pass resolves. The `translateInFlight` flag
   * is set for the duration so the UI can disable the button. See
   * `loadTranslations` for the `segments` contract.
   */
  translate: (
    meetingId: MeetingId,
    language: string,
    segments: Segment[],
  ) => Promise<void>;

  /**
   * Reset all state back to the verbatim view (called when the open meeting
   * changes so stale translations from the previous meeting are dropped).
   */
  reset: () => void;

  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Convert the backend's segment-INDEX-keyed translation map into one keyed by
 * each segment's `start_ms`, using `segments` as the index -> segment lookup.
 * An entry whose index falls outside `segments` (translations fetched against
 * a transcript that has since been replaced) is dropped rather than mapped to
 * the wrong row.
 */
function keyByStartMs(
  indexed: Map<number, string>,
  segments: Segment[],
): Map<number, string> {
  const result = new Map<number, string>();
  for (const [index, text] of indexed) {
    const seg = segments[index];
    if (seg !== undefined) result.set(seg.start_ms, text);
  }
  return result;
}

export const useTranslationsStore = create<TranslationsStore>((set, get) => ({
  selectedLanguage: null,
  translations: new Map(),
  translateInFlight: false,
  lastError: null,
  openMeetingId: null,

  setOpenMeeting: (meetingId) => {
    // Switching meetings clears the translated view so stale translations from
    // the previous meeting are not shown over the new one.
    set({
      openMeetingId: meetingId,
      selectedLanguage: null,
      translations: new Map(),
    });
  },

  showVerbatim: () => {
    set({ selectedLanguage: null, translations: new Map() });
  },

  loadTranslations: async (meetingId, language, segments) => {
    try {
      const map = await getTranslations(meetingId, language);
      set({
        selectedLanguage: language,
        translations: keyByStartMs(map, segments),
        lastError: null,
      });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  translate: async (meetingId, language, segments) => {
    set({ translateInFlight: true, lastError: null });
    try {
      await translateMeeting(meetingId, language);
      // Pass resolved — fetch the now-persisted translations.
      const map = await getTranslations(meetingId, language);
      set({
        selectedLanguage: language,
        translations: keyByStartMs(map, segments),
        translateInFlight: false,
        lastError: null,
      });
    } catch (err) {
      set({ translateInFlight: false, lastError: errorMessage(err) });
    }
  },

  reset: () => {
    set({
      selectedLanguage: null,
      translations: new Map(),
      translateInFlight: false,
      lastError: null,
      openMeetingId: null,
    });
  },

  handleEvent: (event) => {
    // A background pass rewrote the OPEN meeting's transcript segments — a
    // full re-transcribe (`transcript_ready`), or a re-diarize that
    // split/merged speaker-turn segments (`diarization_complete`, #0015).
    // Either way the cached translations are keyed against a segment array
    // that no longer exists, so drop the translated view entirely rather than
    // risk (however unlikely with `start_ms` keying, see the field doc)
    // showing a stale translation over the wrong row. Re-running Translate
    // supplies the correct segments.
    if (
      event.kind === "transcript_ready" ||
      event.kind === "diarization_complete"
    ) {
      const { openMeetingId } = get();
      if (openMeetingId !== null && openMeetingId === event.meeting_id) {
        set({ selectedLanguage: null, translations: new Map() });
      }
      return;
    }
    if (event.kind !== "translation_ready") return;
    const { openMeetingId, selectedLanguage } = get();
    // Only react when the event is for the open meeting and the language
    // matches what the user is viewing (a background translate for a different
    // language should not auto-switch the view).
    if (
      openMeetingId === null ||
      openMeetingId !== event.meeting_id ||
      selectedLanguage === null ||
      selectedLanguage !== event.language
    )
      return;
    // Re-fetch so the translation overlay reflects the newly completed pass,
    // against the meeting's current transcript (a non-reactive read — this
    // runs from an event handler, not a component).
    void get().loadTranslations(
      openMeetingId,
      selectedLanguage,
      activeTranscript(),
    );
  },
}));
