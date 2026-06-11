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
import type { MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type TranslationsStore = {
  /**
   * The language the user has selected to translate into, or `null` for the
   * verbatim view. Drives whether the translated overlay is shown.
   */
  selectedLanguage: string | null;
  /**
   * Cached translations for the currently open meeting + `selectedLanguage`.
   * Map from segment index to translated text. Empty when no translation has
   * been fetched yet (or when the verbatim view is active).
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
   * Does not trigger a new translation pass — call `translate` for that.
   * Useful when opening a meeting that already has translations on disk, or
   * after a `TranslationReady` event refreshes the cache.
   */
  loadTranslations: (
    meetingId: MeetingId,
    language: string,
  ) => Promise<void>;

  /**
   * Trigger a new translation pass, then load the results.
   *
   * Calls `translate_meeting` (which runs the LLM over every segment and
   * writes incremental progress into `translations.json`), then calls
   * `get_translations` once the pass resolves. The `translateInFlight` flag
   * is set for the duration so the UI can disable the button.
   */
  translate: (meetingId: MeetingId, language: string) => Promise<void>;

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

  loadTranslations: async (meetingId, language) => {
    try {
      const map = await getTranslations(meetingId, language);
      set({ selectedLanguage: language, translations: map, lastError: null });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  translate: async (meetingId, language) => {
    set({ translateInFlight: true, lastError: null });
    try {
      await translateMeeting(meetingId, language);
      // Pass resolved — fetch the now-persisted translations.
      const map = await getTranslations(meetingId, language);
      set({
        selectedLanguage: language,
        translations: map,
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
    // Re-fetch so the translation overlay reflects the newly completed pass.
    void get().loadTranslations(openMeetingId, selectedLanguage);
  },
}));
