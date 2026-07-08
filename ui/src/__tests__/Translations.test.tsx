/**
 * Tests for the translated transcript feature (WU4) and its G3c hardening.
 *
 * Covers:
 * - `translate_meeting` command invocation through the translations store.
 * - `get_translations`'s index-keyed result is converted to a `start_ms`-keyed
 *   map against the segments supplied by the caller.
 * - `showVerbatim` clears the selected language and translations map.
 * - `TranslationReady` event refreshes translations for the active meeting +
 *   language (and ignores events for different languages or meetings).
 * - `transcript_ready` / `diarization_complete` for the open meeting clear the
 *   cached translations (a reprocess replaces the segment array the cached
 *   `start_ms` keys were computed against — G3c).
 * - `translation_ready` clears the operation-progress indicator.
 *
 * Tests mock `../ipc/translations` (the seam module), not the generated
 * bindings file (per `architecture/cross-cutting.md` — Automated testing).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({
  WebviewWindow: vi.fn(),
}));

vi.mock("../ipc/translations", () => ({
  translateMeeting: vi.fn().mockResolvedValue(undefined),
  getTranslations: vi.fn().mockResolvedValue(new Map<number, string>()),
}));

// `handleEvent`'s translation_ready re-fetch sources segments from the active
// transcript (a non-reactive read); mock it so the test controls the segment
// array independently of the recording/meetings stores.
vi.mock("../state/active-transcript", () => ({
  activeTranscript: vi.fn(),
}));

import { translateMeeting, getTranslations } from "../ipc/translations";
import { useTranslationsStore } from "../state/translations";
import { useOperationProgressStore } from "../state/operation-progress";
import { activeTranscript } from "../state/active-transcript";
import type { AppEvent, Segment } from "../ipc/bindings";

const MEETING_ID = "00000000-0000-0000-0000-000000000001";

/** Two segments at start_ms 0 and 5_000 — the fixture index -> start_ms map. */
function makeSegments(): Segment[] {
  return [
    { start_ms: 0, end_ms: 1_000, text: "first", words: [], shared_speakers: [] },
    { start_ms: 5_000, end_ms: 6_000, text: "second", words: [], shared_speakers: [] },
  ];
}

function resetStores() {
  useTranslationsStore.setState({
    selectedLanguage: null,
    translations: new Map(),
    translateInFlight: false,
    lastError: null,
    openMeetingId: null,
  });
  useOperationProgressStore.setState({ operations: {} });
  vi.mocked(activeTranscript).mockReturnValue(makeSegments());
}

describe("translations store", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStores();
  });

  it("translate() calls translateMeeting then getTranslations, keying the result by start_ms", async () => {
    // Backend result is INDEX-keyed (segment 0, segment 1).
    const translatedMap = new Map<number, string>([
      [0, "Hola mundo"],
      [1, "Esta es una prueba"],
    ]);
    vi.mocked(getTranslations).mockResolvedValueOnce(translatedMap);

    useTranslationsStore.setState({ openMeetingId: MEETING_ID });
    await useTranslationsStore
      .getState()
      .translate(MEETING_ID, "Spanish", makeSegments());

    expect(translateMeeting).toHaveBeenCalledWith(MEETING_ID, "Spanish");
    expect(getTranslations).toHaveBeenCalledWith(MEETING_ID, "Spanish");
    expect(useTranslationsStore.getState().selectedLanguage).toBe("Spanish");
    // Converted to the segments' start_ms (0 and 5_000), not the raw index.
    expect(useTranslationsStore.getState().translations.get(0)).toBe(
      "Hola mundo",
    );
    expect(useTranslationsStore.getState().translations.get(5_000)).toBe(
      "Esta es una prueba",
    );
    expect(useTranslationsStore.getState().translateInFlight).toBe(false);
  });

  it("translate() drops an index that falls outside the supplied segments", async () => {
    // Index 2 has no corresponding segment in a 2-segment array.
    const translatedMap = new Map<number, string>([[2, "orphaned"]]);
    vi.mocked(getTranslations).mockResolvedValueOnce(translatedMap);

    await useTranslationsStore
      .getState()
      .translate(MEETING_ID, "Spanish", makeSegments());

    expect(useTranslationsStore.getState().translations.size).toBe(0);
  });

  it("loadTranslations() fetches without triggering translateMeeting", async () => {
    const translatedMap = new Map<number, string>([[0, "Bonjour"]]);
    vi.mocked(getTranslations).mockResolvedValueOnce(translatedMap);

    await useTranslationsStore
      .getState()
      .loadTranslations(MEETING_ID, "French", makeSegments());

    expect(translateMeeting).not.toHaveBeenCalled();
    expect(getTranslations).toHaveBeenCalledWith(MEETING_ID, "French");
    expect(useTranslationsStore.getState().selectedLanguage).toBe("French");
    expect(useTranslationsStore.getState().translations.get(0)).toBe("Bonjour");
  });

  it("showVerbatim() clears selectedLanguage and translations", () => {
    useTranslationsStore.setState({
      selectedLanguage: "Spanish",
      translations: new Map([[0, "Hola"]]),
    });

    useTranslationsStore.getState().showVerbatim();

    expect(useTranslationsStore.getState().selectedLanguage).toBeNull();
    expect(useTranslationsStore.getState().translations.size).toBe(0);
  });

  it("setOpenMeeting() resets language and translations on meeting change", () => {
    useTranslationsStore.setState({
      openMeetingId: MEETING_ID,
      selectedLanguage: "Spanish",
      translations: new Map([[0, "Hola"]]),
    });

    const OTHER = "00000000-0000-0000-0000-000000000002";
    useTranslationsStore.getState().setOpenMeeting(OTHER);

    expect(useTranslationsStore.getState().selectedLanguage).toBeNull();
    expect(useTranslationsStore.getState().translations.size).toBe(0);
    expect(useTranslationsStore.getState().openMeetingId).toBe(OTHER);
  });

  it("handleEvent(translation_ready) re-fetches when language matches open meeting", async () => {
    const refreshedMap = new Map<number, string>([[0, "Hola v2"]]);
    vi.mocked(getTranslations).mockResolvedValueOnce(refreshedMap);

    useTranslationsStore.setState({
      openMeetingId: MEETING_ID,
      selectedLanguage: "Spanish",
      translations: new Map([[0, "Hola v1"]]),
    });

    const event: AppEvent = {
      kind: "translation_ready",
      meeting_id: MEETING_ID,
      language: "Spanish",
    };
    useTranslationsStore.getState().handleEvent(event);
    // loadTranslations is async — let its microtask run.
    await Promise.resolve();
    await Promise.resolve();

    expect(getTranslations).toHaveBeenCalledWith(MEETING_ID, "Spanish");
    expect(useTranslationsStore.getState().translations.get(0)).toBe(
      "Hola v2",
    );
  });

  it("handleEvent(translation_ready) ignores events for a different language", async () => {
    useTranslationsStore.setState({
      openMeetingId: MEETING_ID,
      selectedLanguage: "French",
    });

    const event: AppEvent = {
      kind: "translation_ready",
      meeting_id: MEETING_ID,
      language: "Spanish",
    };
    useTranslationsStore.getState().handleEvent(event);
    await Promise.resolve();

    expect(getTranslations).not.toHaveBeenCalled();
  });

  it("handleEvent(translation_ready) ignores events for a different meeting", async () => {
    useTranslationsStore.setState({
      openMeetingId: MEETING_ID,
      selectedLanguage: "Spanish",
    });

    const OTHER = "00000000-0000-0000-0000-000000000099";
    const event: AppEvent = {
      kind: "translation_ready",
      meeting_id: OTHER,
      language: "Spanish",
    };
    useTranslationsStore.getState().handleEvent(event);
    await Promise.resolve();

    expect(getTranslations).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // G3c — a reprocess (re-transcribe and/or re-diarize) replaces the open
  // meeting's segment array; cached translations are keyed against the
  // segments that produced them and must not survive to be misapplied.
  // -------------------------------------------------------------------------

  it.each(["transcript_ready", "diarization_complete"] as const)(
    "handleEvent(%s) clears the translated view for the open meeting",
    (kind) => {
      useTranslationsStore.setState({
        openMeetingId: MEETING_ID,
        selectedLanguage: "Spanish",
        translations: new Map([[0, "Hola"]]),
      });

      const event = (
        kind === "diarization_complete"
          ? { kind, meeting_id: MEETING_ID, speaker_count: 2 }
          : { kind, meeting_id: MEETING_ID }
      ) as AppEvent;
      useTranslationsStore.getState().handleEvent(event);

      const state = useTranslationsStore.getState();
      expect(state.selectedLanguage).toBeNull();
      expect(state.translations.size).toBe(0);
    },
  );

  it("handleEvent(transcript_ready) leaves an unrelated meeting's translations untouched", () => {
    useTranslationsStore.setState({
      openMeetingId: MEETING_ID,
      selectedLanguage: "Spanish",
      translations: new Map([[0, "Hola"]]),
    });

    useTranslationsStore.getState().handleEvent({
      kind: "transcript_ready",
      meeting_id: "00000000-0000-0000-0000-000000000099",
    });

    const state = useTranslationsStore.getState();
    expect(state.selectedLanguage).toBe("Spanish");
    expect(state.translations.get(0)).toBe("Hola");
  });
});

describe("operation-progress store — translation_ready clears indicator", () => {
  beforeEach(() => {
    resetStores();
  });

  it("translation_ready clears the translate op indicator", () => {
    useOperationProgressStore.setState({
      operations: {
        [MEETING_ID]: {
          op: "translate",
          fraction: 0.75,
          label: "Translating…",
        },
      },
    });

    const event: AppEvent = {
      kind: "translation_ready",
      meeting_id: MEETING_ID,
      language: "Spanish",
    };
    useOperationProgressStore.getState().handleEvent(event);

    expect(
      useOperationProgressStore.getState().operationFor(MEETING_ID),
    ).toBeNull();
  });

  /**
   * Regression: the backend now emits TranslationReady on EVERY exit path
   * (success and error) so the operation-progress row is never orphaned.
   * This test verifies the store clears the indicator when translation_ready
   * arrives after a failed translation pass (simulated by first populating the
   * progress row and then firing the event — identical to the success-path
   * signal from the backend's perspective).
   */
  it("translation_ready clears the indicator after a failed translation pass", () => {
    useOperationProgressStore.setState({
      operations: {
        [MEETING_ID]: {
          op: "translate",
          fraction: 0.3,
          label: "Translating… (3/10)",
        },
      },
    });

    // The backend emits TranslationReady even when the pass errors mid-segment.
    const event: AppEvent = {
      kind: "translation_ready",
      meeting_id: MEETING_ID,
      language: "Spanish",
    };
    useOperationProgressStore.getState().handleEvent(event);

    expect(
      useOperationProgressStore.getState().operationFor(MEETING_ID),
    ).toBeNull();
  });
});
