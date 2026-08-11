import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useMeetingsStore } from "../state/meetings";
import { readAutoStartRecordingOnNewMeeting } from "../state/auto-start-recording-settings";
import type { RecordingState } from "../ipc/bindings";

/**
 * Derive the two context-aware transport toggles from the current recording
 * state (#66 — consolidate the former four always-on buttons to two).
 *
 * There are two buttons:
 *   - RECORD: idle with no open draft and the auto-start setting OFF →
 *     "New meeting" (calls `createMeeting` + opens it, no capture yet, never
 *     ASR-gated — creating a draft touches no model); idle with an open draft
 *     OR the auto-start setting ON → "Start" (calls `promote`/`start`,
 *     ASR-gated); recording/paused/stopping/finalising → "Stop" (calls
 *     `stop`).
 *   - PAUSE: "Pause" (calls `pause`) when recording; "Resume" (calls `resume`)
 *     when paused; disabled otherwise (idle / finalising / stopping).
 *
 * `preparing` is the client-only optimistic transient (live-test UX T1): while
 * the first record lazy-loads the ASR model, the record toggle MUST stay
 * disabled so a double-press cannot re-invoke `startRecording` (which the
 * orchestrator rejects with "start called when not idle"). It never applies
 * to the `new_meeting` action (no model/capture involved).
 *
 * Exported for use in unit tests without requiring the Zustand store.
 */
export type RecordAction = "new_meeting" | "start" | "stop";
export type PauseAction = "pause" | "resume";

export type ButtonStates = {
  /** The label/intent of the RECORD toggle. */
  recordAction: RecordAction;
  recordEnabled: boolean;
  /** The label/intent of the PAUSE toggle. */
  pauseAction: PauseAction;
  pauseEnabled: boolean;
};

export function deriveButtonStates(
  state: RecordingState,
  isAsrModelReady: boolean,
  preparing: boolean,
  hasOpenDraft: boolean = false,
  autoStartOnNewMeeting: boolean = false,
): ButtonStates {
  const isIdle = state.kind === "idle";
  const isRecording = state.kind === "recording";
  const isPaused = state.kind === "paused";

  // An open draft or the auto-start setting means this idle press goes
  // straight to recording; otherwise it just creates + opens a prep draft.
  const willImmediatelyRecord = hasOpenDraft || autoStartOnNewMeeting;

  let recordAction: RecordAction;
  let recordEnabled: boolean;
  if (isIdle) {
    if (willImmediatelyRecord) {
      recordAction = "start";
      // Start only from a genuinely idle recorder, with the model ready, and
      // NOT while a start is already in flight (preparing) — a double-press
      // is then impossible.
      recordEnabled = isAsrModelReady && !preparing;
    } else {
      recordAction = "new_meeting";
      // Creating a prep draft touches no model/capture device — never gated.
      recordEnabled = true;
    }
  } else {
    recordAction = "stop";
    // Stop only from recording/paused — disabled while stopping/finalising.
    recordEnabled = isRecording || isPaused;
  }

  // PAUSE: Pause while recording, Resume while paused, disabled otherwise.
  const pauseAction: PauseAction = isPaused ? "resume" : "pause";
  const pauseEnabled = isRecording || isPaused;

  return { recordAction, recordEnabled, pauseAction, pauseEnabled };
}

export function MeetingControls() {
  const state = useRecordingStore((s) => s.state);
  const start = useRecordingStore((s) => s.start);
  const createMeeting = useRecordingStore((s) => s.createMeeting);
  const promote = useRecordingStore((s) => s.promote);
  const stop = useRecordingStore((s) => s.stop);
  const pause = useRecordingStore((s) => s.pause);
  const resume = useRecordingStore((s) => s.resume);
  const preparing = useRecordingStore((s) => s.preparing);
  const autoStartOnNewMeeting = useRecordingStore((s) =>
    readAutoStartRecordingOnNewMeeting(s.settings),
  );
  const isAsrModelReady = useModelsStore((s) => s.isAsrModelReady);
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);
  const openMeetingState = useMeetingsStore((s) => s.openMeetingState);
  const openMeeting = useMeetingsStore((s) => s.open);

  // A currently-open meeting that has never started recording — the "New
  // meeting" prep screen. Only meaningful while idle (a live/finished
  // meeting is never a draft by the time it is open in those states). The
  // `uuid` check guards against a stale `openMeetingState` snapshot (e.g. the
  // brief window right after a stop, before the async re-open resolves) being
  // misread as a draft — trusting it would let "Start" re-promote (and
  // truncate) a meeting that already recorded.
  const hasOpenDraft =
    state.kind === "idle" &&
    openMeetingId !== null &&
    openMeetingState?.meta.uuid === openMeetingId &&
    openMeetingState.meta.recording_started === false;

  const { recordAction, recordEnabled, pauseAction, pauseEnabled } =
    deriveButtonStates(
      state,
      isAsrModelReady,
      preparing,
      hasOpenDraft,
      autoStartOnNewMeeting,
    );

  const recordLabel =
    recordAction === "stop"
      ? "Stop"
      : recordAction === "new_meeting"
        ? "New meeting"
        : preparing
          ? "Preparing…"
          : "Start";
  const pauseLabel = pauseAction === "pause" ? "Pause" : "Resume";

  async function handleRecordClick() {
    if (recordAction === "stop") {
      await stop();
      return;
    }
    if (recordAction === "new_meeting") {
      const meetingId = await createMeeting();
      if (meetingId !== null) await openMeeting(meetingId);
      return;
    }
    // recordAction === "start": either promote the already-open draft, or
    // (the auto-start setting) create + promote a fresh one in one call.
    if (hasOpenDraft && openMeetingId !== null) {
      await promote(openMeetingId, openMeetingState?.meta.title);
    } else {
      await start();
    }
  }

  return (
    <div className="meeting-controls">
      <button
        className="meeting-controls__record"
        data-action={recordAction}
        onClick={() => void handleRecordClick()}
        disabled={!recordEnabled}
        aria-label={recordLabel}
      >
        {recordLabel}
      </button>
      <button
        className="meeting-controls__pause"
        data-action={pauseAction}
        onClick={() => void (pauseAction === "pause" ? pause() : resume())}
        disabled={!pauseEnabled}
        aria-label={pauseLabel}
      >
        {pauseLabel}
      </button>
    </div>
  );
}
