/**
 * Active-transcript data source (U1 — open_meeting restore wiring).
 *
 * There are two transcript sources in the UI:
 *
 *  - the LIVE store (`useRecordingStore.transcript`), appended to by
 *    `transcript_segment` events while recording, and
 *  - a SAVED meeting's restored transcript (`useMeetingsStore.openMeetingState
 *    .transcript`), loaded by `open_meeting`.
 *
 * The transcript pane and the cross-reference must read from whichever is
 * active. The rule: when a saved meeting is open AND nothing is being recorded
 * (`openMeetingId !== null && recordingState.kind === "idle"`), the panes view
 * the SAVED meeting's transcript; otherwise (live recording, or no meeting
 * open) they read the LIVE store. This single selector is the one place that
 * decision is made, so the editor, transcript pane, and cross-ref all agree on
 * the source.
 */
import { useRecordingStore } from "./recording";
import { useMeetingsStore } from "./meetings";
import type { Segment } from "../ipc/bindings";

/**
 * True when the UI is viewing a previously-saved meeting (opened via
 * `open_meeting`) rather than a live recording session. Recording takes
 * precedence: starting/resuming a recording switches the source back to live
 * even if a meeting id is still set.
 */
export function isViewingSavedMeeting(): boolean {
  const openMeetingId = useMeetingsStore.getState().openMeetingId;
  const recordingState = useRecordingStore.getState().state;
  return openMeetingId !== null && recordingState.kind === "idle";
}

/**
 * The transcript currently in view: the opened saved meeting's transcript when
 * viewing a saved meeting, else the live recording transcript.
 *
 * A non-reactive read for callers that only need the latest value at call time
 * (e.g. the editor's hover reporter, which closes over a ref). React components
 * that must re-render on change use {@link useActiveTranscript}.
 */
export function activeTranscript(): Segment[] {
  if (isViewingSavedMeeting()) {
    return useMeetingsStore.getState().openMeetingState?.transcript ?? [];
  }
  return useRecordingStore.getState().transcript;
}

/**
 * Reactive selector hook: the transcript currently in view. Re-renders the
 * consumer when the live transcript grows, when the open meeting changes, or
 * when the recording state flips between live and idle.
 */
export function useActiveTranscript(): Segment[] {
  const liveTranscript = useRecordingStore((s) => s.transcript);
  const recordingKind = useRecordingStore((s) => s.state.kind);
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);
  const savedTranscript = useMeetingsStore(
    (s) => s.openMeetingState?.transcript,
  );

  if (openMeetingId !== null && recordingKind === "idle") {
    return savedTranscript ?? [];
  }
  return liveTranscript;
}
