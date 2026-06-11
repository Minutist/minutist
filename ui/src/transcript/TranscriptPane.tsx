import { useEffect, useRef } from "react";
import { useCrossRefStore } from "../state/cross-ref";
import { useActiveTranscript } from "../state/active-transcript";
import { useMeetingsStore } from "../state/meetings";
import { useRecordingStore } from "../state/recording";
import { useOperationProgressStore } from "../state/operation-progress";
import { writeSegmentDrag } from "../editor/transcript-dnd";
import { speakerColorIndex } from "./speaker-color";
import type { Segment } from "../ipc/bindings";
import "./TranscriptPane.css";

/**
 * Action toolbar at the top of the transcript pane (#67).
 *
 * Holds the two offline reprocessing actions that used to live in the meeting
 * heading: Re-transcribe (re-run ASR over the recording) and Re-identify
 * speakers (re-run diarization). They operate on the OPEN saved meeting and are
 * shown only when one is open and nothing is recording — an offline op must not
 * contend with the live pipeline (the backend refuses unless `Idle`).
 *
 * Each action is disabled while a recording is live OR while its own op is
 * already in flight for the open meeting (read from the operation-progress
 * store, keyed on `meeting_id` + op), so a double-press cannot re-claim the
 * offline slot.
 */
function TranscriptToolbar() {
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);
  const reTranscribe = useMeetingsStore((s) => s.reTranscribe);
  const reDiarize = useMeetingsStore((s) => s.rediarize);
  const recordingState = useRecordingStore((s) => s.state);
  const operation = useOperationProgressStore((s) =>
    openMeetingId !== null ? s.operations[openMeetingId] : undefined,
  );

  // Offline reprocessing only applies to a saved meeting that is open while the
  // recorder is idle (recording / paused / stopping / finalising would contend
  // with the live pipeline). Render nothing otherwise.
  if (openMeetingId === null || recordingState.kind !== "idle") return null;

  const retranscribeInFlight = operation?.op === "re_transcribe";
  const rediarizeInFlight = operation?.op === "rediarize";

  return (
    <div
      className="transcript-pane__toolbar"
      role="toolbar"
      aria-label="Transcript actions"
    >
      <button
        type="button"
        className="transcript-pane__action"
        disabled={retranscribeInFlight}
        onClick={() => void reTranscribe(openMeetingId)}
        title="Re-run speech recognition on this recording (rare; e.g. after changing the language or model)."
      >
        Re-transcribe
      </button>
      <button
        type="button"
        className="transcript-pane__action"
        disabled={rediarizeInFlight}
        onClick={() => void reDiarize(openMeetingId)}
        title="Re-run speaker identification on this recording (rare)."
      >
        Re-identify speakers
      </button>
    </div>
  );
}

/**
 * Format a recording-clock millisecond offset as MM:SS.cc.
 *
 * `start_ms` is a recording-clock offset (ms from the start of the
 * recording), not a wall-clock timestamp.
 *
 * Examples:
 *   0       → "00:00.00"
 *   1234    → "00:01.23"
 *   75400   → "01:15.40"
 *   3723456 → "62:03.45"
 */
export function formatTimestamp(start_ms: number): string {
  const totalCentiseconds = Math.floor(start_ms / 10);
  const cs = totalCentiseconds % 100;
  const totalSeconds = Math.floor(totalCentiseconds / 100);
  const ss = totalSeconds % 60;
  const mm = Math.floor(totalSeconds / 60);

  const pad2 = (n: number) => String(n).padStart(2, "0");
  return `${pad2(mm)}:${pad2(ss)}.${pad2(cs)}`;
}

/**
 * Read-only scrollable transcript view.
 *
 * Auto-scrolls to the bottom on new segments unless the user has scrolled
 * up more than 50 px from the bottom (sticky-bottom behaviour).
 *
 * Cross-reference (Phase 4):
 *   - FR-24: each row is a native drag source — dragging it into the notes
 *     editor inserts a transcript-chip node carrying the segment.
 *   - FR-23: clicking a row publishes a scroll request so the notes editor
 *     scrolls to the nearest-anchored paragraph.
 *   - FR-22: when a notes paragraph is hovered, every segment in the paragraph's
 *     transcript span is highlighted — the half-open `[startIndex, endIndex)`
 *     range published as `highlightedRange` by the cross-ref store.
 *
 * The transcript shown is the ACTIVE transcript (`useActiveTranscript`): a
 * saved meeting's restored segments when viewing a saved meeting (U1), else the
 * live recording transcript.
 */
export function TranscriptPane() {
  const transcript = useActiveTranscript();
  const highlightedRange = useCrossRefStore((s) => s.highlightedRange);
  const clickTranscriptSegment = useCrossRefStore(
    (s) => s.clickTranscriptSegment,
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  // Track whether the user has scrolled away from the bottom.
  const userScrolledUp = useRef(false);

  // When the transcript grows, scroll to bottom if the user has not
  // scrolled away.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!userScrolledUp.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [transcript]);

  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    userScrolledUp.current = distanceFromBottom > 50;
  }

  return (
    <div className="transcript-pane">
      <TranscriptToolbar />
      <div
        className="transcript-pane__scroll"
        ref={scrollRef}
        onScroll={handleScroll}
        aria-label="Transcript"
        aria-live="polite"
      >
        {transcript.length === 0 ? (
          <p className="transcript-pane__empty">
            Transcript will appear here while you record.
          </p>
        ) : (
          <ol className="transcript-pane__list">
            {transcript.map((seg: Segment, idx: number) => {
              const highlighted =
                highlightedRange !== null &&
                idx >= highlightedRange.startIndex &&
                idx < highlightedRange.endIndex;
              // Group consecutive rows by speaker: the labelled chip shows only
              // when the speaker changes from the previous row. Continuation rows
              // keep just the colour dot, so the per-speaker colour persists on
              // every diarized row without the "Speaker X" label repeating.
              const prevSpeaker =
                idx > 0 ? transcript[idx - 1].speaker_id : null;
              const speakerChanged =
                seg.speaker_id != null && seg.speaker_id !== prevSpeaker;
              const dotColor =
                seg.speaker_id != null
                  ? `var(--speaker-${speakerColorIndex(seg.speaker_id)})`
                  : undefined;
              return (
                <li
                  key={idx}
                  className={
                    highlighted
                      ? "transcript-pane__row transcript-pane__row--highlighted"
                      : "transcript-pane__row"
                  }
                  aria-current={highlighted ? "true" : undefined}
                  onClick={() => clickTranscriptSegment(seg)}
                >
                  {/*
                    The timestamp is the drag handle (FR-24): only it is
                    `draggable`, so the segment text stays freely selectable for
                    copy. Clicking anywhere on the row still jumps to the linked
                    paragraph (FR-23, the row's onClick).
                  */}
                  <span
                    className="transcript-pane__timestamp tnum"
                    draggable
                    title="Drag into your notes, or click to jump to the linked paragraph"
                    onDragStart={(e) => {
                      if (e.dataTransfer) writeSegmentDrag(e.dataTransfer, seg);
                    }}
                  >
                    {formatTimestamp(seg.start_ms)}
                  </span>
                  <span className="transcript-pane__text">
                    {/*
                    Phase 6/C: a quiet "Speaker {id}" chip when diarization has
                    assigned this segment a speaker, shown only at the start of a
                    speaker's run (see `speakerChanged`). The id is the diarizer's
                    first-seen label (A / B / …), surfaced verbatim. The
                    per-speaker colour dot's palette slot is resolved by the pure
                    `speakerColorIndex` mapper and passed via the `--dot-color`
                    custom property — tokens only, no hard-coded colour in TSX.
                  */}
                    {speakerChanged && (
                      <span
                        className="transcript-pane__speaker"
                        style={{ ["--dot-color" as string]: dotColor }}
                        aria-label={`Speaker ${seg.speaker_id}`}
                      >
                        <span
                          className="transcript-pane__speaker-dot"
                          aria-hidden="true"
                        />
                        Speaker {seg.speaker_id}
                      </span>
                    )}
                    {/*
                    Continuation row (same speaker as the row above): the colour
                    dot alone, no repeated label. Decorative — the run's leading
                    chip carries the accessible name.
                  */}
                    {seg.speaker_id != null && !speakerChanged && (
                      <span
                        className="transcript-pane__speaker transcript-pane__speaker--cont"
                        style={{ ["--dot-color" as string]: dotColor }}
                        aria-hidden="true"
                      >
                        <span
                          className="transcript-pane__speaker-dot"
                          aria-hidden="true"
                        />
                      </span>
                    )}
                    {seg.text}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </div>
  );
}
