import { useEffect, useRef, useState } from "react";
import { useCrossRefStore } from "../state/cross-ref";
import { useActiveTranscript } from "../state/active-transcript";
import { useMeetingsStore } from "../state/meetings";
import { useRecordingStore } from "../state/recording";
import { useOperationProgressStore } from "../state/operation-progress";
import { useTranslationsStore } from "../state/translations";
import { OUTPUT_LANGUAGES } from "../shell/OutputLanguagePicker";
import { writeSegmentDrag } from "../editor/transcript-dnd";
import { speakerColorIndex } from "./speaker-color";
import type { MeetingId, Segment } from "../ipc/bindings";
import "./TranscriptPane.css";

/**
 * Action toolbar at the top of the transcript pane (#67).
 *
 * Holds the offline Reprocess action (one button that re-transcribes THEN
 * re-diarizes under a single backend claim, #0015) and the translation controls
 * (language selector + Translate / Show original button). All actions operate on
 * the OPEN saved meeting and are shown only when one is open and nothing is
 * recording — an offline op must not contend with the live pipeline (the backend
 * refuses unless `Idle`).
 *
 * Each action is disabled while a recording is live OR while an offline op is
 * already in flight for the open meeting (read from the operation-progress
 * store, keyed on `meeting_id` + op), so a double-press cannot re-claim the
 * offline slot. The merged backend op still reports two progress phases
 * (`re_transcribe` then `rediarize`); the button treats either as "reprocessing".
 *
 * The translation controls:
 *   - A `<select>` pre-seeded to the first language in OUTPUT_LANGUAGES when
 *     no language is selected.
 *   - A "Translate" button (disabled while a translate op is in flight or any
 *     other op occupies the slot for this meeting).
 *   - Once a translated view is active, a "Show original" button flips back to
 *     the verbatim transcript.
 */
/** Format a millisecond duration as `M:SS` for the elapsed / ETA readout. */
function formatClock(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const mm = Math.floor(totalSeconds / 60);
  const ss = totalSeconds % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

/**
 * Progress readout for an in-flight reprocess: a full-width bar plus a text line
 * (phase label, percent, elapsed, and — during the determinate re-transcribe
 * phase — a rough ETA). Before the backend's first `operation_progress` event
 * lands, the op is absent, so it shows an indeterminate "Starting…" state driven
 * by the store's local flag — giving the click immediate visible feedback.
 *
 * Phases: re-transcribe reports a determinate fraction (a bar + ETA); the
 * subsequent diarize phase is indeterminate (an animated bar, no ETA).
 */
function ReprocessProgress({
  meetingId,
  startedMs,
}: {
  meetingId: MeetingId;
  startedMs: number | null;
}) {
  const op = useOperationProgressStore((s) => s.operations[meetingId]);
  // Tick once a second so the elapsed / ETA readout advances live.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const fraction = op?.fraction ?? null;
  const determinate = fraction !== null;
  const pct = determinate
    ? Math.round(Math.min(1, Math.max(0, fraction)) * 100)
    : null;
  const label = op?.label ?? "Starting…";

  const elapsedMs = startedMs !== null ? Math.max(0, nowMs - startedMs) : 0;
  // A rough ETA from the determinate fraction; suppressed below 3 % so the first
  // ticks don't show a wildly swinging estimate.
  const etaMs =
    determinate && (fraction as number) > 0.03
      ? (elapsedMs * (1 - (fraction as number))) / (fraction as number)
      : null;

  return (
    <div
      className="transcript-pane__progress"
      role="status"
      aria-live="polite"
      data-op={op?.op}
    >
      <div
        className="transcript-pane__progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct ?? undefined}
      >
        <div
          className={
            determinate
              ? "transcript-pane__progress-fill"
              : "transcript-pane__progress-fill transcript-pane__progress-fill--indeterminate"
          }
          style={determinate ? { width: `${pct}%` } : undefined}
        />
      </div>
      <span className="transcript-pane__progress-text tnum">
        {label}
        {pct !== null ? ` · ${pct}%` : ""}
        {` · ${formatClock(elapsedMs)} elapsed`}
        {etaMs !== null ? ` · ~${formatClock(etaMs)} left` : ""}
      </span>
    </div>
  );
}

function TranscriptToolbar() {
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);
  const reprocess = useMeetingsStore((s) => s.reprocess);
  const reprocessingId = useMeetingsStore((s) => s.reprocessingId);
  const reprocessStartedMs = useMeetingsStore((s) => s.reprocessStartedMs);
  const recordingState = useRecordingStore((s) => s.state);
  const operation = useOperationProgressStore((s) =>
    openMeetingId !== null ? s.operations[openMeetingId] : undefined,
  );

  const selectedLanguage = useTranslationsStore((s) => s.selectedLanguage);
  const translateInFlight = useTranslationsStore((s) => s.translateInFlight);
  const translate = useTranslationsStore((s) => s.translate);
  const showVerbatim = useTranslationsStore((s) => s.showVerbatim);

  // Translation target language. Starts UNSELECTED (a "Translate to…"
  // placeholder) rather than pre-seeded to the alphabetically-first language, so
  // the control never silently defaults to e.g. Arabic.
  const [pickedLanguage, setPickedLanguage] = useState<string>("");

  // Offline reprocessing only applies to a saved meeting that is open while the
  // recorder is idle (recording / paused / stopping / finalising would contend
  // with the live pipeline). Render nothing otherwise.
  if (openMeetingId === null || recordingState.kind !== "idle") return null;

  // In flight the INSTANT the user clicks (the store's local `reprocessingId`
  // flag, set before the IPC await), then corroborated by the backend's progress
  // events (`re_transcribe` then `rediarize`). The local flag spans the whole
  // pass; the events also feed the progress bar.
  const localReprocessing = reprocessingId === openMeetingId;
  const reprocessInFlight =
    localReprocessing ||
    operation?.op === "re_transcribe" ||
    operation?.op === "rediarize";
  const translateOpInFlight = operation?.op === "translate";
  // Every offline op contends for the same backend claim — reprocess and
  // translate each refuse while the other runs. Disable BOTH buttons whenever
  // any offline op is in flight.
  const anyOfflineOpInFlight =
    reprocessInFlight || translateOpInFlight || translateInFlight;
  const reprocessDisabled = anyOfflineOpInFlight;
  const translateDisabled = anyOfflineOpInFlight;

  return (
    <>
      <div
        className="transcript-pane__toolbar"
        role="toolbar"
        aria-label="Transcript actions"
      >
        <button
          type="button"
          className="transcript-pane__action"
          disabled={reprocessDisabled}
          onClick={() => void reprocess(openMeetingId)}
          title="Re-run speech recognition and speaker identification on this recording, using the transcription language and model set in Settings. Resets any speaker names you have set."
        >
          {reprocessInFlight ? "Reprocessing…" : "Reprocess"}
        </button>
        <span className="transcript-pane__toolbar-sep" aria-hidden="true" />
        {selectedLanguage !== null ? (
          <button
            type="button"
            className="transcript-pane__action"
            onClick={showVerbatim}
            title="Return to the verbatim transcript."
          >
            Show original
          </button>
        ) : (
          <>
            <select
              className="transcript-pane__lang-select"
              aria-label="Translation target language"
              value={pickedLanguage}
              onChange={(e) => setPickedLanguage(e.target.value)}
            >
              <option value="">Translate to…</option>
              {OUTPUT_LANGUAGES.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="transcript-pane__action"
              disabled={translateDisabled || pickedLanguage === ""}
              onClick={() => void translate(openMeetingId, pickedLanguage)}
              title="Translate this transcript into the selected language using the local LLM."
            >
              {translateOpInFlight || translateInFlight
                ? "Translating…"
                : "Translate"}
            </button>
          </>
        )}
      </div>
      {reprocessInFlight && (
        <ReprocessProgress
          meetingId={openMeetingId}
          startedMs={reprocessStartedMs}
        />
      )}
    </>
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
/**
 * The run-leading speaker chip. Shows the speaker's display name when one is
 * set (`speaker_names[label]`), otherwise the bare `Speaker {label}`.
 *
 * `editable` is true only when viewing a saved, finalised meeting — there the
 * chip is a button that opens an inline rename writing through
 * `setSpeakerName`, applying to every row for that label. During a live
 * recording it is false: the diarizer's labels are provisional and get
 * re-lettered (and any names cleared) when the meeting finalises, and there is
 * no persisted metadata to write yet, so the chip is a plain non-interactive
 * label. The chip is never the row's drag handle (the timestamp is) and stops
 * click propagation so renaming does not also trigger the row's jump.
 */
function SpeakerChip({
  label,
  name,
  dotColor,
  editable,
  onRename,
}: {
  label: string;
  name: string | null;
  dotColor: string | undefined;
  editable: boolean;
  onRename: (label: string, name: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(name ?? "");

  function commit() {
    onRename(label, value.trim());
    setEditing(false);
  }

  if (!editable) {
    return (
      <span
        className="transcript-pane__speaker transcript-pane__speaker--readonly"
        style={{ ["--dot-color" as string]: dotColor }}
      >
        <span className="transcript-pane__speaker-dot" aria-hidden="true" />
        {name ?? `Speaker ${label}`}
      </span>
    );
  }

  if (editing) {
    return (
      <input
        className="transcript-pane__speaker-input"
        // eslint-disable-next-line jsx-a11y/no-autofocus -- focus the field the click just opened
        autoFocus
        value={value}
        aria-label={`Name for speaker ${label}`}
        placeholder={`Speaker ${label}`}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => setValue(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            setValue(name ?? "");
            setEditing(false);
          }
        }}
      />
    );
  }

  return (
    <button
      type="button"
      className="transcript-pane__speaker"
      style={{ ["--dot-color" as string]: dotColor }}
      title="Click to rename this speaker"
      onClick={(e) => {
        e.stopPropagation();
        setValue(name ?? "");
        setEditing(true);
      }}
    >
      <span className="transcript-pane__speaker-dot" aria-hidden="true" />
      {name ?? `Speaker ${label}`}
    </button>
  );
}

/**
 * Banner that surfaces uncertain-band voiceprint matches as "Is this
 * \<Name\>?" prompts (#0003 §2.4). Shown only when the matcher places a
 * speaker in the uncertain band (T_reject <= sim < T_accept).
 *
 * Accepting a suggestion writes the name via `setSpeakerName` (same path
 * as the speaker chip rename). Dismissing calls `dismissVoiceprintSuggestion`
 * which calls `reject_match` on the backend to drop that meeting/label
 * contribution from the identity's gallery.
 */
function VoiceprintSuggestionBanner() {
  const suggestions = useMeetingsStore((s) => s.voiceprintSuggestions);
  const setSpeakerName = useMeetingsStore((s) => s.setSpeakerName);
  const dismissVoiceprintSuggestion = useMeetingsStore(
    (s) => s.dismissVoiceprintSuggestion,
  );
  if (suggestions.length === 0) return null;
  return (
    <ul className="transcript-pane__vp-suggestions" aria-label="Speaker identification suggestions">
      {suggestions.map((s) => (
        <li key={`${s.label}-${s.identity_id}`} className="transcript-pane__vp-suggestion">
          <span className="transcript-pane__vp-question">
            Is Speaker {s.label}{" "}
            <strong>{s.display_name}</strong>?
          </span>
          <button
            type="button"
            className="transcript-pane__vp-btn transcript-pane__vp-btn--accept"
            onClick={() => void setSpeakerName(s.label, s.display_name)}
          >
            Yes
          </button>
          <button
            type="button"
            className="transcript-pane__vp-btn transcript-pane__vp-btn--dismiss"
            onClick={() => void dismissVoiceprintSuggestion(s)}
          >
            No
          </button>
        </li>
      ))}
    </ul>
  );
}

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
  // Speaker display-name overlay (label → name) for the open saved meeting,
  // and the writer.
  const speakerNames = useMeetingsStore(
    (s) => s.openMeetingState?.meta?.speaker_names,
  );
  const setSpeakerName = useMeetingsStore((s) => s.setSpeakerName);
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);

  // Translation state.
  const selectedLanguage = useTranslationsStore((s) => s.selectedLanguage);
  const translations = useTranslationsStore((s) => s.translations);
  const setOpenMeeting = useTranslationsStore((s) => s.setOpenMeeting);

  // Notify the translations store when the open meeting changes so it can
  // clear stale translations and re-arm `handleEvent` for the new meeting.
  useEffect(() => {
    setOpenMeeting(openMeetingId);
  }, [openMeetingId, setOpenMeeting]);

  // Renaming is only meaningful — and only persists — on a saved, finalised
  // meeting (open + idle). During a live recording the labels are provisional
  // (re-lettered on stop) and there is no metadata to write, so the chips are
  // display-only. This mirrors `active-transcript`'s saved-vs-live rule.
  const recordingKind = useRecordingStore((s) => s.state.kind);
  const speakerEditable = openMeetingId !== null && recordingKind === "idle";

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
      <VoiceprintSuggestionBanner />
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
                    Phase 6/C: a quiet speaker chip at the start of a speaker's
                    run (see `speakerChanged`). It shows the user-set display
                    name when one exists (`speaker_names[label]`), otherwise the
                    diarizer's first-seen label (A / B / …); clicking it renames
                    the speaker. The colour dot's palette slot is resolved by the
                    pure `speakerColorIndex` mapper and passed via `--dot-color`
                    — tokens only, no hard-coded colour in TSX.
                  */}
                    {speakerChanged && (
                      <SpeakerChip
                        label={seg.speaker_id as string}
                        name={speakerNames?.[seg.speaker_id as string] ?? null}
                        dotColor={dotColor}
                        editable={speakerEditable}
                        onRename={(label, name) =>
                          void setSpeakerName(label, name)
                        }
                      />
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
                    {/*
                    #0002: a quiet marker when the diarizer found this segment
                    spans more than one speaker (it is not split). Count only —
                    naming the co-speakers would clash with the display-name
                    overlay on the chip. Guarded for older/un-diarized segments
                    that predate the field (it is omitted from their JSON).
                  */}
                    {(seg.shared_speakers?.length ?? 0) > 0 && (
                      <span
                        className="transcript-pane__multi-speaker"
                        title="This segment overlaps more than one speaker; it was not split."
                      >
                        {(seg.shared_speakers?.length ?? 0) + 1} speakers
                      </span>
                    )}
                    {/*
                    Translation overlay: when a translated view is active, show
                    the translated text instead of `seg.text`. If this segment
                    has no translation yet (partial pass, gap), fall back to the
                    verbatim text so the row is never blank. A quiet label
                    identifies translated rows to make the substitution explicit.
                  */}
                    {selectedLanguage !== null ? (
                      <>
                        {translations.has(idx)
                          ? translations.get(idx)
                          : seg.text}
                        {translations.has(idx) && (
                          <span
                            className="transcript-pane__translated-label"
                            aria-label={`Translated to ${selectedLanguage}`}
                          >
                            {selectedLanguage}
                          </span>
                        )}
                      </>
                    ) : (
                      seg.text
                    )}
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
