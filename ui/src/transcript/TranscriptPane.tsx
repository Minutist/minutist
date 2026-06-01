import { useEffect, useRef } from "react";
import { useRecordingStore } from "../state/recording";
import { useCrossRefStore } from "../state/cross-ref";
import { writeSegmentDrag } from "../editor/transcript-dnd";
import type { Segment } from "../ipc/bindings";
import "./TranscriptPane.css";

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
 *   - FR-22: when a notes paragraph is hovered, the segment nearest its anchor
 *     is highlighted (`highlightedSegmentIndex` from the cross-ref store).
 */
export function TranscriptPane() {
  const transcript = useRecordingStore((s) => s.transcript);
  const highlightedSegmentIndex = useCrossRefStore(
    (s) => s.highlightedSegmentIndex,
  );
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
    <div
      className="transcript-pane"
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
            const highlighted = idx === highlightedSegmentIndex;
            return (
              <li
                key={idx}
                className={
                  highlighted
                    ? "transcript-pane__row transcript-pane__row--highlighted"
                    : "transcript-pane__row"
                }
                draggable
                aria-current={highlighted ? "true" : undefined}
                title="Drag into your notes, or click to jump to the linked paragraph"
                onDragStart={(e) => {
                  if (e.dataTransfer) writeSegmentDrag(e.dataTransfer, seg);
                }}
                onClick={() => clickTranscriptSegment(seg)}
              >
                <span className="transcript-pane__timestamp tnum">
                  {formatTimestamp(seg.start_ms)}
                </span>
                <span className="transcript-pane__text">{seg.text}</span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
