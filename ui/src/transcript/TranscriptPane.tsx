import { useEffect, useRef } from "react";
import { useRecordingStore } from "../state/recording";
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
 */
export function TranscriptPane() {
  const transcript = useRecordingStore((s) => s.transcript);

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
          {transcript.map((seg: Segment, idx: number) => (
            <li key={idx} className="transcript-pane__row">
              <span className="transcript-pane__timestamp">
                {formatTimestamp(seg.start_ms)}
              </span>
              <span className="transcript-pane__text">{seg.text}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
