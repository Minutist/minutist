/**
 * Meeting-list view (FR-33) — the entry surface before a meeting is open.
 *
 * A quiet index of ruled paper rows in the Editorial Ink language: each row is
 * a meeting showing its title (Fraunces), a stone meta line (date · duration ·
 * speaker count), and a transcript excerpt set in reading italic. Hovering a row
 * reveals its actions: open / rename / delete / re-transcribe / re-summarise.
 *
 * The view consumes `theme.css` tokens only (no hard-coded colour/type) and
 * renders in the DEV shim with sample meetings for visual QA.
 */
import { useEffect, useState } from "react";
import { useMeetingsStore } from "../state/meetings";
import type { MeetingListEntry } from "../state/meetings";
import "./MeetingList.css";

/** Format an RFC3339 start timestamp as a quiet, readable date. */
function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Format a duration (ms) as `H:MM` / `M min` for the meta line. */
export function formatDuration(durationMs: number): string {
  const totalMinutes = Math.round(durationMs / 60_000);
  if (totalMinutes < 60) return `${totalMinutes} min`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return `${hours}:${String(minutes).padStart(2, "0")}`;
}

/** Pluralise the speaker count for the meta line. */
function formatSpeakers(count: number): string {
  if (count <= 0) return "no speakers";
  return count === 1 ? "1 speaker" : `${count} speakers`;
}

type MeetingRowProps = {
  meeting: MeetingListEntry;
  onOpen: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
  onReTranscribe: () => void;
  onReSummarise: () => void;
};

function MeetingRow(props: MeetingRowProps) {
  const { meeting } = props;
  const [renaming, setRenaming] = useState(false);
  const [draftTitle, setDraftTitle] = useState(meeting.title);

  function commitRename() {
    const trimmed = draftTitle.trim();
    setRenaming(false);
    if (trimmed && trimmed !== meeting.title) {
      props.onRename(trimmed);
    } else {
      setDraftTitle(meeting.title);
    }
  }

  return (
    <li className="meeting-list__row">
      <div className="meeting-list__main">
        {renaming ? (
          <input
            className="meeting-list__title-input"
            aria-label="Meeting title"
            value={draftTitle}
            autoFocus
            onChange={(e) => setDraftTitle(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") {
                setDraftTitle(meeting.title);
                setRenaming(false);
              }
            }}
          />
        ) : (
          <button
            type="button"
            className="meeting-list__title"
            onClick={props.onOpen}
          >
            {meeting.title}
          </button>
        )}

        <p className="meeting-list__meta tnum">
          <span>{formatDate(meeting.started_at)}</span>
          <span className="meeting-list__meta-dot" aria-hidden="true">
            ·
          </span>
          <span>{formatDuration(meeting.duration_ms)}</span>
          <span className="meeting-list__meta-dot" aria-hidden="true">
            ·
          </span>
          <span>{formatSpeakers(meeting.speaker_count)}</span>
        </p>

        {meeting.excerpt ? (
          <p className="meeting-list__excerpt">{meeting.excerpt}</p>
        ) : null}
      </div>

      <div className="meeting-list__actions">
        <button
          type="button"
          className="meeting-list__action meeting-list__action--primary"
          onClick={props.onOpen}
        >
          Open
        </button>
        <button
          type="button"
          className="meeting-list__action"
          onClick={() => {
            setDraftTitle(meeting.title);
            setRenaming(true);
          }}
        >
          Rename
        </button>
        <button
          type="button"
          className="meeting-list__action"
          onClick={props.onReTranscribe}
        >
          Re-transcribe
        </button>
        <button
          type="button"
          className="meeting-list__action"
          onClick={props.onReSummarise}
        >
          Re-summarise
        </button>
        <button
          type="button"
          className="meeting-list__action meeting-list__action--danger"
          onClick={props.onDelete}
        >
          Delete
        </button>
      </div>
    </li>
  );
}

export function MeetingList() {
  const meetings = useMeetingsStore((s) => s.meetings);
  const loading = useMeetingsStore((s) => s.loading);
  const refresh = useMeetingsStore((s) => s.refresh);
  const open = useMeetingsStore((s) => s.open);
  const rename = useMeetingsStore((s) => s.rename);
  const remove = useMeetingsStore((s) => s.remove);
  const reTranscribe = useMeetingsStore((s) => s.reTranscribe);
  const reSummarise = useMeetingsStore((s) => s.reSummarise);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <section className="meeting-list ink-reveal" aria-label="Meetings">
      <header className="meeting-list__header">
        <h1 className="meeting-list__heading">Meetings</h1>
        <p className="meeting-list__subhead">
          A quiet index of everything you've recorded.
        </p>
      </header>

      {meetings.length === 0 ? (
        <p className="meeting-list__empty">
          {loading
            ? "Loading meetings…"
            : "No meetings yet. Start a recording to begin."}
        </p>
      ) : (
        <ol className="meeting-list__rows">
          {meetings.map((meeting) => (
            <MeetingRow
              key={meeting.id}
              meeting={meeting}
              onOpen={() => void open(meeting.id)}
              onRename={(title) => void rename(meeting.id, title)}
              onDelete={() => void remove(meeting.id)}
              onReTranscribe={() => void reTranscribe(meeting.id)}
              onReSummarise={() => void reSummarise(meeting.id)}
            />
          ))}
        </ol>
      )}
    </section>
  );
}
