/**
 * Meeting-screen masthead — the open meeting's headline.
 *
 * On the meeting screen the meeting's name is the page's headline: a large
 * Fraunces title with an explicit edit affordance (pencil), sitting above a
 * stone "dateline" (date · duration · speakers). It is the focal element of the
 * screen so it reads as the meeting's identity, not as app chrome.
 *
 * A freshly-recorded meeting carries the orchestrator's default
 * `Recording <ISO timestamp>` name until the user gives it one; that default is
 * shown as a muted "Untitled meeting" placeholder (and rename starts from an
 * empty field) so it is obvious the meeting still wants a name. Click the title
 * to rename — Enter / blur commits, Escape cancels — through the same
 * `useMeetingsStore.rename` seam the home list uses, so the two stay consistent.
 */
import { useEffect, useState } from "react";
import { useMeetingsStore } from "../state/meetings";
import type { MeetingId } from "../ipc/bindings";
import { formatDate, formatDuration, formatSpeakers } from "./MeetingList";
import "./MeetingMasthead.css";

/**
 * Whether a title is still a placeholder rather than a user-chosen name: the
 * orchestrator's auto-generated default (`Recording <ISO timestamp>`, set at
 * `start`), or a "New meeting" prep draft's empty title (set at
 * `create_meeting`, before the user has typed one).
 */
export function isDefaultMeetingTitle(title: string): boolean {
  return title.trim() === "" || /^Recording \d{4}-\d{2}-\d{2}T/.test(title);
}

function PencilIcon() {
  return (
    <svg
      className="meeting-masthead__edit-icon"
      viewBox="0 0 16 16"
      width="15"
      height="15"
      aria-hidden="true"
    >
      <path
        d="M10.8 2.4l2.8 2.8L6 12.8l-3.2.4.4-3.2 7.6-7.6z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function MeetingMasthead({ meetingId }: { meetingId: MeetingId }) {
  // The canonical list entry (same source the home list renders), so a rename
  // here and a rename there stay consistent.
  const entry = useMeetingsStore(
    (s) => s.meetings.find((m) => m.id === meetingId) ?? null,
  );
  const rename = useMeetingsStore((s) => s.rename);

  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  const title = entry?.title ?? null;

  // Keep the draft synced to the canonical title while NOT editing (a list
  // refresh, or switching to another open meeting).
  useEffect(() => {
    if (!editing) setDraft(title ?? "");
  }, [title, editing]);

  // No list entry yet (e.g. mid-finalise, or the workspace opened before the
  // list refreshed) — nothing to show.
  if (entry === null || title === null) return null;

  const isDefault = isDefaultMeetingTitle(title);

  function beginEdit() {
    // Start from an EMPTY field for an un-named meeting so the user types a name
    // over the placeholder rather than editing the auto timestamp.
    setDraft(isDefault ? "" : (title ?? ""));
    setEditing(true);
  }

  function commit() {
    const trimmed = draft.trim();
    setEditing(false);
    if (trimmed && trimmed !== title) {
      void rename(meetingId, trimmed);
    } else {
      setDraft(title ?? "");
    }
  }

  return (
    <div className="meeting-masthead">
      {editing ? (
        <input
          className="meeting-masthead__title-input"
          aria-label="Meeting title"
          placeholder="Name this meeting"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") {
              setDraft(title ?? "");
              setEditing(false);
            }
          }}
        />
      ) : (
        <button
          type="button"
          className="meeting-masthead__title-btn"
          title="Click to rename this meeting"
          aria-label={isDefault ? "Name this meeting" : "Rename this meeting"}
          onClick={beginEdit}
        >
          <span
            className={
              isDefault
                ? "meeting-masthead__title meeting-masthead__title--placeholder"
                : "meeting-masthead__title"
            }
          >
            {isDefault ? "Untitled meeting" : title}
          </span>
          <PencilIcon />
        </button>
      )}

      <p className="meeting-masthead__dateline tnum">
        <span>{formatDate(entry.started_at)}</span>
        <span className="meeting-masthead__dot" aria-hidden="true">
          ·
        </span>
        <span>{formatDuration(entry.duration_ms)}</span>
        <span className="meeting-masthead__dot" aria-hidden="true">
          ·
        </span>
        <span>{formatSpeakers(entry.speaker_count)}</span>
      </p>
    </div>
  );
}
