/**
 * Editable meeting title for the meeting-screen masthead.
 *
 * When a finalised meeting is open, the topbar shows its name here and lets the
 * user rename it WITHOUT returning to the home screen. The title is read from
 * the meetings-list entry (the canonical source the home list also renders) so a
 * rename here and a rename on the home screen stay consistent; editing commits
 * through the same `useMeetingsStore.rename` seam. Click the title to edit;
 * Enter / blur commits, Escape cancels.
 */
import { useEffect, useState } from "react";
import { useMeetingsStore } from "../state/meetings";
import type { MeetingId } from "../ipc/bindings";

export function MeetingTitle({ meetingId }: { meetingId: MeetingId }) {
  // The canonical title from the list entry (null when the meeting is not in the
  // list yet — e.g. mid-finalise — in which case render nothing).
  const title = useMeetingsStore(
    (s) => s.meetings.find((m) => m.id === meetingId)?.title ?? null,
  );
  const rename = useMeetingsStore((s) => s.rename);

  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title ?? "");

  // Keep the draft synced to the canonical title while NOT editing (a list
  // refresh, or switching to another open meeting).
  useEffect(() => {
    if (!editing) setDraft(title ?? "");
  }, [title, editing]);

  if (title === null) return null;

  function commit() {
    const trimmed = draft.trim();
    setEditing(false);
    if (trimmed && trimmed !== title) {
      void rename(meetingId, trimmed);
    } else {
      setDraft(title ?? "");
    }
  }

  return editing ? (
    <input
      className="main-window__meeting-title-input"
      aria-label="Meeting title"
      value={draft}
      autoFocus
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") {
          setDraft(title);
          setEditing(false);
        }
      }}
    />
  ) : (
    <button
      type="button"
      className="main-window__meeting-title"
      title="Click to rename this meeting"
      onClick={() => setEditing(true)}
    >
      {title}
    </button>
  );
}
