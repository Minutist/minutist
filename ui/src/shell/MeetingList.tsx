/**
 * Meeting-list view (FR-33) — the entry surface before a meeting is open.
 *
 * A folder sidebar (left) filters a quiet index of ruled-paper rows (right):
 * each row is a meeting showing its title (Fraunces), a stone meta line (date ·
 * duration · speaker count), and a transcript excerpt set in reading italic.
 * Opening is the row's primary action — the title is a single click, the text
 * area a double-click, plus an explicit Open button; quiet Move-to / Rename /
 * Delete management actions reveal on hover. Re-processing lives in the opened-
 * meeting view, not here.
 *
 * The view consumes `theme.css` tokens only (no hard-coded colour/type) and
 * renders in the DEV shim with sample meetings + folders for visual QA.
 */
import { useEffect, useState } from "react";
import { useMeetingsStore } from "../state/meetings";
import type { MeetingListEntry } from "../state/meetings";
import {
  useCollectionsStore,
  meetingMatchesFilter,
} from "../state/collections";
import type { Collection, CollectionId } from "../state/collections";
import { CollectionsSidebar } from "./CollectionsSidebar";
import { OperationIndicator } from "./OperationIndicator";
import { writeMeetingDrag } from "./meeting-dnd";
import { ContextMenu } from "./ContextMenu";
import type { ContextMenuEntry } from "./ContextMenu";
import { openMeetingFolder } from "../ipc/meetings";
import "./MeetingList.css";

/** Format an RFC3339 start timestamp as a quiet, readable date. */
export function formatDate(iso: string): string {
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
export function formatSpeakers(count: number): string {
  if (count <= 0) return "no speakers";
  return count === 1 ? "1 speaker" : `${count} speakers`;
}

/**
 * "Move to…" popover: files the meeting into a folder (or Unfiled). A backdrop
 * button closes the menu on an outside click; the current folder is marked.
 */
function MoveMenu(props: {
  current: CollectionId | null | undefined;
  collections: Collection[];
  onMove: (collectionId: CollectionId | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const { current, collections, onMove } = props;

  return (
    <span className="meeting-list__move">
      <button
        type="button"
        className="meeting-list__action"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        Move to…
      </button>
      {open && (
        <>
          <button
            type="button"
            className="meeting-list__move-backdrop"
            aria-label="Close menu"
            tabIndex={-1}
            onClick={() => setOpen(false)}
          />
          <div className="meeting-list__move-menu" role="menu">
            <button
              type="button"
              role="menuitem"
              className="meeting-list__move-item"
              aria-current={!current}
              onClick={() => {
                onMove(null);
                setOpen(false);
              }}
            >
              Unfiled
            </button>
            {collections.map((c) => (
              <button
                key={c.id}
                type="button"
                role="menuitem"
                className="meeting-list__move-item"
                aria-current={current === c.id}
                onClick={() => {
                  onMove(c.id);
                  setOpen(false);
                }}
              >
                {c.name}
              </button>
            ))}
            {collections.length === 0 && (
              <span className="meeting-list__move-empty">No folders yet</span>
            )}
          </div>
        </>
      )}
    </span>
  );
}

type MeetingRowProps = {
  meeting: MeetingListEntry;
  collections: Collection[];
  onOpen: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
  onMove: (collectionId: CollectionId | null) => void;
};

function MeetingRow(props: MeetingRowProps) {
  const { meeting } = props;
  const [renaming, setRenaming] = useState(false);
  const [draftTitle, setDraftTitle] = useState(meeting.title);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);

  // A meeting that was never named (empty title in metadata) would otherwise
  // render an invisible, zero-height heading — the row loses its anchor and
  // leads with the dim meta line. Fall back to a stable placeholder for display
  // (the stored title stays empty, so renaming still starts from blank).
  const displayTitle = meeting.title.trim() || "Untitled meeting";

  function commitRename() {
    const trimmed = draftTitle.trim();
    setRenaming(false);
    if (trimmed && trimmed !== meeting.title) {
      props.onRename(trimmed);
    } else {
      setDraftTitle(meeting.title);
    }
  }

  // Enters inline-rename mode. Shared by the row's Rename button and its
  // context-menu entry so both paths land in the exact same edit state.
  function startRename() {
    setDraftTitle(meeting.title);
    setRenaming(true);
  }

  // Right-click menu entries reuse the same handlers/callbacks the row's own
  // buttons and `MoveMenu` already call — no business logic is duplicated
  // here, only the entry list itself.
  const moveItems = [
    {
      label: "Unfiled",
      current: !meeting.collection_id,
      onSelect: () => props.onMove(null),
    },
    ...props.collections.map((c) => ({
      label: c.name,
      current: meeting.collection_id === c.id,
      onSelect: () => props.onMove(c.id),
    })),
  ];
  const menuEntries: ContextMenuEntry[] = [
    { label: "Open", onSelect: props.onOpen },
    { label: "Rename", onSelect: startRename },
    { label: "Delete", onSelect: props.onDelete, danger: true },
    {
      kind: "submenu",
      label: "Move to…",
      items: moveItems,
      emptyLabel: "No folders yet",
    },
    {
      label: "Open storage folder",
      onSelect: () => {
        void openMeetingFolder(meeting.id).catch((err) => {
          console.error("open_meeting_folder failed", err);
        });
      },
    },
  ];

  return (
    <li
      className="meeting-list__row"
      // Drag the row onto a sidebar folder to file it (a parallel path to the
      // "Move to…" menu). Disabled while renaming so the inline input stays
      // usable. The drop target + the actual move live in CollectionsSidebar.
      draggable={!renaming}
      onDragStart={(e) => {
        if (e.dataTransfer) writeMeetingDrag(e.dataTransfer, meeting.id);
      }}
      onContextMenu={(e) => {
        // Keep the native menu on text-editing controls (the inline rename
        // input) so cut/copy/paste still work there; only this row's own
        // surface gets the themed menu (per-surface suppression, not global).
        if ((e.target as HTMLElement).closest("input, textarea")) return;
        e.preventDefault();
        setMenu({ x: e.clientX, y: e.clientY });
      }}
    >
      {/* Double-click anywhere in the meeting's text opens it (the row's
          primary action); the title is also a single-click open. Bound to the
          main area (not the whole row) so double-clicking the quiet management
          actions doesn't also open. */}
      <div
        className="meeting-list__main"
        onDoubleClick={props.onOpen}
        title="Double-click to open"
      >
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
            className={
              meeting.title.trim()
                ? "meeting-list__title"
                : "meeting-list__title meeting-list__title--placeholder"
            }
            onClick={props.onOpen}
          >
            {displayTitle}
          </button>
        )}
        {meeting.recording_started === false && (
          <span className="meeting-list__draft-chip">Draft</span>
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

        {/* Live-test UX T3/T4: a non-blocking per-row indicator for any
            background pass (re-transcribe / re-identify-speakers / summarise)
            running on this meeting. Self-hides when nothing is in flight. */}
        <OperationIndicator meetingId={meeting.id} />
      </div>

      <div className="meeting-list__actions">
        <button
          type="button"
          className="meeting-list__action meeting-list__action--primary"
          onClick={props.onOpen}
        >
          Open
        </button>
        <MoveMenu
          current={meeting.collection_id}
          collections={props.collections}
          onMove={props.onMove}
        />
        <button
          type="button"
          className="meeting-list__action"
          onClick={startRename}
        >
          Rename
        </button>
        <button
          type="button"
          className="meeting-list__action meeting-list__action--danger"
          onClick={props.onDelete}
        >
          Delete
        </button>
      </div>

      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          entries={menuEntries}
          onClose={() => setMenu(null)}
        />
      )}
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
  const setCollection = useMeetingsStore((s) => s.setCollection);

  const collections = useCollectionsStore((s) => s.collections);
  const filter = useCollectionsStore((s) => s.filter);
  const refreshCollections = useCollectionsStore((s) => s.refresh);

  useEffect(() => {
    void refresh();
    void refreshCollections();
  }, [refresh, refreshCollections]);

  const [query, setQuery] = useState("");

  const inFolder = meetings.filter((m) =>
    meetingMatchesFilter(filter, m.collection_id),
  );

  // Client-side title/excerpt search over the folder-filtered set. Case-
  // insensitive substring; empty query is a no-op passthrough.
  const q = query.trim().toLowerCase();
  const visible = q
    ? inFolder.filter(
        (m) =>
          m.title.toLowerCase().includes(q) ||
          (m.excerpt?.toLowerCase().includes(q) ?? false),
      )
    : inFolder;

  // Distinguish "nothing recorded yet" (whole index empty) from "this folder is
  // empty" and from "no search match" so the empty copy is honest.
  const emptyMessage =
    meetings.length === 0
      ? loading
        ? "Loading meetings…"
        : "No meetings yet. Start a recording to begin."
      : q
        ? `No meetings match “${query.trim()}”.`
        : "No meetings in this folder.";

  // Count reads as a plain total normally, or "shown of total" while searching.
  const countLabel = q
    ? `${visible.length} of ${inFolder.length}`
    : `${inFolder.length} ${inFolder.length === 1 ? "meeting" : "meetings"}`;

  return (
    <section className="meeting-list ink-reveal" aria-label="Meetings">
      <header className="meeting-list__header">
        <h1 className="meeting-list__heading">Meetings</h1>
        <div className="meeting-list__tools">
          <input
            type="search"
            className="meeting-list__search"
            placeholder="Search meetings…"
            aria-label="Search meetings"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <span className="meeting-list__count tnum" aria-live="polite">
            {countLabel}
          </span>
        </div>
      </header>

      <div className="meeting-list__body">
        <CollectionsSidebar />

        <div className="meeting-list__rows-col">
          {visible.length === 0 ? (
            <p className="meeting-list__empty">{emptyMessage}</p>
          ) : (
            <ol className="meeting-list__rows">
              {visible.map((meeting) => (
                <MeetingRow
                  key={meeting.id}
                  meeting={meeting}
                  collections={collections}
                  onOpen={() => void open(meeting.id)}
                  onRename={(title) => void rename(meeting.id, title)}
                  onDelete={() => void remove(meeting.id)}
                  onMove={(collectionId) =>
                    void setCollection(meeting.id, collectionId)
                  }
                />
              ))}
            </ol>
          )}
        </div>
      </div>
    </section>
  );
}
