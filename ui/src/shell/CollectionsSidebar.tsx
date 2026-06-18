/**
 * Home-screen folder sidebar.
 *
 * Lists "All meetings", each user folder (with a count), and "Unfiled", and
 * lets the user create / rename / delete folders. Selecting an entry sets the
 * active filter in the collections store; the meeting list reads that filter to
 * show only the matching meetings. Counts are derived from the meetings store's
 * list (each `MeetingListEntry.collection_id`), so they update when a meeting is
 * filed or a list refresh lands.
 */
import { useState, type DragEvent } from "react";
import { useMeetingsStore } from "../state/meetings";
import {
  useCollectionsStore,
  ALL_FILTER,
  UNFILED_FILTER,
  type CollectionFilter,
  type CollectionId,
} from "../state/collections";
import { hasMeetingDrag, readMeetingDrag } from "./meeting-dnd";
import "./CollectionsSidebar.css";

/** Whether two filters select the same entry (for the active highlight). */
function sameFilter(a: CollectionFilter, b: CollectionFilter): boolean {
  if (a.kind === "collection" && b.kind === "collection") return a.id === b.id;
  return a.kind === b.kind;
}

export function CollectionsSidebar() {
  const meetings = useMeetingsStore((s) => s.meetings);
  const collections = useCollectionsStore((s) => s.collections);
  const filter = useCollectionsStore((s) => s.filter);
  const select = useCollectionsStore((s) => s.select);
  const create = useCollectionsStore((s) => s.create);
  const rename = useCollectionsStore((s) => s.rename);
  const remove = useCollectionsStore((s) => s.remove);
  // Filing a dragged meeting reuses the same seam as the row's "Move to…" menu.
  const setCollection = useMeetingsStore((s) => s.setCollection);

  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  // The folder entry currently under a meeting drag (for the drop highlight).
  const [dragOverKey, setDragOverKey] = useState<string | null>(null);

  const total = meetings.length;
  const unfiledCount = meetings.filter((m) => !m.collection_id).length;
  const countFor = (id: string) =>
    meetings.filter((m) => m.collection_id === id).length;

  function commitCreate() {
    const name = draft.trim();
    setCreating(false);
    setDraft("");
    if (name) void create(name);
  }

  function commitRename(id: string) {
    const name = renameDraft.trim();
    setRenamingId(null);
    if (name) void rename(id, name);
  }

  // Drop handlers for filing a dragged meeting onto a folder (`target` = the
  // collection id, or `null` for "Unfiled"). `key` identifies the entry for the
  // drop highlight. Only accepts a meeting drag (`hasMeetingDrag`).
  function dropProps(key: string, target: CollectionId | null) {
    return {
      onDragOver: (e: DragEvent) => {
        if (!hasMeetingDrag(e.dataTransfer)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        if (dragOverKey !== key) setDragOverKey(key);
      },
      onDragLeave: () => setDragOverKey((k) => (k === key ? null : k)),
      onDrop: (e: DragEvent) => {
        e.preventDefault();
        setDragOverKey(null);
        const id = readMeetingDrag(e.dataTransfer);
        if (id) void setCollection(id, target);
      },
    };
  }

  return (
    <nav className="collections-sidebar" aria-label="Folders">
      <div className="collections-sidebar__head">
        <h2 className="collections-sidebar__title">Folders</h2>
        <button
          type="button"
          className="collections-sidebar__new"
          aria-label="New folder"
          title="New folder"
          onClick={() => {
            setDraft("");
            setCreating(true);
          }}
        >
          +
        </button>
      </div>

      <ul className="collections-sidebar__list">
        <li>
          <button
            type="button"
            className={
              "collections-sidebar__item" +
              (sameFilter(filter, ALL_FILTER)
                ? " collections-sidebar__item--active"
                : "")
            }
            onClick={() => select(ALL_FILTER)}
          >
            <span className="collections-sidebar__label">All meetings</span>
            <span className="collections-sidebar__count tnum">{total}</span>
          </button>
        </li>

        {collections.map((c) => (
          <li key={c.id} className="collections-sidebar__folder-row">
            {renamingId === c.id ? (
              <input
                className="collections-sidebar__rename-input"
                aria-label="Folder name"
                value={renameDraft}
                autoFocus
                onChange={(e) => setRenameDraft(e.target.value)}
                onBlur={() => commitRename(c.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitRename(c.id);
                  if (e.key === "Escape") setRenamingId(null);
                }}
              />
            ) : (
              <>
                <button
                  type="button"
                  className={
                    "collections-sidebar__item" +
                    (sameFilter(filter, { kind: "collection", id: c.id })
                      ? " collections-sidebar__item--active"
                      : "") +
                    (dragOverKey === c.id
                      ? " collections-sidebar__item--drop-target"
                      : "")
                  }
                  onClick={() => select({ kind: "collection", id: c.id })}
                  {...dropProps(c.id, c.id)}
                >
                  <span className="collections-sidebar__label">{c.name}</span>
                  <span className="collections-sidebar__count tnum">
                    {countFor(c.id)}
                  </span>
                </button>
                <span className="collections-sidebar__folder-actions">
                  <button
                    type="button"
                    className="collections-sidebar__action"
                    aria-label={`Rename folder ${c.name}`}
                    onClick={() => {
                      setRenameDraft(c.name);
                      setRenamingId(c.id);
                    }}
                  >
                    Rename
                  </button>
                  <button
                    type="button"
                    className="collections-sidebar__action collections-sidebar__action--danger"
                    aria-label={`Delete folder ${c.name}`}
                    onClick={() => void remove(c.id)}
                  >
                    Delete
                  </button>
                </span>
              </>
            )}
          </li>
        ))}

        <li>
          <button
            type="button"
            className={
              "collections-sidebar__item" +
              (sameFilter(filter, UNFILED_FILTER)
                ? " collections-sidebar__item--active"
                : "") +
              (dragOverKey === "unfiled"
                ? " collections-sidebar__item--drop-target"
                : "")
            }
            onClick={() => select(UNFILED_FILTER)}
            {...dropProps("unfiled", null)}
          >
            <span className="collections-sidebar__label collections-sidebar__label--muted">
              Unfiled
            </span>
            <span className="collections-sidebar__count tnum">
              {unfiledCount}
            </span>
          </button>
        </li>

        {creating && (
          <li>
            <input
              className="collections-sidebar__rename-input"
              aria-label="New folder name"
              placeholder="Folder name"
              value={draft}
              autoFocus
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitCreate}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitCreate();
                if (e.key === "Escape") {
                  setCreating(false);
                  setDraft("");
                }
              }}
            />
          </li>
        )}
      </ul>
    </nav>
  );
}
