/**
 * Thin IPC client for notes persistence.
 *
 * The `save_notes` / `load_notes` Tauri commands are generated into
 * `bindings.ts` by Stream S3 (`commands.saveNotes` / `commands.loadNotes`).
 * This module is the single seam the editor uses to persist notes — it wraps
 * the generated commands so callers work with the same `{ meetingId, notesJson,
 * notesMarkdown }` shape and a `NotesDoc | null` result, and so tests mock
 * *this* module (per the architecture testing policy) rather than the generated
 * bindings file.
 *
 * `persistence` owns the on-disk `notes.json` + `notes.md` files (see
 * `architecture/cross-cutting.md` — Filesystem layout); `ipc-bridge` exposes
 * the two commands, routing them directly to `persistence::NotesStore`.
 */
// Route through `./client` (not `./bindings` directly) so the DEV shim's single
// injection point also covers notes load/save. Importing the raw generated
// `commands` from `./bindings` would bypass the shim, so a `vite dev` browser
// with no Tauri backend hits an undefined `invoke` and the editor can't seed.
import { commands, unwrap } from "./client";
import type { NotesDocument as GeneratedNotesDoc } from "./bindings";

/** A persisted notes document as returned by `load_notes`. */
export type NotesDoc = GeneratedNotesDoc;

/** Payload handed to {@link saveNotes}. */
export type SaveNotesPayload = {
  meetingId: string;
  notesJson: string;
  notesMarkdown: string;
};

/**
 * Persist the current notes for a meeting.
 *
 * Resolves once the backend has written `notes.json` + `notes.md`. Rejects on
 * an IPC error; callers (autosave) swallow the rejection and surface it via the
 * recording store's `lastError` if needed.
 */
export async function saveNotes(payload: SaveNotesPayload): Promise<void> {
  unwrap(
    await commands.saveNotes(
      payload.meetingId,
      payload.notesJson,
      payload.notesMarkdown,
    ),
  );
}

/**
 * Load the persisted notes for a meeting, or `null` if none exist yet.
 */
export async function loadNotes(meetingId: string): Promise<NotesDoc | null> {
  return unwrap(await commands.loadNotes(meetingId));
}

/**
 * Apply an incremental Yjs update from the editor's local `Y.Doc` to the
 * meeting's authoritative `notes.ydoc` (B6 WU7).
 *
 * `update` is a lib0 **v1** update (the bytes from a `Y.Doc` `'update'` event /
 * `Y.encodeStateAsUpdate`), passed as a plain number array — the wire shape the
 * `apply_notes_update` command expects (`Vec<u8>` ↔ `number[]`). The backend
 * merges it onto the stored doc, preserving CRDT history, then re-derives
 * `notes.json` and writes the supplied `notes.md`. This is the editor's primary
 * write path once the collaboration binding is active.
 */
export async function applyNotesUpdate(
  meetingId: string,
  update: Uint8Array,
  notesMarkdown: string,
): Promise<void> {
  unwrap(
    await commands.applyNotesUpdate(
      meetingId,
      Array.from(update),
      notesMarkdown,
    ),
  );
}

/**
 * Read the meeting's stored `notes.ydoc` as a lib0 **v1** state update for the
 * editor to apply with `Y.applyUpdate` on open, or `null` when the meeting has
 * no CRDT-backed notes yet (B6 WU7).
 *
 * The bytes come over the wire as a `number[]`; this returns them as a
 * `Uint8Array` ready for `Y.applyUpdate`.
 */
export async function loadNotesYdoc(
  meetingId: string,
): Promise<Uint8Array | null> {
  const state = unwrap(await commands.loadNotesYdoc(meetingId));
  return state === null ? null : Uint8Array.from(state);
}
