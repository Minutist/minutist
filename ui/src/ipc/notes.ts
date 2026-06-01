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
import { commands } from "./bindings";
import type { NotesDoc as GeneratedNotesDoc } from "./bindings";
import { unwrap } from "./client";

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
