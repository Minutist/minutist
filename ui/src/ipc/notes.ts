/**
 * Thin IPC client for notes persistence.
 *
 * The `save_notes` / `load_notes` Tauri commands are NOT yet present in the
 * generated `bindings.ts` — they are added on the backend by Stream S3, which
 * regenerates `bindings.ts` at integration time. Until then this module is the
 * single seam the editor uses to persist notes, and tests mock *this* module
 * (per the architecture testing policy) rather than faking the generated
 * bindings file.
 *
 * Wire contract assumed for S3 (documented for the integrator):
 *
 *   #[tauri::command]
 *   async fn save_notes(
 *       meeting_id: MeetingId,   // hyphenated-lowercase UUID string
 *       notes_json: String,      // serialised Tiptap ProseMirror document JSON
 *       notes_markdown: String,  // markdown export (for summariser / Word paste)
 *   ) -> Result<(), IpcError>;
 *
 *   #[tauri::command]
 *   async fn load_notes(
 *       meeting_id: MeetingId,
 *   ) -> Result<Option<NotesDoc>, IpcError>;
 *
 * where `NotesDoc { notes_json: String, notes_markdown: String }` (a `null`
 * result means "no notes saved yet for this meeting").
 *
 * `persistence` owns the on-disk `notes.json` + `notes.md` files (see
 * `architecture/cross-cutting.md` — Filesystem layout); `ipc-bridge` exposes
 * the two commands. Once S3 regenerates `bindings.ts`, this module is rewired
 * to call `commands.saveNotes` / `commands.loadNotes` and the dynamic invoke
 * below is removed.
 */
import { invoke } from "@tauri-apps/api/core";

/** A persisted notes document as returned by `load_notes`. */
export type NotesDoc = {
  notes_json: string;
  notes_markdown: string;
};

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
  await invoke("save_notes", {
    meetingId: payload.meetingId,
    notesJson: payload.notesJson,
    notesMarkdown: payload.notesMarkdown,
  });
}

/**
 * Load the persisted notes for a meeting, or `null` if none exist yet.
 */
export async function loadNotes(meetingId: string): Promise<NotesDoc | null> {
  return (await invoke("load_notes", { meetingId })) as NotesDoc | null;
}
