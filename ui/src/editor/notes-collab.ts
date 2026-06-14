/**
 * Local-only Yjs collaboration binding for the notes editor (B6 WU7,
 * `planning/DESIGN_notes-crdt.md` §8).
 *
 * Each open meeting gets a per-meeting `Y.Doc`. The editor binds to it via
 * `@tiptap/extension-collaboration` (which binds at the ProseMirror schema
 * level, so the custom nodes — TranscriptChip atom, NoteImage, the
 * ParagraphAnchor attr — sync like any other node). The document flows through
 * the `Y.Doc`, not `getJSON()` → `save_notes`:
 *
 *  - On open, {@link hydrateNotesDoc} reads the backend's stored state
 *    (`load_notes_ydoc`, a lib0 **v1** update) and applies it with
 *    `Y.applyUpdate`, so the editor starts from the persisted document.
 *  - Every local edit fires the `Y.Doc` `'update'` event; {@link NotesDocSync}
 *    debounces those updates (autosave cadence) and ships the incremental bytes
 *    to `apply_notes_update`, which merges them onto the authoritative
 *    `notes.ydoc` and re-derives `notes.json` / `notes.md`.
 *
 * This is local-only: there is NO network provider (no `y-websocket`,
 * `y-webrtc`, etc.). The only transport is the IPC save seam. The connected
 * build adds replication later; the editor binding is identical (D-O2.2).
 *
 * Encoding: the editor speaks lib0 **v1** (`Y.encodeStateAsUpdate`,
 * `Y.applyUpdate`). The durable on-disk blob is v2, but `persistence` does the
 * v1↔v2 translation — this module only ever handles v1. See `persistence::ydoc`.
 */
import * as Y from "yjs";

/** The top-level XML fragment name the editor binds to. */
//
// Must match `persistence::ydoc::PROSEMIRROR_FRAGMENT` (`"prosemirror"`) so the
// Rust-derived `notes.json` and the editor see the same fragment. This is NOT
// `@tiptap/extension-collaboration`'s default `field` (that is `"default"`) — it
// is passed explicitly as the extension's `field` option so the two sides cannot
// silently desync.
export const NOTES_FRAGMENT = "prosemirror";

/** A fresh, empty `Y.Doc` for a meeting's notes. */
export function createNotesDoc(): Y.Doc {
  return new Y.Doc();
}

/**
 * Hydrate a notes `Y.Doc` from the backend's stored state.
 *
 * `loadState` returns the meeting's `notes.ydoc` as a v1 update (or `null` when
 * the meeting has no CRDT-backed notes yet — the doc is then left empty and the
 * first edit seeds it). The applied update is tagged with a non-local origin so
 * the sync layer can ignore it (hydration must not echo straight back to the
 * backend; see {@link NotesDocSync}).
 */
export async function hydrateNotesDoc(
  doc: Y.Doc,
  loadState: () => Promise<Uint8Array | null>,
): Promise<void> {
  const state = await loadState();
  if (state === null || state.length === 0) return;
  Y.applyUpdate(doc, state, HYDRATE_ORIGIN);
}

/** Transaction origin marking an update applied during hydration. */
export const HYDRATE_ORIGIN = Symbol("notes-collab-hydrate");

/**
 * Debounced sink that ships a meeting's local `Y.Doc` updates to the backend.
 *
 * Subscribes to the doc's `'update'` event and, on a debounce window (the
 * autosave cadence), encodes the doc's whole current state as a v1 update and
 * sends it via `send`. Sending whole-state (rather than accumulating raw update
 * blobs) keeps the call idempotent and order-independent — the backend merges
 * it onto `notes.ydoc` regardless — and lets a flush coalesce a burst of
 * keystrokes into one write.
 *
 * Updates whose transaction origin is {@link HYDRATE_ORIGIN} are ignored: a
 * hydrate is the backend's own state coming in, so echoing it back is wasteful.
 */
export class NotesDocSync {
  private readonly doc: Y.Doc;
  private readonly send: (update: Uint8Array) => void;
  private readonly debounceMs: number;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private dirty = false;
  private disposed = false;
  private readonly onUpdate: (update: Uint8Array, origin: unknown) => void;

  constructor(args: {
    doc: Y.Doc;
    send: (update: Uint8Array) => void;
    debounceMs: number;
  }) {
    this.doc = args.doc;
    this.send = args.send;
    this.debounceMs = args.debounceMs;
    this.onUpdate = (_update: Uint8Array, origin: unknown) => {
      // Ignore the hydrate echo — that update is the backend's own state.
      if (origin === HYDRATE_ORIGIN) return;
      this.dirty = true;
      this.schedule();
    };
    this.doc.on("update", this.onUpdate);
  }

  /** Arm the debounce timer (idempotent within a window). */
  private schedule(): void {
    if (this.timer !== null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.emit();
    }, this.debounceMs);
  }

  /** Encode the doc's current whole state and send it, if dirty. */
  private emit(): void {
    if (this.disposed || !this.dirty) return;
    this.dirty = false;
    this.send(Y.encodeStateAsUpdate(this.doc));
  }

  /**
   * Flush any pending update immediately (wired to editor blur, mirroring the
   * legacy autosave flush). No-op when nothing is pending.
   */
  flush(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.emit();
  }

  /**
   * Detach the listener and flush a final time so a pending edit is not lost
   * when the meeting closes or the editor unmounts.
   */
  dispose(): void {
    if (this.disposed) return;
    this.flush();
    this.disposed = true;
    this.doc.off("update", this.onUpdate);
  }
}
