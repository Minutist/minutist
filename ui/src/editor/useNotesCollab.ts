/**
 * React hook owning the per-meeting Yjs document lifecycle for the notes editor
 * (B6 WU7).
 *
 * Returns the `Y.Doc` the editor should bind to for the current meeting, plus a
 * `flush()` the caller wires to editor blur. The hook:
 *
 *  - creates a fresh `Y.Doc` whenever the meeting id changes (so each meeting's
 *    notes are an independent document),
 *  - hydrates it from the backend's stored `notes.ydoc` (`load_notes_ydoc`),
 *  - drives a {@link NotesDocSync} that debounces local edits and ships them to
 *    `apply_notes_update` (the autosave cadence),
 *  - disposes the sync (final flush) when the meeting changes or on unmount.
 *
 * `null` meeting id → no doc (the live entry surface with nothing open); the
 * editor then has no collaboration binding and the legacy JSON path applies.
 *
 * The markdown rendering is supplied lazily by the caller (`getMarkdown`) so the
 * sync writes `notes.md` alongside the update — rendering it needs the editor's
 * typed schema, which this hook does not own.
 */
import { useEffect, useRef, useState } from "react";
import * as Y from "yjs";
import { createNotesDoc, hydrateNotesDoc, NotesDocSync } from "./notes-collab";
import { applyNotesUpdate, loadNotesYdoc } from "../ipc/notes";
import { DEFAULT_AUTOSAVE_INTERVAL_SECS } from "./useAutosave";

export type UseNotesCollabArgs = {
  /** The meeting whose notes the editor is editing, or `null` when none. */
  meetingId: string | null;
  /** Autosave cadence in seconds; falls back to the default when null. */
  intervalSecs: number | null;
  /** Renders the current notes markdown (`notes.md`) for the update write. */
  getMarkdown: () => string;
  /** Optional error sink; receives a rejected `apply_notes_update`. */
  onError?: (err: unknown) => void;
};

export type UseNotesCollabResult = {
  /** The Y.Doc for the current meeting, or `null` when no meeting is open. */
  doc: Y.Doc | null;
  /** Flush any pending update immediately (wire to editor blur). */
  flush: () => void;
};

/** Manage the per-meeting Y.Doc + its debounced backend sync. */
export function useNotesCollab(args: UseNotesCollabArgs): UseNotesCollabResult {
  const { meetingId, intervalSecs, getMarkdown, onError } = args;

  // The live doc, exposed as state so a meeting change recreates the editor
  // (the editor's `useEditor` deps include this doc identity).
  const [doc, setDoc] = useState<Y.Doc | null>(null);
  const syncRef = useRef<NotesDocSync | null>(null);

  // Keep the latest markdown/error callbacks in refs so the per-meeting effect
  // does not re-run (and re-create the doc) on every keystroke.
  const getMarkdownRef = useRef(getMarkdown);
  const onErrorRef = useRef(onError);
  getMarkdownRef.current = getMarkdown;
  onErrorRef.current = onError;

  const debounceMs =
    (intervalSecs && intervalSecs > 0
      ? intervalSecs
      : DEFAULT_AUTOSAVE_INTERVAL_SECS) * 1000;

  useEffect(() => {
    if (meetingId === null) {
      setDoc(null);
      return;
    }

    let cancelled = false;
    const next = createNotesDoc();

    const sync = new NotesDocSync({
      doc: next,
      debounceMs,
      send: (update) => {
        applyNotesUpdate(meetingId, update, getMarkdownRef.current()).catch(
          (err: unknown) => onErrorRef.current?.(err),
        );
      },
    });
    syncRef.current = sync;

    // Hydrate from the stored state, THEN expose the doc so the editor binds to
    // an already-populated document (avoids a flash of empty content, and the
    // hydrate update is tagged so the sync ignores it). A hydrate failure leaves
    // the doc empty rather than throwing — the editor still opens.
    void hydrateNotesDoc(next, () => loadNotesYdoc(meetingId))
      .catch((err: unknown) => onErrorRef.current?.(err))
      .finally(() => {
        if (!cancelled) setDoc(next);
      });

    return () => {
      cancelled = true;
      sync.dispose();
      if (syncRef.current === sync) syncRef.current = null;
      next.destroy();
    };
    // `debounceMs` intentionally excluded: changing the cadence must not tear
    // down and re-create the meeting's doc mid-edit. New cadence applies to the
    // next meeting open. Only the meeting identity drives a doc swap.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meetingId]);

  return {
    doc,
    flush: () => syncRef.current?.flush(),
  };
}
