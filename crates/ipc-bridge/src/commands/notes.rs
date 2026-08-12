//! Notes persistence commands (Phase 3): the Tiptap/ProseMirror notes document, its CRDT
//! update stream, and note-image attachments. Command bodies are factored out from their
//! `#[tauri::command]` wrappers so they are unit-testable without a running Tauri runtime.
use super::*;


// The notes wire type returned by [`load_notes`] is `common::NotesDocument`
// (`notes_json` + `notes_markdown`, both `String`). It carries the
// Tiptap/ProseMirror document as a `String`, not a `serde_json::Value`: a bare
// `serde_json::Value` does not derive `specta::Type`, so it cannot cross the
// tauri-specta boundary directly. The webview owns the (de)serialisation of this
// opaque document; `persistence` stores it verbatim (the Phase-4 transcript-chip
// opacity guarantee). The `String`-over-the-wire choice keeps the IPC contract
// typed without forcing a Rust-side Tiptap model.
//
// `ipc-bridge` re-uses `common::NotesDocument` directly rather than mirroring a
// local copy: a duplicate would emit a second, identical TypeScript type into
// `bindings.ts` (the `NotesDoc`/`NotesDocument` divergence #19, removed here).

/// Persist a meeting's notes (`notes.json` + `notes.md`).
///
/// Routes **directly** to `notes_crdt::NotesStore` against
/// `IpcState::meetings_dir` — notes I/O is independent of the live recording
/// pipeline (see `architecture/components.md`, `persistence` "Phase 3 surface
/// growth — notes"), so the orchestrator is not involved. The blocking
/// filesystem write runs on `spawn_blocking` per the threading model.
///
/// `notes_json` is parsed from a `String` into a `serde_json::Value`; an
/// invalid JSON string is rejected as `AppError::InvalidInput`.
#[tauri::command]
#[specta::specta]
pub async fn save_notes(
    meeting_id: MeetingId,
    notes_json: String,
    notes_markdown: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        save_notes_inner(&meetings_dir, meeting_id, &notes_json, &notes_markdown)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("save_notes task join failed: {e}"),
    })?
}

/// Load a meeting's persisted notes, or `None` when no notes have been saved.
///
/// Routes directly to `notes_crdt::NotesStore`; the loaded opaque
/// `serde_json::Value` is re-serialised back to a `String` for the wire (see
/// [`NotesDocument`]).
#[tauri::command]
#[specta::specta]
pub async fn load_notes(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<Option<NotesDocument>> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || load_notes_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("load_notes task join failed: {e}"),
        })?
}

/// Apply an incremental Yjs update from the editor's local `Y.Doc` (the
/// `'update'` event) onto the meeting's authoritative `notes.ydoc`, then
/// re-derive `notes.json` + write the caller-supplied `notes.md`.
///
/// This is the **primary write path for an open editor** (D-O2.1): with
/// `@tiptap/extension-collaboration` the editor is Yjs-native, so its edits
/// arrive as CRDT updates that MERGE onto the stored doc — preserving the CRDT
/// history that the JSON-rebuild `save_notes` discards. `update` is a lib0 **v1**
/// update (the format the JS `yjs` library emits); the wire type is `Vec<u8>`,
/// exported as `number[]` (matching `save_note_image`'s `bytes` — no base64
/// hop). Routes directly to `notes_crdt::NotesStore::apply_update` on
/// `spawn_blocking`, mirroring `save_notes`.
#[tauri::command]
#[specta::specta]
pub async fn apply_notes_update(
    meeting_id: MeetingId,
    update: Vec<u8>,
    notes_markdown: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        NotesStore::apply_update(&meetings_dir, meeting_id, &update, &notes_markdown)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("apply_notes_update task join failed: {e}"),
    })?
}

/// Read the meeting's current `notes.ydoc` state as a lib0 **v1** update for the
/// editor to apply with `Y.applyUpdate` on open.
///
/// Returns `None` when the meeting has no `notes.ydoc` (the editor then starts
/// empty and its first edit seeds the doc). The wire type is `Option<Vec<u8>>`,
/// exported as `number[] | null`. The stored blob is v2 (durable); persistence
/// re-encodes it as v1 because the JS `yjs` library only accepts v1 over
/// `applyUpdate` (the v1/v2 hops must not be crossed — see
/// `notes_crdt::ydoc`). Routes directly to
/// `notes_crdt::NotesStore::read_ydoc_state` on `spawn_blocking`.
#[tauri::command]
#[specta::specta]
pub async fn load_notes_ydoc(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<Option<Vec<u8>>> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || NotesStore::read_ydoc_state(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("load_notes_ydoc task join failed: {e}"),
        })?
}

// ---------------------------------------------------------------------------
// Notes command bodies — extracted so they can be unit-tested without a
// running Tauri runtime (the round-trip test calls these directly).
// ---------------------------------------------------------------------------

/// Inner body of [`save_notes`]: parse the JSON string and write via
/// `NotesStore`. Returns `AppError` so both the command and the unit test
/// share one error path.
pub(crate) fn save_notes_inner(
    meetings_dir: &std::path::Path,
    meeting_id: MeetingId,
    notes_json: &str,
    notes_markdown: &str,
) -> Result<(), AppError> {
    let value: serde_json::Value =
        serde_json::from_str(notes_json).map_err(|e| AppError::InvalidInput {
            context: format!("notes_json is not valid JSON: {e}"),
        })?;
    NotesStore::save(meetings_dir, meeting_id, &value, notes_markdown)
}

/// Inner body of [`load_notes`]: read via `NotesStore` and re-serialise the
/// opaque document back to a `String` for the wire.
pub(crate) fn load_notes_inner(
    meetings_dir: &std::path::Path,
    meeting_id: MeetingId,
) -> Result<Option<NotesDocument>, AppError> {
    let loaded = NotesStore::load(meetings_dir, meeting_id)?;
    match loaded {
        None => Ok(None),
        Some(data) => {
            let notes_json = serde_json::to_string(&data.json).map_err(|e| AppError::Internal {
                context: format!("failed to re-serialise loaded notes.json: {e}"),
            })?;
            Ok(Some(NotesDocument {
                notes_json,
                notes_markdown: data.markdown,
            }))
        }
    }
}

/// The image extensions a pasted/dropped note image may carry.
///
/// Lower-cased, no leading dot. The frontend derives `ext` from the clipboard
/// `File`'s MIME type / name; this allowlist is the authoritative gate so an
/// arbitrary extension can never reach the filesystem. The matching set is
/// mirrored by the protocol handler's content-type map in `app-main`.
const ALLOWED_IMAGE_EXTS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// Persist a pasted/dropped note image to the meeting's `assets/` directory and
/// return its **portable** reference (the bare `<contenthash>.<ext>` filename)
/// for the frontend to store into `notes.json`.
///
/// Routes **directly** to `persistence::save_note_asset` against
/// `IpcState::meetings_dir` — note assets are independent of the live recording
/// pipeline (see `architecture/components.md`, `persistence` "Note image
/// assets"), so the orchestrator is not involved. The blocking filesystem write
/// runs on `spawn_blocking` per the threading model.
///
/// `ext` is validated against [`ALLOWED_IMAGE_EXTS`] (case-insensitively); a
/// non-image extension is rejected as `AppError::InvalidInput`. The returned
/// filename is portable: it names only the file, not a path or a URL, so the
/// meeting folder (with `assets/`) can be copied to another machine and the
/// notes still resolve. The webview turns the filename into a working
/// `meetingasset:` URL at render time via `convertFileSrc`.
#[tauri::command]
#[specta::specta]
pub async fn save_note_image(
    meeting_id: MeetingId,
    bytes: Vec<u8>,
    ext: String,
    state: State<'_, IpcState>,
) -> AppResult<String> {
    let ext = normalise_image_ext(&ext)?;
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        persistence::save_note_asset(&meetings_dir, meeting_id, &bytes, &ext)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("save_note_image task join failed: {e}"),
    })?
}

/// Validate + normalise a note-image extension to a lower-cased, dot-less form
/// drawn from the [`ALLOWED_IMAGE_EXTS`] allowlist.
///
/// Pure + unit-tested so the allowlist gate is verified without a Tauri runtime.
/// Strips a single leading dot, lower-cases, and rejects anything not on the
/// list as `AppError::InvalidInput`.
pub(crate) fn normalise_image_ext(ext: &str) -> Result<String, AppError> {
    let cleaned = ext.trim().trim_start_matches('.').to_ascii_lowercase();
    if ALLOWED_IMAGE_EXTS.contains(&cleaned.as_str()) {
        Ok(cleaned)
    } else {
        Err(AppError::InvalidInput {
            context: format!(
                "unsupported note image extension {ext:?}; allowed: {ALLOWED_IMAGE_EXTS:?}"
            ),
        })
    }
}

