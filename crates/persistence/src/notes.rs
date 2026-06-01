//! `NotesStore` — standalone reader/writer for `notes.json` + `notes.md`.
//!
//! `NotesStore` is **independent of [`crate::MeetingWriter`]**: there is no
//! shared open file handle. `MeetingWriter` owns `audio.opus`,
//! `transcript.json`, and `metadata.json` while a recording is in flight and
//! never touches the notes files; `NotesStore` only ever reads/writes
//! `notes.json` and `notes.md` and never touches the recording-owned files.
//! This split lets the notes editor autosave (FR-18/FR-35) concurrently with
//! an active recording without contending on a writer.
//!
//! # Opacity guarantee (Phase 4 transcript chips)
//!
//! `notes.json` is stored as an **opaque** [`serde_json::Value`]. The Rust
//! side does not model Tiptap's document shape: unknown or custom node types
//! (e.g. the Phase-4 transcript-chip node) round-trip losslessly because the
//! value is parsed and re-serialised verbatim, never coerced into a typed
//! schema. Do **not** introduce a typed Tiptap document model here.
//!
//! # Atomic writes
//!
//! [`NotesStore::save`] writes each file to a sibling temp file in the meeting
//! folder, then renames it into place. A crash mid-save leaves the previous
//! `notes.json` / `notes.md` intact (the rename is atomic on the same
//! filesystem) and leaves no `.tmp` residue on the success path.

use std::path::{Path, PathBuf};

use meeting_app_common::{AppResult, MeetingId};

use crate::error::Error;

/// The parsed contents of a meeting's notes files.
///
/// `json` is the opaque Tiptap document (`notes.json`); `markdown` is the
/// rendered markdown view (`notes.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct NotesData {
    /// Opaque Tiptap document parsed from `notes.json`.
    pub json: serde_json::Value,
    /// Markdown rendering loaded from `notes.md`.
    pub markdown: String,
}

/// Standalone, stateless store for a meeting's `notes.json` + `notes.md`.
///
/// `NotesStore` holds no open file handle; every call resolves paths from the
/// `(root, meeting_id)` pair. It is deliberately decoupled from
/// [`crate::MeetingWriter`] — see the module docs.
pub struct NotesStore;

impl NotesStore {
    /// Resolve the meeting folder path for `meeting_id` under `root`.
    ///
    /// Mirrors [`crate::MeetingFolder::create`]'s layout (`{root}/{uuid}/`)
    /// **without** creating the directory.
    fn folder_path(root: &Path, meeting_id: MeetingId) -> PathBuf {
        root.join(meeting_id.0.to_string())
    }

    /// Atomically write `notes_json` to `notes.json` and `notes_md` to
    /// `notes.md` inside the existing meeting folder.
    ///
    /// Each file is written to a sibling temp file then renamed into place,
    /// so an interrupted save never leaves a truncated notes file. The temp
    /// files are removed by the rename; no `.tmp` residue remains on success.
    ///
    /// This does **not** create the meeting folder — the folder is owned by
    /// [`crate::MeetingWriter`] and is expected to exist. Sibling files
    /// (`audio.opus`, `transcript.json`, `metadata.json`) are left untouched.
    pub fn save(
        root: &Path,
        meeting_id: MeetingId,
        notes_json: &serde_json::Value,
        notes_md: &str,
    ) -> AppResult<()> {
        let folder = Self::folder_path(root, meeting_id);
        let json_path = folder.join("notes.json");
        let md_path = folder.join("notes.md");

        let json_bytes = serde_json::to_vec_pretty(notes_json)
            .map_err(Error::Serialise)
            .map_err(meeting_app_common::AppError::from)?;

        write_atomic(&json_path, &json_bytes)?;
        write_atomic(&md_path, notes_md.as_bytes())?;

        tracing::debug!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            folder = %folder.display(),
            "notes.json + notes.md saved"
        );

        Ok(())
    }

    /// Load `notes.json` + `notes.md` for `meeting_id` under `root`.
    ///
    /// Returns `Ok(None)` if `notes.json` does not exist (a meeting with no
    /// notes yet). `notes.md` is read as a sibling; an absent `notes.md`
    /// alongside an existing `notes.json` yields an empty markdown string.
    pub fn load(root: &Path, meeting_id: MeetingId) -> AppResult<Option<NotesData>> {
        let folder = Self::folder_path(root, meeting_id);
        let json_path = folder.join("notes.json");
        let md_path = folder.join("notes.md");

        let json_bytes = match std::fs::read(&json_path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::Io(e).into()),
        };

        let json: serde_json::Value = serde_json::from_slice(&json_bytes)
            .map_err(Error::Serialise)
            .map_err(meeting_app_common::AppError::from)?;

        let markdown = match std::fs::read_to_string(&md_path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(Error::Io(e).into()),
        };

        Ok(Some(NotesData { json, markdown }))
    }
}

/// Write `bytes` to `path` atomically: write to a sibling temp file, fsync,
/// then rename into place. The temp file shares `path`'s parent so the rename
/// stays on one filesystem (a cross-device rename would fail).
///
/// On the success path no temp file remains (the rename consumes it). If the
/// rename fails the temp file is removed on a best-effort basis so no `.tmp`
/// residue is left behind.
fn write_atomic(path: &Path, bytes: &[u8]) -> AppResult<()> {
    use std::io::Write;

    let parent = path.parent().ok_or_else(|| {
        meeting_app_common::AppError::Internal {
            context: format!("notes path has no parent: {}", path.display()),
        }
    })?;

    // Temp filename derived from the target so concurrent saves of different
    // files (notes.json vs notes.md) don't collide on a shared temp name.
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "notes".to_string());
    let tmp_path = parent.join(format!("{file_name}.tmp"));

    let write_result = (|| -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        // Best-effort cleanup so a failed write leaves no residue.
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folder::MeetingFolder;
    use serde_json::json;
    use tempfile::TempDir;

    /// A representative Tiptap-shaped document used for round-trip tests.
    fn representative_doc() -> serde_json::Value {
        json!({
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "attrs": { "data-anchor-ms": 1234 },
                    "content": [
                        { "type": "text", "text": "Action items from the meeting:" }
                    ]
                },
                {
                    "type": "bulletList",
                    "content": [
                        {
                            "type": "listItem",
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [
                                        { "type": "text", "text": "Ship the notes store" }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        })
    }

    /// Create the meeting folder on disk and return `(tempdir, root, id)`.
    fn make_meeting() -> (TempDir, std::path::PathBuf, MeetingId) {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path().to_path_buf();
        let id = MeetingId::new();
        // NotesStore writes into an *existing* folder; create it via the
        // owning type so the layout matches production exactly.
        MeetingFolder::create(&root, id).expect("create meeting folder");
        (tempdir, root, id)
    }

    // -----------------------------------------------------------------------
    // 1. Representative-doc save → load round-trip.
    // -----------------------------------------------------------------------

    #[test]
    fn representative_doc_round_trips() {
        let (_tempdir, root, id) = make_meeting();
        let doc = representative_doc();
        let md = "# Notes\n\n- Ship the notes store\n";

        NotesStore::save(&root, id, &doc, md).expect("save");
        let loaded = NotesStore::load(&root, id).expect("load").expect("present");

        assert_eq!(loaded.json, doc, "notes.json did not round-trip");
        assert_eq!(loaded.markdown, md, "notes.md did not round-trip");
    }

    // -----------------------------------------------------------------------
    // 2. Lossless round-trip of a doc with an UNKNOWN/custom node type.
    //    Proves the Phase-4 transcript-chip opacity guarantee.
    // -----------------------------------------------------------------------

    #[test]
    fn unknown_node_type_round_trips_losslessly() {
        let (_tempdir, root, id) = make_meeting();
        // A custom node type the Rust side knows nothing about, carrying
        // arbitrary nested attrs. None of this is modelled in Rust.
        let doc = json!({
            "type": "doc",
            "content": [
                {
                    "type": "transcriptChip",
                    "attrs": {
                        "segmentStartMs": 42000,
                        "segmentEndMs": 45500,
                        "speakerId": "A",
                        "customNested": { "foo": [1, 2, 3], "bar": null, "baz": true }
                    }
                },
                {
                    "type": "someFutureNode",
                    "marks": [{ "type": "weirdMark", "attrs": { "x": 1.5 } }],
                    "content": []
                }
            ]
        });

        NotesStore::save(&root, id, &doc, "").expect("save");
        let loaded = NotesStore::load(&root, id).expect("load").expect("present");

        assert_eq!(
            loaded.json, doc,
            "unknown/custom node types must round-trip losslessly (opacity)"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Absent-file load returns Ok(None).
    // -----------------------------------------------------------------------

    #[test]
    fn absent_notes_load_returns_none() {
        let (_tempdir, root, id) = make_meeting();
        // No save has happened — notes.json does not exist.
        let loaded = NotesStore::load(&root, id).expect("load");
        assert!(loaded.is_none(), "absent notes.json must yield Ok(None)");
    }

    #[test]
    fn absent_meeting_folder_load_returns_none() {
        // Folder itself doesn't exist either — still Ok(None), not an error.
        let tempdir = TempDir::new().expect("tempdir");
        let id = MeetingId::new();
        let loaded = NotesStore::load(tempdir.path(), id).expect("load");
        assert!(loaded.is_none(), "absent folder must yield Ok(None)");
    }

    // -----------------------------------------------------------------------
    // 4. Save into an existing folder leaves recording-owned files untouched.
    // -----------------------------------------------------------------------

    #[test]
    fn save_leaves_sibling_files_untouched() {
        let (_tempdir, root, id) = make_meeting();
        let folder = root.join(id.0.to_string());

        // Seed the recording-owned files with known contents.
        let audio = folder.join("audio.opus");
        let transcript = folder.join("transcript.json");
        let metadata = folder.join("metadata.json");
        std::fs::write(&audio, b"OPUSDATA").expect("seed audio");
        std::fs::write(&transcript, b"[{\"start_ms\":0}]").expect("seed transcript");
        std::fs::write(&metadata, b"{\"title\":\"x\"}").expect("seed metadata");

        NotesStore::save(&root, id, &representative_doc(), "notes body").expect("save");

        assert_eq!(std::fs::read(&audio).unwrap(), b"OPUSDATA");
        assert_eq!(std::fs::read(&transcript).unwrap(), b"[{\"start_ms\":0}]");
        assert_eq!(std::fs::read(&metadata).unwrap(), b"{\"title\":\"x\"}");
    }

    #[test]
    fn save_does_not_create_meeting_folder() {
        // The folder must already exist; saving must not silently create it.
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path().to_path_buf();
        let id = MeetingId::new();
        // Deliberately do NOT create the folder.
        let result = NotesStore::save(&root, id, &representative_doc(), "x");
        assert!(
            result.is_err(),
            "saving into a non-existent meeting folder must error, not create it"
        );
        assert!(
            !root.join(id.0.to_string()).exists(),
            "save must not create the meeting folder"
        );
    }

    // -----------------------------------------------------------------------
    // 5. No `.tmp` residue after a successful save.
    // -----------------------------------------------------------------------

    #[test]
    fn successful_save_leaves_no_tmp_residue() {
        let (_tempdir, root, id) = make_meeting();
        let folder = root.join(id.0.to_string());

        NotesStore::save(&root, id, &representative_doc(), "body").expect("save");

        let residue: Vec<_> = std::fs::read_dir(&folder)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();

        assert!(
            residue.is_empty(),
            "expected no .tmp residue after save, found: {residue:?}"
        );

        // And re-saving over an existing pair stays clean too.
        NotesStore::save(&root, id, &representative_doc(), "body2").expect("re-save");
        let residue_after: Vec<_> = std::fs::read_dir(&folder)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(
            residue_after.is_empty(),
            "expected no .tmp residue after re-save, found: {residue_after:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 6. MeetingFolder path helpers point at the right files.
    // -----------------------------------------------------------------------

    #[test]
    fn folder_helpers_resolve_notes_paths() {
        let tempdir = TempDir::new().expect("tempdir");
        let folder = MeetingFolder::create(tempdir.path(), MeetingId::new()).expect("folder");
        assert!(folder.notes_path().ends_with("notes.json"));
        assert!(folder.notes_md_path().ends_with("notes.md"));
        assert_eq!(folder.notes_path().parent(), Some(folder.path()));
        assert_eq!(folder.notes_md_path().parent(), Some(folder.path()));
    }
}
