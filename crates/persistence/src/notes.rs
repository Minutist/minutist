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
//! schema. Do **not** introduce a typed Tiptap document model here. The Yjs
//! derivation in [`crate::ydoc`] is the single, narrow relaxation of this rule
//! (it walks the document generically; see that module's docs).
//!
//! # CRDT notes storage (O2 — `planning/DESIGN_notes-crdt.md`)
//!
//! `notes.ydoc` is the **authoritative** Yjs (yrs) document state when present;
//! `notes.json` (ProseMirror JSON) and `notes.md` (markdown) are **derived
//! projections** (D-O2.1). [`NotesStore::save`] builds a Yjs doc from the
//! incoming document JSON, writes `notes.ydoc` (a single atomic lib0-v2 blob),
//! then writes the JSON **derived from that doc** as `notes.json` plus the
//! caller-supplied `notes.md` — all three in the one save call (D-O2.4). On
//! [`NotesStore::load`], when `notes.ydoc` exists its derived JSON is returned
//! (the projection self-heals if `notes.json` is missing or stale, exactly as
//! the libsql index self-heals from the folders). A meeting that predates the
//! groundwork has no `notes.ydoc`; it reads straight from `notes.json` and is
//! seeded lazily on open by [`crate::reader::read_meeting_state`].
//!
//! # Atomic writes
//!
//! [`NotesStore::save`] writes each file to a sibling temp file in the meeting
//! folder, then renames it into place. A crash mid-save leaves the previous
//! `notes.ydoc` / `notes.json` / `notes.md` intact (the rename is atomic on the
//! same filesystem) and leaves no `.tmp` residue on the success path. `save`
//! writes `notes.ydoc` first; if the process dies before the projections are
//! rewritten, the next open re-derives them from `notes.ydoc` (self-healing).

use std::path::{Path, PathBuf};

use minutist_common::{AppResult, MeetingId, NoteBlock};

use crate::error::Error;

/// Paragraph attribute carrying the recording-clock anchor, in milliseconds.
///
/// Mirrors `ANCHOR_ATTR` in the editor's `paragraph-anchor.ts`; the editor
/// stores the anchor under this key in the Tiptap document JSON.
const ANCHOR_ATTR: &str = "data-anchor-ms";

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

    /// Atomically persist a meeting's notes: write the authoritative
    /// `notes.ydoc` (Yjs CRDT state) and its derived `notes.json` + the
    /// caller-supplied `notes.md`.
    ///
    /// `notes_json` is the editor's ProseMirror document. `save` builds a Yjs
    /// doc from it (the CRDT becomes authoritative — D-O2.1), encodes that doc
    /// as the single compacted lib0-v2 `notes.ydoc` blob, then writes
    /// `notes.json` **derived from the doc** (a structural round-trip; see
    /// [`crate::ydoc`]) and `notes.md` verbatim. `notes.ydoc` is written first
    /// so that if the process dies before the projections are rewritten the
    /// next open re-derives them (self-healing, D-O2.4). The markdown is
    /// supplied by the caller because rendering it requires the editor's typed
    /// schema, which this crate deliberately does not model.
    ///
    /// Each file is written to a sibling temp file then renamed into place, so
    /// an interrupted save never leaves a truncated notes file. The temp files
    /// are removed by the rename; no `.tmp` residue remains on success.
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
        let ydoc_path = folder.join("notes.ydoc");
        let json_path = folder.join("notes.json");
        let md_path = folder.join("notes.md");

        // The Yjs doc is authoritative: build it from the incoming JSON and
        // derive the JSON projection back from it so on-disk `notes.json`
        // always matches `notes.ydoc` (D-O2.1).
        let doc = crate::ydoc::json_to_ydoc(notes_json);
        let ydoc_bytes = crate::ydoc::encode_ydoc(&doc);
        let derived_json = crate::ydoc::ydoc_to_json(&doc);

        let json_bytes = serde_json::to_vec_pretty(&derived_json)
            .map_err(Error::Serialise)
            .map_err(minutist_common::AppError::from)?;

        // `notes.ydoc` first (authoritative), then the projections.
        write_atomic(&ydoc_path, &ydoc_bytes)?;
        write_atomic(&json_path, &json_bytes)?;
        write_atomic(&md_path, notes_md.as_bytes())?;

        tracing::debug!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            folder = %folder.display(),
            "notes.ydoc + derived notes.json + notes.md saved"
        );

        Ok(())
    }

    /// Load a meeting's notes for `meeting_id` under `root`.
    ///
    /// When `notes.ydoc` is present it is **authoritative**: the returned JSON
    /// is derived from it (D-O2.1), so the document reflects the CRDT state even
    /// if `notes.json` is missing or stale (the projection self-heals — D-O2.4).
    /// When `notes.ydoc` is absent the document is read straight from
    /// `notes.json` (a pre-CRDT meeting not yet seeded — D-O2.7).
    ///
    /// Returns `Ok(None)` only when **neither** `notes.ydoc` nor `notes.json`
    /// exists (a meeting with no notes yet). `notes.md` is read as a sibling; an
    /// absent `notes.md` yields an empty markdown string.
    pub fn load(root: &Path, meeting_id: MeetingId) -> AppResult<Option<NotesData>> {
        let folder = Self::folder_path(root, meeting_id);
        let ydoc_path = folder.join("notes.ydoc");
        let json_path = folder.join("notes.json");
        let md_path = folder.join("notes.md");

        // `notes.ydoc` (authoritative) wins when present; derive the JSON from it.
        let json = match std::fs::read(&ydoc_path) {
            Ok(bytes) => {
                let doc = crate::ydoc::decode_ydoc(&bytes)
                    .map_err(|context| minutist_common::AppError::Internal { context })?;
                crate::ydoc::ydoc_to_json(&doc)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                match std::fs::read(&json_path) {
                    Ok(bytes) => serde_json::from_slice(&bytes)
                        .map_err(Error::Serialise)
                        .map_err(minutist_common::AppError::from)?,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(e) => return Err(Error::Io(e).into()),
                }
            }
            Err(e) => return Err(Error::Io(e).into()),
        };

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
        minutist_common::AppError::Internal {
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

/// Project a Tiptap `notes.json` document into the note paragraphs the
/// summariser weaves (#70).
///
/// Walks the opaque document depth-first and, for every `paragraph` node,
/// emits one [`NoteBlock`] carrying the paragraph's concatenated plain text and
/// its `data-anchor-ms` anchor (a number the editor stamps on the
/// pause-excluding recording clock) when present. Paragraphs are returned in
/// document order; empty / whitespace-only paragraphs are skipped.
///
/// This is a best-effort READ projection, NOT a typed document model: it never
/// constrains what `notes.json` may contain — the store's opacity guarantee
/// (see the module docs) is intact — it simply ignores any node shape it does
/// not recognise. A malformed or non-object document yields an empty vec.
pub fn note_blocks_from_json(doc: &serde_json::Value) -> Vec<NoteBlock> {
    let mut out = Vec::new();
    collect_paragraphs(doc, &mut out);
    out
}

/// Depth-first walk collecting one [`NoteBlock`] per non-empty `paragraph` node.
fn collect_paragraphs(node: &serde_json::Value, out: &mut Vec<NoteBlock>) {
    if let Some(arr) = node.as_array() {
        for child in arr {
            collect_paragraphs(child, out);
        }
        return;
    }
    let Some(obj) = node.as_object() else {
        return;
    };

    if obj.get("type").and_then(|t| t.as_str()) == Some("paragraph") {
        let mut text = String::new();
        collect_text(node, &mut text);
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            let at_ms = obj
                .get("attrs")
                .and_then(|a| a.get(ANCHOR_ATTR))
                .and_then(json_to_ms);
            out.push(NoteBlock {
                at_ms,
                text: trimmed.to_string(),
            });
        }
        // Paragraphs do not nest paragraphs in this schema; the text is already
        // gathered, so do not recurse into the paragraph's own content.
        return;
    }

    // Recurse into `content` to reach paragraphs nested in list items /
    // blockquotes / the top-level `doc`.
    if let Some(content) = obj.get("content") {
        collect_paragraphs(content, out);
    }
}

/// Concatenate the plain text of all descendant `text` nodes of `node`.
fn collect_text(node: &serde_json::Value, out: &mut String) {
    if let Some(arr) = node.as_array() {
        for child in arr {
            collect_text(child, out);
        }
        return;
    }
    let Some(obj) = node.as_object() else {
        return;
    };
    if obj.get("type").and_then(|t| t.as_str()) == Some("text") {
        if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
            out.push_str(t);
        }
    }
    if let Some(content) = obj.get("content") {
        collect_text(content, out);
    }
}

/// Read a JSON anchor value as a millisecond count. The editor writes an
/// integer, but tolerate a float-encoded value (clamped non-negative) too.
fn json_to_ms(v: &serde_json::Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_f64().map(|f| f.max(0.0).round() as u64))
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
    // 0. note_blocks_from_json projection (#70).
    // -----------------------------------------------------------------------

    #[test]
    fn note_blocks_extracts_anchored_and_nested_paragraphs_in_order() {
        let blocks = note_blocks_from_json(&representative_doc());
        // Two paragraphs: the top-level anchored one, then the one nested inside
        // the bullet list's list item — in document order.
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].at_ms, Some(1234));
        assert_eq!(blocks[0].text, "Action items from the meeting:");
        assert_eq!(blocks[1].at_ms, None);
        assert_eq!(blocks[1].text, "Ship the notes store");
    }

    #[test]
    fn note_blocks_skips_empty_paragraphs_and_treats_null_anchor_as_unanchored() {
        let doc = json!({
            "type": "doc",
            "content": [
                // Empty paragraph (no content) — skipped.
                { "type": "paragraph", "attrs": { "data-anchor-ms": null } },
                // Whitespace-only paragraph — skipped.
                {
                    "type": "paragraph",
                    "content": [{ "type": "text", "text": "   " }]
                },
                // Explicit null anchor → un-anchored.
                {
                    "type": "paragraph",
                    "attrs": { "data-anchor-ms": null },
                    "content": [{ "type": "text", "text": "kept" }]
                },
            ]
        });
        let blocks = note_blocks_from_json(&doc);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].at_ms, None);
        assert_eq!(blocks[0].text, "kept");
    }

    #[test]
    fn note_blocks_concatenates_marked_text_runs_within_a_paragraph() {
        // A paragraph split into several text nodes (e.g. a bold run) is joined.
        let doc = json!({
            "type": "doc",
            "content": [{
                "type": "paragraph",
                "attrs": { "data-anchor-ms": 5000 },
                "content": [
                    { "type": "text", "text": "decide " },
                    { "type": "text", "marks": [{ "type": "bold" }], "text": "now" }
                ]
            }]
        });
        let blocks = note_blocks_from_json(&doc);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].at_ms, Some(5000));
        assert_eq!(blocks[0].text, "decide now");
    }

    #[test]
    fn note_blocks_on_malformed_or_empty_doc_is_empty() {
        assert!(note_blocks_from_json(&json!({ "type": "doc", "content": [] })).is_empty());
        assert!(note_blocks_from_json(&json!(null)).is_empty());
        assert!(note_blocks_from_json(&json!("garbage")).is_empty());
        assert!(note_blocks_from_json(&json!({ "type": "doc" })).is_empty());
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
    // 2. Lossless round-trip of a doc with UNKNOWN/custom node types.
    //    Proves the transcript-chip opacity guarantee survives the CRDT
    //    projection: a node type and attribute set the Rust side does not model
    //    round-trips by structure (D-O2.1 — the Yjs derivation walks the
    //    document generically). `notes.json` is now derived from `notes.ydoc`,
    //    so the document is normalised to valid ProseMirror shape — custom
    //    *node types and their attributes* survive, which is what the
    //    transcript-chip / note-image / future-node guarantee requires.
    // -----------------------------------------------------------------------

    #[test]
    fn unknown_node_type_round_trips_losslessly() {
        let (_tempdir, root, id) = make_meeting();
        // Custom node types the Rust side knows nothing about, carrying
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
                    "attrs": { "weird": 1.5, "flag": true, "name": "x" },
                    "content": [
                        { "type": "text", "text": "future content" }
                    ]
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
