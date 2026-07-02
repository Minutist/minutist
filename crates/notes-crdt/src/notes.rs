//! `NotesStore` — standalone reader/writer for `notes.json` + `notes.md`.
//!
//! `NotesStore` is **independent of `persistence`'s `MeetingWriter`**: there is
//! no shared open file handle. `MeetingWriter` owns `audio.opus`,
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
//! seeded lazily on open by `persistence::reader::read_meeting_state`.
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
/// `persistence`'s `MeetingWriter` — see the module docs.
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
    /// `persistence`'s `MeetingWriter` and is expected to exist. Sibling files
    /// (`audio.opus`, `transcript.json`, `metadata.json`) are left untouched.
    ///
    /// # Refusal when `notes.ydoc` already exists
    ///
    /// Rebuilding the Yjs doc from JSON mints a fresh client history, so calling
    /// `save` over an existing `notes.ydoc` would sever its CRDT merge ancestry.
    /// `save` therefore refuses with [`minutist_common::AppError::InvalidInput`]
    /// when `notes.ydoc` is present: an open editor's edits must flow through
    /// [`NotesStore::apply_update`] (which merges, preserving history) instead.
    /// `save` remains the legitimate writer only for the **first** write of a
    /// meeting that has no `notes.ydoc` yet.
    ///
    /// The whole exists-check→build→write sequence runs under this meeting's
    /// [`crate::notes_lock`] so it cannot interleave with a concurrent
    /// `apply_update` / `seed_ydoc_if_needed` on the same meeting (see that
    /// module's docs for the lost-update this closes).
    pub fn save(
        root: &Path,
        meeting_id: MeetingId,
        notes_json: &serde_json::Value,
        notes_md: &str,
    ) -> AppResult<()> {
        let guard = crate::notes_lock::notes_lock(meeting_id);
        let _guard = guard.lock().expect("notes lock poisoned");
        Self::save_locked(root, meeting_id, notes_json, notes_md)
    }

    /// The body of [`NotesStore::save`], run with the caller already holding
    /// this meeting's `notes_lock`. Never call this directly except from
    /// another function that already holds that lock — `std::sync::Mutex` is
    /// not reentrant, so acquiring it twice on one thread deadlocks.
    fn save_locked(
        root: &Path,
        meeting_id: MeetingId,
        notes_json: &serde_json::Value,
        notes_md: &str,
    ) -> AppResult<()> {
        let folder = Self::folder_path(root, meeting_id);
        let ydoc_path = folder.join("notes.ydoc");
        let json_path = folder.join("notes.json");
        let md_path = folder.join("notes.md");

        // `notes.ydoc` is authoritative once it exists; rebuilding it from JSON
        // here would discard its CRDT history. Refuse and route the caller to
        // `apply_update`, which merges.
        if ydoc_path.exists() {
            return Err(minutist_common::AppError::InvalidInput {
                context:
                    "notes.ydoc is authoritative; use apply_update to edit existing CRDT notes"
                        .to_string(),
            });
        }

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

    /// Apply an incremental Yjs **v1** update (as produced by the editor's
    /// `Y.Doc` `'update'` event) onto the meeting's stored `notes.ydoc`, then
    /// atomically rewrite the authoritative blob and its derived `notes.json` +
    /// `notes.md`.
    ///
    /// This is the **primary write path for an open editor** (D-O2.1): the
    /// editor is Yjs-native, so its edits arrive as CRDT updates rather than as
    /// whole-document JSON. The stored `notes.ydoc` is loaded (decoded v2), the
    /// editor's v1 update is merged via [`crate::ydoc::apply_update_v1`] — which
    /// preserves the CRDT history rather than rebuilding the doc from JSON the way
    /// [`NotesStore::save`] does — and the merged doc is re-encoded as the durable
    /// v2 blob. `notes.json` is derived from the merged doc; `notes.md` is the
    /// caller-supplied markdown rendering.
    ///
    /// When `notes.ydoc` is absent a legacy `notes.json` is seeded first (via
    /// [`NotesStore::seed_ydoc_if_needed`]) so the editor's update merges onto the
    /// existing content rather than discarding it. Only when there is **no**
    /// `notes.json` either — a meeting that has never had notes — is the update
    /// applied onto a fresh doc, the editor being the genuine first writer.
    ///
    /// Writes are atomic (sibling temp + fsync + rename), `notes.ydoc` first, so
    /// a mid-save crash leaves the prior files intact and the projections
    /// self-heal on next open. Like [`NotesStore::save`] this does **not** create
    /// the meeting folder.
    ///
    /// The whole seed→load→merge→write sequence runs under this meeting's
    /// [`crate::notes_lock`], so two concurrent callers merging onto the same
    /// meeting — e.g. `sync`'s inbound path (`crates/sync/src/notes_proto.rs`
    /// `apply_inbound`) racing a local editor autosave, or two hub peers
    /// reconciling the same meeting — cannot both load the same base doc and
    /// last-writer-wins on the file. yrs merges are commutative, so serialising
    /// the RMW (rather than the merge itself) is sufficient: each call's result
    /// becomes the next call's base.
    pub fn apply_update(
        root: &Path,
        meeting_id: MeetingId,
        update: &[u8],
        notes_md: &str,
    ) -> AppResult<()> {
        let guard = crate::notes_lock::notes_lock(meeting_id);
        let _guard = guard.lock().expect("notes lock poisoned");
        Self::apply_update_locked(root, meeting_id, update, notes_md)
    }

    /// The body of [`NotesStore::apply_update`], run with the caller already
    /// holding this meeting's `notes_lock`. Never call this directly except
    /// from another function that already holds that lock — `std::sync::Mutex`
    /// is not reentrant, so acquiring it twice on one thread deadlocks.
    fn apply_update_locked(
        root: &Path,
        meeting_id: MeetingId,
        update: &[u8],
        notes_md: &str,
    ) -> AppResult<()> {
        let folder = Self::folder_path(root, meeting_id);
        let ydoc_path = folder.join("notes.ydoc");
        let json_path = folder.join("notes.json");
        let md_path = folder.join("notes.md");

        // Seed from a legacy `notes.json` before merging so a pre-CRDT meeting's
        // content is not dropped on the first incremental write (this is the same
        // migration the on-open seed performs; calling it here closes the gap
        // when an edit reaches `apply_update` before any open-seed has run). Calls
        // the already-locked body directly: the lock is already held by this
        // function's caller, so re-entering the public `seed_ydoc_if_needed`
        // would deadlock.
        Self::seed_ydoc_if_needed_locked(root, meeting_id)?;

        // Load the authoritative doc (or start fresh when the editor is the first
        // writer and there is no legacy `notes.json` to seed from) and MERGE the
        // editor's incremental update onto it — preserving CRDT history, unlike
        // `save`'s rebuild-from-JSON.
        let doc = match std::fs::read(&ydoc_path) {
            Ok(bytes) => crate::ydoc::decode_ydoc(&bytes)
                .map_err(|context| minutist_common::AppError::Internal { context })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => crate::ydoc::new_ydoc(),
            Err(e) => return Err(Error::Io(e).into()),
        };

        crate::ydoc::apply_update_v1(&doc, update)
            .map_err(|context| minutist_common::AppError::Internal { context })?;

        let ydoc_bytes = crate::ydoc::encode_ydoc(&doc);
        let derived_json = crate::ydoc::ydoc_to_json(&doc);
        let json_bytes = serde_json::to_vec_pretty(&derived_json)
            .map_err(Error::Serialise)
            .map_err(minutist_common::AppError::from)?;

        // notes.ydoc first (authoritative), then the projections. notes.json
        // self-heals from notes.ydoc on load; notes.md is a best-effort export
        // and may lag if the process dies between these renames.
        write_atomic(&ydoc_path, &ydoc_bytes)?;
        write_atomic(&json_path, &json_bytes)?;
        write_atomic(&md_path, notes_md.as_bytes())?;

        tracing::debug!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            folder = %folder.display(),
            update_len = update.len(),
            "applied editor Yjs update; rewrote notes.ydoc + derived notes.json + notes.md"
        );

        Ok(())
    }

    /// Read the meeting's current `notes.ydoc` state encoded as a lib0 **v1**
    /// update — the byte form the editor's `Y.applyUpdate` consumes to hydrate
    /// its `Y.Doc` on open.
    ///
    /// Returns `Ok(None)` when `notes.ydoc` is absent (a meeting with no
    /// CRDT-backed notes yet — the editor then starts empty and its first edit
    /// becomes the seed). The stored blob is v2 (durable); this decodes it and
    /// re-encodes as v1 because the JS `yjs` library only accepts v1 over
    /// `applyUpdate` (see [`crate::ydoc`] module docs — the v1/v2 hops must not
    /// be crossed).
    pub fn read_ydoc_state(root: &Path, meeting_id: MeetingId) -> AppResult<Option<Vec<u8>>> {
        let ydoc_path = Self::folder_path(root, meeting_id).join("notes.ydoc");
        match std::fs::read(&ydoc_path) {
            Ok(bytes) => {
                let doc = crate::ydoc::decode_ydoc(&bytes)
                    .map_err(|context| minutist_common::AppError::Internal { context })?;
                Ok(Some(crate::ydoc::encode_state_v1(&doc)))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Io(e).into()),
        }
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
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match std::fs::read(&json_path) {
                Ok(bytes) => serde_json::from_slice(&bytes)
                    .map_err(Error::Serialise)
                    .map_err(minutist_common::AppError::from)?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(Error::Io(e).into()),
            },
            Err(e) => return Err(Error::Io(e).into()),
        };

        let markdown = match std::fs::read_to_string(&md_path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(Error::Io(e).into()),
        };

        Ok(Some(NotesData { json, markdown }))
    }

    /// Lazily seed `notes.ydoc` from an existing `notes.json` for a pre-CRDT
    /// meeting, returning `true` when a seed was written (so the caller flips
    /// `MeetingMeta::notes_format` to `1` and rewrites `metadata.json`).
    ///
    /// The one notes-CRDT migration (D-O2.7), triggered lazily per meeting on
    /// open. It runs when `notes.ydoc` is **absent** and `notes.json` is
    /// present: it reads the JSON, builds a fresh `yrs` doc from it, and writes
    /// `notes.ydoc` (atomically). It is **idempotent** — once `notes.ydoc`
    /// exists this is a no-op — and **build-invariant** (the same path in both
    /// build variants; the free build seeds too, only the sync transport is
    /// gated). A meeting with no `notes.json` is left untouched (nothing to
    /// seed; it stays JSON-readable and migrates the day it first gets notes).
    ///
    /// Seeding a `yrs` doc from JSON gives it a fresh client history — safe only
    /// as a one-time origin for a never-synced document, which is exactly the
    /// migration case (a pre-CRDT meeting has no peer to share ancestry with).
    /// After seeding, `notes.ydoc` is authoritative.
    ///
    /// The exists-check→read→build→write sequence runs under this meeting's
    /// [`crate::notes_lock`] so a concurrent `apply_update`/`save` on the same
    /// meeting cannot interleave with the seed.
    pub fn seed_ydoc_if_needed(root: &Path, meeting_id: MeetingId) -> AppResult<bool> {
        let guard = crate::notes_lock::notes_lock(meeting_id);
        let _guard = guard.lock().expect("notes lock poisoned");
        Self::seed_ydoc_if_needed_locked(root, meeting_id)
    }

    /// The body of [`NotesStore::seed_ydoc_if_needed`], run with the caller
    /// already holding this meeting's `notes_lock`. Never call this directly
    /// except from another function that already holds that lock —
    /// `std::sync::Mutex` is not reentrant, so acquiring it twice on one thread
    /// deadlocks.
    fn seed_ydoc_if_needed_locked(root: &Path, meeting_id: MeetingId) -> AppResult<bool> {
        let folder = Self::folder_path(root, meeting_id);
        let ydoc_path = folder.join("notes.ydoc");
        let json_path = folder.join("notes.json");

        // Idempotent: an existing `notes.ydoc` is already authoritative.
        if ydoc_path.exists() {
            return Ok(false);
        }

        // Nothing to seed when the meeting has never had notes.
        let json_bytes = match std::fs::read(&json_path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(Error::Io(e).into()),
        };

        let json: serde_json::Value = serde_json::from_slice(&json_bytes)
            .map_err(Error::Serialise)
            .map_err(minutist_common::AppError::from)?;

        let doc = crate::ydoc::json_to_ydoc(&json);
        let ydoc_bytes = crate::ydoc::encode_ydoc(&doc);
        write_atomic(&ydoc_path, &ydoc_bytes)?;

        tracing::info!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            folder = %folder.display(),
            "seeded notes.ydoc from notes.json (pre-CRDT migration)"
        );

        Ok(true)
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

    let parent = path
        .parent()
        .ok_or_else(|| minutist_common::AppError::Internal {
            context: format!("notes path has no parent: {}", path.display()),
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

        // A subsequent write over the existing notes.ydoc goes through
        // apply_update (save would refuse); it stays clean too.
        let v1 = NotesStore::read_ydoc_state(&root, id)
            .expect("read state")
            .expect("state present");
        NotesStore::apply_update(&root, id, &v1, "body2").expect("apply update");
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

    // -----------------------------------------------------------------------
    // 7. notes.ydoc is written by save, and the lazy seed migration (D-O2.7).
    // -----------------------------------------------------------------------

    #[test]
    fn save_writes_authoritative_notes_ydoc() {
        let (_tempdir, root, id) = make_meeting();
        let folder = root.join(id.0.to_string());

        NotesStore::save(&root, id, &representative_doc(), "body").expect("save");

        // All three files exist; notes.ydoc is the authoritative blob.
        assert!(
            folder.join("notes.ydoc").exists(),
            "notes.ydoc must be written"
        );
        assert!(
            folder.join("notes.json").exists(),
            "derived notes.json must be written"
        );
        assert!(folder.join("notes.md").exists(), "notes.md must be written");
        let ydoc = std::fs::read(folder.join("notes.ydoc")).expect("read ydoc");
        assert!(!ydoc.is_empty(), "notes.ydoc must carry the encoded state");
    }

    #[test]
    fn save_refuses_when_notes_ydoc_already_exists() {
        // The first save establishes notes.ydoc.
        let (_tempdir, root, id) = make_meeting();
        NotesStore::save(&root, id, &representative_doc(), "body").expect("first save");

        // A second save would rebuild the doc from JSON, severing CRDT history;
        // it must refuse rather than clobber the authoritative blob.
        let err = NotesStore::save(&root, id, &representative_doc(), "body2")
            .expect_err("save over an existing notes.ydoc must refuse");
        assert!(
            matches!(err, minutist_common::AppError::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );

        // The refusal left notes.md (and the rest) at the first-save content.
        let loaded = NotesStore::load(&root, id).expect("load").expect("present");
        assert_eq!(
            loaded.markdown, "body",
            "refused save must not touch notes.md"
        );
    }

    #[test]
    fn apply_update_seeds_legacy_json_before_merging() {
        // A pre-CRDT meeting: notes.json on disk, no notes.ydoc. An edit reaches
        // apply_update before any on-open seed has run.
        let (_tempdir, root, id) = make_meeting();
        let folder = root.join(id.0.to_string());
        let legacy = representative_doc();
        std::fs::write(
            folder.join("notes.json"),
            serde_json::to_vec_pretty(&legacy).unwrap(),
        )
        .expect("seed legacy notes.json");
        assert!(!folder.join("notes.ydoc").exists());

        // Apply an empty (no-op) update from a fresh doc: nothing is added, so if
        // the legacy content survives it can only be because apply_update seeded
        // notes.ydoc from notes.json before merging.
        let empty = crate::ydoc::encode_state_v1(&crate::ydoc::new_ydoc());
        NotesStore::apply_update(&root, id, &empty, "md").expect("apply update");

        let loaded = NotesStore::load(&root, id).expect("load").expect("present");
        assert_eq!(
            loaded.json, legacy,
            "apply_update must seed legacy notes.json before merging, not drop it"
        );
    }

    #[test]
    fn seed_is_idempotent_and_skips_meetings_without_notes() {
        // A meeting with no notes at all: nothing to seed.
        let (_tempdir, root, id) = make_meeting();
        assert!(
            !NotesStore::seed_ydoc_if_needed(&root, id).expect("seed"),
            "a meeting with no notes.json must not be seeded"
        );
        assert!(!root.join(id.0.to_string()).join("notes.ydoc").exists());
    }

    #[test]
    fn seed_builds_ydoc_from_legacy_json_and_is_idempotent() {
        let (_tempdir, root, id) = make_meeting();
        let folder = root.join(id.0.to_string());

        // Simulate a pre-CRDT meeting: notes.json on disk, no notes.ydoc.
        let doc = representative_doc();
        std::fs::write(
            folder.join("notes.json"),
            serde_json::to_vec_pretty(&doc).unwrap(),
        )
        .expect("seed legacy notes.json");
        std::fs::write(folder.join("notes.md"), "# legacy").expect("seed notes.md");
        assert!(!folder.join("notes.ydoc").exists());

        // First seed writes notes.ydoc.
        assert!(
            NotesStore::seed_ydoc_if_needed(&root, id).expect("seed"),
            "first seed must write notes.ydoc"
        );
        assert!(folder.join("notes.ydoc").exists());

        // After seeding, notes.ydoc is authoritative: the load derives the same
        // document (structural round-trip).
        let loaded = NotesStore::load(&root, id).expect("load").expect("present");
        assert_eq!(
            loaded.json, doc,
            "seeded notes.ydoc must derive the same doc"
        );

        // Idempotent: a second seed is a no-op.
        assert!(
            !NotesStore::seed_ydoc_if_needed(&root, id).expect("seed"),
            "second seed must be a no-op once notes.ydoc exists"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Incremental editor-update path (WU7 — apply_update / read_ydoc_state).
    // -----------------------------------------------------------------------

    /// Applying a v1 update produced by `save`'s doc round-trips through the
    /// incremental path: write a doc via `save`, read its v1 state, apply that
    /// onto a fresh meeting via `apply_update`, and confirm the derived JSON
    /// matches. This exercises the full incremental seam end to end on the Rust
    /// side (the JS-produced-update interop is in the UI test).
    #[test]
    fn apply_update_merges_and_re_derives_json() {
        let (_tempdir, root, id) = make_meeting();
        let doc = representative_doc();

        // Establish a notes.ydoc via the legacy save, then read its v1 state.
        NotesStore::save(&root, id, &doc, "md").expect("save");
        let v1 = NotesStore::read_ydoc_state(&root, id)
            .expect("read state")
            .expect("state present");
        assert!(!v1.is_empty(), "v1 state must be non-empty");

        // A second meeting receives that same v1 update as its first write.
        let (_t2, root2, id2) = make_meeting();
        NotesStore::apply_update(&root2, id2, &v1, "md2").expect("apply update");

        let loaded = NotesStore::load(&root2, id2)
            .expect("load")
            .expect("present");
        assert_eq!(
            loaded.json, doc,
            "apply_update must derive the same document"
        );
        assert_eq!(loaded.markdown, "md2");
        assert!(root2.join(id2.0.to_string()).join("notes.ydoc").exists());
    }

    /// `read_ydoc_state` returns `None` for a meeting with no notes.ydoc, so the
    /// editor knows to start empty.
    #[test]
    fn read_ydoc_state_absent_is_none() {
        let (_tempdir, root, id) = make_meeting();
        assert!(
            NotesStore::read_ydoc_state(&root, id)
                .expect("read")
                .is_none(),
            "a meeting with no notes.ydoc must yield None"
        );
    }

    /// The migration interaction (#6): a legacy meeting (notes.json, no
    /// notes.ydoc) is seeded on open, then the editor reads the seeded state and
    /// the legacy content appears. This pins seed-then-read.
    #[test]
    fn legacy_meeting_seeds_then_editor_reads_seeded_state() {
        let (_tempdir, root, id) = make_meeting();
        let folder = root.join(id.0.to_string());

        // Legacy on-disk: notes.json only.
        let doc = representative_doc();
        std::fs::write(
            folder.join("notes.json"),
            serde_json::to_vec_pretty(&doc).unwrap(),
        )
        .expect("seed legacy notes.json");

        // No notes.ydoc yet → editor would have no state to load.
        assert!(NotesStore::read_ydoc_state(&root, id)
            .expect("read")
            .is_none());

        // The on-open seed (D-O2.7) writes notes.ydoc from the legacy JSON.
        assert!(NotesStore::seed_ydoc_if_needed(&root, id).expect("seed"));

        // Now the editor can hydrate from the seeded state, and it carries the
        // legacy content.
        let v1 = NotesStore::read_ydoc_state(&root, id)
            .expect("read")
            .expect("seeded state present");
        let target = crate::ydoc::new_ydoc();
        crate::ydoc::apply_update_v1(&target, &v1).expect("apply seeded state");
        assert_eq!(
            crate::ydoc::ydoc_to_json(&target),
            doc,
            "the seeded state handed to the editor must carry the legacy content"
        );
    }

    // -----------------------------------------------------------------------
    // 9. Concurrency: apply_update must not lose either of two concurrent
    //    updates to the same fresh meeting (F1 review finding).
    // -----------------------------------------------------------------------

    /// Two threads race `apply_update` against the same fresh meeting with two
    /// independent, non-conflicting updates. Before the per-meeting
    /// `notes_lock` serialised the read→merge→write, each thread could load the
    /// same base doc, merge its own update in isolation, and last-writer-wins on
    /// the file — silently dropping the other thread's update. yrs merges are
    /// commutative, so the serialised RMW retains BOTH updates regardless of
    /// which thread's critical section runs first.
    #[test]
    fn concurrent_apply_update_retains_both_updates() {
        let (_tempdir, root, id) = make_meeting();

        // Two independent documents, each carrying one distinguishable
        // paragraph. Their whole-state v1 encodings are the "different yrs
        // updates" applied concurrently below.
        let doc_a = crate::ydoc::json_to_ydoc(&json!({
            "type": "doc",
            "content": [{
                "type": "paragraph",
                "content": [{ "type": "text", "text": "from thread A" }]
            }]
        }));
        let doc_b = crate::ydoc::json_to_ydoc(&json!({
            "type": "doc",
            "content": [{
                "type": "paragraph",
                "content": [{ "type": "text", "text": "from thread B" }]
            }]
        }));
        let update_a = crate::ydoc::encode_state_v1(&doc_a);
        let update_b = crate::ydoc::encode_state_v1(&doc_b);

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));

        let root_a = root.clone();
        let barrier_a = barrier.clone();
        let handle_a = std::thread::spawn(move || {
            barrier_a.wait();
            NotesStore::apply_update(&root_a, id, &update_a, "md-a")
        });

        let root_b = root.clone();
        let barrier_b = barrier.clone();
        let handle_b = std::thread::spawn(move || {
            barrier_b.wait();
            NotesStore::apply_update(&root_b, id, &update_b, "md-b")
        });

        handle_a
            .join()
            .expect("thread a panicked")
            .expect("apply_update a");
        handle_b
            .join()
            .expect("thread b panicked")
            .expect("apply_update b");

        let loaded = NotesStore::load(&root, id).expect("load").expect("present");
        let blocks = note_blocks_from_json(&loaded.json);
        let texts: Vec<&str> = blocks.iter().map(|b| b.text.as_str()).collect();
        assert!(
            texts.contains(&"from thread A"),
            "thread A's concurrent update must survive the serialised RMW, got {texts:?}"
        );
        assert!(
            texts.contains(&"from thread B"),
            "thread B's concurrent update must survive the serialised RMW, got {texts:?}"
        );
    }
}
