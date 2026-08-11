//! Attachment storage for per-meeting documents.
//!
//! Files are stored under `{root}/{meeting_id}/attachments/` as
//! content-addressed originals (`<sha256hex>.<ext>`) with optional converted
//! markdown siblings (`<sha256hex>.md`). The manifest (`attachments.json`) is a
//! `Vec<AttachmentEntry>` maintained by this module; every write goes through an
//! atomic tmp+rename so a crash mid-write leaves no corrupt manifest.
//!
//! # Content addressing and dedup
//!
//! The filename is the SHA-256 hex of the original bytes plus the extension, so
//! uploading the same bytes twice writes one file and one manifest row with the
//! same `hash`. Removing a row is dedup-safe: the original and its markdown
//! sibling are only unlinked when no surviving manifest row references the same
//! hash.
//!
//! # Manifest write lock
//!
//! A process-wide per-meeting `std::sync::Mutex` serialises each manifest
//! read-modify-write so concurrent adds/removes cannot lost-update. The lock is
//! held for the entire RMW (a brief synchronous `std::fs` round-trip); callers
//! drive these from `spawn_blocking` in `ipc-bridge`.
//!
//! # Path-traversal guard
//!
//! [`read_attachment_original`] and [`read_attachment_markdown`] reject any
//! filename containing a path separator or `..` before touching the filesystem
//! (mirroring [`crate::assets`]).

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use minutist_common::{AppError, AppResult, AttachmentEntry, AttachmentId, ConversionState, MeetingId};
use notes_crdt::MeetingFolder;
use sha2::{Digest, Sha256};

use crate::error::Error;

// ---------------------------------------------------------------------------
// Per-meeting manifest write lock
// ---------------------------------------------------------------------------

/// Process-wide registry of per-meeting manifest mutexes.
///
/// Each `MeetingId` gets its own `Mutex<()>`; the RMW (read-manifest,
/// modify, write-atomic) is performed under this lock so concurrent adds or
/// removes cannot interleave and lose an entry.
///
/// The map grows by one `Arc<Mutex<()>>` per meeting touched and entries are
/// never reclaimed. This unbounded-by-meeting-count growth is accepted: it
/// mirrors the offline-mutex precedent elsewhere in this crate, an empty mutex
/// is tiny, and the meeting count is bounded by what a single user accumulates
/// over the app's lifetime (orders of magnitude below any memory concern).
static MANIFEST_LOCKS: OnceLock<Mutex<HashMap<MeetingId, Arc<Mutex<()>>>>> = OnceLock::new();

fn manifest_locks() -> &'static Mutex<HashMap<MeetingId, Arc<Mutex<()>>>> {
    MANIFEST_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Return (or lazily create) the per-meeting manifest lock.
fn manifest_lock(id: MeetingId) -> Arc<Mutex<()>> {
    let mut map = manifest_locks()
        .lock()
        .expect("MANIFEST_LOCKS registry poisoned");
    map.entry(id).or_insert_with(|| Arc::new(Mutex::new(()))).clone()
}

// ---------------------------------------------------------------------------
// Directory / path helpers
// ---------------------------------------------------------------------------

/// `{root}/{meeting_id}/attachments/`
fn attachments_dir(root: &Path, meeting_id: MeetingId) -> PathBuf {
    MeetingFolder::open(root, meeting_id).attachments_dir()
}

/// `{root}/{meeting_id}/attachments/attachments.json`
fn manifest_path(root: &Path, meeting_id: MeetingId) -> PathBuf {
    MeetingFolder::open(root, meeting_id).attachments_manifest_path()
}

// ---------------------------------------------------------------------------
// Path-traversal guard (mirrors assets.rs `is_safe_asset_filename`)
// ---------------------------------------------------------------------------

/// A filename is safe iff it is non-empty, contains no path separator, and is a
/// single normal path component (rejects `.`, `..`, absolute paths, and any
/// embedded separator on either platform).
fn is_safe_filename(filename: &str) -> bool {
    if filename.is_empty() {
        return false;
    }
    if filename.contains('/') || filename.contains('\\') {
        return false;
    }
    if filename.contains("..") {
        return false;
    }
    let mut components = Path::new(filename).components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(c)), None) => c.to_str() == Some(filename),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Store the raw attachment bytes under `attachments/` using a content-addressed
/// filename (`<sha256hex>.<ext>`).
///
/// Creates the `attachments/` subdirectory on demand. If the target file already
/// exists the write is skipped (identical bytes → identical hash). Returns the
/// hex-encoded SHA-256 hash (without the extension) so the caller can build the
/// [`AttachmentEntry`].
pub fn save_attachment_original(
    root: &Path,
    meeting_id: MeetingId,
    bytes: &[u8],
    ext: &str,
) -> AppResult<String> {
    let dir = attachments_dir(root, meeting_id);
    std::fs::create_dir_all(&dir)
        .map_err(Error::Io)
        .map_err(AppError::from)?;

    let hash = format!("{:x}", Sha256::digest(bytes));
    let filename = format!("{hash}.{ext}");
    let target = dir.join(&filename);

    if target.exists() {
        tracing::debug!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            filename = %filename,
            "attachment original already present (dedupe)"
        );
        return Ok(hash);
    }

    minutist_common::fs::write_atomic(&target, bytes)?;

    tracing::info!(
        target: "persistence",
        meeting_id = %meeting_id.0,
        filename = %filename,
        bytes = bytes.len(),
        "attachment original saved"
    );
    Ok(hash)
}

/// Read the raw bytes of an attachment original by its `<hash>.<ext>` filename.
///
/// Rejects any `filename` containing a path separator or `..` before reading
/// (path-traversal guard).
pub fn read_attachment_original(
    root: &Path,
    meeting_id: MeetingId,
    filename: &str,
) -> AppResult<Vec<u8>> {
    if !is_safe_filename(filename) {
        return Err(AppError::InvalidInput {
            context: format!("rejected unsafe attachment filename: {filename:?}"),
        });
    }
    let path = attachments_dir(root, meeting_id).join(filename);
    let bytes = std::fs::read(&path).map_err(Error::Io).map_err(AppError::from)?;
    Ok(bytes)
}

/// Absolute path of an attachment original
/// (`{root}/{meeting_id}/attachments/<filename>`), with the same
/// path-traversal guard as [`read_attachment_original`]. Does NOT read the
/// file — callers that hand the original to the host OS (opening it in the
/// default application) use this to avoid loading the bytes.
pub fn attachment_original_path(
    root: &Path,
    meeting_id: MeetingId,
    filename: &str,
) -> AppResult<PathBuf> {
    if !is_safe_filename(filename) {
        return Err(AppError::InvalidInput {
            context: format!("rejected unsafe attachment filename: {filename:?}"),
        });
    }
    Ok(attachments_dir(root, meeting_id).join(filename))
}

/// Write the converted markdown `md` to `attachments/<hash>.md` atomically.
///
/// Returns the filename (`"<hash>.md"`) so the caller can store it on the
/// manifest row.
pub fn save_attachment_markdown(
    root: &Path,
    meeting_id: MeetingId,
    hash: &str,
    md: &str,
) -> AppResult<String> {
    let dir = attachments_dir(root, meeting_id);
    std::fs::create_dir_all(&dir)
        .map_err(Error::Io)
        .map_err(AppError::from)?;

    let filename = format!("{hash}.md");
    let target = dir.join(&filename);
    minutist_common::fs::write_atomic(&target, md.as_bytes())?;

    tracing::info!(
        target: "persistence",
        meeting_id = %meeting_id.0,
        filename = %filename,
        "attachment markdown saved"
    );
    Ok(filename)
}

/// Read the converted markdown for an attachment by its `<hash>.md` filename.
///
/// Rejects traversal attempts (path-traversal guard).
pub fn read_attachment_markdown(
    root: &Path,
    meeting_id: MeetingId,
    filename: &str,
) -> AppResult<String> {
    if !is_safe_filename(filename) {
        return Err(AppError::InvalidInput {
            context: format!("rejected unsafe attachment markdown filename: {filename:?}"),
        });
    }
    let path = attachments_dir(root, meeting_id).join(filename);
    let bytes = std::fs::read(&path).map_err(Error::Io).map_err(AppError::from)?;
    String::from_utf8(bytes).map_err(|e| AppError::Internal {
        context: format!("attachment markdown is not valid UTF-8: {e}"),
    })
}

/// Best-effort remove of `<hash>.<ext>` and `<hash>.md` from the attachments
/// directory. `NotFound` is silently ignored on each file.
///
/// Only call this when the dedup check has confirmed no surviving manifest row
/// references the same hash (see [`remove_manifest_entry`]).
pub fn unlink_attachment_files(
    root: &Path,
    meeting_id: MeetingId,
    hash: &str,
    ext: &str,
) {
    let dir = attachments_dir(root, meeting_id);
    for filename in [format!("{hash}.{ext}"), format!("{hash}.md")] {
        let path = dir.join(&filename);
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::debug!(
                target: "persistence",
                meeting_id = %meeting_id.0,
                filename = %filename,
                "attachment file unlinked"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                target: "persistence",
                meeting_id = %meeting_id.0,
                filename = %filename,
                "failed to unlink attachment file: {e}"
            ),
        }
    }
}

/// Dedup-safe cleanup of an orphaned `<hash>.md` written by a conversion that
/// finished after its manifest row was removed.
///
/// Takes the per-meeting manifest lock, then unlinks `<hash>.md` ONLY when no
/// surviving manifest row references `hash` (another row sharing the same
/// content hash may legitimately own that markdown). `NotFound` is ignored.
/// The original `<hash>.<ext>` is left untouched — `remove_manifest_entry`
/// already handled it under the same dedup rule.
pub fn unlink_orphan_attachment_markdown(
    root: &Path,
    meeting_id: MeetingId,
    hash: &str,
) -> AppResult<()> {
    let lock = manifest_lock(meeting_id);
    let _guard = lock.lock().expect("manifest lock poisoned");

    let entries = read_manifest_unlocked(root, meeting_id)?;
    if entries.iter().any(|e| e.hash == hash) {
        // A surviving row still references this hash; its markdown is live.
        return Ok(());
    }

    let path = attachments_dir(root, meeting_id).join(format!("{hash}.md"));
    match std::fs::remove_file(&path) {
        Ok(()) => tracing::debug!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            hash = %hash,
            "orphaned attachment markdown unlinked"
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(Error::Io(e).into()),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Manifest CRUD
// ---------------------------------------------------------------------------

/// Read the `attachments.json` manifest for a meeting.
///
/// An absent manifest file is a legitimate empty list (`Ok(vec![])`), mirroring
/// the chat store's not-found short-circuit. A present but unparseable manifest
/// is an error (single manifest file — unlike the per-session chat files, a
/// corrupt manifest hides everything).
pub fn read_manifest(root: &Path, meeting_id: MeetingId) -> AppResult<Vec<AttachmentEntry>> {
    let path = manifest_path(root, meeting_id);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::Io(e).into()),
    };
    let entries: Vec<AttachmentEntry> = serde_json::from_slice(&bytes)
        .map_err(Error::Serialise)
        .map_err(AppError::from)?;
    Ok(entries)
}

/// Append `entry` to the manifest under the per-meeting write lock.
///
/// The write is atomic (tmp+rename). The meeting folder is expected to exist;
/// the `attachments/` subdirectory is created on demand.
pub fn add_manifest_entry(
    root: &Path,
    meeting_id: MeetingId,
    entry: AttachmentEntry,
) -> AppResult<()> {
    let lock = manifest_lock(meeting_id);
    let _guard = lock.lock().expect("manifest lock poisoned");

    let dir = attachments_dir(root, meeting_id);
    std::fs::create_dir_all(&dir).map_err(Error::Io).map_err(AppError::from)?;

    let mut entries = read_manifest_unlocked(root, meeting_id)?;
    entries.push(entry);
    write_manifest_unlocked(root, meeting_id, &entries)
}

/// Update the `conversion` state and `converted_md_filename` of an existing
/// manifest row, under the per-meeting write lock.
///
/// Returns `true` when a row with `id` was found and updated, `false` when the
/// id was absent (e.g. the attachment was removed while a conversion was in
/// flight). A `false` result lets the caller skip a stale event emit and clean
/// up any markdown it just wrote for a row that no longer exists. When the id is
/// absent the manifest is left unchanged (idempotent).
pub fn set_entry_conversion(
    root: &Path,
    meeting_id: MeetingId,
    id: AttachmentId,
    state: ConversionState,
    md_filename: Option<String>,
) -> AppResult<bool> {
    let lock = manifest_lock(meeting_id);
    let _guard = lock.lock().expect("manifest lock poisoned");

    let mut entries = read_manifest_unlocked(root, meeting_id)?;
    let found = match entries.iter_mut().find(|e| e.id == id) {
        Some(row) => {
            row.conversion = state;
            row.converted_md_filename = md_filename;
            true
        }
        None => false,
    };
    write_manifest_unlocked(root, meeting_id, &entries)?;
    Ok(found)
}

/// Set the `awareness` text on an existing manifest row, under the per-meeting
/// write lock.
///
/// Mirrors `set_entry_conversion`. Returns `true` when a row with `id` was
/// found and updated, `false` when the id is absent (e.g. the attachment was
/// removed while the awareness pass was running). A `false` result is not an
/// error — the manifest is left unchanged (idempotent).
pub fn set_entry_awareness(
    root: &Path,
    meeting_id: MeetingId,
    id: AttachmentId,
    awareness: &str,
) -> AppResult<bool> {
    let lock = manifest_lock(meeting_id);
    let _guard = lock.lock().expect("manifest lock poisoned");

    let mut entries = read_manifest_unlocked(root, meeting_id)?;
    let found = match entries.iter_mut().find(|e| e.id == id) {
        Some(row) => {
            row.awareness = Some(awareness.to_owned());
            true
        }
        None => false,
    };
    write_manifest_unlocked(root, meeting_id, &entries)?;
    Ok(found)
}

/// Remove the manifest row for `id`, under the per-meeting write lock.
///
/// After writing the new manifest, performs a dedup-safe unlink: the original
/// and markdown sibling are only removed when no surviving row references the
/// same hash.
///
/// Returns the removed entry (or `None` when absent — idempotent).
pub fn remove_manifest_entry(
    root: &Path,
    meeting_id: MeetingId,
    id: AttachmentId,
) -> AppResult<Option<(AttachmentEntry, bool)>> {
    let lock = manifest_lock(meeting_id);
    let _guard = lock.lock().expect("manifest lock poisoned");

    let mut entries = read_manifest_unlocked(root, meeting_id)?;
    let pos = entries.iter().position(|e| e.id == id);
    let removed = match pos {
        Some(i) => entries.remove(i),
        None => return Ok(None),
    };

    write_manifest_unlocked(root, meeting_id, &entries)?;

    // Dedup-safe: the hash is orphaned iff no surviving row shares it. Computed here
    // under the manifest lock and returned so a caller (the RAG chunk-forget) keys
    // off the SAME decision that gates the markdown unlink — no re-read, no race.
    let hash_orphaned = !entries.iter().any(|e| e.hash == removed.hash);
    if hash_orphaned {
        unlink_attachment_files(root, meeting_id, &removed.hash, &removed.ext);
    }

    Ok(Some((removed, hash_orphaned)))
}

/// Read the `(original_filename, markdown)` of every `Ready` attachment in
/// manifest order. The caller (`ipc-bridge`) assembles the reference-material
/// block and applies the per-attachment budget, so this returns the raw parts
/// rather than a single concatenation.
///
/// Returns an empty `Vec` when no `Ready` attachments exist (the no-attachment
/// summarise path stays byte-identical).
pub fn read_attachments_markdown_parts(
    root: &Path,
    meeting_id: MeetingId,
) -> AppResult<Vec<(String, String)>> {
    let entries = read_manifest(root, meeting_id)?;
    let mut parts = Vec::new();
    for entry in &entries {
        if let ConversionState::Ready = entry.conversion {
            if let Some(md_filename) = &entry.converted_md_filename {
                match read_attachment_markdown(root, meeting_id, md_filename) {
                    Ok(md) => parts.push((entry.original_filename.clone(), md)),
                    Err(e) => {
                        tracing::warn!(
                            target: "persistence",
                            meeting_id = %meeting_id.0,
                            attachment_id = %entry.id.0,
                            "skipping attachment markdown (read error): {e}"
                        );
                    }
                }
            }
        }
    }
    Ok(parts)
}

// ---------------------------------------------------------------------------
// Private helpers (unlocked — caller holds the manifest_lock guard)
// ---------------------------------------------------------------------------

/// Read the manifest without acquiring the per-meeting lock. Must only be
/// called while the caller already holds `manifest_lock(meeting_id)`.
fn read_manifest_unlocked(root: &Path, meeting_id: MeetingId) -> AppResult<Vec<AttachmentEntry>> {
    let path = manifest_path(root, meeting_id);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::Io(e).into()),
    };
    let entries: Vec<AttachmentEntry> = serde_json::from_slice(&bytes)
        .map_err(Error::Serialise)
        .map_err(AppError::from)?;
    Ok(entries)
}

/// Serialise `entries` and write atomically to the manifest path. Must only be
/// called while the caller holds `manifest_lock(meeting_id)`.
fn write_manifest_unlocked(
    root: &Path,
    meeting_id: MeetingId,
    entries: &[AttachmentEntry],
) -> AppResult<()> {
    let path = manifest_path(root, meeting_id);
    let bytes = serde_json::to_vec_pretty(entries)
        .map_err(Error::Serialise)
        .map_err(AppError::from)?;
    minutist_common::fs::write_atomic(&path, &bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::AttachmentId;
    use notes_crdt::MeetingFolder;
    use tempfile::TempDir;

    fn make_meeting() -> (TempDir, PathBuf, MeetingId) {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path().to_path_buf();
        let id = MeetingId::new();
        MeetingFolder::create(&root, id).expect("create meeting folder");
        (tempdir, root, id)
    }

    fn make_entry(hash: &str, ext: &str, name: &str) -> AttachmentEntry {
        AttachmentEntry {
            id: AttachmentId::new(),
            hash: hash.to_string(),
            original_filename: name.to_string(),
            ext: ext.to_string(),
            byte_len: 42,
            added_at: "2026-06-22T10:00:00Z".to_string(),
            conversion: ConversionState::Pending,
            converted_md_filename: None,
            awareness: None,
        }
    }

    // -- File I/O --

    #[test]
    fn save_read_original_round_trip() {
        let (_td, root, id) = make_meeting();
        let bytes = b"hello attachment bytes";
        let hash = save_attachment_original(&root, id, bytes, "pdf").expect("save");
        assert_eq!(hash.len(), 64, "expected 64-char hex hash");

        let filename = format!("{hash}.pdf");
        let read = read_attachment_original(&root, id, &filename).expect("read");
        assert_eq!(read, bytes);
    }

    #[test]
    fn identical_bytes_dedupe_to_one_file() {
        let (_td, root, id) = make_meeting();
        let bytes = b"same bytes";
        let h1 = save_attachment_original(&root, id, bytes, "txt").expect("save 1");
        let h2 = save_attachment_original(&root, id, bytes, "txt").expect("save 2");
        assert_eq!(h1, h2, "same bytes must produce same hash");

        let dir = attachments_dir(&root, id);
        let files: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.ends_with(".tmp"))
            .collect();
        // Only one original file; no manifest yet.
        assert_eq!(files.len(), 1, "deduped: expected 1 file, got {files:?}");
    }

    #[test]
    fn read_original_rejects_traversal() {
        let (_td, root, id) = make_meeting();
        let bytes = b"real file";
        let hash = save_attachment_original(&root, id, bytes, "pdf").expect("save");
        // Good filename works.
        assert!(read_attachment_original(&root, id, &format!("{hash}.pdf")).is_ok());
        // Traversal attempts are rejected.
        for evil in ["../secret", "sub/dir.pdf", "..", ".", "", "a\\b.pdf", "/etc/passwd"] {
            assert!(
                read_attachment_original(&root, id, evil).is_err(),
                "expected traversal rejection for {evil:?}"
            );
        }
    }

    #[test]
    fn save_read_markdown_round_trip() {
        let (_td, root, id) = make_meeting();
        let hash = save_attachment_original(&root, id, b"doc bytes", "pdf").expect("save original");
        let md = "# Document\n\nSome content.";
        let md_filename = save_attachment_markdown(&root, id, &hash, md).expect("save md");
        assert_eq!(md_filename, format!("{hash}.md"));

        let read = read_attachment_markdown(&root, id, &md_filename).expect("read md");
        assert_eq!(read, md);
    }

    #[test]
    fn read_markdown_rejects_traversal() {
        let (_td, root, id) = make_meeting();
        let hash = save_attachment_original(&root, id, b"x", "txt").expect("save");
        save_attachment_markdown(&root, id, &hash, "# Hello").expect("save md");
        for evil in ["../secret.md", "..", "sub/file.md", "", "..\\win.md"] {
            assert!(
                read_attachment_markdown(&root, id, evil).is_err(),
                "expected traversal rejection for {evil:?}"
            );
        }
    }

    // -- Manifest CRUD --

    #[test]
    fn read_absent_manifest_returns_empty() {
        let (_td, root, id) = make_meeting();
        let entries = read_manifest(&root, id).expect("read");
        assert!(entries.is_empty());
    }

    #[test]
    fn add_list_remove_round_trip() {
        let (_td, root, id) = make_meeting();
        let entry = make_entry("aabbcc", "pdf", "report.pdf");
        let eid = entry.id;

        add_manifest_entry(&root, id, entry.clone()).expect("add");
        let list = read_manifest(&root, id).expect("list");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, eid);

        let removed = remove_manifest_entry(&root, id, eid).expect("remove");
        assert!(removed.is_some());
        let list2 = read_manifest(&root, id).expect("list after remove");
        assert!(list2.is_empty());
    }

    #[test]
    fn remove_absent_entry_is_idempotent() {
        let (_td, root, id) = make_meeting();
        let result = remove_manifest_entry(&root, id, AttachmentId::new()).expect("remove");
        assert!(result.is_none(), "absent entry remove should return None");
    }

    #[test]
    fn set_entry_conversion_pending_to_ready() {
        let (_td, root, id) = make_meeting();
        let entry = make_entry("deadbeef", "xlsx", "data.xlsx");
        let eid = entry.id;

        add_manifest_entry(&root, id, entry).expect("add");
        let found = set_entry_conversion(
            &root,
            id,
            eid,
            ConversionState::Ready,
            Some("deadbeef.md".to_string()),
        )
        .expect("set conversion");
        assert!(found, "row was present so the update must report found");

        let updated = read_manifest(&root, id).expect("read after set");
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].conversion, ConversionState::Ready);
        assert_eq!(
            updated[0].converted_md_filename,
            Some("deadbeef.md".to_string())
        );
    }

    #[test]
    fn set_entry_conversion_to_failed() {
        let (_td, root, id) = make_meeting();
        let entry = make_entry("cafebabe", "pptx", "slides.pptx");
        let eid = entry.id;

        add_manifest_entry(&root, id, entry).expect("add");
        let found = set_entry_conversion(
            &root,
            id,
            eid,
            ConversionState::Failed("unsupported layout".to_string()),
            None,
        )
        .expect("set conversion failed");
        assert!(found, "row was present so the update must report found");

        let updated = read_manifest(&root, id).expect("read after set");
        assert_eq!(
            updated[0].conversion,
            ConversionState::Failed("unsupported layout".to_string())
        );
    }

    #[test]
    fn set_entry_conversion_absent_id_reports_false() {
        let (_td, root, id) = make_meeting();
        // One real row so the manifest/dir exists, then update a DIFFERENT id:
        // the absent id must report `false` and leave the present row untouched
        // (models the remove-then-convert race where the row is already gone).
        let present = make_entry("livehash", "pdf", "live.pdf");
        let present_id = present.id;
        add_manifest_entry(&root, id, present).expect("add");

        let found = set_entry_conversion(
            &root,
            id,
            AttachmentId::new(),
            ConversionState::Ready,
            Some("nope.md".to_string()),
        )
        .expect("set conversion on absent id");
        assert!(!found, "absent id must report not-found");

        let entries = read_manifest(&root, id).expect("read");
        assert_eq!(entries.len(), 1, "the present row must survive");
        assert_eq!(entries[0].id, present_id);
        assert_eq!(
            entries[0].conversion,
            ConversionState::Pending,
            "the present row's state must be untouched by the absent-id update"
        );
    }

    // -- set_entry_awareness --

    #[test]
    fn set_entry_awareness_roundtrip() {
        let (_td, root, id) = make_meeting();
        let entry = make_entry("awarehash", "pdf", "brief.pdf");
        let eid = entry.id;

        add_manifest_entry(&root, id, entry).expect("add");
        let found =
            set_entry_awareness(&root, id, eid, "One-sentence summary.\n\nKeywords: foo, bar")
                .expect("set awareness");
        assert!(found, "row was present so the update must report found");

        let updated = read_manifest(&root, id).expect("read after set");
        assert_eq!(updated.len(), 1);
        assert_eq!(
            updated[0].awareness,
            Some("One-sentence summary.\n\nKeywords: foo, bar".to_string())
        );
    }

    #[test]
    fn set_entry_awareness_absent_id_reports_false() {
        let (_td, root, id) = make_meeting();
        // One real row so the manifest/dir exists; update a different id.
        let present = make_entry("livehash2", "pdf", "live2.pdf");
        let present_id = present.id;
        add_manifest_entry(&root, id, present).expect("add");

        let found = set_entry_awareness(&root, id, AttachmentId::new(), "should not apply")
            .expect("set awareness on absent id");
        assert!(!found, "absent id must report not-found");

        let entries = read_manifest(&root, id).expect("read");
        assert_eq!(entries.len(), 1, "the present row must survive");
        assert_eq!(entries[0].id, present_id);
        assert_eq!(
            entries[0].awareness, None,
            "the present row's awareness must be untouched by the absent-id update"
        );
    }

    #[test]
    fn set_entry_awareness_absent_field_deserialises_from_old_manifest() {
        // A manifest serialised without the `awareness` field must still
        // deserialise — verifying the `#[serde(default)]` attribute works.
        let (_td, root, id) = make_meeting();
        let entry = make_entry("oldhash", "txt", "old.txt");
        add_manifest_entry(&root, id, entry).expect("add");

        // Rewrite the manifest JSON with the `awareness` key stripped out.
        let dir = attachments_dir(&root, id);
        let path = dir.join("attachments.json");
        let raw = std::fs::read_to_string(&path).expect("read manifest");
        let stripped = raw
            .lines()
            .filter(|l| !l.contains("\"awareness\""))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, stripped.as_bytes()).expect("write stripped manifest");

        let entries = read_manifest(&root, id).expect("deserialise old manifest");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].awareness, None, "missing field defaults to None");
    }

    // -- Dedup-safe remove --

    #[test]
    fn remove_last_row_unlinks_files() {
        let (_td, root, id) = make_meeting();
        let bytes = b"uniq bytes for this test";
        let hash = save_attachment_original(&root, id, bytes, "pdf").expect("save original");
        save_attachment_markdown(&root, id, &hash, "# doc").expect("save md");

        let mut entry = make_entry(&hash, "pdf", "doc.pdf");
        entry.conversion = ConversionState::Ready;
        entry.converted_md_filename = Some(format!("{hash}.md"));
        let eid = entry.id;
        add_manifest_entry(&root, id, entry).expect("add");

        remove_manifest_entry(&root, id, eid).expect("remove");

        // Both files must be gone.
        let dir = attachments_dir(&root, id);
        assert!(
            !dir.join(format!("{hash}.pdf")).exists(),
            "original must be unlinked after last row removed"
        );
        assert!(
            !dir.join(format!("{hash}.md")).exists(),
            "markdown must be unlinked after last row removed"
        );
    }

    #[test]
    fn remove_one_of_two_same_hash_keeps_files() {
        let (_td, root, id) = make_meeting();
        let bytes = b"shared bytes";
        let hash = save_attachment_original(&root, id, bytes, "txt").expect("save original");

        // Two manifest rows for the same hash (e.g. same file added twice under
        // different names — unusual but must be handled safely).
        let entry_a = make_entry(&hash, "txt", "copy_a.txt");
        let entry_b = make_entry(&hash, "txt", "copy_b.txt");
        let id_a = entry_a.id;
        add_manifest_entry(&root, id, entry_a).expect("add a");
        add_manifest_entry(&root, id, entry_b).expect("add b");

        // Remove only the first row.
        remove_manifest_entry(&root, id, id_a).expect("remove a");

        // The hash file must still exist (the second row still references it).
        let dir = attachments_dir(&root, id);
        assert!(
            dir.join(format!("{hash}.txt")).exists(),
            "original must NOT be unlinked when another row shares the hash"
        );
        // The second row must still be in the manifest.
        let remaining = read_manifest(&root, id).expect("read after remove");
        assert_eq!(remaining.len(), 1);
    }

    // -- Concurrency (lost-update guard) --

    #[test]
    fn concurrent_adds_both_survive() {
        let (_td, root, id) = make_meeting();
        let root_a = root.clone();
        let root_b = root.clone();

        let entry_a = make_entry("hash_aaa", "pdf", "a.pdf");
        let entry_b = make_entry("hash_bbb", "txt", "b.txt");

        let t1 = std::thread::spawn(move || {
            add_manifest_entry(&root_a, id, entry_a).expect("add a");
        });
        let t2 = std::thread::spawn(move || {
            add_manifest_entry(&root_b, id, entry_b).expect("add b");
        });
        t1.join().expect("thread 1");
        t2.join().expect("thread 2");

        let entries = read_manifest(&root, id).expect("read after concurrent adds");
        assert_eq!(
            entries.len(),
            2,
            "both concurrent adds must survive (no lost update)"
        );
    }

    // -- read_attachments_markdown_parts --

    #[test]
    fn parts_empty_when_no_ready_attachments() {
        let (_td, root, id) = make_meeting();
        let entry = make_entry("abc123", "pdf", "pending.pdf");
        add_manifest_entry(&root, id, entry).expect("add");

        let parts = read_attachments_markdown_parts(&root, id).expect("parts");
        assert!(parts.is_empty(), "no Ready attachments → no parts");
    }

    #[test]
    fn parts_empty_for_empty_manifest() {
        let (_td, root, id) = make_meeting();
        let parts = read_attachments_markdown_parts(&root, id).expect("parts");
        assert!(parts.is_empty());
    }

    #[test]
    fn parts_include_ready_attachments_in_order() {
        let (_td, root, id) = make_meeting();

        let hash1 = save_attachment_original(&root, id, b"bytes1", "txt").expect("save 1");
        let hash2 = save_attachment_original(&root, id, b"bytes2", "md").expect("save 2");
        save_attachment_markdown(&root, id, &hash1, "Content one.").expect("save md1");
        save_attachment_markdown(&root, id, &hash2, "Content two.").expect("save md2");

        let mut e1 = make_entry(&hash1, "txt", "first.txt");
        e1.conversion = ConversionState::Ready;
        e1.converted_md_filename = Some(format!("{hash1}.md"));

        let mut e2 = make_entry(&hash2, "md", "second.md");
        e2.conversion = ConversionState::Ready;
        e2.converted_md_filename = Some(format!("{hash2}.md"));

        add_manifest_entry(&root, id, e1).expect("add e1");
        add_manifest_entry(&root, id, e2).expect("add e2");

        let parts = read_attachments_markdown_parts(&root, id).expect("parts");
        assert_eq!(parts.len(), 2, "both Ready attachments returned");
        // Manifest order preserved, with (original_filename, raw markdown) pairs.
        assert_eq!(parts[0].0, "first.txt");
        assert_eq!(parts[0].1, "Content one.");
        assert_eq!(parts[1].0, "second.md");
        assert_eq!(parts[1].1, "Content two.");
    }

    #[test]
    fn successful_write_leaves_no_tmp_residue() {
        let (_td, root, id) = make_meeting();
        let entry = make_entry("noresid", "pdf", "clean.pdf");
        add_manifest_entry(&root, id, entry).expect("add");

        let dir = attachments_dir(&root, id);
        let tmps: Vec<_> = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(tmps.is_empty(), "expected no .tmp residue, found: {tmps:?}");
    }
}
