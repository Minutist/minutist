//! Note image assets — content-addressed files under a meeting's `assets/`.
//!
//! Images pasted or dropped into the notes editor are stored as **separate
//! files** under `{root}/{meeting_id}/assets/`, NOT embedded in `notes.json`.
//! `notes.json` references each asset by **bare filename** only — a portable,
//! machine-independent reference. Because `notes.json` and `assets/` live in the
//! same meeting folder, the folder can be copied to another machine and the
//! notes still resolve: nothing absolute or platform-specific is persisted.
//! See `architecture/cross-cutting.md` — "Note image assets".
//!
//! The filename is the SHA-256 content hash plus the extension
//! (`<hex>.<ext>`), so identical pastes dedupe to one file and an asset's
//! identity is independent of when/where it was written.
//!
//! # Opacity guarantee
//!
//! These are sibling files; `notes.json` is untouched by this module (the
//! editor stores the returned filename into the document, which
//! `notes_crdt::NotesStore` then round-trips verbatim). The Rust side never
//! parses the document to find image references.
//!
//! # Path-traversal guard
//!
//! [`read_note_asset`] REJECTS any `filename` containing a path separator or a
//! `..` component before touching the filesystem, so a request can only ever
//! name a file directly inside the meeting's `assets/` directory — never escape
//! it. The protocol handler in `app-main` relies on this guard.

use std::path::Path;

use minutist_common::{AppResult, MeetingId};
use notes_crdt::MeetingFolder;
use sha2::{Digest, Sha256};

use crate::error::Error;

/// Resolve `{root}/{meeting_id}/assets/`.
fn assets_dir(root: &Path, meeting_id: MeetingId) -> std::path::PathBuf {
    MeetingFolder::open(root, meeting_id).assets_dir()
}

/// Persist a note image `bytes` under the meeting's `assets/` directory and
/// return its **portable** reference: the bare `<contenthash>.<ext>` filename.
///
/// The filename is content-addressed (SHA-256 of `bytes`), so saving the same
/// image twice writes one file and returns the same name (dedupe). The assets
/// directory is created on demand; the meeting folder itself is expected to
/// exist (it is owned by `MeetingWriter`).
///
/// `ext` is the lower-cased extension WITHOUT a leading dot (e.g. `"png"`); the
/// caller (`ipc-bridge`) validates it against the image allowlist. The returned
/// filename is what the editor stores into `notes.json`; the rendered webview
/// URL is derived from it at display time (never persisted).
///
/// The write is atomic-on-rename to a sibling temp file so a crash mid-write
/// leaves no truncated asset; an existing identical file is left in place
/// (content-addressed → byte-identical), so a re-paste is a cheap no-op.
pub fn save_note_asset(
    root: &Path,
    meeting_id: MeetingId,
    bytes: &[u8],
    ext: &str,
) -> AppResult<String> {
    let dir = assets_dir(root, meeting_id);
    std::fs::create_dir_all(&dir)
        .map_err(Error::Io)
        .map_err(minutist_common::AppError::from)?;

    let hash = Sha256::digest(bytes);
    let filename = format!("{:x}.{ext}", hash);
    let target = dir.join(&filename);

    // Content-addressed: an existing file with this name is byte-identical, so a
    // re-paste need not rewrite it.
    if target.exists() {
        tracing::debug!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            filename = %filename,
            "note asset already present (dedupe)"
        );
        return Ok(filename);
    }

    minutist_common::fs::write_atomic(&target, bytes)?;

    tracing::info!(
        target: "persistence",
        meeting_id = %meeting_id.0,
        filename = %filename,
        bytes = bytes.len(),
        "note asset saved"
    );

    Ok(filename)
}

/// Read the bytes of a note image asset by its portable filename.
///
/// REJECTS any `filename` that contains a path separator or a `..`/`.` path
/// component (path-traversal guard, [`AppError::InvalidInput`]) before reading,
/// so the resolved path can only ever be a file directly inside the meeting's
/// `assets/` directory. A missing file surfaces as [`AppError::Io`].
pub fn read_note_asset(root: &Path, meeting_id: MeetingId, filename: &str) -> AppResult<Vec<u8>> {
    if !is_safe_asset_filename(filename) {
        return Err(minutist_common::AppError::InvalidInput {
            context: format!("rejected unsafe asset filename: {filename:?}"),
        });
    }

    let path = assets_dir(root, meeting_id).join(filename);
    let bytes = std::fs::read(&path)
        .map_err(Error::Io)
        .map_err(minutist_common::AppError::from)?;
    Ok(bytes)
}

/// A filename is safe iff it is non-empty, contains no path separator, and is a
/// single normal path component (rejects `.`, `..`, absolute paths, and any
/// embedded separator on either platform).
fn is_safe_asset_filename(filename: &str) -> bool {
    if filename.is_empty() {
        return false;
    }
    // Reject both Unix and Windows separators explicitly (a Windows-style
    // separator would NOT be split by `Path::components` on Unix, so the bare
    // check is required to stay platform-independent).
    if filename.contains('/') || filename.contains('\\') {
        return false;
    }
    if filename.contains("..") {
        return false;
    }
    // The whole string must be exactly one Normal path component.
    let mut components = Path::new(filename).components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(c)), None) => c.to_str() == Some(filename),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notes_crdt::MeetingFolder;
    use tempfile::TempDir;

    fn make_meeting() -> (TempDir, std::path::PathBuf, MeetingId) {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path().to_path_buf();
        let id = MeetingId::new();
        MeetingFolder::create(&root, id).expect("create meeting folder");
        (tempdir, root, id)
    }

    #[test]
    fn save_then_read_round_trips() {
        let (_tempdir, root, id) = make_meeting();
        let bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes".to_vec();

        let filename = save_note_asset(&root, id, &bytes, "png").expect("save");
        assert!(filename.ends_with(".png"));
        // Content-hash filename: 64 hex chars + ".png".
        assert_eq!(filename.len(), 64 + 4);

        let read = read_note_asset(&root, id, &filename).expect("read");
        assert_eq!(read, bytes, "asset bytes did not round-trip");
    }

    #[test]
    fn identical_bytes_dedupe_to_one_file() {
        let (_tempdir, root, id) = make_meeting();
        let bytes = b"identical-image".to_vec();

        let a = save_note_asset(&root, id, &bytes, "jpg").expect("save a");
        let b = save_note_asset(&root, id, &bytes, "jpg").expect("save b");
        assert_eq!(a, b, "identical pastes must produce the same filename");

        let entries: Vec<_> = std::fs::read_dir(root.join(id.0.to_string()).join("assets"))
            .expect("read assets dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.ends_with(".tmp"))
            .collect();
        assert_eq!(entries.len(), 1, "expected one deduped file, got: {entries:?}");
    }

    #[test]
    fn distinct_bytes_produce_distinct_files() {
        let (_tempdir, root, id) = make_meeting();
        let a = save_note_asset(&root, id, b"one", "png").expect("save a");
        let b = save_note_asset(&root, id, b"two", "png").expect("save b");
        assert_ne!(a, b, "different bytes must hash to different filenames");
    }

    #[test]
    fn read_rejects_path_traversal() {
        let (_tempdir, root, id) = make_meeting();
        // Seed a real asset so a successful read is possible; the traversal
        // attempts below must still be rejected before any read.
        let good = save_note_asset(&root, id, b"x", "png").expect("save");
        assert!(read_note_asset(&root, id, &good).is_ok());

        for evil in [
            "../secret.txt",
            "..",
            ".",
            "sub/dir.png",
            "a/../../etc/passwd",
            "/etc/passwd",
            "..\\windows\\system32",
            "dir\\file.png",
            "",
        ] {
            let err = read_note_asset(&root, id, evil);
            assert!(
                err.is_err(),
                "expected traversal/invalid filename {evil:?} to be rejected, got Ok"
            );
        }
    }

    #[test]
    fn read_missing_asset_is_io_error() {
        let (_tempdir, root, id) = make_meeting();
        // A well-formed but non-existent content-hash filename.
        let missing = format!("{}.png", "a".repeat(64));
        let err = read_note_asset(&root, id, &missing);
        assert!(err.is_err(), "missing asset must error, not return empty");
    }

    #[test]
    fn save_does_not_touch_notes_files() {
        // The opacity guarantee: writing an asset must not create or modify
        // notes.json / notes.md.
        let (_tempdir, root, id) = make_meeting();
        let folder = root.join(id.0.to_string());
        save_note_asset(&root, id, b"img", "webp").expect("save");
        assert!(!folder.join("notes.json").exists());
        assert!(!folder.join("notes.md").exists());
    }
}
