//! Shared atomic-file-write primitive.
//!
//! Every crate that persists JSON or binary blobs to disk (`persistence`,
//! `notes-crdt`, `settings`) writes them the same way: to a sibling temp file,
//! `fsync`d, then renamed over the destination. [`write_atomic`] is the one
//! implementation; callers keep their own surrounding logic (directory
//! creation, serialisation, dedup checks) and call this for the actual
//! write-and-commit step.

use std::io::Write;
use std::path::Path;

use crate::{AppError, AppResult};

/// Atomically write `bytes` to `path`.
///
/// Writes to a sibling temp file in `path`'s parent directory (so the final
/// rename stays on one filesystem), flushes, `fsync`s it, then renames it over
/// `path`. The rename is the sole commit point: a crash or kill at any point
/// before it leaves the previous `path` untouched, and a crash after it leaves
/// `path` fully written — never truncated or partially written. The temp
/// file's name carries a random suffix so two concurrent writers targeting the
/// same `path` (e.g. two identical content-addressed uploads racing) never
/// share a temp file and clobber each other's write. The file handle is
/// dropped before the rename (required on Windows, where renaming over an
/// open handle fails).
///
/// On any error — during the write or the rename — the temp file is removed
/// on a best-effort basis so no `.tmp` residue is left behind.
///
/// `path`'s parent directory must already exist; this helper does not create
/// it, since only the caller knows whether that is expected (some callers
/// create the directory on demand first; others treat a missing parent as a
/// caller bug to surface).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> AppResult<()> {
    let parent = path.parent().ok_or_else(|| AppError::Internal {
        context: format!("write_atomic: path has no parent: {}", path.display()),
    })?;
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let tmp_path = parent.join(format!("{file_name}.{}.tmp", uuid::Uuid::new_v4()));

    let write_result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(io_err(e));
    }

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(io_err(e));
    }

    Ok(())
}

fn io_err(e: std::io::Error) -> AppError {
    AppError::Io {
        context: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_bytes() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("file.bin");
        write_atomic(&path, b"hello world").expect("write");
        assert_eq!(std::fs::read(&path).expect("read"), b"hello world");
    }

    #[test]
    fn overwrites_existing_file() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("file.txt");
        write_atomic(&path, b"first").expect("write first");
        write_atomic(&path, b"second").expect("write second");
        assert_eq!(std::fs::read(&path).expect("read"), b"second");
    }

    #[test]
    fn leaves_no_tmp_residue_on_success() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("file.txt");
        write_atomic(&path, b"content").expect("write");

        let residue: Vec<_> = std::fs::read_dir(tmp.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(residue.is_empty(), "expected no .tmp residue, found: {residue:?}");
    }

    #[test]
    fn errors_when_path_has_no_parent() {
        let path = Path::new("");
        let err = write_atomic(path, b"x");
        assert!(err.is_err());
    }

    /// Two threads writing DIFFERENT bytes to the SAME path concurrently must
    /// each see one of the two full writes on disk — never a mix of both
    /// (which the random-suffixed temp name, rather than a shared fixed temp
    /// name, is what prevents: two writers sharing one temp file could
    /// otherwise interleave their writes and rename over a corrupted blend).
    #[test]
    fn concurrent_writers_never_corrupt_the_target() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("shared.bin");

        let a = b"A".repeat(200_000);
        let b = b"B".repeat(200_000);
        let path_a = path.clone();
        let path_b = path.clone();
        let a2 = a.clone();
        let b2 = b.clone();

        let t1 = std::thread::spawn(move || write_atomic(&path_a, &a2));
        let t2 = std::thread::spawn(move || write_atomic(&path_b, &b2));
        t1.join().expect("thread 1").expect("write a");
        t2.join().expect("thread 2").expect("write b");

        let final_bytes = std::fs::read(&path).expect("read final");
        assert!(
            final_bytes == a || final_bytes == b,
            "final content must be exactly one writer's bytes, not a blend of both"
        );
    }

    #[test]
    fn failure_before_rename_leaves_prior_content_intact() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let path = tmp.path().join("file.txt");
        write_atomic(&path, b"original").expect("seed original content");

        // Occupy the whole directory with read-only permissions so the temp
        // file's `File::create` fails before any rename is attempted,
        // regardless of the random temp filename chosen.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(tmp.path()).expect("metadata").permissions();
            perms.set_mode(0o500); // read + execute, no write
            std::fs::set_permissions(tmp.path(), perms.clone()).expect("set readonly");

            let result = write_atomic(&path, b"new content that must not land");

            perms.set_mode(0o700);
            std::fs::set_permissions(tmp.path(), perms).expect("restore writable");

            assert!(
                result.is_err(),
                "write must fail when the temp file cannot be created"
            );
            assert_eq!(
                std::fs::read(&path).expect("read after failed write"),
                b"original",
                "a write that fails before rename must leave the previous content intact"
            );
        }
    }
}
