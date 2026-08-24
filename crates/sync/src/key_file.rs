//! Reading and writing a device's 32-byte secrets on disk.
//!
//! Two files share this shape: the ed25519 device key ([`crate::identity`]) and
//! the account content key ([`crate::content_key`]). Both are exactly 32 raw
//! bytes at the app-data base, owner-only, and both treat a wrong-length file as
//! an error rather than something to silently regenerate. The discipline lives
//! here so a future hardening (Windows per-file ACLs, `fsync`, write-to-temp then
//! rename) lands in one place.

use std::io::Write;
use std::path::Path;

use crate::{Error, Result};

/// Read a 32-byte secret, or `None` when the file does not exist.
///
/// `what` names the secret in any error message (e.g. `"device key"`). A file of
/// the wrong length is an error, never a fresh mint: reminting over a truncated
/// key would strand the device from every peer while looking like a first run.
pub(crate) fn read(path: &Path, what: &str) -> Result<Option<[u8; 32]>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let raw: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
                Error::Identity(format!(
                    "{what} at {path:?} is {} bytes, expected 32",
                    bytes.len()
                ))
            })?;
            Ok(Some(raw))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Identity(format!("reading {what} at {path:?}: {e}"))),
    }
}

/// Write a 32-byte secret, creating the file with owner-only mode `0600`
/// atomically on Unix, so it is never momentarily world-readable.
///
/// `mode` applies only on creation, hence the re-assert for a file that already
/// exists. On Windows the file inherits the parent directory's ACL; per-file ACL
/// tightening is not implemented (the same gap `app-main`'s `write_secret_file`
/// carries).
pub(crate) fn write(path: &Path, bytes: &[u8; 32]) -> std::io::Result<()> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(bytes)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_at_owner_only() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("secret");
        write(&path, &[7u8; 32]).expect("write");
        assert_eq!(read(&path, "secret").expect("read"), Some([7u8; 32]));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = std::fs::metadata(&path).expect("stat");
            assert_eq!(meta.len(), 32);
            assert_eq!(meta.permissions().mode() & 0o777, 0o600);
        }
    }

    #[test]
    fn absent_reads_as_none() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        assert_eq!(
            read(&dir.path().join("absent"), "secret").expect("read"),
            None
        );
    }

    #[test]
    fn a_wrong_length_file_is_an_error_not_a_remint() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("secret");
        std::fs::write(&path, b"short").expect("seed");
        assert!(read(&path, "secret").is_err());
    }

    #[test]
    fn overwrites_an_existing_file_and_keeps_the_mode() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("secret");
        // Seed world-readable, so the re-assert is what has to fix the mode.
        std::fs::write(&path, [1u8; 32]).expect("seed");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
                .expect("loosen");
        }
        write(&path, &[2u8; 32]).expect("overwrite");
        assert_eq!(read(&path, "secret").expect("read"), Some([2u8; 32]));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).expect("stat").permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "an existing file must be tightened");
        }
    }
}
