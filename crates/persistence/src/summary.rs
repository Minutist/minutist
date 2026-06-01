//! `summary.md` write/read helpers.
//!
//! Phase 5's `summariser` produces the markdown summary; Phase 4 lands only
//! the [`crate::MeetingFolder::summary_path`] helper and these I/O functions so
//! the path and storage seam exist before Phase 5 wires the producer.
//!
//! Writes are **atomic** (write to a sibling `*.tmp` in the same dir, fsync,
//! rename into place), matching [`crate::NotesStore`]'s durability story so a
//! crash mid-write never leaves a truncated `summary.md`.

use std::io::Write;
use std::path::Path;

use meeting_app_common::AppResult;

use crate::error::Error;

/// Atomically write `summary_md` to `summary.md` inside the existing meeting
/// folder.
///
/// Writes to a sibling `summary.md.tmp` then renames into place. Does **not**
/// create the meeting folder — the folder is owned by [`crate::MeetingWriter`]
/// and is expected to exist.
pub fn write_summary(meeting_dir: &Path, summary_md: &str) -> AppResult<()> {
    let path = meeting_dir.join("summary.md");
    let tmp_path = meeting_dir.join("summary.md.tmp");

    let write_result = (|| -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(summary_md.as_bytes())?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }

    if let Err(e) = std::fs::rename(&tmp_path, &path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }

    tracing::debug!(
        target: "persistence",
        folder = %meeting_dir.display(),
        "summary.md written"
    );

    Ok(())
}

/// Read `summary.md` from the meeting folder.
///
/// Returns `Ok(None)` when `summary.md` does not exist (a meeting that has not
/// been summarised yet).
pub fn read_summary(meeting_dir: &Path) -> AppResult<Option<String>> {
    let path = meeting_dir.join("summary.md");
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Io(e).into()),
    }
}
