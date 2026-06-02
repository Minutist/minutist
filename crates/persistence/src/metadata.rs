//! `metadata.json` write helpers.
//!
//! Two write entry points share one atomic implementation:
//!
//! - [`write_metadata`] is the **public** atomic writer keyed on the meeting
//!   **directory** (`{root}/{uuid}/`). It is the seam the orchestrator uses to
//!   update `metadata.json`'s `{ speaker_count, diarizer }` after the Phase-6
//!   diarization pass, while `persistence` stays the **sole** writer under
//!   `meetings/{uuid}/`. Like the notes/summary writers it is atomic
//!   (tmp + fsync + rename), so a crash mid-write never leaves a truncated
//!   `metadata.json`, and it leaves sibling files (`audio.opus`,
//!   `transcript.json`, `notes.json`) untouched.
//! - [`write_metadata_to_path`] is the crate-private form used by
//!   [`crate::MeetingWriter::finalise`], which already holds the resolved
//!   `metadata.json` path via [`crate::MeetingFolder::metadata_path`].

use std::io::Write;
use std::path::Path;

use meeting_app_common::{AppResult, MeetingMeta};

use crate::error::{Error, Result};

/// Atomically write `meta` to `metadata.json` inside the existing meeting
/// folder `meeting_dir` (`{root}/{uuid}/`).
///
/// This is the public seam the orchestrator calls to update `metadata.json`
/// (e.g. `{ speaker_count, diarizer }` after diarization) while keeping
/// `persistence` the sole writer under `meetings/{uuid}/`. The write is atomic
/// — it goes to a sibling `metadata.json.tmp`, is fsynced, then renamed into
/// place — so a crash mid-write never truncates the file, and a successful
/// write leaves no `.tmp` residue. It does **not** create the meeting folder
/// (the folder is owned by [`crate::MeetingWriter`] and is expected to exist),
/// and leaves sibling files (`audio.opus` / `transcript.json` / `notes.json`)
/// untouched.
pub fn write_metadata(meeting_dir: &Path, meta: &MeetingMeta) -> AppResult<()> {
    write_metadata_atomic(&meeting_dir.join("metadata.json"), meta)?;

    tracing::debug!(
        target: "persistence",
        folder = %meeting_dir.display(),
        "metadata.json written"
    );

    Ok(())
}

/// Atomically write `meta` to `metadata.json` at an explicit file `path`
/// (crate-internal).
///
/// Used by [`crate::MeetingWriter::finalise`], which already holds the resolved
/// path. Shares the tmp + fsync + rename implementation with [`write_metadata`].
pub(crate) fn write_metadata_to_path(path: &Path, meta: &MeetingMeta) -> Result<()> {
    write_metadata_atomic(path, meta)
}

/// The shared atomic implementation: serialise `meta`, write to a sibling
/// `*.tmp` in the same directory, fsync, then rename into place. On any error
/// the tmp file is removed so no residue is left behind.
fn write_metadata_atomic(path: &Path, meta: &MeetingMeta) -> Result<()> {
    let parent = path
        .parent()
        .ok_or(Error::InvalidState("metadata path has no parent"))?;
    let tmp_path = parent.join("metadata.json.tmp");

    let json = serde_json::to_vec_pretty(meta).map_err(Error::Serialise)?;

    let write_result = (|| -> std::result::Result<(), std::io::Error> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(&json)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e));
    }

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e));
    }

    Ok(())
}
