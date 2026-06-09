//! Rename / delete meeting operations that keep the on-disk folder and the
//! `index.db` row consistent.
//!
//! A rename updates `metadata.json`'s `title` (the authoritative copy) and then
//! the index row. A delete removes the meeting folder and then the index row.
//! In both cases the folder is the source of truth and the index is updated to
//! match, so a crash between the two steps leaves the index stale-but-rebuildable
//! ([`crate::MeetingIndex::rebuild_from_disk`] reconciles it).

use std::path::Path;

use meeting_app_common::{AppResult, MeetingId};

use crate::error::Error;
use crate::index::MeetingIndex;
use crate::reader;

/// Rename a meeting: update `metadata.json`'s `title` in place, then refresh
/// the index row.
///
/// `meetings_root` is `{app-data}/meetings/`; the folder is `{root}/{uuid}/`.
/// Returns `AppError::InvalidInput` (via `Error::MeetingNotFound`) if the
/// folder has no `metadata.json`.
pub async fn rename_meeting(
    meetings_root: &Path,
    index: &MeetingIndex,
    id: MeetingId,
    new_title: &str,
) -> AppResult<()> {
    let folder = meetings_root.join(id.0.to_string());
    let metadata_path = folder.join("metadata.json");

    if !metadata_path.exists() {
        return Err(Error::MeetingNotFound(id).into());
    }

    // Read, mutate title, write back atomically (tmp + rename), all on disk
    // first — the folder is authoritative.
    let mut meta = reader::read_metadata_inner(&folder)?;
    meta.title = new_title.to_string();

    crate::metadata::write_metadata(&folder, &meta)?;

    // Refresh the index row to match the renamed meeting.
    let entry = list_entry_from(&folder)?;
    index.upsert(&entry).await?;

    tracing::info!(
        target: "persistence",
        meeting_id = %id.0,
        new_title,
        "meeting renamed"
    );

    Ok(())
}

/// Delete a meeting: remove the folder recursively, then remove the index row.
///
/// An absent folder is treated as already-deleted (the index row is still
/// removed so the two converge). The index `delete` is a no-op for an absent
/// row.
pub async fn delete_meeting(
    meetings_root: &Path,
    index: &MeetingIndex,
    id: MeetingId,
) -> AppResult<()> {
    let folder = meetings_root.join(id.0.to_string());

    match std::fs::remove_dir_all(&folder) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(Error::Io(e).into()),
    }

    index.delete(id).await?;

    tracing::info!(
        target: "persistence",
        meeting_id = %id.0,
        "meeting deleted"
    );

    Ok(())
}

/// Build a `MeetingListEntry` from a meeting folder's metadata + excerpt.
///
/// Shares the same projection as `rebuild_from_disk` and reuses the excerpt
/// derivation (`crate::index::derive_excerpt`): a one-line `summary.md` blurb
/// once a summary exists, else the first transcript segment (live-test UX T6).
fn list_entry_from(folder: &Path) -> Result<meeting_app_common::MeetingListEntry, Error> {
    let meta = reader::read_metadata_inner(folder)?;
    let excerpt = crate::index::derive_excerpt(folder);

    Ok(meeting_app_common::MeetingListEntry {
        id: meta.uuid,
        title: meta.title,
        started_at: meta.started_at,
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
    })
}
