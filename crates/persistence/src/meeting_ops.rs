//! Rename / delete meeting operations that keep the on-disk folder and the
//! `index.db` row consistent.
//!
//! A rename updates `metadata.json`'s `title` (the authoritative copy) and then
//! the index row. A delete removes the meeting folder and then the index row.
//! In both cases the folder is the source of truth and the index is updated to
//! match, so a crash between the two steps leaves the index stale-but-rebuildable
//! ([`crate::MeetingIndex::rebuild_from_disk`] reconciles it).

use std::path::Path;

use minutist_common::{AppResult, MeetingId};

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

    // Do NOT log `new_title` — a meeting title is user content (issue #0014
    // privacy audit). The meeting id identifies the row for diagnostics; the
    // title must never reach a log line (and thus the crash-file / report
    // excerpt, which capture info+ log lines).
    tracing::info!(
        target: "persistence",
        meeting_id = %id.0,
        "meeting renamed"
    );

    Ok(())
}

/// Set (or clear) a speaker's display name in `metadata.json`.
///
/// Writes `meta.speaker_names[label]`: a non-empty `name` upserts the mapping,
/// an empty `name` removes it (revert to the bare diarizer label). The index
/// row carries no speaker names, so — unlike [`rename_meeting`] — there is
/// nothing to refresh there. Returns the updated map so the caller can reflect
/// it without re-reading.
///
/// `AppError::InvalidInput` if the folder has no `metadata.json`.
pub async fn set_speaker_name(
    meetings_root: &Path,
    id: MeetingId,
    label: &str,
    name: &str,
) -> AppResult<std::collections::BTreeMap<String, String>> {
    let folder = meetings_root.join(id.0.to_string());
    if !folder.join("metadata.json").exists() {
        return Err(Error::MeetingNotFound(id).into());
    }

    let mut meta = reader::read_metadata_inner(&folder)?;
    if name.is_empty() {
        meta.speaker_names.remove(label);
    } else {
        meta.speaker_names.insert(label.to_string(), name.to_string());
    }
    crate::metadata::write_metadata(&folder, &meta)?;

    tracing::info!(
        target: "persistence",
        meeting_id = %id.0,
        label,
        "speaker name set"
    );

    Ok(meta.speaker_names)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use minutist_common::{AudioFormat, MeetingId, MeetingMeta};
    use tempfile::TempDir;

    use crate::folder::MeetingFolder;
    use crate::metadata::write_metadata;
    use crate::reader::read_metadata;

    fn write_meta_with_no_names(root: &std::path::Path) -> MeetingId {
        let id = MeetingId::new();
        let folder = MeetingFolder::create(root, id).expect("create folder");
        let meta = MeetingMeta {
            uuid: id,
            title: "Test meeting".to_string(),
            started_at: "2026-06-14T09:00:00Z".to_string(),
            ended_at: None,
            duration_ms: 60_000,
            speaker_count: 2,
            audio_format: AudioFormat {
                codec: "opus".into(),
                sample_rate: 16_000,
                channels: 1,
                bitrate_kbps: Some(32),
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            speaker_names: std::collections::BTreeMap::new(),
            app_version: "0.0.0".into(),
        };
        write_metadata(folder.path(), &meta).expect("write metadata");
        id
    }

    /// `set_speaker_name` upserts a label→name mapping into `metadata.json`
    /// and the returned map reflects it; a subsequent `read_metadata` confirms
    /// the change was persisted. Clearing with an empty name removes the entry.
    #[tokio::test]
    async fn test_set_speaker_name() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = write_meta_with_no_names(root);
        let folder = root.join(id.0.to_string());

        // Upsert "A" → "Alice".
        let names = super::set_speaker_name(root, id, "A", "Alice")
            .await
            .expect("set_speaker_name");
        assert_eq!(names.get("A").map(String::as_str), Some("Alice"));

        // Confirm the write reached disk.
        let meta = read_metadata(&folder).expect("read_metadata after set");
        assert_eq!(meta.speaker_names.get("A").map(String::as_str), Some("Alice"));

        // A second upsert for a different label adds without clobbering the first.
        let names2 = super::set_speaker_name(root, id, "B", "Bob")
            .await
            .expect("set_speaker_name B");
        assert_eq!(names2.get("A").map(String::as_str), Some("Alice"));
        assert_eq!(names2.get("B").map(String::as_str), Some("Bob"));

        // Clearing "A" with an empty name removes the entry.
        let names3 = super::set_speaker_name(root, id, "A", "")
            .await
            .expect("clear A");
        assert!(!names3.contains_key("A"), "cleared label must be absent");
        assert_eq!(names3.get("B").map(String::as_str), Some("Bob"));

        // Disk reflects the cleared state.
        let meta2 = read_metadata(&folder).expect("read_metadata after clear");
        assert!(!meta2.speaker_names.contains_key("A"));
        assert_eq!(meta2.speaker_names.get("B").map(String::as_str), Some("Bob"));
    }

    /// `set_speaker_name` returns `MeetingNotFound` for a nonexistent meeting.
    #[tokio::test]
    async fn test_set_speaker_name_missing_meeting() {
        let tempdir = TempDir::new().expect("tempdir");
        let missing = MeetingId::new();
        let err = super::set_speaker_name(tempdir.path(), missing, "A", "Alice")
            .await
            .expect_err("missing meeting must error");
        assert!(matches!(err, minutist_common::AppError::InvalidInput { .. }));
    }
}

/// Build a `MeetingListEntry` from a meeting folder's metadata + excerpt.
///
/// Shares the same projection as `rebuild_from_disk` and reuses the excerpt
/// derivation (`crate::index::derive_excerpt`): a one-line `summary.md` blurb
/// once a summary exists, else the first transcript segment (live-test UX T6).
fn list_entry_from(folder: &Path) -> Result<minutist_common::MeetingListEntry, Error> {
    let meta = reader::read_metadata_inner(folder)?;
    let excerpt = crate::index::derive_excerpt(folder);

    Ok(minutist_common::MeetingListEntry {
        id: meta.uuid,
        title: meta.title,
        started_at: meta.started_at,
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
    })
}
