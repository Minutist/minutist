//! Rename / delete meeting operations that keep the on-disk folder and the
//! `index.db` row consistent.
//!
//! A rename updates `metadata.json`'s `title` (the authoritative copy) and then
//! the index row. A delete removes the meeting folder and then the index row.
//! In both cases the folder is the source of truth and the index is updated to
//! match, so a crash between the two steps leaves the index stale-but-rebuildable
//! ([`crate::MeetingIndex::rebuild_from_disk`] reconciles it).

use std::path::Path;

use minutist_common::{AppResult, CollectionId, MeetingId, ProcessingLifecycle};

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

    crate::write_metadata(&folder, &meta)?;

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

/// Set (or clear) the collection a meeting belongs to: update `metadata.json`'s
/// `collection_id` (authoritative) then refresh the index row (whose
/// `collection_id` column is the derived mirror used for filtered listing).
/// `None` clears membership (unfiled).
///
/// Only the meeting id is logged — the collection id is an opaque UUID and the
/// collection's name (user content) lives in `collections.json`, never here.
/// `AppError::InvalidInput` if the folder has no `metadata.json`.
pub async fn set_meeting_collection(
    meetings_root: &Path,
    index: &MeetingIndex,
    id: MeetingId,
    collection_id: Option<CollectionId>,
) -> AppResult<()> {
    let folder = meetings_root.join(id.0.to_string());
    if !folder.join("metadata.json").exists() {
        return Err(Error::MeetingNotFound(id).into());
    }

    let mut meta = reader::read_metadata_inner(&folder)?;
    meta.collection_id = collection_id;
    crate::write_metadata(&folder, &meta)?;

    // Refresh the index row so the derived `collection_id` mirror matches.
    let entry = list_entry_from(&folder)?;
    index.upsert(&entry).await?;

    tracing::info!(
        target: "persistence",
        meeting_id = %id.0,
        "meeting collection set"
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
    crate::write_metadata(&folder, &meta)?;

    tracing::info!(
        target: "persistence",
        meeting_id = %id.0,
        label,
        "speaker name set"
    );

    Ok(meta.speaker_names)
}

/// Apply a meeting's processing-lifecycle state to `metadata.json` (the
/// authoritative copy): read, set `processing`, write back.
///
/// The persistence half of the lifecycle consumer. The state is
/// host-authoritative and arrives over the sync lifecycle exchange; the
/// subscriber that bridges that exchange to this call lives in a crate that
/// depends on both `sync` and `persistence` (`ipc-bridge` / `headless`), since
/// `sync` has no dependency edge to `persistence`. Any racing-claim conflict is
/// resolved upstream in `sync` (lowest `HostRef` wins) before the winning state
/// reaches here, so this writes what it is given — overwriting the inbound
/// placeholder seeded by [`crate::MeetingFolder::ensure`], which starts at
/// `Local` and never guesses `PendingProcessing`
/// (`planning/DESIGN_processing-lifecycle.md` §7 Q4). It is also the local
/// producer for a re-process request (`Processed`/`Local` → `PendingProcessing`).
///
/// `processing` is not mirrored in `index.db` (like `speaker_names`), so unlike
/// [`rename_meeting`] there is no index row to reconcile.
///
/// `AppError::InvalidInput` (via `Error::MeetingNotFound`) if the folder has no
/// `metadata.json`: the sync receive path seeds the placeholder before the
/// lifecycle is applied, so a missing folder is an ordering error.
pub async fn apply_processing_lifecycle(
    meetings_root: &Path,
    id: MeetingId,
    processing: ProcessingLifecycle,
) -> AppResult<()> {
    let folder = meetings_root.join(id.0.to_string());
    if !folder.join("metadata.json").exists() {
        return Err(Error::MeetingNotFound(id).into());
    }

    let mut meta = reader::read_metadata_inner(&folder)?;
    // A stable discriminant for diagnostics. The `HostRef` inside a claim is a
    // device key (not user content) but is omitted to keep the log minimal.
    let state = match &processing {
        ProcessingLifecycle::Local => "local",
        ProcessingLifecycle::PendingProcessing => "pending_processing",
        ProcessingLifecycle::Claimed { .. } => "claimed",
        ProcessingLifecycle::Processed { .. } => "processed",
    };
    meta.processing = processing;
    crate::write_metadata(&folder, &meta)?;

    tracing::info!(
        target: "persistence",
        meeting_id = %id.0,
        state,
        "processing lifecycle applied"
    );

    Ok(())
}

/// Apply a peer-advertised processing-lifecycle state to a meeting we hold, or
/// skip it if that meeting is not present locally yet.
///
/// The lifecycle consumer (the `ipc-bridge` / `headless` subscriber to the sync
/// engine's lifecycle-event stream) calls this for each `(MeetingId,
/// ProcessingLifecycle)` a discovery exchange surfaces. Discovery advertises the
/// PEER's meetings, which we may not have synced yet — for those, applying would
/// be premature (the notes/media receive path seeds the folder, not the
/// lifecycle stream), so this returns `Ok(false)` rather than erroring. For a
/// meeting we do hold it delegates to [`apply_processing_lifecycle`] and returns
/// `Ok(true)`.
///
/// Keeping this in `persistence` keeps the meeting-folder layout owned here: the
/// subscriber never constructs `{meetings_root}/{uuid}` paths itself.
pub async fn apply_synced_lifecycle_if_present(
    meetings_root: &Path,
    id: MeetingId,
    processing: ProcessingLifecycle,
) -> AppResult<bool> {
    if !meetings_root
        .join(id.0.to_string())
        .join("metadata.json")
        .exists()
    {
        return Ok(false);
    }
    apply_processing_lifecycle(meetings_root, id, processing).await?;
    Ok(true)
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
        collection_id: meta.collection_id,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use minutist_common::{
        AudioFormat, HostRef, MeetingId, MeetingMeta, ProcessingClaim, ProcessingLifecycle,
    };
    use tempfile::TempDir;

    use crate::folder::MeetingFolder;
    use crate::write_metadata;
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
            notes_format: 0,
            processing: Default::default(),
            collection_id: None,
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

    /// `apply_processing_lifecycle` overwrites `metadata.json`'s `processing`
    /// with the given host-authoritative state (seeded `Local` → `Claimed` →
    /// `Processed`), persisting each.
    #[tokio::test]
    async fn test_apply_processing_lifecycle() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = write_meta_with_no_names(root);
        let folder = root.join(id.0.to_string());

        // The seeded placeholder is `Local`.
        assert_eq!(
            read_metadata(&folder).expect("read seed").processing,
            ProcessingLifecycle::Local
        );

        // A host claims it.
        let claim = ProcessingClaim {
            host: HostRef("endpoint-xyz".to_string()),
            claimed_at: "2026-06-27T10:00:00Z".to_string(),
            lease_expires_at: "2026-06-27T10:30:00Z".to_string(),
        };
        super::apply_processing_lifecycle(
            root,
            id,
            ProcessingLifecycle::Claimed {
                claim: claim.clone(),
            },
        )
        .await
        .expect("apply claimed");
        assert_eq!(
            read_metadata(&folder).expect("read claimed").processing,
            ProcessingLifecycle::Claimed { claim }
        );

        // Then it completes.
        let processed = ProcessingLifecycle::Processed {
            processed_by: HostRef("endpoint-xyz".to_string()),
            at: "2026-06-27T10:25:00Z".to_string(),
        };
        super::apply_processing_lifecycle(root, id, processed.clone())
            .await
            .expect("apply processed");
        assert_eq!(
            read_metadata(&folder).expect("read processed").processing,
            processed
        );
    }

    /// `apply_processing_lifecycle` returns `MeetingNotFound` for a meeting with
    /// no `metadata.json` (the receive path must seed the folder first).
    #[tokio::test]
    async fn test_apply_processing_lifecycle_missing_meeting() {
        let tempdir = TempDir::new().expect("tempdir");
        let err = super::apply_processing_lifecycle(
            tempdir.path(),
            MeetingId::new(),
            ProcessingLifecycle::PendingProcessing,
        )
        .await
        .expect_err("missing meeting must error");
        assert!(matches!(err, minutist_common::AppError::InvalidInput { .. }));
    }

    /// `apply_synced_lifecycle_if_present` skips a meeting we don't hold
    /// (`Ok(false)`, no error) and applies one we do (`Ok(true)`, metadata
    /// updated) — the consumer-side gating for peer-advertised lifecycle.
    #[tokio::test]
    async fn test_apply_synced_lifecycle_if_present() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();

        // A meeting not present locally is skipped, not an error.
        let absent = MeetingId::new();
        let applied = super::apply_synced_lifecycle_if_present(
            root,
            absent,
            ProcessingLifecycle::PendingProcessing,
        )
        .await
        .expect("skipping an unsynced meeting must not error");
        assert!(!applied, "an unsynced meeting must be skipped");

        // A meeting we hold is applied and its metadata.json updated.
        let id = write_meta_with_no_names(root);
        let folder = root.join(id.0.to_string());
        let processed = ProcessingLifecycle::Processed {
            processed_by: HostRef("endpoint-xyz".to_string()),
            at: "2026-06-27T10:25:00Z".to_string(),
        };
        let applied = super::apply_synced_lifecycle_if_present(root, id, processed.clone())
            .await
            .expect("applying a present meeting must succeed");
        assert!(applied, "a present meeting must be applied");
        assert_eq!(
            read_metadata(&folder).expect("read after apply").processing,
            processed
        );
    }
}
