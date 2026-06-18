//! Collection ("folder") definitions store + operations.
//!
//! A *collection* is a user-facing "folder" that groups meetings (UI label:
//! "Folders"). It is **distinct** from [`crate::MeetingFolder`], which is a
//! single meeting's on-disk directory.
//!
//! # Where the data lives
//!
//! - **Definitions** (id, name, order) are the authoritative
//!   [`minutist_common::Collection`] list in `{app-data}/collections.json`,
//!   owned here. It sits at the app-data root beside `index.db` — NOT under
//!   `index.db`, because the index is a derived cache that `rebuild_from_disk`
//!   wipes; the folder list must survive a rebuild.
//! - **Membership** (which collection a meeting is in) is authoritative in each
//!   meeting's `metadata.json` (`MeetingMeta::collection_id`, written via
//!   [`crate::meeting_ops::set_meeting_collection`]); the `index.db`
//!   `collection_id` column is a derived mirror for filtered listing.
//!
//! Writes are atomic (tmp + fsync + rename), matching the other persistence
//! writers, so a crash mid-write never leaves a truncated `collections.json`.

use std::io::Write;
use std::path::{Path, PathBuf};

use minutist_common::{AppResult, Collection, CollectionId};

use crate::error::{Error, Result};
use crate::index::MeetingIndex;

/// The conventional `collections.json` path under an app-data root (the dir that
/// also holds `index.db` and `meetings/`). Mirrors `index::index_db_path`.
pub fn collections_path(app_data_root: &Path) -> PathBuf {
    app_data_root.join("collections.json")
}

/// Stateless reader/writer for the authoritative collection definitions
/// (`{app-data}/collections.json`).
pub struct CollectionStore;

impl CollectionStore {
    /// Load all collections, ordered by `position` ascending. An absent file is
    /// an empty list (no collections created yet) — not an error.
    pub fn load(app_data_root: &Path) -> AppResult<Vec<Collection>> {
        Ok(Self::load_inner(app_data_root)?)
    }

    fn load_inner(app_data_root: &Path) -> Result<Vec<Collection>> {
        let path = collections_path(app_data_root);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::Io(e)),
        };
        let mut collections: Vec<Collection> = serde_json::from_slice(&bytes)?;
        collections.sort_by_key(|c| c.position);
        Ok(collections)
    }

    /// Create a new collection named `name` (trimmed; must be non-empty),
    /// appended after the current highest `position`. Returns the new
    /// [`Collection`].
    pub fn create(app_data_root: &Path, name: &str) -> AppResult<Collection> {
        Ok(Self::create_inner(app_data_root, name)?)
    }

    fn create_inner(app_data_root: &Path, name: &str) -> Result<Collection> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::InvalidState("collection name must not be empty"));
        }
        let mut collections = Self::load_inner(app_data_root)?;
        let position = collections.iter().map(|c| c.position).max().map_or(0, |m| m + 1);
        let created = Collection {
            id: CollectionId::new(),
            name: name.to_string(),
            position,
        };
        collections.push(created.clone());
        write_collections_atomic(&collections_path(app_data_root), &collections)?;
        tracing::info!(target: "persistence", collection_id = %created.id.0, "collection created");
        Ok(created)
    }

    /// Rename the collection `id` to `name` (trimmed; must be non-empty).
    /// `CollectionNotFound` if no such collection exists.
    pub fn rename(app_data_root: &Path, id: CollectionId, name: &str) -> AppResult<()> {
        Ok(Self::rename_inner(app_data_root, id, name)?)
    }

    fn rename_inner(app_data_root: &Path, id: CollectionId, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::InvalidState("collection name must not be empty"));
        }
        let mut collections = Self::load_inner(app_data_root)?;
        let target = collections
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or(Error::CollectionNotFound(id))?;
        target.name = name.to_string();
        write_collections_atomic(&collections_path(app_data_root), &collections)?;
        tracing::info!(target: "persistence", collection_id = %id.0, "collection renamed");
        Ok(())
    }

    /// Remove the definition of collection `id` from `collections.json`.
    /// Idempotent: removing an absent collection is a no-op. Does NOT touch
    /// meeting membership — see [`delete_collection`] for the full delete.
    fn remove_def(app_data_root: &Path, id: CollectionId) -> Result<()> {
        let mut collections = Self::load_inner(app_data_root)?;
        let before = collections.len();
        collections.retain(|c| c.id != id);
        if collections.len() != before {
            write_collections_atomic(&collections_path(app_data_root), &collections)?;
        }
        Ok(())
    }
}

/// Delete a collection: first clear membership of every meeting filed under it
/// (so no `metadata.json` keeps a dangling `collection_id`), then remove the
/// definition from `collections.json`.
///
/// The affected meetings are found via the index's derived `collection_id`
/// column; each is cleared through [`crate::meeting_ops::set_meeting_collection`]
/// (which rewrites both the authoritative `metadata.json` and the index row).
/// `meetings_root` is `{app-data}/meetings/`.
pub async fn delete_collection(
    app_data_root: &Path,
    meetings_root: &Path,
    index: &MeetingIndex,
    id: CollectionId,
) -> AppResult<()> {
    let affected = index.ids_in_collection(&id).await?;
    for meeting_id in affected {
        crate::meeting_ops::set_meeting_collection(meetings_root, index, meeting_id, None).await?;
    }
    CollectionStore::remove_def(app_data_root, id)?;
    tracing::info!(target: "persistence", collection_id = %id.0, "collection deleted");
    Ok(())
}

/// Atomically write `collections` to `path`: serialise, write to a sibling
/// `*.tmp` in the same directory, fsync, then rename into place. On any error
/// the tmp file is removed so no residue is left behind. Mirrors
/// `metadata::write_metadata_atomic`.
fn write_collections_atomic(path: &Path, collections: &[Collection]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or(Error::InvalidState("collections path has no parent"))?;
    std::fs::create_dir_all(parent).map_err(Error::Io)?;
    let tmp_path = parent.join("collections.json.tmp");

    let json = serde_json::to_vec_pretty(collections).map_err(Error::Serialise)?;

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folder::MeetingFolder;
    use crate::metadata::write_metadata;
    use minutist_common::{AudioFormat, MeetingId, MeetingMeta};
    use tempfile::TempDir;

    fn meta_in(root: &Path, collection_id: Option<CollectionId>) -> MeetingId {
        let id = MeetingId::new();
        let folder = MeetingFolder::create(root, id).expect("create folder");
        let meta = MeetingMeta {
            uuid: id,
            title: "Test".into(),
            started_at: "2026-06-18T09:00:00Z".into(),
            ended_at: None,
            duration_ms: 60_000,
            speaker_count: 1,
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
            collection_id,
            app_version: "0.0.0".into(),
        };
        write_metadata(folder.path(), &meta).expect("write metadata");
        id
    }

    #[test]
    fn load_absent_is_empty() {
        let tmp = TempDir::new().unwrap();
        assert!(CollectionStore::load(tmp.path()).unwrap().is_empty());
    }

    #[test]
    fn create_assigns_increasing_positions_and_persists() {
        let tmp = TempDir::new().unwrap();
        let a = CollectionStore::create(tmp.path(), "  Projects  ").unwrap();
        let b = CollectionStore::create(tmp.path(), "Personal").unwrap();
        assert_eq!(a.name, "Projects", "name is trimmed");
        assert_eq!(a.position, 0);
        assert_eq!(b.position, 1);

        let loaded = CollectionStore::load(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, a.id, "ordered by position ascending");
        assert_eq!(loaded[1].id, b.id);
    }

    #[test]
    fn create_rejects_empty_name() {
        let tmp = TempDir::new().unwrap();
        let err = CollectionStore::create(tmp.path(), "   ").unwrap_err();
        assert!(matches!(err, minutist_common::AppError::InvalidInput { .. }));
    }

    #[test]
    fn rename_changes_name_and_errors_on_missing() {
        let tmp = TempDir::new().unwrap();
        let c = CollectionStore::create(tmp.path(), "Old").unwrap();
        CollectionStore::rename(tmp.path(), c.id, "New").unwrap();
        assert_eq!(CollectionStore::load(tmp.path()).unwrap()[0].name, "New");

        let err = CollectionStore::rename(tmp.path(), CollectionId::new(), "X").unwrap_err();
        assert!(matches!(err, minutist_common::AppError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn delete_clears_membership_then_removes_definition() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let meetings_root = root.join("meetings");
        std::fs::create_dir_all(&meetings_root).unwrap();
        let index = MeetingIndex::open(":memory:").await.expect("index");

        let collection = CollectionStore::create(root, "Acme").unwrap();
        let meeting_id = meta_in(&meetings_root, Some(collection.id));
        // Index the meeting (membership mirror) via the meeting-ops seam.
        crate::meeting_ops::set_meeting_collection(
            &meetings_root,
            &index,
            meeting_id,
            Some(collection.id),
        )
        .await
        .unwrap();
        assert_eq!(
            index.ids_in_collection(&collection.id).await.unwrap(),
            vec![meeting_id],
        );

        delete_collection(root, &meetings_root, &index, collection.id)
            .await
            .unwrap();

        // The definition is gone.
        assert!(CollectionStore::load(root).unwrap().is_empty());
        // The meeting's authoritative membership is cleared (unfiled).
        let meta = crate::reader::read_metadata(&meetings_root.join(meeting_id.0.to_string()))
            .unwrap();
        assert_eq!(meta.collection_id, None);
        // And the index mirror no longer lists it under the (deleted) collection.
        assert!(index.ids_in_collection(&collection.id).await.unwrap().is_empty());
    }
}
