use std::path::{Path, PathBuf};

use meeting_app_common::{AppResult, MeetingId};

use crate::error::Error;

/// Represents the on-disk layout `{root}/{uuid}/`.
///
/// Creating a `MeetingFolder` creates the directory on disk. The caller is
/// responsible for supplying the correct root (e.g. `{app-data}/meetings/`).
/// `persistence` is the only crate that writes under that root.
pub struct MeetingFolder {
    path: PathBuf,
    id: MeetingId,
}

impl MeetingFolder {
    /// Create the per-meeting directory on disk and return a handle.
    ///
    /// Fails with `AppError::InvalidInput` if the directory already exists
    /// (indicates a UUID collision — astronomically unlikely but caught
    /// explicitly to avoid clobbering a prior recording).
    pub fn create(root: &Path, meeting_id: MeetingId) -> AppResult<Self> {
        let path = root.join(meeting_id.0.to_string());

        if path.exists() {
            return Err(Error::FolderExists(path).into());
        }

        std::fs::create_dir_all(&path)
            .map_err(Error::Io)
            .map_err(meeting_app_common::AppError::from)?;

        tracing::info!(
            target: "persistence",
            folder = %path.display(),
            "created meeting folder"
        );

        Ok(Self {
            path,
            id: meeting_id,
        })
    }

    /// Absolute path to the meeting folder.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The `MeetingId` this folder was created for.
    pub fn id(&self) -> MeetingId {
        self.id
    }

    /// Path to `audio.opus` within the folder.
    pub(crate) fn audio_path(&self) -> PathBuf {
        self.path.join("audio.opus")
    }

    /// Path to `metadata.json` within the folder.
    pub(crate) fn metadata_path(&self) -> PathBuf {
        self.path.join("metadata.json")
    }
}
