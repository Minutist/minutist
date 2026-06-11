use std::path::{Path, PathBuf};

use minutist_common::{AppResult, MeetingId};

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
            .map_err(minutist_common::AppError::from)?;

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
    pub fn audio_path(&self) -> PathBuf {
        self.path.join("audio.opus")
    }

    /// Path to `metadata.json` within the folder.
    pub fn metadata_path(&self) -> PathBuf {
        self.path.join("metadata.json")
    }

    /// Path to `transcript.json` within the folder.
    pub fn transcript_path(&self) -> PathBuf {
        self.path.join("transcript.json")
    }

    /// Path to `summary.md` within the folder.
    ///
    /// `summary.md` holds the markdown summary produced by `summariser`
    /// (Phase 5). Phase 4 lands only the path helper and the
    /// [`crate::summary`] read/write I/O; the file is written by `summariser`
    /// via `persistence` in Phase 5.
    pub fn summary_path(&self) -> PathBuf {
        self.path.join("summary.md")
    }

    /// Path to `notes.json` within the folder.
    ///
    /// `notes.json` holds the opaque Tiptap document (see [`crate::NotesStore`]).
    /// It is written by `NotesStore`, never by `MeetingWriter`.
    pub fn notes_path(&self) -> PathBuf {
        self.path.join("notes.json")
    }

    /// Path to `notes.md` within the folder.
    ///
    /// `notes.md` holds the markdown rendering of the notes document, written
    /// alongside `notes.json` by [`crate::NotesStore`].
    pub fn notes_md_path(&self) -> PathBuf {
        self.path.join("notes.md")
    }

    /// Path to `translations.json` within the folder.
    ///
    /// `translations.json` holds per-language translations of transcript
    /// segments, indexed by segment position. It is a derived sidecar written
    /// by the translation commands in `ipc-bridge`; `write_transcript` clears
    /// it automatically when the transcript is replaced. See
    /// [`crate::translations`].
    pub fn translations_path(&self) -> std::path::PathBuf {
        self.path.join("translations.json")
    }

    /// Path to the `assets/` subdirectory within the folder.
    ///
    /// `assets/` holds pasted/dropped note image files (see
    /// [`crate::save_note_asset`]). The files are referenced from `notes.json`
    /// by bare filename (a portable, machine-independent reference) so the
    /// meeting folder — including `assets/` — can be copied to another machine
    /// and the notes still resolve. The directory is created lazily by
    /// [`crate::save_note_asset`], and removed wholesale by
    /// `meeting_ops::delete_meeting`'s `remove_dir_all` (no extra cleanup).
    pub fn assets_dir(&self) -> PathBuf {
        self.path.join("assets")
    }
}
