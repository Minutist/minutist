use std::path::{Path, PathBuf};

use minutist_common::{AppResult, AudioFormat, MeetingId, MeetingMeta};

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

    /// Ensure a meeting folder exists at `{root}/{uuid}/`, seeding a minimal
    /// `metadata.json` when it is absent, and return its path. Idempotent: an
    /// existing folder (and an existing `metadata.json`) is left untouched.
    ///
    /// This is the seam a sync receiver calls before merging an inbound document
    /// for a meeting this device has never seen: a brand-new meeting arriving from
    /// a paired device must not fail for want of its folder. `persistence` stays
    /// the sole owner of folder + projection writes, so the receiver routes the
    /// folder/metadata creation through here rather than touching the layout
    /// itself. The seeded metadata is a placeholder — `uuid`, the current time as
    /// `started_at`, an empty title, and the standard 16 kHz mono Opus
    /// `audio_format` — that the authoritative `metadata.json` overwrites when it
    /// later syncs across (a metadata/media payload, a separate slice). It exists
    /// only so the folder is a valid meeting on disk the moment notes land in it.
    ///
    /// Unlike [`Self::create`], this does NOT fail when the folder already exists;
    /// the converged steady state of a paired device is that the folder is already
    /// there, and re-running a sync must be a no-op.
    pub fn ensure(root: &Path, meeting_id: MeetingId) -> AppResult<PathBuf> {
        let path = root.join(meeting_id.0.to_string());

        let freshly_created = !path.exists();
        std::fs::create_dir_all(&path)
            .map_err(Error::Io)
            .map_err(minutist_common::AppError::from)?;

        let metadata_path = path.join("metadata.json");
        if !metadata_path.exists() {
            let placeholder = MeetingMeta {
                uuid: meeting_id,
                title: String::new(),
                started_at: now_iso8601(),
                ended_at: None,
                duration_ms: 0,
                speaker_count: 0,
                audio_format: AudioFormat {
                    codec: "opus".to_string(),
                    sample_rate: 16_000,
                    channels: 1,
                    bitrate_kbps: Some(32),
                },
                asr_model: None,
                llm_model: None,
                diarizer: None,
                speaker_names: std::collections::BTreeMap::new(),
                notes_format: 0,
                collection_id: None,
                app_version: String::new(),
            };
            crate::write_metadata(&path, &placeholder)?;
        }

        if freshly_created {
            tracing::info!(
                target: "persistence",
                folder = %path.display(),
                "ensured inbound meeting folder (seeded placeholder metadata)"
            );
        }

        Ok(path)
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

/// The current UTC time as an ISO 8601 / RFC 3339 string, matching the
/// `started_at` format the orchestrator writes (`chrono::Utc::now().to_rfc3339()`).
fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read_metadata;

    #[test]
    fn ensure_creates_folder_and_seeds_metadata() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = MeetingId::new();

        let path = MeetingFolder::ensure(dir.path(), id).expect("ensure");
        assert!(path.is_dir(), "the meeting folder must exist");
        assert!(
            path.join("metadata.json").is_file(),
            "a placeholder metadata.json must be seeded"
        );

        let meta = read_metadata(&path).expect("read seeded metadata");
        assert_eq!(meta.uuid, id);
        assert!(
            meta.title.is_empty(),
            "the placeholder carries no title until the authoritative metadata syncs"
        );
    }

    #[test]
    fn ensure_is_idempotent_and_preserves_existing_metadata() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = MeetingId::new();

        // First call seeds the placeholder; overwrite it with a real title to
        // stand in for the authoritative metadata having synced across.
        let path = MeetingFolder::ensure(dir.path(), id).expect("first ensure");
        let mut meta = read_metadata(&path).expect("read");
        meta.title = "Real meeting".to_string();
        crate::write_metadata(&path, &meta).expect("overwrite metadata");

        // A second ensure (a re-sync of the same meeting) must not clobber it.
        let path2 = MeetingFolder::ensure(dir.path(), id).expect("second ensure");
        assert_eq!(path, path2);
        let reloaded = read_metadata(&path2).expect("re-read");
        assert_eq!(
            reloaded.title, "Real meeting",
            "ensure must leave an existing metadata.json untouched"
        );
    }

    #[test]
    fn ensure_succeeds_where_create_would_reject() {
        // `create` rejects a pre-existing folder; `ensure` is the converged-steady
        // -state path and must succeed on it (a re-sync of an already-present
        // meeting).
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = MeetingId::new();
        MeetingFolder::create(dir.path(), id).expect("create");
        assert!(
            MeetingFolder::create(dir.path(), id).is_err(),
            "create must reject an existing folder"
        );
        MeetingFolder::ensure(dir.path(), id).expect("ensure on an existing folder");
    }
}
