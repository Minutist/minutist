//! The libsql `index.db` meeting index — a derived cache over the per-meeting
//! folders.
//!
//! `index.db` holds one row per meeting mirroring
//! [`minutist_common::MeetingListEntry`], so the meeting-list view (FR-33)
//! and search read straight from the index without loading any meeting's full
//! transcript. The index is **rebuildable** from disk ([`MeetingIndex::rebuild_from_disk`])
//! — it never holds authoritative state, only a fast-query projection of the
//! `metadata.json` + transcript files.
//!
//! # Async, no `block_on`
//!
//! libsql is async (tokio). Every operation here is an `async fn`; the crate
//! never calls `block_on`, honouring the threading model in
//! `architecture/cross-cutting.md`. Callers drive these futures on the tokio
//! runtime. The synchronous folder readers ([`crate::reader`]) are called from
//! `rebuild_from_disk` via `tokio::task::spawn_blocking` so blocking `std::fs`
//! reads never occupy an async worker thread.

use std::path::{Path, PathBuf};

use libsql::{Builder, Connection, Database};
use minutist_common::{
    AppResult, AudioFormat, CollectionId, MeetingId, MeetingListEntry, MeetingMeta, Segment,
};

use crate::error::Error;
use crate::{migrations, reader};

/// A handle to the libsql `index.db` database plus a connection.
///
/// Open via [`MeetingIndex::open`] with an explicit `index.db` path (injected
/// by `app-main`; the crate never resolves `{app-data}` itself). `open` runs
/// the forward-only migration runner so the schema is current before any query.
pub struct MeetingIndex {
    // The `Database` owns the underlying connection pool; keep it alive for the
    // lifetime of the index even though queries go through `conn`.
    #[allow(dead_code)]
    db: Database,
    conn: Connection,
    /// Serialises the `BEGIN IMMEDIATE`..`COMMIT`/`ROLLBACK` span of
    /// [`Self::rebuild_transaction`]. `conn` is a bare `libsql::Connection`
    /// shared app-wide via `Arc<MeetingIndex>` with no locking of its own, so
    /// two concurrent calls into a multi-statement transaction could otherwise
    /// interleave their statements on the same connection (`cannot start a
    /// transaction within a transaction`, or a write silently absorbed into —
    /// and lost with — another call's rolled-back transaction). Held for the
    /// full BEGIN..COMMIT span.
    ///
    /// Single-statement methods on `conn` (`upsert`, `delete`) are atomic in
    /// SQLite's autocommit mode on their own, but SQLite transactions are
    /// connection-scoped: a plain statement issued while
    /// `rebuild_transaction` holds an open `BEGIN IMMEDIATE` on this same
    /// connection can be silently absorbed into that transaction — committing
    /// or rolling back together with it — and a concurrent read can
    /// dirty-read its uncommitted rows. `index.db` is a rebuildable cache
    /// (`rebuild_from_disk` recovers it from the meeting folders), so this
    /// window is accepted rather than closed with a lock; every
    /// multi-statement mutation still must hold `tx_lock` for its whole span.
    tx_lock: tokio::sync::Mutex<()>,
}

impl MeetingIndex {
    /// Open (or create) the local `index.db` at `db_path` and migrate it to the
    /// current schema.
    ///
    /// Pass `":memory:"` for an in-memory database (used by tests). A relative
    /// or absolute filesystem path opens-or-creates a file-backed DB.
    pub async fn open(db_path: impl AsRef<Path>) -> AppResult<Self> {
        Ok(Self::open_inner(db_path).await?)
    }

    async fn open_inner(db_path: impl AsRef<Path>) -> Result<Self, Error> {
        let db = Builder::new_local(db_path.as_ref()).build().await?;
        let conn = db.connect()?;
        migrations::run(&conn).await?;
        Ok(Self {
            db,
            conn,
            tx_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// All meetings, most-recent first (ordered by `started_at` descending).
    pub async fn list_meetings(&self) -> AppResult<Vec<MeetingListEntry>> {
        let rows = self
            .conn
            .query(
                "SELECT id, title, started_at, duration_ms, speaker_count, excerpt, collection_id, recording_started, deleted_at
                 FROM meetings
                 ORDER BY started_at DESC",
                (),
            )
            .await
            .map_err(Error::from)?;
        Ok(rows_to_entries(rows).await?)
    }

    /// Meetings whose title or excerpt contains `query` (case-insensitive
    /// substring), most-recent first.
    ///
    /// A trivial `LIKE` scan — adequate for the v1 index size (hundreds of
    /// meetings). Full-text search (FTS5) is a later refinement.
    pub async fn search(&self, query: &str) -> AppResult<Vec<MeetingListEntry>> {
        let pattern = format!("%{}%", escape_like(query));
        let rows = self
            .conn
            .query(
                "SELECT id, title, started_at, duration_ms, speaker_count, excerpt, collection_id, recording_started, deleted_at
                 FROM meetings
                 WHERE title LIKE ?1 ESCAPE '\\' OR excerpt LIKE ?1 ESCAPE '\\'
                 ORDER BY started_at DESC",
                libsql::params![pattern],
            )
            .await
            .map_err(Error::from)?;
        Ok(rows_to_entries(rows).await?)
    }

    /// Insert or replace the index row for `entry`.
    ///
    /// Keyed on `entry.id`; re-upserting an existing meeting overwrites its row
    /// (used after a rename, or when `rebuild_from_disk` repopulates).
    pub async fn upsert(&self, entry: &MeetingListEntry) -> AppResult<()> {
        Ok(self.upsert_inner(entry).await?)
    }

    async fn upsert_inner(&self, entry: &MeetingListEntry) -> Result<(), Error> {
        self.conn
            .execute(
                "INSERT INTO meetings (id, title, started_at, duration_ms, speaker_count, excerpt, collection_id, recording_started, deleted_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(id) DO UPDATE SET
                     title = excluded.title,
                     started_at = excluded.started_at,
                     duration_ms = excluded.duration_ms,
                     speaker_count = excluded.speaker_count,
                     excerpt = excluded.excerpt,
                     collection_id = excluded.collection_id,
                     recording_started = excluded.recording_started,
                     deleted_at = excluded.deleted_at",
                libsql::params![
                    entry.id.0.to_string(),
                    entry.title.clone(),
                    entry.started_at.clone(),
                    entry.duration_ms as i64,
                    entry.speaker_count as i64,
                    entry.excerpt.clone(),
                    entry.collection_id.map(|c| c.0.to_string()),
                    entry.recording_started as i64,
                    entry.deleted_at.clone(),
                ],
            )
            .await?;
        Ok(())
    }

    /// Fetch the current index row for `id`, if any. Used to patch a single
    /// field (e.g. `deleted_at`) onto the existing row without re-deriving
    /// the whole entry from disk (which would re-read `metadata.json` and
    /// re-run `derive_excerpt`'s transcript/summary scan for no reason).
    pub async fn get(&self, id: MeetingId) -> AppResult<Option<MeetingListEntry>> {
        Ok(self.get_inner(id).await?)
    }

    async fn get_inner(&self, id: MeetingId) -> Result<Option<MeetingListEntry>, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, title, started_at, duration_ms, speaker_count, excerpt, collection_id, recording_started, deleted_at
                 FROM meetings WHERE id = ?1",
                libsql::params![id.0.to_string()],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(Some(row_to_entry(&row)?)),
            None => Ok(None),
        }
    }

    /// Delete the index row for `id`. A no-op if the row is absent.
    pub async fn delete(&self, id: MeetingId) -> AppResult<()> {
        Ok(self.delete_inner(id).await?)
    }

    async fn delete_inner(&self, id: MeetingId) -> Result<(), Error> {
        self.conn
            .execute(
                "DELETE FROM meetings WHERE id = ?1",
                libsql::params![id.0.to_string()],
            )
            .await?;
        Ok(())
    }

    /// The ids of all meetings currently filed under `collection_id`.
    ///
    /// Used when a collection is deleted to clear each affected meeting's
    /// membership (the per-meeting `metadata.json` is authoritative, so the
    /// clear must rewrite each one — see `collections::delete_collection`).
    pub async fn ids_in_collection(
        &self,
        collection_id: &CollectionId,
    ) -> AppResult<Vec<MeetingId>> {
        Ok(self.ids_in_collection_inner(collection_id).await?)
    }

    async fn ids_in_collection_inner(
        &self,
        collection_id: &CollectionId,
    ) -> Result<Vec<MeetingId>, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT id FROM meetings WHERE collection_id = ?1",
                libsql::params![collection_id.0.to_string()],
            )
            .await?;
        let mut ids = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            ids.push(parse_meeting_id(&id)?);
        }
        Ok(ids)
    }

    /// Rebuild the index from the per-meeting folders under `meetings_root`.
    ///
    /// Clears the `meetings` table, then scans every immediate subdirectory of
    /// `meetings_root` that contains a `metadata.json`, reads its metadata and
    /// transcript (on a blocking thread), derives a [`MeetingListEntry`], and
    /// upserts it. Folders without a readable `metadata.json` are skipped with a
    /// warning — the index is a cache, so one bad folder must not abort the
    /// whole rebuild.
    ///
    /// # Atomicity (TIMELINE-DRIFT #7)
    ///
    /// The DELETE + repopulate runs inside a single `BEGIN`/`COMMIT`
    /// transaction, so a concurrent `list_meetings`/`search` on the shared
    /// connection observes either the old table or the fully-rebuilt one, never
    /// a half-cleared/half-populated intermediate. On any error the transaction
    /// is rolled back, leaving the previous index contents intact.
    ///
    /// Returns the number of meetings indexed.
    pub async fn rebuild_from_disk(&self, meetings_root: &Path) -> AppResult<usize> {
        Ok(self.rebuild_from_disk_inner(meetings_root).await?)
    }

    async fn rebuild_from_disk_inner(&self, meetings_root: &Path) -> Result<usize, Error> {
        // Enumerate eligible folders BEFORE opening the transaction so the
        // (blocking) directory read does not hold the write lock open longer
        // than necessary. A missing root yields an empty index without ever
        // touching the table.
        let dirs = match std::fs::read_dir(meetings_root) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // No meetings root yet — an empty index is the correct result.
                // Still run the clear in a transaction so a stale table is reset
                // atomically.
                self.rebuild_transaction(Vec::new()).await?;
                return Ok(0);
            }
            Err(e) => return Err(Error::Io(e)),
        };

        let mut entries: Vec<MeetingListEntry> = Vec::new();
        for entry in dirs {
            let entry = entry.map_err(Error::Io)?;
            // The same "is this folder a meeting" predicate `list_meeting_ids`
            // uses — a folder that isn't a UUID-named directory is never a
            // meeting to one enumerator and clutter to another.
            if notes_crdt::folder::parse_meeting_dir(&entry).is_none() {
                continue;
            }
            let path = entry.path();
            if !path.join("metadata.json").exists() {
                continue;
            }

            match entry_from_folder(&path).await {
                Ok(list_entry) => entries.push(list_entry),
                Err(e) => {
                    tracing::warn!(
                        target: "persistence",
                        folder = %path.display(),
                        error = %e,
                        "skipping meeting folder during index rebuild"
                    );
                }
            }
        }

        let indexed = entries.len();
        self.rebuild_transaction(entries).await?;

        tracing::info!(
            target: "persistence",
            root = %meetings_root.display(),
            indexed,
            "index.db rebuilt from disk"
        );

        Ok(indexed)
    }

    /// Atomically replace the `meetings` table contents with `entries`.
    ///
    /// Wraps `DELETE` + the per-entry `INSERT … ON CONFLICT` upserts in a single
    /// transaction so concurrent readers on the shared connection never see a
    /// half-rebuilt table. Rolls back (best-effort) on any error so a failed
    /// rebuild leaves the prior contents intact.
    ///
    /// Holds `tx_lock` for the whole span so a second concurrent call cannot
    /// open its own `BEGIN` on the same connection mid-transaction.
    async fn rebuild_transaction(&self, entries: Vec<MeetingListEntry>) -> Result<(), Error> {
        let _guard = self.tx_lock.lock().await;
        self.conn.execute("BEGIN IMMEDIATE", ()).await?;

        let result = async {
            self.conn.execute("DELETE FROM meetings", ()).await?;
            for entry in &entries {
                self.upsert_inner(entry).await?;
            }
            Ok::<(), Error>(())
        }
        .await;

        match result {
            Ok(()) => {
                if let Err(e) = self.conn.execute("COMMIT", ()).await {
                    // A failed COMMIT can leave the transaction open; roll it
                    // back (best-effort) so the next `tx_lock` holder's BEGIN
                    // IMMEDIATE does not error with a nested transaction.
                    if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                        tracing::warn!(
                            target: "persistence",
                            error = %rb,
                            "index rebuild rollback failed after a COMMIT error"
                        );
                    }
                    return Err(e.into());
                }
                Ok(())
            }
            Err(e) => {
                // Best-effort rollback; the original error is what callers see.
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(
                        target: "persistence",
                        error = %rb,
                        "index rebuild rollback failed after a rebuild error"
                    );
                }
                Err(e)
            }
        }
    }

    /// Self-heal: index any on-disk meeting folder missing from the index,
    /// WITHOUT touching existing rows. Cheap in the common case — one directory
    /// read plus a set-diff; a folder already indexed with a real (non-zero)
    /// duration incurs no file read at all, only a folder already indexed with
    /// a degenerate `duration_ms` (issue 0064) or not indexed yet incurs a
    /// metadata/transcript read + `upsert`.
    ///
    /// Guards in-session visibility against any missed `upsert` (e.g. the
    /// process being killed between finalise and the stop-time index write):
    /// `list_meetings` runs this so a meeting present on disk can never stay
    /// hidden until the next startup `rebuild_from_disk`. Unlike
    /// [`Self::rebuild_from_disk`] it never deletes — a folder removed off-app
    /// is reconciled by the next rebuild, not here.
    ///
    /// A folder with no `metadata.json` but real recording data (an audio
    /// file and/or `transcript.json`) is recovered rather than skipped: a minimal
    /// metadata is synthesised (see [`synthesize_metadata`]) and written before
    /// the folder is indexed via the normal path. This is what makes a
    /// crash/kill mid-recording recoverable even for meetings started before
    /// `MeetingWriter::open` began writing its own initial metadata — `index.db`
    /// stays a derived cache, but `metadata.json` is now guaranteed to exist for
    /// every folder that holds recording data. A folder with neither file is
    /// unrelated clutter and is left alone. Returns the number of orphan
    /// meetings newly indexed (including recovered ones) — a backfilled but
    /// already-indexed meeting does not count (it is a refresh, not an orphan).
    pub async fn reconcile_orphans(&self, meetings_root: &Path) -> AppResult<usize> {
        Ok(self.reconcile_orphans_inner(meetings_root).await?)
    }

    async fn reconcile_orphans_inner(&self, meetings_root: &Path) -> Result<usize, Error> {
        // Indexed ids and their stored duration — the cheap half of the diff,
        // and enough to skip a disk read entirely for any already-indexed
        // meeting whose duration is not degenerate (the index is derived from
        // metadata.json, so a non-zero indexed duration implies a
        // non-degenerate file).
        let known = self.indexed_durations().await?;

        let dirs = match std::fs::read_dir(meetings_root) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(Error::Io(e)),
        };

        let mut indexed = 0usize;
        for entry in dirs {
            let entry = entry.map_err(Error::Io)?;
            // The same "is this folder a meeting" predicate `list_meeting_ids`
            // uses — a folder that isn't a UUID-named directory is never a
            // meeting to one enumerator and clutter to another.
            let Some(meeting_id) = notes_crdt::folder::parse_meeting_dir(&entry) else {
                continue;
            };
            let path = entry.path();
            let id = meeting_id.0.to_string();
            let indexed_duration = known.get(&id).copied();
            if indexed_duration.is_some_and(|d| d != 0) {
                continue;
            }
            let already_known = indexed_duration.is_some();

            if !path.join("metadata.json").exists() {
                if already_known {
                    // Indexed, but metadata.json has since vanished — not
                    // this reconciler's concern; leave the stale row alone.
                    continue;
                }
                let has_recording_data = minutist_common::resolve_audio_path(&path).is_some()
                    || path.join("transcript.json").exists();
                if !has_recording_data {
                    // No metadata and no recording data — unrelated clutter,
                    // not a meeting folder.
                    continue;
                }

                if let Err(e) = synthesize_metadata(&path, meeting_id) {
                    tracing::warn!(
                        target: "persistence",
                        folder = %path.display(),
                        error = %e,
                        "failed to recover incomplete recording during orphan reconcile"
                    );
                    continue;
                }

                tracing::info!(
                    target: "persistence",
                    meeting_id = %id,
                    "recovered incomplete recording (self-heal)"
                );
            } else {
                // metadata.json exists but the indexed (or on-disk, if not yet
                // indexed) duration is degenerate — backfill it from the
                // transcript now that one may exist since it was last checked
                // (issue 0064).
                let backfilled = match backfill_degenerate_duration(meetings_root, meeting_id) {
                    Ok(applied) => applied,
                    Err(e) => {
                        tracing::warn!(
                            target: "persistence",
                            folder = %path.display(),
                            error = %e,
                            "duration backfill failed during orphan reconcile"
                        );
                        false
                    }
                };
                if backfilled {
                    tracing::info!(
                        target: "persistence",
                        meeting_id = %id,
                        "backfilled duration/speaker_count from transcript (self-heal)"
                    );
                } else if already_known {
                    continue;
                }
            }

            match entry_from_folder(&path).await {
                Ok(list_entry) => {
                    self.upsert_inner(&list_entry).await?;
                    // A backfilled-but-already-known row is a refresh, not a
                    // newly-indexed orphan — the return value documents "newly
                    // indexed", and the caller's own already-known count must
                    // not be double-counted.
                    if !already_known {
                        indexed += 1;
                        tracing::info!(
                            target: "persistence",
                            meeting_id = %id,
                            "indexed orphan meeting folder (self-heal)"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "persistence",
                        folder = %path.display(),
                        error = %e,
                        "skipping meeting folder during orphan reconcile"
                    );
                }
            }
        }

        Ok(indexed)
    }

    /// The ids currently in the index and their stored `duration_ms` (string
    /// id form, matching the on-disk folder names). Used by
    /// [`Self::reconcile_orphans`] for the new-vs-known diff and to skip a
    /// disk read for any already-indexed meeting whose duration is not
    /// degenerate.
    async fn indexed_durations(&self) -> Result<std::collections::HashMap<String, u64>, Error> {
        let mut rows = self
            .conn
            .query("SELECT id, duration_ms FROM meetings", ())
            .await?;
        let mut out = std::collections::HashMap::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let duration_ms: i64 = row.get(1)?;
            out.insert(id, duration_ms as u64);
        }
        Ok(out)
    }
}

/// Build a [`MeetingListEntry`] from a meeting folder by reading its metadata
/// and (for the preview excerpt) the first transcript segment.
///
/// The blocking `std::fs` reads run on a `spawn_blocking` thread so they do not
/// occupy an async worker — see the module-level threading note.
async fn entry_from_folder(folder: &Path) -> Result<MeetingListEntry, Error> {
    let folder = folder.to_path_buf();
    tokio::task::spawn_blocking(move || entry_from_folder_blocking(&folder))
        .await
        .map_err(|e| Error::Migration(format!("rebuild join error: {e}")))?
}

/// Synchronous core of [`entry_from_folder`], suitable for `spawn_blocking`.
fn entry_from_folder_blocking(folder: &Path) -> Result<MeetingListEntry, Error> {
    let meta = reader::read_metadata_inner(folder)?;
    let excerpt = derive_excerpt(folder);

    Ok(MeetingListEntry {
        id: meta.uuid,
        title: meta.title,
        started_at: meta.started_at,
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
        collection_id: meta.collection_id,
        recording_started: meta.recording_started,
        deleted_at: meta.deletion.deleted_at(),
    })
}

/// Synthesise and atomically write a minimal `metadata.json` for a folder that
/// holds recording data (an `audio.<ext>` file and/or `transcript.json`) but
/// no metadata — used by [`MeetingIndex::reconcile_orphans`] to recover
/// meetings that never reached `finalise` (a crash/kill mid-recording, most
/// commonly a pre-durability-fix orphan; see `architecture/cross-cutting.md`
/// "`metadata.json` is written at recording start, not only at finalise").
///
/// Neither the audio file nor `transcript.json` records an absolute
/// wall-clock time — transcript segments carry only offsets from the
/// recording's start —
/// so `started_at` is the best available on-disk proxy: the earlier of the two
/// files' modification times, falling back to the current time if neither
/// file's metadata is readable. `duration_ms` is the last transcript segment's
/// end offset (`0` when there is no transcript), and `speaker_count` is the
/// number of distinct `speaker_id` labels seen. The synthesised record uses
/// the codec/sample rate the resolved audio file's extension implies (16 kHz
/// mono either way — this device only ever writes 16 kHz mono, whichever
/// container), or the app's own opus default if the folder has recording
/// data (a transcript) but no audio file. Mirrors the placeholder
/// `audio_format` [`notes_crdt::MeetingFolder::ensure`] seeds for an inbound
/// sync folder — derived from the actual file present, so the synthesised
/// label matches the bytes on disk.
fn synthesize_metadata(folder: &Path, meeting_id: MeetingId) -> Result<(), Error> {
    let segments = reader::read_transcript_inner(folder).unwrap_or_default();
    let started_at = recovered_started_at(folder);
    let (duration_ms, speaker_count) = duration_and_speaker_count(&segments);
    let title = recovered_title(&started_at);
    let codec = match minutist_common::resolve_audio_path(folder)
        .and_then(|p| p.extension().and_then(|e| e.to_str()).map(str::to_string))
        .as_deref()
    {
        Some("m4a") => "aac",
        // "opus", or no audio file at all (transcript-only orphan) — the
        // latter has nothing to mislabel, so the app's own default is fine.
        _ => "opus",
    };

    let meta = MeetingMeta {
        uuid: meeting_id,
        title,
        started_at,
        ended_at: None,
        duration_ms,
        speaker_count,
        audio_format: AudioFormat {
            codec: codec.to_string(),
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
        // A recovered local recording; the lifecycle field defaults to the
        // local-processed state (owned by the sync/producer-gate work).
        processing: Default::default(),
        // Orphan recovery only fires when audio or a transcript already
        // exists on disk — never a draft.
        recording_started: true,
        // A recovered recording is never in the trash — nothing purged this
        // orphan (it still has on-disk data to recover).
        deletion: Default::default(),
    };

    crate::metadata::write_metadata_to_path(&folder.join("metadata.json"), &meta)
}

/// `(duration_ms, speaker_count)` implied by a folder's transcript segments —
/// the last segment's end offset (`0` when there are no segments), and the
/// count of distinct `speaker_id`s. Shared by [`synthesize_metadata`] (no
/// `metadata.json` at all) and [`backfill_degenerate_duration`] (`metadata.json`
/// exists but is degenerate) — the same derivation, two different triggers.
fn duration_and_speaker_count(segments: &[Segment]) -> (u64, u32) {
    let duration_ms = segments.last().map(|s| s.end_ms).unwrap_or(0);
    let speaker_count = segments
        .iter()
        .filter_map(|s| s.speaker_id.as_deref())
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u32;
    (duration_ms, speaker_count)
}

/// Backfill `duration_ms`/`speaker_count` from `transcript.json` for a
/// meeting whose `metadata.json` already exists but is degenerate (issue
/// 0064) — unlike [`synthesize_metadata`], `metadata.json` is present here,
/// so every other field (title, `processing`, `deletion`, ...) must be left
/// untouched. Gated on "duration still zero, transcript now has segments with
/// a real end offset" instead of "metadata.json absent": a meeting stuck on
/// `MeetingFolder::ensure`'s or `synthesize_metadata`'s own placeholder never
/// revisits it once processed, because `duration_ms == 0` alone does not
/// disqualify a `metadata.json` from "existing" for the orphan-recovery gate.
///
/// The predicate-and-mutate run in one `update_metadata_if` closure under one
/// lock acquisition, so there is no read-then-write window a concurrent
/// authoritative writer (lifecycle, `meeting_ops`) could land in between —
/// and re-checking `duration_ms == 0` inside the closure means a transcript
/// whose last segment's `end_ms` is itself `0` returns `Ok(false)` rather
/// than writing `0` and re-triggering this backfill on every future call.
///
/// Returns `Ok(true)` if a backfill was applied, `Ok(false)` if there was
/// nothing to do (already has a real duration, or no transcript segments
/// with a real end offset yet).
fn backfill_degenerate_duration(meetings_root: &Path, meeting_id: MeetingId) -> AppResult<bool> {
    let folder = meetings_root.join(meeting_id.0.to_string());
    let segments = reader::read_transcript_inner(&folder)?;
    let (duration_ms, speaker_count) = duration_and_speaker_count(&segments);
    if duration_ms == 0 {
        return Ok(false);
    }
    let applied = notes_crdt::update_metadata_if(meetings_root, meeting_id, |m| {
        (m.duration_ms == 0).then(|| {
            m.duration_ms = duration_ms;
            m.speaker_count = speaker_count;
        })
    })?;
    Ok(matches!(applied, notes_crdt::MetaUpdate::Applied(())))
}

/// Derive a recovered folder's `started_at`, preferring the earlier of
/// `transcript.json`'s and the meeting's audio file's modification times
/// (see [`synthesize_metadata`] for why neither file carries an absolute
/// timestamp directly), falling back to the current time if neither file's
/// metadata can be read.
fn recovered_started_at(folder: &Path) -> String {
    let transcript_mtime = file_mtime_utc(&folder.join("transcript.json"));
    let audio_mtime = minutist_common::resolve_audio_path(folder)
        .and_then(|p| file_mtime_utc(&p));

    let earliest = match (transcript_mtime, audio_mtime) {
        (Some(t), Some(a)) => Some(t.min(a)),
        (Some(t), None) => Some(t),
        (None, Some(a)) => Some(a),
        (None, None) => None,
    };

    earliest.unwrap_or_else(chrono::Utc::now).to_rfc3339()
}

/// A file's modification time as a UTC `DateTime`, or `None` if the file is
/// absent or its metadata cannot be read.
fn file_mtime_utc(path: &Path) -> Option<chrono::DateTime<chrono::Utc>> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    Some(chrono::DateTime::<chrono::Utc>::from(modified))
}

/// The recovered-meeting title: "Recovered recording <YYYY-MM-DD HH:MM>",
/// derived from `started_at`. Falls back to the raw `started_at` string if it
/// does not parse as RFC 3339 (defensive; `started_at` is always produced by
/// [`recovered_started_at`], which only ever emits RFC 3339).
fn recovered_title(started_at: &str) -> String {
    match chrono::DateTime::parse_from_rfc3339(started_at) {
        Ok(dt) => format!("Recovered recording {}", dt.format("%Y-%m-%d %H:%M")),
        Err(_) => format!("Recovered recording {started_at}"),
    }
}

/// Derive the meeting-list excerpt for a folder (live-test UX T6).
///
/// Prefer a one-line blurb derived from `summary.md` once a summary exists
/// (the summary's opening overview, via [`crate::summary::summary_blurb`]); fall
/// back to the first transcript segment otherwise (the prior behaviour). Shared
/// by the rebuild/reconcile path here and by `meeting_ops` so a restart keeps the
/// summary blurb a finished meeting shows.
pub(crate) fn derive_excerpt(folder: &Path) -> Option<String> {
    if let Ok(Some(md)) = crate::summary::read_summary(folder) {
        if let Some(blurb) = crate::summary::summary_blurb(&md) {
            return Some(blurb);
        }
    }
    reader::read_transcript_inner(folder)
        .ok()
        .and_then(|segs| segs.first().map(|s| truncate_excerpt(&s.text)))
}

/// Truncate a transcript snippet to a bounded preview length (120 chars,
/// on a char boundary), appending an ellipsis when truncated.
fn truncate_excerpt(text: &str) -> String {
    const MAX: usize = 120;
    if text.chars().count() <= MAX {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(MAX).collect();
        format!("{truncated}…")
    }
}

/// Drain a libsql `Rows` cursor into a `Vec<MeetingListEntry>`.
async fn rows_to_entries(mut rows: libsql::Rows) -> Result<Vec<MeetingListEntry>, Error> {
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push(row_to_entry(&row)?);
    }
    Ok(out)
}

/// Decode one index row into a [`MeetingListEntry`].
fn row_to_entry(row: &libsql::Row) -> Result<MeetingListEntry, Error> {
    let id_str: String = row.get(0)?;
    let id = parse_meeting_id(&id_str)?;
    let title: String = row.get(1)?;
    let started_at: String = row.get(2)?;
    let duration_ms: i64 = row.get(3)?;
    let speaker_count: i64 = row.get(4)?;
    // `excerpt` is nullable; libsql maps SQL NULL to an Option.
    let excerpt: Option<String> = row.get(5)?;
    // `collection_id` is nullable (NULL = unfiled); parse the stored UUID string.
    let collection_id: Option<String> = row.get(6)?;
    let collection_id = collection_id.as_deref().map(parse_collection_id).transpose()?;
    let recording_started: i64 = row.get(7)?;
    // `deleted_at` is nullable (NULL = active).
    let deleted_at: Option<String> = row.get(8)?;

    Ok(MeetingListEntry {
        id,
        title,
        started_at,
        duration_ms: duration_ms as u64,
        speaker_count: speaker_count as u32,
        excerpt,
        collection_id,
        recording_started: recording_started != 0,
        deleted_at,
    })
}

/// Parse a stored UUID string back into a [`MeetingId`].
///
/// `MeetingId` is `#[serde(transparent)]` over a hyphenated lowercase UUID
/// string, so round-trip through `serde_json` rather than taking a direct
/// `uuid` dependency (persistence's allowed deps are `common` + its own crates;
/// see `architecture/components.md`).
fn parse_meeting_id(s: &str) -> Result<MeetingId, Error> {
    serde_json::from_str::<MeetingId>(&format!("\"{s}\""))
        .map_err(|e| Error::Migration(format!("invalid meeting id in index: {s} ({e})")))
}

/// Parse a stored UUID string back into a [`CollectionId`] (mirrors
/// [`parse_meeting_id`]; both newtypes are `#[serde(transparent)]` over a UUID
/// string, so round-trip through `serde_json` rather than depending on `uuid`).
fn parse_collection_id(s: &str) -> Result<CollectionId, Error> {
    serde_json::from_str::<CollectionId>(&format!("\"{s}\""))
        .map_err(|e| Error::Migration(format!("invalid collection id in index: {s} ({e})")))
}

/// Escape `LIKE` wildcards in a user-supplied search term so they match
/// literally. Paired with `ESCAPE '\'` in the query.
fn escape_like(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Convenience: the conventional `index.db` filename under an app-data root.
/// `app-main` injects the resolved path; this helper keeps the filename in one
/// place for callers that build the path from the root.
pub fn index_db_path(app_data_root: &Path) -> PathBuf {
    app_data_root.join("index.db")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(n: u32, label: &str) -> MeetingListEntry {
        MeetingListEntry {
            id: MeetingId::new(),
            title: format!("{label} #{n}"),
            started_at: "2026-07-01T10:00:00Z".to_string(),
            duration_ms: 1000,
            speaker_count: 1,
            excerpt: None,
            collection_id: None,
            recording_started: true,
            deleted_at: None,
        }
    }

    /// Two concurrent calls into `rebuild_transaction` (the `BEGIN
    /// IMMEDIATE`..`COMMIT` span) must not interleave on the shared
    /// connection: neither call may error with "cannot start a transaction
    /// within a transaction", and the surviving batch must be complete — never
    /// a partial mix of the two, which would mean one call's writes were
    /// silently lost inside the other's transaction.
    ///
    /// Runs on a real multi-thread runtime (parity with the voiceprints
    /// interleave test) so the two tasks can genuinely race on `conn` rather
    /// than only cooperatively yield on one OS thread.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_rebuild_transactions_do_not_interleave() {
        let index = std::sync::Arc::new(MeetingIndex::open(":memory:").await.unwrap());

        let batch_a: Vec<MeetingListEntry> = (0..5).map(|i| sample_entry(i, "batch-a")).collect();
        let batch_b: Vec<MeetingListEntry> = (0..5).map(|i| sample_entry(i, "batch-b")).collect();

        let task_a = {
            let index = std::sync::Arc::clone(&index);
            tokio::spawn(async move { index.rebuild_transaction(batch_a).await })
        };
        let task_b = {
            let index = std::sync::Arc::clone(&index);
            tokio::spawn(async move { index.rebuild_transaction(batch_b).await })
        };

        let (result_a, result_b) = tokio::join!(task_a, task_b);
        result_a
            .expect("task a must not panic")
            .expect("first rebuild_transaction must not error");
        result_b
            .expect("task b must not panic")
            .expect("second rebuild_transaction must not error");

        let listed = index.list_meetings().await.unwrap();
        assert_eq!(
            listed.len(),
            5,
            "the final rebuild must leave exactly one complete 5-row batch, \
             never an interleaved partial mix of both"
        );
    }
}
