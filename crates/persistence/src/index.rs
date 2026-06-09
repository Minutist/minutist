//! The libsql `index.db` meeting index — a derived cache over the per-meeting
//! folders.
//!
//! `index.db` holds one row per meeting mirroring
//! [`meeting_app_common::MeetingListEntry`], so the meeting-list view (FR-33)
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
use meeting_app_common::{AppResult, MeetingId, MeetingListEntry};

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
        Ok(Self { db, conn })
    }

    /// All meetings, most-recent first (ordered by `started_at` descending).
    pub async fn list_meetings(&self) -> AppResult<Vec<MeetingListEntry>> {
        let rows = self
            .conn
            .query(
                "SELECT id, title, started_at, duration_ms, speaker_count, excerpt
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
                "SELECT id, title, started_at, duration_ms, speaker_count, excerpt
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
                "INSERT INTO meetings (id, title, started_at, duration_ms, speaker_count, excerpt)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(id) DO UPDATE SET
                     title = excluded.title,
                     started_at = excluded.started_at,
                     duration_ms = excluded.duration_ms,
                     speaker_count = excluded.speaker_count,
                     excerpt = excluded.excerpt",
                libsql::params![
                    entry.id.0.to_string(),
                    entry.title.clone(),
                    entry.started_at.clone(),
                    entry.duration_ms as i64,
                    entry.speaker_count as i64,
                    entry.excerpt.clone(),
                ],
            )
            .await?;
        Ok(())
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
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
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
    async fn rebuild_transaction(&self, entries: Vec<MeetingListEntry>) -> Result<(), Error> {
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
                self.conn.execute("COMMIT", ()).await?;
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
    /// read plus a set-diff; only folders not already indexed incur a
    /// metadata/transcript read + `upsert`.
    ///
    /// Guards in-session visibility against any missed `upsert` (e.g. the
    /// process being killed between finalise and the stop-time index write):
    /// `list_meetings` runs this so a meeting present on disk can never stay
    /// hidden until the next startup `rebuild_from_disk`. Unlike
    /// [`Self::rebuild_from_disk`] it never deletes — a folder removed off-app
    /// is reconciled by the next rebuild, not here. Returns the number of orphan
    /// meetings newly indexed.
    pub async fn reconcile_orphans(&self, meetings_root: &Path) -> AppResult<usize> {
        Ok(self.reconcile_orphans_inner(meetings_root).await?)
    }

    async fn reconcile_orphans_inner(&self, meetings_root: &Path) -> Result<usize, Error> {
        // Ids already indexed — the cheap half of the diff.
        let known = self.existing_ids().await?;

        let dirs = match std::fs::read_dir(meetings_root) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(Error::Io(e)),
        };

        let mut indexed = 0usize;
        for entry in dirs {
            let entry = entry.map_err(Error::Io)?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            // The folder name is the meeting uuid; skip already-indexed folders
            // before any (more expensive) metadata read.
            let id = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) if known.contains(name) => continue,
                Some(name) => name.to_string(),
                None => continue,
            };
            if !path.join("metadata.json").exists() {
                continue;
            }

            match entry_from_folder(&path).await {
                Ok(list_entry) => {
                    self.upsert_inner(&list_entry).await?;
                    indexed += 1;
                    tracing::info!(
                        target: "persistence",
                        meeting_id = %id,
                        "indexed orphan meeting folder (self-heal)"
                    );
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

    /// The set of meeting ids currently in the index (string form, matching the
    /// on-disk folder names). Used by [`Self::reconcile_orphans`] for the diff.
    async fn existing_ids(&self) -> Result<std::collections::HashSet<String>, Error> {
        let mut rows = self.conn.query("SELECT id FROM meetings", ()).await?;
        let mut ids = std::collections::HashSet::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            ids.insert(id);
        }
        Ok(ids)
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
    let excerpt = reader::read_transcript_inner(folder)
        .ok()
        .and_then(|segs| segs.first().map(|s| truncate_excerpt(&s.text)));

    Ok(MeetingListEntry {
        id: meta.uuid,
        title: meta.title,
        started_at: meta.started_at,
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
    })
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

    Ok(MeetingListEntry {
        id,
        title,
        started_at,
        duration_ms: duration_ms as u64,
        speaker_count: speaker_count as u32,
        excerpt,
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
