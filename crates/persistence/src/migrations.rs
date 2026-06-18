//! Forward-only schema migration runner for the libsql `index.db`.
//!
//! The index is a **derived cache** (rebuildable from the per-meeting folders
//! — see `architecture/cross-cutting.md` "Filesystem layout"), so migrations
//! never need to preserve data they cannot reconstruct. The runner is still
//! written to be **non-destructive**: each step is additive and runs only when
//! the recorded `schema_version` is below the step's target, so an empty DB and
//! a prior-schema DB both converge on the current schema without losing any
//! rows already present.
//!
//! # `schema_version`
//!
//! A single-row table records the highest applied migration. The runner reads
//! it, applies every migration strictly greater than the stored value in
//! order, and updates the stored value. Running the runner twice is a no-op the
//! second time (idempotent).

use libsql::Connection;

use crate::error::Error;

/// The schema version this build of `persistence` targets. Bump this and add a
/// matching arm in [`apply_migration`] when the schema changes.
pub const CURRENT_VERSION: i64 = 2;

/// Bring `conn`'s schema up to [`CURRENT_VERSION`].
///
/// Creates the `schema_version` bookkeeping table if absent, reads the current
/// version (0 for a fresh DB), then applies each migration `v` for
/// `current < v <= CURRENT_VERSION` in order. Idempotent: a DB already at
/// `CURRENT_VERSION` is left untouched.
pub async fn run(conn: &Connection) -> Result<(), Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (
            id        INTEGER PRIMARY KEY CHECK (id = 1),
            version   INTEGER NOT NULL
        )",
        (),
    )
    .await?;

    let current = read_version(conn).await?;

    for v in (current + 1)..=CURRENT_VERSION {
        apply_migration(conn, v).await?;
        write_version(conn, v).await?;
        tracing::info!(
            target: "persistence",
            from = current,
            to = v,
            "index.db schema migrated"
        );
    }

    Ok(())
}

/// Read the recorded schema version; `0` if no row exists yet (fresh DB).
async fn read_version(conn: &Connection) -> Result<i64, Error> {
    let mut rows = conn
        .query("SELECT version FROM schema_version WHERE id = 1", ())
        .await?;
    match rows.next().await? {
        Some(row) => Ok(row.get::<i64>(0)?),
        None => Ok(0),
    }
}

/// Upsert the recorded schema version to `version`.
async fn write_version(conn: &Connection, version: i64) -> Result<(), Error> {
    conn.execute(
        "INSERT INTO schema_version (id, version) VALUES (1, ?1)
         ON CONFLICT(id) DO UPDATE SET version = excluded.version",
        libsql::params![version],
    )
    .await?;
    Ok(())
}

/// Apply the DDL for a single migration step.
///
/// Each arm is additive. Migration 1 creates the `meetings` index table — the
/// columns mirror [`minutist_common::MeetingListEntry`] so a list query
/// reads straight from the index without touching a meeting folder.
async fn apply_migration(conn: &Connection, version: i64) -> Result<(), Error> {
    match version {
        1 => {
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meetings (
                    id             TEXT PRIMARY KEY,
                    title          TEXT NOT NULL,
                    started_at     TEXT NOT NULL,
                    duration_ms    INTEGER NOT NULL,
                    speaker_count  INTEGER NOT NULL,
                    excerpt        TEXT
                )",
                (),
            )
            .await?;
            // Most-recent-first list queries order by started_at descending.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_meetings_started_at
                 ON meetings (started_at DESC)",
                (),
            )
            .await?;
            Ok(())
        }
        2 => {
            // Add the derived `collection_id` mirror (a meeting's user-facing
            // "folder", authoritative in its `metadata.json`) so the list view
            // can filter by folder straight from the index. Nullable = unfiled.
            conn.execute(
                "ALTER TABLE meetings ADD COLUMN collection_id TEXT",
                (),
            )
            .await?;
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_meetings_collection_id
                 ON meetings (collection_id)",
                (),
            )
            .await?;
            Ok(())
        }
        other => Err(Error::Migration(format!(
            "no migration defined for schema version {other}"
        ))),
    }
}
