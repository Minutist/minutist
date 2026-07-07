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
///
/// Each step's DDL and version bump run inside one `BEGIN IMMEDIATE`/`COMMIT`
/// transaction, so a crash between them can never happen: either both land or
/// neither does, and the next `run` call resumes from the last committed
/// version rather than re-applying (non-idempotent) DDL against a half-applied
/// schema.
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
        conn.execute("BEGIN IMMEDIATE", ()).await?;

        let result = async {
            apply_migration(conn, v).await?;
            write_version(conn, v).await?;
            Ok::<(), Error>(())
        }
        .await;

        match result {
            Ok(()) => {
                if let Err(e) = conn.execute("COMMIT", ()).await {
                    // A failed COMMIT can leave the transaction open; roll it
                    // back (best-effort) so the next `run` iteration's BEGIN
                    // IMMEDIATE does not error with a nested transaction.
                    if let Err(rb) = conn.execute("ROLLBACK", ()).await {
                        tracing::warn!(
                            target: "persistence",
                            error = %rb,
                            "index.db migration rollback failed after a COMMIT error"
                        );
                    }
                    return Err(e.into());
                }
            }
            Err(e) => {
                if let Err(rb) = conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(
                        target: "persistence",
                        error = %rb,
                        "index.db migration rollback failed after a migration error"
                    );
                }
                return Err(e);
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use libsql::Builder;

    async fn open_mem() -> Connection {
        let db = Builder::new_local(":memory:").build().await.unwrap();
        db.connect().unwrap()
    }

    #[tokio::test]
    async fn run_migrates_fresh_db_to_current_version() {
        let conn = open_mem().await;
        run(&conn).await.unwrap();
        assert_eq!(read_version(&conn).await.unwrap(), CURRENT_VERSION);
    }

    /// A migration step that fails *after* changing part of the schema must
    /// leave nothing half-applied: `run` rolls the whole step back and does
    /// not advance `schema_version`, so a later `run` re-applies the step
    /// from scratch. Drives the failure through `run` itself (not a hand-run
    /// BEGIN/ROLLBACK), so it is RED if the per-step `BEGIN IMMEDIATE`/`COMMIT`
    /// wrapping is removed from `run`: without it, migration 2's `ALTER TABLE
    /// … ADD COLUMN` commits in autocommit mode even though the step as a
    /// whole errors, leaving the `collection_id` column behind.
    #[tokio::test]
    async fn run_rolls_back_a_partially_applied_migration_step() {
        let conn = open_mem().await;

        // Bring the DB to schema version 1 — the state migration 2 runs
        // against — using the runner's own building blocks.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL
            )",
            (),
        )
        .await
        .unwrap();
        apply_migration(&conn, 1).await.unwrap();
        write_version(&conn, 1).await.unwrap();

        // Sabotage migration 2's *second* statement (the CREATE INDEX) by
        // pre-creating a table whose name collides with the index. Migration
        // 2's first statement (`ALTER TABLE meetings ADD COLUMN collection_id`)
        // still succeeds and mutates the schema; the following
        // `CREATE INDEX IF NOT EXISTS idx_meetings_collection_id` then fails
        // ("there is already a table named …") — a real mid-step failure that
        // lands only after a partial change.
        conn.execute(
            "CREATE TABLE idx_meetings_collection_id (placeholder INTEGER)",
            (),
        )
        .await
        .unwrap();

        let result = run(&conn).await;
        assert!(
            result.is_err(),
            "migration 2 must fail on the index/table name collision"
        );

        // schema_version must NOT advance past 1: the version bump shares the
        // step's transaction, which rolled back with the DDL.
        assert_eq!(read_version(&conn).await.unwrap(), 1);

        // The partial change — migration 2's ADD COLUMN — must have rolled
        // back. This is the assertion that fails without the wrapping
        // transaction: in autocommit mode the ALTER would persist even though
        // the CREATE INDEX after it errored, leaving a half-applied step.
        let col_probe = conn
            .execute("SELECT collection_id FROM meetings LIMIT 0", ())
            .await;
        assert!(
            col_probe.is_err(),
            "aborted migration 2 must not leave its ADD COLUMN behind"
        );

        // With the sabotage removed, `run` must converge cleanly from the
        // rolled-back state to CURRENT_VERSION — the step re-applies whole.
        conn.execute("DROP TABLE idx_meetings_collection_id", ())
            .await
            .unwrap();
        run(&conn).await.unwrap();
        assert_eq!(read_version(&conn).await.unwrap(), CURRENT_VERSION);
        conn.execute("SELECT collection_id FROM meetings LIMIT 0", ())
            .await
            .unwrap();
    }
}
