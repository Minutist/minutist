//! Forward-only schema migration runner for `voiceprints.db`.
//!
//! Mirrors [`crate::migrations`] (the `index.db` runner) exactly in structure:
//! a `schema_version` bookkeeping table, `read_version` / `write_version` /
//! `apply_migration`, and a `CURRENT_VERSION` constant. Each step is additive
//! and runs only when the recorded version is below the step's target, so an
//! empty DB and a prior-schema DB both converge without data loss.
//!
//! Unlike `index.db`, `voiceprints.db` is **primary, non-rebuildable** data.
//! The runner never drops or truncates existing rows. A migration failure
//! returns an error that the caller maps to enrolment-OFF (see
//! `architecture/cross-cutting.md` — "Voiceprint matching").

use libsql::Connection;

use crate::error::Error;

/// The schema version this build of `persistence` targets for `voiceprints.db`.
/// Bump this and add a matching arm in [`apply_migration`] when the schema changes.
pub const CURRENT_VERSION: i64 = 1;

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
                            "voiceprints.db migration rollback failed after a COMMIT error"
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
                        "voiceprints.db migration rollback failed after a migration error"
                    );
                }
                return Err(e);
            }
        }

        tracing::info!(
            target: "persistence",
            from = current,
            to = v,
            "voiceprints.db schema migrated"
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
/// Migration 1 creates the three §2.9.1 tables:
/// - `voiceprint_identity`: one row per enrolled speaker (stable across renames
///   and merges).
/// - `voiceprint_centroid`: one row per acquisition-condition gallery entry;
///   `embedding` is a cached count-weighted unit-mean of its contributions.
/// - `voiceprint_contribution`: one row per `(meeting, label)` that fed a
///   centroid; makes refinement reversible by letting the store recompute the
///   cached centroid after a contribution is added or removed.
///
/// All three tables use `ON DELETE CASCADE` so deleting an identity removes its
/// centroids, and deleting a centroid removes its contributions — no orphan
/// cleanup required.
async fn apply_migration(conn: &Connection, version: i64) -> Result<(), Error> {
    match version {
        1 => {
            // Identity: a person; stable across renames and merges.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS voiceprint_identity (
                    id           TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    model_id     TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                )",
                (),
            )
            .await?;

            // Centroid: one acquisition condition for an identity.
            // `embedding` is f32 LE bytes; `dim` tracks the expected length;
            // `sample_count` is the sum of contribution counts (cached for quick
            // gallery queries without loading all contributions).
            conn.execute(
                "CREATE TABLE IF NOT EXISTS voiceprint_centroid (
                    id              TEXT PRIMARY KEY,
                    identity_id     TEXT NOT NULL
                        REFERENCES voiceprint_identity(id) ON DELETE CASCADE,
                    embedding       BLOB NOT NULL,
                    dim             INTEGER NOT NULL,
                    sample_count    INTEGER NOT NULL,
                    condition_label TEXT,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                )",
                (),
            )
            .await?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_centroid_identity
                 ON voiceprint_centroid (identity_id)",
                (),
            )
            .await?;

            // Contribution: one (meeting, label) that fed a centroid.
            // Retaining per-contribution centroids (Q#12 decision) makes
            // refinement reversible: drop a contribution, recompute the centroid.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS voiceprint_contribution (
                    id          TEXT PRIMARY KEY,
                    centroid_id TEXT NOT NULL
                        REFERENCES voiceprint_centroid(id) ON DELETE CASCADE,
                    meeting_id  TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    embedding   BLOB NOT NULL,
                    count       INTEGER NOT NULL,
                    created_at  TEXT NOT NULL
                )",
                (),
            )
            .await?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_contribution_centroid
                 ON voiceprint_contribution (centroid_id)",
                (),
            )
            .await?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_contribution_meeting
                 ON voiceprint_contribution (meeting_id)",
                (),
            )
            .await?;

            Ok(())
        }
        other => Err(Error::Migration(format!(
            "no voiceprints migration defined for schema version {other}"
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

    /// A migration step that fails *after* creating some of its tables must
    /// leave nothing half-applied: `run` rolls the whole step back and does
    /// not advance `schema_version`, so a later `run` re-creates the schema
    /// from scratch. Drives the failure through `run` itself, so it is RED if
    /// the per-step `BEGIN IMMEDIATE`/`COMMIT` wrapping is removed: without
    /// it, the `CREATE TABLE`s that ran before the failing statement commit in
    /// autocommit mode and stay behind.
    #[tokio::test]
    async fn run_rolls_back_a_partially_applied_migration_step() {
        let conn = open_mem().await;

        // Sabotage migration 1's *third* statement — `CREATE INDEX IF NOT
        // EXISTS idx_centroid_identity` — by pre-creating a *table* of that
        // name. `IF NOT EXISTS` suppresses a clash only with an existing
        // INDEX, not a TABLE, so the statement errors ("there is already a
        // table named …"). Crucially it fails only after migration 1's first
        // two statements (`CREATE TABLE voiceprint_identity` and
        // `voiceprint_centroid`) have already run — a real partial change.
        conn.execute(
            "CREATE TABLE idx_centroid_identity (placeholder INTEGER)",
            (),
        )
        .await
        .unwrap();

        let result = run(&conn).await;
        assert!(
            result.is_err(),
            "migration 1 must fail on the index/table name collision"
        );

        // schema_version must NOT advance from 0: the version bump shares the
        // step's transaction, which rolled back with the DDL.
        assert_eq!(read_version(&conn).await.unwrap(), 0);

        // The partial change — the two tables created before the failing
        // CREATE INDEX — must have rolled back. This is the assertion that
        // fails without the wrapping transaction: in autocommit mode
        // `voiceprint_identity` would persist even though a later statement
        // in the same step errored, leaving a half-applied migration.
        let table_exists = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='voiceprint_identity'",
                (),
            )
            .await
            .unwrap()
            .next()
            .await
            .unwrap();
        assert!(
            table_exists.is_none(),
            "aborted migration 1 must not leave voiceprint_identity behind"
        );

        // With the sabotage removed, `run` must converge cleanly from the
        // rolled-back state to CURRENT_VERSION — the step re-applies whole.
        conn.execute("DROP TABLE idx_centroid_identity", ())
            .await
            .unwrap();
        run(&conn).await.unwrap();
        assert_eq!(read_version(&conn).await.unwrap(), CURRENT_VERSION);
    }
}
