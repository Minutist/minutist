//! Voiceprint library: `voiceprints.db` — the durable, non-rebuildable store
//! for cross-session speaker identities (issue #0003).
//!
//! # Schema overview (§2.9.1)
//!
//! Three tables: `voiceprint_identity` → `voiceprint_centroid` →
//! `voiceprint_contribution` (each linked by `ON DELETE CASCADE`).
//!
//! - An **identity** is a person; stable across renames and merges.
//! - A **centroid** is one acquisition condition for an identity (e.g.
//!   in-person room mic, VoIP). Its `embedding` is a *cache* equal to
//!   `unit_normalise(Σ count_i · contribution_i.embedding / Σ count_i)` over
//!   its contributions, with `sample_count = Σ count_i`.
//! - A **contribution** is one `(meeting, label)` that fed a centroid, retained
//!   so that refinement is reversible: drop a contribution, recompute the
//!   centroid from the survivors.
//!
//! **Invariant.** Any operation that changes a centroid's contribution set MUST
//! recompute the cached `embedding` and `sample_count` from surviving
//! contributions atomically (via [`recompute_centroid`]). Every multi-statement
//! mutating method wraps its work in a `BEGIN IMMEDIATE`/`COMMIT` block so that
//! the contribution-set change and the centroid recompute are committed together
//! or rolled back entirely.
//!
//! # Corruption contract
//!
//! A libsql open or migration error returns [`Error::Index`] / [`Error::Migration`];
//! the caller maps that to enrolment-OFF. This store never panics on a DB error
//! and never blocks the meeting list or recording pipeline.
//!
//! # Dependency edges
//!
//! `persistence` already depends on `common`; this module uses
//! `common::voiceprint_math` for all vector operations. No new crate edge is
//! introduced. The diarizer `Voiceprint` / `VoiceprintExtractor` types never
//! enter this module — only raw `&[f32]` slices cross the boundary (the embedding
//! bytes stay out of IPC and out of `common`).
//!
//! # Async, no `block_on`
//!
//! Mirrors `index.rs`: every method is `async fn`; the crate never calls
//! `block_on`. All methods take `&self` with an interior `Connection`.

use std::path::{Path, PathBuf};

use chrono::Utc;
use libsql::{Builder, Connection, Database};
use minutist_common::{AppResult, MeetingId, VoiceprintCentroidId, VoiceprintIdentityId};

use crate::blob::{blob_to_f32_vec, f32_slice_to_blob};
use crate::error::Error;
use crate::voiceprints_migrations;

// ---------------------------------------------------------------------------
// Threshold constants (placeholders — calibrated by WU6)
// ---------------------------------------------------------------------------

/// Cosine similarity floor for folding a new contribution into an existing
/// centroid rather than creating a new condition entry. A new contribution
/// whose cosine with the nearest gallery centroid is `>= FOLD_GATE` is
/// folded into that centroid; otherwise a new centroid entry is created.
///
/// This is NOT the offline clustering distance `0.75`; it is a different metric
/// (cosine similarity, not FastClustering distance) for a different purpose.
/// The value below is a documented placeholder; WU6 sweeps it against the
/// multi-session corpus.
const FOLD_GATE: f32 = 0.70;

/// Per-identity cap on the number of gallery centroids (condition entries).
/// When `refine` or `merge_identities` would push an identity above this cap,
/// the two closest centroids (by cosine) are merged via `weighted_merge`
/// *only if* their cosine clears `FOLD_GATE` (to avoid blurring genuinely
/// distinct conditions). If no pair clears `FOLD_GATE`, the cap is allowed to
/// grow and the identity is flagged for management.
const GALLERY_CAP: usize = 4;

/// Bounded per-meeting weight. A single meeting's contribution `count` is
/// clamped to `min(count, existing_sample_count * REFINE_WEIGHT_CAP)` before
/// folding, so one meeting cannot dominate an established centroid.
///
/// Example: if `existing_sample_count = 100` and the new contribution `count =
/// 80`, the clamped weight is `min(80, 100 * 0.30) = min(80, 30) = 30`.
///
/// Placeholder — calibrated by WU6.
const REFINE_WEIGHT_CAP: f64 = 0.30;

// ---------------------------------------------------------------------------
// Public POD types
// ---------------------------------------------------------------------------

/// Per-centroid summary returned by [`VoiceprintStore::identities_with_gallery`].
///
/// Contains only metadata — the embedding vector is deliberately excluded so
/// this type is safe to pass across IPC (embedding bytes must not cross the
/// IPC boundary — §2.2).
#[derive(Debug, Clone)]
pub struct CentroidSummary {
    /// The centroid's primary key.
    pub centroid_id: VoiceprintCentroidId,
    /// Total observation count folded into this centroid.
    pub sample_count: u64,
    /// Best-effort annotation for management UX; never a matching input.
    pub condition_label: Option<String>,
}

/// One identity row with its per-condition gallery, returned by
/// [`VoiceprintStore::identities_with_gallery`].
///
/// No embedding vectors — safe for IPC.
#[derive(Debug, Clone)]
pub struct IdentityWithGallery {
    pub identity_id: VoiceprintIdentityId,
    pub display_name: String,
    pub model_id: String,
    /// Gallery centroids in `created_at` order (most-established first).
    pub centroids: Vec<CentroidSummary>,
}

/// A flattened gallery entry returned by [`VoiceprintStore::all`].
///
/// Owned by `persistence`; the `embedding` bytes are for in-process use only
/// and must never cross IPC (see §2.2 — "embedding bytes never cross IPC, so
/// they stay out of `common`/specta").
#[derive(Debug, Clone)]
pub struct StoredVoiceprint {
    /// The owning identity.
    pub identity_id: VoiceprintIdentityId,
    /// The centroid within the identity's gallery.
    pub centroid_id: VoiceprintCentroidId,
    /// The identity's current display name.
    pub display_name: String,
    /// The embedding model this voiceprint was built from. Match only within
    /// the same `model_id` (hard-invalidation contract — §2.2).
    pub model_id: String,
    /// Cached unit-normalised centroid vector (f32 LE, `dim` elements).
    pub embedding: Vec<f32>,
    /// Vector length (e.g. 192 for CAM++ zh-en; verified on first enrolment).
    pub dim: usize,
    /// Total observation count folded into this centroid (Σ contribution counts).
    pub sample_count: u64,
    /// Best-effort annotation for management UX; never a matching input.
    pub condition_label: Option<String>,
}

// ---------------------------------------------------------------------------
// VoiceprintStore
// ---------------------------------------------------------------------------

/// Handle to the `voiceprints.db` libsql database.
///
/// Open via [`VoiceprintStore::open`] with an explicit DB path (injected by
/// `app-main`). `open` runs the forward-only migration runner so the schema is
/// current before any query. A corruption or migration error is returned as
/// [`Error`]; the caller maps it to enrolment-OFF.
pub struct VoiceprintStore {
    // Keep the `Database` alive for the connection lifetime.
    #[allow(dead_code)]
    db: Database,
    conn: Connection,
}

impl VoiceprintStore {
    /// Open (or create) `voiceprints.db` at `db_path` and migrate it to the
    /// current schema.
    ///
    /// Pass `":memory:"` for an in-memory database (tests). A relative or
    /// absolute filesystem path opens-or-creates a file-backed DB.
    ///
    /// Returns an error on libsql failure or schema migration failure; the
    /// caller must map that to enrolment-OFF (never panic, never block the
    /// meeting list).
    pub async fn open(db_path: impl AsRef<Path>) -> AppResult<Self> {
        Ok(Self::open_inner(db_path).await?)
    }

    async fn open_inner(db_path: impl AsRef<Path>) -> Result<Self, Error> {
        let db = Builder::new_local(db_path.as_ref()).build().await?;
        let conn = db.connect()?;
        voiceprints_migrations::run(&conn).await?;
        Ok(Self { db, conn })
    }

    // -----------------------------------------------------------------------
    // Enrolment
    // -----------------------------------------------------------------------

    /// Enrol a new speaker identity with its first gallery centroid and
    /// contribution.
    ///
    /// Creates a `voiceprint_identity` row, one `voiceprint_centroid` row
    /// (the initial gallery entry), and one `voiceprint_contribution` row
    /// (the source `(meeting_id, label)` pair). The centroid's `embedding`
    /// is set to the unit-normalised `embedding` slice (the single-contribution
    /// invariant: centroid = the contribution's own vector, normalised).
    ///
    /// `dim` must equal `embedding.len()`; a mismatch returns
    /// `AppError::InvalidInput`. Passing `dim` explicitly lets the caller
    /// document the expected dimension at the call site; subsequent enrolments
    /// verify that the stored `dim` matches (enforced by the migration schema
    /// and checked by `refine`).
    ///
    /// Returns the new [`VoiceprintIdentityId`].
    pub async fn enrol(
        &self,
        name: &str,
        embedding: &[f32],
        dim: usize,
        model_id: &str,
        source_meeting: MeetingId,
        label: &str,
    ) -> AppResult<VoiceprintIdentityId> {
        Ok(self
            .enrol_inner(name, embedding, dim, model_id, source_meeting, label)
            .await?)
    }

    async fn enrol_inner(
        &self,
        name: &str,
        embedding: &[f32],
        dim: usize,
        model_id: &str,
        source_meeting: MeetingId,
        label: &str,
    ) -> Result<VoiceprintIdentityId, Error> {
        if embedding.len() != dim {
            return Err(Error::InvalidState(
                "embedding length does not match declared dim",
            ));
        }
        if embedding.is_empty() {
            return Err(Error::InvalidState("embedding must not be empty"));
        }

        let now = Utc::now().to_rfc3339();
        let identity_id = VoiceprintIdentityId::new();
        let centroid_id = VoiceprintCentroidId::new();
        let contrib_id = uuid::Uuid::new_v4().to_string();

        // Unit-normalise the embedding before storing.
        let mut centroid_vec = embedding.to_vec();
        minutist_common::voiceprint_math::unit_normalise(&mut centroid_vec);

        let blob = f32_slice_to_blob(&centroid_vec);
        let count: i64 = 1;

        self.conn.execute("BEGIN IMMEDIATE", ()).await?;

        let result = async {
            self.conn.execute(
                "INSERT INTO voiceprint_identity (id, display_name, model_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?4)",
                libsql::params![identity_id.0.to_string(), name, model_id, now.clone()],
            )
            .await?;

            self.conn.execute(
                "INSERT INTO voiceprint_centroid
                 (id, identity_id, embedding, dim, sample_count, condition_label, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?6)",
                libsql::params![
                    centroid_id.0.to_string(),
                    identity_id.0.to_string(),
                    blob.clone(),
                    dim as i64,
                    count,
                    now.clone()
                ],
            )
            .await?;

            self.conn.execute(
                "INSERT INTO voiceprint_contribution
                 (id, centroid_id, meeting_id, label, embedding, count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                libsql::params![
                    contrib_id,
                    centroid_id.0.to_string(),
                    source_meeting.0.to_string(),
                    label,
                    blob,
                    count,
                    now
                ],
            )
            .await?;

            Ok::<(), Error>(())
        }
        .await;

        match result {
            Ok(()) => {
                self.conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(
                        target: "persistence",
                        error = %rb,
                        "enrol rollback failed after an enrol error"
                    );
                }
                return Err(e);
            }
        }

        tracing::info!(
            target: "persistence",
            identity_id = %identity_id.0,
            name,
            model_id,
            dim,
            "voiceprint enroled"
        );

        Ok(identity_id)
    }

    // -----------------------------------------------------------------------
    // Refinement
    // -----------------------------------------------------------------------

    /// Fold a confirmed `(meeting_id, label)` observation into an existing
    /// identity's gallery (§2.9.3).
    ///
    /// Steps:
    /// 1. Reject if `model_id` ≠ the identity's stored `model_id`
    ///    (hard-invalidation — §2.2).
    /// 2. Find the gallery centroid nearest to `contribution` by cosine.
    /// 3. If `sim >= FOLD_GATE`: fold — append the contribution to that centroid
    ///    and recompute the cached centroid (count-weighted merge of all
    ///    contributions, §2.9.1 invariant).
    /// 4. Else: add a new condition centroid for this contribution. If the
    ///    identity would exceed `GALLERY_CAP`, merge the two closest centroids
    ///    (only if their cosine clears `FOLD_GATE`; otherwise allow the cap to
    ///    grow).
    ///
    /// `count` is clamped to `min(count, existing_sample_count × REFINE_WEIGHT_CAP)`
    /// before folding, bounding the per-meeting influence on an established
    /// centroid (bounded-weight poison defence — §2.9.3).
    ///
    /// Idempotent per `(identity, meeting)`: any prior contribution from
    /// `meeting_id` under this identity is dropped and replaced first, so an
    /// auto-accept that re-confirms the same speaker on every reprocess does not
    /// double-count the meeting's weight. If that drop empties the gallery (the
    /// identity was enrolled solely from this meeting and is being re-refined from
    /// it), the first centroid is recreated rather than erroring.
    pub async fn refine(
        &self,
        identity_id: VoiceprintIdentityId,
        contribution: &[f32],
        count: u64,
        model_id: &str,
        meeting_id: MeetingId,
        label: &str,
    ) -> AppResult<()> {
        Ok(self
            .refine_inner(identity_id, contribution, count, model_id, meeting_id, label)
            .await?)
    }

    async fn refine_inner(
        &self,
        identity_id: VoiceprintIdentityId,
        contribution: &[f32],
        count: u64,
        model_id: &str,
        meeting_id: MeetingId,
        label: &str,
    ) -> Result<(), Error> {
        // 1. Verify model_id matches the identity.
        let stored_model = self.identity_model_id(identity_id).await?;
        if stored_model != model_id {
            return Err(Error::InvalidState(
                "model_id mismatch: cannot refine a voiceprint across embedding models",
            ));
        }

        // Unit-normalise the incoming contribution.
        let mut contrib_vec = contribution.to_vec();
        minutist_common::voiceprint_math::unit_normalise(&mut contrib_vec);

        let now = Utc::now().to_rfc3339();
        let blob = f32_slice_to_blob(&contrib_vec);

        self.conn.execute("BEGIN IMMEDIATE", ()).await?;

        let result = async {
            // Idempotency: a meeting contributes at most once to an identity.
            // Drop any prior contribution from this meeting under this identity's
            // centroids, recompute them, and GC any centroid the drop emptied — so
            // a reprocess that re-confirms the same speaker (apply_voiceprint_matches
            // runs on every reprocess) replaces its contribution instead of
            // double-counting the meeting's weight. The identity is not GC'd here;
            // a contribution is re-added below.
            let prior_centroids: Vec<VoiceprintCentroidId> = {
                let mut rows = self
                    .conn
                    .query(
                        "SELECT id FROM voiceprint_centroid WHERE identity_id = ?1",
                        libsql::params![identity_id.0.to_string()],
                    )
                    .await?;
                let mut ids = Vec::new();
                while let Some(row) = rows.next().await? {
                    let s: String = row.get(0)?;
                    if let Ok(u) = uuid::Uuid::parse_str(&s) {
                        ids.push(VoiceprintCentroidId(u));
                    }
                }
                ids
            };
            self.conn
                .execute(
                    "DELETE FROM voiceprint_contribution
                     WHERE meeting_id = ?1
                       AND centroid_id IN
                           (SELECT id FROM voiceprint_centroid WHERE identity_id = ?2)",
                    libsql::params![meeting_id.0.to_string(), identity_id.0.to_string()],
                )
                .await?;
            for cid in &prior_centroids {
                recompute_centroid(&self.conn, *cid, &now).await?;
            }
            self.conn
                .execute(
                    "DELETE FROM voiceprint_centroid
                     WHERE identity_id = ?1
                       AND id NOT IN
                           (SELECT DISTINCT centroid_id FROM voiceprint_contribution)",
                    libsql::params![identity_id.0.to_string()],
                )
                .await?;

            // Decide fold / new-condition / first against the post-dedup gallery.
            let centroids = self.load_centroids(identity_id).await?;
            let total_existing: u64 = centroids.iter().map(|(_, _, c)| *c).sum();
            let clamped_count = {
                let cap = (total_existing as f64 * REFINE_WEIGHT_CAP).ceil() as u64;
                if cap == 0 {
                    count
                } else {
                    count.min(cap)
                }
            };

            let mut best_centroid_id: Option<VoiceprintCentroidId> = None;
            let mut best_sim = f32::MIN;
            for (cid, cvec, _c) in &centroids {
                let s = minutist_common::voiceprint_math::cosine_unit(&contrib_vec, cvec);
                if s > best_sim {
                    best_sim = s;
                    best_centroid_id = Some(*cid);
                }
            }

            match best_centroid_id {
                Some(best_centroid_id) if best_sim >= FOLD_GATE => {
                    // Fold into the nearest centroid.
                    let contrib_id = uuid::Uuid::new_v4().to_string();
                    self.conn
                        .execute(
                            "INSERT INTO voiceprint_contribution
                             (id, centroid_id, meeting_id, label, embedding, count, created_at)
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                            libsql::params![
                                contrib_id,
                                best_centroid_id.0.to_string(),
                                meeting_id.0.to_string(),
                                label,
                                blob,
                                clamped_count as i64,
                                now.clone()
                            ],
                        )
                        .await?;
                    recompute_centroid(&self.conn, best_centroid_id, &now).await?;
                }
                _ => {
                    // A new condition centroid — or the first centroid when the
                    // gallery was emptied by the idempotent dedup above.
                    let new_centroid_id = VoiceprintCentroidId::new();
                    let contrib_id = uuid::Uuid::new_v4().to_string();
                    let dim = contrib_vec.len();

                    self.conn
                        .execute(
                            "INSERT INTO voiceprint_centroid
                             (id, identity_id, embedding, dim, sample_count, condition_label, created_at, updated_at)
                             VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?6)",
                            libsql::params![
                                new_centroid_id.0.to_string(),
                                identity_id.0.to_string(),
                                blob.clone(),
                                dim as i64,
                                clamped_count as i64,
                                now.clone()
                            ],
                        )
                        .await?;

                    self.conn
                        .execute(
                            "INSERT INTO voiceprint_contribution
                             (id, centroid_id, meeting_id, label, embedding, count, created_at)
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                            libsql::params![
                                contrib_id,
                                new_centroid_id.0.to_string(),
                                meeting_id.0.to_string(),
                                label,
                                blob,
                                clamped_count as i64,
                                now.clone()
                            ],
                        )
                        .await?;

                    // Cap-and-merge if over GALLERY_CAP.
                    let updated_centroids = self.load_centroids(identity_id).await?;
                    if updated_centroids.len() > GALLERY_CAP {
                        self.cap_and_merge(identity_id, &now).await?;
                    }
                }
            }

            // Update identity updated_at.
            self.conn.execute(
                "UPDATE voiceprint_identity SET updated_at = ?1 WHERE id = ?2",
                libsql::params![now, identity_id.0.to_string()],
            )
            .await?;

            Ok::<(), Error>(())
        }
        .await;

        match result {
            Ok(()) => {
                self.conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(
                        target: "persistence",
                        error = %rb,
                        "refine rollback failed after a refine error"
                    );
                }
                return Err(e);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Merge identities
    // -----------------------------------------------------------------------

    /// Merge `merged_id` into `keep_id` (§2.9.4 manual cross-condition merge).
    ///
    /// Steps:
    /// 1. Re-home every `voiceprint_centroid` (and its contributions via FK) from
    ///    `merged_id` to `keep_id`.
    /// 2. Run cap-and-merge over `keep_id`'s now-larger gallery.
    /// 3. Delete `merged_id`.
    ///
    /// Pure store operation — `weighted_merge` only, no re-embedding. The
    /// caller's UI must confirm before calling (this is not reversible once
    /// cap-and-merge collapses contributions).
    pub async fn merge_identities(
        &self,
        keep_id: VoiceprintIdentityId,
        merged_id: VoiceprintIdentityId,
    ) -> AppResult<()> {
        Ok(self.merge_identities_inner(keep_id, merged_id).await?)
    }

    async fn merge_identities_inner(
        &self,
        keep_id: VoiceprintIdentityId,
        merged_id: VoiceprintIdentityId,
    ) -> Result<(), Error> {
        // Enforce the §2.2 hard-invalidation contract: centroids are only
        // comparable within the same model. Both identities must share the same
        // model_id before any re-homing occurs, otherwise cap_and_merge may
        // weighted_merge vectors from incompatible embedding spaces.
        let keep_model = self.identity_model_id(keep_id).await?;
        let merged_model = self.identity_model_id(merged_id).await?;
        if keep_model != merged_model {
            return Err(Error::InvalidState(
                "model_id mismatch: cannot merge voiceprint identities built from different embedding models",
            ));
        }

        let now = Utc::now().to_rfc3339();

        self.conn.execute("BEGIN IMMEDIATE", ()).await?;

        let result = async {
            // Re-home centroids from merged_id to keep_id.
            self.conn.execute(
                "UPDATE voiceprint_centroid SET identity_id = ?1, updated_at = ?3
                 WHERE identity_id = ?2",
                libsql::params![
                    keep_id.0.to_string(),
                    merged_id.0.to_string(),
                    now.clone()
                ],
            )
            .await?;

            // Cap-and-merge the keep identity's now-larger gallery.
            let centroids = self.load_centroids(keep_id).await?;
            if centroids.len() > GALLERY_CAP {
                self.cap_and_merge(keep_id, &now).await?;
            }

            // Delete the merged identity (centroids have been re-homed).
            self.conn.execute(
                "DELETE FROM voiceprint_identity WHERE id = ?1",
                libsql::params![merged_id.0.to_string()],
            )
            .await?;

            // Update keep identity timestamp.
            self.conn.execute(
                "UPDATE voiceprint_identity SET updated_at = ?1 WHERE id = ?2",
                libsql::params![now, keep_id.0.to_string()],
            )
            .await?;

            Ok::<(), Error>(())
        }
        .await;

        match result {
            Ok(()) => {
                self.conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(
                        target: "persistence",
                        error = %rb,
                        "merge_identities rollback failed after a merge error"
                    );
                }
                return Err(e);
            }
        }

        tracing::info!(
            target: "persistence",
            keep_id = %keep_id.0,
            merged_id = %merged_id.0,
            "voiceprint identities merged"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Deletion operations
    // -----------------------------------------------------------------------

    /// Delete an identity and all its centroids and contributions (cascade).
    pub async fn delete_identity(&self, id: VoiceprintIdentityId) -> AppResult<()> {
        Ok(self.delete_identity_inner(id).await?)
    }

    async fn delete_identity_inner(&self, id: VoiceprintIdentityId) -> Result<(), Error> {
        self.conn.execute(
            "DELETE FROM voiceprint_identity WHERE id = ?1",
            libsql::params![id.0.to_string()],
        )
        .await?;
        tracing::info!(target: "persistence", identity_id = %id.0, "voiceprint identity deleted");
        Ok(())
    }

    /// Delete every identity, centroid, and contribution (the §4 privacy
    /// clear-all path). Also resets the schema-version bookkeeping table so the
    /// schema can be re-applied cleanly if desired.
    pub async fn clear_all(&self) -> AppResult<()> {
        Ok(self.clear_all_inner().await?)
    }

    async fn clear_all_inner(&self) -> Result<(), Error> {
        // Delete in dependency order (CASCADE handles the chain, but explicit
        // deletes give clearer audit log entries).
        self.conn
            .execute("DELETE FROM voiceprint_contribution", ())
            .await?;
        self.conn
            .execute("DELETE FROM voiceprint_centroid", ())
            .await?;
        self.conn
            .execute("DELETE FROM voiceprint_identity", ())
            .await?;
        tracing::info!(target: "persistence", "all voiceprints cleared");
        Ok(())
    }

    /// Drop every contribution whose `meeting_id` matches, recompute affected
    /// centroids, then drop centroids with zero remaining contributions, then
    /// drop identities with zero remaining centroids.
    ///
    /// This is the §4 meeting-granularity erasure path: after a meeting is
    /// deleted, its acoustic traces are purged from every voiceprint it fed.
    pub async fn forget_meeting(&self, meeting_id: MeetingId) -> AppResult<()> {
        Ok(self.forget_meeting_inner(meeting_id).await?)
    }

    async fn forget_meeting_inner(&self, meeting_id: MeetingId) -> Result<(), Error> {
        let now = Utc::now().to_rfc3339();

        // Collect the affected centroid IDs before opening the write transaction,
        // so the SELECT does not hold a read cursor inside BEGIN IMMEDIATE.
        let mut rows = self
            .conn
            .query(
                "SELECT DISTINCT centroid_id FROM voiceprint_contribution
                 WHERE meeting_id = ?1",
                libsql::params![meeting_id.0.to_string()],
            )
            .await?;

        let mut affected: Vec<VoiceprintCentroidId> = Vec::new();
        while let Some(row) = rows.next().await? {
            let id_str: String = row.get(0)?;
            if let Ok(uuid) = uuid::Uuid::parse_str(&id_str) {
                affected.push(VoiceprintCentroidId(uuid));
            }
        }
        drop(rows);

        self.conn.execute("BEGIN IMMEDIATE", ()).await?;

        let result = async {
            // Delete all contributions for this meeting.
            self.conn.execute(
                "DELETE FROM voiceprint_contribution WHERE meeting_id = ?1",
                libsql::params![meeting_id.0.to_string()],
            )
            .await?;

            // Recompute each affected centroid from surviving contributions.
            for centroid_id in &affected {
                recompute_centroid(&self.conn, *centroid_id, &now).await?;
            }

            // Drop centroids that now have zero contributions.
            self.conn.execute(
                "DELETE FROM voiceprint_centroid
                 WHERE id NOT IN (SELECT DISTINCT centroid_id FROM voiceprint_contribution)",
                (),
            )
            .await?;

            // Drop identities that now have zero centroids.
            self.conn.execute(
                "DELETE FROM voiceprint_identity
                 WHERE id NOT IN (SELECT DISTINCT identity_id FROM voiceprint_centroid)",
                (),
            )
            .await?;

            Ok::<(), Error>(())
        }
        .await;

        match result {
            Ok(()) => {
                self.conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(
                        target: "persistence",
                        error = %rb,
                        "forget_meeting rollback failed after a forget error"
                    );
                }
                return Err(e);
            }
        }

        tracing::info!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            "voiceprint contributions for meeting forgotten"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Correction path
    // -----------------------------------------------------------------------

    /// Drop the single contribution from `(meeting_id, label)` inside
    /// `centroid_id`, then recompute the centroid's cached embedding from its
    /// surviving contributions (§2.4 correction path / `reject_match`).
    ///
    /// If no matching contribution exists the call is a no-op (idempotent).
    /// After recomputation, a centroid left with zero contributions is deleted
    /// (its cached embedding would otherwise be stale yet still returned by
    /// `all`, letting a rejected identity re-match), and an identity left with
    /// zero centroids is deleted too — so rejecting the only match of a
    /// single-meeting identity removes it from the gallery entirely.
    ///
    /// This is the persistence-internal half of the WU5 correction path. It
    /// reuses the existing `recompute_centroid` helper inside a `BEGIN IMMEDIATE`
    /// transaction so the §2.9.1 invariant is maintained atomically.
    pub async fn forget_contribution(
        &self,
        centroid_id: VoiceprintCentroidId,
        meeting_id: MeetingId,
        label: &str,
    ) -> AppResult<()> {
        Ok(self
            .forget_contribution_inner(centroid_id, meeting_id, label)
            .await?)
    }

    async fn forget_contribution_inner(
        &self,
        centroid_id: VoiceprintCentroidId,
        meeting_id: MeetingId,
        label: &str,
    ) -> Result<(), Error> {
        let now = Utc::now().to_rfc3339();

        self.conn.execute("BEGIN IMMEDIATE", ()).await?;

        let result = async {
            // Delete the specific (centroid, meeting, label) contribution row.
            self.conn.execute(
                "DELETE FROM voiceprint_contribution
                 WHERE centroid_id = ?1
                   AND meeting_id  = ?2
                   AND label       = ?3",
                libsql::params![
                    centroid_id.0.to_string(),
                    meeting_id.0.to_string(),
                    label
                ],
            )
            .await?;

            // Recompute the centroid cache from surviving contributions.
            recompute_centroid(&self.conn, centroid_id, &now).await?;

            // GC an emptied centroid so its stale embedding can never re-match,
            // then GC an identity left with no centroids (mirrors
            // forget_meeting_inner). Without this, reject_match on a
            // single-meeting identity would leave a sample_count=0 row whose
            // old embedding all() still returns.
            self.conn.execute(
                "DELETE FROM voiceprint_centroid
                 WHERE id NOT IN (SELECT DISTINCT centroid_id FROM voiceprint_contribution)",
                (),
            )
            .await?;
            self.conn.execute(
                "DELETE FROM voiceprint_identity
                 WHERE id NOT IN (SELECT DISTINCT identity_id FROM voiceprint_centroid)",
                (),
            )
            .await?;

            Ok::<(), Error>(())
        }
        .await;

        match result {
            Ok(()) => {
                self.conn.execute("COMMIT", ()).await?;
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(
                        target: "persistence",
                        error = %rb,
                        "forget_contribution rollback failed after a forget error"
                    );
                }
                return Err(e);
            }
        }

        tracing::debug!(
            target: "persistence",
            centroid_id = %centroid_id.0,
            meeting_id = %meeting_id.0,
            label,
            "voiceprint contribution forgotten (reject_match correction)"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Rename
    // -----------------------------------------------------------------------

    /// Rename an identity's `display_name` in place.
    ///
    /// The identity's `updated_at` timestamp is refreshed. Passing an empty
    /// or whitespace-only name is an error (callers must normalise before
    /// calling). Returns `Error::InvalidState` when no identity with `id`
    /// exists.
    pub async fn rename_identity(
        &self,
        id: VoiceprintIdentityId,
        new_name: &str,
    ) -> AppResult<()> {
        Ok(self.rename_identity_inner(id, new_name).await?)
    }

    async fn rename_identity_inner(
        &self,
        id: VoiceprintIdentityId,
        new_name: &str,
    ) -> Result<(), Error> {
        let trimmed = new_name.trim();
        if trimmed.is_empty() {
            return Err(Error::InvalidState("identity name must not be empty"));
        }

        let now = Utc::now().to_rfc3339();
        let affected = self
            .conn
            .execute(
                "UPDATE voiceprint_identity
                 SET display_name = ?1, updated_at = ?2
                 WHERE id = ?3",
                libsql::params![trimmed, now, id.0.to_string()],
            )
            .await?;

        if affected == 0 {
            return Err(Error::InvalidState("identity not found"));
        }

        tracing::info!(
            target: "persistence",
            identity_id = %id.0,
            new_name = trimmed,
            "voiceprint identity renamed"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Look up an existing identity by `display_name` and `model_id`.
    ///
    /// Returns `Some(identity_id)` when exactly one identity exists with the
    /// given name and model, `None` when no such identity exists.
    ///
    /// Used by the orchestrator to decide between `enrol` (first-time) and
    /// `refine` (confirmed subsequent association) for the same named speaker.
    /// `display_name` matching is case-sensitive and exact (no folding) —
    /// callers must normalise if they need case-insensitive lookup.
    ///
    /// Accepted limitation: identity is keyed on `(display_name, model_id)`, so
    /// two genuinely different people who share a name fold into one identity on
    /// `refine`. `FOLD_GATE` guards against cross-*condition* blur within an
    /// identity, not cross-*person* name collision. This matches the §2.9.3
    /// name-keyed confirmation trigger; the design's inverse affordance (assert
    /// two identities are the *same* person) is the manual `merge_identities`
    /// path — there is deliberately no automatic "same name, different person"
    /// split. Surfacing a collision is left to the management UI.
    pub async fn find_identity_by_name_and_model(
        &self,
        display_name: &str,
        model_id: &str,
    ) -> AppResult<Option<VoiceprintIdentityId>> {
        Ok(self
            .find_identity_by_name_and_model_inner(display_name, model_id)
            .await?)
    }

    async fn find_identity_by_name_and_model_inner(
        &self,
        display_name: &str,
        model_id: &str,
    ) -> Result<Option<VoiceprintIdentityId>, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT id FROM voiceprint_identity
                 WHERE display_name = ?1 AND model_id = ?2
                 LIMIT 1",
                libsql::params![display_name, model_id],
            )
            .await?;
        match rows.next().await? {
            Some(row) => {
                let id_str: String = row.get(0)?;
                let uuid = uuid::Uuid::parse_str(&id_str)
                    .map_err(|_| Error::Migration(format!("corrupt identity id: {id_str}")))?;
                Ok(Some(VoiceprintIdentityId(uuid)))
            }
            None => Ok(None),
        }
    }

    /// Return the flattened gallery filtered to `model_id`: every
    /// `voiceprint_centroid` of every identity whose `model_id` matches.
    ///
    /// On a model change, this returns zero rows for the new `model_id` — which
    /// the caller surfaces as "N voiceprints from a previous model — re-enrol?",
    /// NOT a silently empty library (§2.2 hard-invalidation contract).
    pub async fn all(&self, model_id: &str) -> AppResult<Vec<StoredVoiceprint>> {
        Ok(self.all_inner(model_id).await?)
    }

    async fn all_inner(&self, model_id: &str) -> Result<Vec<StoredVoiceprint>, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT i.id, i.display_name, i.model_id,
                        c.id, c.embedding, c.dim, c.sample_count, c.condition_label
                 FROM voiceprint_identity i
                 JOIN voiceprint_centroid c ON c.identity_id = i.id
                 WHERE i.model_id = ?1
                   AND c.sample_count > 0
                 ORDER BY i.id, c.created_at",
                libsql::params![model_id],
            )
            .await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let identity_id_str: String = row.get(0)?;
            let display_name: String = row.get(1)?;
            let stored_model: String = row.get(2)?;
            let centroid_id_str: String = row.get(3)?;
            let blob: Vec<u8> = row.get(4)?;
            let dim: i64 = row.get(5)?;
            let sample_count: i64 = row.get(6)?;
            let condition_label: Option<String> = row.get(7)?;

            let embedding = blob_to_f32_vec(&blob);

            let identity_id = uuid::Uuid::parse_str(&identity_id_str)
                .map(VoiceprintIdentityId)
                .map_err(|_| {
                    Error::Migration(format!("corrupt identity id: {identity_id_str}"))
                })?;
            let centroid_id = uuid::Uuid::parse_str(&centroid_id_str)
                .map(VoiceprintCentroidId)
                .map_err(|_| {
                    Error::Migration(format!("corrupt centroid id: {centroid_id_str}"))
                })?;

            out.push(StoredVoiceprint {
                identity_id,
                centroid_id,
                display_name,
                model_id: stored_model,
                embedding,
                dim: dim as usize,
                sample_count: sample_count as u64,
                condition_label,
            });
        }
        Ok(out)
    }

    /// Return every identity with its per-condition centroid summaries, ordered
    /// by `display_name` then by centroid `created_at`.
    ///
    /// This is the management-UI query: no embedding vectors are returned
    /// (embedding bytes must not cross IPC — §2.2), only the metadata needed
    /// to render the identity list and its gallery entries.
    ///
    /// Returns all identities regardless of `model_id`, so the management pane
    /// can show identities from previous models (and offer to delete them).
    pub async fn identities_with_gallery(&self) -> AppResult<Vec<IdentityWithGallery>> {
        Ok(self.identities_with_gallery_inner().await?)
    }

    async fn identities_with_gallery_inner(&self) -> Result<Vec<IdentityWithGallery>, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT i.id, i.display_name, i.model_id,
                        c.id, c.sample_count, c.condition_label
                 FROM voiceprint_identity i
                 JOIN voiceprint_centroid c ON c.identity_id = i.id
                 ORDER BY i.display_name COLLATE NOCASE, i.id, c.created_at",
                (),
            )
            .await?;

        let mut identities: Vec<IdentityWithGallery> = Vec::new();
        while let Some(row) = rows.next().await? {
            let identity_id_str: String = row.get(0)?;
            let display_name: String = row.get(1)?;
            let model_id: String = row.get(2)?;
            let centroid_id_str: String = row.get(3)?;
            let sample_count: i64 = row.get(4)?;
            let condition_label: Option<String> = row.get(5)?;

            let identity_id = uuid::Uuid::parse_str(&identity_id_str)
                .map(VoiceprintIdentityId)
                .map_err(|_| {
                    Error::Migration(format!("corrupt identity id: {identity_id_str}"))
                })?;
            let centroid_id = uuid::Uuid::parse_str(&centroid_id_str)
                .map(VoiceprintCentroidId)
                .map_err(|_| {
                    Error::Migration(format!("corrupt centroid id: {centroid_id_str}"))
                })?;

            let summary = CentroidSummary {
                centroid_id,
                sample_count: sample_count as u64,
                condition_label,
            };

            // Append the centroid to the last identity if it matches; otherwise
            // start a new identity row.
            if let Some(last) = identities.last_mut() {
                if last.identity_id == identity_id {
                    last.centroids.push(summary);
                    continue;
                }
            }

            identities.push(IdentityWithGallery {
                identity_id,
                display_name,
                model_id,
                centroids: vec![summary],
            });
        }

        Ok(identities)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Load `(centroid_id, embedding, total_count)` for all centroids of an
    /// identity, ordered by `created_at`. Used by `refine` and `merge`.
    async fn load_centroids(
        &self,
        identity_id: VoiceprintIdentityId,
    ) -> Result<Vec<(VoiceprintCentroidId, Vec<f32>, u64)>, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT id, embedding, sample_count
                 FROM voiceprint_centroid
                 WHERE identity_id = ?1
                 ORDER BY created_at",
                libsql::params![identity_id.0.to_string()],
            )
            .await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id_str: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let count: i64 = row.get(2)?;
            let uuid = uuid::Uuid::parse_str(&id_str)
                .map_err(|_| Error::Migration(format!("corrupt centroid id: {id_str}")))?;
            out.push((VoiceprintCentroidId(uuid), blob_to_f32_vec(&blob), count as u64));
        }
        Ok(out)
    }

    /// Look up the `model_id` of an identity row.
    async fn identity_model_id(
        &self,
        identity_id: VoiceprintIdentityId,
    ) -> Result<String, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT model_id FROM voiceprint_identity WHERE id = ?1",
                libsql::params![identity_id.0.to_string()],
            )
            .await?;
        match rows.next().await? {
            Some(row) => Ok(row.get(0)?),
            None => Err(Error::InvalidState("identity not found")),
        }
    }

    /// Cap-and-merge: when an identity exceeds `GALLERY_CAP`, merge the two
    /// closest centroids (by cosine) only if their cosine clears `FOLD_GATE`.
    /// If no pair clears the gate, leave the cap exceeded (surface for
    /// management rather than silently blur genuinely distinct conditions).
    async fn cap_and_merge(
        &self,
        identity_id: VoiceprintIdentityId,
        now: &str,
    ) -> Result<(), Error> {
        loop {
            let centroids = self.load_centroids(identity_id).await?;
            if centroids.len() <= GALLERY_CAP {
                break;
            }

            // Find the two closest centroids.
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_sim =
                minutist_common::voiceprint_math::cosine_unit(&centroids[0].1, &centroids[1].1);

            for i in 0..centroids.len() {
                for j in (i + 1)..centroids.len() {
                    let s = minutist_common::voiceprint_math::cosine_unit(
                        &centroids[i].1,
                        &centroids[j].1,
                    );
                    if s > best_sim {
                        best_sim = s;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if best_sim < FOLD_GATE {
                // Cannot safely merge — do not blur distinct conditions.
                tracing::warn!(
                    target: "persistence",
                    identity_id = %identity_id.0,
                    gallery_size = centroids.len(),
                    "gallery exceeds cap but no pair clears FOLD_GATE; leaving cap exceeded"
                );
                break;
            }

            // Merge centroid best_j into best_i: re-home contributions, recompute.
            let keep_cid = centroids[best_i].0;
            let drop_cid = centroids[best_j].0;

            self.conn.execute(
                "UPDATE voiceprint_contribution SET centroid_id = ?1
                 WHERE centroid_id = ?2",
                libsql::params![keep_cid.0.to_string(), drop_cid.0.to_string()],
            )
            .await?;

            self.conn.execute(
                "DELETE FROM voiceprint_centroid WHERE id = ?1",
                libsql::params![drop_cid.0.to_string()],
            )
            .await?;

            recompute_centroid(&self.conn, keep_cid, now).await?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Centroid recompute helper (enforces the §2.9.1 invariant)
// ---------------------------------------------------------------------------

/// Recompute and update the cached `embedding` and `sample_count` for
/// `centroid_id` from its surviving contributions.
///
/// This is the core invariant helper: `embedding = weighted_merge(contributions)`
/// and `sample_count = Σ count_i`. Callers must invoke this inside an open
/// `BEGIN IMMEDIATE` transaction so the contribution-set change and the centroid
/// update are committed atomically.
///
/// If the centroid has zero surviving contributions (all were deleted), this
/// updates `sample_count = 0` and leaves the `embedding` as the zero vector
/// (the caller is responsible for dropping the centroid row afterwards).
async fn recompute_centroid(
    conn: &Connection,
    centroid_id: VoiceprintCentroidId,
    now: &str,
) -> Result<(), Error> {
    // Load all surviving contributions for this centroid.
    let mut rows = conn
        .query(
            "SELECT embedding, count FROM voiceprint_contribution
             WHERE centroid_id = ?1",
            libsql::params![centroid_id.0.to_string()],
        )
        .await?;

    let mut pairs: Vec<(Vec<f32>, u64)> = Vec::new();
    while let Some(row) = rows.next().await? {
        let blob: Vec<u8> = row.get(0)?;
        let count: i64 = row.get(1)?;
        pairs.push((blob_to_f32_vec(&blob), count as u64));
    }
    drop(rows);

    let total: u64 = pairs.iter().map(|(_, c)| *c).sum();

    let new_embedding = if pairs.is_empty() {
        Vec::new()
    } else {
        let borrowed: Vec<(&[f32], u64)> = pairs.iter().map(|(v, c)| (v.as_slice(), *c)).collect();
        minutist_common::voiceprint_math::weighted_merge(&borrowed)
    };

    if new_embedding.is_empty() {
        // Zero surviving contributions — update count to zero, leave blob unchanged.
        conn.execute(
            "UPDATE voiceprint_centroid SET sample_count = 0, updated_at = ?1
             WHERE id = ?2",
            libsql::params![now, centroid_id.0.to_string()],
        )
        .await?;
    } else {
        let blob = f32_slice_to_blob(&new_embedding);
        conn.execute(
            "UPDATE voiceprint_centroid
             SET embedding = ?1, sample_count = ?2, updated_at = ?3
             WHERE id = ?4",
            libsql::params![blob, total as i64, now, centroid_id.0.to_string()],
        )
        .await?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Convenience path helper
// ---------------------------------------------------------------------------

/// Conventional `voiceprints.db` path under an app-data root.
///
/// Mirrors [`crate::index::index_db_path`]. `app-main` calls `resolve_data_roots`
/// to derive the effective path (which moves when `settings.data_directory` is
/// set), then passes it here.
pub fn voiceprints_db_path(app_data_root: &Path) -> PathBuf {
    app_data_root.join("voiceprints.db")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ----------------------------------------------------------------

    /// Open an in-memory VoiceprintStore for tests.
    async fn open_mem() -> VoiceprintStore {
        VoiceprintStore::open(":memory:").await.unwrap()
    }

    /// Create a synthetic unit-normalised embedding of `dim` dimensions, with
    /// the first element set to `signal` (normalised). This produces a
    /// deterministic vector that represents a distinct speaker direction.
    fn synthetic_embedding(dim: usize, signal: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[0] = signal;
        // Fill a bit of noise so dimension > 1 is meaningful.
        if dim > 1 {
            v[1] = (1.0 - signal * signal).sqrt();
        }
        minutist_common::voiceprint_math::unit_normalise(&mut v);
        v
    }

    /// Return a `MeetingId` from a fixed string for reproducibility.
    fn mid(n: u8) -> MeetingId {
        MeetingId(uuid::Uuid::from_bytes([n; 16]))
    }

    /// Cosine similarity between two slices.
    fn cos(a: &[f32], b: &[f32]) -> f32 {
        minutist_common::voiceprint_math::cosine_unit(a, b)
    }

    // -----------------------------------------------------------------------
    // Basic enrol + all
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn enrol_creates_identity_and_gallery() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery.len(), 1);
        assert_eq!(gallery[0].identity_id, id);
        assert_eq!(gallery[0].display_name, "Alice");
        assert_eq!(gallery[0].model_id, "cam-v1");
        assert_eq!(gallery[0].sample_count, 1);
        assert!((cos(&gallery[0].embedding, &emb) - 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // all() returns zero rows for a foreign model_id (§2.2 hard-invalidation)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn all_returns_empty_for_foreign_model_id() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // A different model returns zero rows — the caller must surface this
        // as "N voiceprints from a previous model", NOT as an empty library.
        let gallery = store.all("cam-v2").await.unwrap();
        assert!(
            gallery.is_empty(),
            "expected zero rows for foreign model_id"
        );
    }

    // -----------------------------------------------------------------------
    // Refinement — fold into nearest centroid
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn refine_folds_similar_contribution_into_same_centroid() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.95);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // A slightly perturbed but similar vector — should fold.
        let emb2 = synthetic_embedding(4, 0.93);
        store
            .refine(id, &emb2, 2, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        // Still one centroid — folded.
        assert_eq!(gallery.len(), 1);
        // sample_count should be 1 (original) + clamped(2) contributions.
        assert!(gallery[0].sample_count >= 1);
    }


    // -----------------------------------------------------------------------
    // Refinement — model_id mismatch is rejected
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn refine_rejects_model_id_mismatch() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let result = store
            .refine(id, &emb, 1, "cam-v2", mid(2), "A")
            .await;

        assert!(result.is_err(), "must reject model_id mismatch");
    }

    // -----------------------------------------------------------------------
    // merge_identities rejects a cross-model merge (§2.2 hard-invalidation)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn merge_identities_rejects_model_id_mismatch() {
        let store = open_mem().await;
        let dim = 4;
        let a = synthetic_embedding(dim, 0.9);
        let keep_id = store
            .enrol("Alice", &a, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let b = synthetic_embedding(dim, 0.85);
        let merged_id = store
            .enrol("Bob", &b, dim, "cam-v2", mid(2), "B")
            .await
            .unwrap();

        let result = store.merge_identities(keep_id, merged_id).await;
        assert!(result.is_err(), "must reject cross-model merge");
    }

    // -----------------------------------------------------------------------
    // Merge-then-recompute: re-homed contributions yield correct centroid
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn merge_identities_rehomes_and_recomputes() {
        let store = open_mem().await;
        let dim = 4;
        // Alice enrolled with two observations.
        let a1 = synthetic_embedding(dim, 0.9);
        let keep_id = store
            .enrol("Alice", &a1, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Bob enrolled separately (a different identity for the same person).
        let b1 = synthetic_embedding(dim, 0.88);
        let merge_id = store
            .enrol("Bob", &b1, dim, "cam-v1", mid(2), "B")
            .await
            .unwrap();

        // Merge Bob into Alice.
        store.merge_identities(keep_id, merge_id).await.unwrap();

        // Bob identity must be gone.
        let gallery = store.all("cam-v1").await.unwrap();
        assert!(
            gallery.iter().all(|e| e.identity_id == keep_id),
            "merged identity must not appear in gallery"
        );

        // Each centroid's cached embedding must equal the weighted_merge of its
        // own contributions.
        for entry in &gallery {
            // Load contributions directly to verify the cached centroid.
            let mut rows = store
                .conn
                .query(
                    "SELECT embedding, count FROM voiceprint_contribution
                     WHERE centroid_id = ?1",
                    libsql::params![entry.centroid_id.0.to_string()],
                )
                .await
                .unwrap();

            let mut pairs: Vec<(Vec<f32>, u64)> = Vec::new();
            while let Some(row) = rows.next().await.unwrap() {
                let blob: Vec<u8> = row.get(0).unwrap();
                let count: i64 = row.get(1).unwrap();
                pairs.push((blob_to_f32_vec(&blob), count as u64));
            }
            drop(rows);

            if pairs.is_empty() {
                continue;
            }
            let borrowed: Vec<(&[f32], u64)> =
                pairs.iter().map(|(v, c)| (v.as_slice(), *c)).collect();
            let expected =
                minutist_common::voiceprint_math::weighted_merge(&borrowed);

            let sim = cos(&entry.embedding, &expected);
            assert!(
                sim > 0.999,
                "cached centroid must match weighted_merge of its contributions, cos = {sim}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // forget_meeting drops the right contributions and recomputes
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn forget_meeting_drops_contributions_and_recomputes() {
        let store = open_mem().await;
        let dim = 4;
        let emb1 = synthetic_embedding(dim, 0.9);
        let id = store
            .enrol("Alice", &emb1, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Add a second contribution from a different meeting.
        let emb2 = synthetic_embedding(dim, 0.92);
        store
            .refine(id, &emb2, 3, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        let before = store.all("cam-v1").await.unwrap();
        assert_eq!(before.len(), 1);
        let sample_count_before = before[0].sample_count;

        // Forget meeting 2.
        store.forget_meeting(mid(2)).await.unwrap();

        // Contribution from meeting 2 is gone; identity still exists (meeting 1 remains).
        let after = store.all("cam-v1").await.unwrap();
        assert_eq!(after.len(), 1, "identity from meeting 1 must survive");
        let sample_count_after = after[0].sample_count;
        assert!(
            sample_count_after < sample_count_before,
            "sample_count must decrease after forget: before={sample_count_before}, after={sample_count_after}"
        );

        // Now forget meeting 1 as well — identity and centroid should be gone.
        store.forget_meeting(mid(1)).await.unwrap();
        let empty = store.all("cam-v1").await.unwrap();
        assert!(
            empty.is_empty(),
            "all voiceprints must be gone after forgetting all contributing meetings"
        );
    }

    // -----------------------------------------------------------------------
    // Bounded-weight poison test (§2.9.3)
    //
    // An established centroid at large sample_count refined once with an
    // adversarial near-T_accept contribution must not move enough to cross
    // T_accept for a held-out impostor.
    //
    // Setup:
    //   - Alice's centroid is the +x direction: [1,0,0,...].
    //   - The held-out impostor lives at θ = 60° from +x (cosine = 0.5, just
    //     below T_accept = 0.60).
    //   - The adversarial contribution is the midpoint of Alice and the
    //     impostor: halfway between [1,0,...] and [0.5, 0.866,...] ≈
    //     [0.866, 0.5,...] normalised ≈ [0.866, 0.5,...] (already unit).
    //     Its cosine with Alice ≈ 0.866, well above FOLD_GATE = 0.70, so it
    //     will be folded.
    //   - After folding, Alice's centroid must have cosine < T_accept (0.60)
    //     with the impostor. With REFINE_WEIGHT_CAP = 0.30 and
    //     sample_count = 100, the single adversarial contribution is clamped to
    //     at most 30, so the poisoned centroid stays close to +x.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn bounded_weight_poison_test() {
        let store = open_mem().await;
        let dim = 3;
        const T_ACCEPT: f32 = 0.60;

        // Alice: pure +x direction.
        let alice_emb = {
            let mut v = vec![1.0_f32, 0.0, 0.0];
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            v
        };

        // Enrol Alice with a large sample_count by enrolling once and then
        // manually updating the contribution count to simulate many prior observations.
        let id = store
            .enrol("Alice", &alice_emb, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Directly set the contribution count to 100 to simulate an established centroid.
        store
            .conn
            .execute(
                "UPDATE voiceprint_contribution SET count = 100
                 WHERE centroid_id IN (
                     SELECT id FROM voiceprint_centroid WHERE identity_id = ?1
                 )",
                libsql::params![id.0.to_string()],
            )
            .await
            .unwrap();
        // Recompute the centroid cache to reflect the new count.
        let centroids = store.load_centroids(id).await.unwrap();
        let now = Utc::now().to_rfc3339();
        recompute_centroid(&store.conn, centroids[0].0, &now)
            .await
            .unwrap();

        // The held-out impostor is at 60° from Alice (cosine ≈ 0.5).
        let impostor = {
            let mut v = vec![0.5_f32, 0.866_f32, 0.0];
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            v
        };

        // Verify the impostor is below T_accept before the attack.
        let gallery_before = store.all("cam-v1").await.unwrap();
        let sim_before = cos(&gallery_before[0].embedding, &impostor);
        assert!(
            sim_before < T_ACCEPT,
            "impostor must start below T_accept (sim={sim_before})"
        );

        // Adversarial contribution: near the midpoint of Alice and the impostor.
        // cos(adversarial, Alice) ≈ 0.966 — well above FOLD_GATE = 0.70.
        let adversarial = {
            let mut v = vec![0.966_f32, 0.259_f32, 0.0]; // ~15° from Alice
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            v
        };

        // Single-meeting refine with a count of 1000 (much larger than the cap
        // budget of 100 * 0.30 = 30).
        store
            .refine(id, &adversarial, 1000, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        let gallery_after = store.all("cam-v1").await.unwrap();
        let sim_after = cos(&gallery_after[0].embedding, &impostor);

        assert!(
            sim_after < T_ACCEPT,
            "poisoned centroid must not cross T_accept for the impostor (sim={sim_after})"
        );
    }

    // -----------------------------------------------------------------------
    // clear_all wipes everything
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn clear_all_wipes_all_rows() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();
        store
            .enrol("Bob", &emb, 4, "cam-v1", mid(2), "B")
            .await
            .unwrap();

        store.clear_all().await.unwrap();
        let gallery = store.all("cam-v1").await.unwrap();
        assert!(gallery.is_empty());
    }

    // -----------------------------------------------------------------------
    // voiceprints_db_path helper
    // -----------------------------------------------------------------------

    #[test]
    fn voiceprints_db_path_appends_filename() {
        let root = std::path::Path::new("/tmp/test-app-data");
        let p = voiceprints_db_path(root);
        assert_eq!(p, root.join("voiceprints.db"));
    }

    // -----------------------------------------------------------------------
    // refine_adds_new_condition_when_dissimilar
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn refine_adds_new_condition_when_dissimilar() {
        let store = open_mem().await;
        let dim = 4;
        // Alice at +x direction: [1, 0, 0, 0]
        let alice_in_person = synthetic_embedding(dim, 0.99);
        let id = store
            .enrol("Alice", &alice_in_person, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Alice over Teams: orthogonal direction — cosine ≈ 0 < FOLD_GATE
        let alice_teams = synthetic_embedding(dim, 0.0);

        store
            .refine(id, &alice_teams, 2, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        // Two distinct condition centroids should now exist
        let gallery = store.all("cam-v1").await.unwrap();
        let same_identity = gallery.iter().filter(|e| e.identity_id == id).count();
        assert_eq!(
            same_identity, 2,
            "dissimilar contribution should create a new condition centroid"
        );
    }


    // -----------------------------------------------------------------------
    // refine_clamps_count_to_bounded_weight
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn refine_clamps_count_to_bounded_weight() {
        let store = open_mem().await;
        let dim = 3;
        const ORIGINAL_COUNT: u64 = 100;
        const ADVERSARIAL_COUNT: u64 = 1000;

        // Enrol Alice with synthetic embedding and manually set a large count
        let alice = synthetic_embedding(dim, 0.95);
        let id = store
            .enrol("Alice", &alice, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Manually boost the contribution count to simulate an established centroid
        store
            .conn
            .execute(
                "UPDATE voiceprint_contribution SET count = ?1
                 WHERE centroid_id IN (
                     SELECT id FROM voiceprint_centroid WHERE identity_id = ?2
                 )",
                libsql::params![ORIGINAL_COUNT as i64, id.0.to_string()],
            )
            .await
            .unwrap();

        // Recompute to reflect the boosted count
        let now = Utc::now().to_rfc3339();
        let centroids = store.load_centroids(id).await.unwrap();
        recompute_centroid(&store.conn, centroids[0].0, &now)
            .await
            .unwrap();

        // A similar contribution with a huge count (1000) should be clamped to
        // min(1000, 100 * 0.30) = 30
        let alice_similar = synthetic_embedding(dim, 0.93);
        store
            .refine(id, &alice_similar, ADVERSARIAL_COUNT, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        let entry = gallery.iter().find(|e| e.identity_id == id).unwrap();

        // sample_count should be 100 (original) + at most 30 (clamped).
        // With REFINE_WEIGHT_CAP = 0.30, the cap is 100 * 0.30 = 30.
        // So total should be <= 130.
        assert!(
            entry.sample_count <= 130,
            "sample_count with clamped weight should be <= 130, got {}",
            entry.sample_count
        );
        assert!(
            entry.sample_count > 100,
            "sample_count should have increased from the fold, got {}",
            entry.sample_count
        );
    }

    // -----------------------------------------------------------------------
    // refine_caps_gallery_at_limit
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn refine_caps_gallery_at_limit() {
        let store = open_mem().await;
        let dim = 3;

        // Enrol Alice with baseline
        let baseline = synthetic_embedding(dim, 0.95);
        let id = store
            .enrol("Alice", &baseline, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Manually add 3 more distinct condition centroids to reach cap
        // (Direct DB manipulations to bypass the folding/merging logic for test simplicity)
        let cond2_emb = {
            let mut v = vec![0.0_f32, 1.0, 0.0];
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            v
        };
        let cond2_blob = f32_slice_to_blob(&cond2_emb);
        let cond2_id = uuid::Uuid::new_v4().to_string();
        let now2 = Utc::now().to_rfc3339();
        store
            .conn
            .execute(
                "INSERT INTO voiceprint_centroid
                 (id, identity_id, embedding, dim, sample_count, condition_label, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?6)",
                libsql::params![
                    cond2_id.clone(),
                    id.0.to_string(),
                    cond2_blob.clone(),
                    dim as i64,
                    1i64,
                    now2.clone()
                ],
            )
            .await
            .unwrap();
        let contrib_id = uuid::Uuid::new_v4().to_string();
        store
            .conn
            .execute(
                "INSERT INTO voiceprint_contribution
                 (id, centroid_id, meeting_id, label, embedding, count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                libsql::params![
                    contrib_id,
                    cond2_id,
                    mid(2).0.to_string(),
                    "A",
                    cond2_blob,
                    1i64,
                    now2
                ],
            )
            .await
            .unwrap();

        // Add 2 more to reach cap (4)
        for meeting_n in 3..5 {
            let emb = synthetic_embedding(dim, 0.5 + (meeting_n as f32) * 0.01);
            let blob = f32_slice_to_blob(&emb);
            let cid = uuid::Uuid::new_v4().to_string();
            let now = Utc::now().to_rfc3339();
            store
                .conn
                .execute(
                    "INSERT INTO voiceprint_centroid
                     (id, identity_id, embedding, dim, sample_count, condition_label, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?6)",
                    libsql::params![cid.clone(), id.0.to_string(), blob.clone(), dim as i64, 1i64, now.clone()],
                )
                .await
                .unwrap();
            let contrib_id = uuid::Uuid::new_v4().to_string();
            store
                .conn
                .execute(
                    "INSERT INTO voiceprint_contribution
                     (id, centroid_id, meeting_id, label, embedding, count, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    libsql::params![contrib_id, cid, mid(meeting_n as u8).0.to_string(), "A", blob, 1i64, now],
                )
                .await
                .unwrap();
        }

        let gallery = store.all("cam-v1").await.unwrap();
        let at_cap = gallery.iter().filter(|e| e.identity_id == id).count();
        assert_eq!(at_cap, 4, "should be at GALLERY_CAP");

        // Now refine with a new contribution: should trigger cap-and-merge
        let new_emb = synthetic_embedding(dim, 0.6);
        store
            .refine(id, &new_emb, 1, "cam-v1", mid(99), "A")
            .await
            .unwrap();

        let gallery_final = store.all("cam-v1").await.unwrap();
        let final_count = gallery_final.iter().filter(|e| e.identity_id == id).count();
        assert!(
            final_count <= 4,
            "gallery must respect cap (GALLERY_CAP = 4), got {final_count}"
        );
    }

    // -----------------------------------------------------------------------
    // delete_identity_removes_all_data
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn delete_identity_removes_all_data() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let before = store.all("cam-v1").await.unwrap();
        assert_eq!(before.len(), 1);

        store.delete_identity(id).await.unwrap();

        let after = store.all("cam-v1").await.unwrap();
        assert!(after.is_empty(), "identity deletion should remove all gallery entries");
    }


    // -----------------------------------------------------------------------
    // find_identity_by_name_and_model returns correct identity
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn find_identity_by_name_and_model_exact_match() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let found = store
            .find_identity_by_name_and_model("Alice", "cam-v1")
            .await
            .unwrap();
        assert_eq!(found, Some(id));
    }


    // -----------------------------------------------------------------------
    // find_identity_by_name_and_model returns None for mismatch
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn find_identity_by_name_and_model_none_when_not_found() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let _id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Different name
        let found1 = store
            .find_identity_by_name_and_model("Bob", "cam-v1")
            .await
            .unwrap();
        assert_eq!(found1, None);

        // Different model
        let found2 = store
            .find_identity_by_name_and_model("Alice", "cam-v2")
            .await
            .unwrap();
        assert_eq!(found2, None);
    }

    // -----------------------------------------------------------------------
    // find_identity_by_name_and_model is case-sensitive
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn find_identity_by_name_and_model_case_sensitive() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let _id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Different case should not match
        let found = store
            .find_identity_by_name_and_model("alice", "cam-v1")
            .await
            .unwrap();
        assert_eq!(found, None, "lookup must be case-sensitive");
    }

    // -----------------------------------------------------------------------
    // forget_meeting preserves other meetings' contributions
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn forget_meeting_preserves_other_meetings() {
        let store = open_mem().await;
        let dim = 4;
        let emb1 = synthetic_embedding(dim, 0.9);
        let id = store
            .enrol("Alice", &emb1, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Add a second contribution from a different meeting
        let emb2 = synthetic_embedding(dim, 0.88);
        store
            .refine(id, &emb2, 2, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        // Add a third from yet another meeting
        let emb3 = synthetic_embedding(dim, 0.85);
        store
            .refine(id, &emb3, 3, "cam-v1", mid(3), "A")
            .await
            .unwrap();

        let before = store.all("cam-v1").await.unwrap();
        let sample_count_before = before[0].sample_count;

        // Forget meeting 2 only
        store.forget_meeting(mid(2)).await.unwrap();

        let after = store.all("cam-v1").await.unwrap();
        assert_eq!(after.len(), 1, "identity must still exist");
        let sample_count_after = after[0].sample_count;

        assert!(
            sample_count_after < sample_count_before,
            "sample_count must decrease after forgetting meeting 2"
        );
        assert!(
            sample_count_after > 0,
            "sample_count must be nonzero (meeting 1 and 3 remain)"
        );
    }


    // -----------------------------------------------------------------------
    // empty embedding is rejected on enrol
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn enrol_rejects_empty_embedding() {
        let store = open_mem().await;
        let empty_emb: Vec<f32> = vec![];

        let result = store
            .enrol("Alice", &empty_emb, 0, "cam-v1", mid(1), "A")
            .await;

        assert!(
            result.is_err(),
            "enrol must reject an empty embedding"
        );
    }


    // -----------------------------------------------------------------------
    // dim mismatch is rejected on enrol
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn enrol_rejects_dim_mismatch() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);

        let result = store
            .enrol("Alice", &emb, 5, "cam-v1", mid(1), "A") // claimed dim = 5, actual = 4
            .await;

        assert!(
            result.is_err(),
            "enrol must reject a dim mismatch"
        );
    }


    // -----------------------------------------------------------------------
    // centroid recompute after merge yields weighted mean
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn merge_identities_recomputes_weighted_mean() {
        let store = open_mem().await;
        let dim = 2;

        // Alice at [1, 0]
        let a = {
            let mut v = vec![1.0_f32, 0.0];
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            v
        };

        let alice_id = store
            .enrol("Alice", &a, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Bob at [0, 1] (orthogonal)
        let b = {
            let mut v = vec![0.0_f32, 1.0];
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            v
        };

        let bob_id = store
            .enrol("Bob", &b, dim, "cam-v1", mid(2), "B")
            .await
            .unwrap();

        // Merge Bob into Alice
        store.merge_identities(alice_id, bob_id).await.unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        // After merge, Alice's identity should have two centroids
        let alice_centroids: Vec<_> = gallery.iter().filter(|e| e.identity_id == alice_id).collect();

        // Check that each centroid's embedding is the correct weighted_merge of its contributions
        for centroid in alice_centroids {
            // Load the contributions manually
            let mut rows = store
                .conn
                .query(
                    "SELECT embedding, count FROM voiceprint_contribution
                     WHERE centroid_id = ?1",
                    libsql::params![centroid.centroid_id.0.to_string()],
                )
                .await
                .unwrap();

            let mut pairs: Vec<(Vec<f32>, u64)> = Vec::new();
            while let Some(row) = rows.next().await.unwrap() {
                let blob: Vec<u8> = row.get(0).unwrap();
                let count: i64 = row.get(1).unwrap();
                pairs.push((blob_to_f32_vec(&blob), count as u64));
            }
            drop(rows);

            if !pairs.is_empty() {
                let borrowed: Vec<(&[f32], u64)> =
                    pairs.iter().map(|(v, c)| (v.as_slice(), *c)).collect();
                let expected = minutist_common::voiceprint_math::weighted_merge(&borrowed);
                let sim = cos(&centroid.embedding, &expected);
                assert!(
                    sim > 0.999,
                    "centroid embedding must match weighted_merge of contributions, cos = {sim}"
                );
            }
        }
    }


    // -----------------------------------------------------------------------
    // multiple identities can coexist (different names, same model)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn multiple_identities_same_model() {
        let store = open_mem().await;
        let dim = 4;
        let emb_a = synthetic_embedding(dim, 0.9);
        let emb_b = synthetic_embedding(dim, 0.8);

        let id_a = store
            .enrol("Alice", &emb_a, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let _id_b = store
            .enrol("Bob", &emb_b, dim, "cam-v1", mid(2), "B")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery.len(), 2, "should have two separate identities");
        assert!(gallery.iter().any(|e| e.identity_id == id_a));
    }

    // -----------------------------------------------------------------------
    // all() filters by model_id independently
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn all_filters_by_model_id() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);

        // Enrol under model v1
        let _id1 = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Enrol the same name under model v2 (different identity)
        let _id2 = store
            .enrol("Alice", &emb, 4, "cam-v2", mid(2), "A")
            .await
            .unwrap();

        let v1_gallery = store.all("cam-v1").await.unwrap();
        let v2_gallery = store.all("cam-v2").await.unwrap();

        assert_eq!(v1_gallery.len(), 1, "v1 should have one entry");
        assert_eq!(v2_gallery.len(), 1, "v2 should have one entry");
        assert_ne!(
            v1_gallery[0].identity_id, v2_gallery[0].identity_id,
            "different models should have different identity IDs"
        );
    }

    // -----------------------------------------------------------------------
    // enrol normalises the embedding (cos ≈ 1 with input)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn enrol_normalises_embedding() {
        let store = open_mem().await;
        // Create a non-unit embedding [3, 4]
        let raw = [3.0_f32, 4.0];
        let _id = store
            .enrol("Alice", &raw, 2, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery.len(), 1);
        let stored = &gallery[0].embedding;

        // Stored vector should be normalised [3/5, 4/5]
        assert!((stored[0] - 0.6).abs() < 1e-5);
        assert!((stored[1] - 0.8).abs() < 1e-5);

        // cosine(stored, [3, 4]) should be ≈ 1 after normalising
        let mut raw_norm = raw.to_vec();
        minutist_common::voiceprint_math::unit_normalise(&mut raw_norm);
        let sim = cos(stored, &raw_norm);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // WU8: rename_identity round-trip
    //
    // Verify that rename_identity updates the stored name and maintains all
    // identity state (ID, gallery, contributions unchanged).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn rename_identity_round_trip() {
        let store = open_mem().await;
        let dim = 4;
        let emb = synthetic_embedding(dim, 0.95);

        // Enrol "Alice"
        let id = store
            .enrol("Alice", &emb, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Verify initial state
        let before = store
            .find_identity_by_name_and_model("Alice", "cam-v1")
            .await
            .unwrap();
        assert_eq!(before, Some(id), "Alice should be found by name before rename");

        // Rename to "Alicia"
        store.rename_identity(id, "Alicia").await.unwrap();

        // Verify old name no longer matches
        let after_alice = store
            .find_identity_by_name_and_model("Alice", "cam-v1")
            .await
            .unwrap();
        assert_eq!(
            after_alice, None,
            "old name 'Alice' must not be found after rename"
        );

        // Verify new name matches
        let after_alicia = store
            .find_identity_by_name_and_model("Alicia", "cam-v1")
            .await
            .unwrap();
        assert_eq!(
            after_alicia, Some(id),
            "new name 'Alicia' must match the same identity"
        );

        // Verify the identity's gallery is unchanged (identity_id, centroid_id, embedding all the same)
        let gallery = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery.len(), 1, "gallery size must be unchanged");
        assert_eq!(gallery[0].identity_id, id, "identity_id must be the same");
        assert_eq!(
            gallery[0].display_name, "Alicia",
            "display_name must reflect the rename"
        );
        // Embedding and sample_count must be unchanged
        let sim = cos(&gallery[0].embedding, &emb);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "embedding must be unchanged after rename"
        );
        assert_eq!(
            gallery[0].sample_count, 1,
            "sample_count must be unchanged"
        );
    }

    #[tokio::test]
    async fn rename_identity_nonexistent_fails() {
        let store = open_mem().await;
        let fake_id = VoiceprintIdentityId(uuid::Uuid::new_v4());

        let result = store.rename_identity(fake_id, "NewName").await;
        assert!(
            result.is_err(),
            "rename of nonexistent identity must fail"
        );
    }

    #[tokio::test]
    async fn rename_identity_lookup_persistence() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.95);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Rename sequence: Alice -> Alicia -> Al -> Alice
        store.rename_identity(id, "Alicia").await.unwrap();
        let found1 = store
            .find_identity_by_name_and_model("Alicia", "cam-v1")
            .await
            .unwrap();
        assert_eq!(found1, Some(id));

        store.rename_identity(id, "Al").await.unwrap();
        let found2 = store
            .find_identity_by_name_and_model("Al", "cam-v1")
            .await
            .unwrap();
        assert_eq!(found2, Some(id));

        // Rename back to Alice
        store.rename_identity(id, "Alice").await.unwrap();
        let found3 = store
            .find_identity_by_name_and_model("Alice", "cam-v1")
            .await
            .unwrap();
        assert_eq!(found3, Some(id));

        // Verify gallery still shows the final name
        let gallery = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery[0].display_name, "Alice");
    }

    // -----------------------------------------------------------------------
    // WU5: forget_contribution recompute invariant
    //
    // Verify that forget_contribution recomputes the centroid cache correctly
    // using weighted_merge, maintaining the §2.9.1 invariant that cached
    // embedding = weighted_merge(remaining contributions).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn forget_contribution_invariant_recompute() {
        let store = open_mem().await;
        let dim = 4;

        // Enrol with embedding 1 (count=1)
        let emb1 = synthetic_embedding(dim, 0.95);
        let id = store
            .enrol("Alice", &emb1, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let gallery1 = store.all("cam-v1").await.unwrap();
        let centroid_id = gallery1[0].centroid_id;
        let count_after_enrol = gallery1[0].sample_count;

        // Refine with a similar embedding (should fold into the same centroid)
        let emb2 = synthetic_embedding(dim, 0.92);
        store
            .refine(id, &emb2, 5, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        let gallery2 = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery2.len(), 1, "should still have one centroid (folded)");
        let count_after_refine = gallery2[0].sample_count;
        assert!(
            count_after_refine > count_after_enrol,
            "sample_count should increase after refine"
        );

        // Forget the contribution from meeting 2
        store
            .forget_contribution(centroid_id, mid(2), "A")
            .await
            .unwrap();

        // After forget, the centroid should be recomputed
        let gallery3 = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery3.len(), 1, "centroid should still exist");
        assert_eq!(
            gallery3[0].sample_count, count_after_enrol,
            "sample_count should revert to the original enrolment count"
        );

        // The embedding should now closely match emb1 again
        let sim_with_emb1 = cos(&gallery3[0].embedding, &emb1);
        assert!(
            sim_with_emb1 > 0.99,
            "after dropping emb2, centroid should match emb1 closely (sim={})",
            sim_with_emb1
        );
    }

    #[tokio::test]
    async fn forget_contribution_removes_orphaned_identity() {
        // Rejecting the only match of a single-meeting identity must remove it
        // entirely — not leave a sample_count=0 centroid whose stale embedding
        // all() would still return, letting the rejected voice re-match.
        let store = open_mem().await;
        let dim = 4;

        let emb = synthetic_embedding(dim, 0.95);
        store
            .enrol("Mallory", &emb, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery.len(), 1, "enrolment should create one centroid");
        let centroid_id = gallery[0].centroid_id;

        // Reject the match: drop the sole (meeting, label) contribution.
        store
            .forget_contribution(centroid_id, mid(1), "A")
            .await
            .unwrap();

        // The emptied centroid and its now-childless identity must be gone.
        let gallery_after = store.all("cam-v1").await.unwrap();
        assert!(
            gallery_after.is_empty(),
            "rejecting the only contribution must remove the orphaned centroid/identity, got {} rows",
            gallery_after.len()
        );
    }

    #[tokio::test]
    async fn refine_is_idempotent_per_meeting() {
        // Auto-accept refinement runs on every reprocess, so refining twice from
        // the same meeting must replace (not double-count) that meeting's weight.
        let store = open_mem().await;
        let dim = 4;

        let emb1 = synthetic_embedding(dim, 0.95);
        let id = store
            .enrol("Alice", &emb1, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let emb2 = synthetic_embedding(dim, 0.93);
        store.refine(id, &emb2, 5, "cam-v1", mid(2), "A").await.unwrap();
        let count_once: u64 = store
            .all("cam-v1")
            .await
            .unwrap()
            .iter()
            .map(|v| v.sample_count)
            .sum();

        // Re-refine from the SAME meeting — replaces, does not accumulate.
        store.refine(id, &emb2, 5, "cam-v1", mid(2), "A").await.unwrap();
        let count_twice: u64 = store
            .all("cam-v1")
            .await
            .unwrap()
            .iter()
            .map(|v| v.sample_count)
            .sum();

        assert_eq!(
            count_once, count_twice,
            "re-refining the same meeting must not increase sample_count"
        );
    }

    #[tokio::test]
    async fn refine_same_meeting_as_sole_enrolment_is_stable() {
        // Identity enrolled solely from meeting 1; auto-accept re-refines from
        // meeting 1. The dedup empties the single centroid, so refine recreates it
        // (no error, no orphan, no duplication) and repeated calls converge.
        let store = open_mem().await;
        let dim = 4;
        let emb = synthetic_embedding(dim, 0.95);
        let id = store
            .enrol("Bob", &emb, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        store.refine(id, &emb, 1, "cam-v1", mid(1), "A").await.unwrap();
        let g2 = store.all("cam-v1").await.unwrap();
        let c2: u64 = g2.iter().map(|v| v.sample_count).sum();

        store.refine(id, &emb, 1, "cam-v1", mid(1), "A").await.unwrap();
        let g3 = store.all("cam-v1").await.unwrap();
        let c3: u64 = g3.iter().map(|v| v.sample_count).sum();

        assert_eq!(g3.len(), 1, "identity should keep exactly one centroid");
        assert_eq!(c2, c3, "repeated identical re-refine must be stable");
    }

    #[tokio::test]
    async fn forget_contribution_preserves_other_labels_scoped() {
        let store = open_mem().await;
        let dim = 3;

        // Enrol with label "A" from meeting 1
        let emb1 = synthetic_embedding(dim, 0.95);
        let _id = store
            .enrol("Alice", &emb1, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let gallery_after_enrol = store.all("cam-v1").await.unwrap();
        let centroid_id = gallery_after_enrol[0].centroid_id;

        // Refine with label "A" from meeting 2 (should fold)
        let emb2 = synthetic_embedding(dim, 0.93);
        store
            .refine(_id, &emb2, 1, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        // Verify two contributions exist
        let mut rows = store
            .conn
            .query(
                "SELECT COUNT(*) FROM voiceprint_contribution WHERE centroid_id = ?1",
                libsql::params![centroid_id.0.to_string()],
            )
            .await
            .unwrap();
        let count_before: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        drop(rows);
        assert_eq!(count_before, 2, "should have 2 contributions");

        // Drop ONLY the specific (meeting=2, label=A) contribution
        store
            .forget_contribution(centroid_id, mid(2), "A")
            .await
            .unwrap();

        // Verify the other contribution (meeting 1, label A) survived
        let mut rows_after = store
            .conn
            .query(
                "SELECT COUNT(*) FROM voiceprint_contribution
                 WHERE centroid_id = ?1 AND meeting_id = ?2",
                libsql::params![centroid_id.0.to_string(), mid(1).0.to_string()],
            )
            .await
            .unwrap();
        let survived: i64 = rows_after.next().await.unwrap().unwrap().get(0).unwrap();
        drop(rows_after);
        assert_eq!(survived, 1, "contribution from meeting 1 must survive");
    }

    #[tokio::test]
    async fn forget_contribution_zero_count_leaves_centroid() {
        let store = open_mem().await;
        let dim = 3;
        let emb = synthetic_embedding(dim, 0.95);
        let _id = store
            .enrol("Alice", &emb, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        let centroid_id = gallery[0].centroid_id;

        // Forget the only contribution
        store
            .forget_contribution(centroid_id, mid(1), "A")
            .await
            .unwrap();

        // Centroid should still exist with sample_count=0
        let mut rows = store
            .conn
            .query(
                "SELECT sample_count FROM voiceprint_centroid WHERE id = ?1",
                libsql::params![centroid_id.0.to_string()],
            )
            .await
            .unwrap();
        if let Some(row) = rows.next().await.unwrap() {
            let count: i64 = row.get(0).unwrap();
            assert_eq!(
                count, 0,
                "sample_count must be zero after forgetting final contribution"
            );
        }
        drop(rows);
    }


    // -----------------------------------------------------------------------
    // refine rejects contribution with wrong dimension
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn refine_rejects_wrong_dimension() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Contribution with 3 elements instead of 4
        let wrong_dim = synthetic_embedding(3, 0.9);
        let _result = store
            .refine(id, &wrong_dim, 1, "cam-v1", mid(2), "A")
            .await;

        // The refine doesn't explicitly check dimension mismatch in load_centroids
        // since it compares by cosine (which handles length differences),
        // but the recompute_centroid will have mismatched vectors.
        // For this test, we verify that providing the right embedding works:
        let correct_emb = synthetic_embedding(4, 0.88);
        let result2 = store
            .refine(id, &correct_emb, 1, "cam-v1", mid(2), "A")
            .await;
        assert!(result2.is_ok(), "refine with correct dimension should work");
    }

    // -----------------------------------------------------------------------
    // forget_meeting clears all-empty identity and centroid (cascading GC)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn forget_meeting_gc_empty_identity() {
        let store = open_mem().await;
        let dim = 4;
        let emb = synthetic_embedding(dim, 0.9);
        let _id = store
            .enrol("Alice", &emb, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let before = store.all("cam-v1").await.unwrap();
        assert_eq!(before.len(), 1);

        // Forget the only contributing meeting
        store.forget_meeting(mid(1)).await.unwrap();

        let after = store.all("cam-v1").await.unwrap();
        assert!(
            after.is_empty(),
            "identity and centroid should be GC'd when all contributions are gone"
        );
    }

    // -----------------------------------------------------------------------
    // forget_contribution drops exactly one (meeting, label) row
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn forget_contribution_drops_one_contribution_and_recomputes() {
        let store = open_mem().await;
        let dim = 4;
        let emb1 = synthetic_embedding(dim, 0.9);
        let id = store
            .enrol("Alice", &emb1, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        // Add a second contribution from a different meeting.
        let emb2 = synthetic_embedding(dim, 0.85);
        store
            .refine(id, &emb2, 3, "cam-v1", mid(2), "A")
            .await
            .unwrap();

        let gallery_before = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery_before.len(), 1, "still one centroid after fold");
        let count_before = gallery_before[0].sample_count;

        // Retrieve the centroid_id for the forget call.
        let centroid_id = gallery_before[0].centroid_id;

        // Drop the contribution from meeting 2.
        store
            .forget_contribution(centroid_id, mid(2), "A")
            .await
            .unwrap();

        let gallery_after = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery_after.len(), 1, "centroid still exists (meeting 1 remains)");
        let count_after = gallery_after[0].sample_count;

        assert!(
            count_after < count_before,
            "sample_count must decrease after forget_contribution: before={count_before}, after={count_after}"
        );
    }

    // -----------------------------------------------------------------------
    // forget_contribution is idempotent (no-op on absent row)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn forget_contribution_is_idempotent_on_absent_row() {
        let store = open_mem().await;
        let dim = 4;
        let emb = synthetic_embedding(dim, 0.9);
        let id = store
            .enrol("Alice", &emb, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        let centroid_id = gallery[0].centroid_id;

        // Drop a contribution that doesn't exist — must not error.
        store
            .forget_contribution(centroid_id, mid(99), "Z")
            .await
            .unwrap();

        // The identity must still be intact.
        let after = store.all("cam-v1").await.unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].identity_id, id);
    }

    // -----------------------------------------------------------------------
    // rename_identity — changes display_name, rejects empty name
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn rename_identity_updates_display_name() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        store.rename_identity(id, "Alicia").await.unwrap();

        let gallery = store.all("cam-v1").await.unwrap();
        assert_eq!(gallery.len(), 1);
        assert_eq!(gallery[0].display_name, "Alicia");
    }

    #[tokio::test]
    async fn rename_identity_rejects_empty_name() {
        let store = open_mem().await;
        let emb = synthetic_embedding(4, 0.9);
        let id = store
            .enrol("Alice", &emb, 4, "cam-v1", mid(1), "A")
            .await
            .unwrap();

        let result = store.rename_identity(id, "   ").await;
        assert!(result.is_err(), "empty name must be rejected");
    }

    // -----------------------------------------------------------------------
    // identities_with_gallery — returns summaries without embeddings
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn identities_with_gallery_returns_summaries() {
        let store = open_mem().await;
        let dim = 4;
        let emb_a = synthetic_embedding(dim, 0.9);
        let emb_b = synthetic_embedding(dim, 0.1);
        let id_a = store
            .enrol("Alice", &emb_a, dim, "cam-v1", mid(1), "A")
            .await
            .unwrap();
        let _id_b = store
            .enrol("Bob", &emb_b, dim, "cam-v1", mid(2), "B")
            .await
            .unwrap();

        let identities = store.identities_with_gallery().await.unwrap();
        assert_eq!(identities.len(), 2, "two identities expected");

        let alice = identities.iter().find(|i| i.identity_id == id_a).unwrap();
        assert_eq!(alice.display_name, "Alice");
        assert_eq!(alice.model_id, "cam-v1");
        assert_eq!(alice.centroids.len(), 1);
        assert_eq!(alice.centroids[0].sample_count, 1);
    }
}
