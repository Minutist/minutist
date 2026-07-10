//! Host election for capture-but-unprocessed meetings — the producer-gate
//! ([`planning/DESIGN_producer-gate.md`]).
//!
//! An eligible host (a desktop with a GPU, or the future headless GPU node) runs
//! [`run_election_loop`]: it polls the meetings on disk, claims any a capture device
//! marked [`ProcessingLifecycle::PendingProcessing`] (or whose [`ProcessingLifecycle::Claimed`]
//! lease has expired — a crashed holder, reaped), holds a renewable lease while it
//! runs the pipeline through the [`ElectionDriver`], and writes
//! [`ProcessingLifecycle::Processed`] on success. Every state change rides the
//! existing Discovery exchange (the driver's [`ElectionDriver::advertise`]); there
//! is no new wire message.
//!
//! # Why a trait seam
//!
//! The two collaborators the loop must not depend on directly — the `sync`
//! `SyncEngine` (to advertise) and the `orchestrator` (to reprocess) — sit behind
//! [`ElectionDriver`]. So this crate is a leaf (`common` + `persistence` +
//! `notes-crdt` only), the ONE state machine is reused by both eligible host
//! types, and the contention
//! paths are unit-testable with a mock driver (otherwise untestable until a second
//! GPU host exists).
//!
//! # Convergence, not a settle race
//!
//! Two eligible hosts can briefly claim and process the same meeting; that is
//! ACCEPTED as duplicate-but-idempotent work (same audio → ~same outputs), not
//! prevented by a timer (`DESIGN_producer-gate.md` §10, decision 6.4). The
//! authoritative winner falls out of convergence: `notes_crdt::merge_processing`
//! resolves competing `Claimed`/`Processed` to the lowest `HostRef`, and the
//! Artifacts authority rule resolves the derived bytes. The loop therefore does NOT
//! cancel an in-flight `process()` when it observes a competitor; it only stops
//! *renewing* the lease once GENUINELY superseded ([`superseded`]).
//!
//! # The lease-aware reap (the subtle bit)
//!
//! `merge_processing` resolves two `Claimed` by lowest `HostRef` *unconditionally*
//! (it is clock-independent by design and cannot read `now`), so a stale, expired
//! `Claimed{lower}` re-injected by a peer's buffered advert or the hub's discovery
//! sweep can momentarily clobber a legitimate reap on disk. The loop must NOT treat
//! that as a loss: [`superseded`] is LEASE-AWARE — a lower-`HostRef` claim supersedes
//! only while its lease is LIVE; an expired one is reapable regardless of `HostRef`
//! (`DESIGN_producer-gate.md` §10, review CRITICAL-1). The renewal re-asserts over
//! such a replay rather than aborting.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use minutist_common::{
    AppError, AppResult, HostRef, MeetingId, ProcessingClaim, ProcessingLifecycle,
};
use persistence::meeting_ops::{apply_own_processing_if_not_superseded, update_metadata_if, MetaUpdate};
use tokio::sync::Notify;

/// Tuning for the election loop. Durations are read from `MINUTIST_ELECTION_*`
/// (milliseconds) by [`ElectionConfig::from_env`], else the §10/§7 defaults.
#[derive(Debug, Clone)]
pub struct ElectionConfig {
    /// How often to scan the meetings on disk for claimable candidates.
    pub poll: Duration,
    /// Lease length stamped on a claim (`lease_expires_at = now + lease`).
    pub lease: Duration,
    /// How often the renewal task re-stamps the lease while processing.
    pub renew: Duration,
}

impl Default for ElectionConfig {
    fn default() -> Self {
        Self {
            poll: Duration::from_secs(60),
            lease: Duration::from_secs(30 * 60),
            renew: Duration::from_secs(10 * 60),
        }
    }
}

impl ElectionConfig {
    /// Read overrides from `MINUTIST_ELECTION_POLL_MS` / `_LEASE_MS` / `_RENEW_MS`,
    /// falling back to [`ElectionConfig::default`] for any unset or unparseable
    /// value.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            poll: env_ms("MINUTIST_ELECTION_POLL_MS").unwrap_or(d.poll),
            lease: env_ms("MINUTIST_ELECTION_LEASE_MS").unwrap_or(d.lease),
            renew: env_ms("MINUTIST_ELECTION_RENEW_MS").unwrap_or(d.renew),
        }
    }
}

fn env_ms(key: &str) -> Option<Duration> {
    std::env::var(key)
        .ok()?
        .parse::<u64>()
        .ok()
        .map(Duration::from_millis)
}

/// Whether this host may claim + process. Computed by the binding crate
/// (app-main / headless, which definitively link the GPU probe) and PASSED IN, so
/// this leaf stays pure and testable and its correctness does not hinge on Cargo
/// feature unification (`DESIGN_producer-gate.md` §6.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// A processing-capable host: run the election loop.
    Eligible,
    /// Sync-only (no GPU / a capture device): never claim — park forever.
    ParkSyncOnly,
}

/// The two collaborators the election loop drives, abstracted so this crate
/// depends on neither `sync` nor `orchestrator`.
#[async_trait::async_trait]
pub trait ElectionDriver: Send + Sync {
    /// This host's identity — `HostRef(endpoint_id().to_string())` in production.
    /// Stamped on claims and `Processed`, and the key for the lowest-`HostRef`
    /// tiebreak.
    fn host_ref(&self) -> HostRef;

    /// Propagate local lifecycle state to peers (→ `SyncEngine::discover_*`).
    async fn advertise(&self);

    /// Run the offline pipeline for `meeting_id` (→ `Orchestrator::reprocess`).
    /// The election loop only claims a candidate once its scan has confirmed
    /// `audio.opus` is present in the meeting folder (F4c), so by the time this
    /// is called the audio is there to read.
    async fn process(&self, meeting_id: MeetingId) -> AppResult<()>;

    /// Push the derived artifacts for `meeting_id` to peers BEFORE `Processed` is
    /// advertised, so a consumer never sees `Processed` without retrievable outputs
    /// (`DESIGN_producer-gate.md` §6.7). Default no-op for drivers without an
    /// Artifacts channel (e.g. the mock).
    async fn push_artifacts(&self, _meeting_id: MeetingId) {}
}

// ---------------------------------------------------------------------------
// Pure decision core (no `now()` inside — `now` is passed so it is testable)
// ---------------------------------------------------------------------------

/// `true` iff `lease_expires_at` (RFC 3339) is strictly before `now`. An
/// unparseable timestamp is treated as NOT expired — a lease we cannot evaluate is
/// never reaped out from under its holder (claims are written with valid RFC 3339,
/// so this only guards corruption).
fn lease_expired(lease_expires_at: &str, now: DateTime<Utc>) -> bool {
    match DateTime::parse_from_rfc3339(lease_expires_at) {
        Ok(exp) => now > exp.with_timezone(&Utc),
        Err(_) => false,
    }
}

/// Is `state` available for an eligible host to claim now? `PendingProcessing` (a
/// capture device offered it) or a `Claimed` whose lease has expired (reap a
/// crashed holder). `Local` / `Processed` / a live `Claimed` are not claimable.
fn claimable(state: &ProcessingLifecycle, now: DateTime<Utc>) -> bool {
    match state {
        ProcessingLifecycle::PendingProcessing => true,
        ProcessingLifecycle::Claimed { claim } => lease_expired(&claim.lease_expires_at, now),
        ProcessingLifecycle::Local | ProcessingLifecycle::Processed { .. } => false,
    }
}

/// Has my in-flight claim been GENUINELY superseded — should I stop renewing? True
/// iff `observed` is:
/// - `Processed` by any OTHER host (the outputs already exist), or
/// - a `Claimed` by a host with a strictly LOWER `HostRef` whose lease is STILL
///   LIVE (a live election winner).
///
/// LEASE-AWARE (review CRITICAL-1): a lower-`HostRef` `Claimed` whose lease has
/// EXPIRED is a stale replay of a dead holder, NOT a live winner — it does not
/// supersede, so the renewal re-asserts (reaps) instead of aborting. A higher-
/// `HostRef` live claim does not supersede either (I hold the lower `HostRef`, so I
/// win the tiebreak). My own `Claimed`/`Processed` never supersede me.
fn superseded(observed: &ProcessingLifecycle, me: &HostRef, now: DateTime<Utc>) -> bool {
    match observed {
        ProcessingLifecycle::Processed { processed_by, .. } => processed_by != me,
        ProcessingLifecycle::Claimed { claim } => {
            claim.host != *me
                && !lease_expired(&claim.lease_expires_at, now)
                && claim.host.0 < me.0
        }
        ProcessingLifecycle::Local | ProcessingLifecycle::PendingProcessing => false,
    }
}

/// Build a fresh claim for `host` valid for `lease` from `now`.
fn build_claim(host: &HostRef, now: DateTime<Utc>, lease: Duration) -> ProcessingClaim {
    let lease = chrono::Duration::from_std(lease).unwrap_or_else(|_| chrono::Duration::minutes(30));
    ProcessingClaim {
        host: host.clone(),
        claimed_at: now.to_rfc3339(),
        lease_expires_at: (now + lease).to_rfc3339(),
    }
}

/// Read a meeting's current processing state, or `None` if absent/unreadable.
fn read_processing(meetings_root: &Path, id: MeetingId) -> Option<ProcessingLifecycle> {
    let dir = meetings_root.join(id.0.to_string());
    persistence::read_metadata(&dir).ok().map(|m| m.processing)
}

/// Whether `audio.opus` is present in `id`'s meeting folder.
///
/// In the hub topology, `metadata.json`'s lifecycle state propagates over the
/// Discovery exchange independently of — and typically before — the media blob
/// pull, so a `PendingProcessing`/reapable meeting may not yet have its audio
/// synced in locally (F4c). `process()` (→ `Orchestrator::reprocess`) reads
/// `audio.opus` from disk and has no way to wait for it, so claiming before the
/// blob has arrived just burns a claim/fail cycle. A blocking `std::fs` check,
/// run from [`scan_candidates`] on `spawn_blocking`.
fn audio_present(meetings_root: &Path, id: MeetingId) -> bool {
    meetings_root
        .join(id.0.to_string())
        .join("audio.opus")
        .is_file()
}

/// Scan the meetings on disk for candidates this host may claim now: the state
/// is [`claimable`] AND (F4c) `audio.opus` is already present. A candidate whose
/// audio has not synced in yet is skipped — it remains claimable for a host that
/// already has the audio, or for this host once the blob arrives on a later
/// poll.
///
/// Blocking `std::fs` work (the directory scan + a `metadata.json` read + an
/// `audio.opus` stat per meeting); callers run this on `spawn_blocking`.
fn scan_candidates(meetings_root: &Path, now: DateTime<Utc>) -> Vec<MeetingId> {
    notes_crdt::folder::list_meeting_ids(meetings_root)
        .into_iter()
        .filter(|id| {
            let Some(state) = read_processing(meetings_root, *id) else {
                return false;
            };
            if !claimable(&state, now) {
                return false;
            }
            if !audio_present(meetings_root, *id) {
                tracing::debug!(
                    target: "election",
                    meeting_id = %id.0,
                    "skipping candidate: audio.opus has not synced in yet"
                );
                return false;
            }
            true
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// Run the election loop until cancelled (the task is dropped). On a
/// [`Capability::ParkSyncOnly`] host it parks forever and never claims.
///
/// Each poll: scan the meetings on disk, and for every claimable one attempt a
/// claim → process → `Processed`. A per-meeting failure is logged and skipped so
/// one bad meeting does not stall the rest.
pub async fn run_election_loop(
    config: ElectionConfig,
    driver: Arc<dyn ElectionDriver>,
    meetings_root: PathBuf,
    capability: Capability,
) {
    if capability == Capability::ParkSyncOnly {
        tracing::info!(target: "election", "sync-only host: parking (never claims)");
        std::future::pending::<()>().await;
        return;
    }
    let host = driver.host_ref();
    tracing::info!(target: "election", host = %host.0, "election loop started");
    loop {
        // Scan THEN sleep, so a freshly-eligible host claims an already-pending
        // meeting on startup rather than waiting a full poll interval first. The
        // scan is blocking `std::fs` work (a directory listing plus a
        // `metadata.json` read + an `audio.opus` stat per meeting), so it runs
        // on `spawn_blocking` rather than inline on the async worker.
        let now = Utc::now();
        let scan_root = meetings_root.clone();
        let candidates: Vec<MeetingId> = tokio::task::spawn_blocking(move || {
            scan_candidates(&scan_root, now)
        })
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(target: "election", error = %e, "candidate scan task join failed");
            Vec::new()
        });
        for id in candidates {
            if let Err(e) =
                try_elect_and_process(Arc::clone(&driver), &host, &meetings_root, id, &config)
                    .await
            {
                tracing::warn!(target: "election", meeting_id = %id.0, error = %e, "election attempt failed");
            }
        }
        tokio::time::sleep(config.poll).await;
    }
}

/// Attempt to claim `id` and, if won, process it and write `Processed`.
///
/// The claim is a CONDITIONAL guarded RMW: `Claimed{self}` is written only if the
/// current on-disk state is still claimable (so a concurrent claimant or a state
/// that moved under us cleanly yields). On a win it advertises, spawns a lease
/// renewer, runs `process()` to completion (no mid-process cancel — duplicate work
/// is accepted, §10 6.4), then on success pushes artifacts and writes `Processed`;
/// on failure it releases the claim back to `PendingProcessing` (F4a) rather than
/// leaving it to lapse the full lease.
///
/// `driver` is an owned `Arc` (not a borrow) so the same handle can be cloned into
/// the spawned lease-renewal task, which needs `'static` (F4b — the renewer calls
/// `driver.advertise()` after every successful re-stamp).
async fn try_elect_and_process(
    driver: Arc<dyn ElectionDriver>,
    host: &HostRef,
    meetings_root: &Path,
    id: MeetingId,
    config: &ElectionConfig,
) -> AppResult<()> {
    let now = Utc::now();
    let claim = build_claim(host, now, config.lease);
    let claim_root = meetings_root.to_path_buf();
    let claim_for_write = claim.clone();
    let applied = tokio::task::spawn_blocking(move || {
        update_metadata_if(&claim_root, id, move |m| {
            if claimable(&m.processing, now) {
                m.processing = ProcessingLifecycle::Claimed {
                    claim: claim_for_write.clone(),
                };
                Some(())
            } else {
                None
            }
        })
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("election claim task join failed: {e}"),
    })??;
    if !matches!(applied, MetaUpdate::Applied(())) {
        // Not claimable any more (a peer claimed it, or it moved to Processed) —
        // nothing to do.
        return Ok(());
    }
    tracing::info!(target: "election", meeting_id = %id.0, host = %host.0, "claimed");
    driver.advertise().await;

    // Keep the lease alive while processing. The renewer stops itself once
    // genuinely superseded; we also stop it when process() returns.
    let stop = Arc::new(Notify::new());
    let renewer = tokio::spawn(renewal_loop(
        meetings_root.to_path_buf(),
        id,
        host.clone(),
        config.clone(),
        stop.clone(),
        Arc::clone(&driver),
    ));

    let result = driver.process(id).await;
    stop.notify_one();
    let _ = renewer.await;

    match result {
        Ok(()) => {
            // Record our own `Processed` to local `metadata.json` BEFORE pushing
            // artifacts. `push_artifacts` → `import_artifacts` gates each derived
            // output (`transcript.json` / `summary.md`) on a provable authority
            // (`producer_authority`), which reads the on-disk lifecycle: only a
            // meeting whose `metadata.json` already reads `Processed` yields a
            // non-empty artifact manifest. Writing `Processed` first is therefore
            // what lets the push actually carry the outputs to a peer — pushing
            // while still `Claimed` sends an empty manifest and delivers nothing.
            //
            // Merge-aware terminal write (M2): a peer may already have converged
            // `Processed` (by a lower `HostRef`, synced in while we were
            // mid-process — duplicate-but-idempotent work is accepted, §10 6.4)
            // before we finished. Route the write through the SAME precedence
            // `merge_processing` applies to an inbound peer state
            // (`apply_own_processing_if_not_superseded`), rather than an
            // unconditional overwrite, so our own local write can never regress an
            // already-converged winner. Blocking `std::fs`, run on `spawn_blocking`.
            let processed = ProcessingLifecycle::Processed {
                processed_by: host.clone(),
                at: Utc::now().to_rfc3339(),
            };
            let write_root = meetings_root.to_path_buf();
            let write_outcome = tokio::task::spawn_blocking(move || {
                apply_own_processing_if_not_superseded(&write_root, id, processed)
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("election terminal write task join failed: {e}"),
            })??;
            match write_outcome {
                MetaUpdate::Applied(()) => {
                    tracing::info!(target: "election", meeting_id = %id.0, "processed")
                }
                MetaUpdate::SkippedPredicate => tracing::debug!(
                    target: "election",
                    meeting_id = %id.0,
                    "terminal write skipped: a converged/stronger state is already on disk"
                ),
                MetaUpdate::SkippedAbsent => tracing::debug!(
                    target: "election",
                    meeting_id = %id.0,
                    "terminal write skipped: meeting folder absent"
                ),
            }
            // Push artifacts AFTER the write: the on-disk lifecycle now reads
            // `Processed`, so `import_artifacts` has a provable authority and the
            // manifest carries the real `transcript.json` / `summary.md`. Still
            // before `advertise()`, so peers receive the bytes before they learn
            // `Processed` over discovery (§6.7).
            driver.push_artifacts(id).await;
            driver.advertise().await;
        }
        Err(e) => {
            // F4a: release the claim back to `PendingProcessing` for immediate
            // retry (this host's next poll tick, or a peer's) instead of leaving
            // it to lapse the full (default 30 min) lease — recorder-busy and
            // on-demand-model-load failures both hit this routinely, and
            // recovery latency should not be tied to the hold-lease sizing.
            //
            // An unconditional local write would be convergence-safe on its own
            // terms (`PendingProcessing` is the lowest-ranked non-`Local` state
            // in `merge_processing`, so it can never outrank a peer's `Claimed`/
            // `Processed`), but the release is still routed through the guarded
            // `update_metadata_if` — checking the on-disk claim is still ours —
            // so it matches the claim/renewal discipline the rest of this crate
            // uses and never fires against a state we no longer hold (e.g. a
            // peer's `Processed` that landed on our disk while `process()` ran;
            // we do not cancel in-flight work on supersession, §10 6.4). A
            // short-backoff-lease alternative was considered and rejected: it
            // would need its own tuning knob for no correctness benefit over
            // releasing, which lets the very next poll retry.
            let release_root = meetings_root.to_path_buf();
            let release_host = host.clone();
            let released = tokio::task::spawn_blocking(move || {
                update_metadata_if(&release_root, id, move |m| match &m.processing {
                    ProcessingLifecycle::Claimed { claim } if claim.host == release_host => {
                        m.processing = ProcessingLifecycle::PendingProcessing;
                        Some(())
                    }
                    _ => None,
                })
            })
            .await
            .map_err(|e| AppError::Internal {
                context: format!("election release task join failed: {e}"),
            })?;
            match released {
                Ok(MetaUpdate::Applied(())) => tracing::warn!(
                    target: "election",
                    meeting_id = %id.0,
                    error = %e,
                    "processing failed; released the claim back to PendingProcessing for immediate retry"
                ),
                Ok(_) => tracing::warn!(
                    target: "election",
                    meeting_id = %id.0,
                    error = %e,
                    "processing failed; the claim was already superseded, nothing to release"
                ),
                Err(release_err) => tracing::warn!(
                    target: "election",
                    meeting_id = %id.0,
                    error = %e,
                    release_error = %release_err,
                    "processing failed AND releasing the claim failed; leaving it to lapse"
                ),
            }
        }
    }
    Ok(())
}

/// Re-stamp the lease on the `renew` cadence while we hold the claim, stopping when
/// signalled (process finished) or when genuinely superseded. On every tick that
/// keeps renewing, propagates the fresh lease to peers via `driver.advertise()`
/// (F4b) — without this, a peer's disk only ever sees the ORIGINAL claim-time
/// lease, so a job that outlives the lease gets reaped and duplicated the instant
/// that stale lease expires on the peer's side, with no hub required.
async fn renewal_loop(
    meetings_root: PathBuf,
    id: MeetingId,
    host: HostRef,
    config: ElectionConfig,
    stop: Arc<Notify>,
    driver: Arc<dyn ElectionDriver>,
) {
    loop {
        tokio::select! {
            _ = tokio::time::sleep(config.renew) => {}
            _ = stop.notified() => return,
        }
        // Blocking `std::fs` RMW; run on `spawn_blocking` rather than inline on
        // the async worker.
        let step_root = meetings_root.clone();
        let step_host = host.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            renewal_step(&step_root, id, &step_host, Utc::now(), config.lease)
        })
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(target: "election", meeting_id = %id.0, error = %e, "lease renewal task join failed");
            RenewOutcome::Continue
        });
        if outcome == RenewOutcome::Stop {
            return;
        }
        driver.advertise().await;
    }
}

/// Whether the renewal loop should keep renewing or stop.
#[derive(Debug, PartialEq, Eq)]
enum RenewOutcome {
    /// Lease refreshed (or a transient write error); keep renewing.
    Continue,
    /// Genuinely superseded (a live lower-`HostRef` claim or a `Processed`-by-other)
    /// or the meeting is gone; stop renewing.
    Stop,
}

/// One renewal tick, performed atomically under the per-meeting lock: read the
/// current state, and
/// - if GENUINELY superseded ([`superseded`] — a `Processed`-by-other or a LIVE
///   lower-`HostRef` claim), do not write and return [`RenewOutcome::Stop`];
/// - otherwise re-assert our claim with a fresh lease — covering our own claim (the
///   lease refresh, PRESERVING the original `claimed_at`), a reapable expired/stale
///   claim (the lease-aware reap of a replay that `merge_processing` may have put on
///   our disk), a higher-`HostRef` LIVE claim we win the tiebreak over, and a
///   `PendingProcessing` offer. A `Local` / `Processed` state is never regressed
///   (it should not arise mid-process; if it does, we leave it and keep ticking).
///
/// The decision and the write are ONE closure under ONE lock acquisition (no
/// read-then-write window); the supersession observation rides back out via the
/// closure-captured flag — the observed-state-on-skip contract
/// [`notes_crdt::update_metadata_if`] exists for.
fn renewal_step(
    meetings_root: &Path,
    id: MeetingId,
    host: &HostRef,
    now: DateTime<Utc>,
    lease: Duration,
) -> RenewOutcome {
    let mut lost = false;
    let lease = chrono::Duration::from_std(lease).unwrap_or_else(|_| chrono::Duration::minutes(30));
    let result = update_metadata_if(meetings_root, id, |m| {
        if superseded(&m.processing, host, now) {
            lost = true;
            return None;
        }
        // Re-assert only over a Claimed (mine / reapable / higher-live — the
        // superseding cases already returned above) or a Pending offer; never
        // regress a Local / Processed.
        let reassert = matches!(
            m.processing,
            ProcessingLifecycle::Claimed { .. } | ProcessingLifecycle::PendingProcessing
        );
        if !reassert {
            return None;
        }
        let claimed_at = match &m.processing {
            ProcessingLifecycle::Claimed { claim } if claim.host == *host => {
                claim.claimed_at.clone()
            }
            _ => now.to_rfc3339(),
        };
        m.processing = ProcessingLifecycle::Claimed {
            claim: ProcessingClaim {
                host: host.clone(),
                claimed_at,
                lease_expires_at: (now + lease).to_rfc3339(),
            },
        };
        Some(())
    });
    match result {
        // The meeting's folder is gone — nothing left to renew.
        Ok(MetaUpdate::SkippedAbsent) => RenewOutcome::Stop,
        // Skipped because superseded (stop) or because the state was non-reassertable
        // Local/Processed (harmless; keep ticking until process() signals stop).
        Ok(MetaUpdate::SkippedPredicate) if lost => RenewOutcome::Stop,
        Ok(_) => RenewOutcome::Continue,
        Err(e) => {
            tracing::warn!(target: "election", meeting_id = %id.0, error = %e, "lease renewal write failed");
            RenewOutcome::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use persistence::meeting_ops::{apply_processing_lifecycle, update_metadata};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn host(s: &str) -> HostRef {
        HostRef(s.to_string())
    }

    fn t(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    fn claim_at(h: &str, lease_expires_at: &str) -> ProcessingClaim {
        ProcessingClaim {
            host: host(h),
            claimed_at: "2026-07-01T10:00:00Z".to_string(),
            lease_expires_at: lease_expires_at.to_string(),
        }
    }

    // ----- pure decision core -----

    #[test]
    fn lease_expiry_parses_and_defaults_unparseable_to_live() {
        let now = t("2026-07-01T12:00:00Z");
        assert!(lease_expired("2026-07-01T11:59:59Z", now));
        assert!(!lease_expired("2026-07-01T12:00:01Z", now));
        // Unparseable → NOT expired (never reap what we can't evaluate).
        assert!(!lease_expired("garbage", now));
    }

    #[test]
    fn claimable_is_pending_or_expired_claim_only() {
        let now = t("2026-07-01T12:00:00Z");
        assert!(claimable(&ProcessingLifecycle::PendingProcessing, now));
        assert!(claimable(
            &ProcessingLifecycle::Claimed { claim: claim_at("a", "2026-07-01T11:00:00Z") },
            now
        ));
        // Live claim, Local, Processed → not claimable.
        assert!(!claimable(
            &ProcessingLifecycle::Claimed { claim: claim_at("a", "2026-07-01T13:00:00Z") },
            now
        ));
        assert!(!claimable(&ProcessingLifecycle::Local, now));
        assert!(!claimable(
            &ProcessingLifecycle::Processed { processed_by: host("a"), at: "x".into() },
            now
        ));
    }

    #[test]
    fn superseded_is_lease_aware_and_hostref_aware() {
        let me = host("m");
        let now = t("2026-07-01T12:00:00Z");
        let live = "2026-07-01T13:00:00Z";
        let expired = "2026-07-01T11:00:00Z";

        // Processed by another host supersedes; by me does not.
        assert!(superseded(
            &ProcessingLifecycle::Processed { processed_by: host("a"), at: "x".into() },
            &me,
            now
        ));
        assert!(!superseded(
            &ProcessingLifecycle::Processed { processed_by: me.clone(), at: "x".into() },
            &me,
            now
        ));

        // A LIVE lower-HostRef claim supersedes ("a" < "m").
        assert!(superseded(
            &ProcessingLifecycle::Claimed { claim: claim_at("a", live) },
            &me,
            now
        ));
        // CRITICAL-1: an EXPIRED lower-HostRef claim (a stale replay of a dead
        // holder) does NOT supersede — it is reapable regardless of HostRef.
        assert!(!superseded(
            &ProcessingLifecycle::Claimed { claim: claim_at("a", expired) },
            &me,
            now
        ));
        // A live HIGHER-HostRef claim does not supersede (I win the tiebreak).
        assert!(!superseded(
            &ProcessingLifecycle::Claimed { claim: claim_at("z", live) },
            &me,
            now
        ));
        // My own live claim does not supersede me.
        assert!(!superseded(
            &ProcessingLifecycle::Claimed { claim: claim_at("m", live) },
            &me,
            now
        ));
        // Pending / Local never supersede.
        assert!(!superseded(&ProcessingLifecycle::PendingProcessing, &me, now));
        assert!(!superseded(&ProcessingLifecycle::Local, &me, now));
    }

    // ----- mock-driver loop paths -----

    #[derive(Default)]
    struct MockDriver {
        host: String,
        advertises: AtomicUsize,
        processes: AtomicUsize,
        push_artifacts: AtomicUsize,
        /// Order log so we can assert push_artifacts happens before the Processed advertise.
        order: Mutex<Vec<&'static str>>,
        fail_process: bool,
        /// Real-time delay `process()` awaits before returning, so a test can let
        /// the renewal loop tick (F4b) while a claim is held.
        process_delay: Duration,
        /// Run synchronously inside `process()`, before it returns — lets a test
        /// simulate a peer's stronger state converging onto disk WHILE we are
        /// mid-process (M2).
        on_process: Option<Box<dyn Fn() + Send + Sync>>,
        /// Run synchronously inside `push_artifacts()` — lets a test observe the
        /// on-disk lifecycle at the exact moment artifacts are pushed, which is
        /// what the artifact authority-gate reads.
        on_push_artifacts: Option<Box<dyn Fn() + Send + Sync>>,
    }

    #[async_trait::async_trait]
    impl ElectionDriver for MockDriver {
        fn host_ref(&self) -> HostRef {
            HostRef(self.host.clone())
        }
        async fn advertise(&self) {
            self.advertises.fetch_add(1, Ordering::SeqCst);
            self.order.lock().unwrap().push("advertise");
        }
        async fn process(&self, _id: MeetingId) -> AppResult<()> {
            if !self.process_delay.is_zero() {
                tokio::time::sleep(self.process_delay).await;
            }
            self.processes.fetch_add(1, Ordering::SeqCst);
            self.order.lock().unwrap().push("process");
            if let Some(f) = &self.on_process {
                f();
            }
            if self.fail_process {
                Err(minutist_common::AppError::Internal {
                    context: "mock process failure".to_string(),
                })
            } else {
                Ok(())
            }
        }
        async fn push_artifacts(&self, _id: MeetingId) {
            self.push_artifacts.fetch_add(1, Ordering::SeqCst);
            self.order.lock().unwrap().push("push_artifacts");
            if let Some(f) = &self.on_push_artifacts {
                f();
            }
        }
    }

    fn cfg() -> ElectionConfig {
        // A long renew so the renewer never ticks during a fast mock process.
        ElectionConfig {
            poll: Duration::from_millis(10),
            lease: Duration::from_secs(1800),
            renew: Duration::from_secs(3600),
        }
    }

    async fn seed(root: &Path, id: MeetingId, state: ProcessingLifecycle) {
        notes_crdt::MeetingFolder::ensure(root, id).expect("ensure");
        apply_processing_lifecycle(root, id, state).await.expect("seed state");
    }

    #[tokio::test]
    async fn claims_pending_processes_and_writes_processed() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        seed(root, id, ProcessingLifecycle::PendingProcessing).await;

        let mock = Arc::new(MockDriver { host: "m".into(), ..Default::default() });
        let driver: Arc<dyn ElectionDriver> = mock.clone();
        try_elect_and_process(driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        assert_eq!(mock.processes.load(Ordering::SeqCst), 1);
        assert_eq!(mock.push_artifacts.load(Ordering::SeqCst), 1);
        // Final state is Processed by us.
        assert_eq!(
            read_processing(root, id),
            Some(ProcessingLifecycle::Processed {
                processed_by: host("m"),
                at: read_processed_at(root, id),
            })
        );
        // push_artifacts precedes the Processed advertise (the §6.7 order).
        let order = mock.order.lock().unwrap().clone();
        let pa = order.iter().position(|s| *s == "push_artifacts").unwrap();
        let last_adv = order.iter().rposition(|s| *s == "advertise").unwrap();
        assert!(pa < last_adv, "push_artifacts must precede the Processed advertise: {order:?}");
    }

    #[tokio::test]
    async fn push_artifacts_runs_after_the_processed_write() {
        // Regression (producer-gate artifact-return bug): `push_artifacts` →
        // `import_artifacts` gates each derived output on the on-disk lifecycle
        // reading `Processed` (`producer_authority`). If the push runs while the
        // meeting is still `Claimed`, the artifact manifest is empty and a passive
        // peer receives no `transcript.json` / `summary.md`. Assert the metadata
        // already reads `Processed` at the moment artifacts are pushed.
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        seed(root, id, ProcessingLifecycle::PendingProcessing).await;

        let seen: Arc<Mutex<Option<Option<ProcessingLifecycle>>>> = Arc::new(Mutex::new(None));
        let seen_cl = Arc::clone(&seen);
        let root_cl = root.to_path_buf();
        let mock = Arc::new(MockDriver {
            host: "m".into(),
            on_push_artifacts: Some(Box::new(move || {
                *seen_cl.lock().unwrap() = Some(read_processing(&root_cl, id));
            })),
            ..Default::default()
        });
        let driver: Arc<dyn ElectionDriver> = mock.clone();
        try_elect_and_process(driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        let observed = seen
            .lock()
            .unwrap()
            .clone()
            .expect("push_artifacts must have run");
        assert!(
            matches!(observed, Some(ProcessingLifecycle::Processed { .. })),
            "metadata at push_artifacts time must already read Processed, was {observed:?}"
        );
    }

    #[tokio::test]
    async fn does_not_claim_a_live_foreign_claim() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        let live = (Utc::now() + chrono::Duration::minutes(20)).to_rfc3339();
        seed(
            root,
            id,
            ProcessingLifecycle::Claimed { claim: claim_at("a", &live) },
        )
        .await;

        let mock = Arc::new(MockDriver { host: "m".into(), ..Default::default() });
        let driver: Arc<dyn ElectionDriver> = mock.clone();
        try_elect_and_process(driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        // We never processed and never touched the foreign live claim.
        assert_eq!(mock.processes.load(Ordering::SeqCst), 0);
        assert!(matches!(
            read_processing(root, id),
            Some(ProcessingLifecycle::Claimed { claim }) if claim.host == host("a")
        ));
    }

    #[tokio::test]
    async fn reaps_an_expired_foreign_claim() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        let expired = (Utc::now() - chrono::Duration::minutes(5)).to_rfc3339();
        seed(
            root,
            id,
            ProcessingLifecycle::Claimed { claim: claim_at("a", &expired) },
        )
        .await;

        let mock = Arc::new(MockDriver { host: "m".into(), ..Default::default() });
        let driver: Arc<dyn ElectionDriver> = mock.clone();
        try_elect_and_process(driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        // The expired foreign claim was reaped, processed, and marked Processed by us.
        assert_eq!(mock.processes.load(Ordering::SeqCst), 1);
        assert!(matches!(
            read_processing(root, id),
            Some(ProcessingLifecycle::Processed { processed_by, .. }) if processed_by == host("m")
        ));
    }

    #[tokio::test]
    async fn failed_process_releases_claim_to_pending_processing() {
        // F4a: a failed process() releases the claim back to `PendingProcessing`
        // for immediate retry, rather than leaving it `Claimed` to lapse the full
        // (default 30 min) lease.
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        seed(root, id, ProcessingLifecycle::PendingProcessing).await;

        let mock = Arc::new(MockDriver { host: "m".into(), fail_process: true, ..Default::default() });
        let driver: Arc<dyn ElectionDriver> = mock.clone();
        try_elect_and_process(driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        assert_eq!(mock.processes.load(Ordering::SeqCst), 1);
        assert_eq!(mock.push_artifacts.load(Ordering::SeqCst), 0);
        assert_eq!(
            read_processing(root, id),
            Some(ProcessingLifecycle::PendingProcessing),
            "a failed process must release the claim, not leave it to lapse"
        );
    }

    #[tokio::test]
    async fn renewer_advertises_on_tick_and_failed_process_releases_the_claim() {
        // A short renew so the renewer ticks multiple times while the mock
        // process() is still running (a real `tokio::time::sleep` inside
        // `process()`), proving F4b's wiring: the renewer calls
        // `driver.advertise()` after each successful re-stamp (the re-stamp's OWN
        // correctness — e.g. preserving `claimed_at` — is covered by the
        // `renewal_step_*` unit tests above). The process then fails, proving F4a:
        // the claim is released to `PendingProcessing` rather than left to lapse.
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        seed(root, id, ProcessingLifecycle::PendingProcessing).await;

        let cfg = ElectionConfig {
            poll: Duration::from_millis(10),
            lease: Duration::from_secs(1800),
            renew: Duration::from_millis(15),
        };

        let mock = Arc::new(MockDriver {
            host: "m".into(),
            fail_process: true,
            process_delay: Duration::from_millis(90),
            ..Default::default()
        });
        let driver: Arc<dyn ElectionDriver> = mock.clone();

        try_elect_and_process(driver, &host("m"), root, id, &cfg)
            .await
            .expect("elect");

        // The claim-time advertise (1) plus at least one renewal-tick advertise —
        // a 90 ms process at a 15 ms renew cadence fits several ticks. No
        // Processed advertise fires since process() failed.
        assert!(
            mock.advertises.load(Ordering::SeqCst) >= 2,
            "the renewer must advertise on at least one tick during the slow process"
        );
        assert_eq!(mock.processes.load(Ordering::SeqCst), 1);
        assert_eq!(mock.push_artifacts.load(Ordering::SeqCst), 0);
        assert_eq!(
            read_processing(root, id),
            Some(ProcessingLifecycle::PendingProcessing)
        );
    }

    #[test]
    fn scan_candidates_skips_a_pending_meeting_missing_audio() {
        // F4c: in the hub topology `metadata.json`'s lifecycle state can
        // propagate before the `audio.opus` blob has synced in. A candidate
        // missing its audio must be skipped so it stays `PendingProcessing` for a
        // host that already has the audio (or for this host once it arrives).
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let now = Utc::now();

        let no_audio = MeetingId::new();
        notes_crdt::MeetingFolder::ensure(root, no_audio).expect("ensure");
        update_metadata(root, no_audio, |m| {
            m.processing = ProcessingLifecycle::PendingProcessing;
        })
        .expect("seed no-audio pending");

        let with_audio = MeetingId::new();
        notes_crdt::MeetingFolder::ensure(root, with_audio).expect("ensure");
        update_metadata(root, with_audio, |m| {
            m.processing = ProcessingLifecycle::PendingProcessing;
        })
        .expect("seed with-audio pending");
        std::fs::write(root.join(with_audio.0.to_string()).join("audio.opus"), b"opus")
            .expect("write audio.opus");

        let candidates = scan_candidates(root, now);

        assert_eq!(
            candidates,
            vec![with_audio],
            "a candidate missing audio.opus must be skipped"
        );
    }

    #[tokio::test]
    async fn terminal_write_does_not_regress_a_converged_lower_processed() {
        // M2: while our process() is running, simulate a peer's stronger state
        // (a lower-HostRef Processed, converged via the lifecycle subscriber's
        // merge_processing) landing on our own disk. Our own terminal write must
        // not clobber it back to Processed{self}.
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        seed(root, id, ProcessingLifecycle::PendingProcessing).await;

        let converge_root = root.to_path_buf();
        let mock = Arc::new(MockDriver {
            host: "m".into(),
            on_process: Some(Box::new(move || {
                update_metadata(&converge_root, id, |m| {
                    m.processing = ProcessingLifecycle::Processed {
                        processed_by: host("a"), // "a" < "m" — a lower, stronger winner
                        at: "2026-07-01T10:00:00Z".to_string(),
                    };
                })
                .expect("simulate a converged peer Processed landing mid-process");
            })),
            ..Default::default()
        });
        let driver: Arc<dyn ElectionDriver> = mock.clone();

        try_elect_and_process(driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        // Our own Processed{m} must not have clobbered the already-converged
        // Processed{a}.
        assert!(
            matches!(
                read_processing(root, id),
                Some(ProcessingLifecycle::Processed { processed_by, .. }) if processed_by == host("a")
            ),
            "the terminal write must not regress an already-converged lower-HostRef winner"
        );
    }

    fn read_processed_at(root: &Path, id: MeetingId) -> String {
        match read_processing(root, id) {
            Some(ProcessingLifecycle::Processed { at, .. }) => at,
            other => panic!("expected Processed, got {other:?}"),
        }
    }

    fn read_claim(root: &Path, id: MeetingId) -> ProcessingClaim {
        match read_processing(root, id) {
            Some(ProcessingLifecycle::Claimed { claim }) => claim,
            other => panic!("expected Claimed, got {other:?}"),
        }
    }

    fn claimed_state(h: &str, claimed_at: &str, lease_expires_at: &str) -> ProcessingLifecycle {
        ProcessingLifecycle::Claimed {
            claim: ProcessingClaim {
                host: host(h),
                claimed_at: claimed_at.to_string(),
                lease_expires_at: lease_expires_at.to_string(),
            },
        }
    }

    // ----- renewal_step (the loop core — review W1) -----

    #[tokio::test]
    async fn renewal_step_refreshes_my_lease_preserving_claimed_at() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        let now = Utc::now();
        let old_lease = (now + chrono::Duration::minutes(1)).to_rfc3339();
        seed(root, id, claimed_state("m", "2026-07-01T09:00:00Z", &old_lease)).await;

        let out = renewal_step(root, id, &host("m"), now, Duration::from_secs(1800));

        assert_eq!(out, RenewOutcome::Continue);
        let claim = read_claim(root, id);
        assert_eq!(claim.host, host("m"));
        // W3: the original claim instant is preserved across a renew.
        assert_eq!(claim.claimed_at, "2026-07-01T09:00:00Z");
        // The lease was extended (new expiry is later than the old one-minute one).
        assert!(
            DateTime::parse_from_rfc3339(&claim.lease_expires_at).unwrap()
                > DateTime::parse_from_rfc3339(&old_lease).unwrap()
        );
    }

    #[tokio::test]
    async fn renewal_step_reaps_a_stale_expired_lower_replay() {
        // The CRITICAL-1 case IN the loop: merge_processing may have clobbered our
        // disk with a dead lower-HostRef holder's EXPIRED claim (replayed off a hub
        // sweep). The renew must reap it back to us, not abort.
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        let now = Utc::now();
        let expired = (now - chrono::Duration::minutes(5)).to_rfc3339();
        seed(root, id, claimed_state("a", "2026-07-01T09:00:00Z", &expired)).await;

        let out = renewal_step(root, id, &host("m"), now, Duration::from_secs(1800));

        assert_eq!(out, RenewOutcome::Continue);
        assert_eq!(read_claim(root, id).host, host("m"), "expired lower replay must be reaped");
    }

    #[tokio::test]
    async fn renewal_step_yields_to_a_live_lower_winner() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        let now = Utc::now();
        let live = (now + chrono::Duration::minutes(20)).to_rfc3339();
        seed(root, id, claimed_state("a", "2026-07-01T09:00:00Z", &live)).await;

        let out = renewal_step(root, id, &host("m"), now, Duration::from_secs(1800));

        assert_eq!(out, RenewOutcome::Stop, "a live lower-HostRef claim supersedes us");
        assert_eq!(read_claim(root, id).host, host("a"), "the live winner is left intact");
    }

    #[tokio::test]
    async fn renewal_step_reasserts_over_a_live_higher_claim() {
        // W2: a higher-HostRef live foreign claim on our own disk (a stale-merge
        // artefact — we win the tiebreak) must be re-asserted, not left to lapse.
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        let now = Utc::now();
        let live = (now + chrono::Duration::minutes(20)).to_rfc3339();
        seed(root, id, claimed_state("z", "2026-07-01T09:00:00Z", &live)).await;

        let out = renewal_step(root, id, &host("m"), now, Duration::from_secs(1800));

        assert_eq!(out, RenewOutcome::Continue);
        assert_eq!(read_claim(root, id).host, host("m"), "we win over a higher-HostRef live claim");
    }

    #[tokio::test]
    async fn renewal_step_stops_on_processed_by_other_and_absent() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let now = Utc::now();

        let processed = MeetingId::new();
        seed(
            root,
            processed,
            ProcessingLifecycle::Processed { processed_by: host("a"), at: "2026-07-01T10:00:00Z".into() },
        )
        .await;
        assert_eq!(
            renewal_step(root, processed, &host("m"), now, Duration::from_secs(1800)),
            RenewOutcome::Stop
        );
        // Unchanged — we never regress a Processed.
        assert!(matches!(
            read_processing(root, processed),
            Some(ProcessingLifecycle::Processed { processed_by, .. }) if processed_by == host("a")
        ));

        // An absent meeting (folder gone) stops the renewer.
        assert_eq!(
            renewal_step(root, MeetingId::new(), &host("m"), now, Duration::from_secs(1800)),
            RenewOutcome::Stop
        );
    }
}
