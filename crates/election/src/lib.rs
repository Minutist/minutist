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
//! [`ElectionDriver`]. So this crate is a leaf (`common` + `persistence` only), the
//! ONE state machine is reused by both eligible host types, and the contention
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
use minutist_common::{AppResult, HostRef, MeetingId, ProcessingClaim, ProcessingLifecycle};
use persistence::meeting_ops::{apply_processing_lifecycle, update_metadata_if, MetaUpdate};
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
    /// `audio.opus` has synced in, so a re-electing host has the audio.
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
        tokio::time::sleep(config.poll).await;
        let now = Utc::now();
        let candidates: Vec<MeetingId> = persistence::folder::list_meeting_ids(&meetings_root)
            .into_iter()
            .filter(|id| read_processing(&meetings_root, *id).is_some_and(|s| claimable(&s, now)))
            .collect();
        for id in candidates {
            if let Err(e) = try_elect_and_process(&*driver, &host, &meetings_root, id, &config).await
            {
                tracing::warn!(target: "election", meeting_id = %id.0, error = %e, "election attempt failed");
            }
        }
    }
}

/// Attempt to claim `id` and, if won, process it and write `Processed`.
///
/// The claim is a CONDITIONAL guarded RMW: `Claimed{self}` is written only if the
/// current on-disk state is still claimable (so a concurrent claimant or a state
/// that moved under us cleanly yields). On a win it advertises, spawns a lease
/// renewer, runs `process()` to completion (no mid-process cancel — duplicate work
/// is accepted, §10 6.4), then on success pushes artifacts and writes `Processed`.
async fn try_elect_and_process(
    driver: &dyn ElectionDriver,
    host: &HostRef,
    meetings_root: &Path,
    id: MeetingId,
    config: &ElectionConfig,
) -> AppResult<()> {
    let claim = build_claim(host, Utc::now(), config.lease);
    let applied = update_metadata_if(meetings_root, id, |m| {
        if claimable(&m.processing, Utc::now()) {
            m.processing = ProcessingLifecycle::Claimed {
                claim: claim.clone(),
            };
            Some(())
        } else {
            None
        }
    })?;
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
    ));

    let result = driver.process(id).await;
    stop.notify_one();
    let _ = renewer.await;

    match result {
        Ok(()) => {
            // Artifacts BEFORE advertising Processed (§6.7).
            driver.push_artifacts(id).await;
            apply_processing_lifecycle(
                meetings_root,
                id,
                ProcessingLifecycle::Processed {
                    processed_by: host.clone(),
                    at: Utc::now().to_rfc3339(),
                },
            )
            .await?;
            tracing::info!(target: "election", meeting_id = %id.0, "processed");
            driver.advertise().await;
        }
        Err(e) => {
            // Leave the claim to lapse — a peer (or this host on a later poll) reaps
            // and re-elects. Covers recorder-busy and model-load failures alike.
            tracing::warn!(target: "election", meeting_id = %id.0, error = %e, "processing failed; leaving the claim to lapse");
        }
    }
    Ok(())
}

/// Re-stamp the lease on the `renew` cadence while we hold the claim, stopping when
/// signalled (process finished) or when genuinely superseded. Re-asserts the claim
/// if a stale expired-claim replay clobbered it (the lease-aware reap), but yields
/// to a live lower-`HostRef` claim or a `Processed`-by-other.
async fn renewal_loop(
    meetings_root: PathBuf,
    id: MeetingId,
    host: HostRef,
    config: ElectionConfig,
    stop: Arc<Notify>,
) {
    loop {
        tokio::select! {
            _ = tokio::time::sleep(config.renew) => {}
            _ = stop.notified() => return,
        }
        let now = Utc::now();
        if let Some(state) = read_processing(&meetings_root, id) {
            if superseded(&state, &host, now) {
                tracing::info!(target: "election", meeting_id = %id.0, "claim superseded by a live winner; stopping renewal");
                return;
            }
        }
        let claim = build_claim(&host, now, config.lease);
        let renewed = update_metadata_if(&meetings_root, id, |m| {
            let mine = matches!(&m.processing, ProcessingLifecycle::Claimed { claim } if claim.host == host);
            if mine || claimable(&m.processing, now) {
                m.processing = ProcessingLifecycle::Claimed {
                    claim: claim.clone(),
                };
                Some(())
            } else {
                None
            }
        });
        if let Err(e) = renewed {
            tracing::warn!(target: "election", meeting_id = %id.0, error = %e, "lease renewal write failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            self.processes.fetch_add(1, Ordering::SeqCst);
            self.order.lock().unwrap().push("process");
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
        persistence::MeetingFolder::ensure(root, id).expect("ensure");
        apply_processing_lifecycle(root, id, state).await.expect("seed state");
    }

    #[tokio::test]
    async fn claims_pending_processes_and_writes_processed() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        seed(root, id, ProcessingLifecycle::PendingProcessing).await;

        let driver = MockDriver { host: "m".into(), ..Default::default() };
        try_elect_and_process(&driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        assert_eq!(driver.processes.load(Ordering::SeqCst), 1);
        assert_eq!(driver.push_artifacts.load(Ordering::SeqCst), 1);
        // Final state is Processed by us.
        assert_eq!(
            read_processing(root, id),
            Some(ProcessingLifecycle::Processed {
                processed_by: host("m"),
                at: read_processed_at(root, id),
            })
        );
        // push_artifacts precedes the Processed advertise (the §6.7 order).
        let order = driver.order.lock().unwrap().clone();
        let pa = order.iter().position(|s| *s == "push_artifacts").unwrap();
        let last_adv = order.iter().rposition(|s| *s == "advertise").unwrap();
        assert!(pa < last_adv, "push_artifacts must precede the Processed advertise: {order:?}");
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

        let driver = MockDriver { host: "m".into(), ..Default::default() };
        try_elect_and_process(&driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        // We never processed and never touched the foreign live claim.
        assert_eq!(driver.processes.load(Ordering::SeqCst), 0);
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

        let driver = MockDriver { host: "m".into(), ..Default::default() };
        try_elect_and_process(&driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        // The expired foreign claim was reaped, processed, and marked Processed by us.
        assert_eq!(driver.processes.load(Ordering::SeqCst), 1);
        assert!(matches!(
            read_processing(root, id),
            Some(ProcessingLifecycle::Processed { processed_by, .. }) if processed_by == host("m")
        ));
    }

    #[tokio::test]
    async fn failed_process_does_not_write_processed() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let root = tmp.path();
        let id = MeetingId::new();
        seed(root, id, ProcessingLifecycle::PendingProcessing).await;

        let driver = MockDriver { host: "m".into(), fail_process: true, ..Default::default() };
        try_elect_and_process(&driver, &host("m"), root, id, &cfg())
            .await
            .expect("elect");

        assert_eq!(driver.processes.load(Ordering::SeqCst), 1);
        assert_eq!(driver.push_artifacts.load(Ordering::SeqCst), 0);
        // Left Claimed (by us) to lapse — NOT Processed.
        assert!(matches!(
            read_processing(root, id),
            Some(ProcessingLifecycle::Claimed { claim }) if claim.host == host("m")
        ));
    }

    fn read_processed_at(root: &Path, id: MeetingId) -> String {
        match read_processing(root, id) {
            Some(ProcessingLifecycle::Processed { at, .. }) => at,
            other => panic!("expected Processed, got {other:?}"),
        }
    }
}
