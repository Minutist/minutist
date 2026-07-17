//! Account-mediated peer discovery (`planning/DESIGN_account-peer-source.md`,
//! "B2"; loop v2: `planning/DESIGN_account-refresh-loop-v2.md`).
//!
//! Manual pairing (a shared [`EndpointTicket`](iroh_tickets::endpoint::EndpointTicket))
//! is one way a peer's [`iroh::EndpointAddr`] reaches [`crate::SyncEngine::add_peer`].
//! This module adds a second, account-mediated one: an injected
//! [`AccountEndpointSource`] periodically fetches the other devices on the
//! signed-in account and feeds each into a refresh loop. The two sources are
//! additive — both feed the same [`crate::address_lookup::PeerDirectory`].
//!
//! `crates/sync` takes **no** HTTP or account-service dependency of its own: the
//! consumer (app-main / sync-ffi) supplies an [`AccountEndpointSource`]
//! implementation bound to the device's account credential. This module defines
//! only the trait, the pure reconciliation logic, and the loop that drives it —
//! mirroring the `election` crate's collaborator-behind-a-trait seam.
//!
//! [`run_account_refresh_loop_v2`] drives the loop, built once around
//! [`RefreshSink`] so each new capability (cancellation, source-aware removal,
//! dial-suppression, first-contact dial-kick) is a trait-method addition rather
//! than a signature change.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use minutist_common::AppResult;
use tokio_util::sync::CancellationToken;

/// One device's account-published endpoint: the address another of the
/// account's devices dials to reach it.
#[derive(Debug, Clone, PartialEq)]
pub struct AccountEndpoint {
    /// The account-service device id (opaque; not necessarily the endpoint id).
    pub device_id: String,
    /// The device's hex iroh `EndpointId`.
    pub endpoint_id: String,
    /// The relay URL the device is reachable through.
    pub relay_url: String,
}

/// The injected account-service HTTP seam. `crates/sync` depends on this trait
/// only, never on an HTTP client or the account service directly, so the
/// dependency-table edge stays at `common` (see `architecture/components.md`);
/// the consumer supplies a fetcher already bound to the device's account
/// credential.
#[async_trait::async_trait]
pub trait AccountEndpointSource: Send + Sync {
    /// Fetch every endpoint currently published on this device's account
    /// (`GET /v1/account/devices`), including this device's own entry.
    async fn list_endpoints(&self) -> AppResult<Vec<AccountEndpoint>>;

    /// Publish this device's own endpoint to the account service
    /// (`PUT /v1/account/devices/self/endpoint`), idempotent upsert.
    async fn register_self(&self, endpoint: &AccountEndpoint) -> AppResult<()>;
}

/// Which of `list` this device should add as a peer: every entry whose
/// `endpoint_id` is not `own_endpoint_id`, de-duplicated by `endpoint_id`
/// (first occurrence wins), order preserved.
pub fn peers_to_add(list: &[AccountEndpoint], own_endpoint_id: &str) -> Vec<AccountEndpoint> {
    let mut seen = HashSet::new();
    list.iter()
        .filter(|ep| ep.endpoint_id != own_endpoint_id)
        .filter(|ep| seen.insert(ep.endpoint_id.clone()))
        .cloned()
        .collect()
}

/// The consumer-provided sink [`run_account_refresh_loop_v2`] drives — the one
/// object a consumer implements to wire up account-mediated peer discovery, so
/// each new loop capability lands as a trait-method addition instead of a
/// signature change (`planning/DESIGN_account-refresh-loop-v2.md`).
///
/// Production wires this to [`crate::endpoint::SyncEngineRefreshSink`]; a test
/// records calls instead.
#[async_trait::async_trait]
pub trait RefreshSink: Send + Sync {
    /// Upsert a peer discovered from the account directory (source =
    /// `Account`). Returns whether it was newly added, so the loop can
    /// first-contact-dial a genuinely new peer via [`Self::on_new_peer`].
    fn upsert_account_peer(&self, ep: &AccountEndpoint) -> bool;

    /// Remove an `Account`-sourced peer no longer present in the directory
    /// list (reconcile — it left the account). Source-aware: must never remove
    /// a manually-paired peer.
    fn remove_account_peer(&self, endpoint_id: &str);

    /// Whether the peer is currently dial-suppressed (failed-dial backoff). The
    /// loop consults this before an [`Self::on_new_peer`] dial so a
    /// re-appearing-but-still-unreachable peer is not instant-dialled.
    fn is_suppressed(&self, endpoint_id: &str) -> bool;

    /// First-contact dial of a genuinely new peer, so first-sync is not gated
    /// on the poll cadence. The loop never calls this when [`Self::is_suppressed`]
    /// holds; an implementation MUST also treat it as a no-op if suppressed, as
    /// a second guard. Runs under the loop's cancel scope — dropped on cancel.
    async fn on_new_peer(&self, endpoint_id: &str);

    /// The directory's current `Account`-sourced endpoint ids. Read once at loop
    /// (re)start to seed the reconcile-removal state, so a peer that left the
    /// account during a disabled window (the directory survives a live-process
    /// loop restart) is not treated as absent-from-nothing and left as a
    /// permanent zombie entry.
    fn account_peer_ids(&self) -> Vec<String>;
}

/// Run the account-peer-discovery loop until `cancel` is cancelled.
///
/// On start, registers `self_endpoint` with `source` (best-effort: a failure is
/// logged at `warn` and does NOT abort the loop). The reconcile-removal state is
/// then seeded from
/// `sink.account_peer_ids()` — not empty — so a peer that left the account while
/// this loop was not running (a live `set_enabled(false -> true)` restart, not a
/// full process restart) is still removed on the loop's first tick rather than
/// becoming a zombie entry no future diff ever catches (`sink.account_peer_ids`'s
/// doc; `remove_account_peer` clears on success only, so a missed removal never
/// self-heals).
///
/// Each `interval` tick (the first immediate, `MissedTickBehavior::Delay`
/// thereafter): fetches the account's endpoint list (a failure is logged at
/// `warn` and retried next tick); removes every previously-seen `Account` id
/// absent from the new list; then, for each non-self peer in the list, upserts
/// it and — only for a genuinely new, non-suppressed id — first-contact-dials it
/// via [`RefreshSink::on_new_peer`] under the loop's cancel scope (a cancel
/// during the dial ends the loop immediately, dropping the in-flight dial).
pub async fn run_account_refresh_loop_v2(
    source: Arc<dyn AccountEndpointSource>,
    self_endpoint: AccountEndpoint,
    interval: Duration,
    cancel: CancellationToken,
    sink: Arc<dyn RefreshSink>,
) {
    if let Err(e) = source.register_self(&self_endpoint).await {
        tracing::warn!(
            target: "sync",
            error = %e,
            "registering this device's endpoint with the account service failed; continuing unregistered"
        );
    }

    let mut last: HashSet<String> = sink.account_peer_ids().into_iter().collect();

    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticker.tick() => {
                let list = match source.list_endpoints().await {
                    Ok(list) => list,
                    Err(e) => {
                        tracing::warn!(
                            target: "sync",
                            error = %e,
                            "fetching the account's endpoint list failed; retrying next tick"
                        );
                        continue;
                    }
                };

                let peers = peers_to_add(&list, &self_endpoint.endpoint_id);
                let current: HashSet<String> =
                    peers.iter().map(|ep| ep.endpoint_id.clone()).collect();

                for departed in last.difference(&current) {
                    sink.remove_account_peer(departed);
                }

                for ep in &peers {
                    let was_new = sink.upsert_account_peer(ep);
                    if was_new && !sink.is_suppressed(&ep.endpoint_id) {
                        tokio::select! {
                            _ = cancel.cancelled() => return,
                            _ = sink.on_new_peer(&ep.endpoint_id) => {}
                        }
                    }
                }

                last = current;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio::sync::Notify;
    use tokio::time::timeout;

    fn ep(device: &str, endpoint: &str) -> AccountEndpoint {
        AccountEndpoint {
            device_id: device.to_string(),
            endpoint_id: endpoint.to_string(),
            relay_url: "https://sync.example/relay".to_string(),
        }
    }

    // ----- peers_to_add (pure) -----

    #[test]
    fn peers_to_add_filters_self() {
        let list = vec![ep("me", "self-ep"), ep("other", "peer-ep")];
        let result = peers_to_add(&list, "self-ep");
        assert_eq!(result, vec![ep("other", "peer-ep")]);
    }

    #[test]
    fn peers_to_add_dedups_by_endpoint_id() {
        let list = vec![
            ep("other-a", "peer-ep"),
            ep("other-b", "peer-ep"), // same endpoint id, different device id
        ];
        let result = peers_to_add(&list, "self-ep");
        assert_eq!(result, vec![ep("other-a", "peer-ep")]);
    }

    #[test]
    fn peers_to_add_keeps_multiple_distinct_others() {
        let list = vec![ep("me", "self-ep"), ep("a", "ep-a"), ep("b", "ep-b")];
        let result = peers_to_add(&list, "self-ep");
        assert_eq!(result, vec![ep("a", "ep-a"), ep("b", "ep-b")]);
    }

    #[test]
    fn peers_to_add_empty_list_is_empty() {
        assert!(peers_to_add(&[], "self-ep").is_empty());
    }

    // ----- shared AccountEndpointSource mock -----

    struct MockSource {
        list: Vec<AccountEndpoint>,
        register_calls: AtomicUsize,
        list_calls: AtomicUsize,
        /// When true, the NEXT `list_endpoints` call fails once, then clears.
        fail_next_list: AtomicBool,
    }

    #[async_trait::async_trait]
    impl AccountEndpointSource for MockSource {
        async fn list_endpoints(&self) -> AppResult<Vec<AccountEndpoint>> {
            self.list_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_next_list.swap(false, Ordering::SeqCst) {
                return Err(minutist_common::AppError::Internal {
                    context: "mock list_endpoints failure".to_string(),
                });
            }
            Ok(self.list.clone())
        }

        async fn register_self(&self, _endpoint: &AccountEndpoint) -> AppResult<()> {
            self.register_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    // ----- run_account_refresh_loop_v2 -----

    /// Records every call the loop makes, with configurable seed/suppressed
    /// state. `known` tracks which ids have already been upserted (seeded from
    /// [`Self::seed`]), so [`RefreshSink::upsert_account_peer`]'s was-new
    /// bookkeeping behaves like the real [`crate::address_lookup::PeerDirectory`].
    struct MockRefreshSink {
        known: Mutex<HashSet<String>>,
        suppressed: HashSet<String>,
        seed: Vec<String>,
        upserts: Mutex<Vec<String>>,
        removes: Mutex<Vec<String>>,
        on_new_peer_calls: Mutex<Vec<String>>,
        /// When set, `on_new_peer` blocks on this until notified, so a test can
        /// prove an in-flight dial is cancelled rather than awaited to
        /// completion.
        block_on_new_peer: Option<Arc<Notify>>,
    }

    impl MockRefreshSink {
        fn new() -> Self {
            Self {
                known: Mutex::new(HashSet::new()),
                suppressed: HashSet::new(),
                seed: Vec::new(),
                upserts: Mutex::new(Vec::new()),
                removes: Mutex::new(Vec::new()),
                on_new_peer_calls: Mutex::new(Vec::new()),
                block_on_new_peer: None,
            }
        }

        fn with_seed(mut self, seed: Vec<&str>) -> Self {
            self.seed = seed.into_iter().map(String::from).collect();
            self
        }

        fn with_suppressed(mut self, suppressed: &[&str]) -> Self {
            self.suppressed = suppressed.iter().map(|s| s.to_string()).collect();
            self
        }

        fn with_blocking_on_new_peer(mut self, notify: Arc<Notify>) -> Self {
            self.block_on_new_peer = Some(notify);
            self
        }
    }

    #[async_trait::async_trait]
    impl RefreshSink for MockRefreshSink {
        fn upsert_account_peer(&self, ep: &AccountEndpoint) -> bool {
            self.upserts.lock().unwrap().push(ep.endpoint_id.clone());
            self.known.lock().unwrap().insert(ep.endpoint_id.clone())
        }

        fn remove_account_peer(&self, endpoint_id: &str) {
            self.removes.lock().unwrap().push(endpoint_id.to_string());
            self.known.lock().unwrap().remove(endpoint_id);
        }

        fn is_suppressed(&self, endpoint_id: &str) -> bool {
            self.suppressed.contains(endpoint_id)
        }

        async fn on_new_peer(&self, endpoint_id: &str) {
            self.on_new_peer_calls
                .lock()
                .unwrap()
                .push(endpoint_id.to_string());
            if let Some(notify) = &self.block_on_new_peer {
                notify.notified().await;
            }
        }

        fn account_peer_ids(&self) -> Vec<String> {
            let mut known = self.known.lock().unwrap();
            for id in &self.seed {
                known.insert(id.clone());
            }
            self.seed.clone()
        }
    }

    fn mock_source(list: Vec<AccountEndpoint>) -> Arc<MockSource> {
        Arc::new(MockSource {
            list,
            register_calls: AtomicUsize::new(0),
            list_calls: AtomicUsize::new(0),
            fail_next_list: AtomicBool::new(false),
        })
    }

    /// (a) A non-self peer is upserted, and being genuinely new (was-new) drives
    /// exactly one `on_new_peer` dial-kick even across several ticks (the mock's
    /// `known` set makes every later upsert not-new).
    #[tokio::test]
    async fn refresh_loop_v2_upserts_non_self_peers_and_dials_new_ones() {
        let source = mock_source(vec![ep("me", "self-ep"), ep("peer", "peer-ep")]);
        let sink = Arc::new(MockRefreshSink::new());
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(run_account_refresh_loop_v2(
            source,
            ep("me", "self-ep"),
            Duration::from_millis(5),
            cancel.clone(),
            Arc::clone(&sink) as Arc<dyn RefreshSink>,
        ));

        tokio::time::sleep(Duration::from_millis(40)).await;
        cancel.cancel();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly after cancel")
            .expect("loop task must not panic");

        let upserts = sink.upserts.lock().unwrap();
        assert!(!upserts.is_empty());
        assert!(upserts.iter().all(|id| id == "peer-ep"));
        assert_eq!(
            sink.on_new_peer_calls.lock().unwrap().as_slice(),
            ["peer-ep"],
            "on_new_peer must fire exactly once, on the first genuinely-new upsert"
        );
    }

    /// (b) A genuinely new peer that is already suppressed is upserted but never
    /// dial-kicked.
    #[tokio::test]
    async fn refresh_loop_v2_skips_dial_for_a_suppressed_new_peer() {
        let source = mock_source(vec![ep("me", "self-ep"), ep("peer", "peer-ep")]);
        let sink = Arc::new(MockRefreshSink::new().with_suppressed(&["peer-ep"]));
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(run_account_refresh_loop_v2(
            source,
            ep("me", "self-ep"),
            Duration::from_millis(5),
            cancel.clone(),
            Arc::clone(&sink) as Arc<dyn RefreshSink>,
        ));

        tokio::time::sleep(Duration::from_millis(30)).await;
        cancel.cancel();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly after cancel")
            .expect("loop task must not panic");

        assert!(
            !sink.upserts.lock().unwrap().is_empty(),
            "a suppressed peer is still upserted into the directory"
        );
        assert!(
            sink.on_new_peer_calls.lock().unwrap().is_empty(),
            "a suppressed peer must never be dial-kicked"
        );
    }

    /// (c) An `Account`-sourced id seeded via `account_peer_ids` that is absent
    /// from the first list fetch is reconcile-removed on that first tick.
    #[tokio::test]
    async fn refresh_loop_v2_reconcile_removes_a_seeded_peer_absent_from_the_list() {
        let source = mock_source(vec![ep("me", "self-ep")]); // "stale-ep" is gone
        let sink = Arc::new(MockRefreshSink::new().with_seed(vec!["stale-ep"]));
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(run_account_refresh_loop_v2(
            source,
            ep("me", "self-ep"),
            Duration::from_millis(5),
            cancel.clone(),
            Arc::clone(&sink) as Arc<dyn RefreshSink>,
        ));

        tokio::time::sleep(Duration::from_millis(30)).await;
        cancel.cancel();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly after cancel")
            .expect("loop task must not panic");

        assert_eq!(
            sink.removes.lock().unwrap().as_slice(),
            ["stale-ep"],
            "a seeded account peer absent from the list must be reconcile-removed"
        );
    }

    /// (d) A cancel raised before the first tick stops the loop promptly.
    #[tokio::test]
    async fn refresh_loop_v2_stops_promptly_on_cancel_before_any_tick() {
        let source = mock_source(vec![]);
        let sink = Arc::new(MockRefreshSink::new());
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(run_account_refresh_loop_v2(
            source,
            ep("me", "self-ep"),
            Duration::from_secs(3600), // long enough that a tick never fires first
            cancel.clone(),
            Arc::clone(&sink) as Arc<dyn RefreshSink>,
        ));

        cancel.cancel();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly on cancel even before the first tick")
            .expect("loop task must not panic");
    }

    /// (e) A cancel raised while `on_new_peer` is in-flight (blocked forever,
    /// never notified) still stops the loop promptly — the dial is dropped, not
    /// awaited to completion.
    #[tokio::test]
    async fn refresh_loop_v2_cancel_interrupts_an_in_flight_on_new_peer_dial() {
        let source = mock_source(vec![ep("me", "self-ep"), ep("peer", "peer-ep")]);
        let block = Arc::new(Notify::new());
        let sink = Arc::new(MockRefreshSink::new().with_blocking_on_new_peer(Arc::clone(&block)));
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(run_account_refresh_loop_v2(
            source,
            ep("me", "self-ep"),
            Duration::from_millis(5),
            cancel.clone(),
            Arc::clone(&sink) as Arc<dyn RefreshSink>,
        ));

        // Let the loop tick and enter `on_new_peer`'s blocking wait (never
        // notified — `block` is deliberately never signalled).
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !sink.on_new_peer_calls.lock().unwrap().is_empty(),
            "on_new_peer must have been entered before cancel"
        );
        cancel.cancel();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly even with an in-flight on_new_peer dial")
            .expect("loop task must not panic");
    }
}
