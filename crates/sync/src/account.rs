//! Account-mediated peer discovery (`planning/DESIGN_account-peer-source.md`,
//! "B2").
//!
//! Manual pairing (a shared [`EndpointTicket`](iroh_tickets::endpoint::EndpointTicket))
//! is one way a peer's [`iroh::EndpointAddr`] reaches [`crate::SyncEngine::add_peer`].
//! This module adds a second, account-mediated one: an injected
//! [`AccountEndpointSource`] periodically fetches the other devices on the
//! signed-in account and feeds each into [`run_account_refresh_loop`]'s caller-
//! supplied `add_peer` closure. The two sources are additive — both feed the same
//! [`crate::address_lookup::PeerDirectory`].
//!
//! `crates/sync` takes **no** HTTP or account-service dependency of its own: the
//! consumer (app-main / sync-ffi) supplies an [`AccountEndpointSource`]
//! implementation bound to the device's account credential. This module defines
//! only the trait, the pure reconciliation logic, and the loop that drives it —
//! mirroring the `election` crate's collaborator-behind-a-trait seam.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use minutist_common::AppResult;
use tokio::sync::Notify;

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

/// Run the account-peer-discovery loop until `stop` is notified.
///
/// On start, registers `self_endpoint` with `source` (best-effort: a failure is
/// logged at `warn` and does NOT abort the loop — a device that cannot reach the
/// account service yet should still try to discover and dial the peers it can).
/// Then, on every `interval` tick, fetches the account's endpoint list and calls
/// `add_peer` for each of [`peers_to_add`]. A `list_endpoints` failure is logged
/// at `warn` and retried on the next tick; it never kills the loop.
///
/// The caller MUST supply `stop` (a fresh, not-yet-notified [`Notify`]) and MUST
/// hold a handle to notify it when this loop should end — this loop never
/// creates its own stop handle, so the spawner controls the cancellation seam
/// (the desktop wires it onto the same shared token as the local peers-file poll,
/// so a sync engine re-bind cannot leak this task — `DESIGN_account-peer-source.md`
/// "Design-review refinements").
///
/// `add_peer` is a plain closure (not a `SyncEngine` method) so the loop is
/// testable without a live engine: production wires it to
/// `SyncEngine::add_account_peer`; a test records calls instead.
pub async fn run_account_refresh_loop(
    source: Arc<dyn AccountEndpointSource>,
    self_endpoint: AccountEndpoint,
    interval: Duration,
    stop: Arc<Notify>,
    mut add_peer: impl FnMut(&AccountEndpoint) + Send,
) {
    if let Err(e) = source.register_self(&self_endpoint).await {
        tracing::warn!(
            target: "sync",
            error = %e,
            "registering this device's endpoint with the account service failed; continuing unregistered"
        );
    }

    let mut ticker = tokio::time::interval(interval);
    loop {
        tokio::select! {
            _ = stop.notified() => return,
            _ = ticker.tick() => {
                match source.list_endpoints().await {
                    Ok(list) => {
                        for peer in peers_to_add(&list, &self_endpoint.endpoint_id) {
                            add_peer(&peer);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "sync",
                            error = %e,
                            "fetching the account's endpoint list failed; retrying next tick"
                        );
                    }
                }
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

    // ----- run_account_refresh_loop -----

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

    #[tokio::test]
    async fn refresh_loop_registers_adds_non_self_peers_and_tolerates_a_fetch_error() {
        let source = Arc::new(MockSource {
            list: vec![
                ep("me", "self-ep"),
                ep("peer", "peer-ep"),
                ep("peer-dup", "peer-ep"), // duplicate endpoint id
            ],
            register_calls: AtomicUsize::new(0),
            list_calls: AtomicUsize::new(0),
            fail_next_list: AtomicBool::new(true), // the first tick's fetch fails
        });
        let self_endpoint = ep("me", "self-ep");
        let stop = Arc::new(Notify::new());
        let added: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let added_cl = Arc::clone(&added);
        let stop_cl = Arc::clone(&stop);
        let source_cl = Arc::clone(&source);
        let handle = tokio::spawn(run_account_refresh_loop(
            source_cl,
            self_endpoint,
            Duration::from_millis(5),
            stop_cl,
            move |peer: &AccountEndpoint| {
                added_cl.lock().unwrap().push(peer.endpoint_id.clone());
            },
        ));

        // Let several ticks elapse (the first fails, tolerated; later ones
        // succeed), then stop the loop.
        tokio::time::sleep(Duration::from_millis(40)).await;
        stop.notify_one();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly after stop is notified")
            .expect("loop task must not panic");

        assert_eq!(
            source.register_calls.load(Ordering::SeqCst),
            1,
            "register_self is called exactly once, at start"
        );
        assert!(
            source.list_calls.load(Ordering::SeqCst) >= 2,
            "at least the failing first tick and one successful later tick must have fetched"
        );
        let added = added.lock().unwrap();
        assert!(
            !added.is_empty(),
            "a successful tick must have added the non-self peer at least once"
        );
        assert!(
            added.iter().all(|id| id == "peer-ep"),
            "only the de-duplicated non-self endpoint id is ever added, got {added:?}"
        );
    }

    #[tokio::test]
    async fn refresh_loop_logs_register_self_failure_but_still_runs() {
        struct FailingRegister {
            list: Vec<AccountEndpoint>,
        }
        #[async_trait::async_trait]
        impl AccountEndpointSource for FailingRegister {
            async fn list_endpoints(&self) -> AppResult<Vec<AccountEndpoint>> {
                Ok(self.list.clone())
            }
            async fn register_self(&self, _endpoint: &AccountEndpoint) -> AppResult<()> {
                Err(minutist_common::AppError::Internal {
                    context: "mock register_self failure".to_string(),
                })
            }
        }

        let source = Arc::new(FailingRegister {
            list: vec![ep("me", "self-ep"), ep("peer", "peer-ep")],
        });
        let self_endpoint = ep("me", "self-ep");
        let stop = Arc::new(Notify::new());
        let added: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let added_cl = Arc::clone(&added);
        let stop_cl = Arc::clone(&stop);
        let handle = tokio::spawn(run_account_refresh_loop(
            source,
            self_endpoint,
            Duration::from_millis(5),
            stop_cl,
            move |peer: &AccountEndpoint| {
                added_cl.lock().unwrap().push(peer.endpoint_id.clone());
            },
        ));

        tokio::time::sleep(Duration::from_millis(20)).await;
        stop.notify_one();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly after stop is notified")
            .expect("loop task must not panic");

        assert!(
            !added.lock().unwrap().is_empty(),
            "the loop must still reconcile peers despite a register_self failure"
        );
    }

    #[tokio::test]
    async fn refresh_loop_stops_promptly_on_notify_before_any_tick() {
        let source = Arc::new(MockSource {
            list: vec![],
            register_calls: AtomicUsize::new(0),
            list_calls: AtomicUsize::new(0),
            fail_next_list: AtomicBool::new(false),
        });
        let self_endpoint = ep("me", "self-ep");
        let stop = Arc::new(Notify::new());

        let stop_cl = Arc::clone(&stop);
        let handle = tokio::spawn(run_account_refresh_loop(
            source,
            self_endpoint,
            Duration::from_secs(3600), // long enough that a tick never fires first
            stop_cl,
            |_peer: &AccountEndpoint| {},
        ));

        stop.notify_one();
        timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop must exit promptly on stop even before the first tick")
            .expect("loop task must not panic");
    }
}
