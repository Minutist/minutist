//! Out-of-band peer addressing.
//!
//! Minutist learns a peer's `EndpointId` and reachability (relay URL and/or direct
//! addresses) from the account service, not from DNS/pkarr discovery. iroh 1.0's
//! [`MemoryLookup`] (the successor to 0.x's `StaticProvider`) holds those manually
//! injected entries and satisfies the endpoint's address-lookup contract.
//!
//! [`PeerDirectory`] is a thin wrapper that keeps the naming domain-specific and
//! hides the iroh type from the rest of the crate. The wrapped [`MemoryLookup`] is
//! internally shared and cheap to clone: the same backing store is registered on
//! the endpoint and held here, so a peer added after binding is visible to the
//! next dial.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use iroh::address_lookup::MemoryLookup;
use iroh::{EndpointAddr, EndpointId};

/// How a [`PeerDirectory`] entry was learned, so eviction can stay path-specific.
///
/// Only `Account`-sourced entries are subject to account-reconcile removal (an
/// account peer that drops out of the account's device list); a `Manual` entry
/// (ticket pairing, or the relay-less direct test path) is never touched by
/// that reconcile. `Manual` lumps those two together because they share the
/// same removal semantics: neither is account-reconciled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerSource {
    /// Learned from the account service's device-directory (`crate::account`).
    Account,
    /// Learned any other way: ticket pairing, or the relay-less direct test
    /// path.
    Manual,
}

/// The tracked state of one directory peer: how it was learned, and the exact
/// [`EndpointAddr`] currently registered for it (so [`PeerDirectory::add`] can
/// detect a changed address and apply replace, not union, semantics).
#[derive(Debug, Clone)]
struct Entry {
    source: PeerSource,
    addr: EndpointAddr,
}

/// Holds peers learned out-of-band so the endpoint can resolve and dial them.
///
/// The wrapped [`MemoryLookup`] resolves addresses for the endpoint but exposes
/// no way to enumerate the ids it holds, so a parallel `id -> Entry` map tracks
/// them alongside it: [`Self::ids`] iterates the registered peers, and the
/// [`PeerSource`] tag lets [`Self::remove`] stay source-aware. The tracked
/// [`EndpointAddr`] gives [`Self::add`] replace semantics over
/// [`MemoryLookup`]'s merge. Both the lookup and the map are internally shared
/// and cheap to clone, so an entry added through any clone is visible to all
/// of them.
#[derive(Debug, Clone, Default)]
pub struct PeerDirectory {
    lookup: MemoryLookup,
    entries: Arc<Mutex<BTreeMap<EndpointId, Entry>>>,
}

impl PeerDirectory {
    /// An empty directory. Peers are added later via [`Self::add`].
    pub fn new() -> Self {
        Self {
            lookup: MemoryLookup::new(),
            entries: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Register a peer's full address (id + relay/direct addrs), tagged with how
    /// it was learned. Overwrites any prior entry (and its source tag) for the
    /// same `EndpointId`. Returns whether the id was newly added (`true`) or
    /// already present (`false`), so a caller can distinguish a genuinely new
    /// peer from a re-advertised one (e.g. to gate a first-contact dial).
    ///
    /// **Replace, not union.** [`MemoryLookup::add_endpoint_info`] merges the
    /// address set for an already-known id: it never drops an address. A peer
    /// whose advertised addrs change between polls (direct addrs first
    /// appearing after an initial relay-only entry, or a new ephemeral UDP
    /// port after a restart) would otherwise accumulate every address it has
    /// ever advertised, including dead ones. Those stale entries are extra
    /// dial candidates iroh probes and abandons, aggravating the path churn
    /// that starves its per-remote actor. We clear the lookup entry first
    /// when the address actually changed, so the registered set is exactly
    /// the latest advert; an unchanged re-advert is a no-op (no needless
    /// churn every poll tick).
    pub fn add(&self, addr: EndpointAddr, source: PeerSource) -> bool {
        let id = addr.id;
        let mut entries = self.entries.lock().expect("peer directory entries poisoned");
        let changed = match entries.get(&id) {
            Some(existing) => existing.addr != addr || existing.source != source,
            None => true,
        };
        let was_new = !entries.contains_key(&id);
        if changed {
            if !was_new {
                self.lookup.remove_endpoint_info(id);
            }
            // `EndpointAddr` converts into `EndpointInfo` via `From`, which is
            // what `add_endpoint_info` accepts.
            self.lookup.add_endpoint_info(addr.clone());
            entries.insert(id, Entry { source, addr });
        }
        was_new
    }

    /// Remove `id` only if its current tag equals `source`. A mismatched or
    /// absent id is a no-op. Returns whether an entry was removed.
    ///
    /// This keeps account-reconcile removal (`source = Account`) from evicting
    /// a manually-paired peer that happens to share an id, and vice versa.
    pub fn remove(&self, id: EndpointId, source: PeerSource) -> bool {
        let mut entries = self
            .entries
            .lock()
            .expect("peer directory entries poisoned");
        match entries.get(&id) {
            Some(entry) if entry.source == source => {
                entries.remove(&id);
                drop(entries);
                self.lookup.remove_endpoint_info(id);
                true
            }
            _ => false,
        }
    }

    /// The `EndpointId`s of every registered peer, regardless of source. Used by
    /// the connected `SyncControl` to reconcile a meeting against each known
    /// device.
    pub fn ids(&self) -> Vec<EndpointId> {
        self.entries
            .lock()
            .expect("peer directory entries poisoned")
            .keys()
            .copied()
            .collect()
    }

    /// The hex ids of every `Account`-sourced entry. Used to seed
    /// [`crate::account::run_account_refresh_loop_v2`]'s reconcile-removal state
    /// on (re)start, so a live loop restart does not treat every currently-known
    /// account peer as newly arrived (see the loop's doc comment).
    pub fn account_peer_ids(&self) -> Vec<String> {
        self.entries
            .lock()
            .expect("peer directory entries poisoned")
            .iter()
            .filter(|(_, entry)| entry.source == PeerSource::Account)
            .map(|(id, _)| id.to_string())
            .collect()
    }

    /// The backing lookup to register on the iroh endpoint builder. Cloning shares
    /// the same store, so entries added through [`Self::add`] are seen by the
    /// registered copy.
    pub(crate) fn lookup(&self) -> MemoryLookup {
        self.lookup.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn id() -> EndpointId {
        iroh::SecretKey::generate().public()
    }

    #[test]
    fn add_reports_new_once_then_tracks_address_changes_and_removes_by_source() {
        let dir = PeerDirectory::new();
        let peer = id();
        let relay: iroh::RelayUrl = "https://relay.example/".parse().unwrap();

        // First advert (relay-only) is genuinely new.
        let relay_only = EndpointAddr::new(peer).with_relay_url(relay.clone());
        assert!(dir.add(relay_only.clone(), PeerSource::Account), "first add is new");
        // Identical re-advert is not new and is a lookup no-op.
        assert!(!dir.add(relay_only, PeerSource::Account), "same addr re-add is not new");

        // A later advert that gains a direct addr is not "new" (same id) but the
        // tracked address is updated in place (replace, not union).
        let sock: SocketAddr = "100.82.58.55:41641".parse().unwrap();
        let with_direct = EndpointAddr::new(peer)
            .with_relay_url(relay)
            .with_ip_addr(sock);
        assert!(
            !dir.add(with_direct.clone(), PeerSource::Account),
            "changed addr re-add is not new"
        );
        assert_eq!(
            dir.entries.lock().unwrap().get(&peer).unwrap().addr,
            with_direct,
            "the tracked addr reflects the latest advert"
        );
        assert_eq!(dir.ids(), vec![peer]);
        assert_eq!(dir.account_peer_ids(), vec![peer.to_string()]);

        // Source-aware remove: a mismatched source is a no-op; the matching one evicts.
        assert!(!dir.remove(peer, PeerSource::Manual), "wrong source is a no-op");
        assert!(dir.remove(peer, PeerSource::Account), "matching source removes");
        assert!(dir.ids().is_empty());
    }
}
