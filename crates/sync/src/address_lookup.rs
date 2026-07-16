//! Out-of-band peer addressing.
//!
//! Minutist learns a peer's `EndpointId` and reachability (relay URL and/or direct
//! addresses) from the account service, not from DNS/pkarr discovery. iroh 1.0's
//! [`MemoryLookup`] (the successor to 0.x's `StaticProvider`) holds those manually
//! injected entries and satisfies the endpoint's address-lookup contract.
//!
//! [`PeerDirectory`] is a thin wrapper that keeps the naming domain-specific and
//! hides the iroh type from the rest of the crate. The wrapped [`MemoryLookup`] is
//! internally shared and cheap to clone — the same backing store is registered on
//! the endpoint and held here, so a peer added after binding is visible to the
//! next dial.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use iroh::address_lookup::MemoryLookup;
use iroh::{EndpointAddr, EndpointId};

/// How a [`PeerDirectory`] entry was learned, so eviction can stay path-specific.
///
/// Only `Account`-sourced entries are subject to account-reconcile removal (an
/// account peer that drops out of the account's device list) — a `Manual` entry
/// (ticket pairing, a peers-file, or the relay-less direct test path) is never
/// touched by that reconcile. `Manual` deliberately lumps ticket/peers-file/direct
/// pairing together: they share removal semantics this batch (none of them are
/// account-reconciled), so a finer split is a future additive change if a
/// per-source removal is ever needed for one of them specifically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerSource {
    /// Learned from the account service's device-directory (`crate::account`).
    Account,
    /// Learned any other way: ticket pairing, a peers file, or the relay-less
    /// direct test path.
    Manual,
}

/// Holds peers learned out-of-band so the endpoint can resolve and dial them.
///
/// The wrapped [`MemoryLookup`] resolves addresses for the endpoint but exposes
/// no way to enumerate the ids it holds, so a parallel `id -> source` map is kept
/// alongside it: [`Self::ids`] iterates the registered peers to sync against
/// each, and the [`PeerSource`] tag lets [`Self::remove`] stay source-aware (an
/// account-reconcile removal must never evict a manually-paired peer, and vice
/// versa). Both the lookup and the map are internally shared and cheap to clone,
/// so an entry added through any clone is visible to all of them.
#[derive(Debug, Clone, Default)]
pub struct PeerDirectory {
    lookup: MemoryLookup,
    entries: Arc<Mutex<BTreeMap<EndpointId, PeerSource>>>,
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
    pub fn add(&self, addr: EndpointAddr, source: PeerSource) -> bool {
        let was_new = self
            .entries
            .lock()
            .expect("peer directory entries poisoned")
            .insert(addr.id, source)
            .is_none();
        // `EndpointAddr` converts into `EndpointInfo` via `From`, which is what
        // `add_endpoint_info` accepts.
        self.lookup.add_endpoint_info(addr);
        was_new
    }

    /// Remove `id` if — and only if — its current tag equals `source`. A
    /// mismatched or absent id is a no-op. Returns whether an entry was removed.
    ///
    /// This is the invariant that keeps the two eviction paths (account-reconcile
    /// vs a future per-source removal) from clobbering each other's peers: an
    /// account-reconcile removal (`source = Account`) can never evict a
    /// manually-paired peer that happens to share an id, and vice versa.
    pub fn remove(&self, id: EndpointId, source: PeerSource) -> bool {
        let mut entries = self
            .entries
            .lock()
            .expect("peer directory entries poisoned");
        match entries.get(&id) {
            Some(tag) if *tag == source => {
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
            .filter(|(_, source)| **source == PeerSource::Account)
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
