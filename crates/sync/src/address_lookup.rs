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

use iroh::address_lookup::MemoryLookup;
use iroh::EndpointAddr;

/// Holds peers learned out-of-band so the endpoint can resolve and dial them.
#[derive(Debug, Clone, Default)]
pub struct PeerDirectory {
    lookup: MemoryLookup,
}

impl PeerDirectory {
    /// An empty directory. Peers are added later via [`Self::add`].
    pub fn new() -> Self {
        Self {
            lookup: MemoryLookup::new(),
        }
    }

    /// Register a peer's full address (id + relay/direct addrs). Overwrites any
    /// prior entry for the same `EndpointId`.
    pub fn add(&self, addr: EndpointAddr) {
        // `EndpointAddr` converts into `EndpointInfo` via `From`, which is what
        // `add_endpoint_info` accepts.
        self.lookup.add_endpoint_info(addr);
    }

    /// The backing lookup to register on the iroh endpoint builder. Cloning shares
    /// the same store, so entries added through [`Self::add`] are seen by the
    /// registered copy.
    pub(crate) fn lookup(&self) -> MemoryLookup {
        self.lookup.clone()
    }
}
