//! The iroh endpoint that owns device-to-device sync.
//!
//! [`SyncEngine`] holds a single [`iroh::Endpoint`] bound with:
//!
//! - the device [`DeviceIdentity`]'s secret key,
//! - a relay mode pinning the self-hosted relay ([`SyncConfig::relay_url`]) via
//!   `RelayMode::Custom`, carrying the relay access token when one is configured
//!   ([`SyncConfig::relay_auth_token`]),
//! - a [`MemoryLookup`] for out-of-band peer addressing (peers learned from the
//!   account service rather than DNS/pkarr discovery — the iroh 1.0 successor to
//!   0.x's `StaticProvider`),
//! - TWO ALPNs accepted on one iroh [`Router`]: the sync-update protocol
//!   ([`crate::notes_proto::SYNC_ALPN`], carrying both notes and media-manifest
//!   exchanges, dispatched by a leading [`crate::notes_proto::StreamKind`] tag),
//!   and the blobs protocol ([`iroh_blobs::ALPN`], moving media bytes).
//!
//! The inbound accept side is owned by the [`Router`]. The sync ALPN dispatches
//! to [`AcceptHook`]; the blobs ALPN to [`AuthorizedBlobs`]. BOTH first authorise
//! the remote against the paired-peer [`PeerDirectory`] (sync requires MUTUAL
//! pairing — each device adds the other's ticket), rejecting an unpaired peer
//! before any frame or blob request is served:
//!
//! - [`AcceptHook`] reads the leading stream-kind tag and runs the responder side
//!   of either the notes-sync ([`crate::notes_proto::respond_notes_sync`]) or the
//!   media-manifest ([`crate::media_proto::respond_media_sync`]) protocol.
//! - [`AuthorizedBlobs`] wraps [`iroh_blobs::BlobsProtocol`] and delegates to it
//!   only for a paired remote — closing the same hole [`AcceptHook`] closes, on
//!   the new ALPN: the blobs protocol on its own serves any peer that connects,
//!   so it must NEVER be registered unguarded.
//!
//! `SyncEngine::dial` (crate-internal) dials a peer; [`SyncEngine::sync_notes`] /
//! [`SyncEngine::sync_media`] / [`SyncEngine::sync_artifacts`] dial and run the
//! initiator side for one meeting.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use iroh::endpoint::presets;
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{
    endpoint::Connection, Endpoint, EndpointAddr, EndpointId, RelayConfig, RelayMap, RelayMode,
    RelayUrl,
};
use iroh_tickets::endpoint::EndpointTicket;
use minutist_common::{MeetingId, ProcessingLifecycle};
use tokio::sync::broadcast;

use crate::account::{AccountEndpoint, RefreshSink};
use crate::address_lookup::{PeerDirectory, PeerSource};
#[cfg(feature = "test-support")]
use crate::backoff::BackoffPolicy;
use crate::backoff::BackoffRegistry;
use crate::blobs::BlobStore;
use crate::identity::DeviceIdentity;
use crate::notes_proto::{self, StreamKind, SYNC_ALPN};
use crate::timeouts::ACCEPT_HANDSHAKE_TIMEOUT;
use crate::{artifacts_proto, discovery_proto, media_proto, Error, Result, SyncConfig};

/// Owns the iroh endpoint, the out-of-band peer directory, the content-addressed
/// blob store, and the router that accepts inbound sync connections for one
/// device.
pub struct SyncEngine {
    endpoint: Endpoint,
    router: Router,
    peers: PeerDirectory,
    /// Failed-dial tracking for every peer this device dials, regardless of
    /// source ([`Self::dial`] feeds every outcome). Backs [`Self::is_suppressed`]
    /// and the account-refresh loop's dial-suppression check
    /// ([`SyncEngineRefreshSink::is_suppressed`]).
    backoff: BackoffRegistry,
    /// The content-addressed media-blob store for this device. Held here so the
    /// initiator side ([`Self::sync_media`]) can import/export/download, and kept
    /// alive for the lifetime of the router (the [`iroh_blobs::BlobsProtocol`]
    /// registered on it borrows from a clone of the inner store).
    blobs: BlobStore,
    /// Meetings root the protocols read/write through `persistence`
    /// (`{root}/{meeting_id}/...`). The inbound [`AcceptHook`] shares it.
    meetings_root: PathBuf,
    /// The configured relay URL, kept so [`Self::push_all_to`] can address a peer
    /// relay-only (id + relay) without a stored direct address.
    relay_url: String,
    /// Fires a peer's hex endpoint id at most once per [`PEER_ARRIVAL_DEBOUNCE`]
    /// window of accepted inbound connections from it ("peer arrived"). One sync
    /// "session" (e.g. the desktop's `sync_now` for a meeting) opens several
    /// connections in quick succession — notes, media, and discovery each dial
    /// separately — which [`PeerArrivalTracker`] coalesces into ONE event per
    /// visit rather than one per connection. An always-on hub subscribes via
    /// [`Self::subscribe_peer_events`] and reciprocally pushes (see
    /// [`Self::push_all_to`]); the desktop ignores it. Bounded — a lagging receiver
    /// drops the oldest ids, so the consumer is expected to recover on a
    /// [`broadcast::error::RecvError::Lagged`] by reconciling all known peers
    /// ([`Self::peer_ids`]); no visit is then permanently missed. Hex strings, not
    /// [`EndpointId`], so a consumer (including the mobile FFI wrapper) never
    /// needs an `iroh` type.
    peer_events: broadcast::Sender<String>,
    /// Fires each `(MeetingId, ProcessingLifecycle)` received from a discovery
    /// exchange ([`crate::discovery_proto`]). A consumer in a crate depending on
    /// both `sync` and `persistence` (ipc-bridge / headless) subscribes via
    /// [`Self::subscribe_lifecycle_events`] and persists each via
    /// `persistence::apply_processing_lifecycle` — `sync` has no `persistence`
    /// edge, so it emits rather than writes. Bounded; a consumer that hits
    /// [`broadcast::error::RecvError::Lagged`] recovers by re-running discovery.
    lifecycle_events: broadcast::Sender<(MeetingId, ProcessingLifecycle)>,
}

/// Capacity of the peer-arrived broadcast channel. Small by design: a lagging hub
/// subscriber drops the oldest ids, which the next inbound connection re-fires.
const PEER_EVENTS_CAP: usize = 64;

/// Capacity of the discovery lifecycle-event broadcast channel. Larger than the
/// peer-arrival channel because one discovery exchange emits one event per
/// meeting; a lagging consumer re-runs discovery to recover.
const LIFECYCLE_EVENTS_CAP: usize = 256;

/// Coalescing window for [`PeerArrivalTracker`]. One sync "session" against a
/// peer opens several connections back-to-back — the desktop's `sync_now` for a
/// single meeting dials notes, media, and discovery separately, and the hub's own
/// `push_all_to` does the same per meeting it holds — all normally landing within
/// low single-digit seconds of each other. A peer that has been accepted more
/// recently than this window does not re-fire "peer arrived"; one that has been
/// silent for longer than it is treated as a fresh visit. Deliberately much
/// shorter than the hub's own post-event push debounce
/// (`MINUTIST_HUB_PUSH_DEBOUNCE_MS`, default 15s in `headless`) — that one rate-
/// limits how often a genuinely-repeated arrival triggers a push; this one only
/// suppresses the redundant re-signalling of ONE arrival across its own burst of
/// connections.
const PEER_ARRIVAL_DEBOUNCE: std::time::Duration = std::time::Duration::from_secs(3);

/// Coalesces the burst of inbound connections one sync session opens (notes,
/// media, artifacts, and discovery each dial separately — see
/// [`SyncEngine::peer_events`]) into a single "peer arrived" notification per
/// visit, tracking each peer's most recently accepted-connection time. Shared
/// (cheap to clone) between [`SyncEngine`] and the [`AcceptHook`] the router
/// dispatches inbound connections to.
#[derive(Debug, Clone, Default)]
struct PeerArrivalTracker {
    last_seen: Arc<Mutex<HashMap<EndpointId, Instant>>>,
}

impl PeerArrivalTracker {
    fn new() -> Self {
        Self::default()
    }

    /// Record an accepted connection from `peer` and report whether the caller
    /// should treat it as a fresh arrival: `true` the first time `peer` is seen,
    /// or the first time again after [`PEER_ARRIVAL_DEBOUNCE`] of silence from it.
    fn note_connection(&self, peer: EndpointId) -> bool {
        let now = Instant::now();
        let mut last_seen = self
            .last_seen
            .lock()
            .expect("peer arrival tracker poisoned");
        let fire = match last_seen.get(&peer) {
            Some(seen) => now.duration_since(*seen) >= PEER_ARRIVAL_DEBOUNCE,
            None => true,
        };
        last_seen.insert(peer, now);
        fire
    }
}

/// A static [`iroh::dns::Resolver`] that answers only for the relay host, from IPs
/// the caller pre-resolved. iroh connects to the returned IP with the relay
/// hostname preserved as the TLS SNI, so the relay's hostname cert still verifies.
#[derive(Debug, Clone)]
struct StaticRelayResolver {
    /// The relay host these IPs resolve, lowercased and without a trailing dot.
    host: String,
    v4: Vec<std::net::Ipv4Addr>,
    v6: Vec<std::net::Ipv6Addr>,
}

impl StaticRelayResolver {
    fn matches(&self, host: &str) -> bool {
        host.trim_end_matches('.').eq_ignore_ascii_case(&self.host)
    }
}

impl iroh::dns::Resolver for StaticRelayResolver {
    fn lookup_ipv4(
        &self,
        host: String,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<iroh::dns::BoxIter<std::net::Ipv4Addr>, iroh::dns::DnsError>> + Send>>
    {
        let addrs: Vec<_> = if self.matches(&host) { self.v4.clone() } else { Vec::new() };
        Box::pin(async move { Ok(Box::new(addrs.into_iter()) as iroh::dns::BoxIter<_>) })
    }

    fn lookup_ipv6(
        &self,
        host: String,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<iroh::dns::BoxIter<std::net::Ipv6Addr>, iroh::dns::DnsError>> + Send>>
    {
        let addrs: Vec<_> = if self.matches(&host) { self.v6.clone() } else { Vec::new() };
        Box::pin(async move { Ok(Box::new(addrs.into_iter()) as iroh::dns::BoxIter<_>) })
    }

    fn lookup_txt(
        &self,
        _host: String,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<iroh::dns::BoxIter<iroh::dns::TxtRecordData>, iroh::dns::DnsError>> + Send>>
    {
        // Addressing is account-directory based, not pkarr/TXT discovery, so serve
        // no TXT records — a pkarr lookup simply finds nothing rather than erroring.
        Box::pin(async move { Ok(Box::new(std::iter::empty()) as iroh::dns::BoxIter<_>) })
    }

    fn clear_cache(&self) {}

    fn reset(&self) -> Box<dyn iroh::dns::Resolver> {
        Box::new(self.clone())
    }
}

/// The DNS resolver for Android, where iroh's default (system-config) resolver has
/// no nameservers to read — `/etc/resolv.conf` is absent and the netlink route
/// socket it falls back to is SELinux-denied for untrusted apps — so every in-app
/// lookup fails and the relay never resolves (see the note at the
/// [`SyncEngine::start`] call site).
///
/// Injecting nameservers to query does not survive a full-tunnel VPN (Tailscale
/// MagicDNS intercepts the app's raw UDP:53 to any resolver), so the caller instead
/// resolves the relay host through the OS resolver (which honours the VPN) and
/// passes the IPs as `relay_ips`. A [`StaticRelayResolver`] then serves them for the
/// relay host with no in-app DNS at all.
///
/// Falls back to Cloudflare DoH over `:443` only when no usable IP was injected — a
/// cellular safety net (its certs carry the IPs as SANs, so the IP doubles as the
/// TLS server name; DoH `:443` also survives carrier `:53` hijacking).
///
/// Compiled on every target so the host build type-checks the `iroh::dns` API even
/// though only the Android build calls it.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn android_relay_resolver(relay_host: &str, relay_ips: &[String]) -> iroh::dns::DnsResolver {
    use iroh::dns::DnsProtocol;
    use std::net::IpAddr;

    let mut v4 = Vec::new();
    let mut v6 = Vec::new();
    for ip in relay_ips {
        match ip.parse::<IpAddr>() {
            Ok(IpAddr::V4(a)) => v4.push(a),
            Ok(IpAddr::V6(a)) => v6.push(a),
            Err(e) => {
                tracing::warn!(ip = %ip, error = %e, "skipping unparseable injected relay IP");
            }
        }
    }
    if !v4.is_empty() || !v6.is_empty() {
        return iroh::dns::DnsResolver::custom(StaticRelayResolver {
            host: relay_host.trim_end_matches('.').to_ascii_lowercase(),
            v4,
            v6,
        });
    }

    // No usable injected relay IP → Cloudflare 1.1.1.1 / 1.0.0.1 DoH fallback.
    // Cellular safety net only; a network that blocks outbound 1.1.1.1 (or a
    // full-tunnel VPN) must inject the pre-resolved relay IPs above.
    tracing::info!("no usable injected relay IP; falling back to public DoH resolver");
    iroh::dns::DnsResolver::builder()
        .with_nameserver(
            "1.1.1.1:443".parse().expect("valid DoH nameserver socket addr"),
            DnsProtocol::Https,
        )
        .with_nameserver(
            "1.0.0.1:443".parse().expect("valid DoH nameserver socket addr"),
            DnsProtocol::Https,
        )
        .build()
}

/// Whether a held meeting is materialised enough for [`SyncEngine::adopt_from_peer`]
/// to skip re-syncing it: its authoritative notes CRDT (`notes.ydoc`), `metadata.json`,
/// and `audio.opus` are all present. A held-but-incomplete meeting — any of the three
/// missing (e.g. audio pulled but the notes pull failed on a prior sweep, or vice
/// versa) — fails this and is re-attempted, so adopt self-heals rather than stranding
/// it on folder existence alone. `audio.opus` stands in for media completeness; a
/// meeting genuinely without audio simply re-syncs each sweep — a cheap idempotent
/// manifest exchange, not a blob transfer.
fn meeting_is_materialised(meetings_root: &Path, meeting_id: MeetingId) -> bool {
    let dir = meetings_root.join(meeting_id.0.to_string());
    dir.join("notes.ydoc").is_file()
        && dir.join("metadata.json").is_file()
        && dir.join("audio.opus").is_file()
}

/// Whether a bound direct socket address is worth publishing to the account
/// directory for a peer to dial (0049).
///
/// Keeps addresses a peer on the same tailnet or LAN can plausibly reach —
/// Tailscale CGNAT (100.64.0.0/10), ordinary private LAN ranges (10/8,
/// 192.168/16), and global/public addresses. Drops what no off-host peer can
/// route to: loopback, link-local, unspecified, multicast/broadcast, and the
/// docker-bridge range 172.16.0.0/12 (a container host otherwise advertises
/// ~10 unreachable bridge addrs).
///
/// Heuristic tradeoff: a genuine 172.16/12 LAN would be dropped — accepted on a
/// docker-heavy host where that range is otherwise garbage; refine via
/// default-route interface detection if it ever bites. iroh treats direct
/// addrs opportunistically and falls back to the relay, so an over-filtered set
/// costs a relay hop, never connectivity.
fn is_publishable_direct_addr(addr: &std::net::SocketAddr) -> bool {
    match addr.ip() {
        std::net::IpAddr::V4(ip) => {
            if ip.is_loopback()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_broadcast()
                || ip.is_multicast()
            {
                return false;
            }
            // docker-bridge range 172.16.0.0/12 (172.16.x – 172.31.x).
            let o = ip.octets();
            !(o[0] == 172 && (16..=31).contains(&o[1]))
        }
        std::net::IpAddr::V6(ip) => {
            // fe80::/10 unicast link-local — `Ipv6Addr::is_unicast_link_local`
            // is unstable, so match the prefix directly.
            let is_link_local = (ip.segments()[0] & 0xffc0) == 0xfe80;
            !(ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() || is_link_local)
        }
    }
}

/// Whether `addr` is a Tailscale CGNAT address (100.64.0.0/10) — the marker that
/// this device is on a tailnet mesh (see [`SyncEngine::publishable_direct_addrs`]).
fn is_cgnat_v4(addr: &std::net::SocketAddr) -> bool {
    match addr.ip() {
        std::net::IpAddr::V4(ip) => {
            let o = ip.octets();
            o[0] == 100 && (64..=127).contains(&o[1])
        }
        std::net::IpAddr::V6(_) => false,
    }
}

/// Whether `addr` is an RFC1918 private-LAN address (10/8, 172.16/12, 192.168/16).
fn is_rfc1918_v4(addr: &std::net::SocketAddr) -> bool {
    match addr.ip() {
        std::net::IpAddr::V4(ip) => ip.is_private(),
        std::net::IpAddr::V6(_) => false,
    }
}

impl SyncEngine {
    /// Build the endpoint from `config` and the device `identity`, pinning the
    /// configured relay (with the access token when set), registering the
    /// out-of-band [`PeerDirectory`], and spawning the [`Router`] accept loop on
    /// the [`SYNC_ALPN`].
    ///
    /// Binding opens the QUIC socket; no peer dial happens here. The spawned
    /// router runs the inbound accept loop until [`Self::shutdown`].
    pub async fn start(config: SyncConfig, identity: DeviceIdentity) -> Result<Self> {
        let relay_mode = Self::relay_mode(&config)?;
        let peers = PeerDirectory::new();
        let meetings_root = config.meetings_root.clone();
        let blobs = BlobStore::open(&meetings_root).await?;

        let builder = Endpoint::builder(presets::N0)
            .secret_key(identity.secret_key())
            .relay_mode(relay_mode)
            .alpns(vec![SYNC_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()])
            .address_lookup(peers.lookup());
        // On Android, iroh's default (system-config) DNS resolver has NO
        // nameservers: `/etc/resolv.conf` is absent and the netlink route socket
        // it would fall back to is SELinux-denied for untrusted apps. So every
        // in-app lookup fails — the relay never resolves and the endpoint never
        // homes. Serve the relay from the caller's pre-resolved IPs
        // (`config.relay_ips`) via a static resolver keyed on the relay host
        // (empty → public DoH fallback; see [`android_relay_resolver`]). The relay
        // is the only hostname iroh resolves, and a static resolver survives a
        // full-tunnel VPN that would intercept a raw DNS query. Non-Android keeps
        // iroh's system resolver so the device's DNS (VPN / private DNS / corporate
        // / split-horizon) is honoured.
        #[cfg(target_os = "android")]
        let builder = {
            let relay_host = config
                .relay_url
                .parse::<RelayUrl>()
                .ok()
                .and_then(|u| u.host_str().map(str::to_string))
                .unwrap_or_default();
            builder.dns_resolver(android_relay_resolver(&relay_host, &config.relay_ips))
        };
        let endpoint = builder
            .bind()
            .await
            .map_err(|e| Error::Endpoint(format!("binding iroh endpoint: {e}")))?;

        tracing::info!(
            target: "sync",
            endpoint_id = %endpoint.id(),
            relay = %config.relay_url,
            "sync endpoint bound"
        );

        let (peer_events, _rx) = broadcast::channel(PEER_EVENTS_CAP);
        let (lifecycle_events, _lrx) = broadcast::channel(LIFECYCLE_EVENTS_CAP);
        // Owned solely by the router's `AcceptHook` — `SyncEngine` itself never
        // queries arrival state, only the events `AcceptHook` derives from it.
        let router = Self::build_router(
            &endpoint,
            &blobs,
            &peers,
            &meetings_root,
            peer_events.clone(),
            PeerArrivalTracker::new(),
            lifecycle_events.clone(),
        );

        Ok(Self {
            endpoint,
            router,
            peers,
            backoff: BackoffRegistry::new(config.backoff_policy),
            blobs,
            meetings_root,
            relay_url: config.relay_url,
            peer_events,
            lifecycle_events,
        })
    }

    /// Build the [`Router`] accepting both the sync ALPN ([`AcceptHook`]) and the
    /// blobs ALPN ([`AuthorizedBlobs`]). Both accept hooks share the peer
    /// directory so a peer paired after the router spawned is honoured on its next
    /// inbound connection; the blobs hook also carries a clone of the endpoint so a
    /// media responder can dial the peer back for blobs.
    fn build_router(
        endpoint: &Endpoint,
        blobs: &BlobStore,
        peers: &PeerDirectory,
        meetings_root: &Path,
        peer_events: broadcast::Sender<String>,
        peer_arrivals: PeerArrivalTracker,
        lifecycle_events: broadcast::Sender<(MeetingId, ProcessingLifecycle)>,
    ) -> Router {
        let blobs_protocol = iroh_blobs::BlobsProtocol::new(blobs.inner(), None);
        Router::builder(endpoint.clone())
            .accept(
                SYNC_ALPN,
                AcceptHook::new(
                    meetings_root.to_path_buf(),
                    peers.clone(),
                    blobs.clone(),
                    endpoint.clone(),
                    peer_events,
                    peer_arrivals,
                    lifecycle_events,
                ),
            )
            .accept(
                iroh_blobs::ALPN,
                AuthorizedBlobs::new(blobs_protocol, peers.clone()),
            )
            .spawn()
    }

    /// Build a relay-less engine: `RelayMode::Disabled`, otherwise the same bind +
    /// router as [`Self::start`]. Peers reach each other over the direct addresses
    /// in their [`Self::endpoint_addr`], so no relay (and no relay token) is
    /// involved. Gated behind `test-support` — it is a test/local-only path, not
    /// part of the production sync surface (which always pins the relay).
    #[cfg(feature = "test-support")]
    pub async fn start_direct(identity: DeviceIdentity, meetings_root: PathBuf) -> Result<Self> {
        let peers = PeerDirectory::new();
        let blobs = BlobStore::open(&meetings_root).await?;
        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(identity.secret_key())
            .relay_mode(RelayMode::Disabled)
            .alpns(vec![SYNC_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()])
            .address_lookup(peers.lookup())
            .bind()
            .await
            .map_err(|e| Error::Endpoint(format!("binding iroh endpoint: {e}")))?;
        let (peer_events, _rx) = broadcast::channel(PEER_EVENTS_CAP);
        let (lifecycle_events, _lrx) = broadcast::channel(LIFECYCLE_EVENTS_CAP);
        let router = Self::build_router(
            &endpoint,
            &blobs,
            &peers,
            &meetings_root,
            peer_events.clone(),
            PeerArrivalTracker::new(),
            lifecycle_events.clone(),
        );
        Ok(Self {
            endpoint,
            router,
            peers,
            backoff: BackoffRegistry::new(BackoffPolicy::default()),
            blobs,
            meetings_root,
            // The relay-less test path never addresses a peer relay-only, so it has
            // no relay URL; `push_all_to` is unused here.
            relay_url: String::new(),
            peer_events,
            lifecycle_events,
        })
    }

    /// The bound socket addresses, used to build a direct [`EndpointAddr`] without
    /// a relay. Gated behind `test-support` alongside [`Self::start_direct`].
    #[cfg(feature = "test-support")]
    pub fn bound_sockets(&self) -> Vec<std::net::SocketAddr> {
        self.endpoint.bound_sockets()
    }

    /// Build the endpoint exactly as [`Self::start`], but trusting the relay's
    /// TLS certificate unconditionally instead of verifying it against the
    /// system's CA roots.
    ///
    /// Exists solely for `iroh::test_utils::run_relay_server`'s in-process test
    /// relay, whose certificate is self-signed and unknown to any CA — the same
    /// `CaTlsConfig::insecure_skip_verify` iroh's own relay test suite uses
    /// against that relay. The production relay client (via [`Self::start`])
    /// always verifies the real relay's certificate; this path is gated behind
    /// `test-support` and never reachable from the production build.
    #[cfg(feature = "test-support")]
    pub async fn start_insecure(config: SyncConfig, identity: DeviceIdentity) -> Result<Self> {
        let relay_mode = Self::relay_mode(&config)?;
        let peers = PeerDirectory::new();
        let meetings_root = config.meetings_root.clone();
        let blobs = BlobStore::open(&meetings_root).await?;

        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(identity.secret_key())
            .relay_mode(relay_mode)
            .ca_tls_config(iroh_relay::tls::CaTlsConfig::insecure_skip_verify())
            .alpns(vec![SYNC_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()])
            .address_lookup(peers.lookup())
            .bind()
            .await
            .map_err(|e| Error::Endpoint(format!("binding iroh endpoint: {e}")))?;

        tracing::info!(
            target: "sync",
            endpoint_id = %endpoint.id(),
            relay = %config.relay_url,
            "sync endpoint bound (insecure relay TLS — test relay only)"
        );

        let (peer_events, _rx) = broadcast::channel(PEER_EVENTS_CAP);
        let (lifecycle_events, _lrx) = broadcast::channel(LIFECYCLE_EVENTS_CAP);
        let router = Self::build_router(
            &endpoint,
            &blobs,
            &peers,
            &meetings_root,
            peer_events.clone(),
            PeerArrivalTracker::new(),
            lifecycle_events.clone(),
        );

        Ok(Self {
            endpoint,
            router,
            peers,
            backoff: BackoffRegistry::new(config.backoff_policy),
            blobs,
            meetings_root,
            relay_url: config.relay_url,
            peer_events,
            lifecycle_events,
        })
    }

    /// The configured relay as a [`RelayMode::Custom`], carrying the access token
    /// when [`SyncConfig::relay_auth_token`] is set.
    ///
    /// For a relay with no token the URL is wrapped via `RelayConfig::new`; the
    /// 1.0 relay client presents the token through `RelayConfig::with_auth_token`,
    /// which the relay's `AccessControl` checks on connect.
    fn relay_mode(config: &SyncConfig) -> Result<RelayMode> {
        let url: RelayUrl = config.relay_url.parse().map_err(|e| {
            Error::Endpoint(format!("invalid relay url {:?}: {e}", config.relay_url))
        })?;
        let mut relay = RelayConfig::new(url, None);
        if let Some(token) = &config.relay_auth_token {
            relay = relay.with_auth_token(token.clone());
        }
        let map: RelayMap = relay.into();
        Ok(RelayMode::Custom(map))
    }

    /// This device's [`EndpointId`] (its ed25519 public key) — the address the
    /// account service publishes for peers to dial.
    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// This device's full [`EndpointAddr`] (id + current relay/direct addresses),
    /// the form a peer injects via [`Self::add_peer`] to dial back.
    pub fn endpoint_addr(&self) -> EndpointAddr {
        self.endpoint.addr()
    }

    /// This device's plausibly-reachable direct socket addresses ("ip:port"),
    /// for publishing to the account directory so a same-tailnet/LAN peer can
    /// dial this device directly — no relay, no DNS (0049).
    ///
    /// Filters the endpoint's raw direct-addr set via
    /// [`is_publishable_direct_addr`] so a peer does not burn dial attempts on
    /// addresses it can never route to (loopback, link-local, and the
    /// docker-bridge range a container host otherwise advertises). Order is the
    /// `EndpointAddr`'s (a sorted `BTreeSet`), deduplicated for free.
    ///
    /// **Tailnet-present rule:** if this device has a Tailscale CGNAT
    /// (100.64.0.0/10) address, it is on a tailnet mesh, so the RFC1918
    /// private-LAN addresses are also dropped from the published set. A peer
    /// that can reach this device is then either on the same tailnet (dials the
    /// stable 100.x directly) or off-network (uses the relay). A published LAN
    /// address is otherwise a phantom dial candidate for a peer on a different
    /// L2 — or one whose full-tunnel VPN captures LAN traffic (the phone case):
    /// iroh probes it, abandons it, and re-probes, and that sustained path churn
    /// starves iroh's per-remote actor (a bounded inbox it does not drain while
    /// awaiting path work), dropping the QUIC handshake datagrams so the stream
    /// never opens. Tradeoff: two devices sharing an L2 where one is NOT on the
    /// tailnet lose the direct-LAN path and fall back to the relay — acceptable
    /// for a Tailscale user, and reversible if a same-LAN-no-tailnet case matters.
    pub fn publishable_direct_addrs(&self) -> Vec<String> {
        let kept: Vec<std::net::SocketAddr> = self
            .endpoint_addr()
            .ip_addrs()
            .filter(|addr| is_publishable_direct_addr(addr))
            .copied()
            .collect();
        let has_cgnat = kept.iter().any(is_cgnat_v4);
        kept.into_iter()
            .filter(|addr| !(has_cgnat && is_rfc1918_v4(addr)))
            .map(|addr| addr.to_string())
            .collect()
    }

    /// Register a peer learned out-of-band (manually — a ticket, a peers file,
    /// or the relay-less direct test path) so the endpoint can resolve and dial
    /// it. Tagged [`PeerSource::Manual`]. The [`PeerDirectory`] is shared with
    /// the bound endpoint, so a peer added after binding is picked up on the
    /// next dial.
    pub fn add_peer(&self, addr: EndpointAddr) {
        self.peers.add(addr, PeerSource::Manual);
    }

    /// Register a peer learned from the account service ([`crate::account`]),
    /// addressed by its hex endpoint id and relay URL — the string-keyed
    /// primitive `sync-ffi` wraps for the phone's own account-directory loop
    /// (the in-workspace Rust consumers upsert via [`Self::upsert_account_peer`],
    /// which the account-refresh loop's [`RefreshSink`] drives). Parses both,
    /// builds the same `id + relay + directs` [`EndpointAddr`] shape
    /// [`Self::push_all_to`] dials with, and registers it tagged
    /// [`PeerSource::Account`] (directly, not via [`Self::add_peer`] — that tags
    /// [`PeerSource::Manual`]). No `iroh` type in the signature, so a caller off
    /// the FFI boundary never needs one.
    ///
    /// `direct_addrs` are the peer's published direct socket addresses
    /// ("ip:port"); each is parsed to a [`std::net::SocketAddr`] and
    /// unparseable entries are skipped. When present, iroh dials them directly
    /// (no relay, no DNS) and falls back to the relay if they are unreachable.
    pub fn add_account_peer(
        &self,
        endpoint_id: &str,
        relay_url: &str,
        direct_addrs: &[String],
    ) -> Result<()> {
        let addr = Self::account_peer_addr(endpoint_id, relay_url, direct_addrs)?;
        self.peers.add(addr, PeerSource::Account);
        Ok(())
    }

    /// Upsert a peer learned from the account service, tagged
    /// [`PeerSource::Account`], returning whether it was newly added. The
    /// [`crate::account::RefreshSink`]-facing counterpart of
    /// [`Self::add_account_peer`] (which discards the was-new bool): the loop
    /// needs it to decide whether to first-contact-dial the peer.
    pub fn upsert_account_peer(
        &self,
        endpoint_id: &str,
        relay_url: &str,
        direct_addrs: &[String],
    ) -> Result<bool> {
        let addr = Self::account_peer_addr(endpoint_id, relay_url, direct_addrs)?;
        Ok(self.peers.add(addr, PeerSource::Account))
    }

    /// Parse an account-service `(endpoint_id, relay_url, direct_addrs)` tuple
    /// into the `id + relay + directs` [`EndpointAddr`] shape
    /// [`Self::push_all_to`] dials with. Shared by [`Self::add_account_peer`]
    /// and [`Self::upsert_account_peer`]. Each direct addr is parsed to a
    /// [`std::net::SocketAddr`]; unparseable entries are skipped rather than
    /// failing the whole registration (a bad direct addr must not block the
    /// relay-routed fallback).
    fn account_peer_addr(
        endpoint_id: &str,
        relay_url: &str,
        direct_addrs: &[String],
    ) -> Result<EndpointAddr> {
        let id: EndpointId = endpoint_id.parse().map_err(|e| {
            Error::Protocol(format!("parsing account endpoint id {endpoint_id:?}: {e}"))
        })?;
        let relay: RelayUrl = relay_url.parse().map_err(|e| {
            Error::Endpoint(format!("parsing account relay url {relay_url:?}: {e}"))
        })?;
        let mut addr = EndpointAddr::new(id).with_relay_url(relay);
        for raw in direct_addrs {
            match raw.parse::<std::net::SocketAddr>() {
                Ok(sock) => addr = addr.with_ip_addr(sock),
                Err(e) => tracing::debug!(
                    target: "sync",
                    endpoint_id,
                    addr = %raw,
                    error = %e,
                    "skipping an unparseable account direct addr"
                ),
            }
        }
        Ok(addr)
    }

    /// Remove an `Account`-sourced peer no longer present in the account's
    /// device list (reconcile — it left the account). Source-aware: a no-op
    /// (returns `false`) if `endpoint_id` is absent or was registered any other
    /// way (e.g. [`Self::add_peer_from_ticket`]).
    pub fn remove_account_peer(&self, endpoint_id: &str) -> Result<bool> {
        let id: EndpointId = endpoint_id
            .parse()
            .map_err(|e| Error::Protocol(format!("parsing account endpoint id {endpoint_id:?}: {e}")))?;
        Ok(self.peers.remove(id, PeerSource::Account))
    }

    /// Whether `endpoint_id` is currently dial-suppressed (failed-dial
    /// backoff). Delegates to [`BackoffRegistry::is_suppressed`], fed by every
    /// [`Self::dial`] outcome.
    pub fn is_suppressed(&self, endpoint_id: &str) -> bool {
        self.backoff.is_suppressed(endpoint_id)
    }

    /// The hex ids of every `Account`-sourced peer currently registered. Seeds
    /// [`crate::account::run_account_refresh_loop_v2`]'s reconcile-removal state
    /// on (re)start.
    pub fn account_peer_ids(&self) -> Vec<String> {
        self.peers.account_peer_ids()
    }

    /// This device's shareable ticket: its [`EndpointAddr`] (id + current
    /// relay/direct addresses) encoded as an [`EndpointTicket`] string. The user
    /// copies it to another of their devices, which feeds it to
    /// [`Self::add_peer_from_ticket`] to dial back. The ticket carries only this
    /// device's public addressing — never its secret key.
    pub fn my_ticket(&self) -> String {
        EndpointTicket::new(self.endpoint_addr()).to_string()
    }

    /// Parse a peer's [`EndpointTicket`] string (produced by [`Self::my_ticket`]
    /// on the other device) back into its [`EndpointAddr`] and register it via
    /// [`Self::add_peer`], so the two devices can dial each other.
    ///
    /// Returns the peer's [`EndpointId`] on success. A malformed ticket is an
    /// [`Error::Protocol`].
    pub fn add_peer_from_ticket(&self, ticket: &str) -> Result<EndpointId> {
        let ticket: EndpointTicket = ticket
            .parse()
            .map_err(|e| Error::Protocol(format!("parsing endpoint ticket: {e}")))?;
        let addr: EndpointAddr = ticket.into();
        let id = addr.id;
        self.add_peer(addr);
        Ok(id)
    }

    /// The hex endpoint ids of every peer currently registered in the directory.
    /// The connected `SyncControl` syncs a meeting against each of them. Hex
    /// strings, not [`EndpointId`], so a caller off the FFI boundary (the mobile
    /// wrapper) never needs an `iroh` type; an in-tree caller that needs to dial
    /// uses one of the `*_to_peer` methods, which parse the string back
    /// internally.
    pub fn peer_ids(&self) -> Vec<String> {
        self.peers
            .ids()
            .into_iter()
            .map(|id| id.to_string())
            .collect()
    }

    /// Subscribe to "peer arrived" events: a peer's hex endpoint id, fired at most
    /// once per [`PEER_ARRIVAL_DEBOUNCE`] window of its accepted inbound
    /// connections (see [`Self::peer_events`]). An always-on hub reacts by calling
    /// [`Self::push_all_to_peer`] so a device that reconnects both deposits and
    /// collects (convergence through the hub). The desktop does not subscribe.
    pub fn subscribe_peer_events(&self) -> broadcast::Receiver<String> {
        self.peer_events.subscribe()
    }

    /// Subscribe to discovery lifecycle events: each `(MeetingId,
    /// ProcessingLifecycle)` this device receives from a peer's discovery
    /// exchange (inbound via the accept loop, or outbound via
    /// [`Self::discover_with`]). The consumer — in a crate depending on both
    /// `sync` and `persistence` (ipc-bridge / headless) — persists each via
    /// `persistence::apply_processing_lifecycle`; `sync` has no `persistence`
    /// edge, so it emits rather than writes. Bounded: a consumer that hits
    /// [`broadcast::error::RecvError::Lagged`] recovers by re-running discovery.
    pub fn subscribe_lifecycle_events(
        &self,
    ) -> broadcast::Receiver<(MeetingId, ProcessingLifecycle)> {
        self.lifecycle_events.subscribe()
    }

    /// The meeting ids this device holds on disk — the `{uuid}` folders directly
    /// under [`Self::meetings_root`] (the dot-prefixed `.blobs` store and any
    /// non-UUID entry are skipped).
    pub fn local_meetings(&self) -> Vec<MeetingId> {
        discovery_proto::list_meeting_ids(&self.meetings_root)
    }

    /// Reconcile EVERY meeting this device holds with `peer`, addressed relay-only
    /// (id + the configured relay). This is the hub's reciprocal push: on a peer
    /// arriving it pushes all it holds, so a device that reconnects to deposit one
    /// meeting also collects every other meeting the hub has accumulated. Notes,
    /// media, and derived artifacts are reconciled per meeting; a per-meeting
    /// failure is logged and skipped so one bad meeting does not abort the rest.
    /// Returns how many meetings reconciled without error.
    pub async fn push_all_to(&self, peer: EndpointId) -> Result<usize> {
        let relay: RelayUrl = self.relay_url.parse().map_err(|e| {
            Error::Endpoint(format!(
                "push_all_to needs a relay url, got {:?}: {e}",
                self.relay_url
            ))
        })?;
        let addr = EndpointAddr::new(peer).with_relay_url(relay);
        let meetings = self.local_meetings();
        tracing::debug!(target: "sync", peer = %peer, count = meetings.len(), "pushing all meetings to peer");
        let mut reconciled = 0usize;
        for meeting in meetings {
            if let Err(e) = self.sync_notes(addr.clone(), meeting).await {
                tracing::warn!(target: "sync", peer = %peer, meeting = %meeting.0, error = %e, "push notes failed");
                continue;
            }
            if let Err(e) = self.sync_media(addr.clone(), meeting).await {
                tracing::warn!(target: "sync", peer = %peer, meeting = %meeting.0, error = %e, "push media failed");
                continue;
            }
            // Derived artifacts ride per meeting, AFTER notes+media and BEFORE the
            // final ride-along discovery, so a peer that learns `Processed` via the
            // discovery exchange below has already had the transcript/summary pulled
            // (DESIGN §5 ordering invariant).
            if let Err(e) = self.sync_artifacts(addr.clone(), meeting).await {
                tracing::warn!(target: "sync", peer = %peer, meeting = %meeting.0, error = %e, "push artifacts failed");
                continue;
            }
            reconciled += 1;
        }

        // §7 ride-alongside: after reconciling notes+media, exchange lifecycle
        // with the peer in the same flow (a separate dial, run LAST), so a
        // meeting's processing state follows the meeting it was just pushed in.
        // This is ordering, not atomicity: a device that goes offline between the
        // notes push and this dial has not pushed its lifecycle — the periodic
        // recovery sweep (`discover_all`) backstops that window. Best-effort: a
        // discovery failure does not fail the push (the meetings are reconciled;
        // the lifecycle re-advertises on the next discovery).
        if let Err(e) = self.discover_with(addr.clone()).await {
            tracing::warn!(target: "sync", peer = %peer, error = %e, "ride-along discovery failed");
        }
        Ok(reconciled)
    }

    /// [`Self::push_all_to`] addressing the peer by its hex endpoint-id string —
    /// the form [`Self::peer_ids`] / [`Self::subscribe_peer_events`] hand back, so
    /// the hub's convergence loop (`headless`) never constructs an `iroh` type.
    pub async fn push_all_to_peer(&self, peer_id: &str) -> Result<usize> {
        let id: EndpointId = peer_id
            .parse()
            .map_err(|e| Error::Protocol(format!("parsing peer id {peer_id:?}: {e}")))?;
        self.push_all_to(id).await
    }

    /// Resolve a hex endpoint-id string to a relay-routed [`EndpointAddr`] (id +
    /// the configured relay) — the addressing form [`Self::push_all_to`] uses.
    /// The string-keyed `*_to_peer` / `*_with_peer` methods below take this so an
    /// FFI / non-`iroh` caller (the phone wrapper) can address a paired peer
    /// without constructing `iroh` types.
    fn peer_relay_addr(&self, peer_id: &str) -> Result<EndpointAddr> {
        let id: EndpointId = peer_id
            .parse()
            .map_err(|e| Error::Protocol(format!("parsing peer id {peer_id:?}: {e}")))?;
        let relay: RelayUrl = self.relay_url.parse().map_err(|e| {
            Error::Endpoint(format!(
                "peer dial needs a relay url, got {:?}: {e}",
                self.relay_url
            ))
        })?;
        Ok(EndpointAddr::new(id).with_relay_url(relay))
    }

    /// [`Self::sync_notes`] addressing the peer by its hex endpoint-id string
    /// (relay-routed). The peer must already be paired ([`Self::add_peer_from_ticket`]).
    pub async fn sync_notes_to_peer(&self, peer_id: &str, meeting_id: MeetingId) -> Result<()> {
        self.sync_notes(self.peer_relay_addr(peer_id)?, meeting_id)
            .await
    }

    /// [`Self::sync_media`] addressing the peer by its hex endpoint-id string.
    pub async fn sync_media_to_peer(&self, peer_id: &str, meeting_id: MeetingId) -> Result<()> {
        self.sync_media(self.peer_relay_addr(peer_id)?, meeting_id)
            .await
    }

    /// [`Self::sync_artifacts`] addressing the peer by its hex endpoint-id string.
    pub async fn sync_artifacts_to_peer(&self, peer_id: &str, meeting_id: MeetingId) -> Result<()> {
        self.sync_artifacts(self.peer_relay_addr(peer_id)?, meeting_id)
            .await
    }

    /// [`Self::discover_with`] addressing the peer by its hex endpoint-id string.
    pub async fn discover_with_peer(&self, peer_id: &str) -> Result<Vec<MeetingId>> {
        self.discover_with(self.peer_relay_addr(peer_id)?).await
    }

    /// Dial a peer on the [`SYNC_ALPN`]. The peer must already be resolvable —
    /// either injected via [`Self::add_peer`] or passed as a full
    /// [`EndpointAddr`] carrying its relay/direct addresses. Crate-internal: no
    /// consumer outside `sync` needs the raw [`Connection`] (every real operation
    /// goes through [`Self::sync_notes`] / [`Self::sync_media`] /
    /// [`Self::sync_artifacts`] / [`Self::discover_with`]), so this stays off the
    /// public API rather than widening it with an iroh-typed return. See
    /// [`Self::connect`] for the test-only public seam.
    async fn dial(&self, peer: impl Into<EndpointAddr>) -> Result<Connection> {
        let addr: EndpointAddr = peer.into();
        let id_hex = addr.id.to_string();
        let result = self
            .endpoint
            .connect(addr, SYNC_ALPN)
            .await
            .map_err(|e| Error::Endpoint(format!("dialling peer on sync alpn: {e}")));
        // Universal write side: every dial this device makes — desktop,
        // headless, and the phone's syncs (which flow through this same engine
        // dial) — feeds the backoff registry, regardless of the peer's source.
        self.backoff.on_dial_outcome(&id_hex, result.is_ok());
        result
    }

    /// Test-only public seam wrapping [`Self::dial`] (mirrors [`Self::import_media`]
    /// / [`Self::download_blob`]): lets an integration test prove the raw
    /// connection identity / ALPN negotiation without a full protocol exchange.
    /// Gated behind `test-support` so the production public API carries no
    /// iroh-typed `Connection` return.
    #[cfg(feature = "test-support")]
    pub async fn connect(&self, peer: impl Into<EndpointAddr>) -> Result<Connection> {
        self.dial(peer).await
    }

    /// Reconcile one meeting's notes with `peer`: dial it on the [`SYNC_ALPN`]
    /// and run the initiator side of the notes-sync protocol
    /// ([`notes_proto::initiate_notes_sync`]) against this device's
    /// [`Self::meetings_root`]. On return both sides have merged each other's
    /// missing updates into their `notes.ydoc` (via `persistence`).
    ///
    /// On-demand per-meeting reconciliation: one call, one meeting. The current
    /// cut dials a fresh connection per call.
    /// TODO(OQ-A): a debounce window that coalesces rapid edit bursts into one
    /// reconciliation per meeting, and reuse of a live connection across
    /// meetings, land with the orchestrator wiring (S5).
    pub async fn sync_notes(
        &self,
        peer: impl Into<EndpointAddr>,
        meeting_id: MeetingId,
    ) -> Result<()> {
        let conn = self.dial(peer).await?;
        let result = notes_proto::initiate_notes_sync(&conn, &self.meetings_root, meeting_id).await;
        conn.close(0u32.into(), b"notes-sync-done");
        result
    }

    /// Run a discovery exchange with `peer`: dial it on the [`SYNC_ALPN`] and run
    /// the initiator side ([`discovery_proto::initiate_discovery`]), learning the
    /// peer's `(MeetingId, ProcessingLifecycle)` for every meeting it holds. Each
    /// received state is emitted on the lifecycle-event surface
    /// ([`Self::subscribe_lifecycle_events`]) for the ipc-bridge / headless
    /// subscriber to persist via `persistence::apply_synced_lifecycle_if_present`;
    /// the returned ids are the peer's meeting list (the meeting-list discovery —
    /// the caller reconciles any it lacks via [`Self::sync_notes`] /
    /// [`Self::sync_media`]).
    ///
    /// Per §7 (`planning/DESIGN_processing-lifecycle.md`) discovery rides
    /// alongside a full sync: [`Self::push_all_to`] (the hub) and the desktop's
    /// `sync_now` call this after reconciling a peer's notes/media, so a meeting's
    /// lifecycle travels in the session it is pushed in rather than a skippable
    /// separate round. [`Self::discover_all`] drives it as a standalone recovery
    /// sweep (the hub's periodic re-discovery).
    pub async fn discover_with(&self, peer: impl Into<EndpointAddr>) -> Result<Vec<MeetingId>> {
        let conn = self.dial(peer).await?;
        let result = discovery_proto::initiate_discovery(&conn, &self.meetings_root).await;
        conn.close(0u32.into(), b"discovery-done");
        let theirs = result?;
        let ids = theirs.iter().map(|e| e.meeting_id).collect();
        for entry in theirs {
            let _ = self
                .lifecycle_events
                .send((entry.meeting_id, entry.processing));
        }
        Ok(ids)
    }

    /// The known peers this sweep should dial: every registered id EXCEPT those in
    /// failed-dial backoff ([`BackoffRegistry::is_suppressed`]). A suppressed peer is
    /// skipped without dialling so a stale/unreachable peer does not burn the
    /// per-dial timeout on every sweep (0029 item 6); its backoff window elapsing is
    /// the retry — once `retry_after` passes it re-appears here and is dialled again
    /// (a success then clears the suppression, a failure re-extends it).
    fn peers_to_dial(&self) -> Vec<EndpointId> {
        self.peers
            .ids()
            .into_iter()
            .filter(|id| !self.backoff.is_suppressed(&id.to_string()))
            .collect()
    }

    /// Run a discovery exchange with every known peer NOT in failed-dial backoff,
    /// relay-addressed (id + the configured relay) — the hub's recovery sweep.
    /// Mirrors [`Self::push_all_to`]'s addressing (a hub reaches its peers through
    /// the relay). A dial-suppressed peer is skipped without dialling (see
    /// [`Self::peers_to_dial`]); a per-peer failure is logged and skipped; returns
    /// how many peers were discovered without error.
    ///
    /// This is the scheduled (periodic) re-advertisement that re-applies a
    /// lifecycle state a consumer dropped (broadcast
    /// [`broadcast::error::RecvError::Lagged`], recovered on the next sweep) or
    /// skipped (an advertisement for a meeting not present locally when it fired).
    /// Each received state is emitted on [`Self::subscribe_lifecycle_events`], the
    /// same as [`Self::discover_with`].
    pub async fn discover_all(&self) -> Result<usize> {
        let relay: RelayUrl = self.relay_url.parse().map_err(|e| {
            Error::Endpoint(format!(
                "discover_all needs a relay url, got {:?}: {e}",
                self.relay_url
            ))
        })?;
        // `self.peers.ids()` (not the string-keyed `Self::peer_ids`) — this is
        // internal engine addressing, not the FFI-facing surface. Peers in
        // failed-dial backoff are filtered out by `peers_to_dial`.
        let all = self.peers.ids().len();
        let peers = self.peers_to_dial();
        let skipped = all - peers.len();
        tracing::debug!(target: "sync", count = peers.len(), skipped, "sweeping peers for discovery");
        let mut discovered = 0usize;
        for peer in peers {
            let addr = EndpointAddr::new(peer).with_relay_url(relay.clone());
            match self.discover_with(addr).await {
                Ok(_) => discovered += 1,
                Err(e) => {
                    tracing::warn!(target: "sync", peer = %peer, error = %e, "recovery discovery failed")
                }
            }
        }
        Ok(discovered)
    }

    /// Adopt from `peer_id` as a sync REPLICA: discover the peer's meeting list
    /// (which also applies the peer's lifecycle via
    /// [`Self::subscribe_lifecycle_events`], the same as [`Self::discover_with`]),
    /// then pull every meeting THIS device lacks — notes, media, and artifacts —
    /// from that peer. Returns how many meetings were newly adopted.
    ///
    /// Replica semantics: the caller (the hub) runs no producer/election loop, so
    /// an adopted meeting is never claimed for processing or host-elected — its
    /// lifecycle is applied as-received. Notes are the anchor (they ensure the
    /// folder and carry the doc); media/artifacts ride best-effort. A per-meeting
    /// pull failure is logged and skipped so one bad meeting doesn't abort the set.
    pub async fn adopt_from_peer(&self, peer_id: &str) -> Result<usize> {
        let theirs = self.discover_with_peer(peer_id).await?;
        let mine = discovery_proto::list_meeting_ids(&self.meetings_root);
        let discovered = theirs.len();
        let mut adopted = 0usize;
        let mut recompleted = 0usize;
        let mut skipped_complete = 0usize;
        for meeting_id in theirs {
            // Skip a meeting only when it is already FULLY materialised locally —
            // NOT on mere folder existence. sync_notes creates the folder before
            // media and artifacts run, so a meeting whose notes landed but whose
            // media/artifacts pull FAILED (or vice versa) would, under a
            // folder-existence check, be treated as "held" and skipped on every
            // later sweep, stranded half-synced forever. Re-attempt an incomplete
            // held meeting; notes/media/artifacts sync is idempotent + manifest-
            // diffed, so re-running a complete one would be a cheap no-op (which the
            // completeness skip avoids anyway).
            let held = mine.contains(&meeting_id);
            if held && meeting_is_materialised(&self.meetings_root, meeting_id) {
                skipped_complete += 1;
                continue;
            }
            if let Err(e) = self.sync_notes_to_peer(peer_id, meeting_id).await {
                tracing::warn!(target: "sync", peer = peer_id, meeting_id = %meeting_id.0, error = %e, "adopt: notes pull failed; skipping meeting");
                continue;
            }
            if let Err(e) = self.sync_media_to_peer(peer_id, meeting_id).await {
                tracing::warn!(target: "sync", peer = peer_id, meeting_id = %meeting_id.0, error = %e, "adopt: media pull failed");
            }
            if let Err(e) = self.sync_artifacts_to_peer(peer_id, meeting_id).await {
                tracing::warn!(target: "sync", peer = peer_id, meeting_id = %meeting_id.0, error = %e, "adopt: artifacts pull failed");
            }
            // Count genuinely-new adoptions separately from re-completions of a
            // held-but-incomplete meeting, so the return stays "meetings newly
            // adopted this pass".
            if held {
                recompleted += 1;
            } else {
                adopted += 1;
            }
        }
        // One summary line per peer so an adopt pass is observable without per-meeting
        // spam: how many the peer advertised, how many were newly adopted vs
        // re-completed (half-synced meetings finished) vs skipped as already-complete.
        tracing::debug!(
            target: "sync",
            peer = peer_id,
            discovered,
            adopted,
            recompleted,
            skipped_complete,
            "adopt: pass complete for peer"
        );
        Ok(adopted)
    }

    /// Adopt from every known peer NOT in failed-dial backoff — the hub's periodic
    /// replica sweep. Per-peer [`Self::adopt_from_peer`], relay-addressed like
    /// [`Self::discover_all`]. A per-peer failure is logged and skipped; returns
    /// the total meetings newly adopted across all peers this sweep.
    pub async fn adopt_all(&self) -> Result<usize> {
        // Adopt from every peer CONCURRENTLY: a dead or slow peer (e.g. a
        // backgrounded phone whose dial runs the full timeout) must not delay the
        // sweep for the peers behind it — serialising the sweep let one unreachable
        // peer starve the rest. Each future borrows `&self`; a per-peer failure is
        // logged and contributes zero to the total.
        let counts = futures_util::future::join_all(self.peers_to_dial().into_iter().map(
            |peer| async move {
                match self.adopt_from_peer(&peer.to_string()).await {
                    Ok(n) => n,
                    Err(e) => {
                        tracing::warn!(target: "sync", peer = %peer, error = %e, "adopt sweep: peer discovery failed");
                        0
                    }
                }
            },
        ))
        .await;
        Ok(counts.into_iter().sum())
    }

    /// Reconcile one meeting's media (`audio.opus` + note assets) with `peer`:
    /// dial it on the [`SYNC_ALPN`] and run the initiator side of the
    /// media-manifest protocol ([`media_proto::initiate_media_sync`]) against this
    /// device's [`Self::meetings_root`]. Each side imports its own media into the
    /// blob store, exchanges a manifest of `(relative-path, hash)` pairs, and
    /// pulls the blobs it is missing over the blobs ALPN — exporting each to the
    /// correct per-meeting path and pinning it with a persistent tag. On return
    /// both sides hold byte-identical media for the meeting.
    ///
    /// The remote [`EndpointId`] is taken from the dialled connection so the
    /// downloader dials the same peer back for blobs. Like [`Self::sync_notes`],
    /// this is on-demand per-meeting reconciliation: one call, one meeting, a
    /// fresh connection per call (S5 wires it into the orchestrator).
    pub async fn sync_media(
        &self,
        peer: impl Into<EndpointAddr>,
        meeting_id: MeetingId,
    ) -> Result<()> {
        let conn = self.dial(peer).await?;
        let peer_id = conn.remote_id();
        let result = media_proto::initiate_media_sync(
            &conn,
            &self.blobs,
            &self.endpoint,
            peer_id,
            &self.meetings_root,
            meeting_id,
        )
        .await;
        conn.close(0u32.into(), b"media-sync-done");
        result
    }

    /// Reconcile one meeting's derived artifacts (`transcript.json` + `summary.md`)
    /// with `peer`: dial it on the [`SYNC_ALPN`] and run the initiator side of the
    /// artifact-manifest protocol ([`artifacts_proto::initiate_artifacts_sync`])
    /// against this device's [`Self::meetings_root`]. Each side imports its own
    /// artifacts into the blob store (stamping each entry with the authority for
    /// those exact bytes), exchanges a manifest, and pulls every entry that
    /// strictly supersedes its local copy over the blobs ALPN — exporting it
    /// atomically to the per-meeting path. On return both sides hold the
    /// authoritative copy of each artifact (a stale copy never overwrites a newer
    /// one — `planning/DESIGN_artifacts.md` §2).
    ///
    /// The remote [`EndpointId`] is taken from the dialled connection so the
    /// downloader dials the same peer back for blobs. Like [`Self::sync_media`],
    /// this is on-demand per-meeting reconciliation: one call, one meeting, a fresh
    /// connection per call.
    pub async fn sync_artifacts(
        &self,
        peer: impl Into<EndpointAddr>,
        meeting_id: MeetingId,
    ) -> Result<()> {
        let conn = self.dial(peer).await?;
        let peer_id = conn.remote_id();
        let result = artifacts_proto::initiate_artifacts_sync(
            &conn,
            &self.blobs,
            &self.endpoint,
            peer_id,
            &self.meetings_root,
            meeting_id,
        )
        .await;
        conn.close(0u32.into(), b"artifacts-sync-done");
        result
    }

    /// Import a meeting's media into this device's blob store and return its
    /// [`crate::blobs::Manifest`]. Test-only seam used to stage blobs and to read
    /// a known hash for the blobs-ALPN authorisation test, without going through a
    /// full media reconciliation.
    #[cfg(feature = "test-support")]
    pub async fn import_media(&self, meeting_id: MeetingId) -> Result<crate::blobs::Manifest> {
        self.blobs
            .import_meeting(&self.meetings_root, meeting_id)
            .await
    }

    /// Attempt to download a single blob `hash` from `peer` over the blobs ALPN and
    /// export it to `{meetings_root}/{meeting_id}/{rel}`. Test-only seam that
    /// drives the blobs channel directly so a test can prove the blobs-ALPN
    /// authorisation guard rejects an unpaired peer.
    #[cfg(feature = "test-support")]
    pub async fn download_blob(
        &self,
        peer: EndpointId,
        meeting_id: MeetingId,
        rel: &str,
        hash: crate::blobs::Hash,
    ) -> Result<()> {
        self.blobs
            .download(
                &self.endpoint,
                peer,
                &self.meetings_root,
                meeting_id,
                rel,
                hash,
            )
            .await
            .map(|_| ())
    }

    /// Unpin this device's blobs (media + derived artifacts) for `meeting_id` so
    /// they become GC-eligible — see [`crate::blobs::BlobStore::delete_meeting_blobs`].
    /// Called from the meeting-deletion path (the `ipc-bridge` `delete_meeting`
    /// command, via the connected `SyncControl::delete_meeting_blobs`) after the
    /// on-disk meeting folder is already gone; this only touches the blob store.
    pub async fn delete_meeting_blobs(&self, meeting_id: MeetingId) -> Result<()> {
        self.blobs.delete_meeting_blobs(meeting_id).await
    }

    /// Test-only seam: like [`Self::download_blob`] but with an explicit size cap
    /// in place of the production ceiling, so a test can prove the per-blob
    /// size-cap rejection without a multi-gigabyte payload. See
    /// [`crate::blobs::BlobStore::download_capped_for_test`].
    #[cfg(feature = "test-support")]
    pub async fn download_blob_capped(
        &self,
        peer: EndpointId,
        hash: crate::blobs::Hash,
        max_bytes: u64,
    ) -> Result<()> {
        self.blobs
            .download_capped_for_test(&self.endpoint, peer, hash, max_bytes)
            .await
    }

    /// Shut the router (and its endpoint) down gracefully, draining in-flight
    /// connections. Idempotent at the iroh layer.
    pub async fn shutdown(self) -> Result<()> {
        self.router
            .shutdown()
            .await
            .map_err(|e| Error::Endpoint(format!("shutting down sync router: {e}")))
    }
}

/// The production [`RefreshSink`], wrapping a live [`SyncEngine`] so a consumer
/// driving [`crate::account::run_account_refresh_loop_v2`] just constructs this
/// rather than hand-rolling the trait: `upsert`/`remove`/`is_suppressed`/
/// `account_peer_ids` delegate straight to the matching engine method, and
/// `on_new_peer` first-contact-dials the peer via [`SyncEngine::discover_with_peer`]
/// (a full discovery exchange, so the meeting list + lifecycle also travel on
/// the same dial — a plain connect would only prove reachability).
///
/// **Engine-`Arc` ownership contract.** This sink holds a strong [`Arc<SyncEngine>`],
/// as does the account-refresh loop future it is passed to. A consumer that reclaims
/// sole ownership at shutdown for a graceful drain (`Arc::into_inner` → owning
/// `shutdown(self)`) MUST first `await` the loop future's exit — a signal-only
/// cancel is not enough. No `RefreshSink` method may move this `Arc` into a task
/// that outlives the loop future, or the reclaim silently finds >1 strong ref and
/// skips the graceful path. `on_new_peer` borrows `&self` for exactly one awaited
/// `discover_with_peer`, so it detaches no `Arc`.
pub struct SyncEngineRefreshSink {
    engine: Arc<SyncEngine>,
}

impl SyncEngineRefreshSink {
    /// Wrap `engine`. The engine must outlive the refresh loop driven with this
    /// sink.
    pub fn new(engine: Arc<SyncEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait::async_trait]
impl RefreshSink for SyncEngineRefreshSink {
    fn upsert_account_peer(&self, ep: &AccountEndpoint) -> bool {
        self.engine
            .upsert_account_peer(&ep.endpoint_id, &ep.relay_url, &ep.direct_addrs)
            .unwrap_or_else(|e| {
                tracing::warn!(
                    target: "sync",
                    endpoint_id = %ep.endpoint_id,
                    error = %e,
                    "upserting an account peer failed; treating as not-new"
                );
                false
            })
    }

    fn remove_account_peer(&self, endpoint_id: &str) {
        if let Err(e) = self.engine.remove_account_peer(endpoint_id) {
            tracing::warn!(
                target: "sync",
                endpoint_id,
                error = %e,
                "removing an account peer failed"
            );
        }
    }

    fn is_suppressed(&self, endpoint_id: &str) -> bool {
        self.engine.is_suppressed(endpoint_id)
    }

    async fn on_new_peer(&self, endpoint_id: &str) {
        // Second guard (the loop already skips a suppressed peer before calling):
        // honour the trait contract for any future caller that invokes this
        // without the loop's is_suppressed gate, so a backed-off peer is never
        // instant-dialled.
        if self.engine.is_suppressed(endpoint_id) {
            return;
        }
        // A first-contact dial that fails just waits for the next poll tick —
        // logged at debug, not warn, since a freshly-joined peer being briefly
        // unreachable is the expected common case, not an anomaly.
        if let Err(e) = self.engine.discover_with_peer(endpoint_id).await {
            tracing::debug!(
                target: "sync",
                endpoint_id,
                error = %e,
                "first-contact dial-kick failed; will retry on the next poll tick"
            );
        }
    }

    fn account_peer_ids(&self) -> Vec<String> {
        self.engine.account_peer_ids()
    }
}

/// The inbound-connection handler registered on the [`Router`] for the
/// [`SYNC_ALPN`].
///
/// Authorises the remote against the paired-peer [`PeerDirectory`] before doing
/// anything else: an inbound connection from an `EndpointId` that this device has
/// NOT paired (its ticket is not in the directory) is rejected before a single
/// frame is read, so a holder of the shared relay token who merely learns an
/// `EndpointId` cannot push CRDT updates or media into this device's meetings.
/// Sync therefore requires mutual pairing — each device must add the other's
/// ticket.
///
/// Once authorised it accepts the bidirectional stream the initiator opened, reads
/// the leading one-byte [`StreamKind`] tag, and runs the matching responder —
/// notes ([`notes_proto::respond_notes_sync`]), media
/// ([`media_proto::respond_media_sync`]), discovery
/// ([`discovery_proto::respond_discovery`]), or derived artifacts
/// ([`artifacts_proto::respond_artifacts_sync`]) — against the device's
/// [`SyncEngine::meetings_root`]. The media and artifacts responders also need the
/// blob store and the endpoint (to pull blobs back from the initiator), so the hook
/// carries clones of both (the router spawns a fresh task per connection). A failed
/// exchange is converted to an [`AcceptError`]; it does not bring the router down.
#[derive(Debug, Clone)]
struct AcceptHook {
    meetings_root: PathBuf,
    /// The authorised-peer set, shared with [`SyncEngine`] (cheap-to-clone, same
    /// backing store), so a peer paired after the router spawned is honoured on
    /// the next inbound connection.
    peers: PeerDirectory,
    /// The blob store, for the media responder.
    blobs: BlobStore,
    /// The endpoint, for the media responder's blob pulls.
    endpoint: Endpoint,
    /// Fires the remote's hex id the first time it is authorised in a
    /// [`PEER_ARRIVAL_DEBOUNCE`] window, so an always-on hub can push back (see
    /// [`SyncEngine::subscribe_peer_events`]) once per visit rather than once per
    /// connection.
    peer_events: broadcast::Sender<String>,
    /// Coalesces the burst of connections one sync session opens into a single
    /// [`Self::peer_events`] fire per visit — see [`PeerArrivalTracker`].
    peer_arrivals: PeerArrivalTracker,
    /// Fires each `(MeetingId, ProcessingLifecycle)` received on an inbound
    /// discovery exchange (see [`SyncEngine::subscribe_lifecycle_events`]).
    lifecycle_events: broadcast::Sender<(MeetingId, ProcessingLifecycle)>,
}

impl AcceptHook {
    fn new(
        meetings_root: PathBuf,
        peers: PeerDirectory,
        blobs: BlobStore,
        endpoint: Endpoint,
        peer_events: broadcast::Sender<String>,
        peer_arrivals: PeerArrivalTracker,
        lifecycle_events: broadcast::Sender<(MeetingId, ProcessingLifecycle)>,
    ) -> Self {
        Self {
            meetings_root,
            peers,
            blobs,
            endpoint,
            peer_events,
            peer_arrivals,
            lifecycle_events,
        }
    }
}

impl ProtocolHandler for AcceptHook {
    async fn accept(&self, connection: Connection) -> std::result::Result<(), AcceptError> {
        let peer = connection.remote_id();

        // Authorise before reading any frame: only a peer this device has paired
        // (its ticket is in the directory) may sync. An unpaired peer is rejected.
        if !self.peers.ids().contains(&peer) {
            tracing::debug!(
                target: "sync",
                peer = %peer,
                "rejecting inbound sync connection from an unpaired peer"
            );
            return Err(AcceptError::from_err(Error::Protocol(format!(
                "unpaired peer {peer}"
            ))));
        }

        tracing::info!(target: "sync", peer = %peer, "accepted sync connection");

        // Notify any subscriber (the always-on hub) that this peer is online so it
        // can reciprocally push — but only on a genuine arrival (the first
        // connection from this peer in a PEER_ARRIVAL_DEBOUNCE window), so a
        // burst of connections from one sync session (notes/media/discovery each
        // dial separately) fires ONE event, not one per connection. Best-effort:
        // `send` errors only with no receivers.
        if self.peer_arrivals.note_connection(peer) {
            let _ = self.peer_events.send(peer.to_string());
        }

        self.dispatch(&connection, peer)
            .await
            .map_err(AcceptError::from_err)
    }
}

impl AcceptHook {
    /// Accept the initiator's bidirectional stream, read its leading
    /// [`StreamKind`] tag, and run the matching responder. The accept + tag read
    /// are bounded by [`ACCEPT_HANDSHAKE_TIMEOUT`]: this is the very first
    /// exchange on a fresh connection, so a peer slow here is stalled or hostile
    /// rather than merely on a slow network path.
    async fn dispatch(&self, connection: &Connection, peer: EndpointId) -> Result<()> {
        let (mut send, mut recv) =
            tokio::time::timeout(ACCEPT_HANDSHAKE_TIMEOUT, connection.accept_bi())
                .await
                .map_err(|_| {
                    tracing::warn!(
                        target: "sync",
                        peer = %peer,
                        timeout = ?ACCEPT_HANDSHAKE_TIMEOUT,
                        "accepting the sync bi stream timed out"
                    );
                    Error::Protocol(format!(
                        "accepting sync bi stream timed out after {ACCEPT_HANDSHAKE_TIMEOUT:?}"
                    ))
                })?
                .map_err(|e| Error::Protocol(format!("accepting sync bi stream: {e}")))?;

        let mut tag = [0u8; 1];
        tokio::time::timeout(ACCEPT_HANDSHAKE_TIMEOUT, recv.read_exact(&mut tag))
            .await
            .map_err(|_| {
                tracing::warn!(
                    target: "sync",
                    peer = %peer,
                    timeout = ?ACCEPT_HANDSHAKE_TIMEOUT,
                    "reading the sync stream tag timed out"
                );
                Error::Protocol(format!(
                    "reading sync stream tag timed out after {ACCEPT_HANDSHAKE_TIMEOUT:?}"
                ))
            })?
            .map_err(|e| Error::Protocol(format!("reading sync stream tag: {e}")))?;

        match StreamKind::from_tag(tag[0])? {
            StreamKind::Notes => {
                notes_proto::respond_notes_sync(
                    connection,
                    &mut send,
                    &mut recv,
                    &self.meetings_root,
                )
                .await
            }
            StreamKind::Media => {
                media_proto::respond_media_sync(
                    connection,
                    &mut send,
                    &mut recv,
                    &self.blobs,
                    &self.endpoint,
                    peer,
                    &self.meetings_root,
                )
                .await
            }
            StreamKind::Discovery => {
                let theirs = discovery_proto::respond_discovery(
                    connection,
                    &mut send,
                    &mut recv,
                    &self.meetings_root,
                )
                .await?;
                // Emit each received state for a consumer to persist via
                // `persistence::apply_processing_lifecycle`. Best-effort: `send`
                // errors only when there is no receiver (the desktop with no
                // subscriber), which is fine.
                for entry in theirs {
                    let _ = self
                        .lifecycle_events
                        .send((entry.meeting_id, entry.processing));
                }
                Ok(())
            }
            StreamKind::Artifacts => {
                artifacts_proto::respond_artifacts_sync(
                    connection,
                    &mut send,
                    &mut recv,
                    &self.blobs,
                    &self.endpoint,
                    peer,
                    &self.meetings_root,
                )
                .await
            }
        }
    }
}

/// The inbound-connection handler registered on the [`Router`] for the blobs ALPN
/// ([`iroh_blobs::ALPN`]).
///
/// Wraps [`iroh_blobs::BlobsProtocol`] with the SAME paired-peer authorisation
/// [`AcceptHook`] applies to the sync ALPN. This is a hard security requirement:
/// `BlobsProtocol` on its own serves a blob to ANY peer that connects (its
/// `accept` spawns the provider handler unconditionally), so registering it
/// unguarded would let any holder of the shared relay token who learns an
/// `EndpointId` read this device's meeting media. By rejecting an unpaired remote
/// BEFORE delegating to `BlobsProtocol::accept`, only a mutually-paired peer can
/// fetch a blob.
#[derive(Debug, Clone)]
struct AuthorizedBlobs {
    inner: iroh_blobs::BlobsProtocol,
    peers: PeerDirectory,
}

impl AuthorizedBlobs {
    fn new(inner: iroh_blobs::BlobsProtocol, peers: PeerDirectory) -> Self {
        Self { inner, peers }
    }
}

impl ProtocolHandler for AuthorizedBlobs {
    async fn accept(&self, connection: Connection) -> std::result::Result<(), AcceptError> {
        let peer = connection.remote_id();
        if !self.peers.ids().contains(&peer) {
            tracing::debug!(
                target: "sync",
                peer = %peer,
                "rejecting inbound blobs connection from an unpaired peer"
            );
            return Err(AcceptError::from_err(Error::Protocol(format!(
                "unpaired peer {peer}"
            ))));
        }
        tracing::debug!(target: "sync", peer = %peer, "accepted blobs connection");
        self.inner.accept(connection).await
    }

    async fn shutdown(&self) {
        self.inner.shutdown().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_mode_without_token_is_custom() {
        let config = SyncConfig::new(std::env::temp_dir());
        let mode = SyncEngine::relay_mode(&config).expect("default relay url must parse");
        assert!(matches!(mode, RelayMode::Custom(_)));
    }

    #[test]
    fn relay_mode_with_token_is_custom() {
        let config = SyncConfig::new(std::env::temp_dir()).with_relay_auth_token("secret");
        let mode = SyncEngine::relay_mode(&config).expect("relay url must parse");
        assert!(matches!(mode, RelayMode::Custom(_)));
    }

    // `android_relay_resolver` builds an opaque resolver, so this guards only that
    // the IP-parse split and the empty-list DoH fallback never panic — across
    // injected IPv4/IPv6, unparseable entries, a mix, and an empty list.
    #[test]
    fn android_relay_resolver_builds_for_all_ip_shapes() {
        let cases: &[&[&str]] = &[
            &[],                                        // empty → DoH fallback
            &["220.233.46.218"],                        // IPv4
            &["220.233.46.218", "104.16.0.1"],          // multiple IPv4 (CF-proxied)
            &["2606:4700::6810:1"],                     // IPv6
            &["not-an-ip"],                             // all unparseable → fallback
            &["not-an-ip", "220.233.46.218"],           // mix → keeps the valid one
        ];
        for case in cases {
            let ips: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            let _ = android_relay_resolver("sync.minutist.ai", &ips);
        }
    }

    // The static resolver serves the seeded IPs for the relay host only, matches
    // the host case-insensitively and ignoring a trailing dot, splits v4/v6 across
    // the two lookups, and serves nothing for any other host (so a pkarr lookup
    // finds nothing rather than hitting the network).
    #[tokio::test]
    async fn static_relay_resolver_serves_only_the_relay_host() {
        use iroh::dns::Resolver;
        use std::net::{Ipv4Addr, Ipv6Addr};

        let r = StaticRelayResolver {
            host: "sync.minutist.ai".to_string(),
            v4: vec![Ipv4Addr::new(220, 233, 46, 218)],
            v6: vec![Ipv6Addr::new(0x2606, 0x4700, 0, 0, 0, 0, 0x6810, 0x1)],
        };

        // Relay host (with a trailing dot + odd case) → the seeded v4.
        let v4: Vec<_> = r.lookup_ipv4("SYNC.minutist.ai.".to_string()).await.unwrap().collect();
        assert_eq!(v4, vec![Ipv4Addr::new(220, 233, 46, 218)]);
        // The v6 lookup returns the seeded v6, not the v4.
        let v6: Vec<_> = r.lookup_ipv6("sync.minutist.ai".to_string()).await.unwrap().collect();
        assert_eq!(v6, vec![Ipv6Addr::new(0x2606, 0x4700, 0, 0, 0, 0, 0x6810, 0x1)]);
        // A different host (e.g. a pkarr `dns.iroh.link` lookup) → nothing.
        let other: Vec<_> = r.lookup_ipv4("dns.iroh.link".to_string()).await.unwrap().collect();
        assert!(other.is_empty());
        // No TXT records are ever served.
        let txt = r.lookup_txt("sync.minutist.ai".to_string()).await.unwrap().count();
        assert_eq!(txt, 0);
    }

    // adopt_from_peer skips a held meeting ONLY if fully materialised; a held-but-
    // incomplete meeting must be re-attempted. This guards both stranding shapes:
    // audio-without-notes (the field-test 41-stuck-dirs case) and notes-without-audio
    // (media pull failed after notes landed).
    #[test]
    fn meeting_is_materialised_requires_notes_metadata_and_audio() {
        use std::fs;
        let root = tempfile::TempDir::new().expect("tempdir");
        let mk = |id: MeetingId| {
            let dir = root.path().join(id.0.to_string());
            fs::create_dir_all(&dir).expect("mk meeting dir");
            dir
        };

        // Absent dir → not materialised.
        assert!(!meeting_is_materialised(root.path(), MeetingId(uuid::Uuid::new_v4())));

        // Audio only (media landed, notes/metadata did not) → not materialised.
        let a = MeetingId(uuid::Uuid::new_v4());
        let da = mk(a);
        fs::write(da.join("audio.opus"), b"x").expect("w");
        assert!(!meeting_is_materialised(root.path(), a));
        fs::write(da.join("notes.ydoc"), b"x").expect("w"); // + notes, still no metadata
        assert!(!meeting_is_materialised(root.path(), a));
        fs::write(da.join("metadata.json"), b"{}").expect("w"); // all three
        assert!(meeting_is_materialised(root.path(), a));

        // Notes + metadata but NO audio (media pull failed) → not materialised, so
        // adopt re-attempts the media.
        let b = MeetingId(uuid::Uuid::new_v4());
        let db = mk(b);
        fs::write(db.join("notes.ydoc"), b"x").expect("w");
        fs::write(db.join("metadata.json"), b"{}").expect("w");
        assert!(!meeting_is_materialised(root.path(), b));
    }

    #[test]
    fn empty_relay_url_is_rejected() {
        let mut config = SyncConfig::new(std::env::temp_dir());
        config.relay_url = String::new();
        assert!(SyncEngine::relay_mode(&config).is_err());
    }

    /// F1b: a burst of near-simultaneous connections from ONE peer (mirroring
    /// notes/media/discovery each dialling separately in one sync session) must
    /// coalesce to a single "fire" within the debounce window; a different peer's
    /// arrival is tracked independently and always fires on its first connection.
    #[test]
    fn peer_arrival_tracker_coalesces_a_burst_from_one_peer() {
        let tracker = PeerArrivalTracker::new();
        let peer = iroh::SecretKey::generate().public();

        assert!(
            tracker.note_connection(peer),
            "the first connection from a peer must fire"
        );
        assert!(
            !tracker.note_connection(peer),
            "a second connection within the debounce window must not re-fire"
        );
        assert!(
            !tracker.note_connection(peer),
            "a third connection within the debounce window must not re-fire either"
        );

        let other = iroh::SecretKey::generate().public();
        assert!(
            tracker.note_connection(other),
            "a different peer's first connection must fire independently"
        );
    }

    /// A peer silent for longer than the debounce window is treated as a fresh
    /// visit — the burst-coalescing must not permanently suppress a genuine later
    /// reconnection. Ages the recorded arrival directly (no real sleep) so the
    /// test stays fast and deterministic.
    #[test]
    fn peer_arrival_tracker_refires_after_the_debounce_window_elapses() {
        let tracker = PeerArrivalTracker::new();
        let peer = iroh::SecretKey::generate().public();
        assert!(tracker.note_connection(peer));

        tracker.last_seen.lock().expect("tracker lock").insert(
            peer,
            Instant::now() - PEER_ARRIVAL_DEBOUNCE - std::time::Duration::from_millis(1),
        );

        assert!(
            tracker.note_connection(peer),
            "a connection after the debounce window has elapsed must fire again"
        );
    }

    #[tokio::test]
    async fn ticket_round_trips_an_endpoint_addr() {
        let dir_a = tempfile::TempDir::new().expect("tempdir a");
        let dir_b = tempfile::TempDir::new().expect("tempdir b");
        let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
        let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");
        let engine_a = SyncEngine::start_direct(id_a, dir_a.path().to_path_buf())
            .await
            .expect("engine a");
        let engine_b = SyncEngine::start_direct(id_b, dir_b.path().to_path_buf())
            .await
            .expect("engine b");

        // A exports its ticket; B parses it back and registers A as a peer. The
        // parsed id must equal A's endpoint id, and the peer must appear in B's
        // directory.
        let ticket = engine_a.my_ticket();
        let parsed = engine_b
            .add_peer_from_ticket(&ticket)
            .expect("import ticket");
        assert_eq!(parsed, engine_a.endpoint_id());
        assert_eq!(
            engine_b.peer_ids(),
            vec![engine_a.endpoint_id().to_string()]
        );

        engine_a.shutdown().await.expect("shutdown a");
        engine_b.shutdown().await.expect("shutdown b");
    }

    #[test]
    fn malformed_ticket_is_rejected() {
        // Build a real engine via the relay-less path so the directory exists, then
        // feed it garbage.
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(async {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
            let engine = SyncEngine::start_direct(id, dir.path().to_path_buf())
                .await
                .expect("engine");
            assert!(matches!(
                engine.add_peer_from_ticket("not-a-ticket"),
                Err(Error::Protocol(_))
            ));
            engine.shutdown().await.expect("shutdown");
        });
    }

    #[tokio::test]
    async fn add_account_peer_registers_a_valid_id_and_relay() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
        let engine = SyncEngine::start_direct(id, dir.path().to_path_buf())
            .await
            .expect("engine");

        let other = iroh::SecretKey::generate().public();
        engine
            .add_account_peer(&other.to_string(), "https://sync.example/relay", &[])
            .expect("add account peer");

        assert_eq!(engine.peer_ids(), vec![other.to_string()]);
        engine.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn add_account_peer_accepts_direct_addrs_and_skips_unparseable() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
        let engine = SyncEngine::start_direct(id, dir.path().to_path_buf())
            .await
            .expect("engine");

        let other = iroh::SecretKey::generate().public();
        // A good direct addr and an unparseable one — the peer still registers
        // (the bad addr is skipped, never fails the whole registration).
        engine
            .add_account_peer(
                &other.to_string(),
                "https://sync.example/relay",
                &["100.82.58.55:41641".to_string(), "not-an-addr".to_string()],
            )
            .expect("add account peer with direct addrs");

        assert_eq!(engine.peer_ids(), vec![other.to_string()]);
        engine.shutdown().await.expect("shutdown");
    }

    #[test]
    fn publishable_direct_addr_filter_keeps_reachable_drops_garbage() {
        let keep = |s: &str| is_publishable_direct_addr(&s.parse().unwrap());
        // Tailscale CGNAT, private LAN, and public are published.
        assert!(keep("100.82.58.55:41641"), "tailscale CGNAT");
        assert!(keep("192.168.0.9:41641"), "private LAN");
        assert!(keep("10.1.2.3:41641"), "private LAN 10/8");
        assert!(keep("203.0.113.7:41641"), "public");
        // Loopback, link-local, and docker-bridge are dropped.
        assert!(!keep("127.0.0.1:41641"), "loopback");
        assert!(!keep("169.254.1.1:41641"), "link-local");
        assert!(!keep("172.17.0.1:41641"), "docker bridge");
        assert!(!keep("172.31.255.1:41641"), "docker bridge top of range");
        // A 172.x outside the docker range is a normal public/LAN addr — kept.
        assert!(keep("172.15.0.1:41641"), "172.15 is below the docker range");
        assert!(keep("172.32.0.1:41641"), "172.32 is above the docker range");
        // IPv6: global kept, loopback/link-local dropped.
        assert!(keep("[2001:db8::1]:41641"), "global v6");
        assert!(!keep("[::1]:41641"), "v6 loopback");
        assert!(!keep("[fe80::1]:41641"), "v6 link-local");
    }

    #[test]
    fn cgnat_and_rfc1918_classifiers() {
        let a = |s: &str| s.parse::<std::net::SocketAddr>().unwrap();
        assert!(is_cgnat_v4(&a("100.64.0.1:1")));
        assert!(is_cgnat_v4(&a("100.127.255.255:1")));
        assert!(!is_cgnat_v4(&a("100.63.0.1:1")), "just below CGNAT");
        assert!(!is_cgnat_v4(&a("100.128.0.1:1")), "just above CGNAT");
        assert!(!is_cgnat_v4(&a("203.0.113.1:1")), "public");
        assert!(is_rfc1918_v4(&a("10.0.0.1:1")));
        assert!(is_rfc1918_v4(&a("192.168.0.9:1")));
        assert!(is_rfc1918_v4(&a("172.16.0.1:1")));
        assert!(!is_rfc1918_v4(&a("100.82.58.55:1")), "CGNAT is not RFC1918");
        assert!(!is_rfc1918_v4(&a("203.0.113.1:1")), "public is not RFC1918");
    }

    #[tokio::test]
    async fn add_account_peer_rejects_a_malformed_endpoint_id() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
        let engine = SyncEngine::start_direct(id, dir.path().to_path_buf())
            .await
            .expect("engine");

        assert!(matches!(
            engine.add_account_peer("not-hex", "https://sync.example/relay", &[]),
            Err(Error::Protocol(_))
        ));

        engine.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn add_account_peer_rejects_a_malformed_relay_url() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
        let engine = SyncEngine::start_direct(id, dir.path().to_path_buf())
            .await
            .expect("engine");

        let other = iroh::SecretKey::generate().public();
        assert!(matches!(
            engine.add_account_peer(&other.to_string(), "not a url", &[]),
            Err(Error::Endpoint(_))
        ));

        engine.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn peers_to_dial_excludes_dial_suppressed_peers() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
        let engine = SyncEngine::start_direct(id, dir.path().to_path_buf())
            .await
            .expect("engine");

        let reachable = iroh::SecretKey::generate().public();
        let stale = iroh::SecretKey::generate().public();
        engine.add_peer(EndpointAddr::new(reachable));
        engine.add_peer(EndpointAddr::new(stale));

        // Both are dialled before either is suppressed.
        assert_eq!(engine.peers_to_dial().len(), 2);

        // Drive `stale` past the failure threshold (default `max_fails`) so it is
        // suppressed; the sweep must then skip it while still dialling `reachable`.
        for _ in 0..crate::backoff::BackoffPolicy::default().max_fails {
            engine.backoff.on_dial_outcome(&stale.to_string(), false);
        }
        let to_dial = engine.peers_to_dial();
        assert_eq!(to_dial, vec![reachable], "only the un-suppressed peer is dialled");
        assert!(
            !to_dial.contains(&stale),
            "a dial-suppressed peer must be excluded from the sweep"
        );

        // A success clears suppression → `stale` re-enters the dial set.
        engine.backoff.on_dial_outcome(&stale.to_string(), true);
        assert_eq!(engine.peers_to_dial().len(), 2);

        engine.shutdown().await.expect("shutdown");
    }
}
