//! The iroh endpoint that owns device-to-device sync.
//!
//! [`SyncEngine`] holds one [`iroh::Endpoint`] bound with the device's secret key,
//! a `RelayMode::Custom` pinning the self-hosted relay ([`SyncConfig::relay_url`],
//! plus its access token when configured), and a [`MemoryLookup`] for out-of-band
//! peer addressing rather than DNS/pkarr discovery.
//!
//! Two ALPNs are accepted on one [`Router`]: the sync protocol
//! ([`crate::notes_proto::SYNC_ALPN`], multiplexing notes, media, discovery and
//! artifacts behind a leading [`crate::notes_proto::StreamKind`] tag) and the
//! blobs protocol ([`iroh_blobs::ALPN`], moving bytes).
//!
//! Both accept sides authorise the remote against the paired-peer
//! [`PeerDirectory`] before serving anything, and pairing is mutual: each device
//! must add the other's ticket. [`AcceptHook`] reads the stream-kind tag and runs
//! the matching responder. [`AuthorizedBlobs`] wraps
//! [`iroh_blobs::BlobsProtocol`] and delegates only for a paired remote, which
//! that protocol does not do on its own: registered unguarded it serves any peer
//! that connects.
//!
//! `SyncEngine::dial` dials a peer; [`SyncEngine::sync_notes`],
//! [`SyncEngine::sync_media`] and [`SyncEngine::sync_artifacts`] dial and run the
//! initiator side for one meeting.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use iroh::endpoint::{presets, ConnectionError};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{
    endpoint::Connection, Endpoint, EndpointAddr, EndpointId, RelayConfig, RelayMap, RelayMode,
    RelayUrl,
};
use iroh_tickets::endpoint::EndpointTicket;
use minutist_common::{DeletionState, MeetingId, ProcessingLifecycle};
use tokio::sync::broadcast;

use crate::account::{AccountEndpoint, RefreshSink};
use crate::address_lookup::{PeerDirectory, PeerSource};
#[cfg(feature = "test-support")]
use crate::backoff::BackoffPolicy;
use crate::backoff::BackoffRegistry;
use crate::blobs::{BlobExchange, BlobStore};
use crate::content_key::{ContentKey, FrameCipher};
use crate::enrolment::{EnrolmentStore, PendingEnrolment, Verdict};
use crate::enrolment_proto;
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
    /// Seals and opens every frame on the sync ALPN, derived from the account
    /// content key (`planning/DESIGN_sync-encryption.md` §4). A peer that passes
    /// the ed25519 membership check but holds a different key gets
    /// [`Error::Unauthenticated`] on its first frame and reads nothing.
    ///
    /// Behind a lock because enrolment replaces it at runtime: adopting a key
    /// from a confirmed peer re-derives the cipher, and every in-flight exchange
    /// must pick up the new one. Read out by value rather than held as a guard,
    /// since the frame operations that use it are `async` and holding a guard
    /// across an await is how that deadlocks.
    keys: KeyStore,
    /// The app-data base, for writing a key adopted during enrolment and for the
    /// enrolment record. See [`SyncConfig::app_data_dir`].
    app_data_dir: PathBuf,
    /// The configured relay URL, kept so [`Self::push_all_to`] can address a peer
    /// relay-only (id + relay) without a stored direct address.
    relay_url: String,
    /// Fires a peer's hex endpoint id at most once per [`PEER_ARRIVAL_DEBOUNCE`]
    /// window of accepted inbound connections from it ("peer arrived"). One sync
    /// session (e.g. the desktop's `sync_now` for a meeting) opens several
    /// connections in quick succession, since notes, media and discovery each
    /// dial separately; [`PeerArrivalTracker`] coalesces these into a single
    /// event per visit. An always-on hub subscribes via
    /// [`Self::subscribe_peer_events`] and reciprocally pushes (see
    /// [`Self::push_all_to`]); the desktop ignores it. Bounded, so a lagging
    /// receiver drops the oldest ids and recovers on a
    /// [`broadcast::error::RecvError::Lagged`] by reconciling all known peers
    /// ([`Self::peer_ids`]), so no visit stays permanently missed. Hex strings,
    /// not [`EndpointId`], so a consumer such as the mobile FFI wrapper needs no
    /// `iroh` type.
    peer_events: broadcast::Sender<String>,
    /// Fires each `(MeetingId, ProcessingLifecycle, DeletionState)` received from a discovery
    /// exchange ([`crate::discovery_proto`]). A consumer in a crate depending on
    /// both `sync` and `persistence` (ipc-bridge / headless) subscribes via
    /// [`Self::subscribe_lifecycle_events`] and persists each via
    /// `persistence::apply_processing_lifecycle`; `sync` has no `persistence`
    /// edge, so it emits rather than writes. Bounded; a consumer that hits
    /// [`broadcast::error::RecvError::Lagged`] recovers by re-running discovery.
    lifecycle_events: broadcast::Sender<(MeetingId, ProcessingLifecycle, DeletionState)>,
}

/// The error a device that holds no content key gives for anything but
/// enrolment. Its own state, not the peer's: nothing is wrong with the peer, this
/// device simply has not been enrolled yet.
fn not_enrolled() -> Error {
    Error::Unauthenticated(
        "this device holds no account content key: confirm it from a device that does".to_string(),
    )
}

/// The account content key, shared between the engine and the router-owned
/// accept hook.
///
/// `Arc` because both hold it; `RwLock` because enrolment replaces it on a live
/// engine; `Option` because a device that has not been enrolled holds none and
/// must still bind and serve. Named rather than written out at each use so the
/// poison recovery and the not-enrolled error exist once.
#[derive(Debug, Clone, Default)]
struct KeyStore(Arc<RwLock<Option<ContentKey>>>);

impl KeyStore {
    fn new(key: Option<ContentKey>) -> Self {
        Self(Arc::new(RwLock::new(key)))
    }

    /// The frame cipher, or [`not_enrolled`].
    ///
    /// Derived per call rather than cached: one HKDF-SHA256 expand, on a path
    /// that runs a handful of times per connection and never per frame (the
    /// protocols take a `&FrameCipher` and `frame::Framer` reuses it). Caching
    /// it would mean storing a value derived from another value in the same
    /// struct, with an invariant that they never drift.
    fn cipher(&self) -> Result<FrameCipher> {
        self.read()
            .as_ref()
            .map(ContentKey::frame_cipher)
            .ok_or_else(not_enrolled)
    }

    /// The content key itself. Only enrolment wants this; everything else wants
    /// [`Self::cipher`].
    fn content_key(&self) -> Result<ContentKey> {
        self.read().clone().ok_or_else(not_enrolled)
    }

    fn is_present(&self) -> bool {
        self.read().is_some()
    }

    fn set(&self, key: ContentKey) {
        *self.0.write().unwrap_or_else(|e| e.into_inner()) = Some(key);
    }

    /// A poisoned key lock is recovered rather than propagated: the guarded
    /// value is a key, not an invariant a panicking writer could have left
    /// half-updated, and refusing to read it would take sync down for the
    /// process lifetime.
    fn read(&self) -> std::sync::RwLockReadGuard<'_, Option<ContentKey>> {
        self.0.read().unwrap_or_else(|e| e.into_inner())
    }
}

/// Capacity of the peer-arrived broadcast channel. Small by design: a lagging hub
/// subscriber drops the oldest ids, which the next inbound connection re-fires.
const PEER_EVENTS_CAP: usize = 64;

/// Capacity of the discovery lifecycle-event broadcast channel. Larger than the
/// peer-arrival channel because one discovery exchange emits one event per
/// meeting; a lagging consumer re-runs discovery to recover.
const LIFECYCLE_EVENTS_CAP: usize = 256;

/// Coalescing window for [`PeerArrivalTracker`]. One sync session against a peer
/// opens several connections back-to-back: the desktop's `sync_now` for a single
/// meeting dials notes, media and discovery separately, and the hub's own
/// `push_all_to` does the same per meeting it holds, all normally landing within
/// low single-digit seconds of each other. A peer accepted more recently than
/// this window does not re-fire "peer arrived"; one silent for longer is treated
/// as a fresh visit. Deliberately shorter than the hub's own post-event push
/// debounce (`MINUTIST_HUB_PUSH_DEBOUNCE_MS`, default 15s in `headless`), which
/// rate-limits how often a genuinely repeated arrival triggers a push, whereas
/// this window only suppresses redundant re-signalling of a single arrival
/// across its own burst of connections.
const PEER_ARRIVAL_DEBOUNCE: std::time::Duration = std::time::Duration::from_secs(3);

/// Coalesces the burst of inbound connections one sync session opens (notes,
/// media, artifacts and discovery each dial separately: see
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
    ) -> Pin<
        Box<
            dyn Future<
                    Output = std::result::Result<
                        iroh::dns::BoxIter<std::net::Ipv4Addr>,
                        iroh::dns::DnsError,
                    >,
                > + Send,
        >,
    > {
        let addrs: Vec<_> = if self.matches(&host) {
            self.v4.clone()
        } else {
            Vec::new()
        };
        Box::pin(async move { Ok(Box::new(addrs.into_iter()) as iroh::dns::BoxIter<_>) })
    }

    fn lookup_ipv6(
        &self,
        host: String,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = std::result::Result<
                        iroh::dns::BoxIter<std::net::Ipv6Addr>,
                        iroh::dns::DnsError,
                    >,
                > + Send,
        >,
    > {
        let addrs: Vec<_> = if self.matches(&host) {
            self.v6.clone()
        } else {
            Vec::new()
        };
        Box::pin(async move { Ok(Box::new(addrs.into_iter()) as iroh::dns::BoxIter<_>) })
    }

    fn lookup_txt(
        &self,
        _host: String,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = std::result::Result<
                        iroh::dns::BoxIter<iroh::dns::TxtRecordData>,
                        iroh::dns::DnsError,
                    >,
                > + Send,
        >,
    > {
        // Addressing is account-directory based, not pkarr/TXT discovery, so serve
        // no TXT records, so a pkarr lookup simply finds nothing rather than erroring.
        Box::pin(async move { Ok(Box::new(std::iter::empty()) as iroh::dns::BoxIter<_>) })
    }

    fn clear_cache(&self) {}

    fn reset(&self) -> Box<dyn iroh::dns::Resolver> {
        Box::new(self.clone())
    }
}

/// The DNS resolver for Android, where iroh's system-config resolver has no
/// nameservers to read: `/etc/resolv.conf` is absent and the netlink route socket
/// it falls back to is SELinux-denied for untrusted apps, so every in-app lookup
/// fails and the relay never resolves. See the note at the
/// [`SyncEngine::start`] call site.
///
/// Injecting nameservers does not survive a full-tunnel VPN, since Tailscale
/// MagicDNS intercepts the app's raw UDP:53 to any resolver. The caller instead
/// resolves the relay host through the OS resolver, which honours the VPN, and
/// passes the IPs as `relay_ips` for a [`StaticRelayResolver`] to serve with no
/// in-app DNS at all.
///
/// Falls back to Cloudflare DoH over `:443` only when no usable IP was injected: a
/// cellular safety net, whose certs carry the IPs as SANs so the IP doubles as the
/// TLS server name, and which survives carrier `:53` hijacking.
///
/// Compiled on every target so the host build type-checks the `iroh::dns` API,
/// though only Android calls it.
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
                tracing::warn!(target: "sync", ip = %ip, error = %e, "skipping unparseable injected relay IP");
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
    tracing::info!(target: "sync", "no usable injected relay IP; falling back to public DoH resolver");
    iroh::dns::DnsResolver::builder()
        .with_nameserver(
            "1.1.1.1:443"
                .parse()
                .expect("valid DoH nameserver socket addr"),
            DnsProtocol::Https,
        )
        .with_nameserver(
            "1.0.0.1:443"
                .parse()
                .expect("valid DoH nameserver socket addr"),
            DnsProtocol::Https,
        )
        .build()
}

/// Whether a held meeting is materialised enough for [`SyncEngine::adopt_from_peer`]
/// to skip re-syncing it: its authoritative notes CRDT (`notes.ydoc`), `metadata.json`,
/// and its audio file are all present. A held-but-incomplete meeting, with any of
/// the three missing (e.g. audio pulled but the notes pull failed on a prior sweep,
/// or vice versa), fails this and is re-attempted, so adopt self-heals rather than
/// stranding it on folder existence alone. The audio file (resolved by container,
/// not assumed `audio.opus`: a synced phone recording is `audio.m4a`) stands in for
/// media completeness; a meeting genuinely without audio simply re-syncs each
/// sweep, a cheap idempotent manifest exchange rather than a blob transfer.
fn meeting_is_materialised(meetings_root: &Path, meeting_id: MeetingId) -> bool {
    let dir = meetings_root.join(meeting_id.0.to_string());
    dir.join("notes.ydoc").is_file()
        && dir.join("metadata.json").is_file()
        && minutist_common::resolve_audio_path(&dir).is_some()
}

/// Whether a bound direct socket address is worth publishing to the account
/// directory for a peer to dial (0049).
///
/// Keeps what a peer on the same tailnet or LAN can plausibly reach: Tailscale
/// CGNAT (100.64.0.0/10), private LAN ranges (10/8, 192.168/16), and public
/// addresses. Drops what no off-host peer can route to: loopback, link-local,
/// unspecified, multicast, broadcast, and the docker-bridge range 172.16.0.0/12,
/// which a container host otherwise advertises ten unreachable addresses from.
///
/// That last one is a heuristic: a genuine 172.16/12 LAN is dropped too. Accepted
/// on a docker-heavy host, and refinable via default-route interface detection if
/// it bites. iroh treats direct addrs opportunistically and falls back to the
/// relay, so an over-filtered set costs a hop, never connectivity.
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
            // fe80::/10 unicast link-local; `Ipv6Addr::is_unicast_link_local` is
            // unstable, so match the prefix directly.
            let is_link_local = (ip.segments()[0] & 0xffc0) == 0xfe80;
            !(ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() || is_link_local)
        }
    }
}

/// Whether `addr` is a Tailscale CGNAT address (100.64.0.0/10): the marker that
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

/// QUIC application close code the responder uses when a frame fails to
/// authenticate, so the initiator can tell a wrong content key from a dropped
/// connection.
///
/// The responder is the side that detects it: in every one of the four protocols
/// the initiator writes first, so the initiator's own next read would otherwise
/// surface only "connection lost". Without this the one symptom a user's device
/// reports for an unenrolled peer would be indistinguishable from a flaky
/// network, which defeats the point of a distinct
/// [`Error::Unauthenticated`](crate::Error::Unauthenticated).
const CLOSE_UNAUTHENTICATED: u32 = 0x4d55; // "MU": Minutist, Unauthenticated.

/// Classify an exchange's outcome, then close the connection.
///
/// Order matters and is the reason this is one function rather than two lines at
/// each call site: `Connection::close` overwrites `close_reason()`, so the peer's
/// [`CLOSE_UNAUTHENTICATED`] has to be read off the connection before we close.
///
/// The close code is chosen from the outcome, so an authentication failure is
/// reported symmetrically: whichever side detects it tells the other, rather than
/// only the responder doing so and an initiator that could not open a reply
/// signalling a clean `…-done`.
fn finish_exchange<T>(conn: &Connection, done_reason: &[u8], result: Result<T>) -> Result<T> {
    let result = result.map_err(|e| classify_exchange_error(conn, e));
    match &result {
        Err(Error::Unauthenticated(_)) => {
            conn.close(CLOSE_UNAUTHENTICATED.into(), b"unauthenticated frame")
        }
        _ => conn.close(0u32.into(), done_reason),
    }
    result
}

/// Reclassify an exchange error as [`Error::Unauthenticated`] when the peer
/// closed the connection with [`CLOSE_UNAUTHENTICATED`].
///
/// The initiator's error is whatever its next stream operation produced (usually
/// a read failing with the connection gone), so the verdict has to be read off
/// the connection rather than the error.
fn classify_exchange_error(conn: &Connection, err: Error) -> Error {
    if matches!(err, Error::Unauthenticated(_)) {
        return err;
    }
    match conn.close_reason() {
        Some(ConnectionError::ApplicationClosed(close))
            if close.error_code.into_inner() == u64::from(CLOSE_UNAUTHENTICATED) =>
        {
            Error::Unauthenticated(
                "the peer could not authenticate our frames: it holds a different content key, \
                 so it is not enrolled on this account"
                    .to_string(),
            )
        }
        _ => err,
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
    pub async fn start(
        config: SyncConfig,
        identity: DeviceIdentity,
        content_key: Option<ContentKey>,
    ) -> Result<Self> {
        let relay_mode = Self::relay_mode(&config)?;
        let peers = PeerDirectory::new();
        let meetings_root = config.meetings_root.clone();
        let blobs = BlobStore::open(&meetings_root).await?;

        let builder = Endpoint::builder(presets::N0)
            .secret_key(identity.secret_key())
            .relay_mode(relay_mode)
            .alpns(vec![SYNC_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()])
            .address_lookup(peers.lookup());
        // Android's system-config DNS resolver has no nameservers to read:
        // `/etc/resolv.conf` is absent, and the netlink route socket it falls
        // back to is SELinux-denied for untrusted apps, so every in-app lookup
        // fails and the relay never resolves. Serve it instead from the
        // caller's pre-resolved IPs (`config.relay_ips`) via a static resolver
        // keyed on the relay host (empty → public DoH fallback; see
        // [`android_relay_resolver`]), the only hostname iroh resolves; this
        // also survives a full-tunnel VPN that intercepts a raw DNS query.
        // Non-Android keeps iroh's system resolver so the device's own DNS
        // (VPN / private DNS / corporate / split-horizon) is honoured.
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

        Ok(Self::assemble(
            endpoint,
            peers,
            blobs,
            meetings_root,
            config.app_data_dir,
            content_key,
            BackoffRegistry::new(config.backoff_policy),
            config.relay_url,
        ))
    }

    /// Assemble the engine from an already-bound endpoint: derive the frame
    /// cipher, open the two broadcast channels, and spawn the router.
    ///
    /// Shared by all three `start*` paths, which differ only in how they bind and
    /// in their backoff policy and relay URL. Keeping it in one place means the
    /// `AcceptHook` and `Self` field lists exist once, so a new field is one edit
    /// rather than three that can silently drift.
    // Eight collaborators, all distinct and all needed exactly once here. The
    // alternative is a parameter bundle that only ever exists to be destructured
    // on the next line, which is a lint workaround wearing an abstraction's
    // clothes. `meetings_root` and `app_data_dir` are the pair worth not
    // confusing; `SyncConfig::app_data_dir`'s doc is where that is said.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        endpoint: Endpoint,
        peers: PeerDirectory,
        blobs: BlobStore,
        meetings_root: PathBuf,
        app_data_dir: PathBuf,
        content_key: Option<ContentKey>,
        backoff: BackoffRegistry,
        relay_url: String,
    ) -> Self {
        let (peer_events, _rx) = broadcast::channel(PEER_EVENTS_CAP);
        let (lifecycle_events, _lrx) = broadcast::channel(LIFECYCLE_EVENTS_CAP);
        let keys = KeyStore::new(content_key);
        let router = Self::build_router(
            &endpoint,
            &blobs,
            &peers,
            // `peer_arrivals` is owned solely by the hook; `SyncEngine` never
            // queries arrival state, only the events the hook derives from it.
            AcceptHook {
                meetings_root: meetings_root.clone(),
                peers: peers.clone(),
                blobs: blobs.clone(),
                keys: keys.clone(),
                endpoint: endpoint.clone(),
                peer_events: peer_events.clone(),
                peer_arrivals: PeerArrivalTracker::new(),
                lifecycle_events: lifecycle_events.clone(),
                app_data_dir: app_data_dir.clone(),
            },
        );
        Self {
            endpoint,
            router,
            peers,
            backoff,
            blobs,
            meetings_root,
            keys,
            app_data_dir,
            relay_url,
            peer_events,
            lifecycle_events,
        }
    }

    /// The current frame cipher, by value.
    ///
    /// Cloned out of the lock rather than borrowed: the frame operations that
    /// use it are `async`, and holding a `RwLock` guard across an await is how
    /// that deadlocks. A `FrameCipher` is a 32-byte key, so the clone is free.
    fn cipher(&self) -> Result<FrameCipher> {
        self.keys.cipher()
    }

    /// The current account content key, by value. Only enrolment wants this;
    /// everything else wants [`Self::cipher`].
    fn content_key(&self) -> Result<ContentKey> {
        self.keys.content_key()
    }

    /// The enrolment record, read fresh from disk.
    ///
    /// Not cached: the decision is made by a human in whatever process is in
    /// front of them, which on a headless hub is a one-shot CLI running
    /// alongside this daemon. See [`crate::enrolment`], "Why it is read from
    /// disk every time".
    fn enrolment(&self) -> EnrolmentStore {
        EnrolmentStore::load(&self.app_data_dir)
    }

    /// Tell the engine what the account directory just reported, so a device
    /// with no content key can mint one if it turns out to be the first on its
    /// account (`planning/DESIGN_sync-encryption.md` §3.1).
    ///
    /// **Every consumer that discovers peers must call this**, once per poll,
    /// or a device that holds no key never gets one and every content operation
    /// fails with [`Error::Unauthenticated`] forever. Desktop and the hub reach
    /// it through [`crate::RefreshSink::on_account_poll`]; the phone drives its
    /// own directory loop above the FFI and calls this directly.
    ///
    /// A no-op once a key is held, which is the steady state.
    ///
    /// Minting is deferred to a poll rather than done at startup because startup
    /// cannot know the answer. A device that minted before polling would guess,
    /// and a device joining an existing account guesses wrong every time, ending
    /// up with a key no peer holds and failing every exchange while looking like
    /// a network fault.
    pub fn note_account_peers(&self, has_other_devices: bool) {
        if self.is_enrolled_self() {
            return;
        }
        if has_other_devices {
            tracing::debug!(
                target: "sync",
                "no content key and the account has other devices; waiting to be enrolled"
            );
            return;
        }
        match ContentKey::load_or_mint(&self.app_data_dir) {
            Ok(key) => self.keys.set(key),
            Err(e) => tracing::warn!(
                target: "sync",
                error = %e,
                "could not mint the account content key"
            ),
        }
    }

    /// Whether this device holds the account content key, i.e. has been enrolled.
    /// A device that has not been syncs nothing until a confirmed peer sends it
    /// the key.
    pub fn is_enrolled_self(&self) -> bool {
        self.keys.is_present()
    }

    /// The blob-exchange collaborators for one peer: the frame cipher, blob
    /// store, endpoint, peer id and meetings root that `media_proto` and
    /// `artifacts_proto` both need on their initiator side.
    fn blob_exchange<'a>(&'a self, peer: EndpointId, cipher: &'a FrameCipher) -> BlobExchange<'a> {
        BlobExchange {
            cipher,
            store: &self.blobs,
            endpoint: &self.endpoint,
            peer,
            root: &self.meetings_root,
        }
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
        hook: AcceptHook,
    ) -> Router {
        let blobs_protocol = iroh_blobs::BlobsProtocol::new(blobs.inner(), None);
        Router::builder(endpoint.clone())
            .accept(SYNC_ALPN, hook)
            .accept(
                iroh_blobs::ALPN,
                AuthorizedBlobs::new(blobs_protocol, peers.clone()),
            )
            .spawn()
    }

    /// Build a relay-less engine: `RelayMode::Disabled`, otherwise the same bind +
    /// router as [`Self::start`]. Peers reach each other over the direct addresses
    /// in their [`Self::endpoint_addr`], so no relay (and no relay token) is
    /// involved. Gated behind `test-support`: it is a test/local-only path, not
    /// part of the production sync surface (which always pins the relay).
    #[cfg(feature = "test-support")]
    pub async fn start_direct(
        identity: DeviceIdentity,
        content_key: Option<ContentKey>,
        meetings_root: PathBuf,
    ) -> Result<Self> {
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
        Ok(Self::assemble(
            endpoint,
            peers,
            blobs,
            meetings_root.clone(),
            // The relay-less test path keeps the secrets under the same tempdir.
            meetings_root,
            content_key,
            BackoffRegistry::new(BackoffPolicy::default()),
            // The relay-less test path never addresses a peer relay-only, so it
            // has no relay URL; `push_all_to` is unused here.
            String::new(),
        ))
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
    /// relay, whose certificate is self-signed and unknown to any CA: the same
    /// `CaTlsConfig::insecure_skip_verify` iroh's own relay test suite uses
    /// against that relay. The production relay client (via [`Self::start`])
    /// always verifies the real relay's certificate; this path is gated behind
    /// `test-support` and never reachable from the production build.
    #[cfg(feature = "test-support")]
    pub async fn start_insecure(
        config: SyncConfig,
        identity: DeviceIdentity,
        content_key: Option<ContentKey>,
    ) -> Result<Self> {
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
            "sync endpoint bound (insecure relay TLS, test relay only)"
        );

        Ok(Self::assemble(
            endpoint,
            peers,
            blobs,
            meetings_root,
            config.app_data_dir,
            content_key,
            BackoffRegistry::new(config.backoff_policy),
            config.relay_url,
        ))
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

    /// This device's [`EndpointId`] (its ed25519 public key), the address the
    /// account service publishes for peers to dial.
    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// This device's full [`EndpointAddr`] (id + current relay/direct addresses),
    /// the form a peer injects via [`Self::add_peer`] to dial back.
    pub fn endpoint_addr(&self) -> EndpointAddr {
        self.endpoint.addr()
    }

    /// This device's plausibly-reachable direct socket addresses ("ip:port"), for
    /// publishing to the account directory so a same-tailnet or same-LAN peer can
    /// dial without the relay (0049). Filtered by
    /// [`is_publishable_direct_addr`], so a peer does not burn dials on addresses
    /// it cannot route to.
    ///
    /// A device holding a Tailscale CGNAT (100.64.0.0/10) address also drops its
    /// RFC1918 addresses from the published set: such a peer reaches this device
    /// over the tailnet or over the relay, and a published LAN address would be a
    /// phantom candidate for a peer on another L2, or one whose full-tunnel VPN
    /// captures LAN traffic. iroh probes and re-probes it, and that path churn
    /// starves its per-remote actor until QUIC handshake datagrams drop. The
    /// tradeoff is that two devices sharing an L2 with only one on the tailnet
    /// fall back to the relay.
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

    /// Register a peer learned out-of-band (manually: a ticket, or the
    /// relay-less direct test path) so the endpoint can resolve and dial it.
    /// Tagged [`PeerSource::Manual`]. The [`PeerDirectory`] is shared with the
    /// bound endpoint, so a peer added after binding is picked up on the next
    /// dial.
    pub fn add_peer(&self, addr: EndpointAddr) {
        self.peers.add(addr, PeerSource::Manual);
    }

    /// Register a peer learned from the account service ([`crate::account`]),
    /// addressed by its hex endpoint id and relay URL: the string-keyed
    /// primitive `sync-ffi` wraps for the phone's account-directory loop.
    /// In-workspace consumers use [`Self::upsert_account_peer`] instead, driven by
    /// the account-refresh loop's [`RefreshSink`]. Builds the same
    /// `id + relay + directs` [`EndpointAddr`] shape [`Self::push_all_to`] dials
    /// with and registers it as [`PeerSource::Account`], directly rather than via
    /// [`Self::add_peer`], which would tag it [`PeerSource::Manual`]. No `iroh`
    /// type appears in the signature, so an FFI caller needs none.
    ///
    /// `direct_addrs` are the peer's published socket addresses ("ip:port");
    /// unparseable entries are skipped. When present, iroh dials them directly and
    /// falls back to the relay if they are unreachable.
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
    /// device list (reconcile: it left the account). Source-aware: a no-op
    /// (returns `false`) if `endpoint_id` is absent or was registered any other
    /// way (e.g. [`Self::add_peer_from_ticket`]).
    pub fn remove_account_peer(&self, endpoint_id: &str) -> Result<bool> {
        let id: EndpointId = endpoint_id.parse().map_err(|e| {
            Error::Protocol(format!("parsing account endpoint id {endpoint_id:?}: {e}"))
        })?;
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
    /// device's public addressing, not its secret key.
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

    /// Peers the account directory has offered that the user has not decided
    /// about yet, each with the six-digit code to compare
    /// (`planning/DESIGN_sync-encryption.md` §5).
    ///
    /// This is the list a frontend prompts on, and what `minutist-hub confirm`
    /// prints. Carries no `iroh` or crypto type, so the FFI boundary and the CLI
    /// both consume it directly.
    ///
    /// A peer already confirmed or already refused is absent: the decision
    /// persists, so the user is asked once rather than every poll.
    pub fn pending_enrolments(&self) -> Vec<PendingEnrolment> {
        let peer_ids: Vec<String> = self.peers.ids().iter().map(|id| id.to_string()).collect();
        crate::enrolment::pending_from(self.endpoint.id().as_bytes(), &peer_ids, &self.enrolment())
    }

    /// The code to compare for one specific peer, whatever its current verdict.
    ///
    /// [`Self::pending_enrolments`] covers the prompt; this covers a settings
    /// screen that shows the code for an already-enrolled device, so a user can
    /// re-check one against its screen.
    pub fn safety_code_for(&self, peer_id: &str) -> Result<String> {
        let id: EndpointId = peer_id
            .parse()
            .map_err(|e| Error::Protocol(format!("invalid endpoint id {peer_id:?}: {e}")))?;
        Ok(crate::enrolment::safety_code(
            self.endpoint.id().as_bytes(),
            id.as_bytes(),
        ))
    }

    /// Record that the user confirmed `peer_id` is a device they own.
    ///
    /// Records only: no dial, no network, returns immediately. Handing the peer
    /// the key is [`Self::offer_content_key`], deliberately separate. A user
    /// confirming a device that happens to be asleep must not sit through a dial
    /// timeout to have their decision recorded, and the decision is what has to
    /// be durable, and the transfer is retryable against it for as long as it
    /// stands.
    ///
    /// A caller that wants both in one step calls
    /// [`Self::confirm_and_offer`].
    ///
    /// `decided_at` is an RFC 3339 timestamp from the caller; this crate keeps no
    /// clock.
    pub fn confirm_enrolment(&self, peer_id: &str, decided_at: Option<String>) -> Result<()> {
        self.record_verdict(peer_id, Verdict::Confirmed, decided_at)?;
        tracing::info!(
            target: "sync",
            peer = %peer_id,
            "user confirmed a peer; it may now receive the account content key"
        );
        Ok(())
    }

    /// [`Self::confirm_enrolment`] followed by [`Self::offer_content_key`].
    ///
    /// The confirmation is recorded even when the transfer fails, so an
    /// unreachable peer costs a retry rather than the user's decision.
    pub async fn confirm_and_offer(&self, peer_id: &str, decided_at: Option<String>) -> Result<()> {
        self.confirm_enrolment(peer_id, decided_at)?;
        self.offer_content_key(peer_id).await
    }

    /// Every peer the user has confirmed.
    pub fn confirmed_peers(&self) -> Vec<String> {
        self.enrolment()
            .all()
            .into_iter()
            .filter(|(_, v)| *v == Verdict::Confirmed)
            .map(|(peer, _)| peer)
            .collect()
    }

    /// Hand the content key to every confirmed peer that has not received it,
    /// returning how many succeeded.
    ///
    /// This is what lets a confirmation be made anywhere and still take effect:
    /// a one-shot CLI (or a UI on a device whose peer is asleep) records the
    /// decision to disk, and the running engine finishes the job on its next
    /// pass. Failures are logged and left for the next sweep, since the decision
    /// stands until the user changes it.
    ///
    /// A no-op for a device that holds no key itself, which cannot hand one out.
    pub async fn deliver_pending_keys(&self) -> usize {
        if !self.is_enrolled_self() {
            return 0;
        }
        let mut delivered = 0;
        for peer in self.enrolment().awaiting_key() {
            match self.offer_content_key(&peer).await {
                Ok(()) => delivered += 1,
                Err(e) => tracing::debug!(
                    target: "sync",
                    %peer,
                    error = %e,
                    "could not deliver the content key yet; will retry"
                ),
            }
        }
        delivered
    }

    /// Record that the user rejected `peer_id`, and drop it.
    ///
    /// The refusal persists, so the user is not re-prompted for an endpoint they
    /// have already rejected: re-asking every poll is how a prompt gets trained
    /// into a reflex.
    ///
    /// **Refusing a peer that already received the key does not revoke its
    /// access to content it can already decrypt.** Dropping it stops this device
    /// talking to it, but the key it holds stays valid, because nothing rotates
    /// the account content key. Real revocation needs that rotation plus
    /// re-delivery to the remaining confirmed devices; until it exists, treat
    /// this as "stop dealing with that endpoint", not as un-enrolment.
    ///
    /// Dropping it from the peer directory is what makes the refusal bite, and
    /// it does so in both directions at once, because directory membership is
    /// what the inbound accept hook authorises on and what an outbound dial
    /// needs an address from. A refused endpoint is therefore neither dialled
    /// nor admitted. The account-refresh loop will see it again on its next poll
    /// and must not re-add it, which
    /// [`SyncEngineRefreshSink::upsert_account_peer`] enforces by consulting the
    /// same record.
    ///
    /// Deliberately NOT expressed through the dial-failure backoff. That counter
    /// suppresses only after several consecutive failures and is cleared by any
    /// later success, so one refusal through it would suppress nothing and would
    /// evaporate on the next successful dial. A standing human verdict and a
    /// transient network-health counter have different lifetimes and must not
    /// share a mechanism.
    pub fn refuse_enrolment(&self, peer_id: &str, decided_at: Option<String>) -> Result<()> {
        self.record_verdict(peer_id, Verdict::Refused, decided_at)?;
        if let Ok(id) = peer_id.parse::<EndpointId>() {
            self.peers.remove(id, PeerSource::Account);
            self.peers.remove(id, PeerSource::Manual);
        }
        tracing::warn!(
            target: "sync",
            peer = %peer_id,
            "user refused an endpoint the account directory offered; dropped from the peer directory"
        );
        Ok(())
    }

    /// Whether the user has confirmed `peer_id`.
    pub fn is_enrolled(&self, peer_id: &str) -> bool {
        self.enrolment().is_confirmed(peer_id)
    }

    /// Whether the user has refused `peer_id`. Distinct from "not confirmed":
    /// an undecided peer is still prompted for, a refused one never is.
    pub fn is_refused(&self, peer_id: &str) -> bool {
        self.enrolment().verdict(peer_id) == Some(Verdict::Refused)
    }

    fn record_verdict(
        &self,
        peer_id: &str,
        verdict: Verdict,
        decided_at: Option<String>,
    ) -> Result<()> {
        self.enrolment().record(peer_id, verdict, decided_at)
    }

    /// Dial `peer_id` on the enrolment stream and send it the content key.
    ///
    /// Separate from [`Self::confirm_enrolment`] so a caller can retry the
    /// transfer for a peer that is already confirmed but was unreachable when the
    /// user decided.
    pub async fn offer_content_key(&self, peer_id: &str) -> Result<()> {
        let id: EndpointId = peer_id
            .parse()
            .map_err(|e| Error::Protocol(format!("invalid endpoint id {peer_id:?}: {e}")))?;
        let key = self.content_key()?;
        let store = self.enrolment();

        // Prefer the relay-addressed form the other `*_to_peer` methods use; on
        // the relay-less path (`start_direct`) there is no relay URL, and the
        // directory's own lookup resolves a bare id from what it was told.
        let addr = self
            .peer_relay_addr(peer_id)
            .unwrap_or_else(|_| EndpointAddr::new(id));
        let conn = self.dial(addr).await?;
        let result = enrolment_proto::offer_key(&conn, &store, &key).await;
        let result = finish_exchange(&conn, b"enrolment-done", result);
        match &result {
            Ok(()) => {
                self.record_peer_reachable(id);
                self.enrolment().mark_key_delivered(peer_id)?;
            }
            // The peer answered and declined. It is reachable and healthy; it
            // simply has not confirmed this device yet, which is the normal
            // window between the two halves of a mutual confirmation. Charging
            // that as a dial failure would suppress a good peer after a few
            // polls and stop its real notes and media syncs, for doing exactly
            // what it should.
            Err(Error::Unauthenticated(_)) => self.record_peer_reachable(id),
            Err(_) => self.record_exchange_failure(id),
        }
        result
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
    /// [`Self::discover_with`]). The consumer, in a crate depending on both
    /// `sync` and `persistence` (ipc-bridge / headless), persists each via
    /// `persistence::apply_processing_lifecycle`; `sync` has no `persistence`
    /// edge, so it emits rather than writes. Bounded: a consumer that hits
    /// [`broadcast::error::RecvError::Lagged`] recovers by re-running discovery.
    pub fn subscribe_lifecycle_events(
        &self,
    ) -> broadcast::Receiver<(MeetingId, ProcessingLifecycle, DeletionState)> {
        self.lifecycle_events.subscribe()
    }

    /// The meeting ids this device holds on disk: the `{uuid}` folders directly
    /// under [`Self::meetings_root`] (the dot-prefixed `.blobs` store and any
    /// non-UUID entry are skipped).
    pub fn local_meetings(&self) -> Vec<MeetingId> {
        discovery_proto::list_meeting_ids(&self.meetings_root)
    }

    /// Reconcile every meeting this device holds with `peer`, addressed relay-only
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
        let peer_hex = peer.to_string();
        for meeting in meetings {
            // Cheap early exit, as in `adopt_from_peer`. Returning rather than
            // breaking also skips the ride-along discovery dial below: `dial` would
            // refuse it anyway, and `discover_all` re-advertises the lifecycle.
            if self.backoff.is_suppressed(&peer_hex) {
                tracing::debug!(target: "sync", peer = %peer, reconciled, "push: peer suppressed mid-pass; abandoning the rest");
                return Ok(reconciled);
            }
            if let Err(e) = self.sync_notes(addr.clone(), meeting).await {
                tracing::warn!(target: "sync", peer = %peer, meeting = %meeting.0, error = %e, "push notes failed");
                continue;
            }
            if let Err(e) = self.sync_media(addr.clone(), meeting).await {
                tracing::warn!(target: "sync", peer = %peer, meeting = %meeting.0, error = %e, "push media failed");
                continue;
            }
            // Derived artifacts ride per meeting, after notes+media and before the
            // final ride-along discovery, so a peer that learns `Processed` via the
            // discovery exchange below has already had the transcript/summary
            // pulled (DESIGN §5 ordering invariant).
            if let Err(e) = self.sync_artifacts(addr.clone(), meeting).await {
                tracing::warn!(target: "sync", peer = %peer, meeting = %meeting.0, error = %e, "push artifacts failed");
                continue;
            }
            reconciled += 1;
        }

        // §7 ride-alongside: after reconciling notes+media, exchange lifecycle
        // with the peer in the same flow (a separate dial, run last), so a
        // meeting's processing state follows the meeting it was just pushed in.
        // This is ordering, not atomicity: a device that goes offline between the
        // notes push and this dial has not pushed its lifecycle; the periodic
        // recovery sweep (`discover_all`) backstops that window. Best-effort: a
        // discovery failure does not fail the push (the meetings are reconciled;
        // the lifecycle re-advertises on the next discovery).
        if let Err(e) = self.discover_with(addr.clone()).await {
            tracing::warn!(target: "sync", peer = %peer, error = %e, "ride-along discovery failed");
        }
        Ok(reconciled)
    }

    /// [`Self::push_all_to`] addressing the peer by its hex endpoint-id string,
    /// the form [`Self::peer_ids`] / [`Self::subscribe_peer_events`] hand back, so
    /// the hub's convergence loop (`headless`) never constructs an `iroh` type.
    pub async fn push_all_to_peer(&self, peer_id: &str) -> Result<usize> {
        let id: EndpointId = peer_id
            .parse()
            .map_err(|e| Error::Protocol(format!("parsing peer id {peer_id:?}: {e}")))?;
        self.push_all_to(id).await
    }

    /// Resolve a hex endpoint-id string to a relay-routed [`EndpointAddr`] (id +
    /// the configured relay), the addressing form [`Self::push_all_to`] uses.
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

    /// Dial a peer on the [`SYNC_ALPN`]. The peer must already be resolvable,
    /// either injected via [`Self::add_peer`] or passed as a full
    /// [`EndpointAddr`] carrying its relay or direct addresses. Crate-internal, so
    /// the public API carries no iroh-typed return; [`Self::connect`] is the
    /// test-only seam.
    ///
    /// Refuses a peer in failed-dial backoff before touching the network, so
    /// suppression holds for every exchange rather than only the sweeps that
    /// remember to pre-filter. The refusal is deliberately not recorded as a
    /// failure: recording one re-extends `retry_after`
    /// ([`BackoffRegistry::on_dial_outcome`]), which would make suppression
    /// self-perpetuating and the peer permanently unreachable.
    async fn dial(&self, peer: impl Into<EndpointAddr>) -> Result<Connection> {
        let addr: EndpointAddr = peer.into();
        let id = addr.id;
        if self.backoff.is_suppressed(&id.to_string()) {
            return Err(Error::Suppressed(id.to_string()));
        }
        let result = self
            .endpoint
            .connect(addr, SYNC_ALPN)
            .await
            .map_err(|e| Error::Endpoint(format!("dialling peer on sync alpn: {e}")));
        // Universal write side: every dial this device makes, desktop, headless,
        // and the phone's syncs (which flow through this same engine dial), feeds
        // the backoff registry, regardless of the peer's source.
        //
        // Records failure only. A completed QUIC handshake is not a working peer:
        // the protocol exchange on top of it can still fail every time (a decode
        // error, a truncated stream, an incompatible wire format). Recording
        // success here would wipe the accumulated failure count, since
        // `on_dial_outcome(_, true)` removes the peer's state entirely rather
        // than resetting a counter, leaving such a peer re-dialled at full rate
        // forever.
        if result.is_err() {
            self.backoff.on_dial_outcome(&id.to_string(), false);
        }
        result
    }

    /// Record a per-meeting exchange failure against the backoff registry.
    ///
    /// Failures only, and deliberately so. A sweep runs three streams per meeting
    /// (notes, media, artifacts), and clearing on any one success would let a
    /// working stream mask two broken ones: `on_dial_outcome(_, true)` removes the
    /// peer's state, so a peer whose notes sync succeeds while its media and
    /// artifacts consistently fail would reset to zero on every meeting and never
    /// reach `max_fails`. Only a peer-level health signal clears the count: see
    /// [`Self::record_peer_reachable`].
    fn record_exchange_failure(&self, peer: EndpointId) {
        self.backoff.on_dial_outcome(&peer.to_string(), false);
    }

    /// Clear a peer's accumulated failures after a successful discovery exchange.
    ///
    /// Discovery is the peer-level probe, one exchange per peer per sweep rather
    /// than per meeting, so its success is evidence the peer itself is healthy,
    /// which is the granularity suppression is about. Per-meeting streams
    /// deliberately do not clear (see [`Self::record_exchange_failure`]); without
    /// this the count would only grow, and a recovered peer would stay penalised
    /// until its backoff window elapsed.
    fn record_peer_reachable(&self, peer: EndpointId) {
        self.backoff.on_dial_outcome(&peer.to_string(), true);
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
    /// (`notes_proto::initiate_notes_sync`) against this device's
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
        let peer_id = conn.remote_id();
        let result = notes_proto::initiate_notes_sync(
            &conn,
            &self.cipher()?,
            &self.meetings_root,
            meeting_id,
        )
        .await;
        let result = finish_exchange(&conn, b"notes-sync-done", result);
        if result.is_err() {
            self.record_exchange_failure(peer_id);
        }
        result
    }

    /// Run a discovery exchange with `peer`: dial it on the [`SYNC_ALPN`] and run
    /// the initiator side (`discovery_proto::initiate_discovery`), learning the
    /// peer's `(MeetingId, ProcessingLifecycle, DeletionState)` for every meeting
    /// it holds. Each received state is emitted on
    /// [`Self::subscribe_lifecycle_events`] for the `ipc-bridge` or `headless`
    /// subscriber to persist; the returned ids are the peer's meeting list, and the
    /// caller fetches any it lacks.
    ///
    /// Discovery rides alongside a full sync
    /// (`planning/DESIGN_processing-lifecycle.md` §7): [`Self::push_all_to`] and
    /// the desktop's `sync_now` call it after a peer's notes and media, so a
    /// meeting's lifecycle travels in the session that pushed it rather than a
    /// skippable separate round. [`Self::discover_all`] drives it standalone as the
    /// hub's periodic recovery sweep.
    pub async fn discover_with(&self, peer: impl Into<EndpointAddr>) -> Result<Vec<MeetingId>> {
        let conn = self.dial(peer).await?;
        let peer_id = conn.remote_id();
        let result =
            discovery_proto::initiate_discovery(&conn, &self.cipher()?, &self.meetings_root).await;
        let result = finish_exchange(&conn, b"discovery-done", result);
        match &result {
            Ok(_) => self.record_peer_reachable(peer_id),
            Err(_) => self.record_exchange_failure(peer_id),
        }
        let theirs = result?;
        let ids = theirs.iter().map(|e| e.meeting_id).collect();
        for entry in theirs {
            let _ =
                self.lifecycle_events
                    .send((entry.meeting_id, entry.processing, entry.deletion));
        }
        Ok(ids)
    }

    /// The known peers this sweep should dial: every registered id except those in
    /// failed-dial backoff ([`BackoffRegistry::is_suppressed`]). A suppressed peer is
    /// skipped without dialling so a stale/unreachable peer does not burn the
    /// per-dial timeout on every sweep (0029 item 6); its backoff window elapsing
    /// is the retry, since once `retry_after` passes it re-appears here and is
    /// dialled again (a success then clears the suppression, a failure re-extends it).
    fn peers_to_dial(&self) -> Vec<EndpointId> {
        self.peers
            .ids()
            .into_iter()
            .filter(|id| !self.backoff.is_suppressed(&id.to_string()))
            .collect()
    }

    /// Run a discovery exchange with every known peer not in failed-dial backoff,
    /// relay-addressed (id + the configured relay): the hub's recovery sweep.
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
        // `self.peers.ids()` (not the string-keyed `Self::peer_ids`): this is
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

    /// Adopt from `peer_id` as a sync replica: discover the peer's meeting list,
    /// which also applies its lifecycle as [`Self::discover_with`] does, then pull
    /// notes, media and artifacts for every meeting this device lacks. Returns how
    /// many were newly adopted.
    ///
    /// The caller (the hub) runs no producer or election loop, so an adopted
    /// meeting is never claimed for processing; its lifecycle is applied as
    /// received. Notes are the anchor, carrying the doc; media and artifacts ride
    /// best-effort, and a per-meeting failure is logged and skipped so one bad
    /// meeting does not abort the set.
    ///
    /// `is_purged` stops a permanently-deleted meeting being resurrected. An unheld
    /// id a peer advertises is normally adopted, the peer being ahead, but if this
    /// device purged it the absence is deliberate and pulling it back would undo
    /// the purge from a peer that has not caught up. `sync` has no `persistence`
    /// edge, so `headless` supplies this from `PurgedStore::is_purged`. Soft
    /// deletion needs no such guard: a trashed meeting's folder still exists and
    /// converges over the ordinary lifecycle exchange.
    pub async fn adopt_from_peer(
        &self,
        peer_id: &str,
        is_purged: &impl Fn(MeetingId) -> bool,
    ) -> Result<usize> {
        let theirs = self.discover_with_peer(peer_id).await?;
        let mine = discovery_proto::list_meeting_ids(&self.meetings_root);
        let discovered = theirs.len();
        let mut adopted = 0usize;
        let mut recompleted = 0usize;
        let mut skipped_complete = 0usize;
        let mut skipped_purged = 0usize;
        let mut abandoned = 0usize;
        for (i, meeting_id) in theirs.into_iter().enumerate() {
            // A cheap early exit over the invariant `dial` enforces: once the peer
            // is suppressed every remaining exchange would be refused in-memory
            // anyway, so stop instead of logging a warning per meeting.
            // `peers_to_dial` gates only sweep entry, which is why this re-checks.
            if self.backoff.is_suppressed(peer_id) {
                abandoned = discovered - i;
                break;
            }
            // Skip a meeting only when it is already fully materialised locally,
            // not on mere folder existence. sync_notes creates the folder before
            // media and artifacts run, so a meeting whose notes landed but whose
            // media/artifacts pull failed (or vice versa) would, under a
            // folder-existence check, be treated as held and skipped on every
            // later sweep, stranded half-synced forever. Re-attempt an incomplete
            // held meeting; notes/media/artifacts sync is idempotent and manifest-
            // diffed, so re-running a complete one is a cheap no-op anyway.
            let held = mine.contains(&meeting_id);
            if held && meeting_is_materialised(&self.meetings_root, meeting_id) {
                skipped_complete += 1;
                continue;
            }
            if !held && is_purged(meeting_id) {
                skipped_purged += 1;
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
            skipped_purged,
            abandoned,
            "adopt: pass complete for peer"
        );
        Ok(adopted)
    }

    /// Adopt from every known peer not in failed-dial backoff: the hub's periodic
    /// replica sweep. Per-peer [`Self::adopt_from_peer`], relay-addressed like
    /// [`Self::discover_all`]. A per-peer failure is logged and skipped; returns
    /// the total meetings newly adopted across all peers this sweep. `is_purged`
    /// is threaded through to every [`Self::adopt_from_peer`] call: see its doc.
    pub async fn adopt_all(
        &self,
        is_purged: &(impl Fn(MeetingId) -> bool + Sync),
    ) -> Result<usize> {
        // Adopt from every peer concurrently: a dead or slow peer (e.g. a
        // backgrounded phone whose dial runs the full timeout) must not delay
        // the sweep for the peers behind it; serialising the sweep let one
        // unreachable peer starve the rest. Each future borrows `&self`; a
        // per-peer failure is logged and contributes zero to the total.
        let counts = futures_util::future::join_all(self.peers_to_dial().into_iter().map(
            |peer| async move {
                match self.adopt_from_peer(&peer.to_string(), is_purged).await {
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
    /// media-manifest protocol (`media_proto::initiate_media_sync`) against this
    /// device's [`Self::meetings_root`]. Each side imports its own media into the
    /// blob store, exchanges a manifest of `(relative-path, hash)` pairs, and
    /// pulls the blobs it is missing over the blobs ALPN, exporting each to the
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
        let cipher = self.cipher()?;
        let result = media_proto::initiate_media_sync(
            &conn,
            self.blob_exchange(peer_id, &cipher),
            meeting_id,
        )
        .await;
        let result = finish_exchange(&conn, b"media-sync-done", result);
        if result.is_err() {
            self.record_exchange_failure(peer_id);
        }
        result
    }

    /// Reconcile one meeting's derived artifacts (`transcript.json` + `summary.md`)
    /// with `peer`: dial it on the [`SYNC_ALPN`] and run the initiator side of the
    /// artifact-manifest protocol (`artifacts_proto::initiate_artifacts_sync`)
    /// against this device's [`Self::meetings_root`]. Each side imports its own
    /// artifacts into the blob store (stamping each entry with the authority for
    /// those exact bytes), exchanges a manifest, and pulls every entry that
    /// strictly supersedes its local copy over the blobs ALPN, exporting it
    /// atomically to the per-meeting path. On return both sides hold the
    /// authoritative copy of each artifact (a stale copy never overwrites a newer
    /// one: `planning/DESIGN_artifacts.md` §2).
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
        let cipher = self.cipher()?;
        let result = artifacts_proto::initiate_artifacts_sync(
            &conn,
            self.blob_exchange(peer_id, &cipher),
            meeting_id,
        )
        .await;
        let result = finish_exchange(&conn, b"artifacts-sync-done", result);
        if result.is_err() {
            self.record_exchange_failure(peer_id);
        }
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
    /// they become GC-eligible: see [`crate::blobs::BlobStore::delete_meeting_blobs`].
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
/// driving [`crate::account::run_account_refresh_loop_v2`] constructs this rather
/// than hand-rolling the trait. Most methods delegate to the matching engine
/// method; `on_new_peer` first-contact-dials via
/// [`SyncEngine::discover_with_peer`], a full exchange, so the meeting list and
/// lifecycle travel on the same dial where a plain connect would only prove
/// reachability.
///
/// Ownership contract: this sink and the account-refresh loop future both hold a
/// strong [`Arc<SyncEngine>`]. A consumer reclaiming sole ownership at shutdown for
/// a graceful drain (`Arc::into_inner` into an owning `shutdown(self)`) must await
/// the loop future's exit first; a signal-only cancel leaves a second strong ref
/// and the reclaim silently skips the graceful path. No `RefreshSink` method may
/// move this `Arc` into a task outliving the loop future.
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
    async fn on_account_poll(&self, has_other_devices: bool) {
        // Order matters: a key minted by the first call is what the second has
        // to hand out, so a first-device engine can enrol a peer in the same
        // pass rather than waiting for the next.
        self.engine.note_account_peers(has_other_devices);
        self.engine.deliver_pending_keys().await;
    }

    fn upsert_account_peer(&self, ep: &AccountEndpoint) -> bool {
        // A refused endpoint stays out, however often the directory re-offers
        // it. Without this the next poll would undo `refuse_enrolment`, and the
        // user's decision would last until the following tick.
        if self.engine.is_refused(&ep.endpoint_id) {
            tracing::debug!(
                target: "sync",
                endpoint_id = %ep.endpoint_id,
                "skipping an account peer the user refused"
            );
            return false;
        }
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
        // A first-contact dial that fails just waits for the next poll tick;
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
/// Authorises the remote against the paired-peer [`PeerDirectory`] before reading
/// a single frame, so a holder of the shared relay token who merely learns an
/// `EndpointId` cannot push updates or media into this device's meetings. Pairing
/// is mutual: each device must add the other's ticket.
///
/// It then accepts the initiator's bidirectional stream, reads the leading
/// [`StreamKind`] tag, and runs the matching responder against
/// [`SyncEngine::meetings_root`]. The media and artifacts responders also pull
/// blobs back from the initiator, so the hook carries clones of the blob store and
/// the endpoint; the router spawns a fresh task per connection. A failed exchange
/// becomes an [`AcceptError`] and does not bring the router down.
#[derive(Debug, Clone)]
struct AcceptHook {
    meetings_root: PathBuf,
    /// The authorised-peer set, shared with [`SyncEngine`] (cheap-to-clone, same
    /// backing store), so a peer paired after the router spawned is honoured on
    /// the next inbound connection.
    peers: PeerDirectory,
    /// The blob store, for the media responder.
    blobs: BlobStore,
    /// The content key and frame cipher, shared with [`SyncEngine`] (the same
    /// lock, so a key adopted during enrolment is picked up by both sides).
    /// Every responder seals and opens its frames under the cipher, so an
    /// inbound peer holding a different content key gets
    /// [`Error::Unauthenticated`] on its first frame.
    keys: KeyStore,
    /// The app-data base, for persisting a key adopted during enrolment.
    app_data_dir: PathBuf,
    /// The endpoint, for the media responder's blob pulls.
    endpoint: Endpoint,
    /// Fires the remote's hex id the first time it is authorised in a
    /// [`PEER_ARRIVAL_DEBOUNCE`] window, so an always-on hub can push back (see
    /// [`SyncEngine::subscribe_peer_events`]) once per visit rather than once per
    /// connection.
    peer_events: broadcast::Sender<String>,
    /// Coalesces the burst of connections one sync session opens into a single
    /// [`Self::peer_events`] fire per visit: see [`PeerArrivalTracker`].
    peer_arrivals: PeerArrivalTracker,
    /// Fires each `(MeetingId, ProcessingLifecycle, DeletionState)` received on an inbound
    /// discovery exchange (see [`SyncEngine::subscribe_lifecycle_events`]).
    lifecycle_events: broadcast::Sender<(MeetingId, ProcessingLifecycle, DeletionState)>,
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
        // can reciprocally push, but only on a genuine arrival (the first
        // connection from this peer in a PEER_ARRIVAL_DEBOUNCE window), so a
        // burst of connections from one sync session (notes/media/discovery each
        // dial separately) fires a single event, not one per connection.
        // Best-effort: `send` errors only with no receivers.
        if self.peer_arrivals.note_connection(peer) {
            let _ = self.peer_events.send(peer.to_string());
        }

        let outcome = self.dispatch(&connection, peer).await;
        if let Err(Error::Unauthenticated(ref reason)) = outcome {
            tracing::warn!(
                target: "sync",
                peer = %peer,
                %reason,
                "inbound sync frame failed to authenticate; peer holds a different content key"
            );
        }
        // Closes with `CLOSE_UNAUTHENTICATED` on an authentication failure, so the
        // initiator learns why instead of seeing a bare "connection lost". The
        // responder never sends a `…-done`: the initiator is the side that closes
        // a healthy exchange.
        finish_exchange(&connection, b"", outcome).map_err(AcceptError::from_err)
    }
}

impl AcceptHook {
    /// The enrolment record, read fresh from disk.
    ///
    /// Not cached: the decision is made by a human in whatever process is in
    /// front of them, which on a headless hub is a one-shot CLI running
    /// alongside this daemon. See [`crate::enrolment`], "Why it is read from
    /// disk every time".
    fn enrolment(&self) -> EnrolmentStore {
        EnrolmentStore::load(&self.app_data_dir)
    }

    /// The current frame cipher by value, mirroring [`SyncEngine::cipher`].
    fn cipher(&self) -> Result<FrameCipher> {
        self.keys.cipher()
    }

    /// The blob-exchange collaborators for one inbound peer, mirroring
    /// [`SyncEngine::blob_exchange`].
    fn blob_exchange<'a>(&'a self, peer: EndpointId, cipher: &'a FrameCipher) -> BlobExchange<'a> {
        BlobExchange {
            cipher,
            store: &self.blobs,
            endpoint: &self.endpoint,
            peer,
            root: &self.meetings_root,
        }
    }

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
                    &self.cipher()?,
                    &mut send,
                    &mut recv,
                    &self.meetings_root,
                )
                .await
            }
            StreamKind::Media => {
                media_proto::respond_media_sync(
                    connection,
                    self.blob_exchange(peer, &self.cipher()?),
                    &mut send,
                    &mut recv,
                )
                .await
            }
            StreamKind::Discovery => {
                let theirs = discovery_proto::respond_discovery(
                    connection,
                    &self.cipher()?,
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
                    let _ = self.lifecycle_events.send((
                        entry.meeting_id,
                        entry.processing,
                        entry.deletion,
                    ));
                }
                Ok(())
            }
            StreamKind::Artifacts => {
                artifacts_proto::respond_artifacts_sync(
                    connection,
                    self.blob_exchange(peer, &self.cipher()?),
                    &mut send,
                    &mut recv,
                )
                .await
            }
            StreamKind::Enrolment => {
                // The one stream not sealed under the content key: it delivers
                // that key. `accept_key` refuses unless this device's user has
                // confirmed the sender, and on acceptance the adopted key
                // re-derives the shared cipher so every subsequent exchange, on
                // both sides of this engine, uses it.
                let store = self.enrolment();
                let offered =
                    enrolment_proto::read_offered_key(connection, &mut recv, &store).await?;
                if let Some(bytes) = offered {
                    // Persist and swap BEFORE replying: the initiator treats
                    // ACCEPTED as "enrolled" and returns at once, so any work
                    // left after the reply is a window where it believes an
                    // enrolment that has not landed.
                    self.keys
                        .set(ContentKey::replace(&self.app_data_dir, bytes)?);
                }
                enrolment_proto::send_verdict(connection, &mut send, offered.is_some()).await
            }
        }
    }
}

/// The inbound-connection handler registered on the [`Router`] for the blobs ALPN
/// ([`iroh_blobs::ALPN`]).
///
/// Wraps [`iroh_blobs::BlobsProtocol`] with the same paired-peer authorisation
/// [`AcceptHook`] applies to the sync ALPN. This is a hard security requirement:
/// `BlobsProtocol` on its own serves a blob to any peer that connects (its
/// `accept` spawns the provider handler unconditionally), so registering it
/// unguarded would let any holder of the shared relay token who learns an
/// `EndpointId` read this device's meeting media. Rejecting an unpaired remote
/// before delegating to `BlobsProtocol::accept` means only a mutually-paired
/// peer can fetch a blob.
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
        let config = SyncConfig::new(std::env::temp_dir(), std::env::temp_dir());
        let mode = SyncEngine::relay_mode(&config).expect("default relay url must parse");
        assert!(matches!(mode, RelayMode::Custom(_)));
    }

    #[test]
    fn relay_mode_with_token_is_custom() {
        let config = SyncConfig::new(std::env::temp_dir(), std::env::temp_dir())
            .with_relay_auth_token("secret");
        let mode = SyncEngine::relay_mode(&config).expect("relay url must parse");
        assert!(matches!(mode, RelayMode::Custom(_)));
    }

    // `android_relay_resolver` builds an opaque resolver, so this guards only that
    // the IP-parse split and the empty-list DoH fallback never panic, across
    // injected IPv4/IPv6, unparseable entries, a mix, and an empty list.
    #[test]
    fn android_relay_resolver_builds_for_all_ip_shapes() {
        let cases: &[&[&str]] = &[
            &[],                               // empty → DoH fallback
            &["220.233.46.218"],               // IPv4
            &["220.233.46.218", "104.16.0.1"], // multiple IPv4 (CF-proxied)
            &["2606:4700::6810:1"],            // IPv6
            &["not-an-ip"],                    // all unparseable → fallback
            &["not-an-ip", "220.233.46.218"],  // mix → keeps the valid one
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
        let v4: Vec<_> = r
            .lookup_ipv4("SYNC.minutist.ai.".to_string())
            .await
            .unwrap()
            .collect();
        assert_eq!(v4, vec![Ipv4Addr::new(220, 233, 46, 218)]);
        // The v6 lookup returns the seeded v6, not the v4.
        let v6: Vec<_> = r
            .lookup_ipv6("sync.minutist.ai".to_string())
            .await
            .unwrap()
            .collect();
        assert_eq!(
            v6,
            vec![Ipv6Addr::new(0x2606, 0x4700, 0, 0, 0, 0, 0x6810, 0x1)]
        );
        // A different host (e.g. a pkarr `dns.iroh.link` lookup) → nothing.
        let other: Vec<_> = r
            .lookup_ipv4("dns.iroh.link".to_string())
            .await
            .unwrap()
            .collect();
        assert!(other.is_empty());
        // No TXT records are ever served.
        let txt = r
            .lookup_txt("sync.minutist.ai".to_string())
            .await
            .unwrap()
            .count();
        assert_eq!(txt, 0);
    }

    // adopt_from_peer skips a held meeting only if fully materialised; a held-but-
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
        assert!(!meeting_is_materialised(
            root.path(),
            MeetingId(uuid::Uuid::new_v4())
        ));

        // Audio only (media landed, notes/metadata did not) → not materialised.
        let a = MeetingId(uuid::Uuid::new_v4());
        let da = mk(a);
        fs::write(da.join("audio.opus"), b"x").expect("w");
        assert!(!meeting_is_materialised(root.path(), a));
        fs::write(da.join("notes.ydoc"), b"x").expect("w"); // + notes, still no metadata
        assert!(!meeting_is_materialised(root.path(), a));
        fs::write(da.join("metadata.json"), b"{}").expect("w"); // all three
        assert!(meeting_is_materialised(root.path(), a));

        // Notes + metadata but no audio (media pull failed) → not materialised, so
        // adopt re-attempts the media.
        let b = MeetingId(uuid::Uuid::new_v4());
        let db = mk(b);
        fs::write(db.join("notes.ydoc"), b"x").expect("w");
        fs::write(db.join("metadata.json"), b"{}").expect("w");
        assert!(!meeting_is_materialised(root.path(), b));
    }

    #[test]
    fn empty_relay_url_is_rejected() {
        let mut config = SyncConfig::new(std::env::temp_dir(), std::env::temp_dir());
        config.relay_url = String::new();
        assert!(SyncEngine::relay_mode(&config).is_err());
    }

    /// F1b: a burst of near-simultaneous connections from one peer (mirroring
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
    /// visit; the burst-coalescing must not permanently suppress a genuine later
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
        let engine_a = SyncEngine::start_direct(
            id_a,
            Some(ContentKey::for_tests()),
            dir_a.path().to_path_buf(),
        )
        .await
        .expect("engine a");
        let engine_b = SyncEngine::start_direct(
            id_b,
            Some(ContentKey::for_tests()),
            dir_b.path().to_path_buf(),
        )
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
            let engine = SyncEngine::start_direct(
                id,
                Some(ContentKey::for_tests()),
                dir.path().to_path_buf(),
            )
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
        let engine =
            SyncEngine::start_direct(id, Some(ContentKey::for_tests()), dir.path().to_path_buf())
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
        let engine =
            SyncEngine::start_direct(id, Some(ContentKey::for_tests()), dir.path().to_path_buf())
                .await
                .expect("engine");

        let other = iroh::SecretKey::generate().public();
        // A good direct addr and an unparseable one: the peer still registers
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
        // A 172.x outside the docker range is a normal public/LAN addr: kept.
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
        let engine =
            SyncEngine::start_direct(id, Some(ContentKey::for_tests()), dir.path().to_path_buf())
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
        let engine =
            SyncEngine::start_direct(id, Some(ContentKey::for_tests()), dir.path().to_path_buf())
                .await
                .expect("engine");

        let other = iroh::SecretKey::generate().public();
        assert!(matches!(
            engine.add_account_peer(&other.to_string(), "not a url", &[]),
            Err(Error::Endpoint(_))
        ));

        engine.shutdown().await.expect("shutdown");
    }

    /// Loopback [`EndpointAddr`] for `engine`: its id plus each bound port
    /// against `127.0.0.1`, so two in-process engines can dial each other.
    fn loopback_addr(engine: &SyncEngine) -> EndpointAddr {
        let mut addr = EndpointAddr::new(engine.endpoint_id());
        for sock in engine.bound_sockets() {
            addr = addr.with_ip_addr(std::net::SocketAddr::new(
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                sock.port(),
            ));
        }
        addr
    }

    /// The 0062 invariant, stated where it actually lives: a QUIC dial that
    /// succeeds must not clear failures accumulated by the protocol exchanges on
    /// top of it. `on_dial_outcome(_, true)` removes the peer's state outright, so
    /// recording success on connect resets the count on every attempt and a peer
    /// whose exchanges always fail is never suppressed.
    ///
    /// Two real loopback engines, because the distinguishing case needs a dial
    /// that genuinely completes; a registry-only test cannot express "connect
    /// worked".
    #[tokio::test]
    async fn a_successful_dial_does_not_clear_accumulated_exchange_failures() {
        let dir_a = tempfile::TempDir::new().expect("tempdir a");
        let dir_b = tempfile::TempDir::new().expect("tempdir b");
        let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
        let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");
        let a = SyncEngine::start_direct(
            id_a,
            Some(ContentKey::for_tests()),
            dir_a.path().to_path_buf(),
        )
        .await
        .expect("engine a");
        let b = SyncEngine::start_direct(
            id_b,
            Some(ContentKey::for_tests()),
            dir_b.path().to_path_buf(),
        )
        .await
        .expect("engine b");

        // Mutual pairing so b's accept hook authorises a.
        let b_addr = loopback_addr(&b);
        a.add_peer(b_addr.clone());
        b.add_peer(loopback_addr(&a));

        let b_hex = b.endpoint_id().to_string();
        let max_fails = crate::backoff::BackoffPolicy::default().max_fails;
        assert!(max_fails >= 2, "the test needs room below the threshold");

        // One short of suppression.
        for _ in 0..(max_fails - 1) {
            a.backoff.on_dial_outcome(&b_hex, false);
        }
        assert!(!a.is_suppressed(&b_hex), "not yet at the threshold");

        // A dial that completes at the QUIC layer. Recording it as a success would
        // wipe the count above.
        a.dial(b_addr).await.expect("loopback dial must succeed");

        // So the next failure is the one that crosses the threshold.
        a.backoff.on_dial_outcome(&b_hex, false);
        assert!(
            a.is_suppressed(&b_hex),
            "a successful connect must not have reset the accumulated failures"
        );

        a.shutdown().await.expect("shutdown a");
        b.shutdown().await.expect("shutdown b");
    }

    /// The counterpart: `dial` refuses a suppressed peer before touching the
    /// network, and does not record that refusal as a further failure (which would
    /// re-extend `retry_after` on every rejection and strand the peer for good).
    #[tokio::test]
    async fn dial_refuses_a_suppressed_peer_without_recording_a_failure() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
        let engine =
            SyncEngine::start_direct(id, Some(ContentKey::for_tests()), dir.path().to_path_buf())
                .await
                .expect("engine");

        let peer = iroh::SecretKey::generate().public();
        let peer_hex = peer.to_string();
        for _ in 0..crate::backoff::BackoffPolicy::default().max_fails {
            engine.backoff.on_dial_outcome(&peer_hex, false);
        }
        let before = engine.backoff.fails_for(&peer_hex);

        let err = engine
            .dial(EndpointAddr::new(peer))
            .await
            .expect_err("a suppressed peer must be refused");
        assert!(
            matches!(err, Error::Suppressed(ref id) if id == &peer_hex),
            "expected Error::Suppressed, got {err:?}"
        );
        assert_eq!(
            engine.backoff.fails_for(&peer_hex),
            before,
            "refusing a suppressed peer must not count as another failure"
        );

        engine.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn peers_to_dial_excludes_dial_suppressed_peers() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let id = DeviceIdentity::load_or_generate(dir.path()).expect("identity");
        let engine =
            SyncEngine::start_direct(id, Some(ContentKey::for_tests()), dir.path().to_path_buf())
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
        assert_eq!(
            to_dial,
            vec![reachable],
            "only the un-suppressed peer is dialled"
        );
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
