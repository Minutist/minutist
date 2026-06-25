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
//! [`SyncEngine::connect`] dials a peer; [`SyncEngine::sync_notes`] /
//! [`SyncEngine::sync_media`] dial and run the initiator side for one meeting.

use std::path::{Path, PathBuf};

use iroh::endpoint::presets;
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{
    endpoint::Connection, Endpoint, EndpointAddr, EndpointId, RelayConfig, RelayMap, RelayMode,
    RelayUrl,
};
use iroh_tickets::endpoint::EndpointTicket;
use minutist_common::MeetingId;
use tokio::sync::broadcast;

use crate::address_lookup::PeerDirectory;
use crate::blobs::BlobStore;
use crate::identity::DeviceIdentity;
use crate::notes_proto::{self, StreamKind, SYNC_ALPN};
use crate::{media_proto, Error, Result, SyncConfig};

/// Owns the iroh endpoint, the out-of-band peer directory, the content-addressed
/// blob store, and the router that accepts inbound sync connections for one
/// device.
pub struct SyncEngine {
    endpoint: Endpoint,
    router: Router,
    peers: PeerDirectory,
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
    /// Fires the [`EndpointId`] of a peer each time it opens an authorised inbound
    /// sync connection ("peer arrived"). An always-on hub subscribes via
    /// [`Self::subscribe_peer_events`] and reciprocally pushes (see
    /// [`Self::push_all_to`]); the desktop ignores it. Bounded — a lagging receiver
    /// drops the oldest ids, so the consumer is expected to recover on a
    /// [`broadcast::error::RecvError::Lagged`] by reconciling all known peers
    /// ([`Self::peer_ids`]); no arrival is then permanently missed.
    peer_events: broadcast::Sender<EndpointId>,
}

/// Capacity of the peer-arrived broadcast channel. Small by design: a lagging hub
/// subscriber drops the oldest ids, which the next inbound connection re-fires.
const PEER_EVENTS_CAP: usize = 64;

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

        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(identity.secret_key())
            .relay_mode(relay_mode)
            .alpns(vec![SYNC_ALPN.to_vec(), iroh_blobs::ALPN.to_vec()])
            .address_lookup(peers.lookup())
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
        let router =
            Self::build_router(&endpoint, &blobs, &peers, &meetings_root, peer_events.clone());

        Ok(Self {
            endpoint,
            router,
            peers,
            blobs,
            meetings_root,
            relay_url: config.relay_url,
            peer_events,
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
        peer_events: broadcast::Sender<EndpointId>,
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
        let router =
            Self::build_router(&endpoint, &blobs, &peers, &meetings_root, peer_events.clone());
        Ok(Self {
            endpoint,
            router,
            peers,
            blobs,
            meetings_root,
            // The relay-less test path never addresses a peer relay-only, so it has
            // no relay URL; `push_all_to` is unused here.
            relay_url: String::new(),
            peer_events,
        })
    }

    /// The bound socket addresses, used to build a direct [`EndpointAddr`] without
    /// a relay. Gated behind `test-support` alongside [`Self::start_direct`].
    #[cfg(feature = "test-support")]
    pub fn bound_sockets(&self) -> Vec<std::net::SocketAddr> {
        self.endpoint.bound_sockets()
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

    /// Register a peer learned out-of-band from the account service so the
    /// endpoint can resolve and dial it. The [`PeerDirectory`] is shared with the
    /// bound endpoint, so a peer added after binding is picked up on the next
    /// dial.
    pub fn add_peer(&self, addr: EndpointAddr) {
        self.peers.add(addr);
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

    /// The [`EndpointId`]s of every peer currently registered in the directory.
    /// The connected `SyncControl` syncs a meeting against each of them.
    pub fn peer_ids(&self) -> Vec<EndpointId> {
        self.peers.ids()
    }

    /// Subscribe to "peer arrived" events: the [`EndpointId`] of each peer as it
    /// opens an authorised inbound sync connection. An always-on hub reacts by
    /// calling [`Self::push_all_to`] so a device that reconnects both deposits and
    /// collects (convergence through the hub). The desktop does not subscribe.
    pub fn subscribe_peer_events(&self) -> broadcast::Receiver<EndpointId> {
        self.peer_events.subscribe()
    }

    /// The meeting ids this device holds on disk — the `{uuid}` folders directly
    /// under [`Self::meetings_root`] (the dot-prefixed `.blobs` store and any
    /// non-UUID entry are skipped).
    pub fn local_meetings(&self) -> Vec<MeetingId> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir(&self.meetings_root) else {
            return out;
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                if let Ok(uuid) = uuid::Uuid::parse_str(name) {
                    out.push(MeetingId(uuid));
                }
            }
        }
        out
    }

    /// Reconcile EVERY meeting this device holds with `peer`, addressed relay-only
    /// (id + the configured relay). This is the hub's reciprocal push: on a peer
    /// arriving it pushes all it holds, so a device that reconnects to deposit one
    /// meeting also collects every other meeting the hub has accumulated. Notes and
    /// media are reconciled per meeting; a per-meeting failure is logged and
    /// skipped so one bad meeting does not abort the rest. Returns how many
    /// meetings reconciled without error.
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
            reconciled += 1;
        }
        Ok(reconciled)
    }

    /// Dial a peer on the [`SYNC_ALPN`]. The peer must already be resolvable —
    /// either injected via [`Self::add_peer`] or passed as a full
    /// [`EndpointAddr`] carrying its relay/direct addresses.
    pub async fn connect(&self, peer: impl Into<EndpointAddr>) -> Result<Connection> {
        self.endpoint
            .connect(peer, SYNC_ALPN)
            .await
            .map_err(|e| Error::Endpoint(format!("dialling peer on sync alpn: {e}")))
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
        let conn = self.connect(peer).await?;
        let result = notes_proto::initiate_notes_sync(&conn, &self.meetings_root, meeting_id).await;
        conn.close(0u32.into(), b"notes-sync-done");
        result
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
        let conn = self.connect(peer).await?;
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

    /// Shut the router (and its endpoint) down gracefully, draining in-flight
    /// connections. Idempotent at the iroh layer.
    pub async fn shutdown(self) -> Result<()> {
        self.router
            .shutdown()
            .await
            .map_err(|e| Error::Endpoint(format!("shutting down sync router: {e}")))
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
/// the leading one-byte [`StreamKind`] tag, and runs the matching responder:
/// notes-sync ([`notes_proto::respond_notes_sync`]) or media-manifest
/// ([`media_proto::respond_media_sync`]) against the device's
/// [`SyncEngine::meetings_root`]. The media responder also needs the blob store
/// and the endpoint (to pull blobs back from the initiator), so the hook carries
/// clones of both (the router spawns a fresh task per connection). A failed
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
    /// Fires the remote's id once it is authorised, so an always-on hub can push
    /// back (see [`SyncEngine::subscribe_peer_events`]).
    peer_events: broadcast::Sender<EndpointId>,
}

impl AcceptHook {
    fn new(
        meetings_root: PathBuf,
        peers: PeerDirectory,
        blobs: BlobStore,
        endpoint: Endpoint,
        peer_events: broadcast::Sender<EndpointId>,
    ) -> Self {
        Self {
            meetings_root,
            peers,
            blobs,
            endpoint,
            peer_events,
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
        // can reciprocally push. Best-effort: `send` errors only with no receivers.
        let _ = self.peer_events.send(peer);

        self.dispatch(&connection, peer)
            .await
            .map_err(AcceptError::from_err)
    }
}

impl AcceptHook {
    /// Accept the initiator's bidirectional stream, read its leading
    /// [`StreamKind`] tag, and run the matching responder.
    async fn dispatch(&self, connection: &Connection, peer: EndpointId) -> Result<()> {
        let (mut send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|e| Error::Protocol(format!("accepting sync bi stream: {e}")))?;

        let mut tag = [0u8; 1];
        recv.read_exact(&mut tag)
            .await
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

    #[test]
    fn empty_relay_url_is_rejected() {
        let mut config = SyncConfig::new(std::env::temp_dir());
        config.relay_url = String::new();
        assert!(SyncEngine::relay_mode(&config).is_err());
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
        assert_eq!(engine_b.peer_ids(), vec![engine_a.endpoint_id()]);

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
}
