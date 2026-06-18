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
//! - the notes ([`crate::notes_proto::SYNC_ALPN`]) protocol accepted on an iroh
//!   [`Router`]. The blobs ALPN multiplexes onto the same router in S4.
//!
//! The inbound accept side is owned by the [`Router`], which runs its own accept
//! loop and dispatches each `SYNC_ALPN` connection to [`AcceptHook`]. The hook
//! runs the responder side of the notes-sync protocol
//! ([`crate::notes_proto::respond_notes_sync`]) against the device's meetings
//! root. [`SyncEngine::connect`] dials a peer; [`SyncEngine::sync_notes`] dials
//! and runs the initiator side for one meeting.

use std::path::PathBuf;

use iroh::endpoint::presets;
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{
    endpoint::Connection, Endpoint, EndpointAddr, EndpointId, RelayConfig, RelayMap, RelayMode,
    RelayUrl,
};
use iroh_tickets::endpoint::EndpointTicket;
use minutist_common::MeetingId;

use crate::address_lookup::PeerDirectory;
use crate::identity::DeviceIdentity;
use crate::notes_proto::{self, SYNC_ALPN};
use crate::{Error, Result, SyncConfig};

/// Owns the iroh endpoint, the out-of-band peer directory, and the router that
/// accepts inbound sync connections for one device.
pub struct SyncEngine {
    endpoint: Endpoint,
    router: Router,
    peers: PeerDirectory,
    config: SyncConfig,
    /// Meetings root the notes protocol reads/writes through `persistence`
    /// (`{root}/{meeting_id}/notes.ydoc`). The inbound [`AcceptHook`] shares it.
    meetings_root: PathBuf,
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
        let meetings_root = config.app_data_dir.clone();

        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(identity.secret_key())
            .relay_mode(relay_mode)
            .alpns(vec![SYNC_ALPN.to_vec()])
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

        let router = Router::builder(endpoint.clone())
            .accept(SYNC_ALPN, AcceptHook::new(meetings_root.clone()))
            .spawn();

        Ok(Self {
            endpoint,
            router,
            peers,
            config,
            meetings_root,
        })
    }

    /// Build a relay-less engine: `RelayMode::Disabled`, otherwise the same bind +
    /// router as [`Self::start`]. Peers reach each other over the direct addresses
    /// in their [`Self::endpoint_addr`], so no relay (and no relay token) is
    /// involved. Gated behind `test-support` — it is a test/local-only path, not
    /// part of the production sync surface (which always pins the relay).
    #[cfg(feature = "test-support")]
    pub async fn start_direct(identity: DeviceIdentity, meetings_root: PathBuf) -> Result<Self> {
        let peers = PeerDirectory::new();
        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(identity.secret_key())
            .relay_mode(RelayMode::Disabled)
            .alpns(vec![SYNC_ALPN.to_vec()])
            .address_lookup(peers.lookup())
            .bind()
            .await
            .map_err(|e| Error::Endpoint(format!("binding iroh endpoint: {e}")))?;
        let router = Router::builder(endpoint.clone())
            .accept(SYNC_ALPN, AcceptHook::new(meetings_root.clone()))
            .spawn();
        Ok(Self {
            endpoint,
            router,
            peers,
            config: SyncConfig::new(meetings_root.clone()),
            meetings_root,
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

    /// The relay URL this engine is configured to pin.
    pub fn relay_url(&self) -> &str {
        &self.config.relay_url
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
/// Runs the responder side of the notes-sync protocol
/// ([`notes_proto::respond_notes_sync`]) against the device's
/// [`SyncEngine::meetings_root`], which it carries by clone (the router spawns a
/// fresh task per connection). A failed exchange is logged and converted to an
/// [`AcceptError`]; it does not bring the router down.
#[derive(Debug, Clone)]
struct AcceptHook {
    meetings_root: PathBuf,
}

impl AcceptHook {
    fn new(meetings_root: PathBuf) -> Self {
        Self { meetings_root }
    }
}

impl ProtocolHandler for AcceptHook {
    async fn accept(&self, connection: Connection) -> std::result::Result<(), AcceptError> {
        let peer = connection.remote_id();
        tracing::info!(target: "sync", peer = %peer, "accepted sync connection");

        notes_proto::respond_notes_sync(&connection, &self.meetings_root)
            .await
            .map_err(AcceptError::from_err)
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
        let parsed = engine_b.add_peer_from_ticket(&ticket).expect("import ticket");
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
