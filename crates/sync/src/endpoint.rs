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
//! loop and dispatches each `SYNC_ALPN` connection to the registered handler. The
//! per-connection sync exchange is the S3 hook: [`AcceptHook`] logs the peer and
//! returns, so the connection is accepted (proving identity + ALPN end-to-end)
//! without the notes-update logic. [`SyncEngine::connect`] dials a peer.

use iroh::endpoint::presets;
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{
    endpoint::Connection, Endpoint, EndpointAddr, EndpointId, RelayConfig, RelayMap, RelayMode,
    RelayUrl,
};

use crate::address_lookup::PeerDirectory;
use crate::identity::DeviceIdentity;
use crate::notes_proto::SYNC_ALPN;
use crate::{Error, Result, SyncConfig};

/// Owns the iroh endpoint, the out-of-band peer directory, and the router that
/// accepts inbound sync connections for one device.
pub struct SyncEngine {
    endpoint: Endpoint,
    router: Router,
    peers: PeerDirectory,
    config: SyncConfig,
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
            .accept(SYNC_ALPN, AcceptHook)
            .spawn();

        Ok(Self {
            endpoint,
            router,
            peers,
            config,
        })
    }

    /// Build a relay-less engine: `RelayMode::Disabled`, otherwise the same bind +
    /// router as [`Self::start`]. Peers reach each other over the direct addresses
    /// in their [`Self::endpoint_addr`], so no relay (and no relay token) is
    /// involved. Gated behind `test-support` — it is a test/local-only path, not
    /// part of the production sync surface (which always pins the relay).
    #[cfg(feature = "test-support")]
    pub async fn start_direct(identity: DeviceIdentity) -> Result<Self> {
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
            .accept(SYNC_ALPN, AcceptHook)
            .spawn();
        Ok(Self {
            endpoint,
            router,
            peers,
            config: SyncConfig::new(std::env::temp_dir()),
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

    /// Dial a peer on the [`SYNC_ALPN`]. The peer must already be resolvable —
    /// either injected via [`Self::add_peer`] or passed as a full
    /// [`EndpointAddr`] carrying its relay/direct addresses.
    pub async fn connect(&self, peer: impl Into<EndpointAddr>) -> Result<Connection> {
        self.endpoint
            .connect(peer, SYNC_ALPN)
            .await
            .map_err(|e| Error::Endpoint(format!("dialling peer on sync alpn: {e}")))
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
/// S3 hook: a real notes-sync exchange (accept a bi stream, read/merge Yjs update
/// frames) lands here. For S2 it accepts the connection, logs the peer, and waits
/// for close — enough to prove identity + ALPN negotiation end-to-end. The
/// integration test drives the bytes round-trip by opening its own bi stream from
/// the dial side and reading it back here.
#[derive(Debug, Clone)]
struct AcceptHook;

impl ProtocolHandler for AcceptHook {
    async fn accept(&self, connection: Connection) -> std::result::Result<(), AcceptError> {
        let peer = connection.remote_id();
        tracing::info!(target: "sync", peer = %peer, "accepted sync connection");

        // S3 hook: read the first bi stream and echo it back so the S2 test can
        // assert an end-to-end byte round-trip over the negotiated ALPN. The
        // notes-update merge replaces this body.
        let (mut send, mut recv) = connection.accept_bi().await?;
        let frame = recv
            .read_to_end(MAX_FRAME)
            .await
            .map_err(AcceptError::from_err)?;
        send.write_all(&frame)
            .await
            .map_err(AcceptError::from_err)?;
        send.finish().map_err(AcceptError::from_err)?;

        connection.closed().await;
        Ok(())
    }
}

/// Upper bound on a single inbound frame the S2 accept hook will buffer. Sized for
/// the test's probe payload; S3 sets the real notes-update bound.
const MAX_FRAME: usize = 64 * 1024;

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
}
