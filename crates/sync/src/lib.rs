//! Device-to-device sync (WS4-B).
//!
//! Synchronises a user's own paired devices directly over [iroh] QUIC, dialled
//! through the self-hosted relay at `sync.minutist.ai`.
//!
//! - **Notes** — Yjs CRDT update frames over the sync ALPN protocol
//!   ([`notes_proto`]) on the one [`SyncEngine`] endpoint. The authoritative
//!   document is `notes-crdt`'s `notes.ydoc`; received updates are merged with
//!   [`notes_crdt::NotesStore::apply_update`].
//! - **Meeting media** — audio (`audio.opus`) and note assets, content-addressed
//!   over `iroh-blobs` ([`blobs`]). The two sides exchange a media manifest of
//!   `(relative-path, BLAKE3 hash)` pairs ([`media_proto`]) over the same sync
//!   ALPN, then pull the blobs each is missing over the blobs ALPN, exporting to
//!   the per-meeting paths `notes-crdt` owns (via `MeetingFolder::ensure`).
//!
//! Both exchanges share one ALPN, dispatched by a leading
//! [`notes_proto::StreamKind`] tag, so one paired-peer authorisation point covers
//! both. Framing is shared ([`frame`]).
//!
//! Peers are learned out-of-band from the account service (the same connected-tier
//! surface as `tunnel-client`), so addressing uses iroh's `MemoryLookup` rather
//! than DNS/pkarr discovery. [`account`] carries the account-mediated peer-
//! discovery loop; manual pairing ([`SyncEngine::add_peer_from_ticket`]) is the
//! other, additive source that feeds the same [`address_lookup::PeerDirectory`].
//!
//! This crate is part of the CONNECTED feature surface. It lives in the workspace
//! unconditionally; the free build omits the `app-main -> sync` wiring.
//!
//! [iroh]: https://docs.rs/iroh/1.0.0/iroh/

pub mod account;
pub mod address_lookup;
pub mod artifacts_proto;
pub mod backoff;
pub mod blobs;
pub mod discovery_proto;
pub mod endpoint;
pub mod frame;
pub mod identity;
pub mod media_proto;
pub mod notes_proto;
pub mod peers;
pub(crate) mod timeouts;

use std::path::PathBuf;

pub use account::{
    peers_to_add, run_account_refresh_loop_v2, AccountEndpoint, AccountEndpointSource, RefreshSink,
};
pub use address_lookup::PeerSource;
pub use backoff::{BackoffPolicy, BackoffRegistry};
pub use endpoint::{SyncEngine, SyncEngineRefreshSink};
pub use identity::DeviceIdentity;

/// Errors raised by the sync crate. Converted to [`minutist_common::AppError`] at
/// the IPC boundary (the `From` impl lives here so callers towards `ipc-bridge`
/// get a stable `AppError`).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Loading or generating the persisted device key failed.
    #[error("device identity: {0}")]
    Identity(String),

    /// Building or binding the iroh endpoint failed.
    #[error("endpoint: {0}")]
    Endpoint(String),

    /// A sync protocol exchange failed.
    #[error("protocol: {0}")]
    Protocol(String),

    /// A filesystem operation failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<Error> for minutist_common::AppError {
    fn from(e: Error) -> Self {
        minutist_common::AppError::Internal {
            context: e.to_string(),
        }
    }
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Static configuration for the [`SyncEngine`].
///
/// `Debug` is hand-implemented to redact [`Self::relay_auth_token`]: the token
/// gates admission to the relay's `AccessControl`, so it must never reach a log
/// line through a derived `Debug`.
#[derive(Clone)]
pub struct SyncConfig {
    /// The relay the endpoint pins via `RelayMode::Custom` (e.g.
    /// `https://sync.minutist.ai`).
    pub relay_url: String,

    /// The shared access token the self-hosted relay requires. When set it is
    /// threaded into the relay config via `RelayConfig::with_auth_token`; the
    /// relay's `AccessControl` admits the connection only when it matches. `None`
    /// for relays that do not gate on a token (e.g. a local/direct test).
    pub relay_auth_token: Option<String>,

    /// The directory holding the per-meeting `{uuid}` folders — the meetings
    /// root the app uses (`{app-data}/meetings`), NOT the app-data base. The
    /// notes protocol resolves `{meetings_root}/{uuid}/notes.ydoc` through
    /// `persistence`. The device key is persisted at the app-data BASE
    /// (see [`identity`]), which the caller loads separately.
    pub meetings_root: PathBuf,

    /// The failed-dial backoff policy (threshold + exponential window) the
    /// [`SyncEngine`]'s [`BackoffRegistry`] applies to every dial. Carries no
    /// secret, so it is not redacted in [`Self`]'s hand-written `Debug`.
    pub backoff_policy: BackoffPolicy,

    /// Pre-resolved IPs for the relay host (the host of [`Self::relay_url`]),
    /// injected by the caller. Only consulted on Android, where iroh's default
    /// (system-config) resolver has no nameservers — `/etc/resolv.conf` is absent
    /// and the netlink route socket it falls back to is SELinux-denied for
    /// untrusted apps — so every in-app lookup fails and the relay never resolves.
    /// Injecting nameservers to query does not help under a full-tunnel VPN
    /// (Tailscale MagicDNS intercepts the app's raw UDP:53 to any resolver), so the
    /// mobile layer instead resolves the relay host through the OS resolver
    /// (`InetAddress.getAllByName` — which honours the VPN in every case) and passes
    /// the resulting IPs here. `SyncEngine::start` then serves them from a static
    /// resolver keyed on the relay host, so iroh does no in-app DNS at all and
    /// connects to the IP with the relay hostname preserved as the TLS SNI. Empty
    /// (the default, and every non-Android caller) falls back to a public DoH
    /// resolver as a cellular safety net; non-Android ignores this and keeps iroh's
    /// system resolver. Carries no secret, so it is not redacted in [`Self`]'s
    /// hand-written `Debug`.
    pub relay_ips: Vec<String>,
}

impl std::fmt::Debug for SyncConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redact the relay token; print only whether one is set.
        let redacted = self.relay_auth_token.as_ref().map(|_| "<redacted>");
        f.debug_struct("SyncConfig")
            .field("relay_url", &self.relay_url)
            .field("relay_auth_token", &redacted)
            .field("meetings_root", &self.meetings_root)
            .field("backoff_policy", &self.backoff_policy)
            .field("relay_ips", &self.relay_ips)
            .finish()
    }
}

impl SyncConfig {
    /// The default relay endpoint for the connected tier.
    pub const DEFAULT_RELAY_URL: &'static str = "https://sync.minutist.ai";

    /// Build a config for `meetings_root` (the directory holding the per-meeting
    /// `{uuid}` folders), pinning [`Self::DEFAULT_RELAY_URL`] with no relay auth
    /// token. The token is set later via [`Self::with_relay_auth_token`] once the
    /// account service issues one.
    pub fn new(meetings_root: PathBuf) -> Self {
        Self {
            relay_url: Self::DEFAULT_RELAY_URL.to_string(),
            relay_auth_token: None,
            meetings_root,
            backoff_policy: BackoffPolicy::default(),
            relay_ips: Vec::new(),
        }
    }

    /// Set the relay access token presented to the self-hosted relay.
    pub fn with_relay_auth_token(mut self, token: impl Into<String>) -> Self {
        self.relay_auth_token = Some(token.into());
        self
    }

    /// Override the failed-dial backoff policy. The consumer (desktop/headless)
    /// owns the actual threshold/window values; [`BackoffPolicy::default`] is a
    /// placeholder.
    pub fn with_backoff_policy(mut self, policy: BackoffPolicy) -> Self {
        self.backoff_policy = policy;
        self
    }
}
