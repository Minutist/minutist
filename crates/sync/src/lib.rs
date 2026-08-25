//! Device-to-device sync: direct sync between a user's own paired devices over
//! [iroh] QUIC, dialled through the self-hosted relay at `sync.minutist.ai`.
//!
//! # Protocols
//!
//! One [`SyncEngine`] endpoint accepts two ALPNs. The primary one
//! ([`notes_proto::SYNC_ALPN`]) multiplexes four request/response protocols over a
//! single bidirectional stream, dispatched by a leading
//! [`notes_proto::StreamKind`] tag, so one paired-peer authorisation point (the
//! inbound `AcceptHook`) covers all four. Framing is shared (`frame`).
//!
//! - **Notes** ([`notes_proto`], tag 1): Yjs state-vector/diff exchange. The
//!   authoritative document is `notes-crdt`'s `notes.ydoc`; this crate never
//!   materialises a yrs `Doc` itself.
//! - **Media** ([`media_proto`], tag 2): a manifest of `(path, BLAKE3 hash)` for
//!   `audio.opus` and each note asset, then a pull of what is missing over the
//!   blobs ALPN. Blobs are immutable and content-addressed, so "pull what I lack
//!   by hash" is the whole rule.
//! - **Discovery** ([`discovery_proto`], tag 3): each side advertises the
//!   `(MeetingId, ProcessingLifecycle)` of every meeting it holds. This crate has
//!   no `persistence` edge, so it reads the local state to advertise and emits
//!   received states on [`SyncEngine::subscribe_lifecycle_events`] for a
//!   `persistence`-linked consumer to apply.
//! - **Artifacts** ([`artifacts_proto`], tag 4): `transcript.json` and
//!   `summary.md`. Unlike media these are mutable, so each manifest entry carries
//!   the authority that produced those exact bytes, and a peer entry is pulled
//!   only when it strictly supersedes the local one. That authority travels with
//!   the bytes and is never re-derived from `metadata.json`, whose `Processed`
//!   stamp propagates over Discovery independently: deriving from it would let a
//!   stale relay copy clobber a newer producer copy.
//!
//! The blobs ALPN ([`blobs`]) moves bytes for media and artifacts. Its accept side
//! takes the same paired-peer check as the primary ALPN, which `iroh-blobs`' own
//! handler would not apply.
//!
//! # Payload encryption
//!
//! Every frame on the sync ALPN is sealed with XChaCha20-Poly1305 under a subkey
//! of the account [`ContentKey`], at the one `frame::Framer` chokepoint all
//! four protocols pass through, with the stream's [`notes_proto::StreamKind`] tag
//! as AEAD additional data. A peer that passes the ed25519 membership check but
//! holds a different key therefore reads nothing and gets
//! [`Error::Unauthenticated`] on its first frame. [`SyncEngine::start`] requires
//! a key, so there is no way to bind an unencrypted engine.
//!
//! Blob bytes are not sealed and do not need to be: `iroh-blobs`' provider
//! answers by hash with no enumeration, and a hash reaches a peer only inside a
//! sealed manifest. **A blob hash on any unsealed path is a confidentiality
//! bug**, not untidiness. See `planning/DESIGN_sync-encryption.md` §4.
//!
//! # Peer addressing
//!
//! Peers are learned two ways, both additive into one
//! [`address_lookup::PeerDirectory`] backing iroh's `MemoryLookup`:
//!
//! - Account-mediated discovery ([`account`]), a periodic loop fed by a
//!   consumer-supplied [`account::AccountEndpointSource`], so this crate takes no
//!   HTTP dependency. The live path for all three frontends. Desktop and the hub
//!   drive this crate's loop; the phone (`sync-ffi`) runs its own list-and-add
//!   loop in TS against the same `/v1/account/devices` endpoints, calling
//!   [`SyncEngine::add_account_peer`] per device instead.
//! - Manual ticket exchange ([`SyncEngine::my_ticket`] /
//!   [`SyncEngine::add_peer_from_ticket`]), mutual: each side adds the other's.
//!   No caller left in any frontend. Both stay on `SyncEngine` and in the
//!   `sync-ffi` API surface, unused, pending a coordinated removal.
//!
//! A re-advertised peer's address set replaces rather than unions with the tracked
//! one, so a stale direct address cannot linger as a dead dial candidate that
//! aggravates iroh's per-remote path-churn actor.
//!
//! # Public API
//!
//! [`SyncEngine::start`] binds the endpoint and serves; `start_direct` is the
//! relay-less path for integration tests. Beyond pairing and the four initiator
//! calls, two composite operations drive a full pass:
//!
//! - [`SyncEngine::push_all_to_peer`] pushes every locally-held meeting to one
//!   peer: notes, media, then a discovery dial.
//! - [`SyncEngine::adopt_from_peer`] / [`SyncEngine::adopt_all`] pull: discover a
//!   peer's meetings, then fetch each one this device lacks or holds incompletely.
//!   A held meeting is skipped only when fully materialised (`notes.ydoc` +
//!   `metadata.json` + `audio.opus`), so a half-synced meeting is re-attempted
//!   rather than stranded on folder existence and a sweep self-heals across
//!   restarts.
//!
//! [`SyncEngine::subscribe_peer_events`] and
//! [`SyncEngine::subscribe_lifecycle_events`] are bounded broadcast channels a
//! host reacts to. [`SyncEngine::shutdown`] is the owning, graceful stop. Failed
//! dials are suppressed per-peer by [`backoff::BackoffRegistry`], independent of
//! directory membership.
//!
//! Workspace edges are `common` and `notes-crdt` only, which keeps this crate off
//! the C-heavy graph `persistence` pulls in so it cross-compiles for the phone
//! (`sync-ffi`). See `architecture/components.md` for the dependency table and
//! `Cargo.toml` for per-dependency rationale.
//!
//! [iroh]: https://docs.rs/iroh/1.0.0/iroh/

pub mod account;
pub mod address_lookup;
pub mod artifacts_proto;
pub mod backoff;
pub mod blobs;
pub mod content_key;
pub mod discovery_proto;
pub mod endpoint;
pub mod enrolment;
pub(crate) mod enrolment_proto;
pub(crate) mod frame;
pub mod identity;
pub(crate) mod key_file;
pub mod media_proto;
pub mod notes_proto;
pub(crate) mod timeouts;

use std::path::PathBuf;

pub use account::{
    peers_to_add, run_account_refresh_loop_v2, AccountEndpoint, AccountEndpointSource, RefreshSink,
};
pub use address_lookup::PeerSource;
pub use backoff::{BackoffPolicy, BackoffRegistry};
pub use content_key::ContentKey;
pub use enrolment::{safety_code, EnrolmentStore, Verdict};
pub use endpoint::{PendingEnrolment, SyncEngine, SyncEngineRefreshSink};
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

    /// A frame failed to authenticate under this device's content key. The peer
    /// holds a different key, the bytes were tampered with, or the frame came
    /// from another protocol. Kept distinct from [`Self::Protocol`] so a key
    /// mismatch is diagnosable rather than looking like a malformed peer, and
    /// from [`Self::Suppressed`] because this one warrants attention.
    #[error("unauthenticated: {0}")]
    Unauthenticated(String),

    /// The dial was refused locally: the peer is in failed-dial backoff. Distinct
    /// from [`Self::Endpoint`] so a caller can tell "we declined to try" from "the
    /// peer did not answer". The former is the backoff working as intended, so it
    /// is expected traffic rather than a fault to escalate.
    #[error("peer {0} is in failed-dial backoff")]
    Suppressed(String),

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

    /// The directory holding the per-meeting `{uuid}` folders: the meetings root
    /// the app uses (`{app-data}/meetings`), not the app-data base. The
    /// notes protocol resolves `{meetings_root}/{uuid}/notes.ydoc` through
    /// `persistence`.
    pub meetings_root: PathBuf,

    /// The app-data BASE, where this device's secrets live: the ed25519 device
    /// key ([`identity`]), the account content key ([`content_key`]) and the
    /// enrolment record ([`enrolment`]).
    ///
    /// Distinct from [`Self::meetings_root`], which is `{app-data}/meetings`.
    /// The caller loads the identity and key from here itself, but the engine
    /// needs the path too: adopting a key from a confirmed peer writes it, and
    /// that happens inside the inbound accept path rather than at construction.
    pub app_data_dir: PathBuf,

    /// The failed-dial backoff policy (threshold + exponential window) the
    /// [`SyncEngine`]'s [`BackoffRegistry`] applies to every dial. Carries no
    /// secret, so it is not redacted in [`Self`]'s hand-written `Debug`.
    pub backoff_policy: BackoffPolicy,

    /// Pre-resolved IPs for the relay host (the host of [`Self::relay_url`]),
    /// injected by the caller. Only consulted on Android, where iroh's default
    /// system-config resolver has no nameservers to read: `/etc/resolv.conf` is
    /// absent and the netlink route socket it falls back to is SELinux-denied for
    /// untrusted apps, so every in-app lookup fails and the relay never resolves.
    /// Injecting nameservers to query does not help under a full-tunnel VPN
    /// (Tailscale MagicDNS intercepts the app's raw UDP:53 to any resolver), so the
    /// mobile layer instead resolves the relay host through the OS resolver
    /// (`InetAddress.getAllByName`, which honours the VPN in every case) and passes
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
            .field("app_data_dir", &self.app_data_dir)
            .field("backoff_policy", &self.backoff_policy)
            .field("relay_ips", &self.relay_ips)
            .finish()
    }
}

impl SyncConfig {
    /// The default relay endpoint for the connected tier.
    pub const DEFAULT_RELAY_URL: &'static str = "https://sync.minutist.ai";

    /// Build a config for `app_data_dir` (the base holding this device's
    /// secrets) and `meetings_root` (the directory holding the per-meeting
    /// `{uuid}` folders), pinning [`Self::DEFAULT_RELAY_URL`] with no relay auth
    /// token. The token is set later via [`Self::with_relay_auth_token`] once the
    /// account service issues one.
    pub fn new(app_data_dir: PathBuf, meetings_root: PathBuf) -> Self {
        Self {
            relay_url: Self::DEFAULT_RELAY_URL.to_string(),
            relay_auth_token: None,
            meetings_root,
            app_data_dir,
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
