//! Device-to-device sync (WS4-B): direct sync between a user's own paired
//! devices over [iroh] QUIC, dialled through the self-hosted relay at
//! `sync.minutist.ai`.
//!
//! # Protocols
//!
//! One [`SyncEngine`] endpoint accepts two ALPNs. The primary ALPN
//! ([`notes_proto::SYNC_ALPN`]) multiplexes four request/response protocols over
//! a single bidirectional QUIC stream, dispatched by a leading
//! [`notes_proto::StreamKind`] tag — so one paired-peer authorisation point
//! (the inbound `AcceptHook`) covers all four. Framing is shared ([`frame`]).
//!
//! - **Notes** ([`notes_proto`], tag `1`) — Yjs CRDT state-vector/diff
//!   reconciliation. The authoritative document is `notes-crdt`'s `notes.ydoc`;
//!   received updates are merged with
//!   [`notes_crdt::NotesStore::apply_update`], which re-derives the `notes.json`
//!   / `notes.md` projections. `sync` never materialises a yrs `Doc` itself.
//! - **Media** ([`media_proto`], tag `2`) — a manifest of
//!   `(relative-path, BLAKE3 hash)` pairs for `audio.opus` and each note asset,
//!   then a pull of the missing blobs over the second ALPN ([`blobs`],
//!   `iroh_blobs::ALPN`), exported to the per-meeting paths `notes-crdt` owns
//!   (via `MeetingFolder::ensure`). Blobs are immutable and content-addressed,
//!   so "pull what I lack by hash" is the whole rule.
//! - **Discovery** ([`discovery_proto`], tag `3`) — each side advertises the
//!   `(MeetingId, ProcessingLifecycle)` of every meeting it holds, so a peer
//!   learns both which meetings exist and their host-authoritative processing
//!   state. `sync` has no `persistence` edge, so it only reads the local
//!   `processing` state to advertise and emits received states on
//!   [`SyncEngine::subscribe_lifecycle_events`] for a `persistence`-linked
//!   consumer to apply.
//! - **Artifacts** ([`artifacts_proto`], tag `4`) — reconciles a meeting's
//!   DERIVED outputs (`transcript.json` / `summary.md`, the files processing
//!   produces). Mirrors media's manifest-then-pull shape, but a derived
//!   artifact is MUTABLE (a meeting can be reprocessed), so each manifest entry
//!   carries the authority that produced those exact bytes (`produced_by` host
//!   + `produced_at`); a peer entry is pulled only when it strictly supersedes
//!   the local one. That authority travels WITH the bytes and is never
//!   re-derived from `metadata.json`, whose `Processed` stamp propagates over
//!   Discovery independently of the bytes — deriving from it would let a stale
//!   relay copy clobber a newer producer copy.
//!
//! The second ALPN ([`iroh_blobs::ALPN`], via [`blobs`]) moves blob bytes for
//! both the media and artifacts protocols; its accept side is guarded by the
//! same paired-peer check as the primary ALPN, so it never serves an unpaired
//! remote even though `iroh-blobs`' own protocol handler would.
//!
//! # Peer addressing
//!
//! Peers are learned two ways, additive into the one
//! [`address_lookup::PeerDirectory`], which backs iroh's `MemoryLookup` rather
//! than DNS/pkarr discovery:
//!
//! - manual, out-of-band ticket exchange — [`SyncEngine::my_ticket`] /
//!   [`SyncEngine::add_peer_from_ticket`]; pairing is mutual, each side must add
//!   the other's ticket;
//! - account-mediated discovery ([`account`]) — a periodic loop fed by a
//!   consumer-supplied [`account::AccountEndpointSource`], so this crate takes
//!   no HTTP or account-service dependency of its own.
//!
//! A re-advertised peer's address set REPLACES rather than unions with the
//! tracked one ([`address_lookup::PeerDirectory`]), so a stale direct address
//! (an old ephemeral port, a phantom cross-L2 LAN candidate on a tailnet
//! device) cannot linger as a dead dial candidate that aggravates iroh's
//! per-remote path-churn actor.
//!
//! # Dependency shape
//!
//! This crate depends on `common` (shared types/errors) and `notes-crdt` (the
//! `notes.ydoc` reader/writer and `MeetingFolder`) — nothing else in the
//! workspace. It never depends on `persistence`: the notes-CRDT primitives were
//! extracted into the leaf `notes-crdt` crate specifically so this crate's lib
//! stays off the C-heavy graph `persistence` pulls in (libsql / audiopus /
//! ogg), which lets it cross-compile to `aarch64-linux-android` for the phone
//! companion (the `sync-ffi` crate). `persistence::assets` (note-image
//! round-trips) is reached only as a dev-dependency, by this crate's own
//! integration tests — never by the crate's own lib.
//!
//! This crate is part of the CONNECTED feature surface. It compiles
//! unconditionally as a workspace member; the free build omits only the
//! `app-main -> sync` wiring (`ipc-bridge`'s `SyncControl` trait, backed by a
//! no-op implementation in the free build and this crate's engine in the
//! connected one — see `cross-cutting.md`, "Build variants").
//!
//! # Public API
//!
//! [`SyncEngine::start`] binds the iroh endpoint and starts serving;
//! `start_direct` is the relay-less path used by integration tests
//! (`test-support` feature only). Beyond pairing and the four per-protocol
//! initiator calls (`sync_notes` / `sync_media` / `sync_artifacts` /
//! `discover_with`), two composite operations drive a full reconciliation:
//!
//! - [`SyncEngine::push_all_to_peer`] reconciles every locally-held meeting
//!   (notes, then media, then a discovery dial) to one peer — the PUSH
//!   direction.
//! - [`SyncEngine::adopt_from_peer`] / [`SyncEngine::adopt_all`] are the PULL
//!   direction: discover a peer's meeting list, then pull every meeting this
//!   device lacks OR holds only incompletely (notes + media + artifacts). A
//!   held meeting is skipped only when FULLY materialised (`notes.ydoc` +
//!   `metadata.json` + `audio.opus` all present); a half-synced meeting — audio
//!   pulled but the notes pull failed on a prior sweep, or vice versa — is
//!   re-attempted rather than stranded on folder existence alone, so a sweep
//!   self-heals across restarts. This is the sync-replica hub's backfill path:
//!   the hub runs no producer/election loop, so an adopted meeting is never
//!   claimed for processing, only lifecycle-applied as received. Each pass logs
//!   one per-peer summary (discovered / adopted / recompleted /
//!   skipped_complete) at debug, so a sweep's decision is observable without
//!   per-meeting spam.
//!
//! [`SyncEngine::subscribe_peer_events`] / [`SyncEngine::subscribe_lifecycle_events`]
//! are the two bounded broadcast channels a host (the headless daemon, or a
//! desktop driver) reacts to. [`SyncEngine::shutdown`] is the owning, graceful
//! stop. Failed dials are tracked per-peer by [`backoff::BackoffRegistry`] and
//! exponentially suppressed, independent of directory membership, so a
//! recovery sweep does not burn a per-dial timeout on every pass against an
//! unreachable peer.
//!
//! # Third-party dependencies
//!
//! `iroh` (the QUIC transport, pinned EXACT) and `iroh-tickets` (the
//! `EndpointTicket` round-trip for manual pairing, pinned EXACT alongside the
//! same iroh 1.0 line). `iroh-blobs` (pinned EXACT `=0.103.0`, `fs-store`
//! feature) supplies the BLAKE3 blob store and `BlobsProtocol` handler for the
//! second ALPN; it depends on `iroh ^1.0.0`, and the workspace's `=1.0.0` pin is
//! in that range, so there is exactly one `iroh` in the dependency tree and the
//! endpoint/connection types unify across the accept/connect/download
//! boundary. `yrs` (the same workspace pin as `persistence`) backs the notes
//! CRDT diff. `uuid` decodes the fixed 16-byte meeting id off the wire into a
//! `common::MeetingId`. `futures-util` (workspace-pinned, already a dependency
//! of `tunnel-client`) supplies `StreamExt` over the blob downloader's progress
//! stream — capping a transfer mid-flight once it crosses the per-blob size
//! cap, rather than only discovering an oversized transfer after it has
//! already landed on disk — and `join_all` for concurrent per-peer dialling
//! (a dead or slow peer must not delay discovery/adopt for the peers behind
//! it). `tokio-util` supplies `CancellationToken`, the account-refresh loop's
//! cancel primitive — the same leaf `mcp-server` already carries, so it adds no
//! new crate to the dependency tree.
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

    /// The dial was refused locally: the peer is in failed-dial backoff. Distinct
    /// from [`Self::Endpoint`] so a caller can tell "we declined to try" from "the
    /// peer did not answer" — the former is the backoff working as intended and is
    /// expected traffic, not a fault to escalate.
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
