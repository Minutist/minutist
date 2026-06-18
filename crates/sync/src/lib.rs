//! Device-to-device sync (WS4-B).
//!
//! Synchronises a user's own paired devices directly over [iroh] QUIC, dialled
//! through the self-hosted relay at `sync.minutist.ai`. Two payloads ride the
//! one [`SyncEngine`] endpoint:
//!
//! - **Notes** — Yjs CRDT update frames over a small custom ALPN protocol
//!   ([`notes_proto`]). The authoritative document is `persistence`'s
//!   `notes.ydoc`; received updates are merged with
//!   [`persistence::NotesStore::apply_update`].
//! - **Meeting media** — audio (`audio.opus`) and note assets, content-addressed
//!   via [iroh-blobs] ([`blobs`]).
//!
//! Peers are learned out-of-band from the account service (the same connected-tier
//! surface as `tunnel-client`), so addressing uses iroh's `MemoryLookup` rather
//! than DNS/pkarr discovery.
//!
//! This crate is part of the CONNECTED feature surface. It lives in the workspace
//! unconditionally, but the `app-main -> sync` wiring + the `ipc-bridge` trait
//! injection land in a later slice (S5); the free build omits it.
//!
//! [iroh]: https://docs.rs/iroh/1.0.0/iroh/
//! [iroh-blobs]: https://docs.rs/iroh-blobs/

pub mod blobs;
pub mod endpoint;
pub mod identity;
pub mod notes_proto;

use std::path::PathBuf;

pub use endpoint::SyncEngine;
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
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// The relay the endpoint pins via `RelayMode::Custom` (e.g.
    /// `https://sync.minutist.ai`).
    pub relay_url: String,

    /// The application data directory. The device key is persisted under it
    /// (see [`identity`]); blob media is read from the per-meeting folders below
    /// it via `persistence`.
    pub app_data_dir: PathBuf,
}

impl SyncConfig {
    /// The default relay endpoint for the connected tier.
    pub const DEFAULT_RELAY_URL: &'static str = "https://sync.minutist.ai";

    /// Build a config for `app_data_dir`, pinning [`Self::DEFAULT_RELAY_URL`].
    pub fn new(app_data_dir: PathBuf) -> Self {
        Self {
            relay_url: Self::DEFAULT_RELAY_URL.to_string(),
            app_data_dir,
        }
    }
}
