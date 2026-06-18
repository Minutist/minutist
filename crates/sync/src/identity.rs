//! The device's ed25519 sync identity.
//!
//! iroh identifies an endpoint by its ed25519 key, so each device holds a stable
//! [`iroh::SecretKey`] whose public half is its `EndpointId`. The key is persisted
//! at `{app-data}/sync_node_key` with owner-only `0600` on Unix, mirroring the
//! credential-file discipline `src-tauri/src/tunnel.rs` uses for
//! `tunnel_device.json`.
//!
//! The account/device ids that scope the iroh `EndpointId` to a user's account
//! are the existing `StoredCredential.{account_id, device_id}` (issued during
//! tunnel pairing); S2 reuses those rather than minting separate ids here.

use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// The file under `{app-data}` holding the raw 32-byte ed25519 device key. `0600`
/// on Unix.
const KEY_FILE: &str = "sync_node_key";

/// A device's persistent sync identity — the ed25519 key whose public half is the
/// iroh `EndpointId` peers dial.
pub struct DeviceIdentity {
    #[allow(dead_code)] // TODO(S2): consumed by SyncEngine endpoint construction.
    secret_key: iroh::SecretKey,
}

impl DeviceIdentity {
    /// Path to the persisted key under `app_data_dir`.
    pub fn key_path(app_data_dir: &Path) -> PathBuf {
        app_data_dir.join(KEY_FILE)
    }

    /// Load the device key from `{app-data}/sync_node_key`, generating and
    /// persisting a fresh one on first run.
    ///
    /// TODO(S2): implement — read the 32 raw bytes (`SecretKey::from_bytes`) when
    /// present, else `SecretKey::generate`, write via the app's `0600`
    /// secret-file discipline (the analogue of `StoredCredential::store`), and
    /// return the loaded identity. Stubbed so the crate compiles without the
    /// filesystem + RNG wiring.
    pub fn load_or_generate(_app_data_dir: &Path) -> Result<Self> {
        Err(Error::Identity(
            "load_or_generate not implemented (S2)".into(),
        ))
    }

    /// The ed25519 public key (the iroh `EndpointId` peers dial).
    ///
    /// TODO(S2): expose once `SyncEngine` consumes the identity.
    pub fn public_key(&self) -> iroh::PublicKey {
        self.secret_key.public()
    }
}
