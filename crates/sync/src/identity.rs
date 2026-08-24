//! The device's ed25519 sync identity.
//!
//! iroh identifies an endpoint by its ed25519 key, so each device holds a stable
//! [`iroh::SecretKey`] whose public half is its `EndpointId`. The key is persisted
//! at `{app-data}/sync_node_key` (the raw 32 secret bytes) with owner-only `0600`
//! via `crate::key_file`, which it shares with the account content key.
//!
//! The account/device ids that scope the iroh `EndpointId` to a user's account
//! are the existing `StoredCredential.{account_id, device_id}` (issued during
//! account pairing); S2 reuses those rather than minting separate ids here.

use std::path::{Path, PathBuf};

use crate::{key_file, Error, Result};

/// The file under `{app-data}` holding the raw 32-byte ed25519 device key. `0600`
/// on Unix.
const KEY_FILE: &str = "sync_node_key";

/// A device's persistent sync identity: the ed25519 key whose public half is the
/// iroh `EndpointId` peers dial.
#[derive(Clone)]
pub struct DeviceIdentity {
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
    /// On a present file the raw 32 bytes are read and rebuilt with
    /// [`iroh::SecretKey::from_bytes`] (infallible). On first run a key is minted
    /// with [`iroh::SecretKey::generate`] (the 1.0 no-arg form, seeded internally
    /// from the OS RNG) and written `0600` via `crate::key_file::write`.
    pub fn load_or_generate(app_data_dir: &Path) -> Result<Self> {
        let path = Self::key_path(app_data_dir);
        match key_file::read(&path, "device key")? {
            Some(raw) => Ok(Self {
                secret_key: iroh::SecretKey::from_bytes(&raw),
            }),
            None => {
                let secret_key = iroh::SecretKey::generate();
                key_file::write(&path, &secret_key.to_bytes())
                    .map_err(|e| Error::Identity(format!("writing device key to {path:?}: {e}")))?;
                Ok(Self { secret_key })
            }
        }
    }

    /// The device secret key. Consumed by `SyncEngine` to build the iroh endpoint.
    pub(crate) fn secret_key(&self) -> iroh::SecretKey {
        self.secret_key.clone()
    }

    /// The ed25519 public key (the iroh `EndpointId` peers dial).
    pub fn public_key(&self) -> iroh::PublicKey {
        self.secret_key.public()
    }

    /// The iroh `EndpointId`: an alias for [`Self::public_key`], the form the
    /// account service publishes for out-of-band peer addressing.
    pub fn endpoint_id(&self) -> iroh::EndpointId {
        self.secret_key.public()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_and_reloads_a_stable_key() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let first = DeviceIdentity::load_or_generate(dir.path()).expect("generate");
        let reloaded = DeviceIdentity::load_or_generate(dir.path()).expect("reload");
        assert_eq!(
            first.public_key(),
            reloaded.public_key(),
            "the persisted key must reload identically"
        );
    }

    #[test]
    fn key_file_is_owner_only() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        DeviceIdentity::load_or_generate(dir.path()).expect("generate");
        let path = DeviceIdentity::key_path(dir.path());
        let meta = std::fs::metadata(&path).expect("stat key file");
        assert_eq!(meta.len(), 32, "the key file holds the raw 32 secret bytes");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = meta.permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "the device key must be owner-only");
        }
    }

    #[test]
    fn rejects_a_truncated_key_file() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(DeviceIdentity::key_path(dir.path()), b"short").expect("write");
        assert!(DeviceIdentity::load_or_generate(dir.path()).is_err());
    }
}
