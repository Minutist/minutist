//! Connected-tier account sign-in wiring (WS4-A S5b).
//!
//! Compiled only in the `connected` build. Implements
//! `ipc_bridge::AccountControl` with `tunnel-client`'s device-code pairing
//! client, and owns the secure storage of the issued device credential.
//! `app-main` injects a [`ConnectedAccount`] into `IpcState.connected.account`;
//! the free build never builds this module and uses
//! `ipc_bridge::disabled_account()` instead.
//!
//! # The flow
//!
//! 1. `account_begin_pairing` → [`ConnectedAccount::begin_pairing`]: POST
//!    `/pair/start`, stash the `device_code` + interval, set status `Pairing`,
//!    return the `user_code` + verification URL for the UI to show + open.
//! 2. `account_poll_pairing` → [`ConnectedAccount::poll_pairing`]: POST
//!    `/pair/poll`. While pending, stays `Pairing` (bumping the interval on
//!    `slow_down`); on `authorised` it stores the credential (0600) and moves
//!    to `SignedIn`, which also turns sync on (F5) — there is no separate
//!    enable toggle; on a declined / expired pairing it returns `SignedOut`.
//!
//! # Credential storage
//!
//! The issued credential + account/device ids are stored at
//! `{app-data}/tunnel_device.json` with owner-only `0600` on Unix, the same
//! discipline as the `mcp_token` file (and the same Windows-ACL gap — see
//! `crate::write_secret_file`). The credential is the long-lived device
//! identity; it is never logged and never crosses to the webview. The
//! filename is unchanged from the earlier relay-tunnel design (avoids a
//! migration) even though there is no tunnel any more.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ipc_bridge::{AccountControl, AccountSnapshot, PairingPrompt};
use minutist_common::{AccountStatus, AppError, AppEvent, AppResult};
use serde::{Deserialize, Serialize};
use settings::SettingsHandle;
use tokio::sync::broadcast;
use tunnel_client::{
    next_interval, AccountDirectoryClient, AccountDirectoryError, DeviceCodeClient, PairingError,
    PollOutcome,
};

/// The file under `{app-data}` holding the device credential. 0600 on Unix.
const CREDENTIAL_FILE: &str = "tunnel_device.json";

/// An optional device label sent with `/pair/start` so the registry row is
/// human-readable. The host name is intentionally NOT used (no machine-
/// identifying detail); a fixed product label keeps it simple for v1.
const PAIRING_LABEL: &str = "Minutist desktop";

/// The persisted device credential (`{app-data}/tunnel_device.json`). The
/// `device_credential` is the long-lived secret presented to the
/// account-service; it is never logged.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredCredential {
    device_credential: String,
    account_id: String,
    device_id: String,
}

impl StoredCredential {
    fn path(app_data_dir: &Path) -> PathBuf {
        app_data_dir.join(CREDENTIAL_FILE)
    }

    /// Read the stored credential, or `None` if absent / unreadable / corrupt
    /// (a corrupt file is treated as signed out — the user can sign in again).
    fn load(app_data_dir: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(Self::path(app_data_dir)).ok()?;
        match serde_json::from_str::<Self>(&raw) {
            Ok(c) if !c.device_credential.is_empty() => Some(c),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!(target: "app-main", "tunnel_device.json is corrupt: {e}; treating as signed out");
                None
            }
        }
    }

    /// Persist the credential at `0600`. Reuses the app's `write_secret_file`
    /// discipline (atomic create with owner-only mode on Unix; the Windows-ACL
    /// gap noted on `mcp_token` applies identically here).
    fn store(&self, app_data_dir: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string(self).map_err(std::io::Error::other)?;
        crate::write_secret_file(&Self::path(app_data_dir), &json)
    }
}

/// This device's account credential, as the B4 account-refresh wiring in
/// `sync.rs` needs it: the `mdc_` bearer for the account-directory client and the
/// account-service `device_id` for the self-endpoint registration. Returns `None`
/// when the device is not signed in (no credential file) — the caller then
/// skips the account-discovery loop entirely (there is no other peer-discovery
/// mechanism on desktop any more).
pub(crate) struct DeviceCredentialParts {
    pub device_credential: String,
    pub device_id: String,
}

/// Load the stored device credential for the account-directory client. The
/// `tunnel_device.json` file is the single source of the `mdc_` credential; this
/// is the one accessor `sync.rs` uses so the credential type stays private here.
pub(crate) fn load_device_credential(app_data_dir: &Path) -> Option<DeviceCredentialParts> {
    StoredCredential::load(app_data_dir).map(|c| DeviceCredentialParts {
        device_credential: c.device_credential,
        device_id: c.device_id,
    })
}

/// The in-progress pairing session between `begin_pairing` and a terminal poll.
struct PairingSession {
    device_code: String,
    interval: std::time::Duration,
}

/// The mutable runtime state, guarded by one mutex (all accesses are brief).
struct Runtime {
    status: AccountStatus,
    /// The in-progress pairing, present only between `begin_pairing` and a
    /// terminal poll.
    pairing: Option<PairingSession>,
    /// The stored credential, loaded at startup and refreshed on a successful
    /// pairing. `None` when signed out.
    credential: Option<StoredCredential>,
}

/// The connected `AccountControl` implementation.
pub struct ConnectedAccount {
    settings: SettingsHandle,
    event_tx: broadcast::Sender<AppEvent>,
    app_data_dir: PathBuf,
    runtime: Arc<Mutex<Runtime>>,
}

impl ConnectedAccount {
    /// Build the control. Loads any stored credential; the initial status is
    /// `SignedIn` if one is present, `SignedOut` otherwise.
    pub fn new(
        settings: SettingsHandle,
        event_tx: broadcast::Sender<AppEvent>,
        app_data_dir: PathBuf,
    ) -> Arc<Self> {
        let credential = StoredCredential::load(&app_data_dir);
        let status = if credential.is_some() {
            AccountStatus::SignedIn
        } else {
            AccountStatus::SignedOut
        };
        Arc::new(Self {
            settings,
            event_tx,
            app_data_dir,
            runtime: Arc::new(Mutex::new(Runtime {
                status,
                pairing: None,
                credential,
            })),
        })
    }

    /// Publish a status transition: record it and emit
    /// `AppEvent::AccountStatusChanged` so the pane reflects it live.
    fn set_status(&self, status: AccountStatus) {
        {
            let mut rt = self.runtime.lock().expect("account runtime poisoned");
            if rt.status == status {
                return;
            }
            rt.status = status;
        }
        let _ = self.event_tx.send(AppEvent::AccountStatusChanged { status });
    }

    /// Build the device-code pairing client for the configured api base URL.
    fn pairing_client(&self) -> Result<DeviceCodeClient, PairingError> {
        let api = self.settings.current().relay_api_url;
        DeviceCodeClient::new(api)
    }
}

#[async_trait]
impl AccountControl for ConnectedAccount {
    async fn begin_pairing(&self) -> AppResult<PairingPrompt> {
        let client = self.pairing_client().map_err(map_pairing_err)?;
        let start = client
            .start(Some(PAIRING_LABEL))
            .await
            .map_err(map_pairing_err)?;

        let prompt = PairingPrompt {
            user_code: start.user_code.clone(),
            verification_uri: start.open_url().to_string(),
            code_required: start.verification_uri_complete.is_none(),
        };
        {
            let mut rt = self.runtime.lock().expect("account runtime poisoned");
            rt.pairing = Some(PairingSession {
                device_code: start.device_code.clone(),
                interval: start.initial_interval(),
            });
        }
        self.set_status(AccountStatus::Pairing);
        tracing::info!(target: "app-main", "account: pairing started; awaiting browser approval");
        Ok(prompt)
    }

    async fn poll_pairing(&self) -> AppResult<AccountStatus> {
        let device_code = {
            let rt = self.runtime.lock().expect("account runtime poisoned");
            match &rt.pairing {
                Some(p) => p.device_code.clone(),
                // No pairing in progress: report the current status (idempotent).
                None => return Ok(rt.status),
            }
        };

        let client = self.pairing_client().map_err(map_pairing_err)?;
        match client.poll_once(&device_code).await {
            Ok(PollOutcome::Pending) => Ok(AccountStatus::Pairing),
            Ok(PollOutcome::SlowDown) => {
                // Track the RFC 8628 §3.5 backed-off interval. NOTE: the UI polls
                // at a fixed floor (PAIRING_POLL_INTERVAL_MS) and does not yet read
                // this value back over IPC, so the bump is not surfaced to the
                // client cadence — acceptable because the floor already spaces
                // polls and the relay enforces its own rate limit. Surfacing the
                // interval to the UI is a deferred refinement (would add a field to
                // the poll response / bindings).
                let mut rt = self.runtime.lock().expect("account runtime poisoned");
                if let Some(p) = rt.pairing.as_mut() {
                    p.interval = next_interval(p.interval);
                }
                Ok(AccountStatus::Pairing)
            }
            Ok(PollOutcome::Authorised(issued)) => {
                // Store the credential securely.
                let credential = StoredCredential {
                    device_credential: issued.device_credential,
                    account_id: issued.account_id,
                    device_id: issued.device_id,
                };
                if let Err(e) = credential.store(&self.app_data_dir) {
                    tracing::error!(target: "app-main", "account: failed to persist the device credential: {e}");
                    return Err(AppError::Io {
                        context: "failed to store the device credential".to_string(),
                    });
                }
                {
                    let mut rt = self.runtime.lock().expect("account runtime poisoned");
                    rt.credential = Some(credential);
                    rt.pairing = None;
                }
                // Signing in turns sync on; persist that.
                let _ = self.settings.update(|s| s.connector_enabled = true).await;
                tracing::info!(target: "app-main", "account: signed in");
                self.set_status(AccountStatus::SignedIn);
                Ok(AccountStatus::SignedIn)
            }
            Err(e) => {
                // Terminal pairing error (expired / declined / malformed). Clear
                // the in-progress session and report SignedOut; a fresh
                // begin_pairing restarts. Transport/status errors also land here;
                // they are surfaced to the user as a failed pairing they can retry.
                tracing::warn!(target: "app-main", "account: pairing ended: {e}");
                self.runtime
                    .lock()
                    .expect("account runtime poisoned")
                    .pairing = None;
                self.set_status(AccountStatus::SignedOut);
                Ok(AccountStatus::SignedOut)
            }
        }
    }

    async fn snapshot(&self) -> AccountSnapshot {
        let rt = self.runtime.lock().expect("account runtime poisoned");
        AccountSnapshot {
            status: rt.status,
            account_id: rt.credential.as_ref().map(|c| c.account_id.clone()),
        }
    }

    async fn delete_account(&self) -> AppResult<()> {
        // The stored credential identifies the account to erase. No credential →
        // already signed out; nothing to do (idempotent).
        let credential = {
            let rt = self.runtime.lock().expect("account runtime poisoned");
            rt.credential.clone()
        };
        let Some(credential) = credential else {
            return Ok(());
        };

        // Ask the account service to erase the account. `DELETE /v1/account`
        // removes the rauthy identity (email + credentials) and cascade-deletes
        // every device row (labels, credential hashes, endpoint addressing,
        // direct addresses). The device is identified by its credential.
        let api = self.settings.current().relay_api_url;
        let client = AccountDirectoryClient::new(api, credential.device_credential.clone())
            .map_err(map_account_err)?;
        match client.delete_account().await {
            // Erased, or the credential no longer resolves (the account is
            // already gone) — either way it is not there. Proceed to clear the
            // local state. Any other error leaves the account intact and the
            // operation retryable, so we surface it WITHOUT clearing local state.
            Ok(()) | Err(AccountDirectoryError::Unauthorised) => {}
            Err(e) => return Err(map_account_err(e)),
        }

        // Server-side erasure confirmed: tear down the local device identity.
        let cred_path = StoredCredential::path(&self.app_data_dir);
        if let Err(e) = std::fs::remove_file(&cred_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    target: "app-main",
                    "delete_account: could not remove {}: {e}",
                    cred_path.display()
                );
            }
        }
        {
            let mut rt = self.runtime.lock().expect("account runtime poisoned");
            rt.credential = None;
            rt.pairing = None;
            rt.status = AccountStatus::SignedOut;
        }
        // Turn sync off — nothing should try to reconnect a deleted account.
        let _ = self.settings.update(|s| s.connector_enabled = false).await;
        let _ = self.event_tx.send(AppEvent::AccountStatusChanged {
            status: AccountStatus::SignedOut,
        });
        tracing::info!(target: "app-main", "account erased; device signed out");
        Ok(())
    }
}

/// Map a `tunnel-client` pairing error to the IPC command error. The mapping is
/// coarse — pairing errors carry no oracle — and never includes the credential.
fn map_pairing_err(e: PairingError) -> AppError {
    match e {
        PairingError::Config => AppError::InvalidInput {
            context: "the relay api URL must be https:// (or a loopback http://)".to_string(),
        },
        PairingError::Transport(_) => AppError::Io {
            context: "could not reach the account service".to_string(),
        },
        PairingError::Status { status } => AppError::Internal {
            context: format!("the account service returned status {status}"),
        },
        PairingError::Decode(_) => AppError::Internal {
            context: "the account service returned an unexpected response".to_string(),
        },
        PairingError::Expired => AppError::InvalidInput {
            context: "the pairing code expired; start pairing again".to_string(),
        },
        PairingError::AccessDenied => AppError::InvalidInput {
            context: "pairing was declined".to_string(),
        },
        PairingError::MalformedAuthorisation => AppError::Internal {
            context: "the account service returned a malformed authorisation".to_string(),
        },
    }
}

/// Map an account-directory error to the IPC command error. Coarse (no oracle)
/// and never includes the credential. `Unauthorised` is normally handled as
/// "already gone" before mapping; it is covered here for completeness.
fn map_account_err(e: AccountDirectoryError) -> AppError {
    match e {
        AccountDirectoryError::Config => AppError::InvalidInput {
            context: "the relay api URL must be https:// (or a loopback http://)".to_string(),
        },
        AccountDirectoryError::Transport(_) => AppError::Io {
            context: "could not reach the account service".to_string(),
        },
        AccountDirectoryError::Unauthorised => AppError::Internal {
            context: "the account service rejected the device credential".to_string(),
        },
        AccountDirectoryError::Status { status } => AppError::Internal {
            context: format!("the account service returned status {status}"),
        },
        AccountDirectoryError::Decode(_) => AppError::Internal {
            context: "the account service returned an unexpected response".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_credential_round_trips_at_0600() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let cred = StoredCredential {
            device_credential: "mdc_dev.secret".to_string(),
            account_id: "acct-1".to_string(),
            device_id: "dev-1".to_string(),
        };
        cred.store(dir.path()).expect("store");
        let loaded = StoredCredential::load(dir.path()).expect("load");
        assert_eq!(loaded.device_credential, "mdc_dev.secret");
        assert_eq!(loaded.account_id, "acct-1");
        assert_eq!(loaded.device_id, "dev-1");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(StoredCredential::path(dir.path()))
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "credential file must be owner-only");
        }
    }

    #[test]
    fn externally_seeded_credential_loads_as_paired() {
        // The autonomous e2e harness (roadmap 2.11, scripts/sync-e2e-harness.py)
        // seeds this file directly from WSL instead of running the device-code
        // flow, so the on-disk JSON shape is a contract with that harness. This
        // pins it: a file carrying only the three documented keys — written
        // without a `store()` round-trip — must load as paired. Renaming a
        // `StoredCredential` field fails this test, forcing the harness to change
        // in lockstep rather than silently seeding an unreadable credential.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let seeded =
            r#"{"device_credential":"mdc_seed.secret","account_id":"acct-e2e","device_id":"desktop-e2e"}"#;
        std::fs::write(StoredCredential::path(dir.path()), seeded).expect("write");
        let loaded = StoredCredential::load(dir.path()).expect("seeded credential must load");
        assert_eq!(loaded.device_credential, "mdc_seed.secret");
        assert_eq!(loaded.account_id, "acct-e2e");
        assert_eq!(loaded.device_id, "desktop-e2e");
    }

    #[test]
    fn missing_credential_is_none() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        assert!(StoredCredential::load(dir.path()).is_none());
    }

    #[test]
    fn corrupt_credential_is_treated_as_unpaired() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(StoredCredential::path(dir.path()), "not json").expect("write");
        assert!(StoredCredential::load(dir.path()).is_none());
    }

    fn map(e: PairingError) -> AppError {
        map_pairing_err(e)
    }

    #[test]
    fn pairing_errors_map_without_leaking() {
        assert!(matches!(
            map(PairingError::Expired),
            AppError::InvalidInput { .. }
        ));
        assert!(matches!(
            map(PairingError::AccessDenied),
            AppError::InvalidInput { .. }
        ));
        assert!(matches!(
            map(PairingError::Status { status: 500 }),
            AppError::Internal { .. }
        ));
    }
}
