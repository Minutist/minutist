//! Connected-tier relay tunnel wiring (WS4-A S5b).
//!
//! Compiled only in the `connected` build. Implements `ipc_bridge::TunnelControl`
//! with the `tunnel-client` pairing + reconnect + lifecycle types, and owns the
//! secure storage of the issued device credential. `app-main` injects a
//! [`ConnectedTunnel`] into `IpcState.tunnel`; the free build never builds this
//! module and uses `ipc_bridge::disabled_tunnel()` instead.
//!
//! # The flow
//!
//! 1. `tunnel_begin_pairing` → [`ConnectedTunnel::begin_pairing`]: POST
//!    `/pair/start`, stash the `device_code` + interval, set status `Pairing`,
//!    return the `user_code` + verification URL for the UI to show + open.
//! 2. `tunnel_poll_pairing` → [`ConnectedTunnel::poll_pairing`]: POST
//!    `/pair/poll`. While pending, stays `Pairing` (bumping the interval on
//!    `slow_down`); on `authorised` it stores the credential (0600) and starts
//!    the tunnel lifecycle, moving to `Connecting`/`Online`; on a declined /
//!    expired pairing it returns `Disconnected`.
//! 3. The reconnect loop reports `ConnectionState` transitions, mapped onto
//!    `TunnelStatus` and emitted as `AppEvent::TunnelStatusChanged` so the pane
//!    reflects `Connecting → Online` live. A revoked credential moves to
//!    `NeedsRepair` and the loop stops (no hot-loop).
//!
//! # Credential storage
//!
//! The issued credential + account/device ids are stored at
//! `{app-data}/tunnel_device.json` with owner-only `0600` on Unix, the same
//! discipline as the `mcp_token` file (and the same Windows-ACL gap — see
//! `crate::write_secret_file`). The credential is the long-lived device identity;
//! it is never logged and never crosses to the webview.
//!
//! # The loopback target
//!
//! The tunnel replays relayed MCP requests against the app's own loopback
//! `mcp-server`, applying the app's internal bearer. That bearer + URL come from
//! `IpcState.mcp_info` (the `McpServerInfo` the MCP server fills on bind); the
//! relay's request never carries the user bearer — the app applies its own
//! internal bearer when it replays against loopback (D5).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ipc_bridge::{McpServerInfo, PairingPrompt, TunnelControl, TunnelSnapshot};
use minutist_common::{AppEvent, TunnelStatus};
use serde::{Deserialize, Serialize};
use settings::SettingsHandle;
use tokio::sync::broadcast;
use tunnel_client::{
    next_interval, ConnectionState, DeviceCodeClient, InternalBearer, LoopbackTarget, PairingError,
    PollOutcome, TunnelConfig, TunnelHandle,
};

/// The file under `{app-data}` holding the device credential. 0600 on Unix.
const CREDENTIAL_FILE: &str = "tunnel_device.json";

/// An optional device label sent with `/pair/start` so the registry row is
/// human-readable. The host name is intentionally NOT used (no machine-
/// identifying detail); a fixed product label keeps it simple for v1.
const PAIRING_LABEL: &str = "Minutist desktop";

/// The persisted device credential (`{app-data}/tunnel_device.json`). The
/// `device_credential` is the long-lived secret presented as the tunnel
/// `Hello.device_credential`; it is never logged.
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
    /// (a corrupt file is treated as unpaired — the user can re-pair).
    fn load(app_data_dir: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(Self::path(app_data_dir)).ok()?;
        match serde_json::from_str::<Self>(&raw) {
            Ok(c) if !c.device_credential.is_empty() => Some(c),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!(target: "app-main", "tunnel_device.json is corrupt: {e}; treating as unpaired");
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

/// The in-progress pairing session between `begin_pairing` and a terminal poll.
struct PairingSession {
    device_code: String,
    interval: std::time::Duration,
}

/// The mutable runtime state, guarded by one mutex (all accesses are brief).
struct Runtime {
    status: TunnelStatus,
    /// The in-progress pairing, present only between `begin_pairing` and a
    /// terminal poll.
    pairing: Option<PairingSession>,
    /// The running tunnel lifecycle, present only while the tunnel is up.
    handle: Option<TunnelHandle>,
    /// The stored credential, loaded at startup and refreshed on a successful
    /// pairing. `None` when unpaired.
    credential: Option<StoredCredential>,
}

/// The connected `TunnelControl` implementation.
pub struct ConnectedTunnel {
    settings: SettingsHandle,
    event_tx: broadcast::Sender<AppEvent>,
    app_data_dir: PathBuf,
    /// The loopback MCP endpoint (URL + internal bearer), filled by `app-main`
    /// once the MCP server binds. The tunnel needs it to replay requests.
    mcp_info: Arc<Mutex<Option<McpServerInfo>>>,
    runtime: Arc<Mutex<Runtime>>,
}

impl ConnectedTunnel {
    /// Build the control. Loads any stored credential and, if the connector is
    /// enabled AND a credential is present, starts the tunnel at construction so
    /// a paired+enabled device reconnects on launch.
    pub fn new(
        settings: SettingsHandle,
        event_tx: broadcast::Sender<AppEvent>,
        app_data_dir: PathBuf,
        mcp_info: Arc<Mutex<Option<McpServerInfo>>>,
    ) -> Arc<Self> {
        let credential = StoredCredential::load(&app_data_dir);
        let this = Arc::new(Self {
            settings,
            event_tx,
            app_data_dir,
            mcp_info,
            runtime: Arc::new(Mutex::new(Runtime {
                status: TunnelStatus::Disconnected,
                pairing: None,
                handle: None,
                credential,
            })),
        });

        let enabled = this.settings.current().connector_enabled;
        let has_cred = this
            .runtime
            .lock()
            .expect("tunnel runtime poisoned")
            .credential
            .is_some();
        if enabled && has_cred {
            this.start_tunnel();
        }
        this
    }

    /// Start the tunnel if the connector is enabled, a credential is stored, and
    /// it is not already running. Called by `app-main` once the loopback MCP
    /// server has bound (the tunnel needs that loopback target to replay
    /// against), so a paired+enabled device that launched before the MCP server
    /// was listening connects as soon as it is.
    pub fn retry_start_if_enabled(&self) {
        let (enabled, has_cred, running) = {
            let rt = self.runtime.lock().expect("tunnel runtime poisoned");
            (
                self.settings.current().connector_enabled,
                rt.credential.is_some(),
                rt.handle.is_some(),
            )
        };
        if enabled && has_cred && !running {
            self.start_tunnel();
        }
    }

    /// Publish a status transition: record it and emit
    /// `AppEvent::TunnelStatusChanged` so the pane reflects it live.
    fn set_status(&self, status: TunnelStatus) {
        {
            let mut rt = self.runtime.lock().expect("tunnel runtime poisoned");
            if rt.status == status {
                return;
            }
            rt.status = status;
        }
        let _ = self.event_tx.send(AppEvent::TunnelStatusChanged { status });
    }

    /// Build the device-code pairing client for the configured api base URL.
    fn pairing_client(&self) -> Result<DeviceCodeClient, PairingError> {
        let api = self.settings.current().relay_api_url;
        DeviceCodeClient::new(api)
    }

    /// Build the `TunnelConfig` from the stored credential + the loopback target.
    /// Returns `None` if there is no credential or the MCP server is not yet
    /// listening (no loopback target to replay against).
    fn build_config(&self) -> Option<TunnelConfig> {
        let credential = {
            let rt = self.runtime.lock().expect("tunnel runtime poisoned");
            rt.credential.clone()?
        };
        let info = self.mcp_info.lock().expect("mcp_info poisoned").clone()?;
        // Strip the `/mcp` path to the origin; the relay frame carries the path.
        let base_url = origin_of(&info.url);
        let relay_url = self.settings.current().relay_url;
        Some(TunnelConfig {
            relay_url,
            device_credential: credential.device_credential,
            account_id: credential.account_id,
            loopback: LoopbackTarget::new(base_url, InternalBearer::new(info.token)),
        })
    }

    /// Start (or restart) the tunnel lifecycle. Stops any existing handle first.
    /// Does nothing (logs) if no config can be built yet — e.g. the MCP server
    /// has not bound; the caller re-attempts once it does.
    fn start_tunnel(&self) {
        let Some(config) = self.build_config() else {
            tracing::info!(
                target: "app-main",
                "tunnel: cannot start yet (no credential or loopback target); staying disconnected"
            );
            self.set_status(TunnelStatus::Disconnected);
            return;
        };

        // Tear down any prior handle so we never run two loops.
        self.stop_tunnel_blocking();

        self.set_status(TunnelStatus::Connecting);

        // The state callback runs on the reconnect task; it needs an owned,
        // Send handle back to `self` to emit events. Clone the cheap shared
        // pieces rather than the whole struct.
        let event_tx = self.event_tx.clone();
        let runtime = Arc::clone(&self.runtime);
        let on_state = move |state: ConnectionState| {
            let status = match state {
                ConnectionState::Connecting => TunnelStatus::Connecting,
                ConnectionState::Online => TunnelStatus::Online,
                // Reconnecting maps to Connecting in the UI: the connector is
                // simply not online yet; the app works locally regardless.
                ConnectionState::Reconnecting => TunnelStatus::Connecting,
            };
            let changed = {
                let mut rt = runtime.lock().expect("tunnel runtime poisoned");
                if rt.status == status {
                    false
                } else {
                    rt.status = status;
                    true
                }
            };
            if changed {
                let _ = event_tx.send(AppEvent::TunnelStatusChanged { status });
            }
        };

        let handle = TunnelHandle::start(config, on_state);
        self.runtime.lock().expect("tunnel runtime poisoned").handle = Some(handle);

        // Watch for the loop self-exiting on a revoked credential (NeedsRepair)
        // so the status reflects it even though no further state callback fires.
        self.spawn_exit_watcher();
    }

    /// Spawn a task that awaits the lifecycle's exit (if the loop stops on its
    /// own — a revoked credential). On `NeedsRepair` it sets the status so the UI
    /// prompts a re-pair. On `Cancelled` (normal stop) it does nothing.
    fn spawn_exit_watcher(&self) {
        // We cannot move the JoinHandle out without taking the handle (which the
        // lifecycle owns), so the exit is observed via TunnelHandle::is_finished
        // polling is avoided: instead, the reconnect loop's NeedsRepair path
        // would have left the handle finished. We surface NeedsRepair by polling
        // is_finished on a short cadence only while a handle is present and the
        // status is not already terminal. Bounded, cheap, and stops as soon as a
        // terminal status is set or the handle is taken on stop/restart.
        let runtime = Arc::clone(&self.runtime);
        let event_tx = self.event_tx.clone();
        tauri::async_runtime::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let finished = {
                    let rt = runtime.lock().expect("tunnel runtime poisoned");
                    match &rt.handle {
                        Some(h) => h.is_finished(),
                        None => return, // handle taken (stop/restart) — stop watching
                    }
                };
                if finished {
                    // The loop exited on its own. The only self-exit that is not
                    // a cancel is NeedsRepair (a revoked credential). Mark it and
                    // stop watching.
                    let changed = {
                        let mut rt = runtime.lock().expect("tunnel runtime poisoned");
                        // Drop the finished handle so a subsequent start is clean.
                        rt.handle = None;
                        if rt.status == TunnelStatus::NeedsRepair {
                            false
                        } else {
                            rt.status = TunnelStatus::NeedsRepair;
                            true
                        }
                    };
                    if changed {
                        tracing::warn!(
                            target: "app-main",
                            "tunnel: lifecycle exited on its own (credential revoked); needs re-pair"
                        );
                        let _ = event_tx.send(AppEvent::TunnelStatusChanged {
                            status: TunnelStatus::NeedsRepair,
                        });
                    }
                    return;
                }
            }
        });
    }

    /// Stop the tunnel lifecycle if running, awaiting clean teardown. Synchronous
    /// wrapper used where we are already off the async worker is not available
    /// here, so this takes the handle and spawns the await — the stop signal is
    /// raised immediately and the JoinSet teardown completes in the background.
    /// Callers that need to await teardown use [`stop_tunnel`](Self::stop_tunnel).
    fn stop_tunnel_blocking(&self) {
        let handle = self
            .runtime
            .lock()
            .expect("tunnel runtime poisoned")
            .handle
            .take();
        if let Some(handle) = handle {
            tauri::async_runtime::spawn(async move {
                let _ = handle.stop().await;
            });
        }
    }

    /// Stop the tunnel lifecycle and await its teardown.
    async fn stop_tunnel(&self) {
        let handle = self
            .runtime
            .lock()
            .expect("tunnel runtime poisoned")
            .handle
            .take();
        if let Some(handle) = handle {
            let exit = handle.stop().await;
            tracing::info!(target: "app-main", ?exit, "tunnel: lifecycle stopped");
        }
    }
}

#[async_trait]
impl TunnelControl for ConnectedTunnel {
    async fn begin_pairing(&self) -> Result<PairingPrompt, ipc_bridge::IpcError> {
        let client = self.pairing_client().map_err(map_pairing_err)?;
        let start = client
            .start(Some(PAIRING_LABEL))
            .await
            .map_err(map_pairing_err)?;

        let prompt = PairingPrompt {
            user_code: start.user_code.clone(),
            verification_uri: start.open_url().to_string(),
        };
        {
            let mut rt = self.runtime.lock().expect("tunnel runtime poisoned");
            rt.pairing = Some(PairingSession {
                device_code: start.device_code.clone(),
                interval: start.initial_interval(),
            });
        }
        self.set_status(TunnelStatus::Pairing);
        tracing::info!(target: "app-main", "tunnel: pairing started; awaiting browser approval");
        Ok(prompt)
    }

    async fn poll_pairing(&self) -> Result<TunnelStatus, ipc_bridge::IpcError> {
        let device_code = {
            let rt = self.runtime.lock().expect("tunnel runtime poisoned");
            match &rt.pairing {
                Some(p) => p.device_code.clone(),
                // No pairing in progress: report the current status (idempotent).
                None => return Ok(rt.status),
            }
        };

        let client = self.pairing_client().map_err(map_pairing_err)?;
        match client.poll_once(&device_code).await {
            Ok(PollOutcome::Pending) => Ok(TunnelStatus::Pairing),
            Ok(PollOutcome::SlowDown) => {
                // Bump the stored interval so the UI's next poll backs off.
                let mut rt = self.runtime.lock().expect("tunnel runtime poisoned");
                if let Some(p) = rt.pairing.as_mut() {
                    p.interval = next_interval(p.interval);
                }
                Ok(TunnelStatus::Pairing)
            }
            Ok(PollOutcome::Authorised(issued)) => {
                // Store the credential securely, then start the tunnel.
                let credential = StoredCredential {
                    device_credential: issued.device_credential,
                    account_id: issued.account_id,
                    device_id: issued.device_id,
                };
                if let Err(e) = credential.store(&self.app_data_dir) {
                    tracing::error!(target: "app-main", "tunnel: failed to persist the device credential: {e}");
                    return Err(ipc_bridge::IpcError::Io {
                        context: "failed to store the device credential".to_string(),
                    });
                }
                {
                    let mut rt = self.runtime.lock().expect("tunnel runtime poisoned");
                    rt.credential = Some(credential);
                    rt.pairing = None;
                }
                // Pairing implies the connector should be on; persist that.
                let _ = self.settings.update(|s| s.connector_enabled = true).await;
                tracing::info!(target: "app-main", "tunnel: paired; starting the tunnel");
                self.start_tunnel();
                Ok(self.runtime.lock().expect("tunnel runtime poisoned").status)
            }
            Err(e) => {
                // Terminal pairing error (expired / declined / malformed). Clear
                // the in-progress session and report Disconnected; a fresh
                // begin_pairing restarts. Transport/status errors also land here;
                // they are surfaced to the user as a failed pairing they can retry.
                tracing::warn!(target: "app-main", "tunnel: pairing ended: {e}");
                self.runtime
                    .lock()
                    .expect("tunnel runtime poisoned")
                    .pairing = None;
                self.set_status(TunnelStatus::Disconnected);
                Ok(TunnelStatus::Disconnected)
            }
        }
    }

    async fn set_enabled(&self, enabled: bool) -> Result<TunnelSnapshot, ipc_bridge::IpcError> {
        // Persist the choice.
        let _ = self
            .settings
            .update(|s| s.connector_enabled = enabled)
            .await;

        if enabled {
            let has_cred = self
                .runtime
                .lock()
                .expect("tunnel runtime poisoned")
                .credential
                .is_some();
            if has_cred {
                self.start_tunnel();
            } else {
                // Enabled but unpaired: nothing to dial yet. The pane prompts
                // the user to pair.
                self.set_status(TunnelStatus::Disconnected);
            }
        } else {
            self.stop_tunnel().await;
            self.set_status(TunnelStatus::Disconnected);
        }

        Ok(self.snapshot().await)
    }

    async fn snapshot(&self) -> TunnelSnapshot {
        let rt = self.runtime.lock().expect("tunnel runtime poisoned");
        TunnelSnapshot {
            enabled: self.settings.current().connector_enabled,
            status: rt.status,
            account_id: rt.credential.as_ref().map(|c| c.account_id.clone()),
        }
    }
}

/// Strip a full endpoint URL (`http://127.0.0.1:8765/mcp`) to its origin
/// (`http://127.0.0.1:8765`): scheme + authority, dropping the path. The relay
/// frame carries the path, so the loopback target is the origin only.
fn origin_of(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let authority = rest.split('/').next().unwrap_or(rest);
            format!("{scheme}://{authority}")
        }
        None => url.to_string(),
    }
}

/// Map a `tunnel-client` pairing error to the IPC command error. The mapping is
/// coarse — pairing errors carry no oracle — and never includes the credential.
fn map_pairing_err(e: PairingError) -> ipc_bridge::IpcError {
    match e {
        PairingError::Config => ipc_bridge::IpcError::InvalidInput {
            context: "the relay api URL must be https:// (or a loopback http://)".to_string(),
        },
        PairingError::Transport(_) => ipc_bridge::IpcError::Io {
            context: "could not reach the account service".to_string(),
        },
        PairingError::Status { status } => ipc_bridge::IpcError::Internal {
            context: format!("the account service returned status {status}"),
        },
        PairingError::Decode(_) => ipc_bridge::IpcError::Internal {
            context: "the account service returned an unexpected response".to_string(),
        },
        PairingError::Expired => ipc_bridge::IpcError::InvalidInput {
            context: "the pairing code expired; start pairing again".to_string(),
        },
        PairingError::AccessDenied => ipc_bridge::IpcError::InvalidInput {
            context: "pairing was declined".to_string(),
        },
        PairingError::MalformedAuthorisation => ipc_bridge::IpcError::Internal {
            context: "the account service returned a malformed authorisation".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_strips_path() {
        assert_eq!(
            origin_of("http://127.0.0.1:8765/mcp"),
            "http://127.0.0.1:8765"
        );
        assert_eq!(origin_of("https://host/mcp/extra"), "https://host");
        assert_eq!(origin_of("http://127.0.0.1:8765"), "http://127.0.0.1:8765");
        assert_eq!(origin_of("garbage"), "garbage");
    }

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

    fn map(e: PairingError) -> ipc_bridge::IpcError {
        map_pairing_err(e)
    }

    #[test]
    fn pairing_errors_map_without_leaking() {
        assert!(matches!(
            map(PairingError::Expired),
            ipc_bridge::IpcError::InvalidInput { .. }
        ));
        assert!(matches!(
            map(PairingError::AccessDenied),
            ipc_bridge::IpcError::InvalidInput { .. }
        ));
        assert!(matches!(
            map(PairingError::Status { status: 500 }),
            ipc_bridge::IpcError::Internal { .. }
        ));
    }
}
