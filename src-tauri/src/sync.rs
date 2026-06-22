//! Connected-tier peer-to-peer notes-sync wiring (WS4-B S5).
//!
//! Compiled only in the `connected` build. Implements `ipc_bridge::SyncControl`
//! with the `sync` crate's [`SyncEngine`] (iroh endpoint + the notes-update
//! protocol). `app-main` injects a [`ConnectedSync`] into `IpcState.sync`; the
//! free build never builds this module and uses `ipc_bridge::disabled_sync()`
//! instead. This mirrors the connector tunnel wiring in [`crate::tunnel`].
//!
//! # Lifecycle
//!
//! [`ConnectedSync::new`] returns immediately and spawns engine startup in the
//! background (binding the iroh endpoint is async and `setup` has no entered
//! runtime to block on). Startup runs only when the connector is enabled
//! (`settings.connector_enabled` — sync is part of the same connected tier as
//! the relay tunnel); a disabled connector leaves the engine unbuilt and the
//! status [`SyncStatus::Disabled`]. Startup:
//!
//! 1. loads (or generates) the device [`DeviceIdentity`] under the app-data BASE
//!    (the device key lives at `{app-data}/sync_node_key`, not under
//!    `meetings/`),
//! 2. builds a [`SyncConfig`] over the MEETINGS root (`{app-data}/meetings`, the
//!    same directory the rest of the app uses for per-meeting `{uuid}` folders),
//!    pinning [`sync::SyncConfig::DEFAULT_RELAY_URL`], with the relay auth token
//!    from `MINUTIST_SYNC_TOKEN` / settings when set,
//! 3. calls [`SyncEngine::start`], which binds the endpoint and spawns the
//!    router's inbound accept loop (the responder side of the notes protocol).
//!
//! The status moves `Connecting → Idle` on a successful bind, or `Error` on a
//! failure (also emitted as [`AppEvent::SyncError`]).
//!
//! # Progress events
//!
//! The commands only request actions / read status. Per-meeting transfer
//! progress and completion ride [`AppEvent::SyncProgress`] /
//! [`AppEvent::SyncReady`] / [`AppEvent::SyncError`] on the shared event bus, so
//! the Sync pane reflects a transfer live without polling. [`Self::sync_now`]
//! reconciles a meeting against every registered peer and emits these as it runs.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use ipc_bridge::{IpcError, SyncControl};
use minutist_common::{AppEvent, MeetingId, SyncStatus};
use settings::SettingsHandle;
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use tokio::sync::broadcast;
use tokio::sync::Mutex;

/// The environment variable carrying the self-hosted relay's access token. Takes
/// precedence over the settings value so a deployment / test can inject one
/// without persisting it. The token gates admission to the relay's
/// `AccessControl`; it is never logged.
const RELAY_TOKEN_ENV: &str = "MINUTIST_SYNC_TOKEN";

/// The mutable runtime state, guarded by one async mutex (held only briefly, and
/// across the network awaits in [`SyncEngine::sync_notes`] — the engine's own
/// per-call connection means concurrent `sync_now`s serialise here, acceptable
/// for a manual-trigger surface).
struct Runtime {
    /// The bound engine, present once startup succeeds. `None` while the
    /// connector is disabled or before the background bind completes.
    engine: Option<Arc<SyncEngine>>,
    status: SyncStatus,
}

/// The connected `SyncControl` implementation.
pub struct ConnectedSync {
    event_tx: broadcast::Sender<AppEvent>,
    runtime: Arc<Mutex<Runtime>>,
}

impl ConnectedSync {
    /// Build the control and, when the connector is enabled, spawn engine startup
    /// in the background. Returns immediately; the engine becomes available once
    /// the spawned bind completes (the status moves `Connecting → Idle`).
    ///
    /// `relay_token` is the resolved access token (env-or-settings) the caller
    /// passes in. The two directories are DISTINCT: `app_data_base` holds the
    /// device key (`{app-data}/sync_node_key`); `meetings_dir`
    /// (`{app-data}/meetings`) holds the per-meeting `{uuid}` folders the notes
    /// protocol reads/writes — the same root the rest of the app uses.
    pub fn new(
        settings: SettingsHandle,
        event_tx: broadcast::Sender<AppEvent>,
        app_data_base: PathBuf,
        meetings_dir: PathBuf,
    ) -> Arc<Self> {
        let enabled = settings.current().connector_enabled;
        let initial_status = if enabled {
            SyncStatus::Connecting
        } else {
            SyncStatus::Disabled
        };
        let this = Arc::new(Self {
            event_tx,
            runtime: Arc::new(Mutex::new(Runtime {
                engine: None,
                status: initial_status,
            })),
        });

        if enabled {
            let relay_token = resolve_relay_token(&settings);
            let starter = Arc::clone(&this);
            tauri::async_runtime::spawn(async move {
                starter
                    .start_engine(app_data_base, meetings_dir, relay_token)
                    .await;
            });
        }

        this
    }

    /// Bind the engine in the background, recording the result on the runtime.
    /// A failure sets [`SyncStatus::Error`] and emits [`AppEvent::SyncError`];
    /// it never panics the spawned task.
    ///
    /// The device identity loads from `app_data_base` (the key lives at the
    /// app-data base, never under `meetings/`); the engine config is built over
    /// `meetings_dir` so the notes protocol resolves
    /// `{meetings_dir}/{uuid}/notes.ydoc`.
    async fn start_engine(
        &self,
        app_data_base: PathBuf,
        meetings_dir: PathBuf,
        relay_token: Option<String>,
    ) {
        let identity = match DeviceIdentity::load_or_generate(&app_data_base) {
            Ok(id) => id,
            Err(e) => {
                self.fail(format!("loading the sync device identity: {e}"))
                    .await;
                return;
            }
        };

        let mut config = SyncConfig::new(meetings_dir);
        if let Some(token) = relay_token {
            config = config.with_relay_auth_token(token);
        }

        match SyncEngine::start(config, identity).await {
            Ok(engine) => {
                let mut rt = self.runtime.lock().await;
                rt.engine = Some(Arc::new(engine));
                rt.status = SyncStatus::Idle;
                tracing::info!(target: "app-main", "sync engine started");
            }
            Err(e) => {
                self.fail(format!("starting the sync engine: {e}")).await;
            }
        }
    }

    /// Set the runtime status. Used to mark the engine `Syncing` for the span of
    /// a `sync_now` and restore it to `Idle` afterwards.
    async fn set_status(&self, status: SyncStatus) {
        self.runtime.lock().await.status = status;
    }

    /// Record an error status and emit [`AppEvent::SyncError`].
    async fn fail(&self, message: String) {
        tracing::error!(target: "app-main", "sync: {message}");
        self.runtime.lock().await.status = SyncStatus::Error {
            message: message.clone(),
        };
        let _ = self.event_tx.send(AppEvent::SyncError { context: message });
    }

    /// The bound engine, or an `Unsupported` error when the connector is disabled
    /// and a `Internal` ("still starting") error while the background bind has not
    /// yet completed. Callers use this for every engine-backed operation.
    async fn engine(&self) -> Result<Arc<SyncEngine>, IpcError> {
        let rt = self.runtime.lock().await;
        match (&rt.engine, &rt.status) {
            (Some(engine), _) => Ok(Arc::clone(engine)),
            (None, SyncStatus::Disabled) => Err(IpcError::Unsupported {
                context: "sync is disabled (enable the connector)".to_string(),
            }),
            (None, _) => Err(IpcError::Internal {
                context: "the sync engine is still starting".to_string(),
            }),
        }
    }
}

#[async_trait]
impl SyncControl for ConnectedSync {
    async fn status(&self) -> SyncStatus {
        self.runtime.lock().await.status.clone()
    }

    async fn my_ticket(&self) -> Result<String, IpcError> {
        Ok(self.engine().await?.my_ticket())
    }

    async fn add_peer(&self, ticket: String) -> Result<(), IpcError> {
        let engine = self.engine().await?;
        let peer = engine
            .add_peer_from_ticket(&ticket)
            .map_err(minutist_common::AppError::from)?;
        tracing::info!(target: "app-main", peer = %peer, "sync: registered peer from ticket");
        Ok(())
    }

    async fn sync_now(&self, meeting_id: MeetingId) -> Result<(), IpcError> {
        let engine = self.engine().await?;
        let peers = engine.peer_ids();
        if peers.is_empty() {
            // No paired devices: nothing to reconcile against. Surface a
            // completed-with-nothing-to-do rather than an error — the meeting is
            // already as synced as it can be on a lone device.
            let _ = self
                .event_tx
                .send(AppEvent::SyncReady { meeting_id });
            return Ok(());
        }

        let total = peers.len();
        // Reflect the in-flight transfer in the engine status so a `sync_status`
        // read (e.g. the pane's refresh) observes `Syncing`, alongside the live
        // `SyncProgress` events the UI drives its indicator from. Restored to
        // `Idle` once the loop finishes (a transfer leaves the engine idle, with
        // any failure surfaced via `SyncError`, not a sticky status).
        self.set_status(SyncStatus::Syncing).await;
        let _ = self.event_tx.send(AppEvent::SyncProgress {
            meeting_id,
            label: "Syncing notes…".to_string(),
            fraction: Some(0.0),
        });

        let mut last_err: Option<String> = None;
        for (i, peer) in peers.into_iter().enumerate() {
            match engine.sync_notes(peer, meeting_id).await {
                Ok(()) => {}
                Err(e) => {
                    let message = format!("syncing notes with a peer: {e}");
                    tracing::warn!(target: "app-main", "sync: {message}");
                    last_err = Some(message);
                }
            }
            let _ = self.event_tx.send(AppEvent::SyncProgress {
                meeting_id,
                label: "Syncing notes…".to_string(),
                fraction: Some((i + 1) as f32 / total as f32),
            });
        }

        self.set_status(SyncStatus::Idle).await;
        match last_err {
            None => {
                let _ = self.event_tx.send(AppEvent::SyncReady { meeting_id });
                Ok(())
            }
            Some(message) => {
                // At least one peer failed. Emit the failure on the bus (the UI
                // surfaces it) and return it so the command caller sees it too;
                // peers that succeeded already converged.
                let _ = self.event_tx.send(AppEvent::SyncError {
                    context: message.clone(),
                });
                Err(IpcError::Internal { context: message })
            }
        }
    }
}

/// Resolve the relay access token: `MINUTIST_SYNC_TOKEN` if set and non-empty,
/// otherwise none. (The token is not yet a persisted setting; the env var is the
/// injection point for deployments/tests. Settings precedence lands with the
/// account-service token issuance.)
fn resolve_relay_token(_settings: &SettingsHandle) -> Option<String> {
    match std::env::var(RELAY_TOKEN_ENV) {
        Ok(token) if !token.is_empty() => Some(token),
        _ => None,
    }
}
