//! Connected-tier peer-to-peer notes-sync wiring (WS4-B S5).
//!
//! Compiled only in the `connected` build. Implements `ipc_bridge::SyncControl`
//! with the `sync` crate's [`SyncEngine`] (iroh endpoint + the notes-update
//! protocol). `app-main` injects a [`ConnectedSync`] into `IpcState.sync`; the
//! free build never builds this module and uses `ipc_bridge::disabled_sync()`
//! instead. This mirrors the account wiring in [`crate::account`].
//!
//! # Lifecycle
//!
//! [`ConnectedSync::new`] returns immediately and spawns engine startup in the
//! background (binding the iroh endpoint is async and `setup` has no entered
//! runtime to block on) when `settings.connector_enabled` is already set at
//! construction time (a device that was signed in on a previous run). A
//! signed-out launch leaves the engine unbuilt and the status
//! [`SyncStatus::Disabled`] until [`SyncControl::set_enabled`] is called (F5) —
//! from [`crate::account::ConnectedAccount::poll_pairing`] on a successful
//! sign-in, or from `delete_account` to turn it back off. Both paths converge
//! on the same idempotent [`ConnectedSync::request_start`]: a start is
//! requested at most once (guarded by an atomic flag), so construction-time
//! auto-start and a later sign-in can never double-spawn the engine.
//! Startup:
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
//! reconciles a meeting against every registered peer — both its notes (the CRDT
//! update exchange) and its media (`audio.opus` + assets, as content-addressed
//! blobs) — and emits these as it runs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use async_trait::async_trait;
use election::{Capability, ElectionConfig, ElectionDriver};
use ipc_bridge::{ChatHandles, SyncControl};
use minutist_common::{AppError, AppEvent, AppResult, HostRef, MeetingId, SyncStatus};
use settings::SettingsHandle;
use sync::{
    run_account_refresh_loop_v2, AccountEndpoint, AccountEndpointSource, BackoffPolicy, ContentKey,
    DeviceIdentity, RefreshSink, SyncConfig, SyncEngine, SyncEngineRefreshSink,
};
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// The environment variable carrying the self-hosted relay's access token. Takes
/// precedence over the settings value so a deployment / test can inject one
/// without persisting it. The token gates admission to the relay's
/// `AccessControl`; it is never logged.
const RELAY_TOKEN_ENV: &str = "MINUTIST_SYNC_TOKEN";

/// How often the account-refresh loop (B4) re-fetches the account's device
/// directory to discover newly-registered peers.
const ACCOUNT_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

/// The mutable runtime state, guarded by one async mutex held only briefly. A
/// `sync_now` clones the engine `Arc` out under the lock and runs the per-peer
/// notes + media reconciliation without holding it, re-locking only to flip the
/// `Syncing`/`Idle` status; the engine's own per-call connection bounds
/// concurrent transfers.
struct Runtime {
    /// The bound engine, present once startup succeeds. `None` while the
    /// connector is disabled or before the background bind completes.
    engine: Option<Arc<SyncEngine>>,
    status: SyncStatus,
    /// The cancellation token for the current bind's account-refresh loop v2,
    /// which `select!`s on `cancelled()`. It is the single shared token that
    /// closes issue 0029 item 1's shared-cancel on the desktop, and being
    /// latching, a cancel fired mid-loop-body (between `select!` iterations)
    /// is not missed.
    ///
    /// The `replace()` below `cancel()`s a prior token before adopting the new
    /// one, but that path is NOT reachable today: `start_requested` is a one-shot
    /// latch, so the engine binds exactly once and `replace()` always sees `None`
    /// — `prev.cancel()` never fires on a live loop. `cancel()` is the correct
    /// primitive for WHEN the deferred `set_enabled(false)` re-bind/teardown path
    /// lands (which also needs to AWAIT both tasks' exit — see that follow-up);
    /// today only shutdown (dropping the tasks) stops the loops.
    stop: Option<CancellationToken>,
}

/// The collaborators the producer-gate election loop needs, passed in from
/// `app-main` (which links `sync` + `orchestrator` + `ipc-bridge` + `election`
/// together). Held so [`ConnectedSync::start_engine`] can spawn the loop once the
/// engine binds. `None` disables the gate (the connected E2E tests build the
/// sync control without it).
pub struct ElectionDeps {
    /// The bundle the [`DesktopElectionDriver`] drives `reprocess` through AND
    /// (F4-summary) the post-reprocess summarise pass through — the same
    /// [`ChatHandles`] `app-main` builds for the attachment-conversion VLM,
    /// carrying the orchestrator, the meeting index, the meetings dir, the event
    /// bus, settings, and the shared held-summariser cell.
    pub chat_handles: ChatHandles,
}

/// The connected `SyncControl` implementation.
pub struct ConnectedSync {
    event_tx: broadcast::Sender<AppEvent>,
    runtime: Arc<Mutex<Runtime>>,
    /// When set (production), the engine-bind path spawns the producer-gate election
    /// loop with a [`DesktopElectionDriver`] over these + the bound engine.
    election: Option<ElectionDeps>,
    settings: SettingsHandle,
    app_data_base: PathBuf,
    meetings_dir: PathBuf,
    /// The shared meeting index — handed to the lifecycle subscriber
    /// (`ipc_bridge::spawn_lifecycle_subscriber`) so a synced deletion state
    /// keeps `index.db`'s `deleted_at` mirror in step, exactly like the rest
    /// of the app's writers.
    index: Arc<ipc_bridge::MeetingIndex>,
    /// Guards [`Self::request_start`] so the engine is started at most once:
    /// construction with the connector already enabled, and a later runtime
    /// [`SyncControl::set_enabled`] (F5), both funnel through it — whichever
    /// wins the atomic swap spawns the bind; the other (or a racing duplicate
    /// call) is a no-op.
    start_requested: AtomicBool,
    /// Set once, immediately after construction, so [`Self::request_start`] (an
    /// `&self` method — trait methods never see the owning `Arc`) can obtain an
    /// owned `Arc<Self>` to move into the spawned bind task.
    self_ref: OnceLock<Weak<ConnectedSync>>,
}

impl ConnectedSync {
    /// Build the control and, when the connector is enabled, request engine
    /// startup in the background. Returns immediately; the engine becomes
    /// available once the spawned bind completes (the status moves
    /// `Connecting → Idle`). A connector disabled at construction leaves the
    /// engine unstarted until [`SyncControl::set_enabled`] requests it later
    /// (F5).
    ///
    /// The two directories are DISTINCT: `app_data_base` holds the device key
    /// (`{app-data}/sync_node_key`); `meetings_dir` (`{app-data}/meetings`)
    /// holds the per-meeting `{uuid}` folders the notes protocol reads/writes —
    /// the same root the rest of the app uses. Both are retained on `self` (not
    /// just threaded through this call) so a later runtime enable can start the
    /// engine without the caller re-supplying them.
    pub fn new(
        settings: SettingsHandle,
        event_tx: broadcast::Sender<AppEvent>,
        app_data_base: PathBuf,
        meetings_dir: PathBuf,
        election: Option<ElectionDeps>,
        index: Arc<ipc_bridge::MeetingIndex>,
    ) -> Arc<Self> {
        // `connector_enabled` is the sole gate for auto-starting, but it can drift
        // from the real signed-in state: e.g. `delete_account`'s best-effort
        // credential-file removal can fail (the account is erased server-side but
        // a leftover `tunnel_device.json` survives), leaving a future launch
        // re-derive `AccountStatus::SignedIn` (`ConnectedAccount::new` checks the
        // file, not this setting) while `connector_enabled` stays the `false`
        // `delete_account` persisted — the UI would then show "Signed in" with no
        // way back to an active connector, since nothing but a fresh sign-in ever
        // flips this flag. Treat a present credential as an independent,
        // authoritative "should be enabled" signal, and self-heal the persisted
        // setting to match rather than leaving it silently wrong forever.
        let signed_in = crate::account::load_device_credential(&app_data_base).is_some();
        let persisted_enabled = settings.current().connector_enabled;
        let enabled = persisted_enabled || signed_in;
        let this = Arc::new(Self {
            event_tx,
            runtime: Arc::new(Mutex::new(Runtime {
                engine: None,
                status: SyncStatus::Disabled,
                stop: None,
            })),
            election,
            settings,
            app_data_base,
            meetings_dir,
            index,
            start_requested: AtomicBool::new(false),
            self_ref: OnceLock::new(),
        });
        let _ = this.self_ref.set(Arc::downgrade(&this));

        if enabled {
            let starter = Arc::clone(&this);
            tauri::async_runtime::spawn(async move {
                starter.request_start().await;
            });
        }
        if signed_in && !persisted_enabled {
            let healer = Arc::clone(&this);
            tauri::async_runtime::spawn(async move {
                let _ = healer.settings.update(|s| s.connector_enabled = true).await;
            });
        }

        this
    }

    /// Request the engine start, at most once (F5). The first caller — either
    /// construction with the connector already enabled, or a later
    /// [`SyncControl::set_enabled(true)`](SyncControl::set_enabled) — wins the
    /// atomic flag, marks the status [`SyncStatus::Connecting`] synchronously
    /// (so an immediate `status()` read reflects the request rather than a stale
    /// `Disabled`), and spawns [`Self::start_engine`] in the background. A
    /// second call — the engine already running or its bind already in flight —
    /// is a no-op, so re-enabling an already-started connector can never
    /// double-spawn the engine or its election loop.
    async fn request_start(&self) {
        if self.start_requested.swap(true, Ordering::SeqCst) {
            return;
        }
        {
            let mut rt = self.runtime.lock().await;
            rt.status = SyncStatus::Connecting;
        }
        let Some(this) = self.self_ref.get().and_then(Weak::upgrade) else {
            // Unreachable in practice (`self_ref` is set immediately after the
            // `Arc` is constructed, before any caller can reach `request_start`),
            // but never panic on it — leave the status Connecting; a stuck
            // engine is visible and recoverable (e.g. an app restart), never a
            // crash.
            tracing::error!(target: "app-main", "sync: request_start could not upgrade self_ref");
            return;
        };
        let app_data_base = self.app_data_base.clone();
        let relay_token = resolve_relay_token(&app_data_base, &self.settings);
        let meetings_dir = self.meetings_dir.clone();
        tauri::async_runtime::spawn(async move {
            this.start_engine(app_data_base, meetings_dir, relay_token)
                .await;
        });
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

        // The account content key every enrolled device holds; each frame on the
        // sync ALPN is sealed under it. Minted on first run, which assumes this
        // device is the first on the account: if it is not, the user-confirmed
        // enrolment exchange replaces it (see `sync::content_key`).
        let content_key = match ContentKey::load_or_mint(&app_data_base) {
            Ok(key) => key,
            Err(e) => {
                self.fail(format!("loading the sync content key: {e}"))
                    .await;
                return;
            }
        };

        // Clone the meetings root for the lifecycle subscriber + the election loop
        // before the config consumes it.
        let subscriber_meetings_dir = meetings_dir.clone();
        let election_meetings_dir = meetings_dir.clone();
        let mut config = SyncConfig::new(meetings_dir);
        if let Some(token) = relay_token {
            config = config.with_relay_auth_token(token);
        }
        // Failed-dial backoff policy (2.7). The desktop consumer owns these
        // values; they match the crate placeholder for now — initial estimates,
        // to tune once we have field data on real dial-failure patterns: 3
        // consecutive failed dials before a peer is suppressed, then exponential
        // backoff from 30 s doubling up to a 1 h cap.
        config = config.with_backoff_policy(BackoffPolicy {
            max_fails: 3,
            base: std::time::Duration::from_secs(30),
            cap: std::time::Duration::from_secs(3600),
        });

        match SyncEngine::start(config, identity, content_key).await {
            Ok(engine) => {
                let engine = Arc::new(engine);

                // One cancellation token shared by this bind's background loops
                // (the account-refresh loop v2, when account-paired). Stored in
                // `Runtime` below; a prior bind's token is `cancel()`-led before we
                // replace it. The single shared token that closes issue 0029 item 1
                // on the desktop; latching, so a mid-loop cancel is never missed.
                let stop = CancellationToken::new();

                // Account-mediated peer discovery (B4) is the desktop's ONLY
                // peer-discovery mechanism now: when the device is signed in,
                // publish this device's endpoint to the account directory and add
                // every other account device's endpoint as a peer, refreshing on
                // an interval. (Manual ticket pairing — the peers file, `my_ticket`
                // — existed only as an interim mechanism before this shipped, and
                // has been removed; `sync::SyncEngine::my_ticket`/
                // `add_peer_from_ticket` still exist for the phone client, which
                // has no account-mediated path of its own yet.)
                if let Some(cred) = crate::account::load_device_credential(&app_data_base) {
                    let settings = self.settings.current();
                    match tunnel_client::AccountDirectoryClient::new(
                        settings.relay_api_url,
                        cred.device_credential,
                    ) {
                        Ok(client) => {
                            let source: Arc<dyn AccountEndpointSource> =
                                Arc::new(account_directory::AccountDirectorySource::new(client));
                            let self_endpoint = AccountEndpoint {
                                device_id: cred.device_id,
                                endpoint_id: engine.endpoint_id().to_string(),
                                // The engine homes on SyncConfig::DEFAULT_RELAY_URL —
                                // the sync engine's own iroh relay, unrelated to
                                // settings.relay_api_url (the account-service base).
                                relay_url: SyncConfig::DEFAULT_RELAY_URL.to_string(),
                                // Filtered direct addrs (0049) so a same-tailnet/
                                // LAN peer dials this device directly, no relay/DNS.
                                direct_addrs: engine.publishable_direct_addrs(),
                            };
                            // The engine-backed sink: upsert/remove/is_suppressed/
                            // account_peer_ids delegate to the engine, and
                            // on_new_peer first-contact-dials a genuinely-new peer.
                            let sink: Arc<dyn RefreshSink> =
                                Arc::new(SyncEngineRefreshSink::new(Arc::clone(&engine)));
                            let stop = stop.clone();
                            tauri::async_runtime::spawn(run_account_refresh_loop_v2(
                                source,
                                self_endpoint,
                                ACCOUNT_REFRESH_INTERVAL,
                                stop,
                                sink,
                            ));
                            tracing::info!(target: "app-main", "sync: account-refresh loop started");
                        }
                        Err(e) => tracing::warn!(
                            target: "app-main",
                            error = %e,
                            "sync: account-directory client not built; skipping account discovery"
                        ),
                    }
                }

                // Persist the processing-lifecycle states the engine surfaces from
                // discovery. The subscriber loop lives in `ipc-bridge` (which owns
                // the `persistence` edge); here we only hand it the engine's
                // receiver, so this module keeps no direct `persistence` use.
                ipc_bridge::spawn_lifecycle_subscriber(
                    engine.subscribe_lifecycle_events(),
                    subscriber_meetings_dir,
                    Arc::clone(&self.index),
                    self.event_tx.clone(),
                );

                // Producer-gate election loop (S4): claim + process the
                // `PendingProcessing` meetings a capture device offered, then push
                // the derived artifacts back. Spawned only when the deps were
                // injected (production; the connected E2E tests pass `None`). It
                // self-gates on GPU capability — a CPU-only host parks and never
                // claims — so spawning it unconditionally here is safe.
                if let Some(deps) = &self.election {
                    let driver: Arc<dyn ElectionDriver> = Arc::new(DesktopElectionDriver {
                        engine: Arc::clone(&engine),
                        chat_handles: deps.chat_handles.clone(),
                        host: HostRef(engine.endpoint_id().to_string()),
                    });
                    tauri::async_runtime::spawn(async move {
                        // Probe the GPU capability ONCE, off the async executor (the
                        // probe self-inits the ggml backend and can block).
                        let capability = tokio::task::spawn_blocking(|| {
                            capability_from_gpu(minutist_common::probe_primary_gpu().is_some())
                        })
                        .await
                        .unwrap_or_else(|e| {
                            tracing::warn!(
                                target: "app-main",
                                error = %e,
                                "election: GPU capability probe task join failed; parking sync-only"
                            );
                            Capability::ParkSyncOnly
                        });
                        election::run_election_loop(
                            ElectionConfig::from_env(),
                            driver,
                            election_meetings_dir,
                            capability,
                        )
                        .await;
                    });
                }

                let mut rt = self.runtime.lock().await;
                // Stop a prior bind's loops before adopting this bind's token, so
                // a re-bind never leaves the old peers-poll / account loop running.
                if let Some(prev) = rt.stop.replace(stop) {
                    prev.cancel();
                }
                rt.engine = Some(engine);
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

    /// Record an error status, emit [`AppEvent::SyncError`], and re-arm
    /// [`Self::start_requested`] so a later [`SyncControl::set_enabled(true)`]
    /// can retry the bind. Called only from [`Self::start_engine`]'s two
    /// failure arms (identity load, engine bind) — never from the success
    /// path or from [`Self::request_start`] itself — so the reset targets
    /// exactly "the bind failed" and never re-arms a request that is still in
    /// flight or that already succeeded (either of which would let a second
    /// caller double-spawn the engine).
    ///
    /// The status write happens BEFORE the flag reset, not after: this
    /// terminal write is the last thing this spawned task ever does to
    /// `runtime`, so once the flag is cleared a fresh [`Self::request_start`]
    /// racing in immediately after can safely move the status on to
    /// `Connecting` without risk of this call's `Error` write landing after
    /// it and clobbering it.
    async fn fail(&self, message: String) {
        tracing::error!(target: "app-main", "sync: {message}");
        self.runtime.lock().await.status = SyncStatus::Error {
            message: message.clone(),
        };
        self.start_requested.store(false, Ordering::SeqCst);
        let _ = self.event_tx.send(AppEvent::SyncError { context: message });
    }

    /// The bound engine, or an `Unsupported` error when the connector is disabled
    /// and a `Internal` ("still starting") error while the background bind has not
    /// yet completed. Callers use this for every engine-backed operation.
    async fn engine(&self) -> AppResult<Arc<SyncEngine>> {
        let rt = self.runtime.lock().await;
        match (&rt.engine, &rt.status) {
            (Some(engine), _) => Ok(Arc::clone(engine)),
            (None, SyncStatus::Disabled) => Err(AppError::Unsupported {
                context: "sync is disabled (enable the connector)".to_string(),
            }),
            (None, _) => Err(AppError::Internal {
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

    async fn host_ref(&self) -> HostRef {
        match self.runtime.lock().await.engine {
            Some(ref engine) => HostRef(engine.endpoint_id().to_string()),
            // Not yet bound (disabled, or the bind is still in flight): a
            // fixed placeholder, mirroring `DisabledSync` — a soft-delete/
            // restore performed in this brief window is vanishingly unlikely
            // to race a peer's own toggle of the SAME meeting.
            None => HostRef("local".to_string()),
        }
    }

    async fn sync_now(&self, meeting_id: MeetingId) -> AppResult<()> {
        let engine = self.engine().await?;
        let peers = engine.peer_ids();
        if peers.is_empty() {
            // No paired devices: nothing to reconcile against. Surface a
            // completed-with-nothing-to-do rather than an error — the meeting is
            // already as synced as it can be on a lone device.
            let _ = self.event_tx.send(AppEvent::SyncReady { meeting_id });
            return Ok(());
        }

        // Each peer is reconciled in two phases — notes (the CRDT update
        // exchange) then media (`audio.opus` + assets, content-addressed blobs) —
        // so the progress fraction is over `peers * 2` units. The media phase runs
        // even when the notes phase failed for that peer: the two are independent
        // reconciliations and a notes failure should not strand the recording.
        let steps_per_peer = 2;
        let total = peers.len() * steps_per_peer;
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
        let mut done = 0usize;
        for peer in &peers {
            // Phase 1: notes. The engine dials the peer on the sync ALPN and runs
            // the bidirectional CRDT-update exchange for this meeting.
            if let Err(e) = engine.sync_notes_to_peer(peer, meeting_id).await {
                let message = format!("syncing notes with a peer: {e}");
                tracing::warn!(target: "app-main", "sync: {message}");
                last_err = Some(message);
            }
            done += 1;
            let _ = self.event_tx.send(AppEvent::SyncProgress {
                meeting_id,
                label: "Syncing media…".to_string(),
                fraction: Some(done as f32 / total as f32),
            });

            // Phase 2: media. The engine imports this meeting's media into its
            // content-addressed blob store, exchanges manifests with the peer over
            // the same sync ALPN, and pulls the blobs it is missing over the blobs
            // ALPN — exporting each to the per-meeting `audio.opus` / `assets/*`
            // path and pinning it with a persistent tag for retention.
            if let Err(e) = engine.sync_media_to_peer(peer, meeting_id).await {
                let message = format!("syncing media with a peer: {e}");
                tracing::warn!(target: "app-main", "sync: {message}");
                last_err = Some(message);
            }
            done += 1;
            let _ = self.event_tx.send(AppEvent::SyncProgress {
                meeting_id,
                label: "Syncing media…".to_string(),
                fraction: Some(done as f32 / total as f32),
            });

            // §7 ride-alongside: exchange lifecycle with this peer in the same
            // session, so the meeting just synced carries its processing state
            // (and we learn the peer's for the meetings we hold). Best-effort and
            // NOT a progress step — a discovery failure does not fail the sync; the
            // lifecycle re-advertises on the next sync.
            if let Err(e) = engine.discover_with_peer(peer).await {
                tracing::warn!(target: "app-main", peer = %peer, error = %e, "ride-along discovery failed");
            }
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
                Err(AppError::Internal { context: message })
            }
        }
    }

    async fn delete_meeting_blobs(&self, meeting_id: MeetingId) -> AppResult<()> {
        // Best-effort, and never fails the caller: the meeting folder is already
        // gone by the time `delete_meeting` calls this, so a disabled connector or
        // a not-yet-started engine has nothing left to do.
        let rt = self.runtime.lock().await;
        let Some(engine) = rt.engine.clone() else {
            return Ok(());
        };
        drop(rt);
        if let Err(e) = engine.delete_meeting_blobs(meeting_id).await {
            tracing::warn!(
                target: "app-main",
                meeting_id = %meeting_id.0,
                error = %e,
                "sync: unpinning a deleted meeting's blobs failed (best-effort)"
            );
        }
        Ok(())
    }

    async fn set_enabled(&self, enabled: bool) -> AppResult<()> {
        // F5: request the engine start. The only caller is a successful
        // sign-in (`account_poll_pairing`) — persisting the setting here, not
        // just in the account path, means it survives a restart regardless of
        // which sign-in path called it. `request_start` is idempotent
        // (an atomic-guarded one-shot), so this is safe whether the engine is
        // already running, already starting, or has never been requested.
        //
        // `enabled = false` (only ever called from `delete_account`, alongside
        // `AccountControl::delete_account` persisting
        // `settings.connector_enabled = false` itself) does NOT tear down an
        // already-started engine: `SyncEngine` offers only an owning
        // `shutdown(self)`, and the engine `Arc` here is shared with the
        // spawned election loop and the lifecycle subscriber, so a clean stop
        // would need a cancellation path threaded through both — a known
        // follow-up, not a behaviour this call regresses (the app is about to
        // erase the local credential regardless; a lingering engine has
        // nothing account-scoped left to do until the next launch). So this
        // method only ever persists on the `true` branch; the `false` caller
        // already persisted its own reason for disabling.
        if enabled {
            let _ = self.settings.update(|s| s.connector_enabled = true).await;
            self.request_start().await;
        }
        Ok(())
    }
}

/// Resolve the relay access token: `MINUTIST_SYNC_TOKEN` if set and non-empty,
/// else the paired device's `mdc_` credential, else none.
///
/// The env var stays the override for deployments/tests (e.g. the headless hub,
/// launched with `MINUTIST_SYNC_TOKEN=<its mdc_>`). A GUI desktop has no such env,
/// so it resolves the token from its loaded device credential — the same `mdc_`
/// the flipped relay validates via `/relay-authz`. Without this fallback the
/// desktop presents no relay token and the relay 401s every connection (0043).
/// Mirrors the phone (`capacitor.ts`: `getStoredCredential() ?? RELAY_AUTH_TOKEN`).
fn resolve_relay_token(app_data_base: &Path, _settings: &SettingsHandle) -> Option<String> {
    match std::env::var(RELAY_TOKEN_ENV) {
        Ok(token) if !token.is_empty() => Some(token),
        _ => credential_relay_token(app_data_base),
    }
}

/// The relay token derived from the paired device's stored credential: the
/// `mdc_` device credential if a `tunnel_device.json` is present, else none.
/// Factored out of [`resolve_relay_token`] so the credential-fallback path is
/// testable without touching the process-global `MINUTIST_SYNC_TOKEN` env.
fn credential_relay_token(app_data_base: &Path) -> Option<String> {
    crate::account::load_device_credential(app_data_base).map(|cred| cred.device_credential)
}

/// The desktop's [`ElectionDriver`] (producer-gate S4): drives the election loop's
/// collaborators — the bound [`SyncEngine`] (advertise lifecycle state over
/// discovery; push derived artifacts to peers), the orchestrator (run the offline
/// reprocess pipeline), and — after a successful reprocess — the held-summariser
/// pass (F4-summary), all via the [`ChatHandles`] bundle `app-main` injects. This
/// is the only place that links `election` + `sync` + `ipc-bridge`, which is why
/// the loop abstracts them behind the trait.
struct DesktopElectionDriver {
    engine: Arc<SyncEngine>,
    chat_handles: ChatHandles,
    /// This host's identity (the endpoint id) — the lowest-`HostRef` tiebreak key.
    host: HostRef,
}

#[async_trait]
impl ElectionDriver for DesktopElectionDriver {
    fn host_ref(&self) -> HostRef {
        self.host.clone()
    }

    async fn advertise(&self) {
        // Propagate our lifecycle state (the fresh Claimed / Processed) to every
        // paired peer over the discovery exchange. Best-effort: on failure the
        // state re-advertises on the next loop tick / sync.
        if let Err(e) = self.engine.discover_all().await {
            tracing::warn!(target: "app-main", error = %e, "election: advertise (discover_all) failed");
        }
    }

    async fn process(&self, meeting_id: MeetingId) -> AppResult<()> {
        // The offline pipeline (re-transcribe + diarize) over the synced-in
        // `audio.opus`, rewriting `transcript.json`. `reprocess` takes one
        // offline claim for the whole pass; it does NOT summarise (parity with
        // the standalone offline ops — see `orchestrator::reprocess`'s doc).
        self.chat_handles
            .orchestrator
            .reprocess(&self.chat_handles.index, meeting_id)
            .await?;

        // F4-summary: run the held-summariser pass AFTER reprocess, matching the
        // non-delegated default (`settings.auto_summarise_on_stop`) so a
        // delegated meeting converges with a real `summary.md` rather than a
        // permanently-empty Artifacts slot. Best-effort: reprocess already
        // produced the authoritative transcript + diarization, so a failed
        // summarise must not fail the whole claim (which would leave it to
        // lapse and reprocess a meeting that is otherwise done for no reason).
        if self.chat_handles.settings.current().auto_summarise_on_stop {
            if let Err(e) = ipc_bridge::run_held_summarise(&self.chat_handles, meeting_id).await {
                tracing::warn!(
                    target: "app-main",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "election: post-reprocess summarise failed (best-effort; transcript + diarization remain valid)"
                );
            }
        }

        Ok(())
    }

    async fn push_artifacts(&self, meeting_id: MeetingId) {
        // Push the derived artifacts to every peer BEFORE the loop advertises
        // `Processed`, so a consumer never learns `Processed` without the outputs
        // being retrievable (DESIGN_producer-gate.md §6.7). Best-effort per peer.
        // A peer in failed-dial backoff is skipped without dialling: a
        // stale/unreachable peer would otherwise burn the per-dial timeout on
        // every push, the same cost the discovery sweep avoids (0029 item 6).
        // Skipping is safe for the producer gate — `advertise` (discover_all)
        // skips the same suppressed peer, so it never learns `Processed` without
        // the artifacts being retrievable (DESIGN_producer-gate.md §6.7). The
        // skipped meeting's artifacts are not re-pushed by this path (it fires
        // once per forward `Processed`); they converge when the peer reconnects
        // and pulls, via the `discover_all` backstop, once its backoff clears.
        for peer in self.engine.peer_ids() {
            if self.engine.is_suppressed(&peer) {
                tracing::debug!(target: "app-main", peer = %peer, "election: skipping artifact push to a dial-suppressed peer");
                continue;
            }
            if let Err(e) = self.engine.sync_artifacts_to_peer(&peer, meeting_id).await {
                tracing::warn!(target: "app-main", peer = %peer, error = %e, "election: pushing artifacts to a peer failed");
            }
        }
    }
}

/// Map a GPU-presence probe to the election [`Capability`]. Pure, so the self-gating
/// decision — an eligible host claims + processes; a GPU-less one parks sync-only and
/// never claims — is unit-testable without touching the ggml backend the probe inits.
fn capability_from_gpu(has_gpu: bool) -> Capability {
    if has_gpu {
        Capability::Eligible
    } else {
        Capability::ParkSyncOnly
    }
}

#[cfg(test)]
mod tests {
    //! End-to-end coverage of the connected-tier `SyncControl` glue
    //! (`ConnectedSync`) through the DEPLOYED relay (`sync.minutist.ai`).
    //!
    //! This module compiles only under the `connected` feature (the whole `sync`
    //! module is `#[cfg(feature = "connected")]`), so it is connected-gated by
    //! construction. Where the engine-level `crates/sync/tests/{relay_live,
    //! blobs_live}.rs` drive `SyncEngine` directly, these tests exercise the
    //! app-main lifecycle and the exact methods the Tauri commands delegate to:
    //! two real [`ConnectedSync`] instances, paired directly through the
    //! underlying `SyncEngine` (`my_ticket` / `add_peer_from_ticket` — desktop no
    //! longer exposes these itself) and reconciled THROUGH the [`SyncControl`]
    //! trait's `sync_now` / `status`, proving the IPC surface plus the
    //! `SyncReady`/`SyncProgress` bus events on top of the engine-level coverage.
    //!
    //! GATED: skips (with an `eprintln!`) unless `MINUTIST_SYNC_TOKEN` is set, so
    //! a normal `cargo test` and CI never touch the network. The `sync` module is
    //! declared in `main.rs`, so these live in the BINARY's test target (the `lib`
    //! is a Tauri stub); target the binary to run them:
    //!
    //! ```sh
    //! MINUTIST_SYNC_TOKEN=<relay-access-token> \
    //!   cargo test -p minutist --bin minutist sync::tests -- --nocapture
    //! ```
    //!
    //! ## Engine startup in a `#[tokio::test]`
    //!
    //! These tests drive the REAL [`ConnectedSync::new`], which spawns startup via
    //! [`tauri::async_runtime::spawn`]. That call lazily initialises Tauri's OWN
    //! multi-threaded tokio runtime (a `OnceLock<GlobalRuntime>`), independent of
    //! the test's `#[tokio::test]` runtime, so the spawned `start_engine` future
    //! runs and binds the iroh endpoint even though the test runtime is
    //! current-thread. The test observes the bind by polling [`SyncControl::status`]
    //! (which only locks the `Arc<Mutex<Runtime>>` the spawned task updates) until
    //! it reads [`SyncStatus::Idle`], with a 20 s timeout — so no awaited-start
    //! fallback was needed.

    use std::path::Path;
    use std::time::Duration;

    use super::*;
    use minutist_common::MeetingId;
    use notes_crdt::{MeetingFolder, NotesStore};
    use persistence::save_note_asset;
    use settings::{JsonFileStore, SettingsHandle};

    /// How long to wait for a freshly-built `ConnectedSync`'s background engine
    /// startup (identity load + token-gated relay bind) to reach `Idle`.
    const BIND_TIMEOUT: Duration = Duration::from_secs(20);
    /// Outer guard on a `sync_now` so a relay failure surfaces as a test failure
    /// (timeout) rather than a hang.
    const SYNC_NOW_TIMEOUT: Duration = Duration::from_secs(60);

    /// Read the live relay token from the gating env var, or `None` to skip.
    fn relay_token() -> Option<String> {
        match std::env::var(RELAY_TOKEN_ENV) {
            Ok(t) if !t.is_empty() => Some(t),
            _ => None,
        }
    }

    /// The election capability self-gate (S4): a GPU maps to `Eligible` (claims +
    /// processes); no GPU maps to `ParkSyncOnly` (never claims). Not network-gated.
    #[test]
    fn capability_maps_gpu_presence() {
        assert!(matches!(
            super::capability_from_gpu(true),
            Capability::Eligible
        ));
        assert!(matches!(
            super::capability_from_gpu(false),
            Capability::ParkSyncOnly
        ));
    }

    /// The relay-token credential fallback (0043): a paired desktop with no
    /// `MINUTIST_SYNC_TOKEN` env resolves its relay token from the seeded
    /// `tunnel_device.json` `mdc_` credential; an unpaired dir yields `None`.
    #[test]
    fn credential_relay_token_reads_the_seeded_mdc() {
        // Unpaired: no credential file -> None.
        let empty = tempfile::TempDir::new().expect("tempdir");
        assert_eq!(super::credential_relay_token(empty.path()), None);

        // Paired: a seeded tunnel_device.json -> its `device_credential`.
        let paired = tempfile::TempDir::new().expect("tempdir");
        let mdc = "mdc_deadbeef.cafef00d";
        std::fs::write(
            paired.path().join("tunnel_device.json"),
            format!(
                r#"{{"device_credential":"{mdc}","account_id":"acct-test","device_id":"deadbeef"}}"#
            ),
        )
        .expect("write credential");
        assert_eq!(
            super::credential_relay_token(paired.path()).as_deref(),
            Some(mdc)
        );
    }

    /// `set_enabled(true)` must persist `settings.connector_enabled` and move
    /// the status off `Disabled` WITHOUT needing network access, so the setting
    /// survives a restart regardless of when the engine itself finishes
    /// binding. Does not use `bound_device()` (which polls for a real network
    /// bind) — this only asserts the synchronous, local-only part of
    /// `request_start`.
    #[tokio::test]
    async fn set_enabled_true_persists_the_setting_and_requests_start() {
        let app_data = tempfile::TempDir::new().expect("app-data tempdir");
        let meetings = tempfile::TempDir::new().expect("meetings tempdir");
        let store = JsonFileStore::new(app_data.path().join("settings.store"));
        let settings = SettingsHandle::new(store).expect("build settings handle");
        assert!(!settings.current().connector_enabled, "starts disabled");

        let (event_tx, _events) = broadcast::channel(256);
        let index = Arc::new(
            ipc_bridge::MeetingIndex::open(":memory:")
                .await
                .expect("open in-memory index"),
        );
        let sync = ConnectedSync::new(
            settings.clone(),
            event_tx,
            app_data.path().to_path_buf(),
            meetings.path().to_path_buf(),
            None,
            index,
        );
        assert_eq!(sync.status().await, SyncStatus::Disabled);

        sync.set_enabled(true).await.expect("enable");

        assert!(
            settings.current().connector_enabled,
            "connector_enabled must be persisted, not just flipped in memory"
        );
        // `request_start` marks `Connecting` synchronously, before any network
        // I/O — no need to wait for (or gate on) a real bind here.
        assert_ne!(sync.status().await, SyncStatus::Disabled);
    }

    /// Construction with a signed-in credential (`tunnel_device.json` present)
    /// but `connector_enabled = false` (the drifted state a failed
    /// `delete_account` file removal can leave behind) must both start the
    /// engine anyway AND self-heal the persisted setting, rather than leaving
    /// the device permanently signed-in-but-disabled with no way back short of
    /// hand-editing the settings store.
    #[tokio::test]
    async fn construction_self_heals_a_signed_in_but_disabled_drift() {
        let app_data = tempfile::TempDir::new().expect("app-data tempdir");
        let meetings = tempfile::TempDir::new().expect("meetings tempdir");
        std::fs::write(
            app_data.path().join("tunnel_device.json"),
            r#"{"device_credential":"mdc_dev.secret","account_id":"acct-1","device_id":"dev-1"}"#,
        )
        .expect("seed a signed-in credential");

        let store = JsonFileStore::new(app_data.path().join("settings.store"));
        let settings = SettingsHandle::new(store).expect("build settings handle");
        assert!(
            !settings.current().connector_enabled,
            "drifted state starts with the flag false despite being signed in"
        );

        let (event_tx, _events) = broadcast::channel(256);
        let index = Arc::new(
            ipc_bridge::MeetingIndex::open(":memory:")
                .await
                .expect("open in-memory index"),
        );
        let sync = ConnectedSync::new(
            settings.clone(),
            event_tx,
            app_data.path().to_path_buf(),
            meetings.path().to_path_buf(),
            None,
            index,
        );

        // Both the engine-start request and the settings self-heal are spawned,
        // fire-and-forget tasks (construction itself is synchronous and returns
        // before either has necessarily run) — poll for both rather than
        // assuming a particular scheduling order.
        let started = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if sync.status().await != SyncStatus::Disabled {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            started.is_ok(),
            "engine start must be requested for a signed-in device despite connector_enabled=false"
        );

        let healed = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if settings.current().connector_enabled {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            healed.is_ok(),
            "connector_enabled must be self-healed to true for a signed-in device"
        );
    }

    /// A handle to a built `ConnectedSync` plus its event receiver and the temp
    /// dirs that must outlive it (dropping a `TempDir` deletes its contents).
    struct Device {
        sync: Arc<ConnectedSync>,
        events: broadcast::Receiver<AppEvent>,
        /// The app-data BASE (holds the device key); kept alive for the test.
        _app_data: tempfile::TempDir,
        /// The meetings root (`{uuid}` folders); seeded and projected against.
        meetings: tempfile::TempDir,
    }

    impl Device {
        fn meetings_root(&self) -> &Path {
            self.meetings.path()
        }
    }

    /// Build a `SettingsHandle` with `connector_enabled = true` (the gate
    /// `ConnectedSync::new` checks before it spawns startup). The store is a real
    /// `JsonFileStore` over a tempdir-rooted path; the field defaults to `false`,
    /// so it is flipped through the handle's `update` API (mirroring the field the
    /// `settings` unit test at `crates/settings/src/lib.rs` sets) rather than by
    /// hand-writing a settings file.
    async fn connector_enabled_settings(app_data_base: &Path) -> SettingsHandle {
        let store = JsonFileStore::new(app_data_base.join("settings.store"));
        let handle = SettingsHandle::new(store).expect("build settings handle");
        handle
            .update(|s| s.connector_enabled = true)
            .await
            .expect("enable connector");
        assert!(
            handle.current().connector_enabled,
            "settings handle must report the connector enabled"
        );
        handle
    }

    /// Build a real `ConnectedSync` (engine startup spawned via
    /// `tauri::async_runtime::spawn`) with distinct `app_data_base` / `meetings`
    /// tempdirs and a fresh `broadcast::channel(256)` whose receiver is retained
    /// for event assertions, then poll `status()` until the background bind
    /// reaches `Idle` (or fail on `Error` / the bind timeout). The relay defaults
    /// to `SyncConfig::DEFAULT_RELAY_URL`; the token comes from
    /// `MINUTIST_SYNC_TOKEN` via the production `resolve_relay_token` path.
    async fn bound_device() -> Device {
        let app_data = tempfile::TempDir::new().expect("app-data tempdir");
        let meetings = tempfile::TempDir::new().expect("meetings tempdir");
        let settings = connector_enabled_settings(app_data.path()).await;
        let (event_tx, events) = broadcast::channel(256);
        let index = Arc::new(
            ipc_bridge::MeetingIndex::open(":memory:")
                .await
                .expect("open in-memory index"),
        );

        let sync = ConnectedSync::new(
            settings,
            event_tx,
            app_data.path().to_path_buf(),
            meetings.path().to_path_buf(),
            // No election deps in the relay E2E test — it exercises the SyncControl
            // surface, not the producer-gate loop (which the crates/election tests
            // cover).
            None,
            index,
        );

        let deadline = tokio::time::Instant::now() + BIND_TIMEOUT;
        loop {
            match sync.status().await {
                SyncStatus::Idle => break,
                SyncStatus::Error { message } => panic!("engine failed to bind: {message}"),
                _ => {
                    if tokio::time::Instant::now() >= deadline {
                        panic!("engine did not reach Idle within {BIND_TIMEOUT:?}");
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }

        Device {
            sync,
            events,
            _app_data: app_data,
            meetings,
        }
    }

    /// The seed note's ProseMirror doc and its marker text.
    fn seed_doc(text: &str) -> serde_json::Value {
        serde_json::json!({
            "type": "doc",
            "content": [{ "type": "paragraph",
                "content": [{ "type": "text", "text": text }] }]
        })
    }

    /// Project a meeting's authoritative `notes.ydoc` to ProseMirror JSON via
    /// public `notes_crdt` APIs, so two devices' converged state can be compared
    /// independent of v1 encoding details (mirrors `relay_live.rs::projected`).
    fn projected(root: &Path, meeting: MeetingId) -> serde_json::Value {
        let v1 = NotesStore::read_ydoc_state(root, meeting)
            .expect("read ydoc state")
            .expect("meeting has a notes.ydoc");
        let doc = notes_crdt::ydoc::new_ydoc();
        notes_crdt::ydoc::apply_update_v1(&doc, &v1).expect("apply v1 state");
        notes_crdt::ydoc::ydoc_to_json(&doc)
    }

    /// Drain a receiver and report whether a `SyncReady` for `meeting` was seen.
    /// Non-blocking: `sync_now` has already published all its events by the time
    /// it returns, so every buffered event is present in the channel.
    fn drained_sync_ready(rx: &mut broadcast::Receiver<AppEvent>, meeting: MeetingId) -> bool {
        let mut seen = false;
        while let Ok(ev) = rx.try_recv() {
            if let AppEvent::SyncReady { meeting_id } = ev {
                if meeting_id == meeting {
                    seen = true;
                }
            }
        }
        seen
    }

    /// Two paired `ConnectedSync`s reconcile a meeting (notes + media) through the
    /// deployed relay. Pairing goes directly through the underlying
    /// `SyncEngine::my_ticket` / `add_peer_from_ticket` (the same primitives the
    /// phone client's `sync-ffi` calls — desktop no longer exposes them itself);
    /// `syncNow`/`sync_now` still goes through `SyncControl`, the exact method the
    /// `sync_now` command delegates to. Proves: pairing, notes convergence,
    /// byte-identical media, and the `SyncReady` bus event on the initiator.
    #[tokio::test]
    async fn connected_sync_e2e_through_the_relay() {
        let Some(_token) = relay_token() else {
            eprintln!(
                "SKIP connected_sync_e2e_through_the_relay: set MINUTIST_SYNC_TOKEN to run the live relay test"
            );
            return;
        };
        eprintln!(
            "connected_sync_e2e: using relay {}",
            SyncConfig::DEFAULT_RELAY_URL
        );

        let device_a = bound_device().await;
        let device_b = bound_device().await;

        // Mutual pairing: each side adds the other's ticket directly on the
        // underlying engine (bypassing `SyncControl`, which no longer exposes
        // manual pairing — desktop's production path is account-mediated
        // discovery only now; this test's subject is `sync_now`/relay
        // reconciliation, and pairing is just its setup). Sync requires it —
        // B's accept side rejects an inbound connection from an unpaired peer,
        // so A must be in B's directory (and vice versa) for the
        // reconciliation below to be served.
        let engine_a = device_a
            .sync
            .runtime
            .lock()
            .await
            .engine
            .clone()
            .expect("A must be bound");
        let engine_b = device_b
            .sync
            .runtime
            .lock()
            .await
            .engine
            .clone()
            .expect("B must be bound");
        let ticket_a = engine_a.my_ticket();
        let ticket_b = engine_b.my_ticket();
        engine_a.add_peer_from_ticket(&ticket_b).expect("A pairs B");
        engine_b.add_peer_from_ticket(&ticket_a).expect("B pairs A");

        // Let both endpoints home to the relay (the token-gated relay handshake)
        // before the first dial.
        tokio::time::sleep(Duration::from_secs(3)).await;

        // Seed a meeting on A: notes (first write, no existing `notes.ydoc`) plus a
        // synthetic `audio.opus` and one note asset, so the media phase of
        // `sync_now` has content to reconcile. B holds nothing for this meeting.
        let meeting = MeetingId::new();
        let marker = "hello through ConnectedSync over the relay";
        let json = seed_doc(marker);
        MeetingFolder::ensure(device_a.meetings_root(), meeting).expect("ensure A meeting folder");
        NotesStore::save(device_a.meetings_root(), meeting, &json, marker).expect("seed A notes");
        let audio = b"OggS-synthetic-opus-payload-connected-sync".repeat(128);
        std::fs::write(
            device_a
                .meetings_root()
                .join(meeting.0.to_string())
                .join("audio.opus"),
            &audio,
        )
        .expect("seed A audio.opus");
        let asset_bytes = b"\x89PNG\r\n\x1a\nsynthetic-asset-connected-sync".repeat(32);
        let asset_name = save_note_asset(device_a.meetings_root(), meeting, &asset_bytes, "png")
            .expect("seed A asset");
        eprintln!(
            "connected_sync_e2e: seeded A notes + audio={} bytes + asset {} ({} bytes)",
            audio.len(),
            asset_name,
            asset_bytes.len()
        );

        // A reconciles the meeting against every paired peer (here, B) through the
        // trait's `sync_now` — notes then media — under a 60 s guard so a relay
        // failure is a timeout, not a hang.
        let started = std::time::Instant::now();
        tokio::time::timeout(SYNC_NOW_TIMEOUT, device_a.sync.sync_now(meeting))
            .await
            .expect("sync_now timed out")
            .expect("sync_now failed");
        eprintln!(
            "connected_sync_e2e: sync_now completed in {:?}",
            started.elapsed()
        );

        // B must now hold A's notes, equal at the projection level.
        let a_json = projected(device_a.meetings_root(), meeting);
        let b_json = projected(device_b.meetings_root(), meeting);
        assert_eq!(a_json, b_json, "B must converge to A's notes via the relay");
        assert!(
            serde_json::to_string(&b_json).unwrap().contains(marker),
            "B's converged note must carry the seeded marker text"
        );

        // B must hold A's audio byte-for-byte.
        let b_audio = std::fs::read(
            device_b
                .meetings_root()
                .join(meeting.0.to_string())
                .join("audio.opus"),
        )
        .expect("B must have audio.opus after sync_now");
        assert_eq!(
            b_audio, audio,
            "B's audio must be byte-identical to A's after sync_now"
        );

        // B must hold A's asset byte-for-byte under the same portable filename.
        let b_asset = persistence::read_note_asset(device_b.meetings_root(), meeting, &asset_name)
            .expect("B must have the asset after sync_now");
        assert_eq!(
            b_asset, asset_bytes,
            "B's asset must be byte-identical to A's after sync_now"
        );

        // A's event bus must carry a `SyncReady` for this meeting (the success
        // terminal `sync_now` publishes). The receiver was retained from this
        // device's construction, so it holds every event published since — drain
        // it and look for the terminal `SyncReady`.
        let mut device_a = device_a;
        assert!(
            drained_sync_ready(&mut device_a.events, meeting),
            "A's bus must carry a SyncReady for the meeting after a successful sync_now"
        );

        eprintln!("connected_sync_e2e: PASS — converged through the relay via SyncControl");
    }

    /// A lone `ConnectedSync` with no paired peers: `sync_now` takes the
    /// peer-empty branch — `Ok(())` plus a `SyncReady` on the bus — rather than
    /// erroring. The engine still binds to the relay (gated), so this also
    /// exercises the bind path with the connector enabled.
    #[tokio::test]
    async fn sync_now_with_no_peers_reports_ready() {
        let Some(_token) = relay_token() else {
            eprintln!(
                "SKIP sync_now_with_no_peers_reports_ready: set MINUTIST_SYNC_TOKEN to run the live relay test"
            );
            return;
        };

        let mut device = bound_device().await;

        let meeting = MeetingId::new();
        let marker = "lone device note";
        MeetingFolder::ensure(device.meetings_root(), meeting).expect("ensure meeting folder");
        NotesStore::save(device.meetings_root(), meeting, &seed_doc(marker), marker)
            .expect("seed notes");

        tokio::time::timeout(SYNC_NOW_TIMEOUT, device.sync.sync_now(meeting))
            .await
            .expect("sync_now timed out")
            .expect("sync_now (no peers) must be Ok");

        assert!(
            drained_sync_ready(&mut device.events, meeting),
            "the peer-empty sync_now must publish a SyncReady for the meeting"
        );
    }
}
