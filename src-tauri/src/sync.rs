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
//! reconciles a meeting against every registered peer — both its notes (the CRDT
//! update exchange) and its media (`audio.opus` + assets, as content-addressed
//! blobs) — and emits these as it runs.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use election::{Capability, ElectionConfig, ElectionDriver};
use ipc_bridge::{IpcError, MeetingIndex, SyncControl};
use minutist_common::{AppEvent, AppResult, HostRef, MeetingId, SyncStatus};
use orchestrator::Orchestrator;
use settings::SettingsHandle;
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use tokio::sync::broadcast;
use tokio::sync::Mutex;

/// The environment variable carrying the self-hosted relay's access token. Takes
/// precedence over the settings value so a deployment / test can inject one
/// without persisting it. The token gates admission to the relay's
/// `AccessControl`; it is never logged.
const RELAY_TOKEN_ENV: &str = "MINUTIST_SYNC_TOKEN";

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
}

/// The collaborators the producer-gate election loop needs, passed in from
/// `app-main` (which links `sync` + `orchestrator` + `election` together). Held so
/// [`ConnectedSync::start_engine`] can spawn the loop once the engine binds. `None`
/// disables the gate (the connected E2E tests build the sync control without it).
pub struct ElectionDeps {
    pub orchestrator: Arc<Orchestrator>,
    pub index: Arc<MeetingIndex>,
}

/// The connected `SyncControl` implementation.
pub struct ConnectedSync {
    event_tx: broadcast::Sender<AppEvent>,
    runtime: Arc<Mutex<Runtime>>,
    /// When set (production), the engine-bind path spawns the producer-gate election
    /// loop with a [`DesktopElectionDriver`] over these + the bound engine.
    election: Option<ElectionDeps>,
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
        election: Option<ElectionDeps>,
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
            election,
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

        // Clone the meetings root for the lifecycle subscriber + the election loop
        // before the config consumes it.
        let subscriber_meetings_dir = meetings_dir.clone();
        let election_meetings_dir = meetings_dir.clone();
        let mut config = SyncConfig::new(meetings_dir);
        if let Some(token) = relay_token {
            config = config.with_relay_auth_token(token);
        }

        match SyncEngine::start(config, identity).await {
            Ok(engine) => {
                let engine = Arc::new(engine);
                // Persist the processing-lifecycle states the engine surfaces from
                // discovery. The subscriber loop lives in `ipc-bridge` (which owns
                // the `persistence` edge); here we only hand it the engine's
                // receiver, so this module keeps no direct `persistence` use.
                ipc_bridge::spawn_lifecycle_subscriber(
                    engine.subscribe_lifecycle_events(),
                    subscriber_meetings_dir,
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
                        orchestrator: Arc::clone(&deps.orchestrator),
                        index: Arc::clone(&deps.index),
                        host: HostRef(engine.endpoint_id().to_string()),
                    });
                    tauri::async_runtime::spawn(async move {
                        // Probe the GPU capability ONCE, off the async executor (the
                        // probe self-inits the ggml backend and can block).
                        let capability = tokio::task::spawn_blocking(|| {
                            capability_from_gpu(minutist_common::probe_primary_gpu().is_some())
                        })
                        .await
                        .unwrap_or(Capability::ParkSyncOnly);
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
        for peer in peers {
            // Phase 1: notes. The engine dials the peer on the sync ALPN and runs
            // the bidirectional CRDT-update exchange for this meeting.
            if let Err(e) = engine.sync_notes(peer, meeting_id).await {
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
            if let Err(e) = engine.sync_media(peer, meeting_id).await {
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
            if let Err(e) = engine.discover_with(peer).await {
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

/// The desktop's [`ElectionDriver`] (producer-gate S4): drives the election loop's
/// two collaborators — the bound [`SyncEngine`] (advertise lifecycle state over
/// discovery; push derived artifacts to peers) and the [`Orchestrator`] (run the
/// offline reprocess pipeline). This is the only place that links `election` +
/// `sync` + `orchestrator`, which is why the loop abstracts them behind the trait.
struct DesktopElectionDriver {
    engine: Arc<SyncEngine>,
    orchestrator: Arc<Orchestrator>,
    index: Arc<MeetingIndex>,
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
        // `audio.opus`, rewriting `transcript.json` / `summary.md`. `reprocess`
        // takes one offline claim for the whole pass.
        self.orchestrator.reprocess(&self.index, meeting_id).await
    }

    async fn push_artifacts(&self, meeting_id: MeetingId) {
        // Push the derived artifacts to every peer BEFORE the loop advertises
        // `Processed`, so a consumer never learns `Processed` without the outputs
        // being retrievable (DESIGN_producer-gate.md §6.7). Best-effort per peer.
        for peer in self.engine.peer_ids() {
            if let Err(e) = self
                .engine
                .sync_artifacts_to_peer(&peer.to_string(), meeting_id)
                .await
            {
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
    //! two real [`ConnectedSync`] instances, paired and reconciled THROUGH the
    //! [`SyncControl`] trait (`my_ticket` / `add_peer` / `sync_now` / `status`),
    //! proving the IPC surface plus the `SyncReady`/`SyncProgress` bus events on
    //! top of the engine-level coverage.
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
    use persistence::{save_note_asset, MeetingFolder, NotesStore};
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

        let sync = ConnectedSync::new(
            settings,
            event_tx,
            app_data.path().to_path_buf(),
            meetings.path().to_path_buf(),
            // No election deps in the relay E2E test — it exercises the SyncControl
            // surface, not the producer-gate loop (which the crates/election tests
            // cover).
            None,
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
    /// public `persistence` APIs, so two devices' converged state can be compared
    /// independent of v1 encoding details (mirrors `relay_live.rs::projected`).
    fn projected(root: &Path, meeting: MeetingId) -> serde_json::Value {
        let v1 = NotesStore::read_ydoc_state(root, meeting)
            .expect("read ydoc state")
            .expect("meeting has a notes.ydoc");
        let doc = persistence::ydoc::new_ydoc();
        persistence::ydoc::apply_update_v1(&doc, &v1).expect("apply v1 state");
        persistence::ydoc::ydoc_to_json(&doc)
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
    /// deployed relay, driven entirely through the `SyncControl` trait — the exact
    /// methods the `sync_get_my_ticket` / `sync_add_peer` / `sync_now` commands
    /// delegate to. Proves: pairing via tickets, notes convergence, byte-identical
    /// media, and the `SyncReady` bus event on the initiator.
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

        // Mutual pairing THROUGH the trait: each side adds the other's ticket via
        // `my_ticket` + `add_peer`. Sync requires it — B's accept side rejects an
        // inbound connection from an unpaired peer, so A must be in B's directory
        // (and vice versa) for the reconciliation below to be served.
        let ticket_a = device_a.sync.my_ticket().await.expect("A ticket");
        let ticket_b = device_b.sync.my_ticket().await.expect("B ticket");
        device_a.sync.add_peer(ticket_b).await.expect("A pairs B");
        device_b.sync.add_peer(ticket_a).await.expect("B pairs A");

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
