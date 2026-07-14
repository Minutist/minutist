//! `minutist-hub` — the Minutist headless server daemon.
//!
//! A user-installed, always-on service the user runs on their OWN hardware (a
//! homelab box or GPU workstation). It pairs into the user's device mesh exactly
//! like a desktop and holds meeting plaintext — but on hardware the user owns and
//! controls, so it sits within the same trust boundary as the desktop. It is NOT
//! the hosted relay, which only ever brokers ciphertext.
//!
//! Two capabilities behind one binary, selected by configuration:
//!
//!   1. an always-on **sync hub** — an always-online [`sync::SyncEngine`] peer so
//!      two sometimes-online devices converge through the user's own box, rather
//!      than leaning on the relay's deferred store-and-forward inbox; and
//!   2. (post-launch) a **GPU processing node** that runs the ASR / diarize /
//!      summarise pipeline for meetings captured on GPU-less devices.
//!
//! ## Commands
//!
//! - (no subcommand) — run the daemon: resolve the data root, start the sync
//!   engine bound to the relay, load paired peers from `{data-dir}/peers`, and run
//!   until `SIGTERM` / `SIGINT`.
//! - `print-ticket` — print this device's pairing ticket to stdout and exit (paste
//!   it into a desktop's Sync settings to pair).
//! - `add-peer <ticket>` — register a peer device by its pairing ticket, appending
//!   it to `{data-dir}/peers`. A running daemon re-reads that file periodically, so
//!   the new peer is authorised without a restart.
//! - `status` — print the hub's state as JSON to stdout and exit: endpoint id,
//!   relay, authorised peers, and held meetings each with a content digest of their
//!   notes. A read-only filesystem oracle for automated tests (it does not contact
//!   the running daemon).
//!
//! Timing constants (poll / push-debounce / shutdown-grace) are overridable via
//! env vars in milliseconds (a sub-second test mode) — see the constants below.
//! `MINUTIST_HUB_LOG_JSON=1` switches tracing to a structured JSON formatter.
//!
//! The data root is entirely separate from any desktop's `{app-data}` — the
//! single-writer rule applies per data root, so the daemon must never share a root
//! with another process.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use clap::{Parser, Subcommand};
use iroh_tickets::endpoint::EndpointTicket;
use minutist_common::{AppError, AppResult, MeetingId, ProcessingLifecycle};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use sync::{
    run_account_refresh_loop, AccountEndpoint, AccountEndpointSource, DeviceIdentity, SyncConfig,
    SyncEngine,
};
use tokio::sync::broadcast::{self, error::RecvError};
use tokio::sync::Notify;

/// Default shutdown drain window; override `MINUTIST_HUB_SHUTDOWN_GRACE_MS`.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// Default peers-file re-read interval (honours `add-peer` without a restart);
/// override `MINUTIST_HUB_POLL_MS`.
const PEER_POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Default minimum gap between reciprocal pushes to the same peer (so a rapidly
/// reconnecting peer is not re-pushed every time); override
/// `MINUTIST_HUB_PUSH_DEBOUNCE_MS`.
const PEER_PUSH_DEBOUNCE: Duration = Duration::from_secs(15);

/// Default interval for the recovery discovery sweep — a periodic re-discovery of
/// every known peer so a lifecycle state a consumer dropped (`Lagged`) or skipped
/// (advertised before the meeting's folder had synced in) is eventually
/// re-applied; override `MINUTIST_HUB_DISCOVERY_MS`.
const DISCOVERY_INTERVAL: Duration = Duration::from_secs(300);

/// Interval between account-directory refreshes (fetch peer list + re-register
/// endpoint). Same as the desktop's B4 wiring in `src-tauri/src/sync.rs`.
const ACCOUNT_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// The default account API base URL. Overridable via `--relay-api-url`.
const DEFAULT_API_URL: &str = "https://api.minutist.ai";

/// The filename under `{data-dir}` that holds the seeded device credential.
/// Written by an operator (or the e2e harness) — the hub never runs the
/// interactive device-code flow.
const CREDENTIAL_FILE: &str = "tunnel_device.json";

/// The persisted device credential for account-mediated peer discovery
/// (`{data-dir}/tunnel_device.json`). Seeded by the operator; the headless
/// daemon reads it on startup but never writes it (no interactive pairing).
///
/// The three-field JSON shape is a contract with the seeding harness: the same
/// file the desktop's `app-main` writes on successful pairing. A headless instance
/// is seeded directly from the operator tooling rather than running the
/// device-code flow.
///
/// `device_credential` is the long-lived `mdc_` bearer — never logged.
#[derive(Debug, Deserialize, PartialEq)]
struct StoredCredential {
    device_credential: String,
    account_id: String,
    device_id: String,
}

impl StoredCredential {
    /// Load the stored credential, or `None` when absent / unreadable / corrupt.
    /// A missing file is the normal unauthenticated case; a corrupt file is
    /// treated the same way (the operator can re-seed). Both paths leave the hub
    /// running with peers-file pairing only — no panic, no startup failure.
    fn load(data_dir: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(data_dir.join(CREDENTIAL_FILE)).ok()?;
        match serde_json::from_str::<Self>(&raw) {
            Ok(c) if !c.device_credential.is_empty() => Some(c),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!(
                    target: "hub",
                    error = %e,
                    "tunnel_device.json is corrupt; skipping account-mediated discovery"
                );
                None
            }
        }
    }
}

/// Adapts `tunnel_client::AccountDirectoryClient` onto `sync::AccountEndpointSource`.
///
/// `tunnel-client` stays a near-leaf (no `sync` edge); `headless` is the
/// assembler that depends on both and bridges them. This adapter mirrors the
/// `AccountDirectorySource` in `app-main`'s `src-tauri/src/sync.rs`.
struct AccountDirectorySource {
    client: tunnel_client::AccountDirectoryClient,
}

#[async_trait]
impl AccountEndpointSource for AccountDirectorySource {
    async fn list_endpoints(&self) -> AppResult<Vec<AccountEndpoint>> {
        let devices = self.client.list_devices().await.map_err(|e| AppError::Internal {
            context: format!("account directory list: {e}"),
        })?;
        Ok(devices
            .into_iter()
            .map(|d| AccountEndpoint {
                device_id: d.device_id,
                endpoint_id: d.endpoint_id,
                relay_url: d.relay_url,
            })
            .collect())
    }

    async fn register_self(&self, endpoint: &AccountEndpoint) -> AppResult<()> {
        self.client
            .register_self_endpoint(&endpoint.endpoint_id, &endpoint.relay_url)
            .await
            .map_err(|e| AppError::Internal {
                context: format!("account directory register-self: {e}"),
            })
    }
}

/// A timing default overridable via an env var (milliseconds), so a test mode can
/// collapse the hub's timers to sub-second without touching production defaults.
///
/// A parsed `0` falls back to `default`: one of these timers feeds
/// `tokio::time::interval`, which panics on a zero period, and "as fast as
/// possible" is never a useful hub cadence — so zero is treated as unset.
fn dur_or_env(var: &str, default: Duration) -> Duration {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .map(Duration::from_millis)
        .unwrap_or(default)
}

/// Command-line surface for the daemon.
#[derive(Debug, Parser)]
#[command(
    name = "minutist-hub",
    version,
    about = "Minutist headless server — always-on device-sync hub"
)]
struct Cli {
    /// Absolute path to the daemon's own data directory. Its meetings, logs,
    /// device key, and `peers` file all live under this root, entirely separate
    /// from any desktop's `{app-data}` — the single-writer rule applies per data
    /// root, so the daemon must never share a root with another process.
    #[arg(long)]
    data_dir: PathBuf,

    /// The sync relay to pin. Defaults to the connected-tier relay.
    #[arg(long, default_value_t = SyncConfig::DEFAULT_RELAY_URL.to_string())]
    relay_url: String,

    /// The relay access token. Prefer the `MINUTIST_SYNC_TOKEN` environment
    /// variable — a token passed on the command line is visible in the process
    /// list. Never logged (only whether one is set).
    #[arg(long, env = "MINUTIST_SYNC_TOKEN")]
    relay_token: Option<String>,

    /// The account API base URL for account-mediated peer discovery.
    /// Used to publish this device's endpoint and to fetch the account's other
    /// device endpoints (`GET /v1/account/devices`). Only active when a seeded
    /// device credential is present at `{data-dir}/tunnel_device.json`.
    #[arg(long, default_value = DEFAULT_API_URL)]
    relay_api_url: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print this device's pairing ticket to stdout and exit.
    PrintTicket,
    /// Register a peer device by its pairing ticket (append to `{data-dir}/peers`).
    AddPeer {
        /// The peer's pairing ticket, produced by `print-ticket` on that device
        /// (or shown in a desktop's Sync settings).
        ticket: String,
    },
    /// Print the hub's state as JSON (endpoint id, relay, authorised peers, held
    /// meetings + a content digest of each). A read-only oracle for tests.
    Status,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> AppResult<()> {
    let cli = Cli::parse();
    let data_dir = resolve_data_dir(&cli.data_dir)?;

    match cli.command {
        None => run_daemon(&data_dir, cli.relay_url, cli.relay_token, cli.relay_api_url).await,
        Some(Command::PrintTicket) => print_ticket(&data_dir, cli.relay_url, cli.relay_token).await,
        Some(Command::AddPeer { ticket }) => add_peer(&data_dir, &ticket),
        Some(Command::Status) => print_status(&data_dir, cli.relay_url),
    }
}

/// Run the always-on daemon: start the engine, load paired peers, and serve until
/// asked to stop.
async fn run_daemon(
    data_dir: &Path,
    relay_url: String,
    relay_token: Option<String>,
    api_url: String,
) -> AppResult<()> {
    // Hold the tracing-appender worker guard for the whole process lifetime so the
    // non-blocking file writer is flushed on exit.
    let _log_guard = init_tracing(data_dir);

    tracing::info!(
        target: "hub",
        version = env!("CARGO_PKG_VERSION"),
        data_dir = %data_dir.display(),
        relay_url = %relay_url,
        relay_token = if relay_token.is_some() { "<set>" } else { "<unset>" },
        "minutist-hub starting"
    );

    let (engine, account_args) =
        start_engine(data_dir, relay_url, relay_token, api_url).await?;

    tracing::info!(target: "hub", endpoint_id = %engine.endpoint_id(), "sync engine started");
    // Log the pairing ticket so an operator can pair a desktop with this hub
    // (`journalctl` surfaces it). The ticket carries only public addressing.
    tracing::info!(target: "hub", ticket = %engine.my_ticket(), "pairing ticket");

    // Stable readiness marker for an automated harness / `docker logs`: the hub is
    // bound and accepting. Relay homing follows lazily; an inbound dial that beats
    // it simply retries.
    tracing::info!(target: "hub", "minutist-hub ready");

    // Authorise paired peers from the `peers` file and (if seeded) from the account
    // directory; keep re-reading the peers file so an `add-peer` made while the
    // daemon runs is picked up without a restart. Both discovery mechanisms stop
    // on the daemon's shutdown signal inside serve_until_shutdown.
    let mut seen: HashSet<String> = HashSet::new();
    serve_until_shutdown(&engine, data_dir, &mut seen, account_args).await;

    tracing::info!(target: "hub", "shutdown signal received; draining in-flight sync");
    // Bound the drain: a systemd `stop` must see the process exit promptly. If the
    // engine's graceful shutdown stalls (e.g. the relay actor is mid-reconnect),
    // log it and exit anyway — returning from `main` drops the runtime, which
    // aborts any lingering tasks.
    let grace = dur_or_env("MINUTIST_HUB_SHUTDOWN_GRACE_MS", SHUTDOWN_GRACE);
    match tokio::time::timeout(grace, engine.shutdown()).await {
        Ok(Ok(())) => tracing::info!(target: "hub", "minutist-hub stopped"),
        Ok(Err(e)) => {
            tracing::warn!(target: "hub", error = %e, "sync engine shutdown error; exiting")
        }
        Err(_) => tracing::warn!(
            target: "hub",
            grace_ms = grace.as_millis() as u64,
            "sync engine shutdown did not finish within the grace window; exiting"
        ),
    }
    Ok(())
}

/// Print this device's pairing ticket to stdout and exit. No tracing is
/// initialised so stdout carries only the ticket (suitable for scripting); the
/// engine is bound briefly to obtain the addressed ticket, then shut down.
async fn print_ticket(
    data_dir: &Path,
    relay_url: String,
    relay_token: Option<String>,
) -> AppResult<()> {
    // `print-ticket` never runs the account-refresh loop; pass a dummy api_url.
    let (engine, _account_args) =
        start_engine(data_dir, relay_url, relay_token, DEFAULT_API_URL.to_string()).await?;
    let ticket = engine.my_ticket();
    // Command output (the purpose of this subcommand), not logging — println is
    // the right channel here, distinct from the daemon's tracing.
    println!("{ticket}");
    let grace = dur_or_env("MINUTIST_HUB_SHUTDOWN_GRACE_MS", SHUTDOWN_GRACE);
    let _ = tokio::time::timeout(grace, engine.shutdown()).await;
    Ok(())
}

/// Register a peer by its pairing ticket: validate it, then append it to
/// `{data-dir}/peers` (deduplicated). A running daemon picks it up on its next
/// poll; otherwise it is loaded at the next start.
fn add_peer(data_dir: &Path, ticket: &str) -> AppResult<()> {
    let path = sync::peers::peers_path(data_dir);
    match sync::peers::append(data_dir, ticket) {
        Ok(sync::peers::AppendOutcome::Added) => {
            println!("registered peer in {}", path.display());
            Ok(())
        }
        Ok(sync::peers::AppendOutcome::AlreadyPresent) => {
            println!("peer already registered in {}", path.display());
            Ok(())
        }
        // A malformed ticket is bad operator input; keep the InvalidInput variant
        // (the parse guard in `sync::peers::append` surfaces it as `Protocol`).
        Err(sync::Error::Protocol(msg)) => Err(AppError::InvalidInput { context: msg }),
        // Keep the Io variant + the peers-file path so an operator sees WHICH
        // write failed; the blanket `From` would flatten this to `Internal` and
        // drop the path.
        Err(sync::Error::Io(e)) => Err(AppError::Io {
            context: format!("writing peers file {}: {e}", path.display()),
        }),
        Err(e) => Err(AppError::from(e)),
    }
}

/// JSON shape printed by the `status` subcommand.
#[derive(Serialize)]
struct HubStatus {
    /// `None` when `data_dir` has no persisted device identity yet — `status`
    /// reports that state rather than minting one just to fill this field.
    endpoint_id: Option<String>,
    relay_url: String,
    /// Authorised peers (their `EndpointId`s), resolved from the peers file.
    peers: Vec<String>,
    /// Meetings the hub holds, each with a content digest of its notes.
    meetings: Vec<MeetingStatus>,
}

#[derive(Serialize)]
struct MeetingStatus {
    id: String,
    ydoc_present: bool,
    /// sha256 (hex) of the meeting's notes projected to canonical JSON, or null if
    /// it has no `notes.ydoc` yet. Comparable across converged devices.
    digest: Option<String>,
}

/// Print the hub's state as JSON to stdout and exit. A pure filesystem read (no
/// tracing init, no engine bind, and no identity generation; it does NOT contact
/// the running daemon), so an automated harness can use it as an oracle: which
/// peers are authorised and which meetings the hub holds, with a per-meeting
/// notes digest for convergence assertions.
fn print_status(data_dir: &Path, relay_url: String) -> AppResult<()> {
    let status = build_status(data_dir, relay_url)?;
    let json = serde_json::to_string_pretty(&status).map_err(|e| AppError::Internal {
        context: format!("serialising status: {e}"),
    })?;
    println!("{json}");
    Ok(())
}

/// Build the `status` payload without any side effect on `data_dir`. In
/// particular, `endpoint_id` is `None` on a data root with no persisted device
/// identity yet — a fresh root is reported as fresh, never mutated by a `status`
/// call minting (and persisting) a key just to report on it.
fn build_status(data_dir: &Path, relay_url: String) -> AppResult<HubStatus> {
    let endpoint_id = if DeviceIdentity::key_path(data_dir).exists() {
        Some(
            DeviceIdentity::load_or_generate(data_dir)?
                .endpoint_id()
                .to_string(),
        )
    } else {
        None
    };

    let peers: Vec<String> = sync::peers::read_peer_tickets(data_dir)
        .iter()
        .filter_map(|t| t.parse::<EndpointTicket>().ok())
        .map(|t| iroh::EndpointAddr::from(t).id.to_string())
        .collect();

    let meetings_root = data_dir.join("meetings");
    let mut meetings: Vec<MeetingStatus> = notes_crdt::folder::list_meeting_ids(&meetings_root)
        .into_iter()
        .map(|id| {
            let digest = meeting_digest(&meetings_root, id);
            MeetingStatus {
                id: id.0.to_string(),
                ydoc_present: digest.is_some(),
                digest,
            }
        })
        .collect();
    meetings.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(HubStatus {
        endpoint_id,
        relay_url,
        peers,
        meetings,
    })
}

/// sha256 (hex) of a meeting's `notes.ydoc` PROJECTED to canonical JSON — stable
/// across devices that have converged on the same content (the raw CRDT encoding
/// differs by client history; the projection does not). `None` if no `notes.ydoc`.
fn meeting_digest(meetings_root: &Path, meeting: MeetingId) -> Option<String> {
    let state = notes_crdt::NotesStore::read_ydoc_state(meetings_root, meeting).ok()??;
    let doc = notes_crdt::ydoc::new_ydoc();
    notes_crdt::ydoc::apply_update_v1(&doc, &state).ok()?;
    let json = notes_crdt::ydoc::ydoc_to_json(&doc);
    let bytes = serde_json::to_vec(&json).ok()?;
    Some(format!("{:x}", sha2::Sha256::digest(&bytes)))
}

/// Arguments for the account-refresh loop, built by `start_engine` when a seeded
/// device credential is present. Consumed by `serve_until_shutdown` as a third
/// select! arm alongside the peers-file poll and the discovery sweep.
struct AccountRefreshArgs {
    source: Arc<dyn AccountEndpointSource>,
    self_endpoint: AccountEndpoint,
    /// Shared with `serve_until_shutdown`'s shutdown arm so the loop observes the
    /// daemon's single shutdown signal.
    ///
    /// TODO(0029 item 1): migrate this `Arc<Notify>` stop to
    /// `tokio_util::sync::CancellationToken` together with the desktop B4 loop
    /// (`src-tauri/src/sync.rs`) and the `sync-ffi` consumer — all three must
    /// move at once so the API change is atomic.
    stop: Arc<Notify>,
}

/// Build and start the sync engine for `data_dir` against the given relay.
///
/// When a seeded device credential is present at
/// `{data_dir}/tunnel_device.json`, also prepares the `AccountRefreshArgs`
/// needed by the account-mediated peer-discovery loop. If the credential is
/// absent or the account-directory client cannot be built, logs at `info` and
/// returns `None` — the daemon falls back to peers-file pairing only, with no
/// startup failure.
async fn start_engine(
    data_dir: &Path,
    relay_url: String,
    relay_token: Option<String>,
    api_url: String,
) -> AppResult<(SyncEngine, Option<AccountRefreshArgs>)> {
    // Device identity (0600 ed25519 key) at the data root, generated on first run
    // and reloaded thereafter — the stable identity peers pair against.
    let identity = DeviceIdentity::load_or_generate(data_dir)?;

    // The sync engine reads/writes per-meeting folders under `{data_dir}/meetings`.
    let meetings_root = data_dir.join("meetings");
    std::fs::create_dir_all(&meetings_root).map_err(|e| AppError::Io {
        context: format!("creating meetings root {}: {e}", meetings_root.display()),
    })?;

    let mut config = SyncConfig::new(meetings_root);
    config.relay_url = relay_url.clone();
    if let Some(token) = relay_token {
        config = config.with_relay_auth_token(token);
    }

    // Binding opens the QUIC socket and spawns the inbound accept loop; the relay
    // is dialled lazily, so the engine starts even if the relay is momentarily
    // unreachable.
    let engine = SyncEngine::start(config, identity).await?;

    // Account-mediated peer discovery (5.5b / B4): when a seeded device credential
    // is present, build the account-directory source so serve_until_shutdown can run
    // the refresh loop as a third select! arm alongside the peers-file poll and the
    // discovery sweep. When absent, the daemon falls back to peers-file pairing only.
    let account_args = match StoredCredential::load(data_dir) {
        None => {
            tracing::info!(
                target: "hub",
                "no device credential found; account-mediated peer discovery disabled \
                 (seed {CREDENTIAL_FILE} to enable)"
            );
            None
        }
        Some(cred) => {
            match tunnel_client::AccountDirectoryClient::new(&api_url, cred.device_credential) {
                Err(e) => {
                    tracing::warn!(
                        target: "hub",
                        error = %e,
                        api_url = %api_url,
                        "account-directory client not built; skipping account discovery"
                    );
                    None
                }
                Ok(client) => {
                    let source: Arc<dyn AccountEndpointSource> =
                        Arc::new(AccountDirectorySource { client });
                    let self_endpoint = AccountEndpoint {
                        device_id: cred.device_id,
                        endpoint_id: engine.endpoint_id().to_string(),
                        relay_url,
                    };
                    tracing::info!(
                        target: "hub",
                        account_id = %cred.account_id,
                        device_id = %self_endpoint.device_id,
                        endpoint_id = %self_endpoint.endpoint_id,
                        "account credential loaded; account-refresh loop will start"
                    );
                    Some(AccountRefreshArgs {
                        source,
                        self_endpoint,
                        stop: Arc::new(Notify::new()),
                    })
                }
            }
        }
    };

    Ok((engine, account_args))
}

/// Serve until `SIGTERM` / `SIGINT`. The peers file is re-read on a fixed interval
/// (so `add-peer` is honoured without a restart, and the interval's immediate
/// first tick performs the initial load); on each "peer arrived" event the hub
/// reciprocally pushes every meeting it holds to that peer (debounced per peer) —
/// so a device that reconnects both deposits and collects, converging through the
/// hub — and that push rides a lifecycle discovery in the same flow (§7); and a
/// periodic discovery sweep re-advertises so a lifecycle state a consumer dropped
/// or skipped is re-applied (the recovery driver).
///
/// When `account_args` is `Some`, the account-refresh loop runs as a THIRD select!
/// arm alongside the peers-file poll and the discovery sweep. Both discovery
/// mechanisms (peers-file + account-directory) are additive — they feed the same
/// `PeerDirectory` — and both stop on this function's shutdown signal. The loop is
/// NOT detached: running it inline here means it cannot outlive this function and
/// cannot race `engine.shutdown()` after this returns.
///
/// The lifecycle-event CONSUMER runs in a dedicated spawned task
/// ([`apply_lifecycle_events`]), NOT in this select loop: the loop awaits the
/// emitters (`discover_all` and the `push_all_to` ride-along both emit into the
/// same broadcast), so draining in the same loop would let a sweep larger than the
/// channel cap self-lag while the loop is parked on the sweep producing it — a
/// separate drain keeps up concurrently.
async fn serve_until_shutdown(
    engine: &SyncEngine,
    data_dir: &Path,
    seen: &mut HashSet<String>,
    account_args: Option<AccountRefreshArgs>,
) {
    let mut poll = tokio::time::interval(dur_or_env("MINUTIST_HUB_POLL_MS", PEER_POLL_INTERVAL));
    let debounce = dur_or_env("MINUTIST_HUB_PUSH_DEBOUNCE_MS", PEER_PUSH_DEBOUNCE);
    let mut discovery_poll =
        tokio::time::interval(dur_or_env("MINUTIST_HUB_DISCOVERY_MS", DISCOVERY_INTERVAL));
    let mut peer_events = engine.subscribe_peer_events();
    let meetings_root = data_dir.join("meetings");
    let mut last_push: HashMap<_, Instant> = HashMap::new();

    // Drain discovery lifecycle events in a DEDICATED task so the consumer keeps up
    // while this loop is parked awaiting an emitter (discover_all / the push_all_to
    // ride-along emit into the same broadcast). The engine's sender outlives this
    // fn, so the task is aborted on shutdown rather than seeing `Closed` on its own.
    let lifecycle_task = tokio::spawn(apply_lifecycle_events(
        engine.subscribe_lifecycle_events(),
        meetings_root,
    ));

    // Pin the shutdown future ONCE so its signal handlers persist across loop
    // iterations — recreating them each pass could drop a signal that arrives in
    // the gap, hanging `systemctl stop`.
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    // Account-refresh loop: runs as a third select! arm when account credentials are
    // present, so it shares the daemon's single shutdown signal and cannot outlive
    // this function. When absent, resolves to `std::future::pending()` — that arm
    // never fires but also never breaks the select.
    //
    // Pinned once, like `shutdown`, so the arm persists across loop iterations
    // without recreating the future on each pass.
    let account_stop = account_args.as_ref().map(|a| Arc::clone(&a.stop));
    let account_refresh: Pin<Box<dyn Future<Output = ()> + Send + '_>> =
        match account_args {
            Some(args) => Box::pin(run_account_refresh_loop(
                args.source,
                args.self_endpoint,
                ACCOUNT_REFRESH_INTERVAL,
                args.stop,
                move |ep| {
                    if let Err(e) = engine.add_account_peer(&ep.endpoint_id, &ep.relay_url) {
                        tracing::warn!(
                            target: "hub",
                            error = %e,
                            "account peer add rejected"
                        );
                    }
                },
            )),
            None => Box::pin(std::future::pending()),
        };
    tokio::pin!(account_refresh);

    'serve: loop {
        tokio::select! {
            _ = &mut shutdown => {
                // Notify the account-refresh loop's own stop handle so it exits its
                // internal select cleanly rather than being dropped mid-await. The
                // outer select arm drop handles it regardless, but an explicit notify
                // is cleaner and matches the desktop's pattern.
                if let Some(stop) = &account_stop {
                    stop.notify_one();
                }
                break 'serve;
            }
            // Third arm: account-mediated peer discovery. Runs alongside the peers-file
            // poll. Both feed the same PeerDirectory. This arm fires only when the loop
            // exits (stop notified); under normal operation it parks here.
            _ = &mut account_refresh => {}
            _ = poll.tick() => { sync::peers::reload_into(engine, data_dir, seen); },
            _ = discovery_poll.tick() => {
                // Recovery sweep: re-discover every known peer so a lifecycle state
                // a consumer dropped (Lagged) or skipped (a meeting not present when
                // it was advertised) is re-applied. Raced against shutdown, like the
                // push arm. (The first, immediate tick is a no-op before peers load.)
                tokio::select! {
                    _ = &mut shutdown => {
                        if let Some(stop) = &account_stop {
                            stop.notify_one();
                        }
                        break 'serve;
                    }
                    result = engine.discover_all() => match result {
                        Ok(n) => tracing::debug!(target: "hub", peers = n, "periodic discovery swept peers"),
                        Err(e) => tracing::warn!(target: "hub", error = %e, "periodic discovery failed"),
                    },
                }
            }
            arrived = peer_events.recv() => {
                // Map the event to the peers to reconcile. A normal event names one
                // peer. `Lagged` means arrivals were dropped under load (e.g. while a
                // long push ran) — recover by reconciling EVERY known peer, so no
                // arrival is permanently missed. `Closed` means the engine is gone.
                let peers = match arrived {
                    Ok(peer) => vec![peer],
                    Err(RecvError::Lagged(dropped)) => {
                        tracing::warn!(target: "hub", dropped, "peer-event lag; reconciling all known peers");
                        engine.peer_ids()
                    }
                    Err(RecvError::Closed) => break 'serve,
                };
                for peer in peers {
                    let now = Instant::now();
                    let due = last_push
                        .get(&peer)
                        .map(|t| now.duration_since(*t) >= debounce)
                        .unwrap_or(true);
                    if !due {
                        continue;
                    }
                    last_push.insert(peer.clone(), now);
                    // Race the push against shutdown so a SIGTERM mid-push is honoured
                    // promptly: the push future is dropped (iroh closes the connection;
                    // notes writes are atomic and media is content-addressed, so an
                    // abandoned push is safe and idempotent on the next reconcile).
                    tokio::select! {
                        _ = &mut shutdown => {
                            if let Some(stop) = &account_stop {
                                stop.notify_one();
                            }
                            break 'serve;
                        }
                        result = engine.push_all_to_peer(&peer) => match result {
                            Ok(n) => tracing::info!(target: "hub", peer = %peer, meetings = n, "pushed meetings to arrived peer"),
                            Err(e) => tracing::warn!(target: "hub", peer = %peer, error = %e, "push to arrived peer failed"),
                        },
                    }
                }
            }
        }
    }

    // The engine's lifecycle sender outlives this fn (shutdown is graceful, not a
    // drop), so the drain task won't observe `Closed` on its own — abort it.
    lifecycle_task.abort();
}

/// Drain the engine's discovery lifecycle events and persist each, until the
/// broadcast closes. Runs as a DEDICATED task (spawned by [`serve_until_shutdown`])
/// so it drains CONCURRENTLY with the serve loop's discovery/push awaits — which
/// emit into this same broadcast — instead of self-lagging when a re-discovery
/// sweep emits more than the channel cap while the loop is parked on it. Mirrors
/// `ipc_bridge::lifecycle::run_lifecycle_subscriber` (the hub cannot link
/// `ipc-bridge`). `Lagged` is logged; the periodic `discover_all` sweep
/// re-advertises and this drain re-applies — no self-triggered re-discovery.
async fn apply_lifecycle_events(
    mut lifecycle_events: broadcast::Receiver<(MeetingId, ProcessingLifecycle)>,
    meetings_root: PathBuf,
) {
    loop {
        match lifecycle_events.recv().await {
            Ok((meeting_id, processing)) => {
                match persistence::meeting_ops::apply_synced_lifecycle_if_present(
                    &meetings_root,
                    meeting_id,
                    processing,
                )
                .await
                {
                    Ok(true) => {}
                    Ok(false) => tracing::debug!(target: "hub", meeting_id = %meeting_id.0, "synced lifecycle for a meeting not present locally; skipping (re-applied on a later discovery)"),
                    Err(e) => tracing::warn!(target: "hub", meeting_id = %meeting_id.0, error = %e, "failed to apply synced lifecycle"),
                }
            }
            Err(RecvError::Lagged(dropped)) => {
                tracing::warn!(target: "hub", dropped, "lifecycle-event lag; states recover on the periodic discovery sweep");
            }
            Err(RecvError::Closed) => break,
        }
    }
}

/// Resolve when the process is asked to stop: `SIGTERM` (systemd) or `SIGINT`
/// (Ctrl-C) on Unix; Ctrl-C elsewhere.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler at startup");
        let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler at startup");
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Validate and create the daemon's data root. The path MUST be absolute: the
/// daemon resolves identity, settings, logs, and meetings beneath it, and a
/// relative path would resolve against an unpredictable working directory under
/// systemd.
fn resolve_data_dir(p: &Path) -> AppResult<PathBuf> {
    if !p.is_absolute() {
        return Err(AppError::InvalidInput {
            context: format!("--data-dir must be an absolute path, got {}", p.display()),
        });
    }
    std::fs::create_dir_all(p).map_err(|e| AppError::Io {
        context: format!("creating data dir {}: {e}", p.display()),
    })?;
    Ok(p.to_path_buf())
}

/// Initialise tracing for the daemon's own entry point: a rolling file appender
/// under `{data_dir}/logs/` plus a stderr writer (captured by journald under
/// systemd), both honouring `RUST_LOG` and defaulting to `info`. Returns the
/// non-blocking worker guard, which the caller must hold for the process
/// lifetime.
fn init_tracing(data_dir: &Path) -> tracing_appender::non_blocking::WorkerGuard {
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let log_dir = data_dir.join("logs");
    // Best-effort: if the logs dir can't be created we still get stderr logging.
    let _ = std::fs::create_dir_all(&log_dir);

    let file_appender = tracing_appender::rolling::daily(&log_dir, "minutist-hub.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let json = std::env::var("MINUTIST_HUB_LOG_JSON").is_ok_and(|v| v == "1" || v == "true");

    // Opt-in structured (JSON) output so a harness asserts on fields rather than
    // substring-matching the human formatter; default stays human-readable.
    if json {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().json().with_writer(non_blocking))
            .with(fmt::layer().json().with_writer(std::io::stderr))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
            .with(fmt::layer().with_writer(std::io::stderr))
            .init();
    }

    guard
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_data_dir_is_rejected() {
        let err = resolve_data_dir(Path::new("relative/path"))
            .expect_err("a relative data dir must be rejected");
        assert!(matches!(err, AppError::InvalidInput { .. }));
    }

    #[test]
    fn absolute_data_dir_is_created() {
        let base = tempfile::tempdir().expect("tempdir");
        let target = base.path().join("hub-root");
        let resolved = resolve_data_dir(&target).expect("absolute data dir resolves");
        assert_eq!(resolved, target);
        assert!(target.is_dir(), "the data dir must be created");
    }

    // Peers-file parsing (comment/blank skipping, missing-file → empty) is tested
    // in `sync::peers`, the shared implementation this binary now delegates to.

    #[test]
    fn add_peer_rejects_a_malformed_ticket() {
        let base = tempfile::tempdir().expect("tempdir");
        let err = add_peer(base.path(), "not-a-real-ticket")
            .expect_err("a malformed ticket must be rejected");
        assert!(matches!(err, AppError::InvalidInput { .. }));
        // Nothing should have been written.
        assert!(!sync::peers::peers_path(base.path()).exists());
    }

    #[test]
    fn status_helpers_list_and_digest_meetings() {
        let base = tempfile::tempdir().expect("tempdir");
        let root = base.path();
        assert!(notes_crdt::folder::list_meeting_ids(root).is_empty());

        // A meeting with notes: listed, with a deterministic digest.
        let m = MeetingId(uuid::Uuid::new_v4());
        notes_crdt::MeetingFolder::ensure(root, m).expect("ensure folder");
        let json = serde_json::json!({
            "type": "doc",
            "content": [{ "type": "paragraph",
                "content": [{ "type": "text", "text": "status helper test" }] }]
        });
        notes_crdt::NotesStore::save(root, m, &json, "status helper test").expect("save notes");

        assert_eq!(notes_crdt::folder::list_meeting_ids(root), vec![m]);
        let d1 = meeting_digest(root, m).expect("digest present");
        let d2 = meeting_digest(root, m).expect("digest present again");
        assert_eq!(d1, d2, "digest is deterministic for unchanged content");
        assert_eq!(d1.len(), 64, "sha256 hex digest");

        // A meeting folder with no notes.ydoc → no digest.
        let empty = MeetingId(uuid::Uuid::new_v4());
        notes_crdt::MeetingFolder::ensure(root, empty).expect("ensure empty");
        assert_eq!(meeting_digest(root, empty), None);
    }

    #[test]
    fn status_on_a_fresh_root_reports_no_identity_and_does_not_mint_one() {
        let base = tempfile::tempdir().expect("tempdir");
        let dir = base.path();

        let status = build_status(dir, "wss://relay.example".to_string()).expect("build status");
        assert_eq!(
            status.endpoint_id, None,
            "a fresh root has no identity to report"
        );
        assert!(
            !DeviceIdentity::key_path(dir).exists(),
            "status must not mint (and persist) a device key just to report on it"
        );
    }

    #[test]
    fn status_reports_the_identity_once_one_exists() {
        let base = tempfile::tempdir().expect("tempdir");
        let dir = base.path();
        let identity = DeviceIdentity::load_or_generate(dir).expect("generate identity");

        let status = build_status(dir, "wss://relay.example".to_string()).expect("build status");
        assert_eq!(
            status.endpoint_id.as_deref(),
            Some(identity.endpoint_id().to_string().as_str()),
            "status must report the pre-existing identity"
        );
    }

    #[test]
    fn stored_credential_load_returns_none_when_file_is_absent() {
        let base = tempfile::tempdir().expect("tempdir");
        let result = StoredCredential::load(base.path());
        assert_eq!(result, None, "missing file returns None");
    }

    #[test]
    fn stored_credential_load_parses_present_valid_json() {
        let base = tempfile::tempdir().expect("tempdir");
        let cred_path = base.path().join(CREDENTIAL_FILE);
        let json = serde_json::json!({
            "device_credential": "mdc_test_credential",
            "account_id": "acct_12345",
            "device_id": "dev_67890"
        });
        std::fs::write(&cred_path, serde_json::to_string(&json).expect("serialize")).expect("write");

        let result = StoredCredential::load(base.path()).expect("load credential");
        assert_eq!(result.device_credential, "mdc_test_credential");
        assert_eq!(result.account_id, "acct_12345");
        assert_eq!(result.device_id, "dev_67890");
    }

    #[test]
    fn stored_credential_load_returns_none_when_device_credential_is_empty() {
        let base = tempfile::tempdir().expect("tempdir");
        let cred_path = base.path().join(CREDENTIAL_FILE);
        let json = serde_json::json!({
            "device_credential": "",
            "account_id": "acct_12345",
            "device_id": "dev_67890"
        });
        std::fs::write(&cred_path, serde_json::to_string(&json).expect("serialize")).expect("write");

        let result = StoredCredential::load(base.path());
        assert_eq!(result, None, "empty device_credential returns None");
    }

    #[test]
    fn stored_credential_load_returns_none_when_json_is_corrupt() {
        let base = tempfile::tempdir().expect("tempdir");
        let cred_path = base.path().join(CREDENTIAL_FILE);
        std::fs::write(&cred_path, "not valid json {").expect("write");

        let result = StoredCredential::load(base.path());
        assert_eq!(result, None, "corrupt JSON returns None");
    }
}
