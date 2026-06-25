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
//!
//! The data root is entirely separate from any desktop's `{app-data}` — the
//! single-writer rule applies per data root, so the daemon must never share a root
//! with another process.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use iroh_tickets::endpoint::EndpointTicket;
use minutist_common::{AppError, AppResult};
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use tokio::sync::broadcast::error::RecvError;

/// How long to wait for the sync engine to drain before forcing exit on shutdown.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// How often the running daemon re-reads `{data-dir}/peers` so an `add-peer` made
/// while it is running is honoured without a restart.
const PEER_POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Minimum gap between reciprocal pushes to the same peer, so a peer that
/// reconnects rapidly is not re-pushed on every connection.
const PEER_PUSH_DEBOUNCE: Duration = Duration::from_secs(15);

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
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> AppResult<()> {
    let cli = Cli::parse();
    let data_dir = resolve_data_dir(&cli.data_dir)?;

    match cli.command {
        None => run_daemon(&data_dir, cli.relay_url, cli.relay_token).await,
        Some(Command::PrintTicket) => print_ticket(&data_dir, cli.relay_url, cli.relay_token).await,
        Some(Command::AddPeer { ticket }) => add_peer(&data_dir, &ticket),
    }
}

/// Run the always-on daemon: start the engine, load paired peers, and serve until
/// asked to stop.
async fn run_daemon(
    data_dir: &Path,
    relay_url: String,
    relay_token: Option<String>,
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

    let engine = start_engine(data_dir, relay_url, relay_token).await?;

    tracing::info!(target: "hub", endpoint_id = %engine.endpoint_id(), "sync engine started");
    // Log the pairing ticket so an operator can pair a desktop with this hub
    // (`journalctl` surfaces it). The ticket carries only public addressing.
    tracing::info!(target: "hub", ticket = %engine.my_ticket(), "pairing ticket");

    // Authorise paired peers from the `peers` file, then keep re-reading it so an
    // `add-peer` made while the daemon runs is picked up without a restart.
    let mut seen: HashSet<String> = HashSet::new();
    serve_until_shutdown(&engine, data_dir, &mut seen).await;

    tracing::info!(target: "hub", "shutdown signal received; draining in-flight sync");
    // Bound the drain: a systemd `stop` must see the process exit promptly. If the
    // engine's graceful shutdown stalls (e.g. the relay actor is mid-reconnect),
    // log it and exit anyway — returning from `main` drops the runtime, which
    // aborts any lingering tasks.
    match tokio::time::timeout(SHUTDOWN_GRACE, engine.shutdown()).await {
        Ok(Ok(())) => tracing::info!(target: "hub", "minutist-hub stopped"),
        Ok(Err(e)) => {
            tracing::warn!(target: "hub", error = %e, "sync engine shutdown error; exiting")
        }
        Err(_) => tracing::warn!(
            target: "hub",
            grace_secs = SHUTDOWN_GRACE.as_secs(),
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
    let engine = start_engine(data_dir, relay_url, relay_token).await?;
    let ticket = engine.my_ticket();
    // Command output (the purpose of this subcommand), not logging — println is
    // the right channel here, distinct from the daemon's tracing.
    println!("{ticket}");
    let _ = tokio::time::timeout(SHUTDOWN_GRACE, engine.shutdown()).await;
    Ok(())
}

/// Register a peer by its pairing ticket: validate it, then append it to
/// `{data-dir}/peers` (deduplicated). A running daemon picks it up on its next
/// poll; otherwise it is loaded at the next start.
fn add_peer(data_dir: &Path, ticket: &str) -> AppResult<()> {
    let ticket = ticket.trim();
    // Fail fast on a malformed ticket rather than writing garbage the daemon will
    // later reject.
    ticket
        .parse::<EndpointTicket>()
        .map_err(|e| AppError::InvalidInput {
            context: format!("not a valid pairing ticket: {e}"),
        })?;

    let path = peers_path(data_dir);
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    if existing.lines().any(|line| line.trim() == ticket) {
        println!("peer already registered in {}", path.display());
        return Ok(());
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| AppError::Io {
            context: format!("opening peers file {}: {e}", path.display()),
        })?;
    writeln!(file, "{ticket}").map_err(|e| AppError::Io {
        context: format!("writing peers file {}: {e}", path.display()),
    })?;
    println!("registered peer in {}", path.display());
    Ok(())
}

/// Build and start the sync engine for `data_dir` against the given relay.
async fn start_engine(
    data_dir: &Path,
    relay_url: String,
    relay_token: Option<String>,
) -> AppResult<SyncEngine> {
    // Device identity (0600 ed25519 key) at the data root, generated on first run
    // and reloaded thereafter — the stable identity peers pair against.
    let identity = DeviceIdentity::load_or_generate(data_dir)?;

    // The sync engine reads/writes per-meeting folders under `{data_dir}/meetings`.
    let meetings_root = data_dir.join("meetings");
    std::fs::create_dir_all(&meetings_root).map_err(|e| AppError::Io {
        context: format!("creating meetings root {}: {e}", meetings_root.display()),
    })?;

    let mut config = SyncConfig::new(meetings_root);
    config.relay_url = relay_url;
    if let Some(token) = relay_token {
        config = config.with_relay_auth_token(token);
    }

    // Binding opens the QUIC socket and spawns the inbound accept loop; the relay
    // is dialled lazily, so the engine starts even if the relay is momentarily
    // unreachable.
    Ok(SyncEngine::start(config, identity).await?)
}

/// Serve until `SIGTERM` / `SIGINT`. Two background behaviours run off one select
/// loop: the peers file is re-read on a fixed interval (so `add-peer` is honoured
/// without a restart, and the interval's immediate first tick performs the initial
/// load), and on each "peer arrived" event the hub reciprocally pushes every
/// meeting it holds to that peer (debounced per peer) — so a device that
/// reconnects both deposits and collects, converging through the hub.
async fn serve_until_shutdown(engine: &SyncEngine, data_dir: &Path, seen: &mut HashSet<String>) {
    let mut poll = tokio::time::interval(PEER_POLL_INTERVAL);
    let mut peer_events = engine.subscribe_peer_events();
    let mut last_push: HashMap<_, Instant> = HashMap::new();

    // Pin the shutdown future ONCE so its signal handlers persist across loop
    // iterations — recreating them each pass could drop a signal that arrives in
    // the gap, hanging `systemctl stop`.
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    'serve: loop {
        tokio::select! {
            _ = &mut shutdown => break 'serve,
            _ = poll.tick() => reload_peers(engine, data_dir, seen),
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
                        .map(|t| now.duration_since(*t) >= PEER_PUSH_DEBOUNCE)
                        .unwrap_or(true);
                    if !due {
                        continue;
                    }
                    last_push.insert(peer, now);
                    // Race the push against shutdown so a SIGTERM mid-push is honoured
                    // promptly: the push future is dropped (iroh closes the connection;
                    // notes writes are atomic and media is content-addressed, so an
                    // abandoned push is safe and idempotent on the next reconcile).
                    tokio::select! {
                        _ = &mut shutdown => break 'serve,
                        result = engine.push_all_to(peer) => match result {
                            Ok(n) => tracing::info!(target: "hub", peer = %peer, meetings = n, "pushed meetings to arrived peer"),
                            Err(e) => tracing::warn!(target: "hub", peer = %peer, error = %e, "push to arrived peer failed"),
                        },
                    }
                }
            }
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

/// Read the peers file and authorise any ticket not already applied this run.
/// Malformed lines are logged and skipped (and marked seen so they are not
/// re-warned each poll); the file's absence is normal (no peers paired yet).
fn reload_peers(engine: &SyncEngine, data_dir: &Path, seen: &mut HashSet<String>) {
    for ticket in read_peer_tickets(data_dir) {
        if !seen.insert(ticket.clone()) {
            continue;
        }
        match engine.add_peer_from_ticket(&ticket) {
            Ok(id) => tracing::info!(target: "hub", peer = %id, "authorised paired peer"),
            Err(e) => {
                tracing::warn!(target: "hub", error = %e, "skipping malformed peer ticket")
            }
        }
    }
}

/// Parse `{data-dir}/peers`: one pairing ticket per line; blank lines and
/// `#`-prefixed comments are ignored. A missing file yields an empty list.
fn read_peer_tickets(data_dir: &Path) -> Vec<String> {
    let contents = match std::fs::read_to_string(peers_path(data_dir)) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

/// The peers file path under the data root.
fn peers_path(data_dir: &Path) -> PathBuf {
    data_dir.join("peers")
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
    let file_layer = fmt::layer().with_writer(non_blocking).with_ansi(false);
    let stderr_layer = fmt::layer().with_writer(std::io::stderr);

    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(stderr_layer)
        .init();

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

    #[test]
    fn peers_file_parsing_skips_comments_and_blanks() {
        let base = tempfile::tempdir().expect("tempdir");
        let dir = base.path();
        std::fs::write(
            peers_path(dir),
            "# a comment\n\n  ticket-one  \nticket-two\n   \n# trailing\n",
        )
        .expect("write peers file");
        assert_eq!(
            read_peer_tickets(dir),
            vec!["ticket-one".to_string(), "ticket-two".to_string()]
        );
    }

    #[test]
    fn missing_peers_file_is_empty() {
        let base = tempfile::tempdir().expect("tempdir");
        assert!(read_peer_tickets(base.path()).is_empty());
    }

    #[test]
    fn add_peer_rejects_a_malformed_ticket() {
        let base = tempfile::tempdir().expect("tempdir");
        let err = add_peer(base.path(), "not-a-real-ticket")
            .expect_err("a malformed ticket must be rejected");
        assert!(matches!(err, AppError::InvalidInput { .. }));
        // Nothing should have been written.
        assert!(!peers_path(base.path()).exists());
    }
}
