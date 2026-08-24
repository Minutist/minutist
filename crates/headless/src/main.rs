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
//!   engine bound to the relay, and run until `SIGTERM` / `SIGINT`. Peers are
//!   discovered exclusively via the signed-in account (see `login` below) — an
//!   unauthenticated hub binds and serves but discovers no peers at all.
//! - `login` — sign this hub in to a Minutist account via the same RFC 8628
//!   device-code flow the desktop uses: prints a URL (and a code, when the
//!   server doesn't pre-fill it) to open in any browser, polls until approved,
//!   and persists the issued credential to `{data-dir}/tunnel_device.json`
//!   (0600). This is now the ONLY way a hub gets peers — there is no manual
//!   ticket exchange any more (removed; it existed only as an interim
//!   mechanism before this flow was built).
//! - `status` — print the hub's state as JSON to stdout and exit: endpoint id,
//!   whether it's signed in (and to which account), and held meetings each
//!   with a content digest of their notes. A read-only filesystem oracle for
//!   automated tests (it does not contact the running daemon).
//!
//! Timing constants (poll / push-debounce / shutdown-grace) are overridable via
//! env vars in milliseconds (a sub-second test mode) — see the constants below.
//! `MINUTIST_HUB_LOG_JSON=1` switches tracing to a structured JSON formatter.
//!
//! The data root is entirely separate from any desktop's `{app-data}` — the
//! single-writer rule applies per data root, so the daemon must never share a root
//! with another process.
//!
//! ## Dependency shape
//!
//! `minutist-hub` is a SECOND workspace binary beside `app-main` (`src-tauri`):
//! a standalone `cargo build` target with no Tauri/webview pipeline, so it
//! cross-compiles to a server target and packages as a systemd unit or a
//! minimal OCI image. There is no `app-main -> headless` edge in either
//! direction and no shared code path. Its dependencies are `common`,
//! `persistence`, `notes-crdt`, `sync`, `tunnel-client`, and
//! `account-directory` (the same `AccountDirectorySource` adapter `app-main`
//! uses, so neither binary carries its own copy — see `crates/account-directory`).
//! It takes no `tauri::*` / `ipc-bridge` edge: it wires [`sync::SyncEngine`]
//! directly and carries no command/event surface. `tunnel-client` backs the
//! account-mediated peer-discovery loop (`AccountDirectoryClient` publishes
//! this daemon's endpoint and fetches the account's device list); a seeded
//! `minutist-hub` is always account-capable, so unlike `app-main`'s `connected`
//! Cargo feature, this crate's workspace membership and its edges to `sync` /
//! `tunnel-client` / `account-directory` are unconditional, not feature-gated.
//! A post-launch GPU processing-node role adds `orchestrator` and the
//! ML-runtime crates (`asr-runtime`, `asr-parakeet`, `diarizer`, `summariser`,
//! `model-registry`).

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use minutist_common::{AppError, AppResult, DeletionState, MeetingId, ProcessingLifecycle};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use sync::{
    run_account_refresh_loop_v2, AccountEndpoint, AccountEndpointSource, DeviceIdentity,
    RefreshSink, SyncConfig, SyncEngine, SyncEngineRefreshSink,
};
use tokio::sync::broadcast::{self, error::RecvError};
use tokio_util::sync::CancellationToken;

/// Default shutdown drain window; override `MINUTIST_HUB_SHUTDOWN_GRACE_MS`.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// Default minimum gap between reciprocal pushes to the same peer (so a rapidly
/// reconnecting peer is not re-pushed every time); override
/// `MINUTIST_HUB_PUSH_DEBOUNCE_MS`.
const PEER_PUSH_DEBOUNCE: Duration = Duration::from_secs(15);

/// Default interval for the recovery discovery sweep — a periodic re-discovery of
/// every known peer so a lifecycle state a consumer dropped (`Lagged`) or skipped
/// (advertised before the meeting's folder had synced in) is eventually
/// re-applied; override `MINUTIST_HUB_DISCOVERY_MS`.
const DISCOVERY_INTERVAL: Duration = Duration::from_secs(300);

/// Default interval for the trash auto-purge sweep (7-day TTL) — a hub-run
/// meeting can age out with no desktop open; override
/// `MINUTIST_HUB_TRASH_SWEEP_MS`. Mirrors the desktop's hourly wiring in
/// `src-tauri/src/main.rs`.
const TRASH_SWEEP_INTERVAL: Duration = Duration::from_secs(3600);

/// Interval between account-directory refreshes (fetch peer list + re-register
/// endpoint). Same as the desktop's B4 wiring in `src-tauri/src/sync.rs`.
const ACCOUNT_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// The default account API base URL. Overridable via `--relay-api-url`.
const DEFAULT_API_URL: &str = "https://api.minutist.ai";

/// The filename under `{data-dir}` that holds the device credential issued by
/// `login`. Also accepted if seeded directly by an operator or a test harness
/// (the JSON shape is a stable contract — see [`StoredCredential`]).
const CREDENTIAL_FILE: &str = "tunnel_device.json";

/// The persisted device credential for account-mediated peer discovery
/// (`{data-dir}/tunnel_device.json`), written by [`login`] on a successful
/// device-code pairing (or seeded directly — e.g. by a test harness — since
/// the three-field JSON shape is the same file the desktop's `app-main`
/// writes).
///
/// `device_credential` is the long-lived `mdc_` bearer — never logged.
#[derive(Serialize, Deserialize, PartialEq)]
struct StoredCredential {
    device_credential: String,
    account_id: String,
    device_id: String,
}

impl std::fmt::Debug for StoredCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redact the bearer; mirrors `sync::SyncConfig`'s hand-rolled Debug.
        f.debug_struct("StoredCredential")
            .field("device_credential", &"<redacted>")
            .field("account_id", &self.account_id)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl StoredCredential {
    /// Load the stored credential, or `None` when absent / unreadable / corrupt.
    /// A missing file is the normal not-signed-in case, logged as such; a
    /// present-but-unreadable file (permission denied is the practical case —
    /// e.g. `login` was run as the wrong user against a fixed-user systemd
    /// deployment) is distinct enough to warn about explicitly, since it means
    /// the operator DID provision a credential and it is silently not being
    /// used. Both, like a corrupt file, leave the hub running with no peer
    /// discovery at all — no panic, no startup failure.
    fn load(data_dir: &Path) -> Option<Self> {
        let raw = match std::fs::read_to_string(data_dir.join(CREDENTIAL_FILE)) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => {
                tracing::warn!(
                    target: "hub",
                    error = %e,
                    "tunnel_device.json exists but could not be read; skipping account-mediated discovery"
                );
                return None;
            }
        };
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

    /// Persist the credential atomically: written to a sibling temp file, then
    /// renamed over `{data-dir}/tunnel_device.json` (the rename is the sole
    /// commit point, so a crash or `ENOSPC` mid-write never leaves a
    /// truncated or partial file readable as valid JSON). On unix the temp
    /// file — and so the renamed target, since rename preserves the source's
    /// mode — is created `0600`.
    ///
    /// On other platforms the file carries no per-file ACL of its own: the
    /// only real protection is restricting `{data-dir}`'s own ACL, which is
    /// the installer's job (`packaging/windows/install-service.ps1` locks
    /// `%ProgramData%\minutist-hub` down to SYSTEM/Administrators before the
    /// service ever runs) — the same reliance `app-main`'s `write_secret_file`
    /// has on the desktop's per-user `AppData` rather than a per-file mode.
    fn store(&self, data_dir: &Path) -> std::io::Result<()> {
        use std::io::Write;

        let json = serde_json::to_string(self).map_err(std::io::Error::other)?;
        let path = data_dir.join(CREDENTIAL_FILE);
        let tmp_path = data_dir.join(format!("{CREDENTIAL_FILE}.{}.tmp", uuid::Uuid::new_v4()));

        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let write_result = (|| -> std::io::Result<()> {
            let mut file = opts.open(&tmp_path)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            }
            file.write_all(json.as_bytes())?;
            file.flush()?;
            file.sync_all()
        })();

        if let Err(e) = write_result {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
        std::fs::rename(&tmp_path, &path)
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
#[derive(Parser)]
#[command(
    name = "minutist-hub",
    version,
    about = "Minutist headless server — always-on device-sync hub"
)]
struct Cli {
    /// Absolute path to the daemon's own data directory. Its meetings, logs,
    /// device key, and account credential (`tunnel_device.json`) all live under
    /// this root, entirely separate from any desktop's `{app-data}` — the
    /// single-writer rule applies per data root, so the daemon must never share
    /// a root with another process.
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
    #[arg(long, env = "MINUTIST_HUB_API_URL", default_value = DEFAULT_API_URL)]
    relay_api_url: String,

    #[command(subcommand)]
    command: Option<Command>,
}

impl std::fmt::Debug for Cli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redact the relay token; mirrors `sync::SyncConfig`'s hand-rolled Debug.
        f.debug_struct("Cli")
            .field("data_dir", &self.data_dir)
            .field("relay_url", &self.relay_url)
            .field("relay_token", &self.relay_token.as_ref().map(|_| "<redacted>"))
            .field("relay_api_url", &self.relay_api_url)
            .field("command", &self.command)
            .finish()
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Sign this hub in to a Minutist account (RFC 8628 device-code flow):
    /// prints a URL to open in any browser, polls until approved, and
    /// persists the issued credential to `{data-dir}/tunnel_device.json`.
    Login,
    /// Print the hub's state as JSON (endpoint id, sign-in status, held
    /// meetings + a content digest of each). A read-only oracle for tests.
    Status,
    /// Originate a new meeting in the hub's data directory and print its UUID to
    /// stdout. Used by the e2e harness to seed a meeting on device-A so the hub
    /// can push it to device-B.
    CreateMeeting {
        /// Human-readable title for the new meeting.
        #[arg(long)]
        title: String,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> AppResult<()> {
    let cli = Cli::parse();
    let data_dir = resolve_data_dir(&cli.data_dir)?;

    match cli.command {
        None => run_daemon(&data_dir, cli.relay_url, cli.relay_token, cli.relay_api_url).await,
        Some(Command::Login) => login(&data_dir, &cli.relay_api_url).await,
        Some(Command::Status) => print_status(&data_dir, cli.relay_url),
        Some(Command::CreateMeeting { title }) => create_meeting(&data_dir, &title),
    }
}

/// Run the always-on daemon: start the engine, start account-mediated peer
/// discovery if signed in, and serve until asked to stop.
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
    // Shared with the account-refresh loop's `SyncEngineRefreshSink`, which needs an
    // owned `Arc<SyncEngine>`. Sole ownership is reclaimed below for the consuming
    // `SyncEngine::shutdown(self)` once `serve_until_shutdown` returns and drops the
    // sink.
    let engine = Arc::new(engine);

    tracing::info!(target: "hub", endpoint_id = %engine.endpoint_id(), "sync engine started");
    if account_args.is_none() {
        // `start_engine` already logged the specific reason (no credential file,
        // a corrupt one, or an account-directory client that failed to build) —
        // don't restate a guessed cause here, since telling an operator who IS
        // signed in to "run login" again would be actively misleading.
        tracing::warn!(
            target: "hub",
            "account-mediated peer discovery is not active; see the preceding log line for why"
        );
    }

    // Stable readiness marker for an automated harness / `docker logs`: the hub is
    // bound and accepting. Relay homing follows lazily; an inbound dial that beats
    // it simply retries.
    tracing::info!(target: "hub", "minutist-hub ready");

    // Peers are discovered exclusively via the account-directory loop (when
    // signed in); it stops on the daemon's shutdown signal inside
    // serve_until_shutdown.
    serve_until_shutdown(Arc::clone(&engine), data_dir, account_args).await;

    tracing::info!(target: "hub", "shutdown signal received; draining in-flight sync");
    // Bound the drain: a systemd `stop` must see the process exit promptly. If the
    // engine's graceful shutdown stalls (e.g. the relay actor is mid-reconnect),
    // log it and exit anyway — returning from `main` drops the runtime, which
    // aborts any lingering tasks.
    let grace = dur_or_env("MINUTIST_HUB_SHUTDOWN_GRACE_MS", SHUTDOWN_GRACE);
    // Reclaim sole ownership for the consuming `shutdown(self)`. The only other clone
    // lived in the refresh-loop future, dropped when `serve_until_shutdown` returned,
    // so this is `Some` in practice; a lingering clone would mean a task still holds
    // the engine, so skip the graceful drain (returning drops the runtime and aborts
    // it anyway).
    let engine = match Arc::into_inner(engine) {
        Some(engine) => engine,
        None => {
            tracing::warn!(
                target: "hub",
                "sync engine still referenced at shutdown; skipping graceful drain"
            );
            return Ok(());
        }
    };
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

/// Sign this hub in to a Minutist account via the RFC 8628 device-code flow —
/// the same protocol `src-tauri/src/account.rs` drives for the desktop, here
/// driven as a blocking CLI loop instead of a polled IPC command (there is no
/// UI to poll from). Prints the URL to open (and the code, only when the
/// server didn't pre-fill it — see `PairingStart::open_url` /
/// `code_required`'s reasoning on the desktop side) and blocks until the user
/// approves, is declined, or the code expires. On success, persists the
/// credential and exits 0; the next `run_daemon` picks it up.
async fn login(data_dir: &Path, api_url: &str) -> AppResult<()> {
    let client = tunnel_client::DeviceCodeClient::new(api_url.to_string())
        .map_err(map_pairing_err)?;
    let start = client
        .start(Some("Minutist hub"))
        .await
        .map_err(map_pairing_err)?;

    println!("Open this URL to sign in: {}", start.open_url());
    if start.verification_uri_complete.is_none() {
        println!("Then enter this code: {}", start.user_code);
    }
    println!("This code expires in {} minute(s).", start.expires_in.div_ceil(60));

    // Enforced client-side too, not just left to the server's own
    // `expired_token` response: a CLI run unattended (a provisioning script,
    // `docker exec`) should not be able to block forever on a server that
    // answers `pending` past its own stated expiry.
    let deadline = Instant::now() + Duration::from_secs(start.expires_in);
    let mut interval = start.initial_interval();
    loop {
        match client.poll_once(&start.device_code).await {
            Ok(tunnel_client::PollOutcome::Pending) => {}
            Ok(tunnel_client::PollOutcome::SlowDown) => {
                interval = tunnel_client::next_interval(interval);
            }
            Ok(tunnel_client::PollOutcome::Authorised(issued)) => {
                let credential = StoredCredential {
                    device_credential: issued.device_credential,
                    account_id: issued.account_id,
                    device_id: issued.device_id,
                };
                credential.store(data_dir).map_err(|e| AppError::Io {
                    context: format!("writing {CREDENTIAL_FILE}: {e}"),
                })?;
                println!("Signed in as account {}.", credential.account_id);
                return Ok(());
            }
            Err(e) => return Err(map_pairing_err(e)),
        }
        if Instant::now() >= deadline {
            return Err(AppError::InvalidInput {
                context: "the pairing code expired before it was approved; run login again"
                    .to_string(),
            });
        }
        tokio::time::sleep(interval).await;
    }
}

/// Map a `tunnel-client` pairing error to the CLI's error type. Mirrors
/// `src-tauri/src/account.rs::map_pairing_err` (duplicated rather than shared
/// — `headless` and `src-tauri` are separate binaries with no edge between
/// them, and the mapping is a few lines).
fn map_pairing_err(e: tunnel_client::PairingError) -> AppError {
    use tunnel_client::PairingError;
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
            context: "the pairing code expired; run login again".to_string(),
        },
        PairingError::AccessDenied => AppError::InvalidInput {
            context: "pairing was declined".to_string(),
        },
        PairingError::MalformedAuthorisation => AppError::Internal {
            context: "the account service returned a malformed authorisation".to_string(),
        },
    }
}

/// Originate a new meeting in `{data_dir}/meetings/`: create the folder,
/// seed placeholder metadata, write the first `notes.ydoc`, and print the
/// meeting UUID to stdout. The UUID is the only output — suitable for capture
/// by a harness driver. No daemon interaction; purely a filesystem write.
fn create_meeting(data_dir: &Path, title: &str) -> AppResult<()> {
    let meetings_root = data_dir.join("meetings");
    let id = minutist_common::MeetingId::new();
    notes_crdt::MeetingFolder::ensure(&meetings_root, id)?;
    let notes_json = serde_json::json!({
        "type": "doc",
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": title}]}]
    });
    notes_crdt::NotesStore::save(&meetings_root, id, &notes_json, title)?;
    // Command output (the UUID captured by the e2e runner), not logging.
    println!("{}", id.0);
    Ok(())
}

/// JSON shape printed by the `status` subcommand.
#[derive(Serialize)]
struct HubStatus {
    /// `None` when `data_dir` has no persisted device identity yet — `status`
    /// reports that state rather than minting one just to fill this field.
    endpoint_id: Option<String>,
    relay_url: String,
    /// The signed-in account, or `None` when `login` has not been run (or the
    /// stored credential is corrupt) — the hub then discovers no peers at all.
    account_id: Option<String>,
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
/// the running daemon), so an automated harness can use it as an oracle:
/// sign-in state and which meetings the hub holds, with a per-meeting notes
/// digest for convergence assertions.
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

    let account_id = StoredCredential::load(data_dir).map(|c| c.account_id);

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
        account_id,
        meetings,
    })
}

/// sha256 (hex) of a meeting's `notes.ydoc` PROJECTED to canonical JSON — stable
/// across devices that have converged on the same content (the raw CRDT encoding
/// differs by client history; the projection does not). `None` if no `notes.ydoc`.
/// Snapshot the purged-tombstone set into memory and return a closure over
/// it, for `adopt_all`/`adopt_from_peer`'s `is_purged` check.
///
/// A sweep checks many candidate ids against one snapshot, so reading
/// `purged.json` once per sweep call (here) and checking the in-memory set
/// per id is one file read instead of one per candidate id via repeated
/// `PurgedStore::is_purged` calls. A tombstone-read failure fails OPEN (an
/// empty set, so nothing is treated as purged) — a rare spurious
/// resurrection is far cheaper than permanently starving a legitimate adopt
/// on a transient read error.
fn purged_lookup(data_dir: &Path) -> impl Fn(MeetingId) -> bool + Sync {
    let ids = persistence::purged::PurgedStore::purged_ids(data_dir).unwrap_or_default();
    move |id| ids.contains(&id)
}

fn meeting_digest(meetings_root: &Path, meeting: MeetingId) -> Option<String> {
    let state = notes_crdt::NotesStore::read_ydoc_state(meetings_root, meeting).ok()??;
    let doc = notes_crdt::ydoc::new_ydoc();
    notes_crdt::ydoc::apply_update_v1(&doc, &state).ok()?;
    let json = notes_crdt::ydoc::ydoc_to_json(&doc);
    let bytes = serde_json::to_vec(&json).ok()?;
    Some(format!("{:x}", sha2::Sha256::digest(&bytes)))
}

/// Arguments for the account-refresh loop, built by `start_engine` when a seeded
/// device credential is present. Consumed by `serve_until_shutdown`, which drives
/// [`run_account_refresh_loop_v2`] as a select! arm alongside the discovery sweep,
/// cancelling it on the daemon's shutdown signal.
struct AccountRefreshArgs {
    source: Arc<dyn AccountEndpointSource>,
    self_endpoint: AccountEndpoint,
}

/// Binds the sync engine, trusting the relay's TLS certificate unconditionally
/// when `MINUTIST_HUB_INSECURE_RELAY_TLS` is set instead of verifying it against
/// the system CA roots.
///
/// Exists solely for `hub_e2e`'s in-process test relay
/// (`iroh::test_utils::run_relay_server`), whose self-signed certificate no CA
/// recognises. Gated behind `test-support`: this whole function — env var
/// included — compiles only in a test build, so the escape hatch cannot exist
/// in a shipped binary regardless of the environment it runs in.
#[cfg(feature = "test-support")]
async fn bind_sync_engine(
    config: SyncConfig,
    identity: DeviceIdentity,
) -> sync::Result<SyncEngine> {
    if std::env::var_os("MINUTIST_HUB_INSECURE_RELAY_TLS").is_some() {
        SyncEngine::start_insecure(config, identity).await
    } else {
        SyncEngine::start(config, identity).await
    }
}

/// The production path: always verifies the relay's TLS certificate.
#[cfg(not(feature = "test-support"))]
async fn bind_sync_engine(
    config: SyncConfig,
    identity: DeviceIdentity,
) -> sync::Result<SyncEngine> {
    SyncEngine::start(config, identity).await
}

/// Adds every ticket in `MINUTIST_HUB_TEST_PEERS` (comma-separated) directly to
/// the engine's peer directory via [`SyncEngine::add_peer_from_ticket`] — the
/// same primitive `sync-ffi`'s manual pairing calls, here driven from an env var
/// instead of a UI.
///
/// Exists solely for `hub_e2e`'s local, account-free runs, which have no
/// account service to discover peers through. Gated behind `test-support`: this
/// whole function — env var included — compiles only in a test build, so it
/// cannot exist in a shipped binary regardless of the environment it runs in.
#[cfg(feature = "test-support")]
fn seed_test_peers(engine: &SyncEngine) {
    let Some(raw) = std::env::var_os("MINUTIST_HUB_TEST_PEERS") else {
        return;
    };
    for ticket in raw
        .to_string_lossy()
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        if let Err(e) = engine.add_peer_from_ticket(ticket) {
            tracing::warn!(target: "hub", error = %e, "MINUTIST_HUB_TEST_PEERS: invalid ticket");
        }
    }
}

#[cfg(not(feature = "test-support"))]
fn seed_test_peers(_engine: &SyncEngine) {}

/// Build and start the sync engine for `data_dir` against the given relay.
///
/// When a seeded device credential is present at
/// `{data_dir}/tunnel_device.json`, also prepares the `AccountRefreshArgs`
/// needed by the account-mediated peer-discovery loop. If the credential is
/// absent or the account-directory client cannot be built, logs at `info` and
/// returns `None` — the daemon then discovers no peers at all until
/// `minutist-hub login` provisions one, with no startup failure.
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
    let engine = bind_sync_engine(config, identity).await?;
    seed_test_peers(&engine);

    // Account-mediated peer discovery (5.5b / B4): when a seeded device credential
    // is present, build the account-directory source so serve_until_shutdown can run
    // the refresh loop as a select! arm alongside the discovery sweep. When absent,
    // the daemon discovers no peers at all.
    let account_args = match StoredCredential::load(data_dir) {
        None => {
            tracing::info!(
                target: "hub",
                "no device credential found; account-mediated peer discovery disabled \
                 (run `minutist-hub login` to enable)"
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
                        Arc::new(account_directory::AccountDirectorySource::new(client));
                    let self_endpoint = AccountEndpoint {
                        device_id: cred.device_id,
                        endpoint_id: engine.endpoint_id().to_string(),
                        relay_url,
                        direct_addrs: engine.publishable_direct_addrs(),
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
                    })
                }
            }
        }
    };

    Ok((engine, account_args))
}

/// Serve until `SIGTERM` / `SIGINT`. On each "peer arrived" event the hub
/// reciprocally pushes every meeting it holds to that peer (debounced per peer) —
/// so a device that reconnects both deposits and collects, converging through the
/// hub — and that push rides a lifecycle discovery in the same flow (§7); and a
/// periodic discovery sweep re-advertises so a lifecycle state a consumer dropped
/// or skipped is re-applied (the recovery driver).
///
/// When `account_args` is `Some`, the account-refresh loop runs as a select! arm
/// alongside the discovery sweep — the ONLY peer-discovery mechanism now (manual
/// ticket pairing removed). The loop is NOT detached: running it inline here
/// means it cannot outlive this function and cannot race `engine.shutdown()`
/// after this returns.
///
/// The lifecycle-event CONSUMER runs in a dedicated spawned task
/// ([`apply_lifecycle_events`]), NOT in this select loop: the loop awaits the
/// emitters (`discover_all` and the `push_all_to` ride-along both emit into the
/// same broadcast), so draining in the same loop would let a sweep larger than the
/// channel cap self-lag while the loop is parked on the sweep producing it — a
/// separate drain keeps up concurrently.
async fn serve_until_shutdown(
    engine: Arc<SyncEngine>,
    data_dir: &Path,
    account_args: Option<AccountRefreshArgs>,
) {
    let debounce = dur_or_env("MINUTIST_HUB_PUSH_DEBOUNCE_MS", PEER_PUSH_DEBOUNCE);
    let mut discovery_poll =
        tokio::time::interval(dur_or_env("MINUTIST_HUB_DISCOVERY_MS", DISCOVERY_INTERVAL));
    // Ticks immediately on the first `.tick()`, then every `TRASH_SWEEP_INTERVAL`
    // — so a meeting deleted while the hub was offline is purged promptly on
    // this run's first tick, not just on the hourly cadence thereafter.
    let mut trash_sweep_poll =
        tokio::time::interval(dur_or_env("MINUTIST_HUB_TRASH_SWEEP_MS", TRASH_SWEEP_INTERVAL));
    let mut peer_events = engine.subscribe_peer_events();
    let meetings_root = data_dir.join("meetings");
    let mut last_push: HashMap<_, Instant> = HashMap::new();

    // Drain discovery lifecycle events in a DEDICATED task so the consumer keeps up
    // while this loop is parked awaiting an emitter (discover_all / the push_all_to
    // ride-along emit into the same broadcast). The engine's sender outlives this
    // fn, so the task is aborted on shutdown rather than seeing `Closed` on its own.
    let lifecycle_task = tokio::spawn(apply_lifecycle_events(
        engine.subscribe_lifecycle_events(),
        meetings_root.clone(),
    ));

    // Pin the shutdown future ONCE so its signal handlers persist across loop
    // iterations — recreating them each pass could drop a signal that arrives in
    // the gap, hanging `systemctl stop`.
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    // Account-refresh loop v2: runs as a third select! arm when account credentials
    // are present, driving `run_account_refresh_loop_v2` over the production
    // `SyncEngineRefreshSink` (which shares the engine, so its dial-suppression and
    // source-aware removal act on the same `PeerDirectory` the other arms feed). Its
    // `CancellationToken` is cancelled after the serve loop breaks so the loop shares
    // the daemon's single shutdown signal; when absent, the arm resolves to
    // `std::future::pending()` and the token has no waiter.
    //
    // Pinned once, like `shutdown`, so the arm persists across loop iterations
    // without recreating the future on each pass.
    let account_cancel = CancellationToken::new();
    let account_refresh: Pin<Box<dyn Future<Output = ()> + Send>> = match account_args {
        Some(args) => {
            let sink: Arc<dyn RefreshSink> =
                Arc::new(SyncEngineRefreshSink::new(Arc::clone(&engine)));
            Box::pin(run_account_refresh_loop_v2(
                args.source,
                args.self_endpoint,
                ACCOUNT_REFRESH_INTERVAL,
                account_cancel.clone(),
                sink,
            ))
        }
        None => Box::pin(std::future::pending()),
    };
    tokio::pin!(account_refresh);

    // A meeting this hub already purged must never be re-adopted from a slow
    // peer that hasn't caught up yet (see `persistence::purged`'s design
    // rationale). A tombstone-read failure fails OPEN (treated as not-purged)
    // — a rare spurious resurrection is far cheaper than permanently starving
    // a legitimate adopt on a transient read error.

    'serve: loop {
        tokio::select! {
            _ = &mut shutdown => break 'serve,
            // Account-mediated peer discovery, the only discovery mechanism now.
            // `run_account_refresh_loop_v2` runs until `account_cancel` fires
            // (below, on every shutdown path), so under normal operation this
            // arm never actually resolves — it just parks here. If it ever DID
            // resolve on its own (e.g. a future fatal-auth early return), continuing
            // to poll the same pinned future again next iteration would panic
            // ("resumed after completion"), so treat an unexpected completion as
            // fatal rather than silently re-polling it.
            _ = &mut account_refresh => {
                tracing::error!(target: "hub", "account-refresh loop exited unexpectedly; stopping");
                break 'serve;
            }
            _ = trash_sweep_poll.tick() => {
                // Purge this hub's own expired trash. Blocking `std::fs` work run
                // directly (not `spawn_blocking`): the hub has no dedicated worker
                // pool and every other filesystem scan on this loop (`list_meeting_ids`
                // via discovery, account-directory refresh) is already called the same way.
                match persistence::meeting_ops::sweep_expired_deletions_no_index(
                    &meetings_root,
                    data_dir,
                    minutist_common::TRASH_TTL_DAYS,
                ) {
                    Ok(purged) if !purged.is_empty() => {
                        tracing::info!(target: "hub", count = purged.len(), "trash auto-purge sweep");
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!(target: "hub", error = %e, "trash auto-purge sweep failed"),
                }
            }
            _ = discovery_poll.tick() => {
                // Replica sweep: re-discover every known peer (re-applying a lifecycle
                // a consumer dropped [Lagged] or skipped) AND pull every meeting the
                // hub still lacks — the periodic backfill so a sometimes-online peer
                // converges through the hub. Raced against shutdown, like the push arm.
                // (The first, immediate tick is a no-op before peers load.)
                let lookup = purged_lookup(data_dir);
                tokio::select! {
                    _ = &mut shutdown => break 'serve,
                    result = engine.adopt_all(&lookup) => match result {
                        Ok(n) => tracing::debug!(target: "hub", adopted = n, "periodic replica sweep"),
                        Err(e) => tracing::warn!(target: "hub", error = %e, "periodic replica sweep failed"),
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
                        _ = &mut shutdown => break 'serve,
                        result = engine.push_all_to_peer(&peer) => match result {
                            Ok(n) => tracing::info!(target: "hub", peer = %peer, meetings = n, "pushed meetings to arrived peer"),
                            Err(e) => tracing::warn!(target: "hub", peer = %peer, error = %e, "push to arrived peer failed"),
                        },
                    }
                    // ...and PULL: adopt every meeting the arrived peer has that the
                    // hub lacks (notes+media+artifacts) so the hub mirrors the
                    // account's meetings — the backfill direction (a hub that comes
                    // up after a device recorded must pull that device's history).
                    // Raced against shutdown like the push.
                    let lookup = purged_lookup(data_dir);
                    tokio::select! {
                        _ = &mut shutdown => break 'serve,
                        result = engine.adopt_from_peer(&peer, &lookup) => match result {
                            Ok(n) => tracing::info!(target: "hub", peer = %peer, adopted = n, "adopted meetings from arrived peer"),
                            Err(e) => tracing::warn!(target: "hub", peer = %peer, error = %e, "adopt from arrived peer failed"),
                        },
                    }
                }
            }
        }
    }

    // Cancel the account-refresh loop (latching `CancellationToken`): every shutdown
    // path breaks to here, and the pinned `account_refresh` future is dropped on
    // return regardless, so this only makes the loop's own exit explicit. A no-op
    // when no loop was started (the token has no waiter).
    account_cancel.cancel();
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
    mut lifecycle_events: broadcast::Receiver<(MeetingId, ProcessingLifecycle, DeletionState)>,
    meetings_root: PathBuf,
) {
    loop {
        match lifecycle_events.recv().await {
            Ok((meeting_id, processing, deletion)) => {
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
                // The hub runs no `index.db` — unlike `persistence::meeting_ops`'s
                // index-aware wrapper, call the leaf directly (nothing to mirror).
                match notes_crdt::apply_synced_deletion_if_present(&meetings_root, meeting_id, deletion) {
                    Ok(true) => {}
                    Ok(false) => tracing::debug!(target: "hub", meeting_id = %meeting_id.0, "synced deletion state for a meeting not present locally; skipping"),
                    Err(e) => tracing::warn!(target: "hub", meeting_id = %meeting_id.0, error = %e, "failed to apply synced deletion state"),
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

    #[test]
    fn stored_credential_store_round_trips_at_0600() {
        let base = tempfile::tempdir().expect("tempdir");
        let cred = StoredCredential {
            device_credential: "mdc_dev.secret".to_string(),
            account_id: "acct-1".to_string(),
            device_id: "dev-1".to_string(),
        };
        cred.store(base.path()).expect("store");
        let loaded = StoredCredential::load(base.path()).expect("load");
        assert_eq!(loaded, cred);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(base.path().join(CREDENTIAL_FILE))
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "credential file must be owner-only");
        }
    }

    #[test]
    #[cfg(unix)]
    fn stored_credential_store_tightens_a_pre_existing_looser_mode() {
        use std::os::unix::fs::PermissionsExt;

        let base = tempfile::tempdir().expect("tempdir");
        let path = base.path().join(CREDENTIAL_FILE);
        std::fs::write(&path, b"stale").expect("seed a pre-existing file");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
            .expect("loosen mode");

        let cred = StoredCredential {
            device_credential: "mdc_dev.secret".to_string(),
            account_id: "acct-1".to_string(),
            device_id: "dev-1".to_string(),
        };
        cred.store(base.path()).expect("store over the looser-mode file");

        let mode = std::fs::metadata(&path).expect("metadata").permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "the atomic rename must replace a pre-existing looser-mode file with a 0600 one"
        );
        assert_eq!(StoredCredential::load(base.path()).expect("load"), cred);
    }

    #[test]
    fn pairing_errors_map_without_leaking() {
        assert!(matches!(
            map_pairing_err(tunnel_client::PairingError::Expired),
            AppError::InvalidInput { .. }
        ));
        assert!(matches!(
            map_pairing_err(tunnel_client::PairingError::AccessDenied),
            AppError::InvalidInput { .. }
        ));
        assert!(matches!(
            map_pairing_err(tunnel_client::PairingError::Status { status: 500 }),
            AppError::Internal { .. }
        ));
    }
}
