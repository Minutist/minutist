// Prevents additional console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::Arc;

use ipc_bridge::{spawn_event_forwarder, IpcState};
use settings::{JsonFileStore, SettingsHandle};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, WindowEvent};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod updater;

/// The bundle identifier, mirrored from `tauri.conf.json`. Used for the
/// pre-Tauri log-dir resolution.
const APP_IDENTIFIER: &str = "ai.minutist";

fn main() {
    // ---------------------------------------------------------------------------
    // Logging — file appender + optional console.
    //
    // The app-data path is not yet known at this point (requires a Tauri
    // app handle), so we initialise the subscriber with only a console layer
    // initially.  Once the app handle is available inside setup(), we cannot
    // reconfigure the global subscriber.
    //
    // Resolution: we construct the non-blocking file appender here using the
    // platform-appropriate app-data directory resolved via dirs::data_dir(),
    // which gives the same base path that Tauri's path().app_data_dir() would
    // return.  The Tauri app-data dir is:
    //   Linux:   ~/.local/share/<identifier>
    //   macOS:   ~/Library/Application Support/<identifier>
    //   Windows: %APPDATA%\<identifier>
    //
    // We compute it manually here to allow the subscriber to be set up before
    // the Tauri runtime starts.
    // ---------------------------------------------------------------------------
    let log_dir = resolve_log_dir();
    if let Err(e) = std::fs::create_dir_all(&log_dir) {
        eprintln!("[app-main] failed to create log dir {:?}: {e}", log_dir);
    }

    let file_appender = tracing_appender::rolling::daily(&log_dir, "minutist.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // Default to `info` when neither `RUST_LOG` nor the env is set, so shipped
    // builds capture useful diagnostics (cross-cutting.md "Logging" wants a
    // sensible default plus `RUST_LOG` override). `EnvFilter::from_default_env`
    // would otherwise fall back to ERROR-only and drop every info/debug/warn.
    // `RUST_LOG`, when set, is still honoured verbatim.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false);

    #[cfg(debug_assertions)]
    let subscriber = {
        let console_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(true);
        tracing_subscriber::registry()
            .with(filter)
            .with(file_layer)
            .with(console_layer)
    };

    #[cfg(not(debug_assertions))]
    let subscriber = tracing_subscriber::registry().with(filter).with(file_layer);

    subscriber.init();

    // Keep _guard alive for the duration of the process so the non-blocking
    // writer flushes on exit.
    run(_guard);
}

/// Resolve the logs directory prior to the Tauri runtime being available.
///
/// Mirrors the path Tauri would produce for `BaseDirectory::AppData`:
/// - Linux:   `$XDG_DATA_HOME/<identifier>` or `~/.local/share/<identifier>`
/// - macOS:   `~/Library/Application Support/<identifier>`
/// - Windows: `%APPDATA%\<identifier>`
fn resolve_log_dir() -> PathBuf {
    app_data_root(APP_IDENTIFIER).join("logs")
}

/// The platform app-data root for `identifier`, mirroring Tauri's
/// `BaseDirectory::AppData` resolution (see `resolve_log_dir` docs).
/// Parameterised so the legacy-migration shim resolves BOTH the old and new
/// identifiers' roots with identical platform logic.
fn app_data_root(identifier: &str) -> PathBuf {
    // Attempt to use the same base path as Tauri's AppData directory.
    // tauri uses `dirs` crate internally; we use std env as a fallback.

    #[cfg(target_os = "linux")]
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs_home().map(|h| h.join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));

    #[cfg(target_os = "macos")]
    let base = dirs_home()
        .map(|h| h.join("Library").join("Application Support"))
        .unwrap_or_else(|| PathBuf::from("/tmp"));

    #[cfg(target_os = "windows")]
    let base = std::env::var("APPDATA")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("C:\\Users\\Default\\AppData\\Roaming"));

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    let base = PathBuf::from("/tmp");

    base.join(identifier)
}

#[allow(dead_code)]
fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

/// Clean up log files older than `max_days` in `log_dir`.
///
/// `tracing-appender 0.2.x` does not provide a built-in retention policy.
/// This function is called once at startup to remove stale daily log files.
fn cleanup_old_logs(log_dir: &std::path::Path, max_days: u64) {
    let cutoff = std::time::SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(max_days * 24 * 3600))
        .unwrap_or(std::time::UNIX_EPOCH);

    let read_dir = match std::fs::read_dir(log_dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };

    for entry in read_dir.flatten() {
        let path = entry.path();
        // Only remove files that match our prefix.
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_owned(),
            None => continue,
        };
        if !name.starts_with("minutist.log") {
            continue;
        }
        let modified = match entry.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        if modified < cutoff {
            let _ = std::fs::remove_file(&path);
            tracing::debug!(
                target: "app-main",
                path = %path.display(),
                "removed old log file"
            );
        }
    }
}

/// Resolve the bundled Silero VAD ONNX model and export its absolute path as
/// `MINUTIST_SILERO_PATH` so `vad-chunker::default_model_path()` finds it in
/// an installed package.
///
/// The model is shipped via the `bundle.resources` entry
/// `"../resources/silero/silero_vad_v4.onnx"` in `tauri.conf.json`. Tauri places
/// the file under the resource dir with parent-dir traversal mangled to `_up_`
/// (`tauri-utils::resources::resource_relpath`), and `PathResolver::resolve`
/// applies the SAME mangling to its input, so resolving the original
/// config-relative pattern yields the placed path. We also try the
/// already-mangled relpath as a fallback and pick whichever exists.
///
/// In a dev run (`cargo run`, no bundle) the resource does not resolve to an
/// existing file; we then leave the env var unset and let `vad-chunker`'s
/// source-tree fallback handle it. The var is set once here, early in `setup`,
/// before the orchestrator is constructed (edition 2021 → `set_var` is safe).
fn resolve_silero_model(app: &tauri::AppHandle) {
    use tauri::path::BaseDirectory;

    // Candidate inputs to `resolve(.., Resource)`. The first mirrors the
    // `bundle.resources` config entry verbatim; the second is the mangled
    // relpath the bundler actually writes — both should resolve to the same
    // placed file, but we try both to be robust to resolver edge cases.
    const CANDIDATES: [&str; 2] = [
        "../resources/silero/silero_vad_v4.onnx",
        "_up_/resources/silero/silero_vad_v4.onnx",
    ];

    for candidate in CANDIDATES {
        match app.path().resolve(candidate, BaseDirectory::Resource) {
            Ok(path) if path.is_file() => {
                std::env::set_var("MINUTIST_SILERO_PATH", &path);
                tracing::info!(
                    target: "app-main",
                    path = %path.display(),
                    "bundled Silero VAD model resolved; MINUTIST_SILERO_PATH set"
                );
                return;
            }
            Ok(path) => {
                tracing::debug!(
                    target: "app-main",
                    candidate,
                    resolved = %path.display(),
                    "Silero resource candidate did not resolve to an existing file"
                );
            }
            Err(e) => {
                tracing::debug!(
                    target: "app-main",
                    candidate,
                    error = %e,
                    "failed to resolve Silero resource candidate"
                );
            }
        }
    }

    tracing::debug!(
        target: "app-main",
        "no bundled Silero VAD model found (dev run); leaving MINUTIST_SILERO_PATH unset \
         so vad-chunker uses its source-tree fallback"
    );
}

/// Resolve the MCP bearer token: read the persisted token, or generate + persist
/// a fresh 256-bit CSPRNG one on first enable.
///
/// **Storage.** The token is stored at `{app-data}/mcp_token`. On Unix the file
/// is CREATED with mode `0o600` atomically (no write-then-chmod window). On
/// Windows the file inherits the per-user app-data directory's ACL. The correct
/// API to additionally tighten the Windows file ACL to owner-only is
/// `SetNamedSecurityInfoW` (advapi32), reachable via `windows-sys >= 0.59`
/// (`Win32_Security` + `Win32_Security_Authorization` features). That dep is
/// not yet declared as a direct `[target.'cfg(windows)'.dependencies]` entry in
/// `src-tauri/Cargo.toml`; adding it is deferred to a dedicated Windows-platform
/// hardening commit. The loopback bind + bearer + Host/Origin checks are the
/// primary controls regardless. The token is NEVER logged.
#[cfg(feature = "connected")]
fn resolve_mcp_token(app_data_dir: &std::path::Path) -> String {
    let token_path = app_data_dir.join("mcp_token");

    // Reuse an existing token (so a saved external MCP client config stays valid
    // across runs). A blank/whitespace file is treated as absent.
    if let Ok(existing) = std::fs::read_to_string(&token_path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    // Generate a fresh 256-bit token, hex-encoded.
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let token = hex::encode(bytes);

    if let Err(e) = write_token_file(&token_path, &token) {
        tracing::warn!(
            target: "app-main",
            "failed to persist MCP token to {:?}: {e}; using a per-run token",
            token_path
        );
    } else {
        tracing::info!(target: "app-main", "generated a new MCP bearer token");
    }
    token
}

/// Write the token to `path`, creating the file with owner-only mode `0o600`
/// atomically on Unix (no write-then-chmod window). On Windows the file
/// inherits the parent directory's ACL; per-file ACL tightening via
/// `SetNamedSecurityInfoW` is not yet implemented (see `resolve_mcp_token`).
#[cfg(feature = "connected")]
fn write_token_file(path: &std::path::Path, token: &str) -> std::io::Result<()> {
    use std::io::Write;

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Create with 0600 so the token is never momentarily world-readable.
        // `mode` applies only when the file is CREATED; if it already exists
        // (a blank file we are overwriting) re-assert the mode below.
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Re-assert 0600 in case the file pre-existed with looser perms (the
        // `mode` above is a no-op then). Cheap and closes the pre-existing-file
        // edge.
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(token.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// The live per-instance handles for the running MCP server accept loop.
/// Stored inside [`McpShutdownState`].
#[cfg(feature = "connected")]
struct McpInstanceHandles {
    /// Fires the shutdown signal to the accept loop.
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    /// Resolves once the accept loop has exited and the listener has dropped.
    done_rx: tokio::sync::oneshot::Receiver<()>,
}

/// Live handles to the running MCP server instance (connected build only).
///
/// Held in Tauri's managed state so the settings watcher (which holds an
/// `Arc<McpShutdownState>` clone) can stop the server on `mcp_enabled → false`
/// without touching `IpcState`. `None` when no server is currently running.
/// The settings watcher is the only writer.
///
/// Storing the completion receiver alongside the shutdown sender lets the
/// watcher await the accept-loop exit before re-enabling, preventing a rapid
/// off→on toggle from racing the port release.
#[cfg(feature = "connected")]
struct McpShutdownState(std::sync::Mutex<Option<McpInstanceHandles>>);

/// Parameters passed to [`do_start_mcp_server`] to keep the signature
/// manageable (all values are needed both at boot and on re-enable).
#[cfg(feature = "connected")]
struct McpStartParams {
    orchestrator: Arc<orchestrator::Orchestrator>,
    index: Arc<ipc_bridge::MeetingIndex>,
    meetings_dir: std::path::PathBuf,
    event_tx: tokio::sync::broadcast::Sender<minutist_common::AppEvent>,
    settings: settings::SettingsHandle,
    summariser: Arc<tokio::sync::OnceCell<Arc<ipc_bridge::LlamaSummariser>>>,
    chat_in_flight:
        Arc<std::sync::Mutex<std::collections::HashSet<minutist_common::ChatSessionId>>>,
    app_data_dir: std::path::PathBuf,
    /// A clone of `IpcState::mcp_info`; written on successful bind.
    mcp_info: Arc<std::sync::Mutex<Option<ipc_bridge::McpServerInfo>>>,
    /// Written with the per-instance handles on successful start; cleared on
    /// failure. The settings watcher reads this to stop the server and to
    /// determine achieved state (present = running, absent = not running).
    shutdown_state: Arc<McpShutdownState>,
}

/// Start the MCP server and inter-agent bridge for the current settings
/// snapshot (`settings.mcp_port`, `settings.mcp_write_tools`).
///
/// Creates a new inter-agent bridge driver, resolves the bearer token, creates
/// a fresh shutdown watch pair, and awaits `mcp_server::serve`. On a
/// successful bind the per-instance handles (shutdown sender + completion
/// receiver) are stored in `params.shutdown_state`, `mcp_info` is filled, and
/// `McpServerListening` is emitted. On any failure (summariser load or bind
/// error) the handles slot is cleared and `McpServerStartFailed` is emitted,
/// so the UI drops the "starting…" hint immediately.
///
/// This function is `async` and must be awaited inside the watcher task (not
/// spawned independently) so the watcher serialises stop→start sequentially.
#[cfg(feature = "connected")]
async fn do_start_mcp_server(params: McpStartParams) {
    let settings_now = params.settings.current();
    let allow_writes = settings_now.mcp_write_tools;
    let port = settings_now.mcp_port;

    let chat_handles = ipc_bridge::ChatHandles {
        orchestrator: params.orchestrator.clone(),
        index: params.index.clone(),
        meetings_dir: params.meetings_dir.clone(),
        event_tx: params.event_tx.clone(),
        settings: params.settings.clone(),
        summariser: params.summariser.clone(),
    };

    // Ensure the held summariser is loaded before binding the port: a failure
    // here is an error, not deferred.
    let summariser = match chat_handles.ensure_summariser().await {
        Ok(s) => s,
        Err(e) => {
            let reason = format!("summariser load failed: {e}");
            tracing::error!(target: "app-main", "MCP server not started: {reason}");
            // Clear the handles slot so the watcher knows achieved=false.
            *params
                .shutdown_state
                .0
                .lock()
                .expect("mcp_shutdown poisoned") = None;
            let _ = params
                .event_tx
                .send(minutist_common::AppEvent::McpServerStartFailed { reason });
            return;
        }
    };

    // Create a fresh shutdown watch pair FIRST so the inter-agent driver and
    // the mcp-server accept loop share the same per-instance signal.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let bridge_tx = ipc_bridge::spawn_inter_agent_driver(
        chat_handles,
        params.chat_in_flight,
        allow_writes,
        // The driver subscribes to the same per-instance shutdown signal so
        // disabling the MCP server also stops the driver deterministically.
        shutdown_tx.subscribe(),
    );

    let mcp_registry = Arc::new(agent_tools::ToolRegistry::v1(true));
    let token = resolve_mcp_token(&params.app_data_dir);

    let ctx = Arc::new(
        agent_tools::ToolContext::new(
            params.orchestrator.clone(),
            params.index.clone(),
            params.meetings_dir.clone(),
            summariser as Arc<dyn minutist_common::Summariser>,
            params.event_tx.clone(),
            None, // MCP callers pass meeting_id explicitly
        )
        .with_inter_agent_bridge(bridge_tx),
    );
    let bind_addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));

    match mcp_server::serve(
        mcp_registry,
        ctx,
        mcp_server::McpServerConfig {
            bind_addr,
            bearer_token: token.clone(),
            allow_writes,
        },
        shutdown_rx,
    )
    .await
    {
        Ok((bound, done_rx)) => {
            let url = format!("http://{bound}{}", mcp_server::MCP_PATH);
            // Store handles so the watcher can stop the server and await
            // completion before re-enabling.
            *params
                .shutdown_state
                .0
                .lock()
                .expect("mcp_shutdown poisoned") = Some(McpInstanceHandles {
                shutdown_tx,
                done_rx,
            });
            *params.mcp_info.lock().expect("mcp_info poisoned") =
                Some(ipc_bridge::McpServerInfo {
                    url: url.clone(),
                    token,
                });
            // Emit the listening event so the MCP pane refreshes.
            // The token is NOT carried on the event.
            let _ = params
                .event_tx
                .send(minutist_common::AppEvent::McpServerListening { url });
        }
        Err(e) => {
            let reason = format!("bind failed: {e}");
            tracing::error!(target: "app-main", "MCP server failed to start: {reason}");
            // Clear the handles slot (achieved=false) and notify the UI.
            *params
                .shutdown_state
                .0
                .lock()
                .expect("mcp_shutdown poisoned") = None;
            let _ = params
                .event_tx
                .send(minutist_common::AppEvent::McpServerStartFailed { reason });
        }
    }
}

/// The set of path roots derived from the platform app-data directory and an
/// optional user-configured override, passed to the components that own each
/// directory scope.
///
/// `meetings` and `models` may diverge from `index_db` when
/// `settings.data_directory` is set.
struct DataRoots {
    /// Root for per-meeting folders (`{root}/meetings/`).  Owned by
    /// `persistence`.
    meetings: PathBuf,
    /// Root for per-kind model cache (`{root}/models/`).  Owned by
    /// `model-registry`.
    models: PathBuf,
    /// Parent directory for `index.db`.  Must be the same directory that
    /// `ipc-bridge` opens the DB against.
    index_db: PathBuf,
}

/// Derive `DataRoots` from the platform app-data directory and an optional
/// user-configured data directory override from settings.
///
/// When `data_dir_override` is `None`, all roots are placed under
/// `platform_root`.  When it is `Some(path)`, `meetings`, `models`, and
/// `index_db` are placed under `path` instead; the override must be an
/// absolute path that can be created if absent.  An invalid value (relative,
/// empty, or uncreatable) falls back to `platform_root` with a logged error.
///
/// `settings.store` always stays at the platform root regardless of the
/// override — it is how the override itself is found.  Logs stay at the
/// platform root for the same reason (logging starts before settings load).
/// Both invariants are documented in the `Settings::data_directory` field
/// doc comment.
///
/// The paths are fixed at startup; a change requires an app restart.
fn resolve_data_roots(
    platform_root: &std::path::Path,
    data_dir_override: Option<&std::path::Path>,
) -> DataRoots {
    let effective = match data_dir_override {
        None => platform_root.to_path_buf(),
        Some(p) => {
            if !p.is_absolute() {
                tracing::error!(
                    target: "app-main",
                    path = %p.display(),
                    "settings.data_directory is not absolute; falling back to platform default"
                );
                platform_root.to_path_buf()
            } else if p.as_os_str().is_empty() {
                tracing::error!(
                    target: "app-main",
                    "settings.data_directory is empty; falling back to platform default"
                );
                platform_root.to_path_buf()
            } else {
                match std::fs::create_dir_all(p) {
                    Ok(()) => {
                        tracing::info!(
                            target: "app-main",
                            path = %p.display(),
                            "using settings.data_directory as data root"
                        );
                        p.to_path_buf()
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "app-main",
                            path = %p.display(),
                            "failed to create settings.data_directory ({e}); \
                             falling back to platform default"
                        );
                        platform_root.to_path_buf()
                    }
                }
            }
        }
    };

    DataRoots {
        meetings: effective.join("meetings"),
        models: effective.join("models"),
        index_db: effective.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn no_override_uses_platform_root() {
        let platform = p("/data/app");
        let roots = resolve_data_roots(&platform, None);
        assert_eq!(roots.meetings, p("/data/app/meetings"));
        assert_eq!(roots.models, p("/data/app/models"));
        assert_eq!(roots.index_db, p("/data/app"));
    }

    #[test]
    fn relative_override_falls_back() {
        let platform = p("/data/app");
        let roots = resolve_data_roots(&platform, Some(&p("relative/path")));
        assert_eq!(roots.meetings, p("/data/app/meetings"));
        assert_eq!(roots.models, p("/data/app/models"));
        assert_eq!(roots.index_db, p("/data/app"));
    }

    #[test]
    fn empty_override_falls_back() {
        let platform = p("/data/app");
        let roots = resolve_data_roots(&platform, Some(&p("")));
        assert_eq!(roots.meetings, p("/data/app/meetings"));
        assert_eq!(roots.models, p("/data/app/models"));
        assert_eq!(roots.index_db, p("/data/app"));
    }

    #[test]
    fn valid_absolute_override_is_used() {
        let tmp = std::env::temp_dir().join("minutist_test_override");
        // Pre-create so create_dir_all succeeds.
        std::fs::create_dir_all(&tmp).unwrap();
        let platform = p("/data/app");
        let roots = resolve_data_roots(&platform, Some(&tmp));
        assert_eq!(roots.meetings, tmp.join("meetings"));
        assert_eq!(roots.models, tmp.join("models"));
        assert_eq!(roots.index_db, tmp);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn uncreatable_override_falls_back() {
        // A path beneath a non-existent parent that is impossible to create.
        let bad: PathBuf = if cfg!(unix) {
            p("/proc/this_cannot_be_created_minutist_test/sub")
        } else {
            p("C:\\this_cannot_exist_minutist_test\\sub")
        };
        let platform = p("/data/app");
        let roots = resolve_data_roots(&platform, Some(&bad));
        // Falls back to platform root.
        assert_eq!(roots.meetings, p("/data/app/meetings"));
    }
}

/// The Tauri runtime entry point.
///
/// Accepts the non-blocking writer guard so it stays alive for the process
/// lifetime and the writer flushes on exit.
fn run(_log_guard: tracing_appender::non_blocking::WorkerGuard) {
    let builder = ipc_bridge::bindings_builder();

    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        // Note-image asset protocol (`meetingasset:`). Serves the bytes of a
        // pasted/dropped note image from `{meetings_dir}/<uuid>/assets/<file>`
        // to the webview. The webview renders a stored PORTABLE filename ref via
        // `convertFileSrc(<uuid>/<file>, "meetingasset")`; this handler resolves
        // it back to bytes. NO whole-filesystem exposure — `resolve_note_asset`
        // (in `ipc-bridge`, which owns the `persistence` edge) parses the path
        // into `(meeting_id: Uuid, filename)`, and `persistence::read_note_asset`
        // applies a path-traversal guard, so only a file directly inside a
        // meeting's `assets/` can ever be read. Any validation/read failure →
        // an empty 404 (no detail leaks). See architecture/cross-cutting.md —
        // "Note image assets".
        .register_uri_scheme_protocol(ipc_bridge::MEETING_ASSET_SCHEME, serve_note_asset)
        .invoke_handler(builder.invoke_handler())
        .setup(move |app| {
            // Mount events first so the event channel is ready before any
            // commands run.
            builder.mount_events(app);

            let app_handle = app.handle().clone();

            // Resolve app-data path via Tauri so it matches the platform convention.
            let app_data_dir = app_handle
                .path()
                .app_data_dir()
                .expect("failed to resolve app-data dir");

            // `logs/` is owned by app-main (see architecture/cross-cutting.md
            // "Filesystem layout"). `meetings/` is owned by `persistence`, which
            // creates it on demand via `MeetingFolder::create`; app-main MUST NOT
            // pre-create it here.
            let logs_dir = app_data_dir.join("logs");
            std::fs::create_dir_all(&logs_dir).expect("failed to create logs dir");

            tracing::info!(
                target: "app-main",
                app_data = %app_data_dir.display(),
                "app-data directory resolved"
            );

            // Resolve the bundled Silero VAD model and plumb its path to
            // `vad-chunker` via `MINUTIST_SILERO_PATH` (see
            // architecture/cross-cutting.md "Model lifecycle — Exception:
            // Silero VAD"). MUST run before the orchestrator is constructed and
            // before any recording, because `vad-chunker::default_model_path()`
            // reads this env var at chunker-open time. A no-op in a dev run with
            // no bundle (the source-tree fallback handles that path).
            resolve_silero_model(&app_handle);

            // Prune log files older than 7 days.
            cleanup_old_logs(&logs_dir, 7);

            // Construct settings handle backed by the JSON file store.
            // `settings.store` always lives at the platform root so the override
            // path itself can be stored in settings (see `DataRoots` /
            // `resolve_data_roots`).
            let settings_store = JsonFileStore::new(app_data_dir.join("settings.store"));
            let settings_handle =
                SettingsHandle::new(settings_store).expect("failed to initialise settings");

            tracing::info!(target: "app-main", "settings handle constructed");

            // Resolve the effective data roots for meetings, models, and the
            // index DB.  When `settings.data_directory` is set to a valid
            // absolute path, those roots are placed there rather than under
            // `app_data_dir`.  Logs and `settings.store` always stay at the
            // platform root (bootstrap constraint — see `resolve_data_roots`).
            let data_roots = {
                let current = settings_handle.current();
                resolve_data_roots(
                    &app_data_dir,
                    current.data_directory.as_deref(),
                )
            };

            let meetings_dir = data_roots.meetings;

            tracing::info!(
                target: "app-main",
                meetings = %meetings_dir.display(),
                models = %data_roots.models.display(),
                "data roots resolved"
            );

            // Single event bus shared by the model registry and the
            // orchestrator. The IPC forwarder subscribes once (via
            // Orchestrator::subscribe_events) and sees both orchestrator
            // events (meter / state / transcript) and the registry's
            // ModelDownloadProgress. Constructing the channel here breaks the
            // otherwise-circular dependency (registry is a constructor param of
            // the orchestrator, but the registry needs the sender too).
            let (event_tx, _event_rx) = tokio::sync::broadcast::channel(256);
            // Clone the sender for IpcState so the Phase-5 `summarise_meeting`
            // command can emit `AppEvent::SummaryReady` on the SAME bus the
            // orchestrator/registry broadcast on (the event forwarder subscribes
            // once via Orchestrator::subscribe_events and sees it too).
            let ipc_event_tx = event_tx.clone();
            // Clone for the auto-updater (Phase 7) so it emits UpdateAvailable /
            // UpdateProgress on the SAME bus the forwarder relays to the webview.
            let updater_event_tx = event_tx.clone();

            // Construct the model registry. The manifest is bundled at compile
            // time from resources/models.json; the cache root is the per-kind
            // model directory under app-data (model-registry owns this dir).
            let models_root = data_roots.models;
            let manifest =
                model_registry::load_manifest(include_bytes!("../../resources/models.json"))
                    .expect("bundled resources/models.json is malformed");
            let model_registry = Arc::new(
                model_registry::ModelRegistry::new(models_root, manifest, event_tx.clone())
                    .expect("failed to initialise model registry"),
            );

            tracing::info!(target: "app-main", "model registry constructed");

            // Construct the orchestrator sharing the same event bus. Clone the
            // meetings dir first so the IPC state can route `save_notes` /
            // `load_notes` / `open_meeting` directly to `persistence` against the
            // same root the orchestrator/persistence use.
            let notes_meetings_dir = meetings_dir.clone();
            let orchestrator = Arc::new(orchestrator::Orchestrator::with_event_tx(
                settings_handle.clone(),
                meetings_dir,
                model_registry,
                event_tx,
            ));

            tracing::info!(target: "app-main", "orchestrator constructed");

            // Open the libsql meeting index (`{app-data}/index.db`) via the
            // `ipc-bridge` helper (which owns the `persistence` dependency — see
            // the dependency table in architecture/components.md; app-main does
            // not depend on `persistence` directly). libsql is async; `setup` is
            // not a command handler, so the helper's one-shot block_on at startup
            // is acceptable (the no-block_on rule binds command handlers, not
            // bootstrap). The index is a derived cache — the helper rebuilds it
            // from the per-meeting folders on startup so it converges even if a
            // prior run crashed between a folder write and the index update.
            let (index_db_path, index) =
                ipc_bridge::open_meeting_index(&data_roots.index_db, &notes_meetings_dir);

            tracing::info!(target: "app-main", "meeting index opened");

            // Spawn the event forwarder so orchestrator events reach the webview.
            spawn_event_forwarder(orchestrator.clone(), app_handle.clone());

            // Live-test UX T2: pre-warm the routed ASR model in the background so
            // the FIRST record does not pay the cold ~29 s model load at record
            // time. Best-effort + non-blocking-at-start (no download; a
            // not-yet-downloaded model warms nothing), so it can never delay
            // startup or fail it. `tauri::async_runtime::spawn` (NOT a bare
            // `tokio::spawn`) — `setup` runs with no entered Tokio runtime.
            {
                let prewarm_orchestrator = orchestrator.clone();
                tauri::async_runtime::spawn(async move {
                    prewarm_orchestrator.prewarm_asr().await;
                });
            }

            // Wire the auto-updater (Phase 7): a guarded startup check + the
            // apply-on-accept listener. A no-op when `plugins.updater` is
            // unconfigured (the committed default), so dev/unsigned builds are
            // unaffected.
            updater::start(&app_handle, updater_event_tx);

            // Build the chat tool registry once (Phase 9). `v1(false)` omits the
            // Phase-10 inter-agent bridge tool (the internal agent must not be
            // able to message itself). The held LLM substrate is loaded lazily on
            // first chat/summarise use (see `IpcState::ensure_summariser`).
            let tool_registry = Arc::new(agent_tools::ToolRegistry::v1(false));

            // The shared single-in-flight-turn guard (Phase 9), shared with the
            // Phase-10 inter-agent driver so an external turn and a human turn
            // cannot run on one session at once.
            let chat_in_flight = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));

            // Per-session chat-turn cancel flags (P1); `cancel_chat_turn` raises
            // the flag for a running UI turn.
            let chat_cancel = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

            // The held-summariser cell, shared by the chat command + the
            // inter-agent driver (both load the SAME model once).
            let summariser_cell = Arc::new(tokio::sync::OnceCell::new());

            // The MCP server info slot (URL + token). In the connected build this
            // is filled once the MCP server binds; in the free build it is
            // permanently None (get_mcp_server_info returns None, the MCP pane is
            // absent from the UI bundle).
            let mcp_info: Arc<std::sync::Mutex<Option<ipc_bridge::McpServerInfo>>> =
                Arc::new(std::sync::Mutex::new(None));

            // Register the IPC state so command handlers can access it.
            app.manage(IpcState {
                orchestrator: orchestrator.clone(),
                settings: settings_handle.clone(),
                meetings_dir: notes_meetings_dir.clone(),
                index_db_path,
                index: index.clone(),
                event_tx: ipc_event_tx.clone(),
                summariser: summariser_cell.clone(),
                tool_registry,
                chat_in_flight: chat_in_flight.clone(),
                chat_cancel: chat_cancel.clone(),
                translate_in_flight: Arc::new(std::sync::Mutex::new(
                    std::collections::HashSet::new(),
                )),
                mcp_info: mcp_info.clone(),
            });

            // Log the GPU probe + the resolved default plan at startup so the
            // (estimated) VRAM thresholds in `resolve_gpu_plan` can be validated
            // against real hardware. Best-effort + off the main thread: the probe
            // self-inits the llama backend and can block, so it rides a background
            // `async_runtime::spawn` (NOT a bare `tokio::spawn` — `setup` has no
            // entered runtime). Mirrors the ASR-prewarm spawn above.
            {
                let probe_handle = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    probe_handle.state::<IpcState>().log_gpu_probe();
                });
            }

            // Preload the shared summary/chat LLM in the background so the first
            // Summarise / chat is instant (the model loads in ~seconds the first
            // time otherwise). Gated on `settings.preload_summariser` (default ON)
            // and best-effort: it NEVER downloads — a not-yet-fetched model is
            // left to load on first use. Mirrors the ASR prewarm above;
            // `async_runtime::spawn` because `setup` has no entered tokio runtime.
            {
                let preload_handles = app.state::<IpcState>().chat_handles();
                tauri::async_runtime::spawn(async move {
                    preload_handles.maybe_preload_summariser().await;
                });
            }

            // --- Phase 10: the MCP server + the inter-agent bridge ------------
            //
            // Compiled only in the `connected` build (the free build elides the
            // entire block — no mcp-server crate, no rmcp, no listening socket).
            // `settings.mcp_enabled` (off by default) gates the listener at
            // startup; toggling it at runtime starts or stops the server live
            // (the settings watcher task below drives that). Changing the port
            // or write-tools setting is still restart-required (the running
            // server was built with those values at start time).
            //
            // Achieved-vs-desired state: the watcher derives "is the server
            // running?" from the presence of `shutdown_inner` (Some = running,
            // None = not running), not from `mcp_enabled`. This lets a failed
            // start (bind error or summariser failure) be retried by the user
            // toggling the setting off then on again: desired=true +
            // achieved=false (None) lets `do_start_mcp_server` fire again.
            #[cfg(feature = "connected")]
            {
                // Per-instance handles held in managed state so the settings
                // watcher (which holds an Arc clone) can stop the server
                // without routing through IpcState.
                let shutdown_state: Arc<McpShutdownState> =
                    Arc::new(McpShutdownState(std::sync::Mutex::new(None)));
                app.manage(shutdown_state.clone());

                let settings_now = settings_handle.current();

                // Capture all start-params outside the watcher task. The
                // startup path runs `do_start_mcp_server` inline (not spawned)
                // inside the watcher task body below, which is itself spawned
                // once. The initial start at boot goes through the same watcher
                // task: we seed `prev_enabled = false` so the initial check
                // fires if `mcp_enabled` is true.
                let mut settings_rx = settings_handle.subscribe();

                let watcher_orchestrator = orchestrator.clone();
                let watcher_index = index.clone();
                let watcher_meetings_dir = notes_meetings_dir.clone();
                let watcher_event_tx = ipc_event_tx.clone();
                let watcher_settings = settings_handle.clone();
                let watcher_summariser = summariser_cell.clone();
                let watcher_chat_in_flight = chat_in_flight.clone();
                let watcher_app_data_dir = app_data_dir.clone();
                let watcher_mcp_info = mcp_info.clone();
                let watcher_shutdown = shutdown_state.clone();

                // Whether the server is desired on at the point the watcher
                // last acted. Seeded to `false` so a boot with `mcp_enabled =
                // true` is treated as a false→true transition and fires the
                // initial start.
                let mut prev_desired = false;

                if settings_now.mcp_enabled {
                    tracing::info!(
                        target: "app-main",
                        "MCP server enabled at startup — will start in watcher task"
                    );
                } else {
                    tracing::info!(
                        target: "app-main",
                        "MCP server disabled (settings.mcp_enabled = false)"
                    );
                }

                tauri::async_runtime::spawn(async move {
                    // Handle the initial state: if enabled at boot, act
                    // immediately without waiting for a settings change.
                    let initial_desired = watcher_settings.current().mcp_enabled;
                    if initial_desired && !prev_desired {
                        prev_desired = true;
                        tracing::info!(
                            target: "app-main",
                            "MCP server starting (boot)"
                        );
                        do_start_mcp_server(McpStartParams {
                            orchestrator: watcher_orchestrator.clone(),
                            index: watcher_index.clone(),
                            meetings_dir: watcher_meetings_dir.clone(),
                            event_tx: watcher_event_tx.clone(),
                            settings: watcher_settings.clone(),
                            summariser: watcher_summariser.clone(),
                            chat_in_flight: watcher_chat_in_flight.clone(),
                            app_data_dir: watcher_app_data_dir.clone(),
                            mcp_info: watcher_mcp_info.clone(),
                            shutdown_state: watcher_shutdown.clone(),
                        })
                        .await;
                    }

                    // Watch for `mcp_enabled` changes and start/stop the server
                    // live. Port and write-tools changes are not reacted to here
                    // — those still require a restart (the running server was
                    // built with the port/allow_writes values at start time).
                    loop {
                        if settings_rx.changed().await.is_err() {
                            // SettingsHandle dropped (app exit).
                            break;
                        }
                        let new_desired = settings_rx.borrow_and_update().mcp_enabled;
                        if new_desired == prev_desired {
                            continue;
                        }
                        prev_desired = new_desired;

                        if new_desired {
                            // desired=on. Always attempt start regardless of
                            // achieved state — if a previous start failed
                            // (handles slot is None), this is the retry path.
                            tracing::info!(
                                target: "app-main",
                                "MCP server enabled via settings toggle"
                            );
                            do_start_mcp_server(McpStartParams {
                                orchestrator: watcher_orchestrator.clone(),
                                index: watcher_index.clone(),
                                meetings_dir: watcher_meetings_dir.clone(),
                                event_tx: watcher_event_tx.clone(),
                                settings: watcher_settings.clone(),
                                summariser: watcher_summariser.clone(),
                                chat_in_flight: watcher_chat_in_flight.clone(),
                                app_data_dir: watcher_app_data_dir.clone(),
                                mcp_info: watcher_mcp_info.clone(),
                                shutdown_state: watcher_shutdown.clone(),
                            })
                            .await;
                        } else {
                            // desired=off. Take the handles (if any) and stop.
                            let maybe_handles = watcher_shutdown
                                .0
                                .lock()
                                .expect("mcp_shutdown poisoned")
                                .take();

                            if let Some(handles) = maybe_handles {
                                // Signal the accept loop to stop.
                                let _ = handles.shutdown_tx.send(true);
                                // Await the completion receiver so the port is
                                // fully released before we clear mcp_info or
                                // before a subsequent re-enable can rebind.
                                // Bounded at 5 s (a stalled session should not
                                // block shutdown indefinitely).
                                match tokio::time::timeout(
                                    std::time::Duration::from_secs(5),
                                    handles.done_rx,
                                )
                                .await
                                {
                                    Ok(_) => {
                                        tracing::info!(
                                            target: "app-main",
                                            "MCP server stopped (accept loop exited)"
                                        );
                                    }
                                    Err(_) => {
                                        tracing::warn!(
                                            target: "app-main",
                                            "MCP server stop timed out after 5 s \
                                             — proceeding anyway"
                                        );
                                    }
                                }
                            }

                            // Clear the info slot and notify the UI. Done after
                            // awaiting completion so the UI clears only once the
                            // port is actually free.
                            *watcher_mcp_info
                                .lock()
                                .expect("mcp_info poisoned") = None;
                            let _ = watcher_event_tx.send(
                                minutist_common::AppEvent::McpServerStopped,
                            );
                            tracing::info!(
                                target: "app-main",
                                "MCP server stopped via settings toggle"
                            );
                        }
                    }
                });
            }

            // Build the tray icon.
            build_tray(app)?;

            // Respect start_hidden setting: show the window unless the user
            // prefers the app to start in the tray.
            let start_hidden = {
                let state = app.state::<IpcState>();
                state.settings.current().start_hidden
            };

            if !start_hidden {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }

            tracing::info!(target: "app-main", "setup complete");
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                // Hide rather than close so the user can restore via the tray.
                let _ = window.hide();
                api.prevent_close();
                tracing::debug!(
                    target: "app-main",
                    "CloseRequested intercepted — window hidden, close prevented"
                );
            }
        })
        .run(tauri::generate_context!())
        .expect("error running minutist");
}

/// Handle a `meetingasset:` URI-scheme request: serve a note image asset's
/// bytes from the open meeting's `assets/` directory.
///
/// The request URI path is `/<meeting_id>/<filename>` (as produced by
/// `convertFileSrc(<meeting_id>/<filename>, "meetingasset")` on the frontend).
/// `meetings_dir` is read from the managed [`IpcState`] (the same root every
/// other command resolves against), so there is no separate path computation.
///
/// All parsing + validation + the path-traversal guard live in
/// `ipc_bridge::resolve_note_asset` (which owns the `persistence` edge —
/// `app-main` does not depend on `persistence`). This handler only shapes the
/// HTTP response: a `200` with the inferred `Content-Type` on success, or an
/// empty `404` on ANY failure (malformed path, non-UUID id, traversal attempt,
/// or a missing file) so no detail leaks to the webview.
fn serve_note_asset(
    ctx: tauri::UriSchemeContext<'_, tauri::Wry>,
    request: tauri::http::Request<Vec<u8>>,
) -> tauri::http::Response<Vec<u8>> {
    use tauri::http::{header, Response, StatusCode};

    let meetings_dir = {
        let state = ctx.app_handle().state::<IpcState>();
        state.meetings_dir.clone()
    };

    match ipc_bridge::resolve_note_asset(&meetings_dir, request.uri().path()) {
        Ok(asset) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, asset.content_type)
            // The asset is immutable (content-addressed filename), so it is
            // safe for the webview to cache aggressively.
            .header(header::CACHE_CONTROL, "private, max-age=31536000, immutable")
            .body(asset.bytes)
            .unwrap_or_else(|_| Response::new(Vec::new())),
        Err(e) => {
            tracing::debug!(
                target: "app-main",
                path = %request.uri().path(),
                "meetingasset request rejected: {e}"
            );
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Vec::new())
                .unwrap_or_else(|_| Response::new(Vec::new()))
        }
    }
}

/// Placeholder 32×32 RGBA tray icon — solid blue (#1E64B4).
///
/// Generated at build time.  Replaced by final art in Phase 7.
/// Each pixel is [R=30, G=100, B=180, A=255] repeated 1024 times.
const TRAY_ICON_RGBA: [u8; 32 * 32 * 4] = {
    let mut buf = [0u8; 32 * 32 * 4];
    let mut i = 0;
    while i < 32 * 32 {
        buf[i * 4] = 30;
        buf[i * 4 + 1] = 100;
        buf[i * 4 + 2] = 180;
        buf[i * 4 + 3] = 255;
        i += 1;
    }
    buf
};

fn tray_icon() -> tauri::image::Image<'static> {
    tauri::image::Image::new(&TRAY_ICON_RGBA, 32, 32)
}

/// Construct the tray icon and attach menu event + icon-click handlers.
fn build_tray(app: &tauri::App) -> tauri::Result<()> {
    let open_item = MenuItem::with_id(app, "open", "Open minutist", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&open_item, &quit_item])?;

    TrayIconBuilder::new()
        .icon(tray_icon())
        .tooltip("minutist")
        .menu(&menu)
        // Left-click shows the menu on platforms where that's conventional.
        // An explicit on_tray_icon_event handler below also shows the window
        // on left-click so the app is always reachable.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => {
                show_main_window(app);
            }
            "quit" => {
                tracing::info!(target: "app-main", "quit via tray menu");
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|_tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                // show_main_window needs the AppHandle; obtain it from the tray.
                // The tray event callback doesn't directly give us the app handle,
                // but TrayIcon<R> implements Manager<R>.
                // We use _tray.app_handle() which is available in Tauri 2.
                show_main_window(_tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

/// Show and focus the main window, creating it if it no longer exists.
fn show_main_window(app: &impl Manager<tauri::Wry>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}
