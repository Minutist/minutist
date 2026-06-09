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

    let file_appender = tracing_appender::rolling::daily(&log_dir, "meeting-app.log");
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
    let base = dirs_next();
    base.join("logs")
}

fn dirs_next() -> PathBuf {
    // Attempt to use the same base path as Tauri's AppData directory.
    // tauri uses `dirs` crate internally; we use std env as a fallback.
    let identifier = "net.alelec.meeting-app";

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
        if !name.starts_with("meeting-app.log") {
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
/// `MEETING_APP_SILERO_PATH` so `vad-chunker::default_model_path()` finds it in
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
                std::env::set_var("MEETING_APP_SILERO_PATH", &path);
                tracing::info!(
                    target: "app-main",
                    path = %path.display(),
                    "bundled Silero VAD model resolved; MEETING_APP_SILERO_PATH set"
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
        "no bundled Silero VAD model found (dev run); leaving MEETING_APP_SILERO_PATH unset \
         so vad-chunker uses its source-tree fallback"
    );
}

/// Resolve the MCP bearer token (Phase 10 §4.2): read the persisted token, or
/// generate + persist a fresh 256-bit CSPRNG one on first enable.
///
/// **Storage.** v1 stores the token in `{app-data}/mcp_token`. On Unix the file
/// is CREATED with mode `0o600` atomically (`OpenOptions().mode(0o600)`), so
/// there is no write-then-chmod window in which the token is world-readable
/// (S3). On Windows the file inherits the app-data directory's ACL — the
/// app-data dir is a per-user location, but this code does NOT additionally
/// tighten the file ACL, so the owner-only guarantee holds only on Unix; the
/// Windows wording is scoped accordingly. A documented follow-up hardens this to
/// the OS keychain (e.g. the `keyring` crate); Tauri 2 ships no built-in
/// keychain API, and pulling a cross-platform keychain dependency (with its own
/// platform build concerns) is deferred so it can be reviewed on its own. The
/// token is high-entropy regardless of the at-rest store, and the loopback bind
/// + Host/Origin checks are the primary controls. The token is NEVER logged.
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
/// atomically on Unix (no write-then-chmod window — S3). On Windows the file
/// inherits the parent directory's ACL (no extra tightening in v1).
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
            let meetings_dir = app_data_dir.join("meetings");
            let logs_dir = app_data_dir.join("logs");
            std::fs::create_dir_all(&logs_dir).expect("failed to create logs dir");

            tracing::info!(
                target: "app-main",
                app_data = %app_data_dir.display(),
                "app-data directory resolved"
            );

            // Resolve the bundled Silero VAD model and plumb its path to
            // `vad-chunker` via `MEETING_APP_SILERO_PATH` (see
            // architecture/cross-cutting.md "Model lifecycle — Exception:
            // Silero VAD"). MUST run before the orchestrator is constructed and
            // before any recording, because `vad-chunker::default_model_path()`
            // reads this env var at chunker-open time. A no-op in a dev run with
            // no bundle (the source-tree fallback handles that path).
            resolve_silero_model(&app_handle);

            // Prune log files older than 7 days.
            cleanup_old_logs(&logs_dir, 7);

            // Construct settings handle backed by the JSON file store.
            let settings_store = JsonFileStore::new(app_data_dir.join("settings.store"));
            let settings_handle =
                SettingsHandle::new(settings_store).expect("failed to initialise settings");

            tracing::info!(target: "app-main", "settings handle constructed");

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
            let models_root = app_data_dir.join("models");
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
                ipc_bridge::open_meeting_index(&app_data_dir, &notes_meetings_dir);

            tracing::info!(target: "app-main", "meeting index opened");

            // Spawn the event forwarder so orchestrator events reach the webview.
            spawn_event_forwarder(orchestrator.clone(), app_handle.clone());

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

            // The MCP server info slot (URL + token), filled below when the MCP
            // server is enabled + bound. `get_mcp_server_info` reads it.
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
                mcp_info: mcp_info.clone(),
            });

            // --- Phase 10: the MCP server + the inter-agent bridge ------------
            //
            // Gated on `settings.mcp_enabled` (off by default). When enabled,
            // spawn the loopback Streamable HTTP server from `setup()` via
            // `tauri::async_runtime::spawn` (NOT a bare `tokio::spawn` — `setup`
            // runs on the main thread with no entered tokio runtime; a bare spawn
            // panics — see cross-cutting "Async runtime"). Toggling the setting at
            // runtime is a documented restart-required for v1.
            {
                let settings_now = settings_handle.current();
                if settings_now.mcp_enabled {
                    // The inter-agent driver owns the receiver + the chat turn
                    // (keeping the chat dependency in `ipc-bridge`, not
                    // `mcp-server`); it hands back the SENDER for the bridge tool.
                    let chat_handles = ipc_bridge::ChatHandles {
                        orchestrator: orchestrator.clone(),
                        index: index.clone(),
                        meetings_dir: notes_meetings_dir.clone(),
                        event_tx: ipc_event_tx.clone(),
                        settings: settings_handle.clone(),
                        summariser: summariser_cell.clone(),
                    };
                    // The bridged turn is bounded by the SAME MCP write gate as a
                    // direct `tools/call` (S1), so an external caller cannot reach
                    // a destructive op (retranscribe/rediarize) through the bridge.
                    let allow_writes = settings_now.mcp_write_tools;
                    let bridge_tx = ipc_bridge::spawn_inter_agent_driver(
                        chat_handles,
                        chat_in_flight.clone(),
                        allow_writes,
                    );

                    // The MCP registry is `v1(true)` — includes
                    // `send_to_internal_agent` — and its context carries the
                    // bridge sender. Separate from the UI's `v1(false)` registry.
                    let mcp_registry = Arc::new(agent_tools::ToolRegistry::v1(true));

                    // The bearer token: resolve from the on-disk token file, or
                    // generate + persist a fresh 256-bit one on first enable.
                    let token = resolve_mcp_token(&app_data_dir);
                    let port = settings_now.mcp_port;

                    // The MCP ToolContext needs the held summariser. Build it
                    // lazily inside the spawned task (so a slow first model load
                    // does not block setup); resolve it via the chat handles.
                    let mcp_handles = ipc_bridge::ChatHandles {
                        orchestrator: orchestrator.clone(),
                        index: index.clone(),
                        meetings_dir: notes_meetings_dir.clone(),
                        event_tx: ipc_event_tx.clone(),
                        settings: settings_handle.clone(),
                        summariser: summariser_cell.clone(),
                    };
                    let mcp_event_tx = ipc_event_tx.clone();
                    // A second clone of the bus for the listening-event emit
                    // (the existing forwarder relays it to the webview).
                    let listening_event_tx = ipc_event_tx.clone();
                    let mcp_info_for_task = mcp_info.clone();

                    tauri::async_runtime::spawn(async move {
                        // Build the ToolContext for MCP dispatch. The summariser
                        // backs `resummarise`; load it (downloads on first use).
                        let summariser = match mcp_handles.ensure_summariser().await {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::error!(
                                    target: "app-main",
                                    "MCP server not started: held summariser load failed: {e}"
                                );
                                return;
                            }
                        };
                        let ctx = Arc::new(
                            agent_tools::ToolContext::new(
                                mcp_handles.orchestrator.clone(),
                                mcp_handles.index.clone(),
                                mcp_handles.meetings_dir.clone(),
                                summariser as Arc<dyn meeting_app_common::Summariser>,
                                mcp_event_tx,
                                None, // MCP callers pass meeting_id explicitly
                            )
                            .with_inter_agent_bridge(bridge_tx),
                        );

                        // A never-firing shutdown receiver for v1: the listener
                        // lives for the process. (A future settings-toggle path
                        // would flip this watch instead of restart-required.)
                        let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                        // Keep the sender alive for the process so the watch is
                        // not seen as "dropped" (which would trigger shutdown).
                        std::mem::forget(_shutdown_tx);

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
                            Ok(bound) => {
                                let url = format!("http://{bound}{}", mcp_server::MCP_PATH);
                                // Surface the live endpoint to the Settings pane.
                                *mcp_info_for_task.lock().expect("mcp_info poisoned") =
                                    Some(ipc_bridge::McpServerInfo {
                                        url: url.clone(),
                                        token,
                                    });
                                // Emit the listening event on the shared bus; the
                                // event forwarder relays it to the webview so the
                                // MCP pane refreshes. The token is NOT carried on
                                // the event (revealed only via get_mcp_server_info).
                                let _ = listening_event_tx
                                    .send(meeting_app_common::AppEvent::McpServerListening { url });
                            }
                            Err(e) => {
                                tracing::error!(
                                    target: "app-main",
                                    "MCP server failed to start: {e}"
                                );
                            }
                        }
                    });
                } else {
                    tracing::info!(
                        target: "app-main",
                        "MCP server disabled (settings.mcp_enabled = false)"
                    );
                }
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
        .expect("error running meeting-app");
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
    let open_item = MenuItem::with_id(app, "open", "Open meeting-app", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&open_item, &quit_item])?;

    TrayIconBuilder::new()
        .icon(tray_icon())
        .tooltip("meeting-app")
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
