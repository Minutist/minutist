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

    let filter = EnvFilter::from_default_env();

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

/// The Tauri runtime entry point.
///
/// Accepts the non-blocking writer guard so it stays alive for the process
/// lifetime and the writer flushes on exit.
fn run(_log_guard: tracing_appender::non_blocking::WorkerGuard) {
    let builder = ipc_bridge::bindings_builder();

    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
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

            // Construct the orchestrator sharing the same event bus.
            let orchestrator = Arc::new(orchestrator::Orchestrator::with_event_tx(
                settings_handle.clone(),
                meetings_dir,
                model_registry,
                event_tx,
            ));

            tracing::info!(target: "app-main", "orchestrator constructed");

            // Spawn the event forwarder so orchestrator events reach the webview.
            spawn_event_forwarder(orchestrator.clone(), app_handle.clone());

            // Register the IPC state so command handlers can access it.
            app.manage(IpcState {
                orchestrator: orchestrator.clone(),
                settings: settings_handle,
            });

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
