//! Auto-update integration (Phase 7), driven from Rust via `UpdaterExt`.
//!
//! Owned by app-main per `architecture/cross-cutting.md` "Auto-update". On
//! startup we check the configured endpoint and, if a newer release exists,
//! emit `AppEvent::UpdateAvailable` so the webview can prompt. When the user
//! accepts, the webview emits the `updater://apply` event; we then download
//! (emitting `AppEvent::UpdateProgress`), install, and relaunch.
//!
//! All updater calls are GUARDED: with the committed default `plugins.updater`
//! config (empty `endpoints`, empty `pubkey`), `check()` errors and we log +
//! skip, so dev and unsigned builds run normally. Release builds set the
//! endpoints + minisign `pubkey` (see `architecture/cross-cutting.md`
//! "Auto-update").

use meeting_app_common::{AppError, AppEvent, AppResult};
use tauri::{AppHandle, Listener};
use tauri_plugin_updater::UpdaterExt;
use tokio::sync::broadcast::Sender;

/// Webview → app-main signal that the user accepted the available update.
const APPLY_EVENT: &str = "updater://apply";

/// Wire the updater: register the apply-on-accept listener and spawn the
/// one-shot startup check. Never blocks; never fails the app when the updater
/// is unconfigured (the check is a logged no-op in that case).
pub fn start(app: &AppHandle, event_tx: Sender<AppEvent>) {
    register_apply_listener(app, event_tx.clone());

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        match check(&app).await {
            Ok(Some((version, notes))) => {
                tracing::info!(target: "app-main", version = %version, "update available");
                let _ = event_tx.send(AppEvent::UpdateAvailable { version, notes });
            }
            Ok(None) => tracing::info!(target: "app-main", "updater: already up to date"),
            // Unconfigured / offline / endpoint error: a non-fatal skip.
            Err(e) => tracing::info!(target: "app-main", "updater check skipped: {e}"),
        }
    });
}

/// Check the configured endpoint for a newer release, returning
/// `(version, release_notes)` when one exists.
async fn check(app: &AppHandle) -> AppResult<Option<(String, Option<String>)>> {
    let updater = app.updater().map_err(|e| AppError::Internal {
        context: format!("updater unavailable: {e}"),
    })?;
    let update = updater.check().await.map_err(|e| AppError::Internal {
        context: format!("update check failed: {e}"),
    })?;
    Ok(update.map(|u| (u.version, u.body)))
}

/// Listen for the webview's accept signal and apply the update when it fires.
fn register_apply_listener(app: &AppHandle, event_tx: Sender<AppEvent>) {
    let app_handle = app.clone();
    app.listen_any(APPLY_EVENT, move |_event| {
        let app = app_handle.clone();
        let event_tx = event_tx.clone();
        tauri::async_runtime::spawn(async move {
            if let Err(e) = apply(&app, &event_tx).await {
                tracing::warn!(target: "app-main", "update apply failed: {e}");
                let _ = event_tx.send(AppEvent::ErrorOccurred { error: e });
            }
        });
    });
}

/// Download (emitting `UpdateProgress`), install, and relaunch.
async fn apply(app: &AppHandle, event_tx: &Sender<AppEvent>) -> AppResult<()> {
    let updater = app.updater().map_err(|e| AppError::Internal {
        context: format!("updater unavailable: {e}"),
    })?;
    let maybe_update = updater.check().await.map_err(|e| AppError::Internal {
        context: format!("update check failed: {e}"),
    })?;
    let Some(update) = maybe_update else {
        tracing::info!(target: "app-main", "apply requested but no update is available");
        return Ok(());
    };

    let mut downloaded: u64 = 0;
    update
        .download_and_install(
            |chunk, total| {
                downloaded += chunk as u64;
                let _ = event_tx.send(AppEvent::UpdateProgress {
                    downloaded_bytes: downloaded,
                    total_bytes: total,
                });
            },
            || tracing::info!(target: "app-main", "update downloaded; installing"),
        )
        .await
        .map_err(|e| AppError::Internal {
            context: format!("update download/install failed: {e}"),
        })?;

    tracing::info!(target: "app-main", "update installed; relaunching");
    app.restart();
}
