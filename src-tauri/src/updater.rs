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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use minutist_common::{AppError, AppEvent, AppResult};
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
///
/// `updater://apply` can fire repeatedly (a double-click, or the webview
/// re-emitting). The `in_progress` flag guards against concurrent applies:
/// only the first event that flips it `false -> true` runs `apply`; events
/// arriving while one is in flight log and return without starting a second
/// download/install/`restart` race. The flag is cleared when the in-flight
/// apply finishes (whether it succeeded, skipped, or errored), so a later
/// genuine retry is still allowed.
fn register_apply_listener(app: &AppHandle, event_tx: Sender<AppEvent>) {
    let app_handle = app.clone();
    let in_progress = Arc::new(AtomicBool::new(false));
    app.listen_any(APPLY_EVENT, move |_event| {
        // Claim the in-flight slot; bail if an apply is already running.
        if in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            tracing::info!(target: "app-main", "update apply already in progress; ignoring duplicate request");
            return;
        }

        let app = app_handle.clone();
        let event_tx = event_tx.clone();
        let in_progress = in_progress.clone();
        tauri::async_runtime::spawn(async move {
            let result = apply(&app, &event_tx).await;
            // Release the slot before reacting to the outcome. On the success
            // path `apply` calls `app.restart()` (diverging), so this only runs
            // when the apply did not relaunch — i.e. a skip or an error — at
            // which point a subsequent retry should be permitted.
            in_progress.store(false, Ordering::SeqCst);
            if let Err(e) = result {
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
