//! Event forwarding from the orchestrator broadcast channel to the Tauri
//! webview.
//!
//! `AppEventPayload` is a `#[serde(transparent)]` newtype around
//! `common::AppEvent`. `common` derives `specta::Type` (gated on the
//! `specta` feature, which `ipc-bridge` enables) so `AppEvent` is directly
//! usable as the payload type — no JSON round-trip, no parallel mirror.
//!
//! The wire name on the Tauri event channel is `"app-event-payload"`
//! (derived from this struct's name by `tauri_specta::Event`).
//!
//! # Forwarder lifecycle
//!
//! `spawn_event_forwarder` starts a long-lived tokio task that:
//!   1. Subscribes to the orchestrator's `broadcast::Receiver<AppEvent>`.
//!   2. On each event, wraps it in `AppEventPayload` (zero-cost) and emits.
//!   3. Handles `RecvError::Lagged` with a `tracing::warn!` and continues.
//!   4. Exits when the sender is dropped (orchestrator shut down).

use std::sync::Arc;

use meeting_app_common::AppEvent;
use orchestrator::Orchestrator;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::AppHandle;
use tauri_specta::Event;
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// AppEventPayload — specta-typed wrapper around AppEvent
// ---------------------------------------------------------------------------

/// Typed wrapper that gives `AppEvent` a stable tauri-specta event name.
///
/// The `Event` derive assigns the wire name `"app-event-payload"`. The
/// `#[serde(transparent)]` attribute means the wrapper is invisible on the
/// wire — the JSON shape is exactly `AppEvent`'s own tagged-enum
/// representation.
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
#[serde(transparent)]
pub struct AppEventPayload(pub AppEvent);

// ---------------------------------------------------------------------------
// Event forwarder
// ---------------------------------------------------------------------------

/// Spawn a long-lived task that forwards `AppEvent` broadcasts from the
/// orchestrator to the Tauri webview as typed `AppEventPayload` events.
///
/// The task runs until the orchestrator's broadcast sender is dropped.
/// `RecvError::Lagged` is handled with a `tracing::warn!`; the task
/// continues consuming events after lag.
pub fn spawn_event_forwarder(orchestrator: Arc<Orchestrator>, app_handle: AppHandle) {
    let mut rx: broadcast::Receiver<AppEvent> = orchestrator.subscribe_events();

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let payload = AppEventPayload(event);
                    if let Err(e) = payload.emit(&app_handle) {
                        tracing::warn!(
                            target: "ipc-bridge",
                            "failed to emit AppEventPayload to webview: {e}"
                        );
                    }
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!(
                        target: "ipc-bridge",
                        skipped,
                        "AppEvent forwarder lagged; skipped events"
                    );
                    // Continue — keep consuming after lag.
                }
                Err(broadcast::error::RecvError::Closed) => {
                    tracing::info!(
                        target: "ipc-bridge",
                        "AppEvent broadcast channel closed; forwarder exiting"
                    );
                    break;
                }
            }
        }
    });
}
