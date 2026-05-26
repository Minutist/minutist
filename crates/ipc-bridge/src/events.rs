//! Event forwarding from the orchestrator broadcast channel to the Tauri
//! webview.
//!
//! # Design note — AppEvent wrapper
//!
//! `common::AppEvent` does not derive `specta::Type` because the `common`
//! crate has no `specta` dependency (by architecture design).  The orphan
//! rule prevents adding `impl specta::Type for AppEvent` here.
//!
//! `AppEventPayload` is a specta-typed wrapper defined in `specta_types` as
//! `AppEventType`.  It has identical serde shape to `AppEvent` (same `tag =
//! "kind"`, same `rename_all = "snake_case"`) so JSON round-trips correctly.
//! The `Event` derive on `AppEventPayload` gives tauri-specta the type
//! information it needs to generate the TypeScript listener.
//!
//! The event name on the wire is `"app-event-payload"` (derived from the
//! struct name by the `Event` macro).
//!
//! # Architecture note
//!
//! The spec intended `AppEvent` itself to be registered via `collect_events!`.
//! That requires `common` to gain a `specta` dependency and `AppEvent` to
//! derive `specta::Type`.  Until that architecture-owner change lands, this
//! wrapper pattern is the only viable approach.  See `specta_types.rs` for
//! details.
//!
//! # Forwarder lifecycle
//!
//! `spawn_event_forwarder` starts a long-lived tokio task that:
//!   1. Subscribes to the orchestrator's `broadcast::Receiver<AppEvent>`.
//!   2. On each event, converts `AppEvent` to `AppEventPayload` and emits it.
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

use crate::specta_types::AppEventType;

// ---------------------------------------------------------------------------
// AppEventPayload — specta-typed wrapper around AppEvent
// ---------------------------------------------------------------------------

/// Typed wrapper that gives `AppEventType` a stable tauri-specta event name.
///
/// The `Event` derive assigns the wire name `"app-event-payload"`.
/// The serde shape is identical to `common::AppEvent` because `AppEventType`
/// is a full mirror of `AppEvent`.
///
/// Once `common` gains `specta::Type` derives, this type can be replaced
/// by registering `AppEvent` directly in `collect_events!`.
#[derive(Debug, Clone, Serialize, Deserialize, Type, Event)]
#[serde(transparent)]
pub struct AppEventPayload(pub AppEventType);

impl TryFrom<AppEvent> for AppEventPayload {
    type Error = serde_json::Error;

    /// Convert by round-tripping through JSON.
    ///
    /// Both `AppEvent` and `AppEventType` have identical serde representations,
    /// so this is lossless.  The indirection is needed because the orphan rule
    /// prevents a direct `From<AppEvent> for AppEventType` implementation.
    fn try_from(event: AppEvent) -> Result<Self, Self::Error> {
        let json = serde_json::to_string(&event)?;
        let typed: AppEventType = serde_json::from_str(&json)?;
        Ok(AppEventPayload(typed))
    }
}

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
                Ok(event) => match AppEventPayload::try_from(event) {
                    Ok(payload) => {
                        if let Err(e) = payload.emit(&app_handle) {
                            tracing::warn!(
                                target: "ipc-bridge",
                                "failed to emit AppEventPayload to webview: {e}"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "ipc-bridge",
                            "failed to convert AppEvent to AppEventPayload: {e}"
                        );
                    }
                },
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
