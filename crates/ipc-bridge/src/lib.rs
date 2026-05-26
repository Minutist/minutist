//! `ipc-bridge` — Tauri command + event surface for meeting-app.
//!
//! This is the **only** crate in the workspace that imports `tauri::*`.
//! Every other crate is free of Tauri imports, which keeps them testable
//! without a running Tauri app.
//!
//! ## Phase 1 commands (8 total)
//!
//! | Command | Returns |
//! |---|---|
//! | `list_devices` | `Vec<AudioDeviceType>` |
//! | `start_recording` | `MeetingIdType` |
//! | `pause_recording` | `()` |
//! | `resume_recording` | `()` |
//! | `stop_recording` | `MeetingMetaType` |
//! | `get_recording_state` | `RecordingStateType` |
//! | `get_settings` | `SettingsType` |
//! | `update_settings` | `()` |
//!
//! All commands return `Result<T, IpcError>`.
//!
//! ## Type mirrors
//!
//! Types in command signatures use specta-typed mirrors from `specta_types`
//! (e.g. `AudioDeviceType`, `MeetingIdType`).  This is necessary because
//! `common` and `settings` do not depend on `specta`, so their types don't
//! implement `specta::Type`.  When an architecture-owner commit adds
//! `specta::Type` derives to `common` / `settings`, the mirror types in
//! `specta_types.rs` and the conversion calls in `commands.rs` can be removed.
//!
//! ## Events
//!
//! `AppEventPayload` is a specta-typed wrapper around `AppEventType`, a full
//! mirror of `common::AppEvent`.  The event name on the wire is
//! `"app-event-payload"`.  `spawn_event_forwarder` starts a task that
//! subscribes to the orchestrator's broadcast channel and emits each event to
//! all Tauri windows via JSON round-trip through `AppEventType`.
//!
//! ## Tracing
//!
//! All log calls use `target: "ipc-bridge"`.

pub mod commands;
pub mod error;
pub mod events;
pub mod specta_types;

use std::sync::Arc;

use orchestrator::Orchestrator;
use settings::SettingsHandle;
use tauri_specta::{collect_commands, collect_events, Builder};

pub use error::{Error, IpcError};
pub use events::{spawn_event_forwarder, AppEventPayload};

// ---------------------------------------------------------------------------
// IpcState — Tauri managed state
// ---------------------------------------------------------------------------

/// Tauri managed state shared across all command handlers.
///
/// `app-main` constructs this and passes it to `tauri::Builder::manage`.
pub struct IpcState {
    pub orchestrator: Arc<Orchestrator>,
    pub settings: SettingsHandle,
}

// ---------------------------------------------------------------------------
// bindings_builder — shared builder for app-main and the export helper
// ---------------------------------------------------------------------------

/// Construct a `tauri_specta::Builder` pre-loaded with all Phase 1 commands
/// and the `AppEventPayload` event.
///
/// Both `app-main` (to build the invoke handler) and a bindings-export helper
/// binary can call this function to get the same builder, ensuring the
/// generated TypeScript bindings are always in sync with the runtime handler.
///
/// # Usage
///
/// ```rust,ignore
/// let builder = ipc_bridge::bindings_builder();
///
/// // In app-main — wire into Tauri:
/// tauri::Builder::default()
///     .manage(ipc_state)
///     .invoke_handler(builder.invoke_handler())
///     .setup(move |app| { builder.mount_events(app); Ok(()) });
///
/// // In a bindings-export binary:
/// builder
///     .export(specta_typescript::Typescript::default(), "ui/src/ipc/bindings.ts")
///     .expect("export failed");
/// ```
pub fn bindings_builder() -> Builder<tauri::Wry> {
    Builder::<tauri::Wry>::new()
        .commands(collect_commands![
            commands::list_devices,
            commands::start_recording,
            commands::pause_recording,
            commands::resume_recording,
            commands::stop_recording,
            commands::get_recording_state,
            commands::get_settings,
            commands::update_settings,
        ])
        .events(collect_events![AppEventPayload])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tauri_specta::Event;

    /// Verify that `bindings_builder()` produces a builder with all 8 Phase 1
    /// command names registered, by inspecting the TypeScript export.
    ///
    /// tauri-specta rc.21 does not expose the internal command list publicly.
    /// We use `export_str` to generate the TypeScript bindings string and scan
    /// it for each expected command name.  Each command appears in the TS
    /// as a string literal in the `invoke` call.
    ///
    /// `BigIntExportBehavior::Number` is used to allow `u64` fields (e.g.,
    /// timestamps and byte counts) to export as TypeScript `number` rather
    /// than erroring.  This matches the Handy project's pattern per Phase 1
    #[test]
    fn bindings_builder_registers_all_eight_commands() {
        use specta_typescript::{BigIntExportBehavior, Typescript};

        let builder = bindings_builder();
        let ts = builder
            .export_str(Typescript::default().bigint(BigIntExportBehavior::Number))
            .expect("export_str should succeed for a correctly-configured builder");

        // Each command appears as a string literal in the `invoke(...)` call.
        let expected = [
            "list_devices",
            "start_recording",
            "pause_recording",
            "resume_recording",
            "stop_recording",
            "get_recording_state",
            "get_settings",
            "update_settings",
        ];

        for name in &expected {
            assert!(
                ts.contains(name),
                "expected command '{name}' not found in generated TypeScript:\n{ts}"
            );
        }
    }

    /// Verify that `AppEventPayload` is registered in the builder's event
    /// registry, by checking its `Event::NAME` constant appears in the TS
    /// export.
    #[test]
    fn bindings_builder_registers_app_event_payload() {
        use specta_typescript::{BigIntExportBehavior, Typescript};

        let builder = bindings_builder();
        let ts = builder
            .export_str(Typescript::default().bigint(BigIntExportBehavior::Number))
            .expect("export_str should succeed");

        let event_name = events::AppEventPayload::NAME;
        assert!(
            ts.contains(event_name),
            "AppEventPayload event '{event_name}' not found in generated TypeScript:\n{ts}"
        );
    }
}
