//! Per-crate error type for `ipc-bridge`.
//!
//! [`Error`] carries the error variants that originate inside `ipc-bridge`
//! itself (currently just the event-forwarder's emit failure). It converts to
//! `common::AppError` via [`From<Error> for AppError`], which is the error
//! type that crosses every crate boundary in the workspace and is what every
//! Tauri command returns (`AppResult<T>` = `Result<T, AppError>`). `AppError`
//! derives `specta::Type` behind `common`'s optional `specta` feature, which
//! `ipc-bridge` enables (see `crates/ipc-bridge/Cargo.toml`), so it reaches
//! TypeScript directly — on both the command surface and
//! `AppEvent::ErrorOccurred` on the event bus — with no mirror type needed.

use minutist_common::AppError;
use thiserror::Error;

/// Internal error variants that originate inside `ipc-bridge` itself.
///
/// The only paths that create these are the event-forwarder task and a future
/// export-bindings helper. All public functions convert to [`AppError`] at the
/// boundary.
#[derive(Debug, Error)]
pub enum Error {
    #[error("event forwarder failed to emit: {0}")]
    EventEmit(#[from] tauri::Error),
}

impl From<Error> for AppError {
    fn from(e: Error) -> AppError {
        AppError::Internal {
            context: e.to_string(),
        }
    }
}
