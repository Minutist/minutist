//! Per-crate error type for `ipc-bridge`.
//!
//! [`IpcError`] is the error type exposed on the Tauri command surface.  It
//! derives `specta::Type` so tauri-specta can generate a TypeScript binding.
//!
//! `common::AppError` does not derive `specta::Type` (the `common` crate has
//! no `specta` dependency by design).  The orphan rule prevents adding
//! `impl specta::Type for AppError` here.  `IpcError` therefore re-encodes
//! `AppError` into an equivalent, independently-typed struct that carries the
//! same information and the same serde shape.
//!
//! Downstream code converts via `IpcError::from(app_error)` or the `?`
//! operator on `AppResult<T>`.

use meeting_app_common::AppError;
use serde::{Deserialize, Serialize};
use specta::Type;
use thiserror::Error;

// ---------------------------------------------------------------------------
// IpcError — the Tauri command surface error type
// ---------------------------------------------------------------------------

/// Error type returned from every Tauri command in `ipc-bridge`.
///
/// Carries the same discriminants as `common::AppError` and serialises to the
/// same JSON shape (`{"code": "...", ...}`), so the TypeScript binding is
/// stable even though the derive lives here rather than in `common`.
#[derive(Debug, Clone, Error, Serialize, Deserialize, Type)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum IpcError {
    #[error("I/O error: {context}")]
    Io { context: String },
    #[error("model {model_id} failed to load: {context}")]
    ModelLoad { model_id: String, context: String },
    #[error("model {model_id} not found in registry")]
    ModelNotFound { model_id: String },
    #[error("model download failed: {context}")]
    ModelDownload { context: String },
    #[error("inference failed in {backend}: {context}")]
    Inference { backend: String, context: String },
    #[error("invalid input: {context}")]
    InvalidInput { context: String },
    #[error("operation cancelled")]
    Cancelled,
    #[error("operation not supported: {context}")]
    Unsupported { context: String },
    #[error("internal error: {context}")]
    Internal { context: String },
}

impl From<AppError> for IpcError {
    fn from(e: AppError) -> Self {
        match e {
            AppError::Io { context } => IpcError::Io { context },
            AppError::ModelLoad { model_id, context } => IpcError::ModelLoad { model_id, context },
            AppError::ModelNotFound { model_id } => IpcError::ModelNotFound { model_id },
            AppError::ModelDownload { context } => IpcError::ModelDownload { context },
            AppError::Inference { backend, context } => IpcError::Inference { backend, context },
            AppError::InvalidInput { context } => IpcError::InvalidInput { context },
            AppError::Cancelled => IpcError::Cancelled,
            AppError::Unsupported { context } => IpcError::Unsupported { context },
            AppError::Internal { context } => IpcError::Internal { context },
        }
    }
}

// ---------------------------------------------------------------------------
// Per-crate Error (thiserror) — for errors that originate inside ipc-bridge
// ---------------------------------------------------------------------------

/// Internal error variants that originate inside `ipc-bridge` itself.
///
/// The only paths that create these are the event-forwarder task and a future
/// export-bindings helper.  All public functions convert to [`IpcError`] or
/// [`AppError`] at the boundary.
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

impl From<Error> for IpcError {
    fn from(e: Error) -> IpcError {
        IpcError::Internal {
            context: e.to_string(),
        }
    }
}
