//! Per-crate error type for `ipc-bridge`.
//!
//! [`IpcError`] is the error type exposed on the Tauri command surface.  It
//! derives `specta::Type` so tauri-specta can generate a TypeScript binding.
//!
//! `common::AppError` is the error type that crosses *crate* boundaries (it
//! rides `AppEvent::ErrorOccurred` on the broadcast bus).  It *does* derive
//! `specta::Type` — behind `common`'s optional `specta` feature, which
//! `ipc-bridge` enables (see `crates/ipc-bridge/Cargo.toml`) — so it already
//! reaches TypeScript via the events surface.  [`IpcError`] is a separate,
//! command-surface error so the *command* return type and the *bus* error type
//! can evolve independently; it is hand-mirrored to carry the **same
//! discriminants and the same serde shape** as `AppError`, so the two generated
//! TypeScript unions are byte-identical.
//!
//! Because the mirror is hand-maintained, two guards keep the shapes from
//! drifting (see `tests/error_parity.rs`): the exhaustive `From<AppError> for
//! IpcError` match below makes a new `AppError` variant a compile error here
//! until a matching `IpcError` arm is added, and a serde-parity test asserts
//! every variant serialises to the same tagged JSON through both types.
//!
//! Downstream code converts via `IpcError::from(app_error)` or the `?`
//! operator on `AppResult<T>`.

use minutist_common::AppError;
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
        // Exhaustive on purpose — no catch-all `_` arm. Adding an `AppError`
        // variant in `common` is a compile error here until a matching
        // `IpcError` arm (and variant) is added, which is half the parity
        // guard for FINDING #14 (the other half is the serde-shape test in
        // `tests/error_parity.rs`). Do not collapse this into a `_` arm.
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

impl From<IpcError> for AppError {
    /// Convert back to the shared boundary error. Exhaustive (no catch-all),
    /// like the forward `From<AppError>`, so a new variant is a compile error
    /// here too. Used by the Phase-10 inter-agent driver, which reuses the chat
    /// helpers (`load_or_new_session` / `persist_session`, both `IpcError`-typed)
    /// but ultimately returns `AppResult` over the bridge oneshot.
    fn from(e: IpcError) -> AppError {
        match e {
            IpcError::Io { context } => AppError::Io { context },
            IpcError::ModelLoad { model_id, context } => AppError::ModelLoad { model_id, context },
            IpcError::ModelNotFound { model_id } => AppError::ModelNotFound { model_id },
            IpcError::ModelDownload { context } => AppError::ModelDownload { context },
            IpcError::Inference { backend, context } => AppError::Inference { backend, context },
            IpcError::InvalidInput { context } => AppError::InvalidInput { context },
            IpcError::Cancelled => AppError::Cancelled,
            IpcError::Unsupported { context } => AppError::Unsupported { context },
            IpcError::Internal { context } => AppError::Internal { context },
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
