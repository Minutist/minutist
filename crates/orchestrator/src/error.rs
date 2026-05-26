//! Per-crate error type for `orchestrator`.
//!
//! `From<Error> for AppError` lets errors cross the crate boundary toward
//! the IPC bridge without leaking internal variants.

use meeting_app_common::AppError;

/// Errors that can occur inside the `orchestrator` crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An operation was requested that is invalid in the current state.
    #[error("invalid orchestrator state: {context}")]
    InvalidState { context: String },

    /// A background task panicked and was joined.
    #[error("runner task panicked: {context}")]
    TaskPanicked { context: String },
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::InvalidState { context } => AppError::InvalidInput { context },
            Error::TaskPanicked { context } => AppError::Internal { context },
        }
    }
}
