//! Per-crate error type for the `settings` crate.

use meeting_app_common::AppError;

/// Errors that can occur within the settings crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// JSON serialisation or deserialisation failure.
    #[error("settings serialisation error: {0}")]
    Serialise(#[from] serde_json::Error),

    /// File I/O failure reading or writing the settings file.
    #[error("settings I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The watch channel receiver has been dropped (programming error).
    #[error("settings watch channel closed unexpectedly")]
    ChannelClosed,
}

impl From<Error> for AppError {
    fn from(e: Error) -> AppError {
        match e {
            Error::Serialise(inner) => AppError::Io {
                context: format!("settings JSON: {inner}"),
            },
            Error::Io(inner) => AppError::Io {
                context: format!("settings file: {inner}"),
            },
            Error::ChannelClosed => AppError::Internal {
                context: "settings watch channel closed".to_string(),
            },
        }
    }
}
