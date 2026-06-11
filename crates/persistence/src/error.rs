use minutist_common::AppError;

/// Per-crate error type for the `persistence` crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Opus encoder error: {0}")]
    Opus(String),

    #[error("meeting folder already exists: {0}")]
    FolderExists(std::path::PathBuf),

    #[error("serialisation error: {0}")]
    Serialise(#[from] serde_json::Error),

    #[error("invalid state: {0}")]
    InvalidState(&'static str),

    #[error("Opus decode error: {0}")]
    OpusDecode(String),

    #[error("index database error: {0}")]
    Index(#[from] libsql::Error),

    #[error("meeting not found: {}", .0 .0)]
    MeetingNotFound(minutist_common::MeetingId),

    #[error("schema migration error: {0}")]
    Migration(String),
}

impl From<audiopus::Error> for Error {
    fn from(e: audiopus::Error) -> Self {
        Error::Opus(e.to_string())
    }
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::Io(inner) => AppError::Io {
                context: inner.to_string(),
            },
            Error::Opus(msg) => AppError::Internal { context: msg },
            Error::FolderExists(path) => AppError::InvalidInput {
                context: format!("meeting folder already exists: {}", path.display()),
            },
            Error::Serialise(inner) => AppError::Internal {
                context: inner.to_string(),
            },
            Error::InvalidState(msg) => AppError::InvalidInput {
                context: msg.to_string(),
            },
            Error::OpusDecode(msg) => AppError::Internal {
                context: format!("opus decode: {msg}"),
            },
            Error::Index(inner) => AppError::Internal {
                context: format!("index database: {inner}"),
            },
            Error::MeetingNotFound(id) => AppError::InvalidInput {
                context: format!("meeting not found: {}", id.0),
            },
            Error::Migration(msg) => AppError::Internal {
                context: format!("schema migration: {msg}"),
            },
        }
    }
}

/// Convenience `Result` alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
