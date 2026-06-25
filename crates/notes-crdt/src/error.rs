use minutist_common::AppError;

/// Per-crate error type for the `notes-crdt` crate.
///
/// A deliberately light enum carrying only the variants the notes-CRDT
/// primitives (folder layout, metadata writer, `NotesStore`, the `ydoc`
/// conversions) actually produce. The libsql / audiopus variants live in
/// `persistence::error` — keeping them out of here is what severs this crate's
/// (and `sync`'s) compile-time dependency on the C-heavy persistence graph.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("meeting folder already exists: {0}")]
    FolderExists(std::path::PathBuf),

    #[error("serialisation error: {0}")]
    Serialise(#[from] serde_json::Error),

    #[error("invalid state: {0}")]
    InvalidState(&'static str),

    #[error("meeting not found: {}", .0 .0)]
    MeetingNotFound(minutist_common::MeetingId),
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::Io(inner) => AppError::Io {
                context: inner.to_string(),
            },
            Error::FolderExists(path) => AppError::InvalidInput {
                context: format!("meeting folder already exists: {}", path.display()),
            },
            Error::Serialise(inner) => AppError::Internal {
                context: inner.to_string(),
            },
            Error::InvalidState(msg) => AppError::InvalidInput {
                context: msg.to_string(),
            },
            Error::MeetingNotFound(id) => AppError::InvalidInput {
                context: format!("meeting not found: {}", id.0),
            },
        }
    }
}

/// Convenience `Result` alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
