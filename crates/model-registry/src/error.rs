//! Per-crate error type with conversion to `AppError` at the crate boundary.

use minutist_common::AppError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    #[error("json: {0}")]
    JsonParse(#[from] serde_json::Error),

    #[error("sha256 mismatch for {filename}: expected {expected}, got {actual}")]
    HashMismatch {
        filename: String,
        expected: String,
        actual: String,
    },

    #[error("manifest entry not found: {model_id}")]
    ManifestEntryNotFound { model_id: String },

    #[error("cancelled")]
    Cancelled,
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        use Error::*;
        match e {
            Io(io) => AppError::Io {
                context: io.to_string(),
            },
            Http(http) => AppError::ModelDownload {
                context: http.to_string(),
            },
            JsonParse(j) => AppError::ModelDownload {
                context: format!("manifest parse: {j}"),
            },
            HashMismatch {
                filename,
                expected,
                actual,
            } => AppError::ModelDownload {
                context: format!(
                    "sha256 mismatch for {filename}: expected {expected}, got {actual}"
                ),
            },
            ManifestEntryNotFound { model_id } => AppError::ModelNotFound { model_id },
            Cancelled => AppError::Cancelled,
        }
    }
}
