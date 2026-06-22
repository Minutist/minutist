//! Crate-local error type and its conversion to [`minutist_common::AppError`].
//!
//! Per the cross-cutting rules: one `thiserror`-derived `Error` per crate,
//! mapped to `AppError` at the public boundary via `From`. No `anyhow` in
//! public signatures.

use minutist_common::AppError;

/// Crate-internal error type for `doc-convert`.
///
/// Every public function converts this to [`AppError`] via `?`. The
/// `From<ConvertError> for AppError` impl below is the sole bridge.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("zip decompression limit exceeded: {0}")]
    ZipBomb(String),

    #[error("spreadsheet error: {0}")]
    Spreadsheet(String),

    #[error("HTML conversion error: {0}")]
    Html(String),

    #[error("email parse error: {0}")]
    Email(String),

    #[error("PDF extraction error: {0}")]
    Pdf(String),

    #[error("XML parse error: {0}")]
    Xml(String),

    #[error("text encoding error: {0}")]
    Encoding(String),

    #[error("markdown normalisation error: {0}")]
    Normalise(String),
}

impl From<ConvertError> for AppError {
    fn from(e: ConvertError) -> Self {
        match e {
            ConvertError::Io(inner) => AppError::Io {
                context: inner.to_string(),
            },
            ConvertError::Zip(inner) => AppError::InvalidInput {
                context: format!("zip archive error: {inner}"),
            },
            ConvertError::ZipBomb(msg) => AppError::InvalidInput {
                context: format!("zip bomb rejected: {msg}"),
            },
            ConvertError::Spreadsheet(msg) => AppError::Internal {
                context: format!("spreadsheet: {msg}"),
            },
            ConvertError::Html(msg) => AppError::Internal {
                context: format!("html: {msg}"),
            },
            ConvertError::Email(msg) => AppError::Internal {
                context: format!("email: {msg}"),
            },
            ConvertError::Pdf(msg) => AppError::Internal {
                context: format!("pdf: {msg}"),
            },
            ConvertError::Xml(msg) => AppError::Internal {
                context: format!("xml: {msg}"),
            },
            ConvertError::Encoding(msg) => AppError::Internal {
                context: format!("encoding: {msg}"),
            },
            ConvertError::Normalise(msg) => AppError::Internal {
                context: format!("markdown normalise: {msg}"),
            },
        }
    }
}

/// Convenience alias used throughout this crate.
pub type Result<T> = std::result::Result<T, ConvertError>;
