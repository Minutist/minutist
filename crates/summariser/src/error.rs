//! Per-crate error type. Every variant converts to `common::AppError` via the
//! `From` impl below, so the public summariser surface returns `AppResult`
//! (no `summariser::Error` leaks across the crate boundary).

use meeting_app_common::AppError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load LLM model at {path}: {context}")]
    ModelLoad { path: String, context: String },

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("chat template missing or unusable: {0}")]
    Template(String),

    #[error("prompt exceeds the model context window: {0}")]
    ContextOverflow(String),
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::ModelLoad { path, context } => AppError::ModelLoad {
                model_id: path,
                context,
            },
            Error::Inference(context) => AppError::Inference {
                backend: "summariser".to_string(),
                context,
            },
            // A missing/unusable chat template or an over-long prompt are
            // caller-input problems, surfaced as InvalidInput (never a panic).
            Error::Template(context) => AppError::InvalidInput { context },
            Error::ContextOverflow(context) => AppError::InvalidInput { context },
        }
    }
}
