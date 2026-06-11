//! Per-crate error type. Every variant converts to `common::AppError` via the
//! `From` impl below, so the public diarizer surface returns `AppResult`
//! (no `diarizer::Error` leaks across the crate boundary).

use minutist_common::AppError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load diarization model at {path}: {context}")]
    ModelLoad { path: String, context: String },

    #[error("diarization inference failed: {0}")]
    Inference(String),

    #[error("invalid diarization input: {0}")]
    InvalidInput(String),
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::ModelLoad { path, context } => AppError::ModelLoad {
                model_id: path,
                context,
            },
            Error::Inference(context) => AppError::Inference {
                backend: "diarizer".to_string(),
                context,
            },
            Error::InvalidInput(context) => AppError::InvalidInput { context },
        }
    }
}
