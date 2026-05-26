//! Per-crate error type for audio-capture.
//!
//! `From<Error> for AppError` lets errors cross the crate boundary into the
//! orchestrator / IPC layer without leaking internal variants.

use meeting_app_common::AppError;

/// Errors that can occur inside the `audio-capture` crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No audio input device is available on this host.
    #[error("no audio input device available")]
    NoInputDevice,

    /// The requested device id was not found in the current device list.
    #[error("device not found: {id}")]
    DeviceNotFound { id: String },

    /// cpal failed to enumerate or open a device.
    #[error("cpal error: {context}")]
    Cpal { context: String },

    /// The capture stream is in a state where the requested operation is
    /// not valid (e.g. calling `start` when already started).
    #[error("invalid capture state: {context}")]
    InvalidState { context: String },

    /// rubato resampler failed to initialise or process a frame.
    #[error("resampler error: {context}")]
    Resampler { context: String },
}

impl From<cpal::DevicesError> for Error {
    fn from(e: cpal::DevicesError) -> Self {
        Error::Cpal {
            context: e.to_string(),
        }
    }
}

impl From<cpal::BuildStreamError> for Error {
    fn from(e: cpal::BuildStreamError) -> Self {
        Error::Cpal {
            context: e.to_string(),
        }
    }
}

impl From<cpal::PlayStreamError> for Error {
    fn from(e: cpal::PlayStreamError) -> Self {
        Error::Cpal {
            context: e.to_string(),
        }
    }
}

impl From<cpal::PauseStreamError> for Error {
    fn from(e: cpal::PauseStreamError) -> Self {
        Error::Cpal {
            context: e.to_string(),
        }
    }
}

impl From<cpal::DefaultStreamConfigError> for Error {
    fn from(e: cpal::DefaultStreamConfigError) -> Self {
        Error::Cpal {
            context: e.to_string(),
        }
    }
}

impl From<cpal::SupportedStreamConfigsError> for Error {
    fn from(e: cpal::SupportedStreamConfigsError) -> Self {
        Error::Cpal {
            context: e.to_string(),
        }
    }
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::NoInputDevice => AppError::Unsupported {
                context: "no audio input device available".into(),
            },
            Error::DeviceNotFound { id } => AppError::InvalidInput {
                context: format!("audio device not found: {id}"),
            },
            Error::InvalidState { context } => AppError::InvalidInput { context },
            Error::Cpal { context } | Error::Resampler { context } => {
                AppError::Internal { context }
            }
        }
    }
}
