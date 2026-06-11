//! Per-crate error type. Every variant converts to `common::AppError` via the
//! `From` impl below, so the public `chat-agent` surface returns `AppResult`
//! (no `chat_agent::Error` leaks across the crate boundary). Mirrors
//! `summariser::Error`.

use minutist_common::AppError;

/// Errors raised inside the chat turn engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Rendering the OpenAI-format messages + tools into a prompt failed (the
    /// GGUF's tool template was missing/unusable, or the FFI render errored).
    #[error("chat prompt render failed: {0}")]
    Template(String),

    /// Building/compiling a GBNF grammar from a tool schema failed.
    #[error("tool grammar build failed: {0}")]
    Grammar(String),

    /// A decode / tokenise / context step in the real backend failed.
    #[error("chat inference failed: {0}")]
    Inference(String),

    /// The windowed prompt plus its generation headroom does not fit `n_ctx`,
    /// even after the driver's sliding-window trim — a genuinely-too-large
    /// single turn (the hard floor, §6.2).
    #[error("chat context overflow: {0}")]
    ContextOverflow(String),

    /// The backend returned output that is neither valid final text nor a
    /// parseable tool call (malformed-output handling, §1.3).
    #[error("malformed assistant turn: {0}")]
    MalformedOutput(String),
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::Inference(context) => AppError::Inference {
                backend: "chat-agent".to_string(),
                context,
            },
            // A render/grammar failure, an over-long prompt and malformed model
            // output are all caller-input-shaped problems at this boundary,
            // surfaced as InvalidInput (never a panic) — mirrors `summariser`.
            Error::Template(context)
            | Error::Grammar(context)
            | Error::ContextOverflow(context)
            | Error::MalformedOutput(context) => AppError::InvalidInput { context },
        }
    }
}
