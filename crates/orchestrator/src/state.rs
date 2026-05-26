//! Recording state machine — transition rules.
//!
//! The orchestrator's state is protected by a `tokio::sync::Mutex` so it can
//! be mutated from async command handlers without blocking the executor.
//!
//! Valid transitions:
//!
//! ```text
//! Idle → Recording  (start)
//! Recording → Paused  (pause)
//! Paused → Recording  (resume)
//! Recording | Paused → Stopping  (stop)
//! Stopping → Idle  (runner task completion)
//! ```

use meeting_app_common::{AppError, AppResult, MeetingId, RecordingState};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::Error;

/// The internal run state held under the Mutex.
///
/// This mirrors `RecordingState` from `common` but can carry private fields
/// (e.g. handles to running tasks) that shouldn't be visible in the public API.
#[derive(Debug)]
pub(crate) enum InternalState {
    Idle,
    Recording {
        meeting_id: MeetingId,
        started_at_ms: u64,
    },
    Paused {
        meeting_id: MeetingId,
        paused_at_ms: u64,
        /// Recording clock at the moment pause was requested (ms since
        /// recording start), for computing `started_at_ms` in the public
        /// `RecordingState`.
        started_at_ms: u64,
    },
    Stopping {
        meeting_id: MeetingId,
    },
}

impl InternalState {
    /// Convert to the public `RecordingState` for broadcast.
    pub(crate) fn as_public(&self) -> RecordingState {
        match self {
            InternalState::Idle => RecordingState::Idle,
            InternalState::Recording {
                meeting_id,
                started_at_ms,
            } => RecordingState::Recording {
                meeting_id: *meeting_id,
                started_at_ms: *started_at_ms,
            },
            InternalState::Paused {
                meeting_id,
                paused_at_ms,
                ..
            } => RecordingState::Paused {
                meeting_id: *meeting_id,
                paused_at_ms: *paused_at_ms,
            },
            InternalState::Stopping { meeting_id } => RecordingState::Stopping {
                meeting_id: *meeting_id,
            },
        }
    }
}

/// Validate and execute Idle → Recording.
///
/// Returns `(meeting_id, started_at_ms)` on success.
pub(crate) fn transition_start(state: &mut InternalState) -> AppResult<(MeetingId, u64)> {
    match state {
        InternalState::Idle => {}
        _ => {
            return Err(Error::InvalidState {
                context: "start() called when not Idle".into(),
            }
            .into())
        }
    }

    let meeting_id = MeetingId::new();
    let started_at_ms = now_ms();
    *state = InternalState::Recording {
        meeting_id,
        started_at_ms,
    };
    Ok((meeting_id, started_at_ms))
}

/// Validate and execute Recording → Paused.
///
/// Returns the `meeting_id` of the active recording.
pub(crate) fn transition_pause(state: &mut InternalState) -> AppResult<MeetingId> {
    match state {
        InternalState::Recording {
            meeting_id,
            started_at_ms,
        } => {
            let id = *meeting_id;
            let started = *started_at_ms;
            *state = InternalState::Paused {
                meeting_id: id,
                paused_at_ms: now_ms(),
                started_at_ms: started,
            };
            Ok(id)
        }
        _ => Err(Error::InvalidState {
            context: "pause() called when not Recording".into(),
        }
        .into()),
    }
}

/// Validate and execute Paused → Recording.
///
/// Returns the `meeting_id` of the active recording.
pub(crate) fn transition_resume(state: &mut InternalState) -> AppResult<MeetingId> {
    match state {
        InternalState::Paused {
            meeting_id,
            started_at_ms,
            ..
        } => {
            let id = *meeting_id;
            let started = *started_at_ms;
            *state = InternalState::Recording {
                meeting_id: id,
                started_at_ms: started,
            };
            Ok(id)
        }
        _ => Err(Error::InvalidState {
            context: "resume() called when not Paused".into(),
        }
        .into()),
    }
}

/// Validate and execute Recording | Paused → Stopping.
///
/// Returns the `meeting_id`.
pub(crate) fn transition_stop(state: &mut InternalState) -> AppResult<MeetingId> {
    match state {
        InternalState::Recording { meeting_id, .. } | InternalState::Paused { meeting_id, .. } => {
            let id = *meeting_id;
            *state = InternalState::Stopping { meeting_id: id };
            Ok(id)
        }
        _ => Err(Error::InvalidState {
            context: "stop() called when not Recording or Paused".into(),
        }
        .into()),
    }
}

/// Drive Stopping → Idle once the runner task completes.
pub(crate) fn transition_idle(state: &mut InternalState) -> AppResult<()> {
    match state {
        InternalState::Stopping { .. } => {
            *state = InternalState::Idle;
            Ok(())
        }
        _ => Err(AppError::Internal {
            context: "transition_idle called outside Stopping state".into(),
        }),
    }
}

/// Milliseconds since the Unix epoch (wall clock). Used for recording
/// timestamps; precision is sufficient for human-readable timestamps.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
