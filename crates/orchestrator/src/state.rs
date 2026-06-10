//! Recording state machine — transition rules.
//!
//! The orchestrator's state is protected by a `tokio::sync::Mutex` so it can
//! be mutated from async command handlers without blocking the executor.
//!
//! Valid transitions:
//!
//! ```text
//! Idle → Recording  (start)
//! Offline → Recording  (start PREEMPTS a post-stop repair pass)
//! Recording → Paused  (pause)
//! Paused → Recording  (resume)
//! Recording | Paused → Stopping  (stop — capture stopping)
//! Stopping → Finalising  (background drain/finalise begins)
//! Stopping | Finalising → Idle  (finalise complete)
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
    /// Capture has stopped; the meeting is finalising in the background (drain +
    /// transcript/metadata/audio writes). The recorder is still busy — a new
    /// `start` is refused — but it reports a distinct public state so the UI
    /// stays responsive. See `RecordingState::Finalising`.
    Finalising {
        meeting_id: MeetingId,
    },
    /// An offline operation (re-transcribe / re-diarize, including the automatic
    /// post-stop repair passes) is running. The recorder is not live, but the
    /// slot is **claimed** so a concurrent `re_transcribe` / `rediarize` (or a
    /// user-triggered one) is rejected and cannot clobber the SAME meeting's
    /// `transcript.json` (TIMELINE-DRIFT #5). Restored to `Idle` when the op
    /// completes (including on error). `meeting_id` is surfaced as the public
    /// `Finalising` state's id (see [`InternalState::as_public`]) so the UI shows
    /// a busy indicator while the claim is held.
    ///
    /// **`start` PREEMPTS this state.** A new recording is a new `meeting_id`
    /// (a different `transcript.json`), so the clobber hazard does not apply, and
    /// the user must never be blocked from starting the next meeting while the
    /// previous one's best-effort repair runs. On preempt the in-flight op is
    /// left to finish in the background (it writes the OLD meeting's files, which
    /// is harmless) and its eventual release is a no-op (see
    /// [`transition_offline_release`]); the post-stop chain's remaining passes
    /// self-skip (a fresh claim now fails against `Recording`).
    Offline {
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
            InternalState::Finalising { meeting_id } => RecordingState::Finalising {
                meeting_id: *meeting_id,
            },
            // An offline op (re-transcribe / re-diarize, incl. the automatic
            // post-stop repair passes) holds the claim. Report `Idle`: the
            // recorder is genuinely READY for the next meeting (a `start` here
            // preempts the pass — see `transition_start`), so the transport must
            // NOT gate Start behind a busy state. The repair's progress is shown
            // per-meeting on the meeting-list ROW (via `OperationProgress`), which
            // is where it belongs; the transport reflects only "can I record
            // now?" — and the answer during a background repair is yes.
            InternalState::Offline { .. } => RecordingState::Idle,
        }
    }
}

/// Atomically claim the recorder for an offline operation (re-transcribe /
/// re-diarize). Only valid from `Idle`; any live or already-offline state
/// rejects with `AppError::InvalidInput` so two offline ops (or an offline op
/// racing a `start`) cannot run concurrently and clobber the same meeting's
/// `transcript.json` (TIMELINE-DRIFT #5).
pub(crate) fn transition_offline_claim(
    state: &mut InternalState,
    meeting_id: MeetingId,
) -> AppResult<()> {
    match state {
        InternalState::Idle => {
            *state = InternalState::Offline { meeting_id };
            Ok(())
        }
        _ => Err(AppError::InvalidInput {
            context: "offline operation requires the recorder to be Idle".into(),
        }),
    }
}

/// Release an offline claim. Returns `true` when it actually released an
/// `Offline` claim (→ `Idle`), `false` when the claim had already been preempted
/// (the state is something else, e.g. `Recording`).
///
/// PREEMPTION-SAFE: if `start` has already taken over the slot (`Recording`),
/// the in-flight offline op's late release MUST NOT reset the recorder to `Idle`
/// — that would clobber the new live session. In that case this leaves the state
/// untouched and returns `false`; the caller then suppresses the `Idle`
/// broadcast. A release from any other non-`Offline` state is still forced to
/// `Idle` (defensive: a stray release in a cleanup path must not wedge the
/// recorder), which only the `false`-on-preempt `Recording` case excludes.
pub(crate) fn transition_offline_release(state: &mut InternalState) -> bool {
    match state {
        InternalState::Offline { meeting_id } => {
            tracing::debug!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                "released offline claim → Idle"
            );
            *state = InternalState::Idle;
            true
        }
        // The slot was preempted by a live session — leave it alone.
        InternalState::Recording { .. }
        | InternalState::Paused { .. }
        | InternalState::Stopping { .. }
        | InternalState::Finalising { .. } => false,
        // Truly unexpected (already Idle): force Idle defensively, but report it
        // did not transition from a held claim.
        InternalState::Idle => false,
    }
}

/// Validate and execute `Idle | Offline → Recording`.
///
/// Returns `(meeting_id, started_at_ms)` on success.
///
/// Starting from `Offline` **preempts** a post-stop repair pass (re-transcribe /
/// re-diarize / auto-summarise) for the PREVIOUS meeting: the user must not be
/// blocked from recording the next meeting while best-effort repair runs. The
/// preempted op finishes in the background on its own thread (writing the old
/// meeting's files — a different `transcript.json`, so no clobber) and its
/// release becomes a no-op (see [`transition_offline_release`]). The genuine
/// `Stopping` / `Finalising` drain is NOT preemptible — that is the capture
/// teardown + transcript/metadata write, which must complete first.
pub(crate) fn transition_start(state: &mut InternalState) -> AppResult<(MeetingId, u64)> {
    match state {
        InternalState::Idle | InternalState::Offline { .. } => {}
        _ => {
            return Err(Error::InvalidState {
                context: "start() called while a recording is live or finalising".into(),
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

/// Drive Stopping → Finalising when the background drain/finalise begins.
///
/// Keeps the recorder marked busy (a new `start` is refused) while the meeting
/// finalises off the stop path, but reports a distinct public state so the UI
/// stays responsive instead of showing a blocking "stopping" indicator.
pub(crate) fn transition_finalising(state: &mut InternalState) -> AppResult<MeetingId> {
    match state {
        InternalState::Stopping { meeting_id } => {
            let id = *meeting_id;
            *state = InternalState::Finalising { meeting_id: id };
            Ok(id)
        }
        _ => Err(AppError::Internal {
            context: "transition_finalising called outside Stopping state".into(),
        }),
    }
}

/// Drive Stopping | Finalising → Idle once finalise completes.
pub(crate) fn transition_idle(state: &mut InternalState) -> AppResult<()> {
    match state {
        InternalState::Stopping { .. } | InternalState::Finalising { .. } => {
            *state = InternalState::Idle;
            Ok(())
        }
        _ => Err(AppError::Internal {
            context: "transition_idle called outside Stopping/Finalising state".into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_from_idle_records() {
        let mut state = InternalState::Idle;
        let (id, _) = transition_start(&mut state).expect("idle start");
        assert!(matches!(state, InternalState::Recording { meeting_id, .. } if meeting_id == id));
    }

    #[test]
    fn start_preempts_an_offline_repair_pass() {
        // A post-stop repair pass for meeting A holds the Offline claim.
        let old = MeetingId::new();
        let mut state = InternalState::Offline { meeting_id: old };

        // Starting the next meeting must succeed (NOT block) and take over the
        // slot with a fresh meeting id.
        let (new_id, _) = transition_start(&mut state).expect("start preempts Offline");
        assert_ne!(new_id, old, "the new recording is a distinct meeting");
        assert!(matches!(state, InternalState::Recording { meeting_id, .. } if meeting_id == new_id));
    }

    #[test]
    fn preempted_offline_release_does_not_clobber_the_live_recording() {
        // Slot preempted: state is now Recording (a new meeting).
        let live = MeetingId::new();
        let mut state = InternalState::Recording {
            meeting_id: live,
            started_at_ms: 1,
        };
        // The preempted op's late release must be a NO-OP — it must not reset to
        // Idle and clobber the live recording.
        let released = transition_offline_release(&mut state);
        assert!(!released, "release after preempt reports it did not release");
        assert!(matches!(state, InternalState::Recording { meeting_id, .. } if meeting_id == live));
    }

    #[test]
    fn normal_offline_release_returns_to_idle() {
        let mut state = InternalState::Offline {
            meeting_id: MeetingId::new(),
        };
        let released = transition_offline_release(&mut state);
        assert!(released, "an actual Offline claim release reports true");
        assert!(matches!(state, InternalState::Idle));
    }

    #[test]
    fn start_is_rejected_while_live_or_finalising() {
        let id = MeetingId::new();
        for mut state in [
            InternalState::Recording {
                meeting_id: id,
                started_at_ms: 0,
            },
            InternalState::Paused {
                meeting_id: id,
                paused_at_ms: 0,
                started_at_ms: 0,
            },
            InternalState::Stopping { meeting_id: id },
            InternalState::Finalising { meeting_id: id },
        ] {
            assert!(
                transition_start(&mut state).is_err(),
                "start must be refused from a live/finalising state: {state:?}"
            );
        }
    }
}
