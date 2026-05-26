//! The capture-drain runner task.
//!
//! The runner runs as a `tokio::task::spawn_blocking` thread. It owns:
//! - The `AudioStreams` (sample + meter mpsc receivers from `audio-capture`).
//! - The `MeetingWriter` from `persistence`.
//!
//! It drains both channels and:
//! - forwards sample batches → `MeetingWriter::push_samples`
//! - broadcasts meter frames via the shared `AppEvent` sender
//!
//! Pause / resume pause the *writer* (not the capture stream; the capture
//! stream is paused by the orchestrator separately via `AudioCaptureManager`).
//! Stop finalises the writer and sends the `MeetingMeta` back via oneshot.

use std::time::Duration;

use audio_capture::{AudioFrameBatch, AudioStreams};
use meeting_app_common::{AppEvent, AppResult, AudioMeterFrame, MeetingMeta};
use persistence::MeetingWriter;
use tokio::sync::{broadcast, mpsc, oneshot};

// ---------------------------------------------------------------------------
// Commands sent from orchestrator → runner
// ---------------------------------------------------------------------------

/// Commands the orchestrator sends to the running drain loop.
pub(crate) enum RunnerCommand {
    /// Call `MeetingWriter::pause` and suspend sample writes until `WriterResume`.
    WriterPause,
    /// Call `MeetingWriter::resume` and resume sample writes.
    WriterResume,
    /// Flush, finalise the writer, and exit.
    ///
    /// `meta` is boxed to reduce the size difference between variants.
    Stop {
        meta: Box<MeetingMeta>,
        reply: oneshot::Sender<AppResult<MeetingMeta>>,
    },
}

// ---------------------------------------------------------------------------
// Public handle
// ---------------------------------------------------------------------------

/// Live handle to the drain runner.
pub(crate) struct RunnerHandle {
    pub(crate) cmd_tx: mpsc::Sender<RunnerCommand>,
}

// ---------------------------------------------------------------------------
// Spawn
// ---------------------------------------------------------------------------

/// Spawn the drain runner as a `spawn_blocking` task.
///
/// `streams`           — the two mpsc receivers from `AudioCaptureManager::start`.
/// `writer`            — an already-open `MeetingWriter`.
/// `event_tx`          — the orchestrator's broadcast sender for `AppEvent`.
pub(crate) fn spawn_runner(
    streams: AudioStreams,
    writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
) -> RunnerHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<RunnerCommand>(8);

    tokio::task::spawn_blocking(move || {
        run_drain_loop(streams, writer, event_tx, cmd_rx);
    });

    RunnerHandle { cmd_tx }
}

// ---------------------------------------------------------------------------
// Blocking drain loop
// ---------------------------------------------------------------------------

fn run_drain_loop(
    mut streams: AudioStreams,
    mut writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    mut cmd_rx: mpsc::Receiver<RunnerCommand>,
) {
    let mut writer_paused = false;

    loop {
        // --- 1. Poll commands (non-blocking) ---
        loop {
            match cmd_rx.try_recv() {
                Ok(RunnerCommand::WriterPause) => {
                    if let Err(e) = writer.pause() {
                        tracing::warn!(
                            target: "orchestrator",
                            "MeetingWriter::pause failed: {e}"
                        );
                    }
                    writer_paused = true;
                    tracing::debug!(target: "orchestrator", "runner: writer paused");
                }
                Ok(RunnerCommand::WriterResume) => {
                    if let Err(e) = writer.resume() {
                        tracing::warn!(
                            target: "orchestrator",
                            "MeetingWriter::resume failed: {e}"
                        );
                    }
                    writer_paused = false;
                    tracing::debug!(target: "orchestrator", "runner: writer resumed");
                }
                Ok(RunnerCommand::Stop { meta, reply }) => {
                    tracing::info!(
                        target: "orchestrator",
                        meeting_id = %meta.uuid.0,
                        "runner: stop command received; draining remaining samples"
                    );
                    // Drain any queued samples before finalising.
                    if !writer_paused {
                        drain_samples(&mut streams.samples, &mut writer);
                    }
                    drain_meter(&mut streams.meter, &event_tx);
                    let unboxed = *meta;
                    let result = writer.finalise(unboxed.clone()).map(|_| unboxed);
                    let _ = reply.send(result);
                    tracing::info!(target: "orchestrator", "runner: exiting cleanly");
                    return;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    tracing::warn!(
                        target: "orchestrator",
                        "runner: command channel disconnected; exiting"
                    );
                    return;
                }
            }
        }

        // --- 2. If writer is paused, block on the next command ---
        if writer_paused {
            match cmd_rx.blocking_recv() {
                Some(RunnerCommand::WriterPause) => {
                    // Already paused — no-op.
                }
                Some(RunnerCommand::WriterResume) => {
                    if let Err(e) = writer.resume() {
                        tracing::warn!(
                            target: "orchestrator",
                            "MeetingWriter::resume failed: {e}"
                        );
                    }
                    writer_paused = false;
                    tracing::debug!(target: "orchestrator", "runner: writer resumed");
                }
                Some(RunnerCommand::Stop { meta, reply }) => {
                    tracing::info!(
                        target: "orchestrator",
                        meeting_id = %meta.uuid.0,
                        "runner: stop while writer paused; finalising"
                    );
                    let unboxed = *meta;
                    let result = writer.finalise(unboxed.clone()).map(|_| unboxed);
                    let _ = reply.send(result);
                    return;
                }
                None => {
                    tracing::warn!(
                        target: "orchestrator",
                        "runner: command channel closed while paused; exiting"
                    );
                    return;
                }
            }
            continue;
        }

        // --- 3. Receive one sample batch ---
        match streams.samples.try_recv() {
            Ok(batch) => {
                push_batch(&mut writer, &batch);
            }
            Err(mpsc::error::TryRecvError::Empty) => {
                // Nothing queued; sleep briefly to avoid spin.
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                tracing::info!(
                    target: "orchestrator",
                    "runner: samples channel closed; draining meter and waiting for stop"
                );
                // Drain remaining meter frames and wait for the stop command.
                drain_meter(&mut streams.meter, &event_tx);
                // Block until stop arrives.
                if let Some(RunnerCommand::Stop { meta, reply }) = cmd_rx.blocking_recv() {
                    let unboxed = *meta;
                    let result = writer.finalise(unboxed.clone()).map(|_| unboxed);
                    let _ = reply.send(result);
                }
                tracing::info!(target: "orchestrator", "runner: exiting after samples closed");
                return;
            }
        }

        // --- 4. Forward any pending meter frames ---
        drain_meter_nonblocking(&mut streams.meter, &event_tx);
    }
}

fn push_batch(writer: &mut MeetingWriter, batch: &AudioFrameBatch) {
    if let Err(e) = writer.push_samples(&batch.samples) {
        tracing::error!(
            target: "orchestrator",
            "push_samples failed: {e}"
        );
    }
}

fn drain_samples(rx: &mut mpsc::Receiver<AudioFrameBatch>, writer: &mut MeetingWriter) {
    while let Ok(b) = rx.try_recv() {
        push_batch(writer, &b);
    }
}

fn drain_meter(rx: &mut mpsc::Receiver<AudioMeterFrame>, event_tx: &broadcast::Sender<AppEvent>) {
    while let Ok(mf) = rx.try_recv() {
        broadcast_meter(event_tx, mf);
    }
}

fn drain_meter_nonblocking(
    rx: &mut mpsc::Receiver<AudioMeterFrame>,
    event_tx: &broadcast::Sender<AppEvent>,
) {
    drain_meter(rx, event_tx);
}

fn broadcast_meter(tx: &broadcast::Sender<AppEvent>, frame: AudioMeterFrame) {
    let event = AppEvent::AudioMeter { frame };
    match tx.send(event) {
        Ok(_) => {}
        Err(broadcast::error::SendError(_)) => {
            // No subscribers — silent drop is correct.
        }
    }
    // Lagged subscribers are handled by tokio broadcast internally;
    // callers receive `RecvError::Lagged` and should warn at their end.
}
