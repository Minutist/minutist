//! The capture-drain runner task and ASR worker.
//!
//! The **runner** runs as a `tokio::task::spawn_blocking` thread. It owns:
//! - The `AudioStreams` (sample + meter mpsc receivers from `audio-capture`).
//! - The `MeetingWriter` from `persistence`.
//! - A `VadChunker` for speech-activity detection.
//! - A bounded flush queue to a separate ASR worker task.
//!
//! The runner:
//! - Drains sample batches → `MeetingWriter::push_samples` (audio always saved).
//! - Feeds each batch to `VadChunker::process_samples` → `Vec<VadEvent>`.
//! - Maintains the **batched-VAD accumulator** described in Phase 2
//! - Flushes the accumulator to the ASR worker on size/latency/end-of-stream.
//! - Broadcasts meter frames via the shared `AppEvent` sender.
//!
//! The **ASR worker** runs as a separate `tokio::task::spawn_blocking` thread.
//! It receives `FlushPayload`s from the runner, calls
//! `asr_runtime.transcribe_chunk()`, and re-splits the result proportionally
//! across VAD sub-segments (Phase 2).
//!
//! ## Locked constants
//!
//! - `FLUSH_MIN_SECS = 25.0` — minimum buffer duration before a size-triggered flush.
//! - `LATENCY_WINDOW_SECS = 10.0` — wall-clock seconds of quiet after which a
//!   non-empty buffer is flushed.
//! - `MAX_GAP_MS = 3000` — maximum inter-segment silence gap (zero-padded, not
//!   compacted) inserted between VAD segments in the accumulator.
//!
//! ## Threading
//!
//! Runner → ASR worker: `Arc<Mutex<VecDeque<FlushPayload>>>` (capacity 4) + `Arc<Notify>`.
//! On backpressure the runner drops the OLDEST queued flush (the entry at the
//! front of the deque), enqueues the newest at the back, and emits
//! `AppEvent::ErrorOccurred`. Audio is always preserved in `audio.opus`; only
//! live transcript is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use asr_runtime::{AsrRuntime, AsrRuntimeConfig};
use audio_capture::{AudioFrameBatch, AudioStreams};
use meeting_app_common::{
    AppError, AppEvent, AppResult, AsrBackend, AudioChunk, AudioMeterFrame, MeetingId, MeetingMeta,
    ModelId, ModelStatusState, Segment,
};
use model_registry::ModelRegistry;
use persistence::MeetingWriter;
use tokio::sync::{broadcast, mpsc, oneshot, Notify};
use vad_chunker::{VadChunker, VadConfig, VadEvent};

// ---------------------------------------------------------------------------
// Locked constants (Phase 2 — do NOT change without arch review)
// ---------------------------------------------------------------------------

const FLUSH_MIN_SECS: f32 = 25.0;
const LATENCY_WINDOW_SECS: f32 = 10.0;
const MAX_GAP_MS: u64 = 3000;
const SAMPLE_RATE_HZ: u64 = 16_000;
/// Minimum interval between `AppEvent::RecordingClock` emissions (~5 Hz).
/// See `architecture/cross-cutting.md` — "Notes paragraph-anchor clock".
const RECORDING_CLOCK_MIN_INTERVAL_MS: u64 = 200;
/// Poll interval when no audio is arriving (latency-window check).
const POLL_INTERVAL: Duration = Duration::from_millis(200);
/// Hardcoded ASR model ID (Phase 2 LOCKED CHOICE).
const ASR_MODEL_ID: &str = "qwen3-asr-0.6b-q8_0";
/// Flush dispatch queue capacity. Drop-oldest when this is exceeded.
const FLUSH_CHANNEL_CAP: usize = 4;

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

/// Internal commands sent from the ASR worker back to the runner for writes
/// that must happen on the runner's `MeetingWriter`.
enum WriterCommand {
    /// Write a transcript segment to `transcript.json`.
    WriteSegment(Segment),
}

// ---------------------------------------------------------------------------
// Flush payload: runner → ASR worker
// ---------------------------------------------------------------------------

/// A complete accumulator snapshot sent to the ASR worker.
pub(crate) struct FlushPayload {
    pub(crate) samples: Vec<f32>,
    /// `(start_ms, end_ms)` for each VAD segment in this buffer.
    pub(crate) vad_segments: Vec<(u64, u64)>,
    pub(crate) meeting_id: MeetingId,
}

// ---------------------------------------------------------------------------
// Flush queue: drop-oldest bounded deque + notify
// ---------------------------------------------------------------------------

/// Shared state between the runner (producer) and the ASR worker (consumer).
///
/// `Arc<Notify>` signals the worker that a new payload is available or that
/// the runner has exited (`closed` is set).
/// `Arc<StdMutex<VecDeque<FlushPayload>>>` is the bounded drop-oldest queue.
/// `Arc<AtomicBool>` signals to the worker that the runner has exited so the
/// worker can drain the remaining queue and then exit.
///
/// Choosing `std::sync::Mutex` (not tokio) here because both the runner and
/// the ASR worker run on `spawn_blocking` threads; there is no async context
/// while the mutex is held, so the lighter `std` mutex is correct.
pub(crate) struct FlushQueue {
    pub(crate) deque: Arc<StdMutex<VecDeque<FlushPayload>>>,
    pub(crate) notify: Arc<Notify>,
    /// Set to `true` when the runner has finished; the worker drains remaining
    /// payloads and then exits.
    pub(crate) closed: Arc<AtomicBool>,
    /// Count of payloads currently being processed by the worker.
    /// Incremented when the worker pops a payload; decremented when processing
    /// completes. The runner uses this + queue length = 0 to know all work is done.
    pub(crate) in_flight: Arc<AtomicUsize>,
    /// Set to `true` when the ASR worker thread has exited (either normally or
    /// due to an unrecoverable error). When this flag is set,
    /// `wait_all_processed` returns immediately rather than blocking until
    /// timeout — a dead worker will never decrement `in_flight`.
    pub(crate) worker_exited: Arc<AtomicBool>,
}

impl FlushQueue {
    pub(crate) fn new() -> Self {
        Self {
            deque: Arc::new(StdMutex::new(VecDeque::with_capacity(FLUSH_CHANNEL_CAP + 1))),
            notify: Arc::new(Notify::new()),
            closed: Arc::new(AtomicBool::new(false)),
            in_flight: Arc::new(AtomicUsize::new(0)),
            worker_exited: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Clone the consumer side (shared deque + notify, no ownership transfer).
    pub(crate) fn consumer_clone(&self) -> Self {
        Self {
            deque: Arc::clone(&self.deque),
            notify: Arc::clone(&self.notify),
            closed: Arc::clone(&self.closed),
            in_flight: Arc::clone(&self.in_flight),
            worker_exited: Arc::clone(&self.worker_exited),
        }
    }

    /// Signal the worker that no more payloads will be produced.
    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::Release);
        // Wake the worker so it can observe the closed flag.
        self.notify.notify_one();
    }

    /// Block until the flush queue is empty AND no payload is being processed,
    /// or until the ASR worker thread exits (detected via `worker_exited`).
    ///
    /// Returns `true` if all work completed before `timeout` elapsed, `false`
    /// if the timeout was reached.
    ///
    /// A dead/terminated worker sets `worker_exited = true`, which causes this
    /// function to return immediately so `stop()` is never wedged.
    ///
    /// Called by the runner before finalising the `MeetingWriter` so that the
    /// last flush's transcript segments are written to `transcript.json`.
    pub(crate) fn wait_all_processed(&self, timeout: Duration, poll_interval: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            // If the worker has exited, there is nobody left to decrement
            // `in_flight`; return immediately to avoid hanging forever.
            if self.worker_exited.load(Ordering::Acquire) {
                return true;
            }
            let queue_empty = {
                let deque = self.deque.lock().expect("flush queue mutex poisoned");
                deque.is_empty()
            };
            let idle = queue_empty && self.in_flight.load(Ordering::Acquire) == 0;
            if idle {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(poll_interval);
        }
    }
}

// ---------------------------------------------------------------------------
// Batched-VAD accumulator
// ---------------------------------------------------------------------------

/// Accumulates VAD segments into a buffer that reconstructs the original
/// recording-clock timeline by zero-padding inter-segment gaps (capped at
/// `MAX_GAP_MS` to bound encoder budget on very long silences).
///
/// **NEVER compact** (remove silences between segments) — Qwen3-ASR uses
/// internal silences as sentence-boundary anchors; compaction causes
/// greedy-decode loops.
pub(crate) struct Accumulator {
    pub(crate) samples: Vec<f32>,
    /// Recording-clock ms of the first sample in `samples`.
    pub(crate) buffer_start_ms: Option<u64>,
    /// `(start_ms, end_ms)` of each VAD segment appended so far.
    pub(crate) vad_segments: Vec<(u64, u64)>,
    /// Wall-clock instant the most recent VAD segment ended.
    pub(crate) last_vad_end_at: Option<Instant>,
}

impl Accumulator {
    pub(crate) fn new() -> Self {
        Self {
            samples: Vec::new(),
            buffer_start_ms: None,
            vad_segments: Vec::new(),
            last_vad_end_at: None,
        }
    }

    /// Append a VAD segment, zero-padding the gap from the current buffer tail.
    pub(crate) fn append(&mut self, start_ms: u64, end_ms: u64, seg_samples: &[f32]) {
        let buffer_start = *self.buffer_start_ms.get_or_insert(start_ms);

        // How many samples should be at offset `start_ms` within the buffer.
        let expected_len =
            ((start_ms.saturating_sub(buffer_start)) as usize * SAMPLE_RATE_HZ as usize) / 1000;

        if expected_len > self.samples.len() {
            let gap = expected_len - self.samples.len();
            // Cap inter-segment gap at MAX_GAP_MS worth of samples.
            let max_gap_samples = (MAX_GAP_MS as usize * SAMPLE_RATE_HZ as usize) / 1000;
            let capped = gap.min(max_gap_samples);
            self.samples.extend(std::iter::repeat_n(0.0f32, capped));
        }

        self.samples.extend_from_slice(seg_samples);
        self.vad_segments.push((start_ms, end_ms));
        self.last_vad_end_at = Some(Instant::now());
    }

    /// Duration of buffered audio in seconds.
    pub(crate) fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE_HZ as f32
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Drain the accumulator, returning samples and segment list. Resets state.
    pub(crate) fn drain(&mut self) -> (Vec<f32>, Vec<(u64, u64)>) {
        let samples = std::mem::take(&mut self.samples);
        let segments = std::mem::take(&mut self.vad_segments);
        self.buffer_start_ms = None;
        self.last_vad_end_at = None;
        (samples, segments)
    }
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
/// `streams`        — the two mpsc receivers from `AudioCaptureManager::start`.
/// `writer`         — an already-open `MeetingWriter`.
/// `event_tx`       — the orchestrator's broadcast sender for `AppEvent`.
/// `model_registry` — used to locate the ASR model on first flush.
/// `meeting_id`     — used when emitting `TranscriptSegment` events.
pub(crate) fn spawn_runner(
    streams: AudioStreams,
    writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    model_registry: Arc<ModelRegistry>,
    meeting_id: MeetingId,
) -> RunnerHandle {
    spawn_runner_inner(streams, writer, event_tx, model_registry, meeting_id, None)
}

/// Internal spawn function shared by production and test paths.
fn spawn_runner_inner(
    streams: AudioStreams,
    writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    model_registry: Arc<ModelRegistry>,
    meeting_id: MeetingId,
    prebuilt_backend: Option<Box<dyn AsrBackend + Send>>,
) -> RunnerHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<RunnerCommand>(8);

    // Bounded drop-oldest flush queue: runner (producer) → ASR worker (consumer).
    let flush_queue = FlushQueue::new();
    let worker_flush_queue = flush_queue.consumer_clone();

    // Channel for ASR worker → runner to write transcript segments.
    // Bounded at FLUSH_CHANNEL_CAP * max_segments_per_flush; a rough bound is fine.
    let (writer_cmd_tx, writer_cmd_rx) = mpsc::channel::<WriterCommand>(64);

    // Spawn the ASR worker on a separate spawn_blocking thread.
    let asr_event_tx = event_tx.clone();
    let asr_registry = Arc::clone(&model_registry);
    tokio::task::spawn_blocking(move || {
        run_asr_worker(
            worker_flush_queue,
            prebuilt_backend,
            asr_registry,
            asr_event_tx,
            writer_cmd_tx,
        );
    });

    // Spawn the runner drain loop.
    tokio::task::spawn_blocking(move || {
        run_drain_loop(
            streams,
            writer,
            event_tx,
            cmd_rx,
            flush_queue,
            writer_cmd_rx,
            meeting_id,
        );
    });

    RunnerHandle { cmd_tx }
}

/// Spawn the drain runner with a pre-built ASR backend (test-only path).
///
/// Accepts a `Box<dyn AsrBackend + Send>` that is used directly instead of
/// lazily initialising `AsrRuntime` from the model registry. This allows
/// integration tests to inject a stub backend without a real model file.
///
/// Available only under the `test-source` feature.
#[cfg(any(test, feature = "test-source"))]
pub(crate) fn spawn_runner_with_backend(
    streams: AudioStreams,
    writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    model_registry: Arc<ModelRegistry>,
    meeting_id: MeetingId,
    backend: Box<dyn AsrBackend + Send>,
) -> RunnerHandle {
    spawn_runner_inner(
        streams,
        writer,
        event_tx,
        model_registry,
        meeting_id,
        Some(backend),
    )
}

// ---------------------------------------------------------------------------
// Blocking drain loop (runner)
// ---------------------------------------------------------------------------

fn run_drain_loop(
    mut streams: AudioStreams,
    mut writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    mut cmd_rx: mpsc::Receiver<RunnerCommand>,
    flush_queue: FlushQueue,
    mut writer_cmd_rx: mpsc::Receiver<WriterCommand>,
    meeting_id: MeetingId,
) {
    // Ensure the worker is signalled when this function exits for any reason.
    // `FlushQueueGuard` calls `close()` when dropped.
    struct FlushQueueGuard<'a>(&'a FlushQueue);
    impl Drop for FlushQueueGuard<'_> {
        fn drop(&mut self) {
            self.0.close();
        }
    }
    let _guard = FlushQueueGuard(&flush_queue);
    let mut writer_paused = false;
    let mut acc = Accumulator::new();
    let latency_window = Duration::from_secs_f32(LATENCY_WINDOW_SECS);

    // Throttle for `AppEvent::RecordingClock`. The notes editor stamps
    // paragraph anchors from this event (see `architecture/cross-cutting.md`
    // "Notes paragraph-anchor clock"); ~5 Hz is plenty for that and avoids
    // flooding the broadcast channel. We only emit on the sample-batch receive
    // path below — the paused branch never reaches that path, so the clock is
    // naturally not advanced while paused, matching the pause-EXCLUDING sample
    // clock carried by `batch.end_ms`.
    //
    // Initialised so the first received batch emits immediately rather than
    // waiting out the first throttle window.
    let clock_emit_interval = Duration::from_millis(RECORDING_CLOCK_MIN_INTERVAL_MS);
    let mut last_clock_emit: Option<Instant> = None;

    // Attempt to open the VAD chunker. If the model file is missing we log
    // a warning and operate without VAD (no ASR transcription).
    let mut vad_opt: Option<VadChunker> = {
        let model_path = vad_chunker::default_model_path();
        match VadChunker::open(&model_path, VadConfig::default()) {
            Ok(c) => {
                tracing::info!(
                    target: "orchestrator",
                    "VAD chunker opened from {}",
                    model_path.display()
                );
                Some(c)
            }
            Err(e) => {
                tracing::warn!(
                    target: "orchestrator",
                    "VAD chunker unavailable (model missing?): {e}; ASR disabled for this session"
                );
                None
            }
        }
    };

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
                    if !writer_paused {
                        drain_samples(&mut streams.samples, &mut writer);
                    }
                    drain_meter(&mut streams.meter, &event_tx);

                    finalise_on_stop(
                        meta,
                        reply,
                        &mut vad_opt,
                        &mut acc,
                        &flush_queue,
                        &mut writer_cmd_rx,
                        writer,
                        &event_tx,
                        meeting_id,
                    );
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
                        "runner: stop while writer paused; flushing and finalising"
                    );
                    // No live samples to drain while paused, but the VAD may
                    // still hold an in-progress segment from before the pause —
                    // run the SAME end-of-stream flush + accumulator drain the
                    // Recording-stop path uses so the last utterance is not lost
                    // (TIMELINE-DRIFT #6).
                    drain_meter(&mut streams.meter, &event_tx);
                    finalise_on_stop(
                        meta,
                        reply,
                        &mut vad_opt,
                        &mut acc,
                        &flush_queue,
                        &mut writer_cmd_rx,
                        writer,
                        &event_tx,
                        meeting_id,
                    );
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
                // Push to persistent audio.
                push_batch(&mut writer, &batch);

                // Emit a throttled RecordingClock (~5 Hz) so the notes editor
                // can stamp paragraph anchors on the pause-EXCLUDING sample
                // clock. `batch.end_ms` is exactly that clock (it advances only
                // while samples are flowing; the paused branch never reaches
                // here). See `architecture/cross-cutting.md` — "Notes
                // paragraph-anchor clock".
                let now = Instant::now();
                let should_emit_clock = last_clock_emit
                    .map(|t| now.duration_since(t) >= clock_emit_interval)
                    .unwrap_or(true);
                if should_emit_clock {
                    let _ = event_tx.send(AppEvent::RecordingClock {
                        meeting_id,
                        clock_ms: batch.end_ms,
                    });
                    last_clock_emit = Some(now);
                }

                // Feed to VAD if available.
                if let Some(ref mut vad) = vad_opt {
                    match vad.process_samples(&batch.samples, batch.start_ms) {
                        Ok(events) => {
                            for ev in events {
                                if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                                    acc.append(start_ms, end_ms, &samples);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "orchestrator",
                                "VAD process_samples error: {e}"
                            );
                        }
                    }
                }

                // Size-triggered flush.
                if acc.duration_secs() >= FLUSH_MIN_SECS {
                    let (samples, vad_segments) = acc.drain();
                    let payload = FlushPayload { samples, vad_segments, meeting_id };
                    dispatch_flush(&flush_queue, payload, &event_tx);
                }
            }
            Err(mpsc::error::TryRecvError::Empty) => {
                // Nothing queued; check latency window, then sleep briefly.
                let should_flush_latency = !acc.is_empty()
                    && acc
                        .last_vad_end_at
                        .map(|t| t.elapsed() >= latency_window)
                        .unwrap_or(false);

                if should_flush_latency {
                    tracing::debug!(target: "orchestrator", "runner: latency-window flush");
                    let (samples, vad_segments) = acc.drain();
                    let payload = FlushPayload { samples, vad_segments, meeting_id };
                    dispatch_flush(&flush_queue, payload, &event_tx);
                } else {
                    std::thread::sleep(POLL_INTERVAL);
                }
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                tracing::info!(
                    target: "orchestrator",
                    "runner: samples channel closed; draining meter and waiting for stop"
                );
                drain_meter(&mut streams.meter, &event_tx);
                if let Some(RunnerCommand::Stop { meta, reply }) = cmd_rx.blocking_recv() {
                    // End-of-stream VAD flush.
                    if let Some(ref mut vad) = vad_opt {
                        match vad.flush_end_of_stream() {
                            Ok(events) => {
                                for ev in events {
                                    if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev
                                    {
                                        acc.append(start_ms, end_ms, &samples);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "orchestrator",
                                    "VAD flush_end_of_stream error: {e}"
                                );
                            }
                        }
                    }
                    if !acc.is_empty() {
                        let (samples, vad_segments) = acc.drain();
                        let payload = FlushPayload { samples, vad_segments, meeting_id };
                        dispatch_flush(&flush_queue, payload, &event_tx);
                    }
                    // Wait for the ASR worker to process any remaining flushes.
                    wait_for_asr_worker_drain(
                        &flush_queue,
                        &mut writer_cmd_rx,
                        &mut writer,
                    );
                    // Drain any remaining transcript write commands.
                    drain_writer_commands(&mut writer_cmd_rx, &mut writer);
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

        // --- 5. Drain any transcript write commands from the ASR worker ---
        drain_writer_commands(&mut writer_cmd_rx, &mut writer);
    }
}

// ---------------------------------------------------------------------------
// Flush dispatch helper
// ---------------------------------------------------------------------------

/// Push a flush payload to the ASR worker's bounded drop-oldest queue.
///
/// If the queue is already at capacity (`FLUSH_CHANNEL_CAP`), the **oldest**
/// pending flush (the entry at the front of the deque) is removed and
/// discarded, then the new payload is pushed to the back. An
/// `AppEvent::ErrorOccurred` is emitted so the UI can surface the warning.
///
/// Audio is always preserved in `audio.opus`; only the live transcript is
/// lost for the dropped flush.
fn dispatch_flush(
    flush_queue: &FlushQueue,
    payload: FlushPayload,
    event_tx: &broadcast::Sender<AppEvent>,
) {
    dispatch_flush_inner(flush_queue, payload, event_tx);
}

/// Public wrapper for `dispatch_flush` used by tests.
///
/// Only available under the `test-source` feature (or in `#[cfg(test)]`).
#[cfg(any(test, feature = "test-source"))]
pub(crate) fn dispatch_flush_pub(
    flush_queue: &FlushQueue,
    payload: FlushPayload,
    event_tx: &broadcast::Sender<AppEvent>,
) {
    dispatch_flush_inner(flush_queue, payload, event_tx);
}

fn dispatch_flush_inner(
    flush_queue: &FlushQueue,
    payload: FlushPayload,
    event_tx: &broadcast::Sender<AppEvent>,
) {
    let mut deque = flush_queue
        .deque
        .lock()
        .expect("flush queue mutex poisoned");

    if deque.len() >= FLUSH_CHANNEL_CAP {
        // Drop the OLDEST pending flush (the front of the queue).
        let _dropped = deque.pop_front();
        tracing::warn!(
            target: "orchestrator",
            "ASR flush queue full (backpressure); dropping oldest pending flush \
             (audio.opus unaffected)"
        );
        let _ = event_tx.send(AppEvent::ErrorOccurred {
            error: AppError::Internal {
                context: "ASR flush queue full; oldest pending flush dropped (audio.opus unaffected)".into(),
            },
        });
    }

    deque.push_back(payload);
    drop(deque);

    // Signal the worker that a new payload is available.
    flush_queue.notify.notify_one();
    tracing::debug!(target: "orchestrator", "runner: flush dispatched to ASR worker");
}

// ---------------------------------------------------------------------------
// ASR worker
// ---------------------------------------------------------------------------

/// The ASR worker runs as a dedicated `spawn_blocking` task. It drains
/// `FlushPayload`s from the flush queue, transcribes each with the ASR
/// runtime, emits `AppEvent::TranscriptSegment` events, and sends transcript
/// segments back to the runner via `writer_cmd_tx` for persistence.
///
/// `prebuilt_backend`: when `Some`, the provided backend is used directly
/// (test injection path). When `None`, the `AsrRuntime` is lazy-initialised
/// on the first flush from the `model_registry` (production path).
///
/// One flush is in-flight at a time: the worker pops one payload, processes
/// it fully, then waits for the next notification.
fn run_asr_worker(
    flush_queue: FlushQueue,
    prebuilt_backend: Option<Box<dyn AsrBackend + Send>>,
    model_registry: Arc<ModelRegistry>,
    event_tx: broadcast::Sender<AppEvent>,
    writer_cmd_tx: mpsc::Sender<WriterCommand>,
) {
    // Guard that sets `worker_exited` when this function returns for any reason
    // (clean exit or unwind). This ensures `wait_all_processed` is never wedged
    // by a dead worker that can no longer decrement `in_flight`.
    struct WorkerExitGuard<'a>(&'a Arc<AtomicBool>);
    impl Drop for WorkerExitGuard<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let _exit_guard = WorkerExitGuard(&flush_queue.worker_exited);

    // Use a single-threaded tokio runtime to allow async calls (model_registry.ensure,
    // Notify::notified) from within this spawn_blocking context.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "orchestrator", "ASR worker: failed to build runtime: {e}");
            return;
        }
    };

    // Production path: lazily-initialised AsrRuntime wrapped in an Option.
    // Test path: the prebuilt backend is used directly.
    let mut lazy_runtime: Option<AsrRuntime> = None;

    // Determine which backend to use: either the prebuilt one (test) or a lazy
    // production one. We use a closure-style dispatch below to avoid lifetimes
    // across the Option<Box<dyn AsrBackend>>.
    //
    // The prebuilt backend is moved into its own Option so we can mutably borrow
    // it separately from lazy_runtime.
    let mut prebuilt: Option<Box<dyn AsrBackend + Send>> = prebuilt_backend;

    // `pending_closed`: when true, skip the `notified()` wait and drain the
    // remaining queue immediately (runner has exited, no more notify_one calls).
    let mut pending_closed = false;

    loop {
        if !pending_closed {
            // Wait for a payload to be available, or the runner to signal closed.
            rt.block_on(flush_queue.notify.notified());
        }

        // Pop the oldest pending payload.
        let (payload, is_closed) = {
            let mut deque = flush_queue
                .deque
                .lock()
                .expect("flush queue mutex poisoned");
            let p = deque.pop_front();
            let closed = flush_queue.closed.load(Ordering::Acquire);
            (p, closed)
        };

        // Once the runner signals closed we keep draining without waiting.
        if is_closed {
            pending_closed = true;
        }

        let payload = match payload {
            Some(p) => {
                // Track that we have a payload in-flight; decremented via
                // InFlightGuard below regardless of how this iteration exits.
                flush_queue.in_flight.fetch_add(1, Ordering::AcqRel);
                p
            }
            None => {
                if pending_closed {
                    // Queue is empty and runner has exited — we're done.
                    tracing::debug!(
                        target: "orchestrator",
                        "ASR worker: flush queue closed and drained; exiting"
                    );
                    return;
                }
                // Spurious wakeup; continue waiting.
                continue;
            }
        };

        // Guard that decrements `in_flight` when dropped (on any loop iteration
        // exit path: return, continue, or falling through).
        struct InFlightGuard<'a>(&'a Arc<AtomicUsize>);
        impl Drop for InFlightGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _in_flight_guard = InFlightGuard(&flush_queue.in_flight);

        // Determine the backend to use.
        let backend: &mut dyn AsrBackend = if let Some(ref mut pb) = prebuilt {
            pb.as_mut()
        } else {
            // Lazy-initialise AsrRuntime on the first flush (production path).
            if lazy_runtime.is_none() {
                let result = rt.block_on(init_asr_runtime(&model_registry));
                match result {
                    Ok(Some(runtime)) => {
                        lazy_runtime = Some(runtime);
                    }
                    Ok(None) => {
                        // Model not available; skip this flush.
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(target: "orchestrator", "ASR runtime init failed: {e}");
                        let _ = event_tx.send(AppEvent::ErrorOccurred { error: e });
                        continue;
                    }
                }
            }
            lazy_runtime.as_mut().expect("just initialised")
        };

        // Wrap the per-flush call in `catch_unwind` so a panic inside
        // `transcribe_chunk` is caught, converted to `AppError::Internal`, and
        // emitted as `AppEvent::ErrorOccurred`. The worker then continues to the
        // next flush — one bad flush must not kill the worker or the recording.
        //
        // Per `architecture/cross-cutting.md`: "A panic inside a `spawn_blocking`
        // task must abort the parent orchestrator task and surface as a
        // recoverable `AppError`. The app does not exit on a single bad recording."
        //
        // `AssertUnwindSafe` is required because `&mut dyn AsrBackend`,
        // `broadcast::Sender`, and `mpsc::Sender` are not `UnwindSafe` by default.
        // We uphold the invariant: on unwind we do not use the backend again in
        // the same call (the closure is consumed), and the senders are only used
        // for a `send` before we return from the closure, so there is no risk of
        // leaving them in a broken intermediate state after an unwind.
        let call_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            process_flush_with_backend(payload, backend, &event_tx, &writer_cmd_tx)
        }));

        match call_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                // Backend returned an error; surface it and continue.
                tracing::warn!(target: "orchestrator", "ASR flush error: {e}");
                let _ = event_tx.send(AppEvent::ErrorOccurred { error: e });
            }
            Err(panic_payload) => {
                // Backend panicked; extract the message if possible.
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic payload".to_string()
                };
                tracing::warn!(
                    target: "orchestrator",
                    "ASR worker caught panic in transcribe_chunk: {msg}; \
                     continuing to next flush"
                );
                let _ = event_tx.send(AppEvent::ErrorOccurred {
                    error: AppError::Internal {
                        context: format!("ASR worker panicked: {msg}"),
                    },
                });
            }
        }
        // `_in_flight_guard` drops here, decrementing in_flight.
    }
}

/// Lazily initialise `AsrRuntime` on the first flush.
///
/// Returns `Ok(None)` if the model is not yet available (caller should skip
/// the flush). Returns `Ok(Some(runtime))` when initialisation succeeded.
async fn init_asr_runtime(model_registry: &ModelRegistry) -> AppResult<Option<AsrRuntime>> {
    let model_id = ModelId::from(ASR_MODEL_ID);

    // Fast path: check whether the model is available.
    let model_available = model_registry
        .list_models()
        .into_iter()
        .find(|s| s.id == model_id)
        .map(|s| matches!(s.status, ModelStatusState::Available { .. }))
        .unwrap_or(false);

    if !model_available {
        tracing::debug!(
            target: "orchestrator",
            "ASR model {} not available; skipping flush",
            ASR_MODEL_ID
        );
        return Ok(None);
    }

    let model_dir = model_registry.ensure(&model_id).await.map_err(|e| {
        tracing::warn!(target: "orchestrator", "model ensure failed: {e}");
        e
    })?;

    // Locate gguf and mmproj files. The convention is the first .gguf file
    // in the model dir. Mmproj is the file whose name contains "mmproj".
    let gguf_path = find_file_in_dir(&model_dir, |name| {
        name.ends_with(".gguf") && !name.contains("mmproj")
    })?;
    let mmproj_path = find_file_in_dir(&model_dir, |name| name.contains("mmproj"))?;

    match AsrRuntime::new(&gguf_path, &mmproj_path, AsrRuntimeConfig::default()) {
        Ok(runtime) => {
            tracing::info!(
                target: "orchestrator",
                "ASR runtime initialised from {}",
                model_dir.display()
            );
            Ok(Some(runtime))
        }
        Err(e) => {
            tracing::warn!(target: "orchestrator", "ASR runtime init failed: {e}");
            Err(e)
        }
    }
}

/// Process one flush payload with the provided ASR backend.
fn process_flush_with_backend(
    payload: FlushPayload,
    backend: &mut dyn AsrBackend,
    event_tx: &broadcast::Sender<AppEvent>,
    writer_cmd_tx: &mpsc::Sender<WriterCommand>,
) -> AppResult<()> {
    if payload.vad_segments.is_empty() {
        return Ok(());
    }

    let chunk = AudioChunk {
        samples: payload.samples,
        sample_rate: SAMPLE_RATE_HZ as u32,
        start_ms: payload.vad_segments.first().map(|(s, _)| *s).unwrap_or(0),
        end_ms: payload.vad_segments.last().map(|(_, e)| *e).unwrap_or(0),
    };

    let chunk_segments = match backend.transcribe_chunk(&chunk) {
        Ok(segs) => segs,
        Err(e) => {
            tracing::warn!(target: "orchestrator", "transcribe_chunk failed: {e}");
            return Err(e);
        }
    };

    // Combine all returned segment text into one string for proportional split.
    let combined_text: String = chunk_segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Re-split proportionally across the VAD sub-segments.
    let sub_segments = emit_segments_proportional(&combined_text, &payload.vad_segments);

    for seg in sub_segments {
        // Emit broadcast event.
        let _ = event_tx.send(AppEvent::TranscriptSegment {
            meeting_id: payload.meeting_id,
            segment: seg.clone(),
        });
        // Send to runner for persistence write.
        if let Err(e) = writer_cmd_tx.try_send(WriterCommand::WriteSegment(seg)) {
            tracing::warn!(
                target: "orchestrator",
                "writer command channel full or closed; transcript segment dropped from persistence: {e}"
            );
        }
    }

    Ok(())
}

/// Wait for the ASR worker to finish processing all queued flushes.
///
/// After dispatching the final flush before `stop`, the runner calls this to
/// ensure the ASR worker has processed all pending items and sent back all
/// `WriterCommand::WriteSegment` commands before `finalise()` is called.
///
/// Polls until both the queue is empty AND no payload is in-flight, draining
/// writer commands as they arrive. Times out after 30 s (covers the slowest
/// expected inference on the target hardware).
fn wait_for_asr_worker_drain(
    flush_queue: &FlushQueue,
    writer_cmd_rx: &mut mpsc::Receiver<WriterCommand>,
    writer: &mut MeetingWriter,
) {
    let timeout = Duration::from_secs(30);
    let poll_interval = Duration::from_millis(20);
    let drained = flush_queue.wait_all_processed(timeout, poll_interval);
    if !drained {
        tracing::warn!(
            target: "orchestrator",
            "ASR worker did not drain within 30 s; finalising without waiting further"
        );
    }
    // Drain any writer commands that arrived while we were waiting.
    drain_writer_commands(writer_cmd_rx, writer);
}

/// Drain any pending `WriterCommand`s from the ASR worker and apply them.
fn drain_writer_commands(
    rx: &mut mpsc::Receiver<WriterCommand>,
    writer: &mut MeetingWriter,
) {
    while let Ok(cmd) = rx.try_recv() {
        match cmd {
            WriterCommand::WriteSegment(seg) => {
                if let Err(e) = writer.write_transcript_segment(seg) {
                    tracing::warn!(
                        target: "orchestrator",
                        "write_transcript_segment failed: {e}"
                    );
                }
            }
        }
    }
}

/// Run the end-of-stream finalisation shared by the Recording-stop and
/// Paused-stop branches (TIMELINE-DRIFT #6).
///
/// Both stop paths must flush the VAD end-of-stream (so the last in-progress
/// utterance is closed and dispatched) and drain the accumulator before
/// finalising the writer. Previously only the Recording-stop branch did this;
/// stopping while Paused finalised directly and silently lost the last
/// utterance. This helper centralises the flush → accumulator-drain → ASR-worker
/// wait → transcript-write-drain → `writer.finalise` sequence so the two
/// branches cannot diverge again.
///
/// The caller is responsible for any sample/meter draining that is specific to
/// its branch (the Recording branch drains the live sample stream first; the
/// Paused branch has no pending samples to drain).
#[allow(clippy::too_many_arguments)]
fn finalise_on_stop(
    meta: Box<MeetingMeta>,
    reply: oneshot::Sender<AppResult<MeetingMeta>>,
    vad_opt: &mut Option<VadChunker>,
    acc: &mut Accumulator,
    flush_queue: &FlushQueue,
    writer_cmd_rx: &mut mpsc::Receiver<WriterCommand>,
    mut writer: MeetingWriter,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) {
    // End-of-stream VAD flush closes any in-progress segment.
    if let Some(vad) = vad_opt {
        match vad.flush_end_of_stream() {
            Ok(events) => {
                for ev in events {
                    if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                        acc.append(start_ms, end_ms, &samples);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "orchestrator",
                    "VAD flush_end_of_stream error: {e}"
                );
            }
        }
    }

    // Flush any remaining accumulator content before finalising.
    if !acc.is_empty() {
        let (samples, vad_segments) = acc.drain();
        let payload = FlushPayload { samples, vad_segments, meeting_id };
        dispatch_flush(flush_queue, payload, event_tx);
    }

    // Wait for the ASR worker to process any remaining flushes so that
    // transcript.json is fully written before finalise.
    wait_for_asr_worker_drain(flush_queue, writer_cmd_rx, &mut writer);

    // Drain any remaining transcript write commands.
    drain_writer_commands(writer_cmd_rx, &mut writer);

    let unboxed = *meta;
    let result = writer.finalise(unboxed.clone()).map(|_| unboxed);
    let _ = reply.send(result);
    tracing::info!(target: "orchestrator", "runner: exiting cleanly");
}

/// Find a file in `dir` matching `predicate`. Returns `AppError::Internal` if
/// the directory cannot be read or no matching file is found.
fn find_file_in_dir(
    dir: &std::path::Path,
    predicate: impl Fn(&str) -> bool,
) -> AppResult<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).map_err(|e| AppError::Internal {
        context: format!("read_dir {}: {e}", dir.display()),
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if predicate(name) {
                return Ok(path);
            }
        }
    }
    Err(AppError::Internal {
        context: format!("no matching file found in {}", dir.display()),
    })
}

// ---------------------------------------------------------------------------
// Offline re-transcribe (Phase 4)
// ---------------------------------------------------------------------------

/// A contiguous non-pause region of the decoded PCM, tagged with where it sits
/// on the pause-EXCLUDING clock (TIMELINE-DRIFT #4). `src_start..src_end` index
/// the original (pause-INCLUDING) `pcm`; `excl_start_ms` is the region's start
/// on the pause-excluding timeline (the cumulative duration of all kept audio
/// before it).
struct KeptRegion {
    src_start: usize,
    src_end: usize,
    excl_start_ms: u64,
}

/// Amplitude at or below which a sample counts as "silent" for pause detection.
/// The Opus encoder synthesises exact-zero frames for a pause; the lossy decode
/// reconstructs them as very-low-amplitude samples (the existing persistence
/// pause tests assert the pause region decodes to `< 0.02` peak), so a small
/// epsilon reliably catches encoder pause padding without flagging speech.
const PAUSE_SILENCE_EPS: f32 = 0.02;

/// Minimum length of a near-silent run to be treated as a recording **pause**
/// (and excluded from the pause-excluding timeline). Set comfortably above the
/// live accumulator's `MAX_GAP_MS` (3 s) inter-utterance cap so only genuine
/// user pauses — never a natural quiet gap the live capture clock would have
/// counted — are skipped.
const PAUSE_MIN_MS: u64 = 4000;

/// Split `pcm` into the non-pause regions to feed the offline VAD, excluding
/// encoder-synthesised pause padding from the timeline (TIMELINE-DRIFT #4).
///
/// A "pause" is a run of at least `PAUSE_MIN_MS` of near-silent
/// (`|x| <= PAUSE_SILENCE_EPS`) samples. Each returned [`KeptRegion`] carries
/// the source index range of kept audio plus its start offset on the
/// pause-EXCLUDING clock, which advances only over kept audio. Short quiet gaps
/// (below the threshold) are NOT split out — they remain part of a kept region,
/// matching the live capture clock which counts them.
fn pause_excluding_segments(pcm: &[f32]) -> Vec<KeptRegion> {
    let pause_min_samples = (PAUSE_MIN_MS as usize * SAMPLE_RATE_HZ as usize) / 1000;

    let mut regions: Vec<KeptRegion> = Vec::new();
    let mut excl_start_ms: u64 = 0;
    let mut region_start = 0usize; // start of the current kept region
    let mut i = 0usize;

    while i < pcm.len() {
        // Measure a near-silent run starting at `i`.
        if pcm[i].abs() <= PAUSE_SILENCE_EPS {
            let run_start = i;
            let mut j = i;
            while j < pcm.len() && pcm[j].abs() <= PAUSE_SILENCE_EPS {
                j += 1;
            }
            let run_len = j - run_start;

            if run_len >= pause_min_samples {
                // This silent run is a pause. Close the current kept region at
                // `run_start` (it ends just before the pause) and skip the run.
                if run_start > region_start {
                    let kept_len = run_start - region_start;
                    regions.push(KeptRegion {
                        src_start: region_start,
                        src_end: run_start,
                        excl_start_ms,
                    });
                    excl_start_ms += (kept_len as u64 * 1000) / SAMPLE_RATE_HZ;
                }
                // The next kept region begins after the pause; its
                // pause-excluding start is unchanged (the pause contributed no
                // pause-excluding time), exactly as the live capture clock froze
                // during the pause.
                region_start = j;
            }
            // Else: a short quiet gap — keep it in the current region.
            i = j;
        } else {
            i += 1;
        }
    }

    // Close the trailing kept region.
    if pcm.len() > region_start {
        regions.push(KeptRegion {
            src_start: region_start,
            src_end: pcm.len(),
            excl_start_ms,
        });
    }

    regions
}

/// Re-run transcription over a fully-decoded PCM buffer, reusing the live
/// pipeline's batched-VAD [`Accumulator`] + ASR-dispatch machinery.
///
/// This is the offline counterpart of [`run_drain_loop`]: instead of draining a
/// live capture stream, it feeds an already-decoded `pcm` buffer (the
/// pause-INCLUDING 16 kHz mono samples from
/// `persistence::reader::read_audio_pcm`) through the **same** `VadChunker` →
/// [`Accumulator`] → [`process_flush_with_backend`]-equivalent path. The 30 s
/// encoder-window constraint and the silence-preservation rule therefore hold
/// identically — the accumulator zero-pads inter-utterance gaps (capped at
/// `MAX_GAP_MS`) exactly as the live path does.
///
/// Differences from the live path:
/// - No flush queue / ASR worker thread: the work runs synchronously on the
///   caller's `spawn_blocking` thread, one accumulator flush at a time, so the
///   produced segments can be collected in order and returned to the caller for
///   `transcript.json` rewrite + index upsert.
/// - `AppEvent::TranscriptSegment` is emitted as each segment is produced
///   (same event the live path emits), so the webview's transcript pane
///   appends them live.
///
/// # Pause-EXCLUDING timeline (TIMELINE-DRIFT #4)
///
/// `audio.opus` is recorded **pause-INCLUDING**: the encoder pads every pause
/// with synthesised silence, so `read_audio_pcm` returns a buffer whose duration
/// equals wall-clock recording time. The live transcript, however, is on the
/// **pause-EXCLUDING** capture-sample clock (`Segment::start_ms` —
/// `architecture/cross-cutting.md` "Notes paragraph-anchor clock"): the capture
/// forwarder's sample clock does not advance while paused, so the live VAD never
/// sees the pause silence and post-pause segments continue contiguously on the
/// pause-excluding timeline.
///
/// To produce timestamps that match the live ones, the offline feeder must
/// reproduce that pause-excluding clock. Pause boundaries are **not** persisted
/// anywhere (`MeetingMeta` has no pause map), but the encoder-synthesised pause
/// padding is recoverable from the audio itself: it is a contiguous run of
/// near-silent samples far longer than any natural inter-utterance gap (a user
/// pause is typically many seconds; the live accumulator never zero-pads more
/// than `MAX_GAP_MS`). The feeder therefore detects long near-silent runs
/// (`PAUSE_*` constants below), **skips** them, and feeds only the non-pause
/// audio to the VAD with a clock that advances over kept audio only — exactly
/// reconstructing the pause-excluding capture clock. Combined with the
/// VAD-realignment fix (#3) this yields `Segment::start_ms` matching the live
/// path within frame tolerance.
///
/// Natural quiet gaps (mic live, room quiet) are NOT skipped: they are below the
/// `PAUSE_MIN_MS` length threshold, so they stay on the timeline just as the
/// live capture clock counts them.
///
/// Returns all produced segments in start-time order.
pub(crate) fn re_transcribe_buffer(
    pcm: &[f32],
    backend: &mut dyn AsrBackend,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) -> AppResult<Vec<Segment>> {
    // Same VAD model + config the live path uses. If the VAD model is missing
    // we cannot segment the audio, so re-transcribe yields no segments.
    let model_path = vad_chunker::default_model_path();
    let mut vad = VadChunker::open(&model_path, VadConfig::default()).map_err(|e| {
        tracing::warn!(
            target: "orchestrator",
            "re_transcribe: VAD chunker unavailable ({e}); cannot segment audio"
        );
        e
    })?;

    let mut acc = Accumulator::new();
    let mut produced: Vec<Segment> = Vec::new();

    // Build the pause-excluding feed: the kept (non-pause) sample ranges, each
    // tagged with its start offset on the pause-EXCLUDING clock.
    let kept = pause_excluding_segments(pcm);

    // 1600 samples = 100 ms at 16 kHz — the same batch granularity the live
    // feeder uses, so VAD framing is identical to the live path.
    const BATCH_SAMPLES: usize = 1600;
    for region in &kept {
        let region_pcm = &pcm[region.src_start..region.src_end];
        // `start_ms` runs on the pause-EXCLUDING clock: it begins at the
        // region's excluding-offset and advances only over kept audio. At a
        // region boundary (a skipped pause) it continues from where the
        // previous region left off, so the VAD sees a contiguous pause-excluding
        // timeline — matching the live capture clock.
        let mut start_ms = region.excl_start_ms;
        for chunk in region_pcm.chunks(BATCH_SAMPLES) {
            let duration_ms = (chunk.len() as u64 * 1000) / SAMPLE_RATE_HZ;
            match vad.process_samples(chunk, start_ms) {
                Ok(events) => {
                    for ev in events {
                        if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                            acc.append(start_ms, end_ms, &samples);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(target: "orchestrator", "re_transcribe VAD error: {e}");
                }
            }
            start_ms += duration_ms;

            // Size-triggered flush, identical threshold to the live path.
            if acc.duration_secs() >= FLUSH_MIN_SECS {
                let (samples, vad_segments) = acc.drain();
                transcribe_one_flush(
                    samples,
                    vad_segments,
                    backend,
                    event_tx,
                    meeting_id,
                    &mut produced,
                )?;
            }
        }
    }

    // End-of-stream VAD flush closes any in-progress segment.
    match vad.flush_end_of_stream() {
        Ok(events) => {
            for ev in events {
                if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                    acc.append(start_ms, end_ms, &samples);
                }
            }
        }
        Err(e) => {
            tracing::warn!(target: "orchestrator", "re_transcribe flush_end_of_stream error: {e}");
        }
    }

    // Final flush of whatever remains in the accumulator.
    if !acc.is_empty() {
        let (samples, vad_segments) = acc.drain();
        transcribe_one_flush(
            samples,
            vad_segments,
            backend,
            event_tx,
            meeting_id,
            &mut produced,
        )?;
    }

    Ok(produced)
}

/// Transcribe a single accumulator flush synchronously and append the produced
/// segments to `out`, emitting an `AppEvent::TranscriptSegment` for each.
///
/// Mirrors [`process_flush_with_backend`] (same `AudioChunk` construction +
/// proportional re-split) but collects the segments for the offline caller
/// rather than queueing writer commands.
fn transcribe_one_flush(
    samples: Vec<f32>,
    vad_segments: Vec<(u64, u64)>,
    backend: &mut dyn AsrBackend,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    out: &mut Vec<Segment>,
) -> AppResult<()> {
    if vad_segments.is_empty() {
        return Ok(());
    }

    let chunk = AudioChunk {
        samples,
        sample_rate: SAMPLE_RATE_HZ as u32,
        start_ms: vad_segments.first().map(|(s, _)| *s).unwrap_or(0),
        end_ms: vad_segments.last().map(|(_, e)| *e).unwrap_or(0),
    };

    let chunk_segments = backend.transcribe_chunk(&chunk)?;

    let combined_text: String = chunk_segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let sub_segments = emit_segments_proportional(&combined_text, &vad_segments);

    for seg in sub_segments {
        let _ = event_tx.send(AppEvent::TranscriptSegment {
            meeting_id,
            segment: seg.clone(),
        });
        out.push(seg);
    }

    Ok(())
}

/// Resolve and construct the production `AsrRuntime` for the re-transcribe path.
///
/// Reuses the live-path model-resolution logic ([`init_asr_runtime`]): the ASR
/// model must already be `Available` in the registry. Returns `Ok(None)` when
/// the model is not available (the caller surfaces this as an error, since an
/// explicit user-triggered re-transcribe with no model is a failure, unlike the
/// live path's best-effort skip).
pub(crate) async fn build_asr_runtime_for_retranscribe(
    model_registry: &ModelRegistry,
) -> AppResult<Option<AsrRuntime>> {
    init_asr_runtime(model_registry).await
}

// ---------------------------------------------------------------------------
// Diarizer construction (Phase 6)
// ---------------------------------------------------------------------------

/// Manifest id of the bundled segmentation model (pyannote/segmentation-3.0,
/// MIT). See `resources/models.json` and `architecture/components.md` —
/// `diarizer`.
pub(crate) const DIARIZE_SEG_MODEL_ID: &str = "pyannote-segmentation-3-0";
/// Manifest id of the bundled speaker-embedding model (3D-Speaker CAM++ zh-cn
/// 16k-common, Apache-2.0).
pub(crate) const DIARIZE_EMB_MODEL_ID: &str = "3dspeaker-campplus-zh-cn-16k-common";

/// Lazily build the production `SherpaDiarizer`, mirroring
/// [`build_asr_runtime_for_retranscribe`].
///
/// Resolves both diarize model directories via `ModelRegistry::ensure`
/// (downloading + hash-verifying when absent), locates the single `.onnx` file
/// in each directory, and opens a `SherpaDiarizer` over the
/// `(segmentation, embedding)` pair with `DiarizerConfig::default()`.
///
/// Unlike the ASR runtime, the model directories are *ensured* (not merely
/// checked for `Available`): both the on-stop pass and the user-triggered
/// re-diarize are explicit operations, so a missing model is a downloadable
/// dependency rather than a best-effort skip. A resolution / load failure is an
/// error (`AppError::ModelLoad` / the registry's download error), surfaced to
/// the caller.
pub(crate) async fn build_diarizer(
    model_registry: &ModelRegistry,
) -> AppResult<diarizer::SherpaDiarizer> {
    let seg_id = ModelId::from(DIARIZE_SEG_MODEL_ID);
    let emb_id = ModelId::from(DIARIZE_EMB_MODEL_ID);

    let seg_dir = model_registry.ensure(&seg_id).await?;
    let emb_dir = model_registry.ensure(&emb_id).await?;

    let seg_onnx = find_file_in_dir(&seg_dir, |name| name.ends_with(".onnx"))?;
    let emb_onnx = find_file_in_dir(&emb_dir, |name| name.ends_with(".onnx"))?;

    let diarizer =
        diarizer::SherpaDiarizer::open(&seg_onnx, &emb_onnx, diarizer::DiarizerConfig::default())?;

    tracing::info!(
        target: "orchestrator",
        seg = %seg_onnx.display(),
        emb = %emb_onnx.display(),
        "diarizer initialised"
    );

    Ok(diarizer)
}

// ---------------------------------------------------------------------------
// Proportional word allocation (Phase 2)
// ---------------------------------------------------------------------------

/// Split the transcript `text` across `vad_segments` proportionally by
/// per-segment audio duration.
///
/// Returns one `Segment` per VAD segment. If `text` is empty, each segment
/// carries an empty text string (timestamps are preserved from the VAD layer).
/// The last segment absorbs any word-count rounding remainder.
pub(crate) fn emit_segments_proportional(
    text: &str,
    vad_segments: &[(u64, u64)],
) -> Vec<Segment> {
    if vad_segments.is_empty() {
        return Vec::new();
    }

    let total_ms: u64 = vad_segments
        .iter()
        .map(|(s, e)| e.saturating_sub(*s))
        .sum();

    let words: Vec<&str> = text.split_whitespace().collect();
    let n = vad_segments.len();
    let mut out = Vec::with_capacity(n);
    let mut word_idx = 0usize;

    for (i, (start_ms, end_ms)) in vad_segments.iter().enumerate() {
        let seg_ms = end_ms.saturating_sub(*start_ms);
        let take = if words.is_empty() {
            0
        } else if i == n - 1 {
            words.len() - word_idx
        } else if total_ms == 0 {
            0
        } else {
            let proportion = seg_ms as f64 / total_ms as f64;
            let count = (proportion * words.len() as f64).round() as usize;
            count.min(words.len() - word_idx)
        };

        let seg_text = words[word_idx..word_idx + take].join(" ");
        word_idx += take;

        out.push(Segment {
            start_ms: *start_ms,
            end_ms: *end_ms,
            text: seg_text,
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
        });
    }

    out
}

// ---------------------------------------------------------------------------
// Helpers (unchanged from Phase 1)
// ---------------------------------------------------------------------------

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
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience: ms-to-sample count at 16 kHz.
    fn ms_to_samples(ms: u64) -> usize {
        (ms as usize * SAMPLE_RATE_HZ as usize) / 1000
    }

    // -----------------------------------------------------------------------
    // Test 1: zero-pad between two segments
    // -----------------------------------------------------------------------

    /// Accumulator must zero-pad the gap between two VAD segments so the
    /// buffer length matches `(seg2.start_ms - seg1.start_ms) * 16`.
    #[test]
    fn accumulator_zero_pads_gap_between_two_segments() {
        let mut acc = Accumulator::new();

        // Segment 1: 0 – 1000 ms (1 s).
        let seg1_samples = vec![0.5f32; ms_to_samples(1000)];
        acc.append(0, 1000, &seg1_samples);

        // Segment 2: 2000 – 3000 ms (1 s). Gap = 1000 ms.
        let seg2_samples = vec![0.5f32; ms_to_samples(1000)];
        acc.append(2000, 3000, &seg2_samples);

        // Expected buffer: 3 s = 48_000 samples.
        let expected_len = ms_to_samples(3000);
        assert_eq!(
            acc.samples.len(),
            expected_len,
            "buffer length should be {} (3 s), got {}",
            expected_len,
            acc.samples.len()
        );

        // The gap (1 s, samples 16_000..32_000) must be silence.
        let gap_start = ms_to_samples(1000);
        let gap_end = ms_to_samples(2000);
        for s in &acc.samples[gap_start..gap_end] {
            assert!(
                s.abs() < 1e-6,
                "silence pad violated at offset [{gap_start}..{gap_end}]: {s}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: gap capped at MAX_GAP_MS
    // -----------------------------------------------------------------------

    /// A 5 s gap between segments must be capped at MAX_GAP_MS (3 s).
    #[test]
    fn accumulator_caps_large_gap_at_max_gap_ms() {
        let mut acc = Accumulator::new();

        let seg1 = vec![0.5f32; ms_to_samples(1000)];
        acc.append(0, 1000, &seg1);

        // Segment 2 starts at 6000 ms — gap is 5000 ms, exceeds MAX_GAP_MS=3000 ms.
        let seg2 = vec![0.5f32; ms_to_samples(1000)];
        acc.append(6000, 7000, &seg2);

        // Buffer should be: 1 s speech + 3 s capped silence + 1 s speech = 5 s.
        let expected_len = ms_to_samples(1000) + ms_to_samples(MAX_GAP_MS) + ms_to_samples(1000);
        assert_eq!(
            acc.samples.len(),
            expected_len,
            "buffer should be capped to {expected_len} samples, got {}",
            acc.samples.len()
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: flush triggers on size
    // -----------------------------------------------------------------------

    /// A 25 s+ buffer must be flagged as needing a flush.
    #[test]
    fn accumulator_flush_triggers_on_size() {
        let mut acc = Accumulator::new();
        // Append a 26 s segment.
        let samples = vec![0.0f32; ms_to_samples(26_000)];
        acc.append(0, 26_000, &samples);
        assert!(
            acc.duration_secs() >= FLUSH_MIN_SECS,
            "26 s buffer must meet flush_min_secs={}; got {} s",
            FLUSH_MIN_SECS,
            acc.duration_secs()
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: flush triggers on latency
    // -----------------------------------------------------------------------

    /// After 10 s of quiet (last_vad_end_at sufficiently old), a non-empty
    /// buffer should be flushed.
    ///
    /// We can't sleep 10 s in a unit test, so we set `last_vad_end_at` to
    /// a time well in the past and check the elapsed condition directly.
    #[test]
    fn accumulator_flush_triggers_on_latency() {
        let mut acc = Accumulator::new();
        let seg = vec![0.0f32; ms_to_samples(2000)]; // 2 s — well below flush_min
        acc.append(0, 2000, &seg);

        // Simulate 11 s of elapsed time by setting last_vad_end_at to the past.
        acc.last_vad_end_at = Some(Instant::now() - Duration::from_secs(11));

        let latency_window = Duration::from_secs_f32(LATENCY_WINDOW_SECS);
        let should_flush = !acc.is_empty()
            && acc
                .last_vad_end_at
                .map(|t| t.elapsed() >= latency_window)
                .unwrap_or(false);

        assert!(
            should_flush,
            "latency-window flush should trigger after >10 s of quiet"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: proportional word allocation
    // -----------------------------------------------------------------------

    /// `emit_segments_proportional` must produce N segments matching the VAD
    /// segment count, and the total word count across all segments must equal
    /// the word count of the input text.
    #[test]
    fn proportional_allocation_matches_vad_count_and_total_words() {
        let vad_segments = vec![(0u64, 1000u64), (1000, 3000), (3000, 4000)];
        let text = "alpha beta gamma delta epsilon zeta";
        let word_count = text.split_whitespace().count(); // 6

        let segments = emit_segments_proportional(text, &vad_segments);

        assert_eq!(
            segments.len(),
            vad_segments.len(),
            "segment count must match VAD segment count"
        );

        let total_words: usize = segments
            .iter()
            .map(|s| s.text.split_whitespace().count())
            .sum();
        assert_eq!(
            total_words, word_count,
            "total word count must be preserved across all sub-segments"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: empty text still produces one Segment per VAD segment
    // -----------------------------------------------------------------------

    /// When the ASR produces empty text, `emit_segments_proportional` must
    /// still return one `Segment` per VAD segment (timestamps preserved).
    #[test]
    fn proportional_allocation_empty_text_preserves_segment_count() {
        let vad_segments = vec![(0u64, 1000u64), (1000, 2000)];
        let segments = emit_segments_proportional("", &vad_segments);
        assert_eq!(segments.len(), 2);
        assert!(segments.iter().all(|s| s.text.is_empty()));
    }

    // -----------------------------------------------------------------------
    // Test 7: dispatch_flush drop-oldest behaviour
    // -----------------------------------------------------------------------

    /// When `dispatch_flush` is called with a full queue (FLUSH_CHANNEL_CAP=4),
    /// it must drop the OLDEST entry (front) and retain the newest (back).
    #[test]
    fn dispatch_flush_drops_oldest_when_queue_full() {
        use meeting_app_common::MeetingId;
        let (event_tx, _event_rx) = tokio::sync::broadcast::channel::<AppEvent>(16);

        let flush_queue = FlushQueue::new();
        let meeting_id = MeetingId::new();

        // Fill the queue to capacity with payloads tagged by their index in
        // vad_segments[0].0 (start_ms) so we can identify which was dropped.
        for i in 0..FLUSH_CHANNEL_CAP {
            let payload = FlushPayload {
                samples: vec![0.0f32; 100],
                vad_segments: vec![(i as u64 * 1000, i as u64 * 1000 + 500)],
                meeting_id,
            };
            dispatch_flush(&flush_queue, payload, &event_tx);
        }

        // Queue is now full (4 entries). Dispatching one more should drop index 0.
        let newest_start_ms = FLUSH_CHANNEL_CAP as u64 * 1000;
        let newest_payload = FlushPayload {
            samples: vec![0.0f32; 100],
            vad_segments: vec![(newest_start_ms, newest_start_ms + 500)],
            meeting_id,
        };
        dispatch_flush(&flush_queue, newest_payload, &event_tx);

        // Drain the queue and collect start_ms values.
        let deque = flush_queue.deque.lock().unwrap();
        let starts: Vec<u64> = deque
            .iter()
            .map(|p| p.vad_segments[0].0)
            .collect();

        // The oldest (start_ms = 0) must have been dropped.
        assert!(
            !starts.contains(&0),
            "oldest flush (start_ms=0) must have been dropped; got: {starts:?}"
        );
        // The newest must be present.
        assert!(
            starts.contains(&newest_start_ms),
            "newest flush (start_ms={newest_start_ms}) must be retained; got: {starts:?}"
        );
        // Queue must not exceed FLUSH_CHANNEL_CAP.
        assert_eq!(
            deque.len(),
            FLUSH_CHANNEL_CAP,
            "queue must remain at FLUSH_CHANNEL_CAP after drop-oldest; len={}",
            deque.len()
        );
    }

    // -----------------------------------------------------------------------
    // pause_excluding_segments (TIMELINE-DRIFT #4)
    // -----------------------------------------------------------------------

    /// A buffer with no long silent run is one kept region spanning the whole
    /// buffer, starting at excl_start_ms = 0.
    #[test]
    fn pause_excluding_no_pause_is_single_region() {
        // 2 s of "speech" (non-silent) with a SHORT (1 s) quiet gap in the
        // middle — below PAUSE_MIN_MS, so it must NOT split.
        let mut pcm = vec![0.5f32; ms_to_samples(1000)];
        pcm.extend(std::iter::repeat_n(0.0f32, ms_to_samples(1000))); // 1 s quiet < 4 s
        pcm.extend(std::iter::repeat_n(0.5f32, ms_to_samples(1000)));

        let regions = pause_excluding_segments(&pcm);
        assert_eq!(regions.len(), 1, "short quiet gap must not split the buffer");
        assert_eq!(regions[0].src_start, 0);
        assert_eq!(regions[0].src_end, pcm.len());
        assert_eq!(regions[0].excl_start_ms, 0);
    }

    /// A long near-silent run (a pause) splits the buffer into two kept regions,
    /// and the second region's pause-EXCLUDING start equals the duration of the
    /// FIRST region only — the pause contributes no pause-excluding time.
    #[test]
    fn pause_excluding_splits_on_long_silence_and_excludes_it() {
        // 1 s speech, 6 s pause (> PAUSE_MIN_MS = 4 s), 2 s speech.
        let speech_a = ms_to_samples(1000);
        let pause = ms_to_samples(6000);
        let speech_b = ms_to_samples(2000);

        let mut pcm = vec![0.5f32; speech_a];
        pcm.extend(std::iter::repeat_n(0.0f32, pause));
        pcm.extend(std::iter::repeat_n(0.5f32, speech_b));

        let regions = pause_excluding_segments(&pcm);
        assert_eq!(regions.len(), 2, "long pause must split into two regions");

        // Region 0: the first second of speech, starting at excl 0.
        assert_eq!(regions[0].src_start, 0);
        assert_eq!(regions[0].src_end, speech_a);
        assert_eq!(regions[0].excl_start_ms, 0);

        // Region 1: the post-pause speech. Its source start is AFTER the pause,
        // but its pause-EXCLUDING start is only 1000 ms (the kept duration of
        // region 0) — the 6 s pause is excluded from the timeline.
        assert_eq!(regions[1].src_start, speech_a + pause);
        assert_eq!(regions[1].src_end, pcm.len());
        assert_eq!(
            regions[1].excl_start_ms, 1000,
            "post-pause region must start at the pause-EXCLUDING offset (1 s), \
             not the pause-INCLUDING offset (7 s)"
        );
    }

    /// Trailing/leading pause padding is excluded; a pause at the very start
    /// shifts the first kept region's source start without consuming
    /// pause-excluding time.
    #[test]
    fn pause_excluding_handles_leading_pause() {
        let pause = ms_to_samples(5000);
        let speech = ms_to_samples(1000);
        let mut pcm = vec![0.0f32; pause];
        pcm.extend(std::iter::repeat_n(0.5f32, speech));

        let regions = pause_excluding_segments(&pcm);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].src_start, pause, "leading pause must be skipped");
        assert_eq!(regions[0].excl_start_ms, 0);
    }
}
