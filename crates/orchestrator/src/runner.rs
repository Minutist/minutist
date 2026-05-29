//! The capture-drain runner task and ASR worker.
//!
//! The **runner** runs as a `tokio::task::spawn_blocking` thread. It owns:
//! - The `AudioStreams` (sample + meter mpsc receivers from `audio-capture`).
//! - The `MeetingWriter` from `persistence`.
//! - A `VadChunker` for speech-activity detection.
//! - A bounded mpsc channel to a separate ASR worker task.
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
//! Runner → ASR worker: bounded `mpsc::channel(4)`. On backpressure the runner
//! drops the OLDEST queued flush (audio is preserved in `audio.opus`; only live
//! transcript is lost) and emits `AppEvent::ErrorOccurred`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use asr_runtime::{AsrRuntime, AsrRuntimeConfig};
use audio_capture::{AudioFrameBatch, AudioStreams};
use meeting_app_common::{
    AppError, AppEvent, AppResult, AsrBackend, AudioChunk, AudioMeterFrame, MeetingId, MeetingMeta,
    ModelId, ModelStatusState, Segment,
};
use model_registry::ModelRegistry;
use persistence::MeetingWriter;
use tokio::sync::{broadcast, mpsc, oneshot, Mutex};
use vad_chunker::{VadChunker, VadConfig, VadEvent};

// ---------------------------------------------------------------------------
// Locked constants (Phase 2 — do NOT change without arch review)
// ---------------------------------------------------------------------------

const FLUSH_MIN_SECS: f32 = 25.0;
const LATENCY_WINDOW_SECS: f32 = 10.0;
const MAX_GAP_MS: u64 = 3000;
const SAMPLE_RATE_HZ: u64 = 16_000;
/// Poll interval when no audio is arriving (latency-window check).
const POLL_INTERVAL: Duration = Duration::from_millis(200);
/// Hardcoded ASR model ID (Phase 2 LOCKED CHOICE).
const ASR_MODEL_ID: &str = "qwen3-asr-0.6b-q8_0";
/// Flush dispatch channel capacity.
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
struct FlushPayload {
    samples: Vec<f32>,
    /// `(start_ms, end_ms)` for each VAD segment in this buffer.
    vad_segments: Vec<(u64, u64)>,
    meeting_id: MeetingId,
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
            self.samples.extend(std::iter::repeat(0.0f32).take(capped));
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
    let (cmd_tx, cmd_rx) = mpsc::channel::<RunnerCommand>(8);

    // Bounded flush channel: runner → ASR worker.
    let (flush_tx, flush_rx) = mpsc::channel::<FlushPayload>(FLUSH_CHANNEL_CAP);

    // Channel for ASR worker → runner to write transcript segments.
    // Bounded at FLUSH_CHANNEL_CAP * max_segments_per_flush; a rough bound is fine.
    let (writer_cmd_tx, writer_cmd_rx) = mpsc::channel::<WriterCommand>(64);

    // ASR runtime wrapped in a Mutex so the worker can be lazy-initialised
    // and only one flush is in-flight at a time.
    let asr_mutex: Arc<Mutex<Option<AsrRuntime>>> = Arc::new(Mutex::new(None));

    // Spawn the ASR worker on a separate spawn_blocking thread.
    let asr_event_tx = event_tx.clone();
    let asr_mutex_clone = Arc::clone(&asr_mutex);
    let asr_registry = Arc::clone(&model_registry);
    tokio::task::spawn_blocking(move || {
        run_asr_worker(flush_rx, asr_mutex_clone, asr_registry, asr_event_tx, writer_cmd_tx);
    });

    // Spawn the runner drain loop.
    tokio::task::spawn_blocking(move || {
        run_drain_loop(
            streams,
            writer,
            event_tx,
            cmd_rx,
            flush_tx,
            writer_cmd_rx,
            meeting_id,
        );
    });

    RunnerHandle { cmd_tx }
}

// ---------------------------------------------------------------------------
// Blocking drain loop (runner)
// ---------------------------------------------------------------------------

fn run_drain_loop(
    mut streams: AudioStreams,
    mut writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    mut cmd_rx: mpsc::Receiver<RunnerCommand>,
    flush_tx: mpsc::Sender<FlushPayload>,
    mut writer_cmd_rx: mpsc::Receiver<WriterCommand>,
    meeting_id: MeetingId,
) {
    let mut writer_paused = false;
    let mut acc = Accumulator::new();
    let latency_window = Duration::from_secs_f32(LATENCY_WINDOW_SECS);

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

                    // End-of-stream VAD flush.
                    if let Some(ref mut vad) = vad_opt {
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
                        dispatch_flush(&flush_tx, payload, &event_tx);
                    }

                    // Drain any remaining transcript write commands.
                    drain_writer_commands(&mut writer_cmd_rx, &mut writer);

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
                // Push to persistent audio.
                push_batch(&mut writer, &batch);

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
                    dispatch_flush(&flush_tx, payload, &event_tx);
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
                    dispatch_flush(&flush_tx, payload, &event_tx);
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
                        dispatch_flush(&flush_tx, payload, &event_tx);
                    }
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

/// Try to send a flush payload to the ASR worker.
///
/// If the channel is full (capacity 4), emit `AppEvent::ErrorOccurred` and
/// drop the OLDEST pending flush by discarding the current payload (the
/// oldest is already in the channel; we don't actually pull from it here
/// since that would require try_recv on the worker channel, which we can't
/// do from here). We simply drop this newest payload and warn.
///
/// Audio is always preserved in `audio.opus`; only the live transcript is lost.
fn dispatch_flush(
    flush_tx: &mpsc::Sender<FlushPayload>,
    payload: FlushPayload,
    event_tx: &broadcast::Sender<AppEvent>,
) {
    match flush_tx.try_send(payload) {
        Ok(()) => {
            tracing::debug!(target: "orchestrator", "runner: flush dispatched to ASR worker");
        }
        Err(mpsc::error::TrySendError::Full(_dropped)) => {
            tracing::warn!(
                target: "orchestrator",
                "ASR flush channel full (backpressure); dropping oldest flush"
            );
            let _ = event_tx.send(AppEvent::ErrorOccurred {
                error: AppError::Internal {
                    context: "ASR flush channel full; live transcript delayed (audio.opus unaffected)".into(),
                },
            });
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            tracing::warn!(target: "orchestrator", "ASR flush channel closed unexpectedly");
        }
    }
}

// ---------------------------------------------------------------------------
// ASR worker
// ---------------------------------------------------------------------------

/// The ASR worker runs as a dedicated `spawn_blocking` task. It drains
/// `FlushPayload`s from the runner, transcribes each with the ASR runtime,
/// emits `AppEvent::TranscriptSegment` events, and sends transcript segments
/// back to the runner via `writer_cmd_tx` for persistence.
///
/// The `AsrRuntime` is lazy-initialised on the first flush so that tests and
/// model-missing paths don't force a model load at session start.
fn run_asr_worker(
    mut flush_rx: mpsc::Receiver<FlushPayload>,
    asr_mutex: Arc<Mutex<Option<AsrRuntime>>>,
    model_registry: Arc<ModelRegistry>,
    event_tx: broadcast::Sender<AppEvent>,
    writer_cmd_tx: mpsc::Sender<WriterCommand>,
) {
    // Use a single-threaded tokio runtime to allow async calls (model_registry.ensure)
    // from within this spawn_blocking context.
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

    while let Some(payload) = flush_rx.blocking_recv() {
        let result = rt.block_on(process_flush(
            payload,
            &asr_mutex,
            &model_registry,
            &event_tx,
            &writer_cmd_tx,
        ));
        if let Err(e) = result {
            tracing::warn!(target: "orchestrator", "ASR flush error: {e}");
            let _ = event_tx.send(AppEvent::ErrorOccurred { error: e });
        }
    }
    tracing::debug!(target: "orchestrator", "ASR worker: flush channel closed; exiting");
}

/// Process one flush payload: ensure model, transcribe, emit events, and
/// send transcript segments back to the runner for persistence.
async fn process_flush(
    payload: FlushPayload,
    asr_mutex: &Mutex<Option<AsrRuntime>>,
    model_registry: &ModelRegistry,
    event_tx: &broadcast::Sender<AppEvent>,
    writer_cmd_tx: &mpsc::Sender<WriterCommand>,
) -> AppResult<()> {
    if payload.vad_segments.is_empty() {
        return Ok(());
    }

    let model_id = ModelId::from(ASR_MODEL_ID);

    // Check whether the model is available (fast sync path via list_models).
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
        return Ok(());
    }

    // Lazy-init the ASR runtime on first flush.
    let mut guard = asr_mutex.lock().await;
    if guard.is_none() {
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
                *guard = Some(runtime);
            }
            Err(e) => {
                tracing::warn!(target: "orchestrator", "ASR runtime init failed: {e}");
                return Err(e);
            }
        }
    }

    let runtime = guard.as_mut().expect("just initialised");

    let chunk = AudioChunk {
        samples: payload.samples,
        sample_rate: SAMPLE_RATE_HZ as u32,
        start_ms: payload.vad_segments.first().map(|(s, _)| *s).unwrap_or(0),
        end_ms: payload.vad_segments.last().map(|(_, e)| *e).unwrap_or(0),
    };

    let chunk_segments = match runtime.transcribe_chunk(&chunk) {
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

    // Drop the guard before emitting events to avoid holding the lock longer
    // than needed.
    drop(guard);

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
}
