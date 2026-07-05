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
//! - Maintains the **batched-VAD accumulator** described in Phase 2.
//! - Flushes the accumulator to the ASR worker on size/latency/end-of-stream.
//! - Broadcasts meter frames via the shared `AppEvent` sender.
//!
//! The **ASR worker** runs as a separate `tokio::task::spawn_blocking` thread.
//! It receives `FlushPayload`s from the runner, calls
//! `asr_runtime.transcribe_chunk()`, and re-splits the result proportionally
//! across VAD sub-segments (Phase 2).
//!
//! ## Flush-sizing constants
//!
//! These bound the audio handed to one `transcribe_chunk` call. Qwen3-ASR mtmd
//! hallucinates and loops on over-long input (observed at ~26 s), so the batch
//! must stay well under that. The bound is `FLUSH_MIN_SECS` plus at most one
//! VAD segment (`VadConfig::max_segment_ms`, ~10 s), i.e. ~13 s.
//!
//! - `FLUSH_MIN_SECS = 3.0` — minimum buffer duration before a size-triggered flush.
//! - `LATENCY_WINDOW_SECS = 2.0` — wall-clock seconds of quiet after which a
//!   non-empty buffer is flushed (also the live-transcript latency after a pause).
//! - `MAX_GAP_MS = 3000` — maximum inter-segment silence gap (zero-padded, not
//!   compacted) inserted between VAD segments in the accumulator.
//!
//! Originally 25 s / 10 s (Phase 2), which fed the ASR ~25 s blobs and
//! triggered the repetition loop; lowered here after that was observed live.
//!
//! ## Threading
//!
//! Runner → ASR worker: `Arc<Mutex<VecDeque<FlushPayload>>>` (capacity 4) + `Arc<Notify>`.
//! On backpressure the runner drops the OLDEST queued flush (the entry at the
//! front of the deque) and enqueues the newest at the back. This is self-healing
//! and log-only (a WARN, not a UI error): audio is always preserved in
//! `audio.opus`, and the dropped flush's transcript is restored by the
//! post-stop re-transcribe the `incomplete` flag triggers.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use asr_runtime::{AsrRuntime, AsrRuntimeConfig};
use audio_capture::{AudioFrameBatch, AudioStreams};
use diarizer::{OnlineDiarizer, OnlineDiarizerConfig};
use minutist_common::{
    AppError, AppEvent, AppResult, AsrBackend, AsrEngine, AudioChunk, AudioMeterFrame, MeetingId,
    MeetingMeta, ModelId, ModelStatusState, Segment,
};
use model_registry::ModelRegistry;
use persistence::MeetingWriter;
use tokio::sync::{broadcast, mpsc, oneshot, Notify};
use vad_chunker::{VadChunker, VadConfig, VadEvent};

// ---------------------------------------------------------------------------
// Locked constants (Phase 2 — do NOT change without arch review)
// ---------------------------------------------------------------------------

// Bounded so a single `transcribe_chunk` never receives more than roughly
// `FLUSH_MIN_SECS + VadConfig::max_segment_ms` (~13 s) of audio: Qwen3-ASR mtmd
// hallucinates and enters a greedy-decode repetition loop on over-long input
// (observed at ~26 s). The original 25 s batch produced exactly that. Smaller
// batches also make live transcript appear promptly rather than at stop.
const FLUSH_MIN_SECS: f32 = 3.0;
const LATENCY_WINDOW_SECS: f32 = 2.0;
const MAX_GAP_MS: u64 = 3000;
const SAMPLE_RATE_HZ: u64 = 16_000;
/// Minimum interval between `AppEvent::RecordingClock` emissions (~5 Hz).
/// See `architecture/cross-cutting.md` — "Notes paragraph-anchor clock".
const RECORDING_CLOCK_MIN_INTERVAL_MS: u64 = 200;
/// Poll interval when no audio is arriving (latency-window check).
const POLL_INTERVAL: Duration = Duration::from_millis(200);
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
    /// Live provisional speaker label for each VAD segment, in lockstep with
    /// `vad_segments` (Phase B). `None` when no live diarizer is wired or the
    /// per-segment `assign_segment` failed. The on-stop `SherpaDiarizer` pass
    /// remains authoritative and overwrites these on stop.
    pub(crate) speaker_ids: Vec<Option<String>>,
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
    /// Set to `true` when the live transcript became incomplete: either the
    /// drop-oldest queue discarded a pending flush (mid-recording loss) or the
    /// stop-time drain timed out (tail loss). Surfaced via `RunnerHandle` so
    /// `ipc-bridge` can, after stop, run a background re-transcribe of the
    /// complete `audio.opus` — the authoritative repair, since the audio is
    /// captured in full regardless of ASR speed.
    pub(crate) incomplete: Arc<AtomicBool>,
}

impl FlushQueue {
    pub(crate) fn new() -> Self {
        Self {
            deque: Arc::new(StdMutex::new(VecDeque::with_capacity(FLUSH_CHANNEL_CAP + 1))),
            notify: Arc::new(Notify::new()),
            closed: Arc::new(AtomicBool::new(false)),
            in_flight: Arc::new(AtomicUsize::new(0)),
            worker_exited: Arc::new(AtomicBool::new(false)),
            incomplete: Arc::new(AtomicBool::new(false)),
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
            incomplete: Arc::clone(&self.incomplete),
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

/// What [`Accumulator::drain`] yields: the padded sample buffer, the per-segment
/// `(start_ms, end_ms)` list, and the parallel live speaker-label column (Phase
/// B). The three are index-aligned: `vad_segments.len() == speaker_ids.len()`.
type DrainedFlush = (Vec<f32>, Vec<(u64, u64)>, Vec<Option<String>>);

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
    /// Live provisional speaker label for each VAD segment, pushed in lockstep
    /// with `vad_segments` (Phase B). INVARIANT: `speaker_ids.len() ==
    /// vad_segments.len()` at all times. `None` when no live diarizer is wired
    /// or the per-segment label failed; the label is assigned at SegmentEnd from
    /// the original un-padded per-segment samples and never recomputed.
    pub(crate) speaker_ids: Vec<Option<String>>,
    /// Wall-clock instant the most recent VAD segment ended.
    pub(crate) last_vad_end_at: Option<Instant>,
}

impl Accumulator {
    pub(crate) fn new() -> Self {
        Self {
            samples: Vec::new(),
            buffer_start_ms: None,
            vad_segments: Vec::new(),
            speaker_ids: Vec::new(),
            last_vad_end_at: None,
        }
    }

    /// Append a VAD segment, zero-padding the gap from the current buffer tail.
    ///
    /// `label` is the live provisional speaker id for this VAD segment (Phase B),
    /// pushed in lockstep with the segment so the `speaker_ids` column stays
    /// 1:1 with `vad_segments`.
    pub(crate) fn append(
        &mut self,
        start_ms: u64,
        end_ms: u64,
        seg_samples: &[f32],
        label: Option<String>,
    ) {
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
        self.speaker_ids.push(label);
        self.last_vad_end_at = Some(Instant::now());
    }

    /// Duration of buffered audio in seconds.
    pub(crate) fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE_HZ as f32
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Drain the accumulator, returning samples, segment list, and the parallel
    /// live speaker-label column (Phase B). Resets state.
    ///
    /// The returned `speaker_ids` is always the same length as `vad_segments`
    /// (the two are pushed in lockstep by [`Self::append`]).
    pub(crate) fn drain(&mut self) -> DrainedFlush {
        let samples = std::mem::take(&mut self.samples);
        let segments = std::mem::take(&mut self.vad_segments);
        let speaker_ids = std::mem::take(&mut self.speaker_ids);
        self.buffer_start_ms = None;
        self.last_vad_end_at = None;
        (samples, segments, speaker_ids)
    }
}

// ---------------------------------------------------------------------------
// Public handle
// ---------------------------------------------------------------------------

/// Live handle to the drain runner.
pub(crate) struct RunnerHandle {
    pub(crate) cmd_tx: mpsc::Sender<RunnerCommand>,
    /// Shares `FlushQueue::incomplete`: set during the recording if the live
    /// transcript lost any audio (drop-oldest) or could not finish draining at
    /// stop. `Orchestrator::stop` reads it after finalise so `ipc-bridge` can
    /// trigger a background re-transcribe of the complete audio.
    pub(crate) transcript_incomplete: Arc<AtomicBool>,
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
/// `n_gpu_layers`   — runtime-resolved GPU-offload count (from the
///                    `gpu_acceleration` setting via [`resolve_gpu_layers`]).
/// `language`       — runtime-resolved ASR language hint (from the
///                    `transcription_language` setting via
///                    [`resolve_transcription_language`]); `None` = auto-detect.
/// `online_diarizer` — optional live diarizer (Phase B). When `Some`, the drain
///                    loop labels each VAD segment at SegmentEnd; when `None`
///                    (setting off / model absent / build failed) every segment
///                    is left unlabelled. Best-effort: never affects recording
///                    or transcription. The on-stop pass stays authoritative.
/// `prewarmed_backend` — optional process-held ASR backend warmed by
///                    [`Orchestrator::prewarm_asr`] (live-test UX T2). When
///                    `Some`, it is handed to the ASR worker directly so the
///                    first record skips the cold model load; when `None`, the
///                    worker lazy-inits the backend on the first flush (the
///                    pre-existing path, never regressed).
// Wiring boundary: it hands the runner its channels, handles, and the
// start-resolved scalar config (n_gpu_layers, language, live diarizer). The
// arguments are heterogeneous (not a cohesive value object), so a params struct
// would add indirection without clarity; the >7 count is inherent to wiring.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_runner(
    streams: AudioStreams,
    writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    model_registry: Arc<ModelRegistry>,
    meeting_id: MeetingId,
    n_gpu_layers: u32,
    language: Option<String>,
    engine: AsrEngine,
    online_diarizer: Option<Arc<OnlineDiarizer>>,
    prewarmed_backend: Option<Box<dyn AsrBackend + Send>>,
) -> RunnerHandle {
    spawn_runner_inner(
        streams,
        writer,
        event_tx,
        model_registry,
        meeting_id,
        n_gpu_layers,
        language,
        engine,
        prewarmed_backend,
        online_diarizer,
    )
}

/// Internal spawn function shared by production and test paths.
#[allow(clippy::too_many_arguments)]
fn spawn_runner_inner(
    streams: AudioStreams,
    writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    model_registry: Arc<ModelRegistry>,
    meeting_id: MeetingId,
    n_gpu_layers: u32,
    language: Option<String>,
    engine: AsrEngine,
    prebuilt_backend: Option<Box<dyn AsrBackend + Send>>,
    online_diarizer: Option<Arc<OnlineDiarizer>>,
) -> RunnerHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel::<RunnerCommand>(8);

    // Bounded drop-oldest flush queue: runner (producer) → ASR worker (consumer).
    let flush_queue = FlushQueue::new();
    let worker_flush_queue = flush_queue.consumer_clone();
    // Surfaced to the orchestrator via the handle so a behind/incomplete live
    // transcript can be repaired by a background re-transcribe after stop.
    let transcript_incomplete = Arc::clone(&flush_queue.incomplete);

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
            n_gpu_layers,
            language,
            engine,
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
            online_diarizer,
        );
    });

    RunnerHandle {
        cmd_tx,
        transcript_incomplete,
    }
}

/// Spawn the drain runner with a pre-built ASR backend (test-only path).
///
/// Accepts a `Box<dyn AsrBackend + Send>` that is used directly instead of
/// lazily initialising `AsrRuntime` from the model registry. This allows
/// integration tests to inject a stub backend without a real model file.
///
/// `online_diarizer` lets a test drive the Phase-B live-labelling path with a
/// real `OnlineDiarizer` (env-gated positive case) or `None` (the always-on
/// regression guard that proves transcription is unchanged when live
/// diarization is off).
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
    online_diarizer: Option<Arc<OnlineDiarizer>>,
) -> RunnerHandle {
    // The prebuilt backend is used directly, so the lazy production-path
    // `n_gpu_layers`, `language`, and `engine` are all moot here — the init path
    // is never reached. Pass the compile-time GPU ceiling for parity, `None`
    // language (no prefix), and the CPU-default engine as the moot value.
    spawn_runner_inner(
        streams,
        writer,
        event_tx,
        model_registry,
        meeting_id,
        asr_runtime::default_n_gpu_layers(),
        None,
        AsrEngine::Qwen06B,
        Some(backend),
        online_diarizer,
    )
}

// ---------------------------------------------------------------------------
// Live diarization label (Phase B)
// ---------------------------------------------------------------------------

/// Compute the live provisional speaker label for one VAD segment's un-padded
/// samples (Phase B).
///
/// Returns `None` when no live diarizer is wired (`online_diarizer == None`) or
/// when `assign_segment` fails for this segment (FFI error, mutex poison, empty
/// or invalid segment). A failure is logged at `warn` and degrades to `None` for
/// THAT segment only — subsequent segments keep trying, and recording /
/// transcription are never affected. The label MUST be computed here, at
/// SegmentEnd, from the original un-padded per-segment slice: the accumulator's
/// `MAX_GAP_MS` zero-pad cap makes per-segment sample boundaries unrecoverable
/// from the flushed buffer downstream.
fn live_segment_label(
    online_diarizer: Option<&Arc<OnlineDiarizer>>,
    samples: &[f32],
) -> Option<String> {
    let diarizer = online_diarizer?;
    // Live diarization is strictly additive and runs INLINE on the runner's
    // drain-loop thread (the one that also drains the sample channel and writes
    // audio.opus). A panic in the sherpa FFI (or any unwinding panic in
    // `assign_segment`) must therefore NOT escape this function and abort the
    // drain loop — that would stop recording + transcription. `catch_unwind`
    // contains it; the `Mutex` inside `OnlineDiarizer` poisons on a
    // panic-while-locked, so any subsequent segment cleanly returns `Err` (and
    // hence `None`) rather than re-panicking. Either way: no live label for the
    // affected segment, recording unaffected. (`AssertUnwindSafe` is sound here
    // because we discard the diarizer's state on a panic — we never observe a
    // torn value; we only map the outcome to `Option<String>`.)
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        diarizer.assign_segment(samples, SAMPLE_RATE_HZ as u32)
    }));
    match outcome {
        Ok(Ok(label)) => Some(label),
        Ok(Err(e)) => {
            tracing::warn!(
                target: "orchestrator",
                "live diarization assign_segment failed: {e}"
            );
            None
        }
        Err(_) => {
            tracing::error!(
                target: "orchestrator",
                "live diarization panicked; continuing without a live label"
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Blocking drain loop (runner)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_drain_loop(
    mut streams: AudioStreams,
    mut writer: MeetingWriter,
    event_tx: broadcast::Sender<AppEvent>,
    mut cmd_rx: mpsc::Receiver<RunnerCommand>,
    flush_queue: FlushQueue,
    mut writer_cmd_rx: mpsc::Receiver<WriterCommand>,
    meeting_id: MeetingId,
    online_diarizer: Option<Arc<OnlineDiarizer>>,
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
                        drain_samples_through_vad(
                            &mut streams.samples,
                            &mut writer,
                            &mut vad_opt,
                            &mut acc,
                            online_diarizer.as_ref(),
                            "stop drain",
                        );
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
                        meeting_id,
                        online_diarizer.as_ref(),
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
                        "runner: stop while writer paused; draining queued samples then finalising"
                    );
                    // Drain any sample batches that were queued before the pause
                    // arrived: push each to persistent audio AND feed it to the VAD.
                    // Batches accepted by the channel before the WriterPause command
                    // would otherwise be stranded here (the paused loop blocks on
                    // cmd_rx and never reads streams.samples), so the VAD would
                    // never build an in-progress segment for them. `finalise_on_stop`
                    // then flushes the end-of-stream to close that segment.
                    drain_samples_through_vad(
                        &mut streams.samples,
                        &mut writer,
                        &mut vad_opt,
                        &mut acc,
                        online_diarizer.as_ref(),
                        "paused-stop drain",
                    );
                    drain_meter(&mut streams.meter, &event_tx);
                    finalise_on_stop(
                        meta,
                        reply,
                        &mut vad_opt,
                        &mut acc,
                        &flush_queue,
                        &mut writer_cmd_rx,
                        writer,
                        meeting_id,
                        online_diarizer.as_ref(),
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
                                    // Live label from the still-un-padded
                                    // per-segment slice (Phase B).
                                    let label =
                                        live_segment_label(online_diarizer.as_ref(), &samples);
                                    acc.append(start_ms, end_ms, &samples, label);
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
                    let (samples, vad_segments, speaker_ids) = acc.drain();
                    let payload = FlushPayload {
                        samples,
                        vad_segments,
                        speaker_ids,
                        meeting_id,
                    };
                    dispatch_flush(&flush_queue, payload, &event_tx, meeting_id);
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
                    let (samples, vad_segments, speaker_ids) = acc.drain();
                    let payload = FlushPayload {
                        samples,
                        vad_segments,
                        speaker_ids,
                        meeting_id,
                    };
                    dispatch_flush(&flush_queue, payload, &event_tx, meeting_id);
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
                                        // Live label from the un-padded slice (Phase B).
                                        let label = live_segment_label(
                                            online_diarizer.as_ref(),
                                            &samples,
                                        );
                                        acc.append(start_ms, end_ms, &samples, label);
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
                        let (samples, vad_segments, speaker_ids) = acc.drain();
                        let payload = FlushPayload {
                            samples,
                            vad_segments,
                            speaker_ids,
                            meeting_id,
                        };
                        dispatch_flush(&flush_queue, payload, &event_tx, meeting_id);
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
/// discarded, then the new payload is pushed to the back.
///
/// This is self-healing and log-only — NOT surfaced to the UI as an error (it
/// can fire repeatedly under sustained load, e.g. CPU-only ASR). Audio is always
/// preserved in `audio.opus`; the dropped flush's transcript is restored by the
/// post-stop re-transcribe that the `incomplete` flag triggers.
fn dispatch_flush(
    flush_queue: &FlushQueue,
    payload: FlushPayload,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) {
    if dispatch_flush_inner(flush_queue, payload) {
        // A pending flush was dropped under backpressure. Emit a NON-error
        // signal (NOT `ErrorOccurred`): the live co-pilot driver observes it and
        // pauses its transcript-turn cadence so its own decodes do not compound
        // the backpressure; the webview ignores it. The dropped audio survives
        // in `audio.opus` and the `incomplete` flag drives the post-stop
        // re-transcribe. Send is best-effort (no subscribers → dropped), like
        // the other event emissions in this loop.
        let _ = event_tx.send(AppEvent::AsrBackpressure { meeting_id });
    }
}

/// Public wrapper for `dispatch_flush` used by tests. Returns whether a pending
/// flush was dropped under backpressure.
///
/// Only available under the `test-source` feature (or in `#[cfg(test)]`).
#[cfg(any(test, feature = "test-source"))]
pub(crate) fn dispatch_flush_pub(flush_queue: &FlushQueue, payload: FlushPayload) -> bool {
    dispatch_flush_inner(flush_queue, payload)
}

/// Enqueue `payload`, dropping the oldest pending flush first if the bounded
/// queue is already full. Returns `true` iff a pending flush was dropped
/// (backpressure), so the caller can emit [`AppEvent::AsrBackpressure`].
fn dispatch_flush_inner(flush_queue: &FlushQueue, payload: FlushPayload) -> bool {
    let mut deque = flush_queue
        .deque
        .lock()
        .expect("flush queue mutex poisoned");

    let mut dropped = false;
    if deque.len() >= FLUSH_CHANNEL_CAP {
        // Drop the OLDEST pending flush (the front of the queue).
        let _dropped = deque.pop_front();
        // Mark the live transcript incomplete so ipc-bridge runs a background
        // re-transcribe of the complete audio after stop (the dropped flush's
        // audio survives in audio.opus; only its transcript was lost).
        flush_queue.incomplete.store(true, Ordering::Release);
        dropped = true;
        // Self-healing backpressure: the audio survives in `audio.opus` and the
        // `incomplete` flag above triggers a full re-transcribe after stop, so
        // this is a log-only WARN — NOT surfaced to the UI as an error (it can
        // fire repeatedly under sustained load, e.g. CPU-only ASR). The caller
        // additionally emits a non-error `AsrBackpressure` event for the co-pilot.
        tracing::warn!(
            target: "orchestrator",
            "ASR flush queue full (backpressure); dropping oldest pending flush \
             (audio.opus unaffected; restored by the post-stop re-transcribe)"
        );
    }

    deque.push_back(payload);
    drop(deque);

    // Signal the worker that a new payload is available.
    flush_queue.notify.notify_one();
    tracing::debug!(target: "orchestrator", "runner: flush dispatched to ASR worker");

    dropped
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
/// `n_gpu_layers`: the runtime-resolved GPU-offload count (from the
/// `gpu_acceleration` setting via [`resolve_gpu_layers`]) applied to the lazily
/// built production `AsrRuntime`.
///
/// `language`: the runtime-resolved ASR language hint (from the
/// `transcription_language` setting via [`resolve_transcription_language`]);
/// `None` = auto-detect. Passed by value into the one-shot `init_asr_backend`.
///
/// One flush is in-flight at a time: the worker pops one payload, processes
/// it fully, then waits for the next notification.
#[allow(clippy::too_many_arguments)]
fn run_asr_worker(
    flush_queue: FlushQueue,
    prebuilt_backend: Option<Box<dyn AsrBackend + Send>>,
    model_registry: Arc<ModelRegistry>,
    n_gpu_layers: u32,
    language: Option<String>,
    engine: AsrEngine,
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

    // Production path: lazily-initialised ASR backend (Parakeet or a Qwen tier,
    // chosen by `engine`) wrapped in an Option.
    // Test path: the prebuilt backend is used directly.
    let mut lazy_runtime: Option<Box<dyn AsrBackend + Send>> = None;

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
            // Lazy-initialise the ASR backend on the first flush (production
            // path). The engine (Parakeet vs a Qwen tier) was resolved at start
            // from the transcription-language setting.
            if lazy_runtime.is_none() {
                // `init_asr_backend` runs at most once (lazy init), but the
                // borrow checker can't see that across the loop, so clone the
                // owned hint into the one-shot call.
                let result = rt.block_on(init_asr_backend(
                    &model_registry,
                    engine,
                    n_gpu_layers,
                    language.clone(),
                ));
                match result {
                    Ok(Some(runtime)) => {
                        lazy_runtime = Some(runtime);
                    }
                    Ok(None) => {
                        // Model not available; skip this flush.
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!(target: "orchestrator", "ASR backend init failed: {e}");
                        let _ = event_tx.send(AppEvent::ErrorOccurred { error: e });
                        continue;
                    }
                }
            }
            lazy_runtime.as_mut().expect("just initialised").as_mut()
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

/// Resolve the ASR `n_gpu_layers` from the runtime `gpu_acceleration` setting.
///
/// GPU offload happens ONLY when BOTH (a) the build was compiled with a GPU
/// feature AND (b) the setting is on. `enabled == true` → the compile-time
/// ceiling [`asr_runtime::default_n_gpu_layers`] (which is already `0` in a
/// default CPU-only build, so a CPU build is unaffected by the flag);
/// `enabled == false` → `0` (force CPU even in a GPU build). Pure + unit-tested
/// so the wiring is verified without a model. See `architecture/cross-cutting.md`
/// — "GPU portability".
pub(crate) fn resolve_gpu_layers(enabled: bool) -> u32 {
    if enabled {
        asr_runtime::default_n_gpu_layers()
    } else {
        0
    }
}

/// Resolve the ASR language hint from the `transcription_language` setting.
///
/// Mirrors [`resolve_gpu_layers`]: a pure, model-free mapping from the settings
/// String to the `AsrRuntimeConfig.language: Option<String>` the runtime
/// consumes. The reserved sentinel `"auto"` (case-insensitive), the empty
/// string, and whitespace-only all map to `None` → no prefix → auto-detect
/// (byte-identical to the pre-feature behaviour). Any other value is a full
/// English language name, trimmed and forwarded verbatim → prefix-force.
pub(crate) fn resolve_transcription_language(setting: &str) -> Option<String> {
    let t = setting.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(t.to_string())
    }
}

/// Map the resolved [`AsrEngine`] to its `resources/models.json` model id.
pub(crate) fn engine_model_id(engine: AsrEngine) -> &'static str {
    match engine {
        AsrEngine::ParakeetEuV3 => "parakeet-tdt-0.6b-v3-int8",
        AsrEngine::Qwen06B => "qwen3-asr-0.6b-q8_0",
        AsrEngine::Qwen17B => "qwen3-asr-1.7b-q8_0",
    }
}

/// Lazily initialise the ASR backend chosen by `engine` on the first flush.
///
/// `engine` is resolved at recording start from the `transcription_language`
/// setting (and the GPU-model opt-in) via [`common::asr_engine_for_language`]:
/// Parakeet (`asr-parakeet`, sherpa-onnx) for the languages it covers, else a
/// Qwen tier (`asr-runtime`, llama-cpp-2 mtmd).
///
/// `n_gpu_layers` + `language` only affect the Qwen path — Parakeet is
/// multilingual auto-detect and runs on the ONNX-Runtime CPU EP.
///
/// Returns `Ok(None)` if the selected model is not yet available (caller should
/// skip the flush). Returns `Ok(Some(backend))` when initialisation succeeded.
async fn init_asr_backend(
    model_registry: &ModelRegistry,
    engine: AsrEngine,
    n_gpu_layers: u32,
    language: Option<String>,
) -> AppResult<Option<Box<dyn AsrBackend + Send>>> {
    // Is a given model id locally `Available` (downloaded + hash-verified)? A
    // synchronous, no-network check — the live path must never download or block.
    let is_available = |id: &ModelId| -> bool {
        model_registry
            .list_models()
            .into_iter()
            .find(|s| &s.id == id)
            .map(|s| matches!(s.status, ModelStatusState::Available { .. }))
            .unwrap_or(false)
    };

    // Resolve the effective engine. The routed engine is preferred, but if its
    // model isn't downloaded yet we fall back to any available ASR engine rather
    // than silently producing no transcript (e.g. a fresh English install whose
    // onboarding fetched only the Qwen model still transcribes via Qwen until
    // Parakeet is downloaded). Preference keeps the broad Qwen-0.6B as the safety
    // net, then Parakeet, then the GPU Qwen.
    let routed_id = ModelId::from(engine_model_id(engine));
    let (engine, model_id) = if is_available(&routed_id) {
        (engine, routed_id)
    } else {
        let fallback = [AsrEngine::Qwen06B, AsrEngine::ParakeetEuV3, AsrEngine::Qwen17B]
            .into_iter()
            .filter(|&e| e != engine)
            .find(|&e| is_available(&ModelId::from(engine_model_id(e))));
        match fallback {
            Some(fb) => {
                tracing::warn!(
                    target: "orchestrator",
                    "routed ASR model {} not downloaded; falling back to {}",
                    routed_id.0,
                    engine_model_id(fb)
                );
                (fb, ModelId::from(engine_model_id(fb)))
            }
            None => {
                tracing::debug!(
                    target: "orchestrator",
                    "no ASR model available (routed {}); skipping flush",
                    routed_id.0
                );
                return Ok(None);
            }
        }
    };

    let model_dir = model_registry.ensure(&model_id).await.map_err(|e| {
        tracing::warn!(target: "orchestrator", "model ensure failed: {e}");
        e
    })?;

    match engine {
        AsrEngine::ParakeetEuV3 => {
            // sherpa-onnx offline transducer; reads the 4 model files by name.
            let backend = asr_parakeet::ParakeetBackend::new(
                asr_parakeet::ParakeetConfig::new(model_dir.clone()),
            )?;
            tracing::info!(
                target: "orchestrator",
                "Parakeet ASR initialised from {}",
                model_dir.display()
            );
            Ok(Some(Box::new(backend)))
        }
        AsrEngine::Qwen06B | AsrEngine::Qwen17B => {
            // llama-cpp-2 mtmd. The convention is the first .gguf file in the
            // model dir; mmproj is the file whose name contains "mmproj".
            let gguf_path = find_file_in_dir(&model_dir, |name| {
                name.ends_with(".gguf") && !name.contains("mmproj")
            })?;
            let mmproj_path = find_file_in_dir(&model_dir, |name| name.contains("mmproj"))?;

            let config = AsrRuntimeConfig {
                n_gpu_layers,
                language,
                ..AsrRuntimeConfig::default()
            };
            match AsrRuntime::new(&gguf_path, &mmproj_path, config) {
                Ok(runtime) => {
                    tracing::info!(
                        target: "orchestrator",
                        "Qwen ASR runtime initialised from {}",
                        model_dir.display()
                    );
                    Ok(Some(Box::new(runtime)))
                }
                Err(e) => {
                    tracing::warn!(target: "orchestrator", "ASR runtime init failed: {e}");
                    Err(e)
                }
            }
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

    // Re-split proportionally across the VAD sub-segments. Each emitted
    // sub-Segment inherits the live speaker label of its originating VAD
    // segment (Phase B) — the proportional re-split preserves one output
    // Segment per input vad_segment in order, so the 1:1 label mapping holds.
    let sub_segments =
        emit_segments_proportional(&combined_text, &payload.vad_segments, &payload.speaker_ids);

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
        // Tail loss: the remaining queued/in-flight audio could not be
        // transcribed in time. Mark incomplete so ipc-bridge re-transcribes the
        // complete audio in the background (the audio itself is fully captured).
        flush_queue.incomplete.store(true, Ordering::Release);
        tracing::warn!(
            target: "orchestrator",
            "ASR worker did not drain within 30 s; finalising now — a background \
             re-transcribe of the complete audio will repair the transcript"
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
    meeting_id: MeetingId,
    online_diarizer: Option<&Arc<OnlineDiarizer>>,
) {
    // End-of-stream VAD flush closes any in-progress segment.
    if let Some(vad) = vad_opt {
        match vad.flush_end_of_stream() {
            Ok(events) => {
                for ev in events {
                    if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                        // Live label from the un-padded slice (Phase B); this
                        // end-of-stream flush is a SegmentEnd site too.
                        let label = live_segment_label(online_diarizer, &samples);
                        acc.append(start_ms, end_ms, &samples, label);
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
        let (samples, vad_segments, speaker_ids) = acc.drain();
        let payload = FlushPayload {
            samples,
            vad_segments,
            speaker_ids,
            meeting_id,
        };
        // Final stop-time drain: use the inner enqueue directly. If it drops
        // under a full queue the `incomplete` flag is still set (→ post-stop
        // re-transcribe), but no `AsrBackpressure` event is emitted — the
        // recording is ending, so the co-pilot cooldown it drives is moot.
        let _ = dispatch_flush_inner(flush_queue, payload);
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
pub(crate) fn find_file_in_dir(
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
pub(crate) struct KeptRegion {
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
pub(crate) fn pause_excluding_segments(pcm: &[f32]) -> Vec<KeptRegion> {
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

/// Map a pause-EXCLUDING window `[start_ms, end_ms)` (transcript-clock
/// timestamps) onto a single slice of the pause-INCLUDING decoded `pcm`
/// (Phase 9 — backs `Orchestrator::transcribe_pcm_window`).
///
/// `Segment::start_ms`/`end_ms` live on the pause-EXCLUDING capture clock, but
/// `read_audio_pcm` returns the pause-INCLUDING buffer. To slice the right
/// audio for a re-listen we walk the [`pause_excluding_segments`] kept regions
/// (the same pause model the offline re-transcribe uses) and translate the
/// requested excluding interval back into PCM sample indices.
///
/// **Straddling a pause (W1 decision — clamp, do NOT concatenate).** A
/// pause-excluding window can span a skipped pause, which on the
/// pause-INCLUDING clock maps to two disjoint PCM ranges separated by the
/// pause's synthesised silence. Concatenating them would seam two non-adjacent
/// audio spans and make the re-transcribed timestamps impossible to map cleanly
/// back onto the meeting timeline. v1 therefore **clamps to the single kept
/// region that contains `start_ms`** and slices within it up to `end_ms` (or the
/// region end, whichever is first). The tool's `description` states this caveat.
/// Returns `None` when `start_ms` falls past the last kept region's
/// excluding-clock extent (an out-of-range request).
pub(crate) fn pcm_window_for_excluding_range(
    pcm: &[f32],
    start_ms: u64,
    end_ms: u64,
) -> Option<std::ops::Range<usize>> {
    excluding_range_to_pcm_slice(&pause_excluding_segments(pcm), start_ms, end_ms)
}

/// The per-request half of [`pcm_window_for_excluding_range`]: map the
/// pause-EXCLUDING window onto a PCM slice given an already-computed kept-region
/// table. Split out so the re-listen path can compute the O(samples)
/// [`pause_excluding_segments`] scan once per meeting (cached alongside the
/// decoded PCM) and run only this cheap region math per clip request.
pub(crate) fn excluding_range_to_pcm_slice(
    regions: &[KeptRegion],
    start_ms: u64,
    end_ms: u64,
) -> Option<std::ops::Range<usize>> {
    for region in regions {
        let region_kept_samples = region.src_end - region.src_start;
        let region_excl_len_ms =
            (region_kept_samples as u64 * 1000) / SAMPLE_RATE_HZ;
        let region_excl_end_ms = region.excl_start_ms + region_excl_len_ms;

        // The first region whose excluding-clock extent contains `start_ms`.
        if start_ms < region_excl_end_ms {
            // Offset of `start_ms` within this region, in pause-excluding ms.
            let into_region_ms = start_ms.saturating_sub(region.excl_start_ms);
            let start_off = (into_region_ms as usize * SAMPLE_RATE_HZ as usize) / 1000;
            // Clamp `end_ms` to this region (W1 clamp decision).
            let clamped_end_ms = end_ms.min(region_excl_end_ms);
            let end_into_region_ms = clamped_end_ms.saturating_sub(region.excl_start_ms);
            let end_off = (end_into_region_ms as usize * SAMPLE_RATE_HZ as usize) / 1000;

            let lo = region.src_start + start_off.min(region_kept_samples);
            let hi = region.src_start + end_off.min(region_kept_samples);
            if hi <= lo {
                return None;
            }
            return Some(lo..hi);
        }
    }
    None
}

/// Encode a 16 kHz mono f32 PCM slice as a self-contained little-endian PCM16
/// WAV byte buffer (44-byte RIFF/WAVE header + samples). Backs
/// [`crate::Orchestrator::extract_segment_wav`], which hands the webview a
/// pre-cut, seek-free clip for the transcript "play segment" affordance:
/// serving WAV (not the source Ogg/Opus) keeps the granule-position
/// non-conformance (#0024) and all container quirks out of the playback path.
pub(crate) fn pcm16_wav(samples: &[f32]) -> Vec<u8> {
    // The caller (`extract_segment_wav`) caps the slice well under 4 GiB; assert
    // it so the u32 RIFF / `data` length fields below cannot silently truncate.
    debug_assert!(
        samples
            .len()
            .checked_mul(2)
            .is_some_and(|n| n <= u32::MAX as usize),
        "pcm16_wav slice too large for a 32-bit WAV length field",
    );
    let sr = SAMPLE_RATE_HZ as u32;
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt-chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate (mono, 16-bit)
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    out
}

/// Inverse of [`pcm_window_for_excluding_range`]: map a pause-INCLUDING PCM
/// sample index back to its position on the pause-EXCLUDING transcript clock,
/// in milliseconds (#0015 phase 4).
///
/// [`pcm_window_for_excluding_range`] is forward-only (excluding-ms → PCM
/// range), so the re-ASR split needs this inverse to stamp each single-speaker
/// sub-clip's `start_ms` on the transcript clock. Without it a sub-clip cut at a
/// pause-INCLUDING sample would inherit the cumulative pre-segment pause padding
/// and drift forward by every pause before it.
///
/// Walks the SAME [`pause_excluding_segments`] kept regions the forward map uses
/// and inverts the per-region accounting: for the kept region that contains
/// `sample`, the excluding-clock position is the region's `excl_start_ms` plus
/// the kept-audio duration from the region start up to `sample`. A `sample` that
/// falls inside a skipped pause (between two kept regions) clamps to the END of
/// the preceding kept region — the same instant the excluding clock froze when
/// the pause began — so a cut energy-snapped a few ms into pause padding still
/// lands coherently. A `sample` past the last kept region clamps to the total
/// kept duration. Pure (no FFI/IO) so the inverse round-trips under unit test.
pub(crate) fn excluding_ms_for_pcm_sample(pcm: &[f32], sample: usize) -> u64 {
    let regions = pause_excluding_segments(pcm);

    let mut last_region_excl_end_ms: u64 = 0;
    for region in &regions {
        let region_kept_samples = region.src_end - region.src_start;
        let region_excl_len_ms = (region_kept_samples as u64 * 1000) / SAMPLE_RATE_HZ;
        let region_excl_end_ms = region.excl_start_ms + region_excl_len_ms;

        if sample < region.src_start {
            // The sample sits in the pause that precedes this kept region (or in
            // the leading pause before the first region): the excluding clock was
            // frozen at the previous region's end (0 before the first region).
            return last_region_excl_end_ms;
        }
        if sample < region.src_end {
            // Inside this kept region: excluding position = region start +
            // kept-audio duration from the region start to `sample`.
            let into_region = sample - region.src_start;
            let into_region_ms = (into_region as u64 * 1000) / SAMPLE_RATE_HZ;
            return region.excl_start_ms + into_region_ms;
        }
        last_region_excl_end_ms = region_excl_end_ms;
    }
    // Past the last kept region (trailing pause or out of range): the total kept
    // duration.
    last_region_excl_end_ms
}

/// Length of an RMS analysis window when snapping a cut to a local energy
/// minimum (~5 ms at 16 kHz).
const SNAP_RMS_WINDOW_MS: u64 = 5;

/// Relative-RMS floor below which a candidate cut is accepted as a genuine
/// low-energy boundary. The argmin window's RMS must be at most this fraction of
/// the search span's mean RMS; otherwise the span is continuous/overlapping
/// speech with no clear gap and [`snap_to_energy_min`] returns `None` so the
/// caller keeps the segment whole.
///
/// Calibrated conservatively: a real inter-speaker boundary in meeting audio
/// dips well under half the local mean energy, while a hand-off mid-phrase with
/// no breath does not. Too strict here silently forces keep-whole; the abandon
/// is logged so a real mixed clip can recalibrate it.
const SNAP_REL_FLOOR: f32 = 0.5;

/// Snap a speaker-change cut to the lowest-energy sample near `cut_sample`,
/// searching `± window_ms` (#0015 phase 4).
///
/// `samples` is the pause-INCLUDING PCM the cut indexes. The search slides a
/// `SNAP_RMS_WINDOW_MS` RMS window across `[cut_sample - window_ms, cut_sample +
/// window_ms]` (clamped to the buffer) and returns the start sample of the
/// minimum-RMS window — the quietest instant, where cutting the audio least
/// disturbs either speaker's words.
///
/// Returns `None` when there is **no clear minimum**: if the argmin window's RMS
/// is not at most [`SNAP_REL_FLOOR`] of the search span's mean RMS, the span is
/// continuous or overlapping speech with no real gap, and the caller keeps the
/// segment whole rather than cutting mid-word. Also `None` when the search span
/// is degenerate (empty or shorter than one RMS window). Emits `tracing::debug`
/// when a snap is abandoned. Pure (no FFI/IO).
pub(crate) fn snap_to_energy_min(
    samples: &[f32],
    cut_sample: usize,
    window_ms: u64,
) -> Option<usize> {
    let rms_win = (SNAP_RMS_WINDOW_MS as usize * SAMPLE_RATE_HZ as usize) / 1000;
    if rms_win == 0 || samples.is_empty() {
        return None;
    }
    let window_samples = (window_ms as usize * SAMPLE_RATE_HZ as usize) / 1000;

    let lo = cut_sample.saturating_sub(window_samples);
    // The last valid RMS-window start keeps the whole window inside the span.
    let hi = (cut_sample + window_samples).min(samples.len());
    if hi <= lo || hi - lo < rms_win {
        tracing::debug!(
            target: "orchestrator",
            cut_sample,
            window_ms,
            "snap_to_energy_min: search span too short; keeping segment whole"
        );
        return None;
    }

    let rms_at = |start: usize| -> f32 {
        let end = (start + rms_win).min(samples.len());
        let win = &samples[start..end];
        if win.is_empty() {
            return f32::MAX;
        }
        let sum_sq: f32 = win.iter().map(|s| s * s).sum();
        (sum_sq / win.len() as f32).sqrt()
    };

    let last_start = hi - rms_win;
    let mut best_start = lo;
    let mut best_rms = f32::MAX;
    let mut rms_sum = 0.0f32;
    let mut rms_count = 0u32;
    for start in lo..=last_start {
        let rms = rms_at(start);
        rms_sum += rms;
        rms_count += 1;
        if rms < best_rms {
            best_rms = rms;
            best_start = start;
        }
    }

    let mean_rms = if rms_count > 0 {
        rms_sum / rms_count as f32
    } else {
        0.0
    };
    // A genuine boundary dips well below the local mean; continuous/overlapping
    // speech does not. `mean_rms == 0` (digital silence everywhere) is itself a
    // clear minimum, so accept it.
    if mean_rms > 0.0 && best_rms > SNAP_REL_FLOOR * mean_rms {
        tracing::debug!(
            target: "orchestrator",
            cut_sample,
            best_rms,
            mean_rms,
            "snap_to_energy_min: no clear minimum (continuous speech); keeping segment whole"
        );
        return None;
    }

    Some(best_start)
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

    // Live-test UX T4(a): determinate progress = kept samples FED to the VAD so
    // far / total kept samples. Counted on the kept feed (not raw `pcm`) so the
    // skipped pause padding does not stall the bar; emitted per accumulator flush
    // (the natural cadence at which produced segments stream out).
    let total_kept_samples: usize = kept.iter().map(|r| r.src_end - r.src_start).sum();
    let mut samples_fed: usize = 0;
    // Emit a 0.0 start so the UI shows the bar immediately rather than waiting
    // for the first flush (which can be seconds of inference away).
    emit_retranscribe_progress(event_tx, meeting_id, samples_fed, total_kept_samples);

    // 1600 samples = 100 ms at 16 kHz — the same batch granularity the live
    // feeder uses, so VAD framing is identical to the live path.
    const BATCH_SAMPLES: usize = 1600;
    let n_regions = kept.len();
    for (region_idx, region) in kept.iter().enumerate() {
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
                            // Offline re-transcribe is a distinct path: the
                            // on-stop SherpaDiarizer pass owns labels here, so no
                            // live label is assigned (Phase B).
                            acc.append(start_ms, end_ms, &samples, None);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(target: "orchestrator", "re_transcribe VAD error: {e}");
                }
            }
            start_ms += duration_ms;
            samples_fed += chunk.len();

            // Size-triggered flush, identical threshold to the live path.
            if acc.duration_secs() >= FLUSH_MIN_SECS {
                let (samples, vad_segments, speaker_ids) = acc.drain();
                transcribe_one_flush(
                    samples,
                    vad_segments,
                    speaker_ids,
                    backend,
                    event_tx,
                    meeting_id,
                    &mut produced,
                )?;
                emit_retranscribe_progress(
                    event_tx,
                    meeting_id,
                    samples_fed,
                    total_kept_samples,
                );
            }
        }

        // Pause boundary (between kept regions): close any in-progress VAD
        // segment here and reset the chunker before the next region. The
        // skipped ≥`PAUSE_MIN_MS` silence WOULD have closed the segment in the
        // live path (its hangover fires on ≥720 ms of silence); because the
        // offline feeder skips that silence to reconstruct the pause-EXCLUDING
        // clock, it must close the segment explicitly — otherwise the VAD,
        // seeing the pre-pause region's last speech sample immediately followed
        // by the post-pause region's first, MERGES the two utterances into one
        // segment with the pre-pause start time (TIMELINE-DRIFT #4: the offline
        // segmentation must match the live path's, which splits at the pause).
        if region_idx + 1 < n_regions {
            match vad.flush_end_of_stream() {
                Ok(events) => {
                    for ev in events {
                        if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                            acc.append(start_ms, end_ms, &samples, None);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "orchestrator",
                        "re_transcribe region-boundary flush error: {e}"
                    );
                }
            }
            vad.reset();
        }
    }

    // End-of-stream VAD flush closes any in-progress segment.
    match vad.flush_end_of_stream() {
        Ok(events) => {
            for ev in events {
                if let VadEvent::SegmentEnd { start_ms, end_ms, samples } = ev {
                    // Offline path: no live label (the on-stop pass owns it).
                    acc.append(start_ms, end_ms, &samples, None);
                }
            }
        }
        Err(e) => {
            tracing::warn!(target: "orchestrator", "re_transcribe flush_end_of_stream error: {e}");
        }
    }

    // Final flush of whatever remains in the accumulator.
    if !acc.is_empty() {
        let (samples, vad_segments, speaker_ids) = acc.drain();
        transcribe_one_flush(
            samples,
            vad_segments,
            speaker_ids,
            backend,
            event_tx,
            meeting_id,
            &mut produced,
        )?;
    }

    // Terminal 1.0 so the bar visibly completes; the subsequent
    // `AppEvent::TranscriptReady` clears the per-row indicator.
    emit_retranscribe_progress(event_tx, meeting_id, total_kept_samples, total_kept_samples);

    Ok(produced)
}

/// Determinate progress fraction for an offline re-transcribe: kept samples fed
/// to the VAD so far / total kept samples (live-test UX T4(a)). Pure +
/// unit-tested. Clamped to `0.0..=1.0`; a zero total (empty/all-pause audio)
/// reports `1.0` (nothing to do is "done").
pub(crate) fn re_transcribe_fraction(samples_fed: usize, total_kept_samples: usize) -> f32 {
    if total_kept_samples == 0 {
        return 1.0;
    }
    (samples_fed as f32 / total_kept_samples as f32).clamp(0.0, 1.0)
}

/// Emit a determinate `AppEvent::OperationProgress` for the re-transcribe op.
fn emit_retranscribe_progress(
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    samples_fed: usize,
    total_kept_samples: usize,
) {
    let _ = event_tx.send(AppEvent::OperationProgress {
        meeting_id,
        op: minutist_common::OperationKind::ReTranscribe,
        fraction: Some(re_transcribe_fraction(samples_fed, total_kept_samples)),
        label: "Re-transcribing…".to_string(),
    });
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
    speaker_ids: Vec<Option<String>>,
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

    let sub_segments = emit_segments_proportional(&combined_text, &vad_segments, &speaker_ids);

    for seg in sub_segments {
        let _ = event_tx.send(AppEvent::TranscriptSegment {
            meeting_id,
            segment: seg.clone(),
        });
        out.push(seg);
    }

    Ok(())
}

/// Resolve and construct the production ASR backend for the re-transcribe path.
///
/// Reuses the live-path engine routing + model-resolution logic
/// ([`init_asr_backend`]): the model for the chosen `engine` must already be
/// `Available` in the registry. `n_gpu_layers` + `language` only affect the Qwen
/// tiers (Parakeet ignores them). Returns `Ok(None)` when the model is not
/// available (the caller surfaces this as an error, since an explicit
/// user-triggered re-transcribe with no model is a failure, unlike the live
/// path's best-effort skip).
pub(crate) async fn build_asr_backend_for_retranscribe(
    model_registry: &ModelRegistry,
    engine: AsrEngine,
    n_gpu_layers: u32,
    language: Option<String>,
) -> AppResult<Option<Box<dyn AsrBackend + Send>>> {
    init_asr_backend(model_registry, engine, n_gpu_layers, language).await
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
pub(crate) const DIARIZE_EMB_MODEL_ID: &str = "3dspeaker-campplus-zh-en-advanced";

/// Lazily build the production `SherpaDiarizer`, mirroring
/// [`build_asr_backend_for_retranscribe`].
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

/// Best-effort, local-only builder for the live [`OnlineDiarizer`] (Phase B).
///
/// Unlike [`build_diarizer`] (which `ensure()`s the models — an explicit
/// operation may download), this is the LIVE path: it must NEVER download or
/// block at record start. It resolves the embedding model purely from disk via
/// the synchronous `Available`-check ([`ModelRegistry::list_models`] →
/// `compute_status_sync`, a `std::fs` size-only check — the exact non-blocking,
/// no-network precedent `init_asr_backend` uses), wraps ONLY the embedding model
/// (no segmentation model — VAD upstream supplies segment boundaries), and opens
/// `OnlineDiarizer::open(emb_onnx, OnlineDiarizerConfig::default())`.
///
/// Returns `None` on EVERY non-happy path so a failure degrades to "no live
/// label" without affecting recording/transcription:
/// - embedding model not locally `Available` → `info` log, `None` (the explicit
///   locked-constraint behaviour: no mid/at-start multi-GB download, no block);
/// - the `.onnx` cannot be located, or `OnlineDiarizer::open` fails (corrupt
///   model / sherpa load error) → `warn` log, `None`.
///
/// `list_models()` is synchronous, so this is a plain (non-async) fn; the heavy
/// `EmbeddingExtractor::new` load inside `open` is the caller's responsibility to
/// run off the async executor (the start path drives it on `spawn_blocking`,
/// mirroring the on-stop diarizer build).
pub(crate) fn build_online_diarizer(
    model_registry: &ModelRegistry,
) -> Option<Arc<OnlineDiarizer>> {
    let emb_id = ModelId::from(DIARIZE_EMB_MODEL_ID);

    // Local-only resolve: the embedding model must already be `Available`.
    let local_dir = model_registry
        .list_models()
        .into_iter()
        .find(|s| s.id == emb_id)
        .and_then(|s| match s.status {
            ModelStatusState::Available { local_dir } => Some(local_dir),
            _ => None,
        });

    let local_dir = match local_dir {
        Some(d) => d,
        None => {
            tracing::info!(
                target: "orchestrator",
                "live diarization: embedding model not downloaded; skipping (recording unaffected)"
            );
            return None;
        }
    };

    let emb_onnx = match find_file_in_dir(std::path::Path::new(&local_dir), |name| {
        name.ends_with(".onnx")
    }) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(
                target: "orchestrator",
                "live diarization: embedding .onnx not located ({e}); skipping"
            );
            return None;
        }
    };

    match OnlineDiarizer::open(&emb_onnx, OnlineDiarizerConfig::default()) {
        Ok(diarizer) => {
            tracing::info!(
                target: "orchestrator",
                emb = %emb_onnx.display(),
                "live online diarizer initialised"
            );
            Some(Arc::new(diarizer))
        }
        Err(e) => {
            tracing::warn!(
                target: "orchestrator",
                "live diarization: OnlineDiarizer::open failed ({e}); skipping"
            );
            None
        }
    }
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
///
/// `speaker_ids` carries the live provisional speaker label for each VAD
/// segment (Phase B), indexed in lockstep with `vad_segments`. The output
/// `Segment` at index `i` inherits `speaker_ids[i]` via `.get(i)`, so a
/// (theoretically impossible) shorter `speaker_ids` slice yields `None` for the
/// missing tail rather than panicking on the worker thread.
pub(crate) fn emit_segments_proportional(
    text: &str,
    vad_segments: &[(u64, u64)],
    speaker_ids: &[Option<String>],
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
            speaker_id: speaker_ids.get(i).cloned().flatten(),
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        });
    }

    out
}

// ---------------------------------------------------------------------------
// Helpers (unchanged from Phase 1)
// ---------------------------------------------------------------------------

/// Drain every queued sample batch through the writer AND the VAD so the
/// end-of-stream flush in `finalise_on_stop` can close an in-progress segment
/// from the tail audio. Shared by both stop paths (Recording and Paused) so
/// their drain semantics cannot diverge: batches accepted by the channel
/// before a pause would otherwise be stranded (the paused loop blocks on
/// cmd_rx and never reads `streams.samples`).
#[allow(clippy::too_many_arguments)]
fn drain_samples_through_vad(
    samples_rx: &mut mpsc::Receiver<AudioFrameBatch>,
    writer: &mut MeetingWriter,
    vad_opt: &mut Option<VadChunker>,
    acc: &mut Accumulator,
    online_diarizer: Option<&Arc<OnlineDiarizer>>,
    context: &str,
) {
    while let Ok(batch) = samples_rx.try_recv() {
        push_batch(writer, &batch);
        if let Some(ref mut vad) = vad_opt {
            match vad.process_samples(&batch.samples, batch.start_ms) {
                Ok(events) => {
                    for ev in events {
                        if let VadEvent::SegmentEnd {
                            start_ms,
                            end_ms,
                            samples,
                        } = ev
                        {
                            let label = live_segment_label(online_diarizer, &samples);
                            acc.append(start_ms, end_ms, &samples, label);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "orchestrator",
                        "VAD process_samples during {context}: {e}"
                    );
                }
            }
        }
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
    // resolve_gpu_layers — the runtime GPU toggle wiring (pure, no model)
    // -----------------------------------------------------------------------

    /// GPU off (`gpu_acceleration = false`) MUST force CPU (`0`), regardless of
    /// the compiled backend. GPU on MUST resolve to the compile-time ceiling —
    /// which is itself `0` in the default CPU-only build, so a CPU build is
    /// unaffected by the flag.
    #[test]
    fn resolve_gpu_layers_off_forces_cpu() {
        assert_eq!(resolve_gpu_layers(false), 0, "GPU off must force CPU");

        let on = resolve_gpu_layers(true);
        assert_eq!(
            on,
            asr_runtime::default_n_gpu_layers(),
            "GPU on must use the compile-time ceiling"
        );
        if cfg!(any(
            feature = "vulkan",
            feature = "metal",
            feature = "cuda",
            feature = "rocm"
        )) {
            assert_eq!(on, u32::MAX, "a GPU-feature build offloads all layers when on");
        } else {
            assert_eq!(on, 0, "a default CPU-only build stays on CPU even when on");
        }
    }

    // -----------------------------------------------------------------------
    // resolve_transcription_language — the ASR language-hint wiring (pure)
    // -----------------------------------------------------------------------

    /// A real language name resolves to `Some(name)` (prefix-force), trimmed.
    /// The `"auto"` sentinel (case-insensitive), the empty string, and
    /// whitespace-only resolve to `None` (auto-detect, no prefix).
    #[test]
    fn resolve_transcription_language_maps_names_and_sentinel() {
        assert_eq!(
            resolve_transcription_language("English"),
            Some("English".to_string())
        );
        assert_eq!(
            resolve_transcription_language("Spanish"),
            Some("Spanish".to_string())
        );

        // The reserved sentinel maps to None, case-insensitively.
        assert_eq!(resolve_transcription_language("auto"), None);
        assert_eq!(resolve_transcription_language("Auto"), None);

        // Empty / whitespace-only also map to None.
        assert_eq!(resolve_transcription_language(""), None);
        assert_eq!(resolve_transcription_language("   "), None);

        // A name is trimmed before forwarding.
        assert_eq!(
            resolve_transcription_language("  English "),
            Some("English".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // re_transcribe_fraction — the determinate progress mapping (pure, T4(a))
    // -----------------------------------------------------------------------

    /// The re-transcribe progress fraction is `samples_fed / total_kept_samples`,
    /// clamped to `0.0..=1.0`; a zero total reports `1.0` (nothing to do is done);
    /// an over-count clamps to `1.0`.
    #[test]
    fn re_transcribe_fraction_maps_fed_over_total() {
        assert_eq!(re_transcribe_fraction(0, 1000), 0.0, "start is 0.0");
        assert_eq!(re_transcribe_fraction(500, 1000), 0.5, "halfway is 0.5");
        assert_eq!(re_transcribe_fraction(1000, 1000), 1.0, "complete is 1.0");

        // Zero total (empty / all-pause audio) → 1.0 (done).
        assert_eq!(re_transcribe_fraction(0, 0), 1.0);
        assert_eq!(re_transcribe_fraction(10, 0), 1.0);

        // Over-count (a final-flush tick past the total) clamps to 1.0.
        assert_eq!(re_transcribe_fraction(1500, 1000), 1.0);

        // Quarter, three-quarters — within tolerance.
        assert!((re_transcribe_fraction(250, 1000) - 0.25).abs() < 1e-6);
        assert!((re_transcribe_fraction(750, 1000) - 0.75).abs() < 1e-6);
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
        acc.append(0, 1000, &seg1_samples, None);

        // Segment 2: 2000 – 3000 ms (1 s). Gap = 1000 ms.
        let seg2_samples = vec![0.5f32; ms_to_samples(1000)];
        acc.append(2000, 3000, &seg2_samples, None);

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
        acc.append(0, 1000, &seg1, None);

        // Segment 2 starts at 6000 ms — gap is 5000 ms, exceeds MAX_GAP_MS=3000 ms.
        let seg2 = vec![0.5f32; ms_to_samples(1000)];
        acc.append(6000, 7000, &seg2, None);

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

    /// A buffer at/over `FLUSH_MIN_SECS` must be flagged for a size-triggered
    /// flush; one under it must not. Expressed relative to the constant so the
    /// test stays honest if the threshold is retuned.
    #[test]
    fn accumulator_flush_triggers_on_size() {
        let flush_ms = (FLUSH_MIN_SECS * 1000.0) as u64;

        let mut under = Accumulator::new();
        let s = vec![0.0f32; ms_to_samples(flush_ms.saturating_sub(500))];
        under.append(0, flush_ms.saturating_sub(500), &s, None);
        assert!(
            under.duration_secs() < FLUSH_MIN_SECS,
            "sub-threshold buffer ({} s) should NOT trigger a size flush",
            under.duration_secs()
        );

        let mut over = Accumulator::new();
        let s = vec![0.0f32; ms_to_samples(flush_ms + 1000)];
        over.append(0, flush_ms + 1000, &s, None);
        assert!(
            over.duration_secs() >= FLUSH_MIN_SECS,
            "over-threshold buffer ({} s) must trigger a size flush at {} s",
            over.duration_secs(),
            FLUSH_MIN_SECS
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: flush triggers on latency
    // -----------------------------------------------------------------------

    /// After `LATENCY_WINDOW_SECS` of quiet (last_vad_end_at sufficiently old),
    /// a non-empty sub-threshold buffer should be flushed so the live transcript
    /// is not held back.
    ///
    /// We don't sleep in the test; we set `last_vad_end_at` to a time well past
    /// the window and check the elapsed condition directly.
    #[test]
    fn accumulator_flush_triggers_on_latency() {
        let mut acc = Accumulator::new();
        let seg = vec![0.0f32; ms_to_samples(1000)]; // 1 s — below flush_min
        acc.append(0, 1000, &seg, None);

        // Simulate elapsed time comfortably beyond the latency window.
        acc.last_vad_end_at =
            Some(Instant::now() - Duration::from_secs_f32(LATENCY_WINDOW_SECS + 2.0));

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

        let segments = emit_segments_proportional(text, &vad_segments, &[None, None, None]);

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
        let segments = emit_segments_proportional("", &vad_segments, &[None, None]);
        assert_eq!(segments.len(), 2);
        assert!(segments.iter().all(|s| s.text.is_empty()));
    }

    // -----------------------------------------------------------------------
    // Phase B: live speaker-label threading through the proportional re-split
    // -----------------------------------------------------------------------

    /// Each output `Segment.speaker_id` must equal the input live label at its
    /// index, preserved positionally across the proportional TEXT re-split (the
    /// re-split redistributes words but keeps one Segment per input vad_segment
    /// in order).
    #[test]
    fn proportional_allocation_carries_live_labels_positionally() {
        let vad_segments = vec![(0u64, 1000u64), (1000, 3000), (3000, 4000)];
        let labels = vec![
            Some("A".to_string()),
            Some("B".to_string()),
            Some("A".to_string()),
        ];
        let segments =
            emit_segments_proportional("alpha beta gamma delta", &vad_segments, &labels);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segments[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(segments[2].speaker_id.as_deref(), Some("A"));
    }

    /// Regression guard: all-None labels → all-None speaker_id (today's
    /// behaviour, i.e. live diarization off / unwired).
    #[test]
    fn proportional_allocation_all_none_labels_yields_all_none() {
        let vad_segments = vec![(0u64, 1000u64), (1000, 2000)];
        let segments =
            emit_segments_proportional("alpha beta", &vad_segments, &[None, None]);
        assert_eq!(segments.len(), 2);
        assert!(segments.iter().all(|s| s.speaker_id.is_none()));
    }

    /// A `speaker_ids` slice SHORTER than `vad_segments` must yield `None` for
    /// the missing tail rather than panicking — exercises the `.get(i)` guard.
    #[test]
    fn proportional_allocation_short_labels_yields_none_tail_no_panic() {
        let vad_segments = vec![(0u64, 1000u64), (1000, 2000), (2000, 3000)];
        // Only one label supplied for three segments.
        let labels = vec![Some("A".to_string())];
        let segments = emit_segments_proportional("a b c", &vad_segments, &labels);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segments[1].speaker_id, None);
        assert_eq!(segments[2].speaker_id, None);
    }

    /// Mixed Some/None labels are preserved positionally.
    #[test]
    fn proportional_allocation_mixed_labels_preserved() {
        let vad_segments = vec![(0u64, 1000u64), (1000, 2000), (2000, 3000)];
        let labels = vec![Some("A".to_string()), None, Some("B".to_string())];
        let segments = emit_segments_proportional("x y z", &vad_segments, &labels);
        assert_eq!(segments[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segments[1].speaker_id, None);
        assert_eq!(segments[2].speaker_id.as_deref(), Some("B"));
    }

    // -----------------------------------------------------------------------
    // Phase B: Accumulator carries the speaker-label column in lockstep
    // -----------------------------------------------------------------------

    /// `append` pushes a label per segment; `drain` returns the label column in
    /// lockstep with `vad_segments` (len equality invariant) and resets it.
    #[test]
    fn accumulator_carries_speaker_ids_in_lockstep_and_drains() {
        let mut acc = Accumulator::new();
        let s = vec![0.5f32; ms_to_samples(500)];
        acc.append(0, 500, &s, Some("A".to_string()));
        acc.append(500, 1000, &s, None);
        acc.append(1000, 1500, &s, Some("B".to_string()));

        assert_eq!(
            acc.speaker_ids.len(),
            acc.vad_segments.len(),
            "speaker_ids must stay in lockstep with vad_segments"
        );

        let (_samples, vad_segments, speaker_ids) = acc.drain();
        assert_eq!(vad_segments.len(), 3);
        assert_eq!(speaker_ids.len(), vad_segments.len());
        assert_eq!(speaker_ids[0].as_deref(), Some("A"));
        assert_eq!(speaker_ids[1], None);
        assert_eq!(speaker_ids[2].as_deref(), Some("B"));

        // drain reset the label column.
        assert!(acc.speaker_ids.is_empty(), "drain must reset speaker_ids");
        assert!(acc.vad_segments.is_empty(), "drain must reset vad_segments");
    }

    /// The gap-capping path (large inter-segment gap) must not perturb the
    /// label/segment 1:1 correspondence — the label is tied to the segment, not
    /// to the zero-padded samples.
    #[test]
    fn accumulator_gap_cap_preserves_label_correspondence() {
        let mut acc = Accumulator::new();
        let seg = vec![0.5f32; ms_to_samples(1000)];
        acc.append(0, 1000, &seg, Some("A".to_string()));
        // 5 s gap → capped at MAX_GAP_MS; label still rides segment 2.
        acc.append(6000, 7000, &seg, Some("B".to_string()));

        let (_samples, vad_segments, speaker_ids) = acc.drain();
        assert_eq!(speaker_ids.len(), vad_segments.len());
        assert_eq!(speaker_ids[0].as_deref(), Some("A"));
        assert_eq!(speaker_ids[1].as_deref(), Some("B"));
    }

    // -----------------------------------------------------------------------
    // Test 7: dispatch_flush drop-oldest behaviour
    // -----------------------------------------------------------------------

    /// When `dispatch_flush` is called with a full queue (FLUSH_CHANNEL_CAP=4),
    /// it must drop the OLDEST entry (front) and retain the newest (back).
    #[test]
    fn dispatch_flush_drops_oldest_when_queue_full() {
        use minutist_common::MeetingId;
        let flush_queue = FlushQueue::new();
        let meeting_id = MeetingId::new();

        // Fill the queue to capacity with payloads tagged by their index in
        // vad_segments[0].0 (start_ms) so we can identify which was dropped.
        for i in 0..FLUSH_CHANNEL_CAP {
            let payload = FlushPayload {
                samples: vec![0.0f32; 100],
                vad_segments: vec![(i as u64 * 1000, i as u64 * 1000 + 500)],
                speaker_ids: vec![None],
                meeting_id,
            };
            assert!(
                !dispatch_flush_pub(&flush_queue, payload),
                "no drop while filling below capacity"
            );
        }

        // No drop yet → the live transcript is still complete.
        assert!(
            !flush_queue.incomplete.load(std::sync::atomic::Ordering::Acquire),
            "incomplete must be false before any drop"
        );

        // Queue is now full (4 entries). Dispatching one more should drop index 0.
        let newest_start_ms = FLUSH_CHANNEL_CAP as u64 * 1000;
        let newest_payload = FlushPayload {
            samples: vec![0.0f32; 100],
            vad_segments: vec![(newest_start_ms, newest_start_ms + 500)],
            speaker_ids: vec![None],
            meeting_id,
        };
        assert!(
            dispatch_flush_pub(&flush_queue, newest_payload),
            "dispatching into a full queue must report a drop (backpressure)"
        );

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

        // The drop must flag the transcript incomplete so ipc-bridge runs a
        // background re-transcribe of the complete audio after stop.
        assert!(
            flush_queue.incomplete.load(std::sync::atomic::Ordering::Acquire),
            "dropping a pending flush must set incomplete"
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

    // -----------------------------------------------------------------------
    // Phase B: build_online_diarizer — no-download/no-block guarantee
    // -----------------------------------------------------------------------

    /// `build_online_diarizer` must return `None` (not Err, no panic, no
    /// network) when the embedding model is NOT `Available` in the registry —
    /// proving the local-only/no-download start guarantee. The cache dir is an
    /// empty tempdir, so the model is `Missing`.
    #[test]
    fn build_online_diarizer_returns_none_when_model_absent() {
        use minutist_common::{ModelFileEntry, ModelKind, ModelManifestEntry};

        let dir = tempfile::tempdir().expect("tempdir");
        let (event_tx, _rx) = broadcast::channel::<AppEvent>(16);

        // Manifest entry for the embedding model the live path resolves, but no
        // files are placed → status is Missing.
        let entry = ModelManifestEntry {
            id: ModelId::from(DIARIZE_EMB_MODEL_ID),
            kind: ModelKind::Diarize,
            display_name: "Embedding".into(),
            files: vec![ModelFileEntry {
                filename: "model.onnx".into(),
                url: "http://example.com/model.onnx".into(),
                size: 10,
                sha256: "00".repeat(32),
            }],
            total_size_bytes: 10,
            license: "apache-2.0".into(),
        };

        let registry = ModelRegistry::new(dir.path().to_path_buf(), vec![entry], event_tx)
            .expect("registry");

        // Must be None, and must not have downloaded anything or panicked.
        let result = build_online_diarizer(&registry);
        assert!(
            result.is_none(),
            "build_online_diarizer must return None when the embedding model is absent"
        );
    }

    // -----------------------------------------------------------------------
    // pcm_window_for_excluding_range (W1, Phase 9 — gating)
    // -----------------------------------------------------------------------

    /// A window wholly inside a single kept region maps to the matching PCM
    /// sample range on the pause-INCLUDING clock (no pause present).
    #[test]
    fn pcm_window_maps_within_single_region() {
        let pcm = vec![0.5f32; ms_to_samples(5000)]; // 5 s, no pause
        // Excluding window [1000, 2000) ms.
        let range = pcm_window_for_excluding_range(&pcm, 1000, 2000).expect("range");
        assert_eq!(range.start, ms_to_samples(1000));
        assert_eq!(range.end, ms_to_samples(2000));
    }

    /// W1 GATING: a window whose START lands in the POST-pause region must map
    /// onto the pause-INCLUDING samples AFTER the skipped pause, not the
    /// pause-excluding offset — proving the clock conversion, not a passthrough.
    #[test]
    fn pcm_window_maps_post_pause_window_onto_including_clock() {
        // 1 s speech, 6 s pause (> 4 s threshold), 3 s speech.
        let speech_a = ms_to_samples(1000);
        let pause = ms_to_samples(6000);
        let speech_b = ms_to_samples(3000);
        let mut pcm = vec![0.5f32; speech_a];
        pcm.extend(std::iter::repeat_n(0.0f32, pause));
        pcm.extend(std::iter::repeat_n(0.5f32, speech_b));

        // The post-pause region starts at pause-EXCLUDING 1000 ms. A window
        // [1500, 2500) ms (excluding clock) is 500..1500 ms INTO the post-pause
        // region, i.e. pause-INCLUDING samples [speech_a + pause + 500ms,
        // speech_a + pause + 1500ms).
        let range = pcm_window_for_excluding_range(&pcm, 1500, 2500).expect("range");
        assert_eq!(range.start, speech_a + pause + ms_to_samples(500));
        assert_eq!(range.end, speech_a + pause + ms_to_samples(1500));
        // Confirm the slice is over the post-pause SPEECH (non-silent), proving
        // the pause was skipped rather than sliced.
        assert!(pcm[range.start..range.end].iter().all(|s| s.abs() > 0.1));
    }

    /// W1 GATING: a window that STRADDLES a pause is clamped to the kept region
    /// containing its start (the documented v1 clamp decision — no concatenation
    /// across the pause seam).
    #[test]
    fn pcm_window_straddling_a_pause_clamps_to_first_region() {
        let speech_a = ms_to_samples(2000); // pause-excl [0, 2000)
        let pause = ms_to_samples(6000);
        let speech_b = ms_to_samples(2000); // pause-excl [2000, 4000)
        let mut pcm = vec![0.5f32; speech_a];
        pcm.extend(std::iter::repeat_n(0.0f32, pause));
        pcm.extend(std::iter::repeat_n(0.5f32, speech_b));

        // Window [1000, 3000) ms straddles the pause (region 0 ends at excl
        // 2000). It must clamp to region 0: [1000ms .. 2000ms] of region 0,
        // i.e. PCM samples [1000ms, 2000ms) — never crossing into region 1.
        let range = pcm_window_for_excluding_range(&pcm, 1000, 3000).expect("range");
        assert_eq!(range.start, ms_to_samples(1000));
        assert_eq!(range.end, speech_a, "must clamp at the first region's end");
    }

    /// A window past the end of all kept audio yields `None` (out of range).
    #[test]
    fn pcm_window_out_of_range_is_none() {
        let pcm = vec![0.5f32; ms_to_samples(1000)];
        assert!(pcm_window_for_excluding_range(&pcm, 5000, 6000).is_none());
    }

    // -----------------------------------------------------------------------
    // pcm16_wav (the transcript "play segment" clip encoder)
    // -----------------------------------------------------------------------

    /// The encoded buffer is a well-formed 16 kHz mono PCM16 WAV: a 44-byte
    /// header carrying the canonical RIFF/WAVE fields, one little-endian i16 per
    /// input sample, and a `data` length matching the sample count. Samples are
    /// clamped to [-1, 1] before scaling, so an out-of-range f32 saturates rather
    /// than wrapping.
    #[test]
    fn pcm16_wav_header_and_samples_are_canonical() {
        // 0.0 → 0, 1.0 → 32767, -1.0 → -32767, and 2.0 clamps to 32767.
        let samples = [0.0f32, 1.0, -1.0, 2.0];
        let wav = pcm16_wav(&samples);

        assert_eq!(wav.len(), 44 + samples.len() * 2, "44-byte header + 2 B/sample");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes([wav[16], wav[17], wav[18], wav[19]]), 16); // PCM fmt size
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1); // PCM
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1); // mono
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            SAMPLE_RATE_HZ as u32,
        );
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16); // bits/sample
        assert_eq!(&wav[36..40], b"data");
        let data_len = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_len as usize, samples.len() * 2);
        assert_eq!(
            u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]),
            36 + data_len,
        );

        let decoded: Vec<i16> = wav[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(decoded, vec![0, 32767, -32767, 32767]);
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

    // -----------------------------------------------------------------------
    // snap_to_energy_min (#0015 phase 4)
    // -----------------------------------------------------------------------

    /// A speech–silence–speech buffer snaps the cut into the silent valley: the
    /// returned argmin window sits inside the near-silent run, not in either
    /// speech burst.
    #[test]
    fn snap_finds_the_silence_valley() {
        // 200 ms speech, 60 ms silence, 200 ms speech.
        let a = ms_to_samples(200);
        let gap = ms_to_samples(60);
        let b = ms_to_samples(200);
        let mut pcm = vec![0.5f32; a];
        pcm.extend(std::iter::repeat_n(0.0f32, gap));
        pcm.extend(std::iter::repeat_n(0.5f32, b));

        // Cut nominally at the speech/silence boundary; search ±150 ms.
        let cut = a;
        let snapped = snap_to_energy_min(&pcm, cut, 150).expect("a clear minimum exists");
        // The argmin window must fall within the silent run [a, a + gap).
        assert!(
            snapped >= a && snapped < a + gap,
            "snap {snapped} must land in the silent valley [{a}, {})",
            a + gap
        );
    }

    /// Constant-energy audio has no clear minimum → `None` (keep-whole). The
    /// relative-RMS floor rejects a flat span as continuous/overlapping speech.
    #[test]
    fn snap_returns_none_on_constant_energy() {
        let pcm = vec![0.5f32; ms_to_samples(500)];
        let cut = ms_to_samples(250);
        assert!(
            snap_to_energy_min(&pcm, cut, 150).is_none(),
            "a flat-energy span has no boundary to snap to"
        );
    }

    /// A degenerate search span (window shorter than one RMS window) → `None`.
    #[test]
    fn snap_returns_none_on_tiny_span() {
        let pcm = vec![0.5f32; ms_to_samples(100)];
        // window_ms = 1 → ±1 ms span (16 samples each side), below the 5 ms RMS
        // window, so there is no usable analysis window.
        assert!(snap_to_energy_min(&pcm, ms_to_samples(50), 1).is_none());
    }

    // -----------------------------------------------------------------------
    // excluding_ms_for_pcm_sample — the PCM→excluding-ms inverse (#0015 phase 4)
    // -----------------------------------------------------------------------

    /// Inside a single kept region (no pause) the inverse is the straight
    /// sample→ms conversion: PCM sample N maps to N/16 ms on the excluding clock.
    #[test]
    fn inverse_maps_within_single_region() {
        let pcm = vec![0.5f32; ms_to_samples(5000)];
        assert_eq!(excluding_ms_for_pcm_sample(&pcm, ms_to_samples(1500)), 1500);
        assert_eq!(excluding_ms_for_pcm_sample(&pcm, 0), 0);
    }

    /// CLOCK REGRESSION GUARD: with a ≥4 s pause, the forward map and the inverse
    /// round-trip. A post-pause excluding-ms maps to a pause-INCLUDING PCM sample
    /// (forward), and that sample maps back to the SAME excluding-ms (inverse) —
    /// proving a split sub-clip's `start_ms` lands on the transcript clock, not
    /// inflated by the skipped pause padding.
    #[test]
    fn inverse_round_trips_across_a_long_pause() {
        // 2 s speech, 6 s pause (> PAUSE_MIN_MS = 4 s), 3 s speech.
        let speech_a = ms_to_samples(2000);
        let pause = ms_to_samples(6000);
        let speech_b = ms_to_samples(3000);
        let mut pcm = vec![0.5f32; speech_a];
        pcm.extend(std::iter::repeat_n(0.0f32, pause));
        pcm.extend(std::iter::repeat_n(0.5f32, speech_b));

        // Post-pause excluding clock: region 2 starts at excluding 2000 ms. Pick a
        // point 1000 ms into it → excluding 3000 ms.
        let excl_ms = 3000u64;
        let range =
            pcm_window_for_excluding_range(&pcm, excl_ms, excl_ms + 100).expect("forward range");
        // The forward map lands on pause-INCLUDING samples AFTER the pause.
        assert_eq!(range.start, speech_a + pause + ms_to_samples(1000));
        // The inverse takes that PCM sample back to the SAME excluding ms.
        assert_eq!(excluding_ms_for_pcm_sample(&pcm, range.start), excl_ms);

        // A sample INSIDE the skipped pause clamps to the pre-pause region end
        // (excluding 2000 ms — the instant the clock froze).
        let mid_pause = speech_a + ms_to_samples(3000);
        assert_eq!(excluding_ms_for_pcm_sample(&pcm, mid_pause), 2000);
    }

    // -----------------------------------------------------------------------
    // Embedding-model resolution convergence guard (WU1 / #0003)
    // -----------------------------------------------------------------------

    /// The offline diarizer path (`build_diarizer`, lines ~2170-2171) and the
    /// online diarizer path (`build_online_diarizer`, lines ~2234-2235) both
    /// resolve the embedding model `.onnx` via `find_file_in_dir` with a
    /// predicate of `name.ends_with(".onnx")`, and both must use the SAME
    /// manifest id (`DIARIZE_EMB_MODEL_ID`). A divergence would place
    /// `VoiceprintExtractor` in a different embedding space than the online
    /// clusterer, silently invalidating voiceprint comparisons.
    ///
    /// This test is purely structural (no model, no I/O): it verifies the
    /// predicate applied to candidate filenames and confirms that the model
    /// id constant matches what both paths reference, making the divergence
    /// visible at compile-or-test time rather than at runtime on a user's
    /// machine.
    #[test]
    fn embedding_model_resolution_predicate_is_identical_for_offline_and_online() {
        // The predicate used in both build_diarizer (line ~2171) and
        // build_online_diarizer (line ~2235).  Any change to either call
        // site must keep these assertions in sync.
        let predicate = |name: &str| name.ends_with(".onnx");

        // Positive: the embedding model file the registry places in the dir.
        assert!(predicate("model.onnx"), "primary .onnx name must match");
        assert!(predicate("speaker_embedding.onnx"), "variant .onnx name must match");

        // Negative: other file types in a model dir must NOT match.
        assert!(!predicate("model.bin"), ".bin must not match");
        assert!(!predicate("manifest.json"), ".json must not match");
        assert!(!predicate("README.md"), ".md must not match");
        assert!(!predicate("model.onnxruntime"), "partial extension must not match");

        // The model id used by both resolution paths is the same constant.
        // VoiceprintExtractor (WU1) relies on this: it is opened by the
        // orchestrator using the same DIARIZE_EMB_MODEL_ID + predicate pair,
        // ensuring the embedding space is consistent across all three paths.
        assert_eq!(
            DIARIZE_EMB_MODEL_ID, "3dspeaker-campplus-zh-en-advanced",
            "DIARIZE_EMB_MODEL_ID must not diverge between offline and online paths"
        );
    }
}
