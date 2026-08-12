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

mod asr_worker;
mod audio_helpers;
mod diarizer_build;
mod drain_loop;
mod pause_clock;
mod retranscribe;
#[cfg(test)]
mod tests;
mod word_allocation;

// Every item any of these submodules marks `pub(crate)` is re-exported here so
// the crate-wide call sites in `lib.rs` / `diarize_pipeline.rs` keep using the
// flat `runner::X` paths — `runner`'s public surface is everything reachable
// through this module, regardless of which file underneath defines it.
pub(crate) use asr_worker::*;
pub(crate) use diarizer_build::*;
pub(crate) use drain_loop::*;
pub(crate) use pause_clock::*;
pub(crate) use retranscribe::*;
pub(crate) use word_allocation::*;

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
