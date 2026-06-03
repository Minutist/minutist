//! VAD-based audio chunker for meeting-app.
//!
//! Wraps the Silero VAD model (via `vad-rs`) with an onset/hangover smoother
//! and a partial-frame accumulator. Accepts raw f32 samples at 16 kHz mono
//! and emits [`VadEvent`]s as speech boundaries are detected.
//!
//! # Design constraints (from `architecture/cross-cutting.md`)
//!
//! - 30 ms / 480-sample frames at 16 kHz (Silero v4 requirement).
//! - `Vad::compute` panics on wrong frame size; `process_samples` accumulates
//!   a partial-frame remainder and only feeds complete 480-sample frames.
//! - Smoothing defaults: threshold 0.5, onset 3 frames, hangover 24 frames,
//!   prefill 5 frames.
//! - No `anyhow` in public signatures; per-crate `Error` with `From` into
//!   `AppError` at the boundary.
//! - No `println!` / `eprintln!`; `tracing` with target `"vad-chunker"`.

use std::collections::VecDeque;
use std::path::Path;

use meeting_app_common::{AppError, AppResult};
use vad_rs::Vad;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Sample rate expected by the Silero v4 model and mandated by the workspace.
const SAMPLE_RATE: usize = 16_000;

/// 30 ms frame size at 16 kHz. Silero v4 requires exactly this.
const FRAME_SAMPLES: usize = 480;

/// Frame duration in milliseconds.
const FRAME_MS: u64 = 30;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Per-crate error enum for `vad-chunker`.
///
/// `From<Error>` into `AppError` converts at the crate boundary so callers
/// only ever see `AppError`. `vad-rs` uses `eyre` internally; we convert to
/// `String` at the boundary to avoid pulling `eyre` into our public surface.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load Silero VAD model from {path}: {context}")]
    ModelLoad { path: String, context: String },

    #[error("VAD inference failed: {context}")]
    Inference { context: String },
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::ModelLoad { path, context } => AppError::ModelLoad {
                model_id: path,
                context,
            },
            Error::Inference { context } => AppError::Inference {
                backend: "silero-vad".to_string(),
                context,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for the VAD smoother.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Per-frame raw probability threshold for classifying a frame as voiced.
    /// Default: 0.5.
    pub threshold: f32,
    /// Number of consecutive voiced frames required before declaring speech
    /// onset. Default: 3 (= 90 ms at 30 ms/frame).
    pub onset_frames: usize,
    /// Number of consecutive silent frames that must elapse after the last
    /// voiced frame before the segment is closed. Default: 24 (= 720 ms).
    pub hangover_frames: usize,
    /// Frames of audio captured before the official segment start (pre-roll).
    /// Default: 5 (= 150 ms).
    pub prefill_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            onset_frames: 3,
            hangover_frames: 24,
            prefill_frames: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Public events
// ---------------------------------------------------------------------------

/// Events emitted by [`VadChunker::process_samples`] and
/// [`VadChunker::flush_end_of_stream`].
#[derive(Debug)]
pub enum VadEvent {
    /// A new speech segment has begun.
    SegmentStart {
        /// Recording-clock offset (ms) of the first sample in this segment
        /// (including pre-roll).
        start_ms: u64,
    },
    /// A speech segment has ended.
    SegmentEnd {
        /// Recording-clock offset (ms) of the first sample in this segment.
        start_ms: u64,
        /// Recording-clock offset (ms) one past the last voiced frame.
        end_ms: u64,
        /// 16 kHz mono f32 samples for the entire segment (including pre-roll
        /// and up to the last voiced frame, hangover silence trimmed).
        samples: Vec<f32>,
    },
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// State of an in-progress speech segment.
struct InProgressSegment {
    start_ms: u64,
    samples: Vec<f32>,
    /// End of the last frame classified as voiced (used to trim hangover
    /// silence when closing the segment).
    last_voice_end_ms: u64,
}

/// The smoother state machine — tracks onset and hangover counters.
struct VadSmoother {
    threshold: f32,
    onset_frames: usize,
    hangover_frames: usize,
    /// Consecutive voiced frames seen while outside speech.
    onset_counter: usize,
    /// Remaining hangover frames before the segment closes.
    hangover_counter: usize,
    in_speech: bool,
}

impl VadSmoother {
    fn new(threshold: f32, onset_frames: usize, hangover_frames: usize) -> Self {
        Self {
            threshold,
            onset_frames,
            hangover_frames,
            onset_counter: 0,
            hangover_counter: 0,
            in_speech: false,
        }
    }

    /// Feed one frame's raw VAD probability. Returns the smoother decision.
    fn update(&mut self, prob: f32) -> SmootherDecision {
        let is_voice = prob > self.threshold;

        match (self.in_speech, is_voice) {
            (false, true) => {
                self.onset_counter += 1;
                if self.onset_counter >= self.onset_frames {
                    self.in_speech = true;
                    self.hangover_counter = self.hangover_frames;
                    self.onset_counter = 0;
                    SmootherDecision::SpeechOnset
                } else {
                    SmootherDecision::BuildingOnset
                }
            }
            (false, false) => {
                self.onset_counter = 0;
                SmootherDecision::Silence
            }
            (true, true) => {
                self.hangover_counter = self.hangover_frames;
                SmootherDecision::Speech
            }
            (true, false) => {
                if self.hangover_counter > 0 {
                    self.hangover_counter -= 1;
                }
                if self.hangover_counter == 0 {
                    self.in_speech = false;
                    self.onset_counter = 0;
                    SmootherDecision::SpeechEnd
                } else {
                    SmootherDecision::Hangover
                }
            }
        }
    }
}

/// Internal decision from the smoother for one frame.
#[derive(Debug, PartialEq, Eq)]
enum SmootherDecision {
    /// Silence; not accumulating onset.
    Silence,
    /// Consecutive voiced frames accumulating toward onset.
    BuildingOnset,
    /// Voiced frames threshold crossed: segment just started.
    SpeechOnset,
    /// Inside speech; continuing.
    Speech,
    /// Inside speech but this frame is silent; hangover in progress.
    Hangover,
    /// Hangover exhausted; segment just ended.
    SpeechEnd,
}

// ---------------------------------------------------------------------------
// VadChunker
// ---------------------------------------------------------------------------

/// Stateful VAD chunker.
///
/// Feed 16 kHz mono f32 samples via [`process_samples`](VadChunker::process_samples).
/// The chunker accumulates samples into 480-sample frames, runs the Silero v4
/// VAD, applies onset/hangover smoothing, and emits [`VadEvent`]s.
///
/// Call [`flush_end_of_stream`](VadChunker::flush_end_of_stream) at end-of-stream
/// to close any in-progress segment.
pub struct VadChunker {
    vad: Vad,
    smoother: VadSmoother,
    /// Partial-frame remainder: samples not yet forming a complete 480-sample frame.
    sample_buffer: Vec<f32>,
    /// Recording-clock offset (ms) of the next complete frame's first sample.
    next_frame_start_ms: u64,
    /// Pre-roll ring buffer: holds (frame_start_ms, samples) for the last
    /// `prefill_frames` complete frames.
    prefill: VecDeque<(u64, Vec<f32>)>,
    prefill_frames: usize,
    /// The segment currently being recorded, if any.
    current_segment: Option<InProgressSegment>,
}

impl VadChunker {
    /// Open the Silero VAD model at `model_path` and initialise the chunker.
    ///
    /// # Errors
    ///
    /// Returns `AppError::ModelLoad` if the ONNX model cannot be opened.
    pub fn open(model_path: &Path, config: VadConfig) -> AppResult<Self> {
        let vad = Vad::new(model_path, SAMPLE_RATE).map_err(|e| Error::ModelLoad {
            path: model_path.display().to_string(),
            context: e.to_string(),
        })?;

        tracing::info!(
            target: "vad-chunker",
            path = %model_path.display(),
            threshold = config.threshold,
            onset_frames = config.onset_frames,
            hangover_frames = config.hangover_frames,
            prefill_frames = config.prefill_frames,
            "Silero VAD model loaded"
        );

        Ok(Self {
            vad,
            smoother: VadSmoother::new(
                config.threshold,
                config.onset_frames,
                config.hangover_frames,
            ),
            sample_buffer: Vec::new(),
            next_frame_start_ms: 0,
            prefill: VecDeque::with_capacity(config.prefill_frames + 1),
            prefill_frames: config.prefill_frames,
            current_segment: None,
        })
    }

    /// Feed a slice of 16 kHz mono f32 samples into the chunker.
    ///
    /// `start_ms` is the recording-clock offset (ms) of the first sample in
    /// `samples`. Samples need not be frame-aligned; any remainder is held in
    /// an internal buffer until the next call completes a full 480-sample frame.
    ///
    /// Returns a (possibly empty) list of [`VadEvent`]s emitted during this
    /// call.
    ///
    /// # Errors
    ///
    /// Returns `AppError::Inference` if the VAD model itself fails on a frame.
    pub fn process_samples(
        &mut self,
        samples: &[f32],
        start_ms: u64,
    ) -> AppResult<Vec<VadEvent>> {
        // Re-anchor the frame clock to the caller-supplied `start_ms` on EVERY
        // call (TIMELINE-DRIFT #3).
        //
        // `start_ms` is the recording-clock offset of the first sample in
        // `samples`. The chunker may hold a partial-frame remainder
        // (`sample_buffer`) from the previous call; those `r` buffered samples
        // are the contiguous tail that immediately precedes `samples`, so the
        // first sample of the buffer sits `r` samples (in ms) before `start_ms`.
        // The position the buffer's first sample SHOULD sit at, if this feed is
        // contiguous with the last, is `next_frame_start_ms` (which the drain
        // loop advanced as it consumed frames). Comparing the two detects a gap.
        let buffered_ms = (self.sample_buffer.len() as u64 * 1000) / SAMPLE_RATE as u64;
        let buffer_first_sample_ms = start_ms.saturating_sub(buffered_ms);

        // A discontinuity (dropped capture frames) shows up as the new
        // `start_ms` not lining up with where the internal clock expected the
        // buffer's leading sample. When that happens, the partial-frame
        // remainder and the pre-roll ring buffer hold STALE pre-gap audio whose
        // timestamps no longer relate to the new region; carrying them forward
        // would anchor the next segment's start to the pre-gap timeline (the
        // exact desync this bug is about). Discard them so the new region
        // starts cleanly on the caller's timeline. A tolerance of one frame
        // absorbs ms-rounding of the partial remainder on a contiguous feed.
        let gap = buffer_first_sample_ms.abs_diff(self.next_frame_start_ms) > FRAME_MS;
        if gap {
            self.sample_buffer.clear();
            self.prefill.clear();
            // The new region begins exactly at `start_ms` with an empty buffer.
            self.next_frame_start_ms = start_ms;
        } else {
            // Contiguous feed: re-derive the anchor from the caller's timeline
            // so any sub-frame drift is corrected without perturbing framing.
            self.next_frame_start_ms = buffer_first_sample_ms;
        }

        // Append new samples to the partial-frame buffer.
        self.sample_buffer.extend_from_slice(samples);

        let mut events = Vec::new();

        // Drain complete 480-sample frames from the buffer.
        while self.sample_buffer.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = self.sample_buffer.drain(..FRAME_SAMPLES).collect();
            let frame_start_ms = self.next_frame_start_ms;
            self.next_frame_start_ms += FRAME_MS;

            let frame_events = self.process_frame(&frame, frame_start_ms)?;
            events.extend(frame_events);
        }

        Ok(events)
    }

    /// Signal end-of-stream. If a speech segment is currently in progress, it
    /// is closed and a [`VadEvent::SegmentEnd`] is returned.
    ///
    /// Any partial frame remaining in the internal buffer is discarded (it is
    /// less than 30 ms of audio and insufficient to change the VAD decision).
    ///
    /// # Errors
    ///
    /// Returns `AppError::Inference` if VAD processing of queued frames fails.
    pub fn flush_end_of_stream(&mut self) -> AppResult<Vec<VadEvent>> {
        // Zero-pad and process any partial frame so the VAD sees the final
        // audio, then close the segment.
        let mut events = Vec::new();

        if !self.sample_buffer.is_empty() {
            let mut padded = std::mem::take(&mut self.sample_buffer);
            let frame_start_ms = self.next_frame_start_ms;
            padded.resize(FRAME_SAMPLES, 0.0);
            self.next_frame_start_ms += FRAME_MS;
            let frame_events = self.process_frame(&padded, frame_start_ms)?;
            events.extend(frame_events);
        }

        // Force-close any in-progress segment.
        if let Some(seg) = self.current_segment.take() {
            tracing::debug!(
                target: "vad-chunker",
                start_ms = seg.start_ms,
                end_ms = seg.last_voice_end_ms,
                samples = seg.samples.len(),
                "flush_end_of_stream: closing in-progress segment"
            );
            events.push(VadEvent::SegmentEnd {
                start_ms: seg.start_ms,
                end_ms: seg.last_voice_end_ms,
                samples: seg.samples,
            });
        }

        Ok(events)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Process one complete 480-sample frame through the VAD and smoother.
    fn process_frame(
        &mut self,
        frame: &[f32],
        frame_start_ms: u64,
    ) -> AppResult<Vec<VadEvent>> {
        debug_assert_eq!(frame.len(), FRAME_SAMPLES);

        let frame_end_ms = frame_start_ms + FRAME_MS;

        // Maintain pre-roll ring buffer before the VAD call so it's available
        // immediately when onset fires.
        self.prefill.push_back((frame_start_ms, frame.to_vec()));
        while self.prefill.len() > self.prefill_frames + 1 {
            self.prefill.pop_front();
        }

        // Run Silero VAD.
        let prob = self
            .vad
            .compute(frame)
            .map_err(|e| AppError::from(Error::Inference { context: e.to_string() }))?
            .prob;

        tracing::trace!(
            target: "vad-chunker",
            frame_start_ms,
            frame_end_ms,
            prob,
            "VAD frame"
        );

        let decision = self.smoother.update(prob);
        let mut events = Vec::new();

        match decision {
            SmootherDecision::Silence | SmootherDecision::BuildingOnset => {
                // Not in speech; nothing to do with the segment buffer.
            }

            SmootherDecision::SpeechOnset => {
                // Seed the segment with pre-roll frames.
                let mut segment_samples: Vec<f32> = Vec::new();
                let mut earliest_ms = frame_start_ms;
                for (ms, samples) in &self.prefill {
                    segment_samples.extend_from_slice(samples);
                    earliest_ms = earliest_ms.min(*ms);
                }

                tracing::debug!(
                    target: "vad-chunker",
                    start_ms = earliest_ms,
                    prefill_frames = self.prefill.len(),
                    "speech onset"
                );

                self.current_segment = Some(InProgressSegment {
                    start_ms: earliest_ms,
                    samples: segment_samples,
                    last_voice_end_ms: frame_end_ms,
                });

                events.push(VadEvent::SegmentStart {
                    start_ms: earliest_ms,
                });
            }

            SmootherDecision::Speech => {
                // Continue the current segment.
                if let Some(ref mut seg) = self.current_segment {
                    seg.samples.extend_from_slice(frame);
                    seg.last_voice_end_ms = frame_end_ms;
                }
            }

            SmootherDecision::Hangover => {
                // Keep appending during hangover; don't advance last_voice_end_ms.
                if let Some(ref mut seg) = self.current_segment {
                    seg.samples.extend_from_slice(frame);
                }
            }

            SmootherDecision::SpeechEnd => {
                // Hangover exhausted. Close the segment.
                if let Some(mut seg) = self.current_segment.take() {
                    // Trim the trailing silence appended during hangover.
                    // last_voice_end_ms is the end of the last voiced frame.
                    let trim_ms = frame_end_ms.saturating_sub(seg.last_voice_end_ms);
                    let trim_samples = (trim_ms as usize * SAMPLE_RATE) / 1000;
                    let keep = seg.samples.len().saturating_sub(trim_samples);
                    seg.samples.truncate(keep);

                    tracing::debug!(
                        target: "vad-chunker",
                        start_ms = seg.start_ms,
                        end_ms = seg.last_voice_end_ms,
                        samples = seg.samples.len(),
                        "speech segment ended"
                    );

                    events.push(VadEvent::SegmentEnd {
                        start_ms: seg.start_ms,
                        end_ms: seg.last_voice_end_ms,
                        samples: seg.samples,
                    });
                }
            }
        }

        Ok(events)
    }
}

// ---------------------------------------------------------------------------
// Default model-path resolution
// ---------------------------------------------------------------------------

/// Resolve the bundled Silero VAD model path.
///
/// Priority:
/// 1. `MEETING_APP_SILERO_PATH` environment variable (set at build time or runtime).
/// 2. `{CARGO_MANIFEST_DIR}/../../resources/silero/silero_vad_v4.onnx` so that
///    `cargo test -p vad-chunker` finds the file from the crate's directory.
pub fn default_model_path() -> std::path::PathBuf {
    if let Some(p) = option_env!("MEETING_APP_SILERO_PATH") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../resources/silero/silero_vad_v4.onnx"
    ))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Path to the librispeech_0 fixture (16 kHz mono i16 WAV, ~5.8 s, speech only).
    // Resolved relative to workspace root so that both `cargo test` from any
    // working directory and CI find the same path.
    const FIXTURE_WAV: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/librispeech_0.wav"
    );

    /// Load the fixture WAV as f32 samples at 16 kHz mono.
    fn load_fixture_wav() -> Vec<f32> {
        let mut reader = hound::WavReader::open(FIXTURE_WAV)
            .expect("failed to open test fixture — run from workspace root");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1, "fixture must be mono");
        assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
        reader
            .samples::<i16>()
            .map(|s| s.expect("WAV read error") as f32 / i16::MAX as f32)
            .collect()
    }

    /// Build a VadChunker with the bundled model for testing.
    fn test_chunker() -> VadChunker {
        let path = default_model_path();
        VadChunker::open(&path, VadConfig::default()).expect("failed to load Silero VAD model")
    }

    /// Generate silence samples.
    fn silence(duration_ms: u64) -> Vec<f32> {
        vec![0.0f32; (SAMPLE_RATE as u64 * duration_ms / 1000) as usize]
    }

    /// Collect all events from a slice of samples fed in one call.
    fn process_all(chunker: &mut VadChunker, samples: &[f32], start_ms: u64) -> Vec<VadEvent> {
        chunker
            .process_samples(samples, start_ms)
            .expect("process_samples failed")
    }

    // -----------------------------------------------------------------------
    // Test 1: Real speech + silence + speech produces distinct segments
    // -----------------------------------------------------------------------
    // Compose: librispeech_0 (5.8 s speech) → 700 ms silence → librispeech_0 again.
    // Expect: at least one SegmentEnd for the first speech block and at least
    // one SegmentStart for the second.
    #[test]
    fn test_speech_silence_speech() {
        let speech = load_fixture_wav();
        let silence_pad = silence(700);

        let mut audio = Vec::new();
        audio.extend_from_slice(&speech);
        audio.extend_from_slice(&silence_pad);
        audio.extend_from_slice(&speech);

        let mut chunker = test_chunker();
        let mut all_events = process_all(&mut chunker, &audio, 0);
        all_events.extend(chunker.flush_end_of_stream().expect("flush failed"));

        let ends: usize = all_events
            .iter()
            .filter(|e| matches!(e, VadEvent::SegmentEnd { .. }))
            .count();
        let starts: usize = all_events
            .iter()
            .filter(|e| matches!(e, VadEvent::SegmentStart { .. }))
            .count();

        assert!(
            ends >= 1,
            "expected at least one SegmentEnd; got {ends}"
        );
        assert!(
            starts >= 2,
            "expected at least two SegmentStarts (one per speech region); got {starts}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Sub-onset speech bursts do NOT produce SegmentStart
    // -----------------------------------------------------------------------
    // Verifies that the onset counter blocks premature segment starts.
    //
    // Strategy: configure onset_frames=50 (requiring ~1.5 s of continuous
    // voiced signal) and feed only a short voiced burst (~3 s of speech).
    // Because Silero's LSTM state carries the voiced probability forward for
    // a few frames after speech ends, we use a large onset threshold that
    // clearly exceeds what a brief burst can achieve without sustained input.
    //
    // We don't test onset_frames=3 here because Silero's LSTM keep the
    // probability above 0.5 for 2-3 frames after speech ends, meaning
    // even a 2-frame burst accumulates ~4 consecutive above-threshold
    // frames — enough to fire the default onset of 3. The architectural
    // property being tested is that the onset_frames counter is respected,
    // not that Silero snaps to silence immediately.
    #[test]
    fn test_sub_onset_no_segment_start() {
        // Large onset threshold: requires 50 consecutive voiced frames (~1.5 s).
        let config = VadConfig {
            onset_frames: 50,
            ..VadConfig::default()
        };

        let speech = load_fixture_wav();
        // Use the first 30 voiced frames (900 ms) from the fixture —
        // clearly below the 50-frame onset threshold.
        const FIRST_VOICED_FRAME: usize = 19;
        let burst_start = FIRST_VOICED_FRAME * FRAME_SAMPLES;
        let burst_end = burst_start + FRAME_SAMPLES * 30; // 30 frames < onset_frames=50
        let burst = &speech[burst_start..burst_end];

        let path = default_model_path();
        let mut chunker =
            VadChunker::open(&path, config).expect("failed to load Silero VAD model");

        let mut audio = silence(300);
        audio.extend_from_slice(burst);
        audio.extend_from_slice(&silence(1000));

        let mut all_events = process_all(&mut chunker, &audio, 0);
        all_events.extend(chunker.flush_end_of_stream().expect("flush failed"));

        let starts: usize = all_events
            .iter()
            .filter(|e| matches!(e, VadEvent::SegmentStart { .. }))
            .count();
        assert!(
            starts == 0,
            "30-frame burst (< onset_frames=50) must not produce SegmentStart; got {starts}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Sub-hangover silence inside speech does NOT close the segment
    // -----------------------------------------------------------------------
    // voiced speech → 5 frames of silence (150 ms, < hangover_frames=24) →
    // voiced speech again → flush.
    // We expect at most 1 SegmentEnd total (from the flush). A 5-frame
    // silence gap must NOT close the segment mid-stream.
    //
    // We use frames starting from FIRST_VOICED_FRAME so the VAD sees
    // actual speech from the beginning and onset fires quickly.
    #[test]
    fn test_sub_hangover_silence_does_not_close_segment() {
        let hangover_frames = VadConfig::default().hangover_frames; // 24
        let short_silence_frames = 5usize;
        assert!(short_silence_frames < hangover_frames);

        let speech = load_fixture_wav();
        // Start from where speech is voiced (frame 19 in librispeech_0).
        const FIRST_VOICED_FRAME: usize = 19;
        let speech_offset = FIRST_VOICED_FRAME * FRAME_SAMPLES;
        // 15 frames of voiced audio (onset fires at frame 3; remaining 12 solidly in speech)
        let speech_slice = &speech[speech_offset..speech_offset + FRAME_SAMPLES * 15];

        let mut audio = speech_slice.to_vec();
        audio.extend_from_slice(&silence((short_silence_frames as u64) * FRAME_MS));
        audio.extend_from_slice(speech_slice);

        let mut chunker = test_chunker();
        let mut all_events = process_all(&mut chunker, &audio, 0);
        all_events.extend(chunker.flush_end_of_stream().expect("flush failed"));

        let ends: usize = all_events
            .iter()
            .filter(|e| matches!(e, VadEvent::SegmentEnd { .. }))
            .count();
        // Hangover=24; only a 5-frame gap — must stay inside one segment.
        // At most 1 SegmentEnd total (the flush close).
        assert!(
            ends <= 1,
            "5-frame silence gap (< hangover_frames={hangover_frames}) must not split the segment; got {ends} SegmentEnds"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: flush_end_of_stream emits SegmentEnd for in-progress segment
    // -----------------------------------------------------------------------
    // Feed enough audio to reach onset (which starts at frame 19 in the
    // fixture), then immediately flush. flush_end_of_stream must emit at
    // least one SegmentEnd because a segment is still open.
    #[test]
    fn test_flush_end_of_stream_closes_segment() {
        let speech = load_fixture_wav();

        // librispeech_0 has ~19 frames of leading silence before speech onset.
        // Feed through frame 24 (well into voiced territory) so we are
        // guaranteed to have crossed onset. onset_frames=3; voiced starts
        // at frame 19; onset fires at frame 21; frame 24 is solidly in speech.
        const FRAMES_TO_FEED: usize = 25; // cover frames 0-24 inclusive
        let audio = &speech[..FRAME_SAMPLES * FRAMES_TO_FEED];

        let mut chunker = test_chunker();
        let mid_events = process_all(&mut chunker, audio, 0);
        let flush_events = chunker.flush_end_of_stream().expect("flush failed");

        let all_events: Vec<_> = mid_events.into_iter().chain(flush_events).collect();
        let ends: usize = all_events
            .iter()
            .filter(|e| matches!(e, VadEvent::SegmentEnd { .. }))
            .count();

        assert!(
            ends >= 1,
            "flush_end_of_stream must emit SegmentEnd for in-progress segment; got {ends}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Partial-frame buffering produces identical event stream
    // -----------------------------------------------------------------------
    // Feeding all samples in one call vs feeding 240 samples at a time must
    // produce the same number and type of events.
    #[test]
    fn test_partial_frame_buffering_identical_to_full() {
        let audio = load_fixture_wav();

        // Run A: one call
        let mut chunker_a = test_chunker();
        let mut events_a = chunker_a
            .process_samples(&audio, 0)
            .expect("process_samples A failed");
        events_a.extend(chunker_a.flush_end_of_stream().expect("flush A failed"));

        // Run B: split into 240-sample half-frames
        let mut chunker_b = test_chunker();
        let mut events_b: Vec<VadEvent> = Vec::new();
        for chunk in audio.chunks(240) {
            // The start_ms for each chunk doesn't matter for the structural test;
            // what matters is that the partial-frame accumulator reassembles the
            // same complete 480-sample frames.
            let mut evts = chunker_b
                .process_samples(chunk, 0)
                .expect("process_samples B failed");
            events_b.append(&mut evts);
        }
        events_b.extend(chunker_b.flush_end_of_stream().expect("flush B failed"));

        let count_starts = |evts: &[VadEvent]| -> usize {
            evts.iter()
                .filter(|e| matches!(e, VadEvent::SegmentStart { .. }))
                .count()
        };
        let count_ends = |evts: &[VadEvent]| -> usize {
            evts.iter()
                .filter(|e| matches!(e, VadEvent::SegmentEnd { .. }))
                .count()
        };

        assert_eq!(
            count_starts(&events_a),
            count_starts(&events_b),
            "SegmentStart count must match: one-call vs 240-sample chunks"
        );
        assert_eq!(
            count_ends(&events_a),
            count_ends(&events_b),
            "SegmentEnd count must match: one-call vs 240-sample chunks"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: caller start_ms is honoured on EVERY call (TIMELINE-DRIFT #3)
    // -----------------------------------------------------------------------
    // A dropped-frame gap in the fed samples must re-align the VAD timeline to
    // the caller-supplied `start_ms`, not silently continue from the internal
    // monotonic clock. We feed leading silence, then speech, but on the speech
    // call we jump `start_ms` forward by a large gap (simulating capture frames
    // the upstream dropped, exactly what the live runner can do). The emitted
    // SegmentStart must reflect the post-gap timeline (near the jumped value),
    // NOT the pre-gap monotonic position.
    #[test]
    fn test_start_ms_honoured_after_gap() {
        let speech = load_fixture_wav();
        // Real voiced audio from the fixture, frame-aligned.
        const FIRST_VOICED_FRAME: usize = 19;
        let off = FIRST_VOICED_FRAME * FRAME_SAMPLES;
        // Feed exactly whole frames so there is no partial-frame remainder
        // confounding the start_ms math.
        let lead = &speech[..FRAME_SAMPLES * 10]; // 10 frames = 300 ms, mostly silence
        let voiced = &speech[off..off + FRAME_SAMPLES * 20]; // 20 voiced frames

        let mut chunker = test_chunker();

        // First call: lead silence starting at t=0.
        let _ = process_all(&mut chunker, lead, 0);

        // Simulate a large gap: the next batch's first sample is at 100_000 ms
        // (the capture clock jumped 100 s ahead because frames were dropped).
        const GAP_START_MS: u64 = 100_000;
        let mut events = process_all(&mut chunker, voiced, GAP_START_MS);
        events.extend(chunker.flush_end_of_stream().expect("flush"));

        let first_start = events.iter().find_map(|e| match e {
            VadEvent::SegmentStart { start_ms } => Some(*start_ms),
            _ => None,
        });
        let first_start = first_start.expect("a SegmentStart must be emitted for the voiced run");

        // The start must be on the post-gap timeline: within the voiced region
        // [GAP_START_MS, GAP_START_MS + 20 frames). It must NOT be near the
        // pre-gap position (~300 ms) — that would prove start_ms was ignored.
        let voiced_end_ms = GAP_START_MS + 20 * FRAME_MS;
        // Pre-roll can place the start a few frames before GAP_START_MS.
        let earliest_allowed = GAP_START_MS.saturating_sub(FRAME_MS * 10);
        assert!(
            first_start >= earliest_allowed && first_start <= voiced_end_ms,
            "SegmentStart {first_start} ms is not on the post-gap timeline \
             [{earliest_allowed}, {voiced_end_ms}] — caller start_ms was ignored after the gap"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: contiguous frame-aligned feed advances the clock correctly
    // -----------------------------------------------------------------------
    // When fed contiguous, frame-aligned batches with advancing start_ms, the
    // emitted timestamps must match a single-call feed of the same audio —
    // i.e. re-anchoring on every call does not perturb the no-gap case.
    #[test]
    fn test_contiguous_advancing_start_ms_matches_single_call() {
        let speech = load_fixture_wav();
        // Trim to a whole number of frames so batch boundaries are frame-aligned.
        let frames = speech.len() / FRAME_SAMPLES;
        let audio = &speech[..frames * FRAME_SAMPLES];

        // Run A: one call at start_ms = 0.
        let mut a = test_chunker();
        let mut events_a = process_all(&mut a, audio, 0);
        events_a.extend(a.flush_end_of_stream().expect("flush A"));

        // Run B: 5-frame (150 ms) batches with advancing, contiguous start_ms.
        let mut b = test_chunker();
        let mut events_b: Vec<VadEvent> = Vec::new();
        const BATCH_FRAMES: usize = 5;
        let batch_samples = BATCH_FRAMES * FRAME_SAMPLES;
        let mut start_ms = 0u64;
        for chunk in audio.chunks(batch_samples) {
            let mut evts = process_all(&mut b, chunk, start_ms);
            events_b.append(&mut evts);
            start_ms += (chunk.len() as u64 * 1000) / SAMPLE_RATE as u64;
        }
        events_b.extend(b.flush_end_of_stream().expect("flush B"));

        let starts_a: Vec<u64> = events_a.iter().filter_map(seg_start).collect();
        let starts_b: Vec<u64> = events_b.iter().filter_map(seg_start).collect();
        assert_eq!(
            starts_a, starts_b,
            "contiguous advancing-start_ms feed must yield identical SegmentStart timestamps \
             to a single-call feed"
        );
    }

    fn seg_start(e: &VadEvent) -> Option<u64> {
        match e {
            VadEvent::SegmentStart { start_ms } => Some(*start_ms),
            _ => None,
        }
    }
}
