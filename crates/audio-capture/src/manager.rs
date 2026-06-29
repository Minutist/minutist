//! `AudioCaptureManager` — cpal stream init, pause/resume/stop, and the
//! bridge from the cpal callback thread into async Tokio channels.
//!
//! Threading model (per `architecture/cross-cutting.md`):
//!   • The cpal callback runs on cpal's own thread.  It pushes raw samples
//!     into a bounded [`DropOldestChannel`] (capacity 8).  The callback never
//!     blocks: it uses `try_lock` and, on overflow, evicts the OLDEST queued
//!     frame before enqueuing the newest.  A `tracing::warn!` fires on drop.
//!   • A `tokio::task::spawn_blocking` task drains that channel and forwards
//!     resampled + metered output into bounded `tokio::sync::mpsc` channels
//!     whose capacities are passed in by the caller.

use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::time::Duration;

use cpal::{
    traits::{DeviceTrait, StreamTrait},
    Sample, SizedSample,
};
use minutist_common::{AppResult, AudioDevice, AudioMeterFrame};
use tokio::sync::mpsc;

use crate::device;
use crate::error::Error;
use crate::meter::LevelMeter;
use crate::resample::StreamResampler;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A batch of resampled 16 kHz mono f32 samples with recording-clock offsets.
#[derive(Debug)]
pub struct AudioFrameBatch {
    /// 16 kHz mono f32 samples.
    pub samples: Vec<f32>,
    /// Recording-clock offset of the first sample in milliseconds.
    pub start_ms: u64,
    /// Recording-clock offset one past the last sample in milliseconds.
    pub end_ms: u64,
}

/// Pair of async receivers vended to the orchestrator by `start`.
pub struct AudioStreams {
    /// Resampled 16 kHz mono sample batches.
    pub samples: mpsc::Receiver<AudioFrameBatch>,
    /// Meter frames emitted at ~30 Hz.
    pub meter: mpsc::Receiver<AudioMeterFrame>,
}

// ---------------------------------------------------------------------------
// Internal: raw frame from cpal callback → forwarder
// ---------------------------------------------------------------------------

pub(crate) struct RawFrame {
    pub(crate) samples: Vec<f32>,
}

/// Shared state behind [`DropOldestChannel`].
struct ChannelInner {
    /// Bounded FIFO of pending raw frames. Oldest at the front.
    queue: VecDeque<RawFrame>,
    /// Set once the consumer wants no more frames (stop). The producer then
    /// drops silently and the consumer drains the tail and exits.
    closed: bool,
}

/// Capacity of a per-source raw-frame ring (cpal callback → resample forwarder),
/// counted in cpal callback buffers. Sized to ride a multi-second stall of the
/// forwarder/consumer — chiefly the model-load burst at record start, which
/// otherwise CPU-starves the forwarder, overflows a small ring, and drop-floods
/// (and truncates) the recording. At ~10 ms cpal buffers this is ~10 s of
/// headroom; the forwarder drains the backlog once models finish loading, so no
/// audio is lost — only briefly delayed. (Was 8 ≈ 80 ms, far too small.)
pub(crate) const RAW_RING_CAPACITY: usize = 1024;

/// A bounded SPSC channel that, on overflow, evicts the OLDEST queued frame so
/// the newest data is always retained — the true drop-oldest contract.
///
/// Real-time safety: the cpal callback (the producer) only ever takes the lock
/// via `try_lock` and holds it for the duration of a single push (one
/// `pop_front` + one `push_back`). It never blocks waiting for the consumer.
/// The consumer holds the lock only long enough to pop one frame; the
/// expensive resampling happens after the lock is released, so the producer
/// is essentially never contended. This replaces the previous design, where
/// the consumer held a `Mutex` for the entire session and the callback's
/// drop-oldest path deadlocked on that held lock (so it never dropped the
/// oldest and risked blocking the RT thread).
pub(crate) struct DropOldestChannel {
    inner: Mutex<ChannelInner>,
    /// Signals the consumer that a frame is available or the channel closed.
    not_empty: Condvar,
    capacity: usize,
}

impl DropOldestChannel {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0, "channel capacity must be > 0");
        DropOldestChannel {
            inner: Mutex::new(ChannelInner {
                queue: VecDeque::with_capacity(capacity),
                closed: false,
            }),
            not_empty: Condvar::new(),
            capacity,
        }
    }

    /// Push a frame from the audio callback, evicting the oldest on overflow.
    ///
    /// Never blocks: if the lock is momentarily contended (only possible while
    /// the consumer pops a single frame) the frame is dropped rather than
    /// stalling the RT thread.
    pub(crate) fn push(&self, frame: RawFrame) {
        // `try_lock` keeps the RT thread from ever blocking on the consumer.
        let mut inner = match self.inner.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                tracing::warn!(
                    target = "audio-capture",
                    "cpal→forwarder channel busy; dropping frame to avoid blocking RT callback"
                );
                return;
            }
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };

        if inner.closed {
            return;
        }

        if inner.queue.len() >= self.capacity {
            // Evict the OLDEST frame, then enqueue the newest.
            inner.queue.pop_front();
            tracing::warn!(
                target = "audio-capture",
                "cpal→forwarder channel full (capacity {}); dropped oldest frame",
                self.capacity
            );
        }
        inner.queue.push_back(frame);
        drop(inner);
        self.not_empty.notify_one();
    }

    /// Pop the oldest frame, waiting up to `timeout` for one to arrive.
    ///
    /// Returns `None` on timeout (caller re-checks its stop flag) or once the
    /// channel is closed and drained.
    pub(crate) fn recv_timeout(&self, timeout: Duration) -> Option<RawFrame> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(frame) = inner.queue.pop_front() {
                return Some(frame);
            }
            if inner.closed {
                return None;
            }
            let (next, wait) = self
                .not_empty
                .wait_timeout(inner, timeout)
                .unwrap_or_else(|e| e.into_inner());
            inner = next;
            if wait.timed_out() && inner.queue.is_empty() && !inner.closed {
                return None;
            }
        }
    }

    /// Mark the channel closed and wake the consumer so it can drain and exit.
    pub(crate) fn close(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.closed = true;
        }
        self.not_empty.notify_all();
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Idle,
    Running,
    Paused,
}

/// Newtype promoting `cpal::Stream` to `Send` where the backend does not
/// provide it itself. On Linux (ALSA) and Windows (WASAPI) the stream is
/// already `Send`, so the promotion below is macOS-only; on macOS
/// (CoreAudio) the stream is `!Send` solely because its `StreamInner` holds
/// an `AudioObjectPropertyListener` wrapping `Box<dyn FnMut()>`.
struct StreamHandle(cpal::Stream);

// SAFETY: cpal's macOS `Stream` is `Arc<Mutex<StreamInner>>`; it is `!Send`
// only because `StreamInner` holds an `AudioObjectPropertyListener`
// (`Box<dyn FnMut()>`). That listener is invoked on CoreAudio's own thread
// and reaches `StreamInner` through a `Weak<Mutex<…>>` it locks, so every
// shared access — including listener-vs-drop — is synchronised inside cpal,
// independent of which thread owns the handle. We only ever *move* the
// handle (never share `&StreamHandle` across threads), and
// play/pause/drop carry no CoreAudio thread-affinity requirement, so
// promoting `Send` is safe. `Sync` is deliberately NOT asserted.
#[cfg(target_os = "macos")]
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for StreamHandle {}

/// How a source is captured: a cpal stream, or (the Windows mic path) a WASAPI
/// communications-mode capture thread that exits when the shared `stopped` flag
/// is set.
enum CaptureBackend {
    Cpal(StreamHandle),
    /// Constructed only under `cfg(windows)`; `allow(dead_code)` keeps
    /// non-Windows builds warning-clean.
    #[allow(dead_code)]
    WasapiComms(std::thread::JoinHandle<()>),
}

impl CaptureBackend {
    /// Pause or resume the underlying device. A cpal stream pauses/plays the
    /// hardware; the WASAPI comms thread keeps running and honours the source
    /// `paused` flag instead (dropping frames while paused), so this is a no-op
    /// for it.
    fn set_device_paused(&self, paused: bool) -> AppResult<()> {
        if let CaptureBackend::Cpal(s) = self {
            if paused {
                s.0.pause().map_err(Error::from)?;
            } else {
                s.0.play().map_err(Error::from)?;
            }
        }
        Ok(())
    }
}

/// Live capture resources for one source (mic OR loopback): the capture backend,
/// its pause flag, and the cpal→forwarder channel closed on stop.
struct SourceHandles {
    backend: CaptureBackend,
    paused: Arc<AtomicBool>,
    raw_ch: Arc<DropOldestChannel>,
}

/// Manages audio capture from a single input device, optionally mixing in the
/// system/call (loopback) audio.
pub struct AudioCaptureManager {
    device: Option<cpal::Device>,
    audio_device: Option<AudioDevice>,
    state: StreamState,
    /// Microphone capture resources (the primary source, always present while
    /// `Running`/`Paused`).
    mic: Option<SourceHandles>,
    /// System/call (loopback) capture resources — present only when
    /// `capture_system_audio` was on AND loopback opened successfully.
    loopback: Option<SourceHandles>,
    /// Signals the forwarder/mixer task(s) to exit.
    stopped: Option<Arc<AtomicBool>>,
}

impl AudioCaptureManager {
    /// Enumerate all audio input devices on the default cpal host.
    pub fn list_devices() -> AppResult<Vec<AudioDevice>> {
        device::list_input_devices().map_err(Into::into)
    }

    /// Open (but do not start) capture on the given device.
    ///
    /// `device_id = None` uses the OS default input device.
    pub fn open(device_id: Option<String>) -> AppResult<Self> {
        let cpal_device = device::resolve_device(device_id.as_deref())?;
        // Build the public descriptor (composite id + correct is_default) with
        // the same logic the device picker used, so ids round-trip identically.
        let audio_device = device::describe_device(device_id.as_deref())?;

        tracing::info!(
            target = "audio-capture",
            device = %audio_device.name,
            id = %audio_device.id,
            is_default = audio_device.is_default,
            "opened audio capture device"
        );

        Ok(AudioCaptureManager {
            device: Some(cpal_device),
            audio_device: Some(audio_device),
            state: StreamState::Idle,
            mic: None,
            loopback: None,
            stopped: None,
        })
    }

    /// Start capture.  Returns a pair of receivers the orchestrator polls.
    ///
    /// `sample_capacity` — bound on the `samples` tokio channel.
    /// `meter_capacity`  — bound on the `meter` tokio channel.
    /// `capture_system_audio` — when `true`, ALSO capture the default render
    /// endpoint in loopback mode and SUM it with the mic into the single
    /// `samples` stream (the orchestrator stays unchanged). On platforms where
    /// loopback is unsupported, or if opening loopback fails, capture falls back
    /// to mic-only with a warning — the recording is never failed.
    pub fn start(
        &mut self,
        sample_capacity: usize,
        meter_capacity: usize,
        capture_system_audio: bool,
    ) -> AppResult<AudioStreams> {
        if self.state != StreamState::Idle {
            return Err(Error::InvalidState {
                context: "start() called when not Idle".into(),
            }
            .into());
        }

        let cpal_device = self.device.as_ref().ok_or_else(|| Error::InvalidState {
            context: "no device; call open() first".into(),
        })?;

        let device_name = cpal_device
            .name()
            .unwrap_or_else(|_| "<unknown>".to_string());

        // The forwarder/mixer task(s) share one stop flag.
        let stopped = Arc::new(AtomicBool::new(false));

        // The mic source's raw ring + pause flag (shared by whichever backend).
        let mic_raw_ch = Arc::new(DropOldestChannel::new(RAW_RING_CAPACITY));
        let mic_paused = Arc::new(AtomicBool::new(false));

        // Mic capture backend: prefer the WASAPI communications path on Windows —
        // it returns the OS-processed (beamformed + AEC + noise-suppressed) MONO
        // stream at 16 kHz, so the forwarder resampler is a pass-through and we
        // never average a raw mic array ourselves. Falls back to cpal raw capture
        // off-Windows or if the comms path can't initialise. (v1 captures the
        // default communications device; honouring an explicit non-default mic
        // selection on this path is a follow-up.)
        #[cfg(windows)]
        let wasapi_mic = crate::wasapi_comms::try_start(
            Arc::clone(&mic_paused),
            Arc::clone(&stopped),
            Arc::clone(&mic_raw_ch),
        );
        #[cfg(not(windows))]
        let wasapi_mic: Option<(std::thread::JoinHandle<()>, u32)> = None;

        let (mic_backend, in_rate) = match wasapi_mic {
            Some((handle, rate)) => {
                tracing::info!(
                    target = "audio-capture",
                    device = %device_name,
                    sample_rate = rate,
                    channels = 1usize,
                    capture_system_audio,
                    "starting mic capture (WASAPI communications mode: OS beamforming/AEC/NS → mono)"
                );
                (CaptureBackend::WasapiComms(handle), rate)
            }
            None => {
                let config = device::preferred_config(cpal_device)?;
                let in_rate = config.sample_rate().0;
                let channels = config.channels() as usize;
                tracing::info!(
                    target = "audio-capture",
                    device = %device_name,
                    sample_rate = in_rate,
                    channels,
                    format = ?config.sample_format(),
                    capture_system_audio,
                    "starting mic capture (cpal raw)"
                );
                let mic_stream = build_input_stream(
                    cpal_device,
                    &config,
                    channels,
                    in_rate,
                    Arc::clone(&mic_paused),
                    Arc::clone(&mic_raw_ch),
                )?;
                mic_stream.play().map_err(Error::from)?;
                (CaptureBackend::Cpal(StreamHandle(mic_stream)), in_rate)
            }
        };

        // --- Try to open the loopback source when requested ---
        // A failure (unsupported platform, no render device) degrades to
        // mic-only with a warning; it never fails the recording.
        let loopback_open = if capture_system_audio {
            match crate::loopback::open_loopback() {
                Ok(lb) => Some(lb),
                Err(e) => {
                    tracing::warn!(
                        target = "audio-capture",
                        "system-audio capture requested but loopback unavailable ({e}); \
                         falling back to mic-only"
                    );
                    None
                }
            }
        } else {
            None
        };

        // --- Tokio output channels (the public surface, unchanged) ---
        let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(sample_capacity);
        let (meter_tx, meter_rx) = mpsc::channel::<AudioMeterFrame>(meter_capacity);

        match loopback_open {
            // ---------- Mic + loopback: resample each, then mix ----------
            Some(lb) => {
                tracing::info!(
                    target = "audio-capture",
                    loopback_rate = lb.in_rate,
                    "loopback opened; mixing microphone + system audio \
                     (a starved/idle source is zero-filled so the mic keeps flowing)"
                );
                lb.stream.play().map_err(Error::from)?;
                let lb_stream = StreamHandle(lb.stream);

                // Per-source resampled batch channels feeding the mixer. Bounded
                // so a slow mixer back-pressures without unbounded growth; the
                // RT callbacks remain protected by the upstream drop-oldest ring.
                let (mic_batch_tx, mic_batch_rx) = mpsc::channel::<AudioFrameBatch>(32);
                let (sys_batch_tx, sys_batch_rx) = mpsc::channel::<AudioFrameBatch>(32);

                spawn_resample_forwarder(
                    Arc::clone(&mic_raw_ch),
                    Arc::clone(&stopped),
                    in_rate,
                    mic_batch_tx,
                    "mic",
                );
                spawn_resample_forwarder(
                    Arc::clone(&lb.raw_ch),
                    Arc::clone(&stopped),
                    lb.in_rate,
                    sys_batch_tx,
                    "loopback",
                );
                spawn_mixer(mic_batch_rx, sys_batch_rx, sample_tx, meter_tx);

                self.loopback = Some(SourceHandles {
                    backend: CaptureBackend::Cpal(lb_stream),
                    paused: lb.paused,
                    raw_ch: lb.raw_ch,
                });
            }
            // ---------- Mic only: resample + meter straight to output ----------
            None => {
                spawn_mic_only_forwarder(
                    Arc::clone(&mic_raw_ch),
                    Arc::clone(&stopped),
                    in_rate,
                    sample_tx,
                    meter_tx,
                );
            }
        }

        self.mic = Some(SourceHandles {
            backend: mic_backend,
            paused: mic_paused,
            raw_ch: mic_raw_ch,
        });
        self.stopped = Some(stopped);
        self.state = StreamState::Running;

        Ok(AudioStreams {
            samples: sample_rx,
            meter: meter_rx,
        })
    }

    /// Pause capture.  Samples stop flowing; the cpal stream stays alive.
    pub fn pause(&mut self) -> AppResult<()> {
        match self.state {
            StreamState::Running => {}
            _ => {
                return Err(Error::InvalidState {
                    context: "pause() called when not Running".into(),
                }
                .into())
            }
        }

        for src in [self.mic.as_ref(), self.loopback.as_ref()].into_iter().flatten() {
            src.paused.store(true, Ordering::Relaxed);
            src.backend.set_device_paused(true)?;
        }
        self.state = StreamState::Paused;
        tracing::info!(target = "audio-capture", "capture paused");
        Ok(())
    }

    /// Resume capture after a pause.
    pub fn resume(&mut self) -> AppResult<()> {
        match self.state {
            StreamState::Paused => {}
            _ => {
                return Err(Error::InvalidState {
                    context: "resume() called when not Paused".into(),
                }
                .into())
            }
        }

        for src in [self.mic.as_ref(), self.loopback.as_ref()].into_iter().flatten() {
            src.paused.store(false, Ordering::Relaxed);
            src.backend.set_device_paused(false)?;
        }
        self.state = StreamState::Running;
        tracing::info!(target = "audio-capture", "capture resumed");
        Ok(())
    }

    /// Stop capture and tear down the stream.
    pub fn stop(&mut self) -> AppResult<()> {
        if self.state == StreamState::Idle {
            return Err(Error::InvalidState {
                context: "stop() called when Idle".into(),
            }
            .into());
        }

        // Signal forwarder/mixer task(s) to exit cleanly.
        if let Some(stopped) = &self.stopped {
            stopped.store(true, Ordering::Relaxed);
        }
        // Close each source's ring channel so its forwarder wakes immediately
        // (rather than waiting out its recv timeout) and drains any tail.
        for src in [self.mic.as_ref(), self.loopback.as_ref()].into_iter().flatten() {
            src.raw_ch.close();
        }

        // Drop the streams; cpal stops the callbacks.
        self.mic = None;
        self.loopback = None;
        self.stopped = None;
        self.state = StreamState::Idle;
        tracing::info!(target = "audio-capture", "capture stopped");
        Ok(())
    }

    /// The opened device, or `None` if not opened.
    pub fn current_device(&self) -> Option<&AudioDevice> {
        self.audio_device.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Forwarder + mixer tasks
// ---------------------------------------------------------------------------

/// Mic-only forwarder: drain one source's raw ring, resample to 16 kHz mono,
/// meter, and forward batches straight to the public output channels. This is
/// the original single-source path (behaviour unchanged when
/// `capture_system_audio` is off).
fn spawn_mic_only_forwarder(
    raw_ch: Arc<DropOldestChannel>,
    stopped: Arc<AtomicBool>,
    in_rate: u32,
    sample_tx: mpsc::Sender<AudioFrameBatch>,
    meter_tx: mpsc::Sender<AudioMeterFrame>,
) {
    tokio::task::spawn_blocking(move || {
        let mut resampler = match StreamResampler::new(in_rate) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(target = "audio-capture", "failed to create resampler: {e}");
                return;
            }
        };
        // ~512 samples at 16 kHz ≈ 32 ms window → ~30 Hz meter emission
        let mut meter = LevelMeter::new(512);
        let mut out_clock_samples: u64 = 0; // 16 kHz samples emitted so far

        loop {
            if stopped.load(Ordering::Relaxed) {
                break;
            }
            let frame = match raw_ch.recv_timeout(Duration::from_millis(50)) {
                Some(f) => f,
                None => continue,
            };
            resampler.push(&frame.samples, &mut |chunk| {
                let batch = batch_from(chunk, &mut out_clock_samples);
                if sample_tx.blocking_send(batch).is_err() {
                    tracing::warn!(
                        target = "audio-capture",
                        "sample channel closed; forwarder exiting"
                    );
                    return;
                }
                meter.push(chunk, |mf| {
                    if meter_tx.blocking_send(mf).is_err() {
                        tracing::warn!(target = "audio-capture", "meter channel closed");
                    }
                });
            });
        }

        resampler.finish(&mut |chunk| {
            if !chunk.is_empty() {
                let batch = batch_from(chunk, &mut out_clock_samples);
                let _ = sample_tx.blocking_send(batch);
                meter.push(chunk, |mf| {
                    let _ = meter_tx.blocking_send(mf);
                });
            }
        });
        meter.flush(|mf| {
            let _ = meter_tx.blocking_send(mf);
        });

        tracing::debug!(
            target = "audio-capture",
            out_samples = out_clock_samples,
            "mic-only forwarder task exiting"
        );
    });
}

/// Per-source forwarder used in mixed mode: drain one source's raw ring,
/// resample to 16 kHz mono, and forward `AudioFrameBatch`es into an internal
/// channel feeding the mixer. No metering here — the mixer meters the summed
/// output so the level reflects what is actually recorded.
fn spawn_resample_forwarder(
    raw_ch: Arc<DropOldestChannel>,
    stopped: Arc<AtomicBool>,
    in_rate: u32,
    batch_tx: mpsc::Sender<AudioFrameBatch>,
    source: &'static str,
) {
    tokio::task::spawn_blocking(move || {
        let mut resampler = match StreamResampler::new(in_rate) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    target = "audio-capture",
                    "failed to create {source} resampler: {e}"
                );
                return;
            }
        };
        // The per-source forwarder does not stamp timestamps (the mixer owns the
        // output clock); a throwaway counter satisfies `batch_from`.
        let mut clock: u64 = 0;

        loop {
            if stopped.load(Ordering::Relaxed) {
                break;
            }
            let frame = match raw_ch.recv_timeout(Duration::from_millis(50)) {
                Some(f) => f,
                None => continue,
            };
            resampler.push(&frame.samples, &mut |chunk| {
                let batch = batch_from(chunk, &mut clock);
                if batch_tx.blocking_send(batch).is_err() {
                    tracing::warn!(
                        target = "audio-capture",
                        "{source} mixer channel closed; forwarder exiting"
                    );
                }
            });
        }

        resampler.finish(&mut |chunk| {
            if !chunk.is_empty() {
                let batch = batch_from(chunk, &mut clock);
                let _ = batch_tx.blocking_send(batch);
            }
        });
        tracing::debug!(target = "audio-capture", "{source} resample forwarder exiting");
    });
}

/// Mixer task: drain the mic + loopback batch channels, SUM sample-wise with
/// clamp + zero-fill (per [`crate::mixer::MixState`]), meter the mixed output,
/// and forward `AudioFrameBatch`es into the public `samples` channel.
///
/// Sync strategy: the mixer emits whatever both sources have in common
/// (`min(len)`) on each tick, holding the surplus from the faster source for
/// the next tick. A source that has fallen behind simply contributes its
/// backlog later (small drift, tolerated by transcription); a source that has
/// ENDED is zero-filled on the final flush so the timeline keeps advancing.
fn spawn_mixer(
    mut mic_rx: mpsc::Receiver<AudioFrameBatch>,
    mut sys_rx: mpsc::Receiver<AudioFrameBatch>,
    sample_tx: mpsc::Sender<AudioFrameBatch>,
    meter_tx: mpsc::Sender<AudioMeterFrame>,
) {
    tokio::task::spawn(async move {
        let mut state = crate::mixer::MixState::new();
        let mut meter = LevelMeter::new(512);
        let mut mic_open = true;
        let mut sys_open = true;

        // Drain until BOTH sources have ended and the channels are empty.
        while mic_open || sys_open {
            tokio::select! {
                biased;
                m = mic_rx.recv(), if mic_open => match m {
                    Some(b) => state.push_mic(&b.samples),
                    None => mic_open = false,
                },
                s = sys_rx.recv(), if sys_open => match s {
                    Some(b) => state.push_sys(&b.samples),
                    None => sys_open = false,
                },
            }

            // Emit everything both sources have in common so far.
            while let Some(batch) = state.drain_paired() {
                meter_then_send(&mut meter, &batch.samples, &meter_tx);
                if sample_tx.send(batch).await.is_err() {
                    tracing::warn!(target = "audio-capture", "sample channel closed; mixer exiting");
                    return;
                }
            }

            // Starvation valve: if one source is idle (e.g. loopback with no
            // system audio playing) the paired drain above emits nothing, so
            // flush the live source's backlog beyond the skew cap rather than
            // buffer it forever.
            while let Some(batch) = state.drain_starved() {
                meter_then_send(&mut meter, &batch.samples, &meter_tx);
                if sample_tx.send(batch).await.is_err() {
                    tracing::warn!(target = "audio-capture", "sample channel closed; mixer exiting");
                    return;
                }
            }
        }

        // Both sources ended: flush any surplus, zero-filling the other source.
        if let Some(batch) = state.drain_remaining() {
            meter_then_send(&mut meter, &batch.samples, &meter_tx);
            let _ = sample_tx.send(batch).await;
        }
        meter.flush(|mf| {
            let _ = meter_tx.try_send(mf);
        });
        tracing::debug!(target = "audio-capture", "mixer task exiting");
    });
}

/// Wrap a resampled run in an [`AudioFrameBatch`], advancing `clock` by its
/// length and computing the start/end recording-clock offsets in ms.
fn batch_from(chunk: &[f32], clock: &mut u64) -> AudioFrameBatch {
    let start_ms = *clock * 1000 / crate::resample::TARGET_RATE as u64;
    *clock += chunk.len() as u64;
    let end_ms = *clock * 1000 / crate::resample::TARGET_RATE as u64;
    AudioFrameBatch {
        samples: chunk.to_vec(),
        start_ms,
        end_ms,
    }
}

/// Push a mixed run through the meter, forwarding any emitted meter frames.
fn meter_then_send(
    meter: &mut LevelMeter,
    samples: &[f32],
    meter_tx: &mpsc::Sender<AudioMeterFrame>,
) {
    meter.push(samples, |mf| {
        if meter_tx.try_send(mf).is_err() {
            // Drop a meter frame under back-pressure (same policy as elsewhere).
        }
    });
}

// ---------------------------------------------------------------------------
// cpal stream builder — dispatches on sample format
// ---------------------------------------------------------------------------

pub(crate) fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    channels: usize,
    in_rate: u32,
    paused: Arc<AtomicBool>,
    raw_ch: Arc<DropOldestChannel>,
) -> Result<cpal::Stream, Error> {
    match config.sample_format() {
        cpal::SampleFormat::U8 => {
            build_typed_stream::<u8>(device, config, channels, in_rate, paused, raw_ch)
        }
        cpal::SampleFormat::I8 => {
            build_typed_stream::<i8>(device, config, channels, in_rate, paused, raw_ch)
        }
        cpal::SampleFormat::I16 => {
            build_typed_stream::<i16>(device, config, channels, in_rate, paused, raw_ch)
        }
        cpal::SampleFormat::I32 => {
            build_typed_stream::<i32>(device, config, channels, in_rate, paused, raw_ch)
        }
        cpal::SampleFormat::F32 => {
            build_typed_stream::<f32>(device, config, channels, in_rate, paused, raw_ch)
        }
        fmt => Err(Error::Cpal {
            context: format!("unsupported sample format: {fmt:?}"),
        }),
    }
}

fn build_typed_stream<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    channels: usize,
    in_rate: u32,
    paused: Arc<AtomicBool>,
    raw_ch: Arc<DropOldestChannel>,
) -> Result<cpal::Stream, Error>
where
    T: Sample + SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    let mut mono_buf = Vec::<f32>::new();
    // Suppress unused warning: in_rate is accepted for consistency but not
    // needed now that the recording-clock is tracked by the output resampler.
    let _ = in_rate;

    let stream = device.build_input_stream(
        &config.clone().into(),
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if paused.load(Ordering::Relaxed) {
                return;
            }

            // Mix down to mono f32.
            mono_buf.clear();
            if channels == 1 {
                mono_buf.extend(data.iter().map(|&s| s.to_sample::<f32>()));
            } else {
                let frame_count = data.len() / channels;
                mono_buf.reserve(frame_count);
                for frame in data.chunks_exact(channels) {
                    let sum: f32 = frame.iter().map(|&s| s.to_sample::<f32>()).sum();
                    mono_buf.push(sum / channels as f32);
                }
            }

            raw_ch.push(RawFrame {
                samples: mono_buf.clone(),
            });
        },
        move |err| {
            tracing::error!(target = "audio-capture", "cpal stream error: {err}");
        },
        None,
    )?;

    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-sample frame tagged by `id` (its only sample value), so
    /// tests can assert which frames survived an overflow.
    fn frame(id: f32) -> RawFrame {
        RawFrame { samples: vec![id] }
    }

    fn tag(f: &RawFrame) -> f32 {
        f.samples[0]
    }

    /// Drain everything currently queued (no waiting).
    fn drain(ch: &DropOldestChannel) -> Vec<f32> {
        let mut out = Vec::new();
        while let Some(f) = ch.recv_timeout(Duration::from_millis(0)) {
            out.push(tag(&f));
        }
        out
    }

    /// Under overflow, the OLDEST frames are evicted and the NEWEST are
    /// retained — the drop-oldest contract. With capacity 3 and 6 pushes,
    /// only the last 3 (3,4,5) survive in order.
    #[test]
    fn overflow_drops_oldest_keeps_newest() {
        let ch = DropOldestChannel::new(3);
        for i in 0..6 {
            ch.push(frame(i as f32));
        }
        let got = drain(&ch);
        assert_eq!(
            got,
            vec![3.0, 4.0, 5.0],
            "expected the 3 newest frames in FIFO order; oldest must be dropped"
        );
    }

    /// Without overflow, all frames are delivered oldest-first.
    #[test]
    fn no_overflow_preserves_fifo_order() {
        let ch = DropOldestChannel::new(8);
        for i in 0..4 {
            ch.push(frame(i as f32));
        }
        assert_eq!(drain(&ch), vec![0.0, 1.0, 2.0, 3.0]);
    }

    /// The producer never blocks: push must return even while the consumer
    /// holds the lock. We simulate contention by holding the inner lock and
    /// confirming `push` drops the frame rather than deadlocking. The held
    /// frame count stays at zero (the push could not enqueue).
    #[test]
    fn push_never_blocks_under_contention() {
        let ch = Arc::new(DropOldestChannel::new(4));
        let guard = ch.inner.lock().unwrap();
        // With the lock held, `try_lock` in push must fail and drop the frame
        // without blocking this thread.
        ch.push(frame(42.0));
        // Queue is still empty because the push could not acquire the lock.
        assert!(guard.queue.is_empty());
        drop(guard);
        // After releasing, a subsequent push succeeds.
        ch.push(frame(7.0));
        assert_eq!(drain(&ch), vec![7.0]);
    }

    /// After `close`, the consumer drains any queued tail then returns `None`.
    #[test]
    fn close_drains_tail_then_returns_none() {
        let ch = DropOldestChannel::new(4);
        ch.push(frame(1.0));
        ch.push(frame(2.0));
        ch.close();
        assert_eq!(ch.recv_timeout(Duration::from_millis(0)).map(|f| tag(&f)), Some(1.0));
        assert_eq!(ch.recv_timeout(Duration::from_millis(0)).map(|f| tag(&f)), Some(2.0));
        assert!(ch.recv_timeout(Duration::from_millis(0)).is_none());
    }

    /// Concurrent producer/consumer: a fast producer overflowing a small ring
    /// must never panic and the consumer must keep making progress. The
    /// highest-id frame produced must eventually be observed (newest retained).
    #[test]
    fn concurrent_overflow_no_panic_newest_observed() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        const CAP: usize = 4;
        const N: usize = 5_000;
        let ch = Arc::new(DropOldestChannel::new(CAP));
        let done = Arc::new(AtomicBool::new(false));

        let producer = {
            let ch = Arc::clone(&ch);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                for i in 0..N {
                    ch.push(frame(i as f32));
                }
                done.store(true, Ordering::Release);
                ch.close();
            })
        };

        let consumer = {
            let ch = Arc::clone(&ch);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut last = -1.0f32;
                let mut count = 0usize;
                loop {
                    match ch.recv_timeout(Duration::from_millis(50)) {
                        Some(f) => {
                            let t = tag(&f);
                            // FIFO within the ring: ids only increase.
                            assert!(
                                t > last,
                                "frames must arrive oldest-first: {t} after {last}"
                            );
                            last = t;
                            count += 1;
                        }
                        // recv_timeout returns None on spurious timeout (queue
                        // temporarily empty but producer still running) OR on
                        // post-close drain-complete. Exit ONLY once the producer
                        // has set `done` and a subsequent recv_timeout confirms the
                        // channel is drained (returns None again after close).
                        None => {
                            if done.load(Ordering::Acquire) {
                                // Channel is closed (producer set done and called
                                // close()); the None confirms it is also drained.
                                break;
                            }
                            // Spurious timeout: producer still running but the
                            // small ring was momentarily empty under load. Keep
                            // polling rather than exiting early.
                        }
                    }
                }
                (last, count)
            })
        };

        producer.join().expect("producer panicked");
        let (last, count) = consumer.join().expect("consumer panicked");

        // Some frames were dropped (ring smaller than N) but never more than
        // produced, and the consumer saw a high-id (newest-retained) frame.
        assert!(count <= N, "received {count} > produced {N}");
        assert!(
            last >= (N as f32 - CAP as f32 - 1.0),
            "expected a near-newest frame; last={last}, N={N}"
        );
    }
}
