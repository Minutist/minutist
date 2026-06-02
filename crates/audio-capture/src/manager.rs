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
use meeting_app_common::{AppResult, AudioDevice, AudioMeterFrame};
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

struct RawFrame {
    samples: Vec<f32>,
}

/// Shared state behind [`DropOldestChannel`].
struct ChannelInner {
    /// Bounded FIFO of pending raw frames. Oldest at the front.
    queue: VecDeque<RawFrame>,
    /// Set once the consumer wants no more frames (stop). The producer then
    /// drops silently and the consumer drains the tail and exits.
    closed: bool,
}

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
struct DropOldestChannel {
    inner: Mutex<ChannelInner>,
    /// Signals the consumer that a frame is available or the channel closed.
    not_empty: Condvar,
    capacity: usize,
}

impl DropOldestChannel {
    fn new(capacity: usize) -> Self {
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
    fn push(&self, frame: RawFrame) {
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
    fn recv_timeout(&self, timeout: Duration) -> Option<RawFrame> {
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
    fn close(&self) {
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

/// Manages audio capture from a single input device.
pub struct AudioCaptureManager {
    device: Option<cpal::Device>,
    audio_device: Option<AudioDevice>,
    state: StreamState,
    /// cpal Stream kept alive for the duration of capture.
    _stream: Option<cpal::Stream>,
    /// Signals the cpal callback to stop emitting samples.
    paused: Option<Arc<AtomicBool>>,
    /// Signals the forwarder task to exit.
    stopped: Option<Arc<AtomicBool>>,
    /// cpal→forwarder channel; closed on stop to wake the forwarder promptly.
    raw_ch: Option<Arc<DropOldestChannel>>,
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
            _stream: None,
            paused: None,
            stopped: None,
            raw_ch: None,
        })
    }

    /// Start capture.  Returns a pair of receivers the orchestrator polls.
    ///
    /// `sample_capacity` — bound on the `samples` tokio channel.
    /// `meter_capacity`  — bound on the `meter` tokio channel.
    pub fn start(
        &mut self,
        sample_capacity: usize,
        meter_capacity: usize,
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

        let config = device::preferred_config(cpal_device)?;
        let in_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        tracing::info!(
            target = "audio-capture",
            sample_rate = in_rate,
            channels,
            format = ?config.sample_format(),
            "starting capture stream"
        );

        // --- Bounded ring channel: cpal callback → forwarder (capacity = 8) ---
        let raw_ch = Arc::new(DropOldestChannel::new(8));
        let raw_ch_cb = Arc::clone(&raw_ch);

        // --- Pause/stop flags ---
        let paused = Arc::new(AtomicBool::new(false));
        let paused_cb = Arc::clone(&paused);
        let stopped = Arc::new(AtomicBool::new(false));
        let stopped_fwd = Arc::clone(&stopped);

        // --- Build the cpal stream ---
        let stream = build_input_stream(
            cpal_device,
            &config,
            channels,
            in_rate,
            paused_cb,
            raw_ch_cb,
        )?;
        stream.play().map_err(Error::from)?;

        // --- Tokio output channels ---
        let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(sample_capacity);
        let (meter_tx, meter_rx) = mpsc::channel::<AudioMeterFrame>(meter_capacity);

        // --- spawn_blocking forwarder task ---
        let rx_ch = Arc::clone(&raw_ch);
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
                if stopped_fwd.load(Ordering::Relaxed) {
                    break;
                }

                // Pop one frame (the lock is released inside `recv_timeout`
                // before the expensive resample below, so the RT producer is
                // never contended for long).
                let frame = match rx_ch.recv_timeout(Duration::from_millis(50)) {
                    Some(f) => f,
                    None => continue,
                };

                resampler.push(&frame.samples, &mut |chunk| {
                    let start_ms = out_clock_samples * 1000 / crate::resample::TARGET_RATE as u64;
                    out_clock_samples += chunk.len() as u64;
                    let end_ms = out_clock_samples * 1000 / crate::resample::TARGET_RATE as u64;

                    let batch = AudioFrameBatch {
                        samples: chunk.to_vec(),
                        start_ms,
                        end_ms,
                    };
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

            // Flush tail samples on stop.
            resampler.finish(&mut |chunk| {
                if !chunk.is_empty() {
                    let start_ms = out_clock_samples * 1000 / crate::resample::TARGET_RATE as u64;
                    out_clock_samples += chunk.len() as u64;
                    let end_ms = out_clock_samples * 1000 / crate::resample::TARGET_RATE as u64;
                    let batch = AudioFrameBatch {
                        samples: chunk.to_vec(),
                        start_ms,
                        end_ms,
                    };
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
                "forwarder task exiting"
            );
        });

        self._stream = Some(stream);
        self.paused = Some(paused);
        self.stopped = Some(stopped);
        self.raw_ch = Some(raw_ch);
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

        if let Some(p) = &self.paused {
            p.store(true, Ordering::Relaxed);
        }
        if let Some(stream) = &self._stream {
            stream.pause().map_err(Error::from)?;
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

        if let Some(p) = &self.paused {
            p.store(false, Ordering::Relaxed);
        }
        if let Some(stream) = &self._stream {
            stream.play().map_err(Error::from)?;
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

        // Signal forwarder task to exit cleanly.
        if let Some(stopped) = &self.stopped {
            stopped.store(true, Ordering::Relaxed);
        }
        // Close the channel so the forwarder wakes immediately (rather than
        // waiting out its recv timeout) and drains any tail.
        if let Some(raw_ch) = &self.raw_ch {
            raw_ch.close();
        }

        // Drop the stream; cpal stops the callback.
        self._stream = None;
        self.paused = None;
        self.stopped = None;
        self.raw_ch = None;
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
// cpal stream builder — dispatches on sample format
// ---------------------------------------------------------------------------

fn build_input_stream(
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
            thread::spawn(move || {
                let mut last = -1.0f32;
                let mut count = 0usize;
                while let Some(f) = ch.recv_timeout(Duration::from_millis(50)) {
                    let t = tag(&f);
                    // FIFO within the ring: ids only increase.
                    assert!(t > last, "frames must arrive oldest-first: {t} after {last}");
                    last = t;
                    count += 1;
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
