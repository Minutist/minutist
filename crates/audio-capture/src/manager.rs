//! `AudioCaptureManager` — cpal stream init, pause/resume/stop, and the
//! bridge from the cpal callback thread into async Tokio channels.
//!
//! Threading model (per `architecture/cross-cutting.md`):
//!   • The cpal callback runs on cpal's own thread.  It pushes raw samples
//!     into a `std::sync::mpsc::SyncSender` (bounded capacity 8).  On
//!     overflow the oldest item is dropped and a `tracing::warn!` fires.
//!   • A `tokio::task::spawn_blocking` task drains that channel and forwards
//!     resampled + metered output into bounded `tokio::sync::mpsc` channels
//!     whose capacities are passed in by the caller.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
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

/// A channel pair that drops the *oldest* item when the bounded channel is
/// full, instead of blocking the audio callback.
struct DropOldestChannel {
    tx: std::sync::mpsc::SyncSender<RawFrame>,
    rx: Arc<Mutex<std::sync::mpsc::Receiver<RawFrame>>>,
    capacity: usize,
}

impl DropOldestChannel {
    fn new(capacity: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(capacity);
        DropOldestChannel {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            capacity,
        }
    }

    /// Push a frame, dropping the oldest on overflow.
    fn push(&self, frame: RawFrame) {
        match self.tx.try_send(frame) {
            Ok(()) => {}
            Err(std::sync::mpsc::TrySendError::Full(frame)) => {
                tracing::warn!(
                    target = "audio-capture",
                    "cpal→forwarder channel full (capacity {}); dropping oldest frame",
                    self.capacity
                );
                // Drain one item to make room, then retry.
                {
                    let rx = self.rx.lock().unwrap();
                    let _ = rx.try_recv();
                }
                // Best-effort second attempt; if still full, drop silently.
                let _ = self.tx.try_send(frame);
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                // Consumer gone (stop was called). Silently ignore.
            }
        }
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
        let name = cpal_device.name().unwrap_or_else(|_| "Unknown".into());
        let default_name = cpal::default_host()
            .default_input_device()
            .and_then(|d| d.name().ok());
        let is_default = Some(name.clone()) == default_name;

        tracing::info!(
            target = "audio-capture",
            device = %name,
            is_default,
            "opened audio capture device"
        );

        Ok(AudioCaptureManager {
            device: Some(cpal_device),
            audio_device: Some(AudioDevice {
                id: name.clone(),
                name,
                is_default,
            }),
            state: StreamState::Idle,
            _stream: None,
            paused: None,
            stopped: None,
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
        let rx_arc = Arc::clone(&raw_ch.rx);
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

            let rx = rx_arc.lock().unwrap();
            loop {
                if stopped_fwd.load(Ordering::Relaxed) {
                    break;
                }

                let frame = match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(f) => f,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
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

        // Drop the stream; cpal stops the callback.
        self._stream = None;
        self.paused = None;
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
