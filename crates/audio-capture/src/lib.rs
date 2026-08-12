//! `audio-capture` — cpal-based microphone (and optional system-audio
//! loopback) capture, rubato resampling to 16 kHz mono, level metering, and
//! bounded async delivery to the orchestrator.
//!
//! Owns the audio device, sample-rate negotiation, the capture ring buffer,
//! and device enumeration for the settings UI.
//!
//! Public surface:
//! - [`AudioCaptureManager`] — open/start/pause/resume/stop lifecycle.
//! - [`AudioStreams`] — pair of tokio mpsc receivers (samples + meter).
//! - [`AudioFrameBatch`] — resampled 16 kHz mono f32 batch with timestamps.
//!
//! # Sample rate
//!
//! All output is 16 kHz mono — the rate mtmd's audio encoder expects — so
//! downstream consumers (`vad-chunker`, `asr-runtime`, `asr-parakeet`) never
//! resample.
//!
//! # Back-pressure
//!
//! The capture→forwarder path is a bounded (`RAW_RING_CAPACITY`, ~10 s of
//! buffers) `Mutex<VecDeque>` + `Condvar` ring. The realtime callback pushes
//! via `try_lock` only — it never blocks — and on overflow drops the OLDEST
//! frame rather than the newest. The ring, and the downstream `samples` tokio
//! channel, are both sized deep enough to ride the model-load burst at record
//! start; undersizing either back-pressures into a drop-flood that truncates
//! recordings. The level meter samples in 512-sample (~32 ms, ~30 Hz) windows.
//!
//! # Windows mic capture
//!
//! On Windows the mic is opened via WASAPI **communications mode** (the
//! `wasapi` crate, a `cfg(windows)` dependency): tagging the stream
//! `AudioCategory_Communications` hands capture to the OS/driver voice
//! pipeline (array beamforming + AEC + noise suppression), which delivers an
//! already-processed mono stream — this crate never beamforms or averages a
//! raw mic array itself. Falls back to the cpal raw path if the comms path
//! fails to initialise, and on all other platforms.
//!
//! # Device identity
//!
//! `AudioDevice.id` is an opaque `"{enumeration-index}\u{1f}{name}"` string
//! (ASCII unit separator, which no device name can contain), so same-named
//! ALSA devices resolve to distinct ids; `is_default` is the first
//! name-match. `resolve_device` parses the composite id (index authoritative,
//! name-checked) and falls back to name matching for legacy bare-name ids
//! persisted in `settings.input_device_id`.
//!
//! # System-audio mixing
//!
//! When `capture_system_audio` is enabled (on by default), `start` also opens
//! the default render endpoint in loopback mode as a second capture source
//! and sums it with the microphone into the same `samples` stream, so a
//! Teams-style call transcribes all participants, not just the local user;
//! the public `AudioStreams`/`AudioFrameBatch` shapes are unchanged and
//! diarization downstream separates the speakers.
//!
//! - `loopback` (Windows-only) opens an INPUT stream on the render device,
//!   which cpal's WASAPI backend transparently turns into a loopback capture
//!   (`AUDCLNT_STREAMFLAGS_LOOPBACK`) — reusing the same sample-format
//!   dispatch, mono downmix, and drop-oldest ring as mic capture, with no
//!   extra dependency. On non-Windows platforms, or if the loopback open
//!   fails, the source falls back to mic-only capture; the recording is
//!   never failed outright.
//! - `mixer` resamples each source to 16 kHz mono independently, drains both
//!   per-source batch channels, sums sample-wise with clamping to
//!   `[-1.0, 1.0]`, meters the mixed output, and forwards `AudioFrameBatch`es
//!   downstream. Each tick emits the samples both sources have in common
//!   (`min(len)`); the faster source's surplus carries to the next tick
//!   (small drift is tolerated by transcription), and a source that has
//!   ended is zero-filled on the final flush so the timeline keeps
//!   advancing.
//!
//! Echo cancellation is out of scope: the system-audio toggle (on by
//! default) is the only defence against a mic hearing its own call output —
//! turn it off if that happens.
//!
//! See `architecture/cross-cutting.md` for the threading model this crate
//! implements.

mod device;
pub mod error;
mod loopback;
mod manager;
mod meter;
mod mixer;
mod resample;
/// Windows-only: mic capture via WASAPI communications mode (OS voice DSP →
/// processed mono), preferred over the cpal raw path on Windows.
#[cfg(windows)]
mod wasapi_comms;

#[cfg(any(test, feature = "test-source"))]
pub mod test_source;

pub use manager::{AudioCaptureManager, AudioFrameBatch, AudioStreams};

// Integration-test modules that live alongside the source.
#[cfg(test)]
mod tests {
    use minutist_common::AudioDevice;

    use super::*;

    // ------------------------------------------------------------------
    // Device list shape
    // ------------------------------------------------------------------

    /// `list_devices()` returns `Vec<AudioDevice>` with non-empty `id`
    /// strings.  On Linux CI without an audio device this is skipped.
    #[test]
    #[cfg_attr(target_os = "linux", ignore)]
    fn device_list_shape() {
        let devices = AudioCaptureManager::list_devices().expect("list_devices failed");
        // Headless machines (CI runners) legitimately expose zero input
        // devices; the shape assertions below need at least one to verify,
        // so an empty list is a skip, not a failure.
        if devices.is_empty() {
            eprintln!("skip: no audio input devices on this machine");
            return;
        }
        for d in &devices {
            assert!(!d.id.is_empty(), "device id must not be empty");
            assert!(!d.name.is_empty(), "device name must not be empty");
        }
        // At most one device is the default.
        let defaults: Vec<&AudioDevice> = devices.iter().filter(|d| d.is_default).collect();
        assert!(
            defaults.len() <= 1,
            "at most one device should be marked as default, got {}",
            defaults.len()
        );
    }
}
