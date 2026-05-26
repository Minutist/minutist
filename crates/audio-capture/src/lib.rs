//! `audio-capture` — cpal audio input, rubato resampling to 16 kHz mono,
//! level metering, and bounded async delivery to the orchestrator.
//!
//! Public surface:
//! - [`AudioCaptureManager`] — open/start/pause/resume/stop lifecycle.
//! - [`AudioStreams`] — pair of tokio mpsc receivers (samples + meter).
//! - [`AudioFrameBatch`] — resampled 16 kHz mono f32 batch with timestamps.
//!
//! See `architecture/cross-cutting.md` for the threading model this crate
//! implements.

mod device;
pub mod error;
mod manager;
mod meter;
mod resample;

#[cfg(any(test, feature = "test-source"))]
pub mod test_source;

pub use manager::{AudioCaptureManager, AudioFrameBatch, AudioStreams};

// Integration-test modules that live alongside the source.
#[cfg(test)]
mod tests {
    use meeting_app_common::AudioDevice;

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
        // On a machine with at least one input device we expect a non-empty list.
        assert!(
            !devices.is_empty(),
            "expected at least one audio input device"
        );
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
