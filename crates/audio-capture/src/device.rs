//! Audio device enumeration.
//!
//! Returns `common::AudioDevice` values; does not expose cpal types to callers.

use cpal::traits::{DeviceTrait, HostTrait};
use meeting_app_common::AudioDevice;

use crate::error::Error;

/// Enumerate all available audio input devices on the default cpal host.
///
/// `AudioDevice::id` is the cpal device name (unique per host on Windows and
/// macOS; on Linux ALSA names can collide but are the best stable key cpal
/// exposes). `is_default` reflects the OS default at the moment of the call.
pub(crate) fn list_input_devices() -> Result<Vec<AudioDevice>, Error> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    let devices = host.input_devices()?;
    let mut out = Vec::new();

    for device in devices {
        let name = device.name().unwrap_or_else(|_| "Unknown".into());
        let is_default = Some(name.clone()) == default_name;
        // Use the device name as the stable id; stable enough for single-session
        // round-trips through the IPC layer. On hosts where two devices share a
        // name (rare) the first one wins in `open()`.
        out.push(AudioDevice {
            id: name.clone(),
            name,
            is_default,
        });
    }

    Ok(out)
}

/// Resolve an optional device id to a cpal `Device`.
///
/// `None` → OS default input device.
/// `Some(id)` → first input device whose name matches `id`.
pub(crate) fn resolve_device(device_id: Option<&str>) -> Result<cpal::Device, Error> {
    let host = cpal::default_host();

    match device_id {
        None => host.default_input_device().ok_or(Error::NoInputDevice),
        Some(id) => {
            let devices = host.input_devices()?;
            for device in devices {
                if device.name().ok().as_deref() == Some(id) {
                    return Ok(device);
                }
            }
            Err(Error::DeviceNotFound { id: id.to_owned() })
        }
    }
}

/// Choose the best `SupportedStreamConfig` for an input device.
///
/// Strategy: use the device's default sample rate (avoids forcing hardware
/// into a non-native rate which can fail on Bluetooth / ALSA) and prefer
/// F32 > I16 > I32 > others at that rate. Falls back to the device default
/// if no supported config matches the default rate.
pub(crate) fn preferred_config(
    device: &cpal::Device,
) -> Result<cpal::SupportedStreamConfig, Error> {
    let default_cfg = device.default_input_config()?;
    let target_rate = default_cfg.sample_rate();

    let supported = match device.supported_input_configs() {
        Ok(cfgs) => cfgs,
        Err(e) => {
            tracing::warn!(
                target = "audio-capture",
                "could not enumerate input configs for device, using default: {e}"
            );
            return Ok(default_cfg);
        }
    };

    let score = |fmt: cpal::SampleFormat| match fmt {
        cpal::SampleFormat::F32 => 4,
        cpal::SampleFormat::I16 => 3,
        cpal::SampleFormat::I32 => 2,
        _ => 1,
    };

    let mut best: Option<cpal::SupportedStreamConfigRange> = None;
    for range in supported {
        if range.min_sample_rate() <= target_rate && range.max_sample_rate() >= target_rate {
            let better = match &best {
                None => true,
                Some(cur) => score(range.sample_format()) > score(cur.sample_format()),
            };
            if better {
                best = Some(range);
            }
        }
    }

    match best {
        Some(r) => Ok(r.with_sample_rate(target_rate)),
        None => {
            tracing::warn!(
                target = "audio-capture",
                "no supported config matched device default rate {:?}, using device default",
                target_rate
            );
            Ok(default_cfg)
        }
    }
}
