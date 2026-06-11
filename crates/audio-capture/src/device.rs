//! Audio device enumeration.
//!
//! Returns `common::AudioDevice` values; does not expose cpal types to callers.

use cpal::traits::{DeviceTrait, HostTrait};
use minutist_common::AudioDevice;

use crate::error::Error;

/// Separator between the enumeration index and the device name inside a
/// composite [`AudioDevice::id`].
const ID_SEP: char = '\u{1f}'; // ASCII unit separator — cannot appear in names.

/// Build the stable composite id for the `index`-th input device named `name`.
///
/// cpal exposes no stable hardware id, and on Linux/ALSA several devices can
/// share the same `name`. The enumeration index disambiguates duplicates and
/// is stable for the lifetime of a host's device list (a single session's
/// `list_devices` → `open` round-trip), which is all the IPC contract needs.
/// The name is retained so the id stays human-debuggable and so a caller can
/// still match by name if the index drifts across re-enumeration.
fn make_id(index: usize, name: &str) -> String {
    format!("{index}{ID_SEP}{name}")
}

/// Split a composite id back into `(index, name)`.
///
/// Returns `None` for ids that predate the composite format (no separator);
/// callers then fall back to matching on the bare name.
fn parse_id(id: &str) -> Option<(usize, &str)> {
    let (idx, name) = id.split_once(ID_SEP)?;
    let idx = idx.parse::<usize>().ok()?;
    Some((idx, name))
}

/// Enumerate all available audio input devices on the default cpal host.
///
/// [`AudioDevice::id`] is a composite of the device's enumeration index and
/// its cpal name (see [`make_id`]). The index makes the id unique even when
/// two ALSA devices share a name. `is_default` is matched on the same
/// composite id as the host default rather than on the name alone, so a
/// duplicate-named non-default device is never mis-flagged as default.
pub(crate) fn list_input_devices() -> Result<Vec<AudioDevice>, Error> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    let devices = host.input_devices()?;
    let mut out = Vec::new();
    let mut default_seen = false;

    for (index, device) in devices.enumerate() {
        let name = device.name().unwrap_or_else(|_| "Unknown".into());
        // Mark only the first device whose name matches the host default as the
        // default; later same-named devices are distinct entries and must not
        // also be flagged.
        let is_default =
            !default_seen && Some(name.as_str()) == default_name.as_deref();
        if is_default {
            default_seen = true;
        }
        out.push(AudioDevice {
            id: make_id(index, &name),
            name,
            is_default,
        });
    }

    Ok(out)
}

/// Resolve an optional device id to a cpal `Device`.
///
/// `None` → OS default input device.
/// `Some(id)` → the device selected by the composite id. When the id carries
/// an enumeration index (the current format), that index is authoritative and
/// the name is used only as a consistency check; this disambiguates devices
/// that share a name. Ids without an index (legacy/bare-name) fall back to
/// first-name-match for backward compatibility.
pub(crate) fn resolve_device(device_id: Option<&str>) -> Result<cpal::Device, Error> {
    let host = cpal::default_host();

    let id = match device_id {
        None => return host.default_input_device().ok_or(Error::NoInputDevice),
        Some(id) => id,
    };

    match parse_id(id) {
        Some((index, name)) => {
            let mut devices = host.input_devices()?;
            // Prefer the device at the recorded enumeration index; verify the
            // name still matches to guard against the list having shifted.
            if let Some(device) = devices.nth(index) {
                if device.name().ok().as_deref() == Some(name) {
                    return Ok(device);
                }
            }
            // Index drifted (device list changed); fall back to first name match.
            resolve_by_name(&host, name)
        }
        // Legacy bare-name id: match the first device with that name.
        None => resolve_by_name(&host, id),
    }
}

/// First input device whose cpal name equals `name`.
fn resolve_by_name(host: &cpal::Host, name: &str) -> Result<cpal::Device, Error> {
    let devices = host.input_devices()?;
    for device in devices {
        if device.name().ok().as_deref() == Some(name) {
            return Ok(device);
        }
    }
    Err(Error::DeviceNotFound {
        id: name.to_owned(),
    })
}

/// Build the [`AudioDevice`] describing the device that `device_id` resolves
/// to, including a correct `is_default` flag.
///
/// Shares the composite-id and default-matching logic with
/// [`list_input_devices`] so `open()` reports the same `id`/`is_default` the
/// device picker showed.
pub(crate) fn describe_device(device_id: Option<&str>) -> Result<AudioDevice, Error> {
    let resolved = resolve_device(device_id)?;
    let resolved_name = resolved.name().unwrap_or_else(|_| "Unknown".into());

    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    // Recover the enumeration index for the resolved device so the reported id
    // matches what `list_input_devices` would emit. Match by name and, when an
    // index is available from the supplied id, prefer that exact slot.
    let wanted_index = device_id.and_then(parse_id).map(|(i, _)| i);

    let mut chosen_index = 0usize;
    let mut default_seen = false;
    let mut is_default = false;
    let mut matched = false;

    for (index, device) in host.input_devices()?.enumerate() {
        let name = device.name().unwrap_or_else(|_| "Unknown".into());
        let this_default =
            !default_seen && Some(name.as_str()) == default_name.as_deref();
        if this_default {
            default_seen = true;
        }

        let name_matches = name == resolved_name;
        let index_matches = wanted_index == Some(index);
        // Lock onto the exact slot when we know the index; otherwise the first
        // name match wins (mirrors `resolve_by_name`).
        if (index_matches && name_matches) || (wanted_index.is_none() && name_matches && !matched)
        {
            chosen_index = index;
            is_default = this_default;
            matched = true;
        }
    }

    Ok(AudioDevice {
        id: make_id(chosen_index, &resolved_name),
        name: resolved_name,
        is_default,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_id_round_trips() {
        let id = make_id(3, "USB Microphone");
        let (index, name) = parse_id(&id).expect("composite id must parse");
        assert_eq!(index, 3);
        assert_eq!(name, "USB Microphone");
    }

    #[test]
    fn duplicate_names_get_distinct_ids() {
        // Two ALSA devices commonly share a name; the index must keep ids unique.
        let a = make_id(0, "hw:CARD=PCH");
        let b = make_id(1, "hw:CARD=PCH");
        assert_ne!(a, b, "same-named devices must not share an id");
    }

    #[test]
    fn legacy_bare_name_id_parses_as_none() {
        // Pre-composite ids have no separator and must route to name matching.
        assert!(parse_id("Built-in Microphone").is_none());
    }

    #[test]
    fn id_separator_is_not_a_name_character() {
        // The unit-separator cannot appear in any real device name, so the
        // split point is unambiguous even for names containing other symbols.
        assert!(!"hw:CARD=PCH,DEV=0".contains(ID_SEP));
    }
}
