//! System/call (loopback) audio capture — the DEFAULT RENDER endpoint captured
//! in loopback mode, so a call's far-end participants are recorded alongside
//! the microphone.
//!
//! ## Loopback approach — cpal (no `wasapi` crate)
//!
//! cpal 0.16's WASAPI host already supports loopback **transparently**: building
//! an INPUT stream on a render (output) device sets
//! `AUDCLNT_STREAMFLAGS_LOOPBACK` automatically (see cpal
//! `src/host/wasapi/{mod.rs,device.rs}` — "If you use a WASAPI output device as
//! an input device it will transparently enable loopback mode"). So we open the
//! default OUTPUT device, negotiate its render config, and reuse the exact same
//! [`crate::manager::build_input_stream`] machinery as the mic path — sample
//! format dispatch, mono downmix, and the drop-oldest ring all come for free.
//! No extra `wasapi` dependency is needed.
//!
//! The captured frames are mono-downmixed f32 at the render device's native
//! rate; the manager's per-source resampler converts them to 16 kHz mono before
//! the mixer sums them with the mic.
//!
//! ## Non-Windows
//!
//! Loopback is Windows-only for now. On other platforms [`open_loopback`] is a
//! stub returning [`Error::LoopbackUnsupported`]; the manager logs a warning and
//! falls back to mic-only, never failing the recording. (Linux/macOS loopback
//! — PulseAudio/PipeWire monitor sources, or a virtual aggregate device on
//! macOS — is future work.)
//!
//! ## Echo / AEC (future work)
//!
//! When the microphone also picks the call audio up from the speakers, mixing
//! the loopback in doubles that audio (an echo). v1 handles this only with the
//! opt-in toggle (off by default); the user is advised to turn it off if their
//! mic hears the speakers. Acoustic echo cancellation using the loopback as the
//! reference signal is deliberately deferred — see
//! `architecture/cross-cutting.md` — "Threading model".

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::error::Error;
use crate::manager::DropOldestChannel;

/// A built (but not yet playing) loopback capture source.
///
/// Mirrors the mic side: the cpal stream, its (unused-but-symmetric) pause
/// flag, the raw ring channel the callback pushes into, and the render device's
/// native input rate the resampler must convert from.
pub(crate) struct LoopbackSource {
    pub(crate) stream: cpal::Stream,
    pub(crate) paused: Arc<AtomicBool>,
    pub(crate) raw_ch: Arc<DropOldestChannel>,
    pub(crate) in_rate: u32,
}

/// Open the default render endpoint in loopback mode.
///
/// Windows: builds an input stream on the default OUTPUT device (cpal sets the
/// WASAPI loopback flag automatically). Returns the not-yet-playing source; the
/// caller calls `stream.play()`.
///
/// Non-Windows: returns [`Error::LoopbackUnsupported`].
#[cfg(windows)]
pub(crate) fn open_loopback() -> Result<LoopbackSource, Error> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| Error::LoopbackUnsupported {
            context: "no default render (output) device for loopback".into(),
        })?;

    // The loopback stream uses the RENDER device's output config (a render
    // device has no input configs); cpal captures what the device is playing in
    // that format.
    let config = loopback_config(&device)?;
    let in_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    tracing::info!(
        target = "audio-capture",
        sample_rate = in_rate,
        channels,
        format = ?config.sample_format(),
        device = device.name().unwrap_or_default(),
        "opening render endpoint in loopback mode"
    );

    let raw_ch = Arc::new(DropOldestChannel::new(crate::manager::RAW_RING_CAPACITY));
    let paused = Arc::new(AtomicBool::new(false));

    // Reuse the mic path's stream builder: on a render device cpal's
    // `build_input_stream` enables loopback, downmixes to mono f32, and pushes
    // into the drop-oldest ring exactly as for the mic.
    let stream = crate::manager::build_input_stream(
        &device,
        &config,
        channels,
        in_rate,
        Arc::clone(&paused),
        Arc::clone(&raw_ch),
    )?;

    Ok(LoopbackSource {
        stream,
        paused,
        raw_ch,
        in_rate,
    })
}

#[cfg(not(windows))]
pub(crate) fn open_loopback() -> Result<LoopbackSource, Error> {
    Err(Error::LoopbackUnsupported {
        context: "loopback capture is implemented for Windows (WASAPI) only".into(),
    })
}

/// Choose a capture config for the render device's loopback stream.
///
/// Prefers the render device's default output format (the format the device is
/// actually mixing at, which loopback delivers); falls back to the first
/// supported output config when the default is unavailable.
#[cfg(windows)]
fn loopback_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig, Error> {
    use cpal::traits::DeviceTrait;

    match device.default_output_config() {
        Ok(cfg) => Ok(cfg),
        Err(e) => {
            tracing::warn!(
                target = "audio-capture",
                "no default output config for loopback ({e}); trying supported configs"
            );
            let mut supported = device
                .supported_output_configs()
                .map_err(|err| Error::LoopbackUnsupported {
                    context: format!("render device exposes no output configs: {err}"),
                })?;
            let range = supported.next().ok_or_else(|| Error::LoopbackUnsupported {
                context: "render device exposes no output configs".into(),
            })?;
            Ok(range.with_max_sample_rate())
        }
    }
}
