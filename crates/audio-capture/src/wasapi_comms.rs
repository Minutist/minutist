//! Windows-only microphone capture via WASAPI's **Communications** stream
//! category — the documented way to get the OS/driver voice pipeline
//! (array beamforming + acoustic echo cancellation + noise suppression,
//! downmixed to mono) instead of the raw multi-channel array `cpal` hands us.
//!
//! Why this exists (research artefact
//! `planning/research/windows-mic-array-capture-2026-06.md`): `cpal` opens the
//! default *console* endpoint in default mode and never sets a stream category,
//! so on a multi-mic laptop array it returns raw N-channel audio that we were
//! (wrongly) averaging — comb-filtering the phase-offset elements. The platform
//! must do the array processing, not us. The lever is
//! `IAudioClient2::SetClientProperties({ eCategory = AudioCategory_Communications })`
//! set **before** `Initialize`; the driver then runs its Communications-mode
//! effects and surfaces a processed mono stream.
//!
//! We request **16 kHz mono f32 with `autoconvert`**, so the audio engine both
//! runs the comms processing AND downmixes/resamples to our target rate — we
//! receive clean mono at 16 kHz with no manual downmix or resampling. If the
//! device has no Communications-mode processing the category is a no-op and the
//! engine still autoconverts to mono 16 kHz (just without AEC/NS), so the output
//! is always usable. Any failure here returns `None` and the caller falls back
//! to the `cpal` path.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use wasapi::{
    AudioClientProperties, Direction, DeviceEnumerator, SampleType, StreamCategory, StreamMode,
    WaveFormat,
};

use crate::manager::{DropOldestChannel, RawFrame};

/// Output rate of this path (matches the workspace target — the engine
/// autoconverts to it, so the downstream resampler is a pass-through).
pub(crate) const COMMS_RATE: u32 = 16_000;

/// Client buffer duration (hns = 100 ns units); 30 ms is a safe shared-mode
/// buffer well above the engine period. Event-driven, so callbacks still fire
/// at the engine period regardless.
const BUFFER_DURATION_HNS: i64 = 300_000;

/// Try to start communications-mode mic capture. On success, spawns a capture
/// thread that pushes processed 16 kHz mono `RawFrame`s into `raw_ch` until
/// `stopped` is set, and returns `(join_handle, COMMS_RATE)`. Returns `None` if
/// WASAPI initialisation fails (caller falls back to cpal).
///
/// `paused` is honoured by dropping captured frames while set (the stream keeps
/// running), matching the cpal callback's behaviour.
pub(crate) fn try_start(
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    raw_ch: Arc<DropOldestChannel>,
) -> Option<(std::thread::JoinHandle<()>, u32)> {
    // The capture client is apartment-bound, so init + the loop run on the same
    // thread. The thread signals init success/failure back so the caller can
    // fall back to cpal synchronously before the recording proceeds.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<bool>();
    let handle = std::thread::Builder::new()
        .name("wasapi-comms-capture".into())
        .spawn(move || run(paused, stopped, raw_ch, ready_tx))
        .ok()?;

    match ready_rx.recv_timeout(Duration::from_secs(3)) {
        Ok(true) => Some((handle, COMMS_RATE)),
        // Init failed (thread already returned) or timed out → fall back to cpal.
        _ => None,
    }
}

fn run(
    paused: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    raw_ch: Arc<DropOldestChannel>,
    ready_tx: std::sync::mpsc::Sender<bool>,
) {
    if wasapi::initialize_mta().is_err() {
        let _ = ready_tx.send(false);
        return;
    }

    let init = (|| -> Result<
        (wasapi::AudioClient, wasapi::AudioCaptureClient, wasapi::Handle),
        Box<dyn std::error::Error>,
    > {
        let enumerator = DeviceEnumerator::new()?;
        let device = enumerator.get_default_device(&Direction::Capture)?;
        let mut audio_client = device.get_iaudioclient()?;

        // Engage the OS voice pipeline BEFORE Initialize (must precede it).
        let props = AudioClientProperties::new().set_category(StreamCategory::Communications);
        audio_client.set_properties(props)?;

        // Force processed mono @16 kHz f32 via autoconvert.
        let format = WaveFormat::new(32, 32, &SampleType::Float, COMMS_RATE as usize, 1, None);
        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns: BUFFER_DURATION_HNS,
        };
        audio_client.initialize_client(&format, &Direction::Capture, &mode)?;

        let h_event = audio_client.set_get_eventhandle()?;
        let capture_client = audio_client.get_audiocaptureclient()?;
        audio_client.start_stream()?;
        Ok((audio_client, capture_client, h_event))
    })();

    let (audio_client, capture_client, h_event) = match init {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                target = "audio-capture",
                "WASAPI communications-mode mic capture unavailable ({e}); falling back to cpal"
            );
            let _ = ready_tx.send(false);
            return;
        }
    };
    tracing::info!(
        target = "audio-capture",
        "mic capture via WASAPI communications mode (OS beamforming/AEC/NS → 16 kHz mono)"
    );
    let _ = ready_tx.send(true);

    let mut deque: VecDeque<u8> = VecDeque::new();
    while !stopped.load(Ordering::Relaxed) {
        // Wake on the engine event; the timeout bounds how often we re-check
        // `stopped` if no audio arrives.
        if h_event.wait_for_event(100).is_err() {
            continue;
        }
        if capture_client.read_from_device_to_deque(&mut deque).is_err() {
            tracing::warn!(target = "audio-capture", "WASAPI capture read failed; stopping path");
            break;
        }
        // The stream is mono f32, so consume whole 4-byte samples and leave any
        // trailing partial sample in the deque to complete on the next read.
        let usable = deque.len() - deque.len() % 4;
        if usable == 0 {
            continue;
        }
        // While paused the stream keeps running (matching the cpal callback);
        // drain and discard so the deque can't grow unbounded.
        if paused.load(Ordering::Relaxed) {
            deque.drain(..usable);
            continue;
        }
        let bytes = deque.make_contiguous();
        let samples: Vec<f32> = bytes[..usable]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        deque.drain(..usable);
        raw_ch.push(RawFrame { samples });
    }

    let _ = audio_client.stop_stream();
}
