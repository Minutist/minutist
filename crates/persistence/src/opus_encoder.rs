//! Opus/Ogg encoder for `audio.opus` files.
//!
//! Writes a compliant Opus-in-Ogg bitstream (RFC 7845) to any `std::io::Write`
//! target. Sample rate: 16 kHz mono f32. Bitrate: 32 kbps. Frame size: 20 ms
//! (320 samples).
//!
//! # Pause/resume gap mechanics — option (b), zero-sample padding
//!
//! A single Ogg logical bitstream is used throughout the recording. On pause,
//! the encoder records the wall-clock instant and flushes any pending partial
//! frame. On resume, the elapsed pause duration is converted to the nearest
//! multiple of `FRAME_SAMPLES` (20 ms / 320 samples at 16 kHz) and that many
//! zero-sample (silent) Opus frames are written to the stream. The resulting
//! `audio.opus` contains genuine silent audio for the paused interval, making
//! the decoded duration equal to (recorded_audio + pause_duration) within ±20 ms
//! (one frame).
//!
//! # Why not option (a) — granule jump
//!
//! The `ogg` crate's `PacketWriter::write_packet` accepts an `absgp` that sets
//! the Ogg page's granule position. Jumping the granule forward does encode the
//! gap in the container's timeline metadata, but the Opus decoder itself counts
//! *packets*, not granule positions — it decodes exactly the packets present in
//! the stream and does not interpolate silence for missing packets. Testing
//! confirmed that a granule-only jump produces the same decoded sample count as
//! a stream without the jump: the decoder sees no packets to decode for the gap
//! interval and therefore produces no silence. Option (b) is the correct choice
//! for accurate decoded duration.

use std::io::Write;
use std::time::Instant;

use audiopus::coder::{Encoder as OpusEncoder, GenericCtl};
use audiopus::{Application, Bitrate, Channels, SampleRate};
use ogg::{PacketWriteEndInfo, PacketWriter};

use crate::error::{Error, Result};

/// Samples per Opus frame at 16 kHz — 20 ms.
pub const FRAME_SAMPLES: usize = 320;
/// Sample rate mandated by the architecture (16 kHz mono).
pub const SAMPLE_RATE: u32 = 16_000;
/// Target bitrate in bps (32 kbps).
const BITRATE_BPS: i32 = 32_000;
/// Maximum encoded packet size in bytes (generous upper bound for 32 kbps).
const MAX_PACKET_BYTES: usize = 4000;

/// Internal state of the encoder regarding pause.
enum PauseState {
    /// Actively encoding audio.
    Recording,
    /// Paused; holds the instant the pause started.
    Paused { paused_at: Instant },
}

/// Ogg/Opus encoder that writes directly to a `Write` target.
pub struct OggOpusEncoder<W: Write> {
    encoder: OpusEncoder,
    writer: PacketWriter<'static, W>,
    /// Ogg logical bitstream serial number.
    serial: u32,
    /// Current granule position (cumulative decoded samples written).
    granule: u64,
    /// Samples accumulated but not yet forming a complete frame.
    pending: Vec<f32>,
    /// Pause state.
    pause_state: PauseState,
    /// Whether the Ogg/Opus headers have been written.
    headers_written: bool,
}

impl<W: Write> OggOpusEncoder<W> {
    /// Construct a new encoder. Headers are written lazily on the first call to
    /// `push_samples` (or `write_headers` explicitly).
    pub fn new(writer: W) -> Result<Self> {
        let mut encoder = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip)?;
        encoder.set_bitrate(Bitrate::BitsPerSecond(BITRATE_BPS))?;
        encoder.set_complexity(5)?;

        // Seed the serial number from the current time's subsec nanos xor'd
        // with a constant. `rand` is not a dep in this crate.
        let serial: u32 = {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0x5EED_1234);
            nanos ^ 0xDEAD_BEEF
        };

        Ok(Self {
            encoder,
            writer: PacketWriter::new(writer),
            serial,
            granule: 0,
            pending: Vec::new(),
            pause_state: PauseState::Recording,
            headers_written: false,
        })
    }

    /// Write the Ogg/Opus identification and comment headers.
    ///
    /// Must be called before any audio packets. Called automatically by
    /// `push_samples` if not called explicitly.
    pub fn write_headers(&mut self) -> Result<()> {
        if self.headers_written {
            return Ok(());
        }

        // --- OpusHead packet (RFC 7845 §5.1) ---
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(1); // channel count (mono)
                      // pre-skip: 3840 samples (80 ms) — the Opus encoder lookahead.
        head.extend_from_slice(&3840u16.to_le_bytes());
        // input sample rate (informational field; not the Opus internal rate).
        head.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        head.extend_from_slice(&0u16.to_le_bytes()); // output gain
        head.push(0); // channel mapping family (0 = mono/stereo RTP)

        self.writer
            .write_packet(
                head,
                self.serial,
                PacketWriteEndInfo::EndPage,
                /* absgp = */ 0,
            )
            .map_err(Error::Io)?;

        // --- OpusTags packet (RFC 7845 §5.2) ---
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        let vendor = b"meeting-app persistence v0";
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes()); // user comment list length = 0

        self.writer
            .write_packet(
                tags,
                self.serial,
                PacketWriteEndInfo::EndPage,
                /* absgp = */ 0,
            )
            .map_err(Error::Io)?;

        self.headers_written = true;
        Ok(())
    }

    /// Feed raw f32 PCM samples (16 kHz mono). Encodes full frames immediately;
    /// partial frames are buffered internally and flushed on `finalise`.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        self.write_headers()?;

        if let PauseState::Paused { .. } = self.pause_state {
            return Err(Error::InvalidState(
                "push_samples called while paused; call resume() first",
            ));
        }

        self.pending.extend_from_slice(samples);

        while self.pending.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = self.pending.drain(..FRAME_SAMPLES).collect();
            self.encode_frame(&frame)?;
        }

        Ok(())
    }

    /// Pause encoding. Records the wall-clock time so the gap can be computed
    /// on resume. Flushes any pending partial frame as silence so the stream
    /// boundary is clean.
    pub fn pause(&mut self) -> Result<()> {
        if let PauseState::Paused { .. } = self.pause_state {
            return Err(Error::InvalidState("pause called while already paused"));
        }

        // Flush pending partial frame, zero-padded to a full frame.
        if !self.pending.is_empty() {
            let mut padded = self.pending.drain(..).collect::<Vec<_>>();
            padded.resize(FRAME_SAMPLES, 0.0f32);
            self.encode_frame(&padded)?;
        }

        self.pause_state = PauseState::Paused {
            paused_at: Instant::now(),
        };

        tracing::debug!(
            target: "persistence",
            granule = self.granule,
            "encoder paused"
        );

        Ok(())
    }

    /// Resume after a pause.
    ///
    /// Encodes zero-sample (silent) Opus frames for the elapsed pause duration,
    /// producing genuine silence in the decoded output (option b — see module
    /// doc). The number of silent frames is rounded to the nearest 20 ms
    /// boundary, so the gap accuracy is ±20 ms (well within the ±50 ms budget).
    pub fn resume(&mut self) -> Result<()> {
        let paused_at = match self.pause_state {
            PauseState::Paused { paused_at } => paused_at,
            PauseState::Recording => {
                return Err(Error::InvalidState("resume called while not paused"))
            }
        };

        let elapsed = paused_at.elapsed();

        // Round pause duration to the nearest frame boundary.
        let pause_samples_f = elapsed.as_secs_f64() * SAMPLE_RATE as f64;
        let pause_frames = (pause_samples_f / FRAME_SAMPLES as f64).round() as u64;

        self.pause_state = PauseState::Recording;

        // Reset encoder state so the post-gap audio isn't contaminated by
        // pre-pause residue.
        self.encoder.reset_state()?;

        // Write silent frames to fill the pause interval.
        let silent = vec![0.0f32; FRAME_SAMPLES];
        for _ in 0..pause_frames {
            self.encode_frame(&silent)?;
        }

        tracing::debug!(
            target: "persistence",
            pause_ms = elapsed.as_millis(),
            pause_frames,
            new_granule = self.granule,
            "encoder resumed; silence frames written"
        );

        Ok(())
    }

    /// Flush remaining samples as a zero-padded frame and end the Ogg stream.
    pub fn finalise(mut self) -> Result<()> {
        self.write_headers()?;

        if !self.pending.is_empty() {
            let mut padded = self.pending.drain(..).collect::<Vec<_>>();
            padded.resize(FRAME_SAMPLES, 0.0f32);
            self.encode_frame_end(padded)?;
        } else {
            // Write a silent final frame so the stream is never empty.
            let silent = vec![0.0f32; FRAME_SAMPLES];
            self.encode_frame_end(silent)?;
        }

        Ok(())
    }

    /// Encode one full frame and append it to the Ogg stream (non-final).
    fn encode_frame(&mut self, frame: &[f32]) -> Result<()> {
        debug_assert_eq!(frame.len(), FRAME_SAMPLES);

        let mut buf = vec![0u8; MAX_PACKET_BYTES];
        let n = self.encoder.encode_float(frame, &mut buf)?;
        buf.truncate(n);

        self.granule += FRAME_SAMPLES as u64;

        self.writer
            .write_packet(
                buf,
                self.serial,
                PacketWriteEndInfo::NormalPacket,
                self.granule,
            )
            .map_err(Error::Io)?;

        Ok(())
    }

    /// Encode the final frame and write `EndStream`.
    fn encode_frame_end(&mut self, frame: Vec<f32>) -> Result<()> {
        debug_assert_eq!(frame.len(), FRAME_SAMPLES);

        let mut buf = vec![0u8; MAX_PACKET_BYTES];
        let n = self.encoder.encode_float(&frame, &mut buf)?;
        buf.truncate(n);

        self.granule += FRAME_SAMPLES as u64;

        self.writer
            .write_packet(
                buf,
                self.serial,
                PacketWriteEndInfo::EndStream,
                self.granule,
            )
            .map_err(Error::Io)?;

        Ok(())
    }

    /// The current granule position (total decoded samples, including silence
    /// frames written for pause gaps).
    pub fn granule(&self) -> u64 {
        self.granule
    }
}
