//! Readers for a meeting folder — the Phase-4 read surface.
//!
//! These are deliberately **synchronous** (blocking) `std::fs` reads. Callers
//! that run inside an async context drive them via
//! `tokio::task::spawn_blocking` (the orchestrator and `ipc-bridge` own that
//! decision — see `architecture/cross-cutting.md` "Threading model"). Keeping
//! the readers as plain sync fns means they have no runtime dependency and can
//! be unit-tested without a tokio reactor.
//!
//! All four readers take an explicit `meeting_dir` (the `{root}/{uuid}/`
//! path), so a caller that already holds a [`crate::MeetingFolder`] passes
//! `folder.path()` and a caller that resolved the path some other way (e.g.
//! the index, or `rebuild_from_disk`) passes it directly.
//!
//! # Graduated Opus decoder
//!
//! [`read_audio_pcm`] is the public form of the Opus decoder that previously
//! lived only in the crate's test code. It returns the **pause-INCLUDING**
//! 16 kHz mono f32 buffer — i.e. the silent frames written for pause gaps are
//! present in the output, so the decoded buffer's duration equals wall-clock
//! recording duration. This is the buffer Phase 6 diarization and Phase 4
//! re-transcribe consume; sourcing audio through this reader is why `diarizer`
//! need not depend on `persistence` (the orchestrator reads the PCM and hands
//! it to the `Diarizer` trait).

use std::path::Path;

use audiopus::coder::Decoder;
use audiopus::{Channels, SampleRate};
use minutist_common::{
    AppResult, MeetingMeta, MeetingState, NoteBlock, NotesDocument, Segment,
};
use ogg::PacketReader;

use crate::error::Error;
use crate::notes::{note_blocks_from_json, NotesStore};

/// Maximum number of mono samples a single Opus packet can decode to at any
/// supported frame size (120 ms at 48 kHz = 5760). 16 kHz frames are far
/// smaller, but the decoder needs a buffer sized for the worst case.
const MAX_FRAME_SAMPLES: usize = 5760;

/// The Opus codec's fixed internal sample rate. The `OpusHead` `pre_skip` field
/// is always expressed at this rate (RFC 7845 §5.1).
const OPUS_INTERNAL_RATE_HZ: u64 = 48_000;

/// The decoder's output sample rate (16 kHz mono — workspace standard). Used to
/// scale the 48 kHz `pre_skip` count into output samples.
const SAMPLE_RATE_OUT_HZ: u64 = 16_000;

/// Read and deserialise `metadata.json` from a meeting folder.
///
/// Blocking `std::fs` read; deserialises with `serde_json`.
pub fn read_metadata(meeting_dir: &Path) -> AppResult<MeetingMeta> {
    Ok(read_metadata_inner(meeting_dir)?)
}

/// `read_metadata` in this crate's own error namespace, for the index /
/// meeting-ops callers that already work in `crate::Error`.
pub(crate) fn read_metadata_inner(meeting_dir: &Path) -> Result<MeetingMeta, Error> {
    let path = meeting_dir.join("metadata.json");
    let bytes = std::fs::read(&path).map_err(Error::Io)?;
    let meta: MeetingMeta = serde_json::from_slice(&bytes).map_err(Error::Serialise)?;
    Ok(meta)
}

/// Read and deserialise `transcript.json` from a meeting folder.
///
/// Returns an empty `Vec` when `transcript.json` is absent — a zero-segment
/// meeting writes no transcript file (see [`crate::TranscriptWriter`]), so an
/// absent file is a legitimate empty transcript, not an error.
pub fn read_transcript(meeting_dir: &Path) -> AppResult<Vec<Segment>> {
    Ok(read_transcript_inner(meeting_dir)?)
}

/// `read_transcript` in this crate's own error namespace.
pub(crate) fn read_transcript_inner(meeting_dir: &Path) -> Result<Vec<Segment>, Error> {
    let path = meeting_dir.join("transcript.json");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::Io(e)),
    };
    let segments: Vec<Segment> = serde_json::from_slice(&bytes).map_err(Error::Serialise)?;
    Ok(segments)
}

/// Decode `audio.opus` from a meeting folder into a pause-INCLUDING 16 kHz
/// mono f32 PCM buffer.
///
/// The buffer contains the silent frames written for pause gaps, so its
/// duration equals wall-clock recording duration (the same timeline as the
/// raw `audio.opus` stream). Phase 6 diarization and Phase 4 re-transcribe
/// consume this buffer.
pub fn read_audio_pcm(meeting_dir: &Path) -> AppResult<Vec<f32>> {
    let path = meeting_dir.join("audio.opus");
    let data = std::fs::read(&path).map_err(Error::Io)?;
    decode_opus_ogg(&data).map_err(|e| Error::OpusDecode(e).into())
}

/// Assemble the full restorable [`MeetingState`] for a meeting: metadata +
/// transcript + optional notes.
///
/// Notes are loaded via [`NotesStore::load`] (taking `root` + `meeting_id`)
/// and mapped to the wire-facing [`NotesDocument`] — `notes_json` is the
/// opaque Tiptap document re-serialised to a string, `notes_markdown` the
/// `notes.md` body. A meeting with no saved notes yields `notes: None`.
///
/// `meeting_dir` must be `{root}/{uuid}/`; `read_meeting_state` derives the
/// `(root, meeting_id)` pair `NotesStore` expects from `meta.uuid` and the
/// folder's parent, so the same on-disk layout is honoured.
pub fn read_meeting_state(meeting_dir: &Path) -> AppResult<MeetingState> {
    let meta = read_metadata_inner(meeting_dir)?;
    let transcript = read_transcript_inner(meeting_dir)?;

    // `NotesStore::load` resolves `{root}/{uuid}/` itself, so recover the
    // root from the folder's parent and pass the meeting id from metadata.
    let root = meeting_dir.parent().ok_or(Error::InvalidState(
        "meeting_dir has no parent; cannot resolve notes root",
    ))?;

    let notes = NotesStore::load(root, meta.uuid)?.map(|data| {
        let notes_json = serde_json::to_string(&data.json)
            .unwrap_or_else(|_| "{}".to_string());
        NotesDocument {
            notes_json,
            notes_markdown: data.markdown,
        }
    });

    Ok(MeetingState {
        meta,
        transcript,
        notes,
    })
}

/// Load a meeting's note paragraphs as [`NoteBlock`]s (#70), or an empty vec
/// when the meeting has no notes.
///
/// Reads `notes.json` via [`NotesStore`] and projects it with
/// [`note_blocks_from_json`] — anchored paragraphs carry their `data-anchor-ms`
/// recording-clock timestamp, the rest carry `None`. Mirrors
/// [`read_meeting_state`]'s root derivation: the meeting folder's parent is the
/// notes root and the uuid comes from metadata. Reads the parsed value directly
/// (no markdown round-trip), so the summariser path does not re-parse the
/// re-serialised `notes_json` string.
pub fn read_note_blocks(meeting_dir: &Path) -> AppResult<Vec<NoteBlock>> {
    let meta = read_metadata_inner(meeting_dir)?;
    let root = meeting_dir.parent().ok_or(Error::InvalidState(
        "meeting_dir has no parent; cannot resolve notes root",
    ))?;
    let blocks = NotesStore::load(root, meta.uuid)?
        .map(|data| note_blocks_from_json(&data.json))
        .unwrap_or_default();
    Ok(blocks)
}

/// Decode an Ogg/Opus byte buffer into a 16 kHz mono f32 PCM vector.
///
/// Header packets (`OpusHead`, `OpusTags`) are identified by their RFC 7845
/// magic-byte prefixes and skipped regardless of position in the stream. This
/// handles single streams, double-header sets, and chained logical bitstreams
/// without passing a header packet to the audio decoder (which would return
/// `OPUS_INVALID_PACKET`).
///
/// # Pre-skip trimming
///
/// The encoder declares a `pre_skip` count in the `OpusHead` packet (RFC 7845
/// §5.1) — the number of leading decoded samples that are codec lookahead /
/// priming, not recorded audio. The Opus decoder emits these priming samples at
/// the head of the stream, so decoded sample 0 is **not** recorded sample 0
/// unless the pre-skip is trimmed. We read the actual `pre_skip` field from the
/// `OpusHead` packet (rather than assuming the encoder's nominal value) and drop
/// exactly that many leading decoded samples, so the returned buffer's sample 0
/// aligns with recorded sample 0. Without this trim every decoded buffer is
/// pre-skip-many samples offset and over-long, biasing the diarizer overlay and
/// the offline re-transcribe timeline.
fn decode_opus_ogg(data: &[u8]) -> Result<Vec<f32>, String> {
    let cursor = std::io::Cursor::new(data);
    let mut reader = PacketReader::new(cursor);

    let mut decoder = Decoder::new(SampleRate::Hz16000, Channels::Mono)
        .map_err(|e| format!("decoder init: {e}"))?;

    let mut pcm = Vec::<f32>::new();
    let mut frame_buf = vec![0.0f32; MAX_FRAME_SAMPLES];

    // Pre-skip samples (16 kHz) still to be trimmed off the head of the audio.
    let mut pre_skip_remaining: u64 = 0;

    loop {
        let pkt = match reader.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(format!("ogg read error: {e}")),
        };

        // Identify header packets by magic-byte prefix (RFC 7845 §5.1 /§5.2).
        // Skipping by count (formerly "skip first 2 packets") is fragile for
        // chained streams and double-header sets; magic-byte identification is
        // correct for any conformant Ogg/Opus stream.
        if pkt.data.starts_with(b"OpusHead") {
            pre_skip_remaining = parse_opus_head_pre_skip(&pkt.data);
            continue;
        }
        if pkt.data.starts_with(b"OpusTags") {
            continue;
        }

        let input = audiopus::packet::Packet::try_from(pkt.data.as_slice())
            .map_err(|e| e.to_string())?;
        let output =
            audiopus::MutSignals::try_from(frame_buf.as_mut_slice()).map_err(|e| e.to_string())?;

        let decoded = decoder
            .decode_float(Some(input), output, false)
            .map_err(|e| format!("decode error: {e}"))?;

        let mut frame = &frame_buf[..decoded];
        if pre_skip_remaining > 0 {
            let trim = (pre_skip_remaining as usize).min(frame.len());
            pre_skip_remaining -= trim as u64;
            frame = &frame[trim..];
        }
        pcm.extend_from_slice(frame);
    }

    Ok(pcm)
}

/// Parse the `pre_skip` field from an `OpusHead` packet and convert it from the
/// Opus internal 48 kHz rate to the decoder's 16 kHz output rate.
///
/// RFC 7845 §5.1 layout: 8-byte magic `"OpusHead"`, 1-byte version, 1-byte
/// channel count, then a little-endian `u16` `pre_skip` at byte offset 10. A
/// malformed/short packet yields a `0` pre-skip (no trim) rather than failing
/// the decode — `audio.opus` is the source of truth and a header we cannot
/// parse should degrade to the prior behaviour, not error.
fn parse_opus_head_pre_skip(head: &[u8]) -> u64 {
    if head.len() < 12 || &head[..8] != b"OpusHead" {
        return 0;
    }
    let pre_skip_48k = u16::from_le_bytes([head[10], head[11]]) as u64;
    // The OpusHead pre_skip is always at the 48 kHz internal sample rate
    // regardless of the decode rate (RFC 7845 §5.1). The decoder here outputs
    // 16 kHz, so scale down by 48000/16000 = 3.
    pre_skip_48k * SAMPLE_RATE_OUT_HZ / OPUS_INTERNAL_RATE_HZ
}

#[cfg(test)]
pub(crate) fn decode_opus_ogg_for_test(data: &[u8]) -> Result<Vec<f32>, String> {
    decode_opus_ogg(data)
}

#[cfg(test)]
pub(crate) fn parse_opus_head_pre_skip_for_test(head: &[u8]) -> u64 {
    parse_opus_head_pre_skip(head)
}
