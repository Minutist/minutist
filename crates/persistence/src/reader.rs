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
use meeting_app_common::{
    AppResult, MeetingMeta, MeetingState, NotesDocument, Segment,
};
use ogg::PacketReader;

use crate::error::Error;
use crate::notes::NotesStore;

/// Maximum number of mono samples a single Opus packet can decode to at any
/// supported frame size (120 ms at 48 kHz = 5760). 16 kHz frames are far
/// smaller, but the decoder needs a buffer sized for the worst case.
const MAX_FRAME_SAMPLES: usize = 5760;

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

/// Decode an Ogg/Opus byte buffer into a 16 kHz mono f32 PCM vector.
///
/// Skips the two header packets (`OpusHead`, `OpusTags`) and decodes every
/// audio packet, appending the decoded samples. The silent frames written for
/// pause gaps decode to genuine zero samples, so the returned buffer is
/// pause-including.
fn decode_opus_ogg(data: &[u8]) -> Result<Vec<f32>, String> {
    let cursor = std::io::Cursor::new(data);
    let mut reader = PacketReader::new(cursor);

    let mut decoder = Decoder::new(SampleRate::Hz16000, Channels::Mono)
        .map_err(|e| format!("decoder init: {e}"))?;

    let mut pcm = Vec::<f32>::new();
    let mut frame_buf = vec![0.0f32; MAX_FRAME_SAMPLES];

    // The first two packets are OpusHead and OpusTags — skip them.
    let mut header_packets_seen = 0;

    loop {
        let pkt = match reader.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(format!("ogg read error: {e}")),
        };

        if header_packets_seen < 2 {
            header_packets_seen += 1;
            continue;
        }

        let input = audiopus::packet::Packet::try_from(pkt.data.as_slice())
            .map_err(|e| e.to_string())?;
        let output =
            audiopus::MutSignals::try_from(frame_buf.as_mut_slice()).map_err(|e| e.to_string())?;

        let decoded = decoder
            .decode_float(Some(input), output, false)
            .map_err(|e| format!("decode error: {e}"))?;

        pcm.extend_from_slice(&frame_buf[..decoded]);
    }

    Ok(pcm)
}

#[cfg(test)]
pub(crate) fn decode_opus_ogg_for_test(data: &[u8]) -> Result<Vec<f32>, String> {
    decode_opus_ogg(data)
}
