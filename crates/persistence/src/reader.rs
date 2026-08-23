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
//! path), so a caller that already holds a `notes_crdt::MeetingFolder` passes
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
use notes_crdt::{note_blocks_from_json, NotesStore};
use ogg::PacketReader;

use crate::error::Error;

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
/// meeting-ops callers that already work in `crate::Error`. Delegates to the
/// lifted [`notes_crdt::read_metadata`] (the deserialise now lives in the leaf so
/// the guarded RMW and the mobile path share it), mapping its error into
/// `crate::Error` via the existing `From<notes_crdt::Error>`.
pub(crate) fn read_metadata_inner(meeting_dir: &Path) -> Result<MeetingMeta, Error> {
    Ok(notes_crdt::read_metadata(meeting_dir)?)
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

/// Decode a meeting's audio file into a 16 kHz mono f32 PCM buffer.
///
/// Resolves the file via [`minutist_common::resolve_audio_path`] (0047/0048)
/// and dispatches on its extension rather than a hardcoded name or a
/// self-reported codec label: `.opus` decodes via the Ogg/Opus path below,
/// `.m4a` via [`decode_aac_m4a`]. The desktop's own recordings are always
/// `.opus` and their decode is pause-INCLUDING (the encoder pads every pause
/// gap with silence, so the buffer's duration equals wall-clock recording
/// duration — Phase 6 diarization and Phase 4 re-transcribe depend on this).
/// A synced phone `.m4a` recording carries no such guarantee: the phone's
/// recorder may simply stop encoding across a pause rather than padding it,
/// so an `.m4a` meeting's decoded duration may be shorter than wall-clock.
/// That is a known limitation, not a bug in this function — getting phone
/// recordings transcribable at all is 0047's scope; pause-timeline parity
/// for phone recordings is a separate, unscoped follow-up if it matters in
/// practice.
///
/// **Legacy fallback (issue 0051).** A recording made before phoneapp's
/// honest-format fix stored AAC bytes under the literal name `audio.opus`
/// (the mislabelling 0047 fixes on the write side going forward) — that
/// backlog already exists on disk and synced across devices, so it needs
/// rescuing without a filename migration (which has cross-device/sync-manifest
/// implications this function has no business triggering). If the resolved
/// `.opus` file fails Ogg/Opus decode, this retries it as AAC before giving
/// up. This does not weaken extension-first dispatch as the primary signal —
/// it only engages after the primary decoder has already failed, and
/// [`decode_aac_m4a`] does its own container probe, so genuinely-corrupt
/// data fails cleanly on both attempts rather than being masked.
pub fn read_audio_pcm(meeting_dir: &Path) -> AppResult<Vec<f32>> {
    let not_found = || {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no resolvable audio file in {}", meeting_dir.display()),
        ))
    };
    let path = minutist_common::resolve_audio_path(meeting_dir).ok_or_else(not_found)?;
    let data = std::fs::read(&path).map_err(Error::Io)?;
    // `Error::Io` here (not `Error::InvalidState`/`AppError::InvalidInput`) is
    // deliberate: `ipc-bridge::run_post_stop_passes` matches `InvalidInput` as
    // "recorder busy, try again later" and downgrades the failure to an info
    // log. A missing or unresolvable audio file is a real failure that must
    // stay at `AppError::Io`'s severity, matching what a literal missing
    // `audio.opus` produced before this function resolved the path itself.
    match path.extension().and_then(|e| e.to_str()) {
        Some("opus") => match decode_opus_ogg(&data) {
            Ok(pcm) => Ok(pcm),
            // Legacy fallback (issue 0051): recordings made before phoneapp's
            // honest-format fix stored AAC bytes under the literal name
            // `audio.opus` (the bug 0047 exists to fix, on the write side).
            // Extension-first dispatch is still correct going forward — this
            // is a one-way "did the primary decoder actually fail" recovery,
            // not a case of trusting a self-reported label over the
            // extension. `decode_aac_m4a` does its own container probe, so
            // it fails cleanly (a distinct error) on data that is genuinely
            // not MP4 either — this never masks a real corrupt-Opus file as
            // success.
            Err(opus_err) => match decode_aac_m4a(data) {
                Ok(pcm) => {
                    tracing::warn!(
                        target: "persistence",
                        path = %path.display(),
                        opus_error = %opus_err,
                        "audio.opus failed Ogg/Opus decode but succeeded as AAC — a pre-0047 mislabelled phone recording (issue 0051); its pause-timeline parity with wall-clock is not guaranteed (issue 0050)"
                    );
                    Ok(pcm)
                }
                Err(aac_err) => Err(Error::AudioDecode(format!(
                    "opus decode failed ({opus_err}); AAC fallback also failed ({aac_err})"
                ))
                .into()),
            },
        },
        Some("m4a") => {
            // A phone-recorded meeting's decoded duration has no guaranteed
            // relationship to wall-clock (see this fn's doc comment) — flag it
            // so a diarization/re-listen/voiceprint quality report for a
            // phone meeting can be traced back to this, not treated as
            // unexplained.
            tracing::warn!(
                target: "persistence",
                path = %path.display(),
                "decoding a phone-recorded .m4a meeting — pause-timeline parity with wall-clock is not guaranteed (0047 follow-up: issue 0050)"
            );
            decode_aac_m4a(data).map_err(|e| Error::AudioDecode(e).into())
        }
        _ => Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported audio file extension: {}", path.display()),
        ))
        .into()),
    }
}

/// Assemble the full restorable [`MeetingState`] for a meeting: metadata +
/// transcript + optional notes.
///
/// Notes are loaded via [`NotesStore::load`] (taking `root` + `meeting_id`)
/// and mapped to the wire-facing [`NotesDocument`] — `notes_json` is the
/// opaque Tiptap document re-serialised to a string, `notes_markdown` the
/// `notes.md` body. A meeting with no saved notes yields `notes: None`.
///
/// **Lazy notes-CRDT seed (D-O2.7).** Opening a meeting is also the migration
/// trigger: when `notes.ydoc` is absent but `notes.json` exists, this seeds
/// `notes.ydoc` from the JSON and flips `MeetingMeta::notes_format` to `1`
/// (rewriting `metadata.json`). It is idempotent (a no-op once seeded),
/// build-invariant, and per-meeting — a never-opened meeting is never touched.
/// After seeding, `notes.ydoc` is authoritative and the returned notes are
/// derived from it.
///
/// `meeting_dir` must be `{root}/{uuid}/`; `read_meeting_state` derives the
/// `(root, meeting_id)` pair `NotesStore` expects from `meta.uuid` and the
/// folder's parent, so the same on-disk layout is honoured.
pub fn read_meeting_state(meeting_dir: &Path) -> AppResult<MeetingState> {
    let mut meta = read_metadata_inner(meeting_dir)?;
    let transcript = read_transcript_inner(meeting_dir)?;

    // `NotesStore::load` resolves `{root}/{uuid}/` itself, so recover the
    // root from the folder's parent and pass the meeting id from metadata.
    let root = meeting_dir.parent().ok_or(Error::InvalidState(
        "meeting_dir has no parent; cannot resolve notes root",
    ))?;

    // Lazy one-time migration: seed notes.ydoc from notes.json on first open,
    // then record notes_format = 1 in metadata.json. The seed is idempotent;
    // the metadata flip also self-corrects a meeting whose notes.ydoc was
    // written (e.g. by a prior save) while notes_format still read 0.
    let seeded = NotesStore::seed_ydoc_if_needed(root, meta.uuid)?;
    if (seeded || meeting_dir.join("notes.ydoc").exists()) && meta.notes_format == 0 {
        // Route the flip through the guarded primitive so it takes the
        // per-meeting metadata lock (issue 0025). This is a full-struct RMW, and
        // it fires on exactly the synced meetings that receive `Claimed` /
        // `Processed` over the lifecycle stream — so an unguarded write here
        // would read `processing` before a concurrent lifecycle write and then
        // revert it. `update_metadata` re-reads under the lock, so the returned
        // `meta` reflects any such concurrent write rather than clobbering it.
        meta = crate::meeting_ops::update_metadata(root, meta.uuid, |m| {
            m.notes_format = 1;
            m.clone()
        })?;
    }

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
    let mut decoded_any_packet = false;

    // Pre-skip samples (16 kHz) still to be trimmed off the head of the audio.
    let mut pre_skip_remaining: u64 = 0;

    loop {
        let pkt = match reader.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // A recording killed mid-write (crash, force-quit) never gets its
            // final EOS-flagged page, which trips the `ogg` crate's own I/O
            // error here even though everything up to this point decoded fine
            // (ffmpeg's demuxer tolerates exactly this). Once real audio has
            // already decoded, treat this as an abrupt-but-recoverable end of
            // stream rather than discarding the whole recording; before any
            // audio has decoded this is a genuinely unreadable file and stays
            // fatal.
            Err(e) if decoded_any_packet => {
                tracing::warn!(
                    target: "persistence",
                    error = %e,
                    "ogg stream ended without a final EOS page (recording likely crashed mid-write); keeping the audio decoded so far"
                );
                break;
            }
            // Also keeps a mislabelled AAC-under-`.opus` file (issue 0051)
            // failing at packet 0, so `read_audio_pcm`'s AAC fallback stays
            // reachable — a broader tolerance here would mask that case as a
            // zero-length "successful" Opus decode instead.
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
        decoded_any_packet = true;

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

/// Input block size fed to rubato's FFT resampler in one call, matching
/// `audio-capture::resample`'s tested value. This is an independent one-shot
/// (not streaming) use — the two crates decode from different starting
/// points (a live device stream vs. a whole file already in memory) — so the
/// resample glue is duplicated here rather than sharing an instance.
const RESAMPLE_CHUNK_IN: usize = 1024;

/// A native sample rate outside this range cannot be a real recording (typical
/// audio is 8 kHz–192 kHz); reject it before it reaches rubato, whose
/// `FftFixedIn` sizes an internal FFT buffer from the input/output rate ratio
/// — an untrusted, corrupted, or bit-flipped rate field could otherwise
/// demand a multi-gigabyte allocation and abort the process instead of
/// failing the decode.
const MAX_PLAUSIBLE_SAMPLE_RATE_HZ: u32 = 192_000;
const MIN_PLAUSIBLE_SAMPLE_RATE_HZ: u32 = 1_000;

/// Decode an AAC-in-MP4 (`.m4a`) byte buffer into a 16 kHz mono f32 PCM
/// vector: probe the container, decode every packet on the first audio
/// track, downmix to mono, then resample from the track's native rate to
/// 16 kHz if it differs.
///
/// Unlike [`decode_opus_ogg`], there is no pre-skip/pause-padding contract
/// here — see [`read_audio_pcm`]'s doc comment for what that means for a
/// phone-recorded meeting's decoded duration. There is also no encoder
/// priming-delay trim (the AAC equivalent of Opus's `pre_skip`): symphonia
/// 0.6's `isomp4`/`aac` combination does not implement gapless trimming (its
/// own docs list both as `Gapless: No`), so every decoded `.m4a` carries a
/// small (~1024–2112 sample, i.e. tens of ms) leading offset versus the
/// source recording. Tracked alongside the pause-timeline gap (issue 0050).
fn decode_aac_m4a(data: Vec<u8>) -> Result<Vec<f32>, String> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let cursor = std::io::Cursor::new(data);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("m4a");

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| format!("probing container: {e}"))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| "no audio track in container".to_string())?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| "track has no audio codec parameters".to_string())?
        .clone();
    let native_rate = audio_params
        .sample_rate
        .ok_or_else(|| "track has no sample rate".to_string())?;
    if !(MIN_PLAUSIBLE_SAMPLE_RATE_HZ..=MAX_PLAUSIBLE_SAMPLE_RATE_HZ).contains(&native_rate) {
        return Err(format!(
            "implausible sample rate {native_rate} Hz (expected {MIN_PLAUSIBLE_SAMPLE_RATE_HZ}-{MAX_PLAUSIBLE_SAMPLE_RATE_HZ})"
        ));
    }

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
        .map_err(|e| format!("decoder init: {e}"))?;

    let mut mono = Vec::<f32>::new();
    let mut interleaved = Vec::<f32>::new();
    let mut packets_seen = 0u32;
    let mut packets_failed = 0u32;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(format!("reading packet: {e}")),
        };
        if packet.track_id != track_id {
            continue;
        }
        packets_seen += 1;
        let audio_buf = match decoder.decode(&packet) {
            Ok(buf) => buf,
            // A single malformed packet does not invalidate the whole file —
            // skip it and keep decoding, mirroring symphonia's own examples —
            // but count it: if EVERY packet fails this way, that's a corrupt
            // file, not a legitimately silent/empty one (below).
            Err(SymphoniaError::DecodeError(_)) => {
                packets_failed += 1;
                continue;
            }
            Err(e) => return Err(format!("decoding packet: {e}")),
        };
        let channels = audio_buf.spec().channels().count();
        let n = audio_buf.samples_interleaved();
        if channels == 0 || n == 0 {
            packets_failed += 1;
            continue;
        }
        interleaved.resize(n, 0.0f32);
        audio_buf.copy_to_slice_interleaved(&mut interleaved);
        for frame in interleaved.chunks(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if packets_seen > 0 && packets_failed == packets_seen {
        return Err(format!(
            "every packet on the audio track failed to decode ({packets_failed}/{packets_seen}) — container probed but the encoded audio is corrupt"
        ));
    }
    if packets_failed > 0 {
        tracing::warn!(
            target: "persistence",
            packets_failed,
            packets_seen,
            "some AAC packets failed to decode and were skipped"
        );
    }

    if native_rate == SAMPLE_RATE_OUT_HZ as u32 {
        Ok(mono)
    } else {
        resample_mono_to_16k(&mono, native_rate)
    }
}

/// Resample a whole mono f32 buffer from `native_rate` to 16 kHz via rubato's
/// `FftFixedIn`, processing in fixed-size chunks (the resampler requires a
/// consistent input block size). The final partial chunk is zero-padded to
/// feed the resampler, then the corresponding tail of the OUTPUT — the
/// portion attributable to that padding, computed from the same rate ratio —
/// is trimmed, so the padding never surfaces as fabricated trailing silence
/// or FFT edge-effect ringing in the returned buffer.
fn resample_mono_to_16k(input: &[f32], native_rate: u32) -> Result<Vec<f32>, String> {
    use rubato::{FftFixedIn, Resampler};

    let mut resampler = FftFixedIn::<f32>::new(
        native_rate as usize,
        SAMPLE_RATE_OUT_HZ as usize,
        RESAMPLE_CHUNK_IN,
        1,
        1,
    )
    .map_err(|e| format!("resampler init: {e}"))?;

    let mut out = Vec::with_capacity(
        (input.len() as u64 * SAMPLE_RATE_OUT_HZ / native_rate.max(1) as u64) as usize,
    );
    for chunk in input.chunks(RESAMPLE_CHUNK_IN) {
        let block = if chunk.len() == RESAMPLE_CHUNK_IN {
            chunk.to_vec()
        } else {
            // Final partial chunk: zero-pad to the resampler's fixed input
            // size; the padded portion's corresponding output tail is
            // trimmed below.
            let mut padded = chunk.to_vec();
            padded.resize(RESAMPLE_CHUNK_IN, 0.0);
            padded
        };
        let result = resampler
            .process(&[&block[..]], None)
            .map_err(|e| format!("resampling: {e}"))?;
        out.extend_from_slice(&result[0]);
    }

    // Exact expected output length for `input.len()` real samples at this
    // rate ratio; anything beyond it in `out` is an artefact of the last
    // chunk's zero-padding.
    let expected_len =
        (input.len() as u64 * SAMPLE_RATE_OUT_HZ as u64).div_ceil(native_rate.max(1) as u64)
            as usize;
    out.truncate(expected_len.min(out.len()));
    Ok(out)
}

#[cfg(test)]
pub(crate) fn decode_opus_ogg_for_test(data: &[u8]) -> Result<Vec<f32>, String> {
    decode_opus_ogg(data)
}

#[cfg(test)]
pub(crate) fn decode_aac_m4a_for_test(data: &[u8]) -> Result<Vec<f32>, String> {
    decode_aac_m4a(data.to_vec())
}

#[cfg(test)]
pub(crate) fn parse_opus_head_pre_skip_for_test(head: &[u8]) -> u64 {
    parse_opus_head_pre_skip(head)
}
