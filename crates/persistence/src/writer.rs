use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use minutist_common::{AppResult, AudioFormat, MeetingId, MeetingMeta, Segment};

use crate::error::Error;
use crate::folder::MeetingFolder;
use crate::metadata::write_metadata_to_path;
use crate::opus_encoder::OggOpusEncoder;
use crate::transcript::TranscriptWriter;

/// Writes audio samples and metadata for a single meeting recording.
///
/// `MeetingWriter` is not `Sync` — it is intended to run on the
/// orchestrator's capture-drain task (`spawn_blocking`).
///
/// # Lifecycle
///
/// ```text
/// open() → push_samples()* → [pause() → resume() → push_samples()*]* → finalise()
/// ```
///
/// # Transcript
///
/// `transcript_writer` is created eagerly at `open` time so that
/// `write_transcript_segment` needs no fallible lazy-init path. If no
/// segments are ever written, `finalise` calls `TranscriptWriter::finalise`
/// which is a no-op for an empty buffer, leaving `transcript.json` absent.
pub struct MeetingWriter {
    folder: MeetingFolder,
    encoder: Option<OggOpusEncoder<BufWriter<File>>>,
    format: AudioFormat,
    transcript_writer: Option<TranscriptWriter>,
}

impl MeetingWriter {
    /// Open a new `MeetingWriter` for `meeting_id` under `root`.
    ///
    /// Creates the per-meeting folder and opens `audio.opus` for writing.
    pub fn open(root: &Path, meeting_id: MeetingId, format: AudioFormat) -> AppResult<Self> {
        let folder = MeetingFolder::create(root, meeting_id)?;
        let audio_path = folder.audio_path();

        let file = File::create(&audio_path)
            .map_err(Error::Io)
            .map_err(minutist_common::AppError::from)?;

        let buffered = BufWriter::new(file);
        let encoder = OggOpusEncoder::new(buffered).map_err(minutist_common::AppError::from)?;

        let transcript_writer = TranscriptWriter::open(&folder)?;

        tracing::info!(
            target: "persistence",
            meeting_id = %meeting_id.0,
            audio_path = %audio_path.display(),
            "MeetingWriter opened"
        );

        Ok(Self {
            folder,
            encoder: Some(encoder),
            format,
            transcript_writer: Some(transcript_writer),
        })
    }

    /// Push f32 PCM samples (16 kHz mono) into the encoder.
    ///
    /// The encoder accumulates samples into 20 ms frames and encodes them
    /// to Opus on the fly. This call blocks briefly for each full frame
    /// encoded (well under 2 ms at 16 kHz mono on any modern CPU).
    pub fn push_samples(&mut self, samples: &[f32]) -> AppResult<()> {
        self.encoder_mut()?
            .push_samples(samples)
            .map_err(Into::into)
    }

    /// Flush the encoder and record the pause instant.
    ///
    /// After `pause()`, calls to `push_samples` will return an error until
    /// `resume()` is called.
    pub fn pause(&mut self) -> AppResult<()> {
        self.encoder_mut()?.pause().map_err(Into::into)
    }

    /// Resume after a pause. Injects a granule-position gap into the Ogg
    /// stream equal to the elapsed wall-clock pause time.
    pub fn resume(&mut self) -> AppResult<()> {
        self.encoder_mut()?.resume().map_err(Into::into)
    }

    /// Test-only resume injecting a deterministic pause-frame count (no
    /// wall-clock), delegating to the encoder's `resume_with_pause_frames` so
    /// the pause-INCLUDING silent-frame synthesis is exercised deterministically.
    #[cfg(test)]
    pub fn resume_with_pause_frames(&mut self, pause_frames: u64) -> AppResult<()> {
        self.encoder_mut()?
            .resume_with_pause_frames(pause_frames)
            .map_err(Into::into)
    }

    /// Append a transcript segment to the buffer and flush to `transcript.json`.
    ///
    /// Each call is durable: the segment is buffered then immediately written
    /// to disk so a crash between calls loses at most one flush's worth of
    /// transcript.
    pub fn write_transcript_segment(&mut self, segment: Segment) -> AppResult<()> {
        let tw = self
            .transcript_writer
            .as_mut()
            .ok_or_else(|| minutist_common::AppError::Internal {
                context: "MeetingWriter already finalised".to_string(),
            })?;
        tw.append(segment)?;
        tw.flush()
    }

    /// Finalise the recording: flush the encoder, finalise the transcript,
    /// write `metadata.json`, and return the `MeetingFolder` handle.
    pub fn finalise(mut self, meta: MeetingMeta) -> AppResult<MeetingFolder> {
        let encoder =
            self.encoder
                .take()
                .ok_or_else(|| minutist_common::AppError::Internal {
                    context: "MeetingWriter already finalised".to_string(),
                })?;

        encoder
            .finalise()
            .map_err(minutist_common::AppError::from)?;

        tracing::info!(
            target: "persistence",
            meeting_id = %self.folder.id().0,
            "audio.opus written"
        );

        // Finalise transcript.json (no-op if no segments were written).
        if let Some(tw) = self.transcript_writer.take() {
            tw.finalise()?;
        }

        // Write metadata.json.
        write_metadata_to_path(&self.folder.metadata_path(), &meta)
            .map_err(minutist_common::AppError::from)?;

        tracing::info!(
            target: "persistence",
            meeting_id = %self.folder.id().0,
            "metadata.json written"
        );

        Ok(self.folder)
    }

    // ----- Private helpers -----

    fn encoder_mut(&mut self) -> AppResult<&mut OggOpusEncoder<BufWriter<File>>> {
        self.encoder
            .as_mut()
            .ok_or_else(|| minutist_common::AppError::Internal {
                context: "MeetingWriter already finalised".to_string(),
            })
    }

    /// The `AudioFormat` this writer was opened with.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }
}
