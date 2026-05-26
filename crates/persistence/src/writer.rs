use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use meeting_app_common::{AppResult, AudioFormat, MeetingId, MeetingMeta};

use crate::error::Error;
use crate::folder::MeetingFolder;
use crate::metadata::write_metadata;
use crate::opus_encoder::OggOpusEncoder;

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
pub struct MeetingWriter {
    folder: MeetingFolder,
    encoder: Option<OggOpusEncoder<BufWriter<File>>>,
    format: AudioFormat,
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
            .map_err(meeting_app_common::AppError::from)?;

        let buffered = BufWriter::new(file);
        let encoder = OggOpusEncoder::new(buffered).map_err(meeting_app_common::AppError::from)?;

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

    /// Finalise the recording: flush the encoder, write `metadata.json`, and
    /// return the `MeetingFolder` handle.
    pub fn finalise(mut self, meta: MeetingMeta) -> AppResult<MeetingFolder> {
        let encoder =
            self.encoder
                .take()
                .ok_or_else(|| meeting_app_common::AppError::Internal {
                    context: "MeetingWriter already finalised".to_string(),
                })?;

        encoder
            .finalise()
            .map_err(meeting_app_common::AppError::from)?;

        tracing::info!(
            target: "persistence",
            meeting_id = %self.folder.id().0,
            "audio.opus written"
        );

        // Write metadata.json.
        write_metadata(self.folder.metadata_path(), &meta)
            .map_err(meeting_app_common::AppError::from)?;

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
            .ok_or_else(|| meeting_app_common::AppError::Internal {
                context: "MeetingWriter already finalised".to_string(),
            })
    }

    /// The `AudioFormat` this writer was opened with.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }
}
