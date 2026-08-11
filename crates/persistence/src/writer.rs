use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use minutist_common::{AppResult, AudioFormat, MeetingId, MeetingMeta, Segment};
use notes_crdt::MeetingFolder;

use crate::error::Error;
use crate::metadata::write_metadata_to_path;
use crate::opus_encoder::OggOpusEncoder;
use crate::transcript::TranscriptWriter;

/// Build the on-disk shell of a "New meeting" prep draft: the folder, a draft
/// `metadata.json` (empty title, `recording_started: false`), and a
/// `notes.ydoc` seeded with that draft metadata so title/notes/attachment
/// edits made during prep converge to other devices immediately, before
/// recording ever starts. The Attachments feature and the notes editor both
/// need a real `MeetingId` + folder to exist, so this cannot be a
/// client-side-only draft.
///
/// Shared by [`MeetingWriter::open`] (the auto-start-immediately path: draft
/// then promote, back-to-back, no visible gap) and the "New meeting" IPC
/// command (which promotes later, via [`MeetingWriter::open_for_recording`],
/// on a separate user action).
pub fn create_draft(root: &Path, meeting_id: MeetingId) -> AppResult<MeetingFolder> {
    let folder = MeetingFolder::create(root, meeting_id)?;

    let meta = MeetingMeta {
        uuid: meeting_id,
        title: String::new(),
        started_at: chrono::Utc::now().to_rfc3339(),
        ended_at: None,
        duration_ms: 0,
        speaker_count: 0,
        audio_format: AudioFormat {
            codec: "opus".to_string(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        },
        asr_model: None,
        llm_model: None,
        diarizer: None,
        speaker_names: std::collections::BTreeMap::new(),
        // Yjs-authoritative from creation — the draft's notes.ydoc is seeded
        // below in the same call, unlike a legacy pre-CRDT meeting.
        notes_format: 1,
        collection_id: None,
        processing: Default::default(),
        recording_started: false,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    write_metadata_to_path(&folder.metadata_path(), &meta)
        .map_err(minutist_common::AppError::from)?;

    notes_crdt::meta_crdt::initialise_notes_with_meta(
        folder.meetings_root()?,
        meeting_id,
        None,
        "",
        &meta,
    )?;

    tracing::info!(
        target: "persistence",
        meeting_id = %meeting_id.0,
        "meeting draft created"
    );

    Ok(folder)
}

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
///
/// # `metadata.json` durability
///
/// Both `open` and `open_for_recording` flip `recording_started: true` and
/// stamp the real capture `started_at`/`audio_format` as their last step, so
/// the meeting is a real, indexable, actively-recording folder before a
/// single sample is pushed. `finalise` overwrites the rest of the record.
pub struct MeetingWriter {
    folder: MeetingFolder,
    encoder: Option<OggOpusEncoder<BufWriter<File>>>,
    format: AudioFormat,
    transcript_writer: Option<TranscriptWriter>,
}

impl MeetingWriter {
    /// Open a new `MeetingWriter` for `meeting_id` under `root`, creating the
    /// meeting from scratch (the auto-start-immediately path).
    ///
    /// Composes [`create_draft`] with the shared "start capturing" body — see
    /// [`Self::open_for_recording`] for promoting an EXISTING "New meeting"
    /// prep draft instead.
    pub fn open(root: &Path, meeting_id: MeetingId, format: AudioFormat) -> AppResult<Self> {
        let folder = create_draft(root, meeting_id)?;
        Self::start_capturing(folder, format)
    }

    /// Promote an existing "New meeting" prep draft (created via
    /// [`create_draft`]) to an actively-recording meeting: opens `audio.opus`
    /// for writing into the SAME folder/id, and stamps the real capture
    /// `started_at`/`audio_format` over the draft's placeholder values.
    ///
    /// `AppError::InvalidInput` if no draft folder exists for `meeting_id`
    /// (via `Error::MeetingNotFound`), or if it has already recorded
    /// (`recording_started: true`) — promoting an already-recorded meeting a
    /// second time would `File::create` a fresh, empty `audio.opus` over the
    /// real recording, silently truncating it.
    pub fn open_for_recording(
        root: &Path,
        meeting_id: MeetingId,
        format: AudioFormat,
    ) -> AppResult<Self> {
        let folder = MeetingFolder::open_existing(root, meeting_id)?;
        let meta = crate::reader::read_metadata(folder.path())?;
        if meta.recording_started {
            return Err(minutist_common::AppError::InvalidInput {
                context: format!(
                    "meeting {} has already recorded; open_for_recording only promotes an unstarted draft",
                    meeting_id.0
                ),
            });
        }
        Self::start_capturing(folder, format)
    }

    /// Shared body of [`Self::open`] and [`Self::open_for_recording`]: opens
    /// `audio.opus` + the encoder + the transcript writer on an
    /// already-existing `folder`, then stamps the real capture
    /// `started_at`/`audio_format` via a guarded RMW of `metadata.json` (never
    /// a raw overwrite — a promoted draft may already carry a real user-typed
    /// title/collection from the prep phase that a blind overwrite would
    /// clobber) and mirrors the same two fields into the meta CRDT (issue
    /// 0052) via the granular setters, un-nested after the RMW returns (see
    /// `meeting_ops::rename_meeting`'s lock-ordering note).
    fn start_capturing(folder: MeetingFolder, format: AudioFormat) -> AppResult<Self> {
        let meeting_id = folder.id();
        let audio_path = folder.audio_path();

        let file = File::create(&audio_path)
            .map_err(Error::Io)
            .map_err(minutist_common::AppError::from)?;

        let buffered = BufWriter::new(file);
        let encoder = OggOpusEncoder::new(buffered).map_err(minutist_common::AppError::from)?;

        let transcript_writer = TranscriptWriter::open(&folder)?;

        let meetings_root = folder.meetings_root()?;
        let started_at = chrono::Utc::now().to_rfc3339();
        notes_crdt::update_metadata(meetings_root, meeting_id, |meta| {
            meta.recording_started = true;
            meta.started_at = started_at.clone();
            meta.audio_format = format.clone();
        })?;
        notes_crdt::meta_crdt::edit_meta_ydoc(meetings_root, meeting_id, |doc| {
            notes_crdt::meta_crdt::set_started_at(doc, &started_at);
            notes_crdt::meta_crdt::set_audio_format(doc, &format);
        })?;

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

        // Mirror the now-known fields into notes.ydoc's authored-metadata map
        // (issue 0052) via the granular setters — `notes.ydoc` always already
        // exists by this point (seeded at draft creation, see `create_draft`),
        // so this is an edit, never the one-time `initialise_notes_with_meta`
        // capture write. Granular (not a full `write_descriptive` rewrite) so
        // a concurrent edit to a DIFFERENT field — a synced peer's edit while
        // the recording is still live — is not clobbered. `title` IS
        // re-affirmed here (via `set_title`, not a blind full rewrite): `meta`
        // was built moments ago from whatever is currently the best-known
        // title (a live-typed one, else the on-disk value, else the
        // synthesized default — see `orchestrator::stop`), and a draft's
        // `notes.ydoc` carries NO title key at all until the first real title
        // exists (`write_descriptive` skips an empty title), so this call is
        // what actually gets a title into the CRDT for the common case of a
        // meeting that was never renamed during prep. `started_at`/
        // `audio_format` are untouched here (already set at
        // draft-creation/promotion time).
        let meetings_root = self.folder.meetings_root()?;
        notes_crdt::meta_crdt::edit_meta_ydoc(meetings_root, self.folder.id(), |doc| {
            notes_crdt::meta_crdt::set_title(doc, &meta.title);
            notes_crdt::meta_crdt::set_ended_at(doc, meta.ended_at.as_deref());
            notes_crdt::meta_crdt::set_duration_ms(doc, meta.duration_ms);
            notes_crdt::meta_crdt::set_speaker_count(doc, meta.speaker_count);
            notes_crdt::meta_crdt::set_asr_model(doc, meta.asr_model.as_ref());
            notes_crdt::meta_crdt::set_llm_model(doc, meta.llm_model.as_ref());
            notes_crdt::meta_crdt::set_diarizer(doc, meta.diarizer.as_ref());
            notes_crdt::meta_crdt::set_speaker_names(doc, &meta.speaker_names);
        })?;

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
