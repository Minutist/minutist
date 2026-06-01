//! `TranscriptWriter` — buffered append writer for `transcript.json`.
//!
//! `transcript.json` is a JSON array of `Segment` values. On every `flush`
//! the entire buffer is rewritten to disk (replace, not append). This is
//! acceptable for v1: a 1-hour meeting produces ~120 segments (well under
//! 1 MB), and rewrite-on-flush gives crash-safety without a WAL.
//!
//! Phase 4 may move to JSON Lines if rewrite cost becomes significant; do
//! NOT preempt that here.
//!
//! # Zero-segment meetings
//!
//! If `flush` or `finalise` is called when the buffer is empty, **no file
//! is written** (absent rather than `[]`). `MeetingWriter::finalise` must
//! not error when no transcript was written — the Phase 1 integration test
//! records audio with no ASR model present, so zero-segment finalise must
//! succeed.

use std::io::Write;
use std::path::{Path, PathBuf};

use meeting_app_common::{AppResult, Segment};

use crate::error::Error;
use crate::folder::MeetingFolder;

/// Rewrite `transcript.json` in `meeting_dir` from `segments`.
///
/// The Phase-4 re-transcribe entry point: replaces the file's contents wholesale
/// (the same replace-not-append semantics as [`TranscriptWriter::flush`]). The
/// write is atomic (write to a sibling `transcript.json.tmp`, fsync, rename into
/// place), matching the durability of the notes/summary/metadata writers, so a
/// crash mid-rewrite never leaves a half-written `transcript.json`.
///
/// An **empty** `segments` slice removes any existing `transcript.json` rather
/// than writing `[]`, preserving the crate-wide invariant that a zero-segment
/// meeting has no transcript file (see [`TranscriptWriter`]).
///
/// `meeting_dir` is the `{root}/{uuid}/` folder; it must already exist.
pub fn write_transcript(meeting_dir: &Path, segments: &[Segment]) -> AppResult<()> {
    Ok(write_transcript_inner(meeting_dir, segments)?)
}

fn write_transcript_inner(meeting_dir: &Path, segments: &[Segment]) -> Result<(), Error> {
    let path = meeting_dir.join("transcript.json");

    if segments.is_empty() {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::Io(e)),
        }
        return Ok(());
    }

    let tmp_path = meeting_dir.join("transcript.json.tmp");
    let json = serde_json::to_vec_pretty(segments).map_err(Error::Serialise)?;

    let write_result = (|| -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(&json)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e));
    }

    if let Err(e) = std::fs::rename(&tmp_path, &path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e));
    }

    tracing::info!(
        target: "persistence",
        path = %path.display(),
        segments = segments.len(),
        "transcript.json rewritten"
    );

    Ok(())
}

/// Buffered, append-on-flush writer for `transcript.json`.
///
/// Created by `MeetingWriter::open` (lazily on first `write_transcript_segment`
/// call — see `MeetingWriter` docs). Not `Sync`.
pub struct TranscriptWriter {
    /// Absolute path to `transcript.json` in the meeting folder.
    path: PathBuf,
    /// In-memory accumulation of all segments written so far.
    buffer: Vec<Segment>,
}

impl TranscriptWriter {
    /// Open `transcript.json` (or prepare to create it) within `folder`.
    ///
    /// Does **not** touch the filesystem; actual file creation is deferred
    /// to the first `flush` or `finalise` call.
    pub fn open(folder: &MeetingFolder) -> AppResult<Self> {
        let path = folder.transcript_path();
        tracing::debug!(
            target: "persistence",
            path = %path.display(),
            "TranscriptWriter prepared"
        );
        Ok(Self {
            path,
            buffer: Vec::new(),
        })
    }

    /// Append a single segment to the in-memory buffer.
    ///
    /// Lightweight — does not touch the filesystem. Call `flush` to persist.
    pub fn append(&mut self, segment: Segment) -> AppResult<()> {
        self.buffer.push(segment);
        Ok(())
    }

    /// Rewrite `transcript.json` from the current in-memory buffer.
    ///
    /// If the buffer is empty, this is a no-op (no file is written or
    /// truncated). Idempotent: calling `flush` twice with no intervening
    /// `append` leaves the file unchanged.
    pub fn flush(&mut self) -> AppResult<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.write_to_disk()
    }

    /// Final flush and log. Consumed by `MeetingWriter::finalise`.
    ///
    /// Equivalent to `flush` for non-empty buffers. For an empty buffer
    /// this is a no-op — `transcript.json` is absent rather than `[]`.
    pub fn finalise(mut self) -> AppResult<()> {
        self.flush()?;
        tracing::info!(
            target: "persistence",
            path = %self.path.display(),
            segments = self.buffer.len(),
            "TranscriptWriter finalised"
        );
        Ok(())
    }

    // ----- Private helpers -----

    /// Serialise `self.buffer` and overwrite `self.path`.
    fn write_to_disk(&self) -> AppResult<()> {
        let json = serde_json::to_string_pretty(&self.buffer)
            .map_err(Error::Serialise)
            .map_err(meeting_app_common::AppError::from)?;

        let mut file = std::fs::File::create(&self.path)
            .map_err(Error::Io)
            .map_err(meeting_app_common::AppError::from)?;

        file.write_all(json.as_bytes())
            .map_err(Error::Io)
            .map_err(meeting_app_common::AppError::from)?;

        file.flush()
            .map_err(Error::Io)
            .map_err(meeting_app_common::AppError::from)?;

        tracing::debug!(
            target: "persistence",
            path = %self.path.display(),
            segments = self.buffer.len(),
            "transcript.json flushed"
        );

        Ok(())
    }
}
