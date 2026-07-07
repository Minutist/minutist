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

use minutist_common::{AppResult, Segment};

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
/// **Invariant:** `translations.json` is cleared here, immediately after the
/// transcript is rewritten. A full retranscription renumbers segment indices so
/// any existing per-index translations would point at the wrong segments; clearing
/// them here keeps the two files consistent without relying on callers to
/// remember. Re-diarize does NOT call this function (it only overlays
/// `speaker_id`s without changing indices or text), so translations survive
/// re-diarization unaffected.
///
/// `meeting_dir` is the `{root}/{uuid}/` folder; it must already exist.
pub fn write_transcript(meeting_dir: &Path, segments: &[Segment]) -> AppResult<()> {
    write_transcript_inner(meeting_dir, segments)?;
    // Clear stale translations after the transcript is replaced. An error here
    // is not fatal (the transcript was already written successfully); log and
    // continue so the caller is not forced to handle a cleanup failure.
    if let Err(e) = crate::translations::clear_translations(meeting_dir) {
        tracing::warn!(
            target: "persistence",
            folder = %meeting_dir.display(),
            "failed to clear translations.json after write_transcript: {e}; \
             stale translations may remain until the next retranscription"
        );
    }
    Ok(())
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

    /// Serialise `self.buffer` and atomically overwrite `self.path`.
    ///
    /// Writes to a sibling `transcript.json.tmp`, `fsync`s it, then renames it
    /// into place — the same pattern as [`write_transcript_inner`] — so a crash
    /// mid-write leaves the previous `transcript.json` intact rather than
    /// truncated or half-written.
    fn write_to_disk(&self) -> AppResult<()> {
        let json = serde_json::to_vec_pretty(&self.buffer)
            .map_err(Error::Serialise)
            .map_err(minutist_common::AppError::from)?;

        let tmp_path = self.path.with_extension("json.tmp");

        let write_result = (|| -> Result<(), std::io::Error> {
            let mut file = std::fs::File::create(&tmp_path)?;
            file.write_all(&json)?;
            file.flush()?;
            file.sync_all()?;
            Ok(())
        })();

        if let Err(e) = write_result {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(minutist_common::AppError::from(Error::Io(e)));
        }

        if let Err(e) = std::fs::rename(&tmp_path, &self.path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(minutist_common::AppError::from(Error::Io(e)));
        }

        tracing::debug!(
            target: "persistence",
            path = %self.path.display(),
            segments = self.buffer.len(),
            "transcript.json flushed"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::MeetingId;
    use tempfile::TempDir;

    fn seg(start_ms: u64, end_ms: u64, text: &str) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.to_string(),
            speaker_id: None,
            confidence: None,
            words: vec![],
            shared_speakers: Vec::new(),
        }
    }

    /// A write that fails between the temp-file write and the rename must
    /// leave the previous `transcript.json` untouched.
    ///
    /// Simulates the failure by occupying `write_to_disk`'s `.tmp` target with
    /// a directory before the second flush, so the temp-file open fails and
    /// the rename that would replace `transcript.json` never runs — exercising
    /// exactly the ordering the atomic write depends on: the destination path
    /// is only ever touched by the final `rename`, never by the write itself.
    #[test]
    fn write_to_disk_failure_before_rename_preserves_prior_content() {
        let tempdir = TempDir::new().expect("tempdir");
        let id = MeetingId::new();
        let folder = MeetingFolder::create(tempdir.path(), id).expect("create folder");

        let mut tw = TranscriptWriter::open(&folder).expect("open TranscriptWriter");
        tw.append(seg(0, 100, "old")).expect("append old");
        tw.flush().expect("flush old content");

        let path = folder.path().join("transcript.json");
        let old_json = std::fs::read_to_string(&path).expect("read prior content");

        // Occupy the tmp path with a directory so `File::create` in
        // `write_to_disk` fails before any rename is attempted.
        let tmp_path = path.with_extension("json.tmp");
        std::fs::create_dir(&tmp_path).expect("seed a directory at the tmp path");

        tw.append(seg(200, 300, "new")).expect("append new");
        let result = tw.flush();
        assert!(
            result.is_err(),
            "flush must surface the tmp-file-open failure rather than silently succeeding"
        );

        let content_after_failure =
            std::fs::read_to_string(&path).expect("transcript.json must remain readable");
        assert_eq!(
            content_after_failure, old_json,
            "a write that fails before rename must leave the previous transcript.json intact"
        );
    }
}
