//! `metadata.json` write helpers.
//!
//! The public atomic writer keyed on the meeting **directory**
//! ([`write_metadata`]) lives in the leaf `notes-crdt` crate and is re-exported
//! from `persistence` at its historical path. This module retains only the
//! crate-private path-keyed form used by [`crate::MeetingWriter::open`] and
//! [`crate::MeetingWriter::finalise`], which already hold the resolved
//! `metadata.json` path. Both share the one atomic implementation in
//! `notes_crdt::metadata`.
//!
//! [`write_metadata`]: crate::write_metadata

use std::path::Path;

use minutist_common::{AudioFormat, MeetingId, MeetingMeta};

use crate::error::Result;

/// Atomically write `meta` to `metadata.json` at an explicit file `path`
/// (crate-internal).
///
/// Used by [`crate::MeetingWriter::finalise`], which already holds the resolved
/// path. Delegates to `notes_crdt::write_metadata_atomic` — the same
/// tmp + fsync + rename implementation the public `write_metadata` uses —
/// mapping the leaf's error into `persistence`'s own `Error` so the writer's
/// signature is unchanged.
pub(crate) fn write_metadata_to_path(path: &Path, meta: &MeetingMeta) -> Result<()> {
    Ok(notes_crdt::write_metadata_atomic(path, meta)?)
}

/// Write a minimal in-progress `metadata.json` for a just-opened meeting,
/// unless one is already present at `path`.
///
/// Registers the meeting folder on disk the moment recording starts — rather
/// than only at [`crate::MeetingWriter::finalise`] — so
/// [`crate::MeetingIndex::reconcile_orphans`] can recover it after a crash or
/// kill mid-recording (see `architecture/cross-cutting.md`, "`metadata.json`
/// is written at recording start, not only at finalise"). The stub carries
/// `duration_ms = 0` and a synthesised "Recording <timestamp>" title, matching
/// the default `finalise` itself falls back to when the user never typed a
/// title; `finalise` overwrites this stub with the full record once recording
/// stops normally.
///
/// Guarded on `!path.exists()`: `MeetingFolder::create` already rejects a
/// pre-existing directory, so a real `metadata.json` collision cannot occur on
/// the normal `open` path today, but the guard keeps this helper safe to call
/// defensively without ever downgrading a richer, already-finalised
/// `metadata.json` back to the empty stub.
pub(crate) fn write_initial_metadata_if_absent(
    path: &Path,
    meeting_id: MeetingId,
    audio_format: &AudioFormat,
) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let now = chrono::Utc::now();
    let meta = MeetingMeta {
        uuid: meeting_id,
        title: format!("Recording {}", now.format("%Y-%m-%dT%H:%M:%SZ")),
        started_at: now.to_rfc3339(),
        ended_at: None,
        duration_ms: 0,
        speaker_count: 0,
        audio_format: audio_format.clone(),
        asr_model: None,
        llm_model: None,
        diarizer: None,
        speaker_names: std::collections::BTreeMap::new(),
        notes_format: 0,
        collection_id: None,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    write_metadata_to_path(path, &meta)
}
