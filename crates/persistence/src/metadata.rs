//! `metadata.json` write helpers.
//!
//! The public atomic writer keyed on the meeting **directory** —
//! `notes_crdt::write_metadata` — lives in the leaf `notes-crdt` crate. This
//! module retains only the crate-private path-keyed form used by
//! [`crate::MeetingWriter::finalise`], which already holds the resolved
//! `metadata.json` path. Delegates to the same atomic implementation in
//! `notes_crdt::metadata`.

use std::path::Path;

use minutist_common::MeetingMeta;

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
