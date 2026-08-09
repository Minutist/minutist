//! `notes-crdt` — the notes-CRDT primitives shared by `persistence` and `sync`.
//!
//! A leaf crate (depends only on `common` among workspace crates) carrying:
//!
//! - [`ydoc`]: the Yjs (`yrs`) representation of the notes document — the
//!   authoritative `notes.ydoc` blob and its lossless conversions to/from
//!   ProseMirror JSON, plus the v1/v2 encoding hops.
//! - [`NotesStore`]: the standalone, stateless reader/writer for a meeting's
//!   `notes.ydoc` + its derived `notes.json` / `notes.md`, plus
//!   [`note_blocks_from_json`] (the summariser's note-paragraph projection).
//! - [`MeetingFolder`]: the on-disk `{root}/{uuid}/` layout and its path
//!   helpers, including [`MeetingFolder::ensure`] (the sync-receiver seam).
//! - [`read_metadata`] / [`write_metadata`]: the `metadata.json` reader and the
//!   atomic writer.
//! - [`update_metadata`] / [`update_metadata_if_present`]: the guarded
//!   read-modify-write of `metadata.json` (issue 0025) — read→mutate→write under
//!   [`metadata_lock`], the single entry point every writer uses.
//! - [`metadata_lock`]: the process-wide per-meeting lock that serialises every
//!   `metadata.json` read-modify-write (shared by `persistence::meeting_ops` and
//!   [`MeetingFolder::ensure`]).
//! - [`notes_lock`]: the process-wide per-meeting lock that serialises every
//!   `notes.ydoc` read-modify-write ([`NotesStore::save`], `apply_update`, and
//!   `seed_ydoc_if_needed`) — dedicated to `notes.ydoc`, separate from
//!   `metadata_lock` because the two files have independent writers and
//!   sharing one lock would needlessly serialise unrelated updates.
//! - [`merge_processing`]: the precedence merge for the processing-lifecycle
//!   field, applied on the inbound (synced) write path so a peer's stale state
//!   cannot walk the local one backwards.
//! - [`apply_synced_lifecycle_if_present`]: the guarded read-modify-write that
//!   applies a peer-advertised lifecycle state to a meeting's `metadata.json` via
//!   [`merge_processing`], skipping a meeting not held locally. The single
//!   implementation shared by `persistence` (desktop/hub) and `sync-ffi` (phone)
//!   — the two previously carried independent copies of the same body.
//!
//! These were extracted from `persistence` so `sync` can transport / merge the
//! notes CRDT without pulling in `persistence`'s C-heavy graph (libsql /
//! audiopus / ogg). `persistence` re-exports every symbol below at its existing
//! paths, so its callers (orchestrator, ipc-bridge, agent-tools, app-main) are
//! unchanged.
//!
//! All log calls use `target: "persistence"` so the moved code stays in the
//! same filtered log stream as the rest of the meeting-storage surface.

pub mod error;
pub mod folder;
pub mod lifecycle;
pub mod meta_crdt;
pub mod metadata;
pub mod metadata_lock;
pub mod notes;
pub mod notes_lock;
pub mod ydoc;

pub use error::Error;
pub use folder::MeetingFolder;
pub use lifecycle::{apply_synced_lifecycle_if_present, merge_processing};
pub use metadata::{
    read_metadata, update_metadata, update_metadata_if, update_metadata_if_present, write_metadata,
    write_metadata_atomic, MetaUpdate,
};
pub use metadata_lock::metadata_lock;
pub use notes::{note_blocks_from_json, NotesData, NotesStore};
pub use notes_lock::notes_lock;
