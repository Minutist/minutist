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
//!   [`merge_processing`], skipping (`Ok(false)`) a meeting not held locally. It
//!   uses only primitives already owned by this leaf (`update_metadata_if_present`
//!   + `merge_processing`) — never `libsql` or `index.db` — so it belongs here
//!   rather than in `persistence`, and is the SINGLE implementation of the
//!   merge-and-skip-if-absent logic: `persistence::meeting_ops` re-exports it as
//!   a thin `async` wrapper (its desktop/hub subscribers `.await` it), and
//!   `sync-ffi` calls it directly (the phone has no `persistence` edge).
//! - [`folder::list_meeting_ids`]: the ONE enumeration of "which meetings this
//!   device holds" — the `{uuid}` directories under the meetings root (`.blobs`
//!   and non-UUID entries skipped). It lives beside the folder layout it scans;
//!   `sync`'s discovery exchange, the `headless` hub (via the
//!   `persistence::folder` re-export), and the producer-gate election loop all
//!   read it, so there is no per-consumer copy to drift.
//!   [`folder::parse_meeting_dir`] is the underlying per-entry predicate
//!   `list_meeting_ids` filters through; `persistence`'s own directory scans
//!   (`index::rebuild_from_disk` / `index::reconcile_orphans`, which need their
//!   own `read_dir` error handling and so cannot delegate the whole scan) filter
//!   through the SAME predicate rather than re-validating folder names
//!   themselves, so a folder is never a meeting to one consumer and clutter to
//!   another.
//!
//! # Write serialisation
//!
//! `NotesStore`'s three `notes.ydoc` writers (`save`, `apply_update`,
//! `seed_ydoc_if_needed`) each do a read → merge/rebuild → write over the file.
//! Without serialisation, two concurrent writers of the same meeting — routine
//! when the sync inbound path races a local editor autosave, or when two hub
//! peers reconcile the same meeting — each load the same base doc and
//! last-writer-wins on the file, silently dropping the other's merged update
//! (the atomic tmp+fsync+rename rules out a *torn* file, not a *lost* one).
//! [`notes_lock`] is a process-wide per-meeting `Arc<Mutex<()>>` registry
//! (`std::sync::Mutex`, never held across an `.await` — every guarded
//! read-modify-write is synchronous `std::fs`), the same shape as
//! [`metadata_lock`] but a **dedicated** lock, not shared with it: `notes.ydoc`
//! and `metadata.json` are independent files with independent writers, and
//! sharing one lock would needlessly serialise unrelated updates (e.g. a title
//! rename blocking on an in-flight notes merge). All three writers acquire the
//! lock once at the public entry point and delegate to a `_locked` inner body
//! (`std::sync::Mutex` is not reentrant, so `apply_update` — which calls
//! `seed_ydoc_if_needed` internally — calls that function's `_locked` body
//! directly rather than re-acquiring the lock).
//!
//! # Why this is a leaf
//!
//! These primitives were extracted from `persistence` so `sync` can transport
//! and merge the notes CRDT — including on mobile — without pulling in
//! `persistence`'s C-heavy graph (libsql / audiopus / ogg): keeping that graph
//! out of this leaf is what lets `sync` (which depends on `notes-crdt`, not
//! `persistence`) cross-compile its lib to mobile targets (e.g.
//! `aarch64-linux-android`). Third-party deps are limited accordingly: `yrs`
//! (the Yjs CRDT port), `chrono`, `serde` / `serde_json`, `thiserror`,
//! `tracing` — no `libsql`, `audiopus`, or `ogg`.
//!
//! [`Error`] is a light subset (`Io` / `FolderExists` / `Serialise` /
//! `InvalidState` / `MeetingNotFound`) with a `From<notes_crdt::Error> for
//! common::AppError`; `persistence::error` keeps its own `libsql` / `audiopus`
//! variants and adds the same `From` impl so its `Error` absorbs this leaf's.
//!
//! `persistence` depends on this crate and **re-exports every symbol above at
//! the `persistence::*` paths its consumers already use**
//! (`persistence::{MeetingFolder, NotesStore, NotesData, note_blocks_from_json,
//! write_metadata}` and the `persistence::{ydoc, notes, folder}` modules), and
//! remains the sole writer under `{app-data}/meetings/` — it simply delegates
//! the notes-CRDT bodies to this leaf. Callers of `persistence` (orchestrator,
//! ipc-bridge, agent-tools, app-main) are therefore unaffected by where these
//! primitives actually live.
//!
//! All log calls use `target: "notes-crdt"`, per `architecture/cross-cutting.md`
//! — "Each component uses a static `target` matching the crate name."

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
