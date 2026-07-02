//! Process-wide per-meeting serialisation of `notes.ydoc` read-modify-write.
//!
//! Every writer of a meeting's `notes.ydoc` takes that meeting's lock for the
//! whole read→merge/rebuild→write sequence, so two tasks racing on the same
//! meeting cannot each load the same base doc and last-writer-wins on the file
//! — e.g. two hub peers reconciling the same meeting concurrently
//! ([`crate::notes::NotesStore::apply_update`] via `sync`'s inbound path), or
//! that same inbound merge racing a local editor autosave. The atomic
//! tmp+fsync+rename in [`crate::notes`] already rules out a *torn* file; this
//! lock is what rules out a *lost* update between two full merges. The guarded
//! writers are [`crate::notes::NotesStore::save`],
//! [`crate::notes::NotesStore::apply_update`], and
//! [`crate::notes::NotesStore::seed_ydoc_if_needed`].
//!
//! This is a **dedicated** lock, separate from [`crate::metadata_lock`],
//! deliberately: `notes.ydoc` and `metadata.json` are independent files with
//! independent writers, and sharing one lock would serialise unrelated
//! updates (e.g. a title rename blocking on an in-flight notes merge for the
//! same meeting) for no correctness benefit. It is a [`std::sync::Mutex`], not
//! a `tokio` one, for the same reason `metadata_lock` is: every guarded RMW is
//! synchronous `std::fs` with no `.await` held across the guard, so this adds
//! no `tokio` dependency to `notes-crdt`. Mirrors the `metadata_lock` /
//! `MANIFEST_LOCKS` precedent (`architecture/cross-cutting.md` — Filesystem
//! layout).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use minutist_common::MeetingId;

/// Process-wide registry of per-meeting `notes.ydoc` mutexes.
///
/// Each `MeetingId` gets its own `Mutex<()>`. The map grows by one
/// `Arc<Mutex<()>>` per meeting touched and entries are never reclaimed — the
/// same accepted, bounded-by-user-meeting-count growth as `METADATA_LOCKS` /
/// `MANIFEST_LOCKS` (an empty mutex is tiny; the meeting count is what one
/// user accumulates).
static NOTES_LOCKS: OnceLock<Mutex<HashMap<MeetingId, Arc<Mutex<()>>>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<MeetingId, Arc<Mutex<()>>>> {
    NOTES_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Return (or lazily create) the per-meeting `notes.ydoc` lock.
///
/// The caller locks the returned `Arc<Mutex<()>>` for the duration of its
/// read-merge-write and drops the guard before any `.await`: the guard is a
/// [`std::sync::MutexGuard`], which must never be held across an await point.
/// `std::sync::Mutex` is not reentrant — a caller that already holds this
/// meeting's lock must not call another public `NotesStore` fn that also
/// acquires it (see the `_locked` helpers in `crate::notes`, which assume the
/// lock is already held and never re-acquire it).
pub fn notes_lock(id: MeetingId) -> Arc<Mutex<()>> {
    registry()
        .lock()
        .expect("NOTES_LOCKS registry poisoned")
        .entry(id)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same id always returns the one shared lock; distinct ids get
    /// distinct locks (so unrelated meetings never serialise against each other).
    #[test]
    fn same_id_shares_one_lock_distinct_ids_differ() {
        let a = MeetingId::new();
        let b = MeetingId::new();
        assert!(
            Arc::ptr_eq(&notes_lock(a), &notes_lock(a)),
            "same id must resolve to the same lock"
        );
        assert!(
            !Arc::ptr_eq(&notes_lock(a), &notes_lock(b)),
            "distinct ids must resolve to distinct locks"
        );
    }
}
