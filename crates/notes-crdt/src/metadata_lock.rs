//! Process-wide per-meeting serialisation of `metadata.json` read-modify-write.
//!
//! Every writer of a meeting's `metadata.json` takes that meeting's lock for the
//! whole read→mutate→write sequence, so two tasks racing on the same meeting
//! cannot interleave and lose each other's field update — e.g. the sync
//! lifecycle-event subscriber setting `processing` while a user command renames
//! the meeting. The guarded writers are `persistence::meeting_ops`' RMW
//! operations and [`MeetingFolder::ensure`](crate::MeetingFolder::ensure)'s
//! placeholder seed.
//!
//! The lock lives here, in the leaf `notes-crdt` crate, because it is the lowest
//! crate both writers reach: `persistence` depends on `notes-crdt`, and `ensure`
//! is defined here. It is a [`std::sync::Mutex`], not a `tokio` one, because
//! every guarded RMW is synchronous `std::fs` with no `.await` held across the
//! guard — so this adds no `tokio` dependency to `notes-crdt`. Mirrors the
//! `MANIFEST_LOCKS` precedent for `attachments.json` in `persistence`
//! (`architecture/cross-cutting.md` — Filesystem layout).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use minutist_common::MeetingId;

/// Process-wide registry of per-meeting `metadata.json` mutexes.
///
/// Each `MeetingId` gets its own `Mutex<()>`. The map grows by one
/// `Arc<Mutex<()>>` per meeting touched and entries are never reclaimed — the
/// same accepted, bounded-by-user-meeting-count growth as `MANIFEST_LOCKS` (an
/// empty mutex is tiny; the meeting count is what one user accumulates).
static METADATA_LOCKS: OnceLock<Mutex<HashMap<MeetingId, Arc<Mutex<()>>>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<MeetingId, Arc<Mutex<()>>>> {
    METADATA_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Return (or lazily create) the per-meeting `metadata.json` lock.
///
/// The caller locks the returned `Arc<Mutex<()>>` for the duration of its
/// check-then-read-modify-write and drops the guard before any `.await`: the
/// guard is a [`std::sync::MutexGuard`], which must never be held across an
/// await point.
pub fn metadata_lock(id: MeetingId) -> Arc<Mutex<()>> {
    registry()
        .lock()
        .expect("METADATA_LOCKS registry poisoned")
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
            Arc::ptr_eq(&metadata_lock(a), &metadata_lock(a)),
            "same id must resolve to the same lock"
        );
        assert!(
            !Arc::ptr_eq(&metadata_lock(a), &metadata_lock(b)),
            "distinct ids must resolve to distinct locks"
        );
    }
}
