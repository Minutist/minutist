//! The purged-meeting tombstone set — remembers "I permanently removed this
//! meeting" past the moment its folder stops existing.
//!
//! [`notes_crdt::folder::list_meeting_ids`] (what `sync`'s discovery/adopt
//! machinery uses to learn "which meetings does this device have") is a raw
//! directory scan: presence on disk is the only "I have this" signal, with no
//! memory of "I had this and deliberately removed it." That is fine while a
//! meeting is merely soft-deleted (still on disk, `MeetingMeta::deletion` set)
//! — it converges like any other field. It stops being fine the moment a
//! meeting is actually purged: a peer that has not yet caught up and purges
//! later would look "ahead" on the hub's next `adopt_from_peer` sweep and get
//! pulled right back (`crates/sync/src/endpoint.rs`).
//!
//! This tombstone set closes that gap for the one sweep that matters (the hub
//! is the only device that ever pulls-everything-it-lacks from a peer — see
//! `architecture/components.md`). Every purge (manual or the periodic sweep)
//! records an entry here; the hub's adopt path consults [`PurgedStore::is_purged`]
//! before re-pulling an id it does not hold, so a late/slow peer's still-existing
//! copy cannot undo a purge that already happened. Entries older than a
//! generous slack are dropped ([`PurgedStore::gc`]) so the set does not grow
//! unbounded — 30 days is ample time for any reachable peer to have converged.
//!
//! Sits at the app-data root beside `collections.json` and `index.db` (NOT
//! under `index.db`, for the same reason `collections.json` doesn't: a
//! `rebuild_from_disk` must not wipe it).

use std::path::{Path, PathBuf};

use minutist_common::{AppResult, MeetingId};
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// The conventional `purged.json` path under an app-data root. Mirrors
/// `collections::collections_path`.
pub fn purged_path(app_data_root: &Path) -> PathBuf {
    app_data_root.join("purged.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PurgedEntry {
    id: MeetingId,
    /// RFC 3339 UTC. GC timing only — this set has no merge-arbitration role.
    purged_at: String,
}

/// Stateless reader/writer for the purged-tombstone set (`{app-data}/purged.json`).
pub struct PurgedStore;

impl PurgedStore {
    /// Whether `id` was purged on this device and not yet garbage-collected.
    pub fn is_purged(app_data_root: &Path, id: MeetingId) -> AppResult<bool> {
        Ok(Self::load(app_data_root)?.iter().any(|e| e.id == id))
    }

    /// The full set of not-yet-GC'd purged ids, for a caller that checks many
    /// candidate ids against one snapshot (e.g. an adopt sweep over several
    /// peers' advertised meetings) — one file read instead of one per id via
    /// repeated [`Self::is_purged`] calls.
    pub fn purged_ids(app_data_root: &Path) -> AppResult<std::collections::HashSet<MeetingId>> {
        Ok(Self::load(app_data_root)?.into_iter().map(|e| e.id).collect())
    }

    /// Record that `id` was just purged. Idempotent: re-recording an already-
    /// present id refreshes its `purged_at` rather than duplicating the entry.
    pub fn record(app_data_root: &Path, id: MeetingId) -> AppResult<()> {
        let mut entries = Self::load(app_data_root)?;
        entries.retain(|e| e.id != id);
        entries.push(PurgedEntry {
            id,
            purged_at: chrono::Utc::now().to_rfc3339(),
        });
        Self::write(app_data_root, &entries)
    }

    /// Drop entries older than `older_than_days` — bounds the set's growth.
    /// A malformed `purged_at` is kept (never guess an entry is safe to drop).
    pub fn gc(app_data_root: &Path, older_than_days: i64) -> AppResult<()> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(older_than_days);
        let mut entries = Self::load(app_data_root)?;
        let before = entries.len();
        entries.retain(|e| match chrono::DateTime::parse_from_rfc3339(&e.purged_at) {
            Ok(purged_at) => purged_at.with_timezone(&chrono::Utc) >= cutoff,
            Err(_) => true,
        });
        if entries.len() != before {
            Self::write(app_data_root, &entries)?;
        }
        Ok(())
    }

    fn load(app_data_root: &Path) -> AppResult<Vec<PurgedEntry>> {
        let path = purged_path(app_data_root);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::Io(e).into()),
        };
        Ok(serde_json::from_slice(&bytes).map_err(Error::Serialise)?)
    }

    fn write(app_data_root: &Path, entries: &[PurgedEntry]) -> AppResult<()> {
        let path = purged_path(app_data_root);
        let parent = path
            .parent()
            .ok_or(Error::InvalidState("purged path has no parent"))?;
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
        let json = serde_json::to_vec_pretty(entries).map_err(Error::Serialise)?;
        minutist_common::fs::write_atomic(&path, &json)
            .map_err(|e| Error::Io(std::io::Error::other(e)).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_file_reads_as_not_purged() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(!PurgedStore::is_purged(tmp.path(), MeetingId::new()).unwrap());
    }

    #[test]
    fn record_then_is_purged_round_trips() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let id = MeetingId::new();
        PurgedStore::record(tmp.path(), id).unwrap();
        assert!(PurgedStore::is_purged(tmp.path(), id).unwrap());
        // An unrelated id is unaffected.
        assert!(!PurgedStore::is_purged(tmp.path(), MeetingId::new()).unwrap());
    }

    #[test]
    fn re_recording_the_same_id_does_not_duplicate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let id = MeetingId::new();
        PurgedStore::record(tmp.path(), id).unwrap();
        PurgedStore::record(tmp.path(), id).unwrap();
        assert_eq!(PurgedStore::load(tmp.path()).unwrap().len(), 1);
    }

    #[test]
    fn gc_drops_only_entries_past_the_cutoff() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let fresh = MeetingId::new();
        let stale = MeetingId::new();
        PurgedStore::record(tmp.path(), fresh).unwrap();
        PurgedStore::record(tmp.path(), stale).unwrap();
        // Backdate `stale`'s entry past the cutoff by writing directly.
        let mut entries = PurgedStore::load(tmp.path()).unwrap();
        for e in entries.iter_mut() {
            if e.id == stale {
                e.purged_at = "2000-01-01T00:00:00Z".to_string();
            }
        }
        PurgedStore::write(tmp.path(), &entries).unwrap();

        PurgedStore::gc(tmp.path(), 30).unwrap();
        assert!(PurgedStore::is_purged(tmp.path(), fresh).unwrap());
        assert!(!PurgedStore::is_purged(tmp.path(), stale).unwrap());
    }
}
