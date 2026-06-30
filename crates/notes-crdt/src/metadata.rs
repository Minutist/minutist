//! `metadata.json` read / write / guarded-RMW helpers.
//!
//! - [`read_metadata`] deserialises `metadata.json` into a [`MeetingMeta`].
//! - [`update_metadata`] / [`update_metadata_if_present`] are the **guarded
//!   read-modify-write** entry points (issue 0025): they take the per-meeting
//!   [`metadata_lock`](crate::metadata_lock) across read→mutate→write so no two
//!   writers interleave. Living here (the leaf, beside the lock + the writer)
//!   lets `persistence` and the mobile `sync-ffi` path share one implementation;
//!   `persistence::meeting_ops` re-exports them at their historical paths.
//! - [`write_metadata`] is the **public** atomic writer keyed on the meeting
//!   **directory** (`{root}/{uuid}/`). It is the seam the orchestrator uses to
//!   update `metadata.json`'s `{ speaker_count, diarizer }` after the Phase-6
//!   diarization pass, while `persistence` stays the **sole** writer under
//!   `meetings/{uuid}/`. Like the notes/summary writers it is atomic
//!   (tmp + fsync + rename), so a crash mid-write never leaves a truncated
//!   `metadata.json`, and it leaves sibling files (`audio.opus`,
//!   `transcript.json`, `notes.json`) untouched.
//! - [`write_metadata_atomic`] is the shared atomic implementation, exposed so
//!   `persistence::MeetingWriter::finalise` (which already holds the resolved
//!   `metadata.json` path) can write through the same path.

use std::io::Write;
use std::path::Path;

use minutist_common::{AppResult, MeetingId, MeetingMeta};

use crate::error::{Error, Result};

/// Atomically write `meta` to `metadata.json` inside the existing meeting
/// folder `meeting_dir` (`{root}/{uuid}/`).
///
/// This is the public seam the orchestrator calls to update `metadata.json`
/// (e.g. `{ speaker_count, diarizer }` after diarization) while keeping
/// `persistence` the sole writer under `meetings/{uuid}/`. The write is atomic
/// — it goes to a sibling `metadata.json.tmp`, is fsynced, then renamed into
/// place — so a crash mid-write never truncates the file, and a successful
/// write leaves no `.tmp` residue. It does **not** create the meeting folder
/// (the folder is owned by `persistence::MeetingWriter` and is expected to
/// exist), and leaves sibling files (`audio.opus` / `transcript.json` /
/// `notes.json`) untouched.
pub fn write_metadata(meeting_dir: &Path, meta: &MeetingMeta) -> AppResult<()> {
    write_metadata_atomic(&meeting_dir.join("metadata.json"), meta)?;

    tracing::debug!(
        target: "persistence",
        folder = %meeting_dir.display(),
        "metadata.json written"
    );

    Ok(())
}

/// The shared atomic implementation: serialise `meta`, write to a sibling
/// `*.tmp` in the same directory, fsync, then rename into place. On any error
/// the tmp file is removed so no residue is left behind.
///
/// Exposed (rather than crate-private) so `persistence::MeetingWriter::finalise`
/// can write through this same atomic path while holding the resolved file path
/// directly.
pub fn write_metadata_atomic(path: &Path, meta: &MeetingMeta) -> Result<()> {
    let parent = path
        .parent()
        .ok_or(Error::InvalidState("metadata path has no parent"))?;
    let tmp_path = parent.join("metadata.json.tmp");

    let json = serde_json::to_vec_pretty(meta).map_err(Error::Serialise)?;

    let write_result = (|| -> std::result::Result<(), std::io::Error> {
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

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e));
    }

    Ok(())
}

/// Read and deserialise `metadata.json` from a meeting folder (`{root}/{uuid}/`).
///
/// A blocking `std::fs` read + `serde_json` parse. Lives in this leaf crate so
/// the guarded read-modify-write [`update_metadata`] — and any non-`persistence`
/// caller (the mobile `sync-ffi` path) — can read `metadata.json` without the
/// C-heavy `persistence` graph. `persistence::read_metadata` re-exports it (in
/// its own error namespace).
pub fn read_metadata(meeting_dir: &Path) -> Result<MeetingMeta> {
    let bytes = std::fs::read(meeting_dir.join("metadata.json"))?;
    let meta: MeetingMeta = serde_json::from_slice(&bytes)?;
    Ok(meta)
}

/// Guarded read-modify-write of a meeting's `metadata.json` — the single entry
/// point every `metadata.json` writer uses (issue 0025).
///
/// Takes the per-meeting [`metadata_lock`](crate::metadata_lock), confirms the
/// meeting exists, reads the [`MeetingMeta`], applies `f`, and writes it back
/// atomically — one lock acquisition spanning the existence check and the RMW, so
/// no concurrent writer can interleave and revert a field. The closure returns
/// whatever the caller needs from the mutated `meta` (e.g. `()`, or the updated
/// `speaker_names`), computed under the guard.
///
/// Synchronous by design: the std-`Mutex` guard must never be held across an
/// `.await`. Any async follow-up (e.g. an `index.db` upsert) runs after this
/// returns and the guard drops.
///
/// `AppError::InvalidInput` (via [`Error::MeetingNotFound`]) if the folder has no
/// `metadata.json`. Use [`update_metadata_if_present`] for the consumer case that
/// skips an absent meeting instead of erroring.
///
/// Lives in this leaf crate — co-located with [`metadata_lock`](crate::metadata_lock)
/// and [`write_metadata`] — so `persistence` AND the mobile `sync-ffi` path share
/// ONE guarded-RMW implementation rather than each re-deriving the lock discipline
/// (the de-duplication issue 0025 established). `persistence::meeting_ops`
/// re-exports it at its historical path, so its callers are unchanged.
pub fn update_metadata<F, R>(meetings_root: &Path, id: MeetingId, f: F) -> AppResult<R>
where
    F: FnOnce(&mut MeetingMeta) -> R,
{
    let folder = meetings_root.join(id.0.to_string());
    let lock = crate::metadata_lock(id);
    let _guard = lock.lock().expect("metadata lock poisoned");
    if !folder.join("metadata.json").exists() {
        return Err(Error::MeetingNotFound(id).into());
    }
    let mut meta = read_metadata(&folder)?;
    let out = f(&mut meta);
    write_metadata(&folder, &meta)?;
    Ok(out)
}

/// Like [`update_metadata`], but returns `Ok(None)` (skips) when the meeting is
/// not present locally rather than erroring — the consumer-side contract for a
/// peer-advertised state whose folder may not have synced in yet (the
/// notes/media receive path seeds it). The presence check and the write are one
/// atomic unit under the lock.
pub fn update_metadata_if_present<F, R>(
    meetings_root: &Path,
    id: MeetingId,
    f: F,
) -> AppResult<Option<R>>
where
    F: FnOnce(&mut MeetingMeta) -> R,
{
    let folder = meetings_root.join(id.0.to_string());
    let lock = crate::metadata_lock(id);
    let _guard = lock.lock().expect("metadata lock poisoned");
    if !folder.join("metadata.json").exists() {
        return Ok(None);
    }
    let mut meta = read_metadata(&folder)?;
    let out = f(&mut meta);
    write_metadata(&folder, &meta)?;
    Ok(Some(out))
}

/// The outcome of a conditional guarded RMW ([`update_metadata_if`]).
#[derive(Debug, PartialEq, Eq)]
pub enum MetaUpdate<R> {
    /// The closure committed: it returned `Some(R)` against the current on-disk
    /// state, and the mutated `metadata.json` was written.
    Applied(R),
    /// The closure returned `None`: the current state did not satisfy the
    /// predicate, so nothing was written (any speculative mutation is discarded).
    SkippedPredicate,
    /// The meeting folder has no `metadata.json` (not present locally); the
    /// closure never ran.
    SkippedAbsent,
}

/// Conditional guarded read-modify-write: like [`update_metadata`], but the
/// closure decides — under the per-meeting lock, against the CURRENT on-disk state
/// — whether to commit. `f` returns `Some(R)` to write the mutation it made (→
/// [`MetaUpdate::Applied`]) or `None` to leave `metadata.json` untouched (→
/// [`MetaUpdate::SkippedPredicate`], discarding any mutation it speculatively made);
/// an absent meeting yields [`MetaUpdate::SkippedAbsent`] without running `f`.
///
/// The predicate test and the mutation happen in ONE closure under ONE lock
/// acquisition, so there is no read-then-write window a concurrent writer could
/// slip through — the apply-iff-the-current-state-still-qualifies semantics the
/// producer-gate's claim / re-assert / renew / reap each need (e.g. "set
/// `Claimed{self}` only if currently `PendingProcessing` or an expired `Claimed`").
///
/// A caller that must OBSERVE the current state on the skip path (e.g. to make a
/// lease-aware reap decision when the predicate fails) captures it inside `f`
/// before returning `None`: `f` is handed `&mut MeetingMeta`, so it sees the fresh
/// on-disk state regardless of which branch it takes.
///
/// Synchronous by design (the std-`Mutex` guard is never held across an `.await`),
/// co-located with [`update_metadata`] and [`metadata_lock`](crate::metadata_lock)
/// in this leaf so `persistence` and the `election` crate share one implementation.
pub fn update_metadata_if<F, R>(
    meetings_root: &Path,
    id: MeetingId,
    f: F,
) -> AppResult<MetaUpdate<R>>
where
    F: FnOnce(&mut MeetingMeta) -> Option<R>,
{
    let folder = meetings_root.join(id.0.to_string());
    let lock = crate::metadata_lock(id);
    let _guard = lock.lock().expect("metadata lock poisoned");
    if !folder.join("metadata.json").exists() {
        return Ok(MetaUpdate::SkippedAbsent);
    }
    let mut meta = read_metadata(&folder)?;
    match f(&mut meta) {
        Some(out) => {
            write_metadata(&folder, &meta)?;
            Ok(MetaUpdate::Applied(out))
        }
        None => Ok(MetaUpdate::SkippedPredicate),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::ProcessingLifecycle;

    fn meeting_dir(root: &Path, id: MeetingId) -> std::path::PathBuf {
        root.join(id.0.to_string())
    }

    #[test]
    fn update_metadata_if_applies_when_predicate_commits() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let id = MeetingId::new();
        crate::MeetingFolder::ensure(tmp.path(), id).expect("ensure"); // seeds Local

        let out = update_metadata_if(tmp.path(), id, |m| {
            if matches!(m.processing, ProcessingLifecycle::Local) {
                m.processing = ProcessingLifecycle::PendingProcessing;
                Some("claimed")
            } else {
                None
            }
        })
        .expect("rmw");

        assert_eq!(out, MetaUpdate::Applied("claimed"));
        assert_eq!(
            read_metadata(&meeting_dir(tmp.path(), id))
                .expect("read")
                .processing,
            ProcessingLifecycle::PendingProcessing
        );
    }

    #[test]
    fn update_metadata_if_skips_predicate_observes_state_and_discards_mutation() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let id = MeetingId::new();
        crate::MeetingFolder::ensure(tmp.path(), id).expect("ensure");
        update_metadata(tmp.path(), id, |m| {
            m.processing = ProcessingLifecycle::PendingProcessing;
        })
        .expect("seed pending");

        let mut observed = None;
        let out: MetaUpdate<()> = update_metadata_if(tmp.path(), id, |m| {
            observed = Some(m.processing.clone());
            m.title = "speculative".to_string(); // mutate...
            // Predicate fails (only claim if Local; it is Pending) → skip.
            if matches!(m.processing, ProcessingLifecycle::Local) {
                Some(())
            } else {
                None // ...the mutation above is discarded
            }
        })
        .expect("rmw");

        assert_eq!(out, MetaUpdate::SkippedPredicate);
        // The closure saw the fresh on-disk state on the skip path.
        assert_eq!(observed, Some(ProcessingLifecycle::PendingProcessing));
        // Nothing was written: the speculative title mutation was discarded and the
        // processing state is unchanged.
        let meta = read_metadata(&meeting_dir(tmp.path(), id)).expect("read");
        assert_eq!(meta.title, "", "a None return must discard the mutation");
        assert_eq!(meta.processing, ProcessingLifecycle::PendingProcessing);
    }

    #[test]
    fn update_metadata_if_skips_absent_meeting_without_running_closure() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let id = MeetingId::new(); // no folder seeded
        let out: MetaUpdate<()> = update_metadata_if(tmp.path(), id, |_| {
            panic!("closure must not run for an absent meeting");
        })
        .expect("rmw");
        assert_eq!(out, MetaUpdate::SkippedAbsent);
    }
}
