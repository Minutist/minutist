//! Persisting processing-lifecycle and deletion states received over sync.
//!
//! The connected build's `ConnectedSync` (in app-main) holds the `sync` crate's
//! `SyncEngine` and subscribes to its lifecycle-event stream, handing the
//! `broadcast::Receiver` here. The receiver's item type `(MeetingId,
//! ProcessingLifecycle, DeletionState)` is built from `common` types only, so
//! this lives in `ipc-bridge` — which owns the `persistence` edge — WITHOUT
//! `ipc-bridge` taking a dependency on the `sync` crate. That keeps the
//! trait-seam invariant that holds `sync` a near-leaf (see [`crate::sync`]).
//!
//! The hub (`headless`) runs the equivalent loop in a dedicated task beside its
//! serve loop; it depends on both `sync` and `persistence` directly and does not
//! link `ipc-bridge`.

use std::path::PathBuf;
use std::sync::Arc;

use minutist_common::{AppEvent, DeletionState, MeetingId, ProcessingLifecycle};
use persistence::MeetingIndex;
use tokio::sync::broadcast;

/// Drain the sync engine's lifecycle-event stream, persisting each
/// host-authoritative `(MeetingId, ProcessingLifecycle, DeletionState)` to the
/// meeting's `metadata.json` via
/// [`persistence::meeting_ops::apply_synced_lifecycle_if_present`] and
/// [`persistence::meeting_ops::apply_synced_deletion_if_present`].
///
/// Runs until the broadcast sender is dropped (engine shutdown → `Closed`).
///
/// Recovery semantics (see `planning/DESIGN_processing-lifecycle.md` §8):
/// - `Lagged` is non-fatal and logged. This subscriber does not re-trigger
///   discovery itself (it holds no engine handle). On the desktop, recovery rides
///   the next `sync_now`, which runs `discover_with` per peer (the §7
///   ride-alongside) and re-advertises the authoritative state. (The hub
///   additionally runs a periodic `discover_all` sweep; the desktop has no such
///   sweep — a dedicated desktop recovery driver is a later concern.)
/// - An event for a meeting not present locally is skipped (the notes/media
///   receive path, not this stream, seeds a meeting's folder). It is re-applied
///   once the folder has synced in and discovery next runs (the ride-along on the
///   next `sync_now`).
///
/// Unlike `processing` (no index mirror), a merged deletion state also
/// refreshes the `index.db` row — see
/// [`persistence::meeting_ops::apply_synced_deletion_if_present`]'s doc — so
/// this needs an [`MeetingIndex`] handle the processing half does not.
pub async fn run_lifecycle_subscriber(
    mut rx: broadcast::Receiver<(MeetingId, ProcessingLifecycle, DeletionState)>,
    meetings_dir: PathBuf,
    index: Arc<MeetingIndex>,
    event_tx: broadcast::Sender<AppEvent>,
) {
    loop {
        match rx.recv().await {
            Ok((meeting_id, processing, deletion)) => {
                match persistence::meeting_ops::apply_synced_lifecycle_if_present(
                    &meetings_dir,
                    meeting_id,
                    processing,
                )
                .await
                {
                    // Applied; `apply_synced_lifecycle_if_present` logs the merged state.
                    Ok(true) => {}
                    Ok(false) => tracing::debug!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        "synced lifecycle for a meeting not present locally; skipping (re-applied only on a later discovery)"
                    ),
                    Err(e) => tracing::warn!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        error = %e,
                        "failed to apply synced lifecycle"
                    ),
                }
                match persistence::meeting_ops::apply_synced_deletion_if_present(
                    &meetings_dir,
                    &index,
                    meeting_id,
                    deletion,
                )
                .await
                {
                    // Applied: the local deletion state actually changed (or was
                    // freshly created) — notify the webview so the meeting list
                    // reflects it without waiting for an unrelated refresh.
                    Ok(true) => {
                        let _ = event_tx.send(AppEvent::MeetingDeletionChanged { meeting_id });
                    }
                    Ok(false) => tracing::debug!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        "synced deletion state for a meeting not present locally; skipping"
                    ),
                    Err(e) => tracing::warn!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        error = %e,
                        "failed to apply synced deletion state"
                    ),
                }
            }
            Err(broadcast::error::RecvError::Lagged(dropped)) => {
                tracing::warn!(
                    target: "ipc-bridge",
                    dropped,
                    "lifecycle-event subscriber lagged; dropped states recover on the next sync_now ride-along discovery"
                );
            }
            Err(broadcast::error::RecvError::Closed) => {
                tracing::info!(
                    target: "ipc-bridge",
                    "lifecycle event channel closed; subscriber exiting"
                );
                break;
            }
        }
    }
}

/// Spawn [`run_lifecycle_subscriber`] on the Tauri async runtime.
///
/// Like [`crate::spawn_event_forwarder`], this uses `tauri::async_runtime::spawn`
/// (not `tokio::spawn`): it is started from app-main's engine-startup task, and
/// Tauri's async runtime is tokio-backed.
///
/// The task holds no `JoinHandle` or cancellation token — it exits when the
/// engine's broadcast sender drops (`Closed`). This assumes engine startup is
/// once per process (app-main's `ConnectedSync` binds once); a future
/// engine-restart path MUST cancel or join this task before dropping the old
/// engine, or it would orphan a subscriber.
pub fn spawn_lifecycle_subscriber(
    rx: broadcast::Receiver<(MeetingId, ProcessingLifecycle, DeletionState)>,
    meetings_dir: PathBuf,
    index: Arc<MeetingIndex>,
    event_tx: broadcast::Sender<AppEvent>,
) {
    tauri::async_runtime::spawn(run_lifecycle_subscriber(rx, meetings_dir, index, event_tx));
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::HostRef;

    /// The subscriber persists a present meeting's synced lifecycle AND
    /// deletion state, skips an absent one without dying, and exits when the
    /// sender is dropped.
    #[tokio::test]
    async fn persists_present_skips_absent_and_exits_on_close() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path().to_path_buf();
        let index = Arc::new(MeetingIndex::open(":memory:").await.expect("open index"));

        // A meeting we hold (placeholder seeded → processing = Local).
        let present = MeetingId::new();
        notes_crdt::MeetingFolder::ensure(&root, present).expect("seed present meeting");

        let (tx, rx) = broadcast::channel(16);
        let (event_tx, mut events) = broadcast::channel(16);
        let handle = tokio::spawn(run_lifecycle_subscriber(
            rx,
            root.clone(),
            index.clone(),
            event_tx,
        ));

        // An advertisement for a meeting we do not hold: must be skipped, not fatal.
        tx.send((
            MeetingId::new(),
            ProcessingLifecycle::PendingProcessing,
            DeletionState::default(),
        ))
        .expect("send absent");

        // An advertisement for the meeting we hold: must be applied.
        let processed = ProcessingLifecycle::Processed {
            processed_by: HostRef("endpoint-xyz".to_string()),
            at: "2026-06-27T10:25:00Z".to_string(),
        };
        let deleted = DeletionState {
            deleted: true,
            version: 1,
            by: HostRef("endpoint-xyz".to_string()),
            changed_at: "2026-06-27T10:25:00Z".to_string(),
        };
        tx.send((present, processed.clone(), deleted.clone()))
            .expect("send present");

        // Dropping the sender closes the channel; the loop drains then exits.
        drop(tx);
        handle.await.expect("subscriber task joins");

        let meta = persistence::read_metadata(&root.join(present.0.to_string()))
            .expect("read present metadata");
        assert_eq!(meta.processing, processed);
        assert_eq!(meta.deletion, deleted);

        // The index mirror was refreshed too.
        let listed = index.list_meetings().await.expect("list");
        assert_eq!(listed.len(), 1);
        assert!(listed[0].deleted_at.is_some());

        // The webview was notified so it can refresh without an unrelated event.
        match events.try_recv() {
            Ok(AppEvent::MeetingDeletionChanged { meeting_id }) => {
                assert_eq!(meeting_id, present);
            }
            other => panic!("expected MeetingDeletionChanged, got {other:?}"),
        }
    }
}
