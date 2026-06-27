//! Persisting processing-lifecycle states received over sync.
//!
//! The connected build's `ConnectedSync` (in app-main) holds the `sync` crate's
//! `SyncEngine` and subscribes to its lifecycle-event stream, handing the
//! `broadcast::Receiver` here. The receiver's item type `(MeetingId,
//! ProcessingLifecycle)` is built from `common` types only, so this lives in
//! `ipc-bridge` — which owns the `persistence` edge — WITHOUT `ipc-bridge`
//! taking a dependency on the `sync` crate. That keeps the trait-seam invariant
//! that holds `sync` a near-leaf (see [`crate::sync`]).
//!
//! The hub (`headless`) runs the equivalent loop inline in its serve loop; it
//! depends on both `sync` and `persistence` directly and does not link
//! `ipc-bridge`.

use std::path::PathBuf;

use minutist_common::{MeetingId, ProcessingLifecycle};
use tokio::sync::broadcast;

/// Drain the sync engine's lifecycle-event stream, persisting each
/// host-authoritative `(MeetingId, ProcessingLifecycle)` to the meeting's
/// `metadata.json` via
/// [`persistence::meeting_ops::apply_synced_lifecycle_if_present`].
///
/// Runs until the broadcast sender is dropped (engine shutdown → `Closed`).
///
/// Recovery semantics (v1 — see the known limitations in
/// `planning/DESIGN_processing-lifecycle.md`):
/// - `Lagged` is non-fatal and logged. A dropped event is recovered only when
///   discovery is next driven and re-advertises the authoritative state. Today
///   `discover_with` has NO scheduled caller, so recovery relies on a peer
///   initiating discovery against us (inbound) or a future scheduled caller —
///   there is no automatic re-run here. TODO(recovery): on `Lagged`, trigger a
///   re-discovery once a discovery driver exists.
/// - An event for a meeting not present locally is skipped (the notes/media
///   receive path, not this stream, seeds a meeting's folder). A skipped event
///   is NOT replayed when that folder later syncs in: the meeting's lifecycle
///   stays at its local default until a subsequent discovery re-advertises it.
///   TODO(recovery): re-apply on folder arrival, or couple discovery to
///   notes/media sync completion.
pub async fn run_lifecycle_subscriber(
    mut rx: broadcast::Receiver<(MeetingId, ProcessingLifecycle)>,
    meetings_dir: PathBuf,
) {
    loop {
        match rx.recv().await {
            Ok((meeting_id, processing)) => {
                match persistence::meeting_ops::apply_synced_lifecycle_if_present(
                    &meetings_dir,
                    meeting_id,
                    processing,
                )
                .await
                {
                    // Applied; `apply_processing_lifecycle` logs the transition.
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
            }
            Err(broadcast::error::RecvError::Lagged(dropped)) => {
                tracing::warn!(
                    target: "ipc-bridge",
                    dropped,
                    "lifecycle-event subscriber lagged; dropped states recover only on a re-run discovery (no scheduled caller yet)"
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
    rx: broadcast::Receiver<(MeetingId, ProcessingLifecycle)>,
    meetings_dir: PathBuf,
) {
    tauri::async_runtime::spawn(run_lifecycle_subscriber(rx, meetings_dir));
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::HostRef;

    /// The subscriber persists a present meeting's synced lifecycle, skips an
    /// absent one without dying, and exits when the sender is dropped.
    #[tokio::test]
    async fn persists_present_skips_absent_and_exits_on_close() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path().to_path_buf();

        // A meeting we hold (placeholder seeded → processing = Local).
        let present = MeetingId::new();
        persistence::MeetingFolder::ensure(&root, present).expect("seed present meeting");

        let (tx, rx) = broadcast::channel(16);
        let handle = tokio::spawn(run_lifecycle_subscriber(rx, root.clone()));

        // An advertisement for a meeting we do not hold: must be skipped, not fatal.
        tx.send((MeetingId::new(), ProcessingLifecycle::PendingProcessing))
            .expect("send absent");

        // An advertisement for the meeting we hold: must be applied.
        let processed = ProcessingLifecycle::Processed {
            processed_by: HostRef("endpoint-xyz".to_string()),
            at: "2026-06-27T10:25:00Z".to_string(),
        };
        tx.send((present, processed.clone())).expect("send present");

        // Dropping the sender closes the channel; the loop drains then exits.
        drop(tx);
        handle.await.expect("subscriber task joins");

        let meta = persistence::read_metadata(&root.join(present.0.to_string()))
            .expect("read present metadata");
        assert_eq!(meta.processing, processed);
    }
}
