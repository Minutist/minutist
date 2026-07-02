//! The peer-to-peer notes-sync IPC surface (WS4-B S5).
//!
//! The webview drives device pairing (exchanging shareable tickets) and manual
//! sync through these commands; the sync engine's live state and per-meeting
//! transfer progress ride `AppEvent::{SyncProgress, SyncReady, SyncError}` on
//! the existing event bus (no second event registration).
//!
//! # Why a trait seam
//!
//! `ipc-bridge` must NOT depend on `sync` (the dependency table keeps the sync
//! crate a near-leaf, and `sync` pulls in iroh). The actual endpoint + pairing +
//! notes-sync logic lives in `sync` and is wired in `app-main` behind the
//! `connected` feature. So this module defines [`SyncControl`] — an async trait
//! the command handlers call — and `app-main` injects a connected implementation
//! that holds the `sync` types. The free build (and any connected build before a
//! sync engine is started) gets [`DisabledSync`], a no-op that reports `Disabled`
//! and rejects ticket/peer/sync operations as unsupported, so the commands
//! compile and behave gracefully with no sync engine present.
//!
//! The trait object lives in [`IpcState::sync`](crate::IpcState). The commands
//! never touch `sync` types directly — only [`SyncControl`] and the IPC-facing
//! shapes below. This mirrors the connector tunnel seam in
//! [`crate::tunnel`].

use std::sync::Arc;

use async_trait::async_trait;
use minutist_common::{MeetingId, SyncStatus};

use crate::{IpcError, IpcState};
use tauri::State;

/// The control surface `app-main` injects so the IPC commands can drive the sync
/// engine without `ipc-bridge` depending on `sync`.
///
/// All methods are `async` (they perform network I/O or await the engine). The
/// implementation is responsible for emitting `AppEvent::{SyncProgress,
/// SyncReady, SyncError}` as transfers run; the commands only request actions and
/// read the status.
#[async_trait]
pub trait SyncControl: Send + Sync {
    /// The sync engine's current live state. Never errors — a disabled engine
    /// reports [`SyncStatus::Disabled`].
    async fn status(&self) -> SyncStatus;

    /// This device's shareable endpoint ticket: a string the user copies to
    /// another of their devices so it can dial this one. Carries this device's
    /// public addressing only — not its secret key. The peer feeds it to
    /// [`Self::add_peer`].
    async fn my_ticket(&self) -> Result<String, IpcError>;

    /// Register a peer device from the ticket it produced via [`Self::my_ticket`].
    /// After this, the two devices can sync notes with each other.
    async fn add_peer(&self, ticket: String) -> Result<(), IpcError>;

    /// Trigger a notes sync for one meeting with the paired peers. Progress and
    /// completion arrive on the event bus (`SyncProgress` / `SyncReady`), not the
    /// return value; an error here means the sync could not be started.
    async fn sync_now(&self, meeting_id: MeetingId) -> Result<(), IpcError>;

    /// Best-effort: unpin `meeting_id`'s media + derived-artifact blobs from the
    /// local blob store (so the bytes become GC-eligible) after its folder has
    /// already been deleted. Called from [`crate::commands::delete_meeting`]. A
    /// free build, or a connected build whose sync engine has not started, is a
    /// no-op — [`DisabledSync`] never errors here, since the meeting is gone
    /// either way and there is nothing more this call could accomplish.
    async fn delete_meeting_blobs(&self, meeting_id: MeetingId) -> Result<(), IpcError>;
}

/// The no-op sync control used by the free build and by a connected build with
/// no sync engine wired. Always reports `Disabled` and rejects ticket / peer /
/// sync operations as unsupported, so the commands behave gracefully with no
/// engine present.
pub struct DisabledSync;

/// The message every rejecting [`DisabledSync`] method returns, so the UI shows a
/// single consistent reason.
const SYNC_UNAVAILABLE: &str = "sync not available in this build";

#[async_trait]
impl SyncControl for DisabledSync {
    async fn status(&self) -> SyncStatus {
        SyncStatus::Disabled
    }

    async fn my_ticket(&self) -> Result<String, IpcError> {
        Err(IpcError::Unsupported {
            context: SYNC_UNAVAILABLE.to_string(),
        })
    }

    async fn add_peer(&self, _ticket: String) -> Result<(), IpcError> {
        Err(IpcError::Unsupported {
            context: SYNC_UNAVAILABLE.to_string(),
        })
    }

    async fn sync_now(&self, _meeting_id: MeetingId) -> Result<(), IpcError> {
        Err(IpcError::Unsupported {
            context: SYNC_UNAVAILABLE.to_string(),
        })
    }

    async fn delete_meeting_blobs(&self, _meeting_id: MeetingId) -> Result<(), IpcError> {
        // No blob store in this build — nothing to unpin. Never an error: the
        // meeting folder is already gone by the time this is called, so a free
        // build has genuinely finished the deletion.
        Ok(())
    }
}

/// The default sync control: a [`DisabledSync`] behind an `Arc`. `app-main`
/// constructs `IpcState.sync` with this in the free build (and as the initial
/// value in the connected build before a sync engine is injected).
pub fn disabled_sync() -> Arc<dyn SyncControl> {
    Arc::new(DisabledSync)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// The sync engine's current live status for the Settings → Sync pane.
#[tauri::command]
#[specta::specta]
pub async fn sync_status(state: State<'_, IpcState>) -> Result<SyncStatus, IpcError> {
    Ok(state.sync.status().await)
}

/// This device's shareable ticket string. The UI shows it (and/or a QR) so the
/// user can pair another of their devices, which calls [`sync_add_peer`] with it.
///
/// In the free build (or before any sync wiring) this returns an `Unsupported`
/// error — the Sync pane is absent from that bundle, so the command is never
/// invoked there.
#[tauri::command]
#[specta::specta]
pub async fn sync_get_my_ticket(state: State<'_, IpcState>) -> Result<String, IpcError> {
    state.sync.my_ticket().await
}

/// Register a peer device from its shareable ticket (produced by
/// [`sync_get_my_ticket`] on the other device).
#[tauri::command]
#[specta::specta]
pub async fn sync_add_peer(state: State<'_, IpcState>, ticket: String) -> Result<(), IpcError> {
    state.sync.add_peer(ticket).await
}

/// Trigger a notes sync for one meeting with the paired peers. Progress and
/// completion arrive on the event bus (`AppEvent::SyncProgress` / `SyncReady`).
#[tauri::command]
#[specta::specta]
pub async fn sync_now(state: State<'_, IpcState>, meeting_id: MeetingId) -> Result<(), IpcError> {
    state.sync.sync_now(meeting_id).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disabled_sync_reports_disabled_and_rejects_operations() {
        let s = DisabledSync;
        assert_eq!(s.status().await, SyncStatus::Disabled);
        assert!(matches!(
            s.my_ticket().await,
            Err(IpcError::Unsupported { .. })
        ));
        assert!(matches!(
            s.add_peer("ticket".to_string()).await,
            Err(IpcError::Unsupported { .. })
        ));
        assert!(matches!(
            s.sync_now(MeetingId::new()).await,
            Err(IpcError::Unsupported { .. })
        ));
        assert!(s.delete_meeting_blobs(MeetingId::new()).await.is_ok());
    }

    #[tokio::test]
    async fn disabled_sync_helper_constructs_a_disabled_control() {
        let s = disabled_sync();
        assert_eq!(s.status().await, SyncStatus::Disabled);
    }
}
