//! The connected-tier relay tunnel IPC surface (WS4-A S5b).
//!
//! The webview drives device pairing and the connector toggle through these
//! commands; the tunnel's live state rides `AppEvent::TunnelStatusChanged` on
//! the existing event bus (no second event registration).
//!
//! # Why a trait seam
//!
//! `ipc-bridge` must NOT depend on `tunnel-client` (the dependency table keeps
//! the tunnel crate a near-leaf). The actual pairing + reconnect + lifecycle
//! logic lives in `tunnel-client` and is wired in `app-main` behind the
//! `connected` feature. So this module defines [`TunnelControl`] — an async
//! trait the command handlers call — and `app-main` injects a connected
//! implementation that holds the `tunnel-client` types. The free build (and any
//! connected build before a credential is stored) gets [`DisabledTunnel`], a
//! no-op that reports `Disconnected`, so the commands compile and behave
//! gracefully with no relay present.
//!
//! The trait object lives in [`IpcState::tunnel`](crate::IpcState). The commands
//! never touch `tunnel-client` types directly — only [`TunnelControl`] and the
//! IPC-facing shapes below.

use std::sync::Arc;

use async_trait::async_trait;
use minutist_common::TunnelStatus;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::State;

use crate::{IpcError, IpcState};

/// The codes + URL to show the user when a pairing begins. Returned by
/// [`tunnel_begin_pairing`]. The webview displays `user_code` and opens
/// `verification_uri` in the browser (via `tauri-plugin-opener`).
///
/// Carries no credential — the issued device credential is stored securely by
/// `app-main` on a successful poll and never crosses to the webview.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Type)]
pub struct PairingPrompt {
    /// The short code the user types into the verification page.
    pub user_code: String,
    /// The URL to open in the browser — the pre-filled one when the server
    /// provided it, else the bare verification URL.
    pub verification_uri: String,
}

/// A snapshot of the connector for the Settings → Connection pane. Returned by
/// [`tunnel_status`]. The `account_id` is shown so the user can confirm which
/// account the device is paired to; it is not a secret (it is the rauthy `sub`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Type)]
pub struct TunnelSnapshot {
    /// Whether the user has enabled the connector (`settings.connector_enabled`).
    pub enabled: bool,
    /// The live tunnel state.
    pub status: TunnelStatus,
    /// The account the device is paired to, if a credential is stored. `None`
    /// when unpaired. Never carries the credential itself.
    pub account_id: Option<String>,
}

/// The control surface `app-main` injects so the IPC commands can drive the
/// tunnel without `ipc-bridge` depending on `tunnel-client`.
///
/// All methods are `async` (they perform network I/O or await a lifecycle task).
/// The implementation is responsible for emitting `AppEvent::TunnelStatusChanged`
/// as the status moves; the commands only request transitions and read the
/// snapshot.
#[async_trait]
pub trait TunnelControl: Send + Sync {
    /// Begin a device-code pairing: request a code from the account-service and
    /// return the prompt to show the user. The implementation moves the status to
    /// `Pairing` and emits the change.
    async fn begin_pairing(&self) -> Result<PairingPrompt, IpcError>;

    /// Poll the in-progress pairing once. Returns the current status: `Pairing`
    /// while the user has not approved, `Online`/`Connecting` once the credential
    /// is issued and the tunnel is (re)dialed, or `Disconnected` if pairing was
    /// declined / expired (the implementation surfaces the terminal reason via the
    /// status + a log; a fresh `begin_pairing` restarts it). On a successful
    /// authorisation the implementation stores the credential securely and starts
    /// the tunnel lifecycle.
    async fn poll_pairing(&self) -> Result<TunnelStatus, IpcError>;

    /// Enable or disable the connector. Enabling with a stored credential starts
    /// the tunnel lifecycle; disabling stops it cleanly. Persists the choice via
    /// settings. Returns the resulting snapshot.
    async fn set_enabled(&self, enabled: bool) -> Result<TunnelSnapshot, IpcError>;

    /// The current snapshot (enabled flag + live status + paired account).
    async fn snapshot(&self) -> TunnelSnapshot;
}

/// The no-op tunnel control used by the free build and by a connected build with
/// no relay wiring. Always reports `Disconnected` and rejects pairing as
/// unsupported, so the commands behave gracefully with no relay present.
pub struct DisabledTunnel;

#[async_trait]
impl TunnelControl for DisabledTunnel {
    async fn begin_pairing(&self) -> Result<PairingPrompt, IpcError> {
        Err(IpcError::Unsupported {
            context: "device pairing is only available in the connected build".to_string(),
        })
    }

    async fn poll_pairing(&self) -> Result<TunnelStatus, IpcError> {
        Ok(TunnelStatus::Disconnected)
    }

    async fn set_enabled(&self, _enabled: bool) -> Result<TunnelSnapshot, IpcError> {
        Ok(TunnelSnapshot {
            enabled: false,
            status: TunnelStatus::Disconnected,
            account_id: None,
        })
    }

    async fn snapshot(&self) -> TunnelSnapshot {
        TunnelSnapshot {
            enabled: false,
            status: TunnelStatus::Disconnected,
            account_id: None,
        }
    }
}

/// The default tunnel control: a [`DisabledTunnel`] behind an `Arc`. `app-main`
/// constructs `IpcState.tunnel` with this in the free build (and as the initial
/// value in the connected build before a relay implementation is injected).
pub fn disabled_tunnel() -> Arc<dyn TunnelControl> {
    Arc::new(DisabledTunnel)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Begin device pairing. Returns the `user_code` + verification URL for the UI
/// to display and open in the browser. The user approves in rauthy; the UI then
/// calls [`tunnel_poll_pairing`] until the status is terminal.
///
/// In the free build (or before any relay wiring) this returns an `Unsupported`
/// error — the Connection pane is absent from that bundle, so the command is
/// never invoked there.
#[tauri::command]
#[specta::specta]
pub async fn tunnel_begin_pairing(state: State<'_, IpcState>) -> Result<PairingPrompt, IpcError> {
    state.tunnel.begin_pairing().await
}

/// Poll the in-progress pairing once, returning the current [`TunnelStatus`].
/// The UI polls this on the server's `interval` until the status leaves
/// `Pairing` (reaching `Connecting`/`Online` on success, or `Disconnected` on a
/// declined/expired pairing).
#[tauri::command]
#[specta::specta]
pub async fn tunnel_poll_pairing(state: State<'_, IpcState>) -> Result<TunnelStatus, IpcError> {
    state.tunnel.poll_pairing().await
}

/// Enable or disable the connector. Enabling with a stored credential starts the
/// tunnel; disabling stops it cleanly. Returns the resulting snapshot.
#[tauri::command]
#[specta::specta]
pub async fn set_connector_enabled(
    state: State<'_, IpcState>,
    enabled: bool,
) -> Result<TunnelSnapshot, IpcError> {
    state.tunnel.set_enabled(enabled).await
}

/// The connector snapshot for the Settings → Connection pane: the enabled flag,
/// the live tunnel status, and the paired account (if any).
#[tauri::command]
#[specta::specta]
pub async fn tunnel_status(state: State<'_, IpcState>) -> Result<TunnelSnapshot, IpcError> {
    Ok(state.tunnel.snapshot().await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disabled_tunnel_reports_disconnected_and_rejects_pairing() {
        let t = DisabledTunnel;
        assert!(matches!(
            t.begin_pairing().await,
            Err(IpcError::Unsupported { .. })
        ));
        assert_eq!(t.poll_pairing().await.unwrap(), TunnelStatus::Disconnected);
        let snap = t.snapshot().await;
        assert!(!snap.enabled);
        assert_eq!(snap.status, TunnelStatus::Disconnected);
        assert!(snap.account_id.is_none());
    }

    #[tokio::test]
    async fn disabled_tunnel_helper_constructs_a_disconnected_control() {
        let t = disabled_tunnel();
        assert_eq!(t.snapshot().await.status, TunnelStatus::Disconnected);
    }
}
