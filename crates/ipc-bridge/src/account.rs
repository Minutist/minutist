//! The connected-tier account sign-in IPC surface (WS4-A S5b).
//!
//! Signing in exists to enable cross-device sync (D4) — a device-code pairing,
//! after which every device on the account syncs with every other. It is
//! unrelated to the local MCP server (`mcp-server`), which is not account-gated
//! and is the channel that actually transits meeting content to an external
//! agent's vendor (D5).
//!
//! The webview drives sign-in through these commands; the live state rides
//! `AppEvent::AccountStatusChanged` on the existing event bus (no second event
//! registration).
//!
//! # Why a trait seam
//!
//! `ipc-bridge` must NOT depend on `tunnel-client` (the dependency table keeps
//! it a near-leaf). The actual pairing logic lives in `tunnel-client` and is
//! wired in `app-main` behind the `connected` feature. So this module defines
//! [`AccountControl`] — an async trait the command handlers call — and
//! `app-main` injects a connected implementation that holds the
//! `tunnel-client` types. The free build (and any connected build before a
//! credential is stored) gets [`DisabledAccount`], a no-op that reports
//! `SignedOut`, so the commands compile and behave gracefully with no account
//! service present.
//!
//! The trait object lives in [`IpcState::account`](crate::IpcState). The
//! commands never touch `tunnel-client` types directly — only
//! [`AccountControl`] and the IPC-facing shapes below.

use std::sync::Arc;

use async_trait::async_trait;
use minutist_common::{AccountStatus, AppError, AppResult};
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::State;

use crate::IpcState;

/// The codes + URL to show the user when a pairing begins. Returned by
/// [`account_begin_pairing`]. The webview opens `verification_uri` in the
/// browser (via `tauri-plugin-opener`) and displays `user_code` only when
/// `code_required` is true.
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
    /// Whether `verification_uri` still requires the user to type
    /// `user_code` themselves. False when the URL already carries the code
    /// pre-filled (the server sent `verification_uri_complete`) — showing the
    /// code in that case describes a step that never happens.
    pub code_required: bool,
}

/// A snapshot of this device's sign-in state for the Settings → Sync pane.
/// Returned by [`account_status`]. `account_id` is shown so the user can
/// confirm which account the device is signed in to; it is not a secret (it is
/// the rauthy `sub`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Type)]
pub struct AccountSnapshot {
    /// The live sign-in state.
    pub status: AccountStatus,
    /// The account this device is signed in to, if any. `None` when signed
    /// out. Never carries the credential itself.
    pub account_id: Option<String>,
}

/// The control surface `app-main` injects so the IPC commands can drive
/// account sign-in without `ipc-bridge` depending on `tunnel-client`.
///
/// All methods are `async` (they perform network I/O). The implementation is
/// responsible for emitting `AppEvent::AccountStatusChanged` as the status
/// moves; the commands only request transitions and read the snapshot.
#[async_trait]
pub trait AccountControl: Send + Sync {
    /// Begin a device-code pairing: request a code from the account-service and
    /// return the prompt to show the user. The implementation moves the status
    /// to `Pairing` and emits the change.
    async fn begin_pairing(&self) -> AppResult<PairingPrompt>;

    /// Poll the in-progress pairing once. Returns the current status:
    /// `Pairing` while the user has not approved, `SignedIn` once the
    /// credential is issued, or `SignedOut` if pairing was declined / expired
    /// (the implementation surfaces the terminal reason via the status + a
    /// log; a fresh `begin_pairing` restarts it). On a successful authorisation
    /// the implementation stores the credential securely.
    async fn poll_pairing(&self) -> AppResult<AccountStatus>;

    /// The current snapshot (live status + signed-in account, if any).
    async fn snapshot(&self) -> AccountSnapshot;

    /// Erase the paired account and sign this device out (GDPR Art 17).
    ///
    /// Calls the account-service erase endpoint (`DELETE /v1/account`) with the
    /// stored device credential, then — only once the server confirms the erase
    /// (or reports the account already gone) — forgets the local credential and
    /// moves the status to `SignedOut`, emitting the change. A failed server
    /// call leaves the device signed in and the operation retryable. In the
    /// free build (no credential) this is `Unsupported`.
    async fn delete_account(&self) -> AppResult<()>;
}

/// The no-op account control used by the free build and by a connected build
/// with no account wiring. Always reports `SignedOut` and rejects pairing as
/// unsupported, so the commands behave gracefully with no account service
/// present.
pub struct DisabledAccount;

#[async_trait]
impl AccountControl for DisabledAccount {
    async fn begin_pairing(&self) -> AppResult<PairingPrompt> {
        Err(AppError::Unsupported {
            context: "account sign-in is only available in the connected build".to_string(),
        })
    }

    async fn poll_pairing(&self) -> AppResult<AccountStatus> {
        Ok(AccountStatus::SignedOut)
    }

    async fn snapshot(&self) -> AccountSnapshot {
        AccountSnapshot {
            status: AccountStatus::SignedOut,
            account_id: None,
        }
    }

    async fn delete_account(&self) -> AppResult<()> {
        Err(AppError::Unsupported {
            context: "there is no connected account to delete in this build".to_string(),
        })
    }
}

/// The default account control: a [`DisabledAccount`] behind an `Arc`.
/// `app-main` constructs `IpcState.account` with this in the free build (and
/// as the initial value in the connected build before a real implementation
/// is injected).
pub fn disabled_account() -> Arc<dyn AccountControl> {
    Arc::new(DisabledAccount)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Begin device pairing. Returns the `user_code` + verification URL for the UI
/// to display and open in the browser. The user approves in rauthy; the UI then
/// calls [`account_poll_pairing`] until the status is terminal.
///
/// In the free build this returns an `Unsupported` error — the account UI is
/// absent from that bundle, so the command is never invoked there.
#[tauri::command]
#[specta::specta]
pub async fn account_begin_pairing(state: State<'_, IpcState>) -> AppResult<PairingPrompt> {
    state.connected.account.begin_pairing().await
}

/// Poll the in-progress pairing once, returning the current [`AccountStatus`].
/// The UI polls this on the server's `interval` until the status leaves
/// `Pairing` (reaching `SignedIn` on success, or `SignedOut` on a
/// declined/expired pairing).
#[tauri::command]
#[specta::specta]
pub async fn account_poll_pairing(state: State<'_, IpcState>) -> AppResult<AccountStatus> {
    state.connected.account.poll_pairing().await
}

/// The account snapshot for the Settings → Sync pane: the live sign-in status
/// and the signed-in account (if any).
#[tauri::command]
#[specta::specta]
pub async fn account_status(state: State<'_, IpcState>) -> AppResult<AccountSnapshot> {
    Ok(state.connected.account.snapshot().await)
}

/// Erase the paired account and sign out (GDPR Art 17). `DELETE /v1/account` on
/// the account-service erases the rauthy identity (email + credentials) and
/// every device row; the local device credential is then forgotten. The sync
/// engine is stopped alongside (best-effort). On success the device is fully
/// signed out and the account is gone server-side; a failure leaves the device
/// signed in and the call retryable.
#[tauri::command]
#[specta::specta]
pub async fn delete_account(state: State<'_, IpcState>) -> AppResult<()> {
    state.connected.account.delete_account().await?;
    if let Err(e) = state.connected.sync.set_enabled(false).await {
        tracing::warn!(
            target: "ipc-bridge",
            error = ?e,
            "delete_account: stopping the sync engine failed (best-effort)"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disabled_account_reports_signed_out_and_rejects_pairing() {
        let a = DisabledAccount;
        assert!(matches!(
            a.begin_pairing().await,
            Err(AppError::Unsupported { .. })
        ));
        assert_eq!(a.poll_pairing().await.unwrap(), AccountStatus::SignedOut);
        assert!(matches!(
            a.delete_account().await,
            Err(AppError::Unsupported { .. })
        ));
        let snap = a.snapshot().await;
        assert_eq!(snap.status, AccountStatus::SignedOut);
        assert!(snap.account_id.is_none());
    }

    #[tokio::test]
    async fn disabled_account_helper_constructs_a_signed_out_control() {
        let a = disabled_account();
        assert_eq!(a.snapshot().await.status, AccountStatus::SignedOut);
    }
}
