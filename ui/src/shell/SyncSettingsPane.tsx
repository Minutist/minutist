/**
 * Sync settings pane (WS4-B S5), including account sign-in (WS4-A S5b).
 *
 * Signing in to a Minutist account is the ONLY way cross-device sync turns
 * on — a "Log in" button drives a device-code pairing and a status line
 * reflects whether this device is signed in. There is no separate enable
 * toggle, no pairing code shown when the browser URL that opens already
 * carries it (`code_required`), and no manual device-to-device ticket
 * pairing — every device on the account syncs with this one automatically
 * once signed in.
 *
 * Also shows the live sync engine status (from `sync_status`) and a 'Sync
 * now' action (scoped to the open meeting when available).
 *
 * Honest about the channel (D4): sync is end-to-end between the user's own
 * devices. It is unrelated to the local MCP server (a separate,
 * non-account-gated feature under Connections below), which is the channel
 * that transits meeting content to an external agent's vendor by design (D5).
 *
 * Rendered in the Editorial Ink language using `theme.css` tokens only.
 * Present only in the connected build (VITE_CONNECTED-gated in SettingsDrawer).
 */
import { useEffect, useState } from "react";
import { useAccountStatusStore } from "../state/account-status";
import { useSyncStatusStore } from "../state/sync-status";
import { useMeetingsStore } from "../state/meetings";
import type { AccountStatus, SyncStatus } from "../ipc/bindings";

/** Plain-English label for each account sign-in state. */
function accountStatusLabel(status: AccountStatus): string {
  switch (status) {
    case "signed_out":
      return "Not signed in";
    case "pairing":
      return "Signing in — approve in your browser";
    case "signed_in":
      return "Signed in";
  }
}

/** Plain-English label for each sync engine state. */
function statusLabel(status: SyncStatus): string {
  switch (status.kind) {
    case "disabled":
      return "Disabled";
    case "idle":
      return "Ready";
    case "connecting":
      return "Connecting…";
    case "syncing":
      return "Syncing…";
    case "error":
      return `Error: ${status.message}`;
  }
}

export function SyncSettingsPane() {
  const accountSnapshot = useAccountStatusStore((s) => s.snapshot);
  const userCode = useAccountStatusStore((s) => s.userCode);
  const codeRequired = useAccountStatusStore((s) => s.codeRequired);
  const accountError = useAccountStatusStore((s) => s.lastError);
  const refreshAccount = useAccountStatusStore((s) => s.refresh);
  const beginPairing = useAccountStatusStore((s) => s.beginPairing);
  const deleteAccount = useAccountStatusStore((s) => s.deleteAccount);

  const status = useSyncStatusStore((s) => s.status);
  const inProgress = useSyncStatusStore((s) => s.inProgress);
  const lastError = useSyncStatusStore((s) => s.lastError);
  const refresh = useSyncStatusStore((s) => s.refresh);
  const syncNow = useSyncStatusStore((s) => s.syncNow);

  // The currently open meeting, for the per-meeting 'Sync now' action.
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);

  const [syncNowPending, setSyncNowPending] = useState(false);

  // Fetch status on mount so the pane opens with the live state.
  useEffect(() => {
    void refreshAccount();
    void refresh();
  }, [refreshAccount, refresh]);

  const handleSyncNow = async () => {
    if (!openMeetingId) return;
    setSyncNowPending(true);
    try {
      await syncNow(openMeetingId);
    } finally {
      setSyncNowPending(false);
    }
  };

  const accountState = accountSnapshot?.status ?? "signed_out";
  const account = accountSnapshot?.account_id ?? null;
  const signedIn = account !== null;

  async function confirmDeleteAccount() {
    if (
      !confirm(
        "Delete your Minutist account? This erases your account and every " +
          "paired device from the server — your email, sign-in credentials, and " +
          "this device's pairing. Your local meetings stay on this computer. " +
          "This cannot be undone.",
      )
    )
      return;
    await deleteAccount();
  }

  return (
    <section className="settings-drawer__group" aria-label="Sync">
      <h3 className="settings-drawer__group-title">Sync</h3>

      <div className="settings-drawer__field" aria-label="Account">
        <label>Account</label>
        <div className="settings-drawer__mcp-endpoint">
          <code>{accountStatusLabel(accountState)}</code>
        </div>
        {signedIn && (
          <p className="settings-drawer__hint">
            Signed in as <code>{account}</code>. Every device signed in to
            this account syncs with this one automatically.
          </p>
        )}

        {accountState === "pairing" && userCode && codeRequired && (
          <div className="settings-drawer__field" aria-label="Pairing code">
            <label>Pairing code</label>
            <div className="settings-drawer__mcp-endpoint">
              <code className="settings-drawer__mcp-token">{userCode}</code>
            </div>
            <p className="settings-drawer__hint">
              A browser window should have opened. Sign in to your Minutist
              account and enter the code above to finish.
            </p>
          </div>
        )}
        {accountState === "pairing" && userCode && !codeRequired && (
          <p className="settings-drawer__hint">
            A browser window should have opened with the pairing already
            filled in. Sign in to your Minutist account to finish.
          </p>
        )}

        <div className="settings-drawer__field">
          {signedIn ? (
            <button
              type="button"
              className="settings-drawer__about settings-drawer__about--danger"
              onClick={() => void confirmDeleteAccount()}
            >
              Delete account
            </button>
          ) : (
            <button
              type="button"
              className="settings-drawer__about"
              disabled={accountState === "pairing"}
              onClick={() => void beginPairing()}
            >
              {accountState === "pairing" ? "Signing in…" : "Log in"}
            </button>
          )}
        </div>

        {accountError && (
          <p className="settings-drawer__hint settings-drawer__hint--warn">
            {accountError}
          </p>
        )}
      </div>

      <div className="settings-drawer__field" aria-label="Sync status">
        <label>Status</label>
        <div className="settings-drawer__mcp-endpoint">
          <code>{status !== null ? statusLabel(status) : "—"}</code>
        </div>
        <p className="settings-drawer__hint">
          Every device signed in to your account (above) syncs with this one
          automatically.
        </p>
        {inProgress && (
          <p className="settings-drawer__hint" aria-label="Sync progress">
            {inProgress.label}
            {inProgress.fraction !== null
              ? ` ${Math.round(inProgress.fraction * 100)}%`
              : ""}
          </p>
        )}
      </div>

      {openMeetingId && (
        <div className="settings-drawer__field">
          <button
            type="button"
            className="settings-drawer__about"
            disabled={syncNowPending || status?.kind === "disabled"}
            onClick={() => void handleSyncNow()}
          >
            {syncNowPending ? "Syncing…" : "Sync this meeting now"}
          </button>
          <p className="settings-drawer__hint">
            Pushes and pulls notes for the open meeting to all paired devices.
            Progress appears in the meeting list row while the transfer runs.
          </p>
        </div>
      )}

      {lastError && (
        <p className="settings-drawer__hint settings-drawer__hint--warn">
          {lastError}
        </p>
      )}
    </section>
  );
}
