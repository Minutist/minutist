/**
 * Account sign-in live-state store (WS4-A S5b).
 *
 * Holds the account snapshot (live `AccountStatus` + signed-in account, if
 * any) sourced from the `account_status` command, and drives the device-code
 * pairing flow:
 *
 *  - `beginPairing()` calls `account_begin_pairing`, surfaces the `user_code`,
 *    opens the verification URL in the browser (via `tauri-plugin-opener`), and
 *    starts polling `account_poll_pairing` until the status leaves `pairing`.
 *    A successful pairing also turns sync on — there is no separate enable
 *    toggle.
 *  - `refresh()` re-reads `account_status`.
 *  - `deleteAccount()` erases the account and signs this device out.
 *
 * The backend emits `AppEvent::AccountStatusChanged { status }` as the pairing
 * transitions; routing that through the global dispatcher into this store lets
 * the pane reflect `pairing → signed_in` live without polling. The event
 * carries only the status — the account label comes from `account_status` (it
 * is the non-secret rauthy `sub`); the device credential never crosses to the
 * webview.
 */
import { create } from "zustand";
import { commands, unwrap } from "../ipc/client";
import type { AccountSnapshot, AccountStatus } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

/** Default poll interval (ms) while a pairing is in progress (RFC 8628 floor). */
const PAIRING_POLL_INTERVAL_MS = 5000;

export type AccountStatusStore = {
  /** The live account snapshot, or `null` before the first fetch. */
  snapshot: AccountSnapshot | null;
  /**
   * The short code the user enters at the verification page, set while a pairing
   * is in progress. `null` when not pairing.
   */
  userCode: string | null;
  /**
   * Whether the code still needs to be typed by hand — false when the opened
   * URL already carries it pre-filled (`verification_uri_complete`), in which
   * case the pane must not show the code or claim it needs confirming.
   */
  codeRequired: boolean;
  /** A human-readable error from the last pairing/delete action, or `null`. */
  lastError: string | null;
  /** Re-fetch the snapshot via `account_status`. */
  refresh: () => Promise<void>;
  /** Begin device pairing: show the code, open the URL, and poll to completion. */
  beginPairing: () => Promise<void>;
  /**
   * Erase the paired account and sign this device out via `delete_account`.
   * On success the account is gone server-side and the local credential is
   * forgotten; the snapshot flips to the signed-out state.
   */
  deleteAccount: () => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

/** Patch the snapshot's `status` in place (preserving the account label). */
function withStatus(
  snapshot: AccountSnapshot | null,
  status: AccountStatus,
): AccountSnapshot {
  if (snapshot === null) {
    return { status, account_id: null };
  }
  return { ...snapshot, status };
}

export const useAccountStatusStore = create<AccountStatusStore>((set, get) => ({
  snapshot: null,
  userCode: null,
  codeRequired: false,
  lastError: null,

  refresh: async () => {
    try {
      const snapshot = unwrap(await commands.accountStatus());
      set({ snapshot });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  beginPairing: async () => {
    set({ lastError: null });
    let prompt;
    try {
      prompt = unwrap(await commands.accountBeginPairing());
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
      return;
    }

    // Optimistically reflect the pairing state so the pane shows the code +
    // instructions immediately; the backend confirms via the
    // account_status_changed(Pairing) event and the poll loop below.
    set((s) => ({
      userCode: prompt.user_code,
      codeRequired: prompt.code_required,
      snapshot: withStatus(s.snapshot, "pairing"),
    }));

    // Open the verification URL in the user's browser. Best-effort: the code is
    // shown regardless, so a failed open is not fatal (the user can still type
    // the code at the URL manually).
    try {
      const { openUrl } = await import("@tauri-apps/plugin-opener");
      await openUrl(prompt.verification_uri);
    } catch (err) {
      // Non-fatal — surface a hint but keep polling.
      console.warn("[account] failed to open verification URL:", err);
    }

    // Poll until the status leaves `pairing`. A bounded, self-terminating loop:
    // it stops when authorised (signed_in) or declined/expired (signed_out).
    // The status events also flow through handleEvent, but polling is what
    // advances the backend pairing state.
    const poll = async () => {
      let status: AccountStatus;
      try {
        status = unwrap(await commands.accountPollPairing());
      } catch (err) {
        set({
          lastError: err instanceof Error ? err.message : String(err),
          userCode: null,
          codeRequired: false,
        });
        return;
      }
      set((s) => ({ snapshot: withStatus(s.snapshot, status) }));
      if (status === "pairing") {
        setTimeout(() => void poll(), PAIRING_POLL_INTERVAL_MS);
      } else {
        // Terminal: clear the displayed code and refresh the full snapshot so
        // the account label appears.
        set({ userCode: null, codeRequired: false });
        void get().refresh();
      }
    };
    setTimeout(() => void poll(), PAIRING_POLL_INTERVAL_MS);
  },

  deleteAccount: async () => {
    try {
      unwrap(await commands.deleteAccount());
      // Erased server-side and locally forgotten: reflect the signed-out state.
      set({
        snapshot: { status: "signed_out", account_id: null },
        userCode: null,
        codeRequired: false,
        lastError: null,
      });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  handleEvent: (event) => {
    if (event.kind !== "account_status_changed") return;
    set((s) => ({ snapshot: withStatus(s.snapshot, event.status) }));
  },
}));
