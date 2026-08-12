/**
 * Connector (relay tunnel) live-state store (WS4-A S5b).
 *
 * Holds the connector snapshot (enabled flag + live `TunnelStatus` + paired
 * account) sourced from the `tunnel_status` command, and drives the device-code
 * pairing flow:
 *
 *  - `beginPairing()` calls `tunnel_begin_pairing`, surfaces the `user_code`,
 *    opens the verification URL in the browser (via `tauri-plugin-opener`), and
 *    starts polling `tunnel_poll_pairing` until the status leaves `pairing`.
 *  - `setEnabled(enabled)` calls `set_connector_enabled` (which starts/stops the
 *    tunnel lifecycle and persists the setting) and stores the resulting
 *    snapshot.
 *  - `refresh()` re-reads `tunnel_status`.
 *
 * The backend emits `AppEvent::TunnelStatusChanged { status }` as the reconnect
 * loop / lifecycle transitions; routing that through the global dispatcher into
 * this store lets the pane reflect `Connecting → Online` (and `NeedsRepair`)
 * live without polling. The event carries only the status — the account label
 * comes from `tunnel_status` (it is the non-secret rauthy `sub`); the device
 * credential never crosses to the webview.
 */
import { create } from "zustand";
import { commands, unwrap } from "../ipc/client";
import type { TunnelSnapshot, TunnelStatus } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

/** Default poll interval (ms) while a pairing is in progress (RFC 8628 floor). */
const PAIRING_POLL_INTERVAL_MS = 5000;

export type TunnelStatusStore = {
  /** The live connector snapshot, or `null` before the first fetch. */
  snapshot: TunnelSnapshot | null;
  /**
   * The short code the user enters at the verification page, set while a pairing
   * is in progress. `null` when not pairing.
   */
  userCode: string | null;
  /** A human-readable error from the last pairing/toggle action, or `null`. */
  lastError: string | null;
  /** Re-fetch the snapshot via `tunnel_status`. */
  refresh: () => Promise<void>;
  /** Enable/disable the connector via `set_connector_enabled`. */
  setEnabled: (enabled: boolean) => Promise<void>;
  /** Begin device pairing: show the code, open the URL, and poll to completion. */
  beginPairing: () => Promise<void>;
  /**
   * Erase the paired account and sign this device out via `delete_account`.
   * On success the account is gone server-side and the local credential is
   * forgotten; the snapshot flips to the signed-out (unpaired) state.
   */
  deleteAccount: () => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

/** Patch the snapshot's `status` in place (preserving enabled/account). */
function withStatus(
  snapshot: TunnelSnapshot | null,
  status: TunnelStatus,
): TunnelSnapshot {
  if (snapshot === null) {
    return { enabled: false, status, account_id: null };
  }
  return { ...snapshot, status };
}

export const useTunnelStatusStore = create<TunnelStatusStore>((set, get) => ({
  snapshot: null,
  userCode: null,
  lastError: null,

  refresh: async () => {
    try {
      const snapshot = unwrap(await commands.tunnelStatus());
      set({ snapshot });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setEnabled: async (enabled) => {
    try {
      const snapshot = unwrap(await commands.setConnectorEnabled(enabled));
      set({ snapshot, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  beginPairing: async () => {
    set({ lastError: null });
    let prompt;
    try {
      prompt = unwrap(await commands.tunnelBeginPairing());
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
      return;
    }

    // Optimistically reflect the pairing state so the pane shows the code +
    // instructions immediately; the backend confirms via the
    // tunnel_status_changed(Pairing) event and the poll loop below.
    set((s) => ({
      userCode: prompt.user_code,
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
      console.warn("[connector] failed to open verification URL:", err);
    }

    // Poll until the status leaves `pairing`. A bounded, self-terminating loop:
    // it stops when authorised (Connecting/Online), declined/expired
    // (Disconnected), or NeedsRepair. The status events also flow through
    // handleEvent, but polling is what advances the backend pairing state.
    const poll = async () => {
      let status: TunnelStatus;
      try {
        status = unwrap(await commands.tunnelPollPairing());
      } catch (err) {
        set({
          lastError: err instanceof Error ? err.message : String(err),
          userCode: null,
        });
        return;
      }
      set((s) => ({ snapshot: withStatus(s.snapshot, status) }));
      if (status === "pairing") {
        setTimeout(() => void poll(), PAIRING_POLL_INTERVAL_MS);
      } else {
        // Terminal: clear the displayed code and refresh the full snapshot so
        // the account label appears.
        set({ userCode: null });
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
        snapshot: { enabled: false, status: "disconnected", account_id: null },
        userCode: null,
        lastError: null,
      });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  handleEvent: (event) => {
    if (event.kind !== "tunnel_status_changed") return;
    set((s) => ({ snapshot: withStatus(s.snapshot, event.status) }));
  },
}));
