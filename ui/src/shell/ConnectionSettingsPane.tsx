/**
 * Connection (connector / relay tunnel) settings pane (WS4-A S5b).
 *
 * Lets the user pair this device to their Minutist account and turn the
 * connector on/off. When on and paired, the app dials the hosted relay so an
 * external assistant (Claude web/Desktop, ChatGPT, Codex) you have connected can
 * read your meetings through it.
 *
 * Honest about the boundary (D5): the connector channel transits your meeting
 * content to the assistant's vendor — that is the point of it — so the copy
 * never calls it private. The sync channel (a separate feature) is the one that
 * is end-to-end; these must not be conflated.
 *
 * Rendered in the Editorial Ink language using `theme.css` tokens only. Present
 * only in the connected build (VITE_CONNECTED-gated in SettingsDrawer).
 */
import { useEffect } from "react";
import { useTunnelStatusStore } from "../state/tunnel-status";
import type { TunnelStatus } from "../ipc/bindings";

/** Plain-English label for each tunnel state. */
function statusLabel(status: TunnelStatus): string {
  switch (status) {
    case "disconnected":
      return "Not connected";
    case "pairing":
      return "Pairing — approve in your browser";
    case "connecting":
      return "Connecting…";
    case "online":
      return "Online";
    case "needs_repair":
      return "Disconnected — this device was unpaired; pair again";
  }
}

export function ConnectionSettingsPane() {
  const snapshot = useTunnelStatusStore((s) => s.snapshot);
  const userCode = useTunnelStatusStore((s) => s.userCode);
  const lastError = useTunnelStatusStore((s) => s.lastError);
  const refresh = useTunnelStatusStore((s) => s.refresh);
  const setEnabled = useTunnelStatusStore((s) => s.setEnabled);
  const beginPairing = useTunnelStatusStore((s) => s.beginPairing);

  // Fetch the snapshot on mount so the pane reflects the live state. Status
  // changes after that arrive on the event bus (tunnel_status_changed).
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const enabled = snapshot?.enabled ?? false;
  const status = snapshot?.status ?? "disconnected";
  const account = snapshot?.account_id ?? null;
  const paired = account !== null;

  return (
    <section className="settings-drawer__group" aria-label="Connection">
      <h3 className="settings-drawer__group-title">Connection</h3>

      <label
        className="settings-drawer__toggle"
        title="Dial the Minutist relay so an external assistant you have connected (Claude, ChatGPT, Codex) can read this app's meetings over it. This sends meeting content to that assistant's vendor by design — it is not a private channel. Off by default; needs a paired device."
      >
        <input
          type="checkbox"
          checked={enabled}
          disabled={snapshot === null}
          onChange={(e) => void setEnabled(e.target.checked)}
        />
        <span>Enable the connector</span>
      </label>

      <div className="settings-drawer__field" aria-label="Connection status">
        <label>Status</label>
        <div className="settings-drawer__mcp-endpoint">
          <code>{statusLabel(status)}</code>
        </div>
        {paired && (
          <p className="settings-drawer__hint">
            Paired to account <code>{account}</code>.
          </p>
        )}
      </div>

      {status === "pairing" && userCode && (
        <div className="settings-drawer__field" aria-label="Pairing code">
          <label>Pairing code</label>
          <div className="settings-drawer__mcp-endpoint">
            <code className="settings-drawer__mcp-token">{userCode}</code>
          </div>
          <p className="settings-drawer__hint">
            A browser window should have opened. Sign in to your Minutist account
            and confirm the code above to finish pairing.
          </p>
        </div>
      )}

      <div className="settings-drawer__field">
        <button
          type="button"
          className="settings-drawer__about"
          disabled={status === "pairing"}
          onClick={() => void beginPairing()}
        >
          {paired ? "Pair this device again" : "Pair this device"}
        </button>
      </div>

      <p className="settings-drawer__hint">
        The connector sends the meeting content you ask the assistant about to
        that assistant's vendor — it is not a private channel. It is separate
        from sync, which is end-to-end. The connector stays off until you enable
        it and pair this device.
      </p>

      {lastError && (
        <p className="settings-drawer__hint settings-drawer__hint--warn">
          {lastError}
        </p>
      )}
    </section>
  );
}
