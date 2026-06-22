/**
 * Sync settings pane (WS4-B S5).
 *
 * Shows the user this device's shareable pairing ticket (copyable), a field +
 * button to paste a peer's ticket and call `sync_add_peer`, and a 'Sync now'
 * action (scoped to the open meeting when available, or a global trigger
 * otherwise). The live engine status comes from `sync_status` via the store.
 *
 * Honest about the channel (D4): the sync channel is end-to-end between the
 * user's own paired devices. It is distinct from the connector channel, which
 * transits meeting content to the AI vendor by design and is never called
 * private.
 *
 * Rendered in the Editorial Ink language using `theme.css` tokens only.
 * Present only in the connected build (VITE_CONNECTED-gated in SettingsDrawer).
 */
import { useEffect, useState } from "react";
import { useSyncStatusStore } from "../state/sync-status";
import { useMeetingsStore } from "../state/meetings";
import type { SyncStatus } from "../ipc/bindings";

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
  const status = useSyncStatusStore((s) => s.status);
  const inProgress = useSyncStatusStore((s) => s.inProgress);
  const myTicket = useSyncStatusStore((s) => s.myTicket);
  const lastError = useSyncStatusStore((s) => s.lastError);
  const refresh = useSyncStatusStore((s) => s.refresh);
  const fetchTicket = useSyncStatusStore((s) => s.fetchTicket);
  const addPeer = useSyncStatusStore((s) => s.addPeer);
  const syncNow = useSyncStatusStore((s) => s.syncNow);

  // The currently open meeting, for the per-meeting 'Sync now' action.
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);

  const [peerTicket, setPeerTicket] = useState("");
  const [addPeerPending, setAddPeerPending] = useState(false);
  const [syncNowPending, setSyncNowPending] = useState(false);

  // Fetch status on mount so the pane opens with the live state. The ticket is
  // expensive to fetch (it involves the iroh endpoint); defer it to the user
  // explicitly revealing this section.
  useEffect(() => {
    void refresh();
    void fetchTicket();
  }, [refresh, fetchTicket]);

  const copy = (text: string) => {
    void navigator.clipboard?.writeText(text);
  };

  const handleAddPeer = async () => {
    const t = peerTicket.trim();
    if (!t) return;
    setAddPeerPending(true);
    try {
      await addPeer(t);
      setPeerTicket("");
    } finally {
      setAddPeerPending(false);
    }
  };

  const handleSyncNow = async () => {
    if (!openMeetingId) return;
    setSyncNowPending(true);
    try {
      await syncNow(openMeetingId);
    } finally {
      setSyncNowPending(false);
    }
  };

  return (
    <section className="settings-drawer__group" aria-label="Sync">
      <h3 className="settings-drawer__group-title">Sync</h3>

      <div className="settings-drawer__field" aria-label="Sync status">
        <label>Status</label>
        <div className="settings-drawer__mcp-endpoint">
          <code>{status !== null ? statusLabel(status) : "—"}</code>
        </div>
        {inProgress && (
          <p className="settings-drawer__hint" aria-label="Sync progress">
            {inProgress.label}
            {inProgress.fraction !== null
              ? ` ${Math.round(inProgress.fraction * 100)}%`
              : ""}
          </p>
        )}
      </div>

      {myTicket && (
        <div className="settings-drawer__field" aria-label="This device's ticket">
          <label>This device's ticket</label>
          <div className="settings-drawer__mcp-endpoint">
            <code className="settings-drawer__mcp-url">{myTicket}</code>
            <button
              type="button"
              className="settings-drawer__about"
              onClick={() => copy(myTicket)}
            >
              Copy
            </button>
          </div>
          <p className="settings-drawer__hint">
            On another device, open Settings → Sync and paste this ticket into
            the "Add a peer device" field. Sync is end-to-end between your own
            devices.
          </p>
        </div>
      )}

      <div className="settings-drawer__field" aria-label="Add a peer device">
        <label htmlFor="settings-sync-peer-ticket">Add a peer device</label>
        <div className="settings-drawer__mcp-endpoint">
          <input
            id="settings-sync-peer-ticket"
            type="text"
            className="settings-drawer__mcp-url"
            placeholder="Paste ticket from the other device"
            value={peerTicket}
            onChange={(e) => setPeerTicket(e.target.value)}
            disabled={addPeerPending}
            aria-label="Peer ticket"
          />
          <button
            type="button"
            className="settings-drawer__about"
            disabled={!peerTicket.trim() || addPeerPending}
            onClick={() => void handleAddPeer()}
          >
            {addPeerPending ? "Adding…" : "Add"}
          </button>
        </div>
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
