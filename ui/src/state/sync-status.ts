/**
 * Sync engine live-state store (WS4-B S5).
 *
 * Holds the sync engine's current status (sourced from `sync_status`), this
 * device's shareable ticket, and a pending-add-peer field. The backend emits
 * `AppEvent::SyncProgress / SyncReady / SyncError` while a transfer is
 * running; `SyncReady` also triggers a `sync_ready_toast` notification that
 * the `MainWindow` chrome strip renders.
 *
 * `refresh()` re-reads `sync_status`; called on mount so the pane opens with
 * the live state rather than the initial `null`. `fetchTicket()` fetches
 * `sync_get_my_ticket` on demand (once per session; cached thereafter).
 * `addPeer(ticket)` calls `sync_add_peer` and surfaces any error.
 * `syncNow(meetingId)` calls `sync_now`; progress + completion arrive on the
 * event bus, not in the return value.
 *
 * The sync channel is end-to-end between the user's own paired devices. It is
 * distinct from the connector channel, which transits content to the AI vendor
 * by design and is never called private.
 */
import { create } from "zustand";
import { commands, unwrap } from "../ipc/client";
import type { SyncStatus, MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type SyncStatusStore = {
  /** The live engine status, or `null` before the first fetch. */
  status: SyncStatus | null;
  /**
   * This device's shareable ticket string (the `sync_get_my_ticket` result),
   * or `null` when it has not been fetched yet.
   */
  myTicket: string | null;
  /** A human-readable error from the last action, or `null`. */
  lastError: string | null;
  /**
   * Pending notifications from `SyncReady` events: meeting ids for which a
   * "Synced changes from another device" toast should appear. Cleared by the
   * consumer once displayed.
   */
  pendingReadyNotifications: MeetingId[];
  /** Re-fetch `sync_status`. */
  refresh: () => Promise<void>;
  /** Fetch (and cache) this device's ticket via `sync_get_my_ticket`. */
  fetchTicket: () => Promise<void>;
  /** Register a peer device from its shareable ticket. */
  addPeer: (ticket: string) => Promise<void>;
  /** Trigger a notes sync for one meeting with paired peers. */
  syncNow: (meetingId: MeetingId) => Promise<void>;
  /** Dismiss a pending sync-ready toast by meeting id. */
  dismissReadyNotification: (meetingId: MeetingId) => void;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useSyncStatusStore = create<SyncStatusStore>((set) => ({
  status: null,
  myTicket: null,
  lastError: null,
  pendingReadyNotifications: [],

  refresh: async () => {
    try {
      const status = unwrap(await commands.syncStatus());
      set({ status });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  fetchTicket: async () => {
    try {
      const ticket = unwrap(await commands.syncGetMyTicket());
      set({ myTicket: ticket, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  addPeer: async (ticket) => {
    try {
      unwrap(await commands.syncAddPeer(ticket));
      set({ lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  syncNow: async (meetingId) => {
    try {
      unwrap(await commands.syncNow(meetingId));
      set({ lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  dismissReadyNotification: (meetingId) => {
    set((s) => ({
      pendingReadyNotifications: s.pendingReadyNotifications.filter(
        (id) => id !== meetingId,
      ),
    }));
  },

  handleEvent: (event) => {
    if (event.kind === "sync_progress") {
      // Progress events are consumed by the operation-progress store; the sync
      // store only needs to track terminal events.
      return;
    }
    if (event.kind === "sync_ready") {
      // Queue a toast notification and refresh the live status (a completed
      // transfer may have moved the engine from Syncing → Idle).
      set((s) => ({
        pendingReadyNotifications: [
          ...s.pendingReadyNotifications,
          event.meeting_id,
        ],
      }));
      return;
    }
    if (event.kind === "sync_error") {
      set({ lastError: event.context });
      return;
    }
  },
}));
