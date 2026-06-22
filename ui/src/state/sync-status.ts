/**
 * Sync engine live-state store (WS4-B S5).
 *
 * Holds the sync engine's current status (sourced from `sync_status`), this
 * device's shareable ticket, a pending-add-peer field, and the live in-flight
 * transfer (label + fraction). The backend emits `AppEvent::SyncProgress /
 * SyncReady / SyncError` while a transfer is running: `SyncProgress` updates
 * the in-flight state (the Sync pane renders "Syncing…" from it), and a
 * terminal `SyncReady` / `SyncError` clears it. `SyncReady` also queues a
 * toast the `MainWindow` chrome strip renders, and reloads the meeting list +
 * the open meeting so synced-in content surfaces without a manual refresh.
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
import { useMeetingsStore } from "./meetings";
import type { SyncStatus, MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

/** The live in-flight transfer shown by the Sync pane while a sync runs. */
export type SyncInProgress = {
  /** The meeting the transfer is for. */
  meetingId: MeetingId;
  /** Short human-readable label (e.g. "Syncing notes…"). */
  label: string;
  /** `null` = indeterminate; a `0..=1` value = determinate fraction. */
  fraction: number | null;
};

export type SyncStatusStore = {
  /** The live engine status, or `null` before the first fetch. */
  status: SyncStatus | null;
  /**
   * The in-flight transfer (label + fraction) sourced from `SyncProgress`
   * events, or `null` when nothing is syncing. A terminal `SyncReady` /
   * `SyncError` clears it. The Sync pane shows a "Syncing…" state from this.
   */
  inProgress: SyncInProgress | null;
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
  inProgress: null,
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
      // A transfer made progress: track the in-flight state so the pane shows
      // "Syncing…". The fraction is `Some(f)` for a determinate bar; `null`
      // (None over the wire) for an indeterminate one.
      set({
        inProgress: {
          meetingId: event.meeting_id,
          label: event.label,
          fraction: event.fraction ?? null,
        },
      });
      return;
    }
    if (event.kind === "sync_ready") {
      // Terminal: clear the in-flight state and queue a toast. Reload the
      // meeting list so a synced-in (new or updated) meeting surfaces, and if it
      // is the open meeting, re-read it so its merged notes appear without a
      // manual refresh.
      set((s) => ({
        inProgress: null,
        pendingReadyNotifications: [
          ...s.pendingReadyNotifications,
          event.meeting_id,
        ],
      }));
      const meetings = useMeetingsStore.getState();
      void meetings.refresh();
      if (meetings.openMeetingId === event.meeting_id) {
        void meetings.open(event.meeting_id);
      }
      return;
    }
    if (event.kind === "sync_error") {
      // Terminal: clear the in-flight state and surface the error.
      set({ inProgress: null, lastError: event.context });
      return;
    }
  },
}));
