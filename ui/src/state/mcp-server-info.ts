/**
 * MCP server live-info store (Phase 10 review-fix C1).
 *
 * Holds the live MCP endpoint (URL + bearer token) sourced from the
 * `get_mcp_server_info` command. The backend emits
 * `AppEvent::McpServerListening { url }` once the loopback listener has bound;
 * routing that event through the global dispatcher into this store lets the
 * Settings → MCP pane reflect the live URL WITHOUT a remount (previously the
 * pane only re-fetched on mount / enabled-change, so the event was consumed by
 * no store).
 *
 * The token is sensitive: it is NOT carried on the event (the event has only the
 * URL); the store re-fetches the full info via `get_mcp_server_info`, the one
 * place the token crosses the IPC boundary.
 *
 * `startFailedReason` is set when the backend emits `McpServerStartFailed`,
 * causing the pane to show the error reason instead of the "starting…" hint.
 * It is cleared on every `McpServerListening` (successful start) and on
 * `McpServerStopped` (disabled).
 */
import { create } from "zustand";
import { commands, unwrap } from "../ipc/client";
import type { McpServerInfo } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type McpServerInfoStore = {
  /** The live endpoint info, or `null` when the server is disabled / not bound. */
  info: McpServerInfo | null;
  /**
   * Set when the server failed to start. Causes the pane to show the reason
   * instead of the generic "starting…" hint. `null` when healthy.
   */
  startFailedReason: string | null;
  /** Re-fetch `get_mcp_server_info` (on mount and on the listening event). */
  refresh: () => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useMcpServerInfoStore = create<McpServerInfoStore>((set) => ({
  info: null,
  startFailedReason: null,

  refresh: async () => {
    try {
      const fetched = unwrap(await commands.getMcpServerInfo());
      set({ info: fetched });
    } catch {
      set({ info: null });
    }
  },

  handleEvent: (event) => {
    if (event.kind === "mcp_server_stopped") {
      // The server stopped (accept loop exited); clear endpoint and any error.
      set({ info: null, startFailedReason: null });
      return;
    }
    if (event.kind === "mcp_server_start_failed") {
      // The server failed to bind or load its model; surface the reason so the
      // pane drops the "starting…" hint immediately.
      set({ info: null, startFailedReason: event.reason });
      return;
    }
    if (event.kind !== "mcp_server_listening") return;
    // The listener just bound: clear any prior failure, then re-fetch the full
    // info (URL + token) so the pane reflects the live endpoint.
    set({ startFailedReason: null });
    try {
      void (async () => {
        const fetched = unwrap(await commands.getMcpServerInfo());
        set({ info: fetched });
      })();
    } catch {
      set({ info: null });
    }
  },
}));
