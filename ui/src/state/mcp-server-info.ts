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
 */
import { create } from "zustand";
import { commands, unwrap } from "../ipc/client";
import type { McpServerInfo } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type McpServerInfoStore = {
  /** The live endpoint info, or `null` when the server is disabled / not bound. */
  info: McpServerInfo | null;
  /** Re-fetch `get_mcp_server_info` (on mount and on the listening event). */
  refresh: () => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useMcpServerInfoStore = create<McpServerInfoStore>((set) => ({
  info: null,

  refresh: async () => {
    try {
      const fetched = unwrap(await commands.getMcpServerInfo());
      set({ info: fetched });
    } catch {
      set({ info: null });
    }
  },

  handleEvent: (event) => {
    if (event.kind !== "mcp_server_listening") return;
    // The listener just bound: re-fetch the full info (URL + token) so the pane
    // reflects the live URL live. The token is not on the event.
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
