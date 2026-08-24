/**
 * MCP server settings pane (Phase 10).
 *
 * Lets the user enable the in-process loopback MCP server, set the fixed port,
 * toggle write-tool exposure, and reveal/copy the bearer token, plus a
 * ready-to-run `claude mcp add` command for Claude Code. Claude Desktop has no
 * equivalent path today: its Settings → Connectors only accepts a publicly
 * reachable URL, not a loopback one, and there is currently no desktop
 * extension bridging to a local server (one existed once — `mcpb` — but was
 * removed when external connectivity moved to a hosted relay; that relay is
 * now retired too, and whether to rebuild a bridge is an open question, not
 * something this pane should imply already works).
 *
 * The toggles round-trip through `update_settings` (via the recording store).
 * Enabling/disabling the server takes effect immediately (the backend watches
 * `mcp_enabled` and starts/stops the listener live). Changing the port or
 * write-tools setting is restart-required (the running server was built with
 * those values at startup).
 *
 * The live endpoint URL + token come from `get_mcp_server_info` — the store
 * also refreshes on `mcp_server_listening` (server started) and clears on
 * `mcp_server_stopped` (server disabled). The token is sensitive, masked by
 * default, and only the full value is copyable on an explicit reveal.
 *
 * Rendered in the Editorial Ink language using `theme.css` tokens only.
 */
import { useEffect, useState } from "react";
import { useRecordingStore } from "../state/recording";
import {
  readMcpEnabled,
  readMcpPort,
  readMcpWriteTools,
  DEFAULT_MCP_PORT,
} from "../state/mcp-settings";
import { useMcpServerInfoStore } from "../state/mcp-server-info";

/** The `claude mcp add` command that registers this server with Claude Code. */
function claudeCodeCommand(url: string, token: string): string {
  return `claude mcp add --transport http minutist ${url} --header "Authorization: Bearer ${token}"`;
}

export function McpSettingsPane() {
  const settings = useRecordingStore((s) => s.settings);
  const setMcpEnabled = useRecordingStore((s) => s.setMcpEnabled);
  const setMcpPort = useRecordingStore((s) => s.setMcpPort);
  const setMcpWriteTools = useRecordingStore((s) => s.setMcpWriteTools);

  const enabled = readMcpEnabled(settings);
  const port = readMcpPort(settings);
  const writeTools = readMcpWriteTools(settings);

  // The live endpoint info comes from the shared store (C1): it re-fetches on
  // the `mcp_server_listening` event, so the pane reflects the live URL without
  // a remount. We additionally refresh on mount / enabled-change to cover the
  // case where the listener bound before this pane (and the store) existed.
  const info = useMcpServerInfoStore((s) => s.info);
  const startFailedReason = useMcpServerInfoStore((s) => s.startFailedReason);
  const refreshInfo = useMcpServerInfoStore((s) => s.refresh);
  const [revealed, setRevealed] = useState(false);
  const [restartHint, setRestartHint] = useState(false);

  useEffect(() => {
    void refreshInfo();
  }, [enabled, refreshInfo]);

  // A change to port or write-tools is restart-required (the running server
  // was built with those values). Enabling/disabling is handled live by the
  // backend and does NOT set the restart hint.
  const markRestart = () => setRestartHint(true);

  const copy = (text: string) => {
    void navigator.clipboard?.writeText(text);
  };

  return (
    <section className="settings-drawer__group" aria-label="Connections (MCP)">
      <h3 className="settings-drawer__group-title">Connections (MCP)</h3>

      <label
        className="settings-drawer__toggle"
        title="Run a local Model Context Protocol server so external agents (e.g. Claude Desktop, via the bridge) can read this app's meetings. Loopback-only and bearer-token protected. Off by default."
      >
        <input
          type="checkbox"
          checked={enabled}
          disabled={settings === null}
          onChange={(e) => {
            // No restart hint: enable/disable takes effect live.
            void setMcpEnabled(e.target.checked);
          }}
        />
        <span>Enable MCP server (loopback)</span>
      </label>

      <div className="settings-drawer__field">
        <label htmlFor="settings-mcp-port">Port</label>
        <input
          id="settings-mcp-port"
          type="number"
          min={1}
          max={65535}
          value={port}
          disabled={settings === null || !enabled}
          onChange={(e) => {
            const next = Number.parseInt(e.target.value, 10);
            if (Number.isFinite(next) && next >= 1 && next <= 65535) {
              markRestart();
              void setMcpPort(next);
            }
          }}
        />
        {port !== DEFAULT_MCP_PORT && (
          <button
            type="button"
            className="settings-drawer__about"
            onClick={() => {
              markRestart();
              void setMcpPort(DEFAULT_MCP_PORT);
            }}
          >
            Reset to {DEFAULT_MCP_PORT}
          </button>
        )}
      </div>

      <label
        className="settings-drawer__toggle"
        title="Expose the reversible write tools (set speaker name, rename meeting) over MCP. Off = read-only over MCP. Heavy/destructive operations are never exposed."
      >
        <input
          type="checkbox"
          checked={writeTools}
          disabled={settings === null || !enabled}
          onChange={(e) => {
            markRestart();
            void setMcpWriteTools(e.target.checked);
          }}
        />
        <span>Allow write tools over MCP</span>
      </label>

      {enabled && info && (
        <div className="settings-drawer__field" aria-label="MCP endpoint">
          <label>Endpoint</label>
          <div className="settings-drawer__mcp-endpoint">
            <code className="settings-drawer__mcp-url">{info.url}</code>
            <button
              type="button"
              className="settings-drawer__about"
              onClick={() => copy(info.url)}
            >
              Copy URL
            </button>
          </div>
          <label>Bearer token</label>
          <div className="settings-drawer__mcp-endpoint">
            <code className="settings-drawer__mcp-token">
              {revealed ? info.token : "•".repeat(24)}
            </code>
            <button
              type="button"
              className="settings-drawer__about"
              onClick={() => setRevealed((r) => !r)}
            >
              {revealed ? "Hide" : "Reveal"}
            </button>
            <button
              type="button"
              className="settings-drawer__about"
              onClick={() => copy(info.token)}
            >
              Copy token
            </button>
          </div>
          <label>Claude Code</label>
          <div className="settings-drawer__mcp-endpoint">
            <code className="settings-drawer__mcp-url">
              {claudeCodeCommand(info.url, revealed ? info.token : "•".repeat(24))}
            </code>
            <button
              type="button"
              className="settings-drawer__about"
              onClick={() => copy(claudeCodeCommand(info.url, info.token))}
            >
              Copy command
            </button>
          </div>
          <p className="settings-drawer__hint">
            Run that in a terminal to add this server. Claude Desktop cannot
            reach a local server like this one yet — its Settings &rarr;
            Connectors expects a publicly reachable URL, and there is no
            desktop extension bridging to it at the moment.
          </p>
          <p className="settings-drawer__hint">
            To rotate the token, delete <code>mcp_token</code> in the app-data
            directory and restart the app.
          </p>
        </div>
      )}

      {enabled && !info && !startFailedReason && (
        <p className="settings-drawer__hint">
          The MCP server is enabled but not yet listening. It starts
          automatically in the background — this usually resolves in a few
          seconds.
        </p>
      )}

      {enabled && startFailedReason && (
        <p className="settings-drawer__hint settings-drawer__hint--warn">
          MCP server failed to start: {startFailedReason}. Toggle the server
          off then on again to retry.
        </p>
      )}

      {restartHint && (
        <p className="settings-drawer__hint settings-drawer__hint--warn">
          Restart the app for MCP server changes to take effect.
        </p>
      )}
    </section>
  );
}
