/**
 * MCP server settings pane (Phase 10).
 *
 * Lets the user enable the in-process loopback MCP server, set the fixed port,
 * toggle write-tool exposure, and reveal/copy the bearer token + endpoint URL to
 * paste into an external MCP client's config.
 *
 * The toggles round-trip through `update_settings` (via the recording store);
 * enabling/disabling and the port are restart-required for v1 (the listener is
 * spawned once at startup), so the pane advises a restart when a change is made.
 * The live endpoint URL + token come from `get_mcp_server_info` — the token is
 * sensitive, so it is masked by default and only the full value is copyable on
 * an explicit reveal.
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
import { commands, unwrap } from "../ipc/client";
import type { McpServerInfo } from "../ipc/bindings";

export function McpSettingsPane() {
  const settings = useRecordingStore((s) => s.settings);
  const setMcpEnabled = useRecordingStore((s) => s.setMcpEnabled);
  const setMcpPort = useRecordingStore((s) => s.setMcpPort);
  const setMcpWriteTools = useRecordingStore((s) => s.setMcpWriteTools);

  const enabled = readMcpEnabled(settings);
  const port = readMcpPort(settings);
  const writeTools = readMcpWriteTools(settings);

  const [info, setInfo] = useState<McpServerInfo | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [restartHint, setRestartHint] = useState(false);

  // Fetch the live endpoint info when the pane mounts (and after a re-enable).
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const result = await commands.getMcpServerInfo();
        const fetched = unwrap(result);
        if (!cancelled) setInfo(fetched);
      } catch {
        if (!cancelled) setInfo(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  // A change to enable/port/write-tools is restart-required for v1.
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
            markRestart();
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
          <p className="settings-drawer__hint">
            Paste the URL and token into the Minutist bridge in Claude Desktop
            (Settings &rarr; Extensions). Regenerating the token requires
            replacing it there.
          </p>
        </div>
      )}

      {enabled && !info && (
        <p className="settings-drawer__hint">
          The MCP server is enabled but not yet listening. If you just turned it
          on, restart the app to start the server.
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
