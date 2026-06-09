/**
 * MCP-server settings helpers (Phase 10).
 *
 * Three first-class `Settings` fields owned by the `settings` crate:
 *  - `mcp_enabled` — start the in-process loopback MCP server (off by default).
 *    Toggling at runtime is a documented restart-required for v1 (the listener
 *    is spawned once at startup gated on this flag).
 *  - `mcp_port` — the FIXED loopback port (default 8765, D1), user-editable.
 *  - `mcp_write_tools` — expose the reversible write tools over MCP (off by
 *    default = read-only; D3).
 *
 * Like the other settings helpers these read/write the canonical fields with no
 * augmentation shim. The live endpoint URL + bearer token are NOT settings —
 * they come from the `get_mcp_server_info` command (the token is sensitive and
 * crosses the boundary only on that explicit read).
 */
import type { Settings } from "../ipc/bindings";

/** The fixed default MCP port (mirrors the backend `default_mcp_port`, D1). */
export const DEFAULT_MCP_PORT = 8765;

/**
 * Read the MCP-enabled flag. Defaults to `false` (off) when the field is absent
 * or the snapshot is `null` — matching the backend `#[serde(default)]`.
 */
export function readMcpEnabled(settings: Settings | null): boolean {
  if (settings === null) return false;
  return settings.mcp_enabled === true;
}

/**
 * Read the MCP port. Defaults to {@link DEFAULT_MCP_PORT} when absent or the
 * snapshot is `null` — matching the backend `default_mcp_port`.
 */
export function readMcpPort(settings: Settings | null): number {
  if (settings === null) return DEFAULT_MCP_PORT;
  return settings.mcp_port ?? DEFAULT_MCP_PORT;
}

/**
 * Read the MCP write-tools flag. Defaults to `false` (read-only over MCP) when
 * absent or the snapshot is `null` — matching the backend `#[serde(default)]`.
 */
export function readMcpWriteTools(settings: Settings | null): boolean {
  if (settings === null) return false;
  return settings.mcp_write_tools === true;
}

/** Return a copy of `settings` with the MCP-enabled flag set. */
export function withMcpEnabled(settings: Settings, enabled: boolean): Settings {
  return { ...settings, mcp_enabled: enabled };
}

/** Return a copy of `settings` with the MCP port set. */
export function withMcpPort(settings: Settings, port: number): Settings {
  return { ...settings, mcp_port: port };
}

/** Return a copy of `settings` with the MCP write-tools flag set. */
export function withMcpWriteTools(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, mcp_write_tools: enabled };
}
