/**
 * Connector (relay tunnel) settings helpers (WS4-A S5b).
 *
 * `connector_enabled` is a first-class `Settings` field owned by the `settings`
 * crate (off by default, mirroring `mcp_enabled`). Unlike the MCP toggles, the
 * connector is enabled/disabled through the dedicated `set_connector_enabled`
 * command (which also starts/stops the tunnel lifecycle) rather than a bare
 * `update_settings` round-trip — so this module only exposes the *read* helper;
 * the write goes through the tunnel-status store's `setEnabled`.
 *
 * The connector channel transits meeting content to the AI vendor BY DESIGN
 * (the user asked for it) — it is never described as private (D5).
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the connector-enabled flag. Defaults to `false` (off) when the field is
 * absent or the snapshot is `null` — matching the backend `#[serde(default)]`.
 */
export function readConnectorEnabled(settings: Settings | null): boolean {
  if (settings === null) return false;
  return settings.connector_enabled === true;
}
