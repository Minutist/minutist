/**
 * Live co-pilot (in-meeting agent) settings helpers.
 *
 * The master control is the `settings.live_agent_enabled` field, a tri-state
 * `LiveAgentMode` ("auto" by default). `Off` never runs the co-pilot; `Auto` (the
 * default) runs it when GPU acceleration is active — a usable GPU is present
 * (integrated OR discrete) and acceleration is not forced off; `On` always runs
 * it, even on a CPU-only host (the explicit opt-in, slower refreshes). The gate is
 * `minutist_common::live_agent_should_run`. The field is owned by the `settings`
 * crate and is a first-class member of the generated `Settings` type, so — exactly
 * like the GPU-acceleration helpers — these read/write the canonical field
 * directly. See `architecture/cross-cutting.md` — "Live agent".
 */
import type { LiveAgentMode, Settings } from "../ipc/bindings";

/**
 * Read the live-agent mode from a settings snapshot. Defaults to `"auto"` when the
 * field is absent (an older store, matching the backend `#[serde(default)]` of
 * `Auto`) and when the snapshot is `null` so the control renders the default while
 * settings are still loading.
 */
export function readLiveAgentMode(settings: Settings | null): LiveAgentMode {
  if (settings === null) return "auto";
  return settings.live_agent_enabled ?? "auto";
}

/**
 * Return a copy of `settings` with the live-agent mode set, preserving every
 * other field so the `update_settings` round-trip does not clobber the rest of
 * the store.
 */
export function withLiveAgentMode(
  settings: Settings,
  mode: LiveAgentMode,
): Settings {
  return { ...settings, live_agent_enabled: mode };
}
