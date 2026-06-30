/**
 * Live co-pilot (in-meeting agent) settings helpers.
 *
 * The master control is the `settings.live_agent_enabled` field, a tri-state
 * `LiveAgentMode` ("off" by default). `Off` never runs the co-pilot; `On` runs
 * it whenever a GPU is active (the explicit opt-in, including on a shared-memory
 * integrated GPU); `Auto` runs it only on a DISCRETE GPU with acceleration on,
 * where the held LLM context does not contend with the GPU ASR path for memory.
 * The gate is `minutist_common::live_agent_should_run`. The field is owned by the
 * `settings` crate and is a first-class member of the generated `Settings` type,
 * so — exactly like the GPU-acceleration helpers — these read/write the canonical
 * field directly. See `architecture/cross-cutting.md` — "Live agent".
 */
import type { LiveAgentMode, Settings } from "../ipc/bindings";

/**
 * Read the live-agent mode from a settings snapshot. Defaults to `"off"` when the
 * field is absent (an older store, matching the backend `#[serde(default)]` of
 * `Off`) and when the snapshot is `null` so the control renders the default while
 * settings are still loading.
 */
export function readLiveAgentMode(settings: Settings | null): LiveAgentMode {
  if (settings === null) return "off";
  return settings.live_agent_enabled ?? "off";
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
