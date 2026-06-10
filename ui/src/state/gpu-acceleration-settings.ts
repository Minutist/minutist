/**
 * GPU-acceleration settings helpers.
 *
 * The runtime GPU control is the `settings.gpu_acceleration` field, a tri-state
 * `GpuAcceleration` ("auto" by default). `Auto` probes GPU VRAM at each model
 * load and offloads a model to the GPU only when it fits, else CPU; `On`/`Off`
 * are hard overrides that never consult the probe. GPU offload only ever happens
 * in a build compiled with a GPU feature (Vulkan/Metal/etc.); a CPU-only build
 * always runs on CPU regardless. The field is owned by the `settings` crate and
 * is a first-class member of the generated `Settings` type, so — exactly like
 * the diarization helpers — these read/write the canonical field directly with
 * no augmentation shim. See `architecture/cross-cutting.md` — "GPU portability".
 */
import type { GpuAcceleration, Settings } from "../ipc/bindings";

/**
 * Read the GPU-acceleration mode from a settings snapshot.
 *
 * Defaults to `"auto"` when the field is absent (an older store written before
 * it existed — matching the backend `#[serde(default)]` of `Auto`) and when the
 * snapshot is `null` so the control renders the default while settings are still
 * loading.
 */
export function readGpuAcceleration(settings: Settings | null): GpuAcceleration {
  if (settings === null) return "auto";
  return settings.gpu_acceleration ?? "auto";
}

/**
 * Return a copy of `settings` with the GPU-acceleration mode set, preserving
 * every other field so the `update_settings` round-trip does not clobber the
 * rest of the store.
 */
export function withGpuAcceleration(
  settings: Settings,
  mode: GpuAcceleration,
): Settings {
  return { ...settings, gpu_acceleration: mode };
}
