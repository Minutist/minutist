/**
 * GPU-acceleration settings helpers.
 *
 * The runtime GPU toggle is the `settings.gpu_acceleration` field (on by
 * default). GPU offload happens ONLY when BOTH the build was compiled with a
 * GPU feature (Vulkan/Metal/etc.) AND this setting is true; when false,
 * inference runs on CPU even in a GPU-feature build — the escape hatch for weak
 * GPUs / driver trouble. The field is owned by the `settings` crate and is a
 * first-class member of the generated `Settings` type, so — exactly like the
 * diarization helpers — these read/write the canonical field directly with no
 * augmentation shim. See `architecture/cross-cutting.md` — "GPU portability".
 */
import type { Settings } from "../ipc/bindings";

/**
 * Read the GPU-acceleration flag from a settings snapshot.
 *
 * Defaults to `true` (on) when the field is absent (an older store written
 * before it existed) — matching the backend `#[serde(default)]` of `true` — and
 * to `true` when the snapshot is `null` so the checkbox renders checked while
 * settings are still loading.
 */
export function readGpuAcceleration(settings: Settings | null): boolean {
  if (settings === null) return true;
  return settings.gpu_acceleration !== false;
}

/**
 * Return a copy of `settings` with the GPU-acceleration flag set, preserving
 * every other field so the `update_settings` round-trip does not clobber the
 * rest of the store.
 */
export function withGpuAcceleration(
  settings: Settings,
  enabled: boolean,
): Settings {
  return { ...settings, gpu_acceleration: enabled };
}
