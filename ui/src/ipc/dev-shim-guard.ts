/**
 * DEV-shim activation guard (no sample data — safe to import from the main
 * bundle).
 *
 * Kept separate from `dev-shim.ts` so the heavy sample-data module is only ever
 * reached via a dynamic `import()` and never pulled into the production bundle.
 */

/**
 * True only in a `vite dev` browser with no Tauri backend present.
 *
 * - `import.meta.env.DEV` is statically `false` in production builds, so every
 *   guarded branch dead-code-eliminates there.
 * - `import.meta.env.MODE === "test"` is true only under the Vitest runner,
 *   whose suites mock the IPC seam directly and must not see the shim.
 * - The runtime Tauri check (`__TAURI_INTERNALS__` / `__TAURI__`) means the
 *   real Tauri dev app bypasses the shim and uses the genuine backend.
 */
export function shouldUseDevShim(): boolean {
  if (!import.meta.env.DEV) return false;
  if (import.meta.env.MODE === "test") return false;
  if (typeof window === "undefined") return false;
  return !("__TAURI_INTERNALS__" in window) && !("__TAURI__" in window);
}
