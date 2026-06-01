/**
 * Thin typed wrapper around the generated tauri-specta bindings.
 *
 * Consumers import from here rather than from `bindings.ts` directly so
 * that any future shim (mock in tests, offline-mode stub) has a single
 * injection point.
 */
import { events, commands } from "./bindings";
import type {
  AppEventPayload,
  IpcError,
  Result,
} from "./bindings";
import type { AppEvent } from "./app-event";

// Re-export the generated commands surface verbatim.
export { commands };

// Re-export types that callers commonly need. `AppEvent` is the webview-extended
// union (generated variants + the not-yet-generated `recording_clock`), see
// `./app-event`.
export type { AppEventPayload, AppEvent, IpcError, Result };

// ---------------------------------------------------------------------------
// Typed listen helper
// ---------------------------------------------------------------------------

/**
 * Subscribe to the global `"app-event-payload"` event stream.
 *
 * Returns an unsubscribe function (the resolved `UnlistenFn` from Tauri).
 * Callers are responsible for calling it on unmount to avoid listener leaks.
 *
 * @example
 * ```ts
 * const unlisten = await listenAppEvents((event) => {
 *   store.handleEvent(event);
 * });
 * return () => { unlisten(); };
 * ```
 */
export async function listenAppEvents(
  callback: (event: AppEvent) => void,
): Promise<() => void> {
  return events.appEventPayload.listen((tauriEvent) => {
    // The runtime payload may carry the `recording_clock` variant before the
    // generated bindings include it; widen to the extended `AppEvent` union.
    callback(tauriEvent.payload as AppEvent);
  });
}

// ---------------------------------------------------------------------------
// Result helpers
// ---------------------------------------------------------------------------

/**
 * Unwrap a `Result<T, IpcError>` to `T`, throwing on error.
 *
 * Use this when calling code has a try/catch and does not need to inspect
 * the error variant — it just wants the success value or an exception.
 */
export function unwrap<T>(result: Result<T, IpcError>): T {
  if (result.status === "ok") {
    return result.data;
  }
  throw new IpcCallError(result.error);
}

/**
 * Extract a human-readable message from an `IpcError` discriminated union.
 */
export function ipcErrorMessage(err: IpcError): string {
  switch (err.code) {
    case "io":
      return `IO error: ${err.context}`;
    case "model_load":
      return `Failed to load model "${err.model_id}": ${err.context}`;
    case "model_not_found":
      return `Model not found: ${err.model_id}`;
    case "model_download":
      return `Model download failed: ${err.context}`;
    case "inference":
      return `Inference error (${err.backend}): ${err.context}`;
    case "invalid_input":
      return `Invalid input: ${err.context}`;
    case "cancelled":
      return "Operation cancelled";
    case "unsupported":
      return `Unsupported: ${err.context}`;
    case "internal":
      return `Internal error: ${err.context}`;
  }
}

/**
 * Typed error class wrapping an `IpcError` payload.
 *
 * Thrown by `unwrap()` on error results. Callers that need to inspect the
 * error variant can switch on `error.ipcError.code`.
 */
export class IpcCallError extends Error {
  readonly ipcError: IpcError;

  constructor(ipcError: IpcError) {
    super(ipcErrorMessage(ipcError));
    this.name = "IpcCallError";
    this.ipcError = ipcError;
  }
}
