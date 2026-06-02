/**
 * Thin typed wrapper around the generated tauri-specta bindings.
 *
 * Consumers import from here rather than from `bindings.ts` directly so
 * that any future shim (mock in tests, offline-mode stub) has a single
 * injection point.
 */
import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { events, commands as generatedCommands } from "./bindings";
import type { AppEventPayload, IpcError, Result } from "./bindings";
import type { AppEvent } from "./app-event";
import { shouldUseDevShim } from "./dev-shim-guard";

/**
 * The commands surface consumers import.
 *
 * In a `vite dev` browser with no Tauri backend the DEV-only shim
 * ({@link shouldUseDevShim}) supplies sample data so the themed UI renders for
 * visual QA. Every method delegates through {@link callCommand}, which lazily
 * `import()`s the shim ONLY in that mode — so the shim module (with all its
 * sample data) is loaded as a separate dynamic chunk that the real app and the
 * production build never fetch, and the main bundle stays free of sample data.
 *
 * `shouldUseDevShim()` is evaluated per call (cheap) so behaviour matches the
 * actual runtime; the dynamic import is memoised after the first DEV call.
 */
type Commands = typeof generatedCommands;
type CommandName = keyof Commands;

let devCommandsPromise: Promise<Commands> | null = null;
async function loadDevCommands(): Promise<Commands> {
  if (!devCommandsPromise) {
    devCommandsPromise = import("./dev-shim").then(
      (m) => m.devCommands as unknown as Commands,
    );
  }
  return devCommandsPromise;
}

async function callCommand<K extends CommandName>(
  name: K,
  args: Parameters<Commands[K]>,
): Promise<Awaited<ReturnType<Commands[K]>>> {
  type Out = Awaited<ReturnType<Commands[K]>>;
  if (import.meta.env.DEV && shouldUseDevShim()) {
    const dev = await loadDevCommands();
    return (await (dev[name] as (...a: unknown[]) => Promise<Out>)(
      ...args,
    )) as Out;
  }
  return (await (generatedCommands[name] as (...a: unknown[]) => Promise<Out>)(
    ...args,
  )) as Out;
}

/** Delegating commands surface; see {@link callCommand}. */
export const commands: Commands = {
  listDevices: () => callCommand("listDevices", []),
  startRecording: (deviceId) => callCommand("startRecording", [deviceId]),
  pauseRecording: () => callCommand("pauseRecording", []),
  resumeRecording: () => callCommand("resumeRecording", []),
  stopRecording: () => callCommand("stopRecording", []),
  getRecordingState: () => callCommand("getRecordingState", []),
  getSettings: () => callCommand("getSettings", []),
  updateSettings: (settings) => callCommand("updateSettings", [settings]),
  listModels: () => callCommand("listModels", []),
  ensureModel: (modelId) => callCommand("ensureModel", [modelId]),
  saveNotes: (meetingId, notesJson, notesMarkdown) =>
    callCommand("saveNotes", [meetingId, notesJson, notesMarkdown]),
  loadNotes: (meetingId) => callCommand("loadNotes", [meetingId]),
  // Phase 4 meeting-list + open surface (FR-33). Now present on the generated
  // `commands` object (the temporary "pending generation" path was collapsed
  // once Stream C regenerated `bindings.ts`), so these route through
  // `callCommand` like every other command — the DEV shim and Vitest mocks
  // still intercept here.
  listMeetings: () => callCommand("listMeetings", []),
  openMeeting: (meetingId) => callCommand("openMeeting", [meetingId]),
  renameMeeting: (meetingId, title) =>
    callCommand("renameMeeting", [meetingId, title]),
  deleteMeeting: (meetingId) => callCommand("deleteMeeting", [meetingId]),
  reTranscribe: (meetingId) => callCommand("reTranscribe", [meetingId]),
  // Phase 5 summary surface (FR-30). Now present on the generated `commands`
  // object (the Phase-5 backend JOIN regenerated `bindings.ts` and removed the
  // Phase-4 `re_summarise` stub), so these route through `callCommand` like
  // every other command — the earlier shim-aware `callPendingCommand` raw
  // `invoke` path was collapsed. The DEV shim and the Vitest mocks still
  // intercept here.
  summariseMeeting: (meetingId) =>
    callCommand("summariseMeeting", [meetingId]),
  getSummary: (meetingId) => callCommand("getSummary", [meetingId]),
  saveSummary: (meetingId, summaryMarkdown) =>
    callCommand("saveSummary", [meetingId, summaryMarkdown]),
};

/**
 * Invoke a command that is NOT yet on the generated `commands` surface.
 *
 * Some backend commands land in `ipc-bridge` (and thus in the regenerated
 * `bindings.ts`) in a later stream than the webview slice that calls them. The
 * webview cannot hand-edit `bindings.ts` (A9 — it is generated), so until the
 * backend JOIN regenerates it, those commands route through this thin
 * shim-aware raw-`invoke` path. It mirrors the tauri-specta calling convention
 * exactly so the eventual move to `callCommand` is a no-op for callers:
 *
 *   - the command name is the **snake_case** Tauri command name,
 *   - positional `args` are mapped onto the named-argument envelope by index
 *     using the supplied `argNames` (tauri-specta passes
 *     `TAURI_INVOKE(name, { camelCaseArg })`),
 *   - the result is wrapped in the same `Result<T, IpcError>` envelope the
 *     generated stubs produce.
 *
 * In `vite dev` with no Tauri backend the DEV-only shim
 * ({@link shouldUseDevShim}) answers via its `devPendingCommands` map (keyed by
 * the snake_case name), loaded through the same dynamic `import()` as
 * {@link callCommand} so the production build never bundles the sample data.
 *
 * This was the path the Phase-4 meeting commands and the Phase-5 summary
 * commands used before their bindings regenerated; the Phase-6
 * `rediarize_meeting` command uses it now (Stream S5 regenerates the bindings
 * and folds it into `callCommand`).
 */
let devPendingPromise: Promise<
  Record<string, (...a: unknown[]) => Promise<Result<unknown, IpcError>>>
> | null = null;
async function loadDevPendingCommands(): Promise<
  Record<string, (...a: unknown[]) => Promise<Result<unknown, IpcError>>>
> {
  if (!devPendingPromise) {
    devPendingPromise = import("./dev-shim").then(
      (m) =>
        m.devPendingCommands as Record<
          string,
          (...a: unknown[]) => Promise<Result<unknown, IpcError>>
        >,
    );
  }
  return devPendingPromise;
}

// The single not-yet-generated command Phase 6 needs and its argument name, so
// the named-argument envelope matches what tauri-specta will generate.
const PENDING_ARG_NAMES: Record<string, readonly string[]> = {
  rediarize_meeting: ["meetingId"],
};

export async function callPendingCommand<T>(
  name: string,
  args: unknown[],
): Promise<Result<T, IpcError>> {
  if (import.meta.env.DEV && shouldUseDevShim()) {
    const dev = await loadDevPendingCommands();
    const handler = dev[name];
    if (handler) {
      return (await handler(...args)) as Result<T, IpcError>;
    }
  }
  const argNames = PENDING_ARG_NAMES[name] ?? [];
  const payload: Record<string, unknown> = {};
  argNames.forEach((argName, i) => {
    payload[argName] = args[i];
  });
  try {
    const data = (await tauriInvoke(name, payload)) as T;
    return { status: "ok", data };
  } catch (e) {
    if (e instanceof Error) throw e;
    return { status: "error", error: e as IpcError };
  }
}

// Re-export types that callers commonly need. `AppEvent` is the generated
// event union (re-exported via `./app-event`, which is the webview's single
// import site for the event type).
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
  // DEV-only: with no Tauri backend, drive a representative sample event stream
  // (recording state + transcript + live meter/clock) so the UI populates. The
  // shim is loaded via a dynamic `import()` only in this mode, so production
  // never bundles or fetches it.
  if (import.meta.env.DEV && shouldUseDevShim()) {
    const { startDevEventStream } = await import("./dev-shim");
    return startDevEventStream(callback);
  }
  return events.appEventPayload.listen((tauriEvent) => {
    callback(tauriEvent.payload);
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
