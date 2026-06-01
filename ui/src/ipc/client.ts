/**
 * Thin typed wrapper around the generated tauri-specta bindings.
 *
 * Consumers import from here rather than from `bindings.ts` directly so
 * that any future shim (mock in tests, offline-mode stub) has a single
 * injection point.
 */
import { invoke as TAURI_INVOKE } from "@tauri-apps/api/core";
import { events, commands as generatedCommands } from "./bindings";
import type {
  AppEventPayload,
  IpcError,
  MeetingId,
  Result,
} from "./bindings";
import type { MeetingListEntry, MeetingState } from "./meetings";
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

/**
 * The Phase-4 meeting commands not yet present on the generated `commands`
 * object (Stream C adds them and regenerates `bindings.ts`). Typed here against
 * the canonical `common::MeetingListEntry` / `common::MeetingState` shapes so
 * the seam is fully typed before the regeneration.
 */
export type PendingCommands = {
  listMeetings: () => Promise<Result<MeetingListEntry[], IpcError>>;
  openMeeting: (meetingId: MeetingId) => Promise<Result<MeetingState, IpcError>>;
  renameMeeting: (
    meetingId: MeetingId,
    title: string,
  ) => Promise<Result<null, IpcError>>;
  deleteMeeting: (meetingId: MeetingId) => Promise<Result<null, IpcError>>;
  reTranscribe: (meetingId: MeetingId) => Promise<Result<null, IpcError>>;
  reSummarise: (meetingId: MeetingId) => Promise<Result<null, IpcError>>;
};

/** The dev-shim's runtime surface for the pending commands (camelCase names). */
type DevPending = Record<
  string,
  (...args: unknown[]) => Promise<Result<unknown, IpcError>>
>;

/**
 * Invoke a command that is not yet on the generated `commands` object.
 *
 * In DEV-shim mode the dev shim already supplies the camelCase method; call it.
 * Otherwise invoke the real backend via `TAURI_INVOKE` using the `snake_case`
 * wire name and `args` object, wrapping the call in the same `Result` shape the
 * generated bindings produce. Once Stream C regenerates `bindings.ts`, callers
 * can move to `callCommand`; the wire behaviour is identical.
 */
async function callPendingCommand<T>(
  devName: string,
  wireName: string,
  args: Record<string, unknown>,
): Promise<Result<T, IpcError>> {
  if (import.meta.env.DEV && shouldUseDevShim()) {
    const dev = (await loadDevCommands()) as unknown as DevPending;
    const argList = Object.values(args);
    return (await dev[devName](...argList)) as Result<T, IpcError>;
  }
  try {
    return { status: "ok", data: (await TAURI_INVOKE(wireName, args)) as T };
  } catch (e) {
    if (e instanceof Error) throw e;
    return { status: "error", error: e as IpcError };
  }
}

/** Delegating commands surface; see {@link callCommand}. */
export const commands: Commands & PendingCommands = {
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
  // Phase 4 meeting-list + open surface. These commands are added to the
  // backend by Stream C, which regenerates `bindings.ts` to include them; until
  // that regeneration lands they are not on the generated `commands` object, so
  // they route through `callPendingCommand` (DEV shim in shim-mode, raw
  // `TAURI_INVOKE` with the snake_case wire name otherwise) rather than through
  // `callCommand` (which is keyed on the generated names). Once regenerated,
  // these can fold into `callCommand` with no call-site change.
  listMeetings: () => callPendingCommand("listMeetings", "list_meetings", {}),
  openMeeting: (meetingId) =>
    callPendingCommand("openMeeting", "open_meeting", { meetingId }),
  renameMeeting: (meetingId, title) =>
    callPendingCommand("renameMeeting", "rename_meeting", { meetingId, title }),
  deleteMeeting: (meetingId) =>
    callPendingCommand("deleteMeeting", "delete_meeting", { meetingId }),
  reTranscribe: (meetingId) =>
    callPendingCommand("reTranscribe", "re_transcribe", { meetingId }),
  reSummarise: (meetingId) =>
    callPendingCommand("reSummarise", "re_summarise", { meetingId }),
};

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
