/**
 * Thin typed wrapper around the generated tauri-specta bindings.
 *
 * Consumers import from here rather than from `bindings.ts` directly so
 * that any future shim (mock in tests, offline-mode stub) has a single
 * injection point.
 */
import { events, commands as generatedCommands } from "./bindings";
import { invoke as TAURI_INVOKE } from "@tauri-apps/api/core";
import type { AppEventPayload, IpcError, MeetingId, Result } from "./bindings";
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
type GeneratedCommands = typeof generatedCommands;

/**
 * The Phase-5 summary command surface (`summarise_meeting`, `get_summary`,
 * `save_summary`).
 *
 * These commands are added to `ipc-bridge` by the Phase-5 backend JOIN (Stream
 * S5), which regenerates `bindings.ts` to include them on the generated
 * `commands` object. Until that regeneration lands, the generated `commands`
 * has no entry for them, so this module declares their shape locally and routes
 * them through {@link callPendingCommand} (the shim-aware raw-`invoke` path) —
 * mirroring the Phase-4 meeting commands' "pending generation" path before
 * Stream C regenerated the bindings. The signatures below MUST match the JOIN's
 * generated bindings exactly (see the dev report's S5 JOIN signatures note).
 *
 * Once Stream S5 regenerates `bindings.ts`, these can fold into `callCommand`
 * like every other command and this declaration can be dropped.
 */
type SummaryCommands = {
  summariseMeeting(meetingId: MeetingId): Promise<Result<null, IpcError>>;
  getSummary(meetingId: MeetingId): Promise<Result<string | null, IpcError>>;
  saveSummary(
    meetingId: MeetingId,
    summaryMarkdown: string,
  ): Promise<Result<null, IpcError>>;
};

type Commands = GeneratedCommands & SummaryCommands;
type CommandName = keyof GeneratedCommands;

let devCommandsPromise: Promise<Commands> | null = null;
async function loadDevCommands(): Promise<Commands> {
  if (!devCommandsPromise) {
    devCommandsPromise = import("./dev-shim").then(
      (m) => m.devCommands as unknown as Commands,
    );
  }
  return devCommandsPromise;
}

/**
 * Shim-aware raw-`invoke` path for commands not yet present on the generated
 * `commands` object (the Phase-5 summary surface, until Stream S5 regenerates
 * `bindings.ts`). In a `vite dev` browser with no Tauri backend it routes
 * through the DEV shim; otherwise it calls `TAURI_INVOKE` directly. Tests mock
 * the higher-level `../ipc/summary` seam, so this path is never exercised under
 * Vitest.
 */
async function callPendingCommand<T>(
  name: keyof SummaryCommands,
  tauriCommand: string,
  args: Record<string, unknown>,
): Promise<Result<T, IpcError>> {
  if (import.meta.env.DEV && shouldUseDevShim()) {
    const dev = await loadDevCommands();
    return (dev[name] as (...a: unknown[]) => Promise<Result<T, IpcError>>)(
      ...Object.values(args),
    );
  }
  try {
    return { status: "ok", data: await TAURI_INVOKE(tauriCommand, args) };
  } catch (e) {
    if (e instanceof Error) throw e;
    return { status: "error", error: e as IpcError };
  }
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
  reSummarise: (meetingId) => callCommand("reSummarise", [meetingId]),
  // Phase 5 summary surface (FR-30). Not yet on the generated `commands`
  // object — Stream S5's backend JOIN regenerates `bindings.ts` to add them;
  // until then they route through the shim-aware `callPendingCommand` raw
  // `invoke` path (the same approach the Phase-4 meeting commands used before
  // Stream C regenerated the bindings). The DEV shim still intercepts here.
  summariseMeeting: (meetingId) =>
    callPendingCommand<null>("summariseMeeting", "summarise_meeting", {
      meetingId,
    }),
  getSummary: (meetingId) =>
    callPendingCommand<string | null>("getSummary", "get_summary", {
      meetingId,
    }),
  saveSummary: (meetingId, summaryMarkdown) =>
    callPendingCommand<null>("saveSummary", "save_summary", {
      meetingId,
      summaryMarkdown,
    }),
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
