/**
 * Thin typed wrapper around the generated tauri-specta bindings.
 *
 * Consumers import from here rather than from `bindings.ts` directly so
 * that any future shim (mock in tests, offline-mode stub) has a single
 * injection point.
 */
import { events, commands as generatedCommands } from "./bindings";
import type { AppError, AppEventPayload, Result } from "./bindings";
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
  // Live-test UX T2: pre-warm the ASR model so the first record is not a cold
  // ~29 s load. Best-effort; the DEV shim no-ops it.
  prewarmAsr: () => callCommand("prewarmAsr", []),
  pauseRecording: () => callCommand("pauseRecording", []),
  resumeRecording: () => callCommand("resumeRecording", []),
  // Set the live recording's meeting title (held by the orchestrator, applied at
  // stop). The DEV shim no-ops it (no backend), so the masthead input still works.
  setRecordingTitle: (meetingId, title) =>
    callCommand("setRecordingTitle", [meetingId, title]),
  stopRecording: () => callCommand("stopRecording", []),
  getRecordingState: () => callCommand("getRecordingState", []),
  getSettings: () => callCommand("getSettings", []),
  updateSettings: (settings) => callCommand("updateSettings", [settings]),
  listModels: () => callCommand("listModels", []),
  ensureModel: (modelId) => callCommand("ensureModel", [modelId]),
  saveNotes: (meetingId, notesJson, notesMarkdown) =>
    callCommand("saveNotes", [meetingId, notesJson, notesMarkdown]),
  loadNotes: (meetingId) => callCommand("loadNotes", [meetingId]),
  // CRDT editor binding (B6 WU7): merge an editor-produced lib0-v1 Yjs update
  // onto the meeting's notes.ydoc; read the stored doc as a v1 state update for
  // the editor to hydrate on open. The DEV shim no-ops them (no backend).
  applyNotesUpdate: (meetingId, update, notesMarkdown) =>
    callCommand("applyNotesUpdate", [meetingId, update, notesMarkdown]),
  loadNotesYdoc: (meetingId) => callCommand("loadNotesYdoc", [meetingId]),
  // Persist a pasted/dropped note image; returns the portable filename ref the
  // editor stores into notes.json. The DEV shim no-ops it (no backend).
  saveNoteImage: (meetingId, bytes, ext) =>
    callCommand("saveNoteImage", [meetingId, bytes, ext]),
  // Phase 4 meeting-list + open surface (FR-33). Now present on the generated
  // `commands` object (the temporary "pending generation" path was collapsed
  // once Stream C regenerated `bindings.ts`), so these route through
  // `callCommand` like every other command — the DEV shim and Vitest mocks
  // still intercept here.
  listMeetings: () => callCommand("listMeetings", []),
  openMeeting: (meetingId) => callCommand("openMeeting", [meetingId]),
  renameMeeting: (meetingId, title) =>
    callCommand("renameMeeting", [meetingId, title]),
  setSpeakerName: (meetingId, label, name) =>
    callCommand("setSpeakerName", [meetingId, label, name]),
  deleteMeeting: (meetingId) => callCommand("deleteMeeting", [meetingId]),
  // Themed context menus (#0034) — meeting-list "Open storage folder" entry.
  // Resolves the on-disk directory server-side and hands it to the host OS
  // file explorer via `tauri-plugin-opener`; the DEV shim no-ops it.
  openMeetingFolder: (meetingId) =>
    callCommand("openMeetingFolder", [meetingId]),
  // #0015 — the offline reprocess (re-transcribe + re-diarize under one claim)
  // merges the former `reTranscribe` + `rediarizeMeeting` commands into one.
  reprocess: (meetingId) => callCommand("reprocess", [meetingId]),
  // Collections ("folders"): list/create/rename/delete the folder definitions
  // and file a meeting into one (or unfile it with `null`). The DEV shim keeps
  // an in-memory list so the sidebar renders + mutates under `vite dev`.
  listCollections: () => callCommand("listCollections", []),
  createCollection: (name) => callCommand("createCollection", [name]),
  renameCollection: (collectionId, name) =>
    callCommand("renameCollection", [collectionId, name]),
  deleteCollection: (collectionId) =>
    callCommand("deleteCollection", [collectionId]),
  setMeetingCollection: (meetingId, collectionId) =>
    callCommand("setMeetingCollection", [meetingId, collectionId]),
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
  // Reference-material attachments. Originals are stored content-addressed under
  // the meeting's `attachments/` dir; `add_attachment` enqueues a background
  // doc→markdown conversion (the four attachment events stream the row state).
  // `open_attachment` resolves the original's on-disk path server-side and hands
  // it to the host OS default app via `tauri-plugin-opener`; remove is dedup-safe.
  // The DEV shim and the Vitest mocks intercept at the `../ipc/attachments` seam,
  // not here.
  addAttachment: (meetingId, bytes, ext, originalFilename) =>
    callCommand("addAttachment", [meetingId, bytes, ext, originalFilename]),
  listAttachments: (meetingId) => callCommand("listAttachments", [meetingId]),
  openAttachment: (meetingId, attachmentId) =>
    callCommand("openAttachment", [meetingId, attachmentId]),
  removeAttachment: (meetingId, attachmentId) =>
    callCommand("removeAttachment", [meetingId, attachmentId]),
  // Phase 9 chat surface. The backend (this step, 4a) is wired; the chat UI is
  // a later step (4b). These delegations exist so the typed `commands` surface
  // matches the regenerated `bindings.ts` and consumers have a single seam.
  sendChatMessage: (meetingId, sessionId, message) =>
    callCommand("sendChatMessage", [meetingId, sessionId, message]),
  cancelChatTurn: (sessionId) => callCommand("cancelChatTurn", [sessionId]),
  getChatSession: (meetingId, sessionId) =>
    callCommand("getChatSession", [meetingId, sessionId]),
  listChatSessions: (meetingId) =>
    callCommand("listChatSessions", [meetingId]),
  deleteChatSession: (meetingId, sessionId) =>
    callCommand("deleteChatSession", [meetingId, sessionId]),
  // Phase 10: the live MCP endpoint (URL + bearer token) for the Settings → MCP
  // pane. The token is sensitive and crosses the boundary only on this read.
  getMcpServerInfo: () => callCommand("getMcpServerInfo", []),
  // Translation commands: post-hoc per-segment translation backed by the held
  // LLM. `translateMeeting` triggers the pass (long-running, fire-and-forget
  // from the UI's perspective); `getTranslations` reads the persisted map.
  translateMeeting: (meetingId, targetLanguage) =>
    callCommand("translateMeeting", [meetingId, targetLanguage]),
  getTranslations: (meetingId, targetLanguage) =>
    callCommand("getTranslations", [meetingId, targetLanguage]),
  // Issue #0014: the redacted diagnostic snapshot the "Report a problem" flow
  // pre-fills into a GitHub issue. No telemetry — the user reviews + submits it
  // from their own browser. The dev shim returns sample data for visual QA.
  getDiagnosticReport: () => callCommand("getDiagnosticReport", []),
  // WS4-A S5b: the connected-tier relay tunnel surface. `tunnelBeginPairing`
  // returns the user code + verification URL; `tunnelPollPairing` advances the
  // device-code pairing; `setConnectorEnabled` toggles the connector (starting/
  // stopping the tunnel); `tunnelStatus` reads the snapshot. In the free build
  // (or the dev shim) these are unsupported / inert.
  tunnelBeginPairing: () => callCommand("tunnelBeginPairing", []),
  tunnelPollPairing: () => callCommand("tunnelPollPairing", []),
  setConnectorEnabled: (enabled) =>
    callCommand("setConnectorEnabled", [enabled]),
  tunnelStatus: () => callCommand("tunnelStatus", []),
  // WS4-B S5: the connected-tier peer-to-peer notes sync surface. `syncStatus`
  // reads the engine's live state; `syncGetMyTicket` returns the shareable ticket
  // for pairing another device; `syncAddPeer` registers a peer from its ticket;
  // `syncNow` triggers a notes sync for a meeting. In the free build (or the dev
  // shim) these are unsupported / inert.
  syncStatus: () => callCommand("syncStatus", []),
  syncGetMyTicket: () => callCommand("syncGetMyTicket", []),
  syncAddPeer: (ticket) => callCommand("syncAddPeer", [ticket]),
  syncNow: (meetingId) => callCommand("syncNow", [meetingId]),
  // Issue #0003 voiceprint correction path (§2.4). `rejectMatch` clears the
  // speaker-name overlay for a label in a meeting and drops the contributing
  // (meeting, label) observation from the identity's gallery, so a rejected
  // match does not bias future identifications. `clearAllVoiceprints` is the
  // §4 privacy right-to-erasure path. Both are best-effort: a missing store
  // (enrolment OFF) is a silent no-op on the Rust side.
  rejectMatch: (meetingId, label, identityId, modelId) =>
    callCommand("rejectMatch", [meetingId, label, identityId, modelId]),
  clearAllVoiceprints: () => callCommand("clearAllVoiceprints", []),
  // Issue #0003 WU8 — identity management commands. `listVoiceprints` returns
  // every enrolled identity with per-condition gallery metadata (no embedding
  // bytes — §2.2). The mutation commands (rename / delete / merge / forget) are
  // best-effort no-ops when the store is degraded-to-off.
  listVoiceprints: () => callCommand("listVoiceprints", []),
  mergeVoiceprintIdentities: (keepId, mergedId) =>
    callCommand("mergeVoiceprintIdentities", [keepId, mergedId]),
  renameVoiceprintIdentity: (identityId, newName) =>
    callCommand("renameVoiceprintIdentity", [identityId, newName]),
  deleteVoiceprintIdentity: (identityId) =>
    callCommand("deleteVoiceprintIdentity", [identityId]),
  forgetMeetingVoiceprints: (meetingId) =>
    callCommand("forgetMeetingVoiceprints", [meetingId]),
};

// Re-export types that callers commonly need. `AppEvent` is the generated
// event union (re-exported via `./app-event`, which is the webview's single
// import site for the event type).
export type { AppEventPayload, AppEvent, AppError, Result };

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
 * Unwrap a `Result<T, AppError>` to `T`, throwing on error.
 *
 * Use this when calling code has a try/catch and does not need to inspect
 * the error variant — it just wants the success value or an exception.
 */
export function unwrap<T>(result: Result<T, AppError>): T {
  if (result.status === "ok") {
    return result.data;
  }
  throw new IpcCallError(result.error);
}

/**
 * Extract a human-readable message from an `AppError` discriminated union.
 */
export function ipcErrorMessage(err: AppError): string {
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
 * Typed error class wrapping an `AppError` payload.
 *
 * Thrown by `unwrap()` on error results. Callers that need to inspect the
 * error variant can switch on `error.ipcError.code`.
 */
export class IpcCallError extends Error {
  readonly ipcError: AppError;

  constructor(ipcError: AppError) {
    super(ipcErrorMessage(ipcError));
    this.name = "IpcCallError";
    this.ipcError = ipcError;
  }
}
