/**
 * Zustand store for recording state.
 *
 * This is the sole source of truth for transient UI state. All mutations
 * flow through either IPC command callbacks or `handleEvent`, which is
 * called by the global `useAppEventBridge` hook in `shell/event-listener.tsx`.
 */
import { create } from "zustand";
import { commands, unwrap, ipcErrorMessage } from "../ipc/client";
import type {
  RecordingState,
  AudioDevice,
  Settings,
  Segment,
} from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";
import { useModelsStore } from "./models";
import { withDiarizationEnabled } from "./diarization-settings";
import { withGpuAcceleration } from "./gpu-acceleration-settings";
import { withPreloadSummariser } from "./preload-summariser-settings";
import { withPreferLargeAsrModel } from "./large-asr-model-settings";
import { withCaptureSystemAudio } from "./system-audio-settings";
import { withOnboardingCompleted, withTheme } from "./onboarding-settings";
import { withTranscriptionLanguage } from "./transcription-language-settings";
import { withNotesPaperRules } from "./notes-paper-settings";
import {
  withSummaryPreset,
  withSummarySystemPrompt,
} from "./summary-preset-settings";
import {
  withMcpEnabled,
  withMcpPort,
  withMcpWriteTools,
} from "./mcp-settings";
import type { Theme, SummaryPreset } from "../ipc/bindings";

export type { RecordingState, AudioDevice, AppEvent, Settings, Segment };

export type RecordingStore = {
  state: RecordingState;
  devices: AudioDevice[];
  selectedDeviceId: string | null;
  /**
   * Persisted settings snapshot. Populated on mount via `refreshSettings`;
   * mutations from the UI go through `setSelectedDevice` (etc.) and write
   * back via `commands.updateSettings`.
   */
  settings: Settings | null;
  meter: { peak: number; rms: number };
  lastError: string | null;
  /**
   * Client-only optimistic transient (live-test UX T1): `true` from the moment
   * `start()` is invoked until the backend confirms the recording has begun (the
   * `state_changed` → `recording` event) or `start()` fails. It exists because
   * the FIRST record may lazy-load the ASR model (~29 s) with no other feedback,
   * so the UI would otherwise look dead. While `true` the Start control is
   * disabled (a second press is impossible) and a "Preparing transcription
   * model…" status is shown. NOT part of the Rust `RecordingState` — the backend
   * has no "preparing" state; this is purely a UI affordance over the gap between
   * the start request and the `recording` confirmation.
   */
  preparing: boolean;
  /** Live transcript for the current recording session. Cleared when a new recording starts. */
  transcript: Segment[];
  /**
   * The live capture-sample, pause-**excluding** recording clock in ms — the
   * same timeline as `Segment::start_ms`. Updated by `recording_clock` events
   * (~5 Hz) and reset to `null` when recording stops (transition to
   * idle/stopping).
   *
   * This is the ONLY valid source for notes paragraph anchors. Do NOT derive
   * anchors from `Date.now() - started_at_ms`: that wall-clock delta is
   * pause-including and drifts from the audio/transcript timeline (see
   * `architecture/cross-cutting.md` — "Notes paragraph-anchor clock").
   *
   * `null` while idle (no recording in progress).
   */
  recordingClockMs: number | null;

  // actions
  refreshDevices: () => Promise<void>;
  refreshSettings: () => Promise<void>;
  /**
   * Pre-load the routed ASR model so the FIRST record is not a cold ~29 s load
   * (live-test UX T2). Fire-and-forget when the recording/meeting workspace
   * opens; idempotent and best-effort backend-side (a not-yet-downloaded model
   * warms nothing). app-main also prewarms on startup; this is the in-session
   * trigger for when the window opens after a settings change.
   */
  prewarmAsr: () => Promise<void>;
  start: () => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  setSelectedDevice: (id: string | null) => Promise<void>;
  /**
   * Toggle the Phase-6 diarization-enabled setting (off by default), persisting
   * via `commands.updateSettings` so the choice survives an app restart — the
   * same round-trip-through-settings pattern as `setSelectedDevice`.
   */
  setDiarizationEnabled: (enabled: boolean) => Promise<void>;
  /**
   * Toggle the runtime GPU-acceleration setting (on by default), persisting via
   * `commands.updateSettings` so the choice survives an app restart — the same
   * round-trip-through-settings pattern as `setDiarizationEnabled`. When off,
   * inference runs on CPU even in a GPU-feature build.
   */
  setGpuAcceleration: (enabled: boolean) => Promise<void>;
  /**
   * Toggle preloading the summary/chat LLM at startup (on by default),
   * persisting via `commands.updateSettings` — the same round-trip-through-
   * settings pattern as `setGpuAcceleration`. On: the model is warmed at startup
   * (when downloaded) so the first Summarise / chat is instant. Off: it loads
   * on-demand on first use, keeping idle memory lower.
   */
  setPreloadSummariser: (enabled: boolean) => Promise<void>;
  /**
   * Opt the Qwen ASR branch into the larger 1.7B GPU tier (off by default),
   * persisting via `commands.updateSettings` — the same round-trip-through-
   * settings pattern as `setGpuAcceleration`. Only affects languages that route
   * to Qwen; the Parakeet branch ignores it.
   */
  setPreferLargeAsrModel: (enabled: boolean) => Promise<void>;
  /**
   * Toggle the system-audio (call / loopback) capture setting (off by default),
   * persisting via `commands.updateSettings` so the choice survives an app
   * restart — the same round-trip-through-settings pattern as
   * `setDiarizationEnabled`. When on, the call audio is mixed with the mic so
   * all participants are transcribed; turn it off if the mic also hears the
   * call from the speakers (echo).
   */
  setCaptureSystemAudio: (enabled: boolean) => Promise<void>;
  /**
   * Set the ASR transcription-language hint, persisting via
   * `commands.updateSettings` so the choice survives an app restart — the same
   * round-trip-through-settings pattern as `setTheme`. A full English name
   * forces that language; the sentinel "auto" disables forcing (auto-detect).
   */
  setTranscriptionLanguage: (language: string) => Promise<void>;
  /**
   * Set the UI colour-scheme preference, persisting via `commands.updateSettings`
   * (the same round-trip-through-settings pattern as `setSelectedDevice`). Used
   * by the Phase-7 onboarding quick-settings step.
   */
  setTheme: (theme: Theme) => Promise<void>;
  /**
   * Toggle the notes-editor writing-paper rules (on by default), persisting via
   * `commands.updateSettings` — the same round-trip-through-settings pattern as
   * `setGpuAcceleration`. Presentation-only; the editor reads the field back and
   * toggles a class. The structural oxblood margin rule is unaffected.
   */
  setNotesPaperRules: (enabled: boolean) => Promise<void>;
  /**
   * Mark the first-run onboarding flow complete (Phase 7), persisting
   * `onboarding_completed = true` via `commands.updateSettings`. The shell gate
   * reads this back from the settings snapshot to reveal the main app. Same
   * round-trip-through-settings pattern as `setDiarizationEnabled`.
   */
  setOnboardingCompleted: (completed: boolean) => Promise<void>;
  /**
   * Set the summary prompt preset (Phase 9 — D4), persisting via
   * `commands.updateSettings` — the same round-trip-through-settings pattern as
   * `setTheme`. The selected preset drives the effective summary prompt unless a
   * non-empty custom override is set in `summary_system_prompt`.
   */
  setSummaryPreset: (preset: SummaryPreset) => Promise<void>;
  /**
   * Set the custom summary-prompt override (Phase 9 — D4), persisting via
   * `commands.updateSettings`. An empty string means "use the selected preset";
   * a non-empty value overrides the preset (the backend's
   * `Settings::effective_summary_prompt`).
   */
  setSummarySystemPrompt: (prompt: string) => Promise<void>;
  /**
   * Toggle the in-process MCP server (Phase 10 — off by default), persisting via
   * `commands.updateSettings`. The listener is spawned once at startup gated on
   * this flag, so toggling at runtime is a documented restart-required for v1
   * (the UI advises a restart). Same round-trip-through-settings pattern as
   * `setDiarizationEnabled`.
   */
  setMcpEnabled: (enabled: boolean) => Promise<void>;
  /**
   * Set the MCP server's fixed loopback port (Phase 10 — D1, default 8765),
   * persisting via `commands.updateSettings`. Restart-required like
   * `setMcpEnabled`.
   */
  setMcpPort: (port: number) => Promise<void>;
  /**
   * Toggle MCP write-tool exposure (Phase 10 — D3, off by default = read-only
   * over MCP), persisting via `commands.updateSettings`. With it on, the
   * reversible writes (set speaker name / rename meeting) join `tools/list`;
   * heavy/destructive ops never do. Restart-required like `setMcpEnabled`.
   */
  setMcpWriteTools: (enabled: boolean) => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useRecordingStore = create<RecordingStore>((set, get) => ({
  state: { kind: "idle" },
  devices: [],
  selectedDeviceId: null,
  settings: null,
  meter: { peak: 0, rms: 0 },
  lastError: null,
  preparing: false,
  transcript: [],
  recordingClockMs: null,

  refreshDevices: async () => {
    try {
      const result = await commands.listDevices();
      const devices = unwrap(result);
      set({ devices });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  refreshSettings: async () => {
    try {
      const result = await commands.getSettings();
      const settings = unwrap(result);
      set({
        settings,
        selectedDeviceId: settings.input_device_id ?? null,
      });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  prewarmAsr: async () => {
    // Best-effort: a prewarm failure is non-fatal (the backend keeps the lazy
    // load as the fallback), so swallow errors rather than surfacing lastError.
    try {
      await commands.prewarmAsr();
    } catch {
      // ignore — prewarm is an optimisation, not a required step.
    }
  },

  start: async () => {
    // Guard against a double-press / start-while-busy (live-test UX T1a): only a
    // genuinely idle recorder may start. A second press while Recording would
    // otherwise re-call `startRecording`, which the orchestrator rejects with
    // "start called when not idle" (surfaced as `lastError`). The `preparing`
    // flag covers the window between this call and the `recording` confirmation.
    if (get().state.kind !== "idle" || get().preparing) {
      return;
    }
    // Guard: ASR model must be ready before recording can start.
    if (!useModelsStore.getState().isAsrModelReady) {
      set({ lastError: "ASR model not yet downloaded" });
      return;
    }
    // Optimistic transient (T1a/T1c): disable the control immediately and show
    // "Preparing transcription model…" while the backend opens capture and (on
    // the first record) lazy-loads the ASR model. Cleared on the `recording`
    // state event (see `handleEvent`) or on error below.
    set({ preparing: true, lastError: null });
    try {
      const result = await commands.startRecording(get().selectedDeviceId);
      unwrap(result);
      set({ lastError: null });
    } catch (err) {
      // The start failed before any `recording` event will arrive, so clear the
      // optimistic flag here (the event path won't).
      set({
        preparing: false,
        lastError: err instanceof Error ? err.message : String(err),
      });
    }
  },

  pause: async () => {
    try {
      const result = await commands.pauseRecording();
      unwrap(result);
      set({ lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  resume: async () => {
    try {
      const result = await commands.resumeRecording();
      unwrap(result);
      set({ lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  stop: async () => {
    try {
      const result = await commands.stopRecording();
      unwrap(result);
      set({ lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setSelectedDevice: async (id) => {
    // Update local store immediately for UI responsiveness.
    set({ selectedDeviceId: id });

    // Persist via `update_settings` so the choice survives an app restart.
    // The orchestrator falls back to `settings.input_device_id` when the
    // caller passes `device_id = None` (see Orchestrator::start in
    // crates/orchestrator/src/lib.rs), so persisting is what makes the
    // device selection sticky.
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object. The next setSelectedDevice
      // call after settings load will persist.
      return;
    }
    const next: Settings = { ...current, input_device_id: id };
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setDiarizationEnabled: async (enabled) => {
    // Persist via `update_settings` so the choice survives an app restart, the
    // same round-trip-through-settings pattern `setSelectedDevice` uses. The
    // `diarization_enabled` field is modelled as an augmentation of the
    // generated `Settings` type until the backend JOIN regenerates the bindings
    // (see `./diarization-settings`); the JSON round-trip carries it losslessly.
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withDiarizationEnabled(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setGpuAcceleration: async (enabled) => {
    // Persist via `update_settings` so the choice survives an app restart, the
    // same round-trip-through-settings pattern `setDiarizationEnabled` uses.
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withGpuAcceleration(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setPreloadSummariser: async (enabled) => {
    // Persist via `update_settings`, the same round-trip pattern as
    // `setGpuAcceleration`.
    const current = get().settings;
    if (current === null) {
      return;
    }
    const next = withPreloadSummariser(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setPreferLargeAsrModel: async (enabled) => {
    // Persist via `update_settings`, the same pattern as `setGpuAcceleration`.
    const current = get().settings;
    if (current === null) {
      return;
    }
    const next = withPreferLargeAsrModel(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setCaptureSystemAudio: async (enabled) => {
    // Persist via `update_settings` so the choice survives an app restart, the
    // same round-trip-through-settings pattern `setDiarizationEnabled` uses.
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withCaptureSystemAudio(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setTranscriptionLanguage: async (language) => {
    // Persist via `update_settings` so the choice survives an app restart, the
    // same round-trip-through-settings pattern `setTheme` uses. The sentinel
    // "auto" reaches the store/Rust unchanged (the resolver maps it to None).
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withTranscriptionLanguage(current, language);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setTheme: async (theme) => {
    // Persist via `update_settings` (same round-trip as `setSelectedDevice`).
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withTheme(current, theme);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setNotesPaperRules: async (enabled) => {
    // Persist via `update_settings` (same round-trip as `setGpuAcceleration`).
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withNotesPaperRules(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setOnboardingCompleted: async (completed) => {
    // Persist via `update_settings` so the gate stays satisfied across restarts,
    // the same round-trip-through-settings pattern `setDiarizationEnabled` uses.
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withOnboardingCompleted(current, completed);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setSummaryPreset: async (preset) => {
    // Persist via `update_settings` (same round-trip as `setTheme`).
    const current = get().settings;
    if (current === null) {
      // refreshSettings hasn't completed yet; skip the write to avoid
      // clobbering with a partial object.
      return;
    }
    const next = withSummaryPreset(current, preset);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setSummarySystemPrompt: async (prompt) => {
    // Persist via `update_settings` (same round-trip as `setSummaryPreset`).
    const current = get().settings;
    if (current === null) {
      return;
    }
    const next = withSummarySystemPrompt(current, prompt);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setMcpEnabled: async (enabled) => {
    // Persist via `update_settings` (same round-trip as `setDiarizationEnabled`).
    const current = get().settings;
    if (current === null) {
      return;
    }
    const next = withMcpEnabled(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setMcpPort: async (port) => {
    const current = get().settings;
    if (current === null) {
      return;
    }
    const next = withMcpPort(current, port);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  setMcpWriteTools: async (enabled) => {
    const current = get().settings;
    if (current === null) {
      return;
    }
    const next = withMcpWriteTools(current, enabled);
    try {
      const result = await commands.updateSettings(next);
      unwrap(result);
      set({ settings: next, lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  handleEvent: (event) => {
    switch (event.kind) {
      case "state_changed":
        // Clear the live transcript when a new recording starts so the
        // previous session's transcript is no longer shown. Reset the
        // recording clock to null on any transition out of `recording`
        // (idle/stopping/paused) so notes anchors stop stamping once
        // capture is no longer advancing; the next `recording_clock`
        // event re-populates it while recording or after resume.
        //
        // Any backend state change also resolves the optimistic `preparing`
        // transient (T1): once `recording` arrives the model is loaded and
        // capture is live; any other terminal state (e.g. an error path that
        // left the recorder idle) likewise ends the preparing window.
        if (event.state.kind === "recording") {
          set({ state: event.state, transcript: [], preparing: false });
        } else {
          set({ state: event.state, recordingClockMs: null, preparing: false });
        }
        break;
      case "audio_meter":
        set({ meter: { peak: event.frame.peak, rms: event.frame.rms } });
        break;
      case "recording_clock":
        set({ recordingClockMs: event.clock_ms });
        break;
      case "devices_changed":
        // Trigger a re-fetch; ignore the promise — fire and forget.
        void get().refreshDevices();
        break;
      case "transcript_segment": {
        // Only append segments from the LIVE recording. An offline/background
        // re-transcribe emits `transcript_segment` for a previously-stopped
        // meeting while the recorder is idle — or, if the user started the next
        // meeting (preempting the repair), while the recorder is recording a
        // DIFFERENT meeting. Matching on the live meeting_id keeps those
        // background segments out of the live transcript array in both cases —
        // that pass persists the full transcript and emits `transcript_ready`,
        // which the meetings store handles instead.
        const live = get().state;
        const liveId =
          live.kind === "recording" || live.kind === "paused"
            ? live.meeting_id
            : null;
        if (liveId !== null && liveId === event.meeting_id) {
          set((s) => ({ transcript: [...s.transcript, event.segment] }));
        }
        break;
      }
      case "error_occurred":
        set({ lastError: ipcErrorMessage(event.error) });
        break;
      // model_download_progress is handled by the models store via its own
      // handleEvent dispatch (mounted alongside this bridge in event-listener.tsx).
      // diarization_complete is handled by the meetings store (it re-reads the
      // affected meeting's transcript). summary_ready is handled by the summary
      // store. settings_changed: not handled here.
      default:
        break;
    }
  },
}));
