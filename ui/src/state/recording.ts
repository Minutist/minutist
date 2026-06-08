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
import { withPreferLargeAsrModel } from "./large-asr-model-settings";
import { withCaptureSystemAudio } from "./system-audio-settings";
import { withOnboardingCompleted, withTheme } from "./onboarding-settings";
import { withTranscriptionLanguage } from "./transcription-language-settings";
import { withNotesPaperRules } from "./notes-paper-settings";
import type { Theme } from "../ipc/bindings";

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

  start: async () => {
    // Guard: ASR model must be ready before recording can start.
    if (!useModelsStore.getState().isAsrModelReady) {
      set({ lastError: "ASR model not yet downloaded" });
      return;
    }
    try {
      const result = await commands.startRecording(get().selectedDeviceId);
      unwrap(result);
      set({ lastError: null });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
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

  handleEvent: (event) => {
    switch (event.kind) {
      case "state_changed":
        // Clear the live transcript when a new recording starts so the
        // previous session's transcript is no longer shown. Reset the
        // recording clock to null on any transition out of `recording`
        // (idle/stopping/paused) so notes anchors stop stamping once
        // capture is no longer advancing; the next `recording_clock`
        // event re-populates it while recording or after resume.
        if (event.state.kind === "recording") {
          set({ state: event.state, transcript: [] });
        } else {
          set({ state: event.state, recordingClockMs: null });
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
      case "transcript_segment":
        set((s) => ({ transcript: [...s.transcript, event.segment] }));
        break;
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
