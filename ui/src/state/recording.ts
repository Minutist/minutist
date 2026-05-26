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
  RecordingStateType,
  AudioDeviceType,
  AppEventType,
} from "../ipc/bindings";

export type { RecordingStateType, AudioDeviceType, AppEventType };

export type RecordingStore = {
  state: RecordingStateType;
  devices: AudioDeviceType[];
  selectedDeviceId: string | null;
  meter: { peak: number; rms: number };
  lastError: string | null;

  // actions
  refreshDevices: () => Promise<void>;
  start: () => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  setSelectedDevice: (id: string | null) => void;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEventType) => void;
};

export const useRecordingStore = create<RecordingStore>((set, get) => ({
  state: { kind: "idle" },
  devices: [],
  selectedDeviceId: null,
  meter: { peak: 0, rms: 0 },
  lastError: null,

  refreshDevices: async () => {
    try {
      const result = await commands.listDevices();
      const devices = unwrap(result);
      set({ devices });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  start: async () => {
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

  setSelectedDevice: (id) => {
    set({ selectedDeviceId: id });
  },

  handleEvent: (event) => {
    switch (event.kind) {
      case "state_changed":
        set({ state: event.state });
        break;
      case "audio_meter":
        set({ meter: { peak: event.frame.peak, rms: event.frame.rms } });
        break;
      case "devices_changed":
        // Trigger a re-fetch; ignore the promise — fire and forget.
        void get().refreshDevices();
        break;
      case "error_occurred":
        set({ lastError: ipcErrorMessage(event.error) });
        break;
      // Other event kinds (transcript_segment, diarization_complete,
      // summary_ready, model_download_progress, settings_changed) are not
      // handled in Phase 1. They arrive but produce no state change here.
      default:
        break;
    }
  },
}));
