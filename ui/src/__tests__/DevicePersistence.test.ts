/**
 * Unit test for the device-picker persistence path (PO finding).
 *
 * Verifies that `setSelectedDevice` calls `commands.updateSettings` so the
 * user's choice survives an app restart. The orchestrator falls back to
 * `Settings::input_device_id` when the IPC start_recording call passes
 * `device_id = None`, so the round-trip-through-settings is what makes the
 * choice sticky.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Tauri API mocks — declared before importing any module that pulls in
// `../ipc/bindings`.
// ---------------------------------------------------------------------------
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  Channel: vi.fn(),
}));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({
  WebviewWindow: vi.fn(),
}));

// Mock the generated bindings module. The store's actions call into
// `commands.*`; we intercept each call and assert the shape.
vi.mock("../ipc/bindings", () => {
  const getSettings = vi.fn();
  const updateSettings = vi.fn();
  const listDevices = vi.fn();
  return {
    commands: {
      getSettings,
      updateSettings,
      listDevices,
      startRecording: vi.fn(),
      pauseRecording: vi.fn(),
      resumeRecording: vi.fn(),
      stopRecording: vi.fn(),
      getRecordingState: vi.fn(),
    },
    events: {},
  };
});

import { commands } from "../ipc/bindings";
import { useRecordingStore } from "../state/recording";
import type { Settings } from "../ipc/bindings";

const okSettings = (s: Settings) =>
  Promise.resolve({ status: "ok" as const, data: s });

const okVoid = Promise.resolve({ status: "ok" as const, data: null });

describe("device-picker persistence", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useRecordingStore.setState({
      state: { kind: "idle" },
      devices: [],
      selectedDeviceId: null,
      settings: null,
      meter: { peak: 0, rms: 0 },
      lastError: null,
    });
  });

  it("refreshSettings populates store from getSettings", async () => {
    const stored: Settings = {
      input_device_id: "hw:1,0",
      theme: "dark",
      data_directory: null,
      start_hidden: false,
    };
    vi.mocked(commands.getSettings).mockReturnValueOnce(okSettings(stored));

    await useRecordingStore.getState().refreshSettings();

    expect(commands.getSettings).toHaveBeenCalledOnce();
    expect(useRecordingStore.getState().settings).toEqual(stored);
    expect(useRecordingStore.getState().selectedDeviceId).toBe("hw:1,0");
  });

  it("setSelectedDevice persists via updateSettings", async () => {
    const initial: Settings = {
      input_device_id: null,
      theme: "system",
      data_directory: null,
      start_hidden: false,
    };
    useRecordingStore.setState({ settings: initial });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setSelectedDevice("hw:1,0");

    expect(commands.updateSettings).toHaveBeenCalledWith({
      input_device_id: "hw:1,0",
      theme: "system",
      data_directory: null,
      start_hidden: false,
    });
    expect(useRecordingStore.getState().settings?.input_device_id).toBe(
      "hw:1,0"
    );
    expect(useRecordingStore.getState().selectedDeviceId).toBe("hw:1,0");
  });

  it("setSelectedDevice skips the IPC write when settings are not yet loaded", async () => {
    useRecordingStore.setState({ settings: null });

    await useRecordingStore.getState().setSelectedDevice("hw:1,0");

    expect(commands.updateSettings).not.toHaveBeenCalled();
    // Local UI state still updates so the picker shows the selection.
    expect(useRecordingStore.getState().selectedDeviceId).toBe("hw:1,0");
  });

  it("updateSettings failure surfaces via lastError but does not clobber the local selection", async () => {
    const initial: Settings = {
      input_device_id: null,
      theme: "system",
      data_directory: null,
      start_hidden: false,
    };
    useRecordingStore.setState({ settings: initial });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(
      Promise.resolve({
        status: "error" as const,
        error: { code: "io", context: "disk full" },
      })
    );

    await useRecordingStore.getState().setSelectedDevice("hw:1,0");

    expect(useRecordingStore.getState().selectedDeviceId).toBe("hw:1,0");
    expect(useRecordingStore.getState().lastError).toBeTruthy();
    // Settings snapshot is NOT updated because the write failed.
    expect(useRecordingStore.getState().settings).toEqual(initial);
  });
});
