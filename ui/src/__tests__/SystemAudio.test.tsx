/**
 * System-audio (call / loopback) capture toggle webview tests.
 *
 * Mirrors the GPU-acceleration toggle test (`GpuAcceleration.test.tsx`): the
 * `capture_system_audio` setting round-trips through `commands.updateSettings`
 * at the existing seam, with no new command and no raw invoke. Default-off,
 * persists, and a fresh `getSettings` reads it back. This is a default-suite
 * test: it needs no model, GPU, microphone, or render device — the mocked seams
 * are the fixtures.
 *
 * The UI toggle only controls the setting half; whether loopback actually opens
 * is a backend / platform concern (Windows-only, with mic-only fallback). See
 * `architecture/components.md` — the `audio-capture` section.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { act } from "@testing-library/react";

// ---------------------------------------------------------------------------
// Tauri API mocks — declared before importing any module that pulls in
// `../ipc/bindings`.
// ---------------------------------------------------------------------------
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

vi.mock("../ipc/bindings", () => {
  const updateSettings = vi.fn();
  const getSettings = vi.fn();
  return {
    commands: {
      updateSettings,
      getSettings,
      listDevices: vi.fn(),
      startRecording: vi.fn(),
      pauseRecording: vi.fn(),
      resumeRecording: vi.fn(),
      stopRecording: vi.fn(),
      getRecordingState: vi.fn(),
      listModels: vi.fn(),
      ensureModel: vi.fn(),
    },
    events: {},
  };
});

import { useRecordingStore } from "../state/recording";
import { commands } from "../ipc/bindings";
import {
  readCaptureSystemAudio,
  withCaptureSystemAudio,
} from "../state/system-audio-settings";
import type { Settings } from "../ipc/bindings";

const okVoid = Promise.resolve({ status: "ok" as const, data: null });
const okSettings = (s: Settings) =>
  Promise.resolve({ status: "ok" as const, data: s });

const BASE_SETTINGS: Settings = {
  input_device_id: null,
  theme: "system",
  data_directory: null,
  start_hidden: false,
};

function resetStore() {
  act(() => {
    useRecordingStore.setState({
      state: { kind: "idle" },
      transcript: [],
      settings: null,
      lastError: null,
    });
  });
}

describe("capture_system_audio toggle round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("defaults to off (false) when the field is absent or settings unloaded", () => {
    // An older store / not-yet-loaded snapshot reads as off (matches the
    // backend `#[serde(default)]` of false).
    expect(readCaptureSystemAudio(BASE_SETTINGS)).toBe(false);
    expect(readCaptureSystemAudio(null)).toBe(false);
  });

  it("setCaptureSystemAudio persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setCaptureSystemAudio(true);

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      capture_system_audio: true,
    });
    expect(
      readCaptureSystemAudio(useRecordingStore.getState().settings),
    ).toBe(true);
  });

  it("round-trips back to off, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withCaptureSystemAudio(BASE_SETTINGS, true),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setCaptureSystemAudio(false);
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      capture_system_audio: false,
    });

    // Simulate a reload: getSettings returns the persisted object.
    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(withCaptureSystemAudio(BASE_SETTINGS, false) as Settings),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(
      readCaptureSystemAudio(useRecordingStore.getState().settings),
    ).toBe(false);
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setCaptureSystemAudio(true);
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});
