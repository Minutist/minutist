/**
 * GPU-acceleration toggle webview tests.
 *
 * Mirrors the Phase-6 diarization-toggle test (`Diarization.test.tsx`, section
 * 3): the runtime `gpu_acceleration` setting round-trips through
 * `commands.updateSettings` at the existing seam, with no new command and no
 * raw invoke (rule A9). Default-on, persists, and a fresh `getSettings` reads it
 * back. This is a default-suite test: it needs no model, GPU, or microphone —
 * the mocked seams are the fixtures.
 *
 * GPU offload happens ONLY when BOTH the build has a GPU feature AND this
 * setting is true; the UI toggle only controls the setting half. See
 * `architecture/cross-cutting.md` — "GPU portability".
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

// The recording store rounds the toggle through `commands.updateSettings`; mock
// the generated bindings so the call is observable (mirrors Diarization).
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
  readGpuAcceleration,
  withGpuAcceleration,
} from "../state/gpu-acceleration-settings";
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

describe("gpu_acceleration toggle round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("defaults to on (true) when the field is absent or settings unloaded", () => {
    // An older store / not-yet-loaded snapshot reads as on (matches the
    // backend `#[serde(default)]` of true).
    expect(readGpuAcceleration(BASE_SETTINGS)).toBe(true);
    expect(readGpuAcceleration(null)).toBe(true);
  });

  it("setGpuAcceleration persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setGpuAcceleration(false);

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      gpu_acceleration: false,
    });
    // The store snapshot reflects the persisted value.
    expect(readGpuAcceleration(useRecordingStore.getState().settings)).toBe(
      false,
    );
  });

  it("round-trips back to on, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withGpuAcceleration(BASE_SETTINGS, false),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setGpuAcceleration(true);
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      gpu_acceleration: true,
    });

    // Simulate a reload: getSettings returns the persisted object.
    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(withGpuAcceleration(BASE_SETTINGS, true) as Settings),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(readGpuAcceleration(useRecordingStore.getState().settings)).toBe(
      true,
    );
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setGpuAcceleration(false);
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});
