/**
 * Live co-pilot control webview tests.
 *
 * Mirrors `GpuAcceleration.test.tsx`: the runtime `live_agent_enabled` setting
 * round-trips through `commands.updateSettings` at the existing seam, with no
 * new command and no raw invoke. It is a tri-state `LiveAgentMode` ("off" by
 * default); the control persists a mode and a fresh `getSettings` reads it back.
 * Default-suite test: no model, GPU, or microphone — the mocked seams are the
 * fixtures. Whether the agent actually runs is decided by
 * `minutist_common::live_agent_should_run` (discrete-GPU gate); the UI sets the
 * mode half only. See `architecture/cross-cutting.md` — "Live agent".
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
  readLiveAgentMode,
  withLiveAgentMode,
} from "../state/live-agent-settings";
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

describe("live_agent_enabled mode round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("defaults to off when the field is absent or settings unloaded", () => {
    // An older store / not-yet-loaded snapshot reads as "off" (matches the
    // backend `#[serde(default)]` of `Off` — the co-pilot is opt-in).
    expect(readLiveAgentMode(BASE_SETTINGS)).toBe("off");
    expect(readLiveAgentMode(null)).toBe("off");
  });

  it("setLiveAgentMode persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setLiveAgentMode("on");

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      live_agent_enabled: "on",
    });
    expect(readLiveAgentMode(useRecordingStore.getState().settings)).toBe("on");
  });

  it("round-trips to auto, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withLiveAgentMode(BASE_SETTINGS, "on"),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setLiveAgentMode("auto");
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      live_agent_enabled: "auto",
    });

    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(withLiveAgentMode(BASE_SETTINGS, "auto") as Settings),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(readLiveAgentMode(useRecordingStore.getState().settings)).toBe(
      "auto",
    );
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setLiveAgentMode("on");
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});
