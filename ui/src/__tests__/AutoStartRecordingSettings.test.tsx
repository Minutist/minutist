/**
 * "New meeting" auto-start-recording toggle webview tests.
 *
 * Mirrors the system-audio toggle test (`SystemAudio.test.tsx`): the
 * `auto_start_recording_on_new_meeting` setting round-trips through
 * `commands.updateSettings` at the existing seam, with no new command and no
 * raw invoke. Default-off (the prep screen), persists, and a fresh
 * `getSettings` reads it back.
 *
 * The setting only controls whether `MeetingControls` promotes a fresh draft
 * immediately or opens it as a prep screen; that branch is covered by
 * `MeetingControls.test.tsx`.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { act } from "@testing-library/react";

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
      createMeeting: vi.fn(),
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
  readAutoStartRecordingOnNewMeeting,
  withAutoStartRecordingOnNewMeeting,
} from "../state/auto-start-recording-settings";
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

describe("auto_start_recording_on_new_meeting toggle round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("defaults to off (false, the prep screen) when the field is absent or settings unloaded", () => {
    expect(readAutoStartRecordingOnNewMeeting(BASE_SETTINGS)).toBe(false);
    expect(readAutoStartRecordingOnNewMeeting(null)).toBe(false);
  });

  it("setAutoStartRecordingOnNewMeeting persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setAutoStartRecordingOnNewMeeting(true);

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      auto_start_recording_on_new_meeting: true,
    });
    expect(
      readAutoStartRecordingOnNewMeeting(useRecordingStore.getState().settings),
    ).toBe(true);
  });

  it("round-trips back to off, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withAutoStartRecordingOnNewMeeting(BASE_SETTINGS, true),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setAutoStartRecordingOnNewMeeting(false);
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      auto_start_recording_on_new_meeting: false,
    });

    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(
        withAutoStartRecordingOnNewMeeting(BASE_SETTINGS, false) as Settings,
      ),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(
      readAutoStartRecordingOnNewMeeting(useRecordingStore.getState().settings),
    ).toBe(false);
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setAutoStartRecordingOnNewMeeting(true);
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});
