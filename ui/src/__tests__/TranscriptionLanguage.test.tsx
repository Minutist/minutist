/**
 * ASR transcription-language hint webview tests.
 *
 * Mirrors the system-audio toggle test (`SystemAudio.test.tsx`): the
 * `transcription_language` setting round-trips through `commands.updateSettings`
 * at the existing seam, with no new command and no raw invoke. Defaults to
 * "English", persists a real name, persists the "auto" sentinel unchanged, and a
 * fresh `getSettings` reads it back. This is a default-suite test: it needs no
 * model, GPU, microphone, or render device — the mocked seams are the fixtures.
 *
 * The UI control only sets the setting half; resolving "auto" to no-prefix
 * auto-detect and a real name to a forced prefix is a backend concern (see
 * `architecture/components.md` — the `settings` and `asr-runtime` sections).
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
  readTranscriptionLanguage,
  withTranscriptionLanguage,
} from "../state/transcription-language-settings";
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

describe("transcription_language hint round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("defaults to English when the field is absent or settings unloaded", () => {
    // An older store / not-yet-loaded snapshot reads as "English" (matches the
    // backend default fn).
    expect(readTranscriptionLanguage(BASE_SETTINGS)).toBe("English");
    expect(readTranscriptionLanguage(null)).toBe("English");
  });

  it("setTranscriptionLanguage persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setTranscriptionLanguage("Spanish");

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      transcription_language: "Spanish",
    });
    expect(
      readTranscriptionLanguage(useRecordingStore.getState().settings),
    ).toBe("Spanish");
  });

  it("persists the 'auto' sentinel unchanged", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setTranscriptionLanguage("auto");

    // The sentinel reaches the store/Rust unchanged (the resolver maps it to
    // None backend-side).
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      transcription_language: "auto",
    });
    expect(
      readTranscriptionLanguage(useRecordingStore.getState().settings),
    ).toBe("auto");
  });

  it("round-trips a name, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withTranscriptionLanguage(BASE_SETTINGS, "English"),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setTranscriptionLanguage("Spanish");
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      transcription_language: "Spanish",
    });

    // Simulate a reload: getSettings returns the persisted object.
    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(
        withTranscriptionLanguage(BASE_SETTINGS, "Japanese") as Settings,
      ),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(
      readTranscriptionLanguage(useRecordingStore.getState().settings),
    ).toBe("Japanese");
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setTranscriptionLanguage("Spanish");
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});
