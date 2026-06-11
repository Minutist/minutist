/**
 * Output-language setting round-trip tests.
 *
 * Mirrors TranscriptionLanguage.test.tsx: the `output_language` setting
 * round-trips through `commands.updateSettings` at the existing seam, with no
 * new command and no raw invoke. Defaults to "auto", persists a language name,
 * persists the "auto" sentinel unchanged, and a fresh `getSettings` reads it
 * back.
 *
 * The UI control only sets the setting half; resolving "auto" to the host
 * locale and appending the language instruction to LLM prompts is a backend
 * concern — see `architecture/components.md` — the `ipc-bridge` section.
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
  readOutputLanguage,
  withOutputLanguage,
} from "../state/output-language-settings";
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

describe("output_language setting round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("defaults to 'auto' when the field is absent or settings unloaded", () => {
    // An older store / not-yet-loaded snapshot reads as "auto" (the schema
    // default, matching the Rust default fn).
    expect(readOutputLanguage(BASE_SETTINGS)).toBe("auto");
    expect(readOutputLanguage(null)).toBe("auto");
  });

  it("setOutputLanguage persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setOutputLanguage("French");

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      output_language: "French",
    });
    expect(readOutputLanguage(useRecordingStore.getState().settings)).toBe(
      "French",
    );
  });

  it("persists the 'auto' sentinel unchanged", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setOutputLanguage("auto");

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      output_language: "auto",
    });
    expect(readOutputLanguage(useRecordingStore.getState().settings)).toBe(
      "auto",
    );
  });

  it("round-trips a name, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withOutputLanguage(BASE_SETTINGS, "German"),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setOutputLanguage("Japanese");
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      output_language: "Japanese",
    });

    // Simulate a reload: getSettings returns the persisted object.
    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(withOutputLanguage(BASE_SETTINGS, "Spanish") as Settings),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(readOutputLanguage(useRecordingStore.getState().settings)).toBe(
      "Spanish",
    );
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setOutputLanguage("Italian");
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});
