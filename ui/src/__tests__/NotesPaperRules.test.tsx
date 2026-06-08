/**
 * Notes writing-paper-rules toggle webview tests.
 *
 * Mirrors the GPU-acceleration toggle test: the `notes_paper_rules` setting
 * round-trips through `commands.updateSettings` at the existing seam, with no
 * new command and no raw invoke (rule A9). Default-on, persists, and a fresh
 * `getSettings` reads it back. This is a default-suite test: the mocked seams
 * are the fixtures.
 *
 * Presentation-only: the editor reads the field back and toggles a class that
 * paints the faint horizontal rules; the structural oxblood margin rule is
 * unaffected.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, cleanup } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));
vi.mock("../ipc/notes", () => ({
  saveNotes: vi.fn().mockResolvedValue(undefined),
  loadNotes: vi.fn().mockResolvedValue(null),
}));

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
import { Editor } from "../editor/Editor";
import {
  readNotesPaperRules,
  withNotesPaperRules,
} from "../state/notes-paper-settings";
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

describe("notes_paper_rules toggle round-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("defaults to on (true) when the field is absent or settings unloaded", () => {
    expect(readNotesPaperRules(BASE_SETTINGS)).toBe(true);
    expect(readNotesPaperRules(null)).toBe(true);
  });

  it("setNotesPaperRules persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setNotesPaperRules(false);

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      notes_paper_rules: false,
    });
    expect(readNotesPaperRules(useRecordingStore.getState().settings)).toBe(
      false,
    );
  });

  it("round-trips back to on, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withNotesPaperRules(BASE_SETTINGS, false),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setNotesPaperRules(true);
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      notes_paper_rules: true,
    });

    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(withNotesPaperRules(BASE_SETTINGS, true) as Settings),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(readNotesPaperRules(useRecordingStore.getState().settings)).toBe(
      true,
    );
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setNotesPaperRules(false);
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});

describe("Editor ruled-class application (notes_paper_rules)", () => {
  afterEach(() => {
    act(() => {
      cleanup();
      useRecordingStore.setState({ settings: null });
    });
  });

  /** Mount the Editor with the given settings and return its `.notes-editor` root. */
  async function mountEditor(settings: Settings | null): Promise<HTMLElement> {
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" }, settings });
    });
    let container!: HTMLElement;
    await act(async () => {
      container = render(<Editor />).container;
      await Promise.resolve();
    });
    return container.querySelector(".notes-editor") as HTMLElement;
  }

  it("adds notes-editor--ruled when notes_paper_rules is true", async () => {
    const root = await mountEditor({ ...BASE_SETTINGS, notes_paper_rules: true });
    expect(root.classList.contains("notes-editor--ruled")).toBe(true);
  });

  it("omits notes-editor--ruled when notes_paper_rules is false", async () => {
    const root = await mountEditor({ ...BASE_SETTINGS, notes_paper_rules: false });
    expect(root.classList.contains("notes-editor--ruled")).toBe(false);
  });

  it("defaults to ruled (true) when settings are not yet loaded", async () => {
    const root = await mountEditor(null);
    expect(root.classList.contains("notes-editor--ruled")).toBe(true);
  });
});
