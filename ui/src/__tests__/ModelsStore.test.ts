/**
 * Unit tests for the models Zustand store.
 *
 * Verifies that handleEvent updates downloadInProgress on
 * model_download_progress events, triggers a refresh on completion, and
 * that refreshModels correctly derives isAsrModelReady.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Tauri API mocks — declared before importing any module that pulls in
// bindings.ts
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

vi.mock("../ipc/bindings", () => {
  const listModels = vi.fn();
  const ensureModel = vi.fn();
  return {
    commands: {
      listModels,
      ensureModel,
      listDevices: vi.fn(),
      startRecording: vi.fn(),
      pauseRecording: vi.fn(),
      resumeRecording: vi.fn(),
      stopRecording: vi.fn(),
      getRecordingState: vi.fn(),
      getSettings: vi.fn(),
      updateSettings: vi.fn(),
    },
    events: {},
  };
});

import { commands } from "../ipc/bindings";
import { useModelsStore } from "../state/models";
import type { ModelStatus, AppEvent } from "../ipc/bindings";

const ASR_MODEL_ID = "qwen3-asr-0.6b-q8_0";

function makeAvailableModel(): ModelStatus {
  return {
    id: ASR_MODEL_ID,
    kind: "asr",
    display_name: "Qwen3-ASR 0.6B",
    status: { state: "available", local_dir: "/path/to/model" },
  };
}

function okModels(models: ModelStatus[]) {
  return Promise.resolve({ status: "ok" as const, data: models });
}

describe("ModelsStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useModelsStore.setState({
      models: [],
      isAsrModelReady: false,
      downloadInProgress: {},
    });
  });

  it("refreshModels populates models and derives isAsrModelReady = true when ASR is available", async () => {
    vi.mocked(commands.listModels).mockReturnValueOnce(
      okModels([makeAvailableModel()]),
    );

    await useModelsStore.getState().refreshModels();

    expect(useModelsStore.getState().isAsrModelReady).toBe(true);
    expect(useModelsStore.getState().models).toHaveLength(1);
  });

  it("refreshModels sets isAsrModelReady = false when ASR model is missing", async () => {
    vi.mocked(commands.listModels).mockReturnValueOnce(
      okModels([
        {
          id: ASR_MODEL_ID,
          kind: "asr",
          display_name: "Qwen3-ASR 0.6B",
          status: { state: "missing", bytes_present: 0, bytes_total: 1000000 },
        },
      ]),
    );

    await useModelsStore.getState().refreshModels();

    expect(useModelsStore.getState().isAsrModelReady).toBe(false);
  });

  it("handleEvent updates downloadInProgress on model_download_progress", () => {
    const event: AppEvent = {
      kind: "model_download_progress",
      model_id: ASR_MODEL_ID,
      bytes_done: 300000,
      bytes_total: 1000000,
    };

    useModelsStore.getState().handleEvent(event);

    expect(
      useModelsStore.getState().downloadInProgress[ASR_MODEL_ID],
    ).toEqual({ bytes_done: 300000, bytes_total: 1000000 });
  });

  it("handleEvent removes downloadInProgress entry when bytes_done >= bytes_total", async () => {
    // Pre-populate an in-progress entry.
    useModelsStore.setState({
      downloadInProgress: {
        [ASR_MODEL_ID]: { bytes_done: 500000, bytes_total: 1000000 },
      },
    });

    // Mock the refresh that fires when download completes.
    vi.mocked(commands.listModels).mockReturnValueOnce(
      okModels([makeAvailableModel()]),
    );

    const event: AppEvent = {
      kind: "model_download_progress",
      model_id: ASR_MODEL_ID,
      bytes_done: 1000000,
      bytes_total: 1000000,
    };

    useModelsStore.getState().handleEvent(event);

    // The entry is removed immediately on completion.
    expect(
      useModelsStore.getState().downloadInProgress[ASR_MODEL_ID],
    ).toBeUndefined();
  });

  it("handleEvent handles unknown total (bytes_total null) by keeping bytes_done:0 in progress", () => {
    const event: AppEvent = {
      kind: "model_download_progress",
      model_id: ASR_MODEL_ID,
      bytes_done: 50000,
      bytes_total: null,
    };

    useModelsStore.getState().handleEvent(event);

    expect(
      useModelsStore.getState().downloadInProgress[ASR_MODEL_ID],
    ).toEqual({ bytes_done: 50000, bytes_total: 0 });
  });
});
