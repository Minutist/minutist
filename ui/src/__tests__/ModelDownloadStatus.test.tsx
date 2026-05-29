/**
 * Unit tests for ModelDownloadStatus.
 *
 * Verifies the Download Model button when the ASR model is missing, the
 * progress bar when a download is in progress, and the hidden state when
 * the model is already available.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { act } from "react";

// ---------------------------------------------------------------------------
// Tauri API mocks
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

import { ModelDownloadStatus } from "../shell/ModelDownloadStatus";
import { useModelsStore } from "../state/models";
import type { ModelStatus } from "../ipc/bindings";

const ASR_MODEL_ID = "qwen3-asr-0.6b-q8_0";

function makeMissingModel(): ModelStatus {
  return {
    id: ASR_MODEL_ID,
    kind: "asr",
    display_name: "Qwen3-ASR 0.6B",
    status: { state: "missing", bytes_present: 0, bytes_total: 1000000 },
  };
}

function makeDownloadingModel(bytes_done: number, bytes_total: number): ModelStatus {
  return {
    id: ASR_MODEL_ID,
    kind: "asr",
    display_name: "Qwen3-ASR 0.6B",
    status: { state: "downloading", bytes_done, bytes_total },
  };
}

function makeAvailableModel(): ModelStatus {
  return {
    id: ASR_MODEL_ID,
    kind: "asr",
    display_name: "Qwen3-ASR 0.6B",
    status: { state: "available", local_dir: "/path/to/model" },
  };
}

describe("ModelDownloadStatus", () => {
  beforeEach(() => {
    act(() => {
      useModelsStore.setState({
        models: [],
        isAsrModelReady: false,
        downloadInProgress: {},
      });
    });
  });

  it("shows the Download Model button when ASR model status is missing", () => {
    act(() => {
      useModelsStore.setState({
        models: [makeMissingModel()],
        isAsrModelReady: false,
        downloadInProgress: {},
      });
    });

    render(<ModelDownloadStatus />);

    expect(
      screen.getByRole("button", { name: "Download Model" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("shows a progress bar when a download is in progress (via downloadInProgress)", () => {
    act(() => {
      useModelsStore.setState({
        models: [makeDownloadingModel(250000, 1000000)],
        isAsrModelReady: false,
        downloadInProgress: {
          [ASR_MODEL_ID]: { bytes_done: 250000, bytes_total: 1000000 },
        },
      });
    });

    render(<ModelDownloadStatus />);

    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Download Model" }),
    ).not.toBeInTheDocument();
  });

  it("shows a progress bar when model status is downloading (no downloadInProgress entry)", () => {
    act(() => {
      useModelsStore.setState({
        models: [makeDownloadingModel(100000, 1000000)],
        isAsrModelReady: false,
        downloadInProgress: {},
      });
    });

    render(<ModelDownloadStatus />);

    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("renders nothing when the ASR model is available", () => {
    act(() => {
      useModelsStore.setState({
        models: [makeAvailableModel()],
        isAsrModelReady: true,
        downloadInProgress: {},
      });
    });

    const { container } = render(<ModelDownloadStatus />);
    expect(container).toBeEmptyDOMElement();
  });

  it("displays the model display_name", () => {
    act(() => {
      useModelsStore.setState({
        models: [makeMissingModel()],
        isAsrModelReady: false,
        downloadInProgress: {},
      });
    });

    render(<ModelDownloadStatus />);
    expect(screen.getByText("Qwen3-ASR 0.6B")).toBeInTheDocument();
  });
});
