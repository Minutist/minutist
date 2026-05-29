/**
 * Unit tests for MeetingControls button-state logic.
 *
 * Each test mounts the component with the Zustand store pre-populated to one
 * of the four RecordingState variants and verifies that the four buttons have
 * the correct enabled/disabled state.
 *
 * Phase 2: Start also requires `isAsrModelReady = true`; tests set this
 * explicitly via the models store.
 *
 * The Tauri runtime is not available in jsdom; Tauri API modules are mocked
 * below to prevent import-time failures.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { act } from "react";
import { MeetingControls } from "../shell/MeetingControls";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import type { RecordingState } from "../ipc/bindings";

// ---------------------------------------------------------------------------
// Tauri API mocks — must be declared before any module that imports bindings.ts
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setRecordingState(state: RecordingState) {
  act(() => {
    useRecordingStore.setState({ state });
  });
}

function getButton(name: string): HTMLButtonElement {
  return screen.getByRole("button", { name }) as HTMLButtonElement;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MeetingControls", () => {
  beforeEach(() => {
    // Reset stores to a known baseline before each test.
    act(() => {
      useRecordingStore.setState({
        state: { kind: "idle" },
        devices: [],
        selectedDeviceId: null,
        meter: { peak: 0, rms: 0 },
        lastError: null,
        transcript: [],
      });
      // Phase 2: ASR model is ready by default so Start is enabled in idle.
      useModelsStore.setState({ isAsrModelReady: true });
    });
  });

  it("Idle: Start enabled; Pause, Resume, Stop disabled", () => {
    setRecordingState({ kind: "idle" });
    render(<MeetingControls />);

    expect(getButton("Start")).not.toBeDisabled();
    expect(getButton("Pause")).toBeDisabled();
    expect(getButton("Resume")).toBeDisabled();
    expect(getButton("Stop")).toBeDisabled();
  });

  it("Idle + ASR model not ready: Start disabled", () => {
    act(() => {
      useModelsStore.setState({ isAsrModelReady: false });
    });
    setRecordingState({ kind: "idle" });
    render(<MeetingControls />);

    expect(getButton("Start")).toBeDisabled();
  });

  it("Recording: Pause and Stop enabled; Start and Resume disabled", () => {
    setRecordingState({
      kind: "recording",
      meeting_id: "test-uuid",
      started_at_ms: Date.now(),
    });
    render(<MeetingControls />);

    expect(getButton("Start")).toBeDisabled();
    expect(getButton("Pause")).not.toBeDisabled();
    expect(getButton("Resume")).toBeDisabled();
    expect(getButton("Stop")).not.toBeDisabled();
  });

  it("Paused: Resume and Stop enabled; Start and Pause disabled", () => {
    setRecordingState({
      kind: "paused",
      meeting_id: "test-uuid",
      paused_at_ms: Date.now(),
    });
    render(<MeetingControls />);

    expect(getButton("Start")).toBeDisabled();
    expect(getButton("Pause")).toBeDisabled();
    expect(getButton("Resume")).not.toBeDisabled();
    expect(getButton("Stop")).not.toBeDisabled();
  });

  it("Stopping: all buttons disabled", () => {
    setRecordingState({ kind: "stopping", meeting_id: "test-uuid" });
    render(<MeetingControls />);

    expect(getButton("Start")).toBeDisabled();
    expect(getButton("Pause")).toBeDisabled();
    expect(getButton("Resume")).toBeDisabled();
    expect(getButton("Stop")).toBeDisabled();
  });
});
