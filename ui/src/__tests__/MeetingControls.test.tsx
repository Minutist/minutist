/**
 * Unit tests for MeetingControls button-state logic (#66 — two context-aware
 * toggle buttons replacing the former four always-on Start/Stop/Pause/Resume).
 *
 * Each test mounts the component with the Zustand store pre-populated to one of
 * the RecordingState variants and verifies the RECORD + PAUSE toggle labels and
 * their enabled/disabled state, plus that pressing each invokes the mapped
 * recording-store action.
 *
 * Phase 2: Start also requires `isAsrModelReady = true`; tests set this
 * explicitly via the models store.
 *
 * The Tauri runtime is not available in jsdom; Tauri API modules are mocked
 * below to prevent import-time failures.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { act } from "react";
import {
  MeetingControls,
  deriveButtonStates,
} from "../shell/MeetingControls";
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
        preparing: false,
      });
      // Phase 2: ASR model is ready by default so Start is enabled in idle.
      useModelsStore.setState({ isAsrModelReady: true });
    });
  });

  it("renders exactly two toggle buttons", () => {
    setRecordingState({ kind: "idle" });
    render(<MeetingControls />);
    expect(screen.getAllByRole("button")).toHaveLength(2);
  });

  it("Idle: RECORD shows Start (enabled); PAUSE shows Pause (disabled)", () => {
    setRecordingState({ kind: "idle" });
    render(<MeetingControls />);

    expect(getButton("Start")).not.toBeDisabled();
    expect(getButton("Pause")).toBeDisabled();
    // No standalone Stop / Resume buttons exist anymore.
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Resume" })).toBeNull();
  });

  it("Idle + ASR model not ready: Start disabled", () => {
    act(() => {
      useModelsStore.setState({ isAsrModelReady: false });
    });
    setRecordingState({ kind: "idle" });
    render(<MeetingControls />);

    expect(getButton("Start")).toBeDisabled();
  });

  it("Idle + preparing: RECORD shows 'Preparing…' and is disabled", () => {
    act(() => {
      useRecordingStore.setState({ preparing: true });
    });
    setRecordingState({ kind: "idle" });
    render(<MeetingControls />);

    expect(getButton("Preparing…")).toBeDisabled();
  });

  it("Recording: RECORD shows Stop (enabled); PAUSE shows Pause (enabled)", () => {
    setRecordingState({
      kind: "recording",
      meeting_id: "test-uuid",
      started_at_ms: Date.now(),
    });
    render(<MeetingControls />);

    expect(getButton("Stop")).not.toBeDisabled();
    expect(getButton("Pause")).not.toBeDisabled();
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Resume" })).toBeNull();
  });

  it("Paused: RECORD shows Stop (enabled); PAUSE shows Resume (enabled)", () => {
    setRecordingState({
      kind: "paused",
      meeting_id: "test-uuid",
      paused_at_ms: Date.now(),
    });
    render(<MeetingControls />);

    expect(getButton("Stop")).not.toBeDisabled();
    expect(getButton("Resume")).not.toBeDisabled();
    expect(screen.queryByRole("button", { name: "Pause" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();
  });

  it("Stopping: RECORD shows Stop (disabled); PAUSE shows Pause (disabled)", () => {
    setRecordingState({ kind: "stopping", meeting_id: "test-uuid" });
    render(<MeetingControls />);

    expect(getButton("Stop")).toBeDisabled();
    expect(getButton("Pause")).toBeDisabled();
  });

  // -------------------------------------------------------------------------
  // Action wiring: each toggle invokes the mapped recording-store action.
  // -------------------------------------------------------------------------

  it("RECORD calls start when idle and stop when recording", () => {
    const start = vi.fn();
    const stop = vi.fn();
    act(() => {
      useRecordingStore.setState({ start, stop });
    });

    setRecordingState({ kind: "idle" });
    const { unmount } = render(<MeetingControls />);
    fireEvent.click(getButton("Start"));
    expect(start).toHaveBeenCalledTimes(1);
    unmount();

    setRecordingState({
      kind: "recording",
      meeting_id: "test-uuid",
      started_at_ms: Date.now(),
    });
    render(<MeetingControls />);
    fireEvent.click(getButton("Stop"));
    expect(stop).toHaveBeenCalledTimes(1);
  });

  it("PAUSE calls pause when recording and resume when paused", () => {
    const pause = vi.fn();
    const resume = vi.fn();
    act(() => {
      useRecordingStore.setState({ pause, resume });
    });

    setRecordingState({
      kind: "recording",
      meeting_id: "test-uuid",
      started_at_ms: Date.now(),
    });
    const { unmount } = render(<MeetingControls />);
    fireEvent.click(getButton("Pause"));
    expect(pause).toHaveBeenCalledTimes(1);
    unmount();

    setRecordingState({
      kind: "paused",
      meeting_id: "test-uuid",
      paused_at_ms: Date.now(),
    });
    render(<MeetingControls />);
    fireEvent.click(getButton("Resume"));
    expect(resume).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Pure state-mapping unit tests (no store / DOM).
// ---------------------------------------------------------------------------

describe("deriveButtonStates", () => {
  it("idle (model ready, not preparing): Start enabled; Pause disabled", () => {
    expect(deriveButtonStates({ kind: "idle" }, true, false)).toEqual({
      recordAction: "start",
      recordEnabled: true,
      pauseAction: "pause",
      pauseEnabled: false,
    });
  });

  it("idle (model not ready): Start disabled", () => {
    expect(deriveButtonStates({ kind: "idle" }, false, false).recordEnabled).toBe(
      false,
    );
  });

  it("idle (preparing): Start disabled", () => {
    expect(deriveButtonStates({ kind: "idle" }, true, true).recordEnabled).toBe(
      false,
    );
  });

  it("recording: Stop + Pause both enabled", () => {
    expect(
      deriveButtonStates(
        { kind: "recording", meeting_id: "x", started_at_ms: 0 },
        true,
        false,
      ),
    ).toEqual({
      recordAction: "stop",
      recordEnabled: true,
      pauseAction: "pause",
      pauseEnabled: true,
    });
  });

  it("paused: Stop enabled; Resume enabled", () => {
    expect(
      deriveButtonStates(
        { kind: "paused", meeting_id: "x", paused_at_ms: 0 },
        true,
        false,
      ),
    ).toEqual({
      recordAction: "stop",
      recordEnabled: true,
      pauseAction: "resume",
      pauseEnabled: true,
    });
  });

  it("stopping: Stop disabled; Pause disabled", () => {
    const s = deriveButtonStates(
      { kind: "stopping", meeting_id: "x" },
      true,
      false,
    );
    expect(s.recordAction).toBe("stop");
    expect(s.recordEnabled).toBe(false);
    expect(s.pauseEnabled).toBe(false);
  });
});
