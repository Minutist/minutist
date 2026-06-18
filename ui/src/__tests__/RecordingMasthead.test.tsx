/**
 * Tests for naming a meeting during the live recording: the recording store's
 * `setTitle` echoes locally and pushes to the backend (`set_recording_title`)
 * keyed on the live meeting_id, and the RecordingMasthead renders an editable
 * "Name this meeting" field that drives it.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act, cleanup } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn().mockResolvedValue(undefined),
  Channel: vi.fn(),
}));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { RecordingMasthead } from "../shell/RecordingMasthead";
import { useRecordingStore } from "../state/recording";

beforeEach(() => {
  vi.clearAllMocks();
  act(() => {
    useRecordingStore.setState({
      state: { kind: "recording", meeting_id: "m1", started_at_ms: 0 },
      pendingTitle: "",
      lastError: null,
    });
  });
});
afterEach(() => cleanup());

describe("recording store setTitle", () => {
  it("echoes the title and pushes it to the backend for the live meeting", async () => {
    await act(async () => {
      await useRecordingStore.getState().setTitle("Sprint review");
    });
    expect(useRecordingStore.getState().pendingTitle).toBe("Sprint review");
    expect(invoke).toHaveBeenCalledWith("set_recording_title", {
      meetingId: "m1",
      title: "Sprint review",
    });
  });

  it("echoes locally but does not push to the backend when not recording", async () => {
    act(() => useRecordingStore.setState({ state: { kind: "idle" } }));
    await act(async () => {
      await useRecordingStore.getState().setTitle("X");
    });
    expect(useRecordingStore.getState().pendingTitle).toBe("X");
    expect(invoke).not.toHaveBeenCalledWith(
      "set_recording_title",
      expect.anything(),
    );
  });
});

describe("RecordingMasthead", () => {
  it("renders an editable 'Name this meeting' field and pushes typing", async () => {
    render(<RecordingMasthead />);
    const input = screen.getByPlaceholderText("Name this meeting");
    await act(async () => {
      fireEvent.change(input, { target: { value: "Quarterly planning" } });
    });
    expect(useRecordingStore.getState().pendingTitle).toBe("Quarterly planning");
    expect(invoke).toHaveBeenCalledWith("set_recording_title", {
      meetingId: "m1",
      title: "Quarterly planning",
    });
  });
});
