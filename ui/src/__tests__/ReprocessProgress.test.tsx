/**
 * Tests for the reprocess UX: the meetings store sets an immediate in-flight
 * flag (so the button responds instantly, before any backend progress event)
 * and guards against a double-press, and the transcript toolbar renders a
 * progress bar with a phase label / percent while reprocessing.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { act } from "react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn().mockResolvedValue({}),
  renameMeeting: vi.fn().mockResolvedValue(undefined),
  deleteMeeting: vi.fn().mockResolvedValue(undefined),
  setSpeakerName: vi.fn().mockResolvedValue({}),
  reprocess: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../ipc/collections", () => ({
  listCollections: vi.fn().mockResolvedValue([]),
  createCollection: vi.fn(),
  renameCollection: vi.fn().mockResolvedValue(undefined),
  deleteCollection: vi.fn().mockResolvedValue(undefined),
  setMeetingCollection: vi.fn().mockResolvedValue(undefined),
}));

import { TranscriptPane } from "../transcript/TranscriptPane";
import { useMeetingsStore } from "../state/meetings";
import { useRecordingStore } from "../state/recording";
import { useOperationProgressStore } from "../state/operation-progress";
import * as meetingsIpc from "../ipc/meetings";

beforeEach(() => {
  vi.clearAllMocks();
  act(() => {
    useMeetingsStore.setState({
      meetings: [],
      openMeetingId: null,
      openMeetingState: null,
      reprocessingId: null,
      reprocessStartedMs: null,
      lastError: null,
    });
    useOperationProgressStore.setState({ operations: {} });
    useRecordingStore.setState({ state: { kind: "idle" } });
  });
});
afterEach(() => cleanup());

describe("meetings store reprocess in-flight flag", () => {
  it("sets reprocessingId synchronously and ignores a double-press", async () => {
    let resolveReprocess!: () => void;
    vi.mocked(meetingsIpc.reprocess).mockReturnValueOnce(
      new Promise<void>((r) => {
        resolveReprocess = r;
      }),
    );

    // First press: the flag is set BEFORE the IPC promise resolves.
    const pending = useMeetingsStore.getState().reprocess("m1");
    expect(useMeetingsStore.getState().reprocessingId).toBe("m1");
    expect(useMeetingsStore.getState().reprocessStartedMs).not.toBeNull();

    // Second press while in flight is ignored (guard) — IPC called once.
    await useMeetingsStore.getState().reprocess("m1");
    expect(meetingsIpc.reprocess).toHaveBeenCalledTimes(1);

    // Resolving the pass clears the flag.
    await act(async () => {
      resolveReprocess();
      await pending;
    });
    expect(useMeetingsStore.getState().reprocessingId).toBeNull();
    expect(useMeetingsStore.getState().reprocessStartedMs).toBeNull();
  });
});

describe("transcript toolbar reprocess progress", () => {
  it("shows 'Reprocessing…' + a progress bar immediately, before any backend event", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "m1",
        reprocessingId: "m1",
        reprocessStartedMs: Date.now(),
      });
    });
    render(<TranscriptPane />);
    expect(
      screen.getByRole("button", { name: "Reprocessing…" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    // Before the first backend event the readout is the indeterminate "Starting…".
    expect(screen.getByText(/Starting…/)).toBeInTheDocument();
  });

  it("renders determinate progress (percent + phase) from a progress event", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "m1",
        reprocessingId: "m1",
        reprocessStartedMs: Date.now() - 10_000,
      });
      useOperationProgressStore.setState({
        operations: {
          m1: { op: "re_transcribe", fraction: 0.5, label: "Re-transcribing…" },
        },
      });
    });
    render(<TranscriptPane />);
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "50",
    );
    expect(screen.getByText(/Re-transcribing…/)).toBeInTheDocument();
    expect(screen.getByText(/50%/)).toBeInTheDocument();
  });
});
