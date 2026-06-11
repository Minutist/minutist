/**
 * Phase 6 diarization webview tests.
 *
 * Covers the four Phase-6 acceptance behaviours, all with the IPC layer mocked
 * at the seam (per `architecture/cross-cutting.md` — Automated-testing policy;
 * the generated bindings file is never faked):
 *
 *   1. TranscriptPane renders a "Speaker {id}" chip when `speaker_id` is set and
 *      hides it when `speaker_id` is null.
 *   2. A `diarization_complete` event triggers a SCOPED re-read of THAT
 *      meeting's transcript via `open_meeting` (not the live store).
 *   3. The `diarization_enabled` settings toggle round-trips through
 *      `commands.updateSettings`.
 *   4. The Re-diarize action invokes the `rediarize` seam.
 *
 * This is a default-suite test: it needs no model, GPU, or microphone — the
 * synthetic speaker-tagged segments and the mocked seams are the fixtures.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";

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

// The meetings store imports these from the `../ipc/meetings` seam at module
// load; the mock must expose every named export the store references.
vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn().mockResolvedValue(undefined),
  setSpeakerName: vi.fn().mockResolvedValue({}),
  deleteMeeting: vi.fn().mockResolvedValue(undefined),
  reTranscribe: vi.fn().mockResolvedValue(undefined),
  rediarize: vi.fn().mockResolvedValue(undefined),
}));

// The recording store rounds the toggle through `commands.updateSettings`; mock
// the generated bindings so the call is observable (mirrors DevicePersistence).
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

import { TranscriptPane } from "../transcript/TranscriptPane";
import { useRecordingStore } from "../state/recording";
import { useMeetingsStore } from "../state/meetings";
import { useCrossRefStore } from "../state/cross-ref";
import * as meetingsIpc from "../ipc/meetings";
import { commands } from "../ipc/bindings";
import {
  readDiarizationEnabled,
  withDiarizationEnabled,
} from "../state/diarization-settings";
import {
  speakerColorIndex,
  SPEAKER_PALETTE_SIZE,
} from "../transcript/speaker-color";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Segment, Settings } from "../ipc/bindings";
import type { MeetingState } from "../state/meetings";
import type { AppEvent } from "../ipc/app-event";

function makeSegment(
  start_ms: number,
  text: string,
  speaker_id?: string | null,
): Segment {
  return { start_ms, end_ms: start_ms + 1000, text, speaker_id, words: [] };
}

const okVoid = Promise.resolve({ status: "ok" as const, data: null });
const okSettings = (s: Settings) =>
  Promise.resolve({ status: "ok" as const, data: s });

const BASE_SETTINGS: Settings = {
  input_device_id: null,
  theme: "system",
  data_directory: null,
  start_hidden: false,
};

function resetStores() {
  act(() => {
    useRecordingStore.setState({
      state: { kind: "idle" },
      transcript: [],
      settings: null,
      lastError: null,
    });
    useMeetingsStore.setState({
      meetings: [],
      loading: false,
      openMeetingId: null,
      openMeetingState: null,
      lastError: null,
    });
    useCrossRefStore.setState({ highlightedRange: null, scrollRequest: null });
  });
}

// ---------------------------------------------------------------------------
// 0. speakerColorIndex pure mapping (Phase C)
// ---------------------------------------------------------------------------

describe("speakerColorIndex (Phase C)", () => {
  it("maps single-letter labels by alpha ordinal: A->1, B->2, H->8", () => {
    expect(speakerColorIndex("A")).toBe(1);
    expect(speakerColorIndex("B")).toBe(2);
    expect(speakerColorIndex("H")).toBe(8);
  });

  it("cycles past the palette size: I->1, J->2, and I matches A", () => {
    expect(speakerColorIndex("I")).toBe(1); // 9th label wraps
    expect(speakerColorIndex("J")).toBe(2);
    expect(speakerColorIndex("I")).toBe(speakerColorIndex("A"));
  });

  it("is stable across repeated calls (documents the pure-fn contract)", () => {
    expect(speakerColorIndex("C")).toBe(speakerColorIndex("C"));
  });

  it("falls back to a deterministic, in-range slot for non-alpha ids", () => {
    expect(speakerColorIndex("spk_42")).toBe(speakerColorIndex("spk_42"));
    const slot = speakerColorIndex("spk_42");
    expect(slot).toBeGreaterThanOrEqual(1);
    expect(slot).toBeLessThanOrEqual(SPEAKER_PALETTE_SIZE);
  });

  it("keeps SPEAKER_PALETTE_SIZE in sync with the --speaker-N tokens in BOTH themes", () => {
    // vitest runs with cwd == the `ui` package root.
    const themePath = resolve(process.cwd(), "src/styles/theme.css");
    const css = readFileSync(themePath, "utf8");
    const darkAt = css.indexOf('[data-theme="dark"]');
    expect(darkAt).toBeGreaterThan(0); // both blocks must exist
    const count = (s: string) => (s.match(/--speaker-\d+\s*:/g) ?? []).length;
    // The mapper indexes --speaker-1..N; the light :root AND the dark override
    // must each cover every slot, else a speaker has no colour in one theme.
    expect(count(css.slice(0, darkAt))).toBe(SPEAKER_PALETTE_SIZE);
    expect(count(css.slice(darkAt))).toBe(SPEAKER_PALETTE_SIZE);
  });
});

// ---------------------------------------------------------------------------
// 1. Speaker chip rendering
// ---------------------------------------------------------------------------

describe("TranscriptPane speaker chip (Phase 6)", () => {
  beforeEach(resetStores);

  it("renders a 'Speaker {id}' chip when speaker_id is set", () => {
    act(() => {
      useRecordingStore.setState({
        transcript: [makeSegment(0, "hello there", "A")],
      });
    });
    render(<TranscriptPane />);
    expect(screen.getByText("Speaker A")).toBeInTheDocument();
  });

  it("hides the speaker chip when speaker_id is null/absent", () => {
    act(() => {
      useRecordingStore.setState({
        transcript: [
          makeSegment(0, "no speaker yet", null),
          makeSegment(5_000, "also none"),
        ],
      });
    });
    render(<TranscriptPane />);
    expect(screen.queryByText(/^Speaker /)).not.toBeInTheDocument();
  });

  it("shows chips only for the diarized rows in a mixed transcript", () => {
    act(() => {
      useRecordingStore.setState({
        transcript: [
          makeSegment(0, "tagged", "B"),
          makeSegment(5_000, "untagged", null),
        ],
      });
    });
    render(<TranscriptPane />);
    expect(screen.getByText("Speaker B")).toBeInTheDocument();
    expect(screen.queryByText("Speaker A")).not.toBeInTheDocument();
    // Exactly one chip across two rows.
    expect(screen.getAllByText(/^Speaker /)).toHaveLength(1);
  });

  it("groups consecutive same-speaker rows: the labelled chip shows once per run, continuation rows keep only the dot", () => {
    act(() => {
      useRecordingStore.setState({
        transcript: [
          makeSegment(0, "first", "A"),
          makeSegment(5_000, "second", "A"),
          makeSegment(10_000, "third", "B"),
        ],
      });
    });
    const { container } = render(<TranscriptPane />);
    // Two labelled chips total (A once at its run start, B once), not three.
    expect(screen.getAllByText(/^Speaker /)).toHaveLength(2);
    expect(screen.getByText("Speaker A")).toBeInTheDocument();
    expect(screen.getByText("Speaker B")).toBeInTheDocument();
    // A colour dot on every diarized row (2× A + 1× B = 3).
    expect(
      container.querySelectorAll(".transcript-pane__speaker-dot"),
    ).toHaveLength(3);
    // Exactly one continuation (dot-only) marker — the second "A" row.
    expect(
      container.querySelectorAll(".transcript-pane__speaker--cont"),
    ).toHaveLength(1);
  });

  // --- Phase C: per-speaker colour dot ---------------------------------

  it("renders a colour dot resolving to var(--speaker-1) for speaker A", () => {
    act(() => {
      useRecordingStore.setState({ transcript: [makeSegment(0, "hi", "A")] });
    });
    render(<TranscriptPane />);
    const chip = screen.getByRole("button", { name: "Speaker A" });
    const dot = chip.querySelector(".transcript-pane__speaker-dot");
    expect(dot).not.toBeNull();
    expect(chip.style.getPropertyValue("--dot-color")).toBe(
      "var(--speaker-1)",
    );
  });

  it("keeps the same dot colour for id 'B' across the live -> on-stop relabel", () => {
    // LIVE store path.
    act(() => {
      useRecordingStore.setState({ transcript: [makeSegment(0, "live", "B")] });
    });
    const live = render(<TranscriptPane />);
    expect(
      live.getByRole("button", { name: "Speaker B" }).style.getPropertyValue("--dot-color"),
    ).toBe("var(--speaker-2)");
    live.unmount();

    // On-stop / saved-meeting path: same id "B" routed through the meetings
    // store with a meeting open and recording idle (the saved-meeting branch).
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" }, transcript: [] });
      useMeetingsStore.setState({
        openMeetingId: "m-1",
        openMeetingState: {
          transcript: [makeSegment(0, "saved", "B")],
        } as unknown as MeetingState,
      });
    });
    const saved = render(<TranscriptPane />);
    expect(
      saved.getByRole("button", { name: "Speaker B" }).style.getPropertyValue("--dot-color"),
    ).toBe("var(--speaker-2)");
  });

  it("renders no dot (and no chip) when speaker_id is null/absent", () => {
    act(() => {
      useRecordingStore.setState({
        transcript: [makeSegment(0, "x", null), makeSegment(5_000, "y")],
      });
    });
    const { container } = render(<TranscriptPane />);
    expect(
      container.querySelectorAll(".transcript-pane__speaker-dot"),
    ).toHaveLength(0);
    expect(screen.queryByText(/^Speaker /)).not.toBeInTheDocument();
  });

  it("renders exactly one dot for a mixed transcript, coloured var(--speaker-1)", () => {
    act(() => {
      useRecordingStore.setState({
        transcript: [
          makeSegment(0, "tagged", "A"),
          makeSegment(5_000, "untagged", null),
        ],
      });
    });
    const { container } = render(<TranscriptPane />);
    expect(
      container.querySelectorAll(".transcript-pane__speaker-dot"),
    ).toHaveLength(1);
    expect(
      screen.getByRole("button", { name: "Speaker A" }).style.getPropertyValue("--dot-color"),
    ).toBe("var(--speaker-1)");
  });

  it("exposes the chip as a button named 'Speaker A' and marks the dot aria-hidden", () => {
    act(() => {
      useRecordingStore.setState({ transcript: [makeSegment(0, "hi", "A")] });
    });
    render(<TranscriptPane />);
    const chip = screen.getByRole("button", { name: "Speaker A" });
    const dot = chip.querySelector(".transcript-pane__speaker-dot");
    expect(dot?.getAttribute("aria-hidden")).toBe("true");
  });

  it("overlays the display name from speaker_names instead of the bare label", () => {
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" }, transcript: [] });
      useMeetingsStore.setState({
        openMeetingId: "m-1",
        openMeetingState: {
          meta: { speaker_names: { A: "Alice" } },
          transcript: [makeSegment(0, "hi", "A")],
        } as unknown as MeetingState,
      });
    });
    render(<TranscriptPane />);
    expect(screen.getByRole("button", { name: "Alice" })).toBeInTheDocument();
    expect(screen.queryByText("Speaker A")).not.toBeInTheDocument();
  });

  it("renames a speaker via the chip, writing through the meetings seam", async () => {
    vi.mocked(meetingsIpc.setSpeakerName).mockResolvedValue({ A: "Alice" });
    act(() => {
      useRecordingStore.setState({ state: { kind: "idle" }, transcript: [] });
      useMeetingsStore.setState({
        openMeetingId: "m-1",
        openMeetingState: {
          meta: { speaker_names: {} },
          transcript: [makeSegment(0, "hi", "A")],
        } as unknown as MeetingState,
      });
    });
    render(<TranscriptPane />);

    fireEvent.click(screen.getByRole("button", { name: "Speaker A" }));
    const input = screen.getByLabelText("Name for speaker A");
    fireEvent.change(input, { target: { value: "Alice" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() =>
      expect(meetingsIpc.setSpeakerName).toHaveBeenCalledWith("m-1", "A", "Alice"),
    );
    // The store folds the returned map into the open meeting, so the chip now
    // shows the name.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Alice" })).toBeInTheDocument(),
    );
  });
});

// ---------------------------------------------------------------------------
// 2. diarization_complete triggers a scoped re-read
// ---------------------------------------------------------------------------

describe("meetings store diarization_complete handling (Phase 6)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStores();
  });

  it("re-reads THE EVENT'S meeting via open_meeting when it is the open meeting", async () => {
    const restored: MeetingState = {
      transcript: [makeSegment(0, "now tagged", "A")],
    } as unknown as MeetingState;
    vi.mocked(meetingsIpc.openMeeting).mockResolvedValue(restored);

    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "m-1",
        openMeetingState: {
          transcript: [makeSegment(0, "now tagged", null)],
        } as unknown as MeetingState,
      });
    });

    const event: AppEvent = {
      kind: "diarization_complete",
      meeting_id: "m-1",
      speaker_count: 2,
    };
    act(() => {
      useMeetingsStore.getState().handleEvent(event);
    });

    await waitFor(() => {
      expect(meetingsIpc.openMeeting).toHaveBeenCalledWith("m-1");
      expect(
        useMeetingsStore.getState().openMeetingState?.transcript[0].speaker_id,
      ).toBe("A");
    });
  });

  it("does NOT re-read the open meeting when the event is for a different meeting", () => {
    act(() => {
      useMeetingsStore.setState({
        openMeetingId: "m-1",
        openMeetingState: null,
      });
    });

    act(() => {
      useMeetingsStore.getState().handleEvent({
        kind: "diarization_complete",
        meeting_id: "m-2",
        speaker_count: 3,
      });
    });

    // The open meeting (m-1) is not re-read; only a list refresh fires.
    expect(meetingsIpc.openMeeting).not.toHaveBeenCalled();
    expect(meetingsIpc.listMeetings).toHaveBeenCalled();
  });

  it("ignores non-diarization events", () => {
    act(() => {
      useMeetingsStore.getState().handleEvent({
        kind: "summary_ready",
        meeting_id: "m-1",
      });
    });
    expect(meetingsIpc.openMeeting).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// 3. diarization_enabled toggle round-trips through update_settings
// ---------------------------------------------------------------------------

describe("diarization_enabled toggle round-trip (Phase 6)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStores();
  });

  it("setDiarizationEnabled persists via update_settings, preserving other fields", async () => {
    act(() => {
      useRecordingStore.setState({ settings: BASE_SETTINGS });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setDiarizationEnabled(true);

    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      diarization_enabled: true,
    });
    // The store snapshot reflects the persisted value.
    expect(readDiarizationEnabled(useRecordingStore.getState().settings)).toBe(
      true,
    );
  });

  it("round-trips back to off, and a fresh getSettings reads it back", async () => {
    act(() => {
      useRecordingStore.setState({
        settings: withDiarizationEnabled(BASE_SETTINGS, true),
      });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    await useRecordingStore.getState().setDiarizationEnabled(false);
    expect(commands.updateSettings).toHaveBeenCalledWith({
      ...BASE_SETTINGS,
      diarization_enabled: false,
    });

    // Simulate a reload: getSettings returns the persisted object.
    vi.mocked(commands.getSettings).mockReturnValueOnce(
      okSettings(withDiarizationEnabled(BASE_SETTINGS, false) as Settings),
    );
    await useRecordingStore.getState().refreshSettings();
    expect(readDiarizationEnabled(useRecordingStore.getState().settings)).toBe(
      false,
    );
  });

  it("skips the IPC write when settings are not loaded yet", async () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
    });
    await useRecordingStore.getState().setDiarizationEnabled(true);
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });

  it("defaults to off (false) when the field is absent", () => {
    expect(readDiarizationEnabled(BASE_SETTINGS)).toBe(false);
    expect(readDiarizationEnabled(null)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 4. Re-diarize action invokes the seam (via the store)
// ---------------------------------------------------------------------------

describe("Re-diarize action (Phase 6)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStores();
  });

  it("the meetings-store rediarize action invokes the rediarize seam", async () => {
    await useMeetingsStore.getState().rediarize("m-7");
    expect(meetingsIpc.rediarize).toHaveBeenCalledWith("m-7");
    expect(useMeetingsStore.getState().lastError).toBeNull();
  });

  it("surfaces an error when the rediarize seam rejects", async () => {
    vi.mocked(meetingsIpc.rediarize).mockRejectedValueOnce(
      new Error("no diarizer model"),
    );
    await useMeetingsStore.getState().rediarize("m-7");
    expect(useMeetingsStore.getState().lastError).toBe("no diarizer model");
  });
});
