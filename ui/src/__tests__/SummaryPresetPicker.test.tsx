/**
 * Tests for the summary-prompt preset picker (Phase 9 — D4).
 *
 * The picker lives in the SummaryView: a preset dropdown bound to
 * `settings.summary_preset` + a custom-prompt textarea bound to
 * `settings.summary_system_prompt`. Asserts:
 *   - the preset dropdown is present and reflects the persisted preset,
 *   - changing the preset routes through `update_settings` (via the recording
 *     store's `setSummaryPreset` → `commands.updateSettings`),
 *   - editing the custom prompt routes through `update_settings`,
 *   - the override messaging is shown when a non-empty custom prompt is set
 *     (the UI makes clear the custom prompt overrides the preset).
 *
 * The summary IPC seam is mocked so the SummaryView mounts cleanly; the settings
 * round-trip is asserted through the real recording store against a mocked
 * `commands.updateSettings`.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

vi.mock("../ipc/summary", () => ({
  summariseMeeting: vi.fn().mockResolvedValue(undefined),
  getSummary: vi.fn().mockResolvedValue(null),
  saveSummary: vi.fn().mockResolvedValue(undefined),
}));

// The preset/custom-prompt persistence goes through the recording store, which
// calls `commands.updateSettings` from `../ipc/client`. Mock that surface.
vi.mock("../ipc/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ipc/client")>();
  return {
    ...actual,
    commands: {
      ...actual.commands,
      updateSettings: vi.fn().mockResolvedValue({ status: "ok", data: null }),
    },
  };
});

import { SummaryView } from "../shell/SummaryView";
import { useSummaryStore } from "../state/summary";
import { useModelsStore } from "../state/models";
import { useRecordingStore } from "../state/recording";
import { getSummary } from "../ipc/summary";
import { commands } from "../ipc/client";
import type { Settings } from "../ipc/bindings";

const MEETING = "meeting-0001";

const BASE_SETTINGS: Settings = {
  input_device_id: null,
  theme: "system",
  data_directory: null,
  start_hidden: false,
  summary_preset: "default",
  summary_system_prompt: "",
};

function seed(settings: Settings = BASE_SETTINGS) {
  act(() => {
    useSummaryStore.setState({
      summaryMarkdown: null,
      summarising: false,
      meetingId: null,
      lastError: null,
      editing: false,
      editDraft: "",
      editMeetingId: null,
    });
    useModelsStore.setState({ models: [] });
    useRecordingStore.setState({ settings, lastError: null });
  });
}

describe("Summary preset picker (D4)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    seed();
    vi.mocked(getSummary).mockResolvedValue(null);
  });

  it("renders the preset dropdown reflecting the persisted preset", async () => {
    seed({ ...BASE_SETTINGS, summary_preset: "action_items" });
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    const select = screen.getByLabelText("Preset") as HTMLSelectElement;
    expect(select.value).toBe("action_items");
  });

  it("changing the preset persists via update_settings", async () => {
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    act(() => {
      fireEvent.change(screen.getByLabelText("Preset"), {
        target: { value: "detailed" },
      });
    });

    await waitFor(() =>
      expect(commands.updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({ summary_preset: "detailed" }),
      ),
    );
  });

  it("editing the custom prompt persists via update_settings", async () => {
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    act(() => {
      fireEvent.change(screen.getByLabelText("Custom prompt"), {
        target: { value: "Summarise as bullet points only." },
      });
    });

    await waitFor(() =>
      expect(commands.updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          summary_system_prompt: "Summarise as bullet points only.",
        }),
      ),
    );
  });

  it("shows the override messaging when a non-empty custom prompt is set", async () => {
    seed({
      ...BASE_SETTINGS,
      summary_system_prompt: "Only list action items.",
    });
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    expect(
      screen.getByText(/it overrides the preset/i),
    ).toBeInTheDocument();
  });

  it("indicates the preset drives when no custom prompt is set", async () => {
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    expect(
      screen.getByText(/the selected preset is used/i),
    ).toBeInTheDocument();
  });
});
