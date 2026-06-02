/**
 * Phase 7 first-run onboarding tests.
 *
 * Covers the onboarding gate + flow, with the IPC layer mocked at the seam (per
 * `architecture/cross-cutting.md` — Automated-testing policy; the generated
 * bindings file is mocked, never the higher-level client). Asserted behaviours:
 *
 *   1. Gate — `onboarding_completed=false` shows onboarding, hides the main app;
 *      `true` shows the main app, hides onboarding.
 *   2. Step progression — welcome → model → settings → finish.
 *   3. Finish calls the settings update seam with `onboarding_completed: true`.
 *   4. A not-yet-loaded (settings pending) gate renders without crashing.
 *
 * This is a default-suite test: no model, GPU, or microphone — the mocked seams
 * and synthetic settings snapshots are the fixtures.
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

// The generated bindings: the onboarding flow rounds completion + theme through
// `commands.updateSettings`, so mock it so the call is observable (mirrors
// Diarization.test / DevicePersistence). Default getSettings resolves to a
// completed snapshot; individual tests override per case.
vi.mock("../ipc/bindings", () => {
  const updateSettings = vi.fn();
  const getSettings = vi.fn();
  return {
    commands: {
      updateSettings,
      getSettings,
      listDevices: vi.fn().mockResolvedValue({ status: "ok", data: [] }),
      startRecording: vi.fn(),
      pauseRecording: vi.fn(),
      resumeRecording: vi.fn(),
      stopRecording: vi.fn(),
      getRecordingState: vi.fn(),
      listModels: vi.fn().mockResolvedValue({ status: "ok", data: [] }),
      ensureModel: vi.fn(),
    },
    // The App root mounts `useAppEventBridge`, which subscribes via
    // `events.appEventPayload.listen`; provide a no-op so the bridge attaches
    // cleanly (returning an unlisten fn) rather than logging a subscribe error.
    events: {
      appEventPayload: { listen: vi.fn().mockResolvedValue(() => {}) },
    },
  };
});

// The meetings store loads these from the `../ipc/meetings` seam at module
// load; the mock must expose every named export the store references (the App
// root mounts MainWindow, which mounts the meetings store).
vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn().mockResolvedValue(undefined),
  deleteMeeting: vi.fn().mockResolvedValue(undefined),
  reTranscribe: vi.fn().mockResolvedValue(undefined),
  rediarize: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../ipc/notes", () => ({
  saveNotes: vi.fn().mockResolvedValue(undefined),
  loadNotes: vi.fn().mockResolvedValue(null),
}));

import { App } from "../App";
import { Onboarding } from "../shell/Onboarding";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useMeetingsStore } from "../state/meetings";
import { useOnboardingStore } from "../state/onboarding";
import { commands } from "../ipc/bindings";
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

function settingsWith(onboarding_completed: boolean): Settings {
  return { ...BASE_SETTINGS, onboarding_completed };
}

function resetStores() {
  act(() => {
    useRecordingStore.setState({
      state: { kind: "idle" },
      transcript: [],
      settings: null,
      lastError: null,
    });
    useModelsStore.setState({
      models: [],
      isAsrModelReady: false,
      downloadInProgress: {},
    });
    useMeetingsStore.setState({
      meetings: [],
      loading: false,
      openMeetingId: null,
      openMeetingState: null,
      lastError: null,
    });
    useOnboardingStore.setState({ step: "welcome" });
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(commands.listDevices).mockResolvedValue({ status: "ok", data: [] });
  vi.mocked(commands.listModels).mockResolvedValue({ status: "ok", data: [] });
  resetStores();
});

// ---------------------------------------------------------------------------
// 1. Gate: completed flag decides which surface renders
// ---------------------------------------------------------------------------

describe("onboarding gate (Phase 7)", () => {
  it("completed=false → onboarding renders, main app hidden", async () => {
    vi.mocked(commands.getSettings).mockReturnValue(
      okSettings(settingsWith(false)),
    );

    render(<App />);

    // The onboarding welcome surface appears.
    await waitFor(() =>
      expect(screen.getByText("Welcome to meeting-app")).toBeInTheDocument(),
    );
    // The main app chrome (wordmark) is NOT present.
    expect(screen.queryByText("meeting-app")).not.toBeInTheDocument();
  });

  it("completed=true → main app renders, onboarding hidden", async () => {
    vi.mocked(commands.getSettings).mockReturnValue(
      okSettings(settingsWith(true)),
    );

    render(<App />);

    // The main-window wordmark appears.
    await waitFor(() =>
      expect(screen.getByText("meeting-app")).toBeInTheDocument(),
    );
    // The onboarding dialog is NOT present.
    expect(
      screen.queryByText("Welcome to meeting-app"),
    ).not.toBeInTheDocument();
  });

  it("settings pending (not yet loaded) renders without crashing", () => {
    // getSettings never resolves within the synchronous render; settings stays
    // null. The gate must hold neutral (render nothing app-specific), not throw
    // or flash either surface.
    vi.mocked(commands.getSettings).mockReturnValue(new Promise(() => {}));

    const { container } = render(<App />);

    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText("Welcome to meeting-app")).not.toBeInTheDocument();
    expect(screen.queryByText("meeting-app")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 2. Step progression welcome → model → settings → finish
// ---------------------------------------------------------------------------

describe("onboarding step progression (Phase 7)", () => {
  beforeEach(() => {
    act(() => {
      useRecordingStore.setState({ settings: settingsWith(false) });
    });
  });

  it("advances welcome → model → settings via Continue", () => {
    render(<Onboarding />);

    // Step 1: welcome.
    expect(screen.getByText("Welcome to meeting-app")).toBeInTheDocument();
    expect(useOnboardingStore.getState().step).toBe("welcome");

    // → model.
    act(() => fireEvent.click(screen.getByRole("button", { name: "Continue" })));
    expect(screen.getByText("Speech model")).toBeInTheDocument();
    expect(useOnboardingStore.getState().step).toBe("model");

    // → settings (final step: primary becomes Finish).
    act(() => fireEvent.click(screen.getByRole("button", { name: "Continue" })));
    expect(screen.getByText("A couple of preferences")).toBeInTheDocument();
    expect(useOnboardingStore.getState().step).toBe("settings");
    expect(screen.getByRole("button", { name: "Finish" })).toBeInTheDocument();
  });

  it("Back returns to the previous step", () => {
    render(<Onboarding />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Continue" })));
    expect(useOnboardingStore.getState().step).toBe("model");

    act(() => fireEvent.click(screen.getByRole("button", { name: "Back" })));
    expect(useOnboardingStore.getState().step).toBe("welcome");
    // Back is not offered on the first step.
    expect(screen.queryByRole("button", { name: "Back" })).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 3. Finish persists onboarding_completed=true through the settings seam
// ---------------------------------------------------------------------------

describe("onboarding finish (Phase 7)", () => {
  it("Finish calls the settings update seam with onboarding_completed: true", async () => {
    act(() => {
      useRecordingStore.setState({ settings: settingsWith(false) });
      useOnboardingStore.setState({ step: "settings" });
    });
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    render(<Onboarding />);

    act(() => fireEvent.click(screen.getByRole("button", { name: "Finish" })));

    await waitFor(() => {
      expect(commands.updateSettings).toHaveBeenCalledWith({
        ...settingsWith(false),
        onboarding_completed: true,
      });
    });
    // Other fields are preserved by the round-trip.
    const persisted = vi.mocked(commands.updateSettings).mock.calls[0][0];
    expect(persisted.theme).toBe("system");
  });

  it("Finish reveals the main app once completion is persisted (gate flips)", async () => {
    vi.mocked(commands.getSettings).mockReturnValue(
      okSettings(settingsWith(false)),
    );
    vi.mocked(commands.updateSettings).mockReturnValueOnce(okVoid);

    render(<App />);

    await waitFor(() =>
      expect(screen.getByText("Welcome to meeting-app")).toBeInTheDocument(),
    );

    // Walk to the final step and finish.
    act(() => fireEvent.click(screen.getByRole("button", { name: "Continue" })));
    act(() => fireEvent.click(screen.getByRole("button", { name: "Continue" })));
    act(() => fireEvent.click(screen.getByRole("button", { name: "Finish" })));

    // The store snapshot now reads completed → the gate flips to the main app.
    await waitFor(() =>
      expect(screen.getByText("meeting-app")).toBeInTheDocument(),
    );
    expect(
      screen.queryByText("Welcome to meeting-app"),
    ).not.toBeInTheDocument();
  });

  it("Finish is disabled while settings are still pending (no clobber)", () => {
    act(() => {
      useRecordingStore.setState({ settings: null });
      useOnboardingStore.setState({ step: "settings" });
    });

    render(<Onboarding />);

    const finish = screen.getByRole("button", { name: "Finish" });
    expect(finish).toBeDisabled();
    act(() => fireEvent.click(finish));
    expect(commands.updateSettings).not.toHaveBeenCalled();
  });
});
