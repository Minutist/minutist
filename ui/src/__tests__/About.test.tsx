/**
 * Phase 7 (S6) About-dialog tests.
 *
 * Covers the acceptance item "About dialog lists the bundled-model SPDX
 * licenses + NOTICE/attribution" plus the open/close affordance wired into the
 * main-window header. The IPC layer is mocked at the generated-bindings seam
 * (per `architecture/cross-cutting.md` — Automated-testing policy), mirroring
 * the onboarding tests.
 *
 * Bundled-model rows are derived from the models store (`useModelsStore`),
 * whose `ModelStatus` entries carry `display_name` + `license` straight from
 * `resources/models.json`. The store is seeded here with manifest-faithful
 * fixtures and the rendered list is asserted against them — guarding against
 * the drift that previously let a stale display name linger in a hand-kept
 * mirror.
 *
 * This is a default-suite test: no model, GPU, or microphone.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";

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

// The generated bindings: MainWindow mounts stores that read these at module
// load; provide a minimal commands surface (mirrors Onboarding.test).
vi.mock("../ipc/bindings", () => ({
  commands: {
    updateSettings: vi.fn(),
    getSettings: vi.fn(),
    listDevices: vi.fn().mockResolvedValue({ status: "ok", data: [] }),
    startRecording: vi.fn(),
    pauseRecording: vi.fn(),
    resumeRecording: vi.fn(),
    stopRecording: vi.fn(),
    getRecordingState: vi.fn(),
    listModels: vi.fn().mockResolvedValue({ status: "ok", data: [] }),
    ensureModel: vi.fn(),
  },
  events: {
    appEventPayload: { listen: vi.fn().mockResolvedValue(() => {}) },
  },
}));

vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn(),
  renameMeeting: vi.fn().mockResolvedValue(undefined),
  deleteMeeting: vi.fn().mockResolvedValue(undefined),
  reprocess: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../ipc/notes", () => ({
  saveNotes: vi.fn().mockResolvedValue(undefined),
  loadNotes: vi.fn().mockResolvedValue(null),
  applyNotesUpdate: vi.fn().mockResolvedValue(undefined),
  loadNotesYdoc: vi.fn().mockResolvedValue(null),
}));

import { About } from "../shell/About";
import { MainWindow } from "../shell/MainWindow";
import { APP_VERSION } from "../shell/about-content";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useMeetingsStore } from "../state/meetings";
import { commands } from "../ipc/bindings";
import type { ModelStatus } from "../ipc/bindings";

/**
 * Manifest-faithful fixtures, mirroring the four `resources/models.json`
 * entries (id, kind, display_name, license). The diarization model carries
 * its CURRENT display name + id — the assertions below use it to guard
 * against the stale-mirror drift this refactor removed.
 */
const MANIFEST_MODELS: ModelStatus[] = [
  {
    id: "qwen3-asr-0.6b-q8_0",
    kind: "asr",
    display_name: "Qwen3-ASR 0.6B (Q8_0)",
    status: { state: "available", local_dir: "/models/asr" },
    license: "apache-2.0",
  },
  {
    id: "gemma-4-e4b-it-q4_k_m",
    kind: "llm",
    display_name: "Gemma 4 E4B Instruct (Q4_K_M)",
    status: { state: "available", local_dir: "/models/llm" },
    license: "apache-2.0",
  },
  {
    id: "pyannote-segmentation-3-0",
    kind: "diarize",
    display_name: "pyannote segmentation 3.0",
    status: { state: "available", local_dir: "/models/pyannote" },
    license: "mit",
  },
  {
    id: "3dspeaker-campplus-zh-en-advanced",
    kind: "diarize",
    display_name: "3D-Speaker CAM++ (zh-en 16k common, advanced)",
    status: { state: "available", local_dir: "/models/3dspeaker" },
    license: "apache-2.0",
  },
];

function seedModels(models: ModelStatus[] = MANIFEST_MODELS) {
  act(() => {
    useModelsStore.setState({ models });
  });
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
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  resetStores();
});

// ---------------------------------------------------------------------------
// 1. Content: all four bundled models with their correct SPDX licenses.
// ---------------------------------------------------------------------------

describe("About dialog content (Phase 7 S6)", () => {
  it("lists the bundled models from the store with their SPDX licenses", () => {
    seedModels();
    render(<About onClose={() => {}} />);

    const dialog = screen.getByRole("dialog");

    // Each model's display name (from the manifest-backed store) and its
    // normalised SPDX license is shown.
    const expected: { name: string; spdx: string }[] = [
      { name: "Qwen3-ASR 0.6B (Q8_0)", spdx: "Apache-2.0" },
      { name: "Gemma 4 E4B Instruct (Q4_K_M)", spdx: "Apache-2.0" },
      { name: "pyannote segmentation 3.0", spdx: "MIT" },
      { name: "3D-Speaker CAM++ (zh-en 16k common, advanced)", spdx: "Apache-2.0" },
    ];
    for (const { name } of expected) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
    // Apache-2.0 appears three times, MIT (model) once among the model rows.
    expect(within(dialog, "Apache-2.0").length).toBeGreaterThanOrEqual(3);
    expect(within(dialog, "MIT").length).toBeGreaterThanOrEqual(1);

    // The NOTICE / attribution line is present.
    expect(
      screen.getByText(
        /full MIT and Apache-2\.0 license texts and the accompanying NOTICE/i,
      ),
    ).toBeInTheDocument();
  });

  it("renders the diarization model's CURRENT manifest name (no stale drift)", () => {
    seedModels();
    render(<About onClose={() => {}} />);

    // The current manifest display name is rendered verbatim from the store...
    expect(
      screen.getByText("3D-Speaker CAM++ (zh-en 16k common, advanced)"),
    ).toBeInTheDocument();
    // ...and the previously-hardcoded stale name is gone for good.
    expect(
      screen.queryByText("3D-Speaker CAM++ (zh-cn 16k common)"),
    ).not.toBeInTheDocument();
  });

  it("shows the app name, version, and the major OSS attributions", () => {
    render(<About onClose={() => {}} />);

    expect(
      screen.getByRole("heading", { name: "Minutist" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(new RegExp(`Version ${APP_VERSION.replace(/\./g, "\\.")}`)),
    ).toBeInTheDocument();

    // Major OSS components.
    for (const name of ["Tauri", "llama.cpp", "sherpa-onnx", "Tiptap", "React"]) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
  });
});

/** Count elements with exact text inside a container. */
function within(container: HTMLElement, text: string): HTMLElement[] {
  return Array.from(container.querySelectorAll("*")).filter(
    (el) =>
      el.children.length === 0 && el.textContent?.trim() === text,
  ) as HTMLElement[];
}

// ---------------------------------------------------------------------------
// 2. Open/close affordance from the main window.
// ---------------------------------------------------------------------------

describe("About affordance (Phase 7 S6)", () => {
  it("opening the About control shows the dialog; closing hides it", () => {
    // MainWindow fires refreshModels() on mount; return the manifest models so
    // the async refresh repopulates the store rather than clobbering the seed.
    vi.mocked(commands.listModels).mockResolvedValue({
      status: "ok",
      data: MANIFEST_MODELS,
    });
    seedModels();
    render(<MainWindow />);

    // Closed by default.
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    // Open via the Settings drawer's About affordance (the header About button
    // moved into the drawer footer to keep the masthead a single row).
    act(() =>
      fireEvent.click(screen.getByRole("button", { name: "Settings" })),
    );
    act(() =>
      fireEvent.click(
        screen.getByRole("button", { name: "About Minutist" }),
      ),
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    // Scope to the dialog: MainWindow also renders the model name elsewhere
    // (model-status UI) now that the store is seeded.
    expect(within(dialog, "Qwen3-ASR 0.6B (Q8_0)").length).toBeGreaterThanOrEqual(1);

    // Close via the Close button.
    act(() =>
      fireEvent.click(screen.getByRole("button", { name: "Close" })),
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("clicking the dialog overlay (outside the sheet) closes it", () => {
    const onClose = vi.fn();
    const { container } = render(<About onClose={onClose} />);

    const overlay = container.querySelector(".about-overlay") as HTMLElement;
    act(() => fireEvent.click(overlay));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("clicking inside the sheet does not close the dialog", () => {
    const onClose = vi.fn();
    render(<About onClose={onClose} />);

    act(() => fireEvent.click(screen.getByRole("dialog")));
    expect(onClose).not.toHaveBeenCalled();
  });
});
