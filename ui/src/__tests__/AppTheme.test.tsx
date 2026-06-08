/**
 * App-level colour-theme application tests.
 *
 * `App.tsx` reflects the persisted `settings.theme` onto the document root so
 * `theme.css`'s `:root[data-theme="dark"]` overrides apply: "dark"/"light" are
 * explicit, "system" follows `prefers-color-scheme` and tracks live changes.
 * These assert the setting -> document-root wiring (the substantive half; the
 * SettingsDrawer test only covers the control rendering) at the seam, with a
 * controllable `matchMedia` stub. Default-suite test.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, act, waitFor, cleanup } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

vi.mock("../ipc/bindings", () => {
  const getSettings = vi.fn();
  return {
    commands: {
      updateSettings: vi.fn(),
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
    events: { appEventPayload: { listen: vi.fn().mockResolvedValue(() => {}) } },
  };
});
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
import { useRecordingStore } from "../state/recording";
import { commands } from "../ipc/bindings";
import type { Settings, Theme } from "../ipc/bindings";

const BASE: Settings = {
  input_device_id: null,
  theme: "system",
  data_directory: null,
  start_hidden: false,
  onboarding_completed: true,
};

/** A controllable `prefers-color-scheme: dark` MediaQueryList stub. */
function stubMatchMedia(matches: boolean) {
  const listeners = new Set<(e: { matches: boolean }) => void>();
  const mql = {
    matches,
    media: "(prefers-color-scheme: dark)",
    onchange: null,
    addEventListener: (_type: string, cb: (e: { matches: boolean }) => void) =>
      listeners.add(cb),
    removeEventListener: (_type: string, cb: (e: { matches: boolean }) => void) =>
      listeners.delete(cb),
    addListener: (cb: (e: { matches: boolean }) => void) => listeners.add(cb),
    removeListener: (cb: (e: { matches: boolean }) => void) => listeners.delete(cb),
    dispatchEvent: () => true,
    listenerCount: () => listeners.size,
    emit: (m: boolean) => {
      mql.matches = m;
      listeners.forEach((cb) => cb({ matches: m }));
    },
  };
  // @ts-expect-error — assigning a stub onto window for the test
  window.matchMedia = vi.fn(() => mql);
  return mql;
}

async function renderAppWithTheme(theme: Theme) {
  vi.mocked(commands.getSettings).mockResolvedValue({
    status: "ok",
    data: { ...BASE, theme },
  });
  act(() => {
    useRecordingStore.setState({ state: { kind: "idle" }, settings: null });
  });
  let result!: ReturnType<typeof render>;
  await act(async () => {
    result = render(<App />);
    await Promise.resolve();
  });
  return result;
}

const themeAttr = () => document.documentElement.getAttribute("data-theme");

describe("App colour-theme application", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    document.documentElement.removeAttribute("data-theme");
  });
  afterEach(() => {
    cleanup();
    document.documentElement.removeAttribute("data-theme");
  });

  it("theme 'dark' sets data-theme=dark on the document root", async () => {
    stubMatchMedia(false);
    await renderAppWithTheme("dark");
    await waitFor(() => expect(themeAttr()).toBe("dark"));
  });

  it("theme 'light' clears data-theme (light-first :root defaults)", async () => {
    stubMatchMedia(true); // OS prefers dark, but explicit light must win.
    await renderAppWithTheme("light");
    await waitFor(() => expect(themeAttr()).toBeNull());
  });

  it("theme 'system' follows prefers-color-scheme and tracks live changes", async () => {
    const mql = stubMatchMedia(true); // OS currently prefers dark.
    await renderAppWithTheme("system");
    await waitFor(() => expect(themeAttr()).toBe("dark"));

    // The OS flips to light → the attribute clears without a re-render.
    act(() => mql.emit(false));
    expect(themeAttr()).toBeNull();

    // And a listener was registered (and is the only one) for live tracking.
    expect(mql.listenerCount()).toBe(1);
  });
});
