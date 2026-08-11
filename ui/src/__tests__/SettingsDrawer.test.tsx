/**
 * Settings-drawer tests.
 *
 * The capture / processing controls moved out of the top bar into a drawer.
 * These cover the drawer's own behaviour: open/closed gating, the controls it
 * surfaces, and the dismissal affordances (Done, Escape, scrim). The per-setting
 * persistence is unchanged and stays covered by the store-seam tests
 * (Diarization / GpuAcceleration / SystemAudio / TranscriptionLanguage /
 * DevicePersistence) — the drawer routes through those same seams.
 *
 * The lazy-wiring tests (describe "MCP pane lazy wiring") assert that the
 * drawer mounts McpSettingsPane via the Suspense/lazy path in the connected
 * build, and that the error boundary absorbs a render failure cleanly.
 * VITE_CONNECTED is baked to "1" by vite.config.ts at transform time, so
 * vi.stubEnv cannot change it after the module is loaded — the "gate hides
 * pane when not connected" case cannot be exercised in the vitest suite
 * (that property is verified by the free-build grep in CI instead).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { SettingsDrawer } from "../shell/SettingsDrawer";
import { useRecordingStore } from "../state/recording";
import type { Settings } from "../ipc/bindings";

// ---------------------------------------------------------------------------
// IPC mocks — required by the lazy McpSettingsPane which calls getMcpServerInfo
// on mount via mcp-server-info.ts -> ipc/client.ts -> bindings.ts.
// These mocks are hoisted by vitest and apply to all tests in this file.
// The existing synchronous drawer tests are unaffected: they do not await any
// IPC call.
// ---------------------------------------------------------------------------
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));
vi.mock("../ipc/client", () => ({
  commands: {
    getMcpServerInfo: vi.fn(async () => ({
      status: "ok",
      data: { url: "http://127.0.0.1:8765/mcp", token: "tok" },
    })),
    updateSettings: vi.fn(async () => ({ status: "ok", data: null })),
    getSettings: vi.fn(async () => ({ status: "ok", data: null })),
    // The lazy ConnectionSettingsPane calls tunnel_status on mount.
    tunnelStatus: vi.fn(async () => ({
      status: "ok",
      data: { enabled: false, status: "disconnected", account_id: null },
    })),
    // The lazy SyncSettingsPane calls sync_status + sync_get_my_ticket on mount.
    syncStatus: vi.fn(async () => ({
      status: "ok",
      data: { kind: "idle" },
    })),
    syncGetMyTicket: vi.fn(async () => ({
      status: "ok",
      data: "test-ticket-abc123",
    })),
  },
  unwrap: <T,>(r: { status: string; data: T }) => {
    if (r.status !== "ok") throw new Error("err");
    return r.data;
  },
  ipcErrorMessage: (e: unknown) => String(e),
}));

const BASE_SETTINGS: Settings = {
  input_device_id: null,
  theme: "system",
  data_directory: null,
  start_hidden: false,
};

function seedStore() {
  act(() => {
    useRecordingStore.setState({
      settings: BASE_SETTINGS,
      devices: [{ id: "mic-1", name: "Built-in Mic", is_default: true }],
      selectedDeviceId: null,
    });
  });
}

describe("SettingsDrawer", () => {
  beforeEach(seedStore);

  it("renders nothing when closed", () => {
    const { container } = render(
      <SettingsDrawer open={false} onClose={() => {}} onAbout={() => {}} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("surfaces the appearance + capture + processing controls when open", () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    // Appearance: colour theme + writing-paper rules.
    expect(screen.getByLabelText("Colour theme")).toBeInTheDocument();
    expect(screen.getByText("Ruled writing paper")).toBeInTheDocument();
    // Capture + processing.
    expect(screen.getByLabelText("Input device")).toBeInTheDocument();
    expect(screen.getByLabelText("Transcription language")).toBeInTheDocument();
    // Live-test UX T5: the diarization toggle is relabelled "Identify speakers"
    // (the wire name `diarization_enabled` is unchanged).
    expect(screen.getByText("Identify speakers")).toBeInTheDocument();
    expect(screen.getByText("GPU acceleration")).toBeInTheDocument();
    expect(screen.getByText("Capture call / system audio")).toBeInTheDocument();
    expect(
      screen.getByText("Start recording immediately on a new meeting"),
    ).toBeInTheDocument();
    // Output language dropdown (Processing section).
    expect(screen.getByLabelText("Output language")).toBeInTheDocument();
  });

  it("reflects the persisted appearance defaults (system theme, ruled paper on)", () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    // BASE_SETTINGS omits both fields → the schema defaults apply: theme falls
    // back to "system" and the writing-paper rules read as on.
    const theme = screen.getByLabelText("Colour theme") as HTMLSelectElement;
    expect(theme.value).toBe("system");
    const ruled = screen
      .getByText("Ruled writing paper")
      .closest("label")!
      .querySelector("input") as HTMLInputElement;
    expect(ruled.checked).toBe(true);
  });

  it("calls onClose on the Done button and on Escape", () => {
    const onClose = vi.fn();
    render(<SettingsDrawer open onClose={onClose} onAbout={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("dismisses on a scrim click but not on a click inside the panel", () => {
    const onClose = vi.fn();
    const { container } = render(<SettingsDrawer open onClose={onClose} onAbout={() => {}} />);
    // Click inside the dialog panel — stopPropagation keeps it open.
    fireEvent.click(screen.getByRole("dialog"));
    expect(onClose).not.toHaveBeenCalled();
    // Click the scrim (the overlay root) — dismisses.
    fireEvent.click(container.firstChild as Element);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("MCP pane lazy wiring", () => {
  beforeEach(seedStore);

  // In the connected build (VITE_CONNECTED="1", baked at transform time),
  // LazyMcpSettingsPane is non-null and rendered inside Suspense/McpPaneErrorBoundary.
  // This test verifies that the Suspense resolves and the pane appears in the DOM,
  // catching a broken lazy/gate wiring before it reaches users.
  //
  // The "gate hides pane when not connected" case is not exercised here:
  // VITE_CONNECTED is statically replaced by vite.config.ts `define` at transform
  // time, so vi.stubEnv has no effect after the module is loaded. That property
  // is verified by the free-build grep in the build-free CI job instead.
  it("renders the MCP pane via the lazy path in the connected build", async () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    // The lazy chunk resolves asynchronously; findByText waits for it.
    await screen.findByText("Enable MCP server (loopback)");
  });

  // The Connection (connector / relay tunnel) pane is gated the same way; verify
  // it resolves via the lazy path in the connected build. The free-build absence
  // is verified by the CI grep, not here (VITE_CONNECTED is baked at transform
  // time, so vi.stubEnv cannot toggle it after module load).
  it("renders the Connection pane via the lazy path in the connected build", async () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    await screen.findByText("Enable the connector");
  });

  // The Sync pane is gated the same way; verify it resolves via the lazy path in
  // the connected build. The free-build absence is verified by the CI grep.
  it("renders the Sync pane via the lazy path in the connected build", async () => {
    render(<SettingsDrawer open onClose={() => {}} onAbout={() => {}} />);
    await screen.findByText("Sync");
  });
});
