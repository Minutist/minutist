/**
 * MCP settings pane tests (Phase 10).
 *
 * Covers the pane's own behaviour: the off-by-default reading, the enable +
 * port + write-tools controls routing through the store seams, and the live
 * endpoint reveal/copy. The `get_mcp_server_info` command is mocked so the
 * endpoint row renders without a backend.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import type { Settings } from "../ipc/bindings";

// Mock the IPC client so getMcpServerInfo returns a fixed endpoint and the
// store setters do not hit a real backend.
const updateSettings = vi.fn(
  async (_settings: Settings) => ({ status: "ok", data: null }) as const,
);
vi.mock("../ipc/client", () => ({
  commands: {
    getMcpServerInfo: vi.fn(async () => ({
      status: "ok",
      data: { url: "http://127.0.0.1:8765/mcp", token: "secrettoken123456" },
    })),
    updateSettings: (settings: Settings) => updateSettings(settings),
    getSettings: vi.fn(async () => ({ status: "ok", data: null })),
  },
  unwrap: <T,>(r: { status: string; data: T }) => {
    if (r.status !== "ok") throw new Error("err");
    return r.data;
  },
  ipcErrorMessage: (e: unknown) => String(e),
}));

import { McpSettingsPane } from "../shell/McpSettingsPane";
import { useRecordingStore } from "../state/recording";

const BASE_SETTINGS: Settings = {
  input_device_id: null,
  theme: "system",
  data_directory: null,
  start_hidden: false,
};

function seed(overrides: Partial<Settings> = {}) {
  act(() => {
    useRecordingStore.setState({
      settings: { ...BASE_SETTINGS, ...overrides },
    });
  });
}

describe("McpSettingsPane", () => {
  beforeEach(() => {
    updateSettings.mockClear();
  });

  it("reads off-by-default when the MCP fields are absent", async () => {
    seed();
    render(<McpSettingsPane />);
    const toggle = screen
      .getByText("Enable MCP server (loopback)")
      .closest("label")!
      .querySelector("input") as HTMLInputElement;
    expect(toggle.checked).toBe(false);
    // The endpoint row is hidden while disabled.
    expect(screen.queryByText("Endpoint")).not.toBeInTheDocument();
    // Let the on-mount getMcpServerInfo fetch settle (it sets state).
    await act(async () => {
      await Promise.resolve();
    });
  });

  it("enabling the server persists mcp_enabled and shows a restart hint", async () => {
    seed();
    render(<McpSettingsPane />);
    const toggle = screen
      .getByText("Enable MCP server (loopback)")
      .closest("label")!
      .querySelector("input") as HTMLInputElement;
    fireEvent.click(toggle);
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    const next = updateSettings.mock.calls[0]![0];
    expect(next.mcp_enabled).toBe(true);
    expect(
      screen.getByText(/Restart the app for MCP server changes/i),
    ).toBeInTheDocument();
  });

  it("when enabled, reveals the endpoint URL and masks the token until revealed", async () => {
    seed({ mcp_enabled: true, mcp_port: 8765, mcp_write_tools: false });
    render(<McpSettingsPane />);
    // The URL renders from the mocked getMcpServerInfo.
    await screen.findByText("http://127.0.0.1:8765/mcp");
    // The token is masked initially.
    expect(screen.queryByText("secrettoken123456")).not.toBeInTheDocument();
    // Reveal it.
    fireEvent.click(screen.getByRole("button", { name: "Reveal" }));
    expect(screen.getByText("secrettoken123456")).toBeInTheDocument();
  });

  it("toggling write tools persists mcp_write_tools", async () => {
    seed({ mcp_enabled: true });
    render(<McpSettingsPane />);
    const toggle = screen
      .getByText("Allow write tools over MCP")
      .closest("label")!
      .querySelector("input") as HTMLInputElement;
    fireEvent.click(toggle);
    await waitFor(() => expect(updateSettings).toHaveBeenCalled());
    const next = updateSettings.mock.calls.at(-1)![0];
    expect(next.mcp_write_tools).toBe(true);
  });
});
