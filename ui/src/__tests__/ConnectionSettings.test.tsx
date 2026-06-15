/**
 * Connection (connector / relay tunnel) settings pane tests (WS4-A S5b).
 *
 * Covers the pane's behaviour with the IPC + opener mocked: the pairing flow
 * shows the user code and opens the verification URL, the enable toggle routes
 * through `set_connector_enabled`, and live status transitions (via the event
 * dispatcher) are reflected. The "absent in the free build / no pane" property
 * is verified by the free-build grep in CI (VITE_CONNECTED is baked at transform
 * time, so it cannot be toggled from vitest — same as the MCP pane).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import type { TunnelSnapshot, TunnelStatus } from "../ipc/bindings";

// The opener plugin — assert we open the verification URL during pairing.
const { openUrl } = vi.hoisted(() => ({
  openUrl: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/plugin-opener", () => ({ openUrl }));

// IPC client mock: the four tunnel commands.
type TunnelStatusResult = { status: "ok"; data: TunnelSnapshot };
const tunnelStatus = vi.fn(
  async (): Promise<TunnelStatusResult> => ({
    status: "ok",
    data: { enabled: false, status: "disconnected", account_id: null },
  }),
);
const tunnelBeginPairing = vi.fn(
  async () =>
    ({
      status: "ok",
      data: { user_code: "ABCD-1234", verification_uri: "https://auth.example/device?code=ABCD-1234" },
    }) as const,
);
const tunnelPollPairing = vi.fn(
  async () => ({ status: "ok", data: "online" as TunnelStatus }) as const,
);
const setConnectorEnabled = vi.fn(
  async (enabled: boolean) =>
    ({
      status: "ok",
      data: { enabled, status: "disconnected", account_id: null },
    }) as const,
);

vi.mock("../ipc/client", () => ({
  commands: {
    tunnelStatus: () => tunnelStatus(),
    tunnelBeginPairing: () => tunnelBeginPairing(),
    tunnelPollPairing: () => tunnelPollPairing(),
    setConnectorEnabled: (enabled: boolean) => setConnectorEnabled(enabled),
  },
  unwrap: <T,>(r: { status: string; data: T }) => {
    if (r.status !== "ok") throw new Error("err");
    return r.data;
  },
}));

import { ConnectionSettingsPane } from "../shell/ConnectionSettingsPane";
import { useTunnelStatusStore } from "../state/tunnel-status";

function resetStore() {
  act(() => {
    useTunnelStatusStore.setState({
      snapshot: null,
      userCode: null,
      lastError: null,
    });
  });
}

describe("ConnectionSettingsPane", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    resetStore();
  });

  it("fetches and shows the connector status on mount", async () => {
    render(<ConnectionSettingsPane />);
    // Default snapshot is disconnected.
    await screen.findByText("Not connected");
    expect(tunnelStatus).toHaveBeenCalled();
  });

  it("enabling the connector routes through set_connector_enabled", async () => {
    render(<ConnectionSettingsPane />);
    await screen.findByText("Not connected");
    const toggle = screen
      .getByText("Enable the connector")
      .closest("label")!
      .querySelector("input") as HTMLInputElement;
    fireEvent.click(toggle);
    await waitFor(() => expect(setConnectorEnabled).toHaveBeenCalledWith(true));
  });

  it("pairing shows the user code and opens the verification URL", async () => {
    vi.useFakeTimers();
    render(<ConnectionSettingsPane />);
    // Let the on-mount status fetch settle.
    await act(async () => {
      await Promise.resolve();
    });

    const button = screen.getByRole("button", { name: /Pair this device/ });
    fireEvent.click(button);

    // begin_pairing resolves: the code is shown and the URL is opened.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(tunnelBeginPairing).toHaveBeenCalledOnce();
    expect(screen.getByText("ABCD-1234")).toBeInTheDocument();
    expect(openUrl).toHaveBeenCalledWith(
      "https://auth.example/device?code=ABCD-1234",
    );
    vi.useRealTimers();
  });

  it("reflects a live tunnel_status_changed event", () => {
    act(() => {
      useTunnelStatusStore.setState({
        snapshot: { enabled: true, status: "connecting", account_id: "acct-1" },
      });
    });
    render(<ConnectionSettingsPane />);
    expect(screen.getByText("Connecting…")).toBeInTheDocument();

    // A status event flips it to Online without a remount.
    act(() => {
      useTunnelStatusStore
        .getState()
        .handleEvent({ kind: "tunnel_status_changed", status: "online" });
    });
    expect(screen.getByText("Online")).toBeInTheDocument();
  });

  it("shows the paired account when present", async () => {
    tunnelStatus.mockResolvedValueOnce({
      status: "ok",
      data: { enabled: true, status: "online", account_id: "sub-abc123" },
    });
    render(<ConnectionSettingsPane />);
    await screen.findByText("Online");
    expect(screen.getByText("sub-abc123")).toBeInTheDocument();
  });
});

describe("useTunnelStatusStore event handling", () => {
  beforeEach(resetStore);

  it("tunnel_status_changed patches the status, preserving enabled/account", () => {
    act(() => {
      useTunnelStatusStore.setState({
        snapshot: { enabled: true, status: "connecting", account_id: "acct-1" },
      });
    });
    act(() => {
      useTunnelStatusStore
        .getState()
        .handleEvent({ kind: "tunnel_status_changed", status: "needs_repair" });
    });
    const snap = useTunnelStatusStore.getState().snapshot!;
    expect(snap.status).toBe("needs_repair");
    expect(snap.enabled).toBe(true);
    expect(snap.account_id).toBe("acct-1");
  });

  it("a non-tunnel event does not touch the snapshot", () => {
    act(() => {
      useTunnelStatusStore.setState({
        snapshot: { enabled: true, status: "online", account_id: "acct-1" },
      });
    });
    act(() => {
      useTunnelStatusStore.getState().handleEvent({
        kind: "state_changed",
        state: { kind: "idle" },
      });
    });
    expect(useTunnelStatusStore.getState().snapshot!.status).toBe("online");
  });
});
