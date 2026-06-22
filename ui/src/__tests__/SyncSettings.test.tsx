/**
 * Sync settings pane tests (WS4-B S5).
 *
 * Covers the pane's own behaviour: status display, ticket reveal + copy,
 * add-peer field + button routing through `sync_add_peer`, and `sync_now` for
 * the open meeting. The store's event-handler is tested separately (describe
 * "useSyncStatusStore event handling"). The "absent in the free build / no pane"
 * property is verified by the free-build grep in CI (VITE_CONNECTED is baked at
 * transform time, so it cannot be toggled from vitest — same as the MCP pane).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import type { SyncStatus } from "../ipc/bindings";

// IPC client mock: the four sync commands.
const syncStatus = vi.fn(
  async (): Promise<{ status: "ok"; data: SyncStatus }> => ({
    status: "ok",
    data: { kind: "idle" },
  }),
);
const syncGetMyTicket = vi.fn(
  async () => ({ status: "ok", data: "test-ticket-abc123" }) as const,
);
const syncAddPeer = vi.fn(
  async (_ticket: string) => ({ status: "ok", data: null }) as const,
);
const syncNow = vi.fn(
  async (_meetingId: string) => ({ status: "ok", data: null }) as const,
);

vi.mock("../ipc/client", () => ({
  commands: {
    syncStatus: () => syncStatus(),
    syncGetMyTicket: () => syncGetMyTicket(),
    syncAddPeer: (ticket: string) => syncAddPeer(ticket),
    syncNow: (meetingId: string) => syncNow(meetingId),
  },
  unwrap: <T,>(r: { status: string; data: T }) => {
    if (r.status !== "ok") throw new Error("err");
    return r.data;
  },
}));

// On `sync_ready` the sync store reloads the meeting list (and re-reads the open
// meeting). Mock the meetings IPC seam so those calls are inert in this test.
const listMeetings = vi.fn(async () => []);
const openMeeting = vi.fn(async () => null);
vi.mock("../ipc/meetings", () => ({
  listMeetings: () => listMeetings(),
  openMeeting: () => openMeeting(),
  renameMeeting: vi.fn(),
  setSpeakerName: vi.fn(),
  deleteMeeting: vi.fn(),
  reprocess: vi.fn(),
}));

import { SyncSettingsPane } from "../shell/SyncSettingsPane";
import { useSyncStatusStore } from "../state/sync-status";
import { useMeetingsStore } from "../state/meetings";

function resetStores() {
  act(() => {
    useSyncStatusStore.setState({
      status: null,
      inProgress: null,
      myTicket: null,
      lastError: null,
      pendingReadyNotifications: [],
    });
    useMeetingsStore.setState({ openMeetingId: null });
  });
}

describe("SyncSettingsPane", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStores();
  });

  it("fetches status and ticket on mount", async () => {
    render(<SyncSettingsPane />);
    await waitFor(() => expect(syncStatus).toHaveBeenCalledOnce());
    await waitFor(() => expect(syncGetMyTicket).toHaveBeenCalledOnce());
  });

  it("shows the live status from sync_status", async () => {
    syncStatus.mockResolvedValueOnce({
      status: "ok",
      data: { kind: "connecting" },
    });
    render(<SyncSettingsPane />);
    await screen.findByText("Connecting…");
  });

  it("shows the ticket returned by sync_get_my_ticket", async () => {
    render(<SyncSettingsPane />);
    await screen.findByText("test-ticket-abc123");
  });

  it("shows an error message on status from a kind=error engine", async () => {
    syncStatus.mockResolvedValueOnce({
      status: "ok",
      data: { kind: "error", message: "endpoint bind failed" },
    });
    render(<SyncSettingsPane />);
    await screen.findByText(/Error: endpoint bind failed/);
  });

  it("add-peer button calls sync_add_peer and clears the field", async () => {
    render(<SyncSettingsPane />);
    // Wait for mount fetch to settle.
    await waitFor(() => expect(syncGetMyTicket).toHaveBeenCalled());

    const input = screen.getByLabelText("Peer ticket") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "some-peer-ticket" } });
    fireEvent.click(screen.getByRole("button", { name: /Add$/ }));

    await waitFor(() =>
      expect(syncAddPeer).toHaveBeenCalledWith("some-peer-ticket"),
    );
    // Field clears after a successful add.
    expect(input.value).toBe("");
  });

  it("add-peer button is disabled when the field is empty", async () => {
    render(<SyncSettingsPane />);
    await waitFor(() => expect(syncGetMyTicket).toHaveBeenCalled());
    const button = screen.getByRole("button", { name: /Add$/ });
    expect(button).toBeDisabled();
  });

  it("shows 'Sync this meeting now' only when a meeting is open", async () => {
    render(<SyncSettingsPane />);
    await waitFor(() => expect(syncGetMyTicket).toHaveBeenCalled());
    // No meeting open → button absent.
    expect(
      screen.queryByRole("button", { name: /Sync this meeting now/ }),
    ).not.toBeInTheDocument();

    // Open a meeting → button appears.
    act(() => {
      useMeetingsStore.setState({ openMeetingId: "meeting-123" });
    });
    expect(
      screen.getByRole("button", { name: /Sync this meeting now/ }),
    ).toBeInTheDocument();
  });

  it("'Sync now' calls sync_now with the open meeting id", async () => {
    act(() => {
      useMeetingsStore.setState({ openMeetingId: "meeting-abc" });
    });
    render(<SyncSettingsPane />);
    await waitFor(() => expect(syncGetMyTicket).toHaveBeenCalled());

    fireEvent.click(
      screen.getByRole("button", { name: /Sync this meeting now/ }),
    );
    await waitFor(() =>
      expect(syncNow).toHaveBeenCalledWith("meeting-abc"),
    );
  });

  it("surfaces lastError from the store", () => {
    act(() => {
      useSyncStatusStore.setState({ lastError: "dial timed out" });
    });
    render(<SyncSettingsPane />);
    expect(screen.getByText("dial timed out")).toBeInTheDocument();
  });
});

describe("useSyncStatusStore event handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStores();
  });

  it("sync_ready queues a pending notification for the meeting", () => {
    act(() => {
      useSyncStatusStore
        .getState()
        .handleEvent({ kind: "sync_ready", meeting_id: "mtg-1" });
    });
    expect(
      useSyncStatusStore.getState().pendingReadyNotifications,
    ).toContain("mtg-1");
  });

  it("multiple sync_ready events accumulate independently", () => {
    act(() => {
      const { handleEvent } = useSyncStatusStore.getState();
      handleEvent({ kind: "sync_ready", meeting_id: "mtg-1" });
      handleEvent({ kind: "sync_ready", meeting_id: "mtg-2" });
    });
    const pending = useSyncStatusStore.getState().pendingReadyNotifications;
    expect(pending).toContain("mtg-1");
    expect(pending).toContain("mtg-2");
  });

  it("dismissReadyNotification removes only the specified meeting", () => {
    act(() => {
      useSyncStatusStore.setState({
        pendingReadyNotifications: ["mtg-1", "mtg-2"],
      });
    });
    act(() => {
      useSyncStatusStore.getState().dismissReadyNotification("mtg-1");
    });
    const pending = useSyncStatusStore.getState().pendingReadyNotifications;
    expect(pending).not.toContain("mtg-1");
    expect(pending).toContain("mtg-2");
  });

  it("sync_error sets lastError", () => {
    act(() => {
      useSyncStatusStore
        .getState()
        .handleEvent({ kind: "sync_error", context: "peer unreachable" });
    });
    expect(useSyncStatusStore.getState().lastError).toBe("peer unreachable");
  });

  it("sync_progress sets the in-flight transfer state", () => {
    act(() => {
      useSyncStatusStore.getState().handleEvent({
        kind: "sync_progress",
        meeting_id: "mtg-1",
        label: "Syncing notes…",
        fraction: 0.5,
      });
    });
    const inProgress = useSyncStatusStore.getState().inProgress;
    expect(inProgress).toEqual({
      meetingId: "mtg-1",
      label: "Syncing notes…",
      fraction: 0.5,
    });
    // No terminal side effects yet.
    expect(
      useSyncStatusStore.getState().pendingReadyNotifications,
    ).toHaveLength(0);
    expect(useSyncStatusStore.getState().lastError).toBeNull();
  });

  it("sync_progress with a null fraction is indeterminate", () => {
    act(() => {
      useSyncStatusStore.getState().handleEvent({
        kind: "sync_progress",
        meeting_id: "mtg-1",
        label: "Syncing…",
        fraction: null,
      });
    });
    expect(useSyncStatusStore.getState().inProgress?.fraction).toBeNull();
  });

  it("sync_ready clears the in-flight state and refreshes the meeting list", () => {
    act(() => {
      useSyncStatusStore.setState({
        inProgress: { meetingId: "mtg-1", label: "Syncing…", fraction: 0.5 },
      });
    });
    act(() => {
      useSyncStatusStore
        .getState()
        .handleEvent({ kind: "sync_ready", meeting_id: "mtg-1" });
    });
    expect(useSyncStatusStore.getState().inProgress).toBeNull();
    expect(listMeetings).toHaveBeenCalled();
  });

  it("sync_ready re-reads the open meeting when it matches", () => {
    act(() => {
      useMeetingsStore.setState({ openMeetingId: "mtg-1" });
    });
    act(() => {
      useSyncStatusStore
        .getState()
        .handleEvent({ kind: "sync_ready", meeting_id: "mtg-1" });
    });
    expect(openMeeting).toHaveBeenCalled();
  });

  it("sync_error clears the in-flight state and sets lastError", () => {
    act(() => {
      useSyncStatusStore.setState({
        inProgress: { meetingId: "mtg-1", label: "Syncing…", fraction: null },
      });
    });
    act(() => {
      useSyncStatusStore
        .getState()
        .handleEvent({ kind: "sync_error", context: "boom" });
    });
    expect(useSyncStatusStore.getState().inProgress).toBeNull();
    expect(useSyncStatusStore.getState().lastError).toBe("boom");
  });

  it("an unrelated event does not affect the sync store", () => {
    act(() => {
      useSyncStatusStore.getState().handleEvent({
        kind: "state_changed",
        state: { kind: "idle" },
      });
    });
    expect(
      useSyncStatusStore.getState().pendingReadyNotifications,
    ).toHaveLength(0);
  });
});
