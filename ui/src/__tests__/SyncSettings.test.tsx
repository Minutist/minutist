/**
 * Sync settings pane tests (WS4-B S5), including account sign-in (WS4-A S5b).
 *
 * Covers the pane's own behaviour: account status display, the device-code
 * pairing flow (code shown/opened URL), sign-in/delete-account, sync status
 * display, ticket reveal + copy, add-peer field + button routing through
 * `sync_add_peer`, and `sync_now` for the open meeting. The stores' own
 * event-handler tests live in their own `describe` blocks below. The "absent
 * in the free build / no pane" property is verified by the free-build grep in
 * CI (VITE_CONNECTED is baked at transform time, so it cannot be toggled from
 * vitest — same as the MCP pane).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import type {
  AccountSnapshot,
  AccountStatus,
  PairingPrompt,
  SyncStatus,
} from "../ipc/bindings";

// The opener plugin — assert we open the verification URL during pairing.
const { openUrl } = vi.hoisted(() => ({
  openUrl: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/plugin-opener", () => ({ openUrl }));

// IPC client mock: the account + sync commands.
type AccountStatusResult = { status: "ok"; data: AccountSnapshot };
const accountStatus = vi.fn(
  async (): Promise<AccountStatusResult> => ({
    status: "ok",
    data: { status: "signed_out", account_id: null },
  }),
);
type PairingPromptResult = { status: "ok"; data: PairingPrompt };
const accountBeginPairing = vi.fn(
  async (): Promise<PairingPromptResult> => ({
    status: "ok",
    data: {
      user_code: "ABCD-1234",
      verification_uri: "https://auth.example/device",
      code_required: true,
    },
  }),
);
const accountPollPairing = vi.fn(
  async () => ({ status: "ok", data: "signed_in" as AccountStatus }) as const,
);
const deleteAccount = vi.fn(async () => ({ status: "ok", data: null }) as const);

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
    accountStatus: () => accountStatus(),
    accountBeginPairing: () => accountBeginPairing(),
    accountPollPairing: () => accountPollPairing(),
    deleteAccount: () => deleteAccount(),
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
import { useAccountStatusStore } from "../state/account-status";
import { useSyncStatusStore } from "../state/sync-status";
import { useMeetingsStore } from "../state/meetings";

function resetStores() {
  act(() => {
    useAccountStatusStore.setState({
      snapshot: null,
      userCode: null,
      codeRequired: false,
      lastError: null,
    });
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
    vi.useRealTimers();
    resetStores();
  });

  it("fetches account + sync status and ticket on mount", async () => {
    render(<SyncSettingsPane />);
    await waitFor(() => expect(accountStatus).toHaveBeenCalledOnce());
    await waitFor(() => expect(syncStatus).toHaveBeenCalledOnce());
    await waitFor(() => expect(syncGetMyTicket).toHaveBeenCalledOnce());
  });

  it("shows 'Not signed in' and a Log in button by default", async () => {
    render(<SyncSettingsPane />);
    await screen.findByText("Not signed in");
    expect(screen.getByRole("button", { name: "Log in" })).toBeInTheDocument();
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
    // Field clears after the add RESOLVES — an async state update that lands after
    // the call is made, so wait for it rather than asserting synchronously (the
    // issue-0026 flake: the bare assertion raced the resolve+clear).
    await waitFor(() => expect(input.value).toBe(""));
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

  it("surfaces lastError from the sync store", () => {
    act(() => {
      useSyncStatusStore.setState({ lastError: "dial timed out" });
    });
    render(<SyncSettingsPane />);
    expect(screen.getByText("dial timed out")).toBeInTheDocument();
  });
});

describe("SyncSettingsPane account section", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    resetStores();
  });

  it("pairing shows the user code and opens the verification URL when a code must be typed", async () => {
    vi.useFakeTimers();
    render(<SyncSettingsPane />);
    // Let the on-mount status fetch settle.
    await act(async () => {
      await Promise.resolve();
    });

    const button = screen.getByRole("button", { name: "Log in" });
    fireEvent.click(button);

    // begin_pairing resolves: the code is shown and the URL is opened.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(accountBeginPairing).toHaveBeenCalledOnce();
    expect(screen.getByText("ABCD-1234")).toBeInTheDocument();
    expect(openUrl).toHaveBeenCalledWith("https://auth.example/device");
    vi.useRealTimers();
  });

  it("pairing does not show the code when the opened URL already carries it", async () => {
    accountBeginPairing.mockResolvedValueOnce({
      status: "ok",
      data: {
        user_code: "ABCD-1234",
        verification_uri: "https://auth.example/device?code=ABCD-1234",
        code_required: false,
      },
    });
    vi.useFakeTimers();
    render(<SyncSettingsPane />);
    await act(async () => {
      await Promise.resolve();
    });

    const button = screen.getByRole("button", { name: "Log in" });
    fireEvent.click(button);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.queryByText("ABCD-1234")).not.toBeInTheDocument();
    expect(screen.getByText(/already filled in/i)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it("reflects a live account_status_changed event", () => {
    act(() => {
      useAccountStatusStore.setState({
        snapshot: { status: "pairing", account_id: null },
      });
    });
    render(<SyncSettingsPane />);
    expect(
      screen.getByText("Signing in — approve in your browser"),
    ).toBeInTheDocument();

    // A status event flips it to signed_in without a remount.
    act(() => {
      useAccountStatusStore
        .getState()
        .handleEvent({ kind: "account_status_changed", status: "signed_in" });
    });
    expect(screen.getByText("Signed in")).toBeInTheDocument();
  });

  it("shows the signed-in account and a Delete account button", async () => {
    accountStatus.mockResolvedValueOnce({
      status: "ok",
      data: { status: "signed_in", account_id: "sub-abc123" },
    });
    render(<SyncSettingsPane />);
    await screen.findByText("Signed in");
    expect(screen.getByText("sub-abc123")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Delete account" }),
    ).toBeInTheDocument();
  });

  it("Delete account calls delete_account after confirmation", async () => {
    accountStatus.mockResolvedValueOnce({
      status: "ok",
      data: { status: "signed_in", account_id: "sub-abc123" },
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<SyncSettingsPane />);
    await screen.findByText("Signed in");

    fireEvent.click(screen.getByRole("button", { name: "Delete account" }));
    await waitFor(() => expect(deleteAccount).toHaveBeenCalledOnce());
  });

  it("surfaces lastError from the account store", () => {
    act(() => {
      useAccountStatusStore.setState({ lastError: "pairing expired" });
    });
    render(<SyncSettingsPane />);
    expect(screen.getByText("pairing expired")).toBeInTheDocument();
  });
});

describe("useAccountStatusStore event handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStores();
  });

  it("account_status_changed patches the status, preserving the account id", () => {
    act(() => {
      useAccountStatusStore.setState({
        snapshot: { status: "pairing", account_id: "acct-1" },
      });
    });
    act(() => {
      useAccountStatusStore
        .getState()
        .handleEvent({ kind: "account_status_changed", status: "signed_in" });
    });
    const snap = useAccountStatusStore.getState().snapshot!;
    expect(snap.status).toBe("signed_in");
    expect(snap.account_id).toBe("acct-1");
  });

  it("a non-account event does not touch the snapshot", () => {
    act(() => {
      useAccountStatusStore.setState({
        snapshot: { status: "signed_in", account_id: "acct-1" },
      });
    });
    act(() => {
      useAccountStatusStore.getState().handleEvent({
        kind: "state_changed",
        state: { kind: "idle" },
      });
    });
    expect(useAccountStatusStore.getState().snapshot!.status).toBe(
      "signed_in",
    );
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
