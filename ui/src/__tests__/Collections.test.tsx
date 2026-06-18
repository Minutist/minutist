/**
 * Tests for the collections ("folders") feature: the filter helper, the
 * collections store, the sidebar, and the meeting-list "Move to…" action.
 *
 * The IPC seams (`../ipc/collections`, `../ipc/meetings`) are mocked per the
 * architecture testing policy; the generated bindings file is not faked.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
  once: vi.fn().mockResolvedValue(() => {}),
  emit: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@tauri-apps/api/webviewWindow", () => ({ WebviewWindow: vi.fn() }));

vi.mock("../ipc/collections", () => ({
  listCollections: vi.fn().mockResolvedValue([]),
  createCollection: vi.fn(),
  renameCollection: vi.fn().mockResolvedValue(undefined),
  deleteCollection: vi.fn().mockResolvedValue(undefined),
  setMeetingCollection: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../ipc/meetings", () => ({
  listMeetings: vi.fn().mockResolvedValue([]),
  openMeeting: vi.fn().mockResolvedValue({}),
  renameMeeting: vi.fn().mockResolvedValue(undefined),
  setSpeakerName: vi.fn().mockResolvedValue({}),
  deleteMeeting: vi.fn().mockResolvedValue(undefined),
  reprocess: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../ipc/summary", () => ({
  summariseMeeting: vi.fn().mockResolvedValue(undefined),
  getSummary: vi.fn().mockResolvedValue(null),
  saveSummary: vi.fn().mockResolvedValue(undefined),
}));

import { MeetingList } from "../shell/MeetingList";
import { CollectionsSidebar } from "../shell/CollectionsSidebar";
import {
  useCollectionsStore,
  meetingMatchesFilter,
  ALL_FILTER,
  UNFILED_FILTER,
} from "../state/collections";
import { useMeetingsStore } from "../state/meetings";
import * as collectionsIpc from "../ipc/collections";
import * as meetingsIpc from "../ipc/meetings";

const PROJECTS = { id: "col-projects", name: "Projects", position: 0 };
const PERSONAL = { id: "col-personal", name: "Personal", position: 1 };

const MEETINGS = [
  {
    id: "m1",
    title: "Launch sync",
    started_at: "2026-05-26T14:05:00Z",
    duration_ms: 32 * 60 * 1000,
    speaker_count: 3,
    excerpt: null,
    collection_id: "col-projects",
  },
  {
    id: "m2",
    title: "Standup",
    started_at: "2026-05-18T08:00:00Z",
    duration_ms: 8 * 60 * 1000,
    speaker_count: 1,
    excerpt: null,
    collection_id: null,
  },
];

function seedStores() {
  act(() => {
    useMeetingsStore.setState({
      meetings: MEETINGS,
      loading: false,
      openMeetingId: null,
      openMeetingState: null,
      lastError: null,
    });
    useCollectionsStore.setState({
      collections: [PROJECTS, PERSONAL],
      filter: ALL_FILTER,
      lastError: null,
    });
  });
}

describe("meetingMatchesFilter", () => {
  it("matches All always, Unfiled only when no folder, and a folder by id", () => {
    expect(meetingMatchesFilter(ALL_FILTER, "col-projects")).toBe(true);
    expect(meetingMatchesFilter(ALL_FILTER, null)).toBe(true);

    expect(meetingMatchesFilter(UNFILED_FILTER, null)).toBe(true);
    expect(meetingMatchesFilter(UNFILED_FILTER, undefined)).toBe(true);
    expect(meetingMatchesFilter(UNFILED_FILTER, "col-projects")).toBe(false);

    const f = { kind: "collection", id: "col-projects" } as const;
    expect(meetingMatchesFilter(f, "col-projects")).toBe(true);
    expect(meetingMatchesFilter(f, "col-personal")).toBe(false);
    expect(meetingMatchesFilter(f, null)).toBe(false);
  });
});

describe("collections store", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    act(() => {
      useCollectionsStore.setState({
        collections: [],
        filter: ALL_FILTER,
        lastError: null,
      });
    });
  });

  it("create routes through the seam and refreshes the list", async () => {
    vi.mocked(collectionsIpc.createCollection).mockResolvedValue(PROJECTS);
    vi.mocked(collectionsIpc.listCollections).mockResolvedValue([PROJECTS]);

    await act(async () => {
      await useCollectionsStore.getState().create("Projects");
    });

    expect(collectionsIpc.createCollection).toHaveBeenCalledWith("Projects");
    expect(useCollectionsStore.getState().collections).toEqual([PROJECTS]);
  });

  it("a refresh resets the filter to All when the selected folder is gone", async () => {
    act(() => {
      useCollectionsStore.setState({
        filter: { kind: "collection", id: "col-gone" },
      });
    });
    vi.mocked(collectionsIpc.listCollections).mockResolvedValue([PROJECTS]);

    await act(async () => {
      await useCollectionsStore.getState().refresh();
    });

    expect(useCollectionsStore.getState().filter).toEqual(ALL_FILTER);
  });
});

describe("CollectionsSidebar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    seedStores();
  });

  it("renders All / folders / Unfiled with derived counts", () => {
    render(<CollectionsSidebar />);
    expect(screen.getByText("All meetings")).toBeInTheDocument();
    expect(screen.getByText("Projects")).toBeInTheDocument();
    expect(screen.getByText("Personal")).toBeInTheDocument();
    expect(screen.getByText("Unfiled")).toBeInTheDocument();
    // Counts: 2 total, 1 in Projects, 0 in Personal, 1 unfiled.
    const counts = screen
      .getAllByText(/^\d+$/)
      .map((n) => n.textContent);
    expect(counts).toEqual(["2", "1", "0", "1"]);
  });

  it("selecting a folder sets the filter", () => {
    render(<CollectionsSidebar />);
    fireEvent.click(screen.getByText("Projects"));
    expect(useCollectionsStore.getState().filter).toEqual({
      kind: "collection",
      id: "col-projects",
    });
  });

  it("the + control creates a folder", async () => {
    vi.mocked(collectionsIpc.createCollection).mockResolvedValue({
      id: "col-new",
      name: "Clients",
      position: 2,
    });
    vi.mocked(collectionsIpc.listCollections).mockResolvedValue([
      PROJECTS,
      PERSONAL,
    ]);
    render(<CollectionsSidebar />);
    fireEvent.click(screen.getByRole("button", { name: "New folder" }));
    const input = screen.getByLabelText("New folder name");
    fireEvent.change(input, { target: { value: "Clients" } });
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter" });
    });
    await waitFor(() =>
      expect(collectionsIpc.createCollection).toHaveBeenCalledWith("Clients"),
    );
  });
});

describe("MeetingList folder filtering + move", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(meetingsIpc.listMeetings).mockResolvedValue(MEETINGS);
    vi.mocked(collectionsIpc.listCollections).mockResolvedValue([
      PROJECTS,
      PERSONAL,
    ]);
    seedStores();
  });

  it("shows only the selected folder's meetings", async () => {
    render(<MeetingList />);
    // All: both rows present.
    expect(await screen.findByText("Launch sync")).toBeInTheDocument();
    expect(screen.getByText("Standup")).toBeInTheDocument();

    // Filter to Projects (m1 only).
    act(() => {
      useCollectionsStore.getState().select({
        kind: "collection",
        id: "col-projects",
      });
    });
    expect(screen.getByText("Launch sync")).toBeInTheDocument();
    expect(screen.queryByText("Standup")).not.toBeInTheDocument();
  });

  it("Move to… files a meeting through the meetings seam", async () => {
    render(<MeetingList />);
    await screen.findByText("Standup");

    // Open the move menu on the first row and pick "Personal".
    const moveButtons = screen.getAllByRole("button", { name: "Move to…" });
    fireEvent.click(moveButtons[0]);
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: "Personal" }));
    });
    await waitFor(() =>
      expect(collectionsIpc.setMeetingCollection).toHaveBeenCalledWith(
        "m1",
        "col-personal",
      ),
    );
  });

  it("dragging a meeting row onto a folder files it", async () => {
    render(<MeetingList />);
    await screen.findByText("Standup");

    // Minimal DataTransfer stub (jsdom doesn't implement it): tracks set data
    // and exposes `types` so `hasMeetingDrag` sees the meeting MIME on dragover.
    const store: Record<string, string> = {};
    const dataTransfer = {
      setData: (t: string, v: string) => {
        store[t] = v;
      },
      getData: (t: string) => store[t] ?? "",
      get types() {
        return Object.keys(store);
      },
      dropEffect: "none",
      effectAllowed: "none",
    } as unknown as DataTransfer;

    // Drag the unfiled "Standup" (m2) onto the "Personal" folder.
    const row = screen.getByText("Standup").closest("li");
    const personal = screen.getByText("Personal").closest("button");
    fireEvent.dragStart(row as Element, { dataTransfer });
    fireEvent.dragOver(personal as Element, { dataTransfer });
    await act(async () => {
      fireEvent.drop(personal as Element, { dataTransfer });
    });

    await waitFor(() =>
      expect(collectionsIpc.setMeetingCollection).toHaveBeenCalledWith(
        "m2",
        "col-personal",
      ),
    );
  });
});
