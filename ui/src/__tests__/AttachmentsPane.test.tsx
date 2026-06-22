/**
 * Tests for the attachments pane.
 *
 * Asserts:
 *   - the empty state shows a drop affordance + an "Add files" action,
 *   - the manifest loads on mount (via the `../ipc/attachments` seam) and rows
 *     render with filename / extension badge / conversion state,
 *   - a ready row shows "Open"; a failed row shows the reason + "Open anyway",
 *   - a converting row shows the indeterminate indicator,
 *   - the picker routes a chosen file through `add`,
 *   - Remove routes through `remove`, Open through `open`.
 *
 * The IPC calls are mocked at the `../ipc/attachments` seam (per the architecture
 * testing policy — the seam is mocked, not the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  act,
  waitFor,
} from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

vi.mock("../ipc/attachments", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../ipc/attachments")>();
  return {
    ...actual,
    addAttachment: vi.fn(),
    listAttachments: vi.fn().mockResolvedValue([]),
    openAttachment: vi.fn().mockResolvedValue(undefined),
    removeAttachment: vi.fn().mockResolvedValue(undefined),
  };
});

import { AttachmentsPane } from "../shell/AttachmentsPane";
import { useAttachmentsStore } from "../state/attachments";
import {
  addAttachment,
  listAttachments,
  openAttachment,
  removeAttachment,
} from "../ipc/attachments";
import type { AttachmentEntry } from "../ipc/bindings";

const MEETING = "meeting-0001";

function entry(over: Partial<AttachmentEntry> = {}): AttachmentEntry {
  return {
    id: "att-1",
    hash: "abc123",
    original_filename: "report.xlsx",
    ext: "xlsx",
    byte_len: 48_213,
    added_at: "2026-06-22T00:00:00Z",
    conversion: { state: "ready" },
    converted_md_filename: "abc123.md",
    ...over,
  };
}

function resetStore() {
  act(() => {
    useAttachmentsStore.setState({
      attachments: [],
      meetingId: null,
      loading: false,
      adding: 0,
      lastError: null,
    });
  });
}

describe("AttachmentsPane", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("renders an empty state with a drop affordance and Add files", async () => {
    vi.mocked(listAttachments).mockResolvedValue([]);
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() =>
      expect(listAttachments).toHaveBeenCalledWith(MEETING),
    );
    expect(screen.getByText(/Drop files here/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Add files" }),
    ).toBeInTheDocument();
  });

  it("loads the manifest on mount and renders a ready row with Open", async () => {
    vi.mocked(listAttachments).mockResolvedValue([
      entry({ original_filename: "Q2 report.xlsx" }),
    ]);
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByText("Q2 report.xlsx")).toBeInTheDocument(),
    );
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open" })).toBeInTheDocument();
  });

  it("shows the converting indicator for a pending row", async () => {
    vi.mocked(listAttachments).mockResolvedValue([
      entry({ id: "att-p", conversion: { state: "pending" } }),
    ]);
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByText("report.xlsx")).toBeInTheDocument(),
    );
    expect(screen.getByText("Converting…")).toBeInTheDocument();
    expect(
      screen.getByRole("progressbar", { name: "Converting attachment" }),
    ).toBeInTheDocument();
  });

  it("shows the failure reason and an Open anyway fallback for a failed row", async () => {
    vi.mocked(listAttachments).mockResolvedValue([
      entry({
        id: "att-f",
        conversion: { state: "failed", reason: "no extractable text" },
      }),
    ]);
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByText(/no extractable text/i)).toBeInTheDocument(),
    );
    expect(
      screen.getByRole("button", { name: "Open anyway" }),
    ).toBeInTheDocument();
  });

  it("routes a picked file through add", async () => {
    vi.mocked(listAttachments).mockResolvedValue([]);
    vi.mocked(addAttachment).mockResolvedValue(entry({ id: "att-new" }));
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() => expect(listAttachments).toHaveBeenCalled());

    const file = new File(["data"], "notes.pdf", { type: "application/pdf" });
    const input = document.querySelector(
      ".attachments-pane__file-input",
    ) as HTMLInputElement;
    act(() => {
      fireEvent.change(input, { target: { files: [file] } });
    });

    await waitFor(() =>
      expect(addAttachment).toHaveBeenCalledWith(MEETING, file, "pdf"),
    );
  });

  it("rejects an unsupported file type inline without calling add", async () => {
    vi.mocked(listAttachments).mockResolvedValue([]);
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() => expect(listAttachments).toHaveBeenCalled());

    const file = new File(["x"], "video.mov", { type: "video/quicktime" });
    const input = document.querySelector(
      ".attachments-pane__file-input",
    ) as HTMLInputElement;
    act(() => {
      fireEvent.change(input, { target: { files: [file] } });
    });

    expect(addAttachment).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/unsupported file type/i);
  });

  it("Open invokes the open action", async () => {
    vi.mocked(listAttachments).mockResolvedValue([entry({ id: "att-o" })]);
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Open" })).toBeInTheDocument(),
    );
    act(() => fireEvent.click(screen.getByRole("button", { name: "Open" })));
    await waitFor(() =>
      expect(openAttachment).toHaveBeenCalledWith(
        MEETING,
        expect.objectContaining({ id: "att-o" }),
      ),
    );
  });

  it("Remove invokes the remove action with the attachment id", async () => {
    vi.mocked(listAttachments).mockResolvedValue([entry({ id: "att-r" })]);
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Remove report.xlsx/i }),
      ).toBeInTheDocument(),
    );
    act(() =>
      fireEvent.click(
        screen.getByRole("button", { name: /Remove report.xlsx/i }),
      ),
    );
    await waitFor(() =>
      expect(removeAttachment).toHaveBeenCalledWith(MEETING, "att-r"),
    );
  });

  it("accepts a drag-and-drop of a supported file", async () => {
    vi.mocked(listAttachments).mockResolvedValue([]);
    vi.mocked(addAttachment).mockResolvedValue(entry({ id: "att-drop" }));
    render(<AttachmentsPane meetingId={MEETING} />);
    await waitFor(() => expect(listAttachments).toHaveBeenCalled());

    const pane = screen.getByLabelText("Meeting attachments");
    const file = new File(["x"], "slides.pptx");
    act(() => {
      fireEvent.drop(pane, { dataTransfer: { files: [file] } });
    });

    await waitFor(() =>
      expect(addAttachment).toHaveBeenCalledWith(MEETING, file, "pptx"),
    );
  });
});
