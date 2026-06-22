/**
 * Tests for the attachments store.
 *
 * Asserts:
 *   - `read` loads the manifest (via `list_attachments`) for a meeting,
 *   - `add` invokes `add_attachment` and inserts the returned (Pending) row,
 *   - the four attachment events flip / insert / drop rows, gated on the loaded
 *     meeting (a backgrounded event for another meeting is ignored),
 *   - `remove` drops the row optimistically and rolls back on a failed call.
 *
 * The IPC calls are mocked at the `../ipc/attachments` seam (per the architecture
 * testing policy — do not fake the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../ipc/attachments", () => ({
  addAttachment: vi.fn(),
  listAttachments: vi.fn().mockResolvedValue([]),
  openAttachment: vi.fn().mockResolvedValue(undefined),
  removeAttachment: vi.fn().mockResolvedValue(undefined),
}));

import {
  addAttachment,
  listAttachments,
  openAttachment,
  removeAttachment,
} from "../ipc/attachments";
import { useAttachmentsStore } from "../state/attachments";
import type { AttachmentEntry } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

const MEETING = "meeting-0001";

function entry(over: Partial<AttachmentEntry> = {}): AttachmentEntry {
  return {
    id: "att-1",
    hash: "abc123",
    original_filename: "report.xlsx",
    ext: "xlsx",
    byte_len: 1024,
    added_at: "2026-06-22T00:00:00Z",
    conversion: { state: "pending" },
    converted_md_filename: null,
    ...over,
  };
}

function resetStore() {
  useAttachmentsStore.setState({
    attachments: [],
    meetingId: null,
    loading: false,
    adding: 0,
    lastError: null,
  });
}

describe("useAttachmentsStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("read loads the manifest for a meeting", async () => {
    const rows = [entry({ id: "att-1" }), entry({ id: "att-2" })];
    vi.mocked(listAttachments).mockResolvedValueOnce(rows);

    await useAttachmentsStore.getState().read(MEETING);

    expect(listAttachments).toHaveBeenCalledWith(MEETING);
    expect(useAttachmentsStore.getState().attachments).toEqual(rows);
    expect(useAttachmentsStore.getState().meetingId).toBe(MEETING);
    expect(useAttachmentsStore.getState().loading).toBe(false);
  });

  it("read surfaces an error and clears loading on failure", async () => {
    vi.mocked(listAttachments).mockRejectedValueOnce(new Error("disk gone"));
    await useAttachmentsStore.getState().read(MEETING);
    expect(useAttachmentsStore.getState().lastError).toBe("disk gone");
    expect(useAttachmentsStore.getState().loading).toBe(false);
  });

  it("add invokes add_attachment and inserts the Pending row", async () => {
    useAttachmentsStore.setState({ meetingId: MEETING });
    const added = entry({ id: "att-9", original_filename: "deck.pptx" });
    vi.mocked(addAttachment).mockResolvedValueOnce(added);

    const file = new File(["x"], "deck.pptx");
    await useAttachmentsStore.getState().add(MEETING, file, "pptx");

    expect(addAttachment).toHaveBeenCalledWith(MEETING, file, "pptx");
    expect(useAttachmentsStore.getState().attachments).toEqual([added]);
    expect(useAttachmentsStore.getState().adding).toBe(0);
  });

  it("add does not double-insert when the event raced ahead", async () => {
    const added = entry({ id: "att-9" });
    useAttachmentsStore.setState({ meetingId: MEETING, attachments: [added] });
    vi.mocked(addAttachment).mockResolvedValueOnce(added);

    await useAttachmentsStore
      .getState()
      .add(MEETING, new File(["x"], "f.pdf"), "pdf");

    expect(useAttachmentsStore.getState().attachments).toHaveLength(1);
  });

  it("attachment_added inserts a row for the loaded meeting", () => {
    useAttachmentsStore.setState({ meetingId: MEETING, attachments: [] });
    const event: AppEvent = {
      kind: "attachment_added",
      meeting_id: MEETING,
      attachment: entry({ id: "att-5" }),
    };
    useAttachmentsStore.getState().handleEvent(event);
    expect(useAttachmentsStore.getState().attachments).toHaveLength(1);
    expect(useAttachmentsStore.getState().attachments[0].id).toBe("att-5");
  });

  it("ignores an attachment_added for a different meeting", () => {
    useAttachmentsStore.setState({ meetingId: MEETING, attachments: [] });
    useAttachmentsStore.getState().handleEvent({
      kind: "attachment_added",
      meeting_id: "other-meeting",
      attachment: entry({ id: "att-x" }),
    });
    expect(useAttachmentsStore.getState().attachments).toHaveLength(0);
  });

  it("attachment_converted flips the row to Ready and re-reads the manifest", () => {
    const pending = entry({ id: "att-1", conversion: { state: "pending" } });
    useAttachmentsStore.setState({
      meetingId: MEETING,
      attachments: [pending],
    });
    vi.mocked(listAttachments).mockResolvedValueOnce([
      entry({
        id: "att-1",
        conversion: { state: "ready" },
        converted_md_filename: "abc123.md",
      }),
    ]);

    useAttachmentsStore.getState().handleEvent({
      kind: "attachment_converted",
      meeting_id: MEETING,
      attachment_id: "att-1",
    });

    // Optimistic flip is synchronous; the re-read confirms.
    expect(useAttachmentsStore.getState().attachments[0].conversion.state).toBe(
      "ready",
    );
    expect(listAttachments).toHaveBeenCalledWith(MEETING);
  });

  it("attachment_conversion_failed flips the row to Failed with the reason", () => {
    useAttachmentsStore.setState({
      meetingId: MEETING,
      attachments: [entry({ id: "att-1" })],
    });
    useAttachmentsStore.getState().handleEvent({
      kind: "attachment_conversion_failed",
      meeting_id: MEETING,
      attachment_id: "att-1",
      reason: "password protected",
    });
    const row = useAttachmentsStore.getState().attachments[0];
    expect(row.conversion).toEqual({
      state: "failed",
      reason: "password protected",
    });
  });

  it("attachment_removed drops the row", () => {
    useAttachmentsStore.setState({
      meetingId: MEETING,
      attachments: [entry({ id: "att-1" }), entry({ id: "att-2" })],
    });
    useAttachmentsStore.getState().handleEvent({
      kind: "attachment_removed",
      meeting_id: MEETING,
      attachment_id: "att-1",
    });
    expect(useAttachmentsStore.getState().attachments).toHaveLength(1);
    expect(useAttachmentsStore.getState().attachments[0].id).toBe("att-2");
  });

  it("remove drops the row optimistically and calls remove_attachment", async () => {
    useAttachmentsStore.setState({
      meetingId: MEETING,
      attachments: [entry({ id: "att-1" }), entry({ id: "att-2" })],
    });
    await useAttachmentsStore.getState().remove(MEETING, "att-1");
    expect(removeAttachment).toHaveBeenCalledWith(MEETING, "att-1");
    expect(useAttachmentsStore.getState().attachments.map((a) => a.id)).toEqual([
      "att-2",
    ]);
  });

  it("remove rolls back the row when remove_attachment rejects", async () => {
    const rows = [entry({ id: "att-1" }), entry({ id: "att-2" })];
    useAttachmentsStore.setState({ meetingId: MEETING, attachments: rows });
    vi.mocked(removeAttachment).mockRejectedValueOnce(new Error("locked"));

    await useAttachmentsStore.getState().remove(MEETING, "att-1");

    expect(useAttachmentsStore.getState().attachments).toHaveLength(2);
    expect(useAttachmentsStore.getState().lastError).toBe("locked");
  });

  it("open routes through openAttachment", async () => {
    const e = entry({ id: "att-1" });
    await useAttachmentsStore.getState().open(MEETING, e);
    expect(openAttachment).toHaveBeenCalledWith(MEETING, e);
  });
});
