/**
 * Tests for the summary view (Phase 5, FR-30).
 *
 * Asserts:
 *   - clicking Summarise invokes `summarise_meeting` and enters the in-progress
 *     state (the button shows a progress label and is disabled),
 *   - the rendered `summary.md` markdown shows on the reading sheet,
 *   - Edit → Save persists the edited markdown via `save_summary`.
 *
 * The IPC calls are mocked at the `../ipc/summary` seam (per the architecture
 * testing policy — the seam is mocked, not the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

vi.mock("../ipc/summary", () => ({
  summariseMeeting: vi.fn().mockResolvedValue(undefined),
  getSummary: vi.fn().mockResolvedValue(null),
  saveSummary: vi.fn().mockResolvedValue(undefined),
}));

import { SummaryView, renderSummaryMarkdown } from "../shell/SummaryView";
import { useSummaryStore } from "../state/summary";
import { summariseMeeting, getSummary, saveSummary } from "../ipc/summary";

const MEETING = "meeting-0001";

function resetStore() {
  act(() => {
    useSummaryStore.setState({
      summaryMarkdown: null,
      summarising: false,
      meetingId: null,
      lastError: null,
    });
  });
}

describe("renderSummaryMarkdown", () => {
  it("renders markdown headings and lists to HTML", () => {
    const html = renderSummaryMarkdown("## Decisions\n\n- one\n- two");
    expect(html).toContain("<h2>Decisions</h2>");
    expect(html).toContain("<li>one</li>");
  });
});

describe("SummaryView (FR-30)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("renders an empty state when there is no summary", async () => {
    vi.mocked(getSummary).mockResolvedValue(null);
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));
    expect(screen.getByText(/No summary yet/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Summarise" }),
    ).toBeInTheDocument();
  });

  it("clicking Summarise invokes the command and enters in-progress", async () => {
    let resolveCall: () => void = () => {};
    vi.mocked(summariseMeeting).mockReturnValue(
      new Promise<void>((res) => {
        resolveCall = res;
      }),
    );
    vi.mocked(getSummary).mockResolvedValue(null);

    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalled());

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Summarise" }));
    });

    await waitFor(() =>
      expect(summariseMeeting).toHaveBeenCalledWith(MEETING),
    );

    // In-progress affordance: the action shows a progress label and is disabled.
    const button = screen.getByRole("button", { name: "Summarising…" });
    expect(button).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(/Generating summary/i);

    act(() => resolveCall());
  });

  it("renders the summary markdown on the reading sheet", async () => {
    vi.mocked(getSummary).mockResolvedValue("## Summary\n\nKey outcome.");
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByText("Summary")).toBeInTheDocument(),
    );
    // The heading from the markdown body is rendered (distinct from the
    // view's own "Summary" header heading).
    await waitFor(() =>
      expect(screen.getByText("Key outcome.")).toBeInTheDocument(),
    );
  });

  it("Edit then Save persists the edited markdown via save_summary", async () => {
    vi.mocked(getSummary).mockResolvedValue("original summary");
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByText("original summary")).toBeInTheDocument(),
    );

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    });

    const textarea = screen.getByLabelText(
      "Edit summary markdown",
    ) as HTMLTextAreaElement;
    expect(textarea.value).toBe("original summary");

    act(() => {
      fireEvent.change(textarea, { target: { value: "revised summary" } });
      fireEvent.click(screen.getByRole("button", { name: "Save" }));
    });

    await waitFor(() =>
      expect(saveSummary).toHaveBeenCalledWith(MEETING, "revised summary"),
    );
  });
});
