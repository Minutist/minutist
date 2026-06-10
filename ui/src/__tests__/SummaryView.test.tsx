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
import { useModelsStore } from "../state/models";
import { useOperationProgressStore } from "../state/operation-progress";
import { summariseMeeting, getSummary, saveSummary } from "../ipc/summary";
import type { ModelStatus } from "../ipc/bindings";

const MEETING = "meeting-0001";
const LLM_ID = "gemma-4-e4b-it-q4_k_m";

function llmModel(status: ModelStatus["status"]): ModelStatus {
  return {
    id: LLM_ID,
    kind: "llm",
    display_name: "Gemma 4 E4B",
    status,
    license: "apache-2.0",
  };
}

function resetStore() {
  act(() => {
    useSummaryStore.setState({
      summaryMarkdown: null,
      summarising: false,
      meetingId: null,
      lastError: null,
      editing: false,
      editDraft: "",
      editMeetingId: null,
    });
    useModelsStore.setState({
      models: [],
      isAsrModelReady: true,
      downloadInProgress: {},
      downloadErrors: {},
    });
    useOperationProgressStore.setState({ operations: {} });
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

  it("keeps an in-progress edit draft when the pane is hidden and reshown", async () => {
    // Regression for the summary-pane unmount data-loss path: the draft lives in
    // the store, so hiding the column (which unmounts SummaryView) and reshowing
    // it must restore the in-progress edit rather than discard it.
    vi.mocked(getSummary).mockResolvedValue("original summary");
    const view = render(<SummaryView meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByText("original summary")).toBeInTheDocument(),
    );

    act(() => fireEvent.click(screen.getByRole("button", { name: "Edit" })));
    act(() =>
      fireEvent.change(screen.getByLabelText("Edit summary markdown"), {
        target: { value: "work in progress" },
      }),
    );

    // Hide the pane → unmount.
    act(() => view.unmount());

    // Reshow the pane → remount; the draft survives in the store.
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalled());
    const restored = screen.getByLabelText(
      "Edit summary markdown",
    ) as HTMLTextAreaElement;
    expect(restored.value).toBe("work in progress");
  });

  it("scopes the edit draft to its meeting (a draft for A is not shown for B)", async () => {
    // beginEdit stores editMeetingId; a different open meeting must not inherit
    // the draft / edit mode.
    vi.mocked(getSummary).mockResolvedValue("summary A");
    const view = render(<SummaryView meetingId={MEETING} />);
    await waitFor(() =>
      expect(screen.getByText("summary A")).toBeInTheDocument(),
    );
    act(() => fireEvent.click(screen.getByRole("button", { name: "Edit" })));
    act(() => view.unmount());

    // Open a different meeting: no textarea (not in edit mode for meeting B).
    vi.mocked(getSummary).mockResolvedValue("summary B");
    render(<SummaryView meetingId="meeting-0002" />);
    await waitFor(() =>
      expect(screen.getByText("summary B")).toBeInTheDocument(),
    );
    expect(
      screen.queryByLabelText("Edit summary markdown"),
    ).not.toBeInTheDocument();
  });

  it("shows the model-download phase (not 'Summarising') while the LLM is fetched", async () => {
    vi.mocked(getSummary).mockResolvedValue(null);
    act(() => {
      useSummaryStore.setState({ summarising: true, meetingId: MEETING });
      useModelsStore.setState({
        models: [
          llmModel({ state: "missing", bytes_present: 0, bytes_total: 5_000_000 }),
        ],
        downloadInProgress: {
          [LLM_ID]: { bytes_done: 2_500_000, bytes_total: 5_000_000 },
        },
      });
    });

    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    expect(
      screen.getByText(/Downloading the summarisation model/i),
    ).toHaveTextContent("50%");
    expect(
      screen.getByRole("button", { name: "Downloading model…" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Generating summary/i)).not.toBeInTheDocument();
  });

  it("shows 'Summarising' once the LLM is available", async () => {
    vi.mocked(getSummary).mockResolvedValue(null);
    act(() => {
      useSummaryStore.setState({ summarising: true, meetingId: MEETING });
      useModelsStore.setState({
        models: [llmModel({ state: "available", local_dir: "/models/llm" })],
      });
    });

    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    expect(screen.getByText(/Generating summary/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Summarising…" }),
    ).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // #68(b) — when the pane opens while a summarise is in flight for the open
  // meeting (op == "summarise" in the operation-progress store, e.g. the
  // post-stop auto-summarise), the determinate OperationIndicator bar shows even
  // though THIS pane did not dispatch the summarise (store `summarising` false).
  // -------------------------------------------------------------------------

  it("shows the determinate summarise progress bar when a summarise is in flight on open", async () => {
    vi.mocked(getSummary).mockResolvedValue(null);
    act(() => {
      // The pane did NOT dispatch a summarise (the auto-summarise chain did);
      // the store's `summarising` flag is false, but a `summarise` op is in
      // flight for THIS meeting in the operation-progress store.
      useSummaryStore.setState({ summarising: false, meetingId: null });
      useOperationProgressStore.setState({
        operations: {
          [MEETING]: { op: "summarise", fraction: 0.42, label: "Summarising…" },
        },
      });
    });

    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));

    // The determinate bar (from the shared OperationIndicator) is shown at 42%.
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "42");
    // And the manual Summarise button is disabled + labelled "Summarising…" while
    // the background auto-summarise is in flight (W1 — no double-summarise).
    const summariseBtn = screen.getByRole("button", { name: /Summarising…/ });
    expect(summariseBtn).toBeDisabled();
  });

  it("does not show the summarise progress bar when no summarise op is in flight", async () => {
    vi.mocked(getSummary).mockResolvedValue(null);
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("does not show the bar for a summarise op belonging to a different meeting", async () => {
    vi.mocked(getSummary).mockResolvedValue(null);
    act(() => {
      useOperationProgressStore.setState({
        operations: {
          "other-meeting": {
            op: "summarise",
            fraction: 0.5,
            label: "Summarising…",
          },
        },
      });
    });
    render(<SummaryView meetingId={MEETING} />);
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith(MEETING));
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});
