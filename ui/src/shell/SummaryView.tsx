/**
 * Summary view (Phase 5, FR-30) — the meeting summary surface.
 *
 * Editorial Ink language: the rendered summary reads as a sheet of warm paper
 * (Fraunces headings, Newsreader body, one oxblood accent on the primary
 * action). It renders the `summary.md` markdown for reading, lets the user edit
 * the raw markdown and persist it, and exposes a Summarise action plus an
 * in-progress state while the LLM runs.
 *
 * All mutations route through `useSummaryStore` (which wraps the `../ipc/summary`
 * seam); the component holds only local editor draft state. It consumes
 * `theme.css` tokens only and renders in the DEV shim with sample summary data.
 */
import { useEffect, useMemo } from "react";
import MarkdownIt from "markdown-it";
import { useSummaryStore } from "../state/summary";
import { useModelsStore } from "../state/models";
import type { MeetingId } from "../ipc/bindings";
import "./SummaryView.css";

// A single shared renderer — markdown-only, no raw HTML (the summary is
// model-generated markdown; `html: false` keeps it from injecting markup).
const md = new MarkdownIt({ html: false, linkify: true, typographer: true });

/** Render summary markdown to a sanitised-by-construction HTML string. */
export function renderSummaryMarkdown(markdown: string): string {
  return md.render(markdown);
}

type SummaryViewProps = {
  /** The meeting whose summary is shown. */
  meetingId: MeetingId;
};

export function SummaryView({ meetingId }: SummaryViewProps) {
  const summaryMarkdown = useSummaryStore((s) => s.summaryMarkdown);
  const summarising = useSummaryStore((s) => s.summarising);
  const lastError = useSummaryStore((s) => s.lastError);
  const read = useSummaryStore((s) => s.read);
  const summarise = useSummaryStore((s) => s.summarise);
  const save = useSummaryStore((s) => s.save);
  // Edit state lives in the store (not local useState) so an in-progress draft
  // survives this pane being hidden/unmounted; `editMeetingId` scopes it to its
  // meeting so a draft for one meeting is not shown when another is open.
  const editingFlag = useSummaryStore((s) => s.editing);
  const editDraft = useSummaryStore((s) => s.editDraft);
  const editMeetingId = useSummaryStore((s) => s.editMeetingId);
  const beginEditAction = useSummaryStore((s) => s.beginEdit);
  const setDraft = useSummaryStore((s) => s.setDraft);
  const endEdit = useSummaryStore((s) => s.endEdit);

  // Summarisation needs the LLM; on first use the orchestrator downloads it
  // (multi-GB) before any text is generated. Surface THAT phase distinctly so a
  // long wait does not masquerade as "Summarising…" (which reads as broken).
  const models = useModelsStore((s) => s.models);
  const downloadInProgress = useModelsStore((s) => s.downloadInProgress);
  const llm = models.find((m) => m.kind === "llm");
  const llmReady = llm?.status.state === "available";
  // Only claim the download phase with positive evidence: the LLM is known and
  // not yet available. If the model list isn't loaded (llm undefined), fall back
  // to the plain "Summarising…" state rather than mislabelling.
  const downloadingModel = summarising && llm !== undefined && !llmReady;
  let downloadPct: number | null = null;
  const llmProgress = llm ? downloadInProgress[llm.id] : undefined;
  if (llmProgress && llmProgress.bytes_total > 0) {
    downloadPct = Math.round((100 * llmProgress.bytes_done) / llmProgress.bytes_total);
  } else if (llm?.status.state === "downloading" && llm.status.bytes_total > 0) {
    downloadPct = Math.round((100 * llm.status.bytes_done) / llm.status.bytes_total);
  }

  // Edit mode applies only when the stored draft belongs to THIS meeting.
  const editing = editingFlag && editMeetingId === meetingId;

  // Load the persisted summary when the open meeting changes.
  useEffect(() => {
    void read(meetingId);
  }, [meetingId, read]);

  const renderedHtml = useMemo(
    () => (summaryMarkdown ? renderSummaryMarkdown(summaryMarkdown) : ""),
    [summaryMarkdown],
  );

  function beginEdit() {
    beginEditAction(meetingId, summaryMarkdown ?? "");
  }

  function cancelEdit() {
    endEdit();
  }

  function commitEdit() {
    // Read the live draft from the store (not the render closure) so the save
    // captures the latest keystroke regardless of render timing.
    void save(meetingId, useSummaryStore.getState().editDraft);
    endEdit();
  }

  const hasSummary = summaryMarkdown !== null && summaryMarkdown.trim() !== "";

  return (
    <section className="summary-view ink-reveal" aria-label="Meeting summary">
      <header className="summary-view__header">
        <h2 className="summary-view__heading">Summary</h2>
        <div className="summary-view__actions">
          {editing ? (
            <>
              <button
                type="button"
                className="summary-view__action summary-view__action--primary"
                onClick={commitEdit}
              >
                Save
              </button>
              <button
                type="button"
                className="summary-view__action"
                onClick={cancelEdit}
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className="summary-view__action summary-view__action--primary"
                onClick={() => void summarise(meetingId)}
                disabled={summarising}
              >
                {summarising
                  ? downloadingModel
                    ? "Downloading model…"
                    : "Summarising…"
                  : hasSummary
                    ? "Re-summarise"
                    : "Summarise"}
              </button>
              {hasSummary && (
                <button
                  type="button"
                  className="summary-view__action"
                  onClick={beginEdit}
                  disabled={summarising}
                >
                  Edit
                </button>
              )}
            </>
          )}
        </div>
      </header>

      {lastError && (
        <p className="summary-view__error" role="alert">
          {lastError}
        </p>
      )}

      {summarising && !editing && (
        <p className="summary-view__status" role="status">
          <span className="summary-view__spinner" aria-hidden="true" />
          {downloadingModel
            ? `Downloading the summarisation model (one-time, ~5 GB)${
                downloadPct !== null ? ` — ${downloadPct}%` : "…"
              }`
            : "Generating summary from the transcript and your notes…"}
        </p>
      )}

      {editing ? (
        <textarea
          className="summary-view__editor"
          aria-label="Edit summary markdown"
          value={editDraft}
          onChange={(e) => setDraft(e.target.value)}
          autoFocus
        />
      ) : hasSummary ? (
        <article
          className="summary-view__sheet"
          // markdown-it output with `html: false`; user/model markdown only.
          dangerouslySetInnerHTML={{ __html: renderedHtml }}
        />
      ) : (
        !summarising && (
          <p className="summary-view__empty">
            No summary yet. Run Summarise to generate one from the transcript and
            your notes.
          </p>
        )
      )}
    </section>
  );
}
