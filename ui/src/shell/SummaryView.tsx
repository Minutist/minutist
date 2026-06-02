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
import { useEffect, useMemo, useState } from "react";
import MarkdownIt from "markdown-it";
import { useSummaryStore } from "../state/summary";
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

  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  // Load the persisted summary when the open meeting changes.
  useEffect(() => {
    void read(meetingId);
  }, [meetingId, read]);

  const renderedHtml = useMemo(
    () => (summaryMarkdown ? renderSummaryMarkdown(summaryMarkdown) : ""),
    [summaryMarkdown],
  );

  function beginEdit() {
    setDraft(summaryMarkdown ?? "");
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
  }

  function commitEdit() {
    setEditing(false);
    void save(meetingId, draft);
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
                  ? "Summarising…"
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
          Generating summary from the transcript and your notes…
        </p>
      )}

      {editing ? (
        <textarea
          className="summary-view__editor"
          aria-label="Edit summary markdown"
          value={draft}
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
