/**
 * Live digest panel (Phase 9 — S3).
 *
 * A passive workspace column that renders the current `LiveDigest` for an
 * active meeting. Updated in place each time the backend emits a
 * `live_digest_updated` event; errors are shown inline while the last valid
 * digest is retained. No user input — purely a read surface.
 *
 * Category sections (action items, decisions, open asks, attachment answers,
 * unresolved references) are shown only when the corresponding per-category
 * setting is enabled. Each `LiveDigestItem` carries a `resolved` flag that
 * drives the checked/open state indicator for open asks.
 */
import { useLiveDigestStore } from "../state/liveDigest";
import type { LiveDigest, LiveDigestItem, MeetingId } from "../ipc/bindings";
import "./LiveDigestPanel.css";

type LiveDigestPanelProps = {
  /** The meeting this panel is scoped to. */
  meetingId: MeetingId;
  /** Whether the action-items category is shown (from settings). */
  showActionItems: boolean;
  /** Whether the decisions category is shown (from settings). */
  showDecisions: boolean;
  /** Whether the open-asks category is shown (from settings). */
  showOpenAsks: boolean;
  /** Whether the attachment-answers category is shown (from settings). */
  showAttachmentAnswers: boolean;
  /** Whether the unresolved-references category is shown (from settings). */
  showUnresolvedReferences: boolean;
};

/** Render a single digest item row. */
function DigestItem({
  item,
  showResolved,
}: {
  item: LiveDigestItem;
  showResolved: boolean;
}) {
  return (
    <li
      className={
        "live-digest__item" +
        (item.resolved ? " live-digest__item--resolved" : "")
      }
    >
      {showResolved && (
        <span
          className={
            "live-digest__item-status" +
            (item.resolved
              ? " live-digest__item-status--resolved"
              : " live-digest__item-status--open")
          }
          aria-label={item.resolved ? "Resolved" : "Open"}
        />
      )}
      <span className="live-digest__item-text">{item.text}</span>
      {item.source && (
        <span className="live-digest__item-source">{item.source}</span>
      )}
    </li>
  );
}

/** A labelled category section with its items. Returns null when empty. */
function DigestSection({
  label,
  items,
  showResolved = false,
}: {
  label: string;
  items: LiveDigestItem[];
  showResolved?: boolean;
}) {
  if (items.length === 0) return null;
  return (
    <section className="live-digest__section" aria-label={label}>
      <h3 className="live-digest__section-heading">{label}</h3>
      <ul className="live-digest__list" role="list">
        {items.map((item, idx) => (
          <DigestItem key={idx} item={item} showResolved={showResolved} />
        ))}
      </ul>
    </section>
  );
}

/** Format a generated_at_ms timestamp for display in the panel header. */
function formatGeneratedAt(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function DigestContent({
  digest,
  showActionItems,
  showDecisions,
  showOpenAsks,
  showAttachmentAnswers,
  showUnresolvedReferences,
}: {
  digest: LiveDigest;
  showActionItems: boolean;
  showDecisions: boolean;
  showOpenAsks: boolean;
  showAttachmentAnswers: boolean;
  showUnresolvedReferences: boolean;
}) {
  const hasContent =
    (showActionItems && digest.action_items.length > 0) ||
    (showDecisions && digest.decisions.length > 0) ||
    (showOpenAsks && digest.open_asks.length > 0) ||
    (showAttachmentAnswers && digest.attachment_answers.length > 0) ||
    (showUnresolvedReferences && digest.unresolved_references.length > 0);

  if (!hasContent) {
    return (
      <p className="live-digest__empty">
        Nothing to report yet — the digest will populate as the meeting
        progresses.
      </p>
    );
  }

  return (
    <>
      {showActionItems && (
        <DigestSection label="Action Items" items={digest.action_items} />
      )}
      {showDecisions && (
        <DigestSection label="Decisions" items={digest.decisions} />
      )}
      {showOpenAsks && (
        <DigestSection
          label="Open Questions"
          items={digest.open_asks}
          showResolved
        />
      )}
      {showAttachmentAnswers && (
        <DigestSection
          label="Attachment Answers"
          items={digest.attachment_answers}
        />
      )}
      {showUnresolvedReferences && (
        <DigestSection
          label="Unresolved References"
          items={digest.unresolved_references}
        />
      )}
    </>
  );
}

export function LiveDigestPanel({
  meetingId,
  showActionItems,
  showDecisions,
  showOpenAsks,
  showAttachmentAnswers,
  showUnresolvedReferences,
}: LiveDigestPanelProps) {
  const entry = useLiveDigestStore((s) => s.digestFor(meetingId));

  return (
    <section
      className="live-digest ink-reveal"
      aria-label="Live meeting digest"
    >
      <header className="live-digest__header">
        <h2 className="live-digest__heading">Live Digest</h2>
        {entry?.digest && (
          <span className="live-digest__timestamp">
            {formatGeneratedAt(entry.digest.generated_at_ms)}
          </span>
        )}
      </header>

      <div className="live-digest__body">
        {entry?.lastError && (
          <p className="live-digest__error" role="alert">
            {entry.lastError}
          </p>
        )}

        {entry?.digest ? (
          <DigestContent
            digest={entry.digest}
            showActionItems={showActionItems}
            showDecisions={showDecisions}
            showOpenAsks={showOpenAsks}
            showAttachmentAnswers={showAttachmentAnswers}
            showUnresolvedReferences={showUnresolvedReferences}
          />
        ) : (
          !entry?.lastError && (
            <p className="live-digest__empty">
              Waiting for the first digest…
            </p>
          )
        )}
      </div>
    </section>
  );
}
