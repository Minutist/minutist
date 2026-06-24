/**
 * Tests for the live digest panel (Phase 9 — S3).
 *
 * Asserts:
 * - the waiting state renders when no digest has arrived.
 * - action items, decisions, open asks, attachment answers, and unresolved
 *   references render from a sample digest payload.
 * - the resolved/open state indicator is shown for open_asks items.
 * - resolved open_asks items receive the resolved CSS class.
 * - the last-updated timestamp renders in the header.
 * - a `live_digest_error` shows the error message.
 * - a `live_digest_error` with a prior digest retains the digest items.
 * - per-category visibility props hide/show the section when empty content is
 *   not the reason (tested by toggling showActionItems with items present).
 * - the `source` badge renders on attachment-answer items.
 * - the empty-categories message renders when all categories have no items.
 *
 * No IPC mocks needed — the panel is event-driven and has no IPC seam.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";

import { LiveDigestPanel } from "../shell/LiveDigestPanel";
import { useLiveDigestStore } from "../state/liveDigest";
import type { LiveDigest } from "../ipc/bindings";

const MEETING = "meeting-0001";

function makeDigest(overrides: Partial<LiveDigest> = {}): LiveDigest {
  return {
    meeting_id: MEETING,
    generated_at_ms: 1_750_000_000_000,
    action_items: [],
    decisions: [],
    open_asks: [],
    attachment_answers: [],
    unresolved_references: [],
    ...overrides,
  };
}

/** Render LiveDigestPanel with all categories enabled by default. */
function renderPanel(props: Partial<Parameters<typeof LiveDigestPanel>[0]> = {}) {
  return render(
    <LiveDigestPanel
      meetingId={MEETING}
      showActionItems={true}
      showDecisions={true}
      showOpenAsks={true}
      showAttachmentAnswers={true}
      showUnresolvedReferences={true}
      {...props}
    />,
  );
}

describe("LiveDigestPanel", () => {
  beforeEach(() => {
    useLiveDigestStore.setState({ digests: {} });
  });

  it("shows the waiting state when no digest has arrived", () => {
    renderPanel();
    expect(screen.getByText(/Waiting for the first digest/i)).toBeInTheDocument();
  });

  it("renders action items from a digest payload", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest({
          action_items: [{ text: "Follow up with Alice", resolved: false }],
        }),
      });
    });
    renderPanel();
    expect(screen.getByText("Follow up with Alice")).toBeInTheDocument();
    expect(screen.getByText("Action Items")).toBeInTheDocument();
  });

  it("renders decisions", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest({
          decisions: [{ text: "Adopt TypeScript", resolved: false }],
        }),
      });
    });
    renderPanel();
    expect(screen.getByText("Adopt TypeScript")).toBeInTheDocument();
    expect(screen.getByText("Decisions")).toBeInTheDocument();
  });

  it("renders open asks with resolved/open indicators", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest({
          open_asks: [
            { text: "What is the deadline?", resolved: false },
            { text: "Who leads this?", resolved: true },
          ],
        }),
      });
    });
    renderPanel();
    expect(screen.getByText("What is the deadline?")).toBeInTheDocument();
    expect(screen.getByText("Who leads this?")).toBeInTheDocument();
    // Status indicators for open asks.
    const openIndicators = screen.getAllByLabelText("Open");
    expect(openIndicators).toHaveLength(1);
    const resolvedIndicators = screen.getAllByLabelText("Resolved");
    expect(resolvedIndicators).toHaveLength(1);
  });

  it("resolved items receive the resolved CSS modifier class", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest({
          open_asks: [{ text: "Resolved ask", resolved: true }],
        }),
      });
    });
    renderPanel();
    const item = screen.getByText("Resolved ask").closest("li");
    expect(item).toHaveClass("live-digest__item--resolved");
  });

  it("renders the last-updated timestamp in the header", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest(),
      });
    });
    renderPanel();
    // The timestamp element is present (exact formatting is locale-dependent).
    expect(
      screen.getByRole("banner").querySelector(".live-digest__timestamp"),
    ).toBeInTheDocument();
  });

  it("renders the source badge on attachment-answer items", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest({
          attachment_answers: [
            { text: "Covered in slide 4", resolved: false, source: "deck.pdf" },
          ],
        }),
      });
    });
    renderPanel();
    expect(screen.getByText("Covered in slide 4")).toBeInTheDocument();
    expect(screen.getByText("deck.pdf")).toBeInTheDocument();
  });

  it("shows the error message from live_digest_error", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_error",
        meeting_id: MEETING,
        message: "context overflow",
      });
    });
    renderPanel();
    expect(screen.getByRole("alert")).toHaveTextContent("context overflow");
  });

  it("retains prior digest items when a live_digest_error follows a valid digest", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest({
          decisions: [{ text: "Use Rust", resolved: false }],
        }),
      });
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_error",
        meeting_id: MEETING,
        message: "temporary failure",
      });
    });
    renderPanel();
    // The error is shown.
    expect(screen.getByRole("alert")).toHaveTextContent("temporary failure");
    // The prior decision is still visible.
    expect(screen.getByText("Use Rust")).toBeInTheDocument();
  });

  it("hides a category section when its visibility prop is false", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest({
          action_items: [{ text: "Hidden action", resolved: false }],
        }),
      });
    });
    renderPanel({ showActionItems: false });
    expect(screen.queryByText("Hidden action")).not.toBeInTheDocument();
    expect(screen.queryByText("Action Items")).not.toBeInTheDocument();
  });

  it("renders the empty-categories message when all visible categories have no items", () => {
    act(() => {
      useLiveDigestStore.getState().handleEvent({
        kind: "live_digest_updated",
        meeting_id: MEETING,
        digest: makeDigest(), // all categories empty
      });
    });
    renderPanel();
    expect(
      screen.getByText(/Nothing to report yet/i),
    ).toBeInTheDocument();
  });
});
