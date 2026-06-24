/**
 * Live digest store (Phase 9 — S3).
 *
 * Verifies:
 * - `live_digest_updated` overwrites the entry wholesale (all categories).
 * - a second `live_digest_updated` replaces the prior digest and clears the
 *   error field.
 * - `live_digest_error` stores the message and RETAINS the last valid digest.
 * - `live_digest_error` on an unseen meeting creates an entry with null digest.
 * - `digestFor` returns null for a meeting with no events.
 * - `resolved` flag is preserved as-is on open_asks items (the backend owns
 *   reconciliation; the store adopts the payload verbatim).
 * - an unrelated event kind falls through the switch without modifying state.
 * - digest events for meeting A do not affect meeting B.
 */
import { describe, it, expect, beforeEach } from "vitest";

import { useLiveDigestStore } from "../state/liveDigest";
import type { AppEvent } from "../ipc/bindings";
import type { LiveDigest } from "../ipc/bindings";

// A minimal valid LiveDigest for testing.
function makeDigest(
  meetingId = "m1",
  overrides: Partial<LiveDigest> = {},
): LiveDigest {
  return {
    meeting_id: meetingId,
    generated_at_ms: 1_000_000,
    action_items: [],
    decisions: [],
    open_asks: [],
    attachment_answers: [],
    unresolved_references: [],
    ...overrides,
  };
}

describe("live-digest store", () => {
  beforeEach(() => {
    useLiveDigestStore.setState({ digests: {} });
  });

  it("digestFor returns null for a meeting with no events", () => {
    expect(useLiveDigestStore.getState().digestFor("unknown")).toBeNull();
  });

  it("live_digest_updated overwrites the entry wholesale", () => {
    const digest = makeDigest("m1", {
      action_items: [{ text: "Follow up with Alice", resolved: false }],
      decisions: [{ text: "Use TypeScript", resolved: false }],
    });

    const event: AppEvent = {
      kind: "live_digest_updated",
      meeting_id: "m1",
      digest,
    };
    useLiveDigestStore.getState().handleEvent(event);

    const entry = useLiveDigestStore.getState().digestFor("m1");
    expect(entry).not.toBeNull();
    expect(entry!.digest).toEqual(digest);
    expect(entry!.lastError).toBeNull();
  });

  it("a second live_digest_updated replaces the prior digest", () => {
    const first = makeDigest("m1", {
      action_items: [{ text: "First action", resolved: false }],
      generated_at_ms: 1000,
    });
    const second = makeDigest("m1", {
      action_items: [{ text: "Second action", resolved: false }],
      generated_at_ms: 2000,
    });

    const store = useLiveDigestStore.getState();
    store.handleEvent({
      kind: "live_digest_updated",
      meeting_id: "m1",
      digest: first,
    });
    store.handleEvent({
      kind: "live_digest_updated",
      meeting_id: "m1",
      digest: second,
    });

    const entry = useLiveDigestStore.getState().digestFor("m1");
    expect(entry!.digest!.generated_at_ms).toBe(2000);
    expect(entry!.digest!.action_items[0].text).toBe("Second action");
  });

  it("live_digest_updated clears a prior error for the same meeting", () => {
    // Seed an error first.
    useLiveDigestStore.setState({
      digests: { m1: { digest: null, lastError: "model timeout" } },
    });

    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_updated",
      meeting_id: "m1",
      digest: makeDigest("m1"),
    });

    expect(useLiveDigestStore.getState().digestFor("m1")!.lastError).toBeNull();
  });

  it("live_digest_error stores the message and retains the last valid digest", () => {
    const digest = makeDigest("m1");
    useLiveDigestStore.setState({
      digests: { m1: { digest, lastError: null } },
    });

    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_error",
      meeting_id: "m1",
      message: "context overflow",
    });

    const entry = useLiveDigestStore.getState().digestFor("m1");
    expect(entry!.lastError).toBe("context overflow");
    // Digest is retained, not cleared.
    expect(entry!.digest).toEqual(digest);
  });

  it("live_digest_error on an unseen meeting creates an entry with null digest", () => {
    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_error",
      meeting_id: "m2",
      message: "load failed",
    });

    const entry = useLiveDigestStore.getState().digestFor("m2");
    expect(entry!.digest).toBeNull();
    expect(entry!.lastError).toBe("load failed");
  });

  it("resolved flag is adopted verbatim from the payload (open_asks)", () => {
    const digest = makeDigest("m1", {
      open_asks: [
        { text: "What is the timeline?", resolved: false },
        { text: "Who owns the action?", resolved: true },
      ],
    });

    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_updated",
      meeting_id: "m1",
      digest,
    });

    const openAsks =
      useLiveDigestStore.getState().digestFor("m1")!.digest!.open_asks;
    expect(openAsks[0].resolved).toBe(false);
    expect(openAsks[1].resolved).toBe(true);
  });

  it("all digest categories are stored correctly in one payload", () => {
    const digest = makeDigest("m1", {
      action_items: [{ text: "Action A", resolved: false }],
      decisions: [{ text: "Decision D", resolved: false }],
      open_asks: [{ text: "Ask Q?", resolved: false }],
      attachment_answers: [
        { text: "Answer from slide 3", resolved: false, source: "deck.pdf" },
      ],
      unresolved_references: [
        { text: "GDPR Article 17", resolved: false },
      ],
    });

    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_updated",
      meeting_id: "m1",
      digest,
    });

    const stored = useLiveDigestStore.getState().digestFor("m1")!.digest!;
    expect(stored.action_items).toHaveLength(1);
    expect(stored.decisions).toHaveLength(1);
    expect(stored.open_asks).toHaveLength(1);
    expect(stored.attachment_answers).toHaveLength(1);
    expect(stored.attachment_answers[0].source).toBe("deck.pdf");
    expect(stored.unresolved_references).toHaveLength(1);
  });

  it("an unrelated event kind does not modify state", () => {
    useLiveDigestStore.setState({ digests: {} });

    useLiveDigestStore.getState().handleEvent({
      kind: "summary_ready",
      meeting_id: "m1",
    } as AppEvent);

    // No entry created for m1.
    expect(useLiveDigestStore.getState().digestFor("m1")).toBeNull();
  });

  it("events for meeting A do not affect meeting B", () => {
    const digestA = makeDigest("m1", {
      action_items: [{ text: "A's action", resolved: false }],
    });
    useLiveDigestStore.setState({
      digests: {
        m1: { digest: digestA, lastError: null },
        m2: { digest: makeDigest("m2"), lastError: null },
      },
    });

    // Error for m1 must not touch m2.
    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_error",
      meeting_id: "m1",
      message: "oops",
    });

    expect(useLiveDigestStore.getState().digestFor("m2")!.lastError).toBeNull();
    expect(useLiveDigestStore.getState().digestFor("m2")!.digest).toEqual(
      makeDigest("m2"),
    );
  });
});
