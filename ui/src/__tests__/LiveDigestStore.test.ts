/**
 * Live-agent driver error store.
 *
 * Verifies:
 * - `live_digest_error` stores the message per meeting id.
 * - a second `live_digest_error` for the same meeting replaces the message.
 * - `errorFor` returns null for a meeting with no error.
 * - an unrelated event kind falls through without modifying state.
 * - errors for meeting A do not affect meeting B.
 */
import { describe, it, expect, beforeEach } from "vitest";

import { useLiveDigestStore } from "../state/liveDigest";
import type { AppEvent } from "../ipc/bindings";

describe("live-agent error store", () => {
  beforeEach(() => {
    useLiveDigestStore.setState({ errors: {} });
  });

  it("errorFor returns null for a meeting with no error", () => {
    expect(useLiveDigestStore.getState().errorFor("unknown")).toBeNull();
  });

  it("live_digest_error stores the message for a meeting", () => {
    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_error",
      meeting_id: "m1",
      message: "context overflow",
    });

    expect(useLiveDigestStore.getState().errorFor("m1")).toBe(
      "context overflow",
    );
  });

  it("a second live_digest_error for the same meeting replaces the message", () => {
    const store = useLiveDigestStore.getState();
    store.handleEvent({
      kind: "live_digest_error",
      meeting_id: "m1",
      message: "first error",
    });
    store.handleEvent({
      kind: "live_digest_error",
      meeting_id: "m1",
      message: "second error",
    });

    expect(useLiveDigestStore.getState().errorFor("m1")).toBe("second error");
  });

  it("an unrelated event kind does not modify state", () => {
    useLiveDigestStore.getState().handleEvent({
      kind: "summary_ready",
      meeting_id: "m1",
    } as AppEvent);

    expect(useLiveDigestStore.getState().errorFor("m1")).toBeNull();
  });

  it("errors for meeting A do not affect meeting B", () => {
    useLiveDigestStore.setState({ errors: { m2: "prior m2 error" } });

    useLiveDigestStore.getState().handleEvent({
      kind: "live_digest_error",
      meeting_id: "m1",
      message: "m1 error",
    });

    expect(useLiveDigestStore.getState().errorFor("m2")).toBe(
      "prior m2 error",
    );
  });
});
