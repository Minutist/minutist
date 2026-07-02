/**
 * Live co-pilot store (U4).
 *
 * Verifies:
 * - `messagesFor` returns an empty array for a meeting with no events.
 * - `hasMessages` returns false before any event arrives.
 * - `live_copilot_message` appends to the list in arrival order.
 * - `hasMessages` flips to true after the first message.
 * - a second message is appended (not overwritten).
 * - messages for meeting A do not affect meeting B.
 * - an unrelated event kind falls through without modifying state.
 */
import { describe, it, expect, beforeEach } from "vitest";

import { useLiveCopilotStore } from "../state/liveCopilot";
import type { AppEvent } from "../ipc/bindings";

const M1 = "meeting-0001";
const M2 = "meeting-0002";

function makeEvent(
  meeting_id: string,
  turn_id: number,
  content: string,
): AppEvent {
  return {
    kind: "live_copilot_message",
    meeting_id,
    turn_id,
    role: "assistant",
    content,
  } as AppEvent;
}

describe("live-copilot store", () => {
  beforeEach(() => {
    useLiveCopilotStore.setState({ messages: new Map() });
  });

  it("messagesFor returns empty array for a meeting with no events", () => {
    expect(useLiveCopilotStore.getState().messagesFor(M1)).toEqual([]);
  });

  it("hasMessages returns false before any event arrives", () => {
    expect(useLiveCopilotStore.getState().hasMessages(M1)).toBe(false);
  });

  it("live_copilot_message appends to the list", () => {
    useLiveCopilotStore.getState().handleEvent(makeEvent(M1, 1, "First note."));

    const msgs = useLiveCopilotStore.getState().messagesFor(M1);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].content).toBe("First note.");
    expect(msgs[0].turn_id).toBe(1);
    expect(msgs[0].role).toBe("assistant");
  });

  it("hasMessages flips to true after the first message", () => {
    useLiveCopilotStore.getState().handleEvent(makeEvent(M1, 1, "Hello."));
    expect(useLiveCopilotStore.getState().hasMessages(M1)).toBe(true);
  });

  it("a second message is appended in arrival order", () => {
    const store = useLiveCopilotStore.getState();
    store.handleEvent(makeEvent(M1, 1, "First."));
    store.handleEvent(makeEvent(M1, 2, "Second."));

    const msgs = useLiveCopilotStore.getState().messagesFor(M1);
    expect(msgs).toHaveLength(2);
    expect(msgs[0].content).toBe("First.");
    expect(msgs[1].content).toBe("Second.");
  });

  it("messages for meeting A do not affect meeting B", () => {
    useLiveCopilotStore.getState().handleEvent(makeEvent(M1, 1, "A note."));

    expect(useLiveCopilotStore.getState().messagesFor(M2)).toEqual([]);
    expect(useLiveCopilotStore.getState().hasMessages(M2)).toBe(false);
  });

  it("an unrelated event kind does not modify state", () => {
    useLiveCopilotStore.getState().handleEvent({
      kind: "summary_ready",
      meeting_id: M1,
    } as AppEvent);

    expect(useLiveCopilotStore.getState().messagesFor(M1)).toEqual([]);
  });
});
