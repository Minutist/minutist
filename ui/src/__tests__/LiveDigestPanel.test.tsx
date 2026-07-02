/**
 * Tests for the repurposed live co-pilot feed panel (U4).
 *
 * The panel previously rendered structured `LiveDigest` categories; it now
 * renders transcript-driven co-pilot observations as assistant bubbles. These
 * tests replace the prior digest-rendering suite.
 *
 * Asserts:
 * - the empty state renders when no messages have arrived.
 * - the heading reads "Co-pilot".
 * - a `live_copilot_message` event causes the message content to appear.
 * - the empty state is gone once a message is shown.
 * - a second message is also rendered.
 *
 * No IPC mocks needed — the panel is event-driven and has no IPC seam.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";

import { LiveDigestPanel } from "../shell/LiveDigestPanel";
import { useLiveCopilotStore } from "../state/liveCopilot";
import type { AppEvent } from "../ipc/bindings";

const MEETING = "meeting-0001";

describe("LiveDigestPanel (co-pilot feed)", () => {
  beforeEach(() => {
    useLiveCopilotStore.setState({ messages: new Map() });
  });

  it("shows the empty state when no messages have arrived", () => {
    render(<LiveDigestPanel meetingId={MEETING} />);
    expect(
      screen.getByText(/No co-pilot notes yet/i),
    ).toBeInTheDocument();
  });

  it("renders the Co-pilot heading", () => {
    render(<LiveDigestPanel meetingId={MEETING} />);
    expect(
      screen.getByRole("heading", { name: /Co-pilot/i }),
    ).toBeInTheDocument();
  });

  it("renders message content after a live_copilot_message event", () => {
    render(<LiveDigestPanel meetingId={MEETING} />);

    act(() => {
      useLiveCopilotStore.getState().handleEvent({
        kind: "live_copilot_message",
        meeting_id: MEETING,
        turn_id: 1,
        role: "assistant",
        content: "Action item: follow up with Alice.",
      } as AppEvent);
    });

    expect(screen.getByText(/follow up with Alice/i)).toBeInTheDocument();
    expect(screen.queryByText(/No co-pilot notes yet/i)).not.toBeInTheDocument();
  });

  it("renders multiple messages in arrival order", () => {
    // Both events carry turn_id: 0 — the production shape: the backend
    // hardcodes 0 for every LiveCopilotMessage. The panel keys by array index
    // (append-only), not by turn_id, so duplicate ids do not cause collisions.
    render(<LiveDigestPanel meetingId={MEETING} />);

    act(() => {
      const store = useLiveCopilotStore.getState();
      store.handleEvent({
        kind: "live_copilot_message",
        meeting_id: MEETING,
        turn_id: 0,
        role: "assistant",
        content: "First observation.",
      } as AppEvent);
      store.handleEvent({
        kind: "live_copilot_message",
        meeting_id: MEETING,
        turn_id: 0,
        role: "assistant",
        content: "Second observation.",
      } as AppEvent);
    });

    expect(screen.getByText(/First observation/i)).toBeInTheDocument();
    expect(screen.getByText(/Second observation/i)).toBeInTheDocument();
  });
});
