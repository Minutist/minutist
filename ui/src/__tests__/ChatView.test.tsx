/**
 * Tests for the chat pane (Phase 9).
 *
 * Asserts the pane's rendering + interaction:
 *   - the user / assistant messages render as bubbles, assistant markdown is
 *     rendered,
 *   - typing + Enter sends through the store's `send` (→ the `sendChatMessage`
 *     seam) and the user message shows,
 *   - the send control is disabled while a turn is in flight,
 *   - a streamed token shows with a caret; `chat_turn_complete` finalises it,
 *   - the tool-activity row shows while a tool runs,
 *   - the session switcher lists sessions and the error state surfaces.
 *
 * The chat IPC calls are mocked at the `../ipc/chat` seam (per the architecture
 * testing policy — the seam is mocked, not the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), Channel: vi.fn() }));

vi.mock("../ipc/chat", () => ({
  sendChatMessage: vi.fn().mockResolvedValue("session-1"),
  cancelChatTurn: vi.fn().mockResolvedValue(undefined),
  getChatSession: vi.fn().mockResolvedValue(null),
  listChatSessions: vi.fn().mockResolvedValue([]),
  deleteChatSession: vi.fn().mockResolvedValue(undefined),
}));

import { ChatView, renderChatMarkdown } from "../shell/ChatView";
import { useChatStore } from "../state/chat";
import { sendChatMessage, cancelChatTurn, listChatSessions } from "../ipc/chat";
import type { ChatSession } from "../ipc/chat";

const MEETING = "meeting-0001";
const SESSION = "session-1";

function resetStore() {
  act(() => {
    useChatStore.setState({
      meetingId: null,
      sessionId: null,
      sessions: [],
      messages: [],
      streaming: null,
      inFlight: false,
      toolActivity: null,
      lastError: null,
      historyTrimmed: false,
    });
  });
}

describe("renderChatMarkdown", () => {
  it("renders markdown lists to HTML", () => {
    const html = renderChatMarkdown("- one\n- two");
    expect(html).toContain("<li>one</li>");
  });
});

describe("ChatView (Phase 9)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("renders the empty prompt when there is no conversation", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalledWith(MEETING));
    expect(screen.getByText(/Ask about this meeting/i)).toBeInTheDocument();
  });

  it("renders user and assistant bubbles (assistant markdown rendered)", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());
    act(() => {
      useChatStore.setState({
        sessionId: SESSION,
        messages: [
          { role: "user", content: "Hi there", tool_calls: [], turn_id: 0 },
          {
            role: "assistant",
            content: "**Bold** answer",
            tool_calls: [],
            turn_id: 0,
          },
        ],
      });
    });
    expect(screen.getByText("Hi there")).toBeInTheDocument();
    expect(screen.getByText("Bold").tagName.toLowerCase()).toBe("strong");
  });

  it("typing + Enter sends through the seam and shows the user message", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());

    const input = screen.getByLabelText("Message the meeting agent");
    act(() => {
      fireEvent.change(input, { target: { value: "What was decided?" } });
      fireEvent.keyDown(input, { key: "Enter" });
    });

    await waitFor(() =>
      expect(sendChatMessage).toHaveBeenCalledWith(
        MEETING,
        null,
        "What was decided?",
      ),
    );
    expect(screen.getByText("What was decided?")).toBeInTheDocument();
  });

  it("Shift+Enter does not send", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());
    const input = screen.getByLabelText("Message the meeting agent");
    act(() => {
      fireEvent.change(input, { target: { value: "line one" } });
      fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    });
    expect(sendChatMessage).not.toHaveBeenCalled();
  });

  it("replaces Send with a Stop control while a turn is in flight, and cancels", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());
    act(() => {
      useChatStore.setState({ sessionId: SESSION, inFlight: true });
    });
    // The Send control is gone; a Stop control is shown (P1).
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull();
    const stop = screen.getByRole("button", { name: "Stop" });
    await act(async () => {
      fireEvent.click(stop);
    });
    // Cancelling raises the backend flag and clears the in-flight state.
    expect(cancelChatTurn).toHaveBeenCalledWith(SESSION);
    await waitFor(() => expect(useChatStore.getState().inFlight).toBe(false));
  });

  it("surfaces the history-trimmed notice on chat_context_trimmed", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());
    act(() => {
      useChatStore.setState({ sessionId: SESSION });
    });
    act(() => {
      useChatStore.getState().handleEvent({
        kind: "chat_context_trimmed",
        session_id: SESSION,
        dropped_turns: 2,
      });
    });
    expect(screen.getByText(/Older messages were trimmed/i)).toBeInTheDocument();
  });

  it("shows the streamed text with a caret, then the finalised assistant message", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());
    act(() => {
      useChatStore.setState({ sessionId: SESSION });
    });

    act(() => {
      useChatStore.getState().handleEvent({
        kind: "chat_token",
        session_id: SESSION,
        turn_id: 1,
        token: "streaming partial",
      });
    });
    expect(screen.getByText("streaming partial")).toBeInTheDocument();

    act(() => {
      useChatStore.getState().handleEvent({
        kind: "chat_turn_complete",
        session_id: SESSION,
        turn_id: 1,
        final_text: "the final reconciled answer",
      });
    });
    expect(
      screen.getByText("the final reconciled answer"),
    ).toBeInTheDocument();
    // The streamed partial is gone (replaced by the authoritative message).
    expect(screen.queryByText("streaming partial")).not.toBeInTheDocument();
  });

  it("shows the tool-activity row while a tool runs", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());
    act(() => {
      useChatStore.setState({ sessionId: SESSION });
      useChatStore.getState().handleEvent({
        kind: "chat_tool_call",
        session_id: SESSION,
        turn_id: 1,
        tool: "get_transcript",
        args_json: "{}",
      });
    });
    expect(screen.getByText("get_transcript")).toBeInTheDocument();
    expect(screen.getByText("running…")).toBeInTheDocument();
  });

  it("lists sessions in the switcher", async () => {
    const sample: ChatSession = {
      id: SESSION,
      meeting_id: MEETING,
      title: "Action items",
      messages: [],
      created_at: "2026-06-10T00:00:00Z",
      updated_at: "2026-06-10T00:00:00Z",
    };
    vi.mocked(listChatSessions).mockResolvedValue([sample]);
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalledWith(MEETING));
    await waitFor(() =>
      expect(
        screen.getByRole("option", { name: "Action items" }),
      ).toBeInTheDocument(),
    );
  });

  it("surfaces the error state", async () => {
    render(<ChatView meetingId={MEETING} />);
    await waitFor(() => expect(listChatSessions).toHaveBeenCalled());
    act(() => {
      useChatStore.setState({ lastError: "the model crashed" });
    });
    expect(screen.getByRole("alert")).toHaveTextContent("the model crashed");
  });
});
