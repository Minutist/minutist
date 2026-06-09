/**
 * Tests for the chat-agent store (Phase 9).
 *
 * Asserts the event-reconciliation contract that makes the lossy chat stream
 * safe, plus the IPC actions:
 *   - `chat_token` appends to the streaming buffer (active session only),
 *   - `chat_turn_complete.final_text` REPLACES the streamed buffer (the
 *     lossy-stream guarantee — the authoritative reply wins over accumulated
 *     tokens, even when they disagree),
 *   - `chat_tool_call` / `chat_tool_result` drive the transient tool indicator,
 *   - `chat_error` surfaces the error and clears the in-flight state,
 *   - an event for a NON-active session is ignored (per-session scoping),
 *   - `send` appends the user message optimistically + enters the in-flight
 *     state, and adopts the returned session id,
 *   - `loadSessions` / `openSession` / `deleteSession` route through the seam.
 *
 * The IPC calls are mocked at the `../ipc/chat` seam (per the architecture
 * testing policy — do not fake the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../ipc/chat", () => ({
  sendChatMessage: vi.fn().mockResolvedValue("session-1"),
  getChatSession: vi.fn().mockResolvedValue(null),
  listChatSessions: vi.fn().mockResolvedValue([]),
  deleteChatSession: vi.fn().mockResolvedValue(undefined),
}));

import {
  sendChatMessage,
  getChatSession,
  listChatSessions,
  deleteChatSession,
} from "../ipc/chat";
import type { ChatSession } from "../ipc/chat";
import { useChatStore } from "../state/chat";
import type { AppEvent } from "../ipc/app-event";

const MEETING = "meeting-0001";
const SESSION = "session-1";

function resetStore() {
  useChatStore.setState({
    meetingId: MEETING,
    sessionId: SESSION,
    sessions: [],
    messages: [],
    streaming: null,
    inFlight: false,
    toolActivity: null,
    lastError: null,
  });
}

describe("useChatStore — event reconciliation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("chat_token appends to the streaming buffer for the active session", () => {
    const store = useChatStore.getState();
    store.handleEvent({
      kind: "chat_token",
      session_id: SESSION,
      turn_id: 1,
      token: "Hello",
    });
    store.handleEvent({
      kind: "chat_token",
      session_id: SESSION,
      turn_id: 1,
      token: " world",
    });
    expect(useChatStore.getState().streaming).toBe("Hello world");
    expect(useChatStore.getState().inFlight).toBe(true);
  });

  it("chat_turn_complete REPLACES the streamed buffer with final_text (lossy-stream guarantee)", () => {
    const store = useChatStore.getState();
    // Simulate a corrupted/lossy stream: the accumulated tokens DISAGREE with
    // the authoritative final text (a dropped delta).
    store.handleEvent({
      kind: "chat_token",
      session_id: SESSION,
      turn_id: 1,
      token: "Hel wld", // corrupt partial
    });
    store.handleEvent({
      kind: "chat_turn_complete",
      session_id: SESSION,
      turn_id: 1,
      final_text: "Hello world — the full reconciled reply.",
    });

    const state = useChatStore.getState();
    // The streamed buffer is cleared; the assistant message carries final_text,
    // NOT the corrupt accumulated tokens.
    expect(state.streaming).toBeNull();
    expect(state.inFlight).toBe(false);
    const last = state.messages[state.messages.length - 1];
    expect(last.role).toBe("assistant");
    expect(last.content).toBe("Hello world — the full reconciled reply.");
    expect(last.content).not.toContain("Hel wld");
  });

  it("chat_tool_call then chat_tool_result drive the tool indicator", () => {
    const store = useChatStore.getState();
    store.handleEvent({
      kind: "chat_tool_call",
      session_id: SESSION,
      turn_id: 1,
      tool: "get_transcript",
      args_json: "{}",
    });
    expect(useChatStore.getState().toolActivity).toEqual({
      tool: "get_transcript",
      running: true,
    });

    store.handleEvent({
      kind: "chat_tool_result",
      session_id: SESSION,
      turn_id: 1,
      tool: "get_transcript",
      ok: true,
      summary: "read 42 segments",
    });
    expect(useChatStore.getState().toolActivity).toEqual({
      tool: "get_transcript",
      running: false,
      ok: true,
      summary: "read 42 segments",
    });
  });

  it("the tool indicator is cleared when the turn completes", () => {
    const store = useChatStore.getState();
    store.handleEvent({
      kind: "chat_tool_call",
      session_id: SESSION,
      turn_id: 1,
      tool: "get_summary",
      args_json: "{}",
    });
    store.handleEvent({
      kind: "chat_turn_complete",
      session_id: SESSION,
      turn_id: 1,
      final_text: "Done.",
    });
    expect(useChatStore.getState().toolActivity).toBeNull();
  });

  it("chat_error surfaces the error and clears the in-flight state", () => {
    useChatStore.setState({ streaming: "partial…", inFlight: true });
    useChatStore.getState().handleEvent({
      kind: "chat_error",
      session_id: SESSION,
      message: "the model crashed",
    });
    const state = useChatStore.getState();
    expect(state.lastError).toBe("the model crashed");
    expect(state.inFlight).toBe(false);
    expect(state.streaming).toBeNull();
  });

  it("ignores an event for a NON-active session (per-session scoping)", () => {
    useChatStore.setState({ streaming: "active text", inFlight: true });
    useChatStore.getState().handleEvent({
      kind: "chat_token",
      session_id: "some-other-session",
      turn_id: 1,
      token: "leak",
    });
    // Untouched: the other session's delta must not append to the open buffer.
    expect(useChatStore.getState().streaming).toBe("active text");

    useChatStore.getState().handleEvent({
      kind: "chat_turn_complete",
      session_id: "some-other-session",
      turn_id: 1,
      final_text: "should not appear",
    });
    expect(
      useChatStore
        .getState()
        .messages.some((m) => m.content === "should not appear"),
    ).toBe(false);
  });

  it("ignores chat events when no session is open", () => {
    useChatStore.setState({ sessionId: null, streaming: null });
    useChatStore.getState().handleEvent({
      kind: "chat_token",
      session_id: SESSION,
      turn_id: 1,
      token: "x",
    });
    expect(useChatStore.getState().streaming).toBeNull();
  });

  it("ignores non-chat events", () => {
    useChatStore.setState({ streaming: "kept", inFlight: true });
    useChatStore
      .getState()
      .handleEvent({ kind: "summary_ready", meeting_id: MEETING } as AppEvent);
    expect(useChatStore.getState().streaming).toBe("kept");
  });
});

describe("useChatStore — actions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("send appends the user message, enters in-flight, and adopts the returned session id", async () => {
    useChatStore.setState({ sessionId: null, messages: [] });
    vi.mocked(sendChatMessage).mockResolvedValueOnce("new-session");

    await useChatStore.getState().send("What was decided?");

    const state = useChatStore.getState();
    expect(sendChatMessage).toHaveBeenCalledWith(
      MEETING,
      null,
      "What was decided?",
    );
    const userMsg = state.messages[0];
    expect(userMsg.role).toBe("user");
    expect(userMsg.content).toBe("What was decided?");
    // The streamed buffer is primed and the turn is in flight; the returned id
    // is adopted so streamed events route to it.
    expect(state.streaming).toBe("");
    expect(state.inFlight).toBe(true);
    expect(state.sessionId).toBe("new-session");
  });

  it("send is a no-op while a turn is already in flight (single in-flight turn)", async () => {
    useChatStore.setState({ inFlight: true });
    await useChatStore.getState().send("second message");
    expect(sendChatMessage).not.toHaveBeenCalled();
  });

  it("send ignores empty / whitespace-only input", async () => {
    await useChatStore.getState().send("   ");
    expect(sendChatMessage).not.toHaveBeenCalled();
  });

  it("send surfaces an error and clears in-flight when dispatch rejects", async () => {
    vi.mocked(sendChatMessage).mockRejectedValueOnce(new Error("session busy"));
    await useChatStore.getState().send("hi");
    const state = useChatStore.getState();
    expect(state.inFlight).toBe(false);
    expect(state.streaming).toBeNull();
    expect(state.lastError).toBe("session busy");
  });

  it("loadSessions populates the session list", async () => {
    const sample: ChatSession = {
      id: SESSION,
      meeting_id: MEETING,
      title: "A chat",
      messages: [],
      created_at: "2026-06-10T00:00:00Z",
      updated_at: "2026-06-10T00:00:00Z",
    };
    vi.mocked(listChatSessions).mockResolvedValueOnce([sample]);
    await useChatStore.getState().loadSessions();
    expect(listChatSessions).toHaveBeenCalledWith(MEETING);
    expect(useChatStore.getState().sessions).toEqual([sample]);
  });

  it("openSession loads the session's messages", async () => {
    const sample: ChatSession = {
      id: SESSION,
      meeting_id: MEETING,
      title: "A chat",
      messages: [{ role: "user", content: "hello", turn_id: 0 }],
      created_at: "2026-06-10T00:00:00Z",
      updated_at: "2026-06-10T00:00:00Z",
    };
    vi.mocked(getChatSession).mockResolvedValueOnce(sample);
    await useChatStore.getState().openSession(SESSION);
    expect(getChatSession).toHaveBeenCalledWith(MEETING, SESSION);
    expect(useChatStore.getState().messages).toEqual(sample.messages);
    expect(useChatStore.getState().sessionId).toBe(SESSION);
  });

  it("deleteSession deletes, clears the open conversation, and refreshes the list", async () => {
    useChatStore.setState({
      sessionId: SESSION,
      messages: [{ role: "user", content: "x", turn_id: 0 }],
    });
    await useChatStore.getState().deleteSession(SESSION);
    expect(deleteChatSession).toHaveBeenCalledWith(MEETING, SESSION);
    expect(listChatSessions).toHaveBeenCalledWith(MEETING);
    const state = useChatStore.getState();
    expect(state.sessionId).toBeNull();
    expect(state.messages).toEqual([]);
  });

  it("newSession clears the open session so the next send mints a fresh one", () => {
    useChatStore.setState({
      sessionId: SESSION,
      messages: [{ role: "user", content: "x", turn_id: 0 }],
    });
    useChatStore.getState().newSession();
    const state = useChatStore.getState();
    expect(state.sessionId).toBeNull();
    expect(state.messages).toEqual([]);
  });

  it("setMeeting scopes the chat and loads its sessions, resetting the prior session", async () => {
    useChatStore.setState({
      sessionId: SESSION,
      messages: [{ role: "user", content: "x", turn_id: 0 }],
    });
    await useChatStore.getState().setMeeting("meeting-0002");
    const state = useChatStore.getState();
    expect(state.meetingId).toBe("meeting-0002");
    expect(state.sessionId).toBeNull();
    expect(state.messages).toEqual([]);
    expect(listChatSessions).toHaveBeenCalledWith("meeting-0002");
  });
});
