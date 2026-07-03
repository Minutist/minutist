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
 *   - `live_copilot_message` for the open meeting is folded into the timeline
 *     as an assistant message (the merged co-pilot feed — U2 unification),
 *   - `send` appends the user message optimistically + enters the in-flight
 *     state, and adopts the returned session id,
 *   - `loadSessions` / `openSession` / `deleteSession` route through the seam,
 *   - `setMeeting` auto-opens the meeting's live co-pilot session, if one
 *     exists, instead of leaving the chat blank.
 *
 * The IPC calls are mocked at the `../ipc/chat` seam (per the architecture
 * testing policy — do not fake the generated bindings file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../ipc/chat", () => ({
  sendChatMessage: vi.fn().mockResolvedValue("session-1"),
  cancelChatTurn: vi.fn().mockResolvedValue(undefined),
  getChatSession: vi.fn().mockResolvedValue(null),
  listChatSessions: vi.fn().mockResolvedValue([]),
  deleteChatSession: vi.fn().mockResolvedValue(undefined),
}));

import {
  sendChatMessage,
  cancelChatTurn,
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
    historyTrimmed: false,
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

  it("chat_context_trimmed sets the history-trimmed flag without ending the turn", () => {
    useChatStore.setState({ inFlight: true, historyTrimmed: false });
    useChatStore.getState().handleEvent({
      kind: "chat_context_trimmed",
      session_id: SESSION,
      dropped_turns: 3,
    });
    const state = useChatStore.getState();
    expect(state.historyTrimmed).toBe(true);
    // The turn continues — the trim is informational, not terminal.
    expect(state.inFlight).toBe(true);
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

describe("useChatStore — the merged live co-pilot feed (live_copilot_message)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("appends an assistant message for the open meeting", () => {
    useChatStore.getState().handleEvent({
      kind: "live_copilot_message",
      meeting_id: MEETING,
      turn_id: 3,
      role: "assistant",
      content: "Action item: follow up with Alice.",
    } as AppEvent);

    const state = useChatStore.getState();
    const last = state.messages[state.messages.length - 1];
    expect(last.role).toBe("assistant");
    expect(last.content).toBe("Action item: follow up with Alice.");
    expect(last.turn_id).toBe(3);
  });

  it("ignores a live_copilot_message for a different meeting", () => {
    useChatStore.getState().handleEvent({
      kind: "live_copilot_message",
      meeting_id: "some-other-meeting",
      turn_id: 1,
      role: "assistant",
      content: "should not appear",
    } as AppEvent);

    expect(
      useChatStore
        .getState()
        .messages.some((m) => m.content === "should not appear"),
    ).toBe(false);
  });

  it("does not append an identical repeat of the last assistant message (dedupe guard)", () => {
    useChatStore.setState({
      messages: [
        { role: "assistant", content: "Repeated note.", tool_calls: [], turn_id: 1 },
      ],
    });
    useChatStore.getState().handleEvent({
      kind: "live_copilot_message",
      meeting_id: MEETING,
      turn_id: 2,
      role: "assistant",
      content: "Repeated note.",
    } as AppEvent);

    expect(useChatStore.getState().messages).toHaveLength(1);
  });
});

describe("useChatStore — setMeeting continues the live co-pilot session", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("auto-opens the meeting's is_live session instead of leaving sessionId null", async () => {
    const liveSession: ChatSession = {
      id: "live-session",
      meeting_id: MEETING,
      title: null,
      messages: [
        { role: "assistant", content: "already in progress", tool_calls: [], turn_id: 0 },
      ],
      created_at: "2026-06-10T00:00:00Z",
      updated_at: "2026-06-10T00:00:00Z",
      is_live: true,
    };
    vi.mocked(listChatSessions).mockResolvedValueOnce([liveSession]);
    vi.mocked(getChatSession).mockResolvedValueOnce(liveSession);

    await useChatStore.getState().setMeeting(MEETING);

    expect(getChatSession).toHaveBeenCalledWith(MEETING, "live-session");
    const state = useChatStore.getState();
    expect(state.sessionId).toBe("live-session");
    expect(state.messages).toEqual(liveSession.messages);
  });

  it("leaves sessionId null when no session is live", async () => {
    const finished: ChatSession = {
      id: "old-session",
      meeting_id: MEETING,
      title: "Past chat",
      messages: [],
      created_at: "2026-06-10T00:00:00Z",
      updated_at: "2026-06-10T00:00:00Z",
      is_live: false,
    };
    vi.mocked(listChatSessions).mockResolvedValueOnce([finished]);

    await useChatStore.getState().setMeeting(MEETING);

    expect(getChatSession).not.toHaveBeenCalled();
    expect(useChatStore.getState().sessionId).toBeNull();
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

  it("cancel raises the backend flag and clears the in-flight state (P1)", async () => {
    useChatStore.setState({ sessionId: SESSION, inFlight: true, streaming: "x" });
    await useChatStore.getState().cancel();
    expect(cancelChatTurn).toHaveBeenCalledWith(SESSION);
    const state = useChatStore.getState();
    expect(state.inFlight).toBe(false);
    expect(state.streaming).toBeNull();
  });

  it("cancel is a no-op when no turn is in flight", async () => {
    useChatStore.setState({ sessionId: SESSION, inFlight: false });
    await useChatStore.getState().cancel();
    expect(cancelChatTurn).not.toHaveBeenCalled();
  });

  it("cancel reconciles messages from the persisted session (#57 escape)", async () => {
    // The backend persists the (partial) turn on cancel; cancel re-reads it so
    // the saved messages reconcile into the open session, and the user is never
    // left stuck on a dropped terminal event.
    const persisted: ChatSession = {
      id: SESSION,
      meeting_id: MEETING,
      title: "A chat",
      messages: [
        { role: "user", content: "hi", tool_calls: [], turn_id: 0 },
        { role: "assistant", content: "partial reply…", tool_calls: [], turn_id: 0 },
      ],
      created_at: "2026-06-10T00:00:00Z",
      updated_at: "2026-06-10T00:00:00Z",
    };
    vi.mocked(getChatSession).mockResolvedValueOnce(persisted);
    useChatStore.setState({
      meetingId: MEETING,
      sessionId: SESSION,
      inFlight: true,
      streaming: "partial reply…",
      messages: [{ role: "user", content: "hi", tool_calls: [], turn_id: 0 }],
    });

    await useChatStore.getState().cancel();

    expect(cancelChatTurn).toHaveBeenCalledWith(SESSION);
    expect(getChatSession).toHaveBeenCalledWith(MEETING, SESSION);
    const state = useChatStore.getState();
    expect(state.inFlight).toBe(false);
    expect(state.streaming).toBeNull();
    // The persisted assistant partial is reconciled into the open session.
    expect(state.messages).toEqual(persisted.messages);
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
      messages: [{ role: "user", content: "hello", tool_calls: [], turn_id: 0 }],
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
      messages: [{ role: "user", content: "x", tool_calls: [], turn_id: 0 }],
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
      messages: [{ role: "user", content: "x", tool_calls: [], turn_id: 0 }],
    });
    useChatStore.getState().newSession();
    const state = useChatStore.getState();
    expect(state.sessionId).toBeNull();
    expect(state.messages).toEqual([]);
  });

  it("setMeeting scopes the chat and loads its sessions, resetting the prior session", async () => {
    useChatStore.setState({
      sessionId: SESSION,
      messages: [{ role: "user", content: "x", tool_calls: [], turn_id: 0 }],
    });
    await useChatStore.getState().setMeeting("meeting-0002");
    const state = useChatStore.getState();
    expect(state.meetingId).toBe("meeting-0002");
    expect(state.sessionId).toBeNull();
    expect(state.messages).toEqual([]);
    expect(listChatSessions).toHaveBeenCalledWith("meeting-0002");
  });
});
