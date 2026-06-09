/**
 * Thin IPC client for the chat-agent surface (Phase 9).
 *
 * Wraps the shim-aware `commands` from `./client` (NOT raw `./bindings` — A9),
 * the same single injection point `summary.ts` / `meetings.ts` / `notes.ts`
 * use, so the DEV shim and the Vitest mocks both intercept here. Tests mock
 * THIS module (per `architecture/cross-cutting.md` — Automated-testing policy);
 * they do not fake the generated bindings file.
 *
 * The backend commands (the chat surface) landed with the Phase-9 backend JOIN
 * (commit a3883fb) and are present on the generated `commands` surface:
 *
 *   - `send_chat_message(meeting_id, session_id, message) -> ChatSessionId` —
 *     creates or loads the session, appends the user message, and spawns the
 *     turn on a background task, returning the session id immediately. The turn
 *     streams to the webview via the chat `AppEvent`s (`ChatToken` /
 *     `ChatToolCall` / `ChatToolResult` / `ChatTurnComplete` / `ChatError`); the
 *     resolved promise only confirms dispatch, not completion.
 *   - `get_chat_session(meeting_id, session_id) -> ChatSession | null` — loads
 *     one persisted session, or `null` when it does not exist.
 *   - `list_chat_sessions(meeting_id) -> ChatSession[]` — all sessions for a
 *     meeting, most-recently-updated first.
 *   - `delete_chat_session(meeting_id, session_id) -> ()` — deletes one session
 *     (idempotent).
 *
 * `ChatSession` / `ChatMessage` / `ChatSessionId` are re-exported from the
 * generated `./bindings` (the canonical `common` shapes) so call sites keep
 * importing them from this seam with no change.
 */
import { commands, unwrap } from "./client";
import type {
  ChatSession,
  ChatSessionId,
  ChatMessage,
  ChatRole,
  MeetingId,
} from "./bindings";

export type { ChatSession, ChatSessionId, ChatMessage, ChatRole };

/**
 * Send a user message to the chat agent for a meeting, streaming the reply.
 *
 * Resolves with the (new or existing) session id once the backend has dispatched
 * the turn; the streamed reply arrives over the chat `AppEvent`s and the
 * authoritative final text is `chat_turn_complete.final_text`. Rejects on an IPC
 * error (e.g. `invalid_input { "session busy" }` when a turn is already in
 * flight for the session).
 */
export async function sendChatMessage(
  meetingId: MeetingId | null,
  sessionId: ChatSessionId | null,
  message: string,
): Promise<ChatSessionId> {
  return unwrap(await commands.sendChatMessage(meetingId, sessionId, message));
}

/** Load one persisted chat session for a meeting, or `null` when none exists. */
export async function getChatSession(
  meetingId: MeetingId,
  sessionId: ChatSessionId,
): Promise<ChatSession | null> {
  return unwrap(await commands.getChatSession(meetingId, sessionId));
}

/** List all chat sessions for a meeting, most-recently-updated first. */
export async function listChatSessions(
  meetingId: MeetingId,
): Promise<ChatSession[]> {
  return unwrap(await commands.listChatSessions(meetingId));
}

/** Delete one chat session for a meeting (idempotent). */
export async function deleteChatSession(
  meetingId: MeetingId,
  sessionId: ChatSessionId,
): Promise<void> {
  unwrap(await commands.deleteChatSession(meetingId, sessionId));
}

/**
 * Cancel the in-flight chat turn for a session (P1).
 *
 * Raises the backend's per-session cancel flag; the engine stops between tokens
 * and the turn ends with a terminal `chat_turn_complete` carrying the partial
 * reply (not a `chat_error`). Idempotent — a session with no running turn is a
 * no-op success, so the UI can call it to clear a stuck in-flight state.
 */
export async function cancelChatTurn(sessionId: ChatSessionId): Promise<void> {
  unwrap(await commands.cancelChatTurn(sessionId));
}
