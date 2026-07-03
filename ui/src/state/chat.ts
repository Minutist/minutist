/**
 * Chat-agent store (Phase 9).
 *
 * Holds the transient UI state for the meeting-scoped chat pane: the open
 * session's messages, the in-flight streaming assistant text, a transient
 * tool-activity indicator, the loading/error flags, and the session list for the
 * switcher. All mutations route through the `../ipc/chat` seam (mocked in
 * tests); the store keeps only transient UI state — `persistence::ChatStore`'s
 * on-disk session JSON is authoritative.
 *
 * Event reconciliation (the lossy-broadcast guarantee, see
 * `architecture/cross-cutting.md` — "Agent chat loop"):
 *   - `chat_token` is a PROGRESSIVE HINT: each delta is appended to the
 *     `streaming` buffer for the active session only. A dropped delta (the
 *     broadcast channel skips on lag) would corrupt the assembled text, so the
 *     streamed buffer is NEVER trusted as the final answer.
 *   - `chat_turn_complete.final_text` is AUTHORITATIVE: it REPLACES the streamed
 *     buffer with the full reconciled reply, appended to `messages` as the
 *     assistant message, and clears the in-flight state.
 *   - `chat_tool_call` / `chat_tool_result` drive a transient "running tool X"
 *     indicator (shown while a tool runs, replaced by its one-line result).
 *   - `chat_error` surfaces the error string and clears the in-flight state.
 *
 * KNOWN GAP (narrowed — #57): the TERMINAL events (`chat_turn_complete` /
 * `chat_error`) ride the same lossy broadcast bus as the deltas. If a terminal
 * event is dropped on lag, no terminal event clears the open session's
 * `inFlight`. The ESCAPE is the `Stop` control wired to `cancel()` (the Group-1
 * cancel surface): it raises the backend's cancel flag AND clears `inFlight` /
 * `streaming` / `toolActivity` locally — synchronously, not waiting on the bus —
 * then best-effort re-reads the persisted session (the cancel path persists the
 * partial turn) to reconcile `messages`. So a user is never permanently stuck:
 * pressing Stop always unsticks the UI. Re-opening the session (`openSession`)
 * remains a second recovery path (it re-reads from disk and clears `inFlight`).
 * A timeout watchdog (auto-cancel a turn that never terminates) is deliberately
 * OUT of scope. Do NOT reconcile from disk on the terminal EVENT as-is: the
 * driver emits it DURING the turn, before `persist_session` runs, so that read
 * would race and miss the new turn — only the explicit cancel/openSession reads,
 * which happen after the user acts, are safe.
 *
 * Per-session scoping: every chat event carries a `session_id`. An event whose
 * `session_id` is not the currently-open session is IGNORED, so a turn streaming
 * for a backgrounded session never clobbers the open one (the pane shows a
 * single session at a time).
 *
 * The live co-pilot's proactive feed is folded into this same timeline:
 * `live_copilot_message` (scoped by `meeting_id`, not `session_id` — it is the
 * co-pilot speaking unprompted, not a reply to the open session's turn) is
 * appended to `messages` as an assistant message when it matches the open
 * meeting. There is no separate digest pane; a persisted `"digest"` message
 * (raw transcript context fed to the model) is excluded from the rendered
 * timeline by `ChatView`, not by this store.
 */
import { create } from "zustand";
import {
  sendChatMessage,
  cancelChatTurn,
  getChatSession,
  listChatSessions,
  deleteChatSession,
} from "../ipc/chat";
import type { ChatSession, ChatMessage, ChatSessionId } from "../ipc/chat";
import type { MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

/** A transient indicator for a tool the agent is running / just ran mid-turn. */
export type ToolActivity = {
  /** The tool's name (e.g. `get_transcript`). */
  tool: string;
  /** True while the call is running; false once a result arrived. */
  running: boolean;
  /** Set on a finished call: the one-line human-facing render. */
  summary?: string;
  /** Set on a finished call: whether the tool succeeded. */
  ok?: boolean;
};

export type ChatStore = {
  /** The meeting this chat is scoped to, or `null` (no meeting open). */
  meetingId: MeetingId | null;
  /** The currently-open session id, or `null` when starting a fresh session. */
  sessionId: ChatSessionId | null;
  /** All sessions for the open meeting (the switcher), most-recent first. */
  sessions: ChatSession[];
  /** The open session's persisted messages (system message hidden by the view). */
  messages: ChatMessage[];
  /**
   * The in-flight streamed assistant text, assembled from `chat_token` deltas.
   * A PROGRESSIVE HINT only — `chat_turn_complete.final_text` is authoritative
   * and replaces this. `null` when no turn is streaming.
   */
  streaming: string | null;
  /** True between sending a message and the turn completing (or erroring). */
  inFlight: boolean;
  /** The transient tool-activity indicator for the in-flight turn, or `null`. */
  toolActivity: ToolActivity | null;
  /** Last error surfaced by a chat IPC call or a `chat_error` event. */
  lastError: string | null;
  /**
   * Set when the backend's sliding window evicted older turns to stay within
   * the context budget (`chat_context_trimmed`, P2). The view shows a quiet
   * "history trimmed" affordance; cleared on the next send / session switch.
   */
  historyTrimmed: boolean;

  /**
   * Scope the chat to a meeting (on open); loads its sessions and, if the
   * meeting has an ongoing live co-pilot session (`is_live === true`), opens
   * it so the conversation continues rather than starting blank.
   */
  setMeeting: (meetingId: MeetingId | null) => Promise<void>;
  /** Refresh the session list for the open meeting (the switcher). */
  loadSessions: () => Promise<void>;
  /** Open a session: load its messages and make it the active session. */
  openSession: (sessionId: ChatSessionId) => Promise<void>;
  /** Start a fresh session (clears the open session; the next send creates it). */
  newSession: () => void;
  /** Delete a session, then refresh the list (clears it if it was open). */
  deleteSession: (sessionId: ChatSessionId) => Promise<void>;
  /** Send a user message; appends it optimistically and enters the in-flight state. */
  send: (message: string) => Promise<void>;
  /** Cancel the in-flight turn (P1); the backend ends it with a partial reply. */
  cancel: () => Promise<void>;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** A user `ChatMessage` for the optimistic append on send. */
function userMessage(content: string, turnId: number): ChatMessage {
  return { role: "user", content, tool_calls: [], turn_id: turnId };
}

export const useChatStore = create<ChatStore>((set, get) => ({
  meetingId: null,
  sessionId: null,
  sessions: [],
  messages: [],
  streaming: null,
  inFlight: false,
  toolActivity: null,
  lastError: null,
  historyTrimmed: false,

  setMeeting: async (meetingId) => {
    // Switching meetings resets the open session and its messages — chat is
    // meeting-scoped, so a session for meeting A must not bleed into meeting B.
    set({
      meetingId,
      sessionId: null,
      sessions: [],
      messages: [],
      streaming: null,
      inFlight: false,
      toolActivity: null,
      lastError: null,
      historyTrimmed: false,
    });
    if (meetingId === null) return;
    await get().loadSessions();
    // Continue the meeting's live co-pilot session, if one exists, instead of
    // leaving `sessionId` null — so (re)mounting the chat pane shows the
    // ongoing co-pilot conversation rather than a blank "new session"
    // placeholder. At most one session per meeting has `is_live === true`.
    if (get().meetingId !== meetingId) return; // a later setMeeting won the race
    const liveSession = get().sessions.find((s) => s.is_live === true);
    if (liveSession) {
      await get().openSession(liveSession.id);
    }
  },

  loadSessions: async () => {
    const meetingId = get().meetingId;
    if (meetingId === null) return;
    try {
      const sessions = await listChatSessions(meetingId);
      set({
        sessions: Array.isArray(sessions) ? sessions : [],
        lastError: null,
      });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  openSession: async (sessionId) => {
    const meetingId = get().meetingId;
    if (meetingId === null) return;
    try {
      const session = await getChatSession(meetingId, sessionId);
      // Guard against a race / a since-deleted session: only apply if the
      // session still exists and the meeting is still the one we loaded for.
      if (get().meetingId !== meetingId) return;
      set({
        sessionId,
        messages: session?.messages ?? [],
        streaming: null,
        inFlight: false,
        toolActivity: null,
        lastError: null,
        historyTrimmed: false,
      });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  newSession: () => {
    // A fresh session: the next `send` passes `session_id = null`, so the
    // backend mints a new session and returns its id.
    set({
      sessionId: null,
      messages: [],
      streaming: null,
      inFlight: false,
      toolActivity: null,
      lastError: null,
      historyTrimmed: false,
    });
  },

  deleteSession: async (sessionId) => {
    const meetingId = get().meetingId;
    if (meetingId === null) return;
    try {
      await deleteChatSession(meetingId, sessionId);
      // If the deleted session was open, clear the conversation view.
      if (get().sessionId === sessionId) {
        set({
          sessionId: null,
          messages: [],
          streaming: null,
          inFlight: false,
          toolActivity: null,
          historyTrimmed: false,
        });
      }
      set({ lastError: null });
      await get().loadSessions();
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  send: async (message) => {
    const { meetingId, sessionId, messages, inFlight } = get();
    // Single in-flight turn per session (the backend rejects a second send with
    // `invalid_input { "session busy" }`); guard the UI too.
    if (inFlight) return;
    const trimmed = message.trim();
    if (trimmed === "") return;

    // Optimistically append the user message so it shows immediately, and enter
    // the in-flight / streaming state. The turn_id is provisional (the backend
    // owns the authoritative counter); it correlates the live deltas only.
    const lastTurnId = messages.reduce(
      (max, m) => (m.turn_id > max ? m.turn_id : max),
      -1,
    );
    const provisionalTurnId = lastTurnId + 1;
    set({
      messages: [...messages, userMessage(trimmed, provisionalTurnId)],
      streaming: "",
      inFlight: true,
      toolActivity: null,
      lastError: null,
      historyTrimmed: false,
    });

    try {
      const newSessionId = await sendChatMessage(meetingId, sessionId, trimmed);
      // The backend returns the (new or existing) session id; adopt it so the
      // streamed chat events (keyed on session_id) route to the open session,
      // and so a follow-up send continues the same session.
      if (get().meetingId === meetingId) {
        set({ sessionId: newSessionId });
      }
    } catch (err) {
      // Dispatch failed — leave the in-flight state and surface the error. (On
      // success, `chat_turn_complete` / `chat_error` clear it.)
      set({
        inFlight: false,
        streaming: null,
        toolActivity: null,
        lastError: errorMessage(err),
      });
    }
  },

  cancel: async () => {
    const { sessionId, meetingId, inFlight } = get();
    if (!inFlight || sessionId === null) return;
    try {
      await cancelChatTurn(sessionId);
      // Clear the in-flight state IMMEDIATELY so the UI is never permanently
      // stuck on "Stop" — this is the escape from a dropped terminal event (see
      // the KNOWN-GAP note above). The backend also ends the turn with a terminal
      // `chat_turn_complete` carrying the partial reply, but we do not depend on
      // that event arriving.
      set({ inFlight: false, streaming: null, toolActivity: null });
      // Best-effort reconcile: the cancel path persists the (partial) turn on the
      // backend, so re-read the session from disk to reflect the saved messages —
      // an `openSession`-style reconcile that does NOT depend on the lossy bus.
      // Guard against a race: only apply if the session/meeting are still open.
      if (meetingId !== null) {
        try {
          const session = await getChatSession(meetingId, sessionId);
          if (
            get().meetingId === meetingId &&
            get().sessionId === sessionId &&
            session
          ) {
            set({ messages: session.messages });
          }
        } catch {
          // A reconcile read failure is non-fatal — inFlight is already cleared,
          // so the user is unstuck regardless; the next openSession will reconcile.
        }
      }
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  handleEvent: (event) => {
    // `live_copilot_message` is scoped by MEETING id, not session id (the
    // event predates the concept of "the open session" — it is the live
    // co-pilot speaking proactively) — handle it separately from the
    // session-scoped switch below. Folds the co-pilot's proactive feed into
    // the chat timeline as one continuous conversation instead of a separate
    // pane: appended as an assistant message so it renders as an assistant
    // bubble alongside ordinary chat replies.
    if (event.kind === "live_copilot_message") {
      const { meetingId, messages } = get();
      if (meetingId === null || event.meeting_id !== meetingId) return;
      const last = messages[messages.length - 1];
      // Defensive dedupe: skip if the last message is already this exact
      // assistant reply. In practice this event is emitted only for
      // transcript-driven turns (not the reply to a user-typed message, which
      // already arrives via `chat_turn_complete`), so no duplicate occurs.
      if (last?.role === "assistant" && last.content === event.content) {
        return;
      }
      set({
        messages: [
          ...messages,
          {
            role: "assistant",
            content: event.content,
            tool_calls: [],
            turn_id: event.turn_id,
          },
        ],
      });
      return;
    }

    switch (event.kind) {
      case "chat_token":
      case "chat_tool_call":
      case "chat_tool_result":
      case "chat_turn_complete":
      case "chat_error":
      case "chat_context_trimmed":
        break;
      default:
        return;
    }

    // Per-session scoping: ignore an event for a session that is not the open
    // one. A turn streaming for a backgrounded session must not clobber the
    // open conversation.
    const openSession = get().sessionId;
    if (openSession === null || event.session_id !== openSession) return;

    switch (event.kind) {
      case "chat_token": {
        // Progressive hint: append the delta to the streamed buffer. NEVER
        // trusted as the final answer — `chat_turn_complete` reconciles it.
        const current = get().streaming ?? "";
        set({ streaming: current + event.token, inFlight: true });
        return;
      }
      case "chat_tool_call": {
        // The agent is running a tool — show a transient "running tool X" row.
        set({ toolActivity: { tool: event.tool, running: true } });
        return;
      }
      case "chat_tool_result": {
        // The tool finished — render its one-line summary on the activity row.
        set({
          toolActivity: {
            tool: event.tool,
            running: false,
            ok: event.ok,
            summary: event.summary,
          },
        });
        return;
      }
      case "chat_turn_complete": {
        // AUTHORITATIVE: replace the streamed buffer with the full reconciled
        // reply (lossy-broadcast mitigation — do NOT trust accumulated tokens),
        // append it as the assistant message, and clear the in-flight state.
        const assistant: ChatMessage = {
          role: "assistant",
          content: event.final_text,
          tool_calls: [],
          turn_id: event.turn_id,
        };
        set({
          messages: [...get().messages, assistant],
          streaming: null,
          inFlight: false,
          toolActivity: null,
          lastError: null,
        });
        return;
      }
      case "chat_error": {
        // The turn failed — surface the error and clear any in-flight state. The
        // optimistically-appended user message is left in place (the user can
        // see what they asked); the streamed partial is discarded.
        set({
          streaming: null,
          inFlight: false,
          toolActivity: null,
          lastError: event.message,
        });
        return;
      }
      case "chat_context_trimmed": {
        // The backend evicted older turns to stay within the context budget
        // (P2). Surface a quiet "history trimmed" affordance; the turn continues.
        set({ historyTrimmed: true });
        return;
      }
    }
  },
}));
