/**
 * Chat pane (Phase 9) — the meeting-scoped agent conversation surface.
 *
 * A workspace column (not an overlay), shown alongside the editor / transcript /
 * summary panes for a meeting. It renders the open session's messages as
 * user / assistant bubbles, a compact tool-activity row while the agent runs a
 * tool, a streaming caret while tokens arrive, a send box (Enter to send,
 * disabled while a turn is in flight), a session switcher (new / pick / delete),
 * and an error state.
 *
 * All mutations route through `useChatStore` (which wraps the `../ipc/chat`
 * seam); the component holds only the local input draft. The lossy-stream
 * guarantee lives in the store: the streamed `streaming` buffer is a hint, and
 * the authoritative assistant text arrives on `chat_turn_complete` as the last
 * `messages` entry. Editorial-Ink language: `theme.css` tokens only.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import MarkdownIt from "markdown-it";
import { useChatStore } from "../state/chat";
import type { ChatMessage } from "../ipc/chat";
import type { MeetingId } from "../ipc/bindings";
import "./ChatView.css";

// markdown-only, no raw HTML — assistant replies are model-generated markdown.
const md = new MarkdownIt({ html: false, linkify: true, typographer: true });

/** Render assistant markdown to a sanitised-by-construction HTML string. */
export function renderChatMarkdown(text: string): string {
  return md.render(text);
}

type ChatViewProps = {
  /** The meeting this chat is scoped to. */
  meetingId: MeetingId;
};

/** A short, stable label for a session in the switcher. */
function sessionLabel(title: string | null | undefined, id: string): string {
  if (title && title.trim() !== "") return title;
  return `Session ${id.slice(0, 8)}`;
}

export function ChatView({ meetingId }: ChatViewProps) {
  const sessionId = useChatStore((s) => s.sessionId);
  const sessions = useChatStore((s) => s.sessions);
  const messages = useChatStore((s) => s.messages);
  const streaming = useChatStore((s) => s.streaming);
  const inFlight = useChatStore((s) => s.inFlight);
  const toolActivity = useChatStore((s) => s.toolActivity);
  const lastError = useChatStore((s) => s.lastError);
  const setMeeting = useChatStore((s) => s.setMeeting);
  const openSession = useChatStore((s) => s.openSession);
  const newSession = useChatStore((s) => s.newSession);
  const deleteSession = useChatStore((s) => s.deleteSession);
  const send = useChatStore((s) => s.send);

  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Scope the chat to the open meeting (loads its sessions). Switching meetings
  // resets the open session in the store.
  useEffect(() => {
    void setMeeting(meetingId);
  }, [meetingId, setMeeting]);

  // Stick to the bottom as messages / streamed tokens arrive.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming, toolActivity]);

  // The system message (turn 0) is the persona/context prompt — not shown.
  const visibleMessages = useMemo(
    () => messages.filter((m) => m.role !== "system"),
    [messages],
  );

  function submit() {
    const text = draft.trim();
    if (text === "" || inFlight) return;
    setDraft("");
    void send(text);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  const hasConversation = visibleMessages.length > 0 || streaming !== null;

  return (
    <section className="chat-view ink-reveal" aria-label="Meeting chat">
      <header className="chat-view__header">
        <h2 className="chat-view__heading">Chat</h2>
        <div className="chat-view__session-switch">
          <label className="chat-view__session-label" htmlFor="chat-session">
            Session
          </label>
          <select
            id="chat-session"
            className="chat-view__session-select"
            value={sessionId ?? ""}
            onChange={(e) => {
              const value = e.target.value;
              if (value === "") newSession();
              else void openSession(value);
            }}
          >
            <option value="">New session</option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {sessionLabel(s.title, s.id)}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="chat-view__session-btn"
            onClick={() => newSession()}
            title="Start a new chat session for this meeting"
          >
            New
          </button>
          {sessionId !== null && (
            <button
              type="button"
              className="chat-view__session-btn"
              onClick={() => void deleteSession(sessionId)}
              title="Delete this chat session"
            >
              Delete
            </button>
          )}
        </div>
      </header>

      <div
        className="chat-view__messages"
        ref={scrollRef}
        aria-live="polite"
        aria-label="Conversation"
      >
        {!hasConversation && (
          <p className="chat-view__empty">
            Ask about this meeting — the agent reads the transcript, summary and
            your notes to answer.
          </p>
        )}

        {visibleMessages.map((m: ChatMessage, idx: number) => {
          if (m.role === "tool") {
            // A persisted tool-result message — render as a quiet activity row.
            return (
              <div key={idx} className="chat-view__tool" role="note">
                <span className="chat-view__tool-name">
                  {m.tool_name ?? "tool"}
                </span>
                <span className="chat-view__tool-summary">{m.content}</span>
              </div>
            );
          }
          const isUser = m.role === "user";
          return (
            <div
              key={idx}
              className={
                isUser
                  ? "chat-view__bubble chat-view__bubble--user"
                  : "chat-view__bubble chat-view__bubble--assistant"
              }
            >
              {isUser ? (
                <p className="chat-view__bubble-text">{m.content}</p>
              ) : (
                <div
                  className="chat-view__bubble-text chat-view__markdown"
                  // markdown-it output with `html: false`; model markdown only.
                  dangerouslySetInnerHTML={{
                    __html: renderChatMarkdown(m.content),
                  }}
                />
              )}
            </div>
          );
        })}

        {/* The transient tool-activity indicator for the in-flight turn. */}
        {toolActivity && (
          <div className="chat-view__tool" role="status">
            <span className="chat-view__tool-name">{toolActivity.tool}</span>
            <span className="chat-view__tool-summary">
              {toolActivity.running
                ? "running…"
                : (toolActivity.ok === false ? "failed: " : "") +
                  (toolActivity.summary ?? "done")}
            </span>
          </div>
        )}

        {/* The in-flight streamed assistant text + caret (a progressive hint;
            replaced by the authoritative `chat_turn_complete` message). */}
        {streaming !== null && (
          <div className="chat-view__bubble chat-view__bubble--assistant">
            <div className="chat-view__bubble-text chat-view__markdown">
              {streaming === "" && toolActivity === null ? (
                <span className="chat-view__thinking">Thinking…</span>
              ) : (
                <span
                  // The streamed buffer is rendered as plain text while it
                  // assembles (it may be mid-markdown); the final message is
                  // rendered through markdown once `chat_turn_complete` lands.
                  className="chat-view__stream-text"
                >
                  {streaming}
                </span>
              )}
              <span className="chat-view__caret" aria-hidden="true" />
            </div>
          </div>
        )}
      </div>

      {lastError && (
        <p className="chat-view__error" role="alert">
          {lastError}
        </p>
      )}

      <div className="chat-view__composer">
        <textarea
          className="chat-view__input"
          aria-label="Message the meeting agent"
          placeholder="Ask about this meeting…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
        />
        <button
          type="button"
          className="chat-view__send"
          onClick={submit}
          disabled={inFlight || draft.trim() === ""}
        >
          {inFlight ? "Sending…" : "Send"}
        </button>
      </div>
    </section>
  );
}
