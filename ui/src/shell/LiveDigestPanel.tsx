/**
 * Live co-pilot feed panel (U4).
 *
 * Renders transcript-driven observations and alerts emitted by the live
 * co-pilot as assistant bubbles, using the same `renderChatMarkdown` helper
 * and `chat-view__bubble--assistant` class as the chat pane. The panel
 * appears once the co-pilot has spoken (`useLiveCopilotStore.hasMessages`);
 * until then `MainWindow` keeps it hidden.
 *
 * No user input — purely a read surface for proactive co-pilot output.
 * User-chat replies stream through the chat pane via
 * `ChatToken`/`ChatTurnComplete`; they do not appear here.
 */
import { useLiveCopilotStore } from "../state/liveCopilot";
import { renderChatMarkdown } from "./ChatView";
import type { MeetingId } from "../ipc/bindings";
import "./LiveDigestPanel.css";
import "./ChatView.css";

type LiveDigestPanelProps = {
  /** The meeting this panel is scoped to. */
  meetingId: MeetingId;
};

export function LiveDigestPanel({ meetingId }: LiveDigestPanelProps) {
  const messages = useLiveCopilotStore((s) => s.messagesFor(meetingId));

  return (
    <section
      className="live-digest ink-reveal"
      aria-label="Live co-pilot feed"
    >
      <header className="live-digest__header">
        <h2 className="live-digest__heading">Co-pilot</h2>
      </header>

      <div className="live-digest__body">
        {messages.length === 0 ? (
          <p className="live-digest__empty">No co-pilot notes yet.</p>
        ) : (
          messages.map((msg, i) => (
            <div
              key={i}
              className="chat-view__bubble chat-view__bubble--assistant"
              // renderChatMarkdown returns sanitised-by-construction HTML
              // (markdown-it with html: false); no user-supplied content
              // reaches dangerouslySetInnerHTML.
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{
                __html: renderChatMarkdown(msg.content),
              }}
            />
          ))
        )}
      </div>
    </section>
  );
}
