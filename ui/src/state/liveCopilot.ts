/**
 * Per-meeting live co-pilot message store (U4).
 *
 * Holds the ordered list of `LiveCopilotMessage` events for each meeting,
 * fed exclusively by the global event bridge (`shell/event-listener.tsx`) via
 * `handleEvent`. No IPC seam — the feed is event-driven.
 *
 * Payload semantics: `live_copilot_message` carries a single observation/alert
 * emitted by the co-pilot in response to transcript activity. Messages are
 * appended in arrival order; the store never prunes within a session.
 *
 * Only TRANSCRIPT-driven turns reach the store. User-chat replies travel on
 * the `ChatToken`/`ChatTurnComplete` events keyed by the live session id and
 * are rendered by the chat pane, not here.
 */
import { create } from "zustand";
import type { MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type LiveCopilotMessage = {
  /** Per-session monotonic counter (mirrors the backend turn_id). */
  turn_id: number;
  /** Always "assistant" for transcript-driven observations. */
  role: string;
  /** Markdown text of the observation or alert. */
  content: string;
};

export type LiveCopilotStore = {
  /** Ordered message list per meeting id. */
  messages: Map<MeetingId, LiveCopilotMessage[]>;
  /**
   * Returns the messages for a meeting in arrival order, or an empty array
   * when none have arrived yet.
   */
  messagesFor: (meetingId: MeetingId) => LiveCopilotMessage[];
  /** Returns true once at least one message has arrived for the meeting. */
  hasMessages: (meetingId: MeetingId) => boolean;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useLiveCopilotStore = create<LiveCopilotStore>((set, get) => ({
  messages: new Map(),

  messagesFor: (meetingId) => get().messages.get(meetingId) ?? [],

  hasMessages: (meetingId) => (get().messages.get(meetingId)?.length ?? 0) > 0,

  handleEvent: (event) => {
    switch (event.kind) {
      case "live_copilot_message": {
        set((s) => {
          const prior = s.messages.get(event.meeting_id) ?? [];
          const next = new Map(s.messages);
          next.set(event.meeting_id, [
            ...prior,
            {
              turn_id: event.turn_id,
              role: event.role,
              content: event.content,
            },
          ]);
          return { messages: next };
        });
        break;
      }
      default:
        break;
    }
  },
}));
