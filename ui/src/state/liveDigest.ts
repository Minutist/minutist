/**
 * Per-meeting live-agent driver error store.
 *
 * Holds the last `live_digest_error` message for each meeting, fed
 * exclusively by the global event bridge (`shell/event-listener.tsx`) via
 * `handleEvent`. No IPC seam — event-driven; there is nothing to fetch.
 *
 * The event name kept the wire tag `live_digest_error` for backward
 * compatibility with the Rust `AppEvent::LiveDigestError` variant; it now
 * reports a terminal live-agent-driver failure (worker startup, decode
 * error, or context-capacity exhaustion), not a digest-refresh failure —
 * the digest-panel design it originally served was superseded by the
 * unified co-pilot chat log (`LiveCopilotMessage`) before a digest producer
 * was ever written.
 */
import { create } from "zustand";
import type { MeetingId } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type LiveDigestStore = {
  /** Last error message per meeting id, or absent when none has occurred. */
  errors: Record<MeetingId, string>;
  /** Returns the last error for a meeting, or `null` when none has occurred. */
  errorFor: (meetingId: MeetingId) => string | null;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useLiveDigestStore = create<LiveDigestStore>((set, get) => ({
  errors: {},

  errorFor: (meetingId) => get().errors[meetingId] ?? null,

  handleEvent: (event) => {
    if (event.kind !== "live_digest_error") return;
    set((s) => ({
      errors: { ...s.errors, [event.meeting_id]: event.message },
    }));
  },
}));
