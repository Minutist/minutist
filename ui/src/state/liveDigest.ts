/**
 * Per-meeting live digest store (Phase 9 — S3).
 *
 * Holds the latest `LiveDigest` for each meeting, fed exclusively by the
 * global event bridge (`shell/event-listener.tsx`) via `handleEvent`. No IPC
 * seam — the panel is event-driven; there is nothing to fetch.
 *
 * Payload semantics: `live_digest_updated` carries the FULL replacement digest
 * and the store OVERWRITES wholesale (lossy-broadcast-safe — a dropped
 * intermediate update is recovered on the next backend refresh; the backend
 * carries `resolved` forward across refreshes so the store never reconciles
 * item-level state itself).
 *
 * `live_digest_error` stores the message and RETAINS the last valid digest for
 * that meeting so the panel does not blank on a transient error.
 */
import { create } from "zustand";
import type { MeetingId, LiveDigest } from "../ipc/bindings";
import type { AppEvent } from "../ipc/app-event";

export type LiveDigestEntry = {
  /** The latest authoritative digest, or `null` before the first refresh. */
  digest: LiveDigest | null;
  /** The last error message from `live_digest_error`, or `null` when none. */
  lastError: string | null;
};

export type LiveDigestStore = {
  /** Latest digest entry per meeting id. */
  digests: Record<MeetingId, LiveDigestEntry>;
  /**
   * Returns the entry for a meeting, or `null` when no digest has arrived for
   * it yet. Mirrors the `operationFor` selector pattern.
   */
  digestFor: (meetingId: MeetingId) => LiveDigestEntry | null;
  /** Dispatcher called by the global event listener. */
  handleEvent: (event: AppEvent) => void;
};

export const useLiveDigestStore = create<LiveDigestStore>((set, get) => ({
  digests: {},

  digestFor: (meetingId) => get().digests[meetingId] ?? null,

  handleEvent: (event) => {
    switch (event.kind) {
      case "live_digest_updated": {
        // Wholesale overwrite — the payload is authoritative.
        set((s) => ({
          digests: {
            ...s.digests,
            [event.meeting_id]: {
              digest: event.digest,
              lastError: null,
            },
          },
        }));
        break;
      }
      case "live_digest_error": {
        // Retain the last valid digest; only update the error field.
        set((s) => {
          const prior = s.digests[event.meeting_id];
          return {
            digests: {
              ...s.digests,
              [event.meeting_id]: {
                digest: prior?.digest ?? null,
                lastError: event.message,
              },
            },
          };
        });
        break;
      }
      default:
        break;
    }
  },
}));
