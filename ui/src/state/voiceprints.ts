/**
 * Speaker voiceprint identities store.
 *
 * Holds the enrolled-identity list shown in `VoiceprintPane` (a Settings-drawer
 * section). All mutations route through the `commands.*` IPC seam and reload
 * the list afterward so the UI reflects the authoritative post-mutation state
 * (e.g. a merge's post-merge gallery counts).
 */
import { create } from "zustand";
import { commands, unwrap } from "../ipc/client";
import type { VoiceprintIdentityInfo } from "../ipc/bindings";
import { errorMessage } from "../lib/errors";

export type VoiceprintsStore = {
  identities: VoiceprintIdentityInfo[];
  loading: boolean;
  lastError: string | null;

  /** Reload the identity list from the backend. */
  refresh: () => Promise<void>;
  /** Rename an identity's display name. */
  rename: (id: string, name: string) => Promise<void>;
  /** Delete one identity. */
  remove: (id: string) => Promise<void>;
  /** Merge `mergedId`'s gallery into `keepId`, then reload. */
  merge: (keepId: string, mergedId: string) => Promise<void>;
  /** Clear the entire library (§4 privacy — right to erasure). */
  clearAll: () => Promise<void>;
};

export const useVoiceprintsStore = create<VoiceprintsStore>((set, get) => ({
  identities: [],
  loading: false,
  lastError: null,

  refresh: async () => {
    set({ loading: true, lastError: null });
    try {
      const list = unwrap(await commands.listVoiceprints());
      set({ identities: list, loading: false });
    } catch (err) {
      set({ lastError: errorMessage(err), loading: false });
    }
  },

  rename: async (id, name) => {
    try {
      unwrap(await commands.renameVoiceprintIdentity(id, name));
      set({
        identities: get().identities.map((i) =>
          i.identityId === id ? { ...i, displayName: name } : i,
        ),
        lastError: null,
      });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  remove: async (id) => {
    try {
      unwrap(await commands.deleteVoiceprintIdentity(id));
      set({
        identities: get().identities.filter((i) => i.identityId !== id),
        lastError: null,
      });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  merge: async (keepId, mergedId) => {
    try {
      unwrap(await commands.mergeVoiceprintIdentities(keepId, mergedId));
      set({ lastError: null });
      // Reload to get the post-merge gallery state.
      await get().refresh();
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  clearAll: async () => {
    try {
      unwrap(await commands.clearAllVoiceprints());
      set({ identities: [], lastError: null });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },
}));
