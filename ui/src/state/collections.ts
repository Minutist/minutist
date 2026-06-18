/**
 * Collections ("folders") store.
 *
 * Holds the folder definitions shown in the home-screen sidebar plus the active
 * folder filter applied to the meeting list. Folder *membership* lives on each
 * meeting (`MeetingListEntry.collection_id`, in the meetings store); this store
 * owns only the definitions + the selected filter. All mutations route through
 * the `../ipc/collections` seam (mocked in tests).
 */
import { create } from "zustand";
import {
  listCollections,
  createCollection,
  renameCollection,
  deleteCollection,
} from "../ipc/collections";
import type { Collection, CollectionId } from "../ipc/collections";

export type { Collection, CollectionId };

/** The active meeting-list folder filter. */
export type CollectionFilter =
  | { kind: "all" }
  | { kind: "unfiled" }
  | { kind: "collection"; id: CollectionId };

export const ALL_FILTER: CollectionFilter = { kind: "all" };
export const UNFILED_FILTER: CollectionFilter = { kind: "unfiled" };

export type CollectionsStore = {
  /** Folder definitions, ordered by position. */
  collections: Collection[];
  /** The active folder filter applied to the meeting list. */
  filter: CollectionFilter;
  /** Last error surfaced by a collections IPC call. */
  lastError: string | null;

  /** Refresh the folder list; resets the filter to All if it was deleted. */
  refresh: () => Promise<void>;
  /** Create a folder; returns it (or null on failure). */
  create: (name: string) => Promise<Collection | null>;
  /** Rename a folder. */
  rename: (id: CollectionId, name: string) => Promise<void>;
  /** Delete a folder (its meetings become unfiled). */
  remove: (id: CollectionId) => Promise<void>;
  /** Set the active meeting-list filter. */
  select: (filter: CollectionFilter) => void;
};

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export const useCollectionsStore = create<CollectionsStore>((set, get) => ({
  collections: [],
  filter: ALL_FILTER,
  lastError: null,

  refresh: async () => {
    try {
      const collections = await listCollections();
      const list = Array.isArray(collections) ? collections : [];
      // If the selected folder no longer exists (deleted here or elsewhere),
      // fall back to All so the list never filters to a phantom folder.
      const { filter } = get();
      const stillExists =
        filter.kind !== "collection" || list.some((c) => c.id === filter.id);
      set({
        collections: list,
        filter: stillExists ? filter : ALL_FILTER,
        lastError: null,
      });
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  create: async (name) => {
    try {
      const created = await createCollection(name);
      set({ lastError: null });
      await get().refresh();
      return created;
    } catch (err) {
      set({ lastError: errorMessage(err) });
      return null;
    }
  },

  rename: async (id, name) => {
    try {
      await renameCollection(id, name);
      set({ lastError: null });
      await get().refresh();
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  remove: async (id) => {
    try {
      await deleteCollection(id);
      set({ lastError: null });
      // Refresh resets a now-deleted selected folder back to All.
      await get().refresh();
    } catch (err) {
      set({ lastError: errorMessage(err) });
    }
  },

  select: (filter) => set({ filter }),
}));

/**
 * Whether a meeting (by its `collection_id`) matches the active folder filter.
 * Pure — shared by the meeting list (row filtering) and tests.
 */
export function meetingMatchesFilter(
  filter: CollectionFilter,
  collectionId: CollectionId | null | undefined,
): boolean {
  switch (filter.kind) {
    case "all":
      return true;
    case "unfiled":
      return !collectionId;
    case "collection":
      return collectionId === filter.id;
  }
}
