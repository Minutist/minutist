/**
 * Thin IPC client for collections ("folders") — the home-screen folder sidebar.
 *
 * Wraps the shim-aware `commands` from `./client` (NOT raw `./bindings`), the
 * same single injection point the other seams use, so the DEV shim and the
 * Vitest mocks both intercept here. Tests mock THIS module (per
 * `architecture/cross-cutting.md` — Automated testing policy).
 *
 * A *collection* is a user-facing folder grouping meetings; membership is one
 * folder per meeting (or none). `Collection` / `CollectionId` are re-exported
 * from the generated `./bindings` (the canonical `common` shapes).
 */
import { commands, unwrap } from "./client";
import type { Collection, CollectionId, MeetingId } from "./bindings";

export type { Collection, CollectionId };

/** All collections (folders), ordered by position. */
export async function listCollections(): Promise<Collection[]> {
  return unwrap(await commands.listCollections());
}

/** Create a collection named `name`; returns the created collection. */
export async function createCollection(name: string): Promise<Collection> {
  return unwrap(await commands.createCollection(name));
}

/** Rename a collection. */
export async function renameCollection(
  collectionId: CollectionId,
  name: string,
): Promise<void> {
  unwrap(await commands.renameCollection(collectionId, name));
}

/** Delete a collection; its meetings become unfiled. */
export async function deleteCollection(
  collectionId: CollectionId,
): Promise<void> {
  unwrap(await commands.deleteCollection(collectionId));
}

/** File a meeting into a collection, or unfile it with `null`. */
export async function setMeetingCollection(
  meetingId: MeetingId,
  collectionId: CollectionId | null,
): Promise<void> {
  unwrap(await commands.setMeetingCollection(meetingId, collectionId));
}
