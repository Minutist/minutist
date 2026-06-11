/**
 * Update lifecycle store tests.
 *
 * Covers:
 * - `update_available` event transitions the store from idle → available.
 * - `update_progress` event transitions available → downloading and tracks
 *   bytes (determinate + indeterminate).
 * - `update_available` is a no-op while a download is in progress (guard).
 * - `applyUpdate()` transitions to `applying` and calls the applyFn.
 * - `applyUpdate()` is a no-op while already applying (duplicate-click guard).
 * - `applyUpdate()` reverts to the prior state on applyFn rejection.
 * - `applyUpdate()` is a no-op in the idle state.
 * - Unrelated events leave the update state unchanged.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { useUpdateStore } from "../state/update";
import type { AppEvent } from "../ipc/bindings";

describe("update store", () => {
  beforeEach(() => {
    useUpdateStore.setState({ update: { kind: "idle" } });
  });

  it("update_available transitions idle → available", () => {
    const event: AppEvent = {
      kind: "update_available",
      version: "1.2.3",
      notes: "Bug fixes.",
    };
    useUpdateStore.getState().handleEvent(event);
    const s = useUpdateStore.getState().update;
    expect(s.kind).toBe("available");
    if (s.kind === "available") {
      expect(s.version).toBe("1.2.3");
      expect(s.notes).toBe("Bug fixes.");
    }
  });

  it("update_available with null notes is stored correctly", () => {
    useUpdateStore.getState().handleEvent({
      kind: "update_available",
      version: "0.9.0",
      notes: null,
    });
    const s = useUpdateStore.getState().update;
    expect(s.kind).toBe("available");
    if (s.kind === "available") {
      expect(s.notes).toBeNull();
    }
  });

  it("update_progress transitions available → downloading (determinate)", () => {
    useUpdateStore.setState({
      update: { kind: "available", version: "1.2.3", notes: null },
    });
    useUpdateStore.getState().handleEvent({
      kind: "update_progress",
      downloaded_bytes: 512,
      total_bytes: 1024,
    });
    const s = useUpdateStore.getState().update;
    expect(s.kind).toBe("downloading");
    if (s.kind === "downloading") {
      expect(s.downloadedBytes).toBe(512);
      expect(s.totalBytes).toBe(1024);
      expect(s.version).toBe("1.2.3");
    }
  });

  it("update_progress with null total_bytes is indeterminate", () => {
    useUpdateStore.setState({
      update: { kind: "available", version: "1.0.0", notes: null },
    });
    useUpdateStore.getState().handleEvent({
      kind: "update_progress",
      downloaded_bytes: 100,
      total_bytes: null,
    });
    const s = useUpdateStore.getState().update;
    expect(s.kind).toBe("downloading");
    if (s.kind === "downloading") {
      expect(s.totalBytes).toBeNull();
    }
  });

  it("update_available is a no-op while downloading (duplicate guard)", () => {
    useUpdateStore.setState({
      update: {
        kind: "downloading",
        version: "1.2.3",
        notes: null,
        downloadedBytes: 256,
        totalBytes: 1024,
      },
    });
    useUpdateStore.getState().handleEvent({
      kind: "update_available",
      version: "9.9.9",
      notes: null,
    });
    const s = useUpdateStore.getState().update;
    // State must stay as downloading, not revert to available.
    expect(s.kind).toBe("downloading");
  });

  it("applyUpdate transitions available → applying and calls applyFn", () => {
    useUpdateStore.setState({
      update: { kind: "available", version: "1.0.0", notes: null },
    });
    const applyFn = vi.fn().mockResolvedValue(undefined);
    useUpdateStore.getState().applyUpdate(applyFn);
    expect(useUpdateStore.getState().update.kind).toBe("applying");
    expect(applyFn).toHaveBeenCalledOnce();
  });

  it("applyUpdate transitions downloading → applying and calls applyFn", () => {
    useUpdateStore.setState({
      update: {
        kind: "downloading",
        version: "1.0.0",
        notes: null,
        downloadedBytes: 1024,
        totalBytes: 1024,
      },
    });
    const applyFn = vi.fn().mockResolvedValue(undefined);
    useUpdateStore.getState().applyUpdate(applyFn);
    expect(useUpdateStore.getState().update.kind).toBe("applying");
    expect(applyFn).toHaveBeenCalledOnce();
  });

  it("applyUpdate is a no-op in idle state", () => {
    const applyFn = vi.fn().mockResolvedValue(undefined);
    useUpdateStore.getState().applyUpdate(applyFn);
    expect(applyFn).not.toHaveBeenCalled();
    expect(useUpdateStore.getState().update.kind).toBe("idle");
  });

  it("applyUpdate is a no-op while already applying (duplicate-click guard)", () => {
    useUpdateStore.setState({ update: { kind: "applying" } });
    const applyFn = vi.fn().mockResolvedValue(undefined);
    useUpdateStore.getState().applyUpdate(applyFn);
    expect(applyFn).not.toHaveBeenCalled();
    expect(useUpdateStore.getState().update.kind).toBe("applying");
  });

  it("applyUpdate reverts to prior state on applyFn rejection", async () => {
    const prior = { kind: "available" as const, version: "1.0.0", notes: null };
    useUpdateStore.setState({ update: prior });
    const applyFn = vi.fn().mockRejectedValue(new Error("network error"));
    useUpdateStore.getState().applyUpdate(applyFn);
    // After the rejected promise resolves:
    await vi.waitFor(() =>
      expect(useUpdateStore.getState().update.kind).toBe("available"),
    );
  });

  it("an unrelated event does not change the update state", () => {
    useUpdateStore.setState({
      update: { kind: "available", version: "1.0.0", notes: null },
    });
    useUpdateStore.getState().handleEvent({
      kind: "settings_changed",
    });
    expect(useUpdateStore.getState().update.kind).toBe("available");
  });
});
