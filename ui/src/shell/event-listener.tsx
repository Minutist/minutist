import { useEffect } from "react";
import { listenAppEvents } from "../ipc/client";
import { useRecordingStore } from "../state/recording";

/**
 * Mount the global Tauri event bridge exactly once.
 *
 * This hook subscribes to the `"app-event-payload"` event and dispatches
 * each payload into the Zustand store via `handleEvent`. It must be called
 * from a component that is always mounted (i.e. `App`), never inside a
 * conditionally-rendered subtree.
 *
 * The unlisten function returned by `listenAppEvents` is called on cleanup,
 * so the listener is removed when the component unmounts (HMR, tests, etc.).
 */
export function useAppEventBridge(): void {
  const handleEvent = useRecordingStore((s) => s.handleEvent);

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    listenAppEvents((event) => {
      handleEvent(event);
    })
      .then((fn) => {
        unlisten = fn;
      })
      .catch((err: unknown) => {
        console.error("[event-listener] failed to subscribe:", err);
      });

    return () => {
      unlisten?.();
    };
  // handleEvent is a stable function reference from the Zustand store; the
  // dep array is intentionally exhaustive.
  }, [handleEvent]);
}
