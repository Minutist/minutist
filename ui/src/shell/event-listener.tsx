import { useEffect } from "react";
import { listenAppEvents } from "../ipc/client";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useSummaryStore } from "../state/summary";

/**
 * Mount the global Tauri event bridge exactly once.
 *
 * This hook subscribes to the `"app-event-payload"` event and dispatches
 * each payload into the Zustand stores via their `handleEvent` methods.
 * It must be called from a component that is always mounted (i.e. `App`),
 * never inside a conditionally-rendered subtree.
 *
 * The unlisten function returned by `listenAppEvents` is called on cleanup,
 * so the listener is removed when the component unmounts (HMR, tests, etc.).
 */
export function useAppEventBridge(): void {
  const handleRecordingEvent = useRecordingStore((s) => s.handleEvent);
  const handleModelsEvent = useModelsStore((s) => s.handleEvent);
  const handleSummaryEvent = useSummaryStore((s) => s.handleEvent);

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    listenAppEvents((event) => {
      handleRecordingEvent(event);
      handleModelsEvent(event);
      handleSummaryEvent(event);
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
  // The handleEvent functions are stable Zustand store references; the dep
  // array is intentionally exhaustive.
  }, [handleRecordingEvent, handleModelsEvent, handleSummaryEvent]);
}
