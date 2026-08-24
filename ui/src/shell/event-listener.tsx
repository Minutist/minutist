import { useEffect } from "react";
import { listenAppEvents } from "../ipc/client";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useSummaryStore } from "../state/summary";
import { useMeetingsStore } from "../state/meetings";
import { useChatStore } from "../state/chat";
import { useAttachmentsStore } from "../state/attachments";
import { useMcpServerInfoStore } from "../state/mcp-server-info";
import { useAccountStatusStore } from "../state/account-status";
import { useSyncStatusStore } from "../state/sync-status";
import { useOperationProgressStore } from "../state/operation-progress";
import { useTranslationsStore } from "../state/translations";
import { useUpdateStore } from "../state/update";
import { useLiveDigestStore } from "../state/liveDigest";

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
  const handleMeetingsEvent = useMeetingsStore((s) => s.handleEvent);
  const handleChatEvent = useChatStore((s) => s.handleEvent);
  const handleAttachmentsEvent = useAttachmentsStore((s) => s.handleEvent);
  const handleMcpServerInfoEvent = useMcpServerInfoStore((s) => s.handleEvent);
  const handleAccountStatusEvent = useAccountStatusStore((s) => s.handleEvent);
  const handleSyncStatusEvent = useSyncStatusStore((s) => s.handleEvent);
  const handleOperationProgressEvent = useOperationProgressStore(
    (s) => s.handleEvent,
  );
  const handleTranslationsEvent = useTranslationsStore((s) => s.handleEvent);
  const handleUpdateEvent = useUpdateStore((s) => s.handleEvent);
  const handleLiveDigestEvent = useLiveDigestStore((s) => s.handleEvent);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    // Guards against a mount → unmount → remount race (e.g. React StrictMode's
    // deliberate double-invoke): if cleanup runs before `listenAppEvents`
    // resolves, `unlisten` is still `undefined` when the cleanup fires, so the
    // `unlisten?.()` below is a no-op and the listener that resolves a moment
    // later is never removed — a leaked duplicate subscription. Once cleanup
    // has run, unlisten the just-resolved listener immediately instead of
    // stashing it.
    let cancelled = false;

    listenAppEvents((event) => {
      handleRecordingEvent(event);
      handleModelsEvent(event);
      handleSummaryEvent(event);
      handleMeetingsEvent(event);
      handleChatEvent(event);
      handleAttachmentsEvent(event);
      handleMcpServerInfoEvent(event);
      handleAccountStatusEvent(event);
      handleSyncStatusEvent(event);
      handleOperationProgressEvent(event);
      handleTranslationsEvent(event);
      handleUpdateEvent(event);
      handleLiveDigestEvent(event);
    })
      .then((fn) => {
        if (cancelled) {
          fn();
          return;
        }
        unlisten = fn;
      })
      .catch((err: unknown) => {
        console.error("[event-listener] failed to subscribe:", err);
      });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  // The handleEvent functions are stable Zustand store references; the dep
  // array is intentionally exhaustive.
  }, [
    handleRecordingEvent,
    handleModelsEvent,
    handleSummaryEvent,
    handleMeetingsEvent,
    handleChatEvent,
    handleAttachmentsEvent,
    handleMcpServerInfoEvent,
    handleAccountStatusEvent,
    handleSyncStatusEvent,
    handleOperationProgressEvent,
    handleTranslationsEvent,
    handleUpdateEvent,
    handleLiveDigestEvent,
  ]);
}
