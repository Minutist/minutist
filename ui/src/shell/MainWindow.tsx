import { useEffect, useRef, useState } from "react";
import { Group, Panel, Separator, usePanelRef } from "react-resizable-panels";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { DevicePicker } from "./DevicePicker";
import { MeetingControls } from "./MeetingControls";
import { AudioMeter } from "./AudioMeter";
import { ModelDownloadStatus } from "./ModelDownloadStatus";
import { Editor } from "../editor/Editor";
import { TranscriptPane } from "../transcript/TranscriptPane";
import "./MainWindow.css";

/**
 * Root shell component.
 *
 * Layout:
 *   Header  — title + controls (model download, device picker, transport,
 *             audio meter, error banner).
 *   Body    — a horizontal `react-resizable-panels` Group with two panels:
 *               • Notes editor (primary) — the main view.
 *               • Transcript pane (secondary) — collapsible AND resizable.
 *
 * The transcript panel is `collapsible`; a toolbar button toggles it via the
 * panel's imperative `collapse()` / `expand()` handle, and the `Separator`
 * between the panels lets the user drag to resize (FR-21).
 *
 * The global event bridge (`useAppEventBridge`) is mounted one level up in
 * `App.tsx` so it cannot be accidentally unmounted when this component
 * re-renders.
 */
export function MainWindow() {
  const refreshDevices = useRecordingStore((s) => s.refreshDevices);
  const refreshSettings = useRecordingStore((s) => s.refreshSettings);
  const lastError = useRecordingStore((s) => s.lastError);
  const refreshModels = useModelsStore((s) => s.refreshModels);

  const transcriptPanelRef = usePanelRef();
  const [transcriptCollapsed, setTranscriptCollapsed] = useState(false);

  useEffect(() => {
    // Load persisted settings first so `selectedDeviceId` reflects the
    // user's saved choice; then enumerate devices so the picker is
    // populated. Fetch model list so ModelDownloadStatus renders correctly.
    void refreshSettings();
    void refreshDevices();
    void refreshModels();
  }, [refreshDevices, refreshSettings, refreshModels]);

  function toggleTranscript() {
    const handle = transcriptPanelRef.current;
    if (!handle) return;
    if (handle.isCollapsed()) {
      handle.expand();
      setTranscriptCollapsed(false);
    } else {
      handle.collapse();
      setTranscriptCollapsed(true);
    }
  }

  return (
    <div className="main-window">
      <header className="main-window__header">
        <div className="main-window__title-row">
          <h1>meeting-app</h1>
          <button
            type="button"
            className="main-window__toggle-transcript"
            aria-pressed={transcriptCollapsed}
            onClick={toggleTranscript}
          >
            {transcriptCollapsed ? "Show transcript" : "Hide transcript"}
          </button>
        </div>

        <ModelDownloadStatus />

        <div className="main-window__controls">
          <DevicePicker />
          <MeetingControls />
        </div>

        <section className="main-window__meter">
          <label>Audio level</label>
          <AudioMeter />
        </section>

        {lastError && (
          <div className="main-window__error" role="alert">
            {lastError}
          </div>
        )}
      </header>

      <Group orientation="horizontal" className="main-window__panes">
        <Panel
          id="notes"
          className="main-window__pane main-window__pane--notes"
          minSize="20%"
        >
          <Editor />
        </Panel>

        <Separator className="main-window__resize-handle" />

        <Panel
          id="transcript"
          panelRef={transcriptPanelRef}
          className="main-window__pane main-window__pane--transcript"
          collapsible
          collapsedSize="0%"
          minSize="15%"
          defaultSize="35%"
          onResize={(size) => setTranscriptCollapsed(size.asPercentage <= 0)}
        >
          <TranscriptPane />
        </Panel>
      </Group>
    </div>
  );
}
