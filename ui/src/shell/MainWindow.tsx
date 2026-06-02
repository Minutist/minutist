import { useEffect, useState } from "react";
import { Group, Panel, Separator, usePanelRef } from "react-resizable-panels";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useMeetingsStore } from "../state/meetings";
import { DevicePicker } from "./DevicePicker";
import { MeetingControls } from "./MeetingControls";
import { AudioMeter } from "./AudioMeter";
import { ModelDownloadStatus } from "./ModelDownloadStatus";
import { RecordingStatus } from "./RecordingStatus";
import { MeetingList } from "./MeetingList";
import { SummaryView } from "./SummaryView";
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
  const recordingState = useRecordingStore((s) => s.state);
  const refreshModels = useModelsStore((s) => s.refreshModels);
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);
  const closeMeeting = useMeetingsStore((s) => s.close);

  // The meeting-list is the entry surface (FR-33): shown when no meeting is
  // open and nothing is being recorded. Opening a meeting, or starting a
  // recording, switches to the editor/transcript workspace.
  const inWorkspace = openMeetingId !== null || recordingState.kind !== "idle";

  // The meeting the workspace is operating on: the opened saved meeting, else
  // the live recording's meeting (recording / paused / stopping carry the id).
  // This is the meeting the summary view summarises.
  const activeMeetingId =
    openMeetingId ??
    (recordingState.kind !== "idle" ? recordingState.meeting_id : null);

  const transcriptPanelRef = usePanelRef();
  const [transcriptCollapsed, setTranscriptCollapsed] = useState(false);
  // The summary panel is hidden by default; the header toggle reveals it (FR-30).
  const [summaryOpen, setSummaryOpen] = useState(false);

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
      {/*
        Top bar — calm, hairline-ruled. Left: wordmark. Centre/focal:
        recording status (oxblood dot + elapsed clock). Right: grouped
        transport, audio meter, and device affordance. The most-used action
        (Record/Stop) is the strongest control in MeetingControls; Pause is
        quieter.
      */}
      <header className="main-window__topbar ink-reveal">
        <div className="main-window__brand">
          <span className="main-window__wordmark">meeting-app</span>
        </div>

        <div className="main-window__status">
          <RecordingStatus />
        </div>

        <div className="main-window__controls">
          {/*
            Return to the meeting-list (FR-33 entry surface). Shown only when a
            meeting is open AND nothing is recording — leaving an open meeting
            mid-recording would be ambiguous.
          */}
          {inWorkspace && openMeetingId !== null && recordingState.kind === "idle" && (
            <button
              type="button"
              className="main-window__toggle-transcript"
              onClick={closeMeeting}
            >
              Meetings
            </button>
          )}
          <MeetingControls />
          <div className="main-window__meter" aria-label="Audio level">
            <AudioMeter />
          </div>
          <DevicePicker />
          {inWorkspace && (
            <button
              type="button"
              className="main-window__toggle-transcript"
              aria-pressed={transcriptCollapsed}
              onClick={toggleTranscript}
            >
              {transcriptCollapsed ? "Show transcript" : "Hide transcript"}
            </button>
          )}
          {inWorkspace && activeMeetingId !== null && (
            <button
              type="button"
              className="main-window__toggle-transcript"
              aria-pressed={summaryOpen}
              onClick={() => setSummaryOpen((open) => !open)}
            >
              {summaryOpen ? "Hide summary" : "Summary"}
            </button>
          )}
        </div>
      </header>

      {/*
        Restrained chrome strip below the bar: first-run model download status
        (self-hides once the ASR model is ready) + recoverable errors. Empty
        in the steady state so it does not compete with the writing surface.
      */}
      <div className="main-window__chrome">
        <ModelDownloadStatus />
        {lastError && (
          <div className="main-window__error" role="alert">
            {lastError}
          </div>
        )}
      </div>

      {inWorkspace ? (
        <Group orientation="horizontal" className="main-window__panes ink-reveal">
          <Panel
            id="notes"
            className="main-window__pane main-window__pane--notes"
            minSize="20%"
          >
            <Editor />
          </Panel>

          <Separator className="main-window__resize-handle">
            <span className="main-window__resize-grip" aria-hidden="true" />
          </Separator>

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

          {summaryOpen && activeMeetingId !== null && (
            <>
              <Separator className="main-window__resize-handle">
                <span
                  className="main-window__resize-grip"
                  aria-hidden="true"
                />
              </Separator>
              <Panel
                id="summary"
                className="main-window__pane main-window__pane--summary"
                minSize="20%"
                defaultSize="35%"
              >
                <SummaryView meetingId={activeMeetingId} />
              </Panel>
            </>
          )}
        </Group>
      ) : (
        <MeetingList />
      )}
    </div>
  );
}
