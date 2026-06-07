import { useEffect, useState } from "react";
import { Group, Panel, Separator, usePanelRef } from "react-resizable-panels";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useMeetingsStore } from "../state/meetings";
import { MeetingControls } from "./MeetingControls";
import { AudioMeter } from "./AudioMeter";
import { ModelDownloadStatus } from "./ModelDownloadStatus";
import { RecordingStatus } from "./RecordingStatus";
import { MeetingList } from "./MeetingList";
import { SummaryView } from "./SummaryView";
import { SettingsDrawer } from "./SettingsDrawer";
import { About } from "./About";
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
  const reDiarize = useMeetingsStore((s) => s.rediarize);
  // Re-processing (re-transcribe / re-diarize) is a rare action that lives in
  // the opened-meeting view, not on the meeting list — see MeetingList.
  const reTranscribe = useMeetingsStore((s) => s.reTranscribe);

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
  // The summary is hidden by default; the header toggle reveals it (FR-30) as a
  // reading-width overlay sheet rather than a cramped third pane.
  const [summaryOpen, setSummaryOpen] = useState(false);
  // The About dialog (Phase 7, S6) is hidden by default; a header affordance
  // opens it. Presentational overlay; closing returns to the prior surface.
  const [aboutOpen, setAboutOpen] = useState(false);
  // The Settings drawer holds the capture / processing configuration that used
  // to crowd the top bar (device, language, diarize-on-stop, GPU, system audio).
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    // Load persisted settings first so `selectedDeviceId` reflects the
    // user's saved choice; then enumerate devices so the picker is
    // populated. Fetch model list so ModelDownloadStatus renders correctly.
    void refreshSettings();
    void refreshDevices();
    void refreshModels();
  }, [refreshDevices, refreshSettings, refreshModels]);

  // Close the summary overlay on Escape — parity with the About dialog and the
  // Settings drawer.
  useEffect(() => {
    if (!summaryOpen) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setSummaryOpen(false);
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [summaryOpen]);

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
        {/*
          Lead group — wordmark + recording status, left-aligned together so the
          masthead reads as a single coherent row (brand/status left, actions
          right) instead of a centre-anchored grid whose right cluster wraps.
        */}
        <div className="main-window__lead">
          <span className="main-window__wordmark">meeting-app</span>
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
          {/*
            Phase 6 — re-diarize the OPEN saved meeting from its workspace menu.
            Shown only when a saved meeting is open and nothing is recording (a
            re-diarize must not contend with the live pipeline; the backend
            refuses unless `Idle`). The `diarization_complete` event re-reads the
            open meeting's transcript so the speaker chips appear.
          */}
          {openMeetingId !== null && recordingState.kind === "idle" && (
            <button
              type="button"
              className="main-window__toggle-transcript main-window__reprocess"
              onClick={() => void reTranscribe(openMeetingId)}
              title="Re-run speech recognition on this recording (rare; e.g. after changing the language or model)."
            >
              Re-transcribe
            </button>
          )}
          {openMeetingId !== null && recordingState.kind === "idle" && (
            <button
              type="button"
              className="main-window__toggle-transcript main-window__reprocess"
              onClick={() => void reDiarize(openMeetingId)}
              title="Re-run speaker diarization on this recording (rare)."
            >
              Re-diarize
            </button>
          )}
          <MeetingControls />
          <div className="main-window__meter" aria-label="Audio level">
            <AudioMeter />
          </div>
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
          {/*
            Settings — opens the drawer holding the capture / processing
            configuration (device, language, diarize-on-stop, GPU, system audio)
            that used to crowd the bar. Adds no command; the drawer's controls
            route through the existing settings seams.
          */}
          {/*
            Settings — opens the drawer holding the capture / processing
            configuration (device, language, diarize-on-stop, GPU, system audio)
            + the About affordance (product info, rarely opened, so it lives in
            the drawer rather than crowding the bar). Adds no command; the
            drawer's controls route through the existing settings seams.
          */}
          <button
            type="button"
            className="main-window__toggle-transcript"
            aria-haspopup="dialog"
            aria-expanded={settingsOpen}
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </button>
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
        </Group>
      ) : (
        <MeetingList />
      )}

      {/*
        Summary (FR-30) as a reading-width overlay sheet rather than a third
        body pane: a serif summary column does not fit alongside notes +
        transcript at a comfortable measure. Scrim click / the header toggle
        dismiss it; the SummaryView owns its own Summarise / Edit actions.
      */}
      {summaryOpen && activeMeetingId !== null && (
        <div
          className="summary-overlay"
          onClick={() => setSummaryOpen(false)}
        >
          <div
            className="summary-overlay__sheet ink-reveal"
            role="dialog"
            aria-modal="true"
            aria-label="Meeting summary"
            onClick={(e) => e.stopPropagation()}
          >
            <SummaryView meetingId={activeMeetingId} />
          </div>
        </div>
      )}

      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onAbout={() => {
          setSettingsOpen(false);
          setAboutOpen(true);
        }}
      />

      {aboutOpen && <About onClose={() => setAboutOpen(false)} />}
    </div>
  );
}
