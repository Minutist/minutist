import { useEffect, useLayoutEffect, useState } from "react";
import type { ReactNode } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { useRecordingStore } from "../state/recording";
import { useModelsStore } from "../state/models";
import { useMeetingsStore } from "../state/meetings";
import { MeetingControls } from "./MeetingControls";
import { AudioMeter } from "./AudioMeter";
import { ModelDownloadStatus } from "./ModelDownloadStatus";
import { RecordingStatus } from "./RecordingStatus";
import { MeetingList } from "./MeetingList";
import { SummaryView } from "./SummaryView";
import { ChatView } from "./ChatView";
import { SettingsDrawer } from "./SettingsDrawer";
import { About } from "./About";
import { Editor } from "../editor/Editor";
import { TranscriptPane } from "../transcript/TranscriptPane";
import "./MainWindow.css";

/**
 * Root shell component.
 *
 * Layout:
 *   Header  — wordmark + recording status; transport, audio meter, the
 *             pane-visibility toggle, and Settings.
 *   Body    — a horizontal `react-resizable-panels` Group of up to three
 *             columns the user can show/hide and drag-resize (FR-21/FR-30):
 *               • Notes editor (primary).
 *               • Transcript pane.
 *               • Summary view — a reading column (NOT a popup overlay).
 *
 * Panes are shown/hidden by INCLUDING/EXCLUDING their `Panel` from the Group
 * (with a `Separator` between each pair of visible panes), rather than
 * collapsing to zero width — this keeps a single drag handle between any two
 * visible panes and avoids stacked separators around a hidden middle pane. The
 * percentage min-sizes sum to well under 100 %, so the columns squeeze to fit
 * and the workspace never scrolls horizontally.
 *
 * Per-mode defaults: the live transcript is hidden by default in both modes (a
 * scrolling transcript distracts from note-taking; it is one click away on the
 * toggle). A FINISHED opened meeting defaults to notes + summary (the summary is
 * what you reach for after a meeting); a LIVE recording defaults to notes only.
 *
 * The global event bridge (`useAppEventBridge`) is mounted one level up in
 * `App.tsx` so it cannot be accidentally unmounted when this component
 * re-renders.
 */
/**
 * Interleave resize separators between the visible panes. `slots` carries one
 * entry per pane (a falsy entry = a hidden pane); the result is the visible
 * panels with a single `Separator` between each adjacent pair — so there is
 * exactly one drag handle between any two columns and none stranded beside a
 * hidden pane.
 */
function buildPanes(slots: ReactNode[]): ReactNode[] {
  const panels = slots.filter(Boolean);
  const out: ReactNode[] = [];
  panels.forEach((panel, i) => {
    if (i > 0) {
      out.push(
        <Separator key={`sep-${i}`} className="main-window__resize-handle">
          <span className="main-window__resize-grip" aria-hidden="true" />
        </Separator>,
      );
    }
    out.push(panel);
  });
  return out;
}

export function MainWindow() {
  const refreshDevices = useRecordingStore((s) => s.refreshDevices);
  const refreshSettings = useRecordingStore((s) => s.refreshSettings);
  const prewarmAsr = useRecordingStore((s) => s.prewarmAsr);
  const lastError = useRecordingStore((s) => s.lastError);
  const recordingState = useRecordingStore((s) => s.state);
  const refreshModels = useModelsStore((s) => s.refreshModels);
  const openMeetingId = useMeetingsStore((s) => s.openMeetingId);
  const closeMeeting = useMeetingsStore((s) => s.close);
  const reDiarize = useMeetingsStore((s) => s.rediarize);
  // Re-processing (re-transcribe / re-diarize) is a rare action that lives in
  // the opened-meeting view, not on the meeting list — see MeetingList.
  const reTranscribe = useMeetingsStore((s) => s.reTranscribe);

  // The recorder is actively capturing (or in the brief stopping handoff). Live-
  // test UX T3: `finalising` is deliberately EXCLUDED — once the meeting is
  // finalised the heavy background passes (re-transcribe / re-identify-speakers)
  // run in the background and must NOT keep the recording window open. They
  // surface as a non-blocking per-row indicator on the meeting list instead.
  const isLiveRecording =
    recordingState.kind === "recording" ||
    recordingState.kind === "paused" ||
    recordingState.kind === "stopping";

  // The meeting-list is the entry surface (FR-33): shown when no meeting is
  // open and the recorder is not actively capturing. Opening a meeting, or
  // starting a recording, switches to the editor/transcript workspace; on stop
  // we return to the list as soon as the meeting finalises (T3).
  const inWorkspace = openMeetingId !== null || isLiveRecording;

  // The meeting the workspace is operating on: the opened saved meeting, else
  // the live recording's meeting (recording / paused / stopping carry the id).
  // This is the meeting the summary view summarises.
  const activeMeetingId =
    openMeetingId ?? (isLiveRecording ? recordingState.meeting_id : null);

  // A finished, opened meeting: idle and viewing a saved meeting. Only then is a
  // summary meaningful (you cannot summarise a recording mid-flight), so the
  // summary column is offered only in this mode.
  const isFinishedMeeting =
    openMeetingId !== null && recordingState.kind === "idle";
  const showSummaryPane = isFinishedMeeting;

  // Phase 9 — the chat agent is meeting-scoped: offered whenever the workspace
  // is operating on a concrete meeting (a live recording's meeting or an opened
  // saved meeting), and hidden on the meeting-list entry surface. It needs the
  // `activeMeetingId` to scope the conversation.
  const showChatPane = inWorkspace && activeMeetingId !== null;

  // Pane visibility (FR-21/FR-30). Panes are included/excluded from the Group
  // rather than collapsed to zero width. Reset to the per-mode default whenever
  // the workspace is (re-)entered or the mode flips (see the effect below).
  const [notesShown, setNotesShown] = useState(true);
  const [transcriptShown, setTranscriptShown] = useState(true);
  const [summaryShown, setSummaryShown] = useState(true);
  // Chat is OFF by default (notes are the primary surface); one click on the
  // toggle reveals it.
  const [chatShown, setChatShown] = useState(false);

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
    // Live-test UX T2: pre-warm the ASR model when the workspace opens so the
    // first record is not a cold ~29 s load. Fire-and-forget + idempotent.
    void prewarmAsr();
  }, [refreshDevices, refreshSettings, refreshModels, prewarmAsr]);

  // Reset pane visibility to the per-mode default. The deps are deliberately
  // [inWorkspace, showSummaryPane] only: this fires on entering the workspace
  // and on the finished-meeting↔live-recording mode flip, but NOT on benign
  // re-renders within a mode (e.g. pause/resume), so a transcript the user
  // revealed stays open. (Do NOT add `openMeetingId` here — it would re-default
  // the panes on every meeting switch, which already happens for free via the
  // meeting list remounting the workspace.) The live transcript is HIDDEN by
  // default in both modes — a scrolling transcript distracts from note-taking —
  // and is one click away on the toggle. Finished: notes + summary. Recording:
  // notes only. A LAYOUT effect so the corrected visibility commits before paint
  // (the workspace never flashes the all-panes initial state for a frame).
  useLayoutEffect(() => {
    if (!inWorkspace) return;
    setNotesShown(true);
    setTranscriptShown(false);
    setSummaryShown(true); // only rendered when showSummaryPane (finished meeting)
    setChatShown(false); // chat is one click away on the toggle
  }, [inWorkspace, showSummaryPane]);

  // How many panes are currently visible — used to forbid hiding the last one.
  const visibleCount =
    (notesShown ? 1 : 0) +
    (transcriptShown ? 1 : 0) +
    (showSummaryPane && summaryShown ? 1 : 0) +
    (showChatPane && chatShown ? 1 : 0);

  function togglePane(
    shown: boolean,
    setShown: (next: boolean) => void,
  ) {
    // Never hide the last visible pane — the workspace must show something.
    if (shown && visibleCount <= 1) return;
    setShown(!shown);
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
              className="main-window__header-btn"
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
              className="main-window__header-btn main-window__reprocess"
              onClick={() => void reTranscribe(openMeetingId)}
              title="Re-run speech recognition on this recording (rare; e.g. after changing the language or model)."
            >
              Re-transcribe
            </button>
          )}
          {openMeetingId !== null && recordingState.kind === "idle" && (
            <button
              type="button"
              className="main-window__header-btn main-window__reprocess"
              onClick={() => void reDiarize(openMeetingId)}
              title="Re-run speaker identification on this recording (rare)."
            >
              Re-identify speakers
            </button>
          )}
          <MeetingControls />
          <div className="main-window__meter" aria-label="Audio level">
            <AudioMeter />
          </div>
          {/*
            Pane-visibility toggle (FR-21/FR-30): a segmented control that shows
            or hides each column. A pressed segment = the pane is visible. The
            Summary segment appears only for a finished, opened meeting; the last
            visible pane cannot be hidden (see `togglePane`).
          */}
          {inWorkspace && (
            <div
              className="main-window__view-toggle"
              role="group"
              aria-label="Visible panes"
            >
              <button
                type="button"
                className="main-window__view-seg"
                aria-pressed={notesShown}
                onClick={() => togglePane(notesShown, setNotesShown)}
              >
                Notes
              </button>
              <button
                type="button"
                className="main-window__view-seg"
                aria-pressed={transcriptShown}
                onClick={() => togglePane(transcriptShown, setTranscriptShown)}
              >
                Transcript
              </button>
              {showSummaryPane && (
                <button
                  type="button"
                  className="main-window__view-seg"
                  aria-pressed={summaryShown}
                  onClick={() => togglePane(summaryShown, setSummaryShown)}
                >
                  Summary
                </button>
              )}
              {showChatPane && (
                <button
                  type="button"
                  className="main-window__view-seg"
                  aria-pressed={chatShown}
                  onClick={() => togglePane(chatShown, setChatShown)}
                >
                  Chat
                </button>
              )}
            </div>
          )}
          {/*
            Settings — opens the drawer holding the appearance / capture /
            processing configuration (theme, ruled paper, device, language,
            diarize-on-stop, GPU, system audio) + the About affordance. Adds no
            command; the drawer's controls route through the existing settings
            seams.
          */}
          <button
            type="button"
            className="main-window__header-btn"
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
          {buildPanes([
            notesShown && (
              <Panel
                key="notes"
                id="notes"
                className="main-window__pane main-window__pane--notes"
                minSize="22%"
                defaultSize={showSummaryPane ? "46%" : "62%"}
              >
                <Editor />
              </Panel>
            ),
            transcriptShown && (
              <Panel
                key="transcript"
                id="transcript"
                className="main-window__pane main-window__pane--transcript"
                minSize="16%"
                defaultSize="34%"
              >
                <TranscriptPane />
              </Panel>
            ),
            showSummaryPane && summaryShown && activeMeetingId !== null && (
              <Panel
                key="summary"
                id="summary"
                className="main-window__pane main-window__pane--summary"
                minSize="20%"
                defaultSize="40%"
              >
                <SummaryView meetingId={activeMeetingId} />
              </Panel>
            ),
            showChatPane && chatShown && activeMeetingId !== null && (
              <Panel
                key="chat"
                id="chat"
                className="main-window__pane main-window__pane--chat"
                minSize="20%"
                defaultSize="34%"
              >
                <ChatView meetingId={activeMeetingId} />
              </Panel>
            ),
          ])}
        </Group>
      ) : (
        <MeetingList />
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
