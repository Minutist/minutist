/**
 * DEV-ONLY browser shim for the IPC seam.
 *
 * When the app runs under `vite dev` in a plain browser (no Tauri backend,
 * `window.__TAURI_INTERNALS__` absent), the generated `commands` call
 * `TAURI_INVOKE`, which rejects because there is no Rust side to answer. This
 * module supplies representative sample data so the full themed screen renders
 * for visual QA / screenshots: a meeting in "recording" state with an elapsed
 * clock, ~6 transcript segments, and seeded notes (a heading + paragraphs,
 * including an anchored paragraph so the left-margin timestamp marginalia
 * shows).
 *
 * Guarded by {@link shouldUseDevShim}: it only activates when
 * `import.meta.env.DEV` is true, the runner is NOT Vitest
 * (`import.meta.env.MODE !== "test"`), AND the Tauri global is absent.
 * Production builds (`import.meta.env.DEV === false`) and the Vitest suite
 * (which mocks `../ipc/bindings` directly and asserts on it) are unaffected.
 */
import type {
  AppEvent,
  AudioDevice,
  ChatSession,
  ChatSessionId,
  ModelStatus,
  RecordingState,
  Result,
  Settings,
  Segment,
  IpcError,
  MeetingId,
  MeetingMeta,
  NotesDocument,
} from "./bindings";
import type { MeetingListEntry, MeetingState } from "./meetings";

// The activation guard lives in its own data-free module so the main bundle
// can import it without pulling in any of the sample data below. Re-exported
// here for callers that already have a `dev-shim` import in hand.
export { shouldUseDevShim } from "./dev-shim-guard";

const DEV_MEETING_ID: MeetingId = "dev-meeting-0001";
const ASR_MODEL_ID = "qwen3-asr-0.6b-q8_0";

// A recording started ~7 minutes ago (wall clock), for elapsed-time display.
const STARTED_AT_MS = Date.now() - 7 * 60 * 1000 - 24 * 1000;

const DEV_STATE: RecordingState = {
  kind: "recording",
  meeting_id: DEV_MEETING_ID,
  started_at_ms: STARTED_AT_MS,
};

const DEV_DEVICES: AudioDevice[] = [
  { id: "builtin-mic", name: "MacBook Pro Microphone", is_default: true },
  { id: "usb-podmic", name: "PodMic USB", is_default: false },
  { id: "aggregate", name: "Aggregate Device (2ch)", is_default: false },
];

// DEV-ONLY sample data mirroring resources/models.json — keep in step with the
// manifest when models change. This is the one remaining hand-kept model list,
// and it cannot affect the shipped About dialog (production derives its list
// from the real `list_models` IPC); it only seeds the `vite dev` browser shim.
const DEV_MODELS: ModelStatus[] = [
  {
    id: ASR_MODEL_ID,
    kind: "asr",
    display_name: "Qwen3-ASR 0.6B (Q8_0)",
    status: { state: "available", local_dir: "/dev/models/asr/qwen3-asr" },
    license: "apache-2.0",
  },
  {
    id: "parakeet-tdt-0.6b-v3-int8",
    kind: "asr",
    display_name: "Parakeet TDT 0.6B v3 (int8)",
    status: { state: "available", local_dir: "/dev/models/asr/parakeet-tdt-0.6b-v3" },
    license: "cc-by-4.0",
  },
  {
    id: "qwen3-asr-1.7b-q8_0",
    kind: "asr",
    display_name: "Qwen3-ASR 1.7B (Q8_0)",
    status: { state: "available", local_dir: "/dev/models/asr/qwen3-asr-1.7b" },
    license: "apache-2.0",
  },
  {
    id: "gemma-4-e4b-it-q4_k_m",
    kind: "llm",
    display_name: "Gemma 4 E4B Instruct (Q4_K_M)",
    status: { state: "available", local_dir: "/dev/models/llm/gemma-4-e4b" },
    license: "apache-2.0",
  },
  {
    id: "pyannote-segmentation-3-0",
    kind: "diarize",
    display_name: "pyannote segmentation 3.0",
    status: { state: "available", local_dir: "/dev/models/diarize/pyannote" },
    license: "mit",
  },
  {
    id: "3dspeaker-campplus-zh-en-advanced",
    kind: "diarize",
    display_name: "3D-Speaker CAM++ (zh-en 16k common, advanced)",
    status: { state: "available", local_dir: "/dev/models/diarize/3dspeaker" },
    license: "apache-2.0",
  },
];

const DEV_SETTINGS: Settings = {
  input_device_id: "builtin-mic",
  theme: "light",
  data_directory: null,
  start_hidden: false,
  autosave_interval_secs: 5,
  // Phase 6 diarization toggle. Production default is OFF; the dev shim seeds
  // it ON so the toggle reads as checked under `vite dev` for visual QA. Now a
  // first-class `Settings` field (the Phase-6 JOIN regenerated `bindings.ts`),
  // so the previous augmentation cast collapsed.
  diarization_enabled: true,
  // Phase 7 onboarding gate. Seeded complete so `vite dev` lands on the main
  // app for visual QA; the onboarding flow itself renders in the shim by
  // starting from a non-completed settings snapshot (see Onboarding.tsx note).
  onboarding_completed: true,
  // ASR language hint. Seeded "English" (the schema default) so the language
  // picker renders selected under `vite dev`.
  transcription_language: "English",
  // Phase 8 hybrid-ASR GPU tier opt-in (off by default).
  prefer_large_asr_model: false,
  // Notes-editor writing-paper rules (on by default) — seeded on so the ruled
  // sheet renders under `vite dev` for visual QA.
  notes_paper_rules: true,
};

/**
 * Recording-clock offsets (ms, pause-excluding) for the seeded transcript.
 *
 * Carries `speaker_id`s (first-seen-order labels `A`/`B`, per the diarizer's
 * normalisation) so the transcript pane's per-row "Speaker {id}" chips render
 * under `vite dev` for visual QA of the Phase-6 diarization overlay.
 */
const DEV_TRANSCRIPT: Segment[] = [
  {
    start_ms: 4_200,
    end_ms: 9_800,
    text: "Right, let's get going — thanks everyone for making the time today.",
    speaker_id: "A",
    words: [],
  },
  {
    start_ms: 12_400,
    end_ms: 21_300,
    text: "First item is the launch checklist. We're tracking three open risks against the date.",
    speaker_id: "A",
    words: [],
  },
  {
    start_ms: 24_100,
    end_ms: 33_900,
    text: "The big one is the offline model download — first-run experience needs to feel deliberate, not broken.",
    speaker_id: "B",
    words: [],
  },
  {
    start_ms: 38_600,
    end_ms: 47_200,
    text: "Agreed. I'll own the progress affordance and the retry path. Should land this week.",
    speaker_id: "A",
    words: [],
  },
  {
    start_ms: 51_000,
    end_ms: 61_700,
    text: "Second, the transcript and notes need to read as one document, not two competing panes.",
    speaker_id: "B",
    words: [],
  },
  {
    start_ms: 64_300,
    end_ms: 74_100,
    text: "That's the editorial direction — warm paper, one ink accent, marginal timestamps. Sign-off pending.",
    speaker_id: "B",
    words: [],
  },
];

/**
 * Seeded Tiptap document. One heading + several paragraphs; the second
 * paragraph carries `data-anchor-ms` so the left-margin timestamp marginalia
 * renders without an active recording-clock keystroke.
 */
const DEV_NOTES_JSON = JSON.stringify({
  type: "doc",
  content: [
    {
      type: "heading",
      attrs: { level: 1 },
      content: [{ type: "text", text: "Launch sync — Tuesday" }],
    },
    {
      type: "paragraph",
      content: [
        {
          type: "text",
          text: "Three open risks against the date; offline model download is the one to watch. First-run should feel deliberate.",
        },
      ],
    },
    {
      type: "paragraph",
      attrs: { "data-anchor-ms": 24_100 },
      content: [
        {
          type: "text",
          text: "Owner: progress affordance + retry path. Target: this week.",
        },
      ],
    },
    {
      type: "heading",
      attrs: { level: 2 },
      content: [{ type: "text", text: "Design direction" }],
    },
    {
      type: "paragraph",
      content: [
        {
          type: "text",
          text: "Transcript and notes should read as one document — warm paper, a single oxblood ink accent, timestamps as quiet marginalia.",
        },
      ],
    },
    {
      type: "blockquote",
      content: [
        {
          type: "paragraph",
          content: [
            {
              type: "text",
              text: "It should feel like writing on a fine sheet of paper on a warm desk.",
            },
          ],
        },
      ],
    },
  ],
});

const DEV_NOTES_MD =
  "# Launch sync — Tuesday\n\nThree open risks against the date; offline model download is the one to watch. First-run should feel deliberate.\n\nOwner: progress affordance + retry path. Target: this week.\n\n## Design direction\n\nTranscript and notes should read as one document — warm paper, a single oxblood ink accent, timestamps as quiet marginalia.\n\n> It should feel like writing on a fine sheet of paper on a warm desk.\n";

/**
 * Sample meeting summary (Phase 5, FR-30) so the summary view renders populated
 * under `vite dev`. Markdown with a heading, prose, a decisions list, and an
 * action-items list — the shape the summariser produces from a transcript +
 * notes. Keyed per dev meeting id in {@link devSummaries} so opening different
 * meetings shows distinct content.
 */
const DEV_SUMMARY_MD = `## Summary

The launch sync covered the release checklist with three open risks against the date. The standout concern is the offline model download: the first-run experience needs to feel deliberate rather than broken, since users abandoned earlier builds before the model finished.

### Decisions

- Ship the deliberate first-run progress affordance with an explicit retry path.
- Hold the editorial direction: warm paper, a single oxblood ink accent, timestamps as quiet marginalia.

### Action items

- **Owner** — build the progress affordance and retry path (target: this week).
- Confirm sign-off on the two-pane sheet/transcript treatment.
`;

/** Per-meeting sample summaries; meetings without an entry have no summary yet. */
const devSummaries = new Map<MeetingId, string>([
  [DEV_MEETING_ID, DEV_SUMMARY_MD],
  ["dev-meeting-0002", "## Summary\n\nDesign review signed off the Editorial Ink treatment: warm paper, one oxblood accent, marginal timestamps.\n"],
]);

/**
 * Sample meeting-list rows (FR-33) so the entry surface renders populated under
 * `vite dev`. Dates span a few weeks; durations / speaker counts vary; each has
 * a transcript excerpt for the row preview.
 */
const DEV_MEETINGS: MeetingListEntry[] = [
  {
    id: DEV_MEETING_ID,
    title: "Launch sync — Tuesday",
    started_at: new Date(STARTED_AT_MS).toISOString(),
    duration_ms: 32 * 60 * 1000,
    speaker_count: 3,
    excerpt:
      "Three open risks against the date; offline model download is the one to watch. Owner and retry path agreed for this week.",
  },
  {
    id: "dev-meeting-0002",
    title: "Design review — Editorial Ink",
    started_at: "2026-05-26T14:05:00Z",
    duration_ms: 47 * 60 * 1000,
    speaker_count: 2,
    excerpt:
      "Warm paper, a single oxblood ink accent, timestamps as quiet marginalia. Sign-off on the two-pane sheet treatment.",
  },
  {
    id: "dev-meeting-0003",
    title: "Customer interview — onboarding friction",
    started_at: "2026-05-19T09:30:00Z",
    duration_ms: 58 * 60 * 1000,
    speaker_count: 4,
    excerpt:
      "First-run download felt broken without a deliberate progress affordance; users abandoned before the model finished.",
  },
  {
    id: "dev-meeting-0004",
    title: "Quick standup",
    started_at: "2026-05-18T08:00:00Z",
    duration_ms: 8 * 60 * 1000,
    speaker_count: 5,
    excerpt: null,
  },
];

function ok<T>(data: T): Result<T, IpcError> {
  return { status: "ok", data };
}

/** Build the `open_meeting` restore payload for a dev meeting id. */
function devMeetingState(meetingId: MeetingId): MeetingState {
  const entry =
    DEV_MEETINGS.find((m) => m.id === meetingId) ?? DEV_MEETINGS[0];
  return {
    meta: {
      uuid: entry.id,
      title: entry.title,
      started_at: entry.started_at,
      ended_at: new Date(
        new Date(entry.started_at).getTime() + entry.duration_ms,
      ).toISOString(),
      duration_ms: entry.duration_ms,
      speaker_count: entry.speaker_count,
      audio_format: {
        codec: "opus",
        sample_rate: 16_000,
        channels: 1,
        bitrate_kbps: 32,
      },
      asr_model: null,
      llm_model: null,
      diarizer: null,
      speaker_names: {},
      app_version: "0.0.0-dev",
    },
    transcript: DEV_TRANSCRIPT,
    notes: { notes_json: DEV_NOTES_JSON, notes_markdown: DEV_NOTES_MD },
  };
}

/** A `commands`-shaped object backed entirely by in-memory sample data. */
export const devCommands = {
  async listDevices(): Promise<Result<AudioDevice[], IpcError>> {
    return ok(DEV_DEVICES);
  },
  async startRecording(
    _deviceId: string | null,
  ): Promise<Result<MeetingId, IpcError>> {
    return ok(DEV_MEETING_ID);
  },
  async pauseRecording(): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  async resumeRecording(): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  async stopRecording(): Promise<Result<MeetingMeta, IpcError>> {
    return ok({
      uuid: DEV_MEETING_ID,
      title: "Launch sync — Tuesday",
      started_at: new Date(STARTED_AT_MS).toISOString(),
      ended_at: new Date().toISOString(),
      duration_ms: Date.now() - STARTED_AT_MS,
      speaker_count: 0,
      audio_format: { codec: "opus", sample_rate: 16_000, channels: 1, bitrate_kbps: 32 },
      asr_model: null,
      llm_model: null,
      diarizer: null,
      speaker_names: {},
      app_version: "0.0.0-dev",
    });
  },
  async getRecordingState(): Promise<Result<RecordingState, IpcError>> {
    return ok(DEV_STATE);
  },
  async getSettings(): Promise<Result<Settings, IpcError>> {
    return ok(DEV_SETTINGS);
  },
  async updateSettings(_settings: Settings): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  async listModels(): Promise<Result<ModelStatus[], IpcError>> {
    return ok(DEV_MODELS);
  },
  async ensureModel(_modelId: string): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  async saveNotes(
    _meetingId: MeetingId,
    _notesJson: string,
    _notesMarkdown: string,
  ): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  async loadNotes(
    _meetingId: MeetingId,
  ): Promise<Result<NotesDocument | null, IpcError>> {
    return ok({ notes_json: DEV_NOTES_JSON, notes_markdown: DEV_NOTES_MD });
  },
  // --- Phase 4 meeting-list + open surface (FR-33) ------------------------
  async listMeetings(): Promise<Result<MeetingListEntry[], IpcError>> {
    return ok(DEV_MEETINGS);
  },
  async openMeeting(
    meetingId: MeetingId,
  ): Promise<Result<MeetingState, IpcError>> {
    return ok(devMeetingState(meetingId));
  },
  async renameMeeting(
    _meetingId: MeetingId,
    _title: string,
  ): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  async deleteMeeting(_meetingId: MeetingId): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  async reTranscribe(_meetingId: MeetingId): Promise<Result<null, IpcError>> {
    return ok(null);
  },
  // --- Phase 5 summary surface (FR-30) ------------------------------------
  // (`re_summarise` was removed in Phase 5; the row Summarise action uses
  // `summarise_meeting` below.)
  async summariseMeeting(meetingId: MeetingId): Promise<Result<null, IpcError>> {
    // Seed a summary for this meeting so a follow-up `getSummary` returns
    // content, and notify the live event stream so the store re-reads it (the
    // real backend emits `AppEvent::SummaryReady`).
    devSummaries.set(meetingId, devSummaries.get(meetingId) ?? DEV_SUMMARY_MD);
    devSummaryReadyListeners.forEach((cb) =>
      cb({ kind: "summary_ready", meeting_id: meetingId }),
    );
    return ok(null);
  },
  async getSummary(
    meetingId: MeetingId,
  ): Promise<Result<string | null, IpcError>> {
    return ok(devSummaries.get(meetingId) ?? null);
  },
  async saveSummary(
    meetingId: MeetingId,
    summaryMarkdown: string,
  ): Promise<Result<null, IpcError>> {
    devSummaries.set(meetingId, summaryMarkdown);
    return ok(null);
  },
  // --- Phase 6 re-diarize (FR-11) -----------------------------------------
  // `rediarize_meeting(meeting_id) -> ()`: assign speakers to the meeting's
  // segments, then notify the live event stream so the meetings store re-reads
  // that meeting's transcript (the real backend emits
  // `AppEvent::DiarizationComplete`). Now a first-class `devCommands` entry —
  // the Phase-6 JOIN regenerated `bindings.ts`, so this folded out of the
  // temporary `devPendingCommands` map alongside the `callPendingCommand`
  // collapse (A9).
  async rediarizeMeeting(meetingId: MeetingId): Promise<Result<null, IpcError>> {
    devDiarizationListeners.forEach((cb) =>
      cb({
        kind: "diarization_complete",
        meeting_id: meetingId,
        speaker_count: 2,
      }),
    );
    return ok(null);
  },
  // Phase 9 chat surface. The DEV shim drives a representative streamed turn so
  // the chat pane renders and animates under `vite dev` for visual QA: it adopts
  // (or mints) a session id, then fans out `chat_token` deltas, a tool
  // call/result pair, and an authoritative `chat_turn_complete` into the live
  // event stream — mirroring the real backend's spawn-a-turn-on-a-background-task
  // contract (`send_chat_message` returns the id immediately; the reply streams
  // over the chat `AppEvent`s).
  async sendChatMessage(
    _meetingId: MeetingId | null,
    sessionId: ChatSessionId | null,
    message: string,
  ): Promise<Result<ChatSessionId, IpcError>> {
    const id = sessionId ?? DEV_CHAT_SESSION_ID;
    startDevChatTurn(id, message);
    return ok(id);
  },
  async getChatSession(
    _meetingId: MeetingId,
    sessionId: ChatSessionId,
  ): Promise<Result<ChatSession | null, IpcError>> {
    return ok(devChatSession(sessionId));
  },
  async listChatSessions(
    _meetingId: MeetingId,
  ): Promise<Result<ChatSession[], IpcError>> {
    return ok([devChatSession(DEV_CHAT_SESSION_ID)]);
  },
  async deleteChatSession(
    _meetingId: MeetingId,
    _sessionId: ChatSessionId,
  ): Promise<Result<null, IpcError>> {
    return ok(null);
  },
};

// --- DEV-only chat sample data + streamed turn -----------------------------

const DEV_CHAT_SESSION_ID: ChatSessionId = "dev-chat-0001";

/** A small seeded chat session so the pane has content under `vite dev`. */
function devChatSession(id: ChatSessionId): ChatSession {
  const now = new Date().toISOString();
  return {
    id,
    meeting_id: DEV_MEETING_ID,
    title: "Action items",
    created_at: now,
    updated_at: now,
    messages: [
      {
        role: "user",
        content: "What were the action items from this meeting?",
        turn_id: 0,
      },
      {
        role: "assistant",
        content:
          "There were two action items:\n\n1. **Alex** to circulate the revised budget by Friday.\n2. **Sam** to book the venue for the offsite.",
        turn_id: 0,
      },
    ],
  };
}

/**
 * DEV-only chat-event fan-out so the shim's `sendChatMessage` can stream a turn
 * into the live event stream started by {@link startDevEventStream}. The real
 * backend emits the chat `AppEvent`s on the shared bus; this mirrors them.
 */
const devChatListeners = new Set<(event: AppEvent) => void>();

/** Drive a representative streamed assistant turn for the dev shim. */
function startDevChatTurn(sessionId: ChatSessionId, _message: string): void {
  const turnId = 1;
  const reply =
    "Based on the transcript, the team agreed to ship the beta next sprint and to revisit pricing after the launch.";
  const words = reply.split(" ");
  const fan = (event: AppEvent) =>
    devChatListeners.forEach((cb) => cb(event));

  let i = 0;
  // A tool call/result pair first, then the streamed tokens, then the
  // authoritative completion.
  setTimeout(() => {
    fan({
      kind: "chat_tool_call",
      session_id: sessionId,
      turn_id: turnId,
      tool: "get_transcript",
      args_json: "{}",
    });
  }, 120);
  setTimeout(() => {
    fan({
      kind: "chat_tool_result",
      session_id: sessionId,
      turn_id: turnId,
      tool: "get_transcript",
      ok: true,
      summary: "read 42 segments",
    });
  }, 360);
  const tokenTimer = setInterval(() => {
    if (i >= words.length) {
      clearInterval(tokenTimer);
      fan({
        kind: "chat_turn_complete",
        session_id: sessionId,
        turn_id: turnId,
        final_text: reply,
      });
      return;
    }
    fan({
      kind: "chat_token",
      session_id: sessionId,
      turn_id: turnId,
      token: (i === 0 ? "" : " ") + words[i],
    });
    i += 1;
  }, 90);
}

/**
 * DEV-only `diarization_complete` fan-out so the dev shim's `rediarize_meeting`
 * can notify the live event stream started by {@link startDevEventStream}. The
 * real backend emits `AppEvent::DiarizationComplete`; this mirrors it.
 */
const devDiarizationListeners = new Set<(event: AppEvent) => void>();

/**
 * DEV-only `summary_ready` fan-out so the dev shim's `summariseMeeting` can
 * notify the live event stream started by {@link startDevEventStream}. The
 * real backend emits `AppEvent::SummaryReady`; this mirrors it in the browser.
 */
const devSummaryReadyListeners = new Set<(event: AppEvent) => void>();

/**
 * Drive a representative live event stream into the supplied callback.
 *
 * Emits an initial `state_changed` → recording, a `recording_clock`, each
 * seeded transcript segment, then a gentle repeating meter + clock tick so the
 * recording dot pulses and the elapsed clock keeps advancing for screenshots.
 * Returns an unsubscribe that stops the timers.
 */
export function startDevEventStream(
  callback: (event: AppEvent) => void,
): () => void {
  const timers: ReturnType<typeof setTimeout>[] = [];
  let cancelled = false;

  const emit = (event: AppEvent) => {
    if (!cancelled) callback(event);
  };

  // Forward dev `summary_ready` notifications (fired by the shim's
  // `summariseMeeting`) into this event stream so the summary store re-reads.
  devSummaryReadyListeners.add(emit);
  // Likewise forward dev `diarization_complete` notifications (fired by the
  // shim's `rediarize_meeting`) so the meetings store re-reads the transcript.
  devDiarizationListeners.add(emit);
  // Likewise forward dev chat events (fired by the shim's `sendChatMessage`) so
  // the chat store streams the sample turn under `vite dev`.
  devChatListeners.add(emit);

  // Establish the recording state on the next tick (after stores subscribe).
  timers.push(
    setTimeout(() => {
      emit({ kind: "state_changed", state: DEV_STATE });
      // Seed the transcript.
      DEV_TRANSCRIPT.forEach((segment) => {
        emit({
          kind: "transcript_segment",
          meeting_id: DEV_MEETING_ID,
          segment,
        });
      });
      // Seed the recording clock near the latest segment end.
      emit({
        kind: "recording_clock",
        meeting_id: DEV_MEETING_ID,
        clock_ms: 74_500,
      });
    }, 0),
  );

  // Gentle live meter + clock advance so the rec dot pulses and the clock moves.
  let clockMs = 74_500;
  const tick = setInterval(() => {
    clockMs += 200;
    const peak = 0.18 + Math.abs(Math.sin(clockMs / 700)) * 0.5;
    emit({ kind: "audio_meter", frame: { peak, rms: peak * 0.6 } });
    emit({ kind: "recording_clock", meeting_id: DEV_MEETING_ID, clock_ms: clockMs });
  }, 200);

  return () => {
    cancelled = true;
    devSummaryReadyListeners.delete(emit);
    devDiarizationListeners.delete(emit);
    devChatListeners.delete(emit);
    timers.forEach(clearTimeout);
    clearInterval(tick);
  };
}
