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
  ModelStatus,
  RecordingState,
  Result,
  Settings,
  Segment,
  IpcError,
  MeetingId,
  MeetingMeta,
  NotesDoc,
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

const DEV_MODELS: ModelStatus[] = [
  {
    id: ASR_MODEL_ID,
    kind: "asr",
    display_name: "Qwen3-ASR 0.6B (Q8_0)",
    status: { state: "available", local_dir: "/dev/models/asr/qwen3-asr" },
  },
];

const DEV_SETTINGS: Settings = {
  input_device_id: "builtin-mic",
  theme: "light",
  data_directory: null,
  start_hidden: false,
  autosave_interval_secs: 5,
};

/** Recording-clock offsets (ms, pause-excluding) for the seeded transcript. */
const DEV_TRANSCRIPT: Segment[] = [
  {
    start_ms: 4_200,
    end_ms: 9_800,
    text: "Right, let's get going — thanks everyone for making the time today.",
    words: [],
  },
  {
    start_ms: 12_400,
    end_ms: 21_300,
    text: "First item is the launch checklist. We're tracking three open risks against the date.",
    words: [],
  },
  {
    start_ms: 24_100,
    end_ms: 33_900,
    text: "The big one is the offline model download — first-run experience needs to feel deliberate, not broken.",
    words: [],
  },
  {
    start_ms: 38_600,
    end_ms: 47_200,
    text: "Agreed. I'll own the progress affordance and the retry path. Should land this week.",
    words: [],
  },
  {
    start_ms: 51_000,
    end_ms: 61_700,
    text: "Second, the transcript and notes need to read as one document, not two competing panes.",
    words: [],
  },
  {
    start_ms: 64_300,
    end_ms: 74_100,
    text: "That's the editorial direction — warm paper, one ink accent, marginal timestamps. Sign-off pending.",
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
  ): Promise<Result<NotesDoc | null, IpcError>> {
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
  async reSummarise(_meetingId: MeetingId): Promise<Result<null, IpcError>> {
    return ok(null);
  },
};

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
    timers.forEach(clearTimeout);
    clearInterval(tick);
  };
}
