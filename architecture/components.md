# C4 Level 3 — Components

![Core components](L3_CoreComponents.svg)

![Webview components](L3_WebviewComponents.svg)

The Rust core decomposes into one crate per component. Each crate is the
unit of agent ownership (see [`domain-ownership.md`](domain-ownership.md)).

## Dependency rule

> A component may depend on `common` and on no other component, except
> where this document explicitly grants the dependency.

`common` holds the shared types and trait definitions. Anything else
crossing a boundary either flows through trait objects defined in
`common`, or via the orchestrator wiring everything together.

The explicit cross-component dependencies and the phase each crate first
appears in:

| Crate | First phase | May depend on |
|---|---|---|
| `common` | 1 | (nothing in this workspace) |
| `audio-capture` | 1 | `common` |
| `vad-chunker` | 2 | `common` |
| `asr-runtime` | 2 | `common` |
| `asr-parakeet` | 8 | `common` |
| `diarizer` | 6 | `common` |
| `summariser` | 5 | `common` |
| `persistence` | 1 (minimal) → 4 (full) | `common` |
| `model-registry` | 2 | `common`, `settings` |
| `settings` | 1 | `common` |
| `orchestrator` | 1 (minimal) → 2 (live pipeline) | `common`, `audio-capture`, `vad-chunker`, `asr-runtime`, `asr-parakeet`, `diarizer`, `persistence`, `model-registry`, `settings` |
| `agent-tools` | 9 | `common`, `persistence`, `orchestrator` |
| `chat-agent` | 9 | `common`, `summariser`, `agent-tools` |
| `mcp-server` | 10 | `common`, `agent-tools` |
| `ipc-bridge` | 1 | `common`, `orchestrator`, `persistence`, `summariser`, `settings`, `agent-tools`, `chat-agent` |
| `app-main` (bin) | 1 | `common`, `orchestrator`, `ipc-bridge`, `model-registry`, `settings`, `agent-tools`, `mcp-server`† |

† `mcp-server` is an **optional** edge of `app-main`, gated by the `connected`
Cargo feature (default ON). The free artifact is built with
`--no-default-features` and omits `mcp-server` and its transitive rmcp stack.
See `cross-cutting.md` — "Build variants".

Any PR adding an edge not in this table requires an architecture-doc
update in the same commit. The table tracks **runtime** edges only;
test-only dev-dependencies (e.g. `diarizer → persistence` and
`diarizer → hound` for the over-split eval's audio decode, mirroring
`orchestrator`'s test-only deps) are documented in prose where they are
used, not added here.

### Crates that grow across phases

- **`persistence`** appears in Phase 1 as a minimal writer of
  `audio.opus` + `metadata.json` to a per-meeting folder. Phase 4 grows it
  to the full surface: the folder readers (incl. the graduated
  pause-INCLUDING Opus decoder and the `MeetingState` assembler), the libsql
  `index.db` index + forward-only migration runner + `rebuild_from_disk` +
  self-heal `reconcile_orphans`,
  rename/delete meeting operations, and the `summary.md` path + I/O. It still
  depends only on `common` (libsql / tokio are external crates, not workspace
  components).
- **`orchestrator`** appears in Phase 1 as a tiny state machine for
  start / stop / pause with the audio meter and capture lifecycle. The
  full live pipeline (VAD → ASR → transcript events → diarizer trigger)
  arrives in Phase 2.
- **`ipc-bridge`** appears in Phase 1 with start/stop/pause commands,
  device-list query, audio-meter and state-change events. Grows each
  phase as new domain surface is added.
- **`asr-runtime`** and **`vad-chunker`** both first appear in Phase 2 —
  they're a unit. Phase 1 captures audio but does not transcribe.

## Rust core components

### `common`
**Crate:** `crates/common`
**Owns:** shared types (`MeetingId`, `ChatSessionId`, `ModelId`,
`AudioChunk`, `Segment`,
`WordTimestamp`, `MeetingMeta`, `ModelDescriptor`, `RecordingState`,
`AppEvent`, `AudioDevice`, `AudioMeterFrame`, `AudioFormat`,
`ModelKind`, `ModelManifestEntry`, `ModelFileEntry`, `ModelStatusState`,
`ModelStatus`, `MeetingListEntry`, `NotesDocument`, `NoteBlock`, `MeetingState`,
`InterAgentRequest`, `InterAgentReply`),
trait definitions (`AsrBackend`, `Diarizer`,
`Summariser`), the shared `AppError` enum + `AppResult<T>` alias, and the
`apply_speaker_overlay(&mut [Segment], &BTreeMap<String, String>)` helper — the
single canonical speaker-name overlay (raw diarizer label → display name),
shared by the agent read tools and the summariser input path so a summary
refers to "Alice", not "A".
`NoteBlock { at_ms: Option<u64>, text }` (#70) is a note paragraph for the
summariser — anchored ones carry the `data-anchor-ms` recording-clock
timestamp; `Summariser::summarise` takes `&[NoteBlock]` (not flat markdown) so
notes weave into the transcript at their time.

**Phase 9 precursor — chat-agent shared types.** `ChatSessionId` (a UUID
newtype mirroring `MeetingId`); six chat `AppEvent` variants (`ChatToken`,
`ChatToolCall`, `ChatToolResult`, `ChatTurnComplete`, `ChatError`,
`ChatContextTrimmed { session_id, dropped_turns }` — the last emitted when the
driver's sliding window evicts older turns, P2) that ride the existing
`AppEventPayload` newtype + the single `collect_events![AppEventPayload]`
registration — no new event registration;
`MeetingMeta.speaker_names: BTreeMap<String, String>` (diarizer-label →
display-name overlay, `#[serde(default, skip_serializing_if = …)]` so existing
`metadata.json` still deserialises and the wire shape only grows);
`MeetingMeta.notes_format: u8` (O2 notes-CRDT groundwork — `0` = JSON-only
pre-CRDT, `1` = Yjs `notes.ydoc` authoritative with derived projections;
`#[serde(default)]` so existing `metadata.json` reads as `0`, the same
defaulted-field pattern `speaker_names` used; see
`planning/DESIGN_notes-crdt.md` D-O2.7 and the `persistence` "CRDT notes
storage" section); and the
in-process bridge types `InterAgentRequest` / `InterAgentReply` (referencing
`ChatSessionId`), landed now so Phase 10's MCP `send_to_internal_agent` adds no
`common` change. `ChatToken` is a lossy hint — `ChatTurnComplete.final_text`
carries the full reconciled reply (see `cross-cutting.md` — "Agent chat loop").

**Phase 9 — chat session wire types.** The persisted/wire chat shapes the chat
UI renders and `persistence::ChatStore` serialises: `ChatRole { System, User,
Assistant, Tool }` (serde snake_case); `ToolCallRecord { id, name,
arguments_json }` (one requested tool call, the persisted mirror of
`chat-agent`'s engine `ToolCall`); `ChatMessage { role: ChatRole, content:
String, tool_name: Option<String>, tool_call_id: Option<String>, tool_calls:
Vec<ToolCallRecord>, turn_id: u64 }` (the `turn_id` matches the chat events'
per-session monotonic turn counter; `tool_name`/`tool_call_id` present only on
`Tool` messages; `tool_calls` present only on an `Assistant` message that
requested tools — the carrier that keeps a reloaded multi-tool turn a valid
OpenAI `assistant(tool_calls) → tool(result)` sequence, CQ1; both default-empty
so older `chat/*.json` still deserialises); `ChatSession { id: ChatSessionId,
meeting_id: Option<MeetingId>, title: Option<String>, messages:
Vec<ChatMessage>, created_at: String, updated_at: String }` (RFC 3339
timestamps; absent `meeting_id`/`title` omitted). These are **distinct from**
`chat-agent`'s engine-internal message type (which the engine serialises into
the oaicompat template — its `ChatMessage` likewise carries a `tool_calls:
Vec<ToolCall>` for the assistant turn, plus a `CancelFlag` cancellation signal
and a `TurnOutcome::Cancelled` outcome, P1); the `ipc-bridge` driver maps
between the two at its boundary. All wire types derive `specta::Type` (they
cross tauri-specta).

**Persisted shape — what actually shipped (P3).** The shipped wire/persisted
chat types are the `ChatRole` / `ToolCallRecord` / `ChatMessage` / `ChatSession`
set described above. The earlier plan-draft names `ChatTurn` / `ChatStopReason`
/ `ChatSessionHeader` were **not** built: a session persists as a flat
`Vec<ChatMessage>` (each carrying a monotonic `turn_id`), not a `Vec<ChatTurn>`;
turn termination is conveyed by the `AppEvent::ChatTurnComplete` / `ChatError`
events (and the engine's `TurnOutcome`), not a persisted `ChatStopReason`; and
the session-header fields live inline on `ChatSession` (`id` / `meeting_id` /
`title` / `created_at` / `updated_at`), not a separate `ChatSessionHeader`. The
on-disk `chat/{session_id}.json` is exactly the `ChatSession` JSON.

**Phase 9 precursor — `Summariser: Send + Sync`.** The summariser trait widens
from `Send` to `Send + Sync` (SP0-verified). A held `Arc<dyn Summariser>` is
shared by the one-shot summary path and the chat agent's `resummarise` tool, so
it must cross threads AND be referenced concurrently; with only `Send` an
`Arc<dyn Summariser>` is not `Sync` and the chat tool's `async_trait` `Send`
future bound fails to compile. All impls already satisfy it: `LlamaSummariser`
holds a `LlamaModel` (`unsafe impl Send + Sync`) + a `PathBuf` + config and
builds its `!Sync` `LlamaContext` fresh per call (never stored);
`OllamaSummariser` holds a `reqwest::blocking::Client`; the test stub holds
`Mutex`-guarded fields.

The recorder-lifecycle additions `RecordingState::Finalising` and the
`AppEvent::{MeetingFinalised, TranscriptReady}` variants are documented with
their producers in the `orchestrator`/`ipc-bridge` "Responsive stop" and
re-transcribe notes below.

**VRAM-aware GPU placement — the probe + the pure plan.** `common` now exposes
the GPU auto-detection surface: `probe_primary_gpu() -> Option<GpuProbe>` (behind
the `llama-backend` feature — the same feature that owns the shared
`LlamaBackend`; `None` on a CPU-only build), the `GpuProbe { total_bytes,
free_bytes, is_integrated, name }` snapshot, the tri-state `GpuAcceleration {
Auto, On, Off }` enum (serde snake_case, `Default = Auto`, `specta::Type` — it is
a `Settings` field so it crosses IPC), the `GpuPlan { summariser_gpu, asr_gpu,
effective_prefer_large }` per-model decision, and the **pure**
`resolve_gpu_plan(probe, mode, prefer_large_asr) -> GpuPlan` that the consumers
call at each model-load moment. Private helpers `probe_budget(p) -> u64`
(headroom + free-tighten computation) and `large_asr_fits(asr_headroom,
prefer_large) -> bool` (large-ASR VRAM check) are shared by the `On` and `Auto`
branches inside `resolve_gpu_plan` to avoid duplication; they are not public.
`settings.gpu_acceleration` is now this
`GpuAcceleration` enum (was `bool`; a `deserialize_with` shim migrates a legacy
bool store, `true → Auto` / `false → Off`). `ipc-bridge` and `orchestrator` are
the consumers; the policy + thresholds live in `cross-cutting.md` — "GPU
portability". **No dependency-table edge changes:** the probe + plan live in
`common` (which every crate already depends on) and `probe_primary_gpu` reuses
the existing `llama-backend` feature, so no new crate or `use` edge is
introduced.

**Operation-progress event (live-test UX).** `AppEvent::OperationProgress {
meeting_id, op: OperationKind, fraction: Option<f32>, label: String }` (plus the
`OperationKind { ReTranscribe, Summarise, Rediarize, Finalise, Translate }` enum)
rides the existing `AppEventPayload` newtype + the single
`collect_events![AppEventPayload]` registration — no second registration.
Producers: the orchestrator's `runner::re_transcribe_buffer` emits a DETERMINATE
fraction (kept-samples processed / total) per accumulator flush; `ipc-bridge`'s
`summarise_meeting` emits a DETERMINATE fraction (tokens generated / `max_tokens`)
threaded through `LlamaSummariser::summarise_with_progress`; `ipc-bridge`'s
`translate_meeting` emits a DETERMINATE fraction (segments translated / total
segments); the re-diarize and finalise-drain paths emit INDETERMINATE (`fraction =
None`, one opaque sherpa/drain compute with no progress callback). The webview
clears the per-row indicator on the terminal `TranscriptReady` / `SummaryReady` /
`DiarizationComplete` / `TranslationReady`. See `architecture/cross-cutting.md` —
"Operation progress".

**Phase 7 — shared LlamaBackend (feature-gated).** Behind the optional
`llama-backend` feature (`dep:llama-cpp-2`, OFF by default so the default
`common` build stays pure), `common` exposes
`llama_backend::shared_llama_backend() -> AppResult<&'static LlamaBackend>` — the
single process-wide backend. `LlamaBackend::init()` is global (once per
process), and both `asr-runtime` and `summariser` load GGUF models in the same
app process, so they MUST share one cell; each enables the feature and delegates
to this function. (A private `OnceLock` per crate made whichever initialised
second fail — the record-then-summarise bug fixed in the Phase-7 review pass.)
This adds no workspace dependency edge; `llama-cpp-2` is an external FFI dep.

**Diagnostic report (issue #0014).** `DiagnosticReport { app_version, platform,
gpu, error_class, log_excerpt, backtrace: Option<String> }` (serde, snake_case,
`specta::Type` behind the `specta` feature) is the redacted snapshot the
"Report a problem" flow crosses IPC. Assembled and redacted by `ipc-bridge`'s
`get_diagnostic_report` (log-excerpt / backtrace redaction is owned there, where
the data lives) and pre-filled into a GitHub issue form by the webview's
`issueReport.ts` (the snake_case fields map onto its camelCase `DiagnosticReport`
shape). **No meeting-content field by construction** (no transcript / notes /
title / speaker-name field exists), so meeting content cannot ride this type. No
telemetry: nothing is sent except by the user's explicit browser action. No
dependency-table edge changes — the type lives in `common`.

**Phase 4 precursors.** `MeetingListEntry` (meeting-list row, FR-33),
`NotesDocument { notes_json, notes_markdown }` (the canonical wire-facing
notes carrier — `String` fields because `serde_json::Value` has no
`specta::Type`; `ipc-bridge` uses this type directly — the former local
`NotesDoc` mirror was collapsed into it so only one notes type reaches the TS
bindings), and
`MeetingState { meta, transcript, notes }` (the `open_meeting` restore
payload). Re-transcribe reuses `AppEvent::TranscriptSegment` — no new event.
The local index uses **libsql** (`default-features=false, features=["core"]`;
Gate-A-confirmed building on Linux + Windows MSVC); `index.db` is a derived
cache rebuildable from the per-meeting folders.

**Stable surface — locked.** The trait signatures and event variants in
this crate are the architectural contract that sub-agents implement
against in parallel. Changes here ripple to every other crate and
require an architecture-owner decision plus an update to this document
in the same commit. See [`agent-dispatch.md`](agent-dispatch.md) —
"Prerequisites for parallel dispatch".

**Phase 3 precursor — `AppEvent::RecordingClock { meeting_id, clock_ms }`.**
Additive variant carrying the live capture-sample, pause-*excluding*
recording offset (same timeline as `Segment::start_ms`), emitted throttled
(~5 Hz) by the orchestrator runner. The notes editor stamps paragraph
anchors from this, not from wall-clock `Date.now() - started_at_ms`. See
[`cross-cutting.md`](cross-cutting.md) — "Notes paragraph-anchor clock".

**`specta` feature.** All IPC-crossing types derive `specta::Type` behind
the optional `specta` feature. `ipc-bridge` enables the feature on this
crate (and on `settings`) so the generated TypeScript bindings consume
the canonical types directly — no mirror layer.

### `audio-capture`
**Crate:** `crates/audio-capture`
**Owns:** the audio device, sample-rate negotiation, the capture ring
buffer, device enumeration for the settings UI.

**Inputs:** start/stop commands (from orchestrator); device id (from
settings).
**Outputs:** an async `Stream<Item = AudioFrame>` of f32 samples at the
internal sample rate (**16 kHz mono — mandated; this matches mtmd's
encoder input rate per Spike 1's Q-P0-1**, so downstream consumers do
not resample).

**Back-pressure policy:** the cpal-callback→forwarder channel is a bounded
(8-frame) `Mutex<VecDeque>` + `Condvar` ring. The cpal real-time callback
produces via `try_lock` only (never blocks the RT thread); on overflow it pops
the OLDEST frame and pushes the newest, so drop-oldest is genuinely honoured
(an earlier `sync_channel` design held the lock for the whole session and
silently degraded to drop-newest). Meter window is 512 samples (~32 ms at
16 kHz, ~30 Hz emission rate).

**Device identity:** `AudioDevice.id` is an opaque `String` of the form
`"{enumeration-index}\u{1f}{name}"` (ASCII unit separator, which a device name
cannot contain) so same-named ALSA devices get distinct ids; `is_default` is the
first name-match. `resolve_device` parses the composite id (index authoritative,
name-consistency-checked) and falls back to name matching for legacy bare-name
ids persisted in `settings.input_device_id`.

**System/call audio mixing (loopback source + mixer).** When the
`settings.capture_system_audio` flag is on (ON by default, opt-out), `start` ALSO opens
the default **render** endpoint in **loopback** mode (a second capture source)
and SUMS it with the microphone into the SAME single `samples` stream, so a
Teams-style call transcribes all participants — not just the user. The public
`AudioStreams` / `AudioFrameBatch` shapes are unchanged, so the
orchestrator/runner are untouched; downstream diarization separates the
speakers. `AudioCaptureManager::start` takes a `capture_system_audio: bool`
parameter (the orchestrator passes `settings.current().capture_system_audio`).

- **Loopback source (`loopback.rs`, Windows-only).** Uses **cpal's transparent
  WASAPI loopback**: building an INPUT stream on a render device automatically
  sets `AUDCLNT_STREAMFLAGS_LOOPBACK`, so the existing `build_input_stream`
  machinery (sample-format dispatch, mono downmix, the drop-oldest ring) is
  reused with no extra dependency — **no `wasapi` crate is needed**. On
  non-Windows (`cfg(not(windows))`) the source is a stub returning
  `Error::LoopbackUnsupported`; enabling the toggle there (or any loopback-open
  failure) logs a warning and falls back to mic-only — the recording is never
  failed.
- **Mixer (`mixer.rs`).** Each source resamples to 16 kHz mono independently and
  feeds a bounded per-source batch channel; a mixer task drains both, SUMS
  sample-wise, clamps to `[-1.0, 1.0]`, meters the mixed output, and forwards
  `AudioFrameBatch`es into the public `samples` channel. The RT callbacks keep
  the same `try_lock`/drop-oldest discipline (one ring per source). Sync: the
  mixer emits the samples both sources have in common (`min(len)`) each tick and
  holds the faster source's surplus for the next tick (small drift tolerated by
  transcription); a source that has ENDED is zero-filled on the final flush so
  the timeline keeps advancing. The mixing math (`sum_clamp` + `MixState`) is a
  pure, unit-tested seam since the real capture devices cannot be driven in a
  unit test.

AEC is **future work** — see `cross-cutting.md` "Threading model"; v1 handles
echo only via the toggle (ON by default, opt-out — turn it off when the mic
hears the call from the speakers).

### `vad-chunker`
**Crate:** `crates/vad-chunker`
**Owns:** Silero VAD model lifecycle (via `vad-rs`), the smoothing
wrapper, silence-detection heuristics.

**Inputs:** frame stream from `audio-capture`.
**Outputs:** an async `Stream<Item = AudioChunk>` where each chunk is
bounded by detected silence ≥ the configured threshold and carries
`{start_ms, end_ms, samples}`.

The Silero VAD ONNX file is **vendored** under `resources/silero/`, not
managed by `model-registry`. See
[`cross-cutting.md`](cross-cutting.md) — Model lifecycle.

**Implementation note (Phase 2).** Smoother defaults: threshold 0.5,
onset 3 frames (90 ms), hangover 24 frames (720 ms), prefill 5 frames
(150 ms pre-roll). `process_samples` accumulates a partial-frame buffer
and only feeds the VAD complete 480-sample frames (Silero v4 panics on
any other size). The bundled ONNX is resolved at build time via
`option_env!("MINUTIST_SILERO_PATH")` falling back to
`{CARGO_MANIFEST_DIR}/../../resources/silero/silero_vad_v4.onnx`.

`VadChunker::reset()` restores the chunker to its just-opened state (Silero RNN
hidden state, smoother, partial-frame buffer, pre-roll ring, frame clock, and
any in-progress segment) **without reloading the model**. It is used at a hard
region boundary where the next audio is independent rather than a continuation —
the offline re-transcribe calls it at each detected recording **pause** (see
`orchestrator` — re_transcribe) so a post-pause utterance onsets afresh instead
of merging with the pre-pause one across the skipped silence.

### `asr-runtime`
**Crate:** `crates/asr-runtime`
**Owns:** llama-cpp-2 mtmd binding, the Qwen3-ASR model, the prompt /
template details required to drive it as ASR.

**Implements:** `AsrBackend` from `common`.
**Inputs:** an `AudioChunk`.
**Outputs:** `Vec<Segment>` for that chunk.

**Encoder-window constraint (confirmed by Phase 0 Spike 1).** mtmd's
audio encoder uses a fixed 30 s window. Sub-30 s inputs are
silence-padded internally and the model continues into the padded
region, hallucinating words that weren't in the audio. The `AsrBackend`
trait itself is unaffected — implementations handle the constraint.

`orchestrator` is responsible for shaping its calls to this trait
correctly. The Phase 2 default is the batched-VAD strategy with
silence-preservation (see `cross-cutting.md`, "ASR chunking
constraint"): collect VAD segments into a ≥25 s buffer, **keep the
original inter-utterance silences (zero-padded, capped at ~3 s each)**,
and only then dispatch.

**Output schema.** `asr-runtime` MUST stop generation on `</asr_text>`
in addition to EOG — Qwen3-ASR doesn't always emit EOG for sub-window
audio. See `cross-cutting.md` — ASR chunking constraint.

**Language hint — `AsrRuntimeConfig.language: Option<String>`.** Optional
forcing language (full English name, e.g. `Some("English")`). `None` =
auto-detect (the pre-feature behaviour). When `Some(name)`, the prompt
prefix-forces the language via an assistant-turn prefill appended AFTER
`apply_chat_template` (never inside the user message): the rendered prompt
ends with `language <name><asr_text>`, exactly the wrapper Qwen3-ASR emits
itself, so the model only generates the transcript. `None` produces the
byte-identical pre-feature prompt — the locked "Auto-detect MUST be
byte-identical" guarantee. The hint rides on `AsrRuntimeConfig` only; the
`AsrBackend` trait and the `common` dependency table are unchanged. The
orchestrator resolves it from `settings.transcription_language` at start
(via `resolve_transcription_language`, mirroring `resolve_gpu_layers`).
`Default` is `None` (auto-detect), so the no-arg/test path is unchanged.

**Implementation pattern (Phase 2).** `LlamaBackend` is a process-wide
`OnceLock` singleton; `LlamaModel` + `MtmdContext` are loaded once in
`AsrRuntime::new`; a fresh `LlamaContext` is allocated per
`transcribe_chunk` call (cheap, <100 ms) to guarantee a clean KV cache.
The `</asr_text>` early-stop checks the full concatenated detokenised
string, not per-token, so the tag is caught even when it spans a token
boundary.

**GPU-tier sibling (Phase 8).** `asr-runtime` also drives **Qwen3-ASR-1.7B**
(same mtmd path, official `ggml-org` GGUF + mmproj) as a higher-accuracy /
better-multilingual tier. The tier is requested automatically — the VRAM clamp
in `resolve_gpu_plan` decides whether it fits alongside the summariser; if not,
the 0.6B remains the CPU default. Both share the same `#21847` long-audio
limitation, so the batched-VAD chunking is mandatory for either.

**Auto-language spurious-CJK guard.** When `AsrRuntimeConfig.language` is `None`
(auto-detect), `AsrRuntime` holds a `ScriptHistory` ring buffer (last 8 chunks)
tracking script-class observations (Latin / CJK / Other) per emitted chunk.

*Trigger:* current chunk text is majority-CJK (> 50 % of non-whitespace chars are
CJK codepoints per `is_cjk`) AND the session history is majority-Latin with ≥ 2
prior Latin observations.

*Action:* re-run `transcribe_inner` once with `language = Some("English")`.
`transcribe_inner` now always returns an `InnerResult { text, mean_logprob }` —
the mean log-probability of emitted tokens computed from sample-time logits (read
between `sampler.sample` and `decode()`; log-sum-exp for numerical stability).

*Acceptance:* prefer the forced output only when (a) it passes a plausibility
check — non-empty, no CJK codepoints, non-degenerate 3-char n-gram distribution —
AND (b) its `mean_logprob` exceeds the auto run's by more than `LOGPROB_EPSILON`
(0.05). If the forced run loses or scores are within epsilon, keep the auto result
and emit `tracing::warn` — genuine Chinese utterances in a mixed room are never
silently dropped.

*v1 limitation:* in non-English Latin-script rooms the forced-English retry will
typically score WORSE and be rejected (self-correcting). The forced language here
is hardcoded to `"English"`; a future revision will derive it from the user's
locale/language settings.

*Guard is inert* when `config.language` is `Some(..)` — the `cjk_guard` path is
skipped entirely and `ScriptHistory` is not updated.

### `asr-parakeet`
**Crate:** `crates/asr-parakeet`
**Owns:** the sherpa-onnx offline-transducer binding, the Parakeet TDT 0.6B v3
model, and token→word/segment timestamp aggregation.

**Implements:** `AsrBackend` from `common` (the same trait as `asr-runtime`).
**Inputs:** an `AudioChunk`. **Outputs:** `Vec<Segment>` for that chunk, **with
per-word `start_ms`/`end_ms` populated** — the token-level timestamps the mtmd
path cannot produce.

**Why a separate crate.** Keeps the single-domain rule: `asr-runtime` is the
llama-cpp-2/Qwen domain; `asr-parakeet` is the sherpa-onnx/Parakeet domain.
sherpa-onnx already enters the workspace via `diarizer`; this is its second
consumer (FFI via `sherpa-rs`, the same `=0.6.8` pin). The two ASR backends are
interchangeable behind `Box<dyn AsrBackend + Send>`; the orchestrator selects
one per the resolved transcription language (`runner::build_asr_backend`).

**Timestamps (binding gap, confirmed by the Phase-8 spike).** sherpa-rs 0.6.8
`TransducerRecognizer::transcribe()` returns only the text and drops the
per-token timestamps the C result carries
(`SherpaOnnxGetOfflineStreamResult` → `timestamps` + `tokens`, as used by
`OfflineRecognizerResult`). This crate enables the `sherpa-rs` `sys` feature and
reads the full result directly, then groups Parakeet's sub-word tokens into
words on the leading-space boundary to fill `Segment.words`.

**Language scope + routing.** Parakeet TDT v3 covers 25 European languages
(English + EU). Languages outside that set route to the Qwen `asr-runtime` tiers
instead; `Auto-detect` routes to Qwen (broadest). The pure mapping lives in
`common` (`asr_engine_for_language`) so the UI and the orchestrator agree. See
`cross-cutting.md` — "ASR engine routing".

**License:** CC-BY-4.0 — attribution is shipped in the About dialog (distinct
from the Apache-2.0 Qwen models).

### `diarizer`
**Crate:** `crates/diarizer`
**Owns:** sherpa-onnx binding, the embedding + clustering pipeline.

**Implements:** `Diarizer` from `common`.
**Inputs:** the full buffered audio + the segment array from ASR.
**Outputs:** mutates segments in place, setting `speaker_id`.

The offline `SherpaDiarizer` pass is post-hoc — it runs after the
recording stops or as a user-triggered re-diarize. A SEPARATE, additive
`OnlineDiarizer` runs live (Phase A — see "Phase A — live online
labelling" below); it does not replace the offline pass, which stays
authoritative for the finished transcript.

**Binding pin (confirmed by Phase 0 Spike 4).** `sherpa-rs = 0.6.8`
(Thewh1teagle, MIT) with the `download-binaries` feature for dev and
`static` for Phase 7 bundling. The `sherpa_rs::diarize::Diarize` surface
covers everything needed; no `bindgen` direct-C wrapper required. The
k2-fsa-owned alternative crate `sherpa-onnx = 1.13.x` (Apache-2.0)
should be re-evaluated against `sherpa-rs` before Phase 6 ships.

Cluster IDs returned by the binding are arbitrary `i32`; the impl must
normalise to first-seen-order labels (`A`, `B`, …) before populating
`Segment::speaker_id`. The binding's `eyre::Result` is mapped to
`common::AppError::Inference` at the trait boundary.

**Phase 6 — public surface + model bundle (license-verified 2026-06).** The
crate exposes `SherpaDiarizer::open(seg_onnx, emb_onnx, DiarizerConfig)` and
`impl Diarizer` (`assign_speakers(audio, sample_rate=16000, &mut [Segment])`),
which runs sherpa `Diarize::compute`, relabels first-seen `A`/`B`/…, and
overlays `speaker_id` onto the ASR segments by max-overlap interval-join (no
`common::SpeakerTurn` type — overlay only). It takes RESOLVED model paths and
depends only on `common` (NOT `model-registry`, NOT `persistence`). All
model-registry resolution lives in the orchestrator's `runner::build_diarizer`,
which ensures both model dirs and passes the resolved `&Path`s into
`SherpaDiarizer::open`. Bundled models
(settings-selectable via `model-registry`): **segmentation =
pyannote/segmentation-3.0 (MIT)**; **embedding = 3D-Speaker CAM++ zh-en
16k-common ADVANCED (Apache-2.0, "common" corpus — NOT VoxCeleb)**. (The zh-en
model replaced the Mandarin-only zh-cn one on 2026-06-05: the zh-cn embedding
under-separated English voices, over-splitting a single speaker into 3-4; the
zh-en model opens a usable `cluster_threshold` window — default raised 0.5→0.75.)
This corrects
Spike-4's TitaNet, which is VoxCeleb-trained and not cleanly redistributable in
a paid product; ERes2NetV2 (same license) is the
swap-in accuracy upgrade. The orchestrator owns the lifecycle: it builds the
diarizer (resolving both model dirs via `model-registry`), runs the on-stop pass
(gated on `settings.diarization_enabled`, default off) and the `rediarize`
re-pass, and emits `AppEvent::DiarizationComplete` on its shared bus. Ship the
MIT + Apache NOTICE/attribution (the k2-fsa / HF mirrors don't carry the
upstream notices).

**Implementation (Phase 6 Stream S1).** `SherpaDiarizer::open` constructs the
`sherpa_rs::diarize::Diarize` engine once and holds it behind a `Mutex` (the
`common::Diarizer` trait takes `&self`; sherpa's `compute` takes `&mut self`,
and diarization is single-threaded per call so the mutex is never contended on
the hot path). `DiarizerConfig` maps onto sherpa's `DiarizeConfig`:
`num_clusters = Some(n)` → exact-cluster mode; `None` → `num_clusters = Some(-1)`
(sherpa's "use threshold" sentinel, Spike 4) with `cluster_threshold`, plus
sherpa's `min_duration_on` / `min_duration_off` smoothing. The orchestrator
constructs the diarizer with `DiarizerConfig::default()` (`num_clusters = None`,
`cluster_threshold = 0.75`, `min_duration_on = 0.3`, `min_duration_off = 0.5`,
`min_cluster_share = 0.02`) for BOTH the on-stop pass and the user-triggered
re-diarize pass: at record time the speaker count is unknown, so production uses
threshold/auto-count mode to discover it rather than fixing a cluster count.
There is no `Some(1)` production path. `assign_speakers` rejects any
`sample_rate != 16000` with `AppError::InvalidInput`, short-circuits empty
audio/segments to `0`, runs `Diarize::compute`, and overlays via a pure
`overlay_speakers(&[sherpa::Segment], &mut [Segment], &DiarizerConfig) -> u32`:
per ASR segment it picks the max-total-overlap sherpa CLUSTER (seconds→ms,
half-open `[start_ms, end_ms)`; ties resolve to the lower cluster id; no overlap
→ `speaker_id = None`), then applies the **post-cluster prune + cap** (issue #63:
drop clusters below `min_cluster_share` of the attributed speech duration — or
below the off-by-default `min_cluster_segments` / above the off-by-default
`max_speakers` — and reassign their segments to the nearest surviving cluster),
relabels the surviving `i32` cluster ids to first-seen-order `A`/`B`/… across
segment slice order, and returns the distinct-label count. The prune is the
robust lever against the long-recording over-split (a single distance threshold
cannot separate a drifted same-speaker embedding from a distinct speaker); see
`cross-cutting.md` — "Offline over-split prune". The sherpa `eyre::Result` is
mapped to `Error::ModelLoad`/`Error::Inference` →
`AppError::{ModelLoad,Inference{backend:"diarizer"}}` at the boundary (eyre
arrives transitively via `sherpa-rs`; no separate `eyre` dep). `sherpa-rs =
{ workspace = true }` is added to `crates/diarizer/Cargo.toml`; `hound` and
`persistence` (test-only, the over-split eval's audio/transcript decode) are
dev-dependencies. A second pure public function
`overlay_speakers_from_prior(&mut [Segment], &[(u64, u64, Option<String>)])` carries
the prior diarization onto a freshly ASR-transcribed segment slice by
max-overlap interval-join: for each new segment the prior segment with the
greatest time overlap wins and its `speaker_id` string is copied verbatim
(no re-lettering), so `MeetingMeta.speaker_names` stays keyed correctly after
a re-transcribe. New segments with no prior overlap keep `None`. The
orchestrator's `finalise_retranscribe` calls this before writing the new
`transcript.json`. Tests: the default suite covers `overlay_speakers`
(interval-join, no-overlap=None, tie-break, first-seen relabel, stale-label
clearing) AND the prune/cap (tiny-share drop + reassign, genuine-speaker keep,
segment-count floor, cap-to-largest, never-zero fallback) AND
`overlay_speakers_from_prior` (full-overlap, max-overlap-wins, gap→None,
label-survival, empty-prior, prior-was-None) with no model; the
env-var-gated `tests/accuracy.rs` (`MINUTIST_DIARIZE_SEG_PATH` +
`MINUTIST_DIARIZE_EMB_PATH`, skip-on-unset) runs `assign_speakers` over
committed fixtures (`tests/fixtures/two_speakers_synth.wav` = two distinct
real-speech
speaker clips concatenated, with self-authored ground truth;
`single_speaker_control.wav` = one real speaker repeated), asserting ≥ 80 %
permutation-invariant segment accuracy and exactly one label on the control.

**Phase A — live online labelling (additive).** The crate now ALSO exposes
`OnlineDiarizer::open(embedding_onnx, OnlineDiarizerConfig)`,
`OnlineDiarizer::assign_segment(&[f32] 16 kHz-mono, sample_rate) -> AppResult<String>`,
and `speaker_count() -> AppResult<u32>`. It wraps ONLY the speaker-embedding
model (no segmentation model — VAD upstream supplies the segment boundaries) via
the sherpa `EmbeddingExtractor` (`sherpa_rs::speaker_id`), and delegates
clustering to a pure, FFI-free `OnlineClusterer` (running-mean centroids, cosine
similarity, configurable `similarity_threshold` + optional `max_speakers` cap,
sticky first-seen A/B/C labels). `open` mirrors `SherpaDiarizer::open`'s loading
+ error mapping (`Error::ModelLoad` → `AppError::ModelLoad`); `assign_segment`
reuses the 16 kHz guard, rejects an empty segment as `InvalidInput`, extracts the
embedding (sherpa `eyre` err → `Error::Inference`), assigns a sticky cluster
index, and maps it via `alpha_label`.

The online-vs-offline contract: the offline `SherpaDiarizer` / `common::Diarizer`
pass remains AUTHORITATIVE for the finished transcript; `OnlineDiarizer` is an
additive live hint that emits a sticky label per VAD segment as the segment
closes and NEVER retroactively relabels. The two are independent code paths
sharing only the `alpha_label` first-seen A/B/C generator (now `pub(crate)`) and
the 16 kHz `require_supported_sample_rate` guard.

Why a pure clusterer rather than sherpa's `EmbeddingManager`: the manager has no
running-mean centroids (one fixed vector per name, no update path) and is not
FFI-test-isolable (every method crosses into `sherpa_rs_sys`), so the centroid
update rule and clustering logic could not be exercised model-free — recorded
here so the reviewer sees the decision.

Phase A delivered the diarizer-crate surface only; **Phase B** now wires
`OnlineDiarizer` into the orchestrator (see the `orchestrator` "Phase B — live
diarization wiring" note) WITHOUT adding a dependency edge or a `common`-level
trait: the `orchestrator → diarizer` edge already exists (granted in Phase 6),
`OnlineDiarizer` is re-exported from the `diarizer` crate, and the live path stays
a concrete struct (no second `common` trait — the existing `common::Diarizer`
trait is offline-only and unchanged). No new crate-dependency edge is introduced —
`sherpa-rs` is already a `diarizer` dependency, and `EmbeddingExtractor` /
`ExtractorConfig` live in `sherpa_rs::speaker_id` within the same crate. Tests: the pure `OnlineClusterer` is covered model-free in
`src/online/clusterer.rs` (separation, stickiness, threshold split, centroid
drift, lower-index tie-break, `max_speakers` force-join, dim-mismatch/degenerate
rejection); the env-var-gated `tests/online_embedding.rs`
(`MINUTIST_DIARIZE_EMB_PATH` only — no segmentation model — skip-on-unset)
runs `assign_segment` over committed real-speech fixtures, asserting distinct
sticky labels for two speakers, label reuse on a speaker's repeat, one label for
the single-speaker control, and `InvalidInput` for a non-16 kHz or empty buffer.

### `summariser`
**Crate:** `crates/summariser`
**Owns:** llama-cpp-2 text-LLM lifecycle, summarisation prompts, the
optional external-LLM dispatcher (Ollama / LM Studio).

**Implements:** `Summariser` from `common`.
**Inputs:** transcript + notes (read via `persistence`).
**Outputs:** a markdown summary written via `persistence`.

**Bundled default model (Phase 5, primary-source verified 2026-06).**
**Gemma 4 E4B-it** (`gemma4` arch, **Apache-2.0** — Google moved Gemma 4 off
the restrictive Gemma ToU), the newest on-device Gemma. Loads in the pinned
llama.cpp b8783 (vendored by `llama-cpp-2 =0.1.146`) with no bump; 128K
context fits a 30-min transcript in one pass. Bundle the **text-only**
Q4_K_M GGUF (skip the multimodal `mmproj`). Low-end tier: **Gemma 4 E2B-it**
(same family/loader). Fallback if the Gemma-4 PLE forward-graph bug
(llama.cpp #22243) degrades quality: **IBM Granite 4.1-3b** (Apache-2.0,
dense, no PLE, non-thinking). The model is **settings-selected** — never
hard-coded — so switching is a manifest + `llm_model_id` change.

**Chat-template handling — MODEL-AGNOSTIC (Phase 0 Spike 2).** Use
`LlamaModel::chat_template(None::<&str>)` to read the GGUF's **baked-in**
template, then `LlamaModel::apply_chat_template(template, messages,
add_ass=true)` to render the prompt. Do NOT pull in `tokenizers` and do NOT
hand-build a model-specific scaffold (the old ChatML scaffold only matched
Qwen) — relying on the GGUF's own template keeps the summariser model-agnostic
across Gemma 4 / Qwen / Granite. If the template is missing, fail the request
explicitly (`AppError::InvalidInput`) rather than guessing. For Gemma 4 run
with **thinking disabled** (do not inject the `<|think|>` token); if a future
selected model emits a `<think>` block, strip it before persisting the summary.
The system prompt is folded into a SINGLE `user` turn, NOT a separate `system`
message (several templates, notably Gemma, have no `system` role). That alone is
insufficient: the bundled llama.cpp cannot RENDER a template newer than itself
(Gemma 4 postdates the vendored build), so `apply_chat_template` returns `ffi
error -1` even for a user-only message set. On that failure the summariser falls
back to a hand-built Gemma turn-format prompt (`<bos><start_of_turn>user …
<end_of_turn>` then an open `model` turn) — the format the shipped LLM uses;
other models keep their baked template. BOS is explicit because generation
tokenises `AddBos::Never` and `str_to_token` parses special tokens.

**Prefill must chunk by `n_batch`** — see `cross-cutting.md`, "llama.cpp
prefill batching". Long transcripts exceed `n_batch` (default 512) and
will assert otherwise.

**Use `AddBos::Never` after templating** (the template embeds the BOS
itself). Stop generation on `model.is_eog_token(token)`, which covers
both EOS and `<|im_end|>` for Qwen.

**Implementation (Phase 5 Stream S1).** `LlamaSummariser::open(model_path,
SummariserConfig)` loads the GGUF once (process-wide `LlamaBackend` `OnceLock`
singleton, mirroring `asr-runtime`) and retains the `LlamaModel`; each
`summarise` call allocates a fresh `LlamaContext` sized to `config.n_ctx` /
`config.n_batch`. `SummariserConfig` adds a `threads` field (default
`(num_cpus / 2).clamp(1, 8)`, matching `asr-runtime`) alongside `n_ctx`
(32 768), `n_batch` (512), `max_tokens` (2 048). The chunked-prefill split is a
pure `plan_prefill(prompt_len, n_batch) -> PrefillPlan` function (unit-tested
without a model): it tiles `[0, prompt_len)` into `≤ n_batch` chunks and marks
the final chunk's last token as the sole `logits = true` position. Generation
is greedy with incremental `encoding_rs` detokenisation; a `<think>…</think>`
block (if a model emits one) is stripped before return. The optional
`external-ollama` feature adds `OllamaSummariser` (a `reqwest::blocking`
dispatcher to a local `/api/chat` endpoint); `reqwest` + `serde` are pulled in
only by that feature.

**`translate_segment(text, target_language) -> AppResult<String>`.** A
concrete method on `LlamaSummariser` (not on the `Summariser` trait) that
translates one segment text into the named language. Builds a minimal
single-turn prompt ("Translate … into {language}. Output only the
translation.") and calls the shared `generate_with_config` path with a 512-token
cap (a translated segment is never longer than a full summary). The Gemma
chat-template fallback applies identically. The method is concrete on
`LlamaSummariser` (which always holds a local `LlamaModel`), so there is no
remote-backend path and no remote-backend guard. `ipc-bridge` holds the
concrete `Arc<LlamaSummariser>` and calls
this method per-segment in a `spawn_blocking` translation loop.
Env-gated test: `translate_segment_produces_spanish_translation` requires
`MINUTIST_LLM_MODEL_PATH`; verified 2026-06-12 with Gemma 4 E4B Q4_K_M (~7 s
per segment on CPU). No new dependency edge — `summariser` still depends only
on `common`.

**Phase 9 — model exposure for the chat engine (D5, the ONLY summariser
change).** `LlamaSummariser` gains `pub fn model(&self) -> &LlamaModel`, the
substrate seam the Phase-9 `chat-agent` engine borrows. `summarise()` is
unchanged. `ipc-bridge` holds the concrete `Arc<LlamaSummariser>`, lends
`&LlamaModel` to `chat-agent`'s `LlamaTurnBackend`, and coerces the same handle
to `Arc<dyn Summariser>` for the `agent-tools` `ToolContext`. The model is
`unsafe impl Send + Sync` (`llama-cpp-2`), so it crosses threads and is
referenced concurrently; the chat engine builds its own `!Sync` `LlamaContext`
fresh per turn, exactly as `summarise` does — no GGUF is reloaded per turn.
Keeping this an accessor (not a wrapper) preserves `summarise()` and avoids a
`summariser → chat-agent` edge: `chat-agent` depends on `summariser`, never the
reverse. `pub fn gpu_layers()` (the compile-time GPU-offload ceiling) is also
re-used by `chat-agent`'s `LlamaTurnConfig` default. No new dependency edge —
`summariser` still depends only on `common`.

**Notes weaving + two-phase progress (#69/#70).** The `common::Summariser`
trait's `summarise` now takes `notes: &[NoteBlock]` (was `notes_markdown:
&str`). `NoteBlock { at_ms: Option<u64>, text }` is a `common` vocabulary type;
`persistence::note_blocks_from_json` / `read_note_blocks` project a meeting's
`notes.json` into these (anchored paragraphs carry their `data-anchor-ms`
recording-clock timestamp). When any note is anchored, `render_user_content`
merges the transcript and the anchored notes into ONE time-ordered, `[m:ss]`-
prefixed timeline so the model sees each note beside what was being said when it
was written; un-anchored notes trail the timeline. With no anchored notes the
prior plain transcript + flat `# Notes` block is rendered byte-for-byte (no
extra context tokens). `summarise_with_progress` now reports a two-phase
`SummariseProgress` (`Prefill { done, total }` per prompt chunk, then `Generate
{ done, max }` per token); `ipc-bridge` maps the phases — plus an indeterminate
model-load / context-prepare phase — onto labelled `OperationProgress` (see
`cross-cutting.md` — "Operation progress"). Still depends only on `common`.

**`external-ollama` test coverage + verification.** `OllamaSummariser`'s
deterministic seams are factored into pure functions — `chat_url` (base-URL
normalisation, trailing slash tolerant), `build_chat_request` (the
`ChatRequest` serde shape: system/user roles + `stream: false`), and
`inference_error_for_status` (non-2xx → `Error::Inference` → `AppError::Inference`
with the `"summariser"` backend label) — each covered by `#[cfg(test)]` unit
tests in `ollama.rs` (no live server; the `reqwest` `send()` is the only
untested line). Because the feature is off by default, `cargo test -p
summariser` does not compile these; the gated verification harness
(`scripts/run-tests-windows.ps1`) runs `cargo test -p summariser --features
external-ollama` as an extra step whenever `-Package summariser`, so the ollama
tests are exercised (the feature build reports more tests than the default
build).

### `model-registry`
**Crate:** `crates/model-registry`
**Owns:** the on-disk model cache, the model-manifest schema, download
+ resume + hash verification, version metadata exposed to other
components.

The only component allowed to write to the model directory.

**Manifest:** `resources/models.json` at the repo root (loaded via `include_bytes!`
in `app-main`). Per-kind cache layout: `{app-data}/models/{asr,llm,diarize}/{model-id}/`.
Concurrent `ensure(same_id)` calls are coalesced via an `Arc<Notify>` in-flight map
so each model is downloaded at most once per process lifetime.

**Event source.** `ModelRegistry::new(cache_root, manifest, event_tx)` takes a
`broadcast::Sender<AppEvent>` — the *same* channel the orchestrator broadcasts on
(app-main constructs the channel once and shares it; see `app-main`). The registry
emits `AppEvent::ModelDownloadProgress` directly onto that bus during `ensure`,
throttled to ~10 Hz. So the registry is a legitimate first-class event source
alongside the orchestrator, not solely a path provider — the IPC forwarder's single
subscription sees its progress events too. (This refines `cross-cutting.md` "Model
lifecycle", which still frames the registry as handing out paths: that remains true
for model *files*, but the registry additionally publishes download-progress events.)

Progress is reported against an entry's **aggregate** byte total (the sum of every
file in the manifest entry), not per-file: a multi-file model (e.g. the ASR
`gguf` + `mmproj` pair) drives one monotonic 0→100% bar rather than resetting
between files. A terminal `bytes_done == bytes_total` event is emitted once all
files verify, so a consumer's completion check fires deterministically rather than
depending on a throttled per-chunk emit coinciding with the last byte. Verification
failures (e.g. a SHA-256 mismatch from a stale manifest) are returned to the
`ensure` caller, not the broadcast bus — the webview surfaces them at that seam.
Manifest file URLs MUST pin an immutable commit revision; a moving ref (`main`)
drifts when the upstream repo is re-uploaded and silently breaks hash verification.

### `persistence`
**Crate:** `crates/persistence`
**Owns:** the per-meeting folder layout, the libsql index schema and
migrations, Opus audio encoding, Tiptap JSON I/O.

**Opus encoder pin.** `audiopus = "0.3.0-rc.0"` (the explicit pre-release
tag is required at workspace level; Cargo's semver does not resolve
pre-releases from a `"0.3"` constraint). Container is Ogg via the `ogg`
crate. Phase 1 writes 16 kHz mono 32 kbps.

**CRDT notes dependency — `yrs`.** `yrs = "0.26"` (the Rust port of the Yjs
CRDT) is a direct dependency of `persistence`, used to store the authoritative
per-meeting notes document `notes.ydoc` and derive `notes.json` / `notes.md`
from it (see "CRDT notes storage" below). It is a **third-party** dependency
— like `sha2` / `libsql` / `audiopus` — **not** a crate-to-crate edge, so the
dependency table above is unchanged (`persistence` still depends only on
`common`). `yrs` is pure-Rust with no network surface and is embedded in BOTH
build variants; only the sync *transport* is `connected`-gated (a separate
crate). Durable whole-state blobs use the lib0 v2 encoding. See
`planning/DESIGN_notes-crdt.md` D-O2.1/D-O2.2/D-O2.4.

**Inputs:** typed write commands from orchestrator and IPC bridge.
**Outputs:** typed read responses; emits no events itself.

The only component allowed to read or write under `{app-data}/meetings/`
and `{app-data}/index.db`.

**Phase 1 surface:** writes `audio.opus` (Opus 16 kHz mono 32 kbps, Ogg
container) and `metadata.json` per meeting. Pause/resume inserts zero-sample
(silent) Opus frames so decoded duration equals wall-clock duration including
pauses (±20 ms per frame). The libsql index (`index.db`) and
transcript/notes/summary storage are Phase 4.

**Phase 2 surface growth:** `TranscriptWriter` writes `transcript.json` (JSON array of `Segment`) per meeting. Flushed on each ASR-worker return so a crash mid-recording loses at most one flush's worth of transcript.

**Phase 4 surface growth — `write_transcript(meeting_dir, &[Segment])`.** A free
function (in the `transcript` module) that rewrites `transcript.json` wholesale
from a slice, atomically (tmp + fsync + rename), for the Phase-4 offline
re-transcribe path (`orchestrator::re_transcribe`). An empty slice removes any
existing `transcript.json` rather than writing `[]`, preserving the
"absent-means-empty" invariant `TranscriptWriter` already honours.

**Phase 3 surface growth — notes.** `NotesStore` is a standalone, stateless
reader/writer for `notes.json` + `notes.md`, **independent of `MeetingWriter`**:
there is no shared open file handle. `MeetingWriter` owns `audio.opus` /
`transcript.json` / `metadata.json` while recording and never touches the notes
files; `NotesStore` only ever touches `notes.json` / `notes.md`. This split lets
the editor autosave (FR-18/FR-35) run concurrently with an active recording.

- `NotesStore::save(root, meeting_id, notes_json: &serde_json::Value, notes_md: &str) -> AppResult<()>`
  and `NotesStore::load(root, meeting_id) -> AppResult<Option<NotesData>>`, where
  `NotesData { json: serde_json::Value, markdown: String }`.
- **`notes.json` is stored as an opaque `serde_json::Value`** — the document
  shape is never modelled in Rust. Unknown/custom node types (the Phase-4
  transcript-chip node) round-trip losslessly. This opacity is the Phase-4
  transcript-chip guarantee; do not introduce a typed Tiptap model in this crate.
- Writes are **atomic** (write to a sibling `*.tmp` in the same dir, fsync,
  rename into place); a successful save leaves no `.tmp` residue. Loading an
  absent `notes.json` returns `Ok(None)`. `save` writes into the **existing**
  meeting folder — it does not create the folder and leaves sibling files
  (`audio.opus` / `transcript.json` / `metadata.json`) untouched.
- `MeetingFolder` exposes `notes_path()` / `notes_md_path()` helpers.

**CRDT notes storage (`ydoc` module — O2, `planning/DESIGN_notes-crdt.md`).**
When present, `notes.ydoc` (a single atomic lib0-v2 Yjs/`yrs` whole-state blob)
is the **authoritative** notes document; `notes.json` and `notes.md` are
**derived projections** (D-O2.1). The on-disk file set per meeting is therefore
`notes.ydoc` (authoritative binary) + `notes.json` (derived ProseMirror JSON) +
`notes.md` (derived markdown). `persistence` is the sole owner of all three.

- `NotesStore::save` builds a `yrs` doc from the incoming document JSON, writes
  `notes.ydoc` first (authoritative), then writes `notes.json` **derived from
  that doc** plus the caller-supplied `notes.md` — all three atomically in the
  one save call (D-O2.4). Markdown is caller-supplied because rendering it needs
  the editor's typed schema, which this crate does not model.
- `NotesStore::load` returns the JSON **derived from `notes.ydoc`** when it
  exists (so the projection self-heals if `notes.json` is missing or stale,
  exactly as the libsql index self-heals from the folders); it falls back to
  reading `notes.json` directly when `notes.ydoc` is absent (a pre-CRDT meeting
  not yet seeded). `Ok(None)` only when neither file exists.
- The `ydoc` module owns the JSON↔Yjs conversion (`json_to_ydoc`,
  `ydoc_to_json`, `encode_ydoc`, `decode_ydoc`). It is the **single, narrow**
  relaxation of the notes opacity guarantee: deriving ProseMirror JSON from the
  Yjs `XmlFragment` requires knowing the document is ProseMirror-shaped, but the
  walk is **generic** — element tags, attributes (stored as typed `yrs::Any`),
  text marks, and nesting all round-trip by structure, so unknown/custom nodes
  (transcript-chip atom, note images, future nodes) survive losslessly. No typed
  Tiptap node model is introduced. The mapping matches y-prosemirror (top-level
  `XmlFragment` named `"prosemirror"`, the doc node's children as the fragment's
  children) so the editor-side Yjs binding interops. Because `notes.json` is now
  a derived projection rather than a verbatim store, it is normalised to valid
  ProseMirror shape — custom node *types and attributes* are preserved, which is
  exactly what the transcript-chip guarantee requires.
- A round-trip test suite covers `JSON → yrs → JSON` (and the durable
  `JSON → yrs → v2 blob → yrs → JSON` hop) over the full editor schema —
  StarterKit blocks + marks, Link, lists, blockquote, code block, headings, the
  ParagraphAnchor `data-anchor-ms` attr, the TranscriptChip atom, NoteImage, and
  Table(+row/header/cell). It is the CRDT analogue of the `NotesStore` opacity
  test.

**Note image assets (`assets` module).** Images pasted/dropped into the notes
editor are stored as **separate files** under `{root}/{meeting_id}/assets/`,
NOT embedded in `notes.json`. The `assets` module is the sole writer/reader of
that subdirectory; `notes.json` is untouched (the opacity guarantee holds — the
editor stores only a bare filename into the document, which `NotesStore`
round-trips verbatim).

- `save_note_asset(root, meeting_id, bytes: &[u8], ext: &str) -> AppResult<String>`
  — creates `assets/` on demand and writes the bytes to
  `<sha256(bytes)>.<ext>` (a **content-hash** filename, so identical pastes
  dedupe to one file), via an atomic tmp+rename; returns the bare filename. The
  content hash uses the `sha2` crate, newly a direct dependency of
  `persistence` (already in the workspace dep set — `model-registry` uses it
  for model verification). This is a third-party dependency, not a
  crate-to-crate edge, so the dependency table above is unchanged
  (`persistence` still depends only on `common`).
- `read_note_asset(root, meeting_id, filename: &str) -> AppResult<Vec<u8>>` —
  **REJECTS** any `filename` containing a path separator or a `..` component
  (path-traversal guard, `AppError::InvalidInput`) before reading, so a request
  can only ever name a file directly inside the meeting's `assets/`.
- `MeetingFolder::assets_dir()` exposes the `{folder}/assets` path. The returned
  filename is a **portable** reference: it names only the file, so the meeting
  folder (with `assets/`) can be copied to another machine and the notes still
  resolve. `meeting_ops::delete_meeting`'s `remove_dir_all` removes `assets/`
  with the folder — no separate cleanup. See `cross-cutting.md` — "Note image
  assets".

**Phase 9 surface growth — `ChatStore` (chat session persistence).** `ChatStore`
is a standalone, stateless reader/writer for a meeting's chat sessions under
`{root}/{meeting_id}/chat/{session_id}.json` (one file per session), mirroring
`NotesStore`'s shape — **independent of `MeetingWriter`**, no shared handle.

- `ChatStore::save(root, meeting_id, &common::ChatSession) -> AppResult<()>`
  (atomic tmp+rename in the `chat/` subfolder, created on first save),
  `ChatStore::load(root, meeting_id, session_id) -> AppResult<Option<ChatSession>>`,
  `ChatStore::list(root, meeting_id) -> AppResult<Vec<ChatSession>>`
  (most-recently-updated first; an absent `chat/` folder is an empty list; a
  single unparseable session file is logged and skipped), and
  `ChatStore::delete(root, meeting_id, session_id) -> AppResult<()>` (idempotent).
- The chat driver in `ipc-bridge` persists a session **at turn end** through this
  store; `persistence` stays the **sole writer** under `meetings/`. `delete_meeting`
  already removes the whole meeting folder, so a meeting's chat sessions go with
  it — no separate chat cleanup is required.

**Phase 4 surface growth — readers, libsql index, summary, meeting ops.**
The minimal write-only crate grows to its full read/write surface. The
`libsql` dependency moves from "planned" to declared in
`crates/persistence/Cargo.toml` (the workspace pin already existed;
`tokio` is also now a direct dependency for `spawn_blocking`). No new
cross-component dependency edge is added — `persistence` still depends only
on `common`.

- **Readers (`reader` module), synchronous blocking `std::fs`.** Callers in
  an async context drive them via `tokio::task::spawn_blocking` (the
  threading-model rule). All take an explicit `meeting_dir` (`{root}/{uuid}/`):
  - `read_metadata(meeting_dir) -> AppResult<MeetingMeta>`
  - `read_transcript(meeting_dir) -> AppResult<Vec<Segment>>` — an absent
    `transcript.json` reads as an empty `Vec` (a zero-segment meeting writes
    no file), not an error.
  - `read_audio_pcm(meeting_dir) -> AppResult<Vec<f32>>` — the **graduated
    Opus decoder** (previously test-only). Returns the full **pause-INCLUDING**
    16 kHz mono f32 buffer: the silent frames written for pause gaps decode to
    real zero samples, so the buffer's duration equals wall-clock recording
    duration. This is what Phase 6 diarization and Phase 4 re-transcribe
    consume, and why the orchestrator sources audio through this reader so
    `diarizer` need not depend on `persistence`. The pause-INCLUDING property is
    covered in the **default** suite by a deterministic test
    (`test_read_audio_pcm_includes_silent_gap_deterministic`) that drives the
    actual pause path — `pause()` then a `#[cfg(test)]` `resume_with_pause_frames`
    seam that runs the same `finish_resume` silent-frame synthesis as `resume()`
    but with an injected frame count (no wall-clock sleep) — and asserts the
    decoded buffer spans ~4 s (so the synthesised pause silence was not dropped)
    with the injected-pause region decoding to ~zero. Because it exercises the
    real synthesis path, a regression that stops `resume()` writing silent frames
    fails this test (verified by mutation: the earlier draft pushed the silence
    through the sample stream and did **not** catch that regression).
  - `read_meeting_state(meeting_dir) -> AppResult<MeetingState>` — assembles
    `meta` + `transcript` + optional `notes` (via `NotesStore::load`, mapped to
    `common::NotesDocument`; the opaque `notes.json` value is re-serialised to
    the wire-facing string). This is the `open_meeting` restore payload. It is
    also the **lazy notes-CRDT migration trigger** (D-O2.7): on open, when
    `notes.ydoc` is absent but `notes.json` exists, it seeds `notes.ydoc` from
    the JSON (`NotesStore::seed_ydoc_if_needed`) and flips
    `MeetingMeta::notes_format` to `1` (rewriting `metadata.json`). The seed is
    idempotent (a no-op once `notes.ydoc` exists), build-invariant (the free
    build seeds too; only the sync transport is gated), and per-meeting — a
    never-opened meeting is never touched and stays JSON-readable. After
    seeding, `notes.ydoc` is authoritative.
  - `read_note_blocks(meeting_dir) -> AppResult<Vec<NoteBlock>>` (#70) —
    projects `notes.json` into `common::NoteBlock`s for the summariser via the
    pure `note_blocks_from_json(&Value)`. A best-effort READ projection (one
    block per non-empty `paragraph` node, carrying its `data-anchor-ms` anchor
    when present); it does NOT model the Tiptap schema or weaken the
    `NotesStore` opacity guarantee. Used by `ipc-bridge`'s summarise path and
    `agent-tools`' `resummarise` so notes weave into the transcript at their
    timestamp.
- **libsql index (`index` + `migrations` modules).** `MeetingIndex` opens (or
  creates) `index.db` at an **injected** path (`":memory:"` in tests) and runs
  a **forward-only migration runner** (`migrations::run`): a single-row
  `schema_version` table records the highest applied migration; `run` is
  idempotent and converges both an empty DB and a prior-schema DB onto the
  current schema additively (each step is `CREATE TABLE/INDEX IF NOT EXISTS`),
  so a derived-cache rebuild never loses reconstructable rows. The index holds
  one `meetings` row per `MeetingListEntry`. libsql is **async (tokio)**; the
  index API is `async fn` and the crate **never calls `block_on`**:
  - `MeetingIndex::open(db_path) -> AppResult<MeetingIndex>`
  - `list_meetings() -> AppResult<Vec<MeetingListEntry>>` (most-recent first,
    `started_at DESC`)
  - `search(query) -> AppResult<Vec<MeetingListEntry>>` (case-insensitive
    `LIKE` over title + excerpt; user wildcards escaped to match literally)
  - `upsert(&MeetingListEntry) -> AppResult<()>` (keyed on `id`)
  - `delete(MeetingId) -> AppResult<()>` (no-op when absent)
  - `rebuild_from_disk(meetings_root) -> AppResult<usize>` — `index.db` is a
    **derived cache**: this clears and repopulates the index by scanning every
    `{root}/{uuid}/` folder containing a `metadata.json`, deriving each
    `MeetingListEntry` (excerpt = first transcript segment). One unreadable
    folder is skipped with a warning rather than aborting the rebuild.
  - `reconcile_orphans(meetings_root) -> AppResult<usize>` — the in-session
    **self-heal**: an ADD-only (never deletes) counterpart to
    `rebuild_from_disk`. A `readdir` + set-diff against the indexed ids; only
    folders present on disk but missing from the cache incur a
    `metadata`/transcript read + `upsert`. Called by `ipc-bridge`'s
    `list_meetings` so a meeting can never stay hidden after a missed stop-time
    `upsert` (e.g. the process killed between finalise and the upsert) without
    waiting for the next startup `rebuild_from_disk`.
- **Meeting operations (`meeting_ops` module).** `rename_meeting(root, &index,
  id, new_title)` and `delete_meeting(root, &index, id)` (both `async fn ->
  AppResult<()>`) keep the on-disk folder and the index row consistent: the
  folder is authoritative (rename rewrites `metadata.json` atomically, delete
  removes the folder), then the index row is updated/removed to match. A crash
  between the two steps leaves the index stale-but-rebuildable.
  `set_speaker_name(root, id, label, name) -> AppResult<speaker_names map>`
  is the third op: a read-modify-write of `metadata.json`'s `speaker_names`
  (empty `name` clears the entry). It touches no index row (speaker names are
  not indexed), so unlike rename there is nothing to reconcile. Privacy
  invariant (#0014 audit): these ops log the meeting id (and the diarizer
  `label`), never the `new_title` or speaker `name` — both are user content that
  must not reach a log line (and thus the crash file / report excerpt, which
  capture info+ log lines).
- **Summary hook (`summary` module + `MeetingFolder::summary_path()`).**
  `write_summary(meeting_dir, &str)` (atomic tmp+rename) and
  `read_summary(meeting_dir) -> AppResult<Option<String>>` for `summary.md`.
  Phase 5's `summariser` produces the file; Phase 4 lands only the path helper
  and the I/O seam.

**Phase 6 surface growth — public atomic `write_metadata(meeting_dir,
&MeetingMeta)`.** A public free function in the `metadata` module (re-exported
at the crate root) that **atomically** (tmp + fsync + rename, matching the
notes/summary writers) rewrites `metadata.json` inside an existing
`{root}/{uuid}/` folder. It is the seam the orchestrator uses to update
`metadata.json`'s `{ speaker_count, diarizer }` after the diarization pass while
`persistence` stays the **sole** writer under `meetings/{uuid}/` (the diarizer
itself never touches disk). It does not create the folder and leaves the sibling
files (`audio.opus` / `transcript.json` / `notes.json`) untouched. The Phase-1
`MeetingWriter::finalise` path now also writes through the same atomic
implementation (via the crate-private `write_metadata_to_path`); `meeting_ops`'s
rename re-uses the public function rather than its prior private copy. No new
cross-component dependency edge — `persistence` still depends only on `common`.

**Translations sidecar — `translations.json`.** The `translations` module
holds per-language translations of transcript segments as a derived view. The
sidecar is indexed by `(language, segment_index)` and written by `ipc-bridge`'s
`translate_meeting` command.

- `translations_path(meeting_dir)` — path helper; mirrors the `summary_path()`
  and `notes_path()` helpers on `MeetingFolder`.
- `read_translations(meeting_dir) -> AppResult<HashMap<language, HashMap<index, text>>>`
  — absent file returns empty map.
- `merge_translations(meeting_dir, language, &HashMap<usize, String>)` — atomic
  read-modify-write that adds or overwrites entries for one language, leaving
  other languages untouched. The caller batches segments and flushes on a
  ~200 ms cadence (matching the progress-emit throttle) plus unconditionally
  on loop exit so partial progress survives interruption without O(n²) I/O.
- `clear_translations(meeting_dir)` — removes `translations.json`; idempotent
  on an absent file.

**Invariant:** `write_transcript` calls `clear_translations` after writing the
segment array. A full retranscription renumbers segment indices, so stale
translations would point at the wrong segments; the clear is at the only call
site that replaces all segments. Re-diarize does NOT call `write_transcript`
(only `speaker_id`s change, indices/text are unchanged), so translations survive
re-diarization. No new dependency edge — `persistence` still depends only on
`common`.

### `orchestrator`
**Crate:** `crates/orchestrator`
**Owns:** the live recording state machine. Wires `audio-capture →
vad-chunker → asr-runtime → persistence`, kicks off `diarizer` on stop,
emits typed events (transcript-segment-appended, meter-level,
state-changed) that the IPC bridge forwards to the webview.

**Thorniest crate.** Any change to the live pipeline lives here.
Parallel agents working on capture / VAD / ASR cannot independently
change the orchestrator's wiring — that's an orchestration-owner
decision and needs an architecture-doc update.

**Phase 1 surface (broadcast policy).** `AppEvent` fan-out uses
`broadcast::channel(256)` (~8 s of meter at 30 Hz). Slow subscribers
receive `RecvError::Lagged` from tokio and must warn at their call site;
the orchestrator does not pre-emptively drop subscribers. Meeting
titles use the placeholder convention `"Recording {ISO-8601 start
timestamp}"` until Phase 3/4 rename support lands.

**ASR flush backpressure (Phase 2).** The runner→ASR-worker flush path
uses an `Arc<Mutex<VecDeque<FlushPayload>>>` (capacity 4) + `Arc<Notify>`
instead of a plain `mpsc`. On overflow the runner drops the **oldest**
pending flush (not the newest) from the front of the deque and emits
`AppEvent::ErrorOccurred`. Audio is always preserved in `audio.opus`.

**Panic safety (Phase 2 close-out).** Each per-flush `transcribe_chunk` call is wrapped in `std::panic::catch_unwind`; a panic is caught, converted to `AppError::Internal`, emitted as `AppEvent::ErrorOccurred`, and the worker continues to the next flush. A `worker_exited` flag on `FlushQueue` ensures `stop()` is never wedged by a terminated worker.

**Phase 3 — `AppEvent::RecordingClock` emission.** The runner loop emits a
throttled `AppEvent::RecordingClock { meeting_id, clock_ms }` (~5 Hz; at most one
every 200 ms, tracked by a `last_clock_emit: Instant`) at the sample-batch
receive point, with `clock_ms = batch.end_ms` — the capture-sample,
pause-*excluding* clock (same timeline as `Segment::start_ms`). It is only sent
on the sample-receive path, so the paused branch (which never receives sample
batches) naturally does not advance the clock. The notes editor stamps paragraph
anchors from this event — see `cross-cutting.md` "Notes paragraph-anchor clock".
This is purely an additional event emission; the live pipeline wiring is
unchanged.

**Phase 4 — `Orchestrator::re_transcribe(&MeetingIndex, MeetingId)`.** The
offline re-run of transcription for a previously-recorded meeting (FR-33). It
refuses unless the recorder is `Idle` (returns `AppError::InvalidInput`) — an
offline re-transcribe must not contend with the live pipeline for the ASR model.
It decodes the meeting's `audio.opus` to the pause-INCLUDING 16 kHz mono PCM via
`persistence::reader::read_audio_pcm`, then **reuses the live runner's batched-VAD
machinery**: the same `VadChunker` + `Accumulator` (zero-padded, `MAX_GAP_MS`-capped
silence preservation) + the same `FLUSH_MIN_SECS` size-trigger + the same
proportional re-split (`emit_segments_proportional`) and the same `AsrRuntime`
resolution path (`init_asr_runtime` → model-registry `ensure`) the live worker
uses (`runner::re_transcribe_buffer`). The 30 s encoder-window constraint and the
silence-preservation rule therefore hold identically; the accumulator code is not
re-implemented. To reconstruct the pause-EXCLUDING clock it splits the decoded
PCM into the non-pause regions (a run of ≥`PAUSE_MIN_MS` near-silence is a pause)
and feeds only those, advancing the clock over kept audio only. At each region
(pause) boundary it **flushes and `reset()`s the `VadChunker`** so the pre-pause
utterance is closed there — the skipped silence would have closed it via hangover
in the live path — instead of merging with the post-pause utterance across the
join (the live path splits at the pause; the offline path must match it).
Differences from the live path: no flush queue / ASR-worker
thread — the work runs synchronously on one `spawn_blocking` thread, one
accumulator flush at a time, so segments can be collected in order. As segments
are produced it emits `AppEvent::TranscriptSegment` (the same event the live path
emits), then `finalise_retranscribe` **carries the prior diarization onto the new
segments** via `diarizer::overlay_speakers_from_prior` (time-overlap join against
the old `transcript.json`; see `diarizer` section) so `MeetingMeta.speaker_names`
stays valid without any key remapping — a meeting that was never diarized leaves
all new segments as `None` with no regression. `metadata.json`'s `speaker_count`
is updated to reflect the distinct labels in the new transcript. Then rewrites
`transcript.json` via `persistence::write_transcript` (atomic tmp+rename; an empty
result removes the file), and refreshes the index row (`MeetingIndex::upsert`) so
the meeting-list excerpt reflects the new first segment, then emits
`AppEvent::TranscriptReady { meeting_id }` so the webview re-reads the transcript
(mirroring `DiarizationComplete`). The ASR run is wrapped
in a length-relative timeout (`retranscribe_timeout`: ≈3× real-time, floored 5
min / capped 30 min — generous, since ASR is slower than diarization), so a
wedged run cannot hold the offline claim forever. Unlike the live path's
best-effort skip when no model is present, an explicit user-triggered
re-transcribe with no available model is an error (`AppError::ModelLoad`). The
orchestrator does not own a `MeetingIndex`; the index handle is passed in by
`ipc-bridge` (which owns it in `IpcState`). Besides the user-triggered command,
`ipc-bridge` also spawns this as a **background pass after `stop()`** when the
live transcript fell behind (`take_transcript_incomplete()` — set by the runner
on a drop-oldest flush or a stop-drain timeout), repairing both mid-recording
drops AND a truncated tail from the complete audio; that background invocation
logs and swallows errors (a missing model, or an `InvalidInput` claim-skip when
the recorder is busy) rather than surfacing them, since the live transcript is
already on disk. A failed/skipped background re-transcribe is NOT auto-retried
(the flag is consumed) — the user-triggered command is the recovery.

**Test seam — `re_transcribe_with_backend(&MeetingIndex, MeetingId, Box<dyn
AsrBackend + Send>)`.** A `#[cfg(any(test, feature = "test-source"))]`-gated
sibling of `re_transcribe`, mirroring the live path's
`start_with_streams_and_backend`. It decodes `audio.opus` and drives the **same**
`runner::re_transcribe_buffer` machinery (real Silero VAD + the batched-VAD
accumulator + `transcribe_one_flush` + `write_transcript` + index `upsert`) the
production `re_transcribe` uses, but with a caller-supplied `AsrBackend` stub
instead of resolving a real `AsrRuntime`. Both paths share the private
`ensure_idle_for_retranscribe` (the `Idle`-only invariant) and
`finalise_retranscribe` (transcript rewrite + index-row refresh) helpers, so the
only difference is segment *production*. This lets the **default** test suite
cover the whole offline path over the committed real-speech fixture without a
~1 GB model (see "Integration tests" below). It is compiled out of production
builds, so the public production surface is unchanged.

**Phase 5 — `Orchestrator::ensure_model_path(&ModelId) -> AppResult<PathBuf>`.**
An **additive** thin wrapper over the existing `model-registry` handle
(`ModelRegistry::ensure`, which downloads + verifies when absent) that returns
the resolved per-model **directory**. `ipc-bridge`'s `summarise_meeting` calls
it to locate the selected LLM directory before opening the summariser, keeping
the `model-registry` edge inside the orchestrator. This adds **no**
`orchestrator → summariser` edge — the summariser is loaded in `ipc-bridge`
(the granted `ipc-bridge → summariser` edge), not here.

**Phase 6 — diarization (FR-11): the granted `orchestrator → diarizer` edge,
the on-stop pass, and `Orchestrator::rediarize`.** The orchestrator owns the
diarizer lifecycle (per the `diarizer` section above): `diarizer = { path =
"../diarizer" }` is added to `crates/orchestrator/Cargo.toml`, realising the
`orchestrator → diarizer` edge in the dependency table. A lazy builder
(`runner::build_diarizer`, mirroring `build_asr_runtime_for_retranscribe`)
resolves the two diarize model directories via `model-registry`
(`ModelRegistry::ensure` for `pyannote-segmentation-3-0` +
`3dspeaker-campplus-zh-en-advanced`), locates each `.onnx`, and opens
`SherpaDiarizer::open(seg, emb, DiarizerConfig::default())` — so the
`model-registry` edge stays inside the orchestrator and `diarizer` need not
depend on `persistence` (the orchestrator sources audio through
`persistence::read_audio_pcm`).

- `Orchestrator::rediarize(&MeetingIndex, MeetingId)` — the offline
  user-triggered re-diarize, copying `re_transcribe`'s one-shot idiom: it refuses
  unless `Idle` (`AppError::InvalidInput`), then on `spawn_blocking` decodes the
  pause-INCLUDING PCM (`read_audio_pcm`) + reads `transcript.json`
  (`read_transcript`) and runs `Diarizer::assign_speakers(&audio, 16000, &mut
  segments)` (distinct-count). It rewrites `transcript.json` with the overlaid
  `speaker_id`s (`write_transcript`), updates `metadata.json`'s `{ speaker_count,
  diarizer: Some(ModelDescriptor{..}) }` (`persistence::write_metadata`), refreshes
  the supplied index row's `speaker_count` (`MeetingIndex::upsert`), and emits
  `AppEvent::DiarizationComplete { meeting_id, speaker_count }` on the shared
  `event_tx`. The index handle is passed in by `ipc-bridge` (the orchestrator does
  not own one), exactly as for `re_transcribe`. The (uninterruptible) sherpa
  `compute` is wrapped in a **length-relative timeout** (`diarize_timeout`: ≈1×
  real-time, floored at 2 min / capped at 10 min — sized from `metadata.duration_ms`);
  on timeout `rediarize_inner` returns `AppError::Inference` BEFORE any write, so a
  pathologically slow or wedged pass leaves the meeting un-diarized instead of
  blocking forever. (`tokio` cannot cancel the `spawn_blocking` thread, so a true
  infinite hang leaks one thread until exit; the budget bounds the wait, not the
  thread.)
- **On-stop pass — decoupled, background.** Diarization is NOT run inline in
  `stop()`. `stop()` finalises the meeting and returns it **un-diarized**
  (`speaker_count 0`, `diarizer None`) the instant it is on disk and the recorder
  is back to `Idle`, exposing the user's choice via
  `Orchestrator::diarization_enabled()`. When that is true, `ipc-bridge` — AFTER
  it has indexed the meeting (so visibility is immediate) — **spawns `rediarize`
  in the background**: the on-stop pass IS the re-diarize pass, just
  auto-triggered, so it claims the offline slot, applies the timeout above,
  rewrites `transcript.json` + `metadata.json`, refreshes the index row, and emits
  `DiarizationComplete` when done. A slow or hung diarization therefore can never
  wedge `stop()` or hide the meeting. The flag defaults to **false**. (Previously
  `stop()` ran this pass INLINE/awaited; a hung 30-min diarization then blocked the
  whole stop flow and left the meeting unindexed until the next launch — fixed by
  this decoupling.)
- **Test seam — `rediarize_with_diarizer(&MeetingIndex, MeetingId, Box<dyn
  Diarizer + Send>)`.** A `#[cfg(any(test, feature = "test-source"))]`-gated
  sibling of `rediarize` (mirroring `re_transcribe_with_backend`): both `rediarize`
  and this seam delegate to a shared `rediarize_inner` taking an owned
  `Box<dyn Diarizer + Send>`, so the default suite exercises the full
  decode → assign → `transcript.json` rewrite → `metadata.json` update →
  index-upsert → `DiarizationComplete` wiring with a `StubDiarizer` (NO model).
  `DiarizationComplete` is emitted by the **orchestrator**, not `ipc-bridge`.

**Integration tests** live in `crates/orchestrator/tests/` (per
`cross-cutting.md` — Testing). Phase 1 integration tests:
`start_record_stop` (full lifecycle + pause/resume decoded-duration
accuracy + invalid transitions) and `back_pressure` (slow subscriber
lag and subscriber-gone survivability). Phase 2 integration test:
`transcription_e2e` (env-var-gated end-to-end pipeline: DummyAudioSource
→ VAD → ASR → TranscriptSegment events + transcript.json on disk). Run with
`cargo test -p orchestrator --features test-source`. Phase 4 offline
re-transcribe tests (`re_transcribe`): the gated
`re_transcribe_rewrites_transcript_over_fixture` (records via the real ASR model,
then re-transcribes) plus the **default-suite, model-free**
`re_transcribe_with_stub_backend_rewrites_transcript_over_fixture` — it encodes
the committed LibriSpeech fixture into `audio.opus` via the persistence Opus
encoder, empties `transcript.json`, then runs `re_transcribe_with_backend` with a
`StubAsrBackend` so the real Silero VAD + offline accumulator + transcript
rewrite + index-excerpt refresh are exercised in CI without a model. Phase 6
diarization tests: the **default-suite, model-free** `StubDiarizer` lib tests
(`tests::diarization`) drive the re-diarize inner path
(`rediarize_with_diarizer` → `transcript.json` rewrite with `speaker_id`s +
`metadata.json` `speaker_count` + `DiarizationComplete`); two `stop()` tests
assert diarization is now **decoupled** from `stop()` — both
`diarization_enabled = true` (`stop_with_diarization_enabled_is_decoupled_from_stop`)
and `false` return the meeting un-diarized (`speaker_id == None`, `speaker_count
0`, and no `DiarizationComplete` from `stop()` itself — the background `rediarize`
pass emits it), with `diarization_enabled()` surfacing the toggle for ipc; plus
the env-var-gated `rediarize` integration test
(`MINUTIST_DIARIZE_SEG_PATH` + `MINUTIST_DIARIZE_EMB_PATH`, skip-on-unset)
that stages the two real sherpa models into the registry cache and re-diarizes a
meeting whose audio is the S1 two-speaker fixture.

**Phase B — live diarization wiring (additive).** At record start, gated on
`diarization_enabled` AND the embedding model being locally `Available` (no
download), the orchestrator builds an `Arc<OnlineDiarizer>` (embedding-only) via
a local-only resolver (`runner::build_online_diarizer`, reusing
`DIARIZE_EMB_MODEL_ID` — the SAME embedding model the on-stop `build_diarizer`
uses, so live + offline share one model on disk). The resolver does a synchronous
`Available`-check (`ModelRegistry::list_models` → `compute_status_sync`, a
`std::fs` size-only check — the same non-blocking, no-network precedent
`init_asr_runtime` uses) and NEVER calls `ensure()`; the heavy
`EmbeddingExtractor::new` load runs inside a `spawn_blocking` so the async runtime
is never stalled, mirroring the on-stop diarizer build. The resulting
`Option<Arc<OnlineDiarizer>>` is threaded into the runner (`spawn_runner` →
`run_drain_loop` → `finalise_on_stop`). `assign_segment` is called per VAD segment
at SegmentEnd, on the runner's drain-loop thread, from the still-un-padded
per-segment slice (the accumulator's `MAX_GAP_MS` zero-pad cap makes per-segment
boundaries unrecoverable from the flushed buffer, so the label MUST be assigned
here, not re-derived in the ASR worker). The label rides a parallel
`Option<String>` column: `Accumulator.speaker_ids` → `FlushPayload.speaker_ids`
→ `emit_segments_proportional` → `Segment.speaker_id` (indexed by the same
enumerate `i`, defensively via `.get(i)`), so each re-split sub-Segment inherits
its originating VAD segment's label. Best-effort and additive: setting off / model
absent / open failure / per-segment `assign_segment` error all degrade to "no
label" (logged) with recording and transcription unaffected — no `ensure()`, no
download, no block, no `unwrap` on the diarizer path. The on-stop `SherpaDiarizer`
pass remains AUTHORITATIVE: when `diarization_enabled` is true it rewrites the
whole transcript on stop, overwriting the live labels. No dependency-table change
(the `orchestrator → diarizer` edge already exists; `OnlineDiarizer` is re-exported
from the `diarizer` crate). Tests: model-free default-suite unit tests cover the
label threading through `emit_segments_proportional` (positional carry, all-None
regression, short-slice `.get(i)` guard, mixed Some/None) and the `Accumulator`
label column (lockstep len invariant, drain reset, gap-cap correspondence), plus
`build_online_diarizer_returns_none_when_model_absent` (the no-download guarantee
over an empty cache); the `live_diarization` integration test asserts the None-path
yields all-None `speaker_id` (the "must not break transcription" regression guard),
with an env-var-gated (`MINUTIST_DIARIZE_EMB_PATH`) positive case asserting
non-None live labels.

**Pause/resume command delivery.** `Orchestrator::pause`, `resume`, and `stop`
all deliver their writer commands (`WriterPause`/`WriterResume`/stop) onto the
runner's `cmd_tx` channel via the awaiting `send()` — control commands are
back-pressured, never dropped. The state lock is released before the await so it
is never held across an async yield; the runner never takes the state mutex, so
a busy or exited writer cannot deadlock the caller (a closed channel returns an
error, which is logged). Reliable delivery is what keeps the encoder-pause
silence aligned with the pause-excluding timeline: a lost `WriterPause` would
leave the encoder running, and a lost `WriterResume` would strand it in Paused
with every subsequent `push_samples` failing.

**Stop drains queued samples through the VAD.** Both stop branches of the runner
loop (Recording-stop and paused-stop) drain every sample batch still queued in
`streams.samples` through the persistent writer (`push_batch`) AND the VAD
(`process_samples`) — via the shared `drain_samples_through_vad` helper, so the
two branches cannot diverge — before calling `finalise_on_stop`, whose
end-of-stream flush then closes any in-progress segment from the tail audio.
The paused branch otherwise blocks on `cmd_rx` and never reads
`streams.samples`, so batches accepted before the pause would be stranded and
the recording's final utterance lost.

**Phase 9 — `Orchestrator::transcribe_pcm_window(MeetingId, start_ms, end_ms,
language) -> AppResult<Vec<Segment>>`.** Backs the `agent-tools`
`relisten_section` tool. A **read-only** compute op — it does NOT rewrite
`transcript.json` and does NOT take the offline claim, so it is safe during a
live recording (at a transient second-ASR-model memory cost; the backend is
built fresh inside `spawn_blocking` and dropped after the call). `start_ms`/`end_ms`
are **transcript-clock (pause-EXCLUDING)** timestamps — the only timeline an
agent reading a transcript has. The pause-clock conversion onto the
pause-INCLUDING decoded PCM lives in `runner::pcm_window_for_excluding_range`,
which walks the `pause_excluding_segments` kept regions and **clamps a window
that straddles a pause to the kept region containing its start** (the documented
W1 decision — re-transcribed timestamps cannot be cleanly re-mapped back across a
pause concatenation seam; `pause_excluding_segments` stays `pub(crate)`). ASR
backend resolution reuses the live/re-transcribe engine routing via
`runner::build_asr_backend_for_retranscribe`, keeping the `model-registry` edge
inside the orchestrator — `agent-tools` never reaches `model-registry`. A
`#[cfg(any(test, feature = "test-source"))]` sibling
`transcribe_pcm_window_with_backend` injects an `AsrBackend` stub (mirroring
`re_transcribe_with_backend`) so the window-mapping + read-only behaviour are
covered model-free; the `runner::pcm_window_for_excluding_range` mapping has
gating unit tests over a synthetic PCM with a known mid-window pause.

**Phase 9 — rediarize clears `speaker_names` (§4.4).** The shared
`finalise_diarization` metadata write now also clears
`MeetingMeta.speaker_names` (in the same `write_metadata` it already performs —
no second write). A (re-)diarization pass can re-letter speakers, so a
user-set name map keyed on the OLD letters would silently mis-label; clearing is
the only safe cross-consumer behaviour (an MCP client cannot re-map the way the
UI could). See `cross-cutting.md` "Agent chat loop".

### `agent-tools`
**Crate:** `crates/agent-tools` (Phase 9)
**Owns:** the shared tool layer — one `Tool` trait + one `ToolRegistry`, the
single place a chat-agent / MCP tool is defined. Both consumers (the Phase-9
internal chat agent and the Phase-10 MCP server) drive the SAME registry, so the
"internal agent and an external MCP client use the same tools" constraint is
satisfied by there being exactly one definition site per tool. Edges: `common`,
`persistence`, `orchestrator`.

**Deliberately NOT edges.** No `summariser` edge — the one LLM-using tool
(`resummarise`) drives an `Arc<dyn common::Summariser>` held in `ToolContext`,
constructed by `ipc-bridge`/`app-main` (which own the `summariser` edge; the
bundled impl is `Send + Sync` per SP0). No `model-registry` edge —
`relisten_section` resolves and builds its ASR backend through
`Orchestrator::transcribe_pcm_window`, never by calling `model-registry`. No
`tauri`/`specta` — `serde_json::Value` results cross the IPC boundary as a
`String` in `ipc-bridge`'s event envelope, not here; the `AppError → McpError`
mapping is Phase 10's concern and lives in `mcp-server` (keeps `rmcp` out of this
crate).

**The `Tool` trait** (`Send + Sync`, async `execute`): `name() -> &'static str`
(stable snake_case wire name), `title() -> &'static str` (required — every tool
MUST implement it; a missing impl is a compile error; the title is a short
human-readable label distinct from the snake_case name, projected onto the MCP
`tools/list` `title` field via `Tool::with_title` in `mcp-server`),
`description()`, `input_schema() ->
serde_json::Value` (JSON Schema 2020-12, object root, **no regex `pattern`** — the
vendored llama.cpp schema→GBNF converter rejects PCRE shorthands), `is_write() ->
bool`, `expose_over_mcp() -> bool` (default `!is_write()`), and the async
`execute(&ToolContext, args) -> AppResult<ToolOutput>`. `execute` is async because
the backing ops are async (the orchestrator's offline ops, libsql index queries);
tool bodies still push CPU/fs/inference work onto `spawn_blocking`.
`ToolDescriptor` carries `name`, `title`, `description`, and `input_schema` (pure
projection; `ToolRegistry::descriptors` / `mcp_tool_descriptors_gated` emit it).
The rmcp 1.7 `Tool` type exposes `.with_title(str)` which sets the top-level
`title` field on the MCP tool object (MCP spec 2025-11-25 §tools.title); the
`mcp-server` handler uses this method, not `ToolAnnotations.title`, because the
spec promotes title to a first-class field from revision 2025-11-25 onward.

**`ToolContext`** (Clone): `Arc<Orchestrator>`, `Arc<MeetingIndex>`,
`meetings_dir: PathBuf`, `Arc<dyn Summariser>`, the shared
`broadcast::Sender<AppEvent>`, an optional `default_meeting` (the internal-UI
session scope; MCP leaves it `None` so an MCP caller passes `meeting_id`
explicitly), a per-meeting metadata-write mutex map, and (Phase 10) an optional
inter-agent bridge SENDER. `default_meeting` lets a tool resolve an omitted
`meeting_id` via `resolve_meeting`, but the MODEL must also be TOLD a meeting is
in scope or it asks the user for an id: when the chat is meeting-scoped,
`send_chat_message` / the inter-agent bridge append a "# Current meeting" block
(meeting id + title) to `chat_system_prompt` via
`chat_system_prompt_for_meeting`, instructing the agent to call the tools
(which default to this meeting) rather than ask, AND relax `meeting_id` from the
offered schemas' `required` (`agent_tools::relax_meeting_id_requirement`) so a
schema-respecting model is free to omit it. The context also holds a
per-meeting metadata-write mutex map and (Phase 10) an optional
inter-agent bridge SENDER (`mpsc::Sender<(InterAgentRequest, oneshot)>`, set via
`with_inter_agent_bridge` for the MCP registry context only; `None` for the
internal agent so it cannot message itself). The bridge uses only `common` types
+ tokio channels — no `chat-agent` edge.

**`ToolRegistry::v1(include_inter_agent_bridge: bool)`** registers the 17 base v1
tools in insertion order; `ipc-bridge` passes `false` (the internal agent must
not message itself) and `app-main` passes `true` for the MCP registry instance,
which APPENDS `send_to_internal_agent` (18 tools). `descriptors()` /
`mcp_tool_descriptors()` are pure name/description/schema projections (single
source of truth); `mcp_tool_descriptors()` honours `expose_over_mcp()`.
**`mcp_tool_descriptors_gated(allow_writes)`** (Phase 10) composes the
`mcp_write_tools` setting (D3) on TOP of `expose_over_mcp()`: with it off, write
tools are dropped (reads + the inter-agent tool only); with it on, the reversible
writes join; `retranscribe`/`rediarize` are never `expose_over_mcp` and so never
appear regardless. `mcp_call_allowed(name, allow_writes)` mirrors that gate on
`tools/call` (defence in depth). `dispatch(ctx, name, args)` is the one routing
path: unknown name → `InvalidInput`; shallow arg-shape validation → `InvalidInput`;
then `execute`.

**v1 tools.** Read/compute: `list_meetings`, `search_meetings`, `get_meeting`,
`get_transcript`, `get_transcript_slice`, `get_summary`, `get_notes`,
`get_metadata`, `get_recording_state`, `search_within_transcript`,
`relisten_section`, `resummarise`, `speaker_talk_time`. Writes:
`set_speaker_name`, `rename_meeting` (both MCP-allowlisted — reversible, low
blast radius), `retranscribe_meeting`, `rediarize_meeting` (internal-only — heavy;
holding the offline claim via MCP would block the user's ability to record).
Record-control writes (#62): `start_recording` (optional `device_id`, returns the
new `MeetingId`), `stop_recording` (returns the finished meeting's id + title +
duration), `pause_recording`, `resume_recording` — each dispatches to the
matching `Orchestrator` method (`start`/`stop`/`pause`/`resume`), adding no new
dependency edge. All four are `is_write` AND override `expose_over_mcp() == true`,
so they are **write-gated** like `set_speaker_name`/`rename_meeting`: absent +
rejected when `mcp_write_tools` is OFF (the default), exposed + callable when it
is ON — the deliberate opt-in that lets an external MCP client drive the
record→transcribe→read loop for E2E (off by default, behind the bearer token +
loopback). The internal UI chat (no MCP gate) can always drive them. MCP-only
(Phase 10, `v1(true)`): `send_to_internal_agent` — forwards one
message to the internal chat agent over the bridge channel and returns its reply
(body in `agent-tools`; chat-engine driver in `ipc-bridge::inter_agent`).

**Speaker-name overlay.** `get_transcript`, `get_meeting`,
`search_within_transcript`, and `speaker_talk_time` apply the
`MeetingMeta.speaker_names` map at read time, rewriting a segment's `speaker_id`
label (`"A"`) to its display name (`"Alice"`) where one is set. Presentation-only
— the on-disk transcript is never mutated. `set_speaker_name` writes the map via
`persistence::write_metadata`; `rediarize_meeting` resets it (orchestrator §4.4).

**Write serialization (§4).** `persistence` stays the sole writer under
`meetings/`. `retranscribe_meeting`/`rediarize_meeting` inherit the orchestrator's
offline claim for free (`InvalidInput` when busy). `set_speaker_name` and
`rename_meeting` are read-modify-writes of `metadata.json` that bypass that
claim, so they take a `ToolContext`-owned **per-meeting async mutex** across the
read-modify-write — the one tool-layer-owned write lock. `relisten_section` and
`resummarise` are read-only-with-compute (write nothing). The record-control
tools (`start_recording`/`stop_recording`/`pause_recording`/`resume_recording`)
own no write lock of their own — they delegate straight to the orchestrator's
recording state machine, which serialises lifecycle transitions under its own
lock and rejects an invalid transition with `InvalidInput`.

### `chat-agent`
**Crate:** `crates/chat-agent` (Phase 9)
**Owns:** the stateless, OpenAI-compatible, tool-calling chat TURN engine over
the bundled local LLM. It sits ABOVE both `summariser` (the loaded-model
substrate) and `agent-tools` (the tool descriptors); folding the loop into
`summariser` would force a backwards `summariser → agent-tools` edge. Edges:
`common`, `summariser`, `agent-tools` (+ external `llama-cpp-2`, `serde`,
`serde_json`, `thiserror`, `tracing`, `encoding_rs`).

**Deliberately NOT edges.** No `tauri`/`specta`, no `persistence`/`orchestrator`
directly (the DRIVER reaches those through `agent-tools`), no `model-registry`
(it reuses the held model via the substrate seam), no `common`-trait addition —
the engine types (`ChatEngine`, `ChatMessage`, `TurnOutcome`, `SamplerConfig`,
`TurnBackend`) live in `chat-agent`, not `common`, because no `common`-level
signature names them (the asymmetry with `common::Summariser` is principled:
`Summariser` is named by a `common` type — `agent-tools::ToolContext` — so it
stays in `common`).

**Stateless per call; the driver owns the loop (§1.2/§1.3).** The engine runs
ONE assistant turn: `ChatEngine::run_turn(history: &[ChatMessage],
tool_descriptors: &[agent_tools::ToolDescriptor], cfg: &SamplerConfig, token_cb:
&mut dyn FnMut(&str)) -> AppResult<TurnOutcome>`. It does NOT own the
conversation history, does NOT dispatch tools, and does NOT emit `AppEvent`s
(it holds no broadcast handle). The DRIVER (`ipc-bridge`, a later phase) owns the
`Vec<ChatMessage>` history + the sliding window + the turn loop + the
max-iteration cap, dispatches via `agent_tools::ToolRegistry::dispatch`, appends
a tool-result message, and calls `run_turn` again. A `TurnOutcome` is either
`Final(String)` (a final assistant reply — stop the loop) or
`ToolCalls(Vec<ToolCall>)` (calls for the driver to execute).

**oaicompat tool calling (§0a).** `run_turn` converts the history to an
OpenAI-format `messages_json` and the descriptors to an OpenAI `tools_json`
(`{"type":"function","function":{name,description,parameters:<input_schema>}}`),
then the real backend renders the prompt via
`LlamaModel::apply_chat_template_oaicompat` (the GGUF's own tool template),
generates over a FRESH `LlamaContext` (clean KV cache), streams content via the
`ChatParseStateOaicompat` streaming parser (tool-call JSON is NEVER streamed
through `token_cb`), and does a final authoritative `parse_response_oaicompat`
into a `RawTurn`. The engine maps `RawTurn` → `TurnOutcome`: non-empty tool
calls ⇒ `ToolCalls`; else non-empty text ⇒ `Final`; else malformed →
`AppError::InvalidInput`.

**Sampling (§6.4).** A `temp/top_p/dist(seed)` chain by default; **greedy when
`temperature == 0.0`** (the deterministic test mode). A lazy GBNF grammar
(`json_schema_to_grammar` over the offered-tool schemas, snapped via
`grammar_lazy` on the template's tool-call trigger) is the reliability backstop
for the 4B model — wired but behind `SamplerConfig::grammar_backstop`.

**The substrate seam (D5).** The real turn needs the loaded `LlamaModel`.
`summariser` exposes it via `LlamaSummariser::model() -> &LlamaModel`.
`ipc-bridge` holds the concrete `Arc<LlamaSummariser>`, lends `&LlamaModel` to
`LlamaTurnBackend`, and coerces the same handle to `Arc<dyn Summariser>` for the
`agent-tools` `ToolContext`. The model is `Send + Sync`; no GGUF is reloaded per
turn.

**Testability (the `TurnBackend` seam).** The FFI LLM call is behind a
`TurnBackend` trait (`run(messages_json, tools_json, cfg, token_cb) ->
Result<RawTurn, Error>`). The real `LlamaTurnBackend` uses the oaicompat APIs; a
test stub returns canned text/tool-calls. The engine's turn logic (prompt
assembly, outcome parsing, tool-call extraction, error mapping, the
sliding-window trim, and the CI gate that compiles every registry schema through
`json_schema_to_grammar`) is unit-tested with the stub (no model);
`LlamaTurnBackend` gets a gated test (`#[ignore]` / skip-on-unset
`MINUTIST_LLM_MODEL_PATH`), mirroring the `summariser`/`asr-runtime` gated
tests.

**Context budget (§6.2, "until context full").** A pure `trim_to_budget`
helper + a `fits_budget` check live here even though the DRIVER applies them: it
pins turn 0 (the system prompt + tool list, NOT the full transcript), evicts the
oldest non-pinned turns until the re-tokenised windowed prompt fits `prompt +
max_tokens + reserve <= n_ctx`, and reports a hard floor (`HARD_FLOOR_REJECT`)
when a single turn is genuinely too large (the driver rejects it as
`AppError::InvalidInput`).

### `mcp-server`
**Crate:** `crates/mcp-server` (Phase 10)
**Owns:** the in-process Streamable HTTP MCP server that exposes the Phase-9
`agent-tools` registry to external agents over loopback. It is a SECOND consumer
of that registry and adds **no tools of its own** — it projects
`ToolRegistry::mcp_tool_descriptors_gated(allow_writes)` onto MCP `tools/list`
and `ToolRegistry::dispatch` onto `tools/call`. Any tool logic, schema, or name
living here rather than in `agent-tools` is a reviewer finding. Edges: `common`,
`agent-tools` (+ the external `rmcp` SDK 1.7 and its `hyper`/`hyper-util`/
`http`/`http-body-util`/`tower-service`/`tokio-util` leaf crates — the
`AppError → McpError` mapping is the **only** place rmcp error types are
constructed). SDK: `rmcp` 1.7 (`server`, `macros`,
`transport-streamable-http-server`, `schemars`); rmcp's own hyper-based
`StreamableHttpService` serves the single `/mcp` endpoint — **no `axum`** (Gate-A
SP1: `cargo tree -d` showed no `http`/`hyper`/`tower` skew against Tauri 2.11,
which already resolves the same majors).

**Deliberately NOT edges.** No `tauri`/`specta` (the listener is spawned by
`app-main` via `tauri::async_runtime::spawn`; `mcp-server` takes the registry +
context + a shutdown receiver + bind/token config and serves until shutdown). No
`chat-agent` edge — the inter-agent bridge tool (`send_to_internal_agent`)
reaches the chat engine through a `common`-typed channel held on the
`agent_tools::ToolContext` (the SENDER), whose receiver + the single chat turn
live in `ipc-bridge`. No direct `persistence`/`orchestrator` edge — it drives the
`ToolRegistry`, whose `ToolContext` carries those handles (built by `app-main`).

**`serve(registry, ctx, config, shutdown)`** binds `config.bind_addr`
(`127.0.0.1:{mcp_port}`), wraps rmcp's `StreamableHttpService` in a thin
bearer-check hyper service (401 before rmcp sees the request — the session id is
never the credential), and serves until the `watch` shutdown flips. Host + Origin
validation are rmcp-native (`StreamableHttpServerConfig`: the loopback
`allowed_hosts` default, kept; `allowed_origins` set to the loopback origins so a
cross-origin browser request is a 403). The write-tool exposure gate
(`allow_writes` = `settings.mcp_write_tools`, D3) is applied at projection AND on
call (`mcp_call_allowed`). See `cross-cutting.md` — "MCP transport".

**Tool projection** (`McpToolHandler::list_tools_projection`): for each gated
descriptor, builds an rmcp `Tool` via `Tool::new(name, description, schema)` then
calls `.with_title(title)` (rmcp 1.7 `Tool::with_title`, setting the top-level
`title` field — not `ToolAnnotations.title`) and `.with_annotations(...)` for
`readOnlyHint` / `destructiveHint` / `openWorldHint`. Every projected rmcp `Tool`
carries a non-empty `title` distinct from its snake_case `name` (asserted by two
tests in `handler.rs`: `every_projected_tool_has_non_empty_title_distinct_from_snake_case_name`
checks the descriptor level; `list_tools_projection_rmcp_tools_have_title` checks
the rmcp `Tool` structs AND asserts that `serde_json::to_value` produces a
top-level `"title"` key, confirming spec compliance through the serde layer).
The title originates from `Tool::title()` in `agent-tools` — the single source of
truth.

**The inter-agent tool placement.** `send_to_internal_agent` is DEFINED in
`agent-tools` (registered only on `ToolRegistry::v1(true)`, the MCP registry) so
the single-tool-definition rule holds, but its BODY only `try_send`s an
`InterAgentRequest` over the bridge channel + awaits the reply (with a timeout).
The chat-engine driver that services the channel lives in
`ipc-bridge::inter_agent` (using the INTERNAL `v1(false)` registry so the agent
cannot message itself). This keeps `agent-tools` free of a `chat-agent` edge and
`mcp-server` free of both.

### `settings`
**Crate:** `crates/settings`
**Owns:** the settings schema, validation, change notifications.
Backed by a single JSON file (`{app-data}/settings.store`) read/written via
`serde_json` + `std::fs`; the resolved `PathBuf` is injected by `app-main` at
construction time (no `tauri::*` in this crate). Change notifications use a
`tokio::sync::watch` channel (capacity 1; subscribers always see the latest
value) broadcast directly from `SettingsHandle::update` — not via the
orchestrator.

Single source of truth for runtime configuration. Other components read
settings via this crate; nobody else parses the store directly.

**Phase 3 field — `autosave_interval_secs: u32`.** Notes-editor autosave
cadence (FR-18/FR-35), `#[serde(default = ...)]` defaulting to 5; an older
store JSON written before the field existed deserialises to 5. `Settings`
now carries an explicit `Default` impl (the field's default is non-zero, so
the derived `Default` no longer suffices).

**Phase 5 fields — `summary_system_prompt: String` (FR-28) and
`llm_model_id: Option<ModelId>` (FR-35).** The summary prompt
`#[serde(default = ...)]`-defaults to a structured-summary instruction
(headings / key decisions / action items) the `summariser` forwards verbatim
as the chat `system` message; an older store deserialises to that default.
`llm_model_id` selects the summarisation LLM, `#[serde(default)]`-defaulting
to `None` ("use the bundled default model"); the model is settings-selected,
never hard-coded (switching is a manifest + `llm_model_id` change). `ModelId`
is re-used from `common` — no new dependency edge.

**Phase 6 field — `diarization_enabled: bool` (FR-11).** Gates the post-hoc
diarization pass. `#[serde(default)]`-defaults to `false` (diarization is
post-hoc and off by default); an older store written before the field existed
deserialises to `false`. The orchestrator reads this flag to decide whether to
run the on-stop diarizer pass (and the user-triggered re-diarize), per the
diarizer design above. Added to the hand-written `Default` impl alongside the
Phase-3/Phase-5 fields. No new dependency edge.

**Phase 7 field — `onboarding_completed: bool`.** Gates the first-run
onboarding flow. `#[serde(default)]`-defaults to `false` (first run shows
onboarding); an older store deserialises to `false`. The webview gates the main
UI on it; the onboarding flow's final step sets it `true` through the existing
`update_settings` command (no dedicated `complete_onboarding` command). No new
dependency edge. (Phase 7 also adds
two app-main updater events to `common` — `AppEvent::UpdateAvailable` /
`UpdateProgress` — see `cross-cutting.md` "Auto-update".)

**Field — `gpu_acceleration: GpuAcceleration`.** The runtime GPU-acceleration
mode, now the tri-state `common::GpuAcceleration { Auto, On, Off }` (was `bool`).
`#[serde(default = ...)]`-defaults to `Auto`; an older store written before the
field existed deserialises to `Auto`, and a `deserialize_with` shim migrates a
legacy bool store (`true → Auto`, `false → Off`). Added to the hand-written
`Default` impl (`Auto`). `Auto` probes GPU VRAM at each model load and offloads a
model to the GPU only when it fits; `On`/`Off` are hard overrides that never
consult the probe. GPU offload only ever happens in a build compiled with a GPU
feature (`vulkan`/`metal`/`cuda`/`rocm`); a default CPU-only build is always on
CPU. The orchestrator reads it (`current().gpu_acceleration`) into a `GpuPlan`
via `gpu_plan()` to resolve the live + offline-re-transcribe + re-listen + prewarm
ASR `n_gpu_layers` and ASR tier, and `ipc-bridge`'s held-summariser load reads it
into a `GpuPlan` to resolve the summariser `n_gpu_layers`. No new dependency edge
(the probe + plan live in `common`). See `cross-cutting.md` — "GPU portability".

**Field — `capture_system_audio: bool`.** Whether to capture the system/call
(loopback) audio alongside the mic and MIX them into one transcribed stream, so
a Teams-style call captures all participants. `#[serde(default)]`-defaults to
`false` (opt-in and echo-safe; an older store deserialises to `false`). Added to
the hand-written `Default` impl (`false`). The orchestrator reads it
(`current().capture_system_audio`) and passes it into `AudioCaptureManager::start`,
which opens the loopback source + mixer when on (Windows-only; mic-only fallback
otherwise — see the `audio-capture` section). No new dependency edge.

**Field — `transcription_language: String`.** ASR language hint (Qwen3-ASR).
`#[serde(default = ...)]`-defaults to `"English"` (forces English, fixing the
spurious-Chinese auto-detect bug for the primary user); an older store written
before the field existed deserialises to `"English"`. Added to the hand-written
`Default` impl (`"English"`). It is a `String`, not an enum, deliberately: the
supported set (30 languages + dialects) belongs to `asr-runtime`, not the
settings schema, so a String keeps `settings` decoupled from the ASR language
table and lets `"auto"` be a reserved sentinel rather than a schema variant. The
value is NOT validated against the language table — the UI dropdown constrains it
to valid names, and an unrecognised name simply rides into the prompt prefix and
degrades gracefully (the model treats it as context); only `"auto"`/empty is
special-cased (→ no prefix). The
orchestrator reads it (`current().transcription_language`) and resolves it via
`resolve_transcription_language` to `AsrRuntimeConfig.language: Option<String>`:
the sentinel `"auto"` (case-insensitive), empty, and whitespace-only → `None`
(auto-detect, no prefix = pre-feature behaviour); any other value → the full
English name, trimmed and forwarded verbatim → prefix-force. The same resolver
feeds the live, offline-re-transcribe, and test-source start paths. No new
dependency edge. See the `asr-runtime` "Language hint" note above.

**Phase 9 fields — `chat_system_prompt: String` and the summary prompt
presets (D4).** `chat_system_prompt` `#[serde(default = ...)]`-defaults to a
meeting-notes-assistant instruction the chat engine forwards verbatim as the
session `system` message; an older store deserialises to that default. The
existing summarise feature gains selectable presets: a `SummaryPreset` enum
(`Default` | `FilterChitChat` | `ActionItems` | `Detailed`, serde snake_case,
`Default` impl = `Default`) and a `summary_preset: SummaryPreset` field
(`#[serde(default)]` → `Default`, the prior behaviour). `preset_prompt(preset)
-> &'static str` is a pure function returning the built-in prompt per preset
(`Default` is byte-identical to the prior `summary_system_prompt` default, so
existing behaviour is preserved). `Settings::effective_summary_prompt(&self) ->
String` resolves the prompt: the user's `summary_system_prompt` when it is a
non-empty custom override, else `preset_prompt(self.summary_preset)`.
`ipc-bridge`'s `summarise_meeting` reads `effective_summary_prompt()` (was
`summary_system_prompt`) so the preset picker and the custom override share one
resolution point. Both new fields are added to the hand-written `Default` impl.
No new dependency edge.

**Field — `auto_summarise_on_stop: bool` (#68).** Gates the third post-stop
background pass: when `true`, `ipc-bridge`'s `stop_recording` auto-runs
summarisation AFTER any re-transcribe / re-diarize so the summary is generated
from the final transcript (see the `ipc-bridge` "Decoupled background
post-processing" note). `#[serde(default = ...)]`-defaults to `true` — auto-summarise
is ON by default; an older store written before the field existed deserialises to
`true`, so existing users adopt the new behaviour. Added to the hand-written
`Default` impl (`true`). `ipc-bridge` reads it (`current().auto_summarise_on_stop`)
as the third gate of `post_stop_passes`. No new dependency edge. See
`cross-cutting.md` — "Finalise returns to the meeting list".

**Field — `preload_summariser: bool`.** Whether the shared summary/chat LLM is
warmed at app startup (and kept resident — the held `OnceCell` never unloads).
`#[serde(default = ...)]`-defaults to `true`; an older store deserialises to
`true`. `app-main` reads it via `ChatHandles::maybe_preload_summariser` on a
background startup task (mirroring `prewarm_asr`): when `true` AND the LLM is
already downloaded (checked via `Orchestrator::list_models`, no download), it
calls `ensure_summariser` so the first Summarise / chat is instant; when `false`
the model loads on-demand on first use. It NEVER downloads at startup. See
`cross-cutting.md` — "ASR prewarm".

**Field — `output_language: String`.** Language for all LLM-generated text
(summaries and chat replies). Does NOT affect transcription — the transcript is
always left as-is. The sentinel `"auto"` instructs `ipc-bridge` to resolve the
output language from the host system locale at generation time (via
`sys-locale`); a full English language name (e.g. `"French"`, `"German"`)
passes through verbatim. `#[serde(default = ...)]`-defaults to `"auto"`; an
older store written before the field existed deserialises to `"auto"`. The
resolved language name is appended to the summariser and chat system prompts by
`ipc-bridge` — the transcript itself is never touched. No new dependency edge
on the `settings` crate. See the `ipc-bridge` "Output-language resolution" note.

### `ipc-bridge`
**Crate:** `crates/ipc-bridge`
**Owns:** the Tauri command + event surface. tauri-specta generates
TypeScript types consumed by the webview.

**The only crate that knows about Tauri APIs.** Every other crate is
free of Tauri imports — this is what makes the core testable without a
running Tauri app.

**Phase 1 command surface (8 commands, all `async fn` returning `Result<T, IpcError>`):**
`list_devices`, `start_recording`, `pause_recording`, `resume_recording`,
`stop_recording`, `get_recording_state`, `get_settings`, `update_settings`.

**Phase 2 additions (10 commands total):** `list_models` (`Vec<ModelStatus>`),
`ensure_model` (`()`). Both route through `Orchestrator` — no direct
`model-registry` dependency from `ipc-bridge`.

**Phase 3 additions (12 commands total):** `save_notes`
(`(meeting_id, notes_json, notes_markdown) -> ()`) and `load_notes`
(`(meeting_id) -> Option<NotesDocument>`, `None` when no notes saved). Unlike the
model/recording commands, these route **directly** to `persistence::NotesStore`
— `persistence` is now a real `ipc-bridge` dependency (already granted in the
table above) and the orchestrator is *not* involved: notes I/O is independent of
the live recording pipeline and may run concurrently with an active recording
(see `persistence` "Phase 3 surface growth — notes"). The blocking filesystem
write/read runs on `spawn_blocking`. `IpcState` carries a `meetings_dir:
PathBuf` (a clone of the same `{app-data}/meetings/` root the
orchestrator/persistence use), resolved and injected by `app-main`. The opaque
Tiptap document crosses the wire as a `String` (`common::NotesDocument {
notes_json: String, notes_markdown: String }` — `ipc-bridge` returns the common
type directly rather than a local mirror) because a bare `serde_json::Value`
does not derive `specta::Type`; `save_notes` parses the string to a
`serde_json::Value` before handing it to `NotesStore` and `load_notes`
re-serialises the loaded value back to a string.

**Note image command — `save_note_image` (29 commands total).**
`save_note_image(meeting_id, bytes: Vec<u8>, ext: String) -> String` persists a
pasted/dropped note image and returns the **portable** filename ref the
frontend stores into `notes.json`. Like `save_notes`, it routes **directly** to
`persistence::save_note_asset` (no orchestrator) on `spawn_blocking`; `ext` is
validated against an image allowlist (`png` / `jpg` / `jpeg` / `gif` / `webp`)
and rejected as `AppError::InvalidInput` otherwise. `ipc-bridge` also owns the
**`meetingasset:` asset resolver** (`resolve_note_asset(meetings_dir,
request_path) -> ResolvedNoteAsset`, plus `MEETING_ASSET_SCHEME`): it parses an
asset request path `/<meeting_id>/<filename>` into a `Uuid` + filename and
resolves bytes via `persistence::read_note_asset` (whose path-traversal guard it
relies on). This lives in `ipc-bridge` — not `app-main` — so the `persistence`
edge stays inside `ipc-bridge` (`app-main` does not depend on `persistence`).
See `cross-cutting.md` — "Note image assets".

**Phase 4 — `stop_recording` index upsert (FR-33, in-session visibility).**
`Orchestrator::stop` finalises the meeting folder but deliberately never touches
the `MeetingIndex` (the orchestrator does not own one). To make a just-recorded
meeting appear in `list_meetings` **within the same session** — rather than only
after the next startup `rebuild_from_disk` — the `stop_recording` command, after
`orchestrator.stop()` returns the `MeetingMeta`, builds a `MeetingListEntry` from
that meta (id / title / started_at / duration_ms / speaker_count; `excerpt` from
the first transcript segment via `persistence::read_transcript`, else `None`) and
`upsert`s it into the shared `IpcState::index`. The blocking transcript read runs
on `spawn_blocking`; the async `upsert` is awaited (never `block_on`'d). An
upsert failure is logged and swallowed — the recording is safely on disk and the
index is a derived cache the next startup reconciles, so a failed upsert must not
turn a successful stop into an error. This keeps the orchestrator decoupled from
the index: the index handle lives in `ipc-bridge` (`IpcState`), so the upsert
lives at the command boundary, not in the orchestrator.

**Decoupled background post-processing + self-healing list (drift + truncation
fix).** After the upsert, `stop_recording` runs up to three heavy passes OFF the
stop path, in order, in one fire-and-forget `tokio::spawn` (cloned
`Arc<Orchestrator>` + `Arc<MeetingIndex>` + a `ChatHandles` for the held
summariser), so none can wedge the stop response or hide the meeting: (1) if
`orchestrator.take_transcript_incomplete()`
is true — the live ASR dropped audio (drop-oldest flush queue) or its stop-drain
timed out — `re_transcribe` re-runs ASR over the complete `audio.opus`, the
authoritative transcript (the audio is captured in full regardless of live-ASR
speed); (2) if `orchestrator.diarization_enabled()`, `rediarize` runs AFTER any
re-transcribe so it labels the repaired transcript — each with its own
length-relative timeout (`retranscribe_timeout` / `diarize_timeout`); (3) if
`settings.auto_summarise_on_stop` (default `true`, #68), `run_held_summarise`
auto-summarises the meeting AFTER any re-transcribe / re-diarize so the summary is
generated from the FINAL transcript, emitting the determinate
`OperationProgress { op: Summarise }` + `SummaryReady` exactly as the
user-triggered `summarise_meeting` does (the summarise body is shared — both call
`run_held_summarise`, which resolves the held `LlamaSummariser`, runs the heavy
`summarise_with_progress` on `spawn_blocking`, refreshes the index excerpt, and
emits `SummaryReady`). Passes (1)/(2) refresh the index row and emit their events
on completion (`TranscriptReady` / `DiarizationComplete`); all passes are
best-effort (errors logged, claim-skips logged at info — auto-summarise leaves the
meeting without a summary on failure, recoverable via the Summarise action). While
a re-transcribe / re-diarize pass holds the offline claim the recorder reports the
public **`Idle`** state (`Offline → Idle` in `as_public`), so the transport leaves
Start ENABLED: a `start` here PREEMPTS the pass (`transition_start` accepts `Idle |
Offline`) rather than being refused, because the next meeting is a different
`transcript.json` and the user must never be blocked from recording it. The
preempted pass finishes on its thread (writing the old meeting's files) and its
release is a no-op (preemption-safe `transition_offline_release`); the remaining
chain passes self-skip — re-transcribe/re-diarize because a fresh claim now fails
against `Recording`, auto-summarise (which takes no claim) because it checks
`recorder_is_live()`. And because a derived cache can always drift from
disk, `list_meetings` first calls
`MeetingIndex::reconcile_orphans(meetings_dir)` — a cheap `readdir` + set-diff
that lazily indexes any meeting folder present on disk but missing from the cache
(e.g. the process killed between finalise and the stop-time upsert) — so a
meeting can never stay hidden within a session, even without a restart. Reconcile
is best-effort (a failure logs and serves the cache as-is) and never deletes
(removals are reconciled by the next startup `rebuild_from_disk`).

The pass selection (gating + ordering — re-transcribe before diarize before
auto-summarise) is a pure
`post_stop_passes(needs_retranscribe, needs_diarize, needs_summarise) ->
Vec<PostStopPass>`, and the execution (each pass tolerant of its own error —
`InvalidInput`/busy logged at info, anything else at warn — never aborting the
remaining passes) is `run_post_stop_passes`, which takes the per-pass call as a
closure. Both are extracted from the `#[tauri::command]` body so the orchestration
is unit-tested without a Tauri runtime or a real orchestrator (a recording stub
injects per-pass results; the auto-summarise pass is exercised via a model-free
`StubSummariser` that writes `summary.md` + emits `SummaryReady`).

**Responsive stop — `Finalising` state + `MeetingFinalised` event.** The
in-session drain/finalise (transcribing the live backlog, writing the meeting
files) runs on the runner's own thread, but `stop()` used to keep the UI in
`Stopping` for its whole duration (up to the 30 s drain). `stop()` now, after
dispatching the stop command, broadcasts `RecordingState::Finalising` and flips
to `Idle` only once the runner replies — so the webview stays responsive during
the drain; the record controls treat `finalising` as busy, gating only a NEW
recording, which the state machine enforces (`Recording|Paused → Stopping →
Finalising → Idle`, via `transition_finalising`). On completion `stop()` emits
`Idle` plus `AppEvent::MeetingFinalised { meeting_id }`; the webview's meetings
store refreshes on that event so the just-finalised meeting appears (through
`reconcile_orphans`/`upsert`) with no manual refresh. `RecordingState` gains a
`Finalising` variant and `AppEvent` a `MeetingFinalised` variant — bindings
regenerated.

**Phase 4 additions (18 commands at Phase 4; `re_summarise` removed in Phase 5)
— meeting list / open / actions.** Six commands back the meeting-list view
(FR-33):

- `list_meetings() -> Vec<MeetingListEntry>` — self-heals via
  `MeetingIndex::reconcile_orphans(meetings_dir)` (best-effort), then queries the
  shared libsql index (`MeetingIndex::list_meetings`, most-recent first).
- `open_meeting(meeting_id) -> MeetingState` — assembles the restore payload via
  `persistence::read_meeting_state` (blocking folder reads on `spawn_blocking`);
  the index is **not** consulted (the folder is authoritative for full state).
- `rename_meeting(meeting_id, title) -> ()` and
  `delete_meeting(meeting_id) -> ()` — route to
  `persistence::meeting_ops::{rename_meeting, delete_meeting}`, which keep the
  on-disk folder and the index row consistent.
- `set_speaker_name(meeting_id, label, name) -> speaker_names map` — routes to
  `persistence::meeting_ops::set_speaker_name`; maps a diarizer label to a
  display name in `metadata.json` (empty `name` clears it), returning the
  updated map so the webview re-renders the transcript overlay without a
  reload. The same write is also reachable as the `set_speaker_name` agent
  tool; this is its direct UI path. Label + name capped at 512 chars.
- `re_transcribe(meeting_id) -> ()` — the **only** Phase-4 read/action command
  that routes through the orchestrator (`Orchestrator::re_transcribe`): an
  offline re-run of the live ASR pipeline (see `orchestrator` below). The shared
  `IpcState::index` handle is passed into the call so the orchestrator refreshes
  the index row without owning an index of its own.
- `re_summarise(meeting_id) -> ()` — **a Phase-4 stub, removed in Phase 5.**
  It returned `AppError::Unsupported` as a placeholder until the `summariser`
  landed. Phase 5 replaced it with `summarise_meeting` (below); the meeting-list
  row's Summarise action repoints to that command, so the stub had no caller and
  was deleted.

`IpcState` gains `index_db_path: PathBuf` (resolved by `app-main` via
`persistence::index::index_db_path`) and `index: Arc<MeetingIndex>` — a single
libsql connection opened **once** at startup. libsql's index methods are
`async fn`; the command handlers `await` them and never `block_on` (the
no-`block_on`-in-command-handlers rule). The index is opened (and rebuilt from
disk) at startup by the `ipc_bridge::open_meeting_index` helper, which `app-main`
calls — keeping the `persistence` edge inside `ipc-bridge` so `app-main` does not
acquire a direct `persistence` dependency. That helper drives libsql's async
`open` + `rebuild_from_disk` on a one-shot `block_on` (startup-only; the
no-`block_on` rule binds command handlers, not bootstrap). `MeetingListEntry` /
`MeetingState` are the canonical `common` types (Phase-4 precursors), so the
generated bindings consume them directly with no mirror.

**Phase 5 additions (20 commands total) — summary surface + the `summariser`
edge (FR-30).** The Phase-4 `re_summarise` stub is **removed** and three real
commands land, realising the granted `ipc-bridge → summariser` dependency edge
(`summariser = { path = "../summariser" }` in `ipc-bridge`'s Cargo.toml — already
in the dependency table above):

- `summarise_meeting(meeting_id) -> ()` — resolves the LLM model id via the
  `resolve_llm_model_id(&Settings) -> ModelId` seam (`settings.llm_model_id`,
  else the bundled default `gemma-4-e4b-it-q4_k_m`, exposed as the `pub const
  commands::DEFAULT_LLM_MODEL_ID`),
  resolves the model **directory** via `Orchestrator::ensure_model_path` (so the
  `model-registry` edge stays in the orchestrator — there is **no**
  `orchestrator → summariser` edge), locates the single `.gguf` in that dir
  (skipping any `mmproj-*`), opens a `summariser::LlamaSummariser`, reads the
  transcript (`persistence::read_transcript`) + the notes markdown
  (`read_meeting_state(..).notes`, empty when absent), runs `summarise`, and
  writes `summary.md` (`persistence::write_summary`) — the summariser `open` +
  `summarise` and the folder I/O run on `spawn_blocking` (the threading-model
  rule: inference off the command handler). It then emits
  `AppEvent::SummaryReady { meeting_id }` on the shared `event_tx`.
- `get_summary(meeting_id) -> Option<String>` — reads `summary.md` via
  `persistence::read_summary` (blocking read on `spawn_blocking`); `None` when
  no summary exists.
- `save_summary(meeting_id, summary_markdown) -> ()` — persists an edited
  summary via `persistence::write_summary` (`spawn_blocking`).

`IpcState` gains `event_tx: broadcast::Sender<AppEvent>` — a clone of the
**same** bus `app-main` constructs once and shares with the `ModelRegistry` and
the `Orchestrator` (via `with_event_tx`). Emitting `SummaryReady` here is the
only place `ipc-bridge` produces an event directly; the event forwarder's single
subscription (via `Orchestrator::subscribe_events`) sees it because the channel
is shared. The summary crosses the wire as an opaque markdown `String`;
`summarise_meeting` reuses `AppEvent::SummaryReady` (no new event). A
`summarise_meeting_inner(&dyn Summariser, …)` seam lets the default test suite
exercise the read → summarise → write → event wiring with a `StubSummariser`,
without a model or Tauri runtime (mirroring the orchestrator's re_transcribe
stub-backend seam). The `resolve_llm_model_id` seam is covered by unit tests for
both branches (settings override, default fallback). A manifest-consistency
guard test (`tests/default_model_manifest.rs`) parses `resources/models.json`
and asserts `DEFAULT_LLM_MODEL_ID` stays a `kind = Llm` entry, so a manifest
rename fails a test rather than silently breaking the default summarise path.
That test uses `model-registry` as a **dev-dependency** only (it lives in
`tests/`, touches no `src`): `ipc-bridge` still resolves models exclusively
through `Orchestrator` at runtime, so there is no production `model-registry`
edge in the dependency table above (mirroring `orchestrator`'s test-only
dev-dependencies).

**Phase 6 addition (21 commands total) — re-diarize (FR-11).** One command lands:

- `rediarize_meeting(meeting_id) -> ()` — routes to `Orchestrator::rediarize`
  (the offline re-diarize): decode → `SherpaDiarizer::assign_speakers` →
  `transcript.json` rewrite with `speaker_id`s → `metadata.json` `speaker_count`
  update → index-row refresh → `AppEvent::DiarizationComplete`. The shared
  `IpcState::index` handle is passed into the call so the orchestrator refreshes
  the index row without owning one. The diarizer is built **inside the
  orchestrator** (which holds the granted `orchestrator → diarizer` edge and
  resolves the diarize models via `model-registry`), so there is **no**
  `ipc-bridge → diarizer` Cargo edge — `ipc-bridge` routes via the orchestrator,
  mirroring how the ASR/summariser model-registry edges stay out of `ipc-bridge`.
  `AppEvent::DiarizationComplete` is emitted by the **orchestrator**, not here.

**Phase 9 additions (25 commands total) — the chat agent + the held model.**
Four commands land, realising the granted `ipc-bridge → agent-tools` +
`ipc-bridge → chat-agent` edges, plus the held-model refactor (C2):

- `send_chat_message(meeting_id: Option<MeetingId>, session_id:
  Option<ChatSessionId>, message) -> ChatSessionId` — creates or loads the chat
  `common::ChatSession` (via `persistence::ChatStore`), appends the user message,
  and **spawns the turn on a background `tokio::spawn`**, returning the session id
  immediately. The turn itself runs on `spawn_blocking` (the LLM is FFI-bound);
  tool dispatch re-enters async via a captured `Handle::block_on(registry.
  dispatch(...))` for the dispatch step only (the one async/sync crossing). The
  reply streams to the webview as the chat `AppEvent`s; the updated session is
  persisted by `ChatStore` at turn end. A second send for a session whose turn is
  still running is rejected `InvalidInput { "session busy" }` (single in-flight
  turn per session, tracked in `IpcState::chat_in_flight`).
- `cancel_chat_turn(session_id: ChatSessionId) -> ()` — raises the per-session
  `chat_agent::CancelFlag` registered by `send_chat_message` (held in
  `IpcState::chat_cancel: Arc<Mutex<HashMap<ChatSessionId, CancelFlag>>>`); the
  engine's decode loop checks it between tokens and stops, and the driver ends
  the turn with a terminal `ChatTurnComplete` carrying the partial reply (NOT a
  `ChatError` — cancellation is a user action). Idempotent: a session with no
  running turn is a no-op success (P1).
- `get_chat_session(meeting_id, session_id) -> Option<ChatSession>`,
  `list_chat_sessions(meeting_id) -> Vec<ChatSession>`,
  `delete_chat_session(meeting_id, session_id) -> ()` — route directly to
  `persistence::ChatStore::{load, list, delete}` on `spawn_blocking`.

The **driver loop** is a State-free generic helper (`crate::chat::run_chat_turn`,
generic over `ChatEngine` + a tool-dispatch closure + an emit closure) so the
default test suite drives a full turn — final-only, tool-call-then-final, the
max-iteration cap, and the hard-floor context overflow — with a STUB engine and
STUB tools, no model and no Tauri runtime. It applies `chat_agent::trim_to_budget`
before each engine call (hard-floor → `InvalidInput`; on eviction it snaps the
drop count forward to a user-message group boundary and emits
`ChatContextTrimmed`, CQ2/P2), runs the tool loop with a `MAX_TOOL_ITERATIONS`
cap (the escape offers no tools to force a final answer; exhaustion emits
`ChatError`), appends the **assistant-`tool_calls` message before** the per-call
tool results so the engine renders a valid OpenAI `assistant(tool_calls) →
tool(result)` sequence (CQ1), threads the per-turn `CancelFlag` into each engine
call (a `TurnOutcome::Cancelled` ends the turn with a terminal `ChatTurnComplete`
carrying the partial text, P1), and **injects a per-turn non-zero seed** before
each non-greedy `run_turn` (`chat_agent::SamplerConfig`'s default `seed = 0` is a
fixed/reproducible trap — every non-greedy reply would be verbatim-identical
without this).

**Held model (C2).** `IpcState` gains `summariser: Arc<OnceCell<Arc<LlamaSummariser>>>`
— the LLM GGUF is loaded **once** on first chat/summarise use (via
`IpcState::ensure_summariser`, which resolves the model id + directory through
`Orchestrator::ensure_model_path` and opens the GGUF on `spawn_blocking` with the
GPU-offload count resolved at load time from the VRAM-aware `GpuPlan`
(`plan.summariser_gpu`; see `cross-cutting.md` — "GPU portability") computed from
the `gpu_acceleration` setting) and shared thereafter. `summarise_meeting` was **refactored** to reuse this held
handle instead of constructing a fresh `LlamaSummariser` per call. The chat engine
borrows `&LlamaModel` from the held handle via `LlamaSummariser::model()`; the
`agent-tools` `ToolContext`'s `resummarise` coerces the same handle to
`Arc<dyn Summariser>`. `IpcState` also gains `tool_registry: Arc<ToolRegistry>`
(built once as `ToolRegistry::v1(false)` — the Phase-10 inter-agent bridge tool is
omitted), `chat_in_flight: Arc<Mutex<HashSet<ChatSessionId>>>`, and
`chat_cancel: Arc<Mutex<HashMap<ChatSessionId, chat_agent::CancelFlag>>>` (the
per-session cancel flags `cancel_chat_turn` raises, P1).

The command ledger is now **33** (P6 21 + the four P9 chat commands = 25; P10's
`get_mcp_server_info` = 26; the P9 chat review-fix's `cancel_chat_turn` = 27;
`prewarm_asr` = 28; `save_note_image` = 29; `set_speaker_name` = 30;
`translate_meeting` + `get_translations` = 32; `get_diagnostic_report` = 33),
asserted by the `bindings_builder_registers_expected_command_ledger` test.

**Diagnostic report (`get_diagnostic_report`, issue #0014).** Assembles + REDACTS
the `common::DiagnosticReport` the no-telemetry "Report a problem" flow pre-fills
into a GitHub issue (the webview maps the snake_case binding onto its camelCase
`issueReport.ts` shape and opens the user's browser; nothing is sent
automatically). Log-excerpt / backtrace redaction is owned HERE (`diagnostics`
module), where the data is read: it reads `{logs}/last-crash.txt` when present
(supplying the backtrace + recent-lines excerpt, error class `"panic"`) else the
tail of the rolling `minutist.log*` file (error class `"diagnostic report"`, no
backtrace), and strips meeting-id UUIDs from every text field via a local
`redact` (mirroring `app-main`'s `crash::redact` and the webview's
`redactMeetingPaths` — each crate owns its copy; `ipc-bridge` cannot import
`app-main`). `IpcState` gains `logs_dir` (read-only; `app-main` owns writes),
`app_version`, and `platform` (`"{os} / {arch} / {build}"`, constructed by
`app-main` which owns the `connected` feature), all set by `app-main`. The
`probe_primary_gpu` call (it can block) runs on `spawn_blocking`. No new
dependency edge — `common::probe_primary_gpu` / `resolve_gpu_plan` are already
reached by `log_gpu_probe`.

**Event forwarding:** `spawn_event_forwarder` starts a tokio task that subscribes
to the orchestrator broadcast and emits `AppEventPayload` (event name
`"app-event-payload"`) to all windows.

**tauri-specta pin verified (Q-P1-2):** `tauri-specta = "=2.0.0-rc.21"`,
`specta = "=2.0.0-rc.22"`, `specta-typescript = "0.0.9"` compile cleanly with
`tauri = "2.10"`. No version conflict.

**Specta types (post-P0a):** `common` and `settings` derive `specta::Type`
directly behind their optional `specta` feature, which `ipc-bridge` enables.
The Phase 1 mirror layer (`specta_types.rs`) was deleted; commands and events
use the canonical types. `IpcError` remains a local `specta::Type` mirror of
`AppError` at the boundary (harmless; may be removed in a later cleanup).

**Output-language resolution (`sys-locale` external dependency).** `ipc-bridge`
adds `sys-locale = "0.3"` as a direct external dependency (not a workspace
component edge — it is a third-party crate, so the dependency table above is
unchanged). The `output_language` module exposes `resolve_output_language(setting:
&str) -> Option<String>`: the sentinel `"auto"` calls `sys_locale::get_locale()`,
extracts the primary BCP-47 language subtag, and maps it through a static
subtag→full-name table covering the 15 major languages (en, zh, es, fr, de, it,
pt, ja, ko, ru, nl, ar, hi, pl, tr). An explicit language name passes through
verbatim. Returns `None` for `"auto"` resolving to an unmapped subtag, for an
empty setting, and for the empty string. The resolved name is appended to the
summariser and chat system prompts as `"\n\nRespond entirely in {lang}."` (see
"Summariser and chat injection" in `cross-cutting.md`).

**Translation commands (32 commands total) — translated transcript as derived view.**
Two commands land, using the existing `ipc-bridge → summariser` +
`ipc-bridge → persistence` edges (no new dependency table edges):

- `translate_meeting(meeting_id, target_language) -> ()` — validates
  `target_language` against the 15-language `SUPPORTED_TRANSLATION_LANGUAGES`
  constant (the same set as `output_language::SUBTAG_TO_LANGUAGE` values).
  Rejects a second concurrent call for the same `(meeting_id, target_language)`
  pair via `IpcState::translate_in_flight: Arc<Mutex<HashSet<(MeetingId,
  String)>>>` (mirrors `chat_in_flight`). Emits an indeterminate
  `OperationProgress { op: Translate }` while loading the held summariser, then
  runs the per-segment loop on `spawn_blocking`: for each segment, calls
  `LlamaSummariser::translate_segment(text, target_language)`, accumulates
  the result in a pending batch, and flushes to `translations.json` via
  `persistence::merge_translations` on the same ~200 ms throttle cadence as
  the progress emit (plus unconditionally on loop exit) so partial progress
  survives interruption without O(n²) sidecar rewrites. Emits a determinate
  `OperationProgress` fraction throttled to ~5 Hz. Emits
  `AppEvent::TranslationReady { meeting_id, language }` on every exit path
  (success AND error) so the operation-progress indicator is always cleared.
- `get_translations(meeting_id, target_language) -> HashMap<usize, String>` —
  reads `translations.json` via `persistence::read_translations` on
  `spawn_blocking`, returns the per-language segment map (empty map when no
  translations exist yet). The webview calls this on meeting open and on
  `TranslationReady`.

`AppEvent` gains `TranslationReady { meeting_id: MeetingId, language: String }` in
`common`. `OperationKind` gains `Translate`. Both variants require a `specta::Type`
derivation and are surfaced in the generated TypeScript bindings. The webview's
`operation-progress` store terminal-event handler must clear on `TranslationReady`
(mirrors the existing clears for `SummaryReady` / `DiarizationComplete`).

### `app-main` (bin)
**Crate:** `src-tauri/` (Tauri convention)
**Owns:** the Tauri main binary, tray icon, window management, process
lifetime. Wires the components into a running app.

The thinnest crate — code here should mostly be construction and
plumbing.

**Tracing:** file appender at `{app-data}/logs/minutist.log`, rotated
daily, 7-day retention via startup cleanup. Console layer in debug builds
only. `RUST_LOG`-style filtering via `EnvFilter::from_default_env()`.

**Crash capture (issue #0014).** `src-tauri/src/crash.rs` adds a `tracing`
ring-buffer layer (last 200 log lines in a process-wide static) to the
subscriber and installs a `std::panic::set_hook` that writes a REDACTED
`last-crash.txt` to the logs dir on a panic (version, platform, configured GPU
mode, panic message + location, backtrace, recent ring lines). Every line is
passed through `crash::redact` (meeting-id-UUID strip). See `cross-cutting.md` —
"Logging". `IpcState` is populated with `logs_dir` / `app_version` / `platform`
so `ipc-bridge::get_diagnostic_report` can read the crash file + log tail.

**Browser-open plugin (`tauri-plugin-opener`, #0014).** Registered on the Tauri
builder so the webview's "Report a problem" flow can open the user's default
browser at the pre-filled GitHub issue URL (`opener:allow-open-url` granted in
`capabilities/default.json`). It is an external Tauri plugin (like
`tauri-plugin-fs` / `-store` / `-updater`), not a workspace crate, so it adds no
row to the workspace dependency table. Not an app network operation — the OS
browser makes any request, at the user's click; the D4 no-telemetry claim is
untouched.

**Tray menu:** "Open minutist" (show/focus main window) + "Quit"
(`app.exit(0)`). Left-click on the tray icon shows the main window.
Window close intercepts `CloseRequested` and hides rather than exits.

**Bindings harness:** `cargo run -p minutist --bin generate-bindings`
(alias: `cargo gen-bindings`) writes `ui/src/ipc/bindings.ts` without
starting the GUI. Run after any `ipc-bridge` command/event surface change.

**Phase 9 wiring.** `app-main` builds the chat `ToolRegistry::v1(false)` once and
constructs `IpcState` with it plus the lazily-initialised held-model cell
(`Arc<OnceCell<Arc<LlamaSummariser>>>`, loaded on first chat/summarise use) and the
`chat_in_flight` guard. The held model is owned by `IpcState`; `app-main` does not
load the GGUF at startup. This adds the `agent-tools` (the registry is built here)
+ `chat-agent` (transitively via `ipc-bridge`) dependency rows above.

**`settings.data_directory` path resolution.** After loading settings,
`app-main` calls the pure `resolve_data_roots(platform_root,
settings.data_directory)` helper (unit-tested, in `src-tauri/src/main.rs`) to
derive three path roots: `meetings/`, `models/`, and the `index.db` parent.
When `data_directory` is `None`, all three are under `app_data_dir` (the
platform default — unchanged behaviour). When it is `Some(path)`, the three
roots are placed under `path` instead, which must be an absolute path that can
be created; a relative or uncreatable path falls back to the platform default
with a `tracing::error` and never aborts startup. Two roots are excluded from
the override by bootstrap constraints: `settings.store` (the file that carries
the override) and `logs/` (logging starts before settings load); both always
sit at the platform default root. Data roots are fixed for the lifetime of the
process — changing the setting requires a restart, and existing data is not
migrated automatically. There is currently no UI for this field; it must be set
by editing `settings.store` directly.

**Phase 10 wiring (MCP).** Gated on `settings.mcp_enabled` (off by default).
The shared start logic lives in `do_start_mcp_server` (private, `async` fn in
`app-main`): it first calls `ensure_summariser` (failing early with
`McpServerStartFailed { reason }` if the model load fails), creates a fresh
`watch::Sender<bool>` shutdown pair, spawns the inter-agent driver via
`ipc_bridge::spawn_inter_agent_driver` (passing a `shutdown_rx` clone so the
driver exits deterministically when the server is disabled), builds the MCP
`ToolRegistry::v1(true)` + a `ToolContext` carrying the bridge SENDER, resolves
the bearer token (generate-on-first-enable, persisted to `{app-data}/mcp_token`
with `0600`; OS-keychain hardening is a documented follow-up), and `await`s
`mcp_server::serve` on `127.0.0.1:{mcp_port}`. On success, `serve` returns
`(SocketAddr, oneshot::Receiver<()>)` — the completion receiver resolves when
the accept loop exits and the listener is dropped. The shutdown sender AND
completion receiver are stored together in `McpShutdownState` (Tauri managed
state, an `Arc<McpShutdownState>` the watcher also holds), and `IpcState.mcp_info`
(URL + token, read by `get_mcp_server_info`) is filled; `AppEvent::McpServerListening`
is emitted. On any failure, `McpServerStartFailed { reason }` is emitted and the
handles slot is left `None`.

A settings-watcher task (spawned at startup) subscribes to
`SettingsHandle::subscribe()` and reacts to `mcp_enabled` transitions: on
`false→true`, it calls `do_start_mcp_server` directly (not spawned — the
watcher is itself `async`, so start runs inline and serialises with any
concurrent stop); on `true→false`, it takes the stored handles, fires the
shutdown watch, and **awaits the completion receiver** (bounded at 5 s, logging
a warning on timeout) before clearing `IpcState.mcp_info` and emitting
`AppEvent::McpServerStopped`. Achieved state is tracked via the presence of
the handles slot (`Some` = running, `None` = not running), not from the desired
`mcp_enabled` value — a failed start leaves the slot `None` so a subsequent
off→on toggle retries the start. Enable/disable takes effect immediately with
no restart. Port and `mcp_write_tools` changes are NOT reacted to by the
watcher — those are restart-required (the running server was built with those
values at start time).

`ipc_bridge::spawn_inter_agent_driver` now accepts a `watch::Receiver<bool>`
shutdown signal alongside the existing channel/handles parameters (cross-crate
signature change: `ipc-bridge` → `app-main`). The driver's select loop exits on
either the shutdown signal or all-senders-dropped, whichever fires first.

`common::AppEvent` gains `McpServerStartFailed { reason: String }` — the UI
handles it in `useMcpServerInfoStore` (drops the "starting…" hint, shows the
reason) and in `McpSettingsPane` (renders a `--warn` hint with the reason and
retry guidance). Adds the `mcp-server` dependency row above.

## Webview components

The webview is small enough that ownership maps to directories rather
than packages.

| Component | Lives in | Owns |
|---|---|---|
| Notes editor | `ui/src/editor/` | Tiptap editor, markdown shortcuts, paragraph-anchor extension. |
| Transcript pane | `ui/src/transcript/` | Live-appending transcript view, hover/click cross-reference. Speaker chips carry a live colour dot when diarization labels are present (`speaker-color.ts`: deterministic `speaker_id` → palette slot; colour pairs with the visible label for accessibility). Consecutive rows are grouped: the labelled chip shows once at the start of a speaker's run; continuation rows keep only the colour dot. |
| Meeting shell | `ui/src/shell/` | Window chrome (start/stop/pause, audio meter, meeting list); the pane-visibility toggle; and the Settings drawer (`SettingsDrawer.tsx` — an Appearance group with the colour-theme control + the notes writing-paper-rules toggle, plus input device, transcription language, diarize-on-stop, GPU acceleration, system-audio capture, and a Connections (MCP) pane: `McpSettingsPane.tsx` — enable toggle, fixed port, write-tools toggle, and the live endpoint URL + bearer-token reveal/copy via `get_mcp_server_info`). The summary is a workspace column, not an overlay. The capture/processing/appearance settings live in the drawer rather than the top bar so the masthead stays a single non-overflowing row. The settings controls route through the existing settings seams; the MCP pane adds the one Phase-10 read command `get_mcp_server_info`. |
| IPC client | `ui/src/ipc/` | Typed wrapper around `invoke` + `listen`. Generated stubs from tauri-specta live here. |
| UI state store | `ui/src/state/` | Zustand store. Derived UI state only — transient. Also holds a `settings` snapshot loaded once via `refreshSettings` on mount; user-driven changes (e.g. device selection) round-trip through `commands.updateSettings` so they persist across app restarts. |

The webview's source of truth for typed messages is the generated
`bindings.ts` produced by tauri-specta. Hand-edits to that file are not
allowed.

**Phase 1 implementation notes (Stream G).**
The Zustand store shape is defined in `ui/src/state/recording.ts` as
`RecordingStore` with fields `state`, `devices`, `selectedDeviceId`,
`meter`, and `lastError` plus async action methods and a synchronous
`handleEvent` dispatcher. The global `"app-event-payload"` event listener
is mounted once in `App.tsx` via the `useAppEventBridge` hook
(`ui/src/shell/event-listener.tsx`); it must not be placed inside a
conditionally-rendered subtree. The Vite dev server runs on port 5173
(matching `tauri.conf.json` `devUrl`).

**Phase 2 additions (Stream F).** `RecordingStore` gains `transcript: Segment[]`
(cleared on `state_changed → recording`; appended by `transcript_segment` events).
`ModelsStore` (`ui/src/state/models.ts`) tracks `ModelStatus[]`, `isAsrModelReady`
(derived), and `downloadInProgress` progress map; its `handleEvent` is dispatched
alongside `RecordingStore.handleEvent` from `useAppEventBridge`. The `Start` button
in `MeetingControls` is disabled when `isAsrModelReady` is false; `ModelDownloadStatus`
(`ui/src/shell/`) provides the first-run download flow. `TranscriptPane`
(`ui/src/transcript/`) renders live segments with `MM:SS.cc` timestamps and
sticky-bottom auto-scroll. `MainWindow` uses a two-column 50/50 layout (controls
left, transcript right).

**Phase 3 additions (Stream S2 — notes editor).**

- **Notes editor (`ui/src/editor/`).** A Tiptap v3 WYSIWYG editor is the primary
  view (`Editor.tsx`). It composes `StarterKit` (with `link: false`) +
  `@tiptap/extension-link` + `@tiptap/extension-typography` + the
  `@tiptap/extension-table` family + `tiptap-markdown` via
  `extensions.ts::buildEditorExtensions`. Markdown-shortcut input rules
  (StarterKit + Typography) transform while typing (FR-15/16/20).
- **Paragraph-anchor extension (`ui/src/editor/paragraph-anchor.ts`).** A custom
  Tiptap/ProseMirror extension that registers a nullable `data-anchor-ms`
  attribute on the paragraph node and stamps it on the FIRST keystroke into a
  paragraph, ONLY while `recordingState.kind === "recording"`, from the store's
  `recordingClockMs` (the pause-**excluding** capture clock fed by
  `AppEvent::RecordingClock`) — never `Date.now() - started_at_ms` (FR-19,
  binding correction A4; see `cross-cutting.md` "Notes paragraph-anchor clock").
  Already-anchored paragraphs are never re-stamped; split-created paragraphs
  reset their inherited anchor so the next keystroke stamps fresh. The clock is
  injected as an `AnchorClockSource`, decoupling the extension from the store.
- **Autosave (`ui/src/editor/useAutosave.ts`).** Interval autosave
  (`autosave_interval_secs`, default 5 s) plus flush-on-blur, persisting notes
  through the `save_notes` IPC seam (`ui/src/ipc/notes.ts`). The target meeting
  is `activeMeetingId(state) ?? openMeetingId` — the active recording while
  capturing, otherwise the open saved meeting being viewed (the same document
  identity rule `active-transcript` uses), so edits to a finished/opened meeting
  persist. No-op only when neither exists: the live entry surface with nothing
  open (FR-18).
- **HTML clipboard (`ui/src/editor/clipboard.ts`).** `buildClipboardPayload`
  produces a `text/html` (+ `text/plain`) copy payload — a self-contained UTF-8
  document with internal `data-anchor-ms` attributes stripped — so paste into
  Word retains formatting (FR-17). The editor overrides copy/cut via ProseMirror
  `editorProps.handleDOMEvents`.
- **Issue-report builder (`ui/src/diagnostics/issueReport.ts`).** Pure builder
  for the "Report a problem" flow (#0014, no-telemetry decision O1/U6): given a
  redacted `DiagnosticReport` (version / platform / GPU / error-class / log
  excerpt — by construction no meeting-content field), `buildIssueUrl` composes
  a GitHub issue-form URL (`.github/ISSUE_TEMPLATE/bug-report.yml`) with the
  field ids pre-filled, enforcing an ~8 KB cap by explicitly eliding the
  diagnostics field (never silent) and steering to the clipboard fallback
  (`buildClipboardReport`). `redactMeetingPaths` is the defensive boundary pass
  for meeting-id UUIDs. Log-excerpt redaction proper is owned by the Rust side
  that assembles the report.
- **Report-problem flow (`ui/src/diagnostics/reportProblem.ts` +
  `ui/src/state/report-problem.ts`, #0014 part 2).** `reportProblem` ties the
  pieces together: it calls `get_diagnostic_report`, maps the snake_case binding
  onto the camelCase `issueReport.ts` shape (`fromBinding`), builds the URL, and
  opens the browser via `tauri-plugin-opener`; on an elided URL it writes the
  full report to the clipboard first. `useReportProblemStore` is the shared
  surface seam (in-flight flag + status line) used by the About dialog row and
  the main-window error pane (each error pane carries a "Report a problem"
  button). The store also holds `webviewError`: a window-level
  `error` / `unhandledrejection` handler mounted in `App` records the latest
  uncaught webview error into it, so a frontend crash surfaces in the same error
  pane and feeds the same report flow. No telemetry — the user submits from their
  own browser.
- **`MainWindow` (`ui/src/shell/`)** is a resizable, show/hide multi-column
  layout via `react-resizable-panels` (FR-21/FR-30): up to three columns —
  notes editor (primary), transcript, and the summary reading column. A
  segmented header toggle ("Visible panes") shows or hides each column by
  INCLUDING/EXCLUDING its `Panel` from the Group (a single `Separator` is
  interleaved between each pair of visible panes), rather than collapsing to
  zero width — this avoids stacked separators around a hidden middle pane and
  keeps one drag handle between any two columns. Percentage `minSize`s sum to
  well under 100 %, so the columns squeeze to fit and the workspace never
  scrolls horizontally. The last visible pane cannot be hidden. Per-mode
  defaults: the live transcript is hidden by default in both modes (a scrolling
  transcript distracts from note-taking; it is one click away on the toggle) — a
  finished opened meeting → notes + summary; a live recording → notes only. The
  Group has no `autoSaveId`, so showing/hiding a column re-derives the layout
  from each pane's `defaultSize` — a width the user dragged to is intentionally
  not preserved across a toggle (the squeeze-to-fit model wins over sticky
  widths). The Phase 2 two-column flex layout is replaced.
- **`RecordingStore` additions.** Gains `recordingClockMs: number | null`,
  updated by a new `recording_clock` event case and cleared to `null` on any
  transition out of `recording` (idle/stopping/paused). This is the sole
  anchor-clock source.
- **IPC seams (now in generated bindings — Stream S3).** `save_notes` /
  `load_notes` commands and the `recording_clock` event are wired through
  `ipc-bridge` and present in the regenerated `bindings.ts`. `ui/src/ipc/notes.ts`
  remains the single seam the editor uses to persist notes (it now wraps the
  generated `commands.saveNotes` / `commands.loadNotes` rather than a dynamic
  `invoke`), so tests keep mocking *this* module. `ui/src/ipc/app-event.ts`
  collapsed to a verbatim re-export of the generated `AppEvent` union (the local
  `recording_clock` augmentation is redundant now that the variant is generated).

**Phase 4 additions (Stream B — meeting-list + cross-reference + transcript-chip).**

- **Meeting-list view (`ui/src/shell/MeetingList.tsx` + `.css`, FR-33).** The
  entry surface shown before a meeting is open: a quiet index of ruled paper
  rows (Editorial Ink) listing title / date / duration / speaker-count /
  excerpt, with per-row open / rename / delete / re-transcribe / re-summarise
  actions. `MainWindow` switches between this view and the editor/transcript
  workspace on `useMeetingsStore.openMeetingId` (and the recording state): the
  list shows when no meeting is open and nothing is recording; opening a meeting
  or starting a recording reveals the workspace, and a header "Meetings"
  affordance returns to the list when idle.
- **Cross-reference, paragraph-RANGE granularity (FR-22/23).** On the
  pause-EXCLUDING timeline (`data-anchor-ms` ↔ `Segment.start_ms`, NEVER
  `Date.now()`). `ui/src/editor/hover-bridge.ts` (`NotesHoverBridge`) is a
  presentation-only ProseMirror plugin that reports the hovered paragraph's
  `data-anchor-ms` **and the next anchored paragraph's `data-anchor-ms`** (read
  from the editor DOM in document order), and **mutates no doc / dispatches no
  transaction** (so it cannot touch the A4 stamping logic, exactly like
  `AnchorMarginalia`). `ui/src/state/cross-ref.ts` maps that anchor pair to the
  half-open RANGE of segments whose `start_ms ∈ [anchor(P), anchor(nextP))` —
  through end-of-recording for the last anchored paragraph (FR-22, the locked
  Phase 4 decision; `segmentRangeForAnchors` publishes a
  `{ startIndex, endIndex }` `highlightedRange`, not a single
  `highlightedSegmentIndex`). The transcript pane highlights every row in that
  range (oxblood `--accent-tint` wash + left rule, theme tokens only). Clicking a
  transcript row publishes a scroll request whose `start_ms` the editor resolves
  to the nearest-anchored paragraph via `ui/src/editor/scroll-to-anchor.ts`
  (FR-23, a pure DOM read + `scrollIntoView`).
- **Open-meeting restore wiring (U1, SPEC Phase-4 acceptance).** Opening a saved
  meeting (`useMeetingsStore.open()` → `open_meeting` → `MeetingState`) fully
  restores its notes and transcript into the workspace.
  `ui/src/state/active-transcript.ts` is the single source-of-truth selector:
  when a saved meeting is open AND nothing is recording
  (`openMeetingId !== null && recordingState.kind === "idle"`) the transcript
  pane and the cross-reference read the SAVED meeting's
  `openMeetingState.transcript`; otherwise (live recording, or no meeting open)
  they read the live `useRecordingStore.transcript`. The notes editor hydrates
  from `openMeetingState.notes` in a **production** effect
  (`editor.commands.setContent(JSON.parse(notes.notes_json))`, keyed on the open
  meeting's notes; clears to empty when the open meeting has no notes) — no
  longer gated behind the DEV shim, which now only seeds when no meeting is open.
  Audio restore is **not** wired this phase: a saved meeting opens with its notes
  + transcript + working cross-reference, but a full audio player (and the
  pause-offset seek map from `cross-cutting.md` "Notes paragraph-anchor clock")
  is deferred to a later phase. Test coverage: `TranscriptPane`'s cross-reference
  interactions (FR-22 highlight range, FR-23 click-to-scroll, FR-24 drag-source
  payload) and the `active-transcript.ts` recording-takes-precedence branch are
  under test (`ui/src/__tests__/TranscriptPane.test.tsx`, `ActiveTranscript.test.ts`).
- **Transcript-chip node + DnD (`ui/src/editor/transcript-chip.ts` +
  `transcript-dnd.ts`, FR-24/25).** `TranscriptChip` is a first-class atom block
  node carrying `startMs` / `endMs` / `speakerId` / `text`, registered in
  `editor/extensions.ts`. Native HTML5 drag-and-drop (`transcript-dnd.ts`, MIME
  `application/x-minutist-segment`) carries a dragged transcript segment; the
  editor's `drop` handler inserts a chip (FR-24). The chip survives the
  `notes.json` `getJSON`↔`setContent` round-trip (relies on the Phase-3 opacity
  guarantee) and exports via tiptap-markdown's node `serialize` hook as a fenced
  ```transcript quotation carrying the metadata + segment text (FR-25). The
  transcript pane rows are the drag source.
- **New stores (`ui/src/state/`).** `MeetingsStore` (`meetings.ts`) holds the
  meeting-list rows + the open-meeting state and routes through the
  `ui/src/ipc/meetings.ts` seam; `CrossRefStore` (`cross-ref.ts`) holds the
  transient FR-22 `highlightedRange` (`{ startIndex, endIndex }`) + FR-23
  scroll-request links. `active-transcript.ts` is a derived selector (not a
  store) that picks the live vs. saved-meeting transcript for the panes (U1).
- **IPC seam (`ui/src/ipc/meetings.ts`).** A thin client (mirroring the Phase-3
  `notes.ts`) over the shim-aware `commands` from `./client` — NOT raw
  `./bindings` — for the six Phase-4 commands (`list_meetings`, `open_meeting`,
  `rename_meeting`, `delete_meeting`, `re_transcribe`, `re_summarise`). These
  commands are generated into `bindings.ts` (the `ipc-bridge`/orchestrator JOIN
  added them and regenerated), so `client.ts` routes them uniformly through
  `callCommand` like every other command — the earlier "pending generation"
  raw-`TAURI_INVOKE` shim path was collapsed once the bindings regenerated. The
  DEV shim (`dev-shim.ts`) supplies sample meetings + an opened-meeting payload
  so the list and an open meeting render under `vite dev`. `re_transcribe`
  reuses `AppEvent::TranscriptSegment`; `re_summarise` reuses
  `AppEvent::SummaryReady`.

**Phase 5 additions (Stream S4 — summary view).**

- **Summary view (`ui/src/shell/SummaryView.tsx` + `.css`, FR-30).** A reading
  surface in the Editorial Ink language that renders the meeting's `summary.md`
  markdown (via `markdown-it`, `html: false`) as a paper sheet, exposes a
  Summarise action with an in-progress affordance while the LLM runs, and lets
  the user edit the raw markdown and persist it. It is a workspace **column**
  (not a popup overlay): one of the up-to-three show/hide panes `MainWindow`
  lays out (notes / transcript / summary). The summary column is offered only
  for a FINISHED opened meeting (idle + a saved meeting open) — there is no
  summary mid-recording — and a finished meeting **defaults to notes + summary,
  with the transcript hidden** (the summary is what you reach for after a
  meeting). The meeting it summarises is the open meeting else the live
  recording's `meeting_id`. The meeting-list row's Summarise action (renamed
  from the Phase-4 "Re-summarise" stub button) also runs the real summariser
  through the summary store.
- **Summary store (`ui/src/state/summary.ts`).** Transient UI state only
  (`summaryMarkdown`, `summarising`, `meetingId`, `lastError`) routed through the
  `ui/src/ipc/summary.ts` seam; `summary.md` on disk is authoritative. Its
  `handleEvent` is dispatched alongside `RecordingStore`/`ModelsStore` from
  `useAppEventBridge` and handles `AppEvent::SummaryReady` by re-reading the
  summary (`get_summary`) and leaving the in-progress state — scoped to the
  loaded meeting so an unrelated meeting's event does not clobber the view.
  `save()` rolls back the optimistic markdown on error so the store never
  retains an unsaved edit as if it persisted.
- **IPC seam (`ui/src/ipc/summary.ts`).** A thin client (mirroring `notes.ts` /
  `meetings.ts`) over the shim-aware `commands` from `./client` — NOT raw
  `./bindings` — for the three Phase-5 commands: `summarise_meeting(meeting_id)
  -> ()`, `get_summary(meeting_id) -> Option<String>`, and
  `save_summary(meeting_id, summary_markdown) -> ()`. These commands are added
  to `ipc-bridge` by the Phase-5 backend JOIN (Stream S5), which regenerates
  `bindings.ts`. Until that regeneration lands, `client.ts` routes them through a
  shim-aware `callPendingCommand` raw-`invoke` path (the same approach the
  Phase-4 meeting commands used before Stream C regenerated the bindings); once
  regenerated they fold into `callCommand` like every other command. The DEV
  shim (`dev-shim.ts`) supplies a sample `summary.md` + a `summary_ready`
  fan-out so the view renders and updates under `vite dev`. The summary crosses
  the wire as an opaque markdown `String`; `summarise_meeting` reuses
  `AppEvent::SummaryReady` (no new event).

**Phase 6 additions (Stream S4 — diarization overlay + re-diarize + toggle).**

- **Speaker chip (`ui/src/transcript/TranscriptPane.tsx` + `.css`).** Each
  transcript row renders a quiet "Speaker {id}" chip before its text when the
  segment carries a `speaker_id` (the diarizer's first-seen label `A`/`B`/…,
  already present on `Segment` in `bindings.ts` — no regen). The chip is hidden
  entirely when `speaker_id` is `null`/absent (un-diarized). Editorial Ink:
  `--accent-tint` background, `--rule` hairline, `--stone` ink — tokens only.
  As of Phase B (live diarization wiring) `speaker_id` can now be populated
  during recording by the additive `OnlineDiarizer` (see the `orchestrator`
  "Phase B — live diarization wiring" note), so the chip renders for live
  segments too — no UI change is needed for that (live-label UI consumption is
  Phase C). The on-stop `SherpaDiarizer` pass remains authoritative and rewrites
  the labels on stop. The chip shows the user-set display name when one exists
  (`MeetingMeta.speaker_names[label]`, sourced from `openMeetingState.meta`),
  else the bare label. It is an editable button (inline rename → the
  `set_speaker_name` command) **only when viewing a saved, finalised meeting**
  (`openMeetingId !== null && recording idle`); during a live recording it is a
  display-only span, because the live labels are provisional (re-lettered on
  stop, which also clears `speaker_names`) and there is no finalised metadata to
  write. The timestamp — not the chip — is the row's drag handle, so the chip
  stops click propagation to avoid triggering the row's jump.
- **`diarization_complete` re-read (`ui/src/state/meetings.ts`).** The meetings
  store gains a `handleEvent` (dispatched alongside the recording / models /
  summary stores from `useAppEventBridge`) that, on
  `AppEvent::DiarizationComplete { meeting_id, speaker_count }`, re-reads **that
  meeting's** transcript via `open_meeting` scoped to the **event's**
  `meeting_id` (NOT the live recording store) when it is the open meeting, so
  the restored `openMeetingState.transcript` (the source the transcript pane
  reads for a saved meeting, U1) reflects the new speaker tags; for a
  non-open meeting it refreshes only the list so the row's speaker count
  updates. The recording store does **not** handle this event.
- **Diarization-enabled toggle (`ui/src/state/recording.ts` +
  `ui/src/state/diarization-settings.ts` + `MainWindow.tsx`).** A header
  checkbox ("Diarize on stop", off by default) round-trips the
  `diarization_enabled` setting through `commands.updateSettings`, the same
  round-trip-through-settings pattern as the device selection. The field is
  owned by the `settings` crate and is a first-class member of the generated
  `Settings` type in `bindings.ts` (`diarization_enabled?: boolean`).
  `diarization-settings.ts` reads/writes that canonical field directly and keeps
  `SettingsWithDiarization` only as a named alias of `Settings` so existing call
  sites and tests need no change. It gates the orchestrator's on-stop
  diarization pass; re-diarize is independent of it.
- **Re-diarize action + IPC seam (`ui/src/ipc/meetings.ts::rediarize` +
  `MeetingList.tsx` row action + `MainWindow.tsx` open-meeting workspace menu).**
  `rediarize(meeting_id)` calls the generated `rediarizeMeeting` command, which
  is present on the generated `commands` surface and routes through `callCommand`
  in `ui/src/ipc/client.ts` like every other command (the earlier shim-aware
  `callPendingCommand` raw-`invoke` path was collapsed — A9 — and survives only
  in past-tense comments). The DEV shim (`dev-shim.ts`) supplies sample
  speaker-tagged transcript segments, a `rediarize_meeting` handler, and a
  `diarization_complete` fan-out so the chips and re-read render under
  `vite dev`. Tests mock the `../ipc/meetings` seam. The command is the
  snake-case `rediarize_meeting(meeting_id: MeetingId) -> ()` (camelCase
  `rediarizeMeeting` on the generated surface); it decodes the meeting's
  pause-INCLUDING PCM, runs the `SherpaDiarizer` over the stored segments,
  rewrites `transcript.json` with the overlaid `speaker_id`s, refreshes the
  index row's `speaker_count`, emits
  `AppEvent::DiarizationComplete { meeting_id, speaker_count }`, and (like
  `re_transcribe`) refuses unless the recorder is `Idle`.

**Phase 7 additions (first-run onboarding gate).** `App.tsx` is the gate point:
it fetches `settings` (via the recording store's `refreshSettings`) + the model
list on mount, holds the UI neutral (`return null`) while settings are pending
(so a returning user is never flashed onboarding), then renders `Onboarding`
(`ui/src/shell/Onboarding.tsx`) when `settings.onboarding_completed` is `false`,
else `MainWindow`. The `useAppEventBridge` hook stays mounted ABOVE this gate so
the event listener is never dropped by the conditional render. The onboarding
flow is a 3-step Editorial-Ink sheet (welcome → model download [reuses
`ModelDownloadStatus`] → quick settings [theme + diarization toggle]); its final
step persists `onboarding_completed = true` through the **existing**
`commands.updateSettings` seam (the recording store's single settings path) —
there is NO dedicated `complete_onboarding` command and no raw-`invoke` shim
(rule A9). Onboarding step navigation is a tiny `useOnboardingStore`
(`ui/src/state/onboarding.ts`); completion lives only in persisted settings (the
single source of truth), not in that store.

An **About dialog** (`ui/src/shell/About.tsx` + `about-content.ts`, opened from a
header button in `MainWindow`) satisfies the Phase 7 acceptance item by listing
the bundled-model SPDX licenses + a NOTICE line and the major OSS attributions.
The bundled-model rows are **DERIVED from the manifest** via the models store:
`ModelStatus` now carries a `license` field (populated by `model-registry` from
each `resources/models.json` entry and exposed over IPC), so `About.tsx` reads
`id` / `display_name` / `license` straight from `useModelsStore` and renders an
SPDX-normalised list — there is no hand-mirrored model list to drift (a model
swap flows to About automatically). Only the OSS-component attributions, the app
version, and the NOTICE line remain static in `about-content.ts` (they are not
in the manifest). The `dev-shim` still hand-seeds models for `vite dev` visual
QA, but that path never reaches the shipped dialog.

**Phase 9 — chat pane + summary preset picker.**

- **Chat store (`ui/src/state/chat.ts`, zustand).** Holds the meeting-scoped
  chat pane's transient state: the open session (`sessionId`), its `messages`,
  the in-flight streamed assistant text (`streaming`), a transient
  `toolActivity` indicator, the `sessions` list (the switcher), and
  `inFlight` / `lastError`. Its `handleEvent` is dispatched alongside the other
  stores' from the single `useAppEventBridge` fan-out (one listener, no second
  subscription). **Event-reconciliation rule (the lossy-broadcast guarantee, see
  `cross-cutting.md` — "Agent chat loop"):** `chat_token` deltas APPEND to the
  `streaming` buffer as a progressive hint and are NEVER trusted as the final
  answer; `chat_turn_complete.final_text` is authoritative and REPLACES the
  streamed buffer with the full reconciled reply (appended as the assistant
  message), so a dropped delta on the broadcast channel cannot corrupt the stored
  text. `chat_tool_call` / `chat_tool_result` drive the transient tool indicator;
  `chat_error` surfaces the error and clears the in-flight state. Every chat
  event is per-session scoped — an event whose `session_id` is not the open
  session is ignored, so a backgrounded session's turn never clobbers the open
  one. All IPC routes through the `ui/src/ipc/chat.ts` seam (wrapping the
  shim-aware `commands.*` from `./client`, NOT raw `bindings.ts`), so tests mock
  the seam.
- **Chat pane (`ui/src/shell/ChatView.tsx` + `.css`).** A workspace column (not
  an overlay) wired into `MainWindow`'s `buildPanes` alongside notes / transcript
  / summary, gated on a concrete `activeMeetingId` (a live recording's meeting or
  an opened saved meeting) and hidden on the meeting-list entry surface — chat is
  meeting-scoped. A "Chat" segment is added to the existing pane-visibility
  toggle (off by default; the last visible pane still cannot be hidden). It
  renders user / assistant bubbles (assistant markdown via the Phase-3
  markdown-it, `html: false`), a compact tool-activity row, a streaming caret
  while tokens arrive, an error state, a send box (Enter to send, Shift+Enter for
  a newline, disabled while a turn is in flight), and a session switcher
  (new / pick / delete). Editorial-Ink tokens only.
- **Summary preset picker (D4).** The summary view (`SummaryView.tsx`) gains a
  "Summary prompt" disclosure: a preset `<select>` bound to
  `settings.summary_preset` (the four `SummaryPreset` values, human labels) + a
  custom-prompt `<textarea>` bound to `settings.summary_system_prompt`. A
  non-empty custom prompt OVERRIDES the selected preset (the backend's
  `Settings::effective_summary_prompt`); the UI states this explicitly. Both
  persist through the **existing** `commands.updateSettings` seam via two new
  recording-store actions (`setSummaryPreset` / `setSummarySystemPrompt`) and the
  `ui/src/state/summary-preset-settings.ts` read/with helpers — no new command,
  the same round-trip-through-settings pattern as `setTheme`.

### Design system — "Editorial Ink" (light theme)

A warm-paper, document-centric **light** theme applied across the webview.

- **Token source.** `ui/src/styles/theme.css` is the single source of truth for
  all colour / radius / shadow / type tokens (CSS custom properties). Component
  CSS references these variables only — no hard-coded colour/radius/shadow
  literals. `ui/src/styles/global.css` holds the base layer (warm-desk field,
  oxblood `::selection` + focus-visible ring, the orchestrated load-reveal
  keyframes). Both are imported once from `ui/src/main.tsx`. The accent is a
  single oxblood ink used sparingly (recording dot, links, active/primary
  control, focus, selection); `--stone` is darkened from the brief's value to
  clear 4.5:1 on the paper surface for body-level meta.
- **Fonts (local-first, no CDN).** Bundled via Fontsource:
  `@fontsource-variable/fraunces` (display — app wordmark + editor headings, via
  its `full` axis CSS exposing opsz/wght/SOFT/WONK) and
  `@fontsource-variable/newsreader` (reading body + UI chrome). Italic faces of
  both back blockquotes / emphasis. These are the only two UI font families;
  woff2 files ship as build assets so the app renders offline.
- **Notes sheet (binder paper) + columns.** The notes editor (`ui/src/editor/`)
  renders as a sheet of binder paper that **fills its pane** (no floating card /
  desk margin): a narrow left timestamp gutter, a structural pale-oxblood
  vertical **margin rule** (`--rule-margin`) dividing the gutter from the
  writing column, and — when the `notes_paper_rules` setting is on (default) —
  faint horizontal writing-paper rules (`--rule-line`) pitched to the body
  leading (`--notes-leading`), with headings/lists taking whole-leading space so
  the body re-aligns. The class `notes-editor--ruled` toggles the horizontal
  rules; the margin rule is always shown. The transcript pane
  (`ui/src/transcript/`) and summary view (`ui/src/shell/SummaryView.tsx`) are
  the quiet, recessed `--sheet-quiet` columns. The resizable show/hide
  `react-resizable-panels` structure and panel `id`s (`notes` / `transcript` /
  `summary`) are described under "Phase 4/5 additions". The top bar
  (`ui/src/shell/`) is calm and hairline-ruled: wordmark left, recording status
  focal (oxblood dot, gentle pulse only while recording, plus a tabular elapsed
  clock in `RecordingStatus.tsx`), grouped transport + slim meter + the
  segmented pane-visibility toggle right.
- **Margin-anchor marginalia.** `ui/src/editor/anchor-marginalia.ts` is a
  **presentation-only** ProseMirror decoration extension: it renders each
  anchored paragraph's `data-anchor-ms` value as a quiet timestamp in the sheet's
  left gutter, right-aligned flush against the oxblood margin rule (editorial
  side-note). It adds no node attributes and dispatches no transactions, so it
  cannot interfere with `ParagraphAnchor`'s stamping logic and never shifts the
  text column.
- **Appearance settings.** The Settings drawer's Appearance group exposes the
  colour-theme control (System / Light / Dark — `settings.theme`, applied to the
  document root in `App.tsx`; "System" follows `prefers-color-scheme`) and the
  writing-paper-rules toggle (`settings.notes_paper_rules`). Both are
  presentation-only and round-trip through the existing `update_settings` seam.
- **Dev render shim (DEV-only).** `ui/src/ipc/dev-shim.ts` (sample data) +
  `ui/src/ipc/dev-shim-guard.ts` (`shouldUseDevShim`) let the full app render
  under `vite dev` in a plain browser with no Tauri backend, for visual QA. The
  guard activates only when `import.meta.env.DEV` is true, the runner is not
  Vitest (`MODE !== "test"`), and the Tauri global is absent. `ui/src/ipc/client.ts`
  reaches the shim exclusively through a dynamic `import()`, so the production
  build never bundles or fetches it (the chunk is dead-code-eliminated).
- **Binding on all new views.** Editorial Ink is the webview design language.
  Every view added in later phases — meeting-list, summary, settings, first-run
  / onboarding, model-download UI — MUST consume `theme.css` tokens and reuse
  the established patterns (Fraunces display / Newsreader body, the single
  oxblood accent used sparingly, paper surfaces, hairline rules, restrained
  motion respecting `prefers-reduced-motion`). No new view introduces its own
  palette or type families; a hard-coded colour/font or an off-system pattern
  is a code-review finding. New views should render in the DEV shim with
  representative sample data so they can be visually QA'd the same way.

**Translation UI — translated transcript as derived view (WU4).**

- **`ui/src/ipc/translations.ts` (IPC seam).** Wraps `commands.translateMeeting`
  and `commands.getTranslations` (both added to `client.ts`'s delegating surface
  in WU3). `getTranslations` normalises the JSON-wire `Record<string, string>`
  (JSON object keys are always strings) into a `Map<number, string>` keyed by
  segment index. Tests mock this seam module, not the generated bindings. The DEV
  shim supplies no-op stubs for both commands.
- **`ui/src/state/translations.ts` (Zustand store).** Holds `selectedLanguage`
  (`null` = verbatim view), `translations: Map<number, string>` (the per-segment
  cache for the open meeting + language), `translateInFlight` (blocks the Translate
  button while the backend pass runs), and `openMeetingId` (set by
  `TranscriptPane` via `setOpenMeeting` so `handleEvent` can guard event-scoped
  reloads). Actions: `translate(meetingId, language)` — calls `translateMeeting`
  then `getTranslations` and populates the map; `loadTranslations(meetingId,
  language)` — reads without re-translating (on-open restore); `showVerbatim()`
  — clears `selectedLanguage` and drops the map; `setOpenMeeting(id)` — called on
  meeting open/close to clear stale translations; `reset()` — full reset.
  `handleEvent` reacts to `translation_ready { meeting_id, language }`: if the
  event matches the active meeting + `selectedLanguage`, calls `loadTranslations`
  to refresh the overlay. Dispatched from `useAppEventBridge` alongside the other
  stores.
- **`ui/src/state/operation-progress.ts` (updated).** `translation_ready` added
  to the terminal-event list so the per-row progress indicator clears when a
  translate pass finishes.
- **`TranscriptToolbar` (updated in `TranscriptPane.tsx`).** The toolbar gains:
  a `<select>` pre-seeded to the first language in `OUTPUT_LANGUAGES` (re-used
  from `OutputLanguagePicker`); a Translate button (disabled while any op is
  in-flight, label changes to "Translating…" during the pass); a "Show original"
  button that replaces the selector + Translate pair once a translated view is
  active. A thin `transcript-pane__toolbar-sep` rule divides the reprocess actions
  from the translation controls.
- **Per-segment overlay (updated in `TranscriptPane.tsx`).** When
  `selectedLanguage !== null`, each row renders the translated text from the
  `translations` Map if the index is present, or falls back to `seg.text` for
  segments that have not yet been translated (partial pass). Translated rows show
  a quiet `transcript-pane__translated-label` suffix (the language name, muted
  mono) so the substitution is visible at a glance. A one-tap "Show original"
  in the toolbar flips back to the verbatim view.
- **Test coverage** (`ui/src/__tests__/Translations.test.tsx`): `translate()`
  invokes `translateMeeting` then `getTranslations` and populates the store;
  `loadTranslations()` fetches without a new translation pass; `showVerbatim()`
  clears; `setOpenMeeting()` resets on meeting change; `handleEvent` refreshes
  on matching `translation_ready` and ignores events for different languages or
  meetings; `translation_ready` clears the operation-progress indicator.

## What lives where — quick reference

- **Editing audio capture buffer size:** `audio-capture` crate.
- **Tweaking VAD silence threshold:** `vad-chunker` crate + `settings`
  schema.
- **Changing ASR prompt template:** `asr-runtime` crate.
- **Changing summarisation system prompt default:** `settings` crate +
  `summariser` prompt builder.
- **Adding a new meeting metadata field:** `common` (type), `persistence`
  (storage), `ipc-bridge` (surface).
- **Adding a new Tauri command:** `ipc-bridge` crate. tauri-specta
  regenerates the TS bindings.
- **Adding a new event from backend to webview:** `ipc-bridge` defines
  the event; `orchestrator` (or the relevant crate) emits via an
  abstraction in `common`.
