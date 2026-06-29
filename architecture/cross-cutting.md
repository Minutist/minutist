# Cross-cutting concerns

Concerns that touch every component. Decisions here are binding on all
**production** crates; deviation requires an architecture-doc update.

The `spikes/` crates are exempt — they're throwaway code that proves
upstream APIs work, and don't ship. Spikes may use `anyhow`, `println!`,
unbounded channels, etc. without review findings. Any spike code that
graduates into a production crate is held to these rules at migration
time, not before.

## Async runtime

**Tokio**, multi-threaded scheduler. Tauri requires it; we don't fight
that.

- Long-running native work (ASR / LLM inference, file I/O of meeting
  audio) runs on `tokio::task::spawn_blocking` to keep the multi-threaded
  scheduler responsive for IPC and events.
- The orchestrator owns the long-running task handles. Other components
  expose async functions; the orchestrator chooses how they're driven.
- No use of `block_on` inside Tauri command handlers. Commands return
  futures that the runtime polls.
- Spawning from Tauri's `setup()` hook MUST use `tauri::async_runtime::spawn`,
  NOT a bare `tokio::spawn`: `setup` runs on the main thread with no entered
  Tokio runtime, so `tokio::spawn` panics ("there is no reactor running"). Tauri's
  async runtime is tokio-backed, so tokio primitives (broadcast receivers, etc.)
  work inside it. (The event forwarder is spawned from `setup`.)

## Threading model

| Workload | Where it runs |
|---|---|
| Audio capture callback (cpal) | cpal's own thread; pushes frames into a bounded channel. With system-audio capture on (`settings.capture_system_audio`), a SECOND cpal callback (the render-endpoint loopback source) runs on its own thread and pushes into its own bounded ring — both RT callbacks keep the `try_lock`/drop-oldest discipline. |
| Audio mixer (mic + loopback) | A `spawn`/`spawn_blocking` task draining the two per-source 16 kHz batch channels; SUMS sample-wise, clamps, meters, and forwards the single mixed stream. Only present when system-audio capture is on; mic-only otherwise. Never blocks the RT callbacks (those feed the upstream rings). **Starvation valve:** if one source is idle (e.g. loopback when nothing is playing through the speakers) the mixer must NOT wait to pair samples — past a ~30 ms skew (`mixer::MAX_SKEW_SAMPLES`) it zero-fills the idle source and emits the live one, else the mic is buffered forever (silent transcript + dead meter). The cap also sets the meter/mic-latency cadence on the idle-loopback path (~30 Hz). |
| VAD inference | Runs inline in the single runner drain loop (`spawn_blocking`), which also drains the sample channel and writes audio — not a dedicated VAD task. |
| ASR inference | A dedicated `spawn_blocking` task per active model; chunks queued via bounded channel. |
| Diarization (offline) | One-shot `spawn_blocking` task triggered on stop or user action — the authoritative pass. |
| Diarization (live, Phase B) | Per-VAD-segment `OnlineDiarizer::assign_segment` driven from the runner's drain-loop thread (`spawn_blocking`) at SegmentEnd — gated on the `diarization_enabled` setting AND the embedding model being locally `Available` (no download, no block at start; the heavy `EmbeddingExtractor` load is built on `spawn_blocking` before the runner spawns). Best-effort/additive: any failure degrades to "no label" without affecting recording/transcription. See the live-vs-offline note below. |
| Summarisation | One-shot `spawn_blocking` task triggered by user action. |
| Attachment conversion | A SINGLE long-lived worker task (`tauri::async_runtime::spawn`, same pattern as `spawn_event_forwarder`). Jobs arrive on a **bounded** `tokio::sync::mpsc` channel; the worker processes them one at a time, each on `spawn_blocking`. Best-effort: every error is logged (`target: "ipc-bridge"`); the worker never panics. Back-pressure: `add_attachment` uses `try_send`; a full queue marks the entry `Failed("conversion queue full")` immediately. |
| Persistence writes | `spawn_blocking` per write op for now; revisit if it shows up in profiling. |
| Tauri command handlers | Tokio worker threads. Short-lived, dispatch to the above. |

Bounded channels everywhere. Unbounded queues are not allowed — they
hide back-pressure that the live pipeline needs to surface.

`unsafe` Send/Sync assertions in the workspace (each documents its full
safety argument at the impl site):

- `summariser`: `unsafe impl Send + Sync` for the held `LlamaModel` (see
  the summariser section in components.md).
- `audio-capture`: `unsafe impl Send for StreamHandle`
  (`#[cfg(target_os = "macos")]`) — cpal's CoreAudio stream is `!Send`
  solely via its property-listener closure; every shared access,
  including listener-vs-drop, is synchronised inside cpal
  (`Arc<Mutex<StreamInner>>` + `Weak` upgrade on the listener thread), so
  moving the owned handle across tokio worker threads is safe.
  ALSA/WASAPI streams are `Send` natively; `Sync` is never asserted.
- `ipc-bridge` (`GemmaVlm`): **no `unsafe` impl**. `GemmaVlm` holds only a
  `ChatHandles` (all `Arc` / `PathBuf` / handle fields), so `Send + Sync` is
  auto-derived — it carries no `MtmdContext`. The vision context lives in
  `LlamaSummariser` (summariser crate) as `OnceLock<Mutex<MtmdContext>>`, where
  the `Mutex` serialises the mutating encode path (not a Send/Sync barrier —
  `MtmdContext` is already `unsafe impl Send + Sync` in `llama-cpp-2`). See
  `cross-cutting.md` — "Held model serves vision".

**Live vs. offline diarization (Phase B).** There are two independent
diarization paths. The offline `SherpaDiarizer` / `common::Diarizer`
on-stop (and re-diarize) pass is the SOURCE OF TRUTH for the finished
transcript. The live `OnlineDiarizer` (in `crates/diarizer`,
`src/online`) is an ADDITIVE hint: it emits a sticky first-seen label
("A"/"B"/…) per VAD segment as the segment closes and NEVER
retroactively relabels — live labels are provisional and may disagree
with the authoritative on-stop pass. Like the offline diarizer it is
driven from `spawn_blocking`; its public `&self` methods
(`assign_segment`, `speaker_count`) hold a single `Mutex` over the
`(EmbeddingExtractor, OnlineClusterer)` pair because sherpa's
`compute_speaker_embedding` is `&mut self` (the same `&self`-trait-over-
`&mut`-engine pattern the offline `Mutex<Diarize>` uses). The clustering
itself is a pure, FFI-free running-mean-centroid clusterer; only the
embedding extraction crosses into sherpa. Its cosine-SIMILARITY threshold
(`OnlineClustererConfig::default` = 0.25) is the OPPOSITE orientation to the
offline distance `cluster_threshold` (0.75) and was tuned by a separate sweep
(2026-06-05) on the same real recording + fixtures: 0.25 is the lowest value
that still separates two distinct speakers, maximising single-speaker merging.
The greedy online path has little margin, so live labels stay provisional — the
on-stop pass is the safety net.

**Offline over-split prune (issue #63, 2026-06-10).** On long, acoustically-
varied recordings (room coloration + system-audio loopback + a podcast over a
loudspeaker) the offline pass over-split: one speaker's embeddings drift past the
single distance `cluster_threshold`, minting extra clusters — the field saw 19 /
29 speakers where the truth was a handful. A distance threshold alone cannot
separate "same speaker, drifted" from "different speaker", so the robust fix is a
**post-cluster prune** in `overlay_speakers`, NOT a higher threshold. The
shipped `DiarizerConfig::default()` now carries three additional knobs:
`min_duration_on` / `min_duration_off` (`0.3` / `0.5`, sherpa's own example
values — previously pinned to `0.0`/disabled — bridging short intra-speaker gaps
and dropping sub-300 ms turns inside sherpa) and `min_cluster_share` (`0.02`):
after the interval-join, any cluster winning under 2 % of the attributed speech
DURATION is dropped and its segments reassigned to the nearest surviving cluster
(mirroring pyannote's production `min_cluster_size` reassignment and the 2026
relative-min-cluster-size result, f ≈ 0.01–0.02). A `min_cluster_segments`
floor and a `max_speakers` cap exist but are OFF by default (`0` / `None`) — the
duration-share prune is the primary lever; the segment-count floor would wrongly
fold a genuine speaker who utters one long, high-share segment. The prune is pure
post-processing over sherpa's turns (sherpa-onnx's `FastClustering` exposes no
such knob and returns every cluster it forms). On a 6-min slice of the reported-
19 meeting the shipped config takes the count 9 → 5; the effect compounds over
the full recording. See the journal sweep (2026-06-10) for the count-vs-knob
table and `crates/diarizer/tests/oversplit_eval.rs` for the gated eval harness.
The clean-fixture accuracy test still gives 2 / 1 (balanced speakers sit well
above the 2 % floor).

As of **Phase B** the live path is wired into the orchestrator (see
`components.md` — `orchestrator` "Phase B — live diarization wiring").
The label is assigned per VAD segment at SegmentEnd on the runner's
drain-loop thread and rides a parallel `speaker_ids` column
(`Accumulator` → `FlushPayload` → `emit_segments_proportional` →
`Segment.speaker_id`). Consequently live labels are now emitted on
`AppEvent::TranscriptSegment` and persisted via `WriterCommand::WriteSegment`
(into `transcript.json`) DURING recording. The on-stop pass remains
authoritative: when `diarization_enabled` is true, the whole-transcript
rewrite on stop overwrites the live labels with the offline result. The
wiring adds no dependency edge (the `orchestrator → diarizer` edge pre-exists)
and no `common`-level online trait (the live path is a concrete struct;
the existing `common::Diarizer` trait stays offline-only).

**System/call audio capture + echo (AEC is future work).** When
`settings.capture_system_audio` is on, the render-endpoint loopback is captured
alongside the mic and summed into the single transcribed stream (see
`components.md` — `audio-capture`). If the mic also picks the call audio up from
the speakers, mixing the loopback in doubles that audio (an echo). v1 handles
this only with the toggle (ON by default, opt-out; the UI advises turning it off
when the mic hears the speakers). Acoustic echo cancellation — using the
loopback as the reference signal to subtract the speaker bleed from the mic — is
deliberately **deferred**; it would live in the mixer/capture path. Loopback is
Windows-only (WASAPI) for now; other platforms fall back to mic-only.

## Error handling

Two layers:

1. **Per-crate `Error` enum.** Each crate defines its own `Error` type
   via `thiserror`. Variants are crate-specific. Never `anyhow::Error`
   inside a public function signature.
2. **Boundary conversion.** When errors cross a crate boundary going
   towards the IPC bridge, they're converted to a shared
   `common::AppError` that carries a stable code + display string.
   `From` impls live in the source crate.

The webview never sees a per-crate error shape. At the Tauri command
surface, `AppError` is re-encoded into `ipc-bridge`'s `IpcError` — a
hand-mirrored enum carrying the same discriminants and the same serde
shape (`{"code": "...", ...}`). `IpcError` exists because `common` has
no `specta` dependency by design, so `AppError` cannot derive
`specta::Type`; the derive lives on `IpcError` in `ipc-bridge` instead.
The webview literally receives `IpcError`, which mirrors `AppError`, so
the TypeScript binding stays stable as internal error enums churn.

Panics: never as control flow. A panic inside a `spawn_blocking` task
must abort the parent orchestrator task and surface as a recoverable
`AppError`. The app does not exit on a single bad recording.

## Logging

`tracing` crate. Subscriber configured in `app-main`:

- File appender at `{app-data}/logs/minutist.log`, rotated daily,
  7-day retention.
- Console output in debug builds only.
- `RUST_LOG`-style filtering honoured at startup.

Each component uses a static `target` matching the crate name:
`tracing::info!(target = "asr-runtime", ...)`. The reviewer is expected
to flag log calls without a target — that's how we keep logs
filterable.

No `println!` or `eprintln!` outside test code. Two narrow exceptions:

- **Bootstrap-time fallback before the tracing subscriber is initialised.**
  The `app-main` binary may use `eprintln!` to surface fatal startup errors
  that prevent the subscriber itself from being constructed. Limit to the
  pre-subscriber path.
- **Developer-facing CLI helpers** that intentionally print to stdout as
  their primary output (e.g. `cargo run --bin generate-bindings` writing the
  generated bindings path to the console).

The reviewer is expected to flag any `println!` / `eprintln!` outside these
two carve-outs and outside `#[cfg(test)]`.

**Crash capture (issue #0014, no telemetry).** `app-main` adds a `tracing`
ring-buffer layer (`src-tauri/src/crash.rs`) that retains the last N formatted
log lines in a process-wide static (under the same `EnvFilter`, so it sees the
same info+ lines the file appender does), plus a `std::panic::set_hook` that, on
a panic, writes a REDACTED `{app-data}/logs/last-crash.txt` (app version,
platform, best-effort GPU mode, panic message + location, backtrace, and the
recent ring lines). The hook chains the previous one so the default print/abort
behaviour is preserved. Every line written passes through a meeting-id-UUID
redaction pass (`crash::redact`, mirroring the webview's `redactMeetingPaths`).
By the #0014 privacy audit, no meeting *content* (transcript / notes / title /
speaker text) is logged at any level, so the ring never holds it; the UUID strip
is the defensive boundary for paths. This invariant is enforced at the source:
the ASR backends (`asr-runtime`, `asr-parakeet`) log `text_chars`/`words`
counts, never the transcribed text, and `persistence` logs meeting ids, never
titles. Re-introducing a content-bearing log field would breach this guarantee. The file is read by
`ipc-bridge::get_diagnostic_report` for the "Report a problem" flow — nothing is
sent off the machine; the user reviews + submits from their own browser. This is
NOT telemetry (see "Telemetry" below): there is no automatic transmission and no
network hook.

## IPC contract

Generated by **tauri-specta** at build time. The build step is the
contract:

1. Rust commands annotated with `#[tauri::command]` + tauri-specta
   collector.
2. `cargo run --bin generate-bindings` (or the equivalent build script)
   writes `ui/src/ipc/bindings.ts`.
3. The webview imports from `bindings.ts`; no hand-rolled command
   names.

Events are declared in `common` as enum variants; the IPC bridge owns
the wire encoding. Adding an event requires updating both the enum and
the regen step.

## ASR chunking constraint

Phase 0 Spike 1 confirmed that llama.cpp's mtmd audio encoder uses a
fixed 30 s window. Sub-30 s inputs are silence-padded internally and the
model hallucinates into the pad. This is binding on every `AsrBackend`
caller until upstream issue ggml-org/llama.cpp#20914 lands (multi-phase
streaming work; not in v1's timeframe).

**Verified still binding (2026-06).** A primary-source investigation
confirmed #20914 (realtime/streaming ASR) has NOT landed — its Phase-1
APIs are absent from llama.cpp master (the original monolithic PRs were
rejected; the issue was reopened 2026-06-01) — and the audio encoder
window is still a fixed 30 s everywhere. The pinned `llama-cpp-2 =0.1.146`
already vendors a current llama.cpp (commit `e21cdc11`, build b8783,
2026-04-13) that includes Qwen3-ASR mtmd audio, so there is no version lag
to chase. The silence-preservation and `</asr_text>` early-stop sub-rules
below remain mandatory. The chunk-*sizing* rule was REVISED on 2026-06-04
(see below) after a live test contradicted the Phase-0 ≥25 s guidance.

**Chunk sizing — REVISED 2026-06-04 (supersedes the Phase-0 "≥25 s" rule).**
The orchestrator must bound each `AsrBackend::transcribe_chunk` call to
**roughly 5–13 s** of audio, NOT fill the 30 s window. Phase 0 reasoned that
sub-30 s inputs hallucinate into the internal silence pad, so chunks should be
shaped to ≥25 s. A live recording (2026-06-04) disproved that for the upper
end: a ~26 s chunk drove Qwen3-ASR into a greedy-decode **repetition loop**
(the same failure the silence-preservation rule guards against, but triggered
by over-long input rather than compaction). Short chunks do NOT hallucinate
into the pad in practice because the `</asr_text>` early-stop truncates any
post-transcript continuation. So the binding rule is now an upper bound:
- **VAD force-splits** any single speech segment at `VadConfig::max_segment_ms`
  (10 s) — see `vad-chunker`.
- The **batched-VAD accumulator** flushes at `FLUSH_MIN_SECS` (3 s) or after
  `LATENCY_WINDOW_SECS` (2 s) of quiet, so a `transcribe_chunk` call receives at
  most ~`FLUSH_MIN_SECS + max_segment_ms` ≈ 13 s — see `orchestrator::runner`.

Residual: very short / low-content segments (breaths, single fillers) can still
misfire — e.g. a spurious language switch (Qwen3-ASR auto-detects language per
call and has no hint). A prompt-level language hint is the planned mitigation
(tracked separately), not a chunk-sizing concern.

**Preserve original-timeline silences.** Phase 0 Spike 3 found that
concatenating VAD-trimmed utterances back-to-back into the batched
buffer causes Qwen3-ASR to enter a greedy-decode loop after the first
few words. Reconstructing the inter-utterance silences via zero-padding
between segments restored correct output. Qwen3-ASR appears to use
internal silences as sentence-boundary anchors. The orchestrator MUST
keep original-timeline gaps between VAD segments (cap individual gaps
at ~3 s to bound the 30 s buffer); do not "compact" VAD-bounded audio
before dispatching to mtmd.

**Early-stop on `</asr_text>`.** Qwen3-ASR's output schema wraps the
transcript as `language English<asr_text>...</asr_text>`. The model
does not always emit `<|im_end|>` (EOG) when the audio is shorter than
the 30 s window; it instead generates hallucinated continuation past
the real transcript end. `asr-runtime` MUST stop generation on
`</asr_text>` in addition to EOG.

Alternative strategies (pad-to-30s-per-call, post-filter hallucinated
tail) are documented in Spike 1's README as fallbacks if batched-VAD's
latency profile is unacceptable.

## ASR engine routing

There are two ASR backends behind `common::AsrBackend` (Phase 8): `asr-parakeet`
(sherpa-onnx Parakeet TDT v3 — English + 24 EU languages, per-word timestamps,
CPU) and `asr-runtime` (llama-cpp-2 Qwen3-ASR — 52 languages/dialects, no
timestamps; 0.6B CPU default + optional 1.7B GPU tier).

The engine is chosen **deterministically from the user's `transcription_language`
setting**, never by inspecting the audio (the language isn't known before
transcription). The mapping is a pure function in `common`
(`asr_engine_for_language`) so the orchestrator and the UI agree:

- language ∈ Parakeet's set (English + the 24 EU locales) → **Parakeet** (primary
  — better English/EU accuracy + timestamps);
- language ∈ Qwen-only (Chinese, Japanese, Korean, Arabic, …) → **Qwen**;
- `Auto-detect` (the `""`/`"auto"` sentinel) → **Qwen** (broadest coverage is the
  safe default when the language is unknown).

Within the Qwen branch, the **large tier (1.7B) is requested automatically** —
the call sites pass `true` as the `prefer_large_asr` argument to
`resolve_gpu_plan`, and the VRAM clamp inside that function decides whether the
1.7B fits alongside whatever else is placed on GPU; if not, `effective_prefer_large`
comes back `false` and the 0.6B is used instead. `GpuAcceleration::On` applies the
same clamp. There is no user-facing "prefer large model" toggle: the large tier is
always requested and the VRAM clamp is the sole decider, so no setting can force a
1.7B that would not fit on the available GPU. The orchestrator
resolves the engine once at recording start (and at re-transcribe) in
`runner::build_asr_backend`, mirroring how it already resolves the language hint
and GPU layers. `model-registry` only fetches the model(s) for the selected
engine; pulling all three is opt-in (disk).

## Notes paragraph-anchor clock

Phase 3 binding rule (stress-test correction A4). Notes paragraph anchors
(`data-anchor-ms` on each paragraph, first-keystroke-per-paragraph while
recording) MUST be stamped from the capture-sample, pause-**excluding**
recording clock — the same timeline as `Segment::start_ms`. That value is
exposed to the webview as `AppEvent::RecordingClock { meeting_id, clock_ms }`,
emitted throttled (~5 Hz) from the orchestrator runner loop.

Do **not** derive anchors from `Date.now() - started_at_ms`: that wall-clock
delta is pause-*including* and drifts from the audio/transcript timeline, so
Phase 4 cross-reference (FR-22/23, anchor → nearest transcript segment) would
resolve to the wrong region. `started_at_ms` remains valid for elapsed-time
*display* only.

**Gutter display is a separate, display-only wall-clock.** `data-anchor-ms` (the
cross-reference / summariser timeline) is never shown raw. The notes gutter shows
the **local time-of-day** the note was written: a paired `data-anchor-wall`
attribute (epoch ms) is stamped ALONGSIDE `data-anchor-ms` at anchor time — so it
is correct across pauses, unlike a naive `started_at_ms + offset` conversion —
and `AnchorMarginalia` renders it via `formatWallClock`. Notes predating the
stored wall-clock fall back to deriving a time-of-day from the meeting start +
the offset (pause-naive), then to the bare elapsed offset if no start time is
known. `data-anchor-wall` is presentation-only — it never feeds cross-reference
(FR-22/23) or the summariser, which stay on `data-anchor-ms` — and round-trips
like any paragraph attribute (notes.json opacity + the generic ydoc attr walk),
so no persistence change is needed.

Consequence: `audio.opus` is recorded pause-*including* (the encoder pads each
pause with synthesised silence), while anchors and segment timestamps are
pause-*excluding*. Phase 4 cross-reference (FR-22/23) operates **entirely on the
pause-excluding timeline** (`data-anchor-ms` ↔ `Segment::start_ms`), so it needs
no conversion. The **summariser** relies on the same coincidence (#70): it
merges anchored note paragraphs with transcript segments by comparing
`data-anchor-ms` directly against `Segment::start_ms`, no conversion, to weave
each note in at the time it was written (see `components.md` — `summariser`). Audio-file *seek-to-anchor* (playing the audio at a clicked
anchor) is the only feature that must bridge the two timelines — it needs a
pause-offset map (a list of pause intervals) — and it was **deferred out of
Phase 4** (no audio player shipped this phase). Whatever phase adds audio
playback owns the pause-offset map.

**Offline reprocessing must reproduce the pause-excluding timeline.** Because
`audio.opus` is pause-*including* but `Segment::start_ms` is pause-*excluding*,
`re_transcribe` (which decodes `audio.opus`) MUST reconstruct the pause-excluding
clock or every post-pause segment would be inflated by the pause durations
(breaking the FR-22/23 cross-reference + the diarizer overlay that re-derive from
those timestamps). Since no pause-interval map is persisted yet (see the deferred
seek-to-anchor note above), `re_transcribe` reconstructs it heuristically: it
treats a run of ≥ 4 s of near-silent (`|x| ≤ 0.02`) decoded samples — comfortably
above the live accumulator's 3 s `MAX_GAP_MS` cap — as encoder pause padding and
excludes it from the timeline (the offline clock advances only over kept audio,
exactly as the live capture clock froze during the pause). Decode also trims the
`OpusHead` pre-skip so decoded sample 0 == recorded sample 0. The
`orchestrator/tests/timeline_coherence.rs` test asserts a paused meeting's
re-transcribed post-pause segment lands on the pause-excluding clock (not inflated
by the pause). Limitation: a ≥ 4 s run of genuinely-silent *input* would be
misclassified; a persisted pause-interval map (a `common`/schema change) would
make this exact rather than heuristic — tracked for a later phase.

**Re-diarize re-ASR split must stay on a single clock (#0015 phase 4).** The
offline re-diarize pass re-ASRs each kept mixed Qwen segment into single-speaker
sub-clips at its speaker-change boundaries. The two clocks must never be
compared directly: `SherpaDiarizer::compute_turns` runs over the pause-INCLUDING
PCM, so `SpeakerTurn` ms are pause-INCLUDING, whereas `Segment::start_ms` is
pause-EXCLUDING. The split therefore (1) maps the segment's pause-EXCLUDING
`[start_ms,end_ms)` to a pause-INCLUDING PCM range via
`runner::pcm_window_for_excluding_range`, (2) takes
`diarizer::turn_boundaries_within` cuts on the SAME pause-INCLUDING clock the
turns + PCM share, energy-snaps each cut (`runner::snap_to_energy_min`), slices
the PCM, and re-ASRs each sub-clip, then (3) stamps each sub-clip's `start_ms`
back onto the pause-EXCLUDING transcript clock via the inverse
`runner::excluding_ms_for_pcm_sample` (the forward map is one-way). Without the
inverse, a post-pause sub-clip would inherit the cumulative pre-segment pause
padding and drift forward by every pause before it — breaking the FR-22/23
cross-reference and the notes-anchor join the same way an unconverted
re-transcribe would. The `≥4 s-pause` clock-regression unit test
(`runner::tests::inverse_round_trips_across_a_long_pause`) guards the
forward↔inverse round-trip.

**Offline ops are serialized — but a new recording preempts them.**
`re_transcribe`, `rediarize`, and the merged `reprocess` atomically CLAIM an
internal `Offline` state under the orchestrator lock (rejecting a concurrent
offline op with `AppError::InvalidInput`) and release it on every exit path, so
two offline ops can't race and clobber the SAME meeting's `transcript.json`.

**`reprocess` (#0015 phase 5) takes ONE claim for a re-transcribe + diarize
pass.** It merges `re_transcribe` and `rediarize` into a single offline op under
ONE `claim_offline`/`release_offline`, never re-claiming between the two
sub-steps: it drives their CLAIMED bodies (the post-claim logic, factored out so
the standalone commands and `reprocess` share it) so NO `Idle` window opens
mid-pass. The internal order is **re-transcribe FIRST, then diarize/split/merge
over the fresh transcript, then finalise ONCE** with `finalise_diarization`
semantics (write transcript + `speaker_count` + diarizer descriptor +
`speaker_names` clear). Order matters: a diarize-first order is a guaranteed
lost-update — the re-transcribe finalise's `write_transcript` would clobber the
just-written split. The fresh transcript is persisted between the steps because
the diarize funnel (`run_diarization_blocking`) re-reads `transcript.json` from
disk; that intermediate write is overwritten by the single diarize finalise, so
the re-transcribe is never separately finalised. The timeout budget is NOT
collapsed into one watchdog: each sub-step keeps its own
`retranscribe_timeout(duration_ms)` bound (the ASR run, and the diarize+split
run), and because the two run serially the budgets compose additively — neither
blocking pass is cut off mid-run. `reprocess` does NOT summarise (parity with
the standalone ops; `Summarise` stays a separate post-stop pass gated by
`recorder_is_live`).

However, **`start` PREEMPTS the `Offline` claim** (`transition_start` accepts
`Idle | Offline`): a new recording is a different `meeting_id`/file, so the
clobber hazard does not apply, and the user must never be blocked from recording the next
meeting while the previous one's best-effort repair runs. On preempt the
in-flight op finishes on its own thread (writing the OLD meeting's files —
harmless) and its release is a no-op (`transition_offline_release` returns
`false` and leaves the live `Recording` state intact, suppressing the stray
`Idle` broadcast). `Offline` reports the public **`Idle`** state (NOT
`Finalising`) precisely so the transport leaves Start enabled during the repair;
the repair's progress surfaces per-meeting on the meeting-list ROW via
`OperationProgress`, never as a transport busy-state. The genuine
`Stopping`/`Finalising` drain (capture teardown + transcript/metadata write) is
NOT preemptible — it must complete before any start.

## llama.cpp prefill batching

Phase 0 Spike 2 found that `cparams.n_batch` is a **per-decode hard
limit**, not just an allocation hint. Feeding a prompt longer than
`n_batch` tokens in a single `LlamaBatch` trips
`GGML_ASSERT(n_tokens_all <= cparams.n_batch)` and aborts. The fix is
to chunk the prompt into `n_batch`-sized batches and call `decode` once
per chunk; only the last token of the last chunk needs `logits = true`.

Binding on `summariser`: long transcript + notes prompts will exceed
`n_batch` regularly (default is 512 tokens; a 30-minute transcript can
easily reach 8000+ tokens). The summariser MUST implement
chunked-prefill.

`asr-runtime` is not affected: `mtmd_helper_eval_chunks` performs the
chunking internally for audio-bearing prompts.

## llama.cpp build + version policy

Both `asr-runtime` (mtmd audio) and `summariser` (text) drive llama.cpp
through `llama-cpp-2` (workspace pin `=0.1.146`, `features = ["mtmd"]`).
The native library is **built from source** by `llama-cpp-sys-2` from its
vendored llama.cpp submodule (hence `LIBCLANG_PATH` is required for bindgen
on every clean build, on every platform) — there is no system-lib link.

- **Pin policy.** `llama-cpp-2`/`-sys-2` are pinned with `=EXACT` and bumped
  **deliberately**, never floated — the crate does not follow semver
  meaningfully, so each bump is a separately-verified change. As of 2026-06,
  `=0.1.146` (published 2026-04-30) is the latest published release and
  vendors llama.cpp build b8783 (commit `e21cdc11`, 2026-04-13), which
  already includes the April-2026 audio wave (Qwen3-ASR / Qwen3-Omni,
  Gemma 4 audio). There is no version lag.
- **Going past the latest crate requires a fork.** `llama-cpp-sys-2` has no
  `LLAMA_CPP_SRC`/`PATH` override; to ride a newer llama.cpp than the latest
  crate you must fork `llama-cpp-rs`, bump the submodule, regenerate bindings,
  and reconcile FFI drift (it compiles internal `common_chat_*` C++ with no
  stability contract), wired via `[patch.crates-io]`. Reserve this for a
  specific load-bearing upstream fix; it is not warranted now.
- **Bump/fork verification gotchas (Phase 7 + any future bump).** Re-run the
  gated ASR WER + early-stop tests and the orchestrator pipeline test after
  any crate/submodule change (canary for binding/model drift). Known traps
  past b8783: MSVC LTO break (#22186, after commit 6990e2f → set
  `-DGGML_LTO=OFF` + `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`); Vulkan
  `shaderc ≥ 2025.2` bf16 false-positive (#15344 → pin `shaderc 2024.0`);
  macOS Metal needs `GGML_METAL_EMBED_LIBRARY=ON` to find `default.metallib`
  inside the Tauri `.app` bundle.

## Model lifecycle

Owned by `model-registry`. The contract:

- A model is identified by a stable `ModelId` (e.g.
  `qwen3-asr-0.6b-q8_0`).
- `model-registry::ensure(model_id)` resolves to a local path; downloads
  if absent; verifies hash.
- **First-run provisioning.** The onboarding wizard's model step offers BOTH
  the ASR model and the summarisation LLM (`gemma-4-e4b-it-q4_k_m`) up front,
  each as a per-model `ModelDownloadCard` (progress / retry / ready). Previously
  only the ASR model was provisioned and the LLM lazy-downloaded silently on the
  first summarise (multi-GB, no progress — read as a broken button). Downloads
  remain skippable and continue in the background if the user proceeds. If the
  LLM was skipped, the Summarise action's in-progress UI distinguishes the
  one-time **model-download phase** (with %) from actual summarisation, so the
  multi-GB wait is not mislabelled "Summarising…". A downloaded-but-not-yet-
  loaded LLM still costs an mmap + warmup on the first summarise; #69 surfaces
  THAT as the indeterminate **"Loading the summarisation model…"** phase of the
  summarise progress bar (see "Operation progress"), distinct from the download.
- **Progress UX.** `ensureModel` enters the downloading state optimistically on
  click (seeded at the model's known partial fraction) because resuming a large
  partial spends seconds re-hashing before the first progress event — a lingering
  button reads as a no-op. Relatedly, `ensure`'s validity check skips hashing an
  absent/wrong-size file (size pre-check) rather than reading a multi-GB partial
  in full only to fail.
- Loaded models are owned by the consuming crate (`asr-runtime`,
  `summariser`, `diarizer`). The registry hands out paths, not loaded
  models — we don't want a model cache that two crates can hold
  references into.
- The registry is also an **event source**: it holds the orchestrator's
  shared `broadcast::Sender<AppEvent>` and emits
  `AppEvent::ModelDownloadProgress` (≈10 Hz) during `ensure`, so the
  first-run download UI updates live. See `components.md` —
  `model-registry` "Event source".
- On settings change to model selection, the consuming crate is
  responsible for tearing down its loaded model and reloading. The
  orchestrator coordinates this — there is no recording during a swap.

**Exception: Silero VAD.** The Silero VAD ONNX file (~1.8 MB) lives in
the source tree under `resources/silero/` and is **not** managed by
`model-registry`. It **is bundled as a Tauri resource**: `tauri.conf.json`
`bundle.resources` ships `"../resources/silero/silero_vad_v4.onnx"`, which the
bundler places under the package resource dir at
`_up_/resources/silero/silero_vad_v4.onnx` (parent-dir traversal mangled to
`_up_` by `tauri-utils::resources::resource_relpath`).

Path plumbing: `app-main` resolves the bundled resource at startup (early in
`setup()`, before the orchestrator is constructed and before any recording) via
`app.path().resolve("../resources/silero/silero_vad_v4.onnx",
BaseDirectory::Resource)` — `PathResolver::resolve` applies the same `_up_`
mangling to its input, so resolving the config-relative pattern yields the
placed file. If it resolves to an existing file, app-main exports its absolute
path as the **runtime** env var `MINUTIST_SILERO_PATH`; otherwise (a dev run
with no bundle) it leaves the var unset.

`vad-chunker::default_model_path()` reads, in order: (1) the **runtime**
`MINUTIST_SILERO_PATH` (app-main's injected bundled path); (2) the
**build-time** `MINUTIST_SILERO_PATH` (`option_env!`); (3) a source-tree path
relative to `CARGO_MANIFEST_DIR` (so `cargo run` / `cargo test -p vad-chunker`
work with no env var set). This keeps dev and test runs unchanged while letting
an installed package find the bundled model.

The rationale for keeping it outside the registry: Silero is
small enough that downloading it on first run adds friction without
value; it never changes per-user; and a single-file source asset avoids
forcing every phase that uses VAD to also pull in `model-registry`. This
is the only model file that bypasses the registry.

## ASR prewarm (live-test UX)

The FIRST record lazy-loads the routed ASR backend (the cold Parakeet/Qwen model
load is ~29 s) on the ASR-worker thread's first flush, which looks dead and makes
the live transcript fall behind (forcing the post-stop re-transcribe). To remove
that cold path the orchestrator exposes `prewarm_asr()`: it resolves the engine
from `settings.transcription_language` (+ the GPU-model opt-in) exactly as
`start()` does, builds the backend on `spawn_blocking`, and holds the
`(engine, backend)` pair in a process cache (`Mutex<Option<(AsrEngine, Box<dyn
AsrBackend + Send>)>>`). The first `start()` whose engine matches **takes** the
cached backend and hands it to the runner as the prebuilt backend; a mismatch
(the user changed the language) or an empty cache falls back to the existing lazy
worker-init path, which is never regressed. Prewarm is **idempotent** and
**non-blocking-at-start** (no download — a not-yet-downloaded model warms
nothing) and **best-effort** (a build failure is logged and swallowed). It is
triggered twice for redundancy: once from `app-main`'s `setup` (via
`tauri::async_runtime::spawn`) after the event bus is up, and once from the
webview (`prewarm_asr` command) when the recording/meeting workspace opens. While
the first start is in flight the webview shows a "Preparing transcription model…"
status and disables the Start control (a double-press is then impossible) — see
"Operation progress" below.

**Summariser preload (same shape, for the LLM).** The summary/chat LLM is the
other heavy lazy load (mmap + warmup on the first Summarise / chat). Gated on
`settings.preload_summariser` (default ON), `app-main` warms it on a background
startup task via `ChatHandles::maybe_preload_summariser`, mirroring `prewarm_asr`:
it checks the model is already downloaded (`Orchestrator::list_models`, NEVER
downloading) and, if so, calls `ensure_summariser` to load the shared held
instance early; otherwise it skips (the model loads on first use). The held
`OnceCell` keeps the instance resident for the process lifetime, so once loaded
(preloaded or on-demand) it stays ready — the `preload_summariser` toggle only
chooses startup-warm vs load-on-demand, there is no idle unload.

## Operation progress

Long-running per-meeting operations emit `AppEvent::OperationProgress {
meeting_id, op: OperationKind, fraction: Option<f32>, label: String }` (rides the
existing `AppEventPayload` newtype + the single `collect_events![AppEventPayload]`
registration — no second registration). The webview renders a NON-BLOCKING
per-meeting-row indicator: a determinate bar when `fraction` is `Some` (0..=1), an
indeterminate spinner when `None`. The terminal event for the op clears the
indicator. Producers + determinism:

- **`ReTranscribe` (determinate)** — `orchestrator::runner::re_transcribe_buffer`
  emits per accumulator flush, `fraction = kept-samples-fed / total-kept-samples`
  (a pure `re_transcribe_fraction`, unit-tested). Cleared by `TranscriptReady`.
- **`Summarise` (two-phase determinate + indeterminate lead-in, #69)** —
  `ipc-bridge::run_held_summarise` drives the concrete
  `LlamaSummariser::summarise_with_progress`, which now reports a phased
  `SummariseProgress`. The bar progresses through up to four labelled phases so
  the user is never staring at a silent 0% on a long meeting: (1) indeterminate
  **"Loading the summarisation model…"** around `ensure_summariser` (the
  multi-GB GGUF mmap + warmup, paid on the first summarise of a session —
  including the post-stop auto-summarise); (2) indeterminate **"Preparing the
  model…"** for the `LlamaContext` build (cold-GPU shader compile) before the
  first prefill tick; (3) determinate **"Reading the meeting…"** `Prefill { done,
  total }` as the transcript+notes prompt decodes chunk by chunk; (4) determinate
  **"Writing the summary…"** `Generate { done, max }` per token. The callback is
  throttled to ~5 Hz but always emits on a phase change and at completion. (The
  `common::Summariser::summarise` signature has been widened twice: #70 changed
  `notes_markdown: &str` → `notes: &[NoteBlock]`; the Attachments WS added
  `attachments_markdown: &str` before `system_prompt`. The current four-argument
  signature is `fn summarise(&self, transcript: &[Segment], notes: &[NoteBlock],
  attachments_markdown: &str, system_prompt: &str) -> AppResult<String>`. The
  *progress* method stays concrete on `LlamaSummariser`, which `ipc-bridge` holds.)
  Cleared by `SummaryReady`.
- **`Rediarize` (indeterminate)** — the sherpa diarization `compute` is one opaque
  FFI call with no progress callback, so `fraction = None`. Cleared by
  `DiarizationComplete`.
- **`Finalise` (indeterminate)** — the post-stop drain is opaque, `fraction =
  None`. Cleared by `MeetingFinalised`.
- **`Translate` (determinate)** — `ipc-bridge::translate_meeting` emits per
  segment, `fraction = segments_done / total_segments`, throttled to ~5 Hz (plus
  always on the last segment). Cleared by `TranslationReady`, which is emitted on
  **every exit path** (success and error) so the indicator is never orphaned by a
  mid-segment failure. A partial-failure exit leaves completed segments on disk;
  the UI refetch on the terminal event surfaces them.

## Finalise returns to the meeting list (live-test UX)

On stop the meeting is finalised + index-upserted immediately (the orchestrator
emits `MeetingFinalised` and returns to `Idle` the instant the recording is on
disk); the heavy background passes run AFTER. They run in order in one
fire-and-forget task: (1) re-transcribe if the live transcript fell behind, then
(2) re-diarize — both under the `Offline` claim (which reports the public
**`Idle`** state, so Start stays enabled — see "Offline ops are serialized") —
then (3) **auto-summarise** (#68), gated on `settings.auto_summarise_on_stop`
(default ON; serde-default so an older store adopts it). The auto-summarise step
runs LAST so it summarises the FINAL transcript (after any re-transcribe /
re-diarize), drives the SAME held-summariser path as the user-triggered
`summarise_meeting` (`run_held_summarise`), and emits the determinate
`OperationProgress { op: Summarise }` + `SummaryReady`. It does NOT claim the
offline slot (it reads, never rewrites `transcript.json`); errors are best-effort
(logged — the meeting is left without a summary, recoverable via the Summarise
action). **A new recording preempts the chain**: re-transcribe / re-diarize
self-skip once a recording is live (their fresh `Offline` claim then fails), and
auto-summarise — which takes no claim — checks `recorder_is_live()` and defers
(the manual Summarise action is the recovery), so the previous meeting's repair
never contends with the new live recording's GPU use. The recording window
must **not** stay open for any of these: the webview returns to the home
meeting-list as soon as the recorder leaves the live states
(`recording`/`paused`/`stopping`) — it does NOT gate the window-close on the
offline claim releasing (`Idle`). The background passes surface only as the
non-blocking per-row "Operation progress" indicator above, which the meeting-list
store refreshes on the terminal `MeetingFinalised` / `TranscriptReady` /
`DiarizationComplete` / `SummaryReady` events.

**Auto-summary busy lifecycle (summary pane).** The just-stopped meeting opens
on its finished screen immediately (`MeetingFinalised`), so the user is looking
at the summary pane *before* the determinate `OperationProgress { op: Summarise }`
stream begins — which can be minutes later, since auto-summarise runs LAST, after
any re-transcribe / re-diarize. To stop the pane offering a manual "Summarise"
button (a redundant, racing second run) during that gap, the auto-summary has its
own lifecycle event pair, distinct from the single-slot operation-progress bus
(whose slot the reprocess pass owns while it runs):

- `stop_recording` emits **`AppEvent::SummaryQueued { meeting_id }`** the instant
  it plans an auto-summary (only when `auto_summarise_on_stop` is on), *before*
  spawning the background passes. The webview marks the meeting busy
  (`summary` store `autoPending`) and shows a progress state for the whole queued
  → summarising window. While a reprocess op is in flight the pane names that
  phase ("Finishing the transcript, then summarising…"); once the `summarise` op
  streams, the determinate `OperationIndicator` takes over (read from the
  operation-progress store, keyed on `meeting_id` + `op == Summarise`, so the bar
  shows even though the pane never dispatched the summarise).
- The terminal is **`SummaryReady`** (a summary was written — reveals it) OR
  **`AppEvent::SummaryUnavailable { meeting_id }`**, emitted by the Summarise pass
  when it is deferred (a new recording claimed the model) or fails. Without the
  latter a deferred/failed auto-summary would leave the pane spinning forever;
  it clears `autoPending` so the pane falls back to the manual Summarise action.
  `SummaryUnavailable` also clears the per-row operation-progress indicator (a
  failed `run_held_summarise` may leave a stale `summarise` op).

## Agent chat loop (Phase 9)

The built-in chat agent (`chat-agent` crate, driven by `ipc-bridge`) runs a
multi-turn, tool-calling loop over the bundled LLM. The decided cross-cutting
rules (the engine itself lands in the Phase 9 implementation streams; these
constraints are binding on them):

- **Held model, fresh context per turn.** The `LlamaModel` is loaded once and
  held (an `Arc<dyn Summariser>`/substrate owned by `ipc-bridge`/`app-main`,
  shared with the one-shot summary path) rather than reloaded per call. Each
  assistant turn allocates a fresh `LlamaContext` (clean KV cache); the engine is
  stateless — the driver owns the conversation history and the sliding window.
- **Token streaming is a lossy hint; `final_text` is authoritative.** Generation
  streams `AppEvent::ChatToken { session_id, turn_id, token }` over the shared
  broadcast bus. The bus is lossy (slow subscribers get `Lagged`), so the webview
  MUST treat tokens as a progressive hint only and reconcile against
  `AppEvent::ChatTurnComplete { …, final_text }`, which carries the full reply
  text for the turn. `ChatToolCall` / `ChatToolResult` are emitted around each
  tool dispatch; `ChatError` terminates a turn.
- **Tool calling uses llama-cpp-2's OpenAI-compatible path.** Prompt rendering is
  `apply_chat_template_with_tools_oaicompat` (the GGUF's own tool template);
  tool-call extraction is the streaming `ChatParseStateOaicompat` parser; a
  lazy GBNF grammar (`json_schema_to_grammar` + `LlamaSampler::grammar_lazy`)
  from each tool's input schema is the reliability backstop for the small model.
  A max-tool-iteration cap bounds the loop; malformed tool calls are recovered by
  re-prompting, not by crashing the turn.
- **`Summariser: Send + Sync`.** The held handle crosses threads and is referenced
  concurrently by the summary path and the chat `resummarise` tool, so the trait
  is `Send + Sync` (SP0-verified; see `components.md` — `common`).
- **`speaker_names` and re-diarization.** `MeetingMeta.speaker_names` maps a
  diarizer label (`A`/`B`/…) to a user-set display name. Because re-diarization
  re-clusters and can re-letter speakers, a `rediarize` pass CLEARS `speaker_names`
  (the old label→name mapping is no longer valid); the `set_speaker_name` tool
  re-establishes names afterward. Names are an overlay applied at read time, never
  baked into `transcript.json`. The merged `reprocess` (#0015 phase 5) ALWAYS
  diarizes. When `voiceprint_enrolment_enabled` is ON, a `reprocess` re-identifies
  known speakers from your library: enrolled speakers keep their names across a
  re-letter (via global gallery centroid matching, WU5), and unenrolled speakers
  who were named in this meeting are re-identified via ephemeral centroids and a
  timeline-coherence fallback (WU4 — see below). Where neither the library nor the
  ephemeral centroid matches, the user-typed name is still lost — the accept-and-warn
  default is only partially lifted for unenrolled strangers. When the flag is OFF,
  `speaker_names` is cleared unconditionally on every run (zero behaviour change for
  users who have never opted in).

- **The driver loop (`ipc-bridge`, on `spawn_blocking`).** `chat-agent`'s
  `ChatEngine` is stateless per call; the driver owns the conversation history and
  the turn loop. The loop is a State-free generic helper
  (`ipc_bridge::chat::run_chat_turn`, generic over the engine + a tool-dispatch
  closure + an emit closure) so the default test suite drives a full turn — a
  final-only turn, a tool-call-then-final turn, a multi-tool turn (history-shape
  assertion), the max-iteration cap, a cancelled turn, and a hard-floor context
  overflow — with a stub engine and stub tools, no model and no Tauri runtime. Per
  iteration: apply `chat_agent::trim_to_budget` (hard floor → reject `InvalidInput
  { "message too large for context window" }`), run one engine turn streaming
  `ChatToken`s, then either return the `Final` text (emit `ChatTurnComplete`) or
  dispatch each requested tool and loop. A `MAX_TOOL_ITERATIONS` cap bounds the
  loop: once hit, the engine is re-invoked with NO tools to force a final answer; a
  turn that still cannot finish emits `ChatError`. Tool dispatch re-enters async
  via a captured `Handle::block_on(registry.dispatch(...))` for the dispatch step
  only (§4.5 — the one async/sync crossing).

- **Assistant-`tool_calls` message in history (binding, CQ1).** The OpenAI tool
  protocol the GGUF tool template renders is `assistant(tool_calls) →
  tool(result)*`: a `tool` message MUST be preceded by the assistant message that
  bears the matching `tool_calls` array. When a turn requests tools, the driver
  therefore appends ONE assistant message carrying ALL the requested calls
  (`chat_agent::ChatMessage::assistant_tool_calls`) BEFORE the per-call
  `tool_result` messages — never a bare `tool` after `[system, user]`, which the
  template either hard-errors on or silently degrades. The engine `ChatMessage`
  carries `tool_calls: Vec<ToolCall>` for this; `backend::messages_json`
  serialises it as the OpenAI `tool_calls` array (with `content: null`). The
  carrier is persisted on the wire `common::ChatMessage` (`tool_calls:
  Vec<ToolCallRecord>` on the assistant message, `tool_call_id` on the tool
  message) so a reloaded multi-tool turn reconstructs the same valid sequence.

- **Turn cancellation (binding, P1).** Each turn runs against a
  `chat_agent::CancelFlag` (`Arc<AtomicBool>`). `send_chat_message` registers one
  per session in `IpcState::chat_cancel`; the `cancel_chat_turn(session_id)`
  command raises it. The engine's real decode loop checks the flag BETWEEN decoded
  tokens and, when raised, stops and returns `TurnOutcome::Cancelled { partial }`.
  The driver ends the turn with a terminal `ChatTurnComplete` carrying the partial
  text (cancellation is a user action, not a `ChatError`), clears the in-flight
  guard + the cancel-flag entry, and persists the session. The inter-agent (MCP)
  path drives a fresh never-raised flag (no user cancel surface).

- **Group-boundary eviction (binding, CQ2/P2).** `chat_agent::trim_to_budget` is a
  pure planner that returns the MINIMUM messages to drop after the pinned system
  head. The driver (which owns the message roles) SNAPS that count forward to the
  next user-message boundary before draining, so the survivor at `history[1]` is a
  `User` turn — never an orphan `assistant`/`tool` lead (which, with the CQ1
  assistant-`tool_calls` rule, would be a malformed sequence). On any eviction the
  driver emits `AppEvent::ChatContextTrimmed { session_id, dropped_turns }`; the
  webview shows a quiet "history trimmed" affordance.

- **Per-turn seed (binding).** `chat_agent::SamplerConfig`'s default `seed` is `0`,
  which is FIXED/reproducible — every non-greedy reply would be verbatim-identical.
  The driver therefore injects a per-turn **non-zero** seed (derived from
  wall-clock nanos + a process-wide nonce + the turn id) before each non-greedy
  `run_turn`. The deterministic (greedy, `temperature == 0.0`) profile leaves the
  seed untouched — it is ignored on the greedy path and the test suite relies on
  greedy reproducibility.

- **Chat persistence (`persistence::ChatStore`).** The driver persists the session
  through `ChatStore` at **turn end** (re-loading the on-disk session first so a
  concurrent edit is not clobbered, then appending the turn's produced messages):
  `{meetings_dir}/{meeting_id}/chat/{session_id}.json`, atomic tmp+rename,
  `persistence` the sole writer. A meeting-less session is not persisted (the
  events already delivered the reply). `delete_meeting` removes the meeting folder,
  so chat sessions go with it.

- **Held-model lifecycle (C2).** The LLM GGUF is loaded **once**, lazily, on first
  chat/summarise use into `IpcState::summariser`
  (`Arc<OnceCell<Arc<LlamaSummariser>>>`), and shared by both the chat engine (which
  borrows `&LlamaModel`) and the one-shot `summarise_meeting` path (refactored from
  its prior per-call GGUF load). GPU placement is resolved **at load time** from
  the VRAM-aware `GpuPlan` (`plan.summariser_gpu`; see "GPU portability");
  toggling the setting takes effect on the next process start. Each turn still
  allocates a fresh `LlamaContext` (clean KV cache).
  A single in-flight turn per session is enforced via
  `IpcState::chat_in_flight: Arc<Mutex<HashSet<ChatSessionId>>>`.

## Live in-meeting agent (auto-driver)

The live in-meeting agent (Phase 9 / WU2b) runs a digest-refresh loop during
an active recording, driven by incoming `TranscriptSegment` events and gated by
the `live_agent_min_segments` / `live_agent_min_seconds` cadence settings.
Meeting attachments are pinned in the `LlamaContext` prefix once at recording
start. Cross-cutting rules:

- **HELD context, not fresh per turn.** The Phase-9 chat loop ("Agent chat
  loop" above) allocates a **fresh** `LlamaContext` per assistant turn. The live
  agent holds **one** `LlamaContext` for the entire live session on a dedicated
  single-owned thread (SP-LIVE E2). The attachment prefix is prefilled ONCE at
  recording start (~40 s for a moderately sized slide deck) and the context is
  extended with each digest refresh by appending only the incremental transcript
  tail. The fresh-per-turn pattern would re-pay the ~40 s prefill cost on every
  cadence tick, making live operation unusable. `LlamaContext` is `!Send`; the
  dedicated thread owns it exclusively.

  *Implementation (S2a — `chat-agent::live`).* The held-context loop is
  implemented in `crates/chat-agent/src/live.rs` as `LiveSessionBackend` (the
  testable seam) + `LlamaLiveBackend` (the real impl, `!Send`, borrows
  `&LlamaModel` from the shared `summariser` substrate) + `LiveSession<B>`
  (the driver enforcing prefix-once and tail-only discipline).

  *Implementation (S2b — `ipc-bridge::live_agent`).* The auto-driver is in
  `crates/ipc-bridge/src/live_agent.rs`. It owns:
  - A `tauri::async_runtime` task (the async driver) that subscribes to
    `TranscriptSegment` events, accumulates the tail buffer, and evaluates the
    cadence gate via the pure `should_refresh(new_segments, elapsed_secs,
    in_flight, min_segments, min_seconds) -> bool` function.
  - A dedicated `std::thread` (the worker) that constructs
    `LiveSession<LlamaLiveBackend>` on startup — the `!Send` held-context
    session. The worker borrows `&LlamaModel` from the shared
    `Arc<LlamaSummariser>` (same cell as chat/summarise), using a raw-pointer
    lifetime extension that is safe because the Arc is declared before the
    session on the same stack frame (reverse-declaration drop order guarantees
    the Arc outlives the borrow). The test-only stub `WorkerBackend`
    (`#[cfg(test)]`) exercises the driver protocol without a model.
  - Two bounded `tokio::sync::mpsc` channels (depth 1 each) between the driver
    and worker, enforcing single-in-flight without a separate mutex.
  - The prefix is built on the worker thread (`build_prefix`) at session start,
    then seeded via `seed_prefix_typed` BEFORE the request loop (pin-at-start).
    `build_prefix` calls `persistence::read_attachments_markdown_parts`
    (synchronous filesystem I/O) and must not run on the async driver task.
  - A `startup_cancel: CancelFlag` is created in `spawn_live_agent`, cloned for
    the driver task, and passed to the worker. The driver raises it on any
    shutdown path so a Stop during the ~40 s prefix seed aborts promptly and
    unblocks the driver's join on the worker thread.
  - The worker thread `JoinHandle` is retained by the async driver task and
    joined after the driver loop exits, ensuring the worker is reaped rather
    than leaked.
  The watcher task in `app-main` subscribes to `StateChanged` and calls
  `spawn_live_agent` on `Recording`; it raises the returned `watch::Sender`
  on `Idle` / `Stopping` / `Finalising` to tear down the driver.

  *Error handling policy.* Both `RefreshResult::Err` (decode error, M1/M2
  triggered) and `RefreshResult::CapacityExhausted` are terminal: the driver
  sets a `terminal` flag on receipt, emits one `LiveDigestError` event, and
  dispatches no further refreshes. The worker also stops after a terminal
  result. A single error path covers both cases, consistent with the teardown
  on decode failure (M3).

  *Context overflow policy (v1).* When `LlamaLiveBackend::refresh` returns
  `Error::ContextOverflow` the driver emits ONE `LiveDigestError` event
  (user-visible "context window filled") and sets a permanent
  `capacity_exhausted` flag that stops all further refresh dispatches for the
  session. Re-seeding mid-recording is NOT attempted (it costs another ~40 s
  prefill, starving ASR). The prior digest items are preserved in the event
  store. Recovery is the next recording session (fresh context).
  The driver calls `LiveSession::seed_prefix_typed` and `LiveSession::refresh_typed`
  (returning `Result<_, chat_agent::Error>`) rather than the `AppResult` wrappers,
  so `ContextOverflow` can be matched structurally. The `From<Error> for AppError`
  impl maps `ContextOverflow` to `AppError::InvalidInput`, erasing the variant —
  string-matching over `AppError::Display` would be fragile and is not used.

  *KV retention policy.* The held context accumulates **prefix + transcript tail
  only**. Generated digest-answer tokens are decoded ephemerally into the KV and
  then pruned via `clear_kv_cache_seq` after every refresh (on completion and
  on cancel). This keeps capacity growth proportional to the transcript and
  prevents cancelled partial answers from poisoning subsequent refreshes.
  `clear_kv_cache_seq` returns `Result<bool, KvCacheConversionError>`; a `false` or
  `Err` means the KV state is unrecoverable — the backend returns `Err` and the
  driver tears down the session.

  *Cancellability.* `prefill_prefix` and the tail-prefill loop in `refresh` both
  accept a `&CancelFlag` and check it between decoded chunks. A raised flag during
  the ~40 s prefix prefill prunes any partially-decoded KV range and returns
  `Error::Inference("cancelled")` so the driver can tear down promptly. Both
  the tail-prefill loop (inside `refresh`) and the generation loop already checked
  the flag; the prefix prefill was the missing path.

  *Tail-prefill transactional safety.* The tail-prefill loop in `refresh` captures
  `n_past` before the first batch. A mid-loop decode failure prunes the partial
  range (`clear_kv_cache_seq` to the pre-loop position) and returns `Err` without
  advancing `n_past`, leaving the context consistent. The driver tears down on any
  non-overflow `Err` (the held-context invariant is broken once a decode fails).

- **Cadence gate.** `should_refresh` is a **pure** function with no side effects.
  It returns `true` when ALL of: `new_segments >= min_segments`, `elapsed_secs >=
  min_seconds`, and `!in_flight`. The AND gate (not OR) prevents premature
  refreshes during sparse meetings with few utterances.

- **Standing-list update discipline.** Each refresh prompt includes the prior
  digest (JSON-serialised) so the model UPDATEs existing items (flips `resolved`,
  adds new items) rather than regenerating from scratch. `parse_digest` carries
  forward `resolved = true` from prior items matching by text (case-insensitive)
  even if the model emits `resolved: false` for them (model forgetfulness guard).
  For `open_asks` specifically, the driver accumulates items across refreshes:
  prior unresolved asks not mentioned by the model are carried forward (the model
  may omit them to save tokens), while items the model marks `resolved: true` are
  promoted to resolved and retained. This implements the "tracker maintained across
  refreshes" contract from SP-LIVE. Other categories (action_items, decisions,
  attachment_answers, unresolved_references) apply the base standing-list rule
  (resolved-flag-only carry-forward from matched items).

- **Pin-at-start constraint.** The ~40 s one-time attachment prefill MUST NOT run
  mid-recording (it would starve ASR inference and block the recording UI). The
  prefix is built on the worker thread at session spawn, before the first cadence
  fire. Subsequent `seed_prefix` calls on the same `LiveSession` are no-ops.

- **ASR(CPU) vs LLM(GPU): no contention (SP-LIVE E1 GO).** ASR (`asr-runtime` /
  `asr-parakeet`) runs on CPU via `sherpa-onnx` or llama.cpp CPU layers. The live
  agent (and the summariser) run on the GPU via Vulkan. These are distinct compute
  resources; there is no hardware contention between a live decode refresh and a
  concurrent ASR pass.

- **LLM-vs-LLM contention (v1 known hazard).** The live agent holds a dedicated
  `LlamaContext` (n_ctx = 32 768) but borrows the same `Arc<LlamaSummariser>` model
  (and therefore the same `LlamaModel`) that the chat and post-stop summarise paths
  use. A live digest decode concurrent with a user chat turn or a post-stop summary
  decode is a GPU-level contention: both share the single Vulkan device and the same
  loaded model weights. In v1 this is accepted without a guard because:
  (a) the cadence gate fires at most every `live_agent_min_seconds` (default 45 s);
  (b) post-stop summarise only starts after recording ends (at which point the live
  agent tears down); and (c) chat is user-initiated — simultaneous use is unlikely
  during an active meeting. The `LlamaContext`s are distinct (no shared KV state);
  llama.cpp serialises concurrent `decode` calls via internal locks.
  WU2b should add a coordination guard (e.g. the live agent skips a refresh while
  a foreground chat/summarise decode holds the model) if field testing reveals
  throughput degradation.

- **`LiveAgentMode::Auto` = GPU-acceleration-active gated.**
  `settings.live_agent_enabled` defaults to `Auto`.
  `live_agent_should_run(mode, probe, gpu_acceleration)` in `common` resolves it:
  `Auto` is `true` when the probe is `Some` AND `gpu_acceleration != Off`. This is a
  **GPU-acceleration-active proxy** — the LLM runs on the GPU rather than contending
  with the CPU-bound ASR path. It does NOT inspect `probe.is_integrated` (the AMD
  Radeon 890M, an integrated GPU running Vulkan, is the validated SP-LIVE E1 hardware
  and must resolve `true`). It does NOT consult `resolve_gpu_plan`'s VRAM-budget
  thresholds. `Off` disables unconditionally; `On` enables unconditionally.
  WU2b should refine this to a VRAM-headroom check once the live-context cost is measured.

- **KV quantisation: OFF.** q8_0 KV quantisation costs ~15 % decode throughput for
  memory savings the 36 GB test GPU does not need. Not applied to the live agent
  context. n_ctx = 32 768.

- **Digest panel is PASSIVE.** The live agent never writes to the transcript,
  notes, or metadata. It is a read/compute-only agent in v1. The digest panel
  receives `AppEvent::LiveDigestUpdated` events and updates passively; it does NOT
  interrupt the user or modify any meeting document.

- **Events ride the existing bus.** `LiveDigestUpdated` and `LiveDigestError` ride
  the existing `AppEventPayload` newtype + the single `collect_events![AppEventPayload]`
  registration — no new event registration. Both are lossy-broadcast-safe
  (`LiveDigestUpdated` carries the full replacement digest; a lagged subscriber
  recovers on the next refresh).

## MCP transport (Phase 10)

The `mcp-server` crate exposes the Phase-9 `agent-tools` registry to external
agents over an in-process **Streamable HTTP** MCP server (`rmcp` 1.7, MCP spec
revision 2025-11-25). Binding controls:

- **Single source of truth for tools.** A tool is defined in exactly one place —
  `agent-tools`. `mcp-server` projects `ToolRegistry::mcp_tool_descriptors_gated`
  onto `tools/list` and `ToolRegistry::dispatch` onto `tools/call`. Any tool
  logic / schema / name in `mcp-server` is a reviewer finding. The one rmcp-typed thing in `mcp-server` is the
  `AppError → McpError` mapping (real `AppError` variants only — there is no
  `ContextOverflow`; overflow + "recorder busy" surface as `InvalidInput`).

- **Settings-gated, off by default.** `settings.mcp_enabled` (default `false`)
  gates the listener. `app-main` watches the settings handle via
  `SettingsHandle::subscribe()` and **starts or stops the server live** when
  `mcp_enabled` flips — no restart required for the enable/disable toggle.
  A `McpShutdownState` (Tauri managed state, connected build only) holds the
  `watch::Sender<bool>` for the running server; the watcher fires it on
  disable and spawns a fresh server (with a new inter-agent driver and a new
  shutdown sender) on re-enable. `settings.mcp_port` is a FIXED default
  loopback port (8765, D1 — one instance runs, so a stable port keeps a
  saved client URL valid). Changing the port or `mcp_write_tools` IS
  restart-required (the running server was built with those values at startup).
  On stop, `AppEvent::McpServerStopped` is emitted so the Settings → MCP pane
  clears the endpoint display; on start, `AppEvent::McpServerListening` is
  emitted as before.

- **In-process, not a subprocess.** The listener shares the same
  `Arc<Orchestrator>` / `Arc<MeetingIndex>` / `meetings_dir` / held model / registry
  as the rest of the core, so a second process never opens `index.db` or the
  meeting folders — honouring the Filesystem single-writer rule below. Tool
  dispatch is the SAME async `ToolRegistry::dispatch` the chat loop uses.

- **Loopback + bearer + Host/Origin (the security model).** The server binds
  `127.0.0.1:{mcp_port}` only (never `0.0.0.0`). Every request must carry
  `Authorization: Bearer <token>` (a ≥256-bit CSPRNG token; a thin wrapper service
  returns 401 before rmcp sees the request — the `Mcp-Session-Id` is routing state
  only, never the credential). rmcp's `StreamableHttpServerConfig` enforces the
  `Host` allowlist (loopback default, `rmcp >= 1.4.0` — GHSA-89vp-x53w-74fx,
  DNS-rebinding) and the `Origin` allowlist (set to the loopback origins → 403 on
  a cross-origin browser request). Cautionary precedent: CVE-2025-49596 (MCP
  Inspector RCE) was a localhost MCP service with no auth + browser-reachable —
  exactly what bearer + Host/Origin + loopback prevent.

  **Token storage and file permissions.** The token is stored at
  `{app-data}/mcp_token`, CREATED with mode `0600` atomically on Unix
  (`OpenOptions().mode(0o600)` — no write-then-chmod window); the mode is
  re-asserted after open to cover any pre-existing file. On Windows the file
  inherits the per-user app-data directory's ACL. `app-main` does NOT
  additionally tighten the file ACL on Windows: the correct API is
  `SetNamedSecurityInfoW` (advapi32) with a DACL granting only the process
  owner's SID, reachable via `windows-sys >= 0.59`
  (`Win32_Security` + `Win32_Security_Authorization` features); but
  `windows-sys` is not a direct dep of `src-tauri` (it is only transitive
  through Tauri), so calling it requires adding a
  `[target.'cfg(windows)'.dependencies]` entry — deferred to a dedicated
  Windows-platform hardening commit. Until then the owner-only guarantee is
  Unix-scoped; the Windows per-user app-data directory is the operative
  control. OS-keychain migration (`keyring` crate) is a separate documented
  follow-up. The same writer (`write_secret_file`, formerly `write_token_file`)
  persists the WS4-A S5b **device credential** at `{app-data}/tunnel_device.json`
  with the identical 0600 discipline — and the identical Windows-ACL gap; the
  device credential (`mdc_<device_id>.<secret>`, the long-lived relay device
  identity returned once at pairing) is stored alongside its `account_id` /
  `device_id`, never logged, and never crosses to the webview.

  **Token lifetime and the connected-relay path.** The token is stable across
  restarts: `app-main` reads the existing file on start and reuses it so that
  a saved external MCP-client config stays valid. To rotate the token, delete
  the file and restart the app; the next start generates and persists a fresh
  256-bit token. In the paid connected tier the token doubles as the
  relay↔app shared secret (the hosted proxy authenticates the in-app endpoint
  with the same bearer); rotation therefore also invalidates any active relay
  session. The token is never logged and never on the event bus — it is
  revealed only via the `get_mcp_server_info` command on explicit user
  request.

- **The destructive-write exposure policy (a binding control).** The `Tool`
  trait's `expose_over_mcp()` (default `!is_write()`) is the server-side gate.
  `set_speaker_name` / `rename_meeting` are MCP-allowlisted (reversible, low blast
  radius); `retranscribe_meeting` / `rediarize_meeting` are internal-only (heavy;
  holding the offline claim via MCP would block the user's recording); the
  record-control tools (`start_recording` / `stop_recording` / `pause_recording`
  / `resume_recording`, #62) are MCP-allowlisted **write-gated** control tools —
  `is_write` AND `expose_over_mcp() == true`, so the recording lifecycle is
  driveable over MCP **only when the user turns `mcp_write_tools` ON** (off by
  default, behind the bearer token + loopback bind); this is the deliberate opt-in
  that lets an external client run the record→transcribe→read loop for E2E. No
  destructive tool (`delete_meeting`, notes mutation, summary overwrite) is in the
  v1 registry at all. ON TOP of that, `settings.mcp_write_tools` (D3, default
  `false`) gates the reversible writes: off ⇒ read/compute + the inter-agent tool
  only; on ⇒ the two reversible writes join. The gate is enforced at projection
  AND on call (`mcp_call_allowed`). NOTE: read-only ≠ zero-cost — even with writes
  off, an external agent holding the token can invoke COMPUTE tools
  (`relisten_section` runs ASR; `resummarise` runs the LLM) repeatedly. Each heavy
  compute tool is bounded by a per-call timeout so a single wedged/slow call
  cannot pin a blocking-pool thread + the model indefinitely: `transcribe_pcm_window`
  (relisten) takes a window-length-relative budget (mirroring `re_transcribe`'s
  length-relative timeout; floor 1 min / cap 5 min), and `resummarise` takes a
  fixed 5-min cap; a fired timeout returns `AppError::Inference` cleanly. The v1
  threat model trusts the bearer holder; a per-client rate/concurrency cap (a
  global semaphore across the heavy compute tools) is a documented follow-up.

- **The inter-agent bridge.** `send_to_internal_agent` (MCP-only, in the
  `v1(true)` registry) reaches the internal chat agent through a `common`-typed
  bounded channel on the `ToolContext` (the SENDER), whose receiver + the single
  chat turn live in `ipc-bridge::inter_agent` (driving the INTERNAL `v1(false)`
  registry so the agent cannot message itself). This keeps `mcp-server` free of a
  `chat-agent` edge and the single chat-turn site in `ipc-bridge`. v1 is a
  synchronous request/reply (bounded mpsc(16) `try_send` → "busy"; per-request
  timeout → "timed out"; single-in-flight-per-session → "session busy").
  **The bridge applies the same MCP write gate the direct `tools/call` path
  uses** (binding control): an external caller talking through the bridge must get
  NO broader a write surface than a direct MCP call under the active
  `settings.mcp_write_tools`. The `v1(false)` internal registry still INCLUDES the
  destructive `retranscribe_meeting` / `rediarize_meeting` ops (the internal UI
  agent uses them), so the bridge driver threads `allow_writes` from settings and
  bounds the turn to the gated surface — the engine is offered only
  `mcp_tool_descriptors_gated(allow_writes)` (the model never sees retranscribe/
  rediarize) AND the per-call dispatch rejects a non-allowed tool via
  `mcp_call_allowed` (defence in depth, mirroring `McpToolHandler::call_tool`).
  Both layers reuse the single gate policy in `agent-tools` — the bridge does not
  duplicate it.

- **Threading model row.** *MCP HTTP listener → tokio task spawned from `setup`
  via `tauri::async_runtime::spawn`; rmcp's own hyper-based `StreamableHttpService`
  serves the single `/mcp` endpoint (no `axum`); tool dispatch is the same async
  `ToolRegistry::dispatch` the chat loop uses.*

External MCP clients reach the loopback Streamable HTTP endpoint directly.
Account-based connectivity — a hosted proxy that fronts this endpoint, enabling
Claude web alongside Desktop, plus cross-device meeting sync and calendar
integration — is the planned direction and is out of scope here.

## Configuration

Single source: the `settings` crate, backed by a `serde_json` + `std::fs`
`JsonFileStore` at an injected path (`{app-data}/settings.store`). The crate
has **no `tauri::*` dependency**; `app-main` resolves the path and constructs
the store. (`tauri-plugin-store` is registered as a Tauri plugin in app-main
for the webview's own use, but it is not the settings crate's backing store.)
Other crates hold a `SettingsHandle` and read snapshots via it; nobody parses
the underlying JSON directly. See `components.md` — `settings`.

Settings changes broadcast directly from the `settings` crate via a tokio
`watch` channel (`SettingsHandle::subscribe`). Components that care subscribe.
The orchestrator is not a config bus — it consumes settings the same way
every other component does.

`SettingsHandle::current()` is the authoritative synchronous snapshot and MUST
reflect the latest `update()` whether or not any subscriber is alive — no
component is required to hold a `subscribe()` receiver for `current()` to be
correct (the orchestrator reads `current().diarization_enabled` /
`.input_device_id` directly, with no subscription). `update()` therefore
publishes the new value with `watch::Sender::send_replace`, **not** `send`:
`send` is a no-op that returns `Err` when there are no live receivers, which
would silently leave `current()` stale until the next process start. Persist
before publish: `store.save` runs first so a save failure never publishes a
change.

## Filesystem layout

```
{app-data}/                     platform default root (XDG_DATA_HOME on Linux,
│                               ~/Library/Application Support on macOS,
│                               %APPDATA% on Windows; identifier = ai.minutist)
├── settings.store              owned by `settings` (always at platform root)
├── logs/                       tracing file appender; owned by `app-main`
│                               (always at platform root — logging bootstraps
│                               before settings load)
├── mcp_token                   MCP bearer token (Phase 10); owned by `app-main`
├── tunnel_device.json          relay device credential (WS4-A S5b, connected
│                               build only); owned by `app-main`; 0600
│
│   The four entries below are placed at {app-data} by default.
│   When settings.data_directory is set to a valid absolute path they move
│   to {data_directory}/ instead (see "data_directory override" below).
│
├── index.db                    libsql; owned by `persistence`; derived,
│                               rebuildable cache (see "index.db is a derived
│                               rebuildable cache" below)
├── collections.json            JSON array; owned by `persistence`; durable
│                               collection definitions — NOT a rebuildable cache
├── voiceprints.db              libsql; owned by `persistence`; durable voiceprint
│                               library — NOT a rebuildable cache (see "Voiceprint
│                               matching" below and "index.db is a derived
│                               rebuildable cache" for the contrast)
├── meetings/{uuid}/            owned by `persistence` (and nobody else)
│   ├── audio.opus
│   ├── transcript.json
│   ├── notes.ydoc                 Yjs/yrs CRDT state (authoritative when present)
│   ├── notes.json                 ProseMirror JSON (derived from notes.ydoc)
│   ├── notes.md                   markdown (derived, best-effort)
│   ├── summary.md
│   ├── metadata.json
│   ├── assets/                 pasted/dropped note images (content-hash files)
│   │   └── <sha256>.<ext>
│   └── chat/{session_id}.json  chat sessions (Phase 9)
└── models/                     owned by `model-registry` (and nobody else)
    ├── asr/{model-id}/...      downloaded GGUF + mmproj per manifest entry
    ├── llm/{model-id}/...
    └── diarize/{model-id}/...
```

The model manifest is **not** written into the cache. It is bundled in the
binary (`resources/models.json`, loaded via `include_bytes!` in `app-main`
and parsed by `model_registry::load_manifest`); the cache dir holds only the
downloaded per-kind / per-model files.

Writes to a directory outside a component's owned scope are a review
finding.

**Notes write paths (binding).** Once `notes.ydoc` exists it is authoritative
and only `NotesStore::apply_update` may write it — that path MERGES the editor's
incremental Yjs update, preserving CRDT history. `NotesStore::save` rebuilds the
doc from whole-document JSON (minting a fresh client history) and is therefore
the **first-write-only** writer: it refuses with `AppError::InvalidInput` when a
`notes.ydoc` already exists. `apply_update` seeds a legacy `notes.json` (via
`seed_ydoc_if_needed`) before merging so a pre-CRDT meeting's content is not
dropped on its first incremental write. `notes.json` self-heals from
`notes.ydoc` on load; `notes.md` is a best-effort export that can lag the
authoritative doc after a crash between its rename and the rest.

**`settings.data_directory` override.** `app-main` reads
`settings.data_directory` after loading settings and calls `resolve_data_roots`
to derive the effective paths for `meetings/`, `models/`, `index.db`,
`collections.json`, and `voiceprints.db`. When the field is `Some(path)` and
`path` is an absolute, creatable path, those five entries move under `path/`;
`settings.store` and `logs/` always stay at the platform root (bootstrap
constraints). `collections.json` and `voiceprints.db` sit at the **same
effective root** as `index.db` and move with the `data_directory` override —
they are user-data files that must be co-located with the meetings they
reference. An invalid value (relative, empty, or uncreatable) is logged via
`tracing::error` and falls back to the platform default — startup is never
aborted. The roots are fixed for the process lifetime; a change requires an app
restart. Moving existing data is the user's responsibility (no automatic
migration). There is currently no UI for this field; it must be set by editing
`settings.store` directly.

**`index.db` is a derived, rebuildable cache (binding — Phase 4, A6).** The
per-meeting folders are the **source of truth**; `index.db` (the libsql
meeting-list index) is a query cache derived from each meeting's
`metadata.json` / `transcript.json`. `persistence` opens it lazily and
**rebuilds it from a folder scan on a missing or corrupt DB**
(`MeetingIndex::rebuild_from_disk`, invoked at app start by `ipc-bridge`'s
index bootstrap). A libsql/DB error therefore never risks user data — at worst
the meeting list is briefly stale until the next rebuild (which is also why an
index `upsert` failure on stop is logged-and-swallowed, not fatal). The schema
is versioned and the migration runner is **forward-only** (a `schema_version`
gate; opening an empty DB or a prior-schema DB migrates up without data loss).
Nothing depends on `index.db` being byte-stable or even present.

**Headless server data root (`headless` / `minutist-hub`).** The headless server
(WS4-B; see "Headless server daemon") runs over its OWN data root, supplied as an
absolute path at startup (`--data-dir`), entirely separate from any desktop's
`{app-data}`:

```
{data-dir}/                     absolute; supplied via --data-dir
├── sync_node_key               0600 ed25519 device identity; owned by `sync`
├── peers                       paired-device tickets (one per line); via `add-peer`
├── logs/                       rolling tracing appender (minutist-hub.log)
└── meetings/                   owned by `persistence`, reached through `sync`
    ├── {uuid}/                 per-meeting folders (notes.ydoc, audio.opus, …;
    │                           same layout as the desktop's meetings/)
    └── .blobs/                 iroh-blobs content store (redb); owned by `sync`
```

It has no `settings.store` / `mcp_token` / `tunnel_device.json` / `models/` — the
hub neither serves the UI nor, in the sync-hub role, runs models. The
single-writer rule applies per data root: the daemon must be the sole process
over its root, and must never point at a desktop's `{app-data}`.

### Per-meeting metadata.json write lock

The single-writer rule above keeps two *processes* off one data root; a second,
in-process lock serialises the in-process tasks that read-modify-write a meeting's
`metadata.json` against each other. Every such RMW goes through one guarded helper
— `persistence::meeting_ops::update_metadata(root, id, |meta| {…})` (and its
skip-if-absent sibling `update_metadata_if_present`) — which takes the lock,
reads, applies the closure, and writes atomically, so a caller cannot forget the
lock. The writers routed through it: the `meeting_ops` operations
(`rename_meeting`, `set_meeting_collection`, `set_speaker_name`,
`apply_processing_lifecycle`), `MeetingFolder::ensure`'s placeholder seed,
`read_meeting_state`'s lazy notes-format flip (the one-time `notes.ydoc`
migration on first open — it fires on exactly the synced meetings receiving
`Claimed`/`Processed`, so it must be guarded), the sync lifecycle-event
subscriber, the `orchestrator`'s post-processing RMWs (`finalise_diarization`,
the re-transcribe `speaker_count` update, the voiceprint `speaker_names`
restores), and `agent-tools`' write tools. On a multi-threaded
runtime any of these can run while another runs; without serialisation each does
an independent read→mutate→write and the later write drops the field the earlier
one set.

A process-wide per-meeting `std::sync::Mutex` registry — `METADATA_LOCKS`, in the
leaf `notes-crdt` crate — serialises them, mirroring the attachments
`MANIFEST_LOCKS` (see "Per-meeting manifest write lock"):

    static METADATA_LOCKS: OnceLock<Mutex<HashMap<MeetingId, Arc<Mutex<()>>>>>

It lives in `notes-crdt` because that is the lowest crate both writers reach
(`persistence` depends on `notes-crdt`, and `ensure` is defined there). It is a
`std::sync::Mutex`, not a `tokio` one, because every guarded RMW is synchronous
`std::fs` with no `.await` held across the guard — so it adds no `tokio`
dependency to `notes-crdt`. Each writer takes `notes_crdt::metadata_lock(id)` for
the check-then-RMW and drops the guard before any later `.await` (the `index.db`
upsert in `rename_meeting` / `set_meeting_collection` runs after the guard is
released; the index is a derived cache, reconciled by `rebuild_from_disk`).

Coverage (issue 0025, closed): every `metadata.json` RMW listed above is on this
single registry via `update_metadata`, so the headline lost-update — a local
diarize/reprocess pass racing a remote host's `Claimed`/`Processed` advert and
reverting `processing` — can no longer occur. `agent-tools` no longer keeps its
own instance-scoped lock registry; its write tools route through
`persistence::meeting_ops`, sharing the one lock. The cross-domain edits to
`orchestrator` and `agent-tools` were made under the agreed proposal in
`planning/issues/0025-metadata-lock-orchestrator-followup.md`.
`MeetingWriter::finalise` writes the initial `metadata.json` blind (not an RMW, no
prior on-disk state) and is intentionally not gated. The race is
regression-tested in `crates/persistence/tests/metadata_lock_race.rs`.

## Voiceprint matching

**Scope (issue #0003).** Cross-session speaker voiceprints: a speaker enrolled
by name in one meeting can be automatically re-identified in later meetings,
keeping their name across re-diarization without clearing `speaker_names`.

**`voiceprints.db` is NOT a rebuildable cache.** Unlike `index.db`, a
voiceprint is primary biometric data — it cannot be derived from the
per-meeting folders without re-running the user-guided enrolment flow. A
corrupt or missing `voiceprints.db` degrades enrolment to OFF and logs a
`tracing::error`; it must never block the meeting list or the recording
pipeline. The migration runner is forward-only (same discipline as `index.db`).

**Recompute-from-contributions invariant (§2.9.1 — binding).** The
`voiceprint_centroid.embedding` column is a *cache*, not primary data. Its
value is always equal to
`unit_normalise(Σ count_i · contribution_i.embedding / Σ count_i)`
over the centroid's surviving contributions, and `sample_count = Σ count_i`.
Any operation that adds, removes, or re-homes a contribution (enrol, refine,
merge, forget_meeting) MUST call `recompute_centroid` in the same transaction.
This makes refinement reversible: drop a contribution row, recompute, and the
centroid is back to what it was before that contribution was folded in.

**`model_id` hard-invalidation (§2.2).** Every `voiceprint_identity` row
carries the `model_id` of the embedding model used to build it. Matching and
refinement are valid only within the same model:
- `VoiceprintStore::refine` rejects (returns `Error::InvalidState`) if the
  incoming `model_id` differs from the identity's stored `model_id`.
- `VoiceprintStore::all(model_id)` returns zero rows for a foreign `model_id`.
  The caller MUST surface this as "N voiceprints from a previous model — re-enrol?",
  NOT as a silently empty library. Silently discarding the old voiceprints on a
  model upgrade would give users no indication that re-enrolment is needed.

**Corruption degrade-to-off contract.** A `VoiceprintStore::open` failure
(libsql error or migration error) returns the error to the caller, which maps it
to enrolment-OFF: the voiceprint feature is silently disabled for the session and
a `tracing::error!` is emitted. The meeting list, the recording pipeline, and the
transcript-read path are never blocked. There is no auto-repair path
(unlike `index.db`'s `rebuild_from_disk`) — voiceprints are primary data
and cannot be reconstructed.

**Thresholds (placeholders — WU6 calibration required).** The numbers below
are documented placeholders. They have no grounding in any in-repo sweep; WU6
assembles the labelled multi-session corpus and calibrates them.

| Band | Cosine similarity | Action |
|------|------------------|--------|
| Accept | `sim >= T_accept` (placeholder `0.60`) | Auto-apply the matched display name |
| Uncertain | `T_reject <= sim < T_accept` (placeholder `0.45..0.60`) | Suggest "is this \<Name\>?" — label shows the bare letter until confirmed |
| Reject | `sim < T_reject` (placeholder `0.45`) | No name, anonymous letter only |

**Refinement thresholds (placeholders — WU6):**
- `FOLD_GATE = 0.70` — cosine similarity floor for folding a new contribution
  into an existing centroid rather than creating a new condition entry. Not the
  offline clustering distance `0.75`; a different metric for a different purpose.
- `GALLERY_CAP = 4` — per-identity cap on the number of condition centroids.
  Cap-and-merge merges the two closest centroids only if their cosine clears
  `FOLD_GATE`; if no pair clears the gate, the cap is allowed to grow rather
  than silently blur genuinely distinct conditions.
- `REFINE_WEIGHT_CAP = 0.30` — a single meeting's contribution `count` is
  clamped to `min(count, existing_sample_count × 0.30)` before folding, so one
  adversarial meeting cannot dominate an established centroid. This is the
  bounded-weight poison defence (§2.9.3); the test in `voiceprints::tests` ships
  with the store.

**Asymmetric by design:** `T_accept` is tuned for a low false-accept rate
(labelling a stranger as a known person), NOT at EER. Per the design
(§2.4), the genuine 5th percentile sets `T_reject`; the impostor 99th
percentile sets `T_accept`.

**Assignment policy (WU5 — `orchestrator::matcher`).** Matching a set of fresh
diarizer clusters against the stored gallery is a global assignment problem, not
independent per-cluster thresholding. The algorithm in `orchestrator::matcher`:

1. Scores every `(query_label, identity)` pair by the **maximum cosine over the
   identity's gallery centroids** (§2.9.1 flat-gallery rule). Identity score =
   `max` over its centroids, so a person with an in-person centroid and a Teams
   centroid matches a query from either condition without a blurred-mean penalty.
2. Sorts all `(query, identity, score)` candidates descending by score.
3. Assigns greedily: a candidate is accepted only if (a) neither the query label
   nor the identity has already been assigned, (b) the score is `>= T_reject`,
   and (c) the score beats the **runner-up score for that query by at least
   `MIN_MARGIN`** (placeholder `0.05`). The margin requirement prevents two
   similarly-scored identities from both winning the same cluster.
4. The winning score is classified into the `Accept` / `Uncertain` / `Reject`
   band using the query-side noise guard: when the fresh cluster has fewer than
   `NOISE_GUARD_MIN_WINDOWS` (placeholder `3`) clean windows, `T_ACCEPT_NOISY`
   (placeholder `0.70`) is used instead of `T_ACCEPT`, so a noisy centroid cannot
   auto-accept.

**`orchestrator::matcher` is a pure function** (`assign_identities`): it takes
`&[QueryCluster]` and `&[StoredVoiceprint]` and returns `Vec<AssignedMatch>` with
no side effects, so it is unit-testable with no model and no Tauri runtime.
Tests ship with the module covering: two-clusters-one-identity global assignment,
margin drop, query-side noise guard, identity-score-is-max-over-gallery-centroids,
and empty-input guards.

**`apply_voiceprint_matches` wiring (WU5).** After a `reprocess` (user-triggered
or post-stop background pass) completes, `ipc-bridge` calls
`Orchestrator::apply_voiceprint_matches(meeting_id, store)` when
`voiceprint_enrolment_enabled` is ON. This method:
1. Reads the fresh transcript to collect the distinct diarizer labels.
2. Loads the gallery via `VoiceprintStore::all(model_id)`.
3. Extracts a centroid for each label using the same §2.3.1 cleanliness filter
   and clock mapper as `enrol_voiceprint_claimed` (on `spawn_blocking`).
4. Calls `assign_identities` with the extracted `QueryCluster`s and the gallery.
5. **Accept-band matches**: writes the matched display names back into
   `metadata.json`'s `speaker_names` (a second write on top of
   `finalise_diarization`'s `speaker_names.clear()`; the §2.6 re-map,
   clear-then-restore-matched), AND refines the matched identity with this
   meeting's centroid — §2.9.3 trigger (b). The refine targets the known
   `identity_id` from the assignment (not a name lookup) and `refine` is
   idempotent per `(identity, meeting)`, so running on every reprocess
   strengthens the voiceprint without double-counting. A refine failure is
   logged; the name is still applied.
6. **Uncertain-band matches**: collected into a
   `AppEvent::VoiceprintSuggestions` event emitted on the shared bus. The
   webview presents the "is this \<Name\>?" affordance for each suggestion.
   Confirming calls `set_speaker_name` (which triggers `enrol_voiceprint` for
   refinement — §2.9.3 trigger (c)); dismissing calls `reject_match`.
7. **Reject-band matches and extraction failures**: silently omitted — the label
   keeps its bare diarizer letter.

`apply_voiceprint_matches` is best-effort: errors are logged and the meeting is
left with cleared names, never propagated. The offline claim is still held by the
`reprocess` caller, so no concurrent op can clobber the second metadata write.

**Correction path (`reject_match` — WU5).** `Orchestrator::reject_match`
(called by `ipc-bridge::commands::reject_match`) handles the "this isn't them"
case: (a) it clears `speaker_names[label]` for `meeting_id` (empty-name write),
and (b) it drops the `(meeting_id, label)` contribution from `identity_id`'s
gallery via `VoiceprintStore::forget_contribution` on every centroid that matches.
`forget_contribution` recomputes the centroid cache from surviving contributions
(the §2.9.1 invariant); if dropping the contribution empties the centroid it is
deleted, and an identity left with no centroids is deleted too — so rejecting the
only match of a single-meeting identity removes it entirely and its (now stale)
embedding can never re-match (`all()` additionally filters `sample_count > 0`).
The method is idempotent when no matching contribution exists.

**`clear_all_voiceprints` (§4 privacy).** `ipc-bridge::commands::clear_all_voiceprints`
wraps `VoiceprintStore::clear_all()` — deletes every identity, centroid, and
contribution row. This is the local right-to-erasure path; the E2E sync path must
also purge replicas (a separate sync concern, not in scope here).

**`AppEvent::VoiceprintSuggestions`.** Emitted by
`Orchestrator::apply_voiceprint_matches` (via `finalise_diarization`'s sibling
method) when uncertain-band matches exist. Carries a `Vec<VoiceprintSuggestion>`,
each with the diarizer `label`, `display_name`, `identity_id`, `model_id` (needed
for `reject_match`), and `similarity`. Rides the existing `AppEventPayload`
newtype — no new event registration needed.

**Enrolment-enabled gate (consent obligation — §4 obligation 1).** Enrolment,
re-identification, and the prune-veto are all gated on
`settings.voiceprint_enrolment_enabled` (default `false`). The default-OFF
contract satisfies the collection-time consent obligation (BIPA / GDPR Art. 9):
no voiceprint is created or stored for any speaker until the user has
explicitly opted in. An older settings store written before the field existed
deserialises to `false` via `#[serde(default)]`, so a database upgrade can
never silently activate enrolment. When OFF, `speaker_names.clear()` runs as
before (zero behaviour change for users who have never opted in). When ON, the
reprocess re-map (§2.6) restores matched names instead of clearing them.

**WU3 enrolment-on-rename flow.** When `voiceprint_enrolment_enabled` is ON and
the user renames a speaker label in the UI, `ipc-bridge::set_speaker_name`
calls `Orchestrator::enrol_voiceprint(meeting_id, label, name, &VoiceprintStore)`
after the name write, best-effort (errors are logged and swallowed). The method
takes an offline claim (§2.3 — see below) and, if the model is locally
available, collects all *clean* segments for that label from the stored
transcript (§2.3.1 cleanliness filter: `speaker_id == label`,
`shared_speakers.is_empty()`, duration ≥ 1000 ms), reads the corresponding PCM
windows, and hands them to `VoiceprintExtractor::centroid`, which produces a
single CAM++ embedding. Depending on whether an identity already exists for
the given `display_name + model_id`, the embedding is written via
`VoiceprintStore::enrol` (first association) or `VoiceprintStore::refine`
(confirmed subsequent association — §2.9.3). If the model is absent, or if
the claim is busy, the call returns `Ok(None)` and logs at `debug`; the
rename itself is never blocked.

**WU3b refinement-on-confirm (§2.9.3).** A confirmed association routes to
`VoiceprintStore::refine` instead of `enrol` when an identity already exists
for the same `display_name + model_id`. "Confirmed" is exactly one of:
(a) the user typed/assigned the name via the UI rename (the WU3 path);
(b) an auto-accept match `sim >= T_accept` with the assignment margin AND the
meeting is finalised — `apply_voiceprint_matches` refines the matched
`identity_id` directly; (c) the user accepted an uncertain-band suggestion (WU5).
**Unconfirmed/uncertain matches never refine** — this is the primary slow-poison
defence. For triggers (a)/(c) the rename path resolves the identity via
`VoiceprintStore::find_identity_by_name_and_model` (a `Some(id)` routes to
`refine` rather than `enrol`); trigger (b) already holds the matched
`identity_id`. All three call `refine` with the centroid and the clean-window
count as the contribution weight, and `refine` is idempotent per
`(identity, meeting)` so a repeated reprocess replaces rather than accumulates a
meeting's weight.

The same `spawn_blocking` + offline-claim path used for enrolment (§2.3) is
reused unchanged: the flag, lock discipline, and clock hazards all apply. The
contribution's weight (`count` = number of clean windows) is clamped by
`REFINE_WEIGHT_CAP` inside `VoiceprintStore::refine` to bound a single
meeting's influence on an established centroid. Because contributions are
retained (§2.9.1 invariant), a later `reject_match` (WU5) can drop the
contribution and recompute — refinement is reversible. `FOLD_GATE` and
`REFINE_WEIGHT_CAP` remain placeholder constants (WU6 calibrates them).

**WU4 reprocess re-map (§2.6 — ephemeral centroid + timeline-coherence).** The
`reprocess` path (re-transcribe → diarize → finalise) clears `speaker_names` in
`finalise_diarization` unconditionally. When `voiceprint_enrolment_enabled` is ON,
two additional steps surround this:

1. **Pre-snapshot** (`capture_reprocess_snapshot`): Before the re-transcribe step
   overwrites `transcript.json`, `reprocess_claimed` and `reprocess_with_inputs_claimed`
   capture the current `(speaker_names, segments)` as an optional snapshot. This is
   the only point where the old named state and old segment timestamps coexist — the
   re-transcribe that follows will overwrite both.

2. **Post-finalise re-map** (`apply_ephemeral_remap`): After `finalise_diarization`
   clears names, the snapshot is used to attempt name restoration for each fresh
   diarizer label via two strategies in order:
   - **Centroid matching** (model-dependent): extracts a centroid from the old named
     label's clean segments and from the fresh label's clean segments (using the
     same §2.3.1 filter and clock mapper), then runs `assign_identities` treating
     the old labels' centroids as an ephemeral gallery. Accept-band matches restore
     the name. Skipped gracefully when the embedding model is absent.
   - **Timeline-coherence fallback** (model-free): computes the Jaccard temporal
     overlap between the fresh label's merged speech intervals and each old named
     label's merged speech intervals. If the overlap clears `TIMELINE_JACCARD_THRESHOLD`
     (placeholder `0.50`, calibrated in WU6) and the match is unambiguous (no tie),
     the name is restored. This path makes the re-map testable in the default suite
     without any embedding model.

The result is **clear-then-restore-matched**: `finalise_diarization` always clears
(preserving the invariant for the OFF path and for any label with no match), and the
re-map only restores names where a match is found. Unmatched fresh labels stay
anonymous. The offline claim is still held throughout, so no concurrent op can race
the second metadata write.

`TIMELINE_JACCARD_THRESHOLD = 0.50` is a placeholder pending WU6 calibration. It is
named as a constant (not a magic literal) so WU6 can swap the value at one call site.
The standalone `rediarize` path (no re-transcribe) also runs the ephemeral re-map via
`rediarize_inner_with_snapshot`, but reads the snapshot itself from disk at the top of
the pass (before `run_diarization_blocking` overwrites `transcript.json`).

**Clock-mapper reuse (§2.3 — binding).** Transcript `Segment::start_ms` /
`end_ms` are on the **pause-EXCLUDING** clock (recording wall time minus all
accumulated pause durations at that point). `read_audio_pcm` returns
**pause-INCLUDING** PCM (raw device samples). `enrol_voiceprint_claimed` uses
`runner::pcm_window_for_excluding_range` — the same mapper used by Phase 4
re-ASR — to convert each clean segment's excluding-clock interval to the
correct PCM byte range. The W1 clamping decision is inherited: a segment whose
start falls inside a pause region is silently discarded (returns `None`), not
panicked; the enrolment loop skips `None` windows.

**Offline-claim discipline (§2.3 — binding).** `enrol_voiceprint` calls
`Orchestrator::claim_offline` before touching the transcript or PCM, and
releases the claim in a `Drop` guard on all exit paths. If the claim is
unavailable (i.e., a `reprocess` pass is running), the enrolment returns
`Ok(None)` immediately — it never blocks waiting for the lock. This prevents
the enrolment from reading a partially-written transcript mid-reprocess.

**Pure vector maths.** `common::voiceprint_math` holds the three dependency-free
functions used by both `diarizer` and `persistence`: `unit_normalise`,
`cosine_unit`, and `weighted_merge`. No new crate edge is introduced — both
crates already depend on `common`.

**WU7 — prune-veto (§2.5).** A low-share diarizer cluster that would normally
be dropped by the share-floor or speaker-count cap is kept when the orchestrator
determines it matches an enrolled voiceprint. The veto is computed entirely
OUTSIDE the diarizer (the diarizer must not read the store — no
`diarizer → persistence` edge is created); the orchestrator computes a verdict
list and passes it in as `veto_ids: &[i32]`.

**Orchestrator flow (inside `run_diarization_blocking`):**

1. Before the split-backend enters, `compute_prune_veto_verdicts` identifies
   low-share candidate clusters from the raw `SpeakerTurn` tally — clusters
   whose total attributed duration is below `DiarizerConfig::min_cluster_share`
   of the total speech AND which contribute at least `PRUNE_VETO_MIN_WINDOWS`
   (= 3) complete 1.5 s audio windows (the same noise guard as WU5's
   `NOISE_GUARD_MIN_WINDOWS`). For each such candidate it extracts a centroid via
   `VoiceprintExtractor::centroid` and runs `matcher::assign_identities` against
   the gallery. Clusters where the accept-band verdict fires (`sim >= T_accept`)
   are returned as `Vec<(i32, String)>` (cluster id, matched display name).

2. The `VoiceprintExtractor` is consumed (dropped) before `diarize_split_merge`
   starts the Qwen split backend. Peak VRAM never includes both.

3. `diarize_split_merge` extracts `veto_ids` from the verdict list and passes
   them into `diarizer::overlay_speakers`. After the merge pass it resolves
   vetoed cluster ids back to their first-seen letters via the returned
   cluster→letter map and assembles `veto_names: Vec<(String, String)>`
   (letter → matched display name).

4. After `finalise_diarization` (which always clears `speaker_names`), the
   veto names are written in a second metadata write — the same append-after
   pattern `apply_voiceprint_matches` uses. The offline claim is held throughout,
   so no concurrent op can race the second write.

**Gallery loading.** `load_voiceprint_gallery` is a new async free function in
the orchestrator that calls `VoiceprintStore::all(DIARIZE_EMB_MODEL_ID).await`
and returns a `Vec<StoredVoiceprint>` (best-effort; empty on error). The
`rediarize` public method accepts `voiceprint_store: Option<&persistence::VoiceprintStore>`,
loads the gallery at the top of the call, and threads it down through the chain.
Paths that have no access to the store (the `reprocess` path, stub/test paths)
pass `None`; the prune-veto is then a no-op (empty veto_ids).

**Extractor instantiation.** `build_prune_veto_extractor` is a new async method
on `Orchestrator` that opens a `VoiceprintExtractor` from the embedding model
path (resolved via `DIARIZE_EMB_MODEL_ID`). It is best-effort: it returns
`None` on any failure. When `gallery` is empty (no enrolled speakers) or
`voiceprint_enrolment_enabled` is OFF, the orchestrator skips the extractor call
entirely and passes `None` to `run_diarization_blocking`.

**Diarizer-side invariant (binding).** Vetoed clusters pass through
`surviving_clusters` with two specific exemptions:
- They are not subject to the share-floor or segment-count-floor check.
- They are not subject to the `max_speakers` cap; the cap yields one slot to
  each vetoed cluster by dropping the lowest-share non-vetoed cluster first.
A vetoed cluster that would otherwise survive (share above the floor) is unaffected —
the veto only matters at the boundary.

**`PRUNE_VETO_MIN_WINDOWS: u64 = 3` (diarizer constant — placeholder).** Matches
`matcher::NOISE_GUARD_MIN_WINDOWS`. Calibrated in WU6 alongside the acceptance
thresholds.

**Enrolment-enabled gate (inherited from §4).** The prune-veto path is gated on
`settings.voiceprint_enrolment_enabled` (default `false`). When OFF, `gallery`
is never loaded and the extractor is never opened; `veto_ids` is always `&[]`.

**WU8 — identity management (issue #0003 §2.9.4, §4).** Five new IPC commands
and two new `VoiceprintStore` methods complete the management surface:

- `VoiceprintStore::rename_identity(id, new_name)` — renames in place; trims
  whitespace; rejects blank name. Unit-tested in `persistence::voiceprints::tests`.
- `VoiceprintStore::identities_with_gallery()` — management-UI query: returns
  every identity with per-condition `CentroidSummary` (no embedding bytes). Used
  by `list_voiceprints`; safe for IPC.
- `list_voiceprints` IPC — serialised as `VoiceprintIdentityInfo[]` (camelCase
  via `#[serde(rename_all = "camelCase")]`; `ipc-bridge`-local type, NOT in
  `common`, to keep `persistence` specta-free).
- `merge_voiceprint_identities` / `rename_voiceprint_identity` /
  `delete_voiceprint_identity` / `forget_meeting_voiceprints` IPC — thin
  delegates to the corresponding `VoiceprintStore` methods; all best-effort
  (silent no-op when the store is degraded-to-off).
- `forget_meeting_voiceprints` is the §4 per-meeting erasure path exposed as a
  command. The binding obligation (calling it from `delete_meeting`) is a
  follow-up; the command is wired and callable from the UI now.
- UI: `VoiceprintPane` (React, `ui/src/shell/VoiceprintPane.tsx`) — lists
  identities, rename, delete, clear-all, and a two-step "merge these two people"
  flow with surviving-name choice. Rendered inside `SettingsDrawer`.

## Note image assets

Images pasted or dropped into the notes editor are stored as **files** under
`{app-data}/meetings/{uuid}/assets/<sha256(bytes)>.<ext>`, written/read only by
`persistence` (the `assets` module — `save_note_asset` / `read_note_asset`).
The content-hash filename means identical pastes dedupe to one file.

- **Stored portable reference (binding).** `notes.json` stores the **bare
  filename** as the image node's `src` — NOT a machine-specific absolute path
  and NOT a platform-specific webview URL. Because `notes.json` and the asset
  live in the same meeting folder, the folder (with its `assets/`) can be copied
  to another machine and the notes still resolve. The editor's `getJSON` keeps
  this portable value; the conversion to a working URL happens only at render
  time and is never written back. This keeps the `notes.json` **opacity
  guarantee** intact — the Rust side never parses the document to find images.

- **Rendered URL (per-platform).** At display time the webview converts the
  stored filename into a working URL via Tauri's
  `convertFileSrc("<meeting_id>/<filename>", "meetingasset")`, which yields
  `meetingasset://localhost/<meeting_id>/<filename>` on macOS/Linux and
  `http://meetingasset.localhost/<meeting_id>/<filename>` on Windows (the
  live-test target). The meeting id is supplied by the editor at render time
  (not baked into the document), since the asset always lives under the open
  meeting's folder.

- **Serving mechanism (verified, Tauri 2.11.2).** `app-main` registers a custom
  URI-scheme protocol on the `tauri::Builder` via
  `register_uri_scheme_protocol("meetingasset", handler)`. The synchronous
  handler signature is `Fn(UriSchemeContext<'_, Wry>, http::Request<Vec<u8>>) ->
  http::Response<Vec<u8>>`. It reads `meetings_dir` from the managed `IpcState`,
  delegates parse + read to `ipc_bridge::resolve_note_asset` (which owns the
  `persistence` edge — `app-main` does not depend on `persistence`), sets the
  `Content-Type` from the extension, and returns an empty **404** on ANY
  validation/read failure so no detail leaks.

- **Path-traversal guard (binding).** The protocol exposes **only**
  `{meetings_dir}/<uuid>/assets/<filename>` — never the whole filesystem.
  `resolve_note_asset` parses the request path into a `Uuid` + single filename
  segment (rejecting non-UUID ids and nested paths), and
  `persistence::read_note_asset` rejects any filename containing a path
  separator or `..` before reading.

- **`ext` allowlist.** The `save_note_image` command and the content-type map
  accept only `png` / `jpg` / `jpeg` / `gif` / `webp`; anything else is an
  `AppError::InvalidInput`.

- **Auto-cleanup.** `meeting_ops::delete_meeting`'s `remove_dir_all` removes the
  whole meeting folder, so `assets/` (and its images) are deleted with the
  meeting — no separate asset cleanup path is required.

## Attachments (Attachments WS)

Meeting attachments extend the `persistence` component and the `ipc-bridge`
command surface. The design mirrors the note-image `assets/` pattern (content hash,
atomic write, traversal guard) and adds a JSON manifest with a per-meeting write
lock.

### Attachments storage layout

```
{app-data}/meetings/{uuid}/
    attachments/                       # sub-dir distinct from assets/
        attachments.json               # manifest: Vec<AttachmentEntry>
        <sha256>.<ext>                 # content-addressed original
        <sha256>.md                    # converted markdown sibling
```

`attachments/` is intentionally separate from `assets/` (note-image originals) to
avoid a namespace collision. `meeting_ops::delete_meeting`'s `remove_dir_all`
removes the whole meeting folder including `attachments/` — no separate cleanup
required (same as `assets/`).

### Content addressing and deduplication

The original filename is stored in the manifest (`AttachmentEntry.original_filename`)
but the on-disk file uses `<sha256>.<ext>` (content-hash, mirroring
`assets::save_note_asset`). Identical file bytes dedupe to one `<hash>.<ext>` file
and one `<hash>.md` sibling. `remove_manifest_entry` applies dedup-safe unlink:
`unlink_attachment_files` is called only when no other surviving manifest row shares
the removed entry's hash, so two manifest rows pointing at the same bytes are both
present before either is unlinked.

### Per-meeting manifest write lock

The manifest is a single JSON file; concurrent `add_attachment` /
`remove_attachment` calls for the same meeting must not lost-update. A process-wide
per-meeting `std::sync::Mutex` registry serialises every read-modify-write:

    static MANIFEST_LOCKS: OnceLock<Mutex<HashMap<MeetingId, Arc<Mutex<()>>>>>

Each public manifest function takes the per-meeting lock for the whole RMW (read
the file, mutate, write atomically via tmp + fsync + rename). The RMW is
synchronous `std::fs` on `spawn_blocking`, matching `chat.rs` and `assets.rs`.

### Opening an attachment original (host hand-off)

"Open" hands an attachment original to the HOST OS default application — the
user's PDF reader / Word / Excel / image viewer — NOT a webview navigation or a
custom-URI render. The `open_attachment` command (`ipc-bridge`) resolves the
stored original's on-disk path via `persistence::attachment_original_path` (which
applies the path-traversal guard) and passes it to `tauri-plugin-opener`'s Rust
API (`app.opener().open_path`). The open happens server-side, so no filesystem
path crosses the IPC boundary and no opener capability scope is needed (the
capability system gates only JS-invoked commands, not the Rust manager API). The
originals are real files (content-addressed `attachments/<hash>.<ext>`), so no
temp file is written.

There is no custom URI scheme for attachments: opening is a host hand-off, not a
webview fetch. (Note images still use the separate `meetingasset:` scheme for
inline `<img>` display — a different need.)

### Attachments — parser sandboxing (binding)

`doc-convert`'s `convert_to_markdown` wraps every converter in
`std::panic::catch_unwind` (parser panics on malformed input must surface as a
recoverable `AppError`, not crash the conversion worker — see "Error handling").
Two hard limits are enforced before parsing and returned as `AppError::InvalidInput`
on violation:

1. `MAX_INPUT_BYTES` (50 MiB) checked on `bytes.len()`.
2. Zip-decompression bound for the zip-container formats (`pptx` / `docx` /
   `xlsx` / `ods`): cumulative uncompressed size and entry count tracked via
   `zip`'s `by_index` sizing; abort if a zip-bomb ratio is exceeded.

Textual-content bar for Office formats: `pptx` and `docx` extract text content
(paragraphs, list-item text, table-cell text, plus per-slide pptx speaker
notes) for the summariser feed, not faithful structure. `docx` uses the same
`zip` + `quick-xml` walk as `pptx` (over `word/document.xml`), so no `docx-rs`
production dependency is introduced. Known limitation: for a multi-column
digital PDF, `pdf_oxide`'s default `extract_text` sorts spans into row bands (by
Y then X), which interleaves side-by-side columns rather than reading each
top-to-bottom; all text is captured but cross-column reading order is not
guaranteed (the summariser tolerates this; `pdf_oxide`'s column-aware modes are a
possible future refinement).

`catch_unwind` requires `UnwindSafe`; inputs are wrapped in `AssertUnwindSafe`
(the closure only reads `&[u8]` + `&str`). A caught panic is mapped to
`AppError::InvalidInput { context: "conversion panicked for .<ext>" }` and logged
at `tracing::warn(target: "doc-convert")`.

### Summariser attachments feed

When `ipc-bridge` calls `run_held_summarise`, it reads the meeting's manifest,
concatenates every `Ready` entry's `<hash>.md` under a `## Attachment: <original_filename>`
header in manifest order, and applies the deterministic budget-truncation helper
(per-attachment equal-share, `[truncated]` marker on any trimmed part, budget
derived from `SummariserConfig.n_ctx` minus reserves for transcript + notes +
generation). The assembled string is passed as `attachments_markdown` to `summarise`.
An empty string (no manifest, or no Ready entries) produces byte-identical output to
a run with no attachments — the prepend in `render_user_content` is conditional.

## Held model serves vision

The OCR VLM is the **already-held Gemma-4**, not a second model. `LlamaSummariser`
owns a `LlamaModel` (already `unsafe impl Send + Sync`). A vision `MtmdContext` —
the multimodal projector that maps image tokens into the LM's embedding space —
can be bound to that same loaded model via
`MtmdContext::init_from_file(mmproj_path, &model, …)`, exactly as
`asr-runtime` binds its audio `MtmdContext` to the Qwen3-ASR model. So there is
**no second GGUF**: the OCR VLM reuses the `~5 GiB` Gemma-4 LM weights already
held by the summariser, and the mmproj/encoder (~560 MB) is co-resident only
while an OCR job is active.

**Lazy vision context.** `LlamaSummariser` gains a `vision:
OnceLock<Mutex<MtmdContext>>` and an `ensure_vision(mmproj) ->
AppResult<&Mutex<MtmdContext>>` that builds the `MtmdContext` from the
already-loaded model on first image job, mirroring the `maybe_preload_summariser`
lazy posture. `GemmaVlm` (ipc-bridge) is a thin adapter holding only a
`ChatHandles`; it resolves the held summariser, calls `ensure_vision` +
`image_to_markdown`, and owns no vision state itself. No vision load happens
until an image attachment actually reaches the worker.

**Same ~8 GiB GPU budget.** The VRAM thresholds in `resolve_gpu_plan`
(`SUMMARISER_VRAM_BYTES`, `cross-cutting.md` — "GPU portability") already account
for the full Gemma-4 activation footprint. The mmproj encoder adds ~200–300 MB
of temporary activation while an image is being encoded; the VRAM budget is NOT
widened to accommodate it — the estimate already includes headroom, and the
encoder is short-lived. If a future measurement shows the budget needs revision,
that is a `common` architecture-owner change.

**MtmdContext serialisation (not a Send/Sync barrier).** `MtmdContext` carries
an `unsafe impl Send + Sync` in `llama-cpp-2`, so it is not the threading
constraint: both `LlamaSummariser` (holding `OnceLock<Mutex<MtmdContext>>`
alongside the already-`unsafe impl Send + Sync` `LlamaModel`) and `GemmaVlm`
(holding only `ChatHandles`) derive `Send + Sync` without any new `unsafe impl`.
The `Mutex<MtmdContext>` exists to SERIALISE access: the encode path mutates
internal C state through a shared pointer and runs on the same GPU as
summarise/ASR, so concurrent `eval_chunks` would race that state and contend on
the device. The one bounded conversion worker processes OCR jobs sequentially, so
the mutex is uncontended in practice — a single lock/unlock per image. This
mirrors the offline `SherpaDiarizer`'s `Mutex<Diarize>`: a `&self` trait method
hiding a `Mutex`-guarded `&mut` resource.

## OCR policy for image attachments

**The VLM handles only inputs with no pure-Rust text path.** Every digital
document (`txt`/`md`/`xlsx`/`ods`/`html`/`eml`/`pdf`/`pptx`/`docx`) extracts via
its pure-Rust converter; the VLM is consulted only for direct image attachments
(`png`/`jpg`/`jpeg`/`tiff`). Rationale: digital text extracts instantly and
losslessly, whereas a ~4–14 s/page GPU pass buys nothing for text-bearing
documents and can *introduce* OCR errors; it would also contend with
summarise/ASR on the shared GPU. The single bounded conversion worker serialises
OCR jobs, so OCR and summarise never compete for the `LlamaContext`.

**Image attachments.** The bytes are decoded and re-encoded to PNG (via the
`image` crate) and passed to `vlm.image_to_markdown()`. When `vlm` is `None` (no
held model / mmproj absent), they return `AppError::Unsupported`; the attachment
row shows "Conversion failed" and the user can still open the original via "Open
anyway".

**Scanned / image-only PDFs are NOT supported.** `pdf_oxide` returns
near-empty text for them, and `doc-convert` then returns `AppError::Unsupported`
rather than attempting OCR — the VLM is never invoked for a PDF in this build.
PDF-page OCR needs page rasterisation (a PDFium runtime library bundled per
platform), which is deferred — planning issue 0019.

**Near-empty threshold.** A PDF extraction result is treated as near-empty when
the whitespace-stripped text is shorter than a small constant (100 characters,
in `doc-convert`). Conservative by design: a PDF with a few hundred words of
digital text extracts reliably.

**Deferred.**
- The PDF-image cases — scanned pages, embedded figures, and digital-PDF
  table-structure recovery (`pdf_oxide`'s plain-text extraction flattens table
  grids to cell text) — are tracked in `planning/issues/0019`; all need PDF-page
  rasterisation. Also tracked there: the rare digital PDF whose body font
  `pdf_oxide` cannot decode (it extracts partial text — headings/notes — rather
  than crashing), which a future pdfium/VLM quality-fallback could backstop.
- Which VLM model does the OCR (Gemma-4 chosen; PaddleOCR-VL revisit) is
  `planning/issues/0018`.

## Telemetry

None in v1. Architecture deliberately leaves no telemetry hooks.

If telemetry is added later, it lives in a dedicated `telemetry` crate
with a kill-switch in `settings`, off by default. This requires an
architecture-doc update and an explicit recorded product decision.

## Testing

- Per-crate unit tests live alongside the crate.
- Integration tests that exercise the orchestrator live in
  `crates/orchestrator/tests/`.
- The Tauri main binary is not unit-tested directly; it's wiring.
- Live sync is covered at two layers, all gated on `MINUTIST_SYNC_TOKEN` so a
  normal `cargo test` / CI never touches the network: the engine layer
  (`crates/sync/tests/{relay_live,blobs_live}.rs`) and the app-glue layer
  (`src-tauri/src/sync.rs` ConnectedSync). The headless hub adds a third:
  `crates/headless/tests/hub_e2e.rs` spawns the real `minutist-hub` binary as an
  always-on middle peer and asserts two devices converge **through** it over the
  deployed relay.

Test fixtures (sample WAV files, expected transcripts) live under
`tests/fixtures/` at the repo root and are git-lfs'd if they exceed
~1 MB.

### Automated-testing policy (binding on every phase)

Every phase ships automated tests that cover its acceptance criteria.
This is a phase close-out gate, not a nicety — the PO and `phase-verify`
test-adequacy dimension fail a phase whose acceptance is only manually
demonstrated.

- **Synthetic data is generated where behaviour needs input.** Where a
  test needs a recording, transcript, meeting, or multi-speaker audio that
  doesn't exist as a fixture, generate it deterministically and commit it
  (or a generator) under `tests/fixtures/`. Examples: a synthetic
  multi-utterance recording for VAD/accumulator tests; a hand-labelled
  two-speaker fixture (concatenate two distinct single-speaker clips with
  known boundaries) for diarization accuracy; a synthetic 30-minute
  transcript (`Vec<Segment>`) for summariser chunked-prefill and latency;
  a synthetic meeting folder (audio + transcript + notes + metadata) for
  persistence save/reload. The Silero-VAD-rejects-tones constraint above
  still applies — synthetic *speech-path* audio must be real speech
  (repeat/concatenate the LibriSpeech fixture), not tones.
- **The default suite runs in CI with no manual step and no native
  hardware.** `cargo test --workspace` and `npm test` (the `ui/`
  package's `vitest run` script) must pass on a machine with no model
  files, GPU, or microphone. Tests that need a real
  model, GPU, or native build are **gated behind env vars** (the Phase 2
  `MINUTIST_ASR_MODEL_PATH` pattern) with a no-op skip path. They are run on
  demand either via `scripts/run-tests-windows.ps1` OR directly with
  `make test-integration` (and `-summary`/`-asr`/`-diarize`), which sources a
  git-excluded `tests-local.env` (copied from `tests-local.env.example`) holding
  the real model paths + a `MINUTIST_RECORDINGS_DIR`. Running these against
  real models is how model-integration regressions (e.g. a chat template the
  bundled llama.cpp cannot render) are caught without a full app rebuild — the
  gated summariser test exercises `build_prompt`, and a real-recording variant
  summarises an actual `transcript.json` from the recordings dir.
- **Manual acceptance is additive, never a substitute.** Items that
  genuinely cannot be asserted in software (copy-paste-into-Word fidelity,
  the GPU portability matrix, clean-VM install) are recorded as
  native-hardware evidence in the engineering journal *in addition to* automated
  coverage of everything around them (e.g. the HTML-clipboard serialiser is
  unit-tested even though the paste into Word is checked by hand; the
  updater state machine is tested against a synthetic signed-manifest
  endpoint even though the cross-OS install is run on VMs).
- **Frontend behaviour is tested with Vitest + Testing Library** against
  the generated IPC bindings (mock the Tauri command layer); editor and
  cross-reference interactions assert behaviour, not snapshots.

Two constraints learned from running the gated pipeline tests on native
hardware (Phase 2 close-out):

- **Tests that drive audio through the runner must feed real speech.** The
  runner always instantiates the real Silero VAD, which rejects synthetic
  tones — a 440 Hz sine never produces a `SegmentEnd`, so the accumulator
  never fills and no transcript is emitted. Integration tests that exercise
  the VAD→ASR path use the LibriSpeech fixture, not `DummyAudioSource`.
  `DummyAudioSource` is still valid for back-pressure / metering tests that
  do not assert on VAD output.
- **Event-collection deadlines must tolerate a saturated scheduler.** Cargo
  runs test *binaries* in parallel. When a model-loading test (gated on
  `MINUTIST_ASR_MODEL_PATH`) runs alongside a timing-sensitive one, CPU
  saturation can starve a tight broadcast-drain loop. Size such deadlines in
  seconds, not hundreds of milliseconds.
- **Wall-clock duration assertions compare against the *measured* elapsed, not
  the nominal sleep.** A `sleep(N)` only guarantees *≥ N*; under parallel-binary
  contention it overshoots, and code that records the real elapsed (e.g. the
  Opus encoder padding a pause with silence sized to `paused_at.elapsed()`) will
  then exceed an `N ± ε` window. Capture the actual elapsed in the test and
  assert against it (`pause_resume_decoded_duration_includes_pause_gap` does
  this). Where a deterministic gap is needed without any wall-clock, use a
  `#[cfg(test)]` injection seam (e.g. `OggOpusEncoder::resume_with_pause_frames`).

## Auto-update

Owned by `app-main` (it's process-lifetime work). Uses
`tauri-plugin-updater` against a static HTTPS endpoint serving signed
artefacts. Introduced in Phase 7; no other crate touches updater logic.

Updater status reaches the webview event-driven on the shared `AppEvent` bus:
`AppEvent::UpdateAvailable { version, notes }` when a check finds a newer
release, and `AppEvent::UpdateProgress { downloaded_bytes, total_bytes }` while
an accepted update downloads (mirroring `ModelDownloadProgress`). The verify
step uses the Tauri updater's **minisign** keypair — a separate key from the OS
code-signing certs (Windows EV / Apple Developer ID); the updater rejects an
artefact whose minisign signature does not verify. Per Q7, v1 ships one artefact
per platform built with a portable GPU backend (Vulkan on Windows/Linux, Metal
on macOS) with runtime CPU fallback, so there is no per-backend update fan-out.

The flow is driven entirely from Rust via `UpdaterExt` (no JS updater plugin):
app-main checks on startup and emits `UpdateAvailable`; the webview surfaces it
through the `UpdateBanner` chrome strip (backed by the `update` store's
idle → available → downloading → applying state machine) and, on accept, emits
the `updater://apply` event back; app-main then downloads (emitting
`UpdateProgress`), installs, and relaunches (`AppHandle::restart`). Apply
failures emit `AppEvent::ErrorOccurred`. All updater calls are **guarded** —
the committed `plugins.updater` config has the production `endpoints` URL
(the GitHub releases `latest.json`) but an empty `pubkey`, so `check()` is a
logged no-op until the minisign keypair is activated; dev/unsigned builds are
unaffected. `bundle.createUpdaterArtifacts` is `false` — with it on, the
bundler hard-requires `TAURI_SIGNING_PRIVATE_KEY` at build time, which
contradicts the deferred-keypair posture. Activation is a one-time
maintainer step (documented in `RELEASING.md`): generate the minisign
keypair, keep the private key as the `TAURI_SIGNING_PRIVATE_KEY` CI
secret, paste the public key into `tauri.conf.json` `pubkey`, and set
`createUpdaterArtifacts` to `true` so release builds emit the `.sig`
updater artefacts. The app-wide Tauri 2 capability is `src-tauri/capabilities/default.json`
(`core:default` + `core:event:allow-emit`/`allow-listen`, scoped to the `main`
window) — without a capability a Tauri 2 webview has no IPC access at all, so
this is what lets the webview invoke the tauri-specta commands, receive
`AppEvent` payloads, and emit `updater://apply`. `opener:allow-open-url` is
granted **URL-scoped** (least privilege): `allow: [{ url:
"https://github.com/Minutist/*" }]` confines `openUrl` to the project's GitHub
host (the only URL the app opens is the "Report a problem" issue link), so a
compromised renderer cannot drive it to a `file:`/custom scheme. The build-time
ACL (`gen/schemas/capabilities.json`) is generated from it.

## GPU portability

GPU acceleration is selected at **build time** via Cargo features, all **OFF by
default** so the default `cargo build --workspace` is CPU-only and needs no GPU
SDK installed. Feature names match the backend they enable:

- `vulkan` / `metal` / `cuda` / `rocm` forward to `llama-cpp-2/<backend>` (the
  ASR + summariser path).
- `cuda` / `directml` forward to `sherpa-rs/<backend>` (the diarizer path);
  there is no Vulkan/Metal diarization backend, so on those platforms the
  diarizer stays on the ONNX Runtime CPU EP.

Enabling a feature also offloads work to the device, but the per-model placement
is a **VRAM-aware runtime** decision driven by the tri-state
`settings.gpu_acceleration: GpuAcceleration` (default `Auto`):

- **`Auto`** (default) probes the GPU's reported VRAM at each model-load moment
  and offloads a model to the GPU only when it fits, else CPU.
- **`On`** forces full GPU offload; the VRAM clamp for the large ASR tier still
  applies (a no-probe `On` cannot confirm the 1.7B fits, so it falls back to the
  small tier in that case).
- **`Off`** forces CPU without consulting the probe (the old `false`) — the
  runtime escape hatch for weak GPUs / driver trouble.

In a default CPU-only build the setting has no effect (the compile-time ceiling
is already `0`, and `probe_primary_gpu()` returns `None`).

**The probe + the plan.** `common` owns both halves (so the plan + its tests
build CPU-only):

- `probe_primary_gpu() -> Option<GpuProbe>` (behind the `llama-backend` feature;
  `None` on a CPU-only build / no device) queries the same ggml backend that
  loads the GGUFs and reports `{ total_bytes, free_bytes, is_integrated, name }`
  for the primary device (first discrete GPU, else the integrated one — multi-GPU
  is out of scope).
- `resolve_gpu_plan(probe, mode, prefer_large_asr) -> GpuPlan` is **pure** (the
  probe is an input, so it unit-tests without a GPU) and returns
  `{ summariser_gpu, asr_gpu, effective_prefer_large }`.

**Policy.** Placement is **binary per model** (whole model on GPU or on CPU):
partial layer offload is slower than CPU for models this small, and the existing
`n_gpu_layers` resolution is already binary. The large ASR tier is **requested
automatically** (call sites pass `prefer_large_asr = true`); the VRAM clamp in
`resolve_gpu_plan` decides whether it fits. Under `Auto` the plan **budgets the
summariser FIRST** (it stays resident while an ASR model loads when
`preload_summariser` is on), then budgets ASR against the **remaining** headroom
and downgrades to the small tier (`effective_prefer_large = false`) when the 1.7B
would not fit — running the 1.7B model purely on CPU is strictly worse than the
0.6B CPU default. `On` applies the same VRAM clamp to the large ASR tier, rather
than trusting a now-removed user flag blindly. The decision base is
`total_bytes × headroom` (0.90 discrete, 0.50 integrated), **not `free_bytes`**:
a Vulkan device without `VK_EXT_memory_budget` reports `free == total`, so `free`
is trusted only to *tighten* the budget when it is a credible smaller number.
**A `None` probe (no GPU / probe failed) fails safe to CPU** — a false "fits"
risks an out-of-memory load or a silent host-memory spill.

**The VRAM thresholds are PROVISIONAL ESTIMATES pending live-hardware
validation.** The decision constants live as named `const`s in `common`
(`resolve_gpu_plan`): `SUMMARISER_VRAM_BYTES` (the load-bearing one — Gemma-4-E4B
Q4 weights + KV @ 32K + headroom, ≈ 8 GiB), `ASR_SMALL_VRAM_BYTES`,
`ASR_LARGE_VRAM_BYTES`, and the `DISCRETE_HEADROOM` / `IGPU_HEADROOM` fractions.
They were derived from model file sizes, NOT measured — in particular Gemma's KV
footprint at the 32K context (and its interleaved sliding-window attention) is
calculated. **To tune:** run the app on the target GPU and read the one-shot
startup log line `IpcState::log_gpu_probe` (`target: "app-main"`, fields
`gpu`/`total_mb`/`free_mb`/`integrated`/`summariser_gpu`/`asr_gpu`/`effective_prefer_large`),
compare the reported VRAM against what the model actually needs (e.g. llama.cpp's
reported KV/compute-buffer sizes on load), and adjust the consts. `resolve_gpu_plan`
is pure, so the policy re-tests without a GPU after any change. KNOWN GAPS to
confirm on real hardware: the Gemma KV estimate, whether Vulkan reports
`free == total` here, and the integrated-GPU `0.50` cap.

Wiring: `asr-runtime`'s `AsrRuntimeConfig` and `summariser`'s `SummariserConfig`
each carry a `n_gpu_layers: u32` field whose `Default` is the cfg-gated
compile-time ceiling (`default_n_gpu_layers()` / `gpu_layers()` → `u32::MAX`,
clamped to `i32::MAX` = "all layers", when a GPU feature is compiled in, else
`0`). The model-open site uses `config.n_gpu_layers` for `with_n_gpu_layers(...)`,
and the mtmd `use_gpu` is derived from `config.n_gpu_layers > 0`. Each consumer
computes **one** `GpuPlan` per model-load decision and maps the relevant boolean
to the layer count: the orchestrator's private `gpu_plan()` helper feeds
`runner::resolve_gpu_layers(plan.asr_gpu)` for the live + offline-re-transcribe +
re-listen + prewarm ASR sites (and `plan.effective_prefer_large` selects the ASR
tier via `asr_engine_for_language`), while `ipc-bridge`'s held-summariser load
feeds `commands::resolve_summariser_gpu_layers(plan.summariser_gpu)`. The two
`resolve_*_gpu_layers(enabled: bool)` helpers are unchanged — only their argument
is now the plan boolean instead of the old enum-bool. llama.cpp falls back to CPU
at runtime when no device is present, so a GPU-feature build is still safe on a
CPU-only machine. (Before this, the placement was a single on/off flag mapped
straight to the compile-time ceiling; the VRAM-aware plan now lets `Auto` keep
each model on GPU only when it fits.)

The features fan out through a single chain so the app binary is the only place
a backend is chosen: `minutist` (src-tauri) → `ipc-bridge` → {`summariser`,
`chat-agent`, `orchestrator` → {`asr-runtime`, `diarizer`}}. `ipc-bridge` is the
fan-out point because it sits above `summariser` (direct dep), `chat-agent` (the
held-model chat engine, which forwards `vulkan`/`metal`/`cuda`/`rocm` to its own
`llama-cpp-2` so a GPU build links the chat engine against the same process-wide
`LlamaBackend` as summariser/asr-runtime), and `orchestrator` (which owns
`asr-runtime` + `diarizer`); the orchestrator does NOT depend on summariser
(rule A5), so summariser is reached via ipc-bridge, not orchestrator.

**Q7 — one artefact per platform.** v1 ships a single build per OS with a
portable backend — **Vulkan** on Windows/Linux, **Metal** on macOS — and
relies on llama.cpp's runtime CPU fallback when no compatible device is present.
This avoids a per-backend artefact matrix. CUDA/ROCm/DirectML device-specific
builds are a post-v1 optimization, not a shipped fan-out. CI builds each
selected feature on the appropriate runner (`cargo build --features vulkan`
etc.); the GPU portability matrix (NVIDIA / AMD / Intel iGPU / Apple Silicon /
CPU-only, with WER/RTF/warm-first-segment latency) is recorded as manual
hardware evidence in the engineering journal.

## Output-language injection (summariser and chat)

The `settings.output_language` field controls the language for all LLM-generated
text (summaries and chat replies). The transcript is NEVER affected.

**Injection point.** `ipc-bridge`'s `apply_output_language(prompt, setting) ->
String` is the single call site: it calls `resolve_output_language(setting)` from
the `output_language` module, and when that returns `Some(lang)` it appends
`"\n\nRespond entirely in {lang}."` to the system prompt. Appending AFTER the full
prompt (including any user-customised text) ensures the explicit output-language
setting wins over any conflicting instruction in a custom prompt.

**Summariser path.** `run_held_summarise` (which backs both the direct
`summarise_meeting` command and the post-stop auto-summarise chain) resolves the
effective prompt via `Settings::effective_summary_prompt()` and then passes it
through `apply_output_language` before handing it to the LLM.

**Chat path.** Both `send_chat_message` (the UI chat path) and the inter-agent
bridge driver in `inter_agent.rs` resolve the chat system prompt via
`chat_system_prompt_for_meeting` and then pass it through `apply_output_language`
before starting the turn.

**Resolution.** `resolve_output_language` returns `None` (no instruction) for:
- the sentinel `"auto"` when `sys_locale::get_locale()` is unavailable or the
  primary subtag is not in the 15-language mapping table;
- an empty or whitespace-only setting.
An explicit full English language name (e.g. `"French"`) passes through verbatim.

## Build variants

Two shipping artifacts are produced from one source tree:

| Artifact | Cargo invocation | Vite | Contents |
|---|---|---|---|
| **Connected** (default) | `cargo build` (or `--features connected`) | `VITE_CONNECTED=1` | MCP server, bearer-token generation, MCP settings pane, the relay tunnel (device pairing + connector) + Connection settings pane |
| **Free** | `cargo build --no-default-features [--features <gpu>]` | `VITE_CONNECTED` unset | No MCP server, no rmcp, no listening socket, no tunnel-client, no MCP/Connection panes in the UI bundle |

**Single identifier.** Both artifacts share `ai.minutist` as the bundle identifier and product name. Artifact names in CI are distinguished by a `-free` suffix on the artifact upload name only — no `productName` change.

**Cargo feature.** `connected` is a default feature in `src-tauri/Cargo.toml`. It gates:
- `dep:mcp-server` — the entire MCP server crate + rmcp transitive stack.
- `dep:tunnel-client` — the app-side relay tunnel (pairing + reconnect + lifecycle).
- `dep:async-trait` — the `ConnectedTunnel` impl of `ipc_bridge::TunnelControl`.
- `dep:rand` and `dep:hex` — CSPRNG bearer-token generation (`resolve_mcp_token` in `app-main`).
- The MCP spawn block AND the `ConnectedTunnel` construction + `src-tauri/src/tunnel.rs` module in `app-main`'s `setup()` (`#[cfg(feature = "connected")]`).

The free build compiles `mcp_info` to a permanently-`None` slot; `get_mcp_server_info` returns `None` unconditionally. `IpcState.tunnel` is `ipc_bridge::disabled_tunnel()` (reports `Disconnected`, rejects pairing as `Unsupported`), so the four tunnel commands compile and behave gracefully with no relay present. The `mcp_*` and `connector_enabled` / `relay_url` / `relay_api_url` fields in `Settings` remain (serde compatibility across tier switches — a user who switches from connected to free keeps their settings file intact with the connected fields as inert no-ops).

**Vite flag.** `VITE_CONNECTED` (string `"1"` / unset) controls whether `McpSettingsPane` and `ConnectionSettingsPane` render in the UI. `vite.config.ts` injects this as a `define`-replaced constant: in the free build the false branch of each `React.lazy()` dynamic import is dead-code-eliminated, dropping `McpSettingsPane` / `mcp-settings.ts` AND `ConnectionSettingsPane` / `connector-settings.ts` from the output bundle. (The live-status stores `mcp-server-info.ts` / `tunnel-status.ts` are imported by the always-mounted global event dispatcher, so they stay in the free bundle as inert no-op handlers — they only react to `mcp_*` / `tunnel_*` events the free build never emits.) The default is `"1"` when the env var is absent, so `npm run dev` and `vitest` keep current behaviour without any explicit flag. Verification: `VITE_CONNECTED= npm run build && grep -r "Enable MCP server" dist/` must return no matches.

**Windows build script.** `scripts/build-windows-app.ps1 -Features vulkan` builds the **connected** Vulkan artifact (the `connected` feature is default, so `--features vulkan` implicitly includes it). The free Windows Vulkan build would require `--no-default-features --features vulkan` passed via `$Features`. The `Makefile` `build-free` / `build-free-vulkan` targets show the canonical free invocation on Linux/macOS.

**Honest scope of the free-build claim.** The free artifact excludes `mcp-server`, `rmcp`, and any listening socket. It does NOT guarantee the absence of `hyper` — `hyper` remains via `model-registry → reqwest → hyper`. The claim is "no MCP server / no rmcp / no listening socket", not "no hyper".

**Connected-tier tunnel (`tunnel-client`).** The `connected`-feature gating extends to the app-side relay tunnel (WS4-A): `tunnel-client` is part of the connected surface (the free build has no relay), so `app-main`'s optional edge on it is gated by the same `connected` feature as `mcp-server`, added when the tunnel is wired in WS4-A S5. The crate itself lives in the workspace unconditionally (compiled by the workspace build / `cargo test`) and is simply not pulled into the free binary — the same pattern `mcp-server` followed before Phase 10. The tunnel does **not** add a listening socket: it dials OUTBOUND to the relay (no inbound port), and replays relayed requests against the existing loopback `mcp-server`. The internal `mcp_token` bearer doubles as the relay↔app secret here, applied app-side to the loopback replay only and never sent outbound to the relay (see the "Token storage and file permissions" / "Token lifetime and the connected-relay path" notes above). The tunnel **device credential** secret (`tunnel_device.json`, 0600) is introduced by S5 pairing, not S3b.

## Headless server daemon

The `headless` crate is a SECOND workspace binary (`minutist-hub`) beside
`app-main` — the user-installed headless server (WS4-B): an always-on
device-sync hub now, a GPU processing node post-launch. It is NOT a Tauri binary
and NOT a build variant of `app-main`; it is its own `cargo build` target, always
compiled under `cargo build --workspace` (like `sync`) but never linked by the
desktop free/connected `src-tauri` artefacts. The cross-cutting conventions apply
to it as follows:

- **Entry point + runtime.** `#[tokio::main]` multi-threaded scheduler in the
  daemon's own `main`; plain `tokio::spawn`, **never** `tauri::async_runtime::spawn`
  (there is no Tauri runtime here). All channels bounded, as everywhere.
- **No Tauri / IPC edges.** No `tauri::*` / `tauri-specta` / `ipc-bridge`
  imports (a reviewer finding if any appear). The daemon wires `sync::SyncEngine`
  directly; it carries no command/event surface.
- **Tracing.** Configured in the daemon's own entry point — a stderr writer
  (captured by journald under systemd) honouring `RUST_LOG`, defaulting to
  `info`, plus a rolling file appender under `{data_dir}/logs/` once the data
  root resolves. No `println!` / `eprintln!` for logging — the one-shot CLI
  subcommands (`print-ticket` / `add-peer`) do write their result to stdout via
  `println!`, but that is command output, not logging, and they do not initialise
  the subscriber so stdout stays clean. The relay access token is never logged
  (only whether one is set), mirroring the redacted `SyncConfig` Debug.
- **Configuration + data root.** The data root is an absolute path supplied at
  startup (CLI `--data-dir` / config file / env), resolved BEFORE settings load —
  it is a startup argument, not a settings field. Other settings read from a
  `settings.store` under the daemon's own root.
- **Operator surface.** The daemon has no GUI; pairing is via one-shot CLI
  subcommands. `print-ticket` binds the engine briefly and prints this device's
  pairing ticket to stdout (paste it into a desktop's Sync settings). `add-peer
  <ticket>` validates a peer's ticket and appends it to `{data_dir}/peers` (one
  ticket per line; `#` comments and blank lines ignored). The running daemon
  re-reads `peers` on a fixed interval, so a peer added while it runs is authorised
  without a restart — sync is mutual, so the peer must also add the hub's ticket. A
  Unix-domain-socket admin API is the eventual production surface; this CLI/file
  surface is the first cut.
- **Observability (test/CI instrumentation).** `status` prints the hub's state as
  JSON to stdout (endpoint id, relay, authorised peers, held meetings each with a
  content digest of their notes) — a pure filesystem read, no engine bind, no
  contact with the running daemon, so an automated harness uses it as a convergence
  oracle without `docker exec`'ing into the data dir. The digest is sha256 of the
  notes `ydoc` PROJECTED to canonical JSON, so it is stable across converged
  replicas (the raw CRDT encoding is not). The daemon emits a stable
  `minutist-hub ready` log marker once bound (a harness waits on it instead of
  sleeping). The timing constants are env-overridable in milliseconds
  (`MINUTIST_HUB_POLL_MS` / `MINUTIST_HUB_PUSH_DEBOUNCE_MS` /
  `MINUTIST_HUB_DISCOVERY_MS` / `MINUTIST_HUB_SHUTDOWN_GRACE_MS`) for a sub-second
  test mode, and
  `MINUTIST_HUB_LOG_JSON=1` switches tracing to a structured JSON formatter for
  field-level event assertions. Production defaults are unchanged.
- **Convergence (push-on-reconnect).** `SyncEngine` fires a bounded "peer arrived"
  broadcast (`subscribe_peer_events`) each time a peer opens an authorised inbound
  sync connection. The daemon reacts by calling `SyncEngine::push_all_to(peer)`,
  which reconciles EVERY meeting the hub holds back to that peer (relay-addressed,
  per-meeting notes + media, failures logged-and-skipped), debounced per peer. So a
  device that reconnects to deposit one meeting also collects every other meeting
  the hub accumulated while it was away — true convergence through the hub, not
  passive responding. The reciprocal push is **raced against the shutdown signal**,
  so a `SIGTERM` mid-push is honoured promptly (the push future is dropped — safe
  and idempotent, since notes writes are atomic and media is content-addressed). A
  `Lagged` peer-event (arrivals dropped under load) is recovered by reconciling
  EVERY known peer, so no arrival is permanently missed. The desktop does not
  subscribe to peer events.
- **Lifecycle discovery scheduling (§7 / recovery).** The processing-lifecycle
  exchange (`StreamKind::Discovery`, carrying `(MeetingId, ProcessingLifecycle)`)
  rides the sync flow rather than a separate skippable round: `push_all_to` runs a
  `discover_with` for the peer (a separate dial, last) after reconciling its
  notes/media (the desktop's `sync_now` does the same per peer), so a meeting's
  lifecycle follows the meeting it was pushed in. The hub additionally runs a
  periodic recovery sweep — `SyncEngine::discover_all` (re-discovers every known
  peer, relay-addressed, on the `MINUTIST_HUB_DISCOVERY_MS` interval, raced against
  shutdown) — so a lifecycle state a consumer dropped (broadcast overflow) or
  skipped (advertised before the meeting's folder had synced in) is re-applied on
  the next sweep. The hub's lifecycle CONSUMER runs in a dedicated task beside the
  serve loop (not inline), draining concurrently with the discovery/push awaits
  that emit into the same broadcast — draining in the serve loop would let a sweep
  larger than the channel cap self-lag while the loop is parked on the sweep
  producing it. The desktop has the ride-alongside but no periodic sweep (it
  recovers on its next `sync_now`); a dedicated desktop recovery driver is a later
  concern.
- **Single-writer per data root.** The daemon owns an entirely separate data root
  from any desktop (`settings.store`, `logs/`, `index.db`, `meetings/{uuid}/`
  under it; the device key at the root). Two processes must never share one root
  (libsql's WAL locking is the backstop, not a substitute for the discipline).
  See `containers.md` — "Process model".
- **Error boundary.** Uses `common::AppError` (or a daemon-local `thiserror` type
  converting to it) — never `IpcError` / `tauri-specta` shapes.
- **Trust position.** It holds meeting plaintext, but on the user's own hardware,
  so it sits in the same trust boundary as the desktop (the free-build D4 network
  claim is about Minutist-operated servers seeing content; this is the user's own
  machine). It is NOT the relay, which only ever brokers ciphertext.
- **GPU features (post-launch).** The GPU node forwards backend features to
  `llama-cpp-2` / `sherpa-rs` behind the same Cargo feature names as the desktop,
  conditional on the target GPU (a CPU-only build needs no GPU SDK); see "GPU
  portability". That phase adds the ML-runtime crate edges as a separate
  architecture-doc update.
- **Packaging artefacts.** Static repo files under `packaging/`, NOT built by the
  Rust workspace: a multi-stage `Dockerfile` (its builder stage also produces a
  glibc-compatible binary for bare-metal installs), a systemd unit
  (`minutist-hub.service` — `DynamicUser` + `StateDirectory`, `SIGTERM` stop), and
  the Windows service install via WinSW (`packaging/windows/`), which relies on the
  daemon's cross-platform Ctrl-C shutdown so no Windows-specific Rust code is
  needed. See `packaging/README.md`.

The cleanliness invariant for the free desktop build is unchanged: `cargo build
-p minutist --no-default-features` takes no edge to `headless` and pulls no new
deps. `headless` being a workspace member only means `cargo build --workspace`
compiles it — exactly as `sync` compiles unconditionally while the `app-main ->
sync` edge stays connected-gated.

## What's not decided here

These need decisions but are not yet binding:

- Whether `tracing` ships with a structured-JSON formatter (for future
  log analysis) or stays human-readable. Defer until first time we need
  to grep production logs.
- Auto-update mechanics for native libs (llama.cpp / sherpa-onnx) vs
  models. Currently both ride the app bundle. Revisit at phase 7.
