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
| Diarization (offline) | One-shot `spawn_blocking` task — the authoritative pass — spawned as a background job by `ipc-bridge` *after* stop (decoupled from the stop response), or by user action (re-diarize). |
| Diarization (live, Phase B) | Per-VAD-segment `OnlineDiarizer::assign_segment` driven from the runner's drain-loop thread (`spawn_blocking`) at SegmentEnd — gated on the `diarization_enabled` setting AND the embedding model being locally `Available` (no download, no block at start; the heavy `EmbeddingExtractor` load is built on `spawn_blocking` before the runner spawns). Best-effort/additive: any failure degrades to "no label" without affecting recording/transcription. See the live-vs-offline note below. |
| Summarisation | One-shot `spawn_blocking` task triggered by user action. |
| Persistence writes | `spawn_blocking` per write op for now; revisit if it shows up in profiling. |
| Tauri command handlers | Tokio worker threads. Short-lived, dispatch to the above. |

Bounded channels everywhere. Unbounded queues are not allowed — they
hide back-pressure that the live pipeline needs to surface.

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
rewrite (a background pass `ipc-bridge` spawns *after* stop, not inline in
`stop()`) overwrites the live labels with the offline result. The
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

- File appender at `{app-data}/logs/meeting-app.log`, rotated daily,
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

Within the Qwen branch, the 1.7B tier is used only when the user opts into the
GPU model (a `settings` flag), else the 0.6B. The orchestrator resolves the
engine once at recording start (and at re-transcribe) in
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

**Offline ops are serialized — but a new recording preempts them.**
`re_transcribe` and `rediarize` atomically CLAIM an internal `Offline` state
under the orchestrator lock (rejecting a concurrent re-transcribe / re-diarize
with `AppError::InvalidInput`) and release it on every exit path, so two offline
ops can't race and clobber the SAME meeting's `transcript.json`. However,
**`start` PREEMPTS the `Offline` claim** (`transition_start` accepts `Idle |
Offline`): a new recording is a different `meeting_id`/file, so the clobber
hazard does not apply, and the user must never be blocked from recording the next
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
path as the **runtime** env var `MEETING_APP_SILERO_PATH`; otherwise (a dev run
with no bundle) it leaves the var unset.

`vad-chunker::default_model_path()` reads, in order: (1) the **runtime**
`MEETING_APP_SILERO_PATH` (app-main's injected bundled path); (2) the
**build-time** `MEETING_APP_SILERO_PATH` (`option_env!`); (3) a source-tree path
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
  `common::Summariser::summarise` signature changed for #70 — `notes_markdown:
  &str` → `notes: &[NoteBlock]` — but the *progress* method stays concrete on
  `LlamaSummariser`, which `ipc-bridge` holds.) Cleared by `SummaryReady`.
- **`Rediarize` (indeterminate)** — the sherpa diarization `compute` is one opaque
  FFI call with no progress callback, so `fraction = None`. Cleared by
  `DiarizationComplete`.
- **`Finalise` (indeterminate)** — the post-stop drain is opaque, `fraction =
  None`. Cleared by `MeetingFinalised`.

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
`DiarizationComplete` / `SummaryReady` events. The auto-summarise progress is ALSO
surfaced inside the summary pane: when the user opens `SummaryView` while a
`summarise` op is in flight for that meeting (read from the operation-progress
store, keyed on `meeting_id` + `op == Summarise`), the determinate
`OperationIndicator` bar shows — even when the pane itself did not dispatch the
summarise — and `SummaryReady` then reveals the summary.

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
  baked into `transcript.json`.

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
  gates the listener, spawned once at startup from `app-main`'s `setup()` via
  `tauri::async_runtime::spawn`. `settings.mcp_port` is a FIXED default loopback
  port (8765, D1 — one instance runs, so a stable port keeps a saved client
  URL valid). Toggling at runtime is restart-required for v1.

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
  exactly what bearer + Host/Origin + loopback prevent. The token is stored at
  `{app-data}/mcp_token`, CREATED with mode `0600` atomically on Unix
  (`OpenOptions().mode(0o600)` — no write-then-chmod window); on Windows it
  inherits the per-user app-data directory's ACL (no extra file-ACL tightening in
  v1, so the owner-only guarantee is Unix-scoped). OS-keychain hardening is a
  documented follow-up. Rotating the token is restart-required for v1 (delete the
  file → restart regenerates it); there is no live regenerate command, consistent
  with the rest of the MCP lifecycle (enable / port / write-tools are also
  restart-required). The token is never logged and never on the event bus — it is
  revealed only via the `get_mcp_server_info` command on explicit user request.

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
{app-data}/
├── index.db                    libsql; owned by `persistence`
├── logs/                       tracing file appender; owned by `app-main`
├── meetings/{uuid}/            owned by `persistence` (and nobody else)
│   ├── audio.opus
│   ├── transcript.json
│   ├── notes.json
│   ├── notes.md
│   ├── summary.md
│   ├── metadata.json
│   ├── assets/                 pasted/dropped note images (content-hash files)
│   │   └── <sha256>.<ext>
│   └── chat/{session_id}.json  chat sessions (Phase 9)
├── models/                     owned by `model-registry` (and nobody else)
│   ├── asr/{model-id}/...      downloaded GGUF + mmproj per manifest entry
│   ├── llm/{model-id}/...
│   └── diarize/{model-id}/...
└── settings.store              owned by `settings` (JsonFileStore: serde_json + std::fs)
```

The model manifest is **not** written into the cache. It is bundled in the
binary (`resources/models.json`, loaded via `include_bytes!` in `app-main`
and parsed by `model_registry::load_manifest`); the cache dir holds only the
downloaded per-kind / per-model files.

Writes to a directory outside a component's owned scope are a review
finding.

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
  `MEETING_APP_ASR_MODEL_PATH` pattern) with a no-op skip path. They are run on
  demand either via `scripts/run-tests-windows.ps1` OR directly with
  `make test-integration` (and `-summary`/`-asr`/`-diarize`), which sources a
  git-excluded `tests-local.env` (copied from `tests-local.env.example`) holding
  the real model paths + a `MEETING_APP_RECORDINGS_DIR`. Running these against
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
  `MEETING_APP_ASR_MODEL_PATH`) runs alongside a timing-sensitive one, CPU
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
app-main checks on startup and emits `UpdateAvailable`; the webview prompts and,
on accept, emits the `updater://apply` event back; app-main then downloads
(emitting `UpdateProgress`), installs, and relaunches (`AppHandle::restart`).
All updater calls are **guarded** — the committed default `plugins.updater`
config is `{ "endpoints": [], "pubkey": "" }`, so `check()` is a logged no-op and
dev/unsigned builds are unaffected. Enabling updates is a release step: set the
real `endpoints` + minisign `pubkey` in `tauri.conf.json`, set
`bundle.createUpdaterArtifacts` to `true` (it is **unset** today, so
release builds currently emit no updater artefacts), keep the private key
as the `TAURI_SIGNING_PRIVATE_KEY` CI secret, and enable updater-artefact
signing in the release workflow. The app-wide Tauri 2 capability is `src-tauri/capabilities/default.json`
(`core:default` + `core:event:allow-emit`/`allow-listen`, scoped to the `main`
window) — without a capability a Tauri 2 webview has no IPC access at all, so
this is what lets the webview invoke the tauri-specta commands, receive
`AppEvent` payloads, and emit `updater://apply`. The build-time ACL
(`gen/schemas/capabilities.json`) is generated from it.

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
- **`On`** forces full GPU offload without consulting the probe (the old
  `true`).
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
`n_gpu_layers` resolution is already binary. Under `Auto` the plan **budgets the
summariser FIRST** (it stays resident while an ASR model loads when
`preload_summariser` is on), then budgets ASR against the **remaining** headroom
and downgrades the requested large ASR tier (`effective_prefer_large = false`)
when it would not fit alongside the summariser — running the 1.7B model purely on
CPU is strictly worse than the 0.6B CPU default. The decision base is
`total_bytes × headroom` (0.90 discrete, 0.50 integrated), **not `free_bytes`**:
a Vulkan device without `VK_EXT_memory_budget` reports `free == total`, so `free`
is trusted only to *tighten* the budget when it is a credible smaller number.
**A `None` probe (no GPU / probe failed) fails safe to CPU** — a false "fits"
risks an out-of-memory load or a silent host-memory spill. The VRAM thresholds
in `common` are estimates pending live-hardware evidence; `app-main` logs the
probe + the resolved default plan once at startup (`IpcState::log_gpu_probe`,
`target: "app-main"`) so the numbers can be validated against real devices.

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
a backend is chosen: `meeting-app` (src-tauri) → `ipc-bridge` → {`summariser`,
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

## What's not decided here

These need decisions but are not yet binding:

- Whether `tracing` ships with a structured-JSON formatter (for future
  log analysis) or stays human-readable. Defer until first time we need
  to grep production logs.
- Auto-update mechanics for native libs (llama.cpp / sherpa-onnx) vs
  models. Currently both ride the app bundle. Revisit at phase 7.
