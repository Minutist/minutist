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
no conversion. Audio-file *seek-to-anchor* (playing the audio at a clicked
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

**Offline ops are serialized.** `re_transcribe` and `rediarize` are offline
(require `Idle`) and now atomically CLAIM an internal `Offline` state under the
orchestrator lock (rejecting a concurrent start / re-transcribe / re-diarize with
`AppError::InvalidInput`) and release it on every exit path, so two offline ops
can't race and clobber `transcript.json`.

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
  multi-GB wait is not mislabelled "Summarising…".
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
│   └── metadata.json
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

Enabling a feature also offloads work to the device, but the layer count is a
**runtime** decision driven by the `settings.gpu_acceleration` flag (on by
default). GPU offload happens ONLY when BOTH (a) the build was compiled with a
GPU feature AND (b) `gpu_acceleration` is `true`. When the flag is `false`,
inference runs on CPU (`n_gpu_layers = 0`) even in a GPU-feature build — the
runtime escape hatch for weak GPUs / driver trouble. In a default CPU-only build
the flag has no effect (the compile-time ceiling is already `0`).

Wiring: `asr-runtime`'s `AsrRuntimeConfig` and `summariser`'s `SummariserConfig`
each carry a `n_gpu_layers: u32` field whose `Default` is the cfg-gated
compile-time ceiling (`default_n_gpu_layers()` / `gpu_layers()` → `u32::MAX`,
clamped to `i32::MAX` = "all layers", when a GPU feature is compiled in, else
`0`). The model-open site uses `config.n_gpu_layers` for `with_n_gpu_layers(...)`,
and the mtmd `use_gpu` is derived from `config.n_gpu_layers > 0`. The callers
resolve the runtime value from the setting: the orchestrator
(`runner::resolve_gpu_layers(enabled)`) for the live + offline-re-transcribe ASR
path, and `ipc-bridge`'s `summarise_meeting`
(`resolve_summariser_gpu_layers(enabled)`) for the summariser — each returning
the compile-time ceiling when the flag is on, `0` when off. llama.cpp falls back
to CPU at runtime when no device is present, so a GPU-feature build is still safe
on a CPU-only machine. (Before this, the layer count was purely compile-time; the
runtime flag now lets a GPU build be forced to CPU. Before *that*, the features
compiled a GPU backend but the code hard-coded `n_gpu_layers(0)`, so they
offloaded nothing.)

The features fan out through a single chain so the app binary is the only place
a backend is chosen: `meeting-app` (src-tauri) → `ipc-bridge` → {`summariser`,
`orchestrator` → {`asr-runtime`, `diarizer`}}. `ipc-bridge` is the fan-out point
because it sits above both `summariser` (direct dep) and `orchestrator` (which
owns `asr-runtime` + `diarizer`); the orchestrator does NOT depend on summariser
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
