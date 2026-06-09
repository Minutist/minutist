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
| `ipc-bridge` | 1 | `common`, `orchestrator`, `persistence`, `summariser`, `settings` |
| `app-main` (bin) | 1 | `common`, `orchestrator`, `ipc-bridge`, `model-registry`, `settings` |

Any PR adding an edge not in this table requires an architecture-doc
update in the same commit.

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
**Owns:** shared types (`MeetingId`, `ModelId`, `AudioChunk`, `Segment`,
`WordTimestamp`, `MeetingMeta`, `ModelDescriptor`, `RecordingState`,
`AppEvent`, `AudioDevice`, `AudioMeterFrame`, `AudioFormat`,
`ModelKind`, `ModelManifestEntry`, `ModelFileEntry`, `ModelStatusState`,
`ModelStatus`, `MeetingListEntry`, `NotesDocument`, `MeetingState`),
trait definitions (`AsrBackend`, `Diarizer`,
`Summariser`), the shared `AppError` enum + `AppResult<T>` alias.

The recorder-lifecycle additions `RecordingState::Finalising` and the
`AppEvent::{MeetingFinalised, TranscriptReady}` variants are documented with
their producers in the `orchestrator`/`ipc-bridge` "Responsive stop" and
re-transcribe notes below.

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
`option_env!("MEETING_APP_SILERO_PATH")` falling back to
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
(same mtmd path, official `ggml-org` GGUF + mmproj) as an optional
higher-accuracy / better-multilingual tier, selected only when the user opts
into the GPU model (see `settings`); the 0.6B remains the CPU default. Both
share the same `#21847` long-audio limitation, so the batched-VAD chunking is
mandatory for either.

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
(sherpa's "use threshold" sentinel, Spike 4) with `cluster_threshold`. The
orchestrator constructs the diarizer with `DiarizerConfig::default()`
(`num_clusters = None`, `cluster_threshold = 0.5`) for BOTH the on-stop pass and
the user-triggered re-diarize pass: at record time the speaker count is unknown,
so production uses threshold/auto-count mode to discover it rather than fixing a
cluster count. There is no `Some(1)` production path. `assign_speakers` rejects any
`sample_rate != 16000` with `AppError::InvalidInput`, short-circuits empty
audio/segments to `0`, runs `Diarize::compute`, and overlays via a pure
`overlay_speakers(&[sherpa::Segment], &mut [Segment]) -> u32`: per ASR segment it
picks the max-overlap sherpa turn (seconds→ms, half-open `[start_ms, end_ms)`;
ties resolve to the earlier turn; no overlap → `speaker_id = None`), relabels the
chosen `i32` cluster ids to first-seen-order `A`/`B`/… across segment slice
order, and returns the distinct-label count. The sherpa `eyre::Result` is mapped
to `Error::ModelLoad`/`Error::Inference` → `AppError::{ModelLoad,Inference{backend:"diarizer"}}`
at the boundary (eyre arrives transitively via `sherpa-rs`; no separate `eyre`
dep). `sherpa-rs = { workspace = true }` is added to `crates/diarizer/Cargo.toml`;
`hound` is a dev-dependency for the gated test's WAV decode. Tests: the default
suite covers `overlay_speakers` (interval-join, no-overlap=None, tie-break,
first-seen relabel, stale-label clearing) with no model; the env-var-gated
`tests/accuracy.rs` (`MEETING_APP_DIARIZE_SEG_PATH` + `MEETING_APP_DIARIZE_EMB_PATH`,
skip-on-unset) runs `assign_speakers` over committed
fixtures (`tests/fixtures/two_speakers_synth.wav` = two distinct real-speech
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
(`MEETING_APP_DIARIZE_EMB_PATH` only — no segmentation model — skip-on-unset)
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
    the wire-facing string). This is the `open_meeting` restore payload.
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
emits), rewrites `transcript.json` via `persistence::write_transcript` (atomic
tmp+rename; an empty result removes the file), and refreshes the index row
(`MeetingIndex::upsert`) so the meeting-list excerpt reflects the new first
segment, then emits `AppEvent::TranscriptReady { meeting_id }` so the webview
re-reads the transcript (mirroring `DiarizationComplete`). The ASR run is wrapped
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
(`MEETING_APP_DIARIZE_SEG_PATH` + `MEETING_APP_DIARIZE_EMB_PATH`, skip-on-unset)
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
with an env-var-gated (`MEETING_APP_DIARIZE_EMB_PATH`) positive case asserting
non-None live labels.

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

**Field — `gpu_acceleration: bool`.** The runtime GPU-acceleration toggle.
`#[serde(default = ...)]`-defaults to `true` (GPU on by default); an older store
written before the field existed deserialises to `true`, preserving the prior
compile-time behaviour. Added to the hand-written `Default` impl (`true`). GPU
offload happens ONLY when BOTH (a) the build was compiled with a GPU feature
(`vulkan`/`metal`/`cuda`/`rocm`) AND (b) this flag is `true`; when `false`,
inference runs on CPU (`n_gpu_layers = 0`) even in a GPU-feature build — the
runtime escape hatch for weak GPUs / driver trouble. In a default CPU-only build
the flag has no effect (inference is always on CPU). The orchestrator reads it
(`current().gpu_acceleration`) to resolve the live + offline-re-transcribe ASR
`n_gpu_layers`, and `ipc-bridge`'s `summarise_meeting` reads it to resolve the
summariser `n_gpu_layers`. No new dependency edge. See `cross-cutting.md` —
"GPU portability".

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
fix).** After the upsert, `stop_recording` runs up to two heavy passes OFF the
stop path, in order, in one fire-and-forget `tokio::spawn` (cloned
`Arc<Orchestrator>` + `Arc<MeetingIndex>`), so neither can wedge the stop
response or hide the meeting: (1) if `orchestrator.take_transcript_incomplete()`
is true — the live ASR dropped audio (drop-oldest flush queue) or its stop-drain
timed out — `re_transcribe` re-runs ASR over the complete `audio.opus`, the
authoritative transcript (the audio is captured in full regardless of live-ASR
speed); (2) if `orchestrator.diarization_enabled()`, `rediarize` runs AFTER any
re-transcribe so it labels the repaired transcript — each with its own
length-relative timeout (`retranscribe_timeout` / `diarize_timeout`). Both
refresh the index row and emit their events on completion (`TranscriptReady` /
`DiarizationComplete`; errors logged, claim-skips logged at info). While a pass
holds the offline claim the recorder reports the public `Finalising` busy state
(`Offline → Finalising` in `as_public`, broadcast by `claim_offline`/
`release_offline`), so the webview gates the Start button instead of enabling it
into an `InvalidInput` failure. And because a derived cache can always drift from
disk, `list_meetings` first calls
`MeetingIndex::reconcile_orphans(meetings_dir)` — a cheap `readdir` + set-diff
that lazily indexes any meeting folder present on disk but missing from the cache
(e.g. the process killed between finalise and the stop-time upsert) — so a
meeting can never stay hidden within a session, even without a restart. Reconcile
is best-effort (a failure logs and serves the cache as-is) and never deletes
(removals are reconciled by the next startup `rebuild_from_disk`).

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

The command ledger is now **21** (P5 20 + `rediarize_meeting`), asserted by the
`bindings_builder_registers_expected_command_ledger` test.

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

### `app-main` (bin)
**Crate:** `src-tauri/` (Tauri convention)
**Owns:** the Tauri main binary, tray icon, window management, process
lifetime. Wires the components into a running app.

The thinnest crate — code here should mostly be construction and
plumbing.

**Tracing:** file appender at `{app-data}/logs/meeting-app.log`, rotated
daily, 7-day retention via startup cleanup. Console layer in debug builds
only. `RUST_LOG`-style filtering via `EnvFilter::from_default_env()`.

**Tray menu:** "Open meeting-app" (show/focus main window) + "Quit"
(`app.exit(0)`). Left-click on the tray icon shows the main window.
Window close intercepts `CloseRequested` and hides rather than exits.

**Bindings harness:** `cargo run -p meeting-app --bin generate-bindings`
(alias: `cargo gen-bindings`) writes `ui/src/ipc/bindings.ts` without
starting the GUI. Run after any `ipc-bridge` command/event surface change.

## Webview components

The webview is small enough that ownership maps to directories rather
than packages.

| Component | Lives in | Owns |
|---|---|---|
| Notes editor | `ui/src/editor/` | Tiptap editor, markdown shortcuts, paragraph-anchor extension. |
| Transcript pane | `ui/src/transcript/` | Live-appending transcript view, hover/click cross-reference. Speaker chips carry a live colour dot when diarization labels are present (`speaker-color.ts`: deterministic `speaker_id` → palette slot; colour pairs with the visible label for accessibility). Consecutive rows are grouped: the labelled chip shows once at the start of a speaker's run; continuation rows keep only the colour dot. |
| Meeting shell | `ui/src/shell/` | Window chrome (start/stop/pause, audio meter, meeting list); the pane-visibility toggle; and the Settings drawer (`SettingsDrawer.tsx` — an Appearance group with the colour-theme control + the notes writing-paper-rules toggle, plus input device, transcription language, diarize-on-stop, GPU acceleration, system-audio capture). The summary is a workspace column, not an overlay. The capture/processing/appearance settings live in the drawer rather than the top bar so the masthead stays a single non-overflowing row. All drawer controls route through the existing settings seams; no new command. |
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
  through the `save_notes` IPC seam (`ui/src/ipc/notes.ts`). No-op when there is
  no active recording / MeetingId (FR-18).
- **HTML clipboard (`ui/src/editor/clipboard.ts`).** `buildClipboardPayload`
  produces a `text/html` (+ `text/plain`) copy payload — a self-contained UTF-8
  document with internal `data-anchor-ms` attributes stripped — so paste into
  Word retains formatting (FR-17). The editor overrides copy/cut via ProseMirror
  `editorProps.handleDOMEvents`.
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
  `application/x-meeting-app-segment`) carries a dragged transcript segment; the
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
  the labels on stop.
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
