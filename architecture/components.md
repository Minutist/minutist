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
| `asr-runtime` | 2 | `common`, `model-registry`, `settings` |
| `diarizer` | 6 | `common`, `model-registry` |
| `summariser` | 5 | `common`, `model-registry`, `settings`, `persistence` |
| `persistence` | 1 (minimal) → 4 (full) | `common` |
| `model-registry` | 2 | `common`, `settings` |
| `settings` | 1 | `common` |
| `orchestrator` | 1 (minimal) → 2 (live pipeline) | `common`, `audio-capture`, `vad-chunker`, `asr-runtime`, `diarizer`, `persistence`, `model-registry`, `settings` |
| `ipc-bridge` | 1 | `common`, `orchestrator`, `persistence`, `summariser`, `settings` |
| `app-main` (bin) | 1 | `common`, `orchestrator`, `ipc-bridge`, `model-registry`, `settings` |

Any PR adding an edge not in this table requires an architecture-doc
update in the same commit.

### Crates that grow across phases

- **`persistence`** appears in Phase 1 as a minimal writer of
  `audio.opus` + `metadata.json` to a per-meeting folder. Phase 4 grows it
  to the full surface: the folder readers (incl. the graduated
  pause-INCLUDING Opus decoder and the `MeetingState` assembler), the libsql
  `index.db` index + forward-only migration runner + `rebuild_from_disk`,
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

**Phase 4 precursors.** `MeetingListEntry` (meeting-list row, FR-33),
`NotesDocument { notes_json, notes_markdown }` (the canonical wire-facing
notes carrier — `String` fields because `serde_json::Value` has no
`specta::Type`; `ipc-bridge`'s local `NotesDoc` collapses into this), and
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

**Back-pressure policy:** the cpal-callback→forwarder channel is bounded
at 8 frames (drop-oldest with `tracing::warn!` on overflow); meter window
is 512 samples (~32 ms at 16 kHz, ~30 Hz emission rate).

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

**Implementation pattern (Phase 2).** `LlamaBackend` is a process-wide
`OnceLock` singleton; `LlamaModel` + `MtmdContext` are loaded once in
`AsrRuntime::new`; a fresh `LlamaContext` is allocated per
`transcribe_chunk` call (cheap, <100 ms) to guarantee a clean KV cache.
The `</asr_text>` early-stop checks the full concatenated detokenised
string, not per-token, so the tag is caught even when it spans a token
boundary.

### `diarizer`
**Crate:** `crates/diarizer`
**Owns:** sherpa-onnx binding, the embedding + clustering pipeline.

**Implements:** `Diarizer` from `common`.
**Inputs:** the full buffered audio + the segment array from ASR.
**Outputs:** mutates segments in place, setting `speaker_id`.

Post-hoc only. Not in the live pipeline; runs after the recording stops
or as a user-triggered re-diarize.

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
re-implemented. Differences from the live path: no flush queue / ASR-worker
thread — the work runs synchronously on one `spawn_blocking` thread, one
accumulator flush at a time, so segments can be collected in order. As segments
are produced it emits `AppEvent::TranscriptSegment` (the same event the live path
emits), rewrites `transcript.json` via `persistence::write_transcript` (atomic
tmp+rename; an empty result removes the file), and refreshes the index row
(`MeetingIndex::upsert`) so the meeting-list excerpt reflects the new first
segment. Unlike the live path's best-effort skip when no model is present, an
explicit user-triggered re-transcribe with no available model is an error
(`AppError::ModelLoad`). The orchestrator does not own a `MeetingIndex`; the
index handle is passed in by `ipc-bridge` (which owns it in `IpcState`).

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
rewrite + index-excerpt refresh are exercised in CI without a model.

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
(`(meeting_id) -> Option<NotesDoc>`, `None` when no notes saved). Unlike the
model/recording commands, these route **directly** to `persistence::NotesStore`
— `persistence` is now a real `ipc-bridge` dependency (already granted in the
table above) and the orchestrator is *not* involved: notes I/O is independent of
the live recording pipeline and may run concurrently with an active recording
(see `persistence` "Phase 3 surface growth — notes"). The blocking filesystem
write/read runs on `spawn_blocking`. `IpcState` carries a `meetings_dir:
PathBuf` (a clone of the same `{app-data}/meetings/` root the
orchestrator/persistence use), resolved and injected by `app-main`. The opaque
Tiptap document crosses the wire as a `String` (`NotesDoc { notes_json: String,
notes_markdown: String }`) because a bare `serde_json::Value` does not derive
`specta::Type`; `save_notes` parses the string to a `serde_json::Value` before
handing it to `NotesStore` and `load_notes` re-serialises the loaded value back
to a string.

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

**Phase 4 additions (18 commands total) — meeting list / open / actions.** Six
commands back the meeting-list view (FR-33):

- `list_meetings() -> Vec<MeetingListEntry>` — queries the shared libsql index
  (`MeetingIndex::list_meetings`, most-recent first).
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
- `re_summarise(meeting_id) -> ()` — a Phase-4 **stub** returning
  `AppError::Unsupported`; Phase 5's `summariser` fills it in (it will produce
  `summary.md` and emit `AppEvent::SummaryReady`).

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
| Transcript pane | `ui/src/transcript/` | Live-appending transcript view, hover/click cross-reference. |
| Meeting shell | `ui/src/shell/` | Window chrome, start/stop/pause, audio meter, meeting list. |
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
- **`MainWindow` (`ui/src/shell/`)** is now a collapsible AND resizable two-pane
  layout via `react-resizable-panels` (FR-21): notes editor primary, transcript
  pane secondary. A header toggle collapses/expands the transcript via the
  panel's imperative handle; a `Separator` provides drag-resize. The Phase 2
  two-column flex layout is replaced.
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
  the user edit the raw markdown and persist it. It is an optional third
  workspace pane in `MainWindow` — revealed by a header "Summary" toggle when a
  meeting is open or recording; the meeting it summarises is the open meeting
  else the live recording's `meeting_id`. The meeting-list row's Summarise
  action (renamed from the Phase-4 "Re-summarise" stub button) also runs the
  real summariser through the summary store.
- **Summary store (`ui/src/state/summary.ts`).** Transient UI state only
  (`summaryMarkdown`, `summarising`, `meetingId`, `lastError`) routed through the
  `ui/src/ipc/summary.ts` seam; `summary.md` on disk is authoritative. Its
  `handleEvent` is dispatched alongside `RecordingStore`/`ModelsStore` from
  `useAppEventBridge` and handles `AppEvent::SummaryReady` by re-reading the
  summary (`get_summary`) and leaving the in-progress state — scoped to the
  loaded meeting so an unrelated meeting's event does not clobber the view.
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
- **Two-pane sheet/transcript treatment.** The notes editor (`ui/src/editor/`)
  renders as a centered reading column on a `--sheet` page that lifts off the
  field with `--shadow-sheet` + a hairline edge. The transcript pane
  (`ui/src/transcript/`) is the quiet, recessed `--sheet-quiet` secondary
  column. The collapsible + resizable `react-resizable-panels` structure,
  panel `id`s (`notes` / `transcript`), and the transcript collapse toggle are
  unchanged. The top bar (`ui/src/shell/`) is calm and hairline-ruled: wordmark
  left, recording status focal (oxblood dot, gentle pulse only while recording,
  plus a tabular elapsed clock in `RecordingStatus.tsx`), grouped transport +
  slim meter + device affordance right.
- **Margin-anchor marginalia.** `ui/src/editor/anchor-marginalia.ts` is a
  **presentation-only** ProseMirror decoration extension: it renders each
  anchored paragraph's `data-anchor-ms` value as a quiet timestamp in the sheet's
  left-margin gutter (editorial side-note). It adds no node attributes and
  dispatches no transactions, so it cannot interfere with `ParagraphAnchor`'s
  stamping logic and never shifts the text column.
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
