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
| `notes-crdt` | WS4-B | `common` |
| `persistence` | 1 (minimal) → 4 (full) | `common`, `notes-crdt` |
| `model-registry` | 2 | `common` |
| `settings` | 1 | `common` |
| `orchestrator` | 1 (minimal) → 2 (live pipeline) | `common`, `audio-capture`, `vad-chunker`, `asr-runtime`, `asr-parakeet`, `diarizer`, `persistence`, `model-registry`, `settings` |
| `agent-tools` | 9 | `common`, `persistence`, `notes-crdt`, `rag-retrieval` |
| `chat-agent` | 9 | `common`, `summariser`, `agent-tools` |
| `mcp-server` | 10 | `common`, `agent-tools` |
| `tunnel-client` | WS4-A | (nothing in this workspace) |
| `account-directory` | B4 | `common`, `sync`, `tunnel-client` |
| `sync` | WS4-B | `common`, `notes-crdt` |
| `sync-ffi` | WS4-B (phone) | `common`, `sync`, `notes-crdt` ¶ |
| `election` | WS4-B (producer-gate) | `common`, `persistence`, `notes-crdt` |
| `doc-convert` | Attachments WS | `common` |
| `rag-retrieval` | RAG | `common` |
| `embedder` | RAG | `common`, `llama-cpp-2` |
| `ipc-bridge` | 1 | `common`, `orchestrator`, `persistence`, `notes-crdt`, `summariser`, `settings`, `agent-tools`, `chat-agent`, `doc-convert`, `embedder`, `rag-retrieval` |
| `app-main` (bin) | 1 | `common`, `orchestrator`, `ipc-bridge`, `model-registry`, `settings`, `agent-tools`, `mcp-server`†, `tunnel-client`‡, `sync`§, `election`※, `account-directory`Δ |
| `headless` (bin) | WS4-B | `common`, `persistence`, `notes-crdt`, `sync`, `tunnel-client`⊕ ‖, `account-directory` |

† `mcp-server` is an **optional** edge of `app-main`, gated by the `connected`
Cargo feature (default ON). The free artifact is built with
`--no-default-features` and omits `mcp-server` and its transitive rmcp stack.
See `cross-cutting.md` — "Build variants".

‡ See the `tunnel-client` component section for its dependency shape and
`connected`-feature gating.

§ `sync` is an **optional** edge of `app-main`, gated by the same `connected`
Cargo feature as `mcp-server` / `tunnel-client` (part of the connected-tier
surface — the free build does not sync). See the `sync` section below for its
dependency shape and protocols, and `cross-cutting.md` — "Build variants".
Account-mediated peer discovery (the `AccountEndpointSource` trait seam) is
documented in `crates/sync/src/lib.rs` and the `sync` section below, not
here — it adds no new dependency-table edge. `SyncControl::set_enabled(bool)`
(in `ipc-bridge`) gives the Settings toggle a runtime start/stop path for the
sync engine; `DisabledSync::set_enabled` is a no-op.

※ See the `election` component section for the `ElectionDriver` trait seam
and `Capability` injection.

Δ See the `account-directory` component section for what it adapts and who
shares it.

⊕ See the `headless` (bin) component section — `tunnel-client` is one of its
unconditional (non-feature-gated) dependency edges.

‖ See the `headless` (bin) component section for its status as a second,
independent workspace binary with no `app-main` edge.

¶ `sync-ffi` is the Android FFI wrapper over `sync` (phone companion):
UniFFI-exposes `SyncEngine`'s transport surface to Kotlin, cross-compiled to
`aarch64-linux-android`. Mobile-only; no edge from `app-main` or `headless`.
Takes no workspace edge beyond `common`, `sync`, and `notes-crdt`. Because it
has no `persistence` edge (issue 0016 — the phone stays off `persistence`'s
C-heavy graph), the phone's authoritative local writes (capture, rename)
reimplement `persistence::meeting_ops`' semantics over `notes-crdt` primitives
rather than calling it, including its metadata-then-CRDT lock ordering. See
`crates/sync-ffi/src/lib.rs`.

The notes-sync / media-sync / derived-artifact wire protocols, the pairing
lifecycle, and the third-party dependency rationale (`iroh`, `iroh-blobs`,
`iroh-tickets`, `yrs`) for `sync` live in `crates/sync/src/lib.rs`, not here.

Any PR adding an edge not in this table requires an architecture-doc
update in the same commit. The table tracks **runtime** edges only;
test-only dev-dependencies (e.g. `diarizer → persistence` and
`diarizer → hound` for gated audio-decode tests) are documented in prose
where they are used, not added here.

### Crates that grow across phases

- **`persistence`** writes `audio.opus` + `metadata.json` + `transcript.json`
  to a per-meeting folder and owns the full read/index surface (folder
  readers, the `index.db` migration/rebuild/self-heal cycle, meeting
  rename/delete, collections, RAG cache, voiceprint library). It mirrors
  every authoritative `metadata.json` write into the notes CRDT's meta map
  so sync peers converge on real values instead of a sync-arrival
  placeholder's invented ones (lock-ordering rule against the sync-receive
  projection path documented in its own module doc). Depends on `common`
  and the `notes-crdt` leaf; libsql/tokio remain external crates, not
  workspace components.
- **`orchestrator`** is the state machine for
  start / stop / pause with the audio meter and capture lifecycle, driving the
  full live pipeline (VAD → ASR → transcript events → diarizer trigger).
- **`ipc-bridge`** exposes start/stop/pause commands,
  device-list query, audio-meter and state-change events, plus the
  domain-surface commands catalogued below.
- **`asr-runtime`** and **`vad-chunker`** are a unit — together they form the
  transcription pipeline; `audio-capture` alone captures audio but does not
  transcribe.

## Rust core components

### `common`
**Crate:** `crates/common`

The workspace's shared kernel: the one leaf crate every other crate may
depend on, and the dependency-inversion seam for the whole app. A trait
defined here (`AsrBackend`, `Summariser`, `DocVlm`, `Embedder`) lets a
consumer crate call model-backed inference without a workspace edge to the
crate that implements it; the concrete impl is injected by whichever crate
is allowed to hold both edges. The same pattern keeps heavyweight or
cross-cutting dependencies out of this crate: `HostRef` is an opaque
`String` newtype so `sync`'s `iroh::EndpointId` never has to appear here,
and timestamps are RFC 3339 strings throughout so no time crate enters
`common`.

**Owns:** the shared domain types (meeting/session/chat/attachment/model
identifiers and records, the event bus `AppEvent`, the error type
`AppError`), the architecture's trait definitions (`AsrBackend`,
`Summariser`, `DocVlm`, `Embedder`), pure cross-cutting logic (GPU VRAM
placement, the live-agent gate, ASR-engine selection, voiceprint vector
maths), and the shared atomic-write (`fs::write_atomic`) and
process-wide-`LlamaBackend` primitives other crates build on.

**Dependency rule:** `common` depends on nothing else in this workspace.
Every other crate may depend on `common`; `common` may never depend on
them. Adding, removing, or changing a public item here is an
architecture-owner decision and requires a `components.md` update in the
same commit — the trait signatures and `AppEvent` variants are the stable,
locked surface that parallel sub-agents implement and consume against.

See `crates/common/src/lib.rs` for implementation detail.

### `audio-capture`
**Crate:** `crates/audio-capture`

Cross-platform audio input via cpal — microphone plus optional system-audio
loopback, resampled to 16 kHz mono, level-metered, and delivered to the
orchestrator over bounded async channels.

**Owns:** the audio device, sample-rate negotiation, the capture ring buffer
(drop-oldest under back-pressure, sized to ride the model-load burst at
record start), device enumeration/identity for the settings UI, the Windows
WASAPI-communications-mode fallback path, and the system-audio loopback +
mixer (Windows-only loopback source, sample-wise sum with clamping,
per-source drift tolerance).

**Dependency shape:** `common`-only leaf.

See `crates/audio-capture/src/lib.rs` for implementation detail.

### `vad-chunker`
**Crate:** `crates/vad-chunker`

Silero VAD v4 based speech chunker — turns the raw 16 kHz mono sample
stream from `audio-capture` into speech segments with onset/hangover
smoothing.

**Owns:** the Silero VAD model lifecycle (via `vad-rs`), the 480-sample
framing/partial-buffer accumulator, the smoother state machine,
force-splitting at `max_segment_ms`, timeline-discontinuity handling
(frame-clock re-anchoring, stale-buffer discard on a gap), and the
`reset()` hard-boundary restore used by offline re-transcribe at recording
pauses. The Silero ONNX model is a vendored resource (`resources/silero/`),
not a `model-registry` download.

**Dependency shape:** `common`-only leaf.

See `crates/vad-chunker/src/lib.rs` for implementation detail.

### `asr-runtime`
**Crate:** `crates/asr-runtime`

llama-cpp-2 mtmd binding driving Qwen3-ASR (0.6B default, 1.7B GPU-tier
sibling) as the general-language ASR backend.

**Owns:** `AsrBackend` implementation, the process-wide
`LlamaBackend`/`MtmdContext` lifecycle (per-call fresh `LlamaContext` for a
clean KV cache), the 30 s mtmd encoder-window constraint (the orchestrator,
not this crate, shapes calls via batched-VAD chunking to respect it),
prompt/template construction including the language-hint prefix-force and
the `</asr_text>` stop condition, and the auto-language spurious-CJK guard
(rolling script-history ring buffer, forced-English retry gated on
plausibility + logprob margin).

**Dependency shape:** `common`-only leaf.

See `crates/asr-runtime/src/lib.rs` for implementation detail.

### `asr-parakeet`
**Crate:** `crates/asr-parakeet`

NVIDIA Parakeet TDT 0.6B v3 via sherpa-onnx, the European-language ASR
backend with per-word timestamps.

**Owns:** the sherpa-onnx offline-transducer binding (direct C API via
`sherpa-rs`'s `sys` feature, to recover timestamps that
`TransducerRecognizer::transcribe()` drops), token→word/segment timestamp
aggregation, and the degenerate-repetition output guard (single-word
dominance / low distinct-word-ratio → drop the chunk rather than emit a
hallucinated loop). Separate crate from `asr-runtime` to keep the
sherpa-onnx/Parakeet domain distinct from the llama-cpp-2/Qwen domain
(sherpa-onnx's second consumer after `diarizer`); both implement
`AsrBackend` and the orchestrator routes between them by language (25
European languages here, everything else + auto-detect to `asr-runtime`)
via `common::asr_engine_for_language`. Parakeet is CC-BY-4.0 licensed,
distinct from the Apache-2.0 Qwen models.

**Dependency shape:** `common`-only leaf.

See `crates/asr-parakeet/src/lib.rs` for implementation detail.

### `diarizer`
**Crate:** `crates/diarizer`
**Owns:** the sherpa-onnx diarization binding — the offline (authoritative)
`SherpaDiarizer` embedding + clustering pipeline that assigns `speaker_id`
to ASR segments, the additive live `OnlineDiarizer` hint, and the
voiceprint centroid maths (`Voiceprint`/`VoiceprintExtractor`) built on
`common::voiceprint_math`.

**Inputs:** the full buffered audio + the segment array from ASR, plus
resolved model paths. **Outputs:** mutates segments in place, setting
`speaker_id`.

**Dependency boundary: `common` only** (deliberate). It takes RESOLVED
model paths — it never resolves a model id itself, and never touches
`persistence` at runtime (dev-dependency only, for gated fixture tests).
All `model-registry` resolution and `persistence` audio-decoding live in
the orchestrator (`runner::build_diarizer`, `runner::build_online_diarizer`),
which passes resolved paths / plain PCM in. This keeps the diarizer a pure
compute layer, testable without a `persistence` root or a `model-registry`
handle. The `orchestrator → diarizer` edge is the only crate-dependency
edge this component participates in.

The offline pass is authoritative for the finished transcript; the online
pass is a live, non-retroactive hint used only until the offline pass
overwrites it.

See `crates/diarizer/src/lib.rs` for implementation detail.

### `summariser`
**Crate:** `crates/summariser`

Local-LLM text summarisation — implements `common::Summariser` by driving
a GGUF text model through `llama-cpp-2`.

**Owns:** llama-cpp-2 text-LLM lifecycle (process-wide `LlamaBackend`
singleton, `LlamaSummariser::open` loads the GGUF once, fresh
`LlamaContext` per call), the model-agnostic chat-template rendering path
with a hand-built Gemma-turn-format fallback for when the bundled
llama.cpp can't render a template newer than itself, chunked prefill
(`plan_prefill`, mandatory per `n_batch`), notes-weaving into a single
chronological timeline when notes are anchored to the recording clock,
the attachments-feed prompt section (leading, non-time-woven,
byte-identical when empty), `translate_segment` and
`generate_attachment_awareness` as concrete auxiliary generation
surfaces, the `model()` accessor lent to `chat-agent`, the lazily-built
vision `MtmdContext` for document OCR, and the optional `external-ollama`
dispatcher (`OllamaSummariser`, feature-gated, pulls in
`reqwest`/`serde` only under that feature).

**Dependency shape:** depends only on `common` (plus `llama-cpp-2`); no
new dependency edge from any of the features above — the model-exposure
accessor for `chat-agent` and the vision surface are both additive
without new crate edges.

See `crates/summariser/src/lib.rs` for implementation detail.

### `model-registry`
**Crate:** `crates/model-registry`

The on-disk model cache, model-manifest schema, and download/verification
pipeline for ML model files.

**Owns:** the only crate allowed to write under `{app-data}/models/`;
parses `resources/models.json`; per-kind cache layout (`asr/`, `llm/`,
`diarize/`); in-flight `ensure()` deduplication via a
`tokio::sync::watch` channel; acts as a first-class `AppEvent` source by
emitting `ModelDownloadProgress` on the same shared broadcast channel the
orchestrator uses, throttled to ~10Hz and reported against each entry's
aggregate byte total so multi-file models drive one monotonic bar;
verification failures return to the `ensure` caller rather than the
broadcast bus; manifest URLs must pin an immutable commit revision.

**Dependency shape:** a leaf-ish crate consumed by `app-main`/`ipc-bridge`
for model resolution; no unusual edges.

See `crates/model-registry/src/lib.rs` for implementation detail.

### `notes-crdt`
**Crate:** `crates/notes-crdt`

The notes-CRDT primitives shared by `persistence` and `sync` — the Yjs
(`yrs`) leaf carved out so mobile sync can cross-compile without the
C-heavy graph.

**Owns:** `ydoc` (the Yjs representation + lossless ProseMirror-JSON
conversion + v1/v2 encoding), `NotesStore` (read/write `notes.ydoc`,
derive `notes.json`/`notes.md`), `MeetingFolder` (on-disk `{root}/{uuid}/`
layout, `ensure`), the `metadata.json` reader/atomic-writer plus the
guarded read-modify-write (`update_metadata_if_present` under
`metadata_lock`), `notes_lock` (a dedicated per-meeting lock serialising
the three `notes.ydoc` writers, kept separate from `metadata_lock` since
the two files have independent writers), `apply_synced_lifecycle_if_present`
(the single implementation of inbound-lifecycle merge-and-skip, reused
directly by `sync-ffi` and re-exported by `persistence::meeting_ops`), and
`folder::list_meeting_ids`/`parse_meeting_dir` (the one enumeration of
"which meetings this device holds", shared by `sync`, `headless`, the
election loop, and `persistence`'s own directory scans).

**Dependency shape:** a leaf — depends only on `common` among workspace
crates (`yrs`, `chrono`, `serde`, `thiserror`, `tracing`; no
`libsql`/`audiopus`/`ogg`) — because `sync` depends on `notes-crdt` rather
than `persistence` and must cross-compile to mobile targets. `persistence`
depends on this crate and re-exports every symbol at its existing paths,
so its own callers are unaffected; `notes_crdt::Error` is a light subset
that `persistence::Error` absorbs via `From`.

See `crates/notes-crdt/src/lib.rs` for implementation detail.

### `election`
**Crate:** `crates/election`
**Owns:** the host-election state machine for capture-but-unprocessed
meetings — the producer-gate. An eligible host claims a pending meeting (or
reaps one whose lease expired), holds a renewable lease while it runs the
pipeline, and writes `Processed`, all propagated over the existing Discovery
exchange.

**Dependency edges:** a leaf on `common` + `persistence` + `notes-crdt` only.
The two collaborators the loop must drive — the `sync` `SyncEngine` (to
advertise) and the `orchestrator` (to reprocess) — sit behind the
`ElectionDriver` trait, so this crate carries no `sync` / `orchestrator` /
`tauri` / `ipc-bridge` edge; the ONE state machine is reused by both eligible
host types (desktop-with-GPU and the future headless GPU node) and is
unit-testable with a mock driver. `Capability` (may this host claim?) is
computed by the binding crate (`app-main` / `headless`, which link the GPU
probe) and passed in. `app-main`'s `DesktopElectionDriver` implementation is
gated behind the same `connected` Cargo feature as `sync` / `tunnel-client`.

See `crates/election/src/lib.rs` for implementation detail (the lease-aware
supersession rules, convergence behaviour, correctness tests).

### `persistence`
**Crate:** `crates/persistence`
**Owns:** the sole on-disk state under `{app-data}/meetings/{uuid}/`,
`{app-data}/index.db`, `{app-data}/collections.json`, and
`{app-data}/voiceprints.db` — no other crate in the workspace reads or
writes those paths.

**Depends on:** `common` and the leaf `notes-crdt` crate (the `yrs`/Yjs CRDT
machinery for the authoritative `notes.ydoc` and the per-meeting folder
layout, `MeetingFolder`), which `persistence` re-exports at its historical
paths. The split keeps `persistence`'s C-heavy dependency graph (libsql,
Opus, Ogg) out of `sync`'s compile path, since `sync` needs to transport the
CRDT without pulling in the rest. `yrs` is pure Rust with no network
surface and is embedded in both build variants; only the sync transport is
`connected`-gated.

**Inputs:** typed write commands from `orchestrator` and `ipc-bridge`.
**Outputs:** typed read responses; emits no events itself.

**Write surface:** `audio.opus` + `metadata.json` + `transcript.json`
(`MeetingWriter`), the notes CRDT (`notes.ydoc`, authoritative) plus its
derived `notes.json`/`notes.md` projections, note-image assets,
attachments, chat sessions, translations, and `summary.md`.

**Read/index surface:** the folder readers (including the audio decoder and
the `MeetingState` assembler), the libsql `index.db` meeting index with a
forward-only migration runner, `rebuild_from_disk`, and the self-heal
`reconcile_orphans`; rename/soft-delete/restore/purge meeting operations
(trash — a soft delete only flips `MeetingMeta.deletion`, leaving the folder,
voiceprints, and blobs untouched until restore or purge; purge is the
destructive op, also driven by an automatic 7-day sweep, and records a
`purged.json` tombstone so a hub's peer-adopt sweep can never resurrect it —
see `crates/persistence/src/meeting_ops.rs` and `src/purged.rs`); a
`collections.json` store; a per-meeting RAG cache (`meeting.db`); and a
separate `voiceprints.db` speaker-voiceprint library.

Every write to `metadata.json`'s authoritative fields also mirrors into the
notes CRDT's meta map, so a sync peer that only ever received a
placeholder folder converges onto real values.

See `crates/persistence/src/lib.rs` for implementation detail.

**"New meeting" prep drafts — a meeting that exists before its audio does.**
Creating a new meeting and starting its audio capture are now two separate
steps rather than one: `writer::create_draft(root, meeting_id) ->
AppResult<MeetingFolder>` builds the on-disk shell (folder,
`metadata.json` with `recording_started: false` + an empty title, and a
`notes.ydoc` seeded via `notes_crdt::meta_crdt::initialise_notes_with_meta`)
with NO audio file — the Attachments feature and the notes editor both need a
real `MeetingId` + folder to exist before capture starts, so a prep draft is a
real durable meeting, not a client-side-only placeholder. Two `MeetingWriter`
entry points share a `start_capturing` body that opens `audio.opus` + the
encoder + the transcript writer, then RMWs `metadata.json` (never a raw
overwrite — a promoted draft may already carry a real title/collection from
the prep phase) to flip `recording_started: true` and stamp the real capture
`started_at`/`audio_format`, mirroring the same two fields into the meta CRDT
via new granular `meta_crdt::set_started_at`/`set_audio_format` setters:
- `MeetingWriter::open(root, meeting_id, format)` — the auto-start-immediately
  path: `create_draft` then `start_capturing`, back-to-back, no visible gap.
  Existing external contract unchanged (still mints-and-opens in one call).
- `MeetingWriter::open_for_recording(root, meeting_id, format)` — promotes an
  EXISTING draft (`notes_crdt::MeetingFolder::open_existing`, which errors
  rather than creates if the folder is absent).

`finalise()` correspondingly never (re)initialises the meta CRDT map — since
`notes.ydoc` now always already exists by the time it runs (seeded at draft
creation, for every meeting, no exceptions) — it mirrors `ended_at` /
`duration_ms` / `speaker_count` / `asr_model` / `llm_model` / `diarizer` /
`speaker_names` via `edit_meta_ydoc` + the existing granular setters, same
lock-ordering rule as `meeting_ops::rename_meeting` (RMW first, un-nested).
`title`/`started_at`/`audio_format` are untouched at finalise — they are
draft-creation/promotion-owned, so a concurrent rename during a live
recording (now genuinely possible, since a draft's `notes.ydoc` syncs during
prep, before this device's own recording even starts) cannot be clobbered.

`MeetingMeta.recording_started: bool` (`#[serde(default = "..")]` = `true`)
is the "never recorded" signal on THIS device, deliberately independent of
the `duration_ms == 0` sync-placeholder ambiguity issue 0052 disambiguated —
`notes_crdt::MeetingFolder::ensure`'s inbound-sync placeholder and orphan
recovery (`synthesize_metadata`) both explicitly set it `true` as the safe
default. It is NOT carried in the meta CRDT and so does not converge: a peer
that syncs an origin's still-unpromoted draft (title/notes with no audio)
reads `true` regardless of the origin's real `false` — a known gap, tracked
alongside the wider open question of whether an unpromoted draft should sync
at all before it has real content (today it does, from `create_draft`
onward, unconditionally). Mirrored onto `MeetingListEntry` (`index.db`
migration 3: `recording_started INTEGER NOT NULL DEFAULT 1`) and surfaced as
a "Draft" chip in `MeetingList.tsx` (`meeting.recording_started === false`)
so an abandoned/unpromoted draft is at least identifiable in the list, since
nothing yet garbage-collects one.

### `orchestrator`
**Crate:** `crates/orchestrator`
**Owns:** the live recording state machine (start/pause/resume/stop), the
level meter, and the live pipeline: `audio-capture → vad-chunker →
asr-runtime → persistence`, with an additive live-diarizer hint layered on
top. Kicks off `diarizer`'s on-stop pass, and offers offline companion
operations (re-transcribe, re-diarize, reprocess, read-only relisten/window
extraction, voiceprint enrolment) over a finished meeting. Fans out
`AppEvent` (transcript segments, meter level, state changes, recording
clock, diarization complete, …) on a shared broadcast channel that
`ipc-bridge` forwards to the webview.

**Thorniest crate.** It is the one place `audio-capture`, `vad-chunker`,
`asr-runtime`, `diarizer`, `model-registry`, `persistence`, and `settings`
are wired together into one state machine. A change to the live pipeline's
shape (buffering, flush cadence, event ordering) is an orchestration-owner
decision here — parallel changes to an upstream crate alone cannot settle
it without an architecture-doc update.

**Dependency shape:** `orchestrator → { audio-capture, vad-chunker,
asr-runtime, asr-parakeet, diarizer, model-registry, persistence,
settings }`. It is the only crate below `ipc-bridge` allowed to depend on
`model-registry`, `diarizer`, and `persistence` simultaneously — the live
pipeline needs model resolution, audio persistence, and diarization
clustering together, which keeps the `model-registry` edge out of
`diarizer` and `agent-tools`. No `orchestrator → summariser` edge (the
summariser is loaded by `ipc-bridge`) and no `tauri::*` import (Tauri glue
stays in `ipc-bridge`/`app-main`).

See `crates/orchestrator/src/lib.rs` for implementation detail.

**"New meeting" prep drafts (see `persistence` above).** `start(meeting_id,
device_id) -> AppResult<MeetingId>` no longer mints its own id —
`transition_start` takes a caller-supplied `MeetingId` (an existing draft,
created by `ipc-bridge`'s `create_meeting` command or, for the
auto-start-immediately setting, that same command called immediately
beforehand) and calls `MeetingWriter::open_for_recording` instead of `open`.
No new `RecordingState` variant: a draft with no active capture is simply
never represented there (it stays `Idle`); the orchestrator only learns a
meeting_id exists at the moment `start()` promotes it. `stop()`'s title
fallback now has three tiers, not two: the LIVE `pending_title` (typed during
the recording via `set_recording_title`) wins if set; else whatever title is
already on disk (e.g. one set via `rename_meeting` during the prep phase,
before recording ever started — `stop()`'s `metadata.json` write is a raw
overwrite of the `MeetingMeta` it builds, so this must be read back
explicitly or a prep-phase title would be silently clobbered by the
synthesized default); only when neither is set does it fall back to
`Recording <timestamp>`.

### `agent-tools`
**Crate:** `crates/agent-tools`
**Owns:** the shared tool layer — one `Tool` trait + one `ToolRegistry`, the
single place a chat-agent / MCP tool is defined. Both consumers (the internal
chat agent and the MCP server) drive the same registry, so "the internal
agent and an external MCP client use the same tools" holds by construction.
24 v1 tools: read/compute (list/search/get meetings, transcript, summary,
notes, metadata, recording state, retrieval, relisten, resummarise, talk
time, attachments), two MCP-allowlisted metadata writes (`set_speaker_name`,
`rename_meeting`), `reprocess_meeting` (internal-only), four write-gated
record-control tools, and (MCP registry only) `send_to_internal_agent`.

**Dependencies:** `common`, `persistence`, `notes-crdt`, `rag-retrieval`.
Deliberately no `orchestrator` edge — recording-lifecycle tools drive
`RecordingControl`, a trait object the crate defines; the caller building
`ToolContext` (`ipc-bridge`/`app-main`) supplies a real `Orchestrator`-backed
impl (orphan-rule newtype adapter). Deliberately no `summariser` edge
(`Arc<dyn Summariser>` held on context instead) and no `model-registry`,
`tauri`, or `specta` edge. The two metadata-write tools serialise via
`notes_crdt::metadata_lock`, not a lock this crate owns.

See `crates/agent-tools/src/lib.rs` for implementation detail.

### `chat-agent`
**Crate:** `crates/chat-agent`
**Owns:** the stateless, OpenAI-compatible, tool-calling chat turn engine
over the bundled local LLM, plus the held-context live-session engine
(`live.rs`) for the in-meeting co-pilot — prefix-once prefill, prune-to-prefix
refresh, and a KV-checkpoint snapshot/restore path
(`state_seq_get_data_ext`/`state_seq_set_data_ext`) gated behind real-model
round-trip tests before promotion to the active prune mechanism.

**Dependencies:** `common`, `summariser`, `agent-tools` (+ `llama-cpp-2`).
Sits above both `summariser` (the loaded-model substrate) and `agent-tools`
(the tool descriptors) so the loop never has to fold backward into either.
No `tauri`/`specta`, no direct `persistence`/`orchestrator` (reached only
through `agent-tools`), no `model-registry` (reuses the already-loaded model
via the `summariser` substrate seam). Engine types (`ChatEngine`,
`ChatMessage`, `TurnOutcome`, `SamplerConfig`, `TurnBackend`) live here
rather than `common` because no `common`-level signature names them —
unlike `Summariser`, which `agent_tools::ToolContext` does name and so stays
in `common`.

See `crates/chat-agent/src/lib.rs` for implementation detail.

### `mcp-server`
**Crate:** `crates/mcp-server`
**Owns:** the in-process Streamable HTTP MCP server exposing the
`agent-tools` registry to external agents over loopback. A second consumer
of that registry — it adds no tools of its own, projecting
`ToolRegistry::mcp_tool_descriptors_gated` onto `tools/list` and
`ToolRegistry::dispatch` onto `tools/call`; any tool logic living here
instead of `agent-tools` is a reviewer finding.

**Dependencies:** `common`, `agent-tools` (+ `rmcp` 1.7 and its hyper/http/
tower leaf crates — the one place rmcp error types are constructed). rmcp's
own hyper-based `StreamableHttpService` serves the endpoint directly, no
`axum`. No `chat-agent` edge (the inter-agent bridge reaches the chat engine
through a channel on `ToolContext`, serviced by `ipc-bridge`), no
`tauri`/`specta`, no direct `persistence`/`orchestrator` edge. Because it
depends on `agent-tools` but not `ipc-bridge` (which owns the real
`RecordingControl` adapter), its own tests/example each define a local
`TestRecordingControl` newtype instead of reaching for `ipc-bridge`'s, to
avoid inverting the crate layering.

See `crates/mcp-server/src/lib.rs` for implementation detail.

### `settings`
**Crate:** `crates/settings`
**Owns:** the settings schema, persistence, defaults, and change
notification. `Settings` is backed by a single JSON file
(`{app-data}/settings.store`), read/written whole via `serde_json` +
`std::fs`. Every field carries `#[serde(default = ...)]` so an older store
deserialises missing fields to their default rather than failing — the
schema only ever grows, and a handful of fields use `deserialize_with` to
migrate an old on-disk shape (e.g. `gpu_acceleration`'s legacy bool,
`live_agent_system_prompt`'s legacy prompt upgrade) in place of a
store-wide version field.

**Dependencies:** no `tauri::*` — the resolved `PathBuf` is injected by
`app-main` at construction. Change notifications broadcast via
`tokio::sync::watch` (capacity 1) directly from `SettingsHandle::update`,
not through the orchestrator. Single source of truth for runtime
configuration; nobody else parses the store directly.

See `crates/settings/src/lib.rs` for implementation detail.

**`auto_start_recording_on_new_meeting: bool`** (`#[serde(default = ...)]` =
`false`, the same default for a fresh store AND an older store missing the
field — the new prep-first default applies to everyone, the same pattern
`auto_summarise_on_stop` uses). Off: a new meeting opens the "New meeting"
prep screen (`ipc-bridge`'s `create_meeting`); on: recording starts the
instant a new meeting is created (the legacy behaviour, restored as an
explicit opt-in). See `persistence`/`orchestrator` above.

### `doc-convert`
**Crate:** `crates/doc-convert`
**Owns:** converting attached document bytes to canonical markdown
(`convert_to_markdown`, `supported_exts`), one converter per extension
(txt/md passthrough, csv/tsv, json/yaml/xml/log as fenced blocks, xlsx/ods
via `calamine`, html/htm via `dom_smoothie`+`htmd`, eml via `mail-parser`,
pdf via `pdf_oxide`, pptx/docx via `zip`+`quick-xml`, images via VLM OCR
only). Every conversion runs inside `catch_unwind` with a 50 MiB input cap
and a zip-decompression bomb guard checked before any parser sees the bytes.

**Dependencies:** `common` only (for `AppError`/`AppResult` and the
`DocVlm` trait it calls through for image OCR without taking a workspace
edge on the implementing crate). `image` is a third-party dep for
image-attachment decode/re-encode, not a workspace edge — the
`common`-only rule holds.

See `crates/doc-convert/src/lib.rs` for implementation detail.

### `tunnel-client`
**Crate:** `crates/tunnel-client`
**Owns:** the app-side account client for the connected tier — the RFC 8628
device-code pairing client (`DeviceCodeClient`) used to sign a device in, plus
the account device-directory HTTP client (`AccountDirectoryClient`, `GET
/v1/account/devices`, `PUT .../self/endpoint`, `DELETE /v1/account`). Kept
here (not in `sync`) so this crate stays a `sync`-free near-leaf; `app-main`
and `headless` both adapt it onto `sync::AccountEndpointSource` via the
`account-directory` crate. No outbound relay dial lives here any more — the
former WSS-tunnel/frame-relay machinery (that let a hosted relay reach the
local `mcp-server`) was removed 2026-08-24 when that hosted relay was retired
(D15); see issue 0044.

**Dependency edges:** none in the workspace — a near-leaf over third-party
crates only (`reqwest`). Part of the connected feature surface: compiles
unconditionally as a workspace member; `app-main`'s edge to it is wired only
behind the `connected` Cargo feature, `headless`'s edge is unconditional (a
seeded hub is always account-capable).

See `crates/tunnel-client/src/lib.rs` for implementation detail (the account
client's wire contract and security invariants).

### `account-directory`
**Crate:** `crates/account-directory`
**Owns:** the ONE adapter from `tunnel_client::AccountDirectoryClient` (an
HTTP client) onto `sync::AccountEndpointSource` (the trait `sync` depends on
instead of an HTTP client), so `sync` and `tunnel-client` never take an edge
on each other.

**Dependency edges:** a leaf on `common`, `sync`, `tunnel-client`. Shared by
`app-main` (its account-refresh wiring) and `headless` (its account-discovery
startup path) instead of each defining its own copy — before this crate
existed the two implementations were identical apart from an `AppError`
import qualifier. Part of the connected feature surface wherever `app-main`
wires it in (behind the same `connected` feature as `sync` / `tunnel-client`);
`headless`'s edge to it is unconditional.

See `crates/account-directory/src/lib.rs` for implementation detail.

### `sync`
**Crate:** `crates/sync`
**Owns:** the device-to-device sync engine — an iroh QUIC transport
multiplexing four wire protocols over the crate's ALPNs between a user's own
paired devices: Yjs notes reconciliation (`notes_proto`), content-addressed
meeting-media transfer (`blobs`, via `iroh-blobs`), processing-lifecycle +
trash-state discovery (`discovery_proto`, carrying each meeting's
`ProcessingLifecycle` and `DeletionState` in one frame), and derived-artifact
(`transcript.json` / `summary.md`) reconciliation (`artifacts_proto`) — plus
account-mediated peer discovery (`account`).

**Dependency edges:** `common` + `notes-crdt` only, never `persistence` — the
notes-CRDT primitives live in the leaf `notes-crdt` crate specifically so this
crate's lib stays off the C-heavy graph (libsql / audiopus / ogg) and
cross-compiles to `aarch64-linux-android` for the phone companion
(`sync-ffi`). `persistence::assets` is reached only as a dev-dependency of
this crate's own integration tests. Part of the connected-tier surface: the
crate is an unconditional workspace member, but the `app-main -> sync` edge
(`ipc-bridge`'s `SyncControl` trait) is `connected`-feature-gated — the free
build wires a no-op implementation instead. See `cross-cutting.md` — "Build
variants".

**Account-mediated peer discovery:** `sync::account` defines an
`AccountEndpointSource` trait the consumer implements (the account-service
HTTP fetcher bound to the device's credential); `sync` itself takes no
HTTP/account dependency, so this adds no new dependency-table edge — the
trait is the boundary, mirroring `election::ElectionDriver`. Account-sourced
and manually-paired peers are additive, both feeding the one `PeerDirectory`.

See `crates/sync/src/lib.rs` for implementation detail: the four wire
protocols, the `PeerDirectory` replace-not-union addressing semantics, the
ticket-based pairing lifecycle, blob GC/tagging, the artifact-authority
supersession rule, and the Android relay-DNS pre-resolution mechanism
(`SyncConfig::relay_ips`).

### `rag-retrieval`
**Crate:** `crates/rag-retrieval`

Retrieval-augmented context for the meeting agent — pure chunking and
cosine ranking so large attachments/transcripts are retrieved rather than
pinned in an LLM prefill that is prohibitively slow on the iGPU tier.

**Owns:** `chunk_text` (newline-aligned, char-boundary-safe windows with
overlap), `rank_top_k`/`cosine_unit`/`rrf_fuse` (reusing
`common::voiceprint_math`, since embeddings are L2-normalised so cosine
reduces to a dot product), and `RagChunk`/`DocType` (the
pre-persistence chunk value that `persistence` assigns durable identity
to). It defines and re-exports the `Embedder` seam (from `common`) but
never implements it — the concrete embedder lives in the `embedder`
crate, injected by `ipc-bridge`, which also owns the RAG write path
(attachment-convert time, transcript-finalise, live incremental
indexing) and the two consumers (`agent-tools`' `retrieve_chunks` tool
and the live-agent's per-refresh retrieval).

**Dependency shape:** depends ONLY on `common` — no `llama-cpp-2`, no
model loading — keeping retrieval logic model-agnostic and testable
without a model.

See `crates/rag-retrieval/src/lib.rs` for implementation detail.

### `embedder`
**Crate:** `crates/embedder`

The concrete text-embedding model crate — `Bgem3Embedder`, a
`common::Embedder` implementation backed by a held llama.cpp model
(BGE-M3 by default).

**Owns:** CLS-pooled, 1024-dim, L2-normalised embedding output;
`embed_batch` builds a fresh `!Sync` `LlamaContext` per input text
(keeping the struct itself `Send + Sync`) and reuses the process-wide
shared `LlamaBackend`; GPU features forward straight through to
`llama-cpp-2`. It is the embedding peer of `summariser`/`asr-runtime`/
`diarizer`, constructed and held lazily by `ipc-bridge` so that crate
carries no direct llama edge of its own for embeddings. A gated
real-model retrieval-quality eval adds test-only dev-dependency edges to
`persistence` and `rag-retrieval` (not runtime edges) to catch a
degraded/mis-quantised embedder that stub coverage can't.

**Dependency shape:** depends on `common` (`Embedder`, `AppError`,
`shared_llama_backend`, `voiceprint_math::unit_normalise`) plus
`llama-cpp-2` directly — this is the crate that owns that FFI edge so
`rag-retrieval` and `ipc-bridge` don't have to.

See `crates/embedder/src/lib.rs` for implementation detail.

### `ipc-bridge`
**Crate:** `crates/ipc-bridge`
**Owns:** the Tauri command + event surface (60 commands) and three custom
URI-scheme resolvers (`meetingasset:`, `attachment:`, `meetingrecording:`).
tauri-specta generates the TypeScript bindings the webview consumes.

**The only crate that imports `tauri::*`.** Every other crate stays testable
without a running Tauri app.

**Command surface, by domain:** recording lifecycle; settings; model
list/ensure; notes (JSON + CRDT/Yjs binding) and note images; attachments
(add/list/open/remove + conversion); meeting list/open/rename/
delete(soft)/restore/purge/collections; reprocess (re-transcribe +
re-diarize); summarise; chat
(send/cancel/session CRUD) and the live in-meeting co-pilot; translation;
voiceprint/identity management; diagnostics; and the connected-tier tunnel
pairing, account erasure, and sync commands. `stop_recording` also upserts
the meeting index, self-heals `list_meetings` against on-disk drift, and
fires decoupled background reprocess/auto-summarise passes so a slow pass
can never wedge the stop response or hide the meeting.

**Dependency shape — the systems-engineer composition layer with the
broadest read access of any library crate:** `common`, `orchestrator`,
`persistence`, `notes-crdt`, `summariser`, `settings`, `agent-tools`,
`chat-agent`, `doc-convert`, `embedder`, `rag-retrieval`. Each edge exists
because a domain crate's capability needs a direct command/event surface and
no other crate can own that wiring without leaking `tauri::*` into the
domain crate. It deliberately has **no** `model-registry` or `diarizer`
edge (both are reached only through `Orchestrator`), and **no**
`tunnel-client` / `sync` edge — those connected-tier surfaces are exposed
through `AccountControl` / `SyncControl` trait seams (`DisabledAccount` /
`DisabledSync` in the free build), with `app-main` injecting the concrete
implementation.

**Event bus:** `IpcState::event_tx` is one `broadcast::Sender<AppEvent>`
cloned from `app-main`'s single instance and shared with `Orchestrator` and
`ModelRegistry`. `ipc-bridge` is both the sole forwarder to the webview
(`spawn_event_forwarder`) and a first-class producer on the same bus
(summarise, translate, attachment-conversion, and live-agent events all
emit here directly).

See `crates/ipc-bridge/src/lib.rs` for implementation detail.

**"New meeting" prep drafts (see `persistence`/`orchestrator` above).** New
command `create_meeting() -> AppResult<MeetingId>` routes DIRECTLY to
`persistence::writer::create_draft` (no orchestrator — the meeting has no
live recording state until `start_recording` promotes it), mirroring
`add_attachment`'s direct-to-persistence routing; it also best-effort
upserts the index row (mirroring `stop_recording`'s stop-time upsert) so the
draft appears in `list_meetings` in this session rather than waiting on
`reconcile_orphans`'s next self-heal pass. `start_recording` gains a
`meeting_id: MeetingId` parameter (the draft being promoted), passed through
to `Orchestrator::start`. `set_recording_title`'s doc updated to no longer
claim "the active meeting has no `metadata.json` yet" — every meeting has one
from draft creation onward now; title editing during prep instead reuses the
existing `rename_meeting` command unchanged.

### `app-main` (bin)
**Crate:** `src-tauri` (package `minutist`)

The Tauri assembler binary — wires every other crate into a running app;
the thinnest crate, mostly construction and plumbing.

**Owns:** process lifetime, tray icon (a single programmatically-built
tray, deliberately no declarative `tauri.conf.json` tray to avoid a
duplicate handler-less icon), window management (close hides rather than
exits), tracing setup (file appender + crash-capture ring buffer with
redaction), the `settings.data_directory` path-resolution helper
(`resolve_data_roots`), the MCP server start/stop lifecycle
(feature-independent, gated on a settings toggle, reacting live to
enable/disable but requiring a restart for port/write-tool changes), and
the bindings-generation harness.

**Dependency shape:** the feature-gated `connected` Cargo feature is the
free-vs-paid artifact split — `mcp-server`, `tunnel-client`, `sync`,
`election`, and `account-directory` are all optional edges behind this
single feature, enforced at compile time
(`cargo build -p minutist --no-default-features` produces the free
artifact with none of that code compiled in). The free build injects
`ipc_bridge::disabled_account()` into `IpcState.connected.account` so
`ipc-bridge`'s surface is identical either way.

See `src-tauri/src/main.rs` for implementation detail.

### `headless` (bin)
**Crate:** `crates/headless`
**Owns:** the user-installed headless server daemon, `minutist-hub` — a
SECOND workspace binary beside `app-main`: an always-on device-sync hub now,
and (post-launch) a GPU processing node. It runs on hardware the user owns
and controls, in its own data root, never shared with a desktop's
`{app-data}`; it is not a build variant of `app-main` and shares no code
path with it — there is no `app-main -> headless` edge in either direction.

**Dependency edges:** `common`, `persistence`, `notes-crdt`, `sync`,
`tunnel-client`, `account-directory`. No `tauri::*` / `ipc-bridge` edge: the
daemon wires `sync::SyncEngine` directly and carries no command/event
surface. Unlike `app-main`'s `connected`-gated edges, `headless`'s workspace
membership and its edges to `sync` / `tunnel-client` / `account-directory`
are unconditional (not feature-gated) — a seeded `minutist-hub` is always
account-capable. A post-launch GPU processing-node role adds `orchestrator`
plus the ML-runtime crates (`asr-runtime` / `asr-parakeet` / `diarizer` /
`summariser` / `model-registry`) as a separate table-update commit.

**CLI surface:** the daemon runs by default; `login` signs the hub in to a
Minutist account via the same RFC 8628 device-code flow the desktop uses
(prints a URL to open, polls until approved, persists the credential) — the
ONLY way the hub discovers peers now (manual ticket pairing removed; `sync`'s
underlying `my_ticket`/`add_peer_from_ticket` primitives stay, since the
phone client still uses them via `sync-ffi`, but neither the hub nor the
desktop expose them any more); `status` prints the hub's state (including
sign-in status) as JSON from a pure filesystem read with no engine bind, so
an automated harness uses it as a read-only convergence oracle.

Convergence behaviour, tracing, configuration, and packaging are documented
in `cross-cutting.md` — "Headless server daemon". See
`crates/headless/src/main.rs` for implementation detail.

## Webview components

The webview is small enough that ownership maps to directories rather
than packages.

| Component | Lives in | Owns |
|---|---|---|
| Notes editor | `ui/src/editor/` | Tiptap editor, markdown shortcuts, paragraph-anchor extension. |
| Transcript pane | `ui/src/transcript/` | Live-appending transcript view, hover/click cross-reference. The live audio meter (`AudioMeter.tsx`) renders at the top of this pane. Rows are virtualised (`@tanstack/react-virtual`) and keyed by segment `start_ms`: only rows in the scroll container's visible window (plus overscan) are mounted, and identity follows the segment across a splice/reorder rather than sticking to an array position. `TranscriptRow` is memoised so a live append re-renders only the newly-visible rows. Speaker chips carry a live colour dot when diarization labels are present (`speaker-color.ts`: deterministic `speaker_id` → palette slot). Consecutive rows are grouped: the labelled chip shows once at the start of a speaker's run; continuation rows keep only the dot. |
| Meeting shell | `ui/src/shell/` | Window chrome (start/stop/pause, meeting list), the pane-visibility toggle, and the Settings drawer. The summary is a workspace column, not an overlay. Capture/processing/appearance settings live in the drawer rather than the top bar so the masthead stays a single non-overflowing row. |
| IPC client | `ui/src/ipc/` | Typed wrapper around `invoke` + `listen`. Generated stubs from tauri-specta live here. |
| UI state store | `ui/src/state/` | Zustand stores. Derived UI state only — transient. Also holds a `settings` snapshot loaded once via `refreshSettings`; user-driven changes round-trip through `commands.updateSettings` so they persist across restarts. |

The webview's source of truth for typed messages is the generated
`bindings.ts` produced by tauri-specta. Hand-edits to that file are not
allowed.

### `ui/src/editor/` — notes editor

A Tiptap v3 WYSIWYG editor (`Editor.tsx`) is the primary view: `StarterKit`
(`link: false`) + `@tiptap/extension-link` + `@tiptap/extension-typography` +
the `@tiptap/extension-table` family + `tiptap-markdown`
(`extensions.ts::buildEditorExtensions`), with markdown-shortcut input rules.

- **Paragraph-anchor extension (`paragraph-anchor.ts`).** Registers a
  nullable `data-anchor-ms` attribute on the paragraph node and stamps it on
  the FIRST keystroke into a paragraph, ONLY while
  `recordingState.kind === "recording"`, from the store's `recordingClockMs`
  (the pause-**excluding** capture clock) — **never** `Date.now()` (see
  `cross-cutting.md` "Notes paragraph-anchor clock"). Already-anchored
  paragraphs are never re-stamped; split-created paragraphs reset their
  inherited anchor so the next keystroke stamps fresh. The clock is injected
  as an `AnchorClockSource`, decoupling the extension from the store.
- **Autosave + clipboard.** `useAutosave.ts` runs interval autosave
  (`autosave_interval_secs`, default 5 s) plus flush-on-blur through the
  `save_notes` seam (`ipc/notes.ts`); target meeting is
  `activeMeetingId(state) ?? openMeetingId`, no-op only when neither exists.
  `clipboard.ts::buildClipboardPayload` produces a self-contained `text/html`
  (+ `text/plain`) copy with `data-anchor-ms` stripped, so paste into Word
  keeps formatting.
- **Cross-reference (`hover-bridge.ts`, `scroll-to-anchor.ts`).**
  `NotesHoverBridge` is a **presentation-only** ProseMirror plugin — mutates
  no doc, dispatches no transaction — reporting the hovered paragraph's
  `data-anchor-ms` and the next anchored paragraph's. `state/cross-ref.ts`
  maps that pair to the half-open range of segments whose
  `start_ms ∈ [anchor(P), anchor(nextP))` — through end-of-recording for the
  last anchored paragraph — as a `{ startIndex, endIndex }`
  `highlightedRange` (a range, never a single index); the transcript pane
  highlights every row in it. Clicking a transcript row resolves its
  `start_ms` to the nearest-anchored paragraph and scrolls to it.
- **Transcript-chip node + DnD (`transcript-chip.ts`, `transcript-dnd.ts`).**
  `TranscriptChip` is an atom block node (`startMs`/`endMs`/`speakerId`/
  `text`). HTML5 drag-and-drop (MIME `application/x-minutist-segment`) from
  the transcript pane drops a chip into the editor; it round-trips through
  the `notes.json` JSON cycle and exports via tiptap-markdown's `serialize`
  hook as a fenced ```transcript``` quotation.
- **AttachmentRef node (`attachment-ref.ts`, `attachment-drop.ts`).** A file
  dropped or pasted into the editor — any type — is registered as a normal
  meeting attachment (manifest row → attachments pane → doc-convert markdown
  fed to the summariser), leaving an inline `AttachmentRef` atom node in the
  notes body (transcript-segment drop takes precedence). The node carries a
  **portable** ref (attachment id + on-disk filename + metadata) — never a
  URL — resolved to a display URL at render time via
  `convertFileSrc(<meetingId>/<filename>, "attachment")`. Images render as a
  thumbnail (click → lightbox); other types as a file card (click → OS
  default app). The older `NoteImage` node is unchanged and coexists.
- **Margin marginalia (`anchor-marginalia.ts`).** A **presentation-only**
  decoration (no node attributes, no transactions) rendering each anchored
  paragraph's timestamp in the sheet's left gutter as the **local
  time-of-day** it was written — not the raw recording offset (see
  `cross-cutting.md` "Notes paragraph-anchor clock").

### `ui/src/transcript/` — transcript pane

`TranscriptPane.tsx` renders segments with `MM:SS.cc` timestamps and
sticky-bottom auto-scroll via the row virtualiser's `scrollToIndex` (most
rows are unmounted, so a manual `scrollTop` calculation cannot know their
heights). A row's speaker chip shows the user-set display name
(`MeetingMeta.speaker_names[label]`) or the bare label, editable (inline
rename → `set_speaker_name`) **only when viewing a saved, finalised
meeting** — live labels are provisional (re-lettered on stop, clearing
`speaker_names`), so mid-recording the chip is display-only; the timestamp,
not the chip, is the drag handle. A row whose `Segment::shared_speakers` is
non-empty shows a quiet "N speakers" marker (the segment spans more than one
speaker; it is not split). The toolbar carries a Reprocess action (re-runs
ASR + diarization over the complete recording; refuses unless the recorder
is `Idle`) and a translation control. `state/active-transcript.ts` is the
single source-of-truth selector for which transcript the pane and
cross-reference read: a saved meeting open with nothing recording reads that
meeting's restored transcript; otherwise the live recording store's.

**Translation overlay (`state/translations.ts`).** Holds `selectedLanguage`
(`null` = verbatim) and `translations: Map<number, string>` keyed by segment
`start_ms`, converted from the backend's index-keyed result against the
segments currently in view (an out-of-range index is dropped rather than
mapped to the wrong row). The cache clears whenever the segment array it was
built against is replaced (`transcript_ready` / `diarization_complete` for
the open meeting), since a re-diarize rewrites `transcript.json` and clears
`translations.json` server-side (see the `persistence` translations-sidecar
invariant). An untranslated row falls back to `seg.text`; "Show original"
returns to verbatim.

### `ui/src/shell/` — window chrome and top-level views

- **Meeting list (`MeetingList.tsx`) + folder sidebar
  (`CollectionsSidebar.tsx`).** The entry surface before a meeting is open:
  ruled-paper rows with per-row open/rename/delete/re-transcribe/
  re-summarise actions, plus a sidebar of "All meetings", user folders, and
  "Unfiled" (`state/collections.ts`); a meeting is filed via a "Move to…"
  popover or by dragging the row onto a folder. A row whose
  `recording_started` is `false` carries a "Draft" chip — the only surface
  that identifies an unpromoted prep draft, since nothing garbage-collects
  one.
- **`MainWindow`.** A resizable, show/hide multi-column layout
  (`react-resizable-panels`): up to four columns — notes (primary),
  transcript, summary, chat. The pane-visibility toggle shows/hides a column
  by INCLUDING/EXCLUDING its `Panel` from the Group, **not** collapsing it to
  zero width, keeping exactly one `Separator` between any two visible panes.
  Percentage `minSize`s sum well under 100%, so columns squeeze to fit and
  the workspace never scrolls horizontally; the last visible pane cannot be
  hidden. Defaults: live transcript hidden; a finished opened meeting → notes
  + summary; a live recording → notes only. The Group has no `autoSaveId`, so
  toggling a column re-derives layout from each pane's `defaultSize` —
  squeeze-to-fit wins over a width the user dragged to. A finished open
  meeting shows a back affordance and a masthead band with the meeting name
  (edit-in-place); while recording, the band is an always-editable name field
  (`set_recording_title`, applied at stop).
- **`MeetingControls.tsx` — the three-way record toggle.** One context-aware
  control covering `new_meeting` / `start` / `stop`. `new_meeting` (idle, no
  open draft, `auto_start_recording_on_new_meeting` off — the default)
  creates and opens a prep draft via `useRecordingStore.createMeeting` +
  `useMeetingsStore.open`, with no ASR-readiness gate (creating a draft loads
  no model); `start` (idle with an open draft, or the setting on) promotes the
  draft and IS ASR-gated; `stop` finalises. `MainWindow` excludes an open
  draft from `showSummaryPane` (`meta.recording_started === false` — nothing
  to summarise yet) while `isFinishedMeeting` still matches any open+idle
  meeting, and shows a chrome-strip banner (`main-window__draft-banner`) with
  a Discard action (`close` then `useMeetingsStore.remove`, in that order —
  `remove` does not clear `openMeetingId`). `MeetingMasthead` treats an empty
  title as a placeholder alongside the orchestrator's `Recording <timestamp>`
  default. The notes editor and Attachments pane need no draft-specific
  handling: both operate on any `openMeetingId`.
- **`SummaryView.tsx`.** Renders `summary.md` (`markdown-it`, `html: false`)
  as a paper sheet, with a Summarise action and editable/persistable raw
  markdown. Offered only for a finished opened meeting.
- **`ChatView.tsx`.** A workspace column, gated on a concrete
  `activeMeetingId` and hidden on the meeting-list surface — chat is
  meeting-scoped. Renders bubbles (assistant markdown), a tool-activity row,
  a streaming caret, an error state, a send box (Enter to send, Shift+Enter
  for newline, disabled mid-turn), and a session switcher.
- **`SettingsDrawer.tsx`.** Appearance, input device, transcription
  language, diarize-on-stop, GPU acceleration, system-audio capture, a
  summary prompt-preset picker (a non-empty custom prompt overrides the
  selected preset), a Connection pane (device-code pairing, live status, a
  guarded "Delete account" action — the connector channel transits content to
  the AI vendor by design and the copy never calls it "private"), and an MCP
  settings pane (enable toggle, fixed port, write-tools toggle, endpoint +
  bearer-token reveal/copy). **Every setting here round-trips through the
  existing `commands.updateSettings` seam — no bespoke IPC command per
  toggle.** The two connected-only panes are `VITE_CONNECTED`-gated
  (lazy-loaded, dropped from the free bundle).
- **`Onboarding.tsx`.** `App.tsx` is the gate point: fetch settings + model
  list on mount, hold the UI neutral (`return null`) while pending so a
  returning user is never flashed onboarding, then render `Onboarding` when
  `settings.onboarding_completed` is `false`, else `MainWindow`; the
  app-event listener (`useAppEventBridge`) stays mounted above this gate.
  Onboarding's final step persists `onboarding_completed = true` through the
  same `updateSettings` seam — no dedicated completion command.
- **`About.tsx`.** Bundled-model SPDX licence rows are derived from the
  model manifest via the models store (`ModelStatus.license`) — no
  hand-mirrored list to drift. OSS attributions, app version, and the NOTICE
  line are static.
- **Diagnostics (`ui/src/diagnostics/`).** `issueReport.ts` builds the
  "Report a problem" GitHub issue-form URL from a redacted
  `DiagnosticReport` (no meeting-content field by construction), enforcing
  an ~8 KB cap by explicitly eliding the diagnostics field (never silent)
  and falling back to a clipboard copy. `reportProblem.ts` +
  `state/report-problem.ts` share this flow between the About dialog, the
  main-window error pane, and a window-level `error`/`unhandledrejection`
  handler that captures uncaught webview crashes. No telemetry — the user
  submits from their own browser.

### `ui/src/state/` — Zustand stores

One store per domain (`recording`, `models`, `meetings`, `summary`, `chat`,
`translations`, `collections`, `onboarding`, `cross-ref`,
`operation-progress`, plus the `settings` snapshot). Every store's
`handleEvent` is dispatched from the single `"app-event-payload"` listener
mounted once in `App.tsx` (`useAppEventBridge`, `shell/event-listener.tsx`)
— one subscription only, never inside a conditionally-rendered subtree. Each
`handleEvent` is scoped to the meeting or session the event names, so an
event for a backgrounded meeting/session never clobbers the open one.

- **Settings round-trip.** Any user-driven setting is read once via
  `refreshSettings` and written back through `commands.updateSettings` —
  the only persistence path; no dedicated command is added per toggle.
- **Chat event-reconciliation (the lossy-broadcast guarantee — see
  `cross-cutting.md` "Agent chat loop").** `chat_token` deltas append to the
  `streaming` buffer as a progressive hint and are never trusted as the
  final answer; `chat_turn_complete.final_text` is authoritative and
  replaces the streamed buffer, so a dropped delta cannot corrupt the stored
  text. Every chat event is per-session scoped. Because `send()` starts
  streaming before its dispatch promise resolves with the backend-minted
  session id, an event for a brand-new session can arrive with no id yet to
  check against; such events buffer in `pendingEvents` and replay once
  `send` adopts the id, clearing on every session switch.
- **`active-transcript.ts`** is a derived selector, not a store (above).

### `ui/src/ipc/` — IPC seams

A thin per-domain client (`notes.ts`, `meetings.ts`, `summary.ts`,
`chat.ts`, `translations.ts`, `collections.ts`, …) wraps the shim-aware
`commands` object from `./client` — **never** the raw generated
`./bindings` directly — so tests mock the seam module, not the generated
bindings. `ipc/dev-shim.ts` (sample data) + `dev-shim-guard.ts`
(`shouldUseDevShim`) let the full app render under `vite dev` in a plain
browser with no Tauri backend, for visual QA: the guard activates only when
`import.meta.env.DEV` is true, the runner is not Vitest, and the Tauri
global is absent. `client.ts` reaches the shim exclusively through a
dynamic `import()`, so the production build never bundles or fetches it
(dead-code-eliminated); every new IPC-backed view should render in the DEV
shim with representative sample data so it can be visually QA'd the same way.

### Design system — Editorial Ink (light theme)

A warm-paper, document-centric light theme applied across the webview.

- **Token source.** `ui/src/styles/theme.css` is the single source of truth
  for all colour / radius / shadow / type tokens (CSS custom properties).
  Component CSS references these variables only — **no hard-coded
  colour/radius/shadow literals**. `ui/src/styles/global.css` holds the base
  layer; both are imported once from `ui/src/main.tsx`. The accent is a
  single oxblood ink used sparingly (recording dot, links, active/primary
  control, focus, selection).
- **Fonts (local-first, no CDN).** `@fontsource-variable/fraunces` (display
  — wordmark + editor headings) and `@fontsource-variable/newsreader`
  (reading body + UI chrome) — the **only two** UI font families; woff2
  ships as build assets so the app renders offline.
- **Notes sheet.** The notes editor renders as a sheet of binder paper
  filling its pane (no floating card): a left timestamp gutter, a margin
  rule dividing gutter from writing column, and — when `notes_paper_rules`
  is on (default) — faint horizontal writing-paper rules pitched to the body
  leading. The transcript pane and summary view are the quiet, recessed
  columns.
- **Binding on all new views.** Every view, present or future, must consume
  `theme.css` tokens and reuse the established patterns (Fraunces display /
  Newsreader body, the single oxblood accent used sparingly, paper surfaces,
  hairline rules, restrained motion respecting `prefers-reduced-motion`). No
  view introduces its own palette or type family — a hard-coded colour or
  font is a code-review finding.

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
