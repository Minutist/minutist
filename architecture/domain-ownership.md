# Domain ownership

This file is what makes parallel-agent work tractable. Each Rust core
component is one domain; each domain has one owner role. Agents
assigned to a role work inside that domain without coordinating with
other roles, except via the contracts in
[`components.md`](components.md) and [`cross-cutting.md`](cross-cutting.md).

Names below describe roles, not people. A single agent may hold
multiple roles when work is sequential; multiple agents may share a
role when work is parallel.

## Domain table

| Domain | Owner role | Source paths | Can edit | Can call (without doc update) |
|---|---|---|---|---|
| `common` | architecture-owner | `crates/common/**` | This file too. Adding a type / trait is an architectural change. | Nothing — it's the leaf. |
| `audio-capture` | audio-engineer | `crates/audio-capture/**` | This file too if changing the capture contract. | `common` |
| `vad-chunker` | audio-engineer | `crates/vad-chunker/**` | Same | `common` |
| `asr-runtime` | ml-runtime-engineer | `crates/asr-runtime/**` | Same | `common` |
| `asr-parakeet` | ml-runtime-engineer | `crates/asr-parakeet/**` | Same | `common` |
| `diarizer` | ml-runtime-engineer | `crates/diarizer/**` | Same | `common` |
| `summariser` | ml-runtime-engineer | `crates/summariser/**` | Same | `common` |
| `model-registry` | ml-runtime-engineer | `crates/model-registry/**` | Same | `common` |
| `notes-crdt` | data-engineer | `crates/notes-crdt/**` | Same | `common` |
| `persistence` | data-engineer | `crates/persistence/**` | Same | `common`, `notes-crdt` |
| `doc-convert` | data-engineer | `crates/doc-convert/**` | Same | `common` only — NO other workspace-component edge |
| `rag-retrieval` | ml-runtime-engineer | `crates/rag-retrieval/**` | Same | `common` only (pure retrieval logic; the concrete embedder is the `embedder` crate, injected via the `Embedder` seam) |
| `embedder` | ml-runtime-engineer | `crates/embedder/**` | Same | `common`, `llama-cpp-2` (a model-loading leaf — the embedding peer of `summariser`) |
| `settings` | data-engineer | `crates/settings/**` | Same | `common` |
| `orchestrator` | systems-engineer | `crates/orchestrator/**` | Same | `common` + all live-pipeline crates per table in `components.md` |
| `agent-tools` | systems-engineer | `crates/agent-tools/**` | Same | `common`, `persistence`, `notes-crdt`, `rag-retrieval` |
| `chat-agent` | ml-runtime-engineer | `crates/chat-agent/**` | Same | `common`, `summariser`, `agent-tools` |
| `mcp-server` | systems-engineer | `crates/mcp-server/**` | Same | `common`, `agent-tools` |
| `tunnel-client` | systems-engineer | `crates/tunnel-client/**` | Same | Nothing — it's a near-leaf (re-implements the relay wire frames; takes config, not workspace edges) |
| `account-directory` | systems-engineer | `crates/account-directory/**` | Same | `common`, `sync`, `tunnel-client` (the `AccountEndpointSource` adapter shared by `app-main` and `headless`, so neither carries its own copy) |
| `sync` | systems-engineer | `crates/sync/**` | Same | `common`, `notes-crdt` |
| `sync-ffi` | systems-engineer | `crates/sync-ffi/**` | This file too if changing the FFI wrapper contract. | `common`, `sync` (mobile-only UniFFI wrapper — see the `¶` footnote in `components.md`) |
| `election` | systems-engineer | `crates/election/**` | This file too if changing the `ElectionDriver` trait contract. | `common`, `persistence`, `notes-crdt` (the producer-gate host-election leaf — drives `sync` / `orchestrator` only behind the `ElectionDriver` trait, so it takes no edge to either) |
| `ipc-bridge` | systems-engineer | `crates/ipc-bridge/**` | Same | `common`, `orchestrator`, `persistence`, `notes-crdt`, `summariser`, `settings`, `agent-tools`, `chat-agent`, `doc-convert`, `embedder`, `rag-retrieval` |
| `app-main` (bin) | systems-engineer | `src-tauri/**` | Same | All crates (it's the assembler) |
| `headless` (bin) | systems-engineer | `crates/headless/**` | Same | `common`, `persistence`, `notes-crdt`, `sync`, `tunnel-client`, `account-directory` |
| Webview UI | frontend-engineer | `ui/src/**` | This file too if changing UI domain layout. | `ui/src/ipc/bindings.ts` only — never the Rust source. |

The "Can call without doc update" column is the dependency rule
restated. Adding any other edge requires updating
[`components.md`](components.md) in the same commit.

## Cross-cutting ownership notes

Durable exceptions to the one-domain-one-owner rule — cases where a type,
trait, or invariant is owned by one role but implemented or consumed by
another. Implementation history (which work unit added what, and why) lives
in git log and `planning/journal.md`, not here.

- **`DocVlm` trait (image-attachment OCR).** Owned by `common`
  (architecture-owner) as the injection seam; the concrete `GemmaVlm`
  implementation lives in `ipc-bridge` (systems-engineer) — see `ipc-bridge`'s
  own module doc.
- **VRAM-aware GPU placement.** `common` (architecture-owner) owns the VRAM
  probe and the pure `resolve_gpu_plan()`; `ipc-bridge` and `orchestrator`
  (systems-engineer) are consumers only. See `cross-cutting.md` — "GPU
  portability".
- **Live in-meeting agent.** Gate types (`LiveAgentMode`,
  `live_agent_should_run`) are owned by `common`. The driver lives in
  `ipc-bridge::live_agent` (systems-engineer); the held-context backend lives
  in `chat-agent::live` (ml-runtime-engineer). See `cross-cutting.md` — "Live
  in-meeting agent (auto-driver)" and each crate's own module doc for the
  KV-checkpoint / eviction / routing mechanisms.
- **Voiceprint identity types + maths.** `VoiceprintIdentityId`,
  `VoiceprintCentroidId`, and `common::voiceprint_math` are owned by `common`;
  `diarizer` (embedding extraction) and `persistence` (centroid cache) consume
  them. Neither takes a new edge — both already depend on `common`.
- **Processing-lifecycle + host-election types.** `ProcessingLifecycle`,
  `ProcessingClaim`, and `HostRef` are owned by `common`. `HostRef` is an
  opaque device key — the seam that keeps `iroh` out of `common`; `sync` maps
  it to/from `iroh::EndpointId` at the wire boundary. Transport is `sync`'s
  domain (systems-engineer). `DeletionState` (the trash soft-delete field,
  `MeetingMeta.deletion`) rides the same Discovery stream alongside
  `ProcessingLifecycle`, under the same ownership split and the same
  no-new-dependency-edge constraint — see `crates/notes-crdt/src/lib.rs`'s
  module doc.
- **Per-meeting `metadata.json` write lock.** `persistence::meeting_ops`
  owns the guarded RMW helper; `orchestrator` and `agent-tools`
  (systems-engineer) route their metadata writes through it rather than
  writing directly. No dependency-table change — both already depend on
  `persistence`.

## Role definitions

### `architecture-owner`
The only role allowed to edit `crates/common/**` and
`architecture/**`. In practice this is the human + the
principal-code-reviewer agent in conversation. Changes here are
proposal-and-review, not implement-then-review.

### `audio-engineer`
Owns the realtime audio path before ASR. Capture, sample-rate handling,
ring buffer, VAD chunking. Knowledge expected: `cpal`, Silero VAD, the
back-pressure model for the live pipeline.

### `ml-runtime-engineer`
Owns the ML-runtime crates: ASR (`asr-runtime`, `asr-parakeet`),
diarization (`diarizer`), summarisation (`summariser`), model registry,
RAG retrieval (`rag-retrieval`, `embedder`), and the chat-agent turn
engine (`chat-agent`) — see the domain table above for the authoritative
list. Knowledge expected: `llama-cpp-2` mtmd and text APIs, `sherpa-onnx`,
model file formats, ONNX vs GGUF tradeoffs.

A single agent can hold this role across these crates *if* changes are
sequential. Parallelising within the role requires splitting the crates
among agents — possible because the crates don't depend on each other.

### `data-engineer`
Owns persistence, the notes-CRDT leaf, and settings. Knowledge expected:
libsql, SQLite schema migrations, file-format design (Opus encode, Tiptap
JSON), the Yjs (`yrs`) CRDT and its lib0 v1/v2 encodings, settings store
semantics.

`notes-crdt` is a leaf carved out of `persistence` (depends only on `common`):
the Yjs `ydoc` JSON↔CRDT conversion, the `NotesStore` reader/writer for
`notes.ydoc` + its `notes.json` / `notes.md` projections, the `MeetingFolder`
layout, and the `metadata.json` writer. `persistence` depends on it and
re-exports every symbol at the historical `persistence::*` paths, so the split
is invisible to `persistence`'s consumers. The extraction exists so `sync` can
transport / merge the notes CRDT without pulling in `persistence`'s C-heavy
graph (libsql / audiopus / ogg) — which is what lets `sync`'s lib cross-compile
to mobile targets. `notes-crdt` owns no other crate's domain logic and takes no
edge beyond `common`.

### `systems-engineer`
Owns orchestrator, IPC bridge, and the app-main wiring. The connective
tissue. Knowledge expected: tokio async patterns, Tauri commands and
events, tauri-specta, error-propagation conventions.

This role has the broadest read access; it imports every other crate.
That's by design — orchestration is centralised so the other crates
stay leaf-shaped.

`tunnel-client` (WS4-A) is systems-engineer for the same reason as
`mcp-server`: it is transport/IPC-adjacent connective tissue (the app-side
relay tunnel that bridges to the loopback `mcp-server`), it shares the
`connected`-feature gating and the internal-bearer handoff with `mcp-server`,
and `app-main` (systems-engineer) is the assembler that injects its config and
wires it behind the feature in S5. It owns no domain logic of another crate — it
re-implements the relay's wire frames and forwards HTTP — so it stays a near-leaf
under one role.

`sync` (WS4-B) is systems-engineer for the same reason: it is transport
connective tissue (the device-to-device iroh sync engine), it shares the
`connected`-feature gating with `mcp-server` / `tunnel-client`, and `app-main`
(systems-engineer) is the assembler that injects its config and wires it behind
the feature in S5. It depends only on `common` (shared types) and `notes-crdt`
(the `NotesStore` it reads/merges + `MeetingFolder::ensure` for inbound folders)
— it owns no domain logic of another crate, so it stays a near-leaf under one
role. It does NOT depend on `persistence`: the notes-CRDT primitives were
extracted into the leaf `notes-crdt` crate so `sync`'s lib stays off the C-heavy
graph and cross-compiles to mobile. The meeting-media path (`audio.opus` +
`assets/*`) and the derived-artifact path (`transcript.json` + `summary.md`, plus
the sync-owned authority store under `.blobs/artifacts/`) are written by `sync`
itself under the meetings root after `MeetingFolder::ensure`; `persistence::assets`
(which stays in `persistence`) is reached only as a DEV-dependency by `sync`'s
integration tests.

`headless` (WS4-B) is systems-engineer for the same reason as `sync` /
`tunnel-client` / `mcp-server`: it is connective tissue, not a domain. It is the
user-installed headless server daemon (`minutist-hub`) that wires
`sync::SyncEngine` into a long-running service — an always-on sync hub now, a GPU
processing node post-launch. It owns no domain logic of another crate, is NOT a
Tauri binary, and takes no `ipc-bridge` / `tauri::*` edge. It runs over its own
data root (an absolute path injected at startup via CLI/env), entirely separate
from the desktop's `{app-data}`: it owns `settings.store`, `logs/`, `index.db`,
and `meetings/{uuid}/` under THAT root, and never touches the desktop's
`mcp_token` / `tunnel_device.json` / Tauri-managed paths. The single-writer rule
applies per data root, so the daemon must never share a root with another
process. The post-launch GPU node (adding the ML-runtime crates) is an
architecture-owner decision at that time.

The `status` subcommand is a read-only filesystem oracle: it reports
`endpoint_id: null` on a data root with no persisted `sync_node_key` rather
than generating (and persisting) one just to fill the field, so pointing it at
an unused root cannot silently mint device state. `print-ticket` mints an
identity on first run when one is needed — that side effect is the point of
the command, since a pairing ticket requires an identity to address.

### `frontend-engineer`
Owns everything under `ui/src/`. Knowledge expected: React 19, Tiptap +
ProseMirror, Zustand, generated tauri-specta bindings.

**Output-language UI.** The `OutputLanguagePicker` component and the
`output-language-settings.ts` / `setOutputLanguage` store seam follow the
`LanguagePicker` / `transcription-language-settings.ts` pattern exactly — a
UI-side `OUTPUT_LANGUAGES` constant (15 full English names, alphabetical) plus
the `"auto"` sentinel rendered as "Auto (system)". The picker lives in the
Processing section of `SettingsDrawer`. No model names in user-facing copy.

## Parallel-work rules

These rules let multiple agents work concurrently without coordination:

1. **No cross-domain edits.** An agent working in domain X never edits
   files in domain Y. If a change in Y is required, the agent files an
   architecture issue / proposal and waits.
2. **No `common` edits without architecture review.** Adding a type or
   trait variant in `common` is a one-way door; many downstream changes
   follow. The architecture-owner approves first.
3. **Orchestrator changes are last.** When a new feature touches the
   live pipeline, the component crates change first (independently),
   the orchestrator changes last to wire them together. This keeps the
   orchestrator from being a merge-conflict magnet.
4. **The IPC surface is owned by `systems-engineer`.** Other roles may
   propose new commands / events, but the actual additions land in
   `ipc-bridge` and `common`, not in the consuming crate.
5. **Tests live with the code.** An agent owning a domain owns its
   tests. Integration tests in `crates/orchestrator/tests/` are
   systems-engineer territory.

## When to invoke which role

| Change shape | Role |
|---|---|
| "Add a new field to Segment" | architecture-owner |
| "Tune the VAD threshold default" | audio-engineer + data-engineer (defaults live in `settings`) |
| "Swap the ASR model to Qwen3-ASR-0.6B" | ml-runtime-engineer (model-registry manifest + `asr-runtime` prompt) |
| "Improve summarisation prompt" | ml-runtime-engineer + data-engineer (prompt source: settings) |
| "Index meeting titles for full-text search" | data-engineer (`persistence`) + systems-engineer (`ipc-bridge` surface) |
| "Add a new keyboard shortcut" | frontend-engineer |
| "Refactor the live pipeline to use a different chunking strategy" | systems-engineer (orchestrator) — but specify the new contract in this doc first |
| "Add Ollama dispatcher as a `Summariser` impl" | ml-runtime-engineer (lives in `summariser`, no new edge) |
| "Add a new document format converter" | data-engineer (`doc-convert` — one match arm in `convert_to_markdown` + a fixture test) |
| "Adjust attachment storage layout or manifest schema" | data-engineer (`persistence::attachments`) + architecture-owner if the `AttachmentEntry` shape changes |
| "Add a chat tool" | systems-engineer (`agent-tools` — one `impl Tool` + register in `ToolRegistry::v1`) |
| "Change the agent loop / sampling / tool-call parsing" | ml-runtime-engineer (`chat-agent`) |
| "Expose a tool over MCP" | systems-engineer (the `expose_over_mcp` allowlist + the `mcp_write_tools` gate in `mcp-server`); the tool itself is `agent-tools` |
| "Change the MCP transport / auth" | systems-engineer (`mcp-server`) |
| "Change the VRAM probe or the GPU-plan policy/thresholds" | architecture-owner (`common` — `probe_primary_gpu` / `resolve_gpu_plan`); consumers only re-wire |
| "Add a telemetry hook" | architecture-owner (it's not in scope yet, requires a doc update) |

## Anti-patterns the reviewer flags

- A crate importing another crate that isn't in its allowed list (see
  `components.md` dependency table).
- A crate added/removed, or `common`'s public surface changed, in a
  commit with no `architecture/` touch — caught by the pre-commit hook.
- A new `pub` item in `common` without an entry in this doc.
- `tauri::*` imports outside `ipc-bridge` and `app-main`.
- Filesystem writes outside the owning crate's directory scope (see
  `cross-cutting.md` — Filesystem layout).
- `anyhow::Error` in a public function signature.
- `println!` outside test code.
- Unbounded channels.
