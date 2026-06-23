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
| `model-registry` | ml-runtime-engineer | `crates/model-registry/**` | Same | `common`, `settings` |
| `persistence` | data-engineer | `crates/persistence/**` | Same | `common` |
| `doc-convert` | data-engineer | `crates/doc-convert/**` | Same | `common` only — NO other workspace-component edge |
| `settings` | data-engineer | `crates/settings/**` | Same | `common` |
| `orchestrator` | systems-engineer | `crates/orchestrator/**` | Same | `common` + all live-pipeline crates per table in `components.md` |
| `agent-tools` | systems-engineer | `crates/agent-tools/**` | Same | `common`, `persistence`, `orchestrator` |
| `chat-agent` | ml-runtime-engineer | `crates/chat-agent/**` | Same | `common`, `summariser`, `agent-tools` |
| `mcp-server` | systems-engineer | `crates/mcp-server/**` | Same | `common`, `agent-tools` |
| `tunnel-client` | systems-engineer | `crates/tunnel-client/**` | Same | Nothing — it's a near-leaf (re-implements the relay wire frames; takes config, not workspace edges) |
| `sync` | systems-engineer | `crates/sync/**` | Same | `common`, `persistence` |
| `ipc-bridge` | systems-engineer | `crates/ipc-bridge/**` | Same | `common`, `orchestrator`, `persistence`, `summariser`, `settings`, `agent-tools`, `chat-agent`, `doc-convert` |
| `app-main` (bin) | systems-engineer | `src-tauri/**` | Same | All crates (it's the assembler) |
| Webview UI | frontend-engineer | `ui/src/**` | This file too if changing UI domain layout. | `ui/src/ipc/bindings.ts` only — never the Rust source. |

The "Can call without doc update" column is the dependency rule
restated. Adding any other edge requires updating
[`components.md`](components.md) in the same commit.

## Cross-cutting ownership notes

- **Attachments common types (Attachments WS).** `AttachmentId`, `ConversionState`,
  `AttachmentEntry`, and the four `AppEvent` variants (`AttachmentAdded`,
  `AttachmentConverted`, `AttachmentConversionFailed`, `AttachmentRemoved`) are owned
  by `common` (architecture-owner). Adding them was an architecture-owner change
  (parallel-work rule 2). They follow the additive-field discipline (serde-defaulted,
  specta-derived) and ride the existing `AppEventPayload` + `collect_events!`
  registration — no second event bus.

- **`DocVlm` trait (image-attachment OCR).** `common::DocVlm` is the injection
  seam that lets `doc-convert` call vision inference without taking a workspace
  edge beyond `common`. The trait is owned by `common` (architecture-owner). The
  concrete `GemmaVlm` implementation lives in `ipc-bridge` (systems-engineer):
  it wraps the held `LlamaSummariser` and its lazy `MtmdContext`, and is wired
  through the conversion worker in `ipc-bridge` and `app-main`. `doc-convert`
  (data-engineer) gains `image` as a third-party dep (decode/re-encode image
  attachments to PNG) — NOT a workspace-component edge, so `doc-convert` remains
  a `common`-only leaf. No new row is needed in the dependency table for either
  `GemmaVlm` (it lives inside `ipc-bridge`, which already depends on
  `summariser`) or `image` (third-party). Scanned/image-only PDF OCR — which
  would add `pdfium-render` for page rasterisation — is deferred (planning issue
  0019).

- **`Summariser::summarise` widening (Attachments WS).** The trait gains
  `attachments_markdown: &str` before `system_prompt` — an architecture-owner
  change (the "Change shape" table: trait changes are architecture-owner). All
  impls and call sites are updated in the same commit so the workspace compiles
  throughout; the empty-string path is byte-identical to the prior no-attachment
  behaviour (asserted by a `summariser` unit test).

- **VRAM-aware GPU placement.** `common` (architecture-owner) owns the VRAM
  probe `probe_primary_gpu()` (behind the `llama-backend` feature) + the **pure**
  `resolve_gpu_plan()` and its `GpuProbe` / `GpuAcceleration` / `GpuPlan` types.
  `ipc-bridge` and `orchestrator` (systems-engineer) are **consumers**: each
  calls `resolve_gpu_plan` at a model-load moment and maps the plan to the
  per-model GPU decision (`ipc-bridge` for the summariser; `orchestrator` for
  ASR). The thresholds + policy are documented in `cross-cutting.md` — "GPU
  portability". Changing the probe or the plan is an architecture-owner change in
  `common`; changing only how a consumer uses the plan stays in that consumer.

- **Live in-meeting agent common types (Phase 9 / WU2b).** The following
  types and functions in `common` are owned by `architecture-owner`:
  - `LiveDigestItem { text: String, resolved: bool, source: Option<String> }` —
    one item in a digest category; derives `specta::Type`, crosses IPC.
  - `LiveDigest { meeting_id, generated_at_ms, action_items, decisions,
    open_asks, attachment_answers, unresolved_references }` — the full per-meeting
    digest produced by the live agent on each refresh.
  - `LiveAgentMode { Auto, On, Off }` — user preference for the live agent gate.
    `Auto` resolves to GPU-acceleration-active (see `live_agent_should_run`).
  - `live_agent_should_run(mode, probe, gpu_acceleration) -> bool` — pure gate
    resolution; documented in `components.md` and `cross-cutting.md`.
  - `AppEvent::LiveDigestUpdated` and `AppEvent::LiveDigestError` — the two
    event variants emitted by `ipc-bridge`'s live-agent driver.
  The live-agent driver implementation (S2b) lives in
  `crates/ipc-bridge/src/live_agent.rs` (systems-engineer). The held-context
  backend (S2a) lives in `crates/chat-agent/src/live.rs` (ml-runtime-engineer).
  Neither adds a new dependency edge beyond what is already in the table.
- **Voiceprint identity types + maths (issue #0003 WU0 — one-way door).**
  `VoiceprintIdentityId` and `VoiceprintCentroidId` are UUID newtypes added to
  `common` by the architecture-owner. Adding them is a **one-way-door** per
  parallel-work rule 2 — many downstream changes (persistence schema, IPC
  commands, orchestrator wiring) follow. The dependency table in `components.md`
  is **unchanged**: `diarizer` and `persistence` already depend on `common` and
  gain no new edge here.

  `common::voiceprint_math` (three pure functions: `unit_normalise`,
  `cosine_unit`, `weighted_merge`) is also architecture-owner territory: it is
  the canonical centroid-maths implementation shared by `diarizer` (embedding
  extraction) and `persistence` (centroid cache recomputation). Both crates
  already depend on `common` — no new edge.

  `persistence` (data-engineer) gains `voiceprints.db` within its owned scope
  (`{app-data}/voiceprints.db` — the sixth durable `{app-data}` entry). It
  remains a `common`-only crate; no `diarizer` edge is permitted or needed
  (the pure maths live in `common`; the embedding extraction stays in
  `diarizer`; only the final `Vec<f32>` bytes cross into `persistence`).

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
Owns the four ML-runtime crates: ASR, diarization, summarisation, model
registry. Knowledge expected: `llama-cpp-2` mtmd and text APIs,
`sherpa-onnx`, model file formats, ONNX vs GGUF tradeoffs.

A single agent can hold this role across all four crates *if* changes
are sequential. Parallelising within the role requires splitting the
crates among agents — possible because the crates don't depend on each
other.

### `data-engineer`
Owns persistence and settings. Knowledge expected: libsql, SQLite schema
migrations, file-format design (Opus encode, Tiptap JSON), settings
store semantics.

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
the feature in S5. It depends only on `common` (shared types) and `persistence`
(the notes-CRDT store + meeting-media paths it transports) — it owns no domain
logic of another crate, so it stays a near-leaf under one role.

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
- A `crates/` source change in a commit with no
  `architecture/` touch — caught by the pre-commit hook.
- A new `pub` item in `common` without an entry in this doc.
- `tauri::*` imports outside `ipc-bridge` and `app-main`.
- Filesystem writes outside the owning crate's directory scope (see
  `cross-cutting.md` — Filesystem layout).
- `anyhow::Error` in a public function signature.
- `println!` outside test code.
- Unbounded channels.
