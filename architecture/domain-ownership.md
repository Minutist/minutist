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
| `settings` | data-engineer | `crates/settings/**` | Same | `common` |
| `orchestrator` | systems-engineer | `crates/orchestrator/**` | Same | `common` + all live-pipeline crates per table in `components.md` |
| `agent-tools` | systems-engineer | `crates/agent-tools/**` | Same | `common`, `persistence`, `orchestrator` |
| `chat-agent` | ml-runtime-engineer | `crates/chat-agent/**` | Same | `common`, `summariser`, `agent-tools` |
| `mcp-server` | systems-engineer | `crates/mcp-server/**` | Same | `common`, `agent-tools` |
| `tunnel-client` | systems-engineer | `crates/tunnel-client/**` | Same | Nothing — it's a near-leaf (re-implements the relay wire frames; takes config, not workspace edges) |
| `ipc-bridge` | systems-engineer | `crates/ipc-bridge/**` | Same | `common`, `orchestrator`, `persistence`, `summariser`, `settings`, `agent-tools`, `chat-agent` |
| `app-main` (bin) | systems-engineer | `src-tauri/**` | Same | All crates (it's the assembler) |
| Webview UI | frontend-engineer | `ui/src/**` | This file too if changing UI domain layout. | `ui/src/ipc/bindings.ts` only — never the Rust source. |

The "Can call without doc update" column is the dependency rule
restated. Adding any other edge requires updating
[`components.md`](components.md) in the same commit.

## Cross-cutting ownership notes

- **VRAM-aware GPU placement.** `common` (architecture-owner) owns the VRAM
  probe `probe_primary_gpu()` (behind the `llama-backend` feature) + the **pure**
  `resolve_gpu_plan()` and its `GpuProbe` / `GpuAcceleration` / `GpuPlan` types.
  `ipc-bridge` and `orchestrator` (systems-engineer) are **consumers**: each
  calls `resolve_gpu_plan` at a model-load moment and maps the plan to the
  per-model GPU decision (`ipc-bridge` for the summariser; `orchestrator` for
  ASR). The thresholds + policy are documented in `cross-cutting.md` — "GPU
  portability". Changing the probe or the plan is an architecture-owner change in
  `common`; changing only how a consumer uses the plan stays in that consumer.

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
