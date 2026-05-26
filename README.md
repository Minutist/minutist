# meeting-app

Local-first desktop meeting-notes application. Records meetings,
transcribes them on-device, takes hand-typed notes alongside, summarises
with a local LLM. Cross-platform (Windows / macOS / Linux).

**Status:** pre-prototype. Phase 0 spikes in progress.

## Architecture

See [`architecture/`](architecture/). The C4 model diagrams and docs there
are authoritative for component boundaries, interfaces, and ownership.
Read [`architecture/README.md`](architecture/README.md) first.

## Workspace layout

```
architecture/     C4 docs + Structurizr DSL + rendered SVGs
crates/
  common/         shared types and trait definitions
spikes/
  asr/            llama-cpp-2 mtmd ASR spike
  llm/            llama-cpp-2 text LLM spike
  vad-loop/       Silero VAD + ASR end-to-end
  diarize/        sherpa-onnx diarization spike
scripts/
  render-architecture.sh   regenerate SVGs from workspace.dsl
.githooks/
  pre-commit      architecture drift guard (install: see below)
```

The `spikes/` crates are deliberately throwaway. Once Phase 0 exits, the
patterns that work move into `crates/common` and the real application
crates listed in [`architecture/components.md`](architecture/components.md).

## Setup

After cloning, install the architecture pre-commit hook:

```bash
git config core.hooksPath .githooks
```

The hook fails any commit that touches source under `crates/`,
`src-tauri/`, or `ui/src/` without also touching `architecture/`. See
[`architecture/README.md`](architecture/README.md) for the rationale.

## Build

```bash
cargo build --workspace
```

Models, native dependencies, and platform-specific build instructions
land here as Phase 0 progresses.

## Rendering the architecture diagrams

```bash
scripts/render-architecture.sh
```

Requires Docker. Uses `structurizr/structurizr` to export the DSL to
Mermaid, then `minlag/mermaid-cli` to convert to SVG. SVGs are committed
alongside [`architecture/workspace.dsl`](architecture/workspace.dsl).
