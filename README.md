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

After cloning, configure git for this repo:

```bash
git config core.hooksPath .githooks   # architecture drift guard
git config pull.rebase true           # never merge-pull
git config merge.ff only              # no merge commits; ff-only
```

- **Hook.** The hook fails any commit that touches source under
  `crates/`, `src-tauri/`, or `ui/src/` without also touching
  `architecture/`. See [`architecture/README.md`](architecture/README.md)
  for the rationale.
- **Rebase + ff-only.** Linear history is required. No merge commits.
  See
  [`architecture/agent-dispatch.md`](architecture/agent-dispatch.md) —
  Branch and merge convention.

## Build

```bash
cargo build --workspace
```

Models, native dependencies, and platform-specific build instructions
land here as Phase 0 progresses.

## CI

GitHub Actions under [`.github/workflows/`](.github/workflows/):

- **`test.yml`** — push / PR. Runs `cargo test --workspace` plus the UI
  build and Vitest suite across ubuntu / windows / macos. CPU-only; no GPU
  feature flags.
- **`build.yml`** — push to `main` + manual dispatch. Builds the one
  release bundle v1 ships per platform (Vulkan on Windows/Linux, Metal on
  macOS) and uploads it as a build artifact. Builds succeed unsigned;
  signing applies only when the signing secrets are present.
- **`release.yml`** — `v*` tags. Same per-OS bundle build with signing +
  notarization (gated on secrets), publishes a GitHub Release and the
  `tauri-plugin-updater` `latest.json` manifest.

[`scripts/generate-update-manifest.py`](scripts/generate-update-manifest.py)
(run via `uv run`) is a fallback for assembling `latest.json` by hand from
built bundles and their `.sig` files, for off-CI release work.

Cross-OS signed builds and the GPU hardware matrix are validated in CI / on
hardware, not locally. The matrix lives in
[`architecture/cross-cutting.md`](architecture/cross-cutting.md) under "GPU
portability".

## Rendering the architecture diagrams

```bash
scripts/render-architecture.sh
```

Requires Docker. Uses `structurizr/structurizr` to export the DSL to
Mermaid, then `minlag/mermaid-cli` to convert to SVG. SVGs are committed
alongside [`architecture/workspace.dsl`](architecture/workspace.dsl).

## Running spikes on native Windows

Phase 0 §4 exit criteria require each spike to run on Windows AND Linux.
The Linux side is verified in WSL/native. The Windows side uses
[`scripts/run-spike-windows.ps1`](scripts/run-spike-windows.ps1), which
robocopies the repo to `C:\Users\anl\meeting-app`, sets up the MSVC
dev shell via vswhere + Launch-VsDevShell.ps1, and runs `cargo build`
plus the spike binary.

From WSL:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File 'C:\Users\anl\meeting-app\scripts\run-spike-windows.ps1' \
  -Spike asr -Run
```

The script must already exist on the Windows side (the `\\wsl.localhost\…`
UNC path isn't reliably executable). For the first invocation, sync the
script over with `cp /home/anl/meeting-app/scripts/run-spike-windows.ps1
/mnt/c/Users/anl/meeting-app/scripts/run-spike-windows.ps1` or run
`-SyncOnly` from any already-Windows-side copy first.

Toolchain expectations: Rust on PATH, Visual Studio Build Tools 2022,
LLVM (for `libclang.dll`), Vulkan SDK (optional, only needed for the
`vulkan` feature). Edit the env-var paths at the top of the script if
your install differs.
