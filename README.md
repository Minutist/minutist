# meeting-app

Local-first desktop meeting-notes application. Records meetings,
transcribes them on-device, takes hand-typed notes alongside, summarises
with a local LLM. Cross-platform (Windows / macOS / Linux).

**Status:** in development. The full pipeline (record → transcribe → notes →
summarise → opt-in diarize) is implemented across the workspace crates and the
Tauri 2 + React app; native Windows/macOS/Linux builds run. Distribution
(code-signing, auto-update) and on-hardware validation are the remaining work.

## Architecture

See [`architecture/`](architecture/). The C4 model diagrams and docs there
are authoritative for component boundaries, interfaces, and ownership.
Read [`architecture/README.md`](architecture/README.md) first.

## Workspace layout

```
architecture/     C4 docs + Structurizr DSL + rendered SVGs
crates/
  common/         shared types + trait definitions (the architectural contract)
  audio-capture/  cpal capture + device enumeration
  vad-chunker/    Silero VAD + batched-VAD chunking
  asr-runtime/    llama-cpp-2 mtmd ASR
  summariser/     llama-cpp-2 text-LLM summarisation
  diarizer/       sherpa-onnx speaker diarization (opt-in)
  model-registry/ model download + SHA-256 verification
  persistence/    meeting folders, Opus audio, transcript/notes/metadata, libsql index
  settings/       settings schema, store, change notifications
  orchestrator/   recording lifecycle + the live VAD→ASR pipeline
  ipc-bridge/     Tauri command + event surface (tauri-specta bindings)
src-tauri/        Tauri 2 app shell (app-main): bootstrap, tray, updater, capabilities
ui/               React 19 + Tiptap + Zustand frontend
spikes/           throwaway Phase-0 API-proof crates (asr, llm, vad-loop, diarize)
scripts/          build / test / render helpers
.githooks/
  pre-commit      architecture drift guard (install: see below)
```

Component boundaries, the dependency table, and ownership are authoritative in
[`architecture/components.md`](architecture/components.md). The `spikes/` crates
are throwaway Phase-0 API proofs, exempt from the cross-cutting rules.

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

## Development

Common tasks are wrapped in the [`Makefile`](Makefile) — run `make` (or
`make help`) for the full list:

| Command | Does |
|---|---|
| `make build` | debug build of the whole workspace |
| `make test` | the full default suite — `cargo test --workspace` + the UI build + Vitest |
| `make clippy` / `make fmt` | lint / format |
| `make bindings` | regenerate `ui/src/ipc/bindings.ts` from the Rust IPC surface |
| `make render-arch` | re-render the C4 SVGs (needs Docker) |
| `make clean` / `make clean-all` | remove `target/` + `ui/dist` (`clean-all` also drops `ui/node_modules`) |

`bindgen` (pulled in by `llama-cpp-2`) needs `libclang`, so the Makefile sets
`LIBCLANG_PATH=/usr/lib/llvm-18/lib`; override it in the environment if your LLVM
lives elsewhere. Models download on first run; GPU feature flags and the
single-portable-backend artefact policy are documented in
[`architecture/cross-cutting.md`](architecture/cross-cutting.md) under "GPU
portability".

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

## Native Windows builds + tests

These run on the Windows side from WSL via `powershell.exe`. The scripts
robocopy-mirror the repo to `C:\Users\anl\meeting-app`, set up the MSVC dev
shell (vswhere + `Launch-VsDevShell.ps1`), and build/test there. The frontend is
prebuilt in WSL and synced, so Node is not required on Windows.

- **App build** — `make windows-build` (CPU) or `make windows-build-vulkan`
  (GPU), wrapping
  [`scripts/build-windows-app.ps1`](scripts/build-windows-app.ps1). Output is a
  run-from-folder at `dist-windows\meeting-app[-vulkan]\` (the exe + native DLLs
  + the bundled Silero model) plus a zip; run `meeting-app.exe` directly, no
  install. It builds with `cargo tauri build --no-bundle` so the webview serves
  the embedded frontend over `tauri://` — a bare `cargo build` is dev-mode and
  points the webview at the (absent) Vite dev server.
- **Gated tests** —
  [`scripts/run-tests-windows.ps1`](scripts/run-tests-windows.ps1) runs the
  env-var-gated MSVC tests (`-Package <crate> -Ignored`).
- **Boot smoke** — [`scripts/smoke-windows.ps1`](scripts/smoke-windows.ps1) and
  [`scripts/verify-windows-zip.ps1`](scripts/verify-windows-zip.ps1) launch the
  built app / unzipped artifact briefly and capture the startup log.

Toolchain: Rust on PATH, Visual Studio Build Tools 2022, LLVM (`libclang.dll`),
and — for the `vulkan` feature — the Vulkan SDK + `ninja`. Edit the env-var paths
at the top of the scripts if your install differs.
