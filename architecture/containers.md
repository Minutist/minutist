# C4 Level 2 — Containers

![Containers](L2_Containers.svg)

## Containers

| Container | Tech | Lifetime | Responsibility |
|---|---|---|---|
| **Webview UI** | React 19 + TypeScript + Tiptap, inside a Tauri webview | Per-window | Notes editor, transcript pane, meeting controls, audio meter, meeting-list view. No business logic. |
| **Rust core** | Tauri 2 main process | Per app instance | All native subsystems — audio, ASR, diarization, summarisation, persistence, settings. Exposes a typed command + event surface. |
| **llama.cpp** | Bundled native lib, Vulkan / Metal / CPU | Loaded per model | ASR (Qwen3-ASR via the mtmd module) and summary-LLM inference. |
| **sherpa-onnx** | Bundled native lib, ONNX Runtime | Loaded on demand | Diarization (pyannote/segmentation-3.0 segmentation + 3D-Speaker CAM++ speaker embeddings + clustering). |
| **libsql index** | SQLite file at `{app-data}/index.db` | Per app instance, persistent | Fast list / search over per-meeting metadata. |
| **Meeting filesystem** | Per-meeting directories under `{app-data}/meetings/{uuid}/` | Persistent | Authoritative store for audio, transcript, notes, summary, metadata. |

## Why two native libs and not one

`llama.cpp` and `sherpa-onnx` ship as separate native dependencies because
they serve different inference shapes:

- llama.cpp is a single backend that we use for both ASR and the
  summary LLM (the value of the architectural pivot — one runtime for
  two workloads).
- sherpa-onnx is the diarization pipeline. We do not own enough audio /
  ML engineering capacity to reimplement the pyannote/segmentation-3.0
  (MIT) segmentation model and the 3D-Speaker CAM++ (Apache-2.0) speaker
  embeddings plus clustering. It's a hard external dependency at the
  runtime layer.

The cost is an extra 50-80 MB of native libs in the bundle and a second
backend to keep up to date. Worth it.

## Data flow

```
mic ─▶ Rust core ─▶ webview              (audio meter; transcript events)
       Rust core ─▶ filesystem            (audio.opus, transcript.json,
                                            notes.json, summary.md)
       Rust core ─▶ libsql                (metadata index)
       Rust core ◀─ webview               (commands: start/stop, save notes,
                                            run summary, open meeting)
       Rust core ◀▶ llama.cpp             (ASR + summary inference, FFI)
       Rust core ◀▶ sherpa-onnx           (diarization inference, FFI)
```

The webview never reaches a native lib or the filesystem directly. All
its writes go through the Rust core's IPC surface, and all its reads
come back as typed events.

## Process model

Single process. Tauri runs the Rust core; the webview runs in an
embedded webview engine (WebView2 / WebKit / WebKitGTK) hosted by the
same process. No worker subprocesses in v1.

If profiling shows the ASR or summarisation workload starves the UI
thread, the first mitigation is to move that work to a dedicated tokio
task pool (already the plan — see `cross-cutting.md`). Spinning up a
subprocess only becomes a question if pinning whole CPU cores to ASR is
needed; that decision is deferred until evidence demands it.

## Bundle topology

**Aspirational — not yet wired.** The layout below is the intended
packaged shape. It is *not* the current state: `src-tauri/tauri.conf.json`
has no `bundle.resources` key today, so none of these `resources/`
sub-trees are actually placed into a packaged build yet. Productionising
the bundle (resource declarations, per-platform native-lib staging) is
open work.

```
meeting-app(.exe)                          Tauri binary (Rust core + webview host)
├── resources/                             (intended; no bundle.resources key today)
│   ├── llama/                             llama.cpp shared libs (per platform)
│   ├── sherpa/                            sherpa-onnx shared libs
│   └── ui/                                Built React bundle
└── (no models bundled — downloaded on first run)
```

Models are not in the bundle. The installer is targeted at ~50-100 MB;
the first-run flow downloads ~2-4 GB of model weights to the app data
directory.
