# meeting-app

Local-first desktop meeting-notes application. Records meetings, transcribes them on-device, takes hand-typed notes alongside, summarises with a local LLM. Cross-platform (Windows / macOS / Linux).

**Status:** pre-prototype. Phase 0 spikes in progress.

## Workspace layout

```
crates/
  common/         shared types
spikes/
  asr/            llama-cpp-2 mtmd ASR spike
  llm/            llama-cpp-2 text LLM spike
  vad-loop/       Silero VAD + ASR end-to-end
  diarize/        sherpa-onnx diarization spike
```

The `spikes/` crates are deliberately throwaway. Once Phase 0 exits, the patterns that work move into `crates/common` and a real application crate.

## Build

```bash
cargo build --workspace
```

Models, native dependencies, and platform-specific build instructions land here as Phase 0 progresses.
