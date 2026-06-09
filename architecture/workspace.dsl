/*
 * meeting-app — Structurizr workspace
 *
 * Authoritative source for the C4 model. Rendered SVGs in this directory are
 * derived; do not edit them directly. Edit this DSL and re-run
 * scripts/render-architecture.sh.
 *
 * Levels modelled:
 *   1. System Context — what the meeting-app system talks to.
 *   2. Containers     — the deployable / runtime units that make up the app.
 *   3. Components     — the Rust crates inside the core container, with the
 *                       interfaces and traits that define their boundaries.
 *
 * Updates: any structural change (new container, new component, changed
 * interface) MUST land in the same commit as the source change. The
 * pre-commit hook flags drift; the reviewer agent treats out-of-domain edits
 * as findings.
 */

workspace "meeting-app" "Local-first desktop meeting-notes application." {

    !identifiers hierarchical

    model {

        // ----------------------------------------------------------------
        // People and external systems
        // ----------------------------------------------------------------

        user = person "User" "A meeting participant taking notes while audio is captured and transcribed locally."

        microphone = softwareSystem "Microphone (OS device)" "System-provided audio input device, surfaced through the OS audio API (WASAPI / CoreAudio / ALSA / PulseAudio)." {
            tags "External"
        }

        modelHost = softwareSystem "Model files on disk" "GGUF and ONNX model files cached under the app data directory. Downloaded on first run from a vendored manifest." {
            tags "External"
        }

        updateServer = softwareSystem "Update endpoint" "Static HTTPS host serving signed update artefacts for tauri-plugin-updater. Updater is implemented (wired in app-main via UpdaterExt); release config — endpoints + minisign pubkey — is pending." {
            tags "External"
        }

        externalLlm = softwareSystem "External LLM (Ollama / LM Studio)" "Optional user-configured local LLM backend reachable over loopback HTTP. Default disabled." {
            tags "External" "Optional"
        }

        // ----------------------------------------------------------------
        // The meeting-app system
        // ----------------------------------------------------------------

        meetingApp = softwareSystem "meeting-app" "Tauri desktop application. Records audio, transcribes locally via llama.cpp, takes hand-typed notes alongside, summarises with a local LLM." {

            // ------------------------------------------------------------
            // Containers
            // ------------------------------------------------------------

            webview = container "Webview UI" "React 19 + TypeScript + Tiptap. The notes editor, transcript pane, and meeting controls. Runs inside the Tauri webview." "React / TypeScript / Tiptap" {
                tags "Container" "Frontend"

                editor       = component "Notes editor" "Tiptap WYSIWYG editor with markdown shortcuts. Owns paragraph-anchor timestamps." "Tiptap / @tiptap/react"
                transcriptUi = component "Transcript pane" "Live-appending transcript view. Per-segment speaker chip with a live colour dot (deterministic speaker_id -> palette slot) when diarization labels are present. Read-only; emits hover/click cross-reference events." "React"
                meetingShell = component "Meeting shell" "Window chrome, start/stop/pause controls, audio meter, meeting-list view." "React"
                ipcClient    = component "IPC client" "Typed wrapper around Tauri invoke + event APIs. Generated from Rust types via tauri-specta." "TypeScript / tauri-specta"
                uiState      = component "UI state store" "Zustand store. Holds derived UI state; transient only, no persistence." "Zustand"
            }

            core = container "Rust core" "Tauri main process. Hosts all native subsystems and exposes them to the webview via typed Tauri commands and events." "Rust / Tauri 2" {
                tags "Container" "Backend"

                // ------------------------------------------------------------
                // Components — one crate per component, depend only on `common`.
                // ------------------------------------------------------------

                common = component "common" "Shared interface types and trait definitions: Segment, AudioChunk, MeetingId, AsrBackend, Diarizer, Summariser, error types. The only crate every other component is allowed to depend on." "Rust crate: crates/common" {
                    tags "Component" "Shared"
                }

                audioCapture = component "audio-capture" "cpal-backed capture loop. Owns the audio device, sample-rate negotiation, ring buffer. Produces f32 frames at a fixed internal sample rate." "Rust crate: crates/audio-capture"

                vadChunker = component "vad-chunker" "Silero VAD via vad-rs, plus the smoothing wrapper. Converts a frame stream into silence-bounded AudioChunks with start_ms / end_ms." "Rust crate: crates/vad-chunker"

                asrRuntime = component "asr-runtime" "llama-cpp-2 (mtmd module) bound to Qwen3-ASR GGUF (0.6B CPU default + optional 1.7B GPU tier). Implements AsrBackend. Owns the Qwen ASR model lifecycle." "Rust crate: crates/asr-runtime"

                asrParakeet = component "asr-parakeet" "sherpa-onnx offline-transducer bound to Parakeet TDT 0.6B v3 (English + 24 EU langs). Implements AsrBackend with per-word timestamps. Primary engine for its languages; orchestrator routes by transcription language." "Rust crate: crates/asr-parakeet"

                diarizer = component "diarizer" "sherpa-onnx integration. Offline SherpaDiarizer (implements Diarizer) is the authoritative post-hoc pass on buffered audio after stop. Additive live OnlineDiarizer labels VAD segments during recording via embedding + a pure online clusterer (orchestrator-wired in Phase B, best-effort, overwritten by the on-stop pass)." "Rust crate: crates/diarizer"

                summariser = component "summariser" "llama-cpp-2 (text) for summary generation. Implements Summariser. Owns the text-LLM model lifecycle; also exposes the optional external-LLM dispatcher." "Rust crate: crates/summariser"

                modelRegistry = component "model-registry" "Model download, hash verification, on-disk catalogue. Owns the model cache directory; everything that needs a model goes through it." "Rust crate: crates/model-registry"

                persistence = component "persistence" "Per-meeting folder layout, libsql index, Opus audio encoding, Tiptap-JSON storage. Owns disk and database." "Rust crate: crates/persistence"

                orchestrator = component "meeting-orchestrator" "The live recording state machine. Wires audio-capture → vad-chunker → asr-runtime → persistence, emits typed events for the UI. The only crate that depends on multiple other components." "Rust crate: crates/orchestrator"

                agentTools = component "agent-tools" "Shared tool layer. One Tool trait + ToolRegistry; the single place a tool is defined; driven by both the chat agent and (Phase 10) the MCP server." "Rust crate: crates/agent-tools"

                chatAgent = component "chat-agent" "Stateless, OpenAI-compatible, tool-calling chat TURN engine over the bundled local LLM. Reuses the summariser model substrate + the agent-tools descriptors; the driver (ipc-bridge) owns history + the loop + tool dispatch." "Rust crate: crates/chat-agent"

                settings = component "settings" "Settings schema, validation, change notifications. Persists via tauri-plugin-store." "Rust crate: crates/settings"

                ipcBridge = component "ipc-bridge" "Tauri command + event surface. tauri-specta generates the TypeScript bindings consumed by the webview's IPC client. The only crate that knows about Tauri APIs." "Rust crate: crates/ipc-bridge"

                appMain = component "app-main" "Tauri main binary. Wires components, owns process lifetime, handles tray icon and window management." "Rust crate: src-tauri (bin)"
            }

            llamaNative = container "llama.cpp" "Bundled native library compiled with Vulkan / Metal / CPU backends. Used by both ASR and summary LLMs via llama-cpp-2." "C++ / Vulkan / Metal" {
                tags "Container" "Native"
            }

            sherpaNative = container "sherpa-onnx" "Bundled native library for diarization (pyannote/segmentation-3.0 segmentation + 3D-Speaker CAM++ embeddings + clustering)." "C++ / ONNX Runtime" {
                tags "Container" "Native"
            }

            sqliteDb = container "libsql index" "SQLite file at {app-data}/index.db. Mirrors per-meeting metadata for fast list + search." "libsql" {
                tags "Container" "Storage"
            }

            meetingFs = container "Meeting filesystem" "Per-meeting directory under {app-data}/meetings/{uuid}/. Holds audio.opus, transcript.json, notes.json, notes.md, summary.md, metadata.json." "Filesystem" {
                tags "Container" "Storage"
            }
        }

        // ----------------------------------------------------------------
        // Relationships — System Context (Level 1)
        // ----------------------------------------------------------------

        user -> meetingApp "Records meetings, types notes, runs summaries"
        meetingApp -> microphone "Captures audio via OS audio API"
        meetingApp -> modelHost "Reads GGUF + ONNX model files"
        meetingApp -> updateServer "Fetches signed updates" "HTTPS"
        meetingApp -> externalLlm "Optional: dispatches summary requests" "HTTP / loopback"

        // ----------------------------------------------------------------
        // Relationships — Container (Level 2)
        // ----------------------------------------------------------------

        user -> meetingApp.webview "Interacts with editor + controls"
        meetingApp.webview -> meetingApp.core "Tauri commands + events (typed via tauri-specta)"
        meetingApp.core -> microphone "Captures audio frames"
        meetingApp.core -> meetingApp.llamaNative "ASR + LLM inference" "FFI via llama-cpp-2"
        meetingApp.core -> meetingApp.sherpaNative "Diarization + Parakeet ASR" "FFI via sherpa-rs"
        meetingApp.core -> meetingApp.sqliteDb "Reads/writes meeting index" "libsql"
        meetingApp.core -> meetingApp.meetingFs "Reads/writes per-meeting files" "std::fs"
        meetingApp.core -> modelHost "Reads / downloads model files"
        meetingApp.core -> updateServer "Polls + applies signed updates"
        meetingApp.core -> externalLlm "Optional summary dispatch" "HTTP"

        // ----------------------------------------------------------------
        // Relationships — Component (Level 3) — INSIDE the Rust core
        // ----------------------------------------------------------------

        // Every component (except common) depends on common. Drawn only once to
        // keep the diagram readable; the dependency rule is enforced by the
        // workspace conventions, not by every edge here.
        meetingApp.core.orchestrator -> meetingApp.core.common "Uses interface types"
        meetingApp.core.agentTools   -> meetingApp.core.common "Uses interface types"
        meetingApp.core.chatAgent    -> meetingApp.core.common "Uses interface types"
        meetingApp.core.appMain      -> meetingApp.core.common "Uses interface types"
        meetingApp.core.ipcBridge    -> meetingApp.core.common "Uses interface types"

        // Live pipeline. Orchestrator wires the dataflow.
        meetingApp.core.audioCapture -> microphone "Captures audio" "cpal"
        meetingApp.core.orchestrator -> meetingApp.core.audioCapture "Starts/stops capture; consumes frames"
        meetingApp.core.orchestrator -> meetingApp.core.vadChunker "Feeds frames; consumes AudioChunks"
        meetingApp.core.orchestrator -> meetingApp.core.asrRuntime "Dispatches chunks via AsrBackend trait (Qwen tiers; non-Parakeet languages)"
        meetingApp.core.orchestrator -> meetingApp.core.asrParakeet "Dispatches chunks via AsrBackend trait (Parakeet languages; primary)"
        meetingApp.core.orchestrator -> meetingApp.core.persistence "Streams audio + segments to disk"
        meetingApp.core.orchestrator -> meetingApp.core.diarizer "On stop: assigns speakers via Diarizer trait (authoritative); during recording: live per-segment labels via OnlineDiarizer (additive)"
        meetingApp.core.orchestrator -> meetingApp.core.ipcBridge "Emits transcript / meter / state events"

        // Model lifecycle.
        meetingApp.core.asrRuntime  -> meetingApp.core.modelRegistry "Resolves + loads ASR model"
        meetingApp.core.asrParakeet -> meetingApp.core.modelRegistry "Resolves + loads Parakeet ONNX model"
        meetingApp.core.modelRegistry -> modelHost "Reads / downloads model files"

        // FFI boundaries.
        meetingApp.core.asrRuntime  -> meetingApp.llamaNative "Inference" "llama-cpp-2 FFI"
        meetingApp.core.summariser  -> meetingApp.llamaNative "Inference" "llama-cpp-2 FFI"
        meetingApp.core.diarizer    -> meetingApp.sherpaNative "Inference" "sherpa-rs FFI"

        // Persistence.
        meetingApp.core.persistence -> meetingApp.sqliteDb "Index reads/writes" "libsql"
        meetingApp.core.persistence -> meetingApp.meetingFs "Per-meeting file I/O"

        // Summarisation triggered by user action; orchestrator is bypassed once
        // the meeting is stopped — summariser reads from persistence directly.
        meetingApp.core.summariser  -> meetingApp.core.persistence "Reads transcript + notes; writes summary.md"
        meetingApp.core.summariser  -> externalLlm "Optional dispatch" "HTTP"

        // Settings.
        meetingApp.core.settings    -> meetingApp.meetingFs "Persists user preferences" "tauri-plugin-store"
        meetingApp.core.orchestrator -> meetingApp.core.settings "Reads runtime config"
        meetingApp.core.asrRuntime   -> meetingApp.core.settings "Reads model selection"

        // Shared tool layer (Phase 9). One Tool trait + ToolRegistry, driven by
        // both the chat agent and (Phase 10) the MCP server. Reads meeting
        // artefacts through persistence; runs re-transcribe / rediarize /
        // transcribe_pcm_window through the orchestrator (which keeps the
        // model-registry edge — agent-tools has none).
        meetingApp.core.agentTools -> meetingApp.core.persistence "Reads meeting artefacts; writes via existing writers"
        meetingApp.core.agentTools -> meetingApp.core.orchestrator "Re-transcribe / rediarize / transcribe_pcm_window"

        // Chat agent (Phase 9). The stateless turn engine sits ABOVE both the
        // summariser substrate (borrows the loaded LlamaModel via the D5 seam)
        // and the agent-tools descriptors (for the oaicompat prompt + grammar).
        // The driver (ipc-bridge, a later phase) owns history + the loop + tool
        // dispatch.
        meetingApp.core.chatAgent -> meetingApp.core.summariser "Reuses the loaded model substrate (LlamaSummariser::model)"
        meetingApp.core.chatAgent -> meetingApp.core.agentTools "Reads tool descriptors for the prompt + grammar"

        // IPC bridge — the ONLY crate that knows about Tauri APIs.
        meetingApp.core.ipcBridge -> meetingApp.core.orchestrator "Invokes commands; subscribes to events"
        meetingApp.core.ipcBridge -> meetingApp.core.persistence "Meeting list / load / delete"
        meetingApp.core.ipcBridge -> meetingApp.core.summariser  "Triggers Summarise"
        meetingApp.core.ipcBridge -> meetingApp.core.settings    "Get / set settings"
        meetingApp.core.appMain   -> meetingApp.core.ipcBridge   "Mounts command handlers"
        meetingApp.core.appMain   -> meetingApp.core.orchestrator "Owns lifetime"

        // Webview ↔ ipc-bridge.
        meetingApp.webview.ipcClient -> meetingApp.core.ipcBridge "invoke + listen" "Tauri IPC"
        meetingApp.webview.editor       -> meetingApp.webview.ipcClient
        meetingApp.webview.transcriptUi -> meetingApp.webview.ipcClient
        meetingApp.webview.meetingShell -> meetingApp.webview.ipcClient
        meetingApp.webview.editor       -> meetingApp.webview.uiState
        meetingApp.webview.transcriptUi -> meetingApp.webview.uiState
        meetingApp.webview.meetingShell -> meetingApp.webview.uiState
    }

    views {

        systemContext meetingApp "L1_SystemContext" {
            include *
            autolayout lr
            description "Level 1 — meeting-app and the things outside it."
        }

        container meetingApp "L2_Containers" {
            include *
            autolayout tb
            description "Level 2 — runtime containers and bundled native dependencies."
        }

        component meetingApp.core "L3_CoreComponents" {
            include *
            autolayout tb
            description "Level 3 — Rust crates inside the core process. One crate per component; each is the unit of agent ownership."
        }

        component meetingApp.webview "L3_WebviewComponents" {
            include *
            autolayout tb
            description "Level 3 — React components inside the webview. UI ownership."
        }

        styles {
            element "Person" {
                shape Person
                background #1168bd
                color #ffffff
            }
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Future" {
                background #cccccc
                color #555555
            }
            element "Optional" {
                background #b0b0b0
                color #ffffff
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Frontend" {
                background #438dd5
                color #ffffff
            }
            element "Backend" {
                background #2e6cae
                color #ffffff
            }
            element "Native" {
                background #555555
                color #ffffff
                shape Hexagon
            }
            element "Storage" {
                background #6b6b6b
                color #ffffff
                shape Cylinder
            }
            element "Component" {
                background #85bbf0
                color #000000
            }
            element "Shared" {
                background #f5cf6b
                color #000000
            }
        }
    }
}
