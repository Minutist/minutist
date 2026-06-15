/*
 * minutist — Structurizr workspace
 *
 * Authoritative source for the C4 model. Rendered SVGs in this directory are
 * derived; do not edit them directly. Edit this DSL and re-run
 * scripts/render-architecture.sh.
 *
 * Levels modelled:
 *   1. System Context — what the minutist system talks to.
 *   2. Containers     — the deployable / runtime units that make up the app.
 *   3. Components     — the Rust crates inside the core container, with the
 *                       interfaces and traits that define their boundaries.
 *
 * Updates: any structural change (new container, new component, changed
 * interface) MUST land in the same commit as the source change. The
 * pre-commit hook flags drift; the reviewer agent treats out-of-domain edits
 * as findings.
 */

workspace "Minutist" "Local-first desktop meeting-notes application." {

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

        mcpClient = softwareSystem "External MCP client" "Connects over Streamable HTTP (loopback). Reads meetings + messages the internal agent over the agent-tools registry. Optional; requires the MCP server to be enabled." {
            tags "External" "Optional"
        }
        externalLlm = softwareSystem "External LLM (Ollama / LM Studio)" "Optional user-configured local LLM backend reachable over loopback HTTP. Default disabled." {
            tags "External" "Optional"
        }

        // ----------------------------------------------------------------
        // The minutist system
        // ----------------------------------------------------------------

        minutist = softwareSystem "Minutist" "Tauri desktop application. Records audio, transcribes locally via llama.cpp, takes hand-typed notes alongside, summarises with a local LLM." {

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

                mcpServer = component "mcp-server" "In-process Streamable HTTP MCP server (loopback). Projects the agent-tools registry onto tools/list / tools/call; bearer + Host/Origin auth. Settings-gated, off by default." "Rust crate: crates/mcp-server"

                tunnelClient = component "tunnel-client" "App-side half of the connected-tier relay tunnel (WS4-A). Dials the hosted relay OUTBOUND over WSS, re-implements the relay's postcard wire frames, and replays relayed MCP requests against the loopback mcp-server with the internal bearer. No workspace edge; connected-feature gated; wired into app-main in S5." "Rust crate: crates/tunnel-client"

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

            meetingFs = container "Meeting filesystem" "Per-meeting directory under {app-data}/meetings/{uuid}/. Holds audio.opus, transcript.json, notes.ydoc (authoritative CRDT), notes.json + notes.md (derived), summary.md, metadata.json." "Filesystem" {
                tags "Container" "Storage"
            }
        }

        // ----------------------------------------------------------------
        // Relationships — System Context (Level 1)
        // ----------------------------------------------------------------

        user -> minutist "Records meetings, types notes, runs summaries"
        minutist -> microphone "Captures audio via OS audio API"
        minutist -> modelHost "Reads GGUF + ONNX model files"
        minutist -> updateServer "Fetches signed updates" "HTTPS"
        minutist -> externalLlm "Optional: dispatches summary requests" "HTTP / loopback"
        mcpClient -> minutist "Reads meetings + messages the internal agent over MCP" "Streamable HTTP / loopback"

        // ----------------------------------------------------------------
        // Relationships — Container (Level 2)
        // ----------------------------------------------------------------

        user -> minutist.webview "Interacts with editor + controls"
        minutist.webview -> minutist.core "Tauri commands + events (typed via tauri-specta)"
        minutist.core -> microphone "Captures audio frames"
        minutist.core -> minutist.llamaNative "ASR + LLM inference" "FFI via llama-cpp-2"
        minutist.core -> minutist.sherpaNative "Diarization + Parakeet ASR" "FFI via sherpa-rs"
        minutist.core -> minutist.sqliteDb "Reads/writes meeting index" "libsql"
        minutist.core -> minutist.meetingFs "Reads/writes per-meeting files" "std::fs"
        minutist.core -> modelHost "Reads / downloads model files"
        minutist.core -> updateServer "Polls + applies signed updates"
        minutist.core -> externalLlm "Optional summary dispatch" "HTTP"
        mcpClient -> minutist.core "tools/list + tools/call (bearer + Host/Origin)" "Streamable HTTP / loopback"

        // ----------------------------------------------------------------
        // Relationships — Component (Level 3) — INSIDE the Rust core
        // ----------------------------------------------------------------

        // Every component (except common) depends on common. Drawn only once to
        // keep the diagram readable; the dependency rule is enforced by the
        // workspace conventions, not by every edge here.
        minutist.core.orchestrator -> minutist.core.common "Uses interface types"
        minutist.core.agentTools   -> minutist.core.common "Uses interface types"
        minutist.core.chatAgent    -> minutist.core.common "Uses interface types"
        minutist.core.appMain      -> minutist.core.common "Uses interface types"
        minutist.core.ipcBridge    -> minutist.core.common "Uses interface types"

        // Live pipeline. Orchestrator wires the dataflow.
        minutist.core.audioCapture -> microphone "Captures audio" "cpal"
        minutist.core.orchestrator -> minutist.core.audioCapture "Starts/stops capture; consumes frames"
        minutist.core.orchestrator -> minutist.core.vadChunker "Feeds frames; consumes AudioChunks"
        minutist.core.orchestrator -> minutist.core.asrRuntime "Dispatches chunks via AsrBackend trait (Qwen tiers; non-Parakeet languages)"
        minutist.core.orchestrator -> minutist.core.asrParakeet "Dispatches chunks via AsrBackend trait (Parakeet languages; primary)"
        minutist.core.orchestrator -> minutist.core.persistence "Streams audio + segments to disk"
        minutist.core.orchestrator -> minutist.core.diarizer "On stop: assigns speakers via Diarizer trait (authoritative); during recording: live per-segment labels via OnlineDiarizer (additive)"
        minutist.core.orchestrator -> minutist.core.ipcBridge "Emits transcript / meter / state events"

        // Model lifecycle.
        minutist.core.asrRuntime  -> minutist.core.modelRegistry "Resolves + loads ASR model"
        minutist.core.asrParakeet -> minutist.core.modelRegistry "Resolves + loads Parakeet ONNX model"
        minutist.core.modelRegistry -> modelHost "Reads / downloads model files"

        // FFI boundaries.
        minutist.core.asrRuntime  -> minutist.llamaNative "Inference" "llama-cpp-2 FFI"
        minutist.core.summariser  -> minutist.llamaNative "Inference" "llama-cpp-2 FFI"
        minutist.core.diarizer    -> minutist.sherpaNative "Inference" "sherpa-rs FFI"

        // Persistence.
        minutist.core.persistence -> minutist.sqliteDb "Index reads/writes" "libsql"
        minutist.core.persistence -> minutist.meetingFs "Per-meeting file I/O"

        // Summarisation triggered by user action; orchestrator is bypassed once
        // the meeting is stopped — summariser reads from persistence directly.
        minutist.core.summariser  -> minutist.core.persistence "Reads transcript + notes; writes summary.md"
        minutist.core.summariser  -> externalLlm "Optional dispatch" "HTTP"

        // Settings.
        minutist.core.settings    -> minutist.meetingFs "Persists user preferences" "tauri-plugin-store"
        minutist.core.orchestrator -> minutist.core.settings "Reads runtime config"
        minutist.core.asrRuntime   -> minutist.core.settings "Reads model selection"

        // Shared tool layer (Phase 9). One Tool trait + ToolRegistry, driven by
        // both the chat agent and (Phase 10) the MCP server. Reads meeting
        // artefacts through persistence; runs re-transcribe / rediarize /
        // transcribe_pcm_window through the orchestrator (which keeps the
        // model-registry edge — agent-tools has none).
        minutist.core.agentTools -> minutist.core.persistence "Reads meeting artefacts; writes via existing writers"
        minutist.core.agentTools -> minutist.core.orchestrator "Re-transcribe / rediarize / transcribe_pcm_window"

        // Chat agent (Phase 9). The stateless turn engine sits ABOVE both the
        // summariser substrate (borrows the loaded LlamaModel via the D5 seam)
        // and the agent-tools descriptors (for the oaicompat prompt + grammar).
        // The driver (ipc-bridge, a later phase) owns history + the loop + tool
        // dispatch.
        minutist.core.chatAgent -> minutist.core.summariser "Reuses the loaded model substrate (LlamaSummariser::model)"
        minutist.core.chatAgent -> minutist.core.agentTools "Reads tool descriptors for the prompt + grammar"

        // MCP server (Phase 10). A SECOND consumer of the agent-tools registry —
        // projects it onto tools/list / tools/call; no chat-agent edge (the
        // inter-agent bridge tool reaches the chat engine via a common-typed
        // channel whose driver lives in ipc-bridge). app-main spawns the listener.
        minutist.core.mcpServer -> minutist.core.common "Uses interface types"
        minutist.core.mcpServer -> minutist.core.agentTools "Projects the registry; dispatches tools/call"
        minutist.core.appMain   -> minutist.core.mcpServer  "Spawns the listener via tauri::async_runtime::spawn (settings-gated)"
        minutist.core.appMain   -> minutist.core.tunnelClient "Runs device pairing + the reconnect/lifecycle (connected-gated, WS4-A S5b); injects ConnectedTunnel as IpcState.tunnel"

        // IPC bridge — the ONLY crate that knows about Tauri APIs.
        minutist.core.ipcBridge -> minutist.core.orchestrator "Invokes commands; subscribes to events"
        minutist.core.ipcBridge -> minutist.core.persistence "Meeting list / load / delete"
        minutist.core.ipcBridge -> minutist.core.summariser  "Triggers Summarise; holds the LLM substrate"
        minutist.core.ipcBridge -> minutist.core.settings    "Get / set settings"
        minutist.core.ipcBridge -> minutist.core.agentTools  "Builds the registry + context; dispatches tools"
        minutist.core.ipcBridge -> minutist.core.chatAgent   "Holds the engine; drives the turn loop"
        minutist.core.appMain   -> minutist.core.ipcBridge   "Mounts command handlers"
        minutist.core.appMain   -> minutist.core.orchestrator "Owns lifetime"
        minutist.core.appMain   -> minutist.core.agentTools  "Wires the tool registry"

        // Webview ↔ ipc-bridge.
        minutist.webview.ipcClient -> minutist.core.ipcBridge "invoke + listen" "Tauri IPC"
        minutist.webview.editor       -> minutist.webview.ipcClient
        minutist.webview.transcriptUi -> minutist.webview.ipcClient
        minutist.webview.meetingShell -> minutist.webview.ipcClient
        minutist.webview.editor       -> minutist.webview.uiState
        minutist.webview.transcriptUi -> minutist.webview.uiState
        minutist.webview.meetingShell -> minutist.webview.uiState
    }

    views {

        systemContext minutist "L1_SystemContext" {
            include *
            autolayout lr
            description "Level 1 — minutist and the things outside it."
        }

        container minutist "L2_Containers" {
            include *
            autolayout tb
            description "Level 2 — runtime containers and bundled native dependencies."
        }

        component minutist.core "L3_CoreComponents" {
            include *
            autolayout tb
            description "Level 3 — Rust crates inside the core process. One crate per component; each is the unit of agent ownership."
        }

        component minutist.webview "L3_WebviewComponents" {
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
