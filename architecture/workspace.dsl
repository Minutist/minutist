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

        irohRelay = softwareSystem "Sync relay (sync.minutist.ai)" "Self-hosted iroh relay. Brokers QUIC connectivity (NAT traversal / fallback) between a user's paired devices. Sees ciphertext only — never meeting plaintext. Server-side only, never shipped." {
            tags "External" "Optional"
        }

        connectorRelay = softwareSystem "Connected-tier relay (minutist-relay)" "Hosted backend behind mcp.minutist.ai (WSS tunnel relay, dialled outbound by tunnel-client to replay MCP requests against the app's loopback mcp-server) and api.minutist.ai (RFC 8628 device-code pairing API). Server-side only, never shipped; user-overridable for self-hosting." {
            tags "External" "Optional"
        }

        phoneCompanion = softwareSystem "Phone companion (minutist-mobile)" "Android Capacitor app in the sibling minutist-mobile repo. Bundles the sync-ffi .so and calls its UniFFI surface to join the same paired-device sync mesh as the desktop and the headless hub." {
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
                transcriptUi = component "Transcript pane" "Live-appending transcript view, with the live audio meter rendered above it. Per-segment speaker chip with a live colour dot (deterministic speaker_id -> palette slot) when diarization labels are present. Read-only; emits hover/click cross-reference events." "React"
                meetingShell = component "Meeting shell" "Window chrome, start/stop/pause controls, meeting-list view." "React"
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

                embedder = component "embedder" "BGE-M3 text embedder: the common::Embedder impl over a held llama.cpp model (CLS-pooled, 1024-dim, L2-normalised). Model-loading leaf, the embedding peer of summariser/asr-runtime. (RAG Phase B)" "Rust crate: crates/embedder"

                ragRetrieval = component "rag-retrieval" "Pure retrieval logic: char/turn chunking + cosine ranking + RRF fusion. Depends only on common; the concrete embedder is injected via the common::Embedder seam. (RAG Phase A/B)" "Rust crate: crates/rag-retrieval"

                modelRegistry = component "model-registry" "Model download, hash verification, on-disk catalogue. Owns the model cache directory; everything that needs a model goes through it." "Rust crate: crates/model-registry"

                notesCrdt = component "notes-crdt" "Leaf carrying the notes-CRDT primitives extracted from persistence: the Yjs (yrs) notes.ydoc + its JSON/markdown projections (NotesStore, ydoc), the MeetingFolder layout, and the metadata.json writer. No libsql/audiopus/ogg, so sync can transport the CRDT and cross-compile to mobile. persistence re-exports it at the historical paths." "Rust crate: crates/notes-crdt"

                persistence = component "persistence" "Per-meeting folder layout, libsql index, Opus audio encoding, Tiptap-JSON storage. Owns disk and database; delegates the notes-CRDT primitives to notes-crdt and re-exports them." "Rust crate: crates/persistence"

                orchestrator = component "meeting-orchestrator" "The live recording state machine. Wires audio-capture → vad-chunker → asr-runtime → persistence, emits typed events for the UI. The only crate that depends on multiple other components." "Rust crate: crates/orchestrator"

                agentTools = component "agent-tools" "Shared tool layer. One Tool trait + ToolRegistry; the single place a tool is defined; driven by both the chat agent and (Phase 10) the MCP server." "Rust crate: crates/agent-tools"

                chatAgent = component "chat-agent" "Stateless, OpenAI-compatible, tool-calling chat TURN engine over the bundled local LLM. Reuses the summariser model substrate + the agent-tools descriptors; the driver (ipc-bridge) owns history + the loop + tool dispatch." "Rust crate: crates/chat-agent"

                mcpServer = component "mcp-server" "In-process Streamable HTTP MCP server (loopback). Projects the agent-tools registry onto tools/list / tools/call; bearer + Host/Origin auth. Settings-gated, off by default." "Rust crate: crates/mcp-server"

                tunnelClient = component "tunnel-client" "App-side half of the connected-tier relay tunnel (WS4-A). Dials the hosted relay OUTBOUND over WSS, re-implements the relay's postcard wire frames, and replays relayed MCP requests against the loopback mcp-server with the internal bearer. No workspace edge; connected-feature gated; wired into app-main in S5." "Rust crate: crates/tunnel-client"

                sync = component "sync" "Device-to-device sync engine (WS4-B): iroh QUIC transport over a custom SYNC_ALPN. Exchanges Yjs notes-update frames, content-addressed meeting media (audio + note assets, over a second iroh-blobs ALPN), the processing-lifecycle Discovery exchange, and derived-artifact (transcript.json / summary.md) reconciliation. A near-leaf: depends only on common + notes-crdt, never persistence, which keeps its lib cross-compilable to mobile targets. Connected-feature gated; wired into app-main in S5." "Rust crate: crates/sync"

                election = component "election" "Host-election state machine for the producer gate (WS4-B): claims a claimable meeting (PendingProcessing, or a Claimed past its lease) with audio already synced in, runs the pipeline, and writes Processed — via the ElectionDriver trait. A leaf (common + persistence + notes-crdt): the sync (advertise) and orchestrator (process) collaborators sit behind the trait, so this crate takes no edge to either and the one state machine is reused by both eligible host types. Connected-feature gated; wired into app-main in S4." "Rust crate: crates/election"

                settings = component "settings" "Settings schema, validation, change notifications. Persists to a single JSON file at {app-data}/settings.store via serde_json + std::fs; no tauri dependency." "Rust crate: crates/settings"

                docConvert = component "doc-convert" "Converts attached document bytes (PDF, XLSX, PPTX, DOCX, HTML, EML, ODS, CSV/TSV, JSON/YAML/XML, txt/md/log) to canonical markdown; routes images (PNG/JPEG/TIFF) to an injected OCR backend. Pure-Rust in-process; catch_unwind sandboxed. Public surface: convert_to_markdown + supported_exts." "Rust crate: crates/doc-convert"

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

            headlessHub = container "Minutist Server" "User-installed headless daemon (minutist-hub). An always-on sync hub other devices converge through, and post-launch a GPU processing node. Pairs into the device mesh like a desktop; holds meeting plaintext in its own data root, on hardware the user owns. Optional; not the relay." "Rust / tokio (headless)" {
                tags "Container" "Backend"
            }

            syncFfiBridge = container "sync-ffi (phone bridge)" "UniFFI wrapper over sync::SyncEngine (crates/sync-ffi), cross-compiled to an aarch64-linux-android cdylib and bundled by the separate phone companion app. Mobile-only: not linked by app-main or headless; a workspace member built as its own artifact, not part of the Rust core process." "Rust crate: crates/sync-ffi (cdylib, aarch64-linux-android)" {
                tags "Container" "Native"
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
        minutist -> irohRelay "Syncs paired devices (NAT traversal / relay fallback; ciphertext only)" "QUIC / HTTPS"
        phoneCompanion -> minutist "Syncs meetings/notes/lifecycle with paired devices" "QUIC (iroh)"
        minutist -> connectorRelay "Dials the relay tunnel (device pairing + MCP relay) and the account/pairing API (connected tier, opt-in)" "WSS / HTTPS"

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
        minutist.core -> minutist.headlessHub "Reconciles notes + media + derived artifacts over iroh QUIC (mutual device sync)"
        minutist.core -> irohRelay "NAT traversal / relay fallback (ciphertext only)" "QUIC"
        minutist.headlessHub -> irohRelay "NAT traversal / relay fallback (ciphertext only)" "QUIC"
        minutist.core.tunnelClient -> connectorRelay "Dials the relay tunnel outbound (WSS); pairing client posts to the account API" "WSS / HTTPS"
        phoneCompanion -> minutist.syncFfiBridge "Bundles the .so; calls the UniFFI sync surface" "JNI"
        minutist.syncFfiBridge -> phoneCompanion "Pushes lifecycle/peer-arrival events to registered listeners" "JNI callback"

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
        minutist.core.notesCrdt    -> minutist.core.common "Uses interface types"
        minutist.core.sync         -> minutist.core.common "Uses interface types"
        minutist.core.election     -> minutist.core.common "Uses interface types"
        minutist.core.embedder     -> minutist.core.common "Uses interface types (Embedder, voiceprint_math, shared_llama_backend)"
        minutist.core.ragRetrieval -> minutist.core.common "Uses interface types (Embedder seam, voiceprint_math)"

        // Live pipeline. Orchestrator wires the dataflow.
        minutist.core.audioCapture -> microphone "Captures audio" "cpal"
        minutist.core.orchestrator -> minutist.core.audioCapture "Starts/stops capture; consumes frames"
        minutist.core.orchestrator -> minutist.core.vadChunker "Feeds frames; consumes AudioChunks"
        minutist.core.orchestrator -> minutist.core.asrRuntime "Dispatches chunks via AsrBackend trait (Qwen tiers; non-Parakeet languages)"
        minutist.core.orchestrator -> minutist.core.asrParakeet "Dispatches chunks via AsrBackend trait (Parakeet languages; primary)"
        minutist.core.orchestrator -> minutist.core.persistence "Streams audio + segments to disk"
        minutist.core.orchestrator -> minutist.core.diarizer "On stop: assigns speakers via Diarizer trait (authoritative); during recording: live per-segment labels via OnlineDiarizer (additive)"
        minutist.core.orchestrator -> minutist.core.ipcBridge "Emits transcript / meter / state events"

        // Model lifecycle. asr-runtime / asr-parakeet depend only on common; all
        // model-registry resolution lives in the orchestrator (runner::build_diarizer
        // / init_asr_runtime), which hands the resolved model path to the ASR/diarizer
        // backend rather than either resolving its own model.
        minutist.core.orchestrator -> minutist.core.modelRegistry "Resolves model dirs (ensure/list) for ASR + diarizer"
        minutist.core.modelRegistry -> modelHost "Reads / downloads model files"

        // FFI boundaries.
        minutist.core.asrRuntime  -> minutist.llamaNative "Inference" "llama-cpp-2 FFI"
        minutist.core.summariser  -> minutist.llamaNative "Inference" "llama-cpp-2 FFI"
        minutist.core.embedder    -> minutist.llamaNative "Embedding inference" "llama-cpp-2 FFI"
        minutist.core.diarizer    -> minutist.sherpaNative "Inference" "sherpa-rs FFI"

        // Persistence.
        minutist.core.persistence -> minutist.core.notesCrdt "Delegates the notes-CRDT primitives; re-exports them"
        minutist.core.persistence -> minutist.sqliteDb "Index reads/writes" "libsql"
        minutist.core.persistence -> minutist.meetingFs "Per-meeting file I/O"
        minutist.core.notesCrdt   -> minutist.meetingFs "notes.ydoc / notes.json / notes.md + metadata.json I/O"

        // Summarisation triggered by user action; orchestrator is bypassed once
        // the meeting is stopped. summariser takes transcript/notes as plain
        // parameters (Summariser::summarise) — it has no persistence edge itself;
        // ipc-bridge reads via persistence::read_transcript / writes via
        // persistence::write_summary around the call (see the ipcBridge -> persistence
        // edge below).
        minutist.core.summariser  -> externalLlm "Optional dispatch" "HTTP"

        // Settings. The settings crate has no tauri dependency; it persists
        // directly to a JSON file at {app-data}/settings.store via std::fs (the
        // path is injected by app-main, outside the per-meeting filesystem).
        minutist.core.orchestrator -> minutist.core.settings "Reads runtime config"

        // Shared tool layer (Phase 9). One Tool trait + ToolRegistry, driven by
        // both the chat agent and (Phase 10) the MCP server. Reads meeting
        // artefacts through persistence; runs re-transcribe / rediarize /
        // transcribe_pcm_window through the orchestrator (which keeps the
        // model-registry edge — agent-tools has none).
        minutist.core.agentTools -> minutist.core.persistence "Reads meeting artefacts; writes via existing writers"
        minutist.core.agentTools -> minutist.core.notesCrdt "Reads/writes metadata.json + notes.ydoc via the lifted primitives"
        minutist.core.agentTools -> minutist.core.orchestrator "Re-transcribe / rediarize / transcribe_pcm_window"
        minutist.core.agentTools -> minutist.core.ragRetrieval "rrf_fuse for the retrieve_chunks tool"

        // RAG (Phase B/D). ipc-bridge drives the write path (chunk + embed + persist
        // to the per-meeting meeting.db) and holds the embedder; agent-tools'
        // retrieve_chunks queries it, and the live-agent worker both retrieves and
        // incrementally indexes transcript per refresh (Phase D). embedder is the
        // model-loading leaf; rag-retrieval is pure logic.
        minutist.core.ipcBridge -> minutist.core.embedder "Constructs + holds the BGE-M3 embedder"
        minutist.core.ipcBridge -> minutist.core.ragRetrieval "chunk_text (write path) + rrf_fuse (live-agent retrieval)"

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

        // Sync engine + producer-gate election (WS4-B). sync is a near-leaf
        // (common + notes-crdt only, never persistence) so its lib cross-compiles
        // to mobile; election is a leaf (common + persistence + notes-crdt) — the
        // sync (advertise) and orchestrator (process) collaborators it drives sit
        // behind the ElectionDriver trait, so neither is a workspace edge of this
        // crate. app-main injects the connected implementations of both behind
        // the same `connected` feature as mcp-server / tunnel-client.
        minutist.core.sync      -> minutist.core.notesCrdt "Reads/merges the authoritative notes.ydoc via NotesStore; MeetingFolder::ensure for inbound folders"
        minutist.core.election  -> minutist.core.persistence "Scans candidates (folder::list_meeting_ids) and claims/renews/reaps via the guarded update_metadata_if re-export"
        minutist.core.election  -> minutist.core.notesCrdt "Reads the projected notes.ydoc state for claim/renew/reap bookkeeping"
        minutist.core.appMain   -> minutist.core.sync "Injects the connected SyncControl (ConnectedSync); the free build wires disabled_sync() instead (connected-gated)"
        minutist.core.appMain   -> minutist.core.election "Spawns run_election_loop with the DesktopElectionDriver (connected-gated)"
        minutist.headlessHub    -> minutist.core.sync "Wires SyncEngine into the always-on hub daemon"
        minutist.headlessHub    -> minutist.core.tunnelClient "AccountDirectoryClient: publishes this hub's endpoint, fetches the account's device list (unconditional, not feature-gated)"
        minutist.syncFfiBridge  -> minutist.core.sync "Wraps SyncEngine's transport + lifecycle surface via UniFFI"
        minutist.syncFfiBridge  -> minutist.core.notesCrdt "Reads/writes metadata.json + notes.ydoc via the lifted primitives (persistence-free)"

        // IPC bridge — the ONLY crate that knows about Tauri APIs.
        minutist.core.ipcBridge -> minutist.core.orchestrator "Invokes commands; subscribes to events"
        minutist.core.ipcBridge -> minutist.core.persistence "Meeting list / load / delete"
        minutist.core.ipcBridge -> minutist.core.notesCrdt "Reads/writes metadata.json + notes.ydoc directly for the leaf's own command surface"
        minutist.core.ipcBridge -> minutist.core.summariser  "Triggers Summarise; holds the LLM substrate"
        minutist.core.ipcBridge -> minutist.core.settings    "Get / set settings"
        minutist.core.ipcBridge -> minutist.core.agentTools  "Builds the registry + context; dispatches tools"
        minutist.core.ipcBridge -> minutist.core.chatAgent   "Holds the engine; drives the turn loop"
        minutist.core.ipcBridge -> minutist.core.docConvert  "Enqueues attachment conversion jobs; calls convert_to_markdown in the bounded worker"
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
