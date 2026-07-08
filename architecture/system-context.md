# C4 Level 1 — System Context

![System Context](L1_SystemContext.svg)

## The system

**minutist** is a desktop application, free by default: nothing it does
requires network access out of the box. A connected tier (opt-in, off by
default) adds device-to-device sync and an external-MCP connector; both are
covered below as adjacent systems, not as a change to the free build's
no-network default.

## Actors and external dependencies

| External | Why it's external |
|---|---|
| **User** | The only human in the loop. Drives recording, types notes, triggers summaries. |
| **Microphone (OS device)** | Provided by the operating system. The app talks to it via WASAPI (Windows), CoreAudio (macOS), or ALSA/PulseAudio (Linux). |
| **Model files on disk** | GGUF (ASR + summary LLM) and ONNX (diarization + Parakeet ASR) files cached under the app data directory. The app reads them; on first run it downloads them from a vendored manifest. Treated as external because the lifetime is decoupled from the app version. |
| **Update endpoint** | Static HTTPS host for signed updates. The auto-updater is implemented (wired in `app-main` via `UpdaterExt`); the release config — endpoints and minisign pubkey — is still pending, so shipped builds do not yet poll a live endpoint. |
| **External LLM (optional)** | An Ollama or LM Studio instance the user has running locally. Off by default. When enabled, summarisation can dispatch over loopback HTTP instead of running the bundled LLM. |
| **External MCP client (optional, connected tier)** | Connects to the in-process MCP server over Streamable HTTP (loopback) when the server is enabled. Reads meetings and messages the internal agent over the `agent-tools` registry. |
| **Sync relay — sync.minutist.ai (optional, connected tier)** | Self-hosted iroh relay brokering QUIC connectivity (NAT traversal / fallback) between a user's own paired devices. Sees ciphertext only, never meeting plaintext. |
| **Connected-tier relay — minutist-relay (optional, connected tier)** | Hosted backend behind mcp.minutist.ai (a WSS tunnel `tunnel-client` dials outbound, replaying relayed MCP requests against the app's loopback MCP server) and api.minutist.ai (the device-pairing API). |
| **Phone companion — minutist-mobile (optional, connected tier)** | The Android companion app. Bundles the `sync` engine's UniFFI wrapper and joins the same paired-device sync mesh as the desktop over iroh QUIC. |

## Out of scope at this level

These are deliberately not actors:

- Calendar / meeting platforms (Zoom, Meet, Teams). The app does not
  integrate with them. It listens to the microphone.
- Cloud transcription or summarisation run on Minutist-operated
  infrastructure. The connected tier's relays broker ciphertext (sync) or
  proxy MCP tool-call bytes to a client the user configured (the connector) —
  neither runs ASR, diarization, or summarisation itself.
- Multi-tenant collaboration between different people's accounts. Sync is
  device-to-device for one user's own paired devices (optionally including
  their own headless server) — not shared/collaborative editing across users.

## Trust boundaries

- The user's filesystem is the trust boundary for the free build:
  everything inside it (audio, transcripts, notes, summaries, models) is
  the user's data and never leaves the machine except via explicit export.
- The connected tier (opt-in, off by default) extends that boundary to the
  user's own paired devices and their own optional headless server
  (`crates/headless`, the "Minutist Server" container in `containers.md`) —
  hardware the user owns, so it sits within the same trust boundary as the
  desktop. The sync relay sits outside that boundary but sees ciphertext
  only. The connected-tier relay also sits outside it: enabling the
  connector means the channel transits content to whatever external MCP
  client the user configured, by design — never described as "private"
  once enabled.
- Network access is opt-in per feature. Update polling becomes active
  once the release config sets a real update endpoint (the committed
  default leaves `endpoints` empty, so `check()` is a no-op); external-LLM
  dispatch, sync, and the connector are all off by default.

## Why this level exists

To force the scope question. If a future feature proposal adds an
external system here — e.g. "we'll sync meetings to Dropbox" — that's
visible on the L1 diagram and triggers a spec-level review before any
component work.
