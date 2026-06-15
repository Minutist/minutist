//! App-side tunnel client (WS4-A S3b).
//!
//! The device half of the connected-tier tunnel. It dials the hosted relay over
//! WSS, completes the device handshake, then forwards each relayed MCP request
//! to the app's own loopback `mcp-server` and streams the response back to the
//! relay. The relay is the OAuth resource server and the public MCP endpoint;
//! this client keeps a single MCP implementation (the app's) behind it.
//!
//! # Public surface
//!
//! [`run_tunnel`] is the entry point: given a [`TunnelConfig`] it performs one
//! connect attempt, runs until the connection closes, and returns. Reconnection
//! and lifecycle (start/stop on the `connected` toggle) are S5 and live in
//! `app-main`; this crate exposes a single clean connect-and-run so that seam is
//! a `loop { run_tunnel(..).await; backoff().await }` wrapper later.
//!
//! # Injection, not coupling
//!
//! The crate takes no workspace edge. The relay URL, the device credential, the
//! account id, and the loopback target (base URL + internal bearer) are all
//! passed in as configuration. `app-main` (S5) sources the loopback target from
//! `ipc-bridge::McpServerInfo`; this crate never imports `mcp-server` or
//! `ipc-bridge` — it talks to the loopback server over HTTP like any client.
//!
//! # Security
//!
//! The internal loopback bearer is held in [`InternalBearer`], whose `Debug`
//! redacts the value. It is attached only to the outbound loopback HTTP request
//! and is never serialised into a tunnel frame nor logged. Request bodies and
//! response bodies are never logged; only `request_id`, method, path, and status
//! are traced.

mod frame;
mod lifecycle;
mod loopback;
mod pairing;
mod reconnect;
mod run;

pub use frame::{
    Frame, FrameError, Hello, HelloAck, HelloErr, HelloErrReason, RequestFrame, RequestId,
    ResponseChunk, ResponseEnd, ResponseError, ResponseStart, PROTOCOL_VERSION,
};
pub use lifecycle::TunnelHandle;
pub use loopback::{InternalBearer, LoopbackTarget};
pub use pairing::{
    next_interval, DeviceCodeClient, IssuedDeviceCredential, PairingError, PairingStart,
    PollOutcome, MIN_POLL_INTERVAL_SECS, SLOW_DOWN_INCREMENT_SECS,
};
pub use reconnect::{reconnect_loop, ConnectionState, ReconnectExit, BACKOFF_INITIAL, BACKOFF_MAX};
pub use run::{run_tunnel, run_tunnel_with_observer, TunnelConfig, TunnelError};
