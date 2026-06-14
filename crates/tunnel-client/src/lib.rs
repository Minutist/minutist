//! App-side tunnel client (WS4-A S3b).
//!
//! The device half of the connected-tier tunnel. It dials the hosted relay over
//! WSS, completes the device handshake, then forwards each relayed MCP request
//! to the app's own loopback `mcp-server` and streams the response back to the
//! relay. The relay is the OAuth resource server and the public MCP endpoint;
//! this client keeps a single MCP implementation (the app's) behind it.
//!
//! # Injection, not coupling
//!
//! The crate takes no workspace edge. The relay URL, the device credential, the
//! account id, and the loopback target (base URL + internal bearer) are all
//! passed in as configuration. `app-main` (S5) sources the loopback target from
//! `ipc-bridge::McpServerInfo`; this crate never imports `mcp-server` or
//! `ipc-bridge` — it talks to the loopback server over HTTP like any client.

mod loopback;

pub use loopback::{InternalBearer, LoopbackTarget};
