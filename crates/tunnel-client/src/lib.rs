//! App-side tunnel client for the connected tier.
//!
//! The device half of the connected-tier tunnel. It dials the hosted relay over
//! WSS, completes the device handshake, then forwards each relayed MCP request
//! to the app's own loopback `mcp-server` and streams the response back to the
//! relay. The relay is the OAuth resource server and the public MCP endpoint;
//! this client keeps a single MCP implementation (the app's) behind it.
//!
//! Also bundled here: [`AccountDirectoryClient`], the account device-directory
//! HTTP client (`GET /v1/account/devices`, `PUT
//! /v1/account/devices/self/endpoint`, both bearer-authed with the device's
//! `mdc_` credential). It lives here rather than in `sync` so this crate keeps
//! no `sync` edge; the caller (`app-main`, via the `account-directory` adapter
//! crate) adapts its [`DeviceEndpointEntry`] DTO onto `sync::AccountEndpointSource`.
//!
//! # Public surface
//!
//! [`run_tunnel`] is the entry point: given a [`TunnelConfig`] it performs one
//! connect attempt, runs until the connection closes, and returns.
//! [`run_tunnel_with_observer`] is the same connect-and-run plus a one-shot
//! callback fired the instant the relay acknowledges the `Hello`, used by
//! [`reconnect_loop`] to learn "this credential has worked". [`reconnect_loop`]
//! supervises `run_tunnel_with_observer` with capped exponential backoff +
//! jitter and reports [`ConnectionState`]; [`TunnelHandle`] wraps that loop with
//! start/stop lifecycle (completion-handle discipline: `stop()` awaits the
//! task, including the in-flight-request `JoinSet` abort, before returning).
//! [`DeviceCodeClient`] (in `pairing`) is the app-side RFC 8628 device-code
//! client used to obtain the `mdc_` credential in the first place.
//!
//! # Injection, not coupling
//!
//! The crate takes no workspace edge — it is a near-leaf consumer of
//! third-party crates only, part of the connected feature surface (the free
//! build has no relay). It lives in the workspace unconditionally (so it always
//! builds and tests) but is only wired into `app-main` behind the `connected`
//! Cargo feature. The relay URL, the device credential, the account id, and the
//! loopback target (base URL + internal bearer) are all passed in as
//! configuration. `app-main` sources the loopback target from
//! `ipc-bridge::McpServerInfo`; this crate never imports `mcp-server` or
//! `ipc-bridge` — it talks to the loopback server over HTTP like any client.
//!
//! # Wire contract
//!
//! The relay lives in a separate private repo, so the two ends do not share a
//! crate. The `frame` module re-implements the relay's `Frame` enum
//! byte-for-byte: `PROTOCOL_VERSION` carried in `Hello`, postcard
//! (`default-features = false` + `alloc`) with one frame per binary WebSocket
//! message. The variant order in `Frame` is part of the contract, since postcard
//! encodes the enum discriminant by index — it must not be reordered without a
//! coordinated change to the relay's encoder. The match is pinned by a committed
//! cross-impl fixture (`tests/fixtures/relay_frames.txt`, the relay encoder's
//! hex for a known frame set) that a unit test asserts this crate's encoding
//! equals and decodes back.
//!
//! Handshake + demux: dial, send `Hello`, await `HelloAck` (or fail on
//! `HelloErr`). A single writer task drains a bounded outbound channel onto the
//! socket; the read loop receives `Request` frames and spawns a bounded
//! per-request task (a `Semaphore`-capped pool) that replays the request against
//! the loopback target and streams the HTTP response back as `ResponseStart →
//! ResponseChunk* → ResponseEnd` (or `ResponseError`). Concurrent requests
//! multiplex by `request_id`. No unbounded channels or task spawning: the
//! inflight semaphore and the bounded outbound channel bound the work, and the
//! read loop drains the per-request `JoinSet` as entries complete rather than
//! only at session end, so a long-lived connection does not accumulate one
//! `JoinSet` entry per request ever served.
//!
//! # Security
//!
//! The internal loopback bearer is held in [`InternalBearer`], whose `Debug`
//! redacts the value. It is attached only to the outbound loopback HTTP request
//! (via `HeaderMap::insert`, which replaces rather than appends — appending
//! would let a relay-supplied `authorization` ride alongside it) and is never
//! serialised into a tunnel frame nor logged. Request and response bodies are
//! never logged; only `request_id`, method, path, and status are traced.
//! Response bodies are streamed, not buffered whole. Loopback response headers
//! crossing back to the relay, and inbound request headers used to build the
//! loopback call, are both allowlisted (`FORWARDED_RESPONSE_HEADERS` /
//! `FORWARDED_REQUEST_HEADERS`), mirroring the relay's own header filtering so
//! a future header addition on either side can't cross the trust boundary by
//! default. `run_tunnel` refuses a non-`wss://` relay URL before dialing;
//! plaintext `ws://` is tolerated only for a loopback host.

mod account;
mod frame;
mod lifecycle;
mod loopback;
mod pairing;
mod reconnect;
mod run;

pub use account::{AccountDirectoryClient, AccountDirectoryError, DeviceEndpointEntry};
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
