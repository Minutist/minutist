//! Dial-out, handshake, and the request/response demux loop.

use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::Message;

use crate::frame::{Frame, Hello, HelloErrReason, PROTOCOL_VERSION};
use crate::loopback::LoopbackTarget;

/// Upper bound on a single WebSocket message, matching the relay's
/// `MAX_TUNNEL_MESSAGE_BYTES` (4 MiB). Bounds the allocation a frame can force.
const MAX_TUNNEL_MESSAGE_BYTES: usize = 4 * 1024 * 1024;

/// Maximum number of relayed requests replayed against the loopback server
/// concurrently. Bounds spawned work — a flood of `Request` frames cannot spawn
/// unbounded tasks (cross-cutting rule: bounded concurrency, no unbounded
/// channels/tasks). Excess requests wait for a permit before their task starts.
const MAX_INFLIGHT_REQUESTS: usize = 32;

/// Capacity of the outbound frame channel feeding the single writer task. A
/// bounded channel applies back-pressure onto the per-request tasks when the
/// socket is slow rather than buffering unboundedly.
const OUTBOUND_CHANNEL_CAPACITY: usize = 256;

/// Configuration for one tunnel session.
#[derive(Clone, Debug)]
pub struct TunnelConfig {
    /// The relay rendezvous URL, e.g. `wss://mcp.minutist.ai/tunnel`.
    pub relay_url: String,
    /// The device credential the relay maps to an account.
    pub device_credential: String,
    /// The account this device serves.
    pub account_id: String,
    /// Where to replay relayed requests, and the internal bearer to apply.
    pub loopback: LoopbackTarget,
}

/// A failure running the tunnel.
#[derive(Debug, thiserror::Error)]
pub enum TunnelError {
    /// The relay URL scheme is not acceptable. The tunnel carries the loopback
    /// bearer and MCP content, so an off-machine relay must be `wss://`; a
    /// cleartext `ws://` relay is refused outright (and only tolerated for a
    /// loopback host) rather than dialed.
    #[error("relay_url must use wss:// (ws:// is allowed only for a loopback host)")]
    Config,
    /// The WSS dial or upgrade failed.
    #[error("failed to connect to the relay")]
    Connect(#[source] tokio_tungstenite::tungstenite::Error),
    /// A WebSocket transport error during the session.
    #[error("websocket transport error")]
    Transport(#[source] tokio_tungstenite::tungstenite::Error),
    /// The relay sent a frame that could not be decoded.
    #[error("undecodable frame from the relay")]
    Decode(#[source] crate::frame::FrameError),
    /// A frame could not be encoded for sending.
    #[error("failed to encode a frame")]
    Encode(#[source] crate::frame::FrameError),
    /// The relay rejected the device `Hello`.
    #[error("relay rejected the device: {0:?}")]
    HelloRejected(HelloErrReason),
    /// The relay sent a protocol violation during the handshake (wrong first
    /// frame, a text message, or an early close).
    #[error("handshake protocol violation: {0}")]
    Handshake(&'static str),
    /// The relay sent a protocol violation AFTER the handshake (a device→relay
    /// frame, a text message, or an oversized frame on the established session).
    #[error("tunnel protocol violation: {0}")]
    Protocol(&'static str),
    /// Building the loopback HTTP client failed.
    #[error("failed to build the loopback HTTP client")]
    Client(#[source] reqwest::Error),
}

/// Connect to the relay, complete the device handshake, and run the demux loop
/// until the connection closes. One connect attempt; returns on clean close or
/// the first transport/protocol error. Reconnection is the caller's concern (S5).
pub async fn run_tunnel(config: TunnelConfig) -> Result<(), TunnelError> {
    run_tunnel_with_observer(config, || {}).await
}

/// As [`run_tunnel`], but calls `on_handshake` exactly once the moment the
/// relay acknowledges the `Hello` (before the demux loop runs).
///
/// The S5 reconnect supervisor uses this to learn that *this credential has
/// worked*: an `AuthFailed` after a handshake means the device was revoked
/// (re-pair), whereas an `AuthFailed` before any handshake is a bad credential.
/// It is also the precise point to report the UI `Online` state, rather than
/// inferring it from a session that merely ran for a while.
pub async fn run_tunnel_with_observer<F>(
    config: TunnelConfig,
    on_handshake: F,
) -> Result<(), TunnelError>
where
    F: FnOnce(),
{
    // Refuse a cleartext relay before dialing: the tunnel carries the loopback
    // bearer and MCP content, so an encrypted wss:// channel is required for any
    // off-machine relay. A ws:// URL is tolerated only for a loopback host (a
    // cleartext tunnel that never leaves the machine — used by the test harness).
    if !relay_scheme_is_acceptable(&config.relay_url) {
        return Err(TunnelError::Config);
    }

    let client = reqwest::Client::builder()
        .build()
        .map_err(TunnelError::Client)?;

    let (ws_stream, _response) = tokio_tungstenite::connect_async(&config.relay_url)
        .await
        .map_err(TunnelError::Connect)?;
    let (mut ws_tx, mut ws_rx) = ws_stream.split();

    // --- Handshake: send Hello, await HelloAck/HelloErr ---
    let hello = Frame::Hello(Hello {
        version: PROTOCOL_VERSION,
        device_credential: config.device_credential.clone(),
        account_id: config.account_id.clone(),
    });
    let hello_bytes = hello.encode().map_err(TunnelError::Encode)?;
    ws_tx
        .send(Message::Binary(hello_bytes.into()))
        .await
        .map_err(TunnelError::Transport)?;

    match read_handshake_reply(&mut ws_rx).await? {
        HandshakeReply::Ack => {
            tracing::info!(target: "tunnel-client", account = %config.account_id, "tunnel: handshake acknowledged");
            on_handshake();
        }
        HandshakeReply::Err(reason) => {
            tracing::warn!(target: "tunnel-client", ?reason, "tunnel: relay rejected the device");
            return Err(TunnelError::HelloRejected(reason));
        }
    }

    // --- Pump: one writer task drains the outbound channel; the main loop reads
    // inbound frames and spawns bounded per-request tasks. ---
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<Frame>(OUTBOUND_CHANNEL_CAPACITY);
    let writer = tokio::spawn(async move {
        while let Some(frame) = outbound_rx.recv().await {
            let bytes = match frame.encode() {
                Ok(b) => b,
                Err(error) => {
                    tracing::error!(target: "tunnel-client", %error, "tunnel: outbound frame encode failed; dropping");
                    continue;
                }
            };
            if ws_tx.send(Message::Binary(bytes.into())).await.is_err() {
                tracing::info!(target: "tunnel-client", "tunnel: write failed; writer stopping");
                break;
            }
        }
        // Best-effort close on the way out.
        let _ = ws_tx.close().await;
    });

    let inflight = Arc::new(Semaphore::new(MAX_INFLIGHT_REQUESTS));
    let client = Arc::new(client);
    let loopback = Arc::new(config.loopback);

    // Owns the per-request tasks so they can be aborted when the session ends:
    // once the relay connection is gone their responses can never be delivered,
    // and leaving them detached would strand work across the S5 reconnect loop.
    let mut tasks = tokio::task::JoinSet::new();

    let loop_result = read_loop(
        &mut ws_rx,
        &outbound_tx,
        &inflight,
        &client,
        &loopback,
        &mut tasks,
    )
    .await;

    // Abort any still-in-flight per-request tasks (their connection is gone).
    tasks.shutdown().await;
    // Dropping outbound_tx lets the writer drain and finish.
    drop(outbound_tx);
    let _ = writer.await;

    loop_result
}

/// Whether `url` is an acceptable relay URL: `wss://` for any host, or `ws://`
/// only when the host is a loopback address (`127.0.0.1`, `::1`, `localhost`),
/// where cleartext never leaves the machine. Any other scheme is rejected.
///
/// The scheme is the text before the first `://`; the host is the authority up
/// to the first `/`, `:`, or end. A malformed URL (no `://`) is rejected.
fn relay_scheme_is_acceptable(url: &str) -> bool {
    let Some((scheme, rest)) = url.split_once("://") else {
        return false;
    };
    if scheme.eq_ignore_ascii_case("wss") {
        return true;
    }
    if scheme.eq_ignore_ascii_case("ws") {
        // Authority up to the first '/', then the host part. A bracketed IPv6
        // host (`[::1]:port`) keeps its colons inside the brackets.
        let authority = rest.split('/').next().unwrap_or("");
        let host = if let Some(after) = authority.strip_prefix('[') {
            after.split(']').next().unwrap_or("")
        } else {
            authority.split(':').next().unwrap_or("")
        };
        return host == "127.0.0.1" || host == "::1" || host.eq_ignore_ascii_case("localhost");
    }
    false
}

enum HandshakeReply {
    Ack,
    Err(HelloErrReason),
}

/// Read frames until the relay's reply to our `Hello`. A binary `HelloAck` /
/// `HelloErr` resolves the handshake; transport keepalive (Ping/Pong) is
/// skipped; anything else is a protocol violation.
async fn read_handshake_reply<S>(ws_rx: &mut S) -> Result<HandshakeReply, TunnelError>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while let Some(message) = ws_rx.next().await {
        match message.map_err(TunnelError::Transport)? {
            Message::Binary(bytes) => match Frame::decode(&bytes).map_err(TunnelError::Decode)? {
                Frame::HelloAck(_) => return Ok(HandshakeReply::Ack),
                Frame::HelloErr(err) => return Ok(HandshakeReply::Err(err.reason)),
                _ => {
                    return Err(TunnelError::Handshake(
                        "first reply was not HelloAck/HelloErr",
                    ))
                }
            },
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Close(_) => {
                return Err(TunnelError::Handshake("relay closed before HelloAck"))
            }
            Message::Text(_) => return Err(TunnelError::Handshake("relay sent a text message")),
            Message::Frame(_) => continue,
        }
    }
    Err(TunnelError::Handshake(
        "relay closed before a handshake reply",
    ))
}

/// The inbound read loop. Dispatches `Request` frames to bounded per-request
/// tasks, answers `Ping` with `Pong`, and returns on close. Returns `Ok(())` on
/// a clean close, `Err` on a transport/decode error.
async fn read_loop<S>(
    ws_rx: &mut S,
    outbound_tx: &mpsc::Sender<Frame>,
    inflight: &Arc<Semaphore>,
    client: &Arc<reqwest::Client>,
    loopback: &Arc<LoopbackTarget>,
    tasks: &mut tokio::task::JoinSet<()>,
) -> Result<(), TunnelError>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while let Some(message) = ws_rx.next().await {
        let message = message.map_err(TunnelError::Transport)?;
        match message {
            Message::Binary(bytes) => {
                if bytes.len() > MAX_TUNNEL_MESSAGE_BYTES {
                    tracing::warn!(
                        target: "tunnel-client",
                        len = bytes.len(),
                        "tunnel: oversized inbound frame; closing"
                    );
                    return Err(TunnelError::Protocol("inbound frame exceeds size cap"));
                }
                let frame = Frame::decode(&bytes).map_err(TunnelError::Decode)?;
                match frame {
                    Frame::Request(request) => {
                        spawn_request(
                            tasks,
                            request,
                            outbound_tx.clone(),
                            Arc::clone(inflight),
                            Arc::clone(client),
                            Arc::clone(loopback),
                        );
                    }
                    Frame::Ping(nonce) => {
                        // Best-effort; a dropped Pong under congestion is
                        // harmless (transport ping/pong also covers liveness).
                        let _ = outbound_tx.try_send(Frame::Pong(nonce));
                    }
                    Frame::Pong(_) => {}
                    // Device→relay-only or already-handled control frames coming
                    // FROM the relay are a protocol violation; close.
                    Frame::Hello(_)
                    | Frame::HelloAck(_)
                    | Frame::HelloErr(_)
                    | Frame::ResponseStart(_)
                    | Frame::ResponseChunk(_)
                    | Frame::ResponseEnd(_)
                    | Frame::ResponseError(_) => {
                        tracing::warn!(target: "tunnel-client", "tunnel: relay sent a device→relay frame; closing");
                        return Err(TunnelError::Protocol("relay sent a device→relay frame"));
                    }
                }
            }
            Message::Close(_) => return Ok(()),
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            Message::Text(_) => {
                tracing::warn!(target: "tunnel-client", "tunnel: relay sent a text message; closing");
                return Err(TunnelError::Protocol("relay sent a text message"));
            }
        }
    }
    Ok(())
}

/// Spawn a bounded task to replay one request against the loopback server and
/// stream the response back. The task acquires an inflight permit FIRST (so the
/// number of concurrent loopback requests is capped) and is keyed by
/// `request_id`, which it echoes on every response frame.
fn spawn_request(
    tasks: &mut tokio::task::JoinSet<()>,
    request: crate::frame::RequestFrame,
    outbound_tx: mpsc::Sender<Frame>,
    inflight: Arc<Semaphore>,
    client: Arc<reqwest::Client>,
    loopback: Arc<LoopbackTarget>,
) {
    tasks.spawn(async move {
        // Bound concurrency: hold a permit for the request's lifetime.
        let _permit = match inflight.acquire_owned().await {
            Ok(p) => p,
            Err(_) => return, // semaphore closed (shutdown)
        };
        crate::loopback::loopback_replay(request, &outbound_tx, &client, &loopback).await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopback::{InternalBearer, LoopbackTarget};

    fn config_with(relay_url: &str) -> TunnelConfig {
        TunnelConfig {
            relay_url: relay_url.to_string(),
            device_credential: "cred".to_string(),
            account_id: "acct".to_string(),
            loopback: LoopbackTarget::new("http://127.0.0.1:8765", InternalBearer::new("tok")),
        }
    }

    #[test]
    fn wss_required_for_remote_ws_only_for_loopback() {
        // wss:// is accepted for any host.
        assert!(relay_scheme_is_acceptable("wss://mcp.minutist.ai/tunnel"));
        assert!(relay_scheme_is_acceptable("WSS://mcp.minutist.ai/tunnel"));
        // ws:// is accepted only for a loopback host.
        assert!(relay_scheme_is_acceptable("ws://127.0.0.1:8080/tunnel"));
        assert!(relay_scheme_is_acceptable("ws://localhost:8080/tunnel"));
        assert!(relay_scheme_is_acceptable("ws://[::1]:8080/tunnel"));
        // ws:// to an off-machine host is refused.
        assert!(!relay_scheme_is_acceptable("ws://mcp.minutist.ai/tunnel"));
        // Other schemes and malformed URLs are refused.
        assert!(!relay_scheme_is_acceptable(
            "https://mcp.minutist.ai/tunnel"
        ));
        assert!(!relay_scheme_is_acceptable("mcp.minutist.ai/tunnel"));
    }

    #[tokio::test]
    async fn run_tunnel_refuses_cleartext_remote_relay() {
        // A cleartext ws:// relay to an off-machine host is refused before any
        // dial is attempted.
        let err = run_tunnel(config_with("ws://mcp.minutist.ai/tunnel"))
            .await
            .expect_err("cleartext remote relay must be refused");
        assert!(matches!(err, TunnelError::Config));
    }
}
