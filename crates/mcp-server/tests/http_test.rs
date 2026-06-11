//! Focused HTTP integration test for the MCP server transport (Phase 10 §12.2).
//!
//! Drives a live `serve(..)` instance over loopback with a raw TCP client (no
//! heavy HTTP-client dependency) to verify the transport-level controls that the
//! pure unit tests cannot exercise end-to-end:
//!
//! - the listener binds loopback,
//! - a request with NO `Authorization` header is a 401 (the bearer wrapper runs
//!   before rmcp),
//! - a request with a wrong bearer token is a 401,
//! - a request with a bad `Origin` is a 403 (rmcp's Origin validation, which we
//!   configure to the loopback origins).
//!
//! The full MCP protocol round-trip (`tools/list` / `tools/call`) against a real
//! TS SDK client is the env-gated/manual gate;
//! the tool-projection + dispatch logic is unit-tested in `agent-tools` and the
//! handler module.

use std::sync::Arc;

use agent_tools::{ToolContext, ToolRegistry};
use mcp_server::{serve, McpServerConfig};
use minutist_common::{AppEvent, AppResult, NoteBlock, Segment, Summariser};
use orchestrator::test_support::test_orchestrator;
use persistence::MeetingIndex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{broadcast, watch};

/// A no-op `Summariser` stub — the HTTP-layer test never dispatches a tool, so
/// the summariser is unused, but `ToolContext::new` requires one.
struct StubSummariser;

impl Summariser for StubSummariser {
    fn summarise(
        &self,
        _transcript: &[Segment],
        _notes: &[NoteBlock],
        _system_prompt: &str,
    ) -> AppResult<String> {
        Ok(String::new())
    }
}

const TOKEN: &str = "test-bearer-token-deadbeef";

/// Build a real `ToolContext` (test-source orchestrator + in-memory index +
/// stub summariser) and start the MCP server on `127.0.0.1:0`. Returns the
/// bound port, shutdown sender, and completion receiver.
async fn start_server_with_done(
    allow_writes: bool,
) -> (u16, watch::Sender<bool>, tokio::sync::oneshot::Receiver<()>) {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let meetings_dir = tempdir.path().join("meetings");
    std::fs::create_dir_all(&meetings_dir).expect("meetings dir");

    let index = Arc::new(MeetingIndex::open(":memory:").await.expect("index"));
    let orchestrator = Arc::new(test_orchestrator(meetings_dir.clone()));
    let summariser: Arc<dyn Summariser> = Arc::new(StubSummariser);
    let (event_tx, _rx) = broadcast::channel::<AppEvent>(16);

    // A bridge channel so the v1(true) registry's inter-agent tool has a sender;
    // the receiver is dropped (no driver) since this test does not call it.
    let (bridge_tx, _bridge_rx) = tokio::sync::mpsc::channel(1);
    let ctx = Arc::new(
        ToolContext::new(
            orchestrator,
            index,
            meetings_dir,
            summariser,
            event_tx,
            None,
        )
        .with_inter_agent_bridge(bridge_tx),
    );
    let registry = Arc::new(ToolRegistry::v1(true));

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (bound, done_rx) = serve(
        registry,
        ctx,
        McpServerConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            bearer_token: TOKEN.to_string(),
            allow_writes,
        },
        shutdown_rx,
    )
    .await
    .expect("serve must bind");

    // Keep the tempdir alive for the server's lifetime by leaking it: the test
    // process is short-lived and the OS reclaims it. (Dropping it here would
    // remove the meetings dir out from under the running server.)
    std::mem::forget(tempdir);

    assert_eq!(bound.ip().to_string(), "127.0.0.1", "must bind loopback");
    (bound.port(), shutdown_tx, done_rx)
}

/// Convenience wrapper that discards the completion receiver (existing tests
/// that do not need to await completion).
async fn start_server(allow_writes: bool) -> (u16, watch::Sender<bool>) {
    let (port, shutdown, _done) = start_server_with_done(allow_writes).await;
    (port, shutdown)
}

/// Send a raw HTTP/1.1 request to `127.0.0.1:{port}` and return the status line.
async fn raw_request(port: u16, request: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect");
    stream.write_all(request.as_bytes()).await.expect("write");
    stream.flush().await.expect("flush");

    let mut buf = Vec::new();
    // Read until the server closes or we have the headers; a small read is
    // enough for the status line + headers.
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let text = String::from_utf8_lossy(&buf);
    text.lines().next().unwrap_or("").to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_bearer_is_401() {
    let (port, _shutdown) = start_server(false).await;
    // A POST to /mcp with NO Authorization header.
    let req = format!(
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
    );
    let status = raw_request(port, &req).await;
    assert!(
        status.contains("401"),
        "missing bearer must be 401, got: {status:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_bearer_is_401() {
    let (port, _shutdown) = start_server(false).await;
    let req = format!(
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer wrong-token\r\n\
         Content-Type: application/json\r\nAccept: application/json, text/event-stream\r\n\
         Content-Length: 2\r\nConnection: close\r\n\r\n{{}}"
    );
    let status = raw_request(port, &req).await;
    assert!(
        status.contains("401"),
        "wrong bearer must be 401, got: {status:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_origin_is_403_even_with_valid_bearer() {
    let (port, _shutdown) = start_server(false).await;
    // Valid bearer, but a cross-origin browser Origin → rmcp Origin validation
    // rejects with 403 (the bearer check passes first, then rmcp enforces).
    let req = format!(
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {TOKEN}\r\n\
         Origin: https://evil.example.com\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
    );
    let status = raw_request(port, &req).await;
    assert!(
        status.contains("403"),
        "a cross-origin request must be 403, got: {status:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graceful_shutdown_stops_accepting() {
    let (port, shutdown) = start_server(false).await;
    // The server is up: an unauthenticated request gets a 401 (not a connection
    // refusal).
    let req = format!(
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
    );
    let status = raw_request(port, &req).await;
    assert!(
        status.contains("401"),
        "server up before shutdown: {status:?}"
    );

    // Flip the shutdown flag and give the accept loop a tick to break.
    shutdown.send(true).expect("send shutdown");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // New connections are refused (the listener was dropped). A connect may
    // succeed-then-reset or refuse outright depending on timing; either way the
    // status line is NOT a 401 from a live handler.
    let connect = tokio::net::TcpStream::connect(("127.0.0.1", port)).await;
    if let Ok(mut stream) = connect {
        let _ = stream.write_all(req.as_bytes()).await;
        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).await.unwrap_or(0);
        assert_eq!(
            n, 0,
            "a post-shutdown connection must not get a live response"
        );
    }
}

/// Verify that the completion receiver resolves after shutdown, and that the
/// port can be re-bound immediately after awaiting it (deterministic
/// stop→start: no port-in-use race).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_completion_and_port_rebind() {
    let (port, shutdown, done_rx) = start_server_with_done(false).await;

    // The server is alive.
    let req = format!(
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
    );
    let status = raw_request(port, &req).await;
    assert!(status.contains("401"), "server must be alive: {status:?}");

    // Shut it down and await the completion receiver (bounded: 2 s).
    shutdown.send(true).expect("send shutdown");
    tokio::time::timeout(std::time::Duration::from_secs(2), done_rx)
        .await
        .expect("completion must resolve within 2 s")
        .expect("done_tx must not be dropped without sending");

    // Re-bind the SAME explicit port — must succeed immediately because the
    // completion receiver confirms the listener was dropped.
    let bind_addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let rebound = tokio::net::TcpListener::bind(bind_addr).await;
    assert!(rebound.is_ok(), "port {port} must be rebindable after completion: {rebound:?}");
}
