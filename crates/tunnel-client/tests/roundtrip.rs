//! End-to-end round-trip test for the app-side tunnel client.
//!
//! Stands up two in-process fakes:
//!   - a **fake relay**: a tokio-tungstenite WebSocket server that performs the
//!     Hello/HelloAck handshake, then sends a `Request` frame and collects the
//!     `ResponseStart`/`ResponseChunk`/`ResponseEnd` (or `ResponseError`) the
//!     client streams back;
//!   - a **fake loopback `mcp-server`**: a raw HTTP/1.1 server (tokio
//!     `TcpListener`) standing in for the app's loopback MCP endpoint. It records
//!     the request line, the `Authorization` header, and the body, and replies
//!     with a canned response.
//!
//! The tests assert: a relayed request reaches the loopback WITH the internal
//! bearer attached and the response streams back; a `HelloErr` fails cleanly; a
//! loopback connection failure yields a `ResponseError` frame; and the internal
//! bearer never appears in any frame sent to the relay.

use std::collections::BTreeMap;

use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message;

use tunnel_client::{
    run_tunnel, Frame, HelloErr, HelloErrReason, InternalBearer, LoopbackTarget, RequestFrame,
    TunnelConfig,
};

const INTERNAL_BEARER: &str = "internal-loopback-secret-token";

/// What the fake loopback server recorded about the single request it received.
#[derive(Debug, Default, Clone)]
struct LoopbackCapture {
    request_line: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

/// Start a fake loopback HTTP/1.1 server. Returns its `http://127.0.0.1:PORT`
/// origin and a channel that yields what it captured for the first request.
/// `response` is the raw HTTP/1.1 response bytes it replies with; pass `None` to
/// have it accept the TCP connection then immediately drop it (simulating a
/// loopback that refuses to respond).
async fn start_fake_loopback(
    response: Option<&'static str>,
) -> (String, oneshot::Receiver<LoopbackCapture>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 8192];
        let mut data = Vec::new();
        // Read until we have headers + the full body (Content-Length aware, but
        // the test bodies are small so a couple of reads suffice).
        loop {
            let n = socket.read(&mut buf).await.unwrap();
            if n == 0 {
                break;
            }
            data.extend_from_slice(&buf[..n]);
            if let Some(pos) = find_headers_end(&data) {
                let header_str = String::from_utf8_lossy(&data[..pos]).to_string();
                let content_len = content_length(&header_str);
                let body_start = pos + 4;
                if data.len() >= body_start + content_len {
                    break;
                }
            }
        }

        let text = String::from_utf8_lossy(&data).to_string();
        let request_line = text.lines().next().unwrap_or_default().to_string();
        let authorization = text
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
            .map(|l| {
                l.split_once(':')
                    .map(|x| x.1)
                    .unwrap_or("")
                    .trim()
                    .to_string()
            });
        let body = data
            .iter()
            .copied()
            .skip(find_headers_end(&data).map(|p| p + 4).unwrap_or(data.len()))
            .collect::<Vec<u8>>();

        let _ = tx.send(LoopbackCapture {
            request_line,
            authorization,
            body,
        });

        match response {
            Some(resp) => {
                socket.write_all(resp.as_bytes()).await.unwrap();
                socket.flush().await.unwrap();
            }
            None => {
                // Drop the socket without replying.
            }
        }
    });

    (format!("http://{addr}"), rx)
}

fn find_headers_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|w| w == b"\r\n\r\n")
}

fn content_length(headers: &str) -> usize {
    headers
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':').map(|x| x.1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

/// What the fake relay collected from the client across a session.
#[derive(Debug, Default)]
struct RelayCapture {
    /// Every frame the client sent AFTER the Hello (the response frames).
    response_frames: Vec<Frame>,
    /// The Hello the client sent.
    hello: Option<Frame>,
    /// The handshake outcome the client observed (whether it kept the socket
    /// open past HelloAck).
    handshake_acked: bool,
}

/// Start a fake relay. On a client connection it reads the Hello, replies with
/// `ack_frame` (HelloAck or HelloErr), and — if it acked — sends `request` then
/// collects response frames until the socket closes. Returns the relay URL and a
/// channel yielding the capture.
async fn start_fake_relay(
    ack_frame: Frame,
    request: Option<RequestFrame>,
) -> (String, oneshot::Receiver<RelayCapture>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        let (mut ws_tx, mut ws_rx) = ws.split();

        let mut capture = RelayCapture::default();

        // First inbound frame must be the Hello.
        if let Some(Ok(Message::Binary(bytes))) = ws_rx.next().await {
            capture.hello = Frame::decode(&bytes).ok();
        }

        let acked = matches!(ack_frame, Frame::HelloAck(_));
        capture.handshake_acked = acked;
        ws_tx
            .send(Message::Binary(ack_frame.encode().unwrap().into()))
            .await
            .unwrap();

        if acked {
            if let Some(req) = request {
                ws_tx
                    .send(Message::Binary(
                        Frame::Request(req).encode().unwrap().into(),
                    ))
                    .await
                    .unwrap();
            }
            // Collect response frames until the client closes the socket.
            while let Some(msg) = ws_rx.next().await {
                match msg {
                    Ok(Message::Binary(bytes)) => {
                        if let Ok(frame) = Frame::decode(&bytes) {
                            let is_end =
                                matches!(frame, Frame::ResponseEnd(_) | Frame::ResponseError(_));
                            capture.response_frames.push(frame);
                            if is_end {
                                break;
                            }
                        }
                    }
                    Ok(Message::Close(_)) | Err(_) => break,
                    _ => {}
                }
            }
            // Clean WS close so the client's reader returns Ok rather than
            // observing a transport reset.
            let _ = ws_tx.send(Message::Close(None)).await;
        }

        let _ = tx.send(capture);
    });

    (format!("ws://{addr}"), rx)
}

fn sample_headers() -> BTreeMap<String, String> {
    let mut h = BTreeMap::new();
    h.insert("content-type".to_string(), "application/json".to_string());
    h.insert("accept".to_string(), "application/json".to_string());
    h
}

#[tokio::test]
async fn relayed_request_reaches_loopback_with_bearer_and_response_streams_back() {
    let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 17\r\n\r\n{\"result\":\"ok\"}\n\n";
    let (loopback_origin, loopback_rx) = start_fake_loopback(Some(response)).await;

    let request = RequestFrame {
        request_id: 99,
        method: "POST".into(),
        path: "/mcp".into(),
        headers: sample_headers(),
        body: br#"{"jsonrpc":"2.0","method":"tools/list","id":1}"#.to_vec(),
    };
    let (relay_url, relay_rx) = start_fake_relay(
        Frame::HelloAck(tunnel_client::HelloAck {
            account_id: "sub-1".into(),
        }),
        Some(request),
    )
    .await;

    let config = TunnelConfig {
        relay_url,
        device_credential: "dev-cred".into(),
        account_id: "sub-1".into(),
        loopback: LoopbackTarget::new(loopback_origin, InternalBearer::new(INTERNAL_BEARER)),
    };

    run_tunnel(config).await.expect("tunnel run");

    let loopback = loopback_rx.await.expect("loopback capture");
    let relay = relay_rx.await.expect("relay capture");

    // The request reached the loopback server with the right line + bearer.
    assert!(
        loopback.request_line.starts_with("POST /mcp HTTP/1.1"),
        "unexpected request line: {}",
        loopback.request_line
    );
    assert_eq!(
        loopback.authorization.as_deref(),
        Some(&*format!("Bearer {INTERNAL_BEARER}")),
        "internal bearer not attached to loopback request"
    );
    assert_eq!(
        loopback.body,
        br#"{"jsonrpc":"2.0","method":"tools/list","id":1}"#.to_vec(),
        "request body not forwarded verbatim"
    );

    // The response streamed back as Start -> Chunk(s) -> End, all keyed to id 99.
    let frames = &relay.response_frames;
    assert!(
        matches!(frames.first(), Some(Frame::ResponseStart(s)) if s.request_id == 99 && s.status == 200)
    );
    assert!(matches!(frames.last(), Some(Frame::ResponseEnd(e)) if e.request_id == 99));
    let body: Vec<u8> = frames
        .iter()
        .filter_map(|f| match f {
            Frame::ResponseChunk(c) => Some(c.data.clone()),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(body, b"{\"result\":\"ok\"}\n\n", "streamed body mismatch");

    // SECURITY: the internal bearer must never appear in any frame sent back to
    // the relay.
    assert_no_bearer_in_frames(frames);
    // ... nor in the Hello.
    if let Some(hello) = &relay.hello {
        assert_no_bearer_in_frames(std::slice::from_ref(hello));
    }
}

#[tokio::test]
async fn hello_err_fails_cleanly() {
    let (relay_url, relay_rx) = start_fake_relay(
        Frame::HelloErr(HelloErr {
            reason: HelloErrReason::AuthFailed,
        }),
        None,
    )
    .await;

    let config = TunnelConfig {
        relay_url,
        device_credential: "bad-cred".into(),
        account_id: "sub-1".into(),
        loopback: LoopbackTarget::new("http://127.0.0.1:1", InternalBearer::new(INTERNAL_BEARER)),
    };

    let err = run_tunnel(config).await.expect_err("should reject");
    assert!(
        matches!(
            err,
            tunnel_client::TunnelError::HelloRejected(HelloErrReason::AuthFailed)
        ),
        "unexpected error: {err:?}"
    );
    let _ = relay_rx.await; // relay finished
}

#[tokio::test]
async fn loopback_failure_yields_response_error() {
    // Bind a listener, get its port, then drop it so the connect is refused.
    let dead = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_addr = dead.local_addr().unwrap();
    drop(dead);
    let loopback_origin = format!("http://{dead_addr}");

    let request = RequestFrame {
        request_id: 5,
        method: "POST".into(),
        path: "/mcp".into(),
        headers: BTreeMap::new(),
        body: b"{}".to_vec(),
    };
    let (relay_url, relay_rx) = start_fake_relay(
        Frame::HelloAck(tunnel_client::HelloAck {
            account_id: "sub-1".into(),
        }),
        Some(request),
    )
    .await;

    let config = TunnelConfig {
        relay_url,
        device_credential: "dev-cred".into(),
        account_id: "sub-1".into(),
        loopback: LoopbackTarget::new(loopback_origin, InternalBearer::new(INTERNAL_BEARER)),
    };

    run_tunnel(config).await.expect("tunnel run");

    let relay = relay_rx.await.expect("relay capture");
    let frames = &relay.response_frames;
    assert_eq!(
        frames.len(),
        1,
        "expected a single ResponseError, got {frames:?}"
    );
    assert!(
        matches!(&frames[0], Frame::ResponseError(e) if e.request_id == 5),
        "expected ResponseError for id 5, got {:?}",
        frames[0]
    );
    // The error message must carry no secret.
    assert_no_bearer_in_frames(frames);
}

/// Assert the internal bearer (and its `Bearer …` header form) does not appear
/// in the postcard encoding of any of the given frames.
fn assert_no_bearer_in_frames(frames: &[Frame]) {
    let secret = INTERNAL_BEARER.as_bytes();
    let bearer_form = format!("Bearer {INTERNAL_BEARER}");
    let bearer = bearer_form.as_bytes();
    for frame in frames {
        let bytes = frame.encode().unwrap();
        assert!(
            !contains_subslice(&bytes, secret),
            "internal bearer found in a frame sent to the relay: {frame:?}"
        );
        assert!(
            !contains_subslice(&bytes, bearer),
            "Bearer-form internal bearer found in a frame sent to the relay: {frame:?}"
        );
    }
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}
