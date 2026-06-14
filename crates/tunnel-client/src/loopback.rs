//! The loopback target: where relayed requests are replayed, and the internal
//! bearer applied to them.

use std::collections::BTreeMap;
use std::fmt;

use futures_util::StreamExt;
use tokio::sync::mpsc;

use crate::frame::{Frame, RequestFrame, ResponseChunk, ResponseEnd, ResponseError, ResponseStart};

/// The app's internal loopback bearer token. Newtyped so its `Debug` redacts the
/// value — it must never appear in logs and is attached only to the outbound
/// loopback HTTP request, never serialised into a tunnel frame.
#[derive(Clone)]
pub struct InternalBearer(String);

impl InternalBearer {
    /// Wrap the raw token. Take the value from `ipc-bridge::McpServerInfo.token`.
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }

    /// The `Authorization` header value (`Bearer <token>`) to attach to the
    /// loopback request. Crate-internal — there is no public accessor for the
    /// raw secret.
    pub(crate) fn header_value(&self) -> String {
        format!("Bearer {}", self.0)
    }
}

impl fmt::Debug for InternalBearer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("InternalBearer(<redacted>)")
    }
}

/// Response headers forwarded from the loopback `mcp-server` back to the
/// untrusted relay. Everything else is dropped before building `ResponseStart`,
/// mirroring the relay's inbound request-header filtering: a future
/// `set-cookie` / `authorization` echo must not transit the trust boundary.
/// Names are lower-cased here and compared against the lower-cased response
/// header names. The set covers what the MCP-over-HTTP / SSE protocol needs:
/// the content descriptors, cache-control, and the SSE/MCP session headers.
const FORWARDED_RESPONSE_HEADERS: &[&str] = &[
    "content-type",
    "content-length",
    "cache-control",
    "mcp-session-id",
    "last-event-id",
];

/// Where to replay relayed MCP requests: the app's loopback `mcp-server` base
/// URL plus the internal bearer it expects.
///
/// `base_url` is the scheme+authority of the loopback server, e.g.
/// `http://127.0.0.1:8765` (NO trailing slash, NO path — the relay frame carries
/// the path, e.g. `/mcp`). `app-main` builds this from `McpServerInfo`: the
/// `McpServerInfo.url` is the full `…/mcp` endpoint, so the caller strips the
/// path to the origin and keeps the token as the bearer.
#[derive(Clone, Debug)]
pub struct LoopbackTarget {
    pub base_url: String,
    pub internal_bearer: InternalBearer,
}

impl LoopbackTarget {
    /// Build a target from the loopback origin and the internal bearer.
    pub fn new(base_url: impl Into<String>, internal_bearer: InternalBearer) -> Self {
        Self {
            base_url: base_url.into(),
            internal_bearer,
        }
    }

    /// Join the origin with a request path. The relay path is already absolute
    /// (`/mcp`); a missing leading slash is tolerated.
    pub(crate) fn url_for(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        if path.starts_with('/') {
            format!("{base}{path}")
        } else {
            format!("{base}/{path}")
        }
    }
}

/// Replay one relayed request against the loopback `mcp-server` and stream the
/// response back over `outbound_tx` as `ResponseStart` → `ResponseChunk*` →
/// `ResponseEnd`, or a single `ResponseError` on a local failure.
///
/// The internal bearer is attached HERE — to the loopback HTTP request only — and
/// never travels back to the relay. The response body is consumed as a byte
/// stream (`bytes_stream`) so a long/SSE body is forwarded chunk-by-chunk rather
/// than buffered whole. Only `request_id`, method, path, and status are traced;
/// bodies are never logged.
pub(crate) async fn loopback_replay(
    request: RequestFrame,
    outbound_tx: &mpsc::Sender<Frame>,
    client: &reqwest::Client,
    target: &LoopbackTarget,
) {
    let request_id = request.request_id;
    let url = target.url_for(&request.path);
    tracing::debug!(request_id, method = %request.method, path = %request.path, "tunnel: replaying request to loopback");

    let method = match reqwest::Method::from_bytes(request.method.as_bytes()) {
        Ok(m) => m,
        Err(_) => {
            send_error(outbound_tx, request_id, "invalid request method").await;
            return;
        }
    };

    let mut builder = client.request(method, &url);
    // Forward the relay's filtered headers verbatim (content-type / accept).
    for (name, value) in &request.headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    // Apply the internal bearer LAST so it cannot be shadowed by a forwarded
    // header. The relay strips the inbound authorization, but assert our own.
    builder = builder.header(
        reqwest::header::AUTHORIZATION,
        target.internal_bearer.header_value(),
    );
    builder = builder.body(request.body);

    let response = match builder.send().await {
        Ok(r) => r,
        Err(error) => {
            // `error` is logged WITHOUT the URL bearer (reqwest errors do not
            // carry header values); keep the message coarse for the relay.
            tracing::warn!(request_id, %error, "tunnel: loopback request failed");
            send_error(outbound_tx, request_id, "loopback request failed").await;
            return;
        }
    };

    let status = response.status().as_u16();
    let headers = filter_response_headers(response.headers());
    tracing::debug!(request_id, status, "tunnel: loopback responded");

    if outbound_tx
        .send(Frame::ResponseStart(ResponseStart {
            request_id,
            status,
            headers,
        }))
        .await
        .is_err()
    {
        return; // writer gone; nothing more to do
    }

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if bytes.is_empty() {
                    continue;
                }
                if outbound_tx
                    .send(Frame::ResponseChunk(ResponseChunk {
                        request_id,
                        data: bytes.to_vec(),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Err(error) => {
                // Mid-stream failure: the relay already saw ResponseStart, so a
                // ResponseError now terminates the request cleanly.
                tracing::warn!(request_id, %error, "tunnel: loopback body stream error");
                send_error(outbound_tx, request_id, "loopback body stream error").await;
                return;
            }
        }
    }

    let _ = outbound_tx
        .send(Frame::ResponseEnd(ResponseEnd { request_id }))
        .await;
}

/// Project the loopback response headers down to the [`FORWARDED_RESPONSE_HEADERS`]
/// allowlist (names lower-cased), dropping everything else before it crosses to
/// the relay. A header whose value is not valid UTF-8 is skipped.
fn filter_response_headers(
    response_headers: &reqwest::header::HeaderMap,
) -> BTreeMap<String, String> {
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in response_headers {
        let name = name.as_str().to_ascii_lowercase();
        if !FORWARDED_RESPONSE_HEADERS.contains(&name.as_str()) {
            continue;
        }
        if let Ok(v) = value.to_str() {
            headers.insert(name, v.to_string());
        }
    }
    headers
}

/// Send a terminal `ResponseError` for a request. `message` is a fixed,
/// non-sensitive string — it never carries user/MCP content.
async fn send_error(outbound_tx: &mpsc::Sender<Frame>, request_id: u64, message: &str) {
    let _ = outbound_tx
        .send(Frame::ResponseError(ResponseError {
            request_id,
            message: message.to_string(),
        }))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_bearer() {
        let bearer = InternalBearer::new("super-secret-token");
        let shown = format!("{bearer:?}");
        assert!(
            !shown.contains("super-secret-token"),
            "bearer leaked in Debug"
        );
        assert!(shown.contains("redacted"));
    }

    #[test]
    fn debug_of_target_redacts_bearer() {
        let target = LoopbackTarget::new("http://127.0.0.1:8765", InternalBearer::new("tok"));
        let shown = format!("{target:?}");
        assert!(
            !shown.contains("tok"),
            "bearer leaked via LoopbackTarget Debug"
        );
    }

    #[test]
    fn url_join_handles_slashes() {
        let t = LoopbackTarget::new("http://127.0.0.1:8765/", InternalBearer::new("x"));
        assert_eq!(t.url_for("/mcp"), "http://127.0.0.1:8765/mcp");
        let t = LoopbackTarget::new("http://127.0.0.1:8765", InternalBearer::new("x"));
        assert_eq!(t.url_for("mcp"), "http://127.0.0.1:8765/mcp");
    }

    #[test]
    fn header_value_is_bearer_scheme() {
        assert_eq!(InternalBearer::new("abc").header_value(), "Bearer abc");
    }

    #[test]
    fn response_header_filter_drops_non_allowlisted() {
        use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

        let mut raw = HeaderMap::new();
        raw.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        raw.insert("mcp-session-id", HeaderValue::from_static("sess-1"));
        // A sensitive header the loopback server must never echo to the relay.
        raw.insert("set-cookie", HeaderValue::from_static("sid=secret"));
        raw.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer leaked"),
        );

        let filtered = filter_response_headers(&raw);

        assert_eq!(
            filtered.get("content-type").map(String::as_str),
            Some("text/event-stream")
        );
        assert_eq!(
            filtered.get("mcp-session-id").map(String::as_str),
            Some("sess-1")
        );
        assert!(
            !filtered.contains_key("set-cookie"),
            "set-cookie must not be forwarded to the relay"
        );
        assert!(
            !filtered.contains_key("authorization"),
            "authorization must not be forwarded to the relay"
        );
    }
}
