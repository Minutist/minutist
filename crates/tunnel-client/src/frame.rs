//! The tunnel wire protocol — the app-side re-implementation of the relay's
//! frame contract.
//!
//! This module mirrors `minutist-relay/crates/mcp-relay/src/tunnel/frame.rs`
//! byte-for-byte. The two ends live in separate repositories and do **not**
//! share a crate (EXECUTION.md X9 — only this app-side client lives in the
//! public workspace), so the types are re-implemented here and pinned to the
//! relay's encoding by the cross-impl fixture test in
//! `tests/fixtures/relay_frames.txt`.
//!
//! # Encoding contract (must not drift)
//!
//! - Each frame is the [`Frame`] enum serialised with **postcard**
//!   (default-features=false + alloc) and sent as one **binary** WebSocket
//!   message. No length prefix — WebSocket framing delimits messages.
//! - postcard encodes an enum discriminant by its **variant index**, so the
//!   variant ORDER below is the wire contract. Reordering, inserting, or
//!   removing a variant is a breaking change and MUST bump [`PROTOCOL_VERSION`]
//!   in lockstep with the relay.
//! - Struct field order and types likewise match the relay; `BTreeMap` keeps the
//!   header encoding deterministic (sorted keys) so round-trips are stable.
//!
//! # Security (binding)
//!
//! The inbound user OAuth bearer is **never** present in any frame. The relay
//! forwards only the MCP request semantics (method/path/body + content
//! negotiation headers); the app applies its own internal loopback bearer when
//! it replays the request against `mcp-server`. That secret stays on the device
//! and never transits the tunnel.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The tunnel protocol version, announced in [`Hello`]. Must equal the relay's
/// `PROTOCOL_VERSION`; the relay rejects a `Hello` whose `version` differs.
pub const PROTOCOL_VERSION: u32 = 1;

/// Correlates a request with its response frames across the multiplexed tunnel.
/// Assigned by the relay (monotonic per connection); the device echoes it
/// unchanged on every response frame.
pub type RequestId = u64;

/// One frame on the tunnel. Serialised with postcard, one frame per binary
/// WebSocket message. **The variant order is the wire contract** — postcard
/// encodes the discriminant by index. Keep this identical to the relay's
/// `Frame` and do not reorder without a [`PROTOCOL_VERSION`] bump on both ends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Frame {
    /// Device → relay, first frame after the WS upgrade. Authenticates the
    /// device and claims an account.
    Hello(Hello),
    /// Relay → device, on a successful [`Frame::Hello`].
    HelloAck(HelloAck),
    /// Relay → device, on a rejected [`Frame::Hello`]. The relay closes the
    /// connection immediately after sending this.
    HelloErr(HelloErr),
    /// Either direction: liveness probe. The peer replies with [`Frame::Pong`]
    /// echoing the nonce.
    Ping(u64),
    /// Either direction: reply to [`Frame::Ping`], echoing its nonce.
    Pong(u64),
    /// Relay → device: one gated MCP HTTP request to replay against the app's
    /// loopback `mcp-server`. The whole request body is carried inline.
    Request(RequestFrame),
    /// Device → relay: the start of a response — status and headers.
    ResponseStart(ResponseStart),
    /// Device → relay: one body chunk for an in-progress response.
    ResponseChunk(ResponseChunk),
    /// Device → relay: the response body is complete.
    ResponseEnd(ResponseEnd),
    /// Device → relay: the device could not produce a response (loopback
    /// unreachable, replay failed). Terminates the request; the relay maps it to
    /// a 502. Distinct from a real HTTP error *status*, which is a normal
    /// [`Frame::ResponseStart`] carrying that status.
    ResponseError(ResponseError),
}

/// First frame from the device: its credential and the account it claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    /// The protocol version the device speaks. Must equal [`PROTOCOL_VERSION`].
    pub version: u32,
    /// The device credential the relay maps to an account.
    pub device_credential: String,
    /// The account the device claims to serve.
    pub account_id: String,
}

/// Relay's acknowledgement of a successful [`Hello`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelloAck {
    /// The account the connection is now registered under.
    pub account_id: String,
}

/// Relay's rejection of a [`Hello`]. The connection is closed after this frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelloErr {
    /// A machine-readable reason. Coarse by design — carries no secret material.
    pub reason: HelloErrReason,
}

/// Why a [`Hello`] was rejected. Coarse by design — never leaks which of the
/// credential/account was wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HelloErrReason {
    /// The device credential / account pair did not authenticate.
    AuthFailed,
    /// The device announced a protocol version the relay does not speak.
    UnsupportedVersion,
}

/// A gated MCP HTTP request to replay against the app's loopback `mcp-server`.
/// Carries request semantics only — never the inbound user bearer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestFrame {
    /// Correlates the response frames. Assigned by the relay.
    pub request_id: RequestId,
    /// The HTTP method (e.g. `POST`, `GET`).
    pub method: String,
    /// The path-and-query to replay against the loopback server, e.g. `/mcp`.
    pub path: String,
    /// Filtered request headers (lowercased names); the inbound `authorization`
    /// header is deliberately absent. `BTreeMap` for deterministic encoding.
    pub headers: BTreeMap<String, String>,
    /// The request body bytes. Empty for a body-less GET. Carried whole.
    pub body: Vec<u8>,
}

/// The start of a response: the HTTP status and headers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseStart {
    /// Correlates with the originating [`RequestFrame::request_id`].
    pub request_id: RequestId,
    /// The HTTP status code from the loopback server.
    pub status: u16,
    /// Response headers (lowercased names).
    pub headers: BTreeMap<String, String>,
}

/// One chunk of a streaming response body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseChunk {
    /// Correlates with the originating [`RequestFrame::request_id`].
    pub request_id: RequestId,
    /// A slice of the body bytes, in order. The concatenation of all chunks for
    /// a `request_id` is the full body.
    pub data: Vec<u8>,
}

/// Marks a response body complete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseEnd {
    /// Correlates with the originating [`RequestFrame::request_id`].
    pub request_id: RequestId,
}

/// The device failed to produce a response for a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseError {
    /// Correlates with the originating [`RequestFrame::request_id`].
    pub request_id: RequestId,
    /// A short, non-sensitive description for the relay log. Never carries
    /// user/MCP content.
    pub message: String,
}

/// A failure encoding or decoding a [`Frame`].
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// postcard could not serialise the frame.
    #[error("frame encode failed")]
    Encode(#[source] postcard::Error),
    /// postcard could not deserialise the bytes into a frame (truncated,
    /// corrupt, or an unknown variant from a mismatched protocol version).
    #[error("frame decode failed")]
    Decode(#[source] postcard::Error),
}

impl Frame {
    /// The [`RequestFrame::request_id`] this frame correlates with, if any.
    /// `Some` only for the device→relay response frames (`ResponseStart`,
    /// `ResponseChunk`, `ResponseEnd`, `ResponseError`); `Hello`/`Ping`/`Pong` and
    /// the relay→device frames carry no request id.
    pub(crate) fn request_id(&self) -> Option<RequestId> {
        match self {
            Frame::ResponseStart(r) => Some(r.request_id),
            Frame::ResponseChunk(r) => Some(r.request_id),
            Frame::ResponseEnd(r) => Some(r.request_id),
            Frame::ResponseError(r) => Some(r.request_id),
            _ => None,
        }
    }

    /// Serialise this frame to the bytes of one binary WebSocket message.
    pub fn encode(&self) -> Result<Vec<u8>, FrameError> {
        postcard::to_allocvec(self).map_err(FrameError::Encode)
    }

    /// Parse one binary WebSocket message's bytes into a frame.
    pub fn decode(bytes: &[u8]) -> Result<Self, FrameError> {
        postcard::from_bytes(bytes).map_err(FrameError::Decode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(frame: Frame) {
        let bytes = frame.encode().expect("encode");
        let decoded = Frame::decode(&bytes).expect("decode");
        assert_eq!(frame, decoded, "round-trip changed the frame");
    }

    fn sample_headers() -> BTreeMap<String, String> {
        let mut h = BTreeMap::new();
        h.insert("content-type".to_string(), "application/json".to_string());
        h.insert("accept".to_string(), "text/event-stream".to_string());
        h
    }

    /// The exact value set encoded in `tests/fixtures/relay_frames.txt` by the
    /// relay's own encoder. Kept in lockstep with that file — same names, same
    /// values.
    fn fixture_frames() -> Vec<(&'static str, Frame)> {
        vec![
            (
                "hello",
                Frame::Hello(Hello {
                    version: PROTOCOL_VERSION,
                    device_credential: "dev-secret".into(),
                    account_id: "sub-123".into(),
                }),
            ),
            (
                "hello_ack",
                Frame::HelloAck(HelloAck {
                    account_id: "sub-123".into(),
                }),
            ),
            (
                "hello_err_auth",
                Frame::HelloErr(HelloErr {
                    reason: HelloErrReason::AuthFailed,
                }),
            ),
            (
                "hello_err_ver",
                Frame::HelloErr(HelloErr {
                    reason: HelloErrReason::UnsupportedVersion,
                }),
            ),
            ("ping", Frame::Ping(42)),
            ("pong", Frame::Pong(42)),
            (
                "request",
                Frame::Request(RequestFrame {
                    request_id: 7,
                    method: "POST".into(),
                    path: "/mcp".into(),
                    headers: sample_headers(),
                    body: br#"{"jsonrpc":"2.0","method":"tools/list","id":1}"#.to_vec(),
                }),
            ),
            (
                "request_empty",
                Frame::Request(RequestFrame {
                    request_id: 8,
                    method: "GET".into(),
                    path: "/mcp".into(),
                    headers: BTreeMap::new(),
                    body: Vec::new(),
                }),
            ),
            (
                "response_start",
                Frame::ResponseStart(ResponseStart {
                    request_id: 7,
                    status: 200,
                    headers: sample_headers(),
                }),
            ),
            (
                "response_chunk",
                Frame::ResponseChunk(ResponseChunk {
                    request_id: 7,
                    data: b"event: message\ndata: {}\n\n".to_vec(),
                }),
            ),
            (
                "response_end",
                Frame::ResponseEnd(ResponseEnd { request_id: 7 }),
            ),
            (
                "response_error",
                Frame::ResponseError(ResponseError {
                    request_id: 7,
                    message: "loopback connection refused".into(),
                }),
            ),
        ]
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
            .collect()
    }

    #[test]
    fn round_trips_every_variant() {
        for (_, frame) in fixture_frames() {
            round_trip(frame);
        }
    }

    /// The byte-level cross-impl pin: this crate's encoding of each known frame
    /// must equal the relay-produced bytes in the fixture file, AND the fixture
    /// bytes must decode back into the expected value. If either side fails the
    /// app-side client would be wire-incompatible with the relay.
    #[test]
    fn matches_relay_produced_bytes() {
        let fixture = include_str!("../tests/fixtures/relay_frames.txt");
        let mut expected: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for line in fixture.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, hex) = line.split_once(' ').expect("name + hex");
            expected.insert(name, hex);
        }

        for (name, frame) in fixture_frames() {
            let relay_hex = expected
                .get(name)
                .unwrap_or_else(|| panic!("fixture missing entry for {name}"));
            // Our encoding equals the relay's bytes.
            let ours = frame.encode().expect("encode");
            assert_eq!(
                hex(&ours),
                *relay_hex,
                "encoding of `{name}` diverged from the relay fixture"
            );
            // The relay's bytes decode into the expected value.
            let decoded = Frame::decode(&unhex(relay_hex)).expect("decode relay bytes");
            assert_eq!(decoded, frame, "relay bytes for `{name}` decoded wrong");
        }

        assert_eq!(
            expected.len(),
            fixture_frames().len(),
            "fixture file and fixture_frames() drifted in entry count"
        );
    }

    #[test]
    fn decode_rejects_truncated_bytes() {
        let bytes = Frame::Ping(1).encode().unwrap();
        let truncated = &bytes[..bytes.len() - 1];
        assert!(matches!(
            Frame::decode(truncated),
            Err(FrameError::Decode(_))
        ));
    }

    #[test]
    fn decode_rejects_empty_bytes() {
        assert!(matches!(Frame::decode(&[]), Err(FrameError::Decode(_))));
    }
}
