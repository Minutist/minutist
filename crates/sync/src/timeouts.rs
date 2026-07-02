//! Timeout budgets bounding the inbound accept/responder path.
//!
//! A paired peer is trusted for content, not for liveness: nothing stops it (a
//! stalled device, a bad network path, or a compromised paired device) from
//! opening a connection and then never writing or reading again. Every await on
//! that path used to be unbounded, so a single stalled peer pinned a router task
//! (and, on the always-on hub, one of its limited worker threads) forever — a
//! slowloris against a target that is expected to stay up. Each constant here
//! wraps one `tokio::time::timeout` call on the responder side; a peer that blows
//! its budget is logged at `warn` and the exchange returns [`crate::Error::Protocol`],
//! which the caller already treats as an ordinary failure — the connection is
//! then dropped exactly as it would be for a malformed frame or an unpaired peer.
//!
//! [`FRAME_IO_TIMEOUT`] is applied inside [`crate::frame::read_frame`] /
//! [`crate::frame::write_frame`] themselves, so it bounds both the initiator and
//! the responder side of every protocol without a per-call-site change.

use std::time::Duration;

/// Bound on the very first step of an inbound connection: accepting the
/// bidirectional stream and reading its one-byte
/// [`crate::notes_proto::StreamKind`] tag. Short — this is the first exchange on
/// a fresh connection, so a peer slow here is stalled or hostile, not merely on a
/// slow network path.
pub(crate) const ACCEPT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(20);

/// Bound on a single length-prefixed frame read or write ([`crate::frame`]).
/// Applies to every notes/media/discovery/artifacts frame on both the initiator
/// and responder side, so a peer that stops reading or writing mid-frame cannot
/// pin the stream indefinitely.
pub(crate) const FRAME_IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Bound on a responder's final `conn.closed().await` — the wait for the
/// initiator to finish reading the responder's last write and close the
/// connection. Generous relative to [`FRAME_IO_TIMEOUT`] because, from the
/// initiator's side, it spans however long its own last frame read plus the
/// blob pulls the two `DONE` handshakes wait on take, not just one frame.
pub(crate) const RESPONDER_CLOSE_TIMEOUT: Duration = Duration::from_secs(120);

/// Bound on pulling one blob (media or derived artifact) from a peer over the
/// blobs ALPN. Generous — meeting audio can run tens of megabytes even under the
/// per-blob size cap ([`crate::blobs::MAX_BLOB_BYTES`]) — but finite, so a peer
/// that advertises a blob and then serves it too slowly (or not at all) cannot
/// pin the downloader forever.
pub(crate) const BLOB_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// Bound on the final one-byte DONE read in the media/artifacts completion
/// handshake ([`crate::media_proto::finish_done`] /
/// [`crate::artifacts_proto::finish_done`]), shared by both the initiator and
/// responder side of that call. The read spans however long the PEER takes to
/// finish pulling every blob IT lacks over the separate blobs ALPN before it
/// writes its DONE byte — a manifest can list more than one blob, so this is
/// set well above a single [`BLOB_DOWNLOAD_TIMEOUT`] rather than reusing it
/// directly, while still being finite so a peer that never sends DONE cannot
/// pin the handshake forever.
pub(crate) const PEER_PULL_TIMEOUT: Duration = Duration::from_secs(1800);
