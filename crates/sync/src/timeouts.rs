//! Timeout budgets for the inbound accept/responder path.
//!
//! A paired peer is trusted for content, not liveness: nothing stops a stalled or
//! hostile one from opening a connection and never reading or writing again. Each
//! constant bounds one `tokio::time::timeout`. Blowing a budget logs at `warn` and
//! returns [`crate::Error::Protocol`], so the connection drops exactly as it would
//! for a malformed frame.
//!
//! [`FRAME_IO_TIMEOUT`] applies inside [`crate::frame::read_frame`] and
//! [`crate::frame::write_frame`], so it covers both sides of every protocol.

use std::time::Duration;

/// Accepting the inbound bi stream and reading its one-byte
/// [`crate::notes_proto::StreamKind`] tag. Short: this is the first exchange on a
/// fresh connection, so a peer slow here is stalled, not on a slow path.
pub(crate) const ACCEPT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(20);

/// One length-prefixed frame read or write, on either side of any protocol.
pub(crate) const FRAME_IO_TIMEOUT: Duration = Duration::from_secs(30);

/// A responder's final `conn.closed().await`. Spans the initiator's own last frame
/// read plus the blob pulls its `DONE` handshake waits on, not one frame, so it
/// sits well above [`FRAME_IO_TIMEOUT`].
pub(crate) const RESPONDER_CLOSE_TIMEOUT: Duration = Duration::from_secs(120);

/// Pulling one blob over the blobs ALPN. Meeting audio reaches tens of megabytes
/// even under [`crate::blobs::MAX_BLOB_BYTES`], hence the size.
pub(crate) const BLOB_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// The final DONE read in the media/artifacts completion handshake. Spans however
/// long the peer takes to pull every blob it lacks, and a manifest may list
/// several, so it sits well above [`BLOB_DOWNLOAD_TIMEOUT`].
pub(crate) const PEER_PULL_TIMEOUT: Duration = Duration::from_secs(1800);
