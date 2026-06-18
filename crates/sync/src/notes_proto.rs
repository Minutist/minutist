//! The Yjs notes-update exchange protocol.
//!
//! A small custom iroh protocol (ALPN [`SYNC_ALPN`]) ships opaque Yjs CRDT update
//! frames between paired devices. The local authoritative document is `persistence`'s
//! `notes.ydoc`; the protocol:
//!
//! - reads the local state as a v1 update with
//!   [`persistence::NotesStore::read_ydoc_state`],
//! - exchanges update frames with the peer over a bi-directional QUIC stream,
//! - merges a received update with [`persistence::NotesStore::apply_update`]
//!   (which itself calls [`persistence::ydoc::apply_update_v1`]),
//!
//! so both devices converge without a central server. CRDT merge is commutative,
//! so exchange order does not matter.

/// ALPN for the notes-update protocol. Bumping the suffix is a wire break.
pub const SYNC_ALPN: &[u8] = b"minutist/sync/notes/1";

/// The notes-sync protocol handler.
///
/// TODO(S3): implement `iroh::protocol::ProtocolHandler` to accept a peer
/// connection, exchange v1 update frames (own length-framing over the bi stream),
/// and merge received frames into the meeting's `notes.ydoc` via `persistence`.
/// Carries the `persistence` root + an apply sink; stubbed here so the ALPN and
/// the module shape are pinned without the connection logic.
pub struct NotesSyncProto;
