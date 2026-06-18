//! Meeting-media blob sync.
//!
//! Audio and note assets are large and immutable, so they ride iroh-blobs'
//! content-addressed (BLAKE3) transfer rather than the notes update protocol. The
//! source paths come from `persistence`:
//!
//! - audio at [`persistence::MeetingFolder::audio_path`] (`audio.opus`),
//! - note assets via [`persistence::read_note_asset`] /
//!   [`persistence::save_note_asset`].
//!
//! A device adds a blob to its store (yielding a BLAKE3 hash), publishes the hash
//! to the peer out-of-band, and the peer downloads it by hash from the known
//! provider endpoint. The blobs protocol is multiplexed onto the same iroh
//! endpoint as the notes protocol (`iroh_blobs::ALPN`).

/// The meeting-media blob-sync handle.
///
/// TODO(S4): hold an `iroh_blobs` store and a `BlobsProtocol` registered on the
/// shared `Router`; expose `publish(meeting_id, kind) -> Hash` (add the
/// `persistence` source file to the store) and `fetch(hash, provider)` (download
/// then export back onto the meeting folder). Stubbed here so the module shape is
/// pinned without the iroh-blobs store wiring.
pub struct BlobSync;
