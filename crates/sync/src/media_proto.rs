//! The media-manifest exchange protocol (WS4-B S4).
//!
//! After (or alongside) the notes reconciliation, two paired devices reconcile a
//! meeting's media (`audio.opus` and each note asset) by exchanging a
//! [`Manifest`] of `(relative-path, BLAKE3 hash)` pairs over a fresh bidirectional
//! stream on the existing notes ALPN ([`crate::notes_proto::SYNC_ALPN`]), then each
//! side pulls the blobs it is missing from the other over the blobs ALPN
//! ([`iroh_blobs::ALPN`]). The two ALPNs share one [`iroh::Endpoint`] and
//! [`iroh::protocol::Router`]; the manifest names what to pull, the blobs channel
//! moves the bytes.
//!
//! # Wire protocol
//!
//! One reconciliation runs over a single bidirectional QUIC stream, driven by the
//! dialling side (the *initiator*); the accepting side is the *responder*. Each
//! side imports its own meeting media first (which both yields its manifest and
//! stages its blobs in the store so the peer can fetch them), exchanges manifests,
//! then pulls every peer entry it does not already hold by content hash over the
//! blobs ALPN. The pulls run over a separate ALPN connection, so both sides pull
//! concurrently; a final one-byte DONE handshake on the manifest stream makes
//! each side wait for the other to finish before the connection is torn down, so
//! the initiator does not return (and the caller does not observe) before the
//! responder's pull has landed:
//!
//! ```text
//! initiator                                    responder
//! ---------                                    ---------
//! REQUEST  ->  meeting_id (16) + manifest(init)
//!                                        read REQUEST; ensure folder; import
//!                                   <-  MANIFEST    manifest(resp)
//! read manifest(resp)
//! pull init-missing blobs (blobs ALPN)   pull resp-missing blobs (blobs ALPN)
//! DONE     ->  1 byte                <-  DONE        1 byte
//! read DONE (waits for resp's pull)      read DONE (waits for init's pull)
//! close connection ----------------->    conn.closed() resolves; responder returns
//! ```
//!
//! Because blobs are content-addressed, a re-run once both sides hold the same
//! media is a no-op (every manifest entry is already held; nothing is pulled).
//!
//! # Authorisation
//!
//! This protocol multiplexes onto the paired-peer-authorised notes ALPN, so the
//! responder side is reached only after [`crate::endpoint`]'s `AcceptHook` has
//! confirmed the remote is paired. The blob pulls go over the blobs ALPN, whose
//! own accept side is guarded by the same [`PeerDirectory`] check
//! (`AuthorizedBlobs` in [`crate::endpoint`]), so an unpaired peer can neither
//! push a manifest nor serve a blob.
//!
//! [`Manifest`]: crate::blobs::Manifest
//! [`PeerDirectory`]: crate::address_lookup::PeerDirectory

use std::collections::HashSet;

use iroh::endpoint::{Connection, RecvStream, SendStream};
use minutist_common::MeetingId;
use uuid::Uuid;

use crate::blobs::BlobExchange;
use crate::blobs::Manifest;
use crate::notes_proto::StreamKind;
use crate::timeouts::{FRAME_IO_TIMEOUT, PEER_PULL_TIMEOUT, RESPONDER_CLOSE_TIMEOUT};
use crate::{Error, Result};

/// Decode a [`Manifest`] from a received frame body.
fn decode_manifest(bytes: &[u8]) -> Result<Manifest> {
    serde_json::from_slice(bytes)
        .map_err(|e| Error::Protocol(format!("decoding media manifest: {e}")))
}

/// Encode a [`Manifest`] to a frame body.
fn encode_manifest(manifest: &Manifest) -> Result<Vec<u8>> {
    serde_json::to_vec(manifest)
        .map_err(|e| Error::Protocol(format!("encoding media manifest: {e}")))
}

/// Pull every blob in `peer_manifest` that this device does not already hold
/// (by `(rel_path, hash)`) from `peer`, exporting each to its per-meeting path.
///
/// `local_manifest` is this device's own manifest for the meeting (the media it
/// just imported); an entry already present in it, with the same relative path
/// and content hash, is skipped. `peer_manifest` has been validated against path
/// traversal before this runs.
async fn pull_missing(
    ex: BlobExchange<'_>,
    meeting_id: MeetingId,
    local_manifest: &Manifest,
    peer_manifest: &Manifest,
) -> Result<()> {
    let held: HashSet<(&str, [u8; 32])> = local_manifest
        .entries
        .iter()
        .map(|e| (e.rel_path.as_str(), *e.hash.as_bytes()))
        .collect();

    for entry in &peer_manifest.entries {
        if held.contains(&(entry.rel_path.as_str(), *entry.hash.as_bytes())) {
            continue;
        }
        ex.store
            .download(
                ex.endpoint,
                ex.peer,
                ex.root,
                meeting_id,
                &entry.rel_path,
                entry.hash,
            )
            .await?;
    }
    Ok(())
}

/// Run the *initiator* (dialling) side of one media reconciliation for
/// `meeting_id` over `conn`, against the meetings `root`.
///
/// Imports this device's own media for the meeting (staging it in the blob store
/// and producing the local manifest), opens a bi stream, writes the
/// [`StreamKind::Media`] tag, sends the REQUEST (meeting id + local manifest),
/// reads the peer's manifest, then pulls every peer blob it lacks over the blobs
/// ALPN. The caller closes `conn` after this returns: the initiator is the last
/// reader on this stream.
///
/// `peer` is the remote [`EndpointId`] the downloader dials over the shared
/// endpoint. It must equal `conn.remote_id()`; the caller passes the id it dialled.
pub(crate) async fn initiate_media_sync(
    conn: &Connection,
    ex: BlobExchange<'_>,
    meeting_id: MeetingId,
) -> Result<()> {
    let framer = ex.framer(StreamKind::Media);
    let local_manifest = ex.store.import_meeting(ex.root, meeting_id).await?;

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| Error::Protocol(format!("opening media-sync bi stream: {e}")))?;

    // Tag this stream as a media exchange so the responder dispatches correctly.
    send.write_all(&[StreamKind::Media as u8])
        .await
        .map_err(|e| Error::Protocol(format!("writing media stream tag: {e}")))?;

    // REQUEST: meeting id (fixed 16 bytes) then our manifest.
    send.write_all(meeting_id.0.as_bytes())
        .await
        .map_err(|e| Error::Protocol(format!("writing meeting id: {e}")))?;
    framer
        .write(&mut send, &encode_manifest(&local_manifest)?)
        .await?;

    // Read the peer's manifest and pull what we lack over the blobs ALPN.
    let peer_manifest = decode_manifest(&framer.read(&mut recv).await?)?;
    peer_manifest.validate()?;
    pull_missing(ex, meeting_id, &local_manifest, &peer_manifest).await?;

    // Signal our pull is complete, then wait for the responder's DONE so we do not
    // return (tearing down the connection) before its pull has landed.
    finish_done(&mut send, &mut recv).await
}

/// Run the *responder* (accepting) side of one media reconciliation against the
/// meetings `root`, over a bi stream the accept hook has already accepted and
/// whose leading [`StreamKind`] tag it has already consumed.
///
/// Reads the REQUEST (meeting id + initiator manifest), replies with its own
/// manifest, then pulls every initiator blob it lacks over the blobs ALPN. The
/// fs blob store's export creates the meeting folder itself the moment there is
/// a blob to write, so a brand-new meeting needs no folder pre-created. Parks
/// on [`Connection::closed`] so the router does not drop the connection before
/// the initiator has read the manifest.
///
/// `peer` is the remote [`EndpointId`] (`conn.remote_id()`), already authorised by
/// the notes-ALPN accept hook before this runs.
///
/// [`Connection::closed`]: iroh::endpoint::Connection::closed
pub(crate) async fn respond_media_sync(
    conn: &Connection,
    ex: BlobExchange<'_>,
    send: &mut SendStream,
    recv: &mut RecvStream,
) -> Result<()> {
    let framer = ex.framer(StreamKind::Media);
    // REQUEST: meeting id then the initiator's manifest. Bounded by
    // FRAME_IO_TIMEOUT like every other frame read on this stream: this one is a
    // raw fixed-size read rather than a length-prefixed frame, so it needs its own
    // explicit bound to stay off the unbounded-await slowloris surface.
    let mut id_buf = [0u8; 16];
    tokio::time::timeout(FRAME_IO_TIMEOUT, recv.read_exact(&mut id_buf))
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                peer = %ex.peer,
                timeout = ?FRAME_IO_TIMEOUT,
                "reading the media-sync meeting id timed out"
            );
            Error::Protocol(format!(
                "reading media-sync meeting id timed out after {FRAME_IO_TIMEOUT:?}"
            ))
        })?
        .map_err(|e| Error::Protocol(format!("reading meeting id: {e}")))?;
    let meeting_id = MeetingId(Uuid::from_bytes(id_buf));
    let peer_manifest = decode_manifest(&framer.read(recv).await?)?;
    peer_manifest.validate()?;

    // No `MeetingFolder::ensure` here: a content-less peer meeting must not
    // materialise a blank-placeholder folder just because its id was named in a
    // stream header. `import_meeting`/`pull_missing` need no pre-created
    // directory: `resolve_audio_path`/`assets_dir.is_dir()` return `None`/`false`
    // on an absent folder, and the fs blob store's own export creates the parent
    // directory once there is an actual blob to write. If the folder already
    // exists with real `notes.ydoc` content (from an earlier or same-session
    // notes sync), repair a still-blank metadata.json now rather than waiting
    // on the next notes sweep.
    crate::notes_proto::project_meta_best_effort(ex.root, meeting_id);

    // Import our own media (stages our blobs for the peer to fetch + our manifest).
    let local_manifest = ex.store.import_meeting(ex.root, meeting_id).await?;

    // Reply with our manifest so the initiator can pull what it lacks.
    framer
        .write(send, &encode_manifest(&local_manifest)?)
        .await?;

    // Pull what we lack from the initiator over the blobs ALPN.
    pull_missing(ex, meeting_id, &local_manifest, &peer_manifest).await?;

    // Signal our pull is complete, then wait for the initiator's DONE so neither
    // side observes completion before both pulls have landed.
    finish_done(send, recv).await?;

    // Hold the connection open until the initiator closes; returning here would let
    // the router drop and abort the stream. Bounded by RESPONDER_CLOSE_TIMEOUT so
    // a stalled or hostile initiator that never closes cannot pin this task
    // forever.
    tokio::time::timeout(RESPONDER_CLOSE_TIMEOUT, conn.closed())
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                peer = %ex.peer,
                timeout = ?RESPONDER_CLOSE_TIMEOUT,
                "media-sync responder timed out waiting for the initiator to close"
            );
            Error::Protocol(format!(
                "media-sync responder wait for close timed out after {RESPONDER_CLOSE_TIMEOUT:?}"
            ))
        })?;
    Ok(())
}

/// The completion handshake: write a one-byte DONE, finish the send direction,
/// then read the peer's one-byte DONE. The write and read do not deadlock: a
/// single byte fits the QUIC send buffer, so both sides write before either
/// blocks on the read.
///
/// The read is bounded by [`PEER_PULL_TIMEOUT`], not [`FRAME_IO_TIMEOUT`]:
/// unlike every other read on this stream, this one legitimately spans however
/// long the peer takes to pull every blob it lacks over the separate blobs ALPN
/// before it writes its byte, not just one frame's transfer time.
async fn finish_done(send: &mut SendStream, recv: &mut RecvStream) -> Result<()> {
    send.write_all(&[DONE])
        .await
        .map_err(|e| Error::Protocol(format!("writing media-sync done: {e}")))?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing media-sync send: {e}")))?;
    let mut done = [0u8; 1];
    tokio::time::timeout(PEER_PULL_TIMEOUT, recv.read_exact(&mut done))
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                timeout = ?PEER_PULL_TIMEOUT,
                "reading the peer's media-sync done byte timed out"
            );
            Error::Protocol(format!(
                "reading media-sync done timed out after {PEER_PULL_TIMEOUT:?}"
            ))
        })?
        .map_err(|e| Error::Protocol(format!("reading media-sync done: {e}")))?;
    if done[0] != DONE {
        return Err(Error::Protocol(format!(
            "unexpected media-sync done byte {}",
            done[0]
        )));
    }
    Ok(())
}

/// The completion-handshake byte exchanged after both sides finish pulling.
const DONE: u8 = 0xD0;
