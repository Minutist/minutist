//! The derived-artifact exchange protocol ([`StreamKind::Artifacts`], WS4-B).
//!
//! Two paired devices reconcile a meeting's DERIVED outputs — `transcript.json`
//! (ASR + diarization labels) and `summary.md` — the files processing produces.
//! It mirrors [`crate::media_proto`]: a [`ArtifactManifest`] is exchanged over a
//! fresh bidirectional stream on the sync ALPN ([`crate::notes_proto::SYNC_ALPN`]),
//! then each side pulls the entries it should take over the blobs ALPN
//! ([`iroh_blobs::ALPN`]). It differs from media in ONE load-bearing way: a media
//! blob is immutable and content-addressed, so "pull what I lack by hash" is the
//! whole rule; a derived artifact is MUTABLE (a meeting can be reprocessed), so the
//! manifest stamps each entry with the authority that produced those exact bytes
//! (`produced_by` host + `produced_at`), and the pull takes a peer's entry only
//! when it **strictly supersedes** the local one (`planning/DESIGN_artifacts.md`
//! §2). The authority travels WITH the bytes and is never re-derived from
//! `metadata.json` (whose `Processed` stamp propagates over Discovery independently
//! of the bytes — the relay-clobber, §2 C1).
//!
//! # Wire protocol
//!
//! One reconciliation runs over a single bidirectional QUIC stream, driven by the
//! dialling side (the *initiator*); the accepting side is the *responder*. The
//! shape is identical to [`crate::media_proto`]: import locally (yielding the
//! manifest + staging blobs for the peer), exchange manifests, pull over the blobs
//! ALPN, then a one-byte DONE handshake so neither side tears the connection down
//! before both pulls have landed.
//!
//! ```text
//! initiator                                    responder
//! ---------                                    ---------
//! REQUEST  ->  meeting_id (16) + manifest(init)
//!                                        read REQUEST; ensure folder; import
//!                                   <-  MANIFEST    manifest(resp)
//! read manifest(resp)
//! pull superseding (blobs ALPN)          pull superseding (blobs ALPN)
//! DONE     ->  1 byte                <-  DONE        1 byte
//! read DONE (waits for resp's pull)      read DONE (waits for init's pull)
//! close connection ----------------->    conn.closed() resolves; responder returns
//! ```
//!
//! A re-run once both sides hold the authoritative copy is a no-op: every peer
//! entry is byte-identical (skipped) or does not supersede the local one.
//!
//! # Authorisation
//!
//! Like media, this multiplexes onto the paired-peer-authorised sync ALPN (the
//! responder is reached only after [`crate::endpoint`]'s `AcceptHook` confirms the
//! remote is paired), and the blob pulls go over the blobs ALPN guarded by the same
//! [`PeerDirectory`](crate::address_lookup::PeerDirectory) check.
//!
//! # Processed-without-bytes
//!
//! A device advertises an artifact entry only for a file present on disk
//! ([`BlobStore::import_artifacts`]), so a relay that learned `Processed` over
//! Discovery but lacks the bytes contributes NO entry and never offers a
//! `Processed`-with-no-transcript copy. A consumer in that position simply receives
//! nothing this round (fetch-pending) and pulls on a later sync once a holder is
//! reachable — not an error (§4).

use std::path::Path;

use iroh::endpoint::{Connection, RecvStream, SendStream};
use iroh::{Endpoint, EndpointId};
use minutist_common::{HostRef, MeetingId, ProcessingLifecycle};
use uuid::Uuid;

use crate::blobs::{record_artifact_authority, ArtifactManifest, BlobStore};
use crate::frame::{read_frame, write_frame};
use crate::notes_proto::StreamKind;
use crate::timeouts::{FRAME_IO_TIMEOUT, PEER_PULL_TIMEOUT, RESPONDER_CLOSE_TIMEOUT};
use crate::{discovery_proto, Error, Result};

/// The completion-handshake byte exchanged after both sides finish pulling.
/// Distinct from the media handshake byte purely as a debugging aid — it rides a
/// separate stream, so no collision is possible.
const DONE: u8 = 0xA4;

/// Decode an [`ArtifactManifest`] from a received frame body.
fn decode_manifest(bytes: &[u8]) -> Result<ArtifactManifest> {
    serde_json::from_slice(bytes)
        .map_err(|e| Error::Protocol(format!("decoding artifact manifest: {e}")))
}

/// Encode an [`ArtifactManifest`] to a frame body.
fn encode_manifest(manifest: &ArtifactManifest) -> Result<Vec<u8>> {
    serde_json::to_vec(manifest)
        .map_err(|e| Error::Protocol(format!("encoding artifact manifest: {e}")))
}

/// The producer authority to stamp on locally-produced artifacts: a meeting's
/// local `Processed { processed_by, at }`, or `None` when the meeting is not
/// locally `Processed` (a capture-only device, or a consumer that has not received
/// the derived state). Used only as the byte-coherent fallback when the
/// authority store holds no record for the on-disk bytes — see
/// [`BlobStore::import_artifacts`].
///
/// The underlying `metadata.json` read is intentionally lock-free: its failure
/// mode is conservative — a torn/absent/corrupt read yields `Local` → `None` →
/// the artifact simply is not advertised this round (fetch-pending, self-heals on
/// the next sync), never a clobber — so it does not need the metadata lock.
fn producer_authority(root: &Path, meeting_id: MeetingId) -> Option<(HostRef, String)> {
    match discovery_proto::read_local_processing(root, meeting_id) {
        ProcessingLifecycle::Processed { processed_by, at } => Some((processed_by, at)),
        _ => None,
    }
}

/// Pull every peer artifact entry that should overwrite the local copy, exporting
/// each to its per-meeting path and recording the authority that arrived with the
/// bytes.
///
/// For each peer entry: if we advertise a provable copy, take the peer's only when
/// its bytes differ AND it strictly supersedes ours
/// ([`crate::blobs::ArtifactEntry::supersedes`] — strict `>` on `produced_at`, ties
/// to the lowest `produced_by` HostRef). If we advertise NO provable copy, pull only
/// when we genuinely lack the file on disk: a file present but unstampable (bytes
/// produced before a `Processed` flip, or a lost/corrupt authority record) is NOT
/// overwritten with a peer copy we cannot prove is newer — that would silently lose
/// possibly-newer local bytes. The local file is kept (stuck until its authority is
/// re-established) and the situation logged. `peer_manifest` has been validated
/// before this runs.
async fn pull_superseding(
    store: &BlobStore,
    endpoint: &Endpoint,
    peer: EndpointId,
    meetings_root: &Path,
    meeting_id: MeetingId,
    local_manifest: &ArtifactManifest,
    peer_manifest: &ArtifactManifest,
) -> Result<()> {
    for entry in &peer_manifest.entries {
        let should_pull = match local_manifest.entry(&entry.rel_path) {
            // We advertise a provable copy: take the peer's only if its bytes differ
            // AND it strictly supersedes ours (byte-identical → nothing to do).
            Some(ours) => ours.hash != entry.hash && entry.supersedes(ours),
            // We advertise no provable copy. Pull ONLY if we genuinely lack the file
            // on disk; if a file IS present but we could not stamp it, refuse to
            // overwrite it with a copy we cannot prove is newer (silent-loss guard).
            None => {
                let local_path = meetings_root
                    .join(meeting_id.0.to_string())
                    .join(&entry.rel_path);
                if local_path.exists() {
                    tracing::warn!(
                        target: "sync",
                        meeting_id = %meeting_id.0,
                        rel = %entry.rel_path,
                        "peer offers an artifact we hold on disk but cannot stamp (no provable authority); keeping local bytes, not overwriting"
                    );
                    false
                } else {
                    true
                }
            }
        };
        if !should_pull {
            continue;
        }

        store
            .download_artifact(
                endpoint,
                peer,
                meetings_root,
                meeting_id,
                &entry.rel_path,
                entry.hash,
            )
            .await?;

        // Record the authority that arrived WITH these bytes so a later import on
        // this device re-advertises it faithfully (never re-derived from
        // metadata.json — DESIGN §2 C1).
        record_artifact_authority(
            meetings_root,
            meeting_id,
            &entry.rel_path,
            entry.hash,
            entry.produced_by.clone(),
            entry.produced_at.clone(),
        )?;

        // TODO(DESIGN §4, build-order step 5 — consumer read-sync): a received
        // transcript.json bypasses persistence's `write_transcript`, which clears
        // the per-device `translations.json`; reconcile (clear/regenerate) the
        // stale translations and emit a reload signal so an open UI re-reads. Both
        // land with the consumer read-sync slice; `sync` has no UI/persistence edge
        // to drive them here. This is a content-correctness gap (a stale translation
        // would key to the wrong segments), NOT polish — it is a RELEASE GATE: the
        // consumer read-sync slice must land before artifact sync + translations are
        // both enabled in a shipping build.
    }
    Ok(())
}

/// Run the *initiator* (dialling) side of one artifact reconciliation for
/// `meeting_id` over `conn`, against the meetings `root`.
///
/// Imports this device's own artifacts for the meeting (staging them in the blob
/// store and producing the local manifest, each entry authority-stamped), opens a
/// bi stream, writes the [`StreamKind::Artifacts`] tag, sends the REQUEST (meeting
/// id + local manifest), reads the peer's manifest, then pulls every superseding
/// peer entry over the blobs ALPN. The caller closes `conn` after this returns —
/// the initiator is the last reader on this stream.
///
/// `peer` is the remote [`EndpointId`] the downloader dials; it must equal
/// `conn.remote_id()` (the caller passes the id it dialled).
pub async fn initiate_artifacts_sync(
    conn: &Connection,
    store: &BlobStore,
    endpoint: &Endpoint,
    peer: EndpointId,
    root: &Path,
    meeting_id: MeetingId,
) -> Result<()> {
    let local_manifest = store
        .import_artifacts(root, meeting_id, producer_authority(root, meeting_id))
        .await?;

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| Error::Protocol(format!("opening artifacts-sync bi stream: {e}")))?;

    // Tag this stream as an artifacts exchange so the responder dispatches correctly.
    send.write_all(&[StreamKind::Artifacts as u8])
        .await
        .map_err(|e| Error::Protocol(format!("writing artifacts stream tag: {e}")))?;

    // REQUEST: meeting id (fixed 16 bytes) then our manifest.
    send.write_all(meeting_id.0.as_bytes())
        .await
        .map_err(|e| Error::Protocol(format!("writing meeting id: {e}")))?;
    write_frame(&mut send, &encode_manifest(&local_manifest)?).await?;

    // Read the peer's manifest and pull what supersedes ours over the blobs ALPN.
    let peer_manifest = decode_manifest(&read_frame(&mut recv).await?)?;
    peer_manifest.validate()?;
    pull_superseding(
        store,
        endpoint,
        peer,
        root,
        meeting_id,
        &local_manifest,
        &peer_manifest,
    )
    .await?;

    // Signal our pull is complete, then wait for the responder's DONE so we do not
    // return (tearing down the connection) before its pull has landed.
    finish_done(&mut send, &mut recv).await
}

/// Run the *responder* (accepting) side of one artifact reconciliation against the
/// meetings `root`, over a bi stream the accept hook has already accepted and whose
/// leading [`StreamKind`] tag it has already consumed.
///
/// Reads the REQUEST (meeting id + initiator manifest), ensures the meeting folder
/// exists, imports its own artifacts (staging blobs + the local manifest), replies
/// with that manifest, then pulls every superseding initiator entry over the blobs
/// ALPN. Parks on [`Connection::closed`] so the router does not drop the connection
/// before the initiator has read the manifest.
///
/// `peer` is the remote [`EndpointId`] (`conn.remote_id()`), already authorised by
/// the sync-ALPN accept hook before this runs.
///
/// [`Connection::closed`]: iroh::endpoint::Connection::closed
pub async fn respond_artifacts_sync(
    conn: &Connection,
    send: &mut SendStream,
    recv: &mut RecvStream,
    store: &BlobStore,
    endpoint: &Endpoint,
    peer: EndpointId,
    root: &Path,
) -> Result<()> {
    // REQUEST: meeting id then the initiator's manifest. Bounded by
    // FRAME_IO_TIMEOUT like every other frame read on this stream — this one is
    // a raw fixed-size read rather than a length-prefixed frame, so it needs its
    // own explicit bound to stay off the unbounded-await slowloris surface.
    let mut id_buf = [0u8; 16];
    tokio::time::timeout(FRAME_IO_TIMEOUT, recv.read_exact(&mut id_buf))
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                peer = %peer,
                timeout = ?FRAME_IO_TIMEOUT,
                "reading the artifacts-sync meeting id timed out"
            );
            Error::Protocol(format!(
                "reading artifacts-sync meeting id timed out after {FRAME_IO_TIMEOUT:?}"
            ))
        })?
        .map_err(|e| Error::Protocol(format!("reading meeting id: {e}")))?;
    let meeting_id = MeetingId(Uuid::from_bytes(id_buf));
    let peer_manifest = decode_manifest(&read_frame(recv).await?)?;
    peer_manifest.validate()?;

    // A meeting syncing artifacts to a device that lacks its folder must not fail
    // for want of the directory; `notes-crdt` owns the folder/metadata creation.
    notes_crdt::MeetingFolder::ensure(root, meeting_id)
        .map_err(|e| Error::Protocol(format!("ensuring inbound meeting folder: {e}")))?;

    // `ensure` only wrote the arrival-time placeholder; if this meeting's
    // notes.ydoc already carries the origin's real title/duration (from an
    // earlier or same-session notes sync), project it over metadata.json now
    // rather than leaving artifacts attached to a blank placeholder
    // indefinitely — mirrors `notes_proto::apply_inbound`. Best-effort: a
    // projection failure must not fail the artifacts sync; the next notes
    // sweep re-applies it.
    if let Err(e) = notes_crdt::meta_crdt::project_ydoc_meta_into_metadata(root, meeting_id) {
        tracing::warn!(
            target: "sync",
            meeting_id = %meeting_id.0,
            error = %e,
            "projecting synced metadata over metadata.json failed; will retry next sweep"
        );
    }

    // Import our own artifacts (stages our blobs for the peer + our manifest).
    let local_manifest = store
        .import_artifacts(root, meeting_id, producer_authority(root, meeting_id))
        .await?;

    // Reply with our manifest so the initiator can pull what supersedes its copy.
    write_frame(send, &encode_manifest(&local_manifest)?).await?;

    // Pull what supersedes our copy from the initiator over the blobs ALPN.
    pull_superseding(
        store,
        endpoint,
        peer,
        root,
        meeting_id,
        &local_manifest,
        &peer_manifest,
    )
    .await?;

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
                peer = %peer,
                timeout = ?RESPONDER_CLOSE_TIMEOUT,
                "artifacts-sync responder timed out waiting for the initiator to close"
            );
            Error::Protocol(format!(
                "artifacts-sync responder wait for close timed out after {RESPONDER_CLOSE_TIMEOUT:?}"
            ))
        })?;
    Ok(())
}

/// The completion handshake: write a one-byte DONE, finish the send direction, then
/// read the peer's one-byte DONE. A single byte fits the QUIC send buffer, so both
/// sides write before either blocks on the read (no deadlock).
///
/// The read is bounded by [`PEER_PULL_TIMEOUT`], not [`FRAME_IO_TIMEOUT`]: unlike
/// every other read on this stream, this one legitimately spans however long the
/// peer takes to pull every artifact that supersedes its copy over the blobs ALPN
/// before it writes its byte, not just one frame's transfer time.
async fn finish_done(send: &mut SendStream, recv: &mut RecvStream) -> Result<()> {
    send.write_all(&[DONE])
        .await
        .map_err(|e| Error::Protocol(format!("writing artifacts-sync done: {e}")))?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing artifacts-sync send: {e}")))?;
    let mut done = [0u8; 1];
    tokio::time::timeout(PEER_PULL_TIMEOUT, recv.read_exact(&mut done))
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                timeout = ?PEER_PULL_TIMEOUT,
                "reading the peer's artifacts-sync done byte timed out"
            );
            Error::Protocol(format!(
                "reading artifacts-sync done timed out after {PEER_PULL_TIMEOUT:?}"
            ))
        })?
        .map_err(|e| Error::Protocol(format!("reading artifacts-sync done: {e}")))?;
    if done[0] != DONE {
        return Err(Error::Protocol(format!(
            "unexpected artifacts-sync done byte {}",
            done[0]
        )));
    }
    Ok(())
}
