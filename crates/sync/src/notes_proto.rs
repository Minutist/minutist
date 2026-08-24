//! The Yjs notes-update exchange protocol.
//!
//! A small custom iroh protocol (ALPN [`SYNC_ALPN`]) reconciles one meeting's
//! Yjs CRDT document between two of a user's paired devices. The authoritative
//! local document is `notes-crdt`'s `notes.ydoc`. The exchange is a
//! state-vector / minimal-diff reconciliation:
//!
//! - each side reads its local state as a lib0-**v1** whole-state update with
//!   [`notes_crdt::NotesStore::read_ydoc_state`] (an absent `notes.ydoc` is the
//!   empty-document state),
//! - each side derives its [state vector] from that v1 state
//!   ([`yrs::encode_state_vector_from_update_v1`]) and sends it to the peer,
//! - the peer answers with the **minimal** update covering everything the sender
//!   has not yet seen ([`yrs::diff_updates_v1`] of the peer's local state against
//!   the sender's state vector),
//! - each side merges the inbound diff with
//!   [`notes_crdt::NotesStore::apply_update`], which re-derives `notes.json` and
//!   `notes.md` (`notes-crdt` owns projection-writing; sync never touches it).
//!
//! yrs merge is commutative and idempotent, so the exchange is order-independent
//! and re-running it is a no-op once both sides have converged.
//!
//! # Wire protocol
//!
//! One reconciliation runs over a single bidirectional QUIC stream, driven by
//! the dialling side (the *initiator*); the accepting side is the *responder*.
//! The exchange strictly alternates so neither side blocks on a read while the
//! other also blocks: no deadlock on the one stream. The **initiator is the
//! last reader**, so it holds the inbound diff in hand before it closes the
//! connection. The responder parks on [`Connection::closed`] until that close
//! arrives, so its applied write is never aborted by a premature teardown.
//!
//! ```text
//! initiator                                   responder
//! ---------                                   ---------
//! REQUEST  ->  meeting_id (16) + sv_init
//!                                       read REQUEST
//!                                  <-  SV       sv_resp
//! read sv_resp
//! DIFF     ->  diff(init_local, sv_resp), finish send
//!                                       read DIFF; apply (resp converges)
//!                                  <-  DIFF     diff(resp_local, sv_init), finish
//! read DIFF; apply (init converges)
//! close connection ------------------> conn.closed() resolves; responder returns
//! ```
//!
//! Every variable-length field is a sealed frame: a `u32` big-endian byte length
//! then that many bytes, opened under the account content key
//! (`crate::frame::Framer`). The
//! `meeting_id` is a fixed 16-byte UUID (no prefix). A diff that carries no
//! changes (the peer had nothing the receiver lacked) is recognised by
//! [`is_noop_update`] and skipped rather than written, so an up-to-date or
//! empty meeting touches no disk.
//!
//! [state vector]: https://docs.rs/yrs/0.26.0/yrs/struct.StateVector.html
//! [`Connection::closed`]: iroh::endpoint::Connection::closed

use std::path::Path;

use iroh::endpoint::{Connection, RecvStream, SendStream};
use minutist_common::MeetingId;
use notes_crdt::NotesStore;
use uuid::Uuid;
use yrs::updates::decoder::Decode;
use yrs::updates::encoder::Encode;
use yrs::Update;

use crate::content_key::FrameCipher;
use crate::frame::Framer;
use crate::timeouts::{FRAME_IO_TIMEOUT, RESPONDER_CLOSE_TIMEOUT};
use crate::{Error, Result};

/// ALPN for the sync-update protocol. Bumping the suffix is a wire break.
///
/// Every sync exchange multiplexes onto this one ALPN: notes reconciliation,
/// the media-manifest exchange ([`crate::media_proto`]), discovery
/// ([`crate::discovery_proto`]), and the derived-artifact exchange
/// ([`crate::artifacts_proto`]). The initiator writes a one-byte [`StreamKind`]
/// tag as the first byte of each bidirectional stream, and the accept hook
/// dispatches on it, so one paired-peer authorisation point (the notes-ALPN
/// accept hook) covers all of them.
pub const SYNC_ALPN: &[u8] = b"minutist/sync/notes/2";

/// The first byte of a sync bidirectional stream, selecting the protocol that
/// runs over it. Lets notes and media reconciliation share one ALPN and one
/// authorised accept loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamKind {
    /// Yjs notes reconciliation (`initiate_notes_sync` / `respond_notes_sync`).
    Notes = 1,
    /// Media-manifest exchange ([`crate::media_proto`]).
    Media = 2,
    /// Meeting-list + processing-lifecycle discovery
    /// ([`crate::discovery_proto`]). Each side sends the
    /// `(MeetingId, ProcessingLifecycle)` of every meeting it holds, so a peer
    /// learns both which meetings exist and their host-authoritative processing
    /// state. Appended as tag `3`: the tag is the wire contract, so new variants
    /// must only ever be added at the end. An older peer rejects an unknown tag
    /// via [`Self::from_tag`] rather than mis-dispatching.
    Discovery = 3,
    /// Derived-artifact exchange ([`crate::artifacts_proto`]): the
    /// processor→consumer sync of a meeting's `transcript.json` + `summary.md`
    /// (the outputs processing produces). Mirrors [`Self::Media`]: a manifest of
    /// `(relative-path, hash, produced_by, produced_at)` entries, then a
    /// content-addressed blob pull over the blobs ALPN. The authority is stamped
    /// per entry, bound to the bytes, so a stale relay copy can never clobber a
    /// newer producer copy. Appended as tag `4` (append-only).
    Artifacts = 4,
}

impl StreamKind {
    /// Decode a stream-kind tag byte, rejecting an unknown value as a protocol
    /// error rather than silently mis-dispatching.
    pub fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Notes),
            2 => Ok(Self::Media),
            3 => Ok(Self::Discovery),
            4 => Ok(Self::Artifacts),
            other => Err(Error::Protocol(format!("unknown sync stream kind {other}"))),
        }
    }
}

/// The local v1 whole-state update for `meeting_id`, or the empty-document state
/// when the meeting has no `notes.ydoc` yet.
///
/// `read_ydoc_state` returns `None` for a never-noted meeting; the empty v1
/// update derived from a fresh `yrs` doc stands in so the state-vector / diff
/// math has a valid operand on both sides.
fn local_v1_state(root: &Path, meeting_id: MeetingId) -> Result<Vec<u8>> {
    match NotesStore::read_ydoc_state(root, meeting_id) {
        Ok(Some(state)) => Ok(state),
        Ok(None) => Ok(empty_v1_state()),
        Err(e) => Err(Error::Protocol(format!(
            "reading local notes.ydoc state: {e}"
        ))),
    }
}

/// The lib0-v1 whole-state update of an empty Yjs document: the operand for a
/// meeting whose `notes.ydoc` does not exist yet.
fn empty_v1_state() -> Vec<u8> {
    // `yrs::Update::default().encode_v1()` is the empty update; deriving it from
    // an empty `StateVector` keeps this in lock-step with how `notes-crdt`
    // encodes whole state (`encode_state_as_update_v1` of the default SV).
    yrs::Update::default().encode_v1()
}

/// The encoded state vector of a lib0-v1 whole-state update.
fn state_vector_of(v1_state: &[u8]) -> Result<Vec<u8>> {
    yrs::encode_state_vector_from_update_v1(v1_state)
        .map_err(|e| Error::Protocol(format!("encoding state vector: {e}")))
}

/// The minimal lib0-v1 update carrying everything in `v1_state` that the peer
/// (described by `peer_sv`) has not yet observed.
fn diff_against(v1_state: &[u8], peer_sv: &[u8]) -> Result<Vec<u8>> {
    yrs::diff_updates_v1(v1_state, peer_sv)
        .map_err(|e| Error::Protocol(format!("computing minimal diff: {e}")))
}

/// Whether a lib0-v1 update carries no changes: no inserted blocks and no
/// deletions. Such a diff is the peer saying it is already up to date, so
/// applying it would be a pure no-op merge.
fn is_noop_update(diff: &[u8]) -> Result<bool> {
    let update =
        Update::decode_v1(diff).map_err(|e| Error::Protocol(format!("decoding diff: {e}")))?;
    Ok(update.is_empty() && update.delete_set().is_empty())
}

/// Merge an inbound v1 diff into the meeting's `notes.ydoc` via `notes-crdt`,
/// preserving the existing `notes.md` projection.
///
/// `apply_update` re-derives `notes.json` from the merged doc but takes the
/// markdown verbatim: rendering markdown needs the editor's typed schema, which
/// neither `sync` nor `notes-crdt` models, so a CRDT sync must not blank the
/// existing export (the editor re-renders it on next save).
///
/// A no-op diff (nothing the peer had that we lacked) is skipped before
/// touching disk. A change-carrying diff first ensures the meeting folder via
/// [`notes_crdt::MeetingFolder::ensure`], so a brand-new meeting syncing to a
/// device that lacks its folder does not fail for want of the directory;
/// `notes-crdt` owns the folder/metadata creation (a placeholder `metadata.json`
/// the authoritative one later overwrites).
fn apply_inbound(root: &Path, meeting_id: MeetingId, diff: &[u8]) -> Result<()> {
    if is_noop_update(diff)? {
        return Ok(());
    }
    notes_crdt::MeetingFolder::ensure(root, meeting_id)
        .map_err(|e| Error::Protocol(format!("ensuring inbound meeting folder: {e}")))?;
    let notes_md = match NotesStore::load(root, meeting_id) {
        Ok(Some(data)) => data.markdown,
        Ok(None) => String::new(),
        Err(e) => return Err(Error::Protocol(format!("loading notes.md: {e}"))),
    };
    NotesStore::apply_update(root, meeting_id, diff, &notes_md)
        .map_err(|e| Error::Protocol(format!("applying inbound update: {e}")))?;

    // The merged `notes.ydoc` now carries any inbound descriptive-metadata map ops
    // (0052: the map rides inside the same doc). Project them over
    // `metadata.json` so a synced meeting shows real dates, title, and codec
    // instead of the arrival-time placeholder `MeetingFolder::ensure` wrote.
    project_meta_best_effort(root, meeting_id);
    Ok(())
}

/// Best-effort projection of a meeting's `notes.ydoc` descriptive metadata over
/// `metadata.json`. Shared by every responder that may touch a meeting whose
/// own notes sync has not (yet) run: `apply_inbound` here, and
/// `media_proto`/`artifacts_proto`'s responders. A failure must not fail the
/// caller's sync; the next sweep re-applies it. Cheap to call unconditionally,
/// since `project_ydoc_meta_into_metadata` is a no-op read when there is no
/// `notes.ydoc` yet, or when the projection changes nothing.
pub(crate) fn project_meta_best_effort(root: &Path, meeting_id: MeetingId) {
    if let Err(e) = notes_crdt::meta_crdt::project_ydoc_meta_into_metadata(root, meeting_id) {
        tracing::warn!(
            target: "sync",
            meeting_id = %meeting_id.0,
            error = %e,
            "projecting synced metadata over metadata.json failed; will retry next sweep"
        );
    }
}

/// Run the *initiator* (dialling) side of one notes reconciliation for
/// `meeting_id` over `conn`, against the meetings `root`.
///
/// Opens a bi stream, writes the [`StreamKind::Notes`] tag, sends the REQUEST
/// (meeting id + local state vector), reads the peer's state vector, sends the
/// DIFF the peer is missing (so the responder converges), then reads the DIFF we
/// are missing and applies it (so we converge). The caller closes `conn` after
/// this returns: the initiator is the last reader, so closing then is safe. See
/// the module wire-protocol diagram.
pub(crate) async fn initiate_notes_sync(
    conn: &Connection,
    cipher: &FrameCipher,
    root: &Path,
    meeting_id: MeetingId,
) -> Result<()> {
    let framer = Framer::new(cipher, StreamKind::Notes);
    let local_state = local_v1_state(root, meeting_id)?;
    let local_sv = state_vector_of(&local_state)?;

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| Error::Protocol(format!("opening notes-sync bi stream: {e}")))?;

    // Tag this stream as a notes exchange so the responder dispatches correctly.
    send.write_all(&[StreamKind::Notes as u8])
        .await
        .map_err(|e| Error::Protocol(format!("writing notes stream tag: {e}")))?;

    // REQUEST: meeting id (fixed 16 bytes) then our state vector.
    send.write_all(meeting_id.0.as_bytes())
        .await
        .map_err(|e| Error::Protocol(format!("writing meeting id: {e}")))?;
    framer.write(&mut send, &local_sv).await?;

    // Learn the peer's state vector, then send it the diff it is missing.
    let peer_sv = framer.read(&mut recv).await?;
    let diff_for_peer = diff_against(&local_state, &peer_sv)?;
    framer.write(&mut send, &diff_for_peer).await?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing notes-sync send: {e}")))?;

    // Read the diff we are missing and merge it last (initiator is last reader).
    let diff_for_us = framer.read(&mut recv).await?;
    apply_inbound(root, meeting_id, &diff_for_us)?;
    Ok(())
}

/// Run the *responder* (accepting) side of one notes reconciliation against the
/// meetings `root`, over a bi stream the accept hook has already accepted and
/// whose leading [`StreamKind`] tag it has already consumed.
///
/// Reads the REQUEST (meeting id + initiator state vector), replies with its own
/// state vector, reads the initiator's DIFF and applies it (so we converge), then
/// sends the DIFF the initiator is missing. Finally parks on
/// [`Connection::closed`] so the router does not drop the connection (aborting our
/// last write) before the initiator has read it. See the module wire-protocol
/// diagram.
///
/// [`Connection::closed`]: iroh::endpoint::Connection::closed
pub(crate) async fn respond_notes_sync(
    conn: &Connection,
    cipher: &FrameCipher,
    send: &mut SendStream,
    recv: &mut RecvStream,
    root: &Path,
) -> Result<()> {
    let framer = Framer::new(cipher, StreamKind::Notes);
    // REQUEST: meeting id then the initiator's state vector. Bounded by
    // FRAME_IO_TIMEOUT like every other frame read on this stream: this one is a
    // raw fixed-size read rather than a length-prefixed frame, so it needs its own
    // explicit bound to stay off the unbounded-await slowloris surface.
    let mut id_buf = [0u8; 16];
    tokio::time::timeout(FRAME_IO_TIMEOUT, recv.read_exact(&mut id_buf))
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                peer = %conn.remote_id(),
                timeout = ?FRAME_IO_TIMEOUT,
                "reading the notes-sync meeting id timed out"
            );
            Error::Protocol(format!(
                "reading notes-sync meeting id timed out after {FRAME_IO_TIMEOUT:?}"
            ))
        })?
        .map_err(|e| Error::Protocol(format!("reading meeting id: {e}")))?;
    let meeting_id = MeetingId(Uuid::from_bytes(id_buf));
    let init_sv = framer.read(recv).await?;

    let local_state = local_v1_state(root, meeting_id)?;
    let local_sv = state_vector_of(&local_state)?;

    // Reply with our state vector so the initiator can diff against it.
    framer.write(send, &local_sv).await?;

    // Apply the initiator's diff (we converge), then send the diff it is missing.
    let diff_for_us = framer.read(recv).await?;
    apply_inbound(root, meeting_id, &diff_for_us)?;

    let diff_for_init = diff_against(&local_state, &init_sv)?;
    framer.write(send, &diff_for_init).await?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing notes-sync send: {e}")))?;

    // Hold the connection open until the initiator finishes reading our diff and
    // closes; returning here would let the router drop and abort the stream.
    // Bounded by RESPONDER_CLOSE_TIMEOUT so an initiator that never closes (a
    // stalled or hostile peer) cannot pin this task forever.
    tokio::time::timeout(RESPONDER_CLOSE_TIMEOUT, conn.closed())
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                peer = %conn.remote_id(),
                timeout = ?RESPONDER_CLOSE_TIMEOUT,
                "notes-sync responder timed out waiting for the initiator to close"
            );
            Error::Protocol(format!(
                "notes-sync responder wait for close timed out after {RESPONDER_CLOSE_TIMEOUT:?}"
            ))
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use yrs::updates::decoder::Decode;

    #[test]
    fn stream_kind_tags_are_the_wire_contract() {
        // The tag byte is the wire contract: 1/2/3/4 are fixed and append-only, so
        // this test fails if a future change reorders or renumbers a variant.
        assert_eq!(StreamKind::from_tag(1).unwrap(), StreamKind::Notes);
        assert_eq!(StreamKind::from_tag(2).unwrap(), StreamKind::Media);
        assert_eq!(StreamKind::from_tag(3).unwrap(), StreamKind::Discovery);
        assert_eq!(StreamKind::from_tag(4).unwrap(), StreamKind::Artifacts);
        // An unknown tag (an old peer seeing a future variant, or garbage) is a
        // protocol error, never a silent mis-dispatch.
        assert!(matches!(StreamKind::from_tag(0), Err(Error::Protocol(_))));
        assert!(matches!(StreamKind::from_tag(5), Err(Error::Protocol(_))));
    }

    #[test]
    fn empty_state_has_empty_state_vector() {
        // The empty-document v1 state encodes a state vector with no client
        // entries, so a diff against it is the whole of any non-empty document.
        let sv = state_vector_of(&empty_v1_state()).expect("encode sv");
        let decoded = yrs::StateVector::decode_v1(&sv).expect("decode sv");
        assert!(
            decoded.is_empty(),
            "empty doc must have an empty state vector"
        );
    }

    #[test]
    fn diff_against_empty_sv_is_whole_state() {
        // A document's diff against the empty state vector equals re-applying its
        // whole state: the operand for a peer that has never seen the meeting.
        let doc = notes_crdt::ydoc::json_to_ydoc(&serde_json::json!({
            "type": "doc",
            "content": [{ "type": "paragraph",
                "content": [{ "type": "text", "text": "hi" }] }]
        }));
        let v1 = notes_crdt::ydoc::encode_state_v1(&doc);
        let empty_sv = state_vector_of(&empty_v1_state()).expect("empty sv");
        let diff = diff_against(&v1, &empty_sv).expect("diff");

        let target = notes_crdt::ydoc::new_ydoc();
        notes_crdt::ydoc::apply_update_v1(&target, &diff).expect("apply diff");
        assert_eq!(
            notes_crdt::ydoc::ydoc_to_json(&target),
            notes_crdt::ydoc::ydoc_to_json(&doc),
            "diff against empty sv must reconstruct the whole document"
        );
    }

    #[test]
    fn diff_against_own_sv_is_empty() {
        // A document reconciled against its own state vector owes nothing.
        let doc = notes_crdt::ydoc::json_to_ydoc(&serde_json::json!({
            "type": "doc",
            "content": [{ "type": "paragraph",
                "content": [{ "type": "text", "text": "x" }] }]
        }));
        let v1 = notes_crdt::ydoc::encode_state_v1(&doc);
        let sv = state_vector_of(&v1).expect("sv");
        let diff = diff_against(&v1, &sv).expect("diff");

        // Applying the (empty) diff is a no-op merge.
        let target =
            notes_crdt::ydoc::decode_ydoc(&notes_crdt::ydoc::encode_ydoc(&doc)).expect("decode");
        let before = notes_crdt::ydoc::ydoc_to_json(&target);
        notes_crdt::ydoc::apply_update_v1(&target, &diff).expect("apply empty diff");
        assert_eq!(
            notes_crdt::ydoc::ydoc_to_json(&target),
            before,
            "diff against own sv must be a no-op"
        );
    }

    #[test]
    fn garbage_update_bytes_are_a_protocol_error_not_a_panic() {
        // Garbage in a frame body (a peer sending non-lib0 bytes) must surface as
        // an Error::Protocol from the decode helpers, never a panic.
        let garbage = [0xFFu8, 0x00, 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x42];
        assert!(matches!(state_vector_of(&garbage), Err(Error::Protocol(_))));
        assert!(matches!(is_noop_update(&garbage), Err(Error::Protocol(_))));
        assert!(matches!(
            diff_against(&garbage, &garbage),
            Err(Error::Protocol(_))
        ));
    }
}
