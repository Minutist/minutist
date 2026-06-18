//! The Yjs notes-update exchange protocol.
//!
//! A small custom iroh protocol (ALPN [`SYNC_ALPN`]) reconciles one meeting's
//! Yjs CRDT document between two of a user's paired devices. The authoritative
//! local document is `persistence`'s `notes.ydoc`. The exchange is a
//! state-vector / minimal-diff reconciliation:
//!
//! - each side reads its local state as a lib0-**v1** whole-state update with
//!   [`persistence::NotesStore::read_ydoc_state`] (an absent `notes.ydoc` is the
//!   empty-document state),
//! - each side derives its [state vector] from that v1 state
//!   ([`yrs::encode_state_vector_from_update_v1`]) and sends it to the peer,
//! - the peer answers with the **minimal** update covering everything the sender
//!   has not yet seen ([`yrs::diff_updates_v1`] of the peer's local state against
//!   the sender's state vector),
//! - each side merges the inbound diff with
//!   [`persistence::NotesStore::apply_update`], which re-derives `notes.json` /
//!   `notes.md` (`persistence` owns projection-writing — sync never touches it).
//!
//! yrs merge is commutative and idempotent, so the exchange is order-independent
//! and re-running it is a no-op once both sides have converged.
//!
//! # Wire protocol
//!
//! One reconciliation runs over a single bidirectional QUIC stream and is driven
//! by the dialling side (the *initiator*); the accepting side is the
//! *responder*. The exchange strictly alternates so neither side blocks on a
//! read while the other also blocks — no deadlock on the one stream — and the
//! **initiator is the last reader**, so it holds the inbound diff in hand before
//! it closes the connection. The responder parks on [`Connection::closed`] until
//! that close arrives, so its applied write is never aborted by a premature
//! teardown:
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
//! Every variable-length field is a frame: a `u32` big-endian byte length
//! followed by that many bytes ([`read_frame`] / [`write_frame`]). The
//! `meeting_id` is a fixed 16-byte UUID (no prefix). A diff that carries no
//! changes (the peer had nothing the receiver lacked — the common
//! already-converged case) is recognised by [`is_noop_update`] and skipped
//! rather than written, so an up-to-date or empty meeting touches no disk.
//!
//! [state vector]: https://docs.rs/yrs/0.26.0/yrs/struct.StateVector.html
//! [`Connection::closed`]: iroh::endpoint::Connection::closed

use std::path::Path;

use iroh::endpoint::{Connection, RecvStream, SendStream};
use minutist_common::MeetingId;
use persistence::NotesStore;
use uuid::Uuid;
use yrs::updates::decoder::Decode;
use yrs::updates::encoder::Encode;
use yrs::Update;

use crate::{Error, Result};

/// ALPN for the notes-update protocol. Bumping the suffix is a wire break.
pub const SYNC_ALPN: &[u8] = b"minutist/sync/notes/1";

/// Upper bound on a single length-prefixed frame the protocol will buffer, in
/// bytes. A whole-document Yjs update or state vector for a meeting's notes is
/// far smaller than this; the cap bounds a hostile/buggy peer's allocation.
const MAX_FRAME: usize = 8 * 1024 * 1024;

/// Read one length-prefixed frame from `recv`: a `u32` big-endian length then
/// that many bytes. Rejects a length over [`MAX_FRAME`] before allocating.
async fn read_frame(recv: &mut RecvStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf)
        .await
        .map_err(|e| Error::Protocol(format!("reading frame length: {e}")))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME {
        return Err(Error::Protocol(format!(
            "frame length {len} exceeds cap {MAX_FRAME}"
        )));
    }
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf)
        .await
        .map_err(|e| Error::Protocol(format!("reading {len}-byte frame body: {e}")))?;
    Ok(buf)
}

/// Write one length-prefixed frame to `send`: a `u32` big-endian length then the
/// bytes. Rejects a body over [`MAX_FRAME`] so both directions share the cap.
async fn write_frame(send: &mut SendStream, bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_FRAME {
        return Err(Error::Protocol(format!(
            "frame body {} exceeds cap {MAX_FRAME}",
            bytes.len()
        )));
    }
    let len = bytes.len() as u32;
    send.write_all(&len.to_be_bytes())
        .await
        .map_err(|e| Error::Protocol(format!("writing frame length: {e}")))?;
    send.write_all(bytes)
        .await
        .map_err(|e| Error::Protocol(format!("writing frame body: {e}")))?;
    Ok(())
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

/// The lib0-v1 whole-state update of an empty Yjs document — the operand for a
/// meeting whose `notes.ydoc` does not exist yet.
fn empty_v1_state() -> Vec<u8> {
    // `yrs::Update::default().encode_v1()` is the empty update; deriving it from
    // an empty `StateVector` keeps this in lock-step with how `persistence`
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

/// Whether a lib0-v1 update carries no changes at all — no inserted blocks and
/// no deletions. Such a diff is the peer telling us "you are already up to date";
/// applying it would be a pure no-op merge.
fn is_noop_update(diff: &[u8]) -> Result<bool> {
    let update =
        Update::decode_v1(diff).map_err(|e| Error::Protocol(format!("decoding diff: {e}")))?;
    Ok(update.is_empty() && update.delete_set().is_empty())
}

/// Merge an inbound v1 diff into the meeting's `notes.ydoc` via `persistence`,
/// preserving the existing `notes.md` projection.
///
/// `persistence::NotesStore::apply_update` re-derives `notes.json` from the
/// merged doc but takes the markdown verbatim (rendering markdown needs the
/// editor's typed schema, which neither `sync` nor `persistence` models). We
/// therefore carry the meeting's current `notes.md` through unchanged: a CRDT
/// sync must not blank an existing markdown export, and the editor re-renders it
/// on next save.
///
/// A no-op diff (the peer had nothing we lacked) is skipped: there is no merge to
/// perform, and skipping avoids a `notes.ydoc` rewrite. Only a diff that actually
/// carries changes reaches `apply_update`.
///
/// Before that merge, the meeting folder is ensured on disk via
/// [`persistence::MeetingFolder::ensure`]: a brand-new meeting syncing to a device
/// that lacks its folder must not fail for want of the directory. `persistence`
/// owns the folder/metadata creation (it seeds a placeholder `metadata.json` the
/// authoritative one later overwrites); `sync` only triggers it. Ensuring runs
/// only for a change-carrying diff, so an already-converged or empty meeting still
/// touches no disk.
fn apply_inbound(root: &Path, meeting_id: MeetingId, diff: &[u8]) -> Result<()> {
    if is_noop_update(diff)? {
        return Ok(());
    }
    persistence::MeetingFolder::ensure(root, meeting_id)
        .map_err(|e| Error::Protocol(format!("ensuring inbound meeting folder: {e}")))?;
    let notes_md = match NotesStore::load(root, meeting_id) {
        Ok(Some(data)) => data.markdown,
        Ok(None) => String::new(),
        Err(e) => return Err(Error::Protocol(format!("loading notes.md: {e}"))),
    };
    NotesStore::apply_update(root, meeting_id, diff, &notes_md)
        .map_err(|e| Error::Protocol(format!("applying inbound update: {e}")))
}

/// Run the *initiator* (dialling) side of one notes reconciliation for
/// `meeting_id` over `conn`, against the meetings `root`.
///
/// Opens a bi stream, sends the REQUEST (meeting id + local state vector), reads
/// the peer's state vector, sends the DIFF the peer is missing (so the responder
/// converges), then reads the DIFF we are missing and applies it (so we
/// converge). The caller closes `conn` after this returns — the initiator is the
/// last reader, so closing then is safe. See the module wire-protocol diagram.
pub async fn initiate_notes_sync(
    conn: &Connection,
    root: &Path,
    meeting_id: MeetingId,
) -> Result<()> {
    let local_state = local_v1_state(root, meeting_id)?;
    let local_sv = state_vector_of(&local_state)?;

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| Error::Protocol(format!("opening notes-sync bi stream: {e}")))?;

    // REQUEST: meeting id (fixed 16 bytes) then our state vector.
    send.write_all(meeting_id.0.as_bytes())
        .await
        .map_err(|e| Error::Protocol(format!("writing meeting id: {e}")))?;
    write_frame(&mut send, &local_sv).await?;

    // Learn the peer's state vector, then send it the diff it is missing.
    let peer_sv = read_frame(&mut recv).await?;
    let diff_for_peer = diff_against(&local_state, &peer_sv)?;
    write_frame(&mut send, &diff_for_peer).await?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing notes-sync send: {e}")))?;

    // Read the diff we are missing and merge it last (initiator is last reader).
    let diff_for_us = read_frame(&mut recv).await?;
    apply_inbound(root, meeting_id, &diff_for_us)?;
    Ok(())
}

/// Run the *responder* (accepting) side of one notes reconciliation over `conn`,
/// against the meetings `root`.
///
/// Accepts the bi stream the initiator opened, reads the REQUEST (meeting id +
/// initiator state vector), replies with its own state vector, reads the
/// initiator's DIFF and applies it (so we converge), then sends the DIFF the
/// initiator is missing. Finally parks on [`Connection::closed`] so the
/// router does not drop the connection (aborting our last write) before the
/// initiator has read it. See the module wire-protocol diagram.
///
/// [`Connection::closed`]: iroh::endpoint::Connection::closed
pub async fn respond_notes_sync(conn: &Connection, root: &Path) -> Result<()> {
    let (mut send, mut recv) = conn
        .accept_bi()
        .await
        .map_err(|e| Error::Protocol(format!("accepting notes-sync bi stream: {e}")))?;

    // REQUEST: meeting id then the initiator's state vector.
    let mut id_buf = [0u8; 16];
    recv.read_exact(&mut id_buf)
        .await
        .map_err(|e| Error::Protocol(format!("reading meeting id: {e}")))?;
    let meeting_id = MeetingId(Uuid::from_bytes(id_buf));
    let init_sv = read_frame(&mut recv).await?;

    let local_state = local_v1_state(root, meeting_id)?;
    let local_sv = state_vector_of(&local_state)?;

    // Reply with our state vector so the initiator can diff against it.
    write_frame(&mut send, &local_sv).await?;

    // Apply the initiator's diff (we converge), then send the diff it is missing.
    let diff_for_us = read_frame(&mut recv).await?;
    apply_inbound(root, meeting_id, &diff_for_us)?;

    let diff_for_init = diff_against(&local_state, &init_sv)?;
    write_frame(&mut send, &diff_for_init).await?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing notes-sync send: {e}")))?;

    // Hold the connection open until the initiator finishes reading our diff and
    // closes; returning here would let the router drop and abort the stream.
    conn.closed().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use yrs::updates::decoder::Decode;

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
        let doc = persistence::ydoc::json_to_ydoc(&serde_json::json!({
            "type": "doc",
            "content": [{ "type": "paragraph",
                "content": [{ "type": "text", "text": "hi" }] }]
        }));
        let v1 = persistence::ydoc::encode_state_v1(&doc);
        let empty_sv = state_vector_of(&empty_v1_state()).expect("empty sv");
        let diff = diff_against(&v1, &empty_sv).expect("diff");

        let target = persistence::ydoc::new_ydoc();
        persistence::ydoc::apply_update_v1(&target, &diff).expect("apply diff");
        assert_eq!(
            persistence::ydoc::ydoc_to_json(&target),
            persistence::ydoc::ydoc_to_json(&doc),
            "diff against empty sv must reconstruct the whole document"
        );
    }

    #[test]
    fn diff_against_own_sv_is_empty() {
        // A document reconciled against its own state vector owes nothing.
        let doc = persistence::ydoc::json_to_ydoc(&serde_json::json!({
            "type": "doc",
            "content": [{ "type": "paragraph",
                "content": [{ "type": "text", "text": "x" }] }]
        }));
        let v1 = persistence::ydoc::encode_state_v1(&doc);
        let sv = state_vector_of(&v1).expect("sv");
        let diff = diff_against(&v1, &sv).expect("diff");

        // Applying the (empty) diff is a no-op merge.
        let target =
            persistence::ydoc::decode_ydoc(&persistence::ydoc::encode_ydoc(&doc)).expect("decode");
        let before = persistence::ydoc::ydoc_to_json(&target);
        persistence::ydoc::apply_update_v1(&target, &diff).expect("apply empty diff");
        assert_eq!(
            persistence::ydoc::ydoc_to_json(&target),
            before,
            "diff against own sv must be a no-op"
        );
    }
}
