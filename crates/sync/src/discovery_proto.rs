//! Meeting-list + processing-lifecycle discovery ([`StreamKind::Discovery`]).
//!
//! Each side of a paired connection advertises the `(MeetingId,
//! ProcessingLifecycle)` of every meeting it holds; both learn the other's
//! meeting list AND each meeting's host-authoritative processing state. This
//! completes the deferred meeting-list discovery (a device learns *which*
//! meetings a peer has, not only notes for meetings it already knows) and carries
//! the processing lifecycle the capture-but-unprocessed workstream consumes
//! (`planning/DESIGN_processing-lifecycle.md` §7).
//!
//! # Wire protocol
//!
//! One exchange runs over a single bidirectional QUIC stream tagged
//! [`StreamKind::Discovery`], driven by the initiator. A single length-prefixed
//! frame ([`crate::frame`]) in each direction carries the JSON-encoded entry
//! list:
//!
//! ```text
//! initiator                                responder
//! ---------                                ---------
//! ENTRIES  ->  json([{meeting_id, processing}, …]), finish
//!                                    read ENTRIES
//!                               <-  ENTRIES  json([…]), finish
//! read ENTRIES
//! close connection ----------------> conn.closed() resolves; responder returns
//! ```
//!
//! Each side returns the PEER's entries; the caller emits them on the
//! [`SyncEngine`](crate::SyncEngine) lifecycle-event surface so a consumer in a
//! crate that depends on both `sync` and `persistence` (ipc-bridge / headless)
//! can persist them via `persistence::apply_processing_lifecycle`. `sync` itself
//! never writes `metadata.json` (it has no `persistence` dependency edge) — it
//! only *reads* the local `processing` to advertise, and emits the received ones.

use std::path::Path;

use iroh::endpoint::{Connection, RecvStream, SendStream};
use minutist_common::{MeetingId, MeetingMeta, ProcessingLifecycle};
use serde::{Deserialize, Serialize};

use crate::frame::{read_frame, write_frame};
use crate::notes_proto::StreamKind;
use crate::timeouts::RESPONDER_CLOSE_TIMEOUT;
use crate::{Error, Result};

/// One meeting's identity + its host-authoritative processing state, as carried
/// on a [`StreamKind::Discovery`] stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryEntry {
    /// The meeting id.
    pub meeting_id: MeetingId,
    /// The meeting's processing-lifecycle state, read from its `metadata.json`.
    pub processing: ProcessingLifecycle,
}

/// Enumerate the meeting ids this device holds on disk — delegates to the canonical
/// [`notes_crdt::folder::list_meeting_ids`] (the scan lives beside the meeting-folder
/// layout it reads). Used by
/// [`SyncEngine::local_meetings`](crate::SyncEngine::local_meetings) and the
/// discovery exchange.
pub(crate) fn list_meeting_ids(root: &Path) -> Vec<MeetingId> {
    notes_crdt::folder::list_meeting_ids(root)
}

/// Read a meeting's `processing` from its `metadata.json`, defaulting to
/// [`ProcessingLifecycle::Local`] when the file is absent or unparseable — the
/// same conservative default the inbound placeholder uses (§7 Q4: a meeting we
/// cannot classify is treated as local, never guessed adoptable). `sync` reads
/// `metadata.json` only to learn the local state to advertise; the authoritative
/// writer is `persistence`. Shared with [`crate::artifacts_proto`], which maps a
/// local `Processed` to the producer authority it stamps on its artifact manifest.
pub(crate) fn read_local_processing(root: &Path, id: MeetingId) -> ProcessingLifecycle {
    let path = root.join(id.0.to_string()).join("metadata.json");
    match std::fs::read(&path) {
        Ok(bytes) => match serde_json::from_slice::<MeetingMeta>(&bytes) {
            Ok(meta) => meta.processing,
            Err(error) => {
                // A corrupt/truncated metadata.json advertises Local (§7 Q4 —
                // never guess an adoptable state), but surface it: every other
                // metadata reader treats a parse failure as an error, so a silent
                // Local here would hide a corrupt local file.
                tracing::warn!(
                    target: "sync",
                    meeting_id = %id.0,
                    %error,
                    "discovery: metadata.json parse failed; advertising Local"
                );
                ProcessingLifecycle::default()
            }
        },
        Err(_) => ProcessingLifecycle::default(),
    }
}

/// The local entries to advertise: every meeting on disk paired with its
/// processing state.
fn local_entries(root: &Path) -> Vec<DiscoveryEntry> {
    list_meeting_ids(root)
        .into_iter()
        .map(|meeting_id| DiscoveryEntry {
            processing: read_local_processing(root, meeting_id),
            meeting_id,
        })
        .collect()
}

/// Encode an entry list as one JSON frame body.
fn encode(entries: &[DiscoveryEntry]) -> Result<Vec<u8>> {
    serde_json::to_vec(entries)
        .map_err(|e| Error::Protocol(format!("encoding discovery entries: {e}")))
}

/// Decode an entry list from a JSON frame body.
fn decode(bytes: &[u8]) -> Result<Vec<DiscoveryEntry>> {
    serde_json::from_slice(bytes)
        .map_err(|e| Error::Protocol(format!("decoding discovery entries: {e}")))
}

/// Run the *initiator* (dialling) side of a discovery exchange over `conn`: open
/// a bi stream, tag it [`StreamKind::Discovery`], advertise our entries, then
/// read the peer's. Returns the PEER's entries (the caller emits them). The
/// caller closes `conn` after this returns — the initiator is the last reader.
pub async fn initiate_discovery(conn: &Connection, root: &Path) -> Result<Vec<DiscoveryEntry>> {
    let ours = local_entries(root);

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| Error::Protocol(format!("opening discovery bi stream: {e}")))?;

    send.write_all(&[StreamKind::Discovery as u8])
        .await
        .map_err(|e| Error::Protocol(format!("writing discovery stream tag: {e}")))?;
    write_frame(&mut send, &encode(&ours)?).await?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing discovery send: {e}")))?;

    let theirs = decode(&read_frame(&mut recv).await?)?;
    Ok(theirs)
}

/// Run the *responder* (accepting) side over a bi stream the accept hook has
/// accepted and whose leading [`StreamKind`] tag it has consumed. Reads the
/// initiator's entries, replies with ours, and parks on [`Connection::closed`]
/// so the router does not drop the connection (aborting our reply) before the
/// initiator has read it. Returns the initiator's entries (the caller emits).
///
/// [`Connection::closed`]: iroh::endpoint::Connection::closed
pub async fn respond_discovery(
    conn: &Connection,
    send: &mut SendStream,
    recv: &mut RecvStream,
    root: &Path,
) -> Result<Vec<DiscoveryEntry>> {
    let theirs = decode(&read_frame(recv).await?)?;

    let ours = local_entries(root);
    write_frame(send, &encode(&ours)?).await?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing discovery send: {e}")))?;

    // Bounded by RESPONDER_CLOSE_TIMEOUT, mirroring the notes/media responders —
    // a stalled or hostile initiator that never closes cannot pin this task
    // forever.
    tokio::time::timeout(RESPONDER_CLOSE_TIMEOUT, conn.closed())
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                peer = %conn.remote_id(),
                timeout = ?RESPONDER_CLOSE_TIMEOUT,
                "discovery responder timed out waiting for the initiator to close"
            );
            Error::Protocol(format!(
                "discovery responder wait for close timed out after {RESPONDER_CLOSE_TIMEOUT:?}"
            ))
        })?;
    Ok(theirs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::{HostRef, ProcessingClaim};

    fn entry(processing: ProcessingLifecycle) -> DiscoveryEntry {
        DiscoveryEntry {
            meeting_id: MeetingId::new(),
            processing,
        }
    }

    #[test]
    fn entries_round_trip_through_json_frame() {
        let entries = vec![
            entry(ProcessingLifecycle::Local),
            entry(ProcessingLifecycle::PendingProcessing),
            entry(ProcessingLifecycle::Claimed {
                claim: ProcessingClaim {
                    host: HostRef("endpoint-a".into()),
                    claimed_at: "2026-06-27T10:00:00Z".into(),
                    lease_expires_at: "2026-06-27T10:30:00Z".into(),
                },
            }),
            entry(ProcessingLifecycle::Processed {
                processed_by: HostRef("endpoint-a".into()),
                at: "2026-06-27T10:25:00Z".into(),
            }),
        ];
        let decoded = decode(&encode(&entries).expect("encode")).expect("decode");
        assert_eq!(decoded, entries);
    }

    #[test]
    fn decode_rejects_garbage_as_protocol_error() {
        assert!(matches!(decode(b"not json"), Err(Error::Protocol(_))));
    }

    // The UUID-dir scan itself is tested canonically in `notes_crdt::folder`
    // (`list_meeting_ids_finds_uuid_dirs_and_skips_others`); this module only
    // delegates to it.

    #[test]
    fn read_local_processing_defaults_to_local_when_metadata_absent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let id = MeetingId::new();
        std::fs::create_dir(tmp.path().join(id.0.to_string())).expect("mk dir");
        // No metadata.json → Local (never guesses an adoptable state).
        assert_eq!(
            read_local_processing(tmp.path(), id),
            ProcessingLifecycle::Local
        );
    }
}
