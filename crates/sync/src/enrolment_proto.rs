//! The account content-key transfer ([`StreamKind::Enrolment`], tag 5).
//!
//! One frame, one direction: a device that holds the account content key and
//! whose user has confirmed the far side sends it the key. The receiver stores
//! it only if its own user independently confirmed the sender. Both halves of
//! that are load-bearing, and the two confirmations are separate records on
//! separate devices — neither side takes the other's word for having asked.
//!
//! ```text
//! holder (initiator)                        joiner (responder)
//! ------------------                        ------------------
//! user confirmed joiner?  no -> never dial
//!         yes
//! OFFER  ->  32 raw key bytes
//!                                     user confirmed holder?  no -> refuse
//!                                             yes
//!                                     persist key, swap cipher   <- all state
//!                                <-  ACCEPTED (1) | REFUSED (0)      lands first
//! ```
//!
//! # Why this stream is not sealed
//!
//! Every other stream kind is sealed under the account content key. This one
//! cannot be: it exists to deliver that key to a device which by definition does
//! not hold it yet. The exception is deliberate and is protected by three other
//! things — QUIC/TLS mutual authentication to both ed25519 identity keys, the
//! paired-peer membership check the accept hook already applies, and a locally
//! recorded user confirmation for this exact peer on BOTH sides.
//!
//! The confirmation is what makes this safe rather than merely authenticated.
//! Membership alone is what the account directory controls, and trusting it is
//! the hole this whole design closes: a compromised directory can publish an
//! endpoint that passes the membership check, so membership must never be
//! sufficient to receive or hand over the key.
//!
//! # Why the receiver checks too
//!
//! It would be tempting to let the holder decide alone, since it is the one
//! giving something away. That is wrong in a way worth stating: a device that
//! adopts a key offered by a rogue then SEALS ITS OWN OUTBOUND FRAMES under that
//! key, so the rogue reads everything it subsequently sends. Receiving a key is
//! not the safe direction, and the responder refuses without its own
//! confirmation.

use iroh::endpoint::{Connection, RecvStream, SendStream};

use crate::content_key::ContentKey;
use crate::enrolment::EnrolmentStore;
use crate::notes_proto::StreamKind;
use crate::timeouts::{FRAME_IO_TIMEOUT, RESPONDER_CLOSE_TIMEOUT};
use crate::{Error, Result};

/// The responder's verdict byte.
const ACCEPTED: u8 = 1;
const REFUSED: u8 = 0;

/// Raw length of the key on the wire. Fixed, so there is no length prefix to
/// mis-parse and nothing a peer can declare.
const KEY_LEN: usize = 32;

/// Send the account content key to `peer`, which this device's user has
/// confirmed.
///
/// The caller must have checked the confirmation; this asserts it again rather
/// than trusting the call site, because handing out the key is the one
/// irreversible act in the protocol and a missing check here has no second line
/// of defence on this side.
pub(crate) async fn offer_key(
    conn: &Connection,
    store: &EnrolmentStore,
    key: &ContentKey,
) -> Result<()> {
    let peer = conn.remote_id().to_string();
    if !store.is_confirmed(&peer) {
        return Err(Error::Unauthenticated(format!(
            "refusing to send the content key to {peer}: not user-confirmed"
        )));
    }

    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| Error::Protocol(format!("opening enrolment bi stream: {e}")))?;

    send.write_all(&[StreamKind::Enrolment as u8])
        .await
        .map_err(|e| Error::Protocol(format!("writing enrolment stream tag: {e}")))?;
    send.write_all(key.as_bytes())
        .await
        .map_err(|e| Error::Protocol(format!("writing the content key: {e}")))?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing the enrolment stream: {e}")))?;

    // The peer's verdict. Read rather than assumed so a refusal is visible here
    // and can be surfaced, instead of the holder believing it enrolled a device
    // that rejected it.
    let mut verdict = [0u8; 1];
    tokio::time::timeout(FRAME_IO_TIMEOUT, recv.read_exact(&mut verdict))
        .await
        .map_err(|_| {
            Error::Protocol(format!(
                "waiting for the enrolment verdict timed out after {FRAME_IO_TIMEOUT:?}"
            ))
        })?
        .map_err(|e| Error::Protocol(format!("reading the enrolment verdict: {e}")))?;

    match verdict[0] {
        ACCEPTED => {
            tracing::info!(target: "sync", %peer, "peer adopted the account content key");
            Ok(())
        }
        REFUSED => Err(Error::Unauthenticated(format!(
            "peer {peer} refused the content key: it has not confirmed this device"
        ))),
        other => Err(Error::Protocol(format!(
            "unknown enrolment verdict byte {other} from {peer}"
        ))),
    }
}

/// Read an offered content key, returning it only if this device's user has
/// confirmed the sender.
///
/// Deliberately does NOT write the verdict or touch any state: the caller
/// persists the key and updates its in-memory cipher FIRST, then calls
/// [`send_verdict`]. That ordering matters. The initiator treats `ACCEPTED` as
/// "the peer is enrolled" and returns immediately, so anything this device still
/// has to do afterwards is a window where the initiator believes an enrolment
/// that has not finished landing.
pub(crate) async fn read_offered_key(
    conn: &Connection,
    recv: &mut RecvStream,
    store: &EnrolmentStore,
) -> Result<Option<[u8; 32]>> {
    let peer = conn.remote_id().to_string();

    let mut bytes = [0u8; KEY_LEN];
    tokio::time::timeout(FRAME_IO_TIMEOUT, recv.read_exact(&mut bytes))
        .await
        .map_err(|_| {
            Error::Protocol(format!(
                "reading the offered content key timed out after {FRAME_IO_TIMEOUT:?}"
            ))
        })?
        .map_err(|e| Error::Protocol(format!("reading the offered content key: {e}")))?;

    // The check that matters. An unconfirmed sender may well be a legitimately
    // paired peer as far as the directory is concerned; that is exactly the
    // condition this refuses on.
    if !store.is_confirmed(&peer) {
        tracing::warn!(
            target: "sync",
            %peer,
            "refused an offered content key: this device has not confirmed that peer"
        );
        return Ok(None);
    }
    Ok(Some(bytes))
}

/// Tell the initiator whether the key was adopted, then hold the connection open
/// until it has read that.
///
/// Called only after the adopting device has finished persisting the key and
/// swapping its cipher, so `ACCEPTED` means "already enrolled", not "about to
/// be".
pub(crate) async fn send_verdict(
    conn: &Connection,
    send: &mut SendStream,
    accepted: bool,
) -> Result<()> {
    let byte = if accepted { ACCEPTED } else { REFUSED };
    send.write_all(&[byte])
        .await
        .map_err(|e| Error::Protocol(format!("writing the enrolment verdict: {e}")))?;
    send.finish()
        .map_err(|e| Error::Protocol(format!("finishing the enrolment stream: {e}")))?;
    if accepted {
        tracing::info!(
            target: "sync",
            peer = %conn.remote_id(),
            "adopted the account content key from a confirmed peer"
        );
    }
    park_until_closed(conn).await
}

/// Hold the connection open until the initiator has read our verdict and closed.
///
/// Returning before then lets the router drop the connection and abort the
/// stream, so the initiator sees "connection lost" instead of the verdict. Every
/// other responder in this crate parks the same way; bounded by
/// [`RESPONDER_CLOSE_TIMEOUT`] so a peer that never closes cannot pin the task.
async fn park_until_closed(conn: &Connection) -> Result<()> {
    tokio::time::timeout(RESPONDER_CLOSE_TIMEOUT, conn.closed())
        .await
        .map(|_| ())
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                peer = %conn.remote_id(),
                timeout = ?RESPONDER_CLOSE_TIMEOUT,
                "initiator never closed the enrolment connection"
            );
            Error::Protocol(format!(
                "enrolment initiator did not close within {RESPONDER_CLOSE_TIMEOUT:?}"
            ))
        })
}
