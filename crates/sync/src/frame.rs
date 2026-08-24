//! Length-prefixed, authenticated framing shared by the sync protocols.
//!
//! Every variable-length field on a sync stream is a frame: a `u32` big-endian
//! byte length followed by that many bytes. The declared length is validated
//! against [`MAX_FRAME`] before any buffer is allocated, so a hostile or buggy
//! peer cannot trigger a multi-gigabyte allocation.
//!
//! The body is sealed with XChaCha20-Poly1305 under the account content key
//! ([`crate::content_key`]), so this one chokepoint gives every protocol payload
//! confidentiality and integrity against a peer that passes the ed25519
//! membership check but holds no content key
//! (`planning/DESIGN_sync-encryption.md` §4). A `Framer` binds a cipher to one
//! stream's [`StreamKind`] tag for the life of an exchange, which is what stops
//! a frame lifted from one protocol being replayed into another and means the
//! tag cannot be got wrong at one of several call sites in the same function.
//!
//! Blob payloads travel over the blobs ALPN, not a frame, so they are not sealed
//! here. They do not need to be: a peer can only fetch a blob it can name, and
//! blob hashes reach a peer only inside a manifest, which is a frame. See
//! `DESIGN_sync-encryption.md` §4.1, and treat a blob hash on any unsealed path
//! as a confidentiality bug.

use iroh::endpoint::{RecvStream, SendStream};

use crate::content_key::{FrameCipher, NONCE_LEN, SEAL_OVERHEAD};
use crate::notes_proto::StreamKind;
use crate::timeouts::FRAME_IO_TIMEOUT;
use crate::{Error, Result};

/// Upper bound on a single frame's plaintext, in bytes. A whole-document Yjs
/// update or state vector, and a media manifest, are far smaller than this; the
/// cap bounds a hostile/buggy peer's allocation. Blob payloads travel over the
/// blobs ALPN, not a frame, so this cap does not affect media size.
///
/// The cap is on the plaintext, so the wire length a peer may declare is this
/// plus the seal's nonce and tag (`MAX_SEALED_FRAME`).
pub(crate) const MAX_FRAME: usize = 8 * 1024 * 1024;

/// Upper bound on the sealed body actually on the wire: [`MAX_FRAME`] of
/// plaintext plus the nonce and tag the seal adds.
pub(crate) const MAX_SEALED_FRAME: usize = MAX_FRAME + SEAL_OVERHEAD;

/// Validate a frame's big-endian `u32` length prefix, returning the length as a
/// `usize` to allocate. Both bounds are checked before anything is allocated.
///
/// Over [`MAX_SEALED_FRAME`] is [`Error::Protocol`]: a hostile or buggy peer
/// trying to make us allocate. Under [`SEAL_OVERHEAD`] is
/// [`Error::Unauthenticated`]: too short to be a sealed frame at all, so it
/// cannot authenticate, and saying so here means `open` is only ever handed a
/// body that could plausibly be one.
pub(crate) fn checked_frame_len(len_buf: [u8; 4]) -> Result<usize> {
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_SEALED_FRAME {
        return Err(Error::Protocol(format!(
            "frame length {len} exceeds cap {MAX_SEALED_FRAME}"
        )));
    }
    if len < SEAL_OVERHEAD {
        return Err(Error::Unauthenticated(format!(
            "sealed frame is {len} bytes, below the {SEAL_OVERHEAD}-byte minimum"
        )));
    }
    Ok(len)
}

/// Reads and writes the frames of one stream, sealing each under the account
/// content key with the stream's [`StreamKind`] tag as additional data.
///
/// Built once per exchange and passed down, so every frame on a stream is bound
/// to that stream's protocol by construction.
pub(crate) struct Framer<'a> {
    cipher: &'a FrameCipher,
    aad: u8,
}

impl<'a> Framer<'a> {
    /// Bind `cipher` to `kind`'s tag for the life of one exchange.
    pub(crate) fn new(cipher: &'a FrameCipher, kind: StreamKind) -> Self {
        Self {
            cipher,
            aad: kind as u8,
        }
    }

    /// Read one frame from `recv` and open it: a `u32` big-endian length, that
    /// many sealed bytes, then the AEAD open. Rejects a length over
    /// [`MAX_SEALED_FRAME`] before allocating.
    ///
    /// Bounded by [`FRAME_IO_TIMEOUT`]: a peer that stops writing mid-frame (the
    /// length prefix or the body) cannot pin the stream, or the task behind it,
    /// indefinitely. Applies on both the initiator and responder side, since
    /// both read the peer's frames this way.
    ///
    /// An open failure is [`Error::Unauthenticated`], not [`Error::Protocol`]:
    /// the peer holds a different content key, or the bytes were altered, or the
    /// frame belongs to another protocol.
    pub(crate) async fn read(&self, recv: &mut RecvStream) -> Result<Vec<u8>> {
        let (nonce, body) = tokio::time::timeout(FRAME_IO_TIMEOUT, read_sealed(recv))
            .await
            .unwrap_or_else(|_| {
                tracing::warn!(
                    target: "sync",
                    timeout = ?FRAME_IO_TIMEOUT,
                    "reading a sync frame timed out; dropping connection"
                );
                Err(Error::Protocol(format!(
                    "reading frame timed out after {FRAME_IO_TIMEOUT:?}"
                )))
            })?;
        self.cipher.open(self.aad, &nonce, body)
    }

    /// Seal `bytes` and write it to `send` as a `u32` big-endian length then the
    /// sealed body. Rejects a plaintext over [`MAX_FRAME`] so both directions
    /// share the cap.
    ///
    /// Bounded by [`FRAME_IO_TIMEOUT`], mirroring [`Self::read`]: a peer that
    /// stops reading (so the QUIC send buffer never drains) cannot pin the writer
    /// forever.
    pub(crate) async fn write(&self, send: &mut SendStream, bytes: &[u8]) -> Result<()> {
        if bytes.len() > MAX_FRAME {
            return Err(Error::Protocol(format!(
                "frame body {} exceeds cap {MAX_FRAME}",
                bytes.len()
            )));
        }
        let sealed = self.cipher.seal(self.aad, bytes)?;
        tokio::time::timeout(FRAME_IO_TIMEOUT, write_sealed(send, &sealed))
            .await
            .unwrap_or_else(|_| {
                tracing::warn!(
                    target: "sync",
                    timeout = ?FRAME_IO_TIMEOUT,
                    "writing a sync frame timed out; dropping connection"
                );
                Err(Error::Protocol(format!(
                    "writing frame timed out after {FRAME_IO_TIMEOUT:?}"
                )))
            })
    }
}

/// Read one wire frame, returning its nonce and its still-sealed body.
///
/// Splitting the nonce out here rather than after the fact lets
/// [`FrameCipher::open`] decrypt the body buffer in place, with no second
/// allocation or copy of a frame that can reach [`MAX_FRAME`].
async fn read_sealed(recv: &mut RecvStream) -> Result<([u8; NONCE_LEN], Vec<u8>)> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf)
        .await
        .map_err(|e| Error::Protocol(format!("reading frame length: {e}")))?;
    let len = checked_frame_len(len_buf)?;
    let mut nonce = [0u8; NONCE_LEN];
    recv.read_exact(&mut nonce)
        .await
        .map_err(|e| Error::Protocol(format!("reading frame nonce: {e}")))?;
    let mut body = vec![0u8; len - NONCE_LEN];
    recv.read_exact(&mut body)
        .await
        .map_err(|e| Error::Protocol(format!("reading {len}-byte frame body: {e}")))?;
    Ok((nonce, body))
}

async fn write_sealed(send: &mut SendStream, sealed: &[u8]) -> Result<()> {
    let len = sealed.len() as u32;
    send.write_all(&len.to_be_bytes())
        .await
        .map_err(|e| Error::Protocol(format!("writing frame length: {e}")))?;
    send.write_all(sealed)
        .await
        .map_err(|e| Error::Protocol(format!("writing frame body: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_frame_length_is_rejected_before_allocating() {
        // A `u32` length prefix beyond the sealed cap must be rejected as a
        // protocol error, not used to allocate a multi-gigabyte buffer.
        // `u32::MAX` is the worst case a hostile peer can put on the wire.
        assert!(matches!(
            checked_frame_len(u32::MAX.to_be_bytes()),
            Err(Error::Protocol(_))
        ));
        // The exact cap is fine; one over it is not. The wire length is the
        // plaintext cap plus the seal's nonce and tag, so a full-size frame is
        // legal and must not be rejected as oversized.
        assert!(checked_frame_len((MAX_SEALED_FRAME as u32).to_be_bytes()).is_ok());
        assert!(matches!(
            checked_frame_len((MAX_SEALED_FRAME as u32 + 1).to_be_bytes()),
            Err(Error::Protocol(_))
        ));
    }

    #[test]
    fn a_length_below_the_seal_overhead_cannot_authenticate() {
        // Too short to hold a nonce and a tag, so it is not a sealed frame and is
        // rejected as unauthenticated rather than read and handed to `open`.
        assert!(matches!(
            checked_frame_len(((SEAL_OVERHEAD - 1) as u32).to_be_bytes()),
            Err(Error::Unauthenticated(_))
        ));
        // The exact overhead is a legal empty frame.
        assert!(checked_frame_len((SEAL_OVERHEAD as u32).to_be_bytes()).is_ok());
    }
}
