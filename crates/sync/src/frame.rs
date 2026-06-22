//! Length-prefixed framing shared by the sync protocols.
//!
//! Every variable-length field on a sync stream is a frame: a `u32` big-endian
//! byte length followed by that many bytes. A frame's declared length is checked
//! against [`MAX_FRAME`] BEFORE any buffer is allocated, so a hostile or buggy
//! peer cannot trigger a multi-gigabyte allocation. The notes protocol
//! ([`crate::notes_proto`]) and the media protocol ([`crate::media_proto`]) both
//! frame their payloads this way.

use iroh::endpoint::{RecvStream, SendStream};

use crate::{Error, Result};

/// Upper bound on a single length-prefixed frame, in bytes. A whole-document Yjs
/// update or state vector, and a media manifest, are far smaller than this; the
/// cap bounds a hostile/buggy peer's allocation. Blob payloads themselves do NOT
/// transit a frame — they travel over the blobs ALPN — so this cap is unaffected
/// by media size.
pub const MAX_FRAME: usize = 8 * 1024 * 1024;

/// Validate a frame's big-endian `u32` length prefix against [`MAX_FRAME`],
/// returning the length as a `usize` to allocate. A length over the cap is an
/// [`Error::Protocol`] — the guard that bounds a hostile/buggy peer's allocation,
/// checked BEFORE any buffer is allocated.
pub fn checked_frame_len(len_buf: [u8; 4]) -> Result<usize> {
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME {
        return Err(Error::Protocol(format!(
            "frame length {len} exceeds cap {MAX_FRAME}"
        )));
    }
    Ok(len)
}

/// Read one length-prefixed frame from `recv`: a `u32` big-endian length then
/// that many bytes. Rejects a length over [`MAX_FRAME`] before allocating.
pub async fn read_frame(recv: &mut RecvStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf)
        .await
        .map_err(|e| Error::Protocol(format!("reading frame length: {e}")))?;
    let len = checked_frame_len(len_buf)?;
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf)
        .await
        .map_err(|e| Error::Protocol(format!("reading {len}-byte frame body: {e}")))?;
    Ok(buf)
}

/// Write one length-prefixed frame to `send`: a `u32` big-endian length then the
/// bytes. Rejects a body over [`MAX_FRAME`] so both directions share the cap.
pub async fn write_frame(send: &mut SendStream, bytes: &[u8]) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_frame_length_is_rejected_before_allocating() {
        // A `u32` length prefix beyond MAX_FRAME must be rejected as a protocol
        // error, not used to allocate a multi-gigabyte buffer. `u32::MAX` is the
        // worst case a hostile peer can put on the wire.
        assert!(matches!(
            checked_frame_len(u32::MAX.to_be_bytes()),
            Err(Error::Protocol(_))
        ));
        // The exact cap is fine; one over it is not.
        assert!(checked_frame_len((MAX_FRAME as u32).to_be_bytes()).is_ok());
        assert!(matches!(
            checked_frame_len((MAX_FRAME as u32 + 1).to_be_bytes()),
            Err(Error::Protocol(_))
        ));
    }
}
