//! Little-endian `f32` ↔ BLOB helpers shared by the libsql stores.
//!
//! Embedding vectors (voiceprints, RAG) are stored as packed little-endian `f32`
//! BLOBs. Hoisted here so both [`crate::voiceprints`] and [`crate::rag`] use one
//! definition rather than duplicating the conversion.

/// Serialise an `f32` slice to a little-endian byte blob for SQLite storage.
pub(crate) fn f32_slice_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Deserialise a little-endian byte blob back to `Vec<f32>`.
///
/// Any trailing bytes that do not form a complete `f32` are silently discarded.
pub(crate) fn blob_to_f32_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
