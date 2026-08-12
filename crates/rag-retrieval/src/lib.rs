//! `rag-retrieval` — retrieval-augmented context for the meeting agent.
//!
//! # Why this exists
//!
//! On the integrated-GPU tier (the shipped target), feeding a whole document or
//! transcript to the LLM is infeasible: prefill is quadratic and slow (~10 min
//! for 20k tokens on the AMD 890M — SP-LIVE E5). So large attachments and long
//! transcripts cannot be pinned in context. The escape is **retrieval**: chunk
//! the source once at attach time (a cheap encoder pass — SP-LIVE E6 embedded a
//! 150k-char doc set in ~32 s), and at query time feed the LLM ONLY the few
//! relevant chunks. See `planning/RAG_RETRIEVAL_PLAN.md`.
//!
//! # What this crate is (and isn't)
//!
//! PURE retrieval logic — chunking ([`chunk_text`]) and cosine ranking
//! ([`rank_top_k`], reusing `common::voiceprint_math::cosine_unit`; embeddings
//! are L2-normalised, so cosine similarity is the plain dot product), driven
//! through the [`Embedder`] seam (defined in `common` alongside
//! `Summariser`/`DocVlm`, and re-exported here for convenience). It depends
//! ONLY on `common`; it does NOT load models or pull in `llama-cpp-2`.
//!
//! [`chunk_text`] produces newline-aligned, char-boundary-safe windows with
//! overlap (≈256-token chunks). [`RagChunk`]/[`DocType`] are the
//! pre-persistence chunk value (attachment or transcript text); `persistence`
//! assigns the durable row and identity columns when it stores them.
//!
//! The concrete embedder (BGE-M3 by default) is provided by the `embedder`
//! crate against `common::Embedder`; `rag-retrieval` itself never depends on
//! it, keeping the retrieval logic model-agnostic and llama-free. `ipc-bridge`
//! constructs and holds the concrete embedder, and owns the RAG write path
//! (its `rag_index` module): chunks are produced at attachment-convert time
//! (skipped when the content hash is already indexed) and at every
//! transcript-finalise point, plus incrementally during a live recording so
//! earlier turns become retrievable before the meeting stops. Chunks are
//! embedded via the held [`Embedder`] and persisted per-meeting (libsql +
//! FTS5) through `persistence`'s `RagStore`, which records the embedder's
//! `model_id` alongside each chunk. RAG indexing is a rebuildable cache:
//! failures there are logged and swallowed rather than failing attachment
//! conversion or the transcript-finalise flow. The index is consumed by the
//! chat agent's `retrieve_chunks` tool (`agent-tools`) and by the live
//! agent's per-refresh retrieval in `ipc-bridge`.

mod chunk;
mod score;

pub use chunk::{chunk_text, Chunk};
pub use minutist_common::Embedder;
pub use score::{cosine_unit, rank_top_k, rrf_fuse, unit_normalise};

use serde::{Deserialize, Serialize};

/// Which source document a chunk came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocType {
    /// An attachment (doc-convert markdown).
    Attachment,
    /// The meeting transcript.
    Transcript,
}

/// An embeddable chunk of a meeting's source material.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RagChunk {
    /// Whether this chunk is attachment or transcript text.
    pub doc_type: DocType,
    /// Identifies the source document — the attachment content-hash, or
    /// `"transcript"` for the meeting transcript.
    pub source_id: String,
    /// The chunk text (what gets embedded and, when retrieved, fed to the LLM).
    pub text: String,
    /// Byte offset of the chunk within its source document.
    pub byte_offset: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctype_serialises_snake_case() {
        assert_eq!(
            serde_json::to_string(&DocType::Attachment).unwrap(),
            "\"attachment\""
        );
        assert_eq!(
            serde_json::to_string(&DocType::Transcript).unwrap(),
            "\"transcript\""
        );
    }

    #[test]
    fn rag_chunk_round_trips() {
        let c = RagChunk {
            doc_type: DocType::Transcript,
            source_id: "transcript".into(),
            text: "Alice: let's ship it.".into(),
            byte_offset: 42,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: RagChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
