//! `RagStore` — the per-meeting RAG retrieval cache held in `meeting.db`.
//!
//! Holds the retrieval index for ONE meeting: chunked attachment + transcript
//! text, their embedding vectors, and an FTS5 lexical index. The db lives at
//! `{meetings_dir}/{meeting_id}/meeting.db` and is a **rebuildable cache** —
//! re-chunk + re-embed the on-disk markdown / transcript to reconstruct it. One
//! db per meeting, so rows are not keyed by `meeting_id`. (`meeting.db` is the
//! intended future home for more per-meeting state; see
//! `planning/DESIGN_meeting_db.md`. Phase B occupies it with the RAG tables only.)
//!
//! # Hybrid retrieval — two legs, fused by the caller
//!
//! Retrieval is split into [`RagStore::retrieve_dense`] (brute-force cosine over
//! the stored vectors, reusing `common::voiceprint_math::cosine_unit`) and
//! [`RagStore::retrieve_lexical`] (FTS5 `bm25()`). The caller fuses the two ranked
//! lists (Reciprocal Rank Fusion). The fusion lives in the retrieval tool so this
//! crate takes **no `rag-retrieval` edge** — only `common` + libsql.
//!
//! # Idempotency
//!
//! [`RagStore::index_source`] replaces *all* chunks for a `source_id`
//! (delete-then-insert in one transaction), so re-indexing a changed source is
//! clean. For content-addressed sources (an attachment's `source_id` is its
//! content hash) the caller can skip re-embedding entirely via
//! [`RagStore::has_source`].
//!
//! # Async, no `block_on`
//!
//! Mirrors `index.rs` / `voiceprints.rs`: every public method is `async fn` over
//! an interior `Connection`; the crate never calls `block_on`. Multi-statement
//! mutations wrap their work in `BEGIN IMMEDIATE` / `COMMIT` (rolled back on error).

use std::path::Path;

use libsql::{Builder, Connection, Database};
use minutist_common::{
    voiceprint_math::{cmp_desc_finite_first, cosine_unit},
    AppResult,
};
use notes_crdt::MeetingFolder;

use crate::blob::{blob_to_f32_vec, f32_slice_to_blob};
use crate::error::Error;

/// Path to a meeting's RAG cache db, `{meetings_dir}/{meeting_id}/meeting.db`.
///
/// The single owner of the `meeting.db` filename + per-meeting layout — the write
/// path (`ipc-bridge::rag_index`) and the retrieval tool (`agent-tools`) both
/// resolve the path through here rather than hand-joining the literal.
pub fn meeting_db_path(
    meetings_dir: &Path,
    meeting_id: minutist_common::MeetingId,
) -> std::path::PathBuf {
    MeetingFolder::open(meetings_dir, meeting_id)
        .path()
        .join("meeting.db")
}

/// One chunk to index: the pre-persistence value plus its (L2-normalised)
/// embedding. The owning `source_id` / `doc_type` / `model_id` are passed once to
/// [`RagStore::index_source`], not repeated per chunk.
pub struct NewChunk<'a> {
    /// The chunk text (also fed to the LLM when this chunk is retrieved).
    pub text: &'a str,
    /// Byte offset of the chunk within its source document (provenance).
    pub byte_offset: u64,
    /// The chunk's embedding vector (MUST be L2-normalised — the `Embedder` contract).
    pub embedding: &'a [f32],
}

/// A chunk returned from a retrieval leg, carrying that leg's score.
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    /// Opaque chunk id (used by the caller to fuse the dense + lexical legs).
    pub chunk_id: String,
    /// `"attachment"` or `"transcript"`.
    pub doc_type: String,
    /// The source document id (attachment content hash, or `"transcript"`).
    pub source_id: String,
    /// The chunk text.
    pub text: String,
    /// Byte offset within the source document.
    pub byte_offset: u64,
    /// Dense cosine similarity (`retrieve_dense`) or the `bm25()` score
    /// (`retrieve_lexical`, lower = better). Rank-based fusion ignores the
    /// magnitude, so the two scales never mix.
    pub score: f32,
}

/// The per-meeting RAG cache.
pub struct RagStore {
    // Keep the `Database` alive for the connection lifetime.
    #[allow(dead_code)]
    db: Database,
    conn: Connection,
}

impl RagStore {
    /// Open (or create) the RAG cache at `db_path` and ensure its schema.
    ///
    /// Pass `":memory:"` for an in-memory database (tests); a filesystem path
    /// opens-or-creates a file-backed db. Returns an error on libsql failure; the
    /// caller treats RAG as unavailable for that meeting (never panics, never
    /// blocks recording).
    pub async fn open(db_path: impl AsRef<Path>) -> AppResult<Self> {
        Ok(Self::open_inner(db_path).await?)
    }

    async fn open_inner(db_path: impl AsRef<Path>) -> Result<Self, Error> {
        let db = Builder::new_local(db_path.as_ref()).build().await?;
        let conn = db.connect()?;
        // A brief busy timeout so an overlapping writer (the attach-index and the
        // post-stop transcript-index can race on one meeting.db) retries instead of
        // failing its BEGIN IMMEDIATE with SQLITE_BUSY. `query` (not `execute`): the
        // assignment form returns the new value as a row.
        let _ = conn.query("PRAGMA busy_timeout = 5000", ()).await?;
        create_schema(&conn).await?;
        Ok(Self { db, conn })
    }

    /// True if any chunk for `source_id` is already indexed. Lets the caller skip
    /// re-embedding an unchanged, content-addressed source.
    pub async fn has_source(&self, source_id: &str) -> AppResult<bool> {
        Ok(self.has_source_inner(source_id).await?)
    }

    async fn has_source_inner(&self, source_id: &str) -> Result<bool, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT 1 FROM rag_chunk WHERE source_id = ?1 LIMIT 1",
                libsql::params![source_id],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    }

    /// Replace all chunks for `source_id` with `chunks` (delete-then-insert in one
    /// transaction). `doc_type` is `"attachment"` or `"transcript"`; `model_id`
    /// records the embedder that produced the vectors (so a query only scores
    /// against vectors from the same model). Returns the number of chunks inserted.
    pub async fn index_source(
        &self,
        source_id: &str,
        doc_type: &str,
        model_id: &str,
        chunks: &[NewChunk<'_>],
    ) -> AppResult<usize> {
        Ok(self
            .index_source_inner(source_id, doc_type, model_id, chunks)
            .await?)
    }

    async fn index_source_inner(
        &self,
        source_id: &str,
        doc_type: &str,
        model_id: &str,
        chunks: &[NewChunk<'_>],
    ) -> Result<usize, Error> {
        self.conn.execute("BEGIN IMMEDIATE", ()).await?;
        let result = async {
            self.delete_source_rows(source_id).await?;
            self.insert_chunk_rows(source_id, doc_type, model_id, chunks)
                .await
        }
        .await;
        match result {
            Ok(n) => {
                self.conn.execute("COMMIT", ()).await?;
                Ok(n)
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(target: "persistence", error = %rb, "RAG index rollback failed");
                }
                Err(e)
            }
        }
    }

    /// Append `chunks` to `source_id` WITHOUT deleting existing chunks, in one
    /// transaction. Unlike [`Self::index_source`] (delete-then-insert), this grows a
    /// source incrementally — used by the live-agent incremental transcript indexer,
    /// which appends only the turns that have newly sealed (see
    /// [`Self::max_byte_offset`]). The caller is responsible for not re-appending
    /// already-indexed chunks. Returns the number of chunks inserted.
    pub async fn append_source_chunks(
        &self,
        source_id: &str,
        doc_type: &str,
        model_id: &str,
        chunks: &[NewChunk<'_>],
    ) -> AppResult<usize> {
        Ok(self
            .append_source_chunks_inner(source_id, doc_type, model_id, chunks)
            .await?)
    }

    async fn append_source_chunks_inner(
        &self,
        source_id: &str,
        doc_type: &str,
        model_id: &str,
        chunks: &[NewChunk<'_>],
    ) -> Result<usize, Error> {
        if chunks.is_empty() {
            return Ok(0);
        }
        self.conn.execute("BEGIN IMMEDIATE", ()).await?;
        let result = self
            .insert_chunk_rows(source_id, doc_type, model_id, chunks)
            .await;
        match result {
            Ok(n) => {
                self.conn.execute("COMMIT", ()).await?;
                Ok(n)
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(target: "persistence", error = %rb, "RAG append rollback failed");
                }
                Err(e)
            }
        }
    }

    /// The highest `byte_offset` indexed for `source_id`, or `None` when the source
    /// has no chunks. The live-agent incremental indexer uses this as the watermark:
    /// a turn-packed chunk with a strictly greater offset has not been indexed yet.
    pub async fn max_byte_offset(&self, source_id: &str) -> AppResult<Option<u64>> {
        Ok(self.max_byte_offset_inner(source_id).await?)
    }

    async fn max_byte_offset_inner(&self, source_id: &str) -> Result<Option<u64>, Error> {
        let mut rows = self
            .conn
            .query(
                "SELECT MAX(byte_offset) FROM rag_chunk WHERE source_id = ?1",
                libsql::params![source_id],
            )
            .await?;
        match rows.next().await? {
            // MAX over no rows yields a single NULL row → `Option<i64>` is `None`.
            Some(row) => Ok(row.get::<Option<i64>>(0)?.map(|v| v as u64)),
            None => Ok(None),
        }
    }

    /// Insert the chunk / FTS / embedding rows for `chunks`. The caller owns the
    /// surrounding transaction, so [`Self::index_source`] can delete-first while
    /// [`Self::append_source_chunks`] inserts-only.
    async fn insert_chunk_rows(
        &self,
        source_id: &str,
        doc_type: &str,
        model_id: &str,
        chunks: &[NewChunk<'_>],
    ) -> Result<usize, Error> {
        for c in chunks {
            let id = uuid::Uuid::new_v4().to_string();
            self.conn
                .execute(
                    "INSERT INTO rag_chunk (id, doc_type, source_id, chunk_text, byte_offset)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    libsql::params![id.clone(), doc_type, source_id, c.text, c.byte_offset as i64],
                )
                .await?;
            self.conn
                .execute(
                    "INSERT INTO rag_chunk_fts (chunk_text, chunk_id) VALUES (?1, ?2)",
                    libsql::params![c.text, id.clone()],
                )
                .await?;
            self.conn
                .execute(
                    "INSERT INTO rag_embedding (chunk_id, embedding, dim, model_id)
                     VALUES (?1, ?2, ?3, ?4)",
                    libsql::params![
                        id,
                        f32_slice_to_blob(c.embedding),
                        c.embedding.len() as i64,
                        model_id
                    ],
                )
                .await?;
        }
        Ok(chunks.len())
    }

    /// Remove every chunk (and its embedding + FTS row) for `source_id`. Returns
    /// the number of `rag_chunk` rows deleted.
    pub async fn forget_source(&self, source_id: &str) -> AppResult<u64> {
        Ok(self.forget_source_inner(source_id).await?)
    }

    async fn forget_source_inner(&self, source_id: &str) -> Result<u64, Error> {
        self.conn.execute("BEGIN IMMEDIATE", ()).await?;
        let result = self.delete_source_rows(source_id).await;
        match result {
            Ok(n) => {
                self.conn.execute("COMMIT", ()).await?;
                Ok(n)
            }
            Err(e) => {
                if let Err(rb) = self.conn.execute("ROLLBACK", ()).await {
                    tracing::warn!(target: "persistence", error = %rb, "RAG forget rollback failed");
                }
                Err(e)
            }
        }
    }

    /// Delete the FTS, embedding, and chunk rows for `source_id`. The caller holds
    /// the transaction. FTS5 is not foreign-keyed, and the FK cascade is not relied
    /// upon (the pragma is off by default), so all three tables are cleared explicitly.
    async fn delete_source_rows(&self, source_id: &str) -> Result<u64, Error> {
        self.conn
            .execute(
                "DELETE FROM rag_chunk_fts
                 WHERE chunk_id IN (SELECT id FROM rag_chunk WHERE source_id = ?1)",
                libsql::params![source_id],
            )
            .await?;
        self.conn
            .execute(
                "DELETE FROM rag_embedding
                 WHERE chunk_id IN (SELECT id FROM rag_chunk WHERE source_id = ?1)",
                libsql::params![source_id],
            )
            .await?;
        let deleted = self
            .conn
            .execute(
                "DELETE FROM rag_chunk WHERE source_id = ?1",
                libsql::params![source_id],
            )
            .await?;
        Ok(deleted)
    }

    /// Dense leg: rank chunks by cosine similarity to `query_embedding` (assumed
    /// L2-normalised, so cosine reduces to a dot product), returning the top `k`.
    /// Scores ONLY against vectors stored under `model_id` and of the same
    /// dimension — a vector from a different embedder (or dimension) is skipped
    /// rather than scored on a truncated dot product, so a model swap degrades to
    /// "no comparable vectors" instead of silently corrupting the ranking.
    /// Brute-force over the meeting's vectors (a few hundred chunks, sub-ms).
    /// `k == 0` or an empty query returns empty.
    pub async fn retrieve_dense(
        &self,
        query_embedding: &[f32],
        model_id: &str,
        k: usize,
    ) -> AppResult<Vec<RetrievedChunk>> {
        Ok(self
            .retrieve_dense_inner(query_embedding, model_id, k)
            .await?)
    }

    async fn retrieve_dense_inner(
        &self,
        query_embedding: &[f32],
        model_id: &str,
        k: usize,
    ) -> Result<Vec<RetrievedChunk>, Error> {
        if k == 0 || query_embedding.is_empty() {
            return Ok(Vec::new());
        }
        let mut rows = self
            .conn
            .query(
                "SELECT c.id, c.doc_type, c.source_id, c.chunk_text, c.byte_offset, e.embedding
                 FROM rag_chunk c JOIN rag_embedding e ON e.chunk_id = c.id
                 WHERE e.model_id = ?1",
                libsql::params![model_id],
            )
            .await?;
        let mut scored: Vec<RetrievedChunk> = Vec::new();
        while let Some(row) = rows.next().await? {
            let blob: Vec<u8> = row.get(5)?;
            let v = blob_to_f32_vec(&blob);
            // Defence in depth behind the model_id filter: never score against a
            // foreign-dimension vector (cosine_unit would silently truncate it).
            if v.len() != query_embedding.len() {
                continue;
            }
            let score = cosine_unit(query_embedding, &v);
            let byte_offset: i64 = row.get(4)?;
            scored.push(RetrievedChunk {
                chunk_id: row.get(0)?,
                doc_type: row.get(1)?,
                source_id: row.get(2)?,
                text: row.get(3)?,
                byte_offset: byte_offset as u64,
                score,
            });
        }
        // Descending; non-finite scores sink to the bottom so a degenerate
        // embedding can't corrupt the ordering (shared comparator).
        scored.sort_by(|a, b| cmp_desc_finite_first(a.score, b.score));
        scored.truncate(k);
        Ok(scored)
    }

    /// Lexical leg: FTS5 `bm25()` over the chunk text, best-ranked first, top `k`.
    /// The query is sanitised into a quoted-token MATCH expression so FTS5
    /// operators in user text cannot inject. `k == 0` or a query with no usable
    /// tokens returns empty.
    pub async fn retrieve_lexical(
        &self,
        query_text: &str,
        k: usize,
    ) -> AppResult<Vec<RetrievedChunk>> {
        Ok(self.retrieve_lexical_inner(query_text, k).await?)
    }

    async fn retrieve_lexical_inner(
        &self,
        query_text: &str,
        k: usize,
    ) -> Result<Vec<RetrievedChunk>, Error> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let match_query = fts_match_query(query_text);
        if match_query.is_empty() {
            return Ok(Vec::new());
        }
        let mut rows = self
            .conn
            .query(
                "SELECT c.id, c.doc_type, c.source_id, c.chunk_text, c.byte_offset, bm25(rag_chunk_fts)
                 FROM rag_chunk_fts
                 JOIN rag_chunk c ON c.id = rag_chunk_fts.chunk_id
                 WHERE rag_chunk_fts MATCH ?1
                 ORDER BY bm25(rag_chunk_fts)
                 LIMIT ?2",
                libsql::params![match_query, k as i64],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let byte_offset: i64 = row.get(4)?;
            let bm25: f64 = row.get(5)?;
            out.push(RetrievedChunk {
                chunk_id: row.get(0)?,
                doc_type: row.get(1)?,
                source_id: row.get(2)?,
                text: row.get(3)?,
                byte_offset: byte_offset as u64,
                score: bm25 as f32,
            });
        }
        Ok(out)
    }
}

/// Create the RAG tables if absent. Plain self-content FTS5 (the chunk text is
/// stored in both `rag_chunk` and `rag_chunk_fts`, kept in sync within the write
/// transaction) — no external-content table + triggers, so there is no
/// trigger-portability assumption. `chunk_id` is an UNINDEXED FTS column used only
/// to join FTS hits back to `rag_chunk`.
async fn create_schema(conn: &Connection) -> Result<(), Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_chunk (
            id          TEXT PRIMARY KEY,
            doc_type    TEXT NOT NULL,
            source_id   TEXT NOT NULL,
            chunk_text  TEXT NOT NULL,
            byte_offset INTEGER NOT NULL
        )",
        (),
    )
    .await?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_chunk_source ON rag_chunk (source_id)",
        (),
    )
    .await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_embedding (
            chunk_id  TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            dim       INTEGER NOT NULL,
            model_id  TEXT NOT NULL
        )",
        (),
    )
    .await?;
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunk_fts USING fts5 (chunk_text, chunk_id UNINDEXED)",
        (),
    )
    .await?;
    Ok(())
}

/// Build a safe FTS5 `MATCH` expression from free user text: each whitespace
/// token is wrapped in double quotes (a string literal — internal quotes doubled)
/// and joined by spaces (implicit AND), so FTS5 operators in the query cannot
/// inject. Returns an empty string when the query has no usable tokens.
fn fts_match_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|tok| format!("\"{}\"", tok.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        minutist_common::voiceprint_math::unit_normalise(&mut v);
        v
    }

    async fn store() -> RagStore {
        RagStore::open(":memory:").await.expect("open in-memory")
    }

    #[tokio::test]
    async fn index_retrieve_dense_ranks_nearest_first() {
        let s = store().await;
        let a = unit(vec![1.0, 0.0, 0.0]);
        let b = unit(vec![0.0, 1.0, 0.0]);
        s.index_source(
            "transcript",
            "transcript",
            "bge-m3-q8_0",
            &[
                NewChunk { text: "alpha", byte_offset: 0, embedding: &a },
                NewChunk { text: "beta", byte_offset: 10, embedding: &b },
            ],
        )
        .await
        .expect("index");

        let q = unit(vec![0.95, 0.05, 0.0]);
        let hits = s.retrieve_dense(&q, "bge-m3-q8_0", 2).await.expect("dense");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].text, "alpha", "nearest vector first");
        assert!(hits[0].score >= hits[1].score);
    }

    #[tokio::test]
    async fn lexical_matches_and_sanitises() {
        let s = store().await;
        let e = unit(vec![1.0, 0.0]);
        s.index_source(
            "transcript",
            "transcript",
            "m",
            &[
                NewChunk { text: "the migration plan is owned by Priya", byte_offset: 0, embedding: &e },
                NewChunk { text: "unrelated text about coffee", byte_offset: 40, embedding: &e },
            ],
        )
        .await
        .expect("index");

        let hits = s.retrieve_lexical("migration", 5).await.expect("lexical");
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("migration"));

        // "OR" is quoted to a literal term, NOT the FTS5 OR operator: sanitised,
        // this requires migration AND OR AND coffee, which no single chunk has —
        // so it does not behave as `migration OR coffee` (which would match both).
        let not_or = s.retrieve_lexical("migration OR coffee", 5).await.expect("no error");
        assert!(not_or.len() < 2, "OR neutralised to a literal term");

        // A bare quote is an FTS5 syntax error unsanitised; quoted away it just
        // returns empty rather than erroring.
        let bare = s.retrieve_lexical("\"", 5).await.expect("bare quote sanitised");
        assert!(bare.is_empty());
    }

    #[tokio::test]
    async fn reindex_replaces_and_forget_removes() {
        let s = store().await;
        let e = unit(vec![1.0, 0.0]);
        s.index_source("att1", "attachment", "m", &[NewChunk { text: "v1", byte_offset: 0, embedding: &e }])
            .await
            .expect("index v1");
        assert!(s.has_source("att1").await.unwrap());

        // Re-index the same source: replace, not accumulate.
        s.index_source(
            "att1",
            "attachment",
            "m",
            &[
                NewChunk { text: "v2a", byte_offset: 0, embedding: &e },
                NewChunk { text: "v2b", byte_offset: 4, embedding: &e },
            ],
        )
        .await
        .expect("reindex");
        let all = s.retrieve_dense(&e, "m", 10).await.unwrap();
        assert_eq!(all.len(), 2, "re-index replaced the prior chunk set");
        assert!(all.iter().all(|c| c.text != "v1"));

        let removed = s.forget_source("att1").await.unwrap();
        assert_eq!(removed, 2);
        assert!(!s.has_source("att1").await.unwrap());
        assert!(s.retrieve_dense(&e, "m", 10).await.unwrap().is_empty());
        assert!(s.retrieve_lexical("v2a", 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn append_grows_source_and_tracks_watermark() {
        let s = store().await;
        let e = unit(vec![1.0, 0.0]);
        assert_eq!(
            s.max_byte_offset("transcript").await.unwrap(),
            None,
            "empty source has no watermark"
        );

        s.append_source_chunks(
            "transcript",
            "transcript",
            "m",
            &[
                NewChunk { text: "first turn", byte_offset: 0, embedding: &e },
                NewChunk { text: "second turn", byte_offset: 50, embedding: &e },
            ],
        )
        .await
        .expect("append 1");
        assert_eq!(s.max_byte_offset("transcript").await.unwrap(), Some(50));

        // A second append GROWS the source (does not replace, unlike index_source).
        s.append_source_chunks(
            "transcript",
            "transcript",
            "m",
            &[NewChunk { text: "third turn", byte_offset: 120, embedding: &e }],
        )
        .await
        .expect("append 2");
        assert_eq!(s.max_byte_offset("transcript").await.unwrap(), Some(120));
        let all = s.retrieve_dense(&e, "m", 10).await.unwrap();
        assert_eq!(all.len(), 3, "append accumulates across calls");

        // An empty append is a no-op and leaves the watermark unchanged.
        assert_eq!(
            s.append_source_chunks("transcript", "transcript", "m", &[])
                .await
                .unwrap(),
            0
        );
        assert_eq!(s.max_byte_offset("transcript").await.unwrap(), Some(120));
    }
}
