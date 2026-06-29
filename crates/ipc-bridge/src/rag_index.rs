//! Best-effort RAG write-path: chunk + embed + persist attachment / transcript
//! text into the meeting's `meeting.db`.
//!
//! RAG is a rebuildable cache, so every failure here logs and is swallowed — it
//! must never fail attachment conversion or the post-stop flow. Reused by the
//! attach hook (`attachments::run_convert_job`), the post-stop transcript pass and
//! the standalone reprocess command (`commands`). Embedding is synchronous FFI and
//! runs on `spawn_blocking`; persistence is async libsql.

use std::path::Path;
use std::sync::Arc;

use minutist_common::{apply_speaker_overlay, AppError, AppResult, Embedder, MeetingId, Segment};
use persistence::meeting_db_path;

use crate::chat_runtime::ChatHandles;

/// ~256-token chunks (≈1024 chars) with ~20% overlap — the SP-LIVE E6 setting.
/// Used for attachment markdown; the transcript is chunked by speaker turn instead.
const CHUNK_CHARS: usize = 1024;
const CHUNK_OVERLAP: usize = 200;

/// `source_id` for transcript chunks appended incrementally DURING a live recording.
/// Kept distinct from the canonical `"transcript"` source the post-stop pass writes,
/// so the live append and the post-stop delete-then-insert never collide on a shared
/// key. The post-stop [`index_transcript`] forgets it once the full set is written.
const TRANSCRIPT_LIVE_SOURCE: &str = "transcript_live";

/// Embed `chunks` (`(text, byte_offset)` pairs) and replace `source_id`'s chunks in
/// the meeting's `meeting.db`.
///
/// When `skip_if_indexed`, returns early WITHOUT loading the embedder or embedding
/// if the source is already present — used for content-addressed attachments whose
/// chunks never change, so a re-attach of identical content costs one query, not a
/// full BGE-M3 pass. Empty `chunks` is a no-op.
async fn index_chunks(
    handles: &ChatHandles,
    meeting_id: MeetingId,
    source_id: &str,
    doc_type: &'static str,
    chunks: Vec<(String, u64)>,
    skip_if_indexed: bool,
) -> AppResult<usize> {
    let store =
        persistence::RagStore::open(meeting_db_path(&handles.meetings_dir, meeting_id)).await?;
    // Cheap skip BEFORE loading the model: a content-addressed source already in
    // the index has byte-identical chunks, so there is nothing to re-embed.
    if skip_if_indexed && store.has_source(source_id).await? {
        return Ok(0);
    }
    if chunks.is_empty() {
        return Ok(0);
    }
    // Loading the embedder is what downloads BGE-M3 on first use; do it only once
    // the skip check has passed. Embedding is sync FFI → off the async runtime.
    let embedder = handles.ensure_embedder().await.map_err(AppError::from)?;
    // Capture the model id before `embedder` moves into spawn_blocking.
    let model_id = embedder.model_id().to_string();
    let owned: Vec<String> = chunks.iter().map(|(t, _)| t.clone()).collect();
    let embeddings = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        embedder.embed_batch(&refs)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("embed task join failed: {e}"),
    })??;
    // The 1:1 contract is load-bearing — `index_source` replaces the whole source,
    // so a short result would silently persist a truncated index.
    if embeddings.len() != chunks.len() {
        return Err(AppError::Inference {
            backend: "embedder".into(),
            context: format!(
                "embedder returned {} vectors for {} chunks",
                embeddings.len(),
                chunks.len()
            ),
        });
    }
    let new_chunks: Vec<persistence::NewChunk> = chunks
        .iter()
        .zip(&embeddings)
        .map(|((text, byte_offset), embedding)| persistence::NewChunk {
            text,
            byte_offset: *byte_offset,
            embedding,
        })
        .collect();
    store
        .index_source(source_id, doc_type, &model_id, &new_chunks)
        .await
}

/// Attach-time hook: index an attachment's converted markdown into `meeting.db`.
/// Best-effort. The `source_id` is the attachment's content hash, so an already-
/// indexed identical attachment is skipped (no re-embed).
pub async fn index_attachment(handles: &ChatHandles, meeting_id: MeetingId, hash: &str) {
    let filename = format!("{hash}.md");
    let md = match persistence::read_attachment_markdown(&handles.meetings_dir, meeting_id, &filename)
    {
        Ok(md) => md,
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                error = %e,
                "RAG attach-index: reading converted markdown failed (skipped)"
            );
            return;
        }
    };
    let chunks: Vec<(String, u64)> = rag_retrieval::chunk_text(&md, CHUNK_CHARS, CHUNK_OVERLAP)
        .into_iter()
        .map(|c| (c.text, c.byte_offset as u64))
        .collect();
    match index_chunks(handles, meeting_id, hash, "attachment", chunks, true).await {
        Ok(n) => tracing::info!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            chunks = n,
            "indexed attachment for retrieval"
        ),
        Err(e) => tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            error = %e,
            "RAG attach-index failed (best-effort)"
        ),
    }
}

/// Attachment-remove hook: forget a now-orphaned source's chunks from `meeting.db`.
/// Best-effort — logs and returns on error. The caller is responsible for the dedup
/// check (only call this once the content hash is no longer referenced by any
/// surviving attachment). A meeting that was never indexed has no db → no-op.
pub async fn forget_attachment(meetings_dir: &Path, meeting_id: MeetingId, source_id: &str) {
    let db = meeting_db_path(meetings_dir, meeting_id);
    if !db.exists() {
        return;
    }
    let store = match persistence::RagStore::open(&db).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                error = %e,
                "RAG forget: opening meeting.db failed (skipped)"
            );
            return;
        }
    };
    if let Err(e) = store.forget_source(source_id).await {
        tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            error = %e,
            "RAG forget_source failed (best-effort)"
        );
    }
}

/// Transcript-finalise hook: index the meeting transcript into `meeting.db`. Called
/// at every point the transcript is finalised — the post-stop pass AND the
/// standalone reprocess command — so the index never goes stale relative to the
/// displayed transcript. Returns `Err` so the caller can log it. The `"transcript"`
/// source is replaced wholesale on each call.
pub async fn index_transcript(handles: &ChatHandles, meeting_id: MeetingId) -> AppResult<()> {
    let meeting_dir = handles.meetings_dir.join(meeting_id.0.to_string());
    let mut segments = persistence::read_transcript(&meeting_dir)?;
    if segments.is_empty() {
        return Ok(());
    }
    // Resolve raw diarizer labels (e.g. "SPEAKER_00") to display names via the same
    // canonical overlay the read tools / summariser use, so retrieved chunk text
    // matches what the user sees. A no-op until names are set (e.g. after reprocess).
    if let Ok(meta) = persistence::read_metadata(&meeting_dir) {
        apply_speaker_overlay(&mut segments, &meta.speaker_names);
    }
    let chunks = chunk_transcript_turns(&segments, CHUNK_CHARS);
    let n = index_chunks(handles, meeting_id, "transcript", "transcript", chunks, false).await?;
    // Drop the transient live-append chunks now the canonical "transcript" set is
    // written. Done AFTER the full index so a live append that landed during it is
    // also cleared; the rare append that lands after this forget self-heals on the
    // next reprocess (best-effort, like the rest of the RAG write path).
    if let Ok(store) =
        persistence::RagStore::open(meeting_db_path(&handles.meetings_dir, meeting_id)).await
    {
        if let Err(e) = store.forget_source(TRANSCRIPT_LIVE_SOURCE).await {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                error = %e,
                "RAG: clearing transient live transcript chunks failed (best-effort)"
            );
        }
    }
    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        chunks = n,
        "indexed transcript for retrieval"
    );
    Ok(())
}

/// Live-recording incremental transcript indexer: append the turns that have newly
/// **sealed** (scrolled out of the live window) into `store`, embedding only those.
///
/// Re-chunks the on-disk transcript each call (cheap string work), drops the trailing
/// partial chunk (still inside the live window — re-chunked next call), and appends
/// only chunks whose `byte_offset` is beyond the indexed watermark
/// ([`persistence::RagStore::max_byte_offset`]). Greedy turn-packing makes every
/// earlier chunk prefix-stable, so the watermark cleanly separates new from indexed —
/// already-indexed turns are never re-embedded.
///
/// Writes to a distinct [`TRANSCRIPT_LIVE_SOURCE`] (not the canonical `"transcript"`),
/// so it never collides with the post-stop full re-index; that pass forgets the live
/// source once the canonical set is written, making these chunks transient. Raw
/// diarizer labels are kept (NO `apply_speaker_overlay`): the overlay is a no-op
/// during a live recording anyway, and skipping it keeps `byte_offset`
/// overlay-invariant — applying a different-length display name to an already-sealed
/// turn would otherwise shift offsets and silently desync the watermark. A torn read
/// of the in-place-rewritten `transcript.json` mid-flush yields a parse `Err` the
/// caller swallows and retries next refresh (benign). Returns the number of chunks
/// appended. The caller (live-agent worker) supplies the open `store` + the
/// already-resolved `embedder`.
pub async fn index_transcript_incremental(
    store: &persistence::RagStore,
    meetings_dir: &Path,
    meeting_id: MeetingId,
    embedder: &Arc<dyn Embedder>,
) -> AppResult<usize> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    let segments = persistence::read_transcript(&meeting_dir)?;
    if segments.is_empty() {
        return Ok(0);
    }
    let mut chunks = chunk_transcript_turns(&segments, CHUNK_CHARS);
    // Drop the trailing partial chunk — more turns will extend it, and it is still
    // inside the live window. Everything before it is sealed + prefix-stable.
    chunks.pop();
    if chunks.is_empty() {
        return Ok(0);
    }
    let watermark = store.max_byte_offset(TRANSCRIPT_LIVE_SOURCE).await?;
    let fresh: Vec<(String, u64)> = chunks
        .into_iter()
        .filter(|(_, off)| watermark.is_none_or(|w| *off > w))
        .collect();
    if fresh.is_empty() {
        return Ok(0);
    }
    let model_id = embedder.model_id().to_string();
    let owned: Vec<String> = fresh.iter().map(|(t, _)| t.clone()).collect();
    let emb = embedder.clone();
    let embeddings = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        emb.embed_batch(&refs)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("incremental embed task join failed: {e}"),
    })??;
    if embeddings.len() != fresh.len() {
        return Err(AppError::Inference {
            backend: "embedder".into(),
            context: format!(
                "embedder returned {} vectors for {} chunks",
                embeddings.len(),
                fresh.len()
            ),
        });
    }
    let to_append: Vec<persistence::NewChunk> = fresh
        .iter()
        .zip(&embeddings)
        .map(|((text, byte_offset), embedding)| persistence::NewChunk {
            text,
            byte_offset: *byte_offset,
            embedding,
        })
        .collect();
    store
        .append_source_chunks(TRANSCRIPT_LIVE_SOURCE, "transcript", &model_id, &to_append)
        .await
}

/// Chunk the transcript by speaker turn (issue #0015): pack consecutive turns
/// (`"speaker: text"`) up to ~`chunk_chars`, so a chunk holds whole turns and breaks
/// on a turn boundary rather than mid-utterance. A single turn longer than the budget
/// becomes its own chunk (the embedder caps tokens). `byte_offset` indexes into the
/// assembled per-turn text, for provenance.
fn chunk_transcript_turns(segments: &[Segment], chunk_chars: usize) -> Vec<(String, u64)> {
    let mut chunks: Vec<(String, u64)> = Vec::new();
    let mut cur = String::new();
    let mut cur_start = 0u64;
    let mut offset = 0u64;
    for seg in segments {
        let line = match &seg.speaker_id {
            Some(spk) => format!("{spk}: {}", seg.text),
            None => seg.text.clone(),
        };
        // Flush before this turn would push the chunk past the budget, but always
        // keep at least one turn per chunk (an over-budget lone turn stands alone).
        if !cur.is_empty() && cur.len() + 1 + line.len() > chunk_chars {
            chunks.push((std::mem::take(&mut cur), cur_start));
        }
        if cur.is_empty() {
            cur_start = offset;
        } else {
            cur.push('\n');
            offset += 1;
        }
        cur.push_str(&line);
        offset += line.len() as u64;
    }
    if !cur.is_empty() {
        chunks.push((cur, cur_start));
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(speaker: Option<&str>, text: &str) -> Segment {
        Segment {
            start_ms: 0,
            end_ms: 0,
            text: text.to_string(),
            speaker_id: speaker.map(str::to_string),
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        }
    }

    #[test]
    fn turn_chunking_packs_turns_and_breaks_on_boundaries() {
        let segments = vec![
            seg(Some("A"), "hello there"),
            seg(Some("B"), "general kenobi"),
            seg(Some("A"), "you are a bold one"),
        ];
        // Tiny budget forces one turn per chunk; each chunk is a whole turn.
        let chunks = chunk_transcript_turns(&segments, 10);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].0, "A: hello there");
        assert_eq!(chunks[1].0, "B: general kenobi");
        // Offsets are non-decreasing.
        assert!(chunks[0].1 <= chunks[1].1 && chunks[1].1 <= chunks[2].1);

        // A generous budget packs all turns into one chunk on turn-joined lines.
        let one = chunk_transcript_turns(&segments, 10_000);
        assert_eq!(one.len(), 1);
        assert!(one[0].0.contains("A: hello there\nB: general kenobi\n"));
    }

    #[test]
    fn turn_chunking_empty_is_empty() {
        assert!(chunk_transcript_turns(&[], 1024).is_empty());
    }

    /// Deterministic embedder for the incremental-index test (ranking is irrelevant
    /// here; we assert append counts + the watermark).
    struct StubEmbedder;

    impl Embedder for StubEmbedder {
        fn embed_batch(&self, texts: &[&str]) -> AppResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }
        fn dim(&self) -> usize {
            4
        }
        fn model_id(&self) -> &str {
            "stub-embed"
        }
    }

    #[tokio::test]
    async fn incremental_index_appends_only_newly_sealed_turns() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mid = MeetingId::new();
        let meeting_dir = tmp.path().join(mid.0.to_string());
        std::fs::create_dir_all(&meeting_dir).expect("mkdir");
        let store = persistence::RagStore::open(":memory:").await.expect("open");
        let embedder: Arc<dyn Embedder> = Arc::new(StubEmbedder);

        // Each turn longer than CHUNK_CHARS, so every turn is its own chunk and the
        // most recent turn is always the unsealed trailing partial.
        let long = "x".repeat(CHUNK_CHARS + 100);
        let turns = |n: usize| -> Vec<Segment> {
            (0..n)
                .map(|i| seg(Some("A"), &format!("{i} {long}")))
                .collect()
        };

        // 3 turns → chunks [t0, t1, t2]; t2 is the trailing partial → 2 sealed.
        persistence::write_transcript(&meeting_dir, &turns(3)).expect("write 3");
        let n = index_transcript_incremental(&store, tmp.path(), mid, &embedder)
            .await
            .expect("inc 1");
        assert_eq!(n, 2, "two sealed turns indexed; the third is the trailing partial");

        // A 4th turn seals t2 → only t2 is beyond the watermark → 1 appended.
        persistence::write_transcript(&meeting_dir, &turns(4)).expect("write 4");
        let n = index_transcript_incremental(&store, tmp.path(), mid, &embedder)
            .await
            .expect("inc 2");
        assert_eq!(n, 1, "only the newly-sealed turn is appended (no re-embed)");

        // No new turns → nothing sealed beyond the watermark.
        let n = index_transcript_incremental(&store, tmp.path(), mid, &embedder)
            .await
            .expect("inc 3");
        assert_eq!(n, 0, "idempotent when nothing new has sealed");

        // Exactly the three sealed turns are persisted (the trailing t3 is not).
        let all = store
            .retrieve_dense(&[1.0, 0.0, 0.0, 0.0], "stub-embed", 100)
            .await
            .unwrap();
        assert_eq!(all.len(), 3, "only sealed turns are persisted");
    }

    /// Records the batch size of each `embed_batch` call, so a test can prove the
    /// incremental indexer embeds ONLY newly-sealed chunks (never re-embeds).
    struct CountingEmbedder {
        calls: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    }

    impl Embedder for CountingEmbedder {
        fn embed_batch(&self, texts: &[&str]) -> AppResult<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.len());
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }
        fn dim(&self) -> usize {
            4
        }
        fn model_id(&self) -> &str {
            "stub-embed"
        }
    }

    #[tokio::test]
    async fn incremental_index_packs_turns_and_never_re_embeds() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mid = MeetingId::new();
        let meeting_dir = tmp.path().join(mid.0.to_string());
        std::fs::create_dir_all(&meeting_dir).expect("mkdir");
        let store = persistence::RagStore::open(":memory:").await.expect("open");
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let embedder: Arc<dyn Embedder> = Arc::new(CountingEmbedder {
            calls: calls.clone(),
        });

        // ~400-char turns → two pack into one ~801-char chunk (< CHUNK_CHARS=1024); a
        // third would overflow and flush. So packing genuinely packs (unlike the
        // one-turn-per-chunk case above), exercising the prefix-stability assumption.
        let turns = |n: usize| -> Vec<Segment> {
            (0..n)
                .map(|i| seg(Some("S"), &format!("turn {i} {}", "y".repeat(390))))
                .collect()
        };

        // 5 turns → [c0=(t0,t1), c1=(t2,t3), c2=(t4 trailing)] → pop → 2 sealed.
        persistence::write_transcript(&meeting_dir, &turns(5)).expect("write 5");
        assert_eq!(
            index_transcript_incremental(&store, tmp.path(), mid, &embedder)
                .await
                .unwrap(),
            2
        );
        assert_eq!(
            *calls.lock().unwrap(),
            vec![2],
            "first call embeds the two sealed packed chunks"
        );

        // 7 turns → c2=(t4,t5) now sealed; t6 trailing. Only c2 is beyond the watermark.
        persistence::write_transcript(&meeting_dir, &turns(7)).expect("write 7");
        assert_eq!(
            index_transcript_incremental(&store, tmp.path(), mid, &embedder)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            *calls.lock().unwrap(),
            vec![2, 1],
            "second call embeds ONLY the newly-sealed chunk — c0/c1 are not re-embedded \
             (proves prefix-stable packing + the watermark)"
        );

        let all = store
            .retrieve_dense(&[1.0, 0.0, 0.0, 0.0], "stub-embed", 100)
            .await
            .unwrap();
        assert_eq!(all.len(), 3, "three sealed packed chunks, no duplicates");
    }
}
