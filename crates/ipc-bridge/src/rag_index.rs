//! Best-effort RAG write-path: chunk + embed + persist attachment / transcript
//! text into the meeting's `meeting.db`.
//!
//! RAG is a rebuildable cache, so every failure here logs and is swallowed — it
//! must never fail attachment conversion or the post-stop flow. Reused by the
//! attach hook (`attachments::run_convert_job`), the post-stop transcript pass and
//! the standalone reprocess command (`commands`). Embedding is synchronous FFI and
//! runs on `spawn_blocking`; persistence is async libsql.

use std::path::Path;

use minutist_common::{apply_speaker_overlay, AppError, AppResult, MeetingId, Segment};
use persistence::meeting_db_path;

use crate::chat_runtime::ChatHandles;

/// ~256-token chunks (≈1024 chars) with ~20% overlap — the SP-LIVE E6 setting.
/// Used for attachment markdown; the transcript is chunked by speaker turn instead.
const CHUNK_CHARS: usize = 1024;
const CHUNK_OVERLAP: usize = 200;

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
    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        chunks = n,
        "indexed transcript for retrieval"
    );
    Ok(())
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
}
