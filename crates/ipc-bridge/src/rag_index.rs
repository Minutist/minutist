//! Best-effort RAG write-path: chunk + embed + persist attachment / transcript
//! text into the meeting's `meeting.db`.
//!
//! RAG is a rebuildable cache, so every failure here logs and is swallowed — it
//! must never fail attachment conversion or the post-stop flow. Reused by the
//! attach hook (`attachments::run_convert_job`) and the post-stop transcript pass
//! (`commands::stop_recording`). Embedding is synchronous FFI and runs on
//! `spawn_blocking`; persistence is async libsql.

use std::path::{Path, PathBuf};

use minutist_common::{AppError, AppResult, MeetingId, Segment};

use crate::chat_runtime::ChatHandles;
use crate::commands::DEFAULT_EMBED_MODEL_ID;

/// ~256-token chunks (≈1024 chars) with ~20% overlap — the SP-LIVE E6 setting.
const CHUNK_CHARS: usize = 1024;
const CHUNK_OVERLAP: usize = 200;

/// Path to a meeting's RAG cache db, `{meetings_dir}/{meeting_id}/meeting.db`.
fn meeting_db_path(meetings_dir: &Path, meeting_id: MeetingId) -> PathBuf {
    meetings_dir
        .join(meeting_id.0.to_string())
        .join("meeting.db")
}

/// Chunk `text`, embed the chunks, and replace `source_id`'s chunks in the
/// meeting's `meeting.db`. Returns the number of chunks indexed (0 for empty text).
async fn index_source(
    handles: &ChatHandles,
    meeting_id: MeetingId,
    source_id: &str,
    doc_type: &'static str,
    text: &str,
) -> AppResult<usize> {
    let chunks = rag_retrieval::chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP);
    if chunks.is_empty() {
        return Ok(0);
    }
    let embedder = handles.ensure_embedder().await.map_err(AppError::from)?;
    // Embedding is synchronous FFI — run it off the async runtime. Own the chunk
    // texts so the closure is `'static`.
    let owned: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        embedder.embed_batch(&refs)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("embed task join failed: {e}"),
    })??;
    let store =
        persistence::RagStore::open(meeting_db_path(&handles.meetings_dir, meeting_id)).await?;
    let new_chunks: Vec<persistence::NewChunk> = chunks
        .iter()
        .zip(&embeddings)
        .map(|(c, e)| persistence::NewChunk {
            text: &c.text,
            byte_offset: c.byte_offset as u64,
            embedding: e,
        })
        .collect();
    store
        .index_source(source_id, doc_type, DEFAULT_EMBED_MODEL_ID, &new_chunks)
        .await
}

/// Attach-time hook: index an attachment's converted markdown into `meeting.db`.
/// Best-effort — logs and returns on any error. The `source_id` is the
/// attachment's content hash, so re-attaching identical content is idempotent.
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
    match index_source(handles, meeting_id, hash, "attachment", &md).await {
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
/// Best-effort — logs and returns on error. The caller is responsible for the
/// dedup check (only call this once the content hash is no longer referenced by
/// any surviving attachment). A meeting that was never indexed has no db → no-op.
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

/// Post-stop hook: index the meeting transcript into `meeting.db`. Returns `Err`
/// so the caller can log it alongside the other post-stop passes. Idempotent: the
/// `"transcript"` source is replaced wholesale on each call (e.g. after reprocess).
pub async fn index_transcript(handles: &ChatHandles, meeting_id: MeetingId) -> AppResult<()> {
    let meeting_dir = handles.meetings_dir.join(meeting_id.0.to_string());
    let segments = persistence::read_transcript(&meeting_dir)?;
    if segments.is_empty() {
        return Ok(());
    }
    let text = transcript_to_text(&segments);
    let n = index_source(handles, meeting_id, "transcript", "transcript", &text).await?;
    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        chunks = n,
        "indexed transcript for retrieval"
    );
    Ok(())
}

/// Assemble transcript segments into one text blob for chunking. Speaker-prefixed
/// lines give the embedder turn context; `byte_offset`s index into this blob.
fn transcript_to_text(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|s| match &s.speaker_id {
            Some(spk) => format!("{spk}: {}", s.text),
            None => s.text.clone(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
