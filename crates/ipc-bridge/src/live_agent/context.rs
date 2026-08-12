//! Context-assembly helpers the worker calls into: the one-time system
//! prompt prefix, RAG retrieval for per-turn context injection, and the
//! post-eviction recap loader.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chat_agent::TurnMarkers;
use minutist_common::{ChatRole, Embedder, MeetingId};
use persistence::{ChatStore, RagStore, RetrievedChunk};
use rag_retrieval::rrf_fuse;
use tokio::sync::OnceCell;

use super::{EVICT_RECAP_CHARS, EVICT_RECAP_LINE_CAP, EVICT_RECAP_TURNS};

/// Build the one-time system-prompt prefix for the co-pilot keep-alive session.
///
/// The prefix is prefilled ONCE at session start and held for the session
/// lifetime; each subsequent `converse` call appends a new turn to the KV
/// cache rather than re-prefilling. The content is the plain co-pilot persona
/// from settings, optionally followed by a compact attachment-awareness block.
///
/// The prefix is a **complete, closed** system/user turn:
/// `<bos>{open}user\n{system}[awareness]{close}\n`. This allows `append_turn`
/// (invoked via `session.converse`) to treat `n_past == prefix_len` as a clean
/// boundary with no prior open turn to close — matching `append_turn`'s
/// first-turn framing contract, which does NOT prepend a close marker when
/// starting from the prefix.
///
/// `markers` must be the `TurnMarkers` detected from the loaded model at worker
/// start — never hardcoded model-specific strings.
///
/// `awareness_block` is the pre-formatted, sanitised attachment-awareness text
/// (one `- filename: summary\n` line per ready attachment). When non-empty it is
/// inserted between the system prompt and the close marker, separated by a blank
/// line and headed `## Attached documents (retrieve details on demand)`. An empty
/// `awareness_block` produces the same prefix as if no attachments exist.
///
/// Note: awareness is loaded at worker startup only. An attachment added DURING
/// a live session is not reflected here until the session restarts. The
/// mid-session re-seed path (A1 dirty-prefix / eviction-rebuild) is deferred.
pub(crate) fn build_prefix(
    s: &settings::Settings,
    markers: &TurnMarkers,
    awareness_block: &str,
) -> String {
    let mut prefix = String::new();
    // BOS + a self-contained closed user turn carrying the system prompt.
    // The close marker terminates the turn so `append_turn` can begin a fresh
    // user turn immediately, with no dangling open-turn state in the KV.
    prefix.push_str("<bos>");
    prefix.push_str(&markers.turn_open);
    prefix.push_str("user\n");
    prefix.push_str(&s.live_agent_system_prompt);
    if !awareness_block.is_empty() {
        prefix.push_str("\n\n## Attached documents (retrieve details on demand)\n");
        prefix.push_str(awareness_block);
    }
    prefix.push_str(&markers.turn_close);
    prefix.push('\n');
    prefix
}

/// Cap the retrieval query to the most recent slice of the discussion — it is the
/// "what's being talked about now" focus; older context is what we retrieve.
const QUERY_CHAR_CAP: usize = 2000;

/// Per-session RAG context held by the worker: drives both retrieval (each refresh
/// embeds the recent window + reads the cache) and incremental transcript indexing
/// (sealed turns appended as the meeting runs). The shared embedder cell is peeked —
/// `None` until the background load completes.
///
/// `store` is a single libsql connection used for BOTH the per-refresh reads and the
/// incremental-index write; this is sound ONLY because the worker is a single-in-flight
/// current-thread loop (the read and the append never overlap). Moving the incremental
/// index off the loop (e.g. `tokio::spawn`) would need its own connection.
pub(crate) struct LiveRetrieval {
    pub(crate) embedder_cell: Arc<OnceCell<Arc<dyn Embedder>>>,
    pub(crate) store: RagStore,
    /// Meeting folder root — the incremental indexer reads `transcript.json` under it.
    pub(crate) meetings_dir: PathBuf,
    /// Top-k chunks fused across the dense + lexical legs (tier-scaled).
    pub(crate) k: usize,
    /// Upper bound on injected-context characters. A backstop — `k` is the
    /// dominant knob, since each chunk is ~1 KB.
    pub(crate) char_budget: usize,
}

/// Scale the configured retrieval `k` to the GPU tier. An integrated GPU pays a
/// quadratic per-refresh prefill, so it gets roughly half the chunks (floored so
/// retrieval never collapses to a single hit); a discrete GPU uses the full `k`.
/// `k == 0` disables retrieval on both tiers.
pub(crate) fn tier_scaled_k(base_k: usize, is_integrated: bool) -> usize {
    if base_k == 0 {
        return 0;
    }
    if is_integrated {
        (base_k / 2).max(3).min(base_k)
    } else {
        base_k
    }
}

/// The last `n` characters of `s` (char-boundary safe), or all of `s` when shorter.
/// `n == 0` yields `""`.
pub(crate) fn tail_chars(s: &str, n: usize) -> &str {
    if n == 0 {
        return "";
    }
    let count = s.chars().count();
    if count <= n {
        return s;
    }
    let start = s
        .char_indices()
        .nth(count - n)
        .map(|(i, _)| i)
        .unwrap_or(0);
    &s[start..]
}

/// A human-readable heading for a retrieved chunk, by document type. (The
/// attachment `source_id` is a content hash, not a filename, so the heading is
/// generic until a hash→filename lookup is wired in.)
fn retrieval_source_label(c: &RetrievedChunk) -> &'static str {
    match c.doc_type.as_str() {
        "attachment" => "From an attached document",
        "transcript" => "Earlier in the meeting",
        _ => "Relevant context",
    }
}

/// Retrieve the chunks relevant to the recent discussion and format them as a
/// tail-injected context block, or `None` when retrieval is unavailable / empty.
///
/// Query = the recent transcript window (`recent`, this refresh's new segments),
/// capped to the last [`QUERY_CHAR_CAP`] characters. The dense (cosine) and
/// lexical (FTS5) legs are fused by RRF. No dedup against the live window is
/// needed: `recent` is reset each refresh, and the incremental indexer only seals
/// turns that have already scrolled out of the window, so an indexed transcript
/// chunk can never duplicate what is in the current window. Survivors are packed
/// up to `char_budget`.
pub(crate) async fn build_retrieval_block(rc: &LiveRetrieval, recent: &str) -> Option<String> {
    // Peek the shared cell — `None` until the background load completes.
    let embedder = rc.embedder_cell.get().cloned()?;
    if rc.k == 0 {
        return None;
    }
    let query = tail_chars(recent, QUERY_CHAR_CAP);
    if query.trim().is_empty() {
        return None;
    }
    // Embed the query OFF the runtime thread (sync FFI ~180 ms) so the background
    // embedder load and the worker channels keep progressing during it.
    let q = query.to_string();
    let emb = embedder.clone();
    let qvec = match tokio::task::spawn_blocking(move || emb.embed(&q)).await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            tracing::warn!(
                target: "ipc-bridge",
                error = %e,
                "live-agent: query embed failed; skipping context injection"
            );
            return None;
        }
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                error = %e,
                "live-agent: query embed task join failed; skipping context injection"
            );
            return None;
        }
    };
    let model_id = embedder.model_id();
    // A cache-read failure disables injection for this refresh only (best-effort).
    let dense = rc
        .store
        .retrieve_dense(&qvec, model_id, rc.k)
        .await
        .unwrap_or_default();
    let lexical = rc
        .store
        .retrieve_lexical(query, rc.k)
        .await
        .unwrap_or_default();
    if dense.is_empty() && lexical.is_empty() {
        return None;
    }

    // Fuse by chunk_id (RRF ignores the per-leg score scales), then map the fused
    // ids back to their chunk. Both legs return the same chunk fields, so either
    // copy works.
    let mut by_id: HashMap<String, RetrievedChunk> = HashMap::new();
    for c in dense.iter().chain(lexical.iter()) {
        by_id.entry(c.chunk_id.clone()).or_insert_with(|| c.clone());
    }
    let dense_ids: Vec<String> = dense.iter().map(|c| c.chunk_id.clone()).collect();
    let lexical_ids: Vec<String> = lexical.iter().map(|c| c.chunk_id.clone()).collect();
    let fused = rrf_fuse(&[&dense_ids, &lexical_ids], rc.k);

    let mut block = String::new();
    let mut used = 0usize;
    for id in fused {
        let Some(chunk) = by_id.get(&id) else {
            continue;
        };
        if used + chunk.text.len() > rc.char_budget {
            break;
        }
        let label = retrieval_source_label(chunk);
        block.push_str(&format!("## {label}\n{}\n\n", chunk.text.trim()));
        used += chunk.text.len();
    }
    if block.is_empty() {
        None
    } else {
        Some(format!(
            "Relevant context (attachments + earlier transcript):\n\n{block}"
        ))
    }
}

/// Read the last `EVICT_RECAP_TURNS` User and Assistant messages from the
/// meeting's persisted live `ChatSession` and format them as a size-capped
/// block to prepend after a KV eviction.
///
/// Returns `Some(recap)` on success, or `None` on a load failure (best-effort:
/// the session is reset without a recap rather than failing the turn). The
/// caller sanitises the resulting string with `sanitise_untrusted` before
/// injecting it into the model prompt.
///
/// Roles included: `ChatRole::User` and `ChatRole::Assistant`. `ChatRole::Digest`
/// (transcript auto-injections) and `ChatRole::Tool` are excluded — they are
/// bulky and less useful as conversation context after eviction.
///
/// Ordering: most-recent first, so that trimming the block by the whole-block
/// character cap always preserves the newest context. After trimming, the block
/// is reversed before returning so the model reads it in chronological order.
pub(crate) fn load_eviction_recap(meetings_dir: &Path, meeting_id: MeetingId) -> Option<String> {
    let now = chrono::Utc::now().to_rfc3339();
    let chat_session = match ChatStore::load_or_create_live(meetings_dir, meeting_id, &now) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                "live-agent eviction: chat session load failed; resetting without recap: {e}"
            );
            return None;
        }
    };

    // Collect the last EVICT_RECAP_TURNS User + Assistant messages (most-recent-first).
    let relevant: Vec<&minutist_common::ChatMessage> = chat_session
        .messages
        .iter()
        .rev()
        .filter(|m| {
            matches!(m.role, ChatRole::User | ChatRole::Assistant)
        })
        .take(EVICT_RECAP_TURNS)
        .collect();

    if relevant.is_empty() {
        return None;
    }

    // Format each entry as "{role}: {content}", truncated to EVICT_RECAP_LINE_CAP.
    // Accumulate most-recent-first until we reach EVICT_RECAP_CHARS.
    let mut lines: Vec<String> = Vec::with_capacity(relevant.len());
    let mut total_chars = 0usize;
    for msg in &relevant {
        let role_label = match msg.role {
            ChatRole::User => "User",
            ChatRole::Assistant => "Assistant",
            _ => continue,
        };
        let content = &msg.content;
        let line_raw = format!("{role_label}: {content}");
        // Per-line cap: truncate at a char boundary.
        let line = if line_raw.chars().count() > EVICT_RECAP_LINE_CAP {
            let cap_byte = line_raw
                .char_indices()
                .nth(EVICT_RECAP_LINE_CAP)
                .map(|(i, _)| i)
                .unwrap_or(line_raw.len());
            line_raw[..cap_byte].to_string()
        } else {
            line_raw
        };
        if total_chars + line.len() > EVICT_RECAP_CHARS {
            // The recap budget is full; stop here (older entries are less useful).
            break;
        }
        total_chars += line.len();
        lines.push(line);
    }

    if lines.is_empty() {
        return None;
    }

    // lines is most-recent-first; reverse to chronological order before joining.
    lines.reverse();
    Some(lines.join("\n"))
}
