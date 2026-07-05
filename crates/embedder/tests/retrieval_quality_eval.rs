//! Gated real-model retrieval-quality eval for the BGE-M3 embedder.
//!
//! The crate's other coverage uses a stub embedder and so only proves the
//! plumbing. This loads the REAL bge-m3 GGUF, embeds a small planted-fact corpus,
//! and runs paraphrased queries that share little vocabulary with their gold
//! chunk — so a hit proves *semantic* retrieval, not keyword overlap. It is the
//! only test that catches a degraded / mis-quantised / wrong-pooling embedder (the
//! failure a community GGUF can silently introduce: near-zero or scrambled
//! vectors), and it would have caught the gated-model outage where a missing model
//! fails the load.
//!
//! It mirrors the dense + lexical + RRF fusion of the chat-window retrieval path,
//! reimplemented here because that helper is crate-private.
//!
//! `#[ignore]`d (the project convention for real-model tests — see
//! `chat-agent/tests/real_model.rs`, `asr-runtime`): the default `cargo test`
//! never loads a model. Run it explicitly with the model path set:
//!
//!   MINUTIST_BGE_M3_PATH=/path/to/bge-m3-Q8_0.gguf \
//!   cargo test -p embedder --test retrieval_quality_eval -- --ignored --nocapture

use std::collections::HashMap;
use std::path::PathBuf;

use embedder::Bgem3Embedder;
use minutist_common::Embedder;
use persistence::{NewChunk, RagStore};

/// The model GGUF, or `None` (→ skip) when the gating var is unset/empty.
fn model_path() -> Option<PathBuf> {
    std::env::var("MINUTIST_BGE_M3_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Planted-fact corpus (indices 0–5) plus unrelated distractors (6–9). Each
/// fact is one chunk; the distractors raise the bar so a top-k hit is meaningful.
const CORPUS: &[&str] = &[
    // 0 — schema-migration owner
    "Priya Nair leads the telemetry-schema migration, moving services off the legacy \
     metrics format onto the new event schema; she is the decision-maker for that pipeline.",
    // 1 — billing owner
    "Marcus Webb owns the billing reconciliation workstream.",
    // 2 — ship date
    "The telemetry-schema migration has a committed target ship date of 14 March.",
    // 3 — per-pod compute cap
    "The staging cluster is capped at 8 vCPUs per pod; load tests that exceed it get OOM-killed.",
    // 4 — offline support cut
    "We decided not to support offline sync in version 1; it was dropped to protect the March deadline.",
    // 5 — system of record
    "Postgres remains the system of record; the event store is only a derived cache.",
    // distractors
    "The office coffee machine will be serviced on Thursday afternoon.",
    "Marketing requested new brand colours for the Q3 landing page refresh.",
    "The team offsite is tentatively booked for the second week of April.",
    "Please remember to submit expense reports before the end of the month.",
];

/// `(query, gold chunk index)`. Paraphrased to share little vocabulary with the
/// gold chunk, so a hit proves semantic retrieval rather than keyword matching.
const QUERIES: &[(&str, usize)] = &[
    ("Who is responsible for moving us onto the new metrics format?", 0),
    ("When are we aiming to have that out the door?", 2),
    ("What is the per-pod compute limit on the staging boxes?", 3),
    ("Are we shipping any offline support this release?", 4),
    ("Who is handling the billing reconciliation?", 1),
];

#[tokio::test]
#[ignore = "requires MINUTIST_BGE_M3_PATH — the bge-m3 GGUF"]
async fn bge_m3_retrieves_paraphrased_facts() {
    let Some(path) = model_path() else {
        eprintln!("skip: set MINUTIST_BGE_M3_PATH to the bge-m3 GGUF to run this eval");
        return;
    };
    if !path.exists() {
        eprintln!("skip: MINUTIST_BGE_M3_PATH does not exist: {}", path.display());
        return;
    }

    // CPU (n_gpu_layers = 0) — the eval is small and must run anywhere.
    let embedder = Bgem3Embedder::open(&path, "bge-m3-q8_0", 0).expect("load bge-m3");
    let model_id = embedder.model_id().to_string();

    // Embed + index the corpus (one source, all attachment chunks).
    let embeddings: Vec<Vec<f32>> = CORPUS
        .iter()
        .map(|t| embedder.embed(t).expect("embed chunk"))
        .collect();
    let store = RagStore::open(":memory:").await.expect("open store");
    let chunks: Vec<NewChunk> = CORPUS
        .iter()
        .zip(&embeddings)
        .enumerate()
        .map(|(i, (text, emb))| NewChunk {
            text,
            byte_offset: (i * 1000) as u64,
            embedding: emb.as_slice(),
        })
        .collect();
    store
        .index_source("brief", "attachment", &model_id, &chunks)
        .await
        .expect("index");

    const K: usize = 3;
    let mut hits = 0usize;
    for (query, gold) in QUERIES {
        let qvec = embedder.embed(query).expect("embed query");
        let dense = store
            .retrieve_dense(&qvec, &model_id, K * 2)
            .await
            .expect("dense");
        let lexical = store.retrieve_lexical(query, K * 2).await.expect("lexical");

        let dense_ids: Vec<&str> = dense.iter().map(|c| c.chunk_id.as_str()).collect();
        let lexical_ids: Vec<&str> = lexical.iter().map(|c| c.chunk_id.as_str()).collect();
        let fused = rag_retrieval::rrf_fuse(&[&dense_ids[..], &lexical_ids[..]], K);

        let text_by_id: HashMap<&str, &str> = dense
            .iter()
            .chain(lexical.iter())
            .map(|c| (c.chunk_id.as_str(), c.text.as_str()))
            .collect();
        let gold_text = CORPUS[*gold];
        let rank = fused
            .iter()
            .position(|cid| text_by_id.get(cid).copied() == Some(gold_text));
        match rank {
            Some(r) => {
                hits += 1;
                eprintln!("✓ gold@{gold} at rank {} — {query:?}", r + 1);
            }
            None => eprintln!("✗ gold@{gold} NOT in top-{K} — {query:?}"),
        }
    }

    let recall = hits as f32 / QUERIES.len() as f32;
    eprintln!("recall@{K} = {recall:.2} ({hits}/{})", QUERIES.len());
    // Degradation gate, not a precise benchmark: a healthy bge-m3 scores 0.8–1.0
    // here (measured: 4/5 rank-1, the lone miss being the hardest paraphrase —
    // "out the door" for "ship date", which shares no date words). Random ranking
    // over 10 chunks at k=3 would be ~0.3, so a scrambled / mis-quantised /
    // wrong-pooling embedder collapses well below this floor. The 0.6 floor keeps
    // one-query margin so a hard paraphrase or a model-mirror difference does not
    // flip CI, while still catching a genuinely broken embedder.
    assert!(
        recall >= 0.6,
        "bge-m3 recall@{K} = {recall:.2} is below the 0.6 floor — the embedder is likely \
         degraded/mis-quantised (scrambled or near-zero vectors), not just imperfect"
    );
}
