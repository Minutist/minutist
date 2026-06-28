//! Cosine ranking over chunk embeddings.
//!
//! Reuses the shared vector math in `common::voiceprint_math` (the same
//! `cosine_unit` / `unit_normalise` the diarizer/voiceprint path uses) rather
//! than duplicating it. Embeddings are assumed L2-normalised (the `Embedder`
//! normalises its output), so cosine reduces to the dot product.

pub use minutist_common::voiceprint_math::{cosine_unit, unit_normalise};

/// Rank `(id, embedding)` items against `query` by cosine similarity, returning
/// the top `k` as `(id, score)` in descending score order.
///
/// `query` and every embedding are assumed L2-normalised and of equal length. An
/// empty `query` returns empty (no meaningful ranking). Ties keep their original
/// relative order (stable sort); non-finite scores sort to the bottom. `k == 0`
/// returns empty.
pub fn rank_top_k(query: &[f32], items: &[(usize, Vec<f32>)], k: usize) -> Vec<(usize, f32)> {
    if query.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> = items
        .iter()
        .map(|(id, v)| {
            debug_assert_eq!(query.len(), v.len(), "embedding dim mismatch");
            (*id, cosine_unit(query, v))
        })
        .collect();
    // Descending; non-finite scores sink to the bottom (shared comparator) so a
    // degenerate embedding can't make the sort non-transitive.
    scored.sort_by(|a, b| minutist_common::voiceprint_math::cmp_desc_finite_first(a.1, b.1));
    scored.truncate(k);
    scored
}

/// Reciprocal Rank Fusion of several ranked lists into one.
///
/// Each leg in `legs` is a list of opaque keys already sorted best-first (e.g.
/// the dense cosine ranking and the lexical `bm25` ranking of the same corpus).
/// A key's fused score is `Σ_legs 1 / (RRF_K + rank)` over the legs it appears in
/// (`rank` 1-based), which rewards keys ranked highly by *either* leg without
/// mixing the two incomparable score scales. Returns the distinct keys best-first,
/// truncated to `k`; ties keep first-seen order (stable). `k == 0` returns empty.
pub fn rrf_fuse<K: Clone + Eq + std::hash::Hash>(legs: &[&[K]], k: usize) -> Vec<K> {
    /// The standard RRF damping constant; larger flattens the rank weighting.
    const RRF_K: f32 = 60.0;
    let mut scores: std::collections::HashMap<K, f32> = std::collections::HashMap::new();
    let mut order: Vec<K> = Vec::new(); // first-seen order, for a stable tie-break
    for leg in legs {
        for (rank, key) in leg.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            scores
                .entry(key.clone())
                .and_modify(|s| *s += contribution)
                .or_insert_with(|| {
                    order.push(key.clone());
                    contribution
                });
        }
    }
    order.sort_by(|a, b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        unit_normalise(&mut v);
        v
    }

    #[test]
    fn ranks_most_similar_first() {
        let query = unit(vec![1.0, 0.0, 0.0]);
        let items = vec![
            (10, unit(vec![0.0, 1.0, 0.0])), // orthogonal → ~0
            (11, unit(vec![1.0, 0.1, 0.0])), // close → high
            (12, unit(vec![-1.0, 0.0, 0.0])), // opposite → ~-1
        ];
        let ranked = rank_top_k(&query, &items, 3);
        assert_eq!(ranked[0].0, 11, "closest vector ranks first");
        assert_eq!(ranked[2].0, 12, "opposite vector ranks last");
    }

    #[test]
    fn top_k_truncates() {
        let query = unit(vec![1.0, 0.0]);
        let items = vec![
            (1, unit(vec![1.0, 0.0])),
            (2, unit(vec![0.9, 0.1])),
            (3, unit(vec![0.0, 1.0])),
        ];
        assert_eq!(rank_top_k(&query, &items, 2).len(), 2);
        assert_eq!(rank_top_k(&query, &items, 0).len(), 0);
    }

    #[test]
    fn non_finite_score_sinks_to_bottom() {
        // A NaN cosine (degenerate embedding) must not corrupt the ordering of the
        // finite items — it sorts last, not somewhere in the middle.
        let query = unit(vec![1.0, 0.0]);
        let items = vec![
            (1, vec![f32::NAN, f32::NAN]), // NaN score
            (2, unit(vec![1.0, 0.0])),     // best finite match
            (3, unit(vec![0.0, 1.0])),     // orthogonal
        ];
        let ranked = rank_top_k(&query, &items, 3);
        assert_eq!(ranked[0].0, 2, "finite best ranks first despite the NaN item");
        assert_eq!(ranked[2].0, 1, "non-finite item sinks to the bottom");
    }

    #[test]
    fn empty_query_returns_empty() {
        let items = vec![(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])];
        assert!(rank_top_k(&[], &items, 5).is_empty());
    }

    #[test]
    fn rrf_fuse_rewards_agreement_and_dedupes() {
        // "b" is ranked highly by both legs → it should win; keys stay distinct.
        let dense = ["a", "b", "c"];
        let lexical = ["b", "d", "a"];
        let fused = rrf_fuse(&[&dense[..], &lexical[..]], 3);
        assert_eq!(fused[0], "b", "agreement across both legs ranks first");
        assert_eq!(fused.len(), 3);
        let mut distinct = fused.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), fused.len(), "no duplicate keys");
    }

    #[test]
    fn rrf_fuse_k_zero_and_single_leg() {
        assert!(rrf_fuse(&[&["a", "b"][..]], 0).is_empty());
        // A single leg passes through in order.
        assert_eq!(rrf_fuse(&[&["x", "y"][..]], 5), vec!["x", "y"]);
    }
}
