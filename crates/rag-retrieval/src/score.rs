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
    // Sort descending. A non-finite score (e.g. from a degenerate embedding)
    // would make `partial_cmp` non-transitive and scramble the whole ordering, so
    // map it to the bottom before comparing.
    let key = |s: f32| if s.is_finite() { s } else { f32::NEG_INFINITY };
    scored.sort_by(|a, b| key(b.1).partial_cmp(&key(a.1)).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
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
}
