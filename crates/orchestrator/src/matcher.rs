//! Voiceprint matcher — global greedy assignment of fresh diarizer clusters to
//! stored gallery identities (issue #0003, §2.4 + §2.9.1).
//!
//! # Design
//!
//! Matching is a **global assignment problem**, not independent per-cluster
//! thresholding. A first-over-threshold approach would allow two clusters to
//! both claim one identity, or one identity to be claimed by two clusters.
//!
//! The algorithm is a greedy assignment with a margin requirement:
//! 1. Score every `(query_label, identity)` pair by the maximum cosine over the
//!    identity's gallery centroids (§2.9.1 flat-gallery rule).
//! 2. Sort all pairs by descending score.
//! 3. In score order, assign a pair only if (a) neither the query label nor the
//!    identity has already been assigned, (b) the score is `>= T_reject`, and
//!    (c) the score beats the next-best unassigned score for that query by at
//!    least `MIN_MARGIN`.
//!
//! # Threshold constants (§2.4 — placeholders pending WU6)
//!
//! `T_ACCEPT`, `T_REJECT`, and `MIN_MARGIN` are **documented placeholders** with
//! no grounding in an in-repo corpus sweep. WU6 assembles the labelled
//! multi-session corpus and calibrates them. The names are intentionally stable
//! so the WU6 calibration can swap the values without changing call sites.
//!
//! # Query-side noise guard
//!
//! A fresh cluster built from few, short segments is noisy — exactly the case
//! where a spurious high cosine is most likely. When `query_window_count` is
//! below `NOISE_GUARD_MIN_WINDOWS`, the effective accept threshold is raised
//! to `T_ACCEPT_NOISY` (a tighter threshold), so a noisy query cannot auto-accept
//! and falls to the uncertain band or reject band instead.
//!
//! # Dependency
//!
//! `matcher` sees only `common::voiceprint_math` (via direct `&[f32]` slices) and
//! the `persistence::StoredVoiceprint` type (which `orchestrator` is already
//! allowed to use — it owns the `persistence` dependency). No `diarizer` import
//! here; the caller passes a plain `&[f32]` query centroid extracted upstream.

use std::collections::{HashMap, HashSet};

use minutist_common::VoiceprintIdentityId;
use persistence::StoredVoiceprint;

// ---------------------------------------------------------------------------
// Threshold constants (§2.4 — placeholders pending WU6 calibration)
// ---------------------------------------------------------------------------

/// Cosine similarity floor for auto-accepting a match and applying the name.
///
/// A false-accept (labelling a stranger as a known person) is worse than a miss,
/// so `T_ACCEPT` is tuned for a low false-accept rate (impostor 99th-percentile
/// bound), NOT at EER. **Placeholder — WU6 calibrates from a multi-session corpus.**
pub const T_ACCEPT: f32 = 0.60;

/// Cosine similarity below which a match is unconditionally rejected.
///
/// A similarity in `[T_REJECT, T_ACCEPT)` enters the uncertain band: the name is
/// *suggested* but not auto-applied (the UI shows "is this <Name>?"). Below
/// `T_REJECT` the cluster gets no name at all.
///
/// **Placeholder — WU6 calibrates as the genuine 5th percentile.**
pub const T_REJECT: f32 = 0.45;

/// Noisy-query accept threshold: used in place of `T_ACCEPT` when a fresh
/// cluster has fewer than `NOISE_GUARD_MIN_WINDOWS` clean windows (§2.4 query-
/// side noise guard). A higher bar prevents a noisy centroid from auto-accepting.
///
/// **Placeholder — WU6 calibrates.**
pub const T_ACCEPT_NOISY: f32 = 0.70;

/// Minimum number of clean PCM windows behind a fresh cluster centroid before
/// the normal `T_ACCEPT` bar applies. Below this count, `T_ACCEPT_NOISY` is
/// used instead.
///
/// **Placeholder — WU6 calibrates.**
pub const NOISE_GUARD_MIN_WINDOWS: u64 = 3;

/// Minimum cosine margin by which a winning match must beat the runner-up
/// candidate for the same query label. Mirrors the `LOGPROB_EPSILON` margin
/// pattern in the ASR runtime: prevents two similarly-scored identities from
/// both grabbing one cluster.
///
/// **Placeholder — WU6 calibrates.**
pub const MIN_MARGIN: f32 = 0.05;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Band decision for one (query label → identity) match.
#[derive(Debug, Clone, PartialEq)]
pub enum MatchBand {
    /// `sim >= T_ACCEPT` (or `T_ACCEPT_NOISY` for noisy queries): auto-apply
    /// the name.
    Accept,
    /// `T_REJECT <= sim < T_ACCEPT`: suggest the name with an "is this <Name>?"
    /// affordance. The label stays anonymous until the user confirms.
    Uncertain,
    /// `sim < T_REJECT`: no name; the cluster stays as its bare diarizer letter.
    Reject,
}

/// One assigned match from [`assign_identities`].
#[derive(Debug, Clone)]
pub struct AssignedMatch {
    /// The diarizer label (e.g. `"A"`, `"B"`) for the fresh cluster.
    pub query_label: String,
    /// The matched stored identity.
    pub identity_id: VoiceprintIdentityId,
    /// Display name of the matched identity at match time.
    pub display_name: String,
    /// The cosine similarity of the best gallery centroid for this identity
    /// against the query centroid.
    pub similarity: f32,
    /// The band decision (accept / uncertain / reject).
    pub band: MatchBand,
}

/// Input descriptor for one fresh diarizer cluster.
#[derive(Debug, Clone)]
pub struct QueryCluster {
    /// The diarizer label (e.g. `"A"`) assigned by the current diarization pass.
    pub label: String,
    /// The L2-unit-normalised centroid vector for this cluster.
    pub centroid: Vec<f32>,
    /// Number of clean PCM windows used to build `centroid`. Used by the
    /// query-side noise guard (§2.4): a low count raises the effective accept
    /// threshold.
    pub window_count: u64,
}

// ---------------------------------------------------------------------------
// Per-cluster match (collisions allowed) — merge-pass support
// ---------------------------------------------------------------------------

/// Per-cluster best-identity match, with collisions allowed.
///
/// Unlike [`assign_identities`] (which is a global injective assignment — no
/// identity can be claimed by two clusters), this function returns the
/// **independent argmax** for each query cluster: the identity with the highest
/// cosine over its gallery centroids, accepted only when:
///
/// - `sim >= T_ACCEPT` (or `T_ACCEPT_NOISY` when `window_count <
///   NOISE_GUARD_MIN_WINDOWS`), AND
/// - the winning identity beats the runner-up identity for the same query by at
///   least `MIN_MARGIN`.
///
/// Two clusters can independently match the **same** identity (a collision);
/// that is exactly the signal the library-informed merge pass needs to detect
/// two diarizer clusters that belong to one enrolled speaker.
///
/// Returns one entry per query cluster: `(query_label, Option<(identity_id,
/// sim)>)`. An entry with `None` means no identity cleared the accept threshold
/// + margin for that cluster.
///
/// Pure function; reuses the same threshold constants as `assign_identities`.
/// No new dependency edge (already in the orchestrator).
pub fn match_each_cluster(
    queries: &[QueryCluster],
    gallery: &[StoredVoiceprint],
) -> Vec<(String, Option<(VoiceprintIdentityId, f32)>)> {
    if queries.is_empty() || gallery.is_empty() {
        return queries
            .iter()
            .map(|q| (q.label.clone(), None))
            .collect();
    }

    // Group gallery centroids by identity (same as assign_identities).
    let mut identity_centroids: HashMap<VoiceprintIdentityId, Vec<&[f32]>> = HashMap::new();
    for entry in gallery {
        identity_centroids
            .entry(entry.identity_id)
            .or_default()
            .push(&entry.embedding);
    }

    let mut result: Vec<(String, Option<(VoiceprintIdentityId, f32)>)> =
        Vec::with_capacity(queries.len());

    for query in queries {
        if query.centroid.is_empty() {
            result.push((query.label.clone(), None));
            continue;
        }

        // Score every identity independently for this query.
        let mut scores: Vec<(VoiceprintIdentityId, f32)> = identity_centroids
            .iter()
            .filter_map(|(&id, centroids)| {
                let best_sim = centroids
                    .iter()
                    .map(|c| {
                        if c.is_empty() {
                            -1.0f32
                        } else {
                            minutist_common::voiceprint_math::cosine_unit(&query.centroid, c)
                        }
                    })
                    .fold(f32::NEG_INFINITY, f32::max);
                // Only keep candidates above the reject floor (T_REJECT).
                if best_sim >= T_REJECT {
                    Some((id, best_sim))
                } else {
                    None
                }
            })
            .collect();

        if scores.is_empty() {
            result.push((query.label.clone(), None));
            continue;
        }

        // Sort descending so [0] is the winner and [1] (if any) is the runner-up.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (winner_id, winner_sim) = scores[0];
        let runner_up_sim = scores.get(1).map(|s| s.1).unwrap_or(0.0f32);
        let margin = winner_sim - runner_up_sim.max(0.0);

        if margin < MIN_MARGIN {
            // Two identities score too similarly — cannot decide; skip.
            result.push((query.label.clone(), None));
            continue;
        }

        // Query-side noise guard.
        let effective_accept = if query.window_count < NOISE_GUARD_MIN_WINDOWS {
            T_ACCEPT_NOISY
        } else {
            T_ACCEPT
        };

        if winner_sim >= effective_accept {
            result.push((query.label.clone(), Some((winner_id, winner_sim))));
        } else {
            // In the uncertain band — not an accept; skip for merge purposes.
            result.push((query.label.clone(), None));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// Assign fresh diarizer clusters to stored gallery identities (§2.4 / §2.9.1).
///
/// Each query in `queries` is matched against every centroid in `gallery`
/// (already filtered to the active `model_id` by the caller via
/// `VoiceprintStore::all(model_id)`). Identity score = max cosine over the
/// identity's centroids (§2.9.1 flat-gallery rule).
///
/// Returns one [`AssignedMatch`] per query that reaches at least `T_REJECT`.
/// Queries below `T_REJECT` for every identity are silently omitted — they
/// stay anonymous.
///
/// The assignment is injective (one-to-one): no identity can be claimed by two
/// queries, and no query can be assigned to two identities.
///
/// # Argument
///
/// - `queries`: the fresh clusters from the current diarization pass.
/// - `gallery`: the stored gallery from `VoiceprintStore::all(model_id)`, already
///   filtered to the embedding model in use.
pub fn assign_identities(
    queries: &[QueryCluster],
    gallery: &[StoredVoiceprint],
) -> Vec<AssignedMatch> {
    if queries.is_empty() || gallery.is_empty() {
        return Vec::new();
    }

    // Build a flat (query_idx, identity_id, identity_name, best_sim) table.
    // Identity score = max cosine over that identity's gallery centroids (§2.9.1).
    struct Candidate {
        query_idx: usize,
        identity_id: VoiceprintIdentityId,
        display_name: String,
        similarity: f32,
    }

    // Group gallery centroids by identity so we can take max over them.
    let mut identity_centroids: HashMap<VoiceprintIdentityId, (String, Vec<&[f32]>)> =
        HashMap::new();
    for entry in gallery {
        identity_centroids
            .entry(entry.identity_id)
            .or_insert_with(|| (entry.display_name.clone(), Vec::new()))
            .1
            .push(&entry.embedding);
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    for (q_idx, query) in queries.iter().enumerate() {
        if query.centroid.is_empty() {
            continue;
        }
        for (identity_id, (display_name, centroids)) in &identity_centroids {
            // Identity score = max cosine over its gallery centroids.
            let best_sim = centroids
                .iter()
                .map(|c| {
                    if c.is_empty() {
                        -1.0f32
                    } else {
                        minutist_common::voiceprint_math::cosine_unit(&query.centroid, c)
                    }
                })
                .fold(f32::NEG_INFINITY, f32::max);

            if best_sim >= T_REJECT {
                candidates.push(Candidate {
                    query_idx: q_idx,
                    identity_id: *identity_id,
                    display_name: display_name.clone(),
                    similarity: best_sim,
                });
            }
        }
    }

    // Sort descending by similarity for greedy assignment.
    candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

    let mut assigned_queries: HashSet<usize> = HashSet::new();
    let mut assigned_identities: HashSet<VoiceprintIdentityId> = HashSet::new();
    let mut result: Vec<AssignedMatch> = Vec::new();

    for candidate in &candidates {
        // Skip if already assigned.
        if assigned_queries.contains(&candidate.query_idx)
            || assigned_identities.contains(&candidate.identity_id)
        {
            continue;
        }

        let query = &queries[candidate.query_idx];

        // Margin check: find the runner-up score for the same query among
        // unassigned identities (other than this one).
        let runner_up = candidates
            .iter()
            .filter(|c| {
                c.query_idx == candidate.query_idx
                    && c.identity_id != candidate.identity_id
                    && !assigned_identities.contains(&c.identity_id)
            })
            .map(|c| c.similarity)
            .fold(f32::NEG_INFINITY, f32::max);

        let margin = candidate.similarity - runner_up.max(0.0);
        if margin < MIN_MARGIN {
            // The winner doesn't beat the runner-up by enough; skip this
            // assignment. (The runner-up for this query, if any, will be
            // evaluated in a later iteration.)
            continue;
        }

        // Query-side noise guard: low-count centroids use T_ACCEPT_NOISY.
        let effective_accept = if query.window_count < NOISE_GUARD_MIN_WINDOWS {
            T_ACCEPT_NOISY
        } else {
            T_ACCEPT
        };

        let band = if candidate.similarity >= effective_accept {
            MatchBand::Accept
        } else {
            // similarity >= T_REJECT already enforced at candidate-building time.
            MatchBand::Uncertain
        };

        assigned_queries.insert(candidate.query_idx);
        assigned_identities.insert(candidate.identity_id);

        result.push(AssignedMatch {
            query_label: query.label.clone(),
            identity_id: candidate.identity_id,
            display_name: candidate.display_name.clone(),
            similarity: candidate.similarity,
            band,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::{VoiceprintCentroidId, VoiceprintIdentityId};
    use persistence::StoredVoiceprint;

    /// Build a unit-normalised vector pointing at angle `theta_rad` in the
    /// first two dimensions.
    fn unit_vec2(theta_rad: f32) -> Vec<f32> {
        let mut v = vec![theta_rad.cos(), theta_rad.sin()];
        minutist_common::voiceprint_math::unit_normalise(&mut v);
        v
    }

    /// Build a [`QueryCluster`] with a fixed direction vector and window count.
    fn qc(label: &str, theta_rad: f32, windows: u64) -> QueryCluster {
        QueryCluster {
            label: label.to_string(),
            centroid: unit_vec2(theta_rad),
            window_count: windows,
        }
    }

    /// Build a [`StoredVoiceprint`] for an identity that has a single centroid.
    fn gal_entry(
        identity_id: VoiceprintIdentityId,
        centroid_id: VoiceprintCentroidId,
        name: &str,
        theta_rad: f32,
    ) -> StoredVoiceprint {
        StoredVoiceprint {
            identity_id,
            centroid_id,
            display_name: name.to_string(),
            model_id: "test".to_string(),
            embedding: unit_vec2(theta_rad),
            dim: 2,
            sample_count: 10,
            condition_label: None,
        }
    }

    // Fixed-UUID test identity helpers — parse known nil-adjacent UUIDs so
    // comparisons are stable across calls within a single test.
    fn id_alice() -> VoiceprintIdentityId {
        minutist_common::VoiceprintIdentityId(
            "00000000-0000-0000-0000-000000000001".parse().expect("uuid"),
        )
    }
    fn id_bob() -> VoiceprintIdentityId {
        minutist_common::VoiceprintIdentityId(
            "00000000-0000-0000-0000-000000000002".parse().expect("uuid"),
        )
    }
    fn cid(n: u64) -> VoiceprintCentroidId {
        minutist_common::VoiceprintCentroidId(
            format!("00000000-0000-0000-0000-{n:012x}")
                .parse()
                .expect("valid uuid"),
        )
    }

    // -----------------------------------------------------------------------
    // Basic accept: one query, one identity, well above T_ACCEPT
    // -----------------------------------------------------------------------

    #[test]
    fn single_accept_match() {
        let queries = vec![qc("A", 0.0, 10)];
        let gallery = vec![gal_entry(id_alice(), cid(1), "Alice", 0.0)];

        let matches = assign_identities(&queries, &gallery);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].query_label, "A");
        assert_eq!(matches[0].display_name, "Alice");
        assert_eq!(matches[0].band, MatchBand::Accept);
    }

    // -----------------------------------------------------------------------
    // Reject: query is far from all identities (sim < T_REJECT)
    // -----------------------------------------------------------------------

    #[test]
    fn query_below_t_reject_is_omitted() {
        use std::f32::consts::PI;
        // Alice is at 0°; query is at 90° (cosine = 0.0, well below T_REJECT = 0.45).
        let queries = vec![qc("A", PI / 2.0, 10)];
        let gallery = vec![gal_entry(id_alice(), cid(1), "Alice", 0.0)];

        let matches = assign_identities(&queries, &gallery);
        assert!(matches.is_empty(), "query below T_REJECT must not appear in results");
    }

    // -----------------------------------------------------------------------
    // Two clusters, one identity: global assignment prevents double-grab
    //
    // Both "A" and "B" are close to Alice. Only the better-scoring one (A,
    // closer) should win Alice; B must be left unmatched.
    // -----------------------------------------------------------------------

    #[test]
    fn two_clusters_one_identity_only_best_wins() {
        use std::f32::consts::PI;
        // Alice at 0°; A at 5° (very close), B at 15° (closer than T_REJECT but below A).
        let angle_a = 5.0_f32 * PI / 180.0;
        let angle_b = 15.0_f32 * PI / 180.0;
        let queries = vec![qc("A", angle_a, 10), qc("B", angle_b, 10)];
        let gallery = vec![gal_entry(id_alice(), cid(1), "Alice", 0.0)];

        let matches = assign_identities(&queries, &gallery);

        // At most one match (Alice can only be assigned once).
        assert!(
            matches.len() <= 1,
            "identity can only be assigned to one query"
        );
        if let Some(m) = matches.first() {
            // The assigned query must be "A" (closest to Alice).
            assert_eq!(m.query_label, "A");
            assert_eq!(m.identity_id, id_alice());
        }
    }

    // -----------------------------------------------------------------------
    // Margin check: two identities both above threshold for one query,
    // but too close together — the assignment is skipped.
    // -----------------------------------------------------------------------

    #[test]
    fn margin_too_small_drops_assignment() {
        use std::f32::consts::PI;
        // Query at 0°; Alice at 1°, Bob at 2° — both well above T_REJECT,
        // margin between them ≈ cos(1°) - cos(2°) ≈ 0.9998 - 0.9994 ≈ 0.0004
        // which is below MIN_MARGIN = 0.05.
        let queries = vec![qc("A", 0.0, 10)];
        let alice = gal_entry(id_alice(), cid(1), "Alice", 1.0_f32 * PI / 180.0);
        let bob = gal_entry(id_bob(), cid(2), "Bob", 2.0_f32 * PI / 180.0);
        let gallery = vec![alice, bob];

        let matches = assign_identities(&queries, &gallery);

        // Margin between Alice and Bob for query A is too small — neither wins.
        assert!(
            matches.is_empty(),
            "assignment must be dropped when margin is below MIN_MARGIN, got {:?}",
            matches.iter().map(|m| &m.query_label).collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------
    // Query-side noise guard: low-count query cannot auto-accept
    // -----------------------------------------------------------------------

    #[test]
    fn noisy_query_falls_to_uncertain_band() {
        // Similarity just above T_ACCEPT (0.60) but below T_ACCEPT_NOISY (0.70).
        // With window_count < NOISE_GUARD_MIN_WINDOWS, this should be Uncertain.
        //
        // cos(θ) = 0.63 => θ ≈ 50.9°
        let theta = 50.9_f32.to_radians();
        let queries = vec![qc(
            "A",
            theta,
            NOISE_GUARD_MIN_WINDOWS - 1, // below the guard
        )];
        let gallery = vec![gal_entry(id_alice(), cid(1), "Alice", 0.0)];

        let matches = assign_identities(&queries, &gallery);

        // The match should exist (sim > T_REJECT) but be Uncertain due to noise guard.
        // Whether it has a result depends on the margin — with a single identity there
        // is no runner-up so margin = candidate.similarity - 0.0 >= MIN_MARGIN for any
        // reasonable similarity. Check band.
        if let Some(m) = matches.first() {
            if m.similarity >= T_ACCEPT && m.similarity < T_ACCEPT_NOISY {
                assert_eq!(
                    m.band,
                    MatchBand::Uncertain,
                    "low-count query above T_ACCEPT but below T_ACCEPT_NOISY must be Uncertain"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Two queries, two identities: both get assigned (no conflict)
    // -----------------------------------------------------------------------

    #[test]
    fn two_queries_two_identities_both_assigned() {
        use std::f32::consts::PI;
        // Alice at 0°, Bob at 90° — orthogonal identities.
        // Query A at 5° (near Alice), Query B at 85° (near Bob).
        let queries = vec![qc("A", 5.0_f32 * PI / 180.0, 10), qc("B", 85.0_f32 * PI / 180.0, 10)];
        let alice = gal_entry(id_alice(), cid(1), "Alice", 0.0);
        let bob = gal_entry(id_bob(), cid(2), "Bob", PI / 2.0);
        let gallery = vec![alice, bob];

        let matches = assign_identities(&queries, &gallery);

        // Both should get assigned to distinct identities.
        assert_eq!(matches.len(), 2, "both queries should be assigned");
        let labels: Vec<&str> = matches.iter().map(|m| m.query_label.as_str()).collect();
        assert!(labels.contains(&"A"));
        assert!(labels.contains(&"B"));
        // A goes to Alice, B goes to Bob.
        for m in &matches {
            if m.query_label == "A" {
                assert_eq!(m.identity_id, id_alice());
            } else {
                assert_eq!(m.identity_id, id_bob());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Identity score = max over multiple gallery centroids (§2.9.1)
    // -----------------------------------------------------------------------

    #[test]
    fn identity_score_is_max_over_gallery_centroids() {
        use std::f32::consts::PI;
        // Alice has two centroids: one at 0° and one at 60°.
        // Query A is at 55° — closer to Alice's 60° centroid (cos(5°) ≈ 0.996)
        // than to her 0° centroid (cos(55°) ≈ 0.574). The identity score must
        // use the max, so Alice scores ≈ 0.996 against A.
        let angle_q = 55.0_f32 * PI / 180.0;
        let queries = vec![qc("A", angle_q, 10)];

        let alice_cond1 = gal_entry(id_alice(), cid(1), "Alice", 0.0); // 0°
        let alice_cond2 = gal_entry(id_alice(), cid(2), "Alice", PI / 3.0); // 60°
        let gallery = vec![alice_cond1, alice_cond2];

        let matches = assign_identities(&queries, &gallery);

        assert_eq!(matches.len(), 1);
        // Similarity should be the max-over-centroids value, not the 0°-centroid value.
        assert!(
            matches[0].similarity > T_ACCEPT,
            "identity score must use max over centroids, got sim = {}",
            matches[0].similarity
        );
        assert_eq!(matches[0].band, MatchBand::Accept);
    }

    // -----------------------------------------------------------------------
    // Empty inputs return empty results
    // -----------------------------------------------------------------------

    #[test]
    fn empty_queries_returns_empty() {
        let gallery = vec![gal_entry(id_alice(), cid(1), "Alice", 0.0)];
        let matches = assign_identities(&[], &gallery);
        assert!(matches.is_empty());
    }

    #[test]
    fn empty_gallery_returns_empty() {
        let queries = vec![qc("A", 0.0, 10)];
        let matches = assign_identities(&queries, &[]);
        assert!(matches.is_empty());
    }

    // -----------------------------------------------------------------------
    // match_each_cluster — collision-allowed per-cluster matcher
    // -----------------------------------------------------------------------

    /// Two clusters both close to the same identity: both must return that
    /// identity (collision allowed). This is the key difference from
    /// `assign_identities`, which would give it to only the closer one.
    #[test]
    fn match_each_cluster_two_clusters_same_identity() {
        use std::f32::consts::PI;
        // Alice at 0°; A at 5°, B at 10° — both well above T_ACCEPT.
        let queries = vec![
            qc("A", 5.0_f32 * PI / 180.0, 10),
            qc("B", 10.0_f32 * PI / 180.0, 10),
        ];
        let gallery = vec![gal_entry(id_alice(), cid(1), "Alice", 0.0)];

        let results = match_each_cluster(&queries, &gallery);
        assert_eq!(results.len(), 2);

        for (label, matched) in &results {
            let (id, _sim) = matched
                .as_ref()
                .unwrap_or_else(|| panic!("cluster {label} must match Alice"));
            assert_eq!(
                *id,
                id_alice(),
                "cluster {label} must match Alice (collision allowed)"
            );
        }
    }

    /// Two clusters matching DIFFERENT identities must NOT be merged — each
    /// gets its own distinct identity.
    #[test]
    fn match_each_cluster_two_clusters_different_identities() {
        use std::f32::consts::PI;
        // Alice at 0°, Bob at 90°; A near Alice, B near Bob.
        let queries = vec![
            qc("A", 5.0_f32 * PI / 180.0, 10),
            qc("B", 85.0_f32 * PI / 180.0, 10),
        ];
        let alice = gal_entry(id_alice(), cid(1), "Alice", 0.0);
        let bob = gal_entry(id_bob(), cid(2), "Bob", PI / 2.0);
        let gallery = vec![alice, bob];

        let results = match_each_cluster(&queries, &gallery);
        assert_eq!(results.len(), 2);

        let a_match = results.iter().find(|(l, _)| l == "A").and_then(|(_, m)| m.as_ref());
        let b_match = results.iter().find(|(l, _)| l == "B").and_then(|(_, m)| m.as_ref());

        assert!(a_match.is_some(), "A must match some identity");
        assert!(b_match.is_some(), "B must match some identity");
        assert_ne!(
            a_match.unwrap().0,
            b_match.unwrap().0,
            "A and B must match DIFFERENT identities"
        );
    }

    /// A cluster that matches no identity stays None.
    #[test]
    fn match_each_cluster_unenrolled_stays_none() {
        use std::f32::consts::PI;
        // Alice at 0°; query at 90° — cosine = 0.0, below T_REJECT.
        let queries = vec![qc("A", PI / 2.0, 10)];
        let gallery = vec![gal_entry(id_alice(), cid(1), "Alice", 0.0)];

        let results = match_each_cluster(&queries, &gallery);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].1.is_none(),
            "cluster far from all gallery entries must return None"
        );
    }

    /// Margin check applies per-cluster: two identities scoring too similarly
    /// for one cluster means neither is returned for that cluster.
    #[test]
    fn match_each_cluster_margin_too_small_drops_match() {
        use std::f32::consts::PI;
        // Alice at 1°, Bob at 2°; query at 0° — margin between Alice and Bob
        // is tiny (≈ cos(1°) - cos(2°) ≈ 0.0004), below MIN_MARGIN = 0.05.
        let queries = vec![qc("A", 0.0, 10)];
        let alice = gal_entry(id_alice(), cid(1), "Alice", 1.0_f32 * PI / 180.0);
        let bob = gal_entry(id_bob(), cid(2), "Bob", 2.0_f32 * PI / 180.0);
        let gallery = vec![alice, bob];

        let results = match_each_cluster(&queries, &gallery);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].1.is_none(),
            "margin below MIN_MARGIN must produce no match for that cluster"
        );
    }
}
