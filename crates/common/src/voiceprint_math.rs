//! Pure, dependency-free vector maths for voiceprint centroids.
//!
//! Three functions consumed by two crates that must not form a direct edge:
//! - `diarizer` calls [`unit_normalise`] and [`cosine_unit`] when building
//!   per-cluster centroids from re-embedded audio windows.
//! - `persistence::VoiceprintStore` calls [`unit_normalise`] and
//!   [`weighted_merge`] when folding or merging the stored centroid cache in
//!   `voiceprints.db`.
//!
//! Both crates already depend on `common`, so housing the maths here adds
//! **no new dependency-table edge** (see `architecture/components.md`).
//!
//! ## Design reference
//!
//! § 2.9.2 of `planning/issues/0003-voiceprints-design.md` specifies the
//! three functions. Their behaviour intentionally mirrors the proven
//! `OnlineClusterer` maths in `crates/diarizer/src/online/clusterer.rs`
//! (the `cos > 0.999` centroid-aligns-with-sample-mean discipline), but as
//! standalone free functions that carry no clusterer state.
//!
//! ## Welford vs count-weighted merge
//!
//! [`weighted_merge`] is **not** the Welford one-observation running-mean
//! (`c += (u - c) / (n + 1)`) used by `OnlineClusterer::update_centroid`.
//! Welford grows a *single* centroid one embedding at a time; `weighted_merge`
//! combines N *already-established* centroids each backed by a different
//! observation count. The two operations are mathematically distinct:
//!
//! ```text
//! Welford:        c_new = c + (u - c) / (n + 1)
//! Weighted mean:  v = Σ count_i · centroid_i / Σ count_i, then unit-normalise v
//! ```
//!
//! `persistence` uses `weighted_merge` to recompute the cached centroid
//! from its stored contributions after a fold, add, or merge, making
//! refinement reversible (drop a contribution, call `weighted_merge` over the
//! survivors, store the result).

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// L2-normalise `v` in place. No-op when the norm is zero or non-finite
/// (a zero vector or a vector containing non-finite components).
///
/// After a successful normalisation every element satisfies
/// `v[i] = v[i] / ||v||` and `Σ v[i]² ≈ 1.0`.
///
/// The no-op-on-degenerate contract mirrors the reject-on-degenerate
/// behaviour of the private `unit_normalise` helper in
/// `diarizer::online::clusterer`, but as an in-place mutation (no
/// allocation) returning nothing — the caller already owns the buffer.
pub fn unit_normalise(v: &mut [f32]) {
    let sum_sq: f32 = v.iter().map(|&x| x * x).sum();
    let norm = sum_sq.sqrt();
    // Guard: reject the zero vector and any case where the squared-sum
    // overflows to +inf (still a finite but very large input) — both make
    // the cosine undefined, so leave v unchanged.
    if norm.is_finite() && norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two unit vectors `a` and `b`.
///
/// Both inputs are assumed to already be L2-normalised (i.e. `||a|| ≈ 1`
/// and `||b|| ≈ 1`). Under that assumption the cosine reduces to the plain
/// dot product: `cos(a, b) = Σ a[i] · b[i]`.
///
/// Lengths must match; if they differ the shorter slice governs (extra
/// elements in the longer are ignored). Returns `0.0` for empty slices.
///
/// This mirrors `cosine_unit_vs_centroid` in
/// `diarizer::online::clusterer` simplified for two unit inputs (no
/// centroid-norm division required).
pub fn cosine_unit(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Count-weighted mean of N established centroids, then L2-normalised.
///
/// Each element of `centroids` is a `(vector, count)` pair where `count`
/// is the number of observations behind that centroid. The result is:
///
/// ```text
/// v = Σ count_i · centroid_i  /  Σ count_i
/// unit_normalise(v)
/// ```
///
/// This is the merge operation for `VoiceprintStore`: after folding a new
/// contribution or re-homing contributions during a manual merge, the store
/// recomputes the cached `voiceprint_centroid.embedding` by calling this
/// function over its surviving contributions. Contributions with `count = 0`
/// contribute nothing and are handled gracefully (weight zero, skipped).
///
/// ## Not Welford
///
/// This is **not** the Welford one-observation running-mean
/// (`c += (u - c) / (n + 1)`) used by `OnlineClusterer`. Welford grows a
/// single centroid one embedding at a time; this function combines N
/// already-established centroid means with different observation counts.
/// Confusing the two leads to incorrect centroid caches — see the module-level
/// documentation for the distinction.
///
/// ## Degenerate inputs
///
/// - Empty `centroids` slice → returns an empty `Vec`.
/// - All counts zero → the weighted sum is the zero vector; the
///   [`unit_normalise`] no-op contract applies and the zero vector is returned.
/// - A single element → equivalent to a copy of that centroid followed by
///   [`unit_normalise`] (the existing centroid may already be unit-length, but
///   is renormalised for safety).
pub fn weighted_merge(centroids: &[(&[f32], u64)]) -> Vec<f32> {
    // Guard: nothing to merge.
    if centroids.is_empty() {
        return Vec::new();
    }
    let dim = centroids[0].0.len();
    if dim == 0 {
        return Vec::new();
    }

    let mut acc = vec![0.0f32; dim];
    let mut total_count = 0u64;

    for &(vec, count) in centroids {
        if count == 0 {
            continue;
        }
        let w = count as f32;
        for (a, &v) in acc.iter_mut().zip(vec.iter()) {
            *a += w * v;
        }
        total_count += count;
    }

    if total_count > 0 {
        let inv = 1.0 / (total_count as f32);
        for a in acc.iter_mut() {
            *a *= inv;
        }
    }
    // L2-normalise; no-op on the zero vector (all counts zero).
    unit_normalise(&mut acc);
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for float comparisons.
    const EPS: f32 = 1e-5;

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    // --- unit_normalise -------------------------------------------------------

    #[test]
    fn unit_normalise_produces_unit_vector() {
        let mut v = [3.0_f32, 4.0];
        unit_normalise(&mut v);
        // Expected: [0.6, 0.8]
        assert!((norm(&v) - 1.0).abs() < EPS);
        assert!((v[0] - 0.6).abs() < EPS);
        assert!((v[1] - 0.8).abs() < EPS);
    }

    #[test]
    fn unit_normalise_no_op_on_zero_vector() {
        let mut v = [0.0_f32, 0.0, 0.0];
        unit_normalise(&mut v);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn unit_normalise_no_op_on_non_finite_norm() {
        // A vector whose squared-sum overflows to +inf.
        let big = f32::MAX;
        let mut v = [big, big];
        unit_normalise(&mut v);
        // Norm overflows to +inf — leave unchanged.
        assert_eq!(v, [big, big]);
    }

    // --- cosine_unit ----------------------------------------------------------

    #[test]
    fn cosine_unit_identical_returns_one() {
        let a = [1.0_f32, 0.0, 0.0];
        assert!((cosine_unit(&a, &a) - 1.0).abs() < EPS);
    }

    #[test]
    fn cosine_unit_orthogonal_returns_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(cosine_unit(&a, &b).abs() < EPS);
    }

    #[test]
    fn cosine_unit_opposite_returns_minus_one() {
        let a = [1.0_f32, 0.0];
        let b = [-1.0_f32, 0.0];
        assert!((cosine_unit(&a, &b) - (-1.0)).abs() < EPS);
    }

    // --- running_mean_centroid_aligns_with_sample_mean (cos > 0.999) ----------
    //
    // Retargets the `running_mean_centroid_drift` discipline from
    // `diarizer::online::clusterer` at the `common` module: the count-weighted
    // mean of unit vectors must have cos > 0.999 to the plain mean of those
    // same unit vectors.

    #[test]
    fn weighted_merge_aligns_with_plain_mean_cos_gt_0999() {
        // Unit-normalise four slightly-varying 2-D vectors (mirrors the
        // clusterer test) then build a weighted_merge and check alignment.
        let raw: &[[f32; 2]] = &[
            [1.0, 0.0],
            [0.96, 0.28],
            [0.94, 0.34],
            [0.92, 0.39],
        ];

        // Unit-normalise each raw sample.
        let units: Vec<[f32; 2]> = raw
            .iter()
            .map(|s| {
                let n = norm(s);
                [s[0] / n, s[1] / n]
            })
            .collect();

        // Plain arithmetic mean of the unit vectors.
        let mut plain_mean = [0.0_f32; 2];
        for u in &units {
            plain_mean[0] += u[0];
            plain_mean[1] += u[1];
        }
        plain_mean[0] /= units.len() as f32;
        plain_mean[1] /= units.len() as f32;

        // weighted_merge with equal counts of 1.
        let pairs: Vec<(&[f32], u64)> = units.iter().map(|u| (u.as_slice(), 1u64)).collect();
        let merged = weighted_merge(&pairs);

        // Cosine between merged and plain_mean must exceed 0.999.
        let dot: f32 = merged
            .iter()
            .zip(plain_mean.iter())
            .map(|(&m, &p)| m * p)
            .sum();
        let mn = norm(&plain_mean);
        let cos = dot / (norm(&merged) * mn);

        assert!(
            cos > 0.999,
            "weighted_merge should align with the plain unit mean, cos = {cos}"
        );
    }

    // --- weighted_merge equal-count test (two means → their plain unit-mean) --

    #[test]
    fn weighted_merge_equal_counts_gives_plain_unit_mean() {
        // Two unit vectors at 0° and 90°.
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];

        let merged = weighted_merge(&[(&a, 5), (&b, 5)]);

        // Expected: unit normalised [0.5, 0.5] = [√2/2, √2/2]
        let expected = 2.0_f32.sqrt() / 2.0;
        assert!((merged[0] - expected).abs() < EPS, "merged[0] = {}", merged[0]);
        assert!((merged[1] - expected).abs() < EPS, "merged[1] = {}", merged[1]);
        // Result must be unit length.
        assert!((norm(&merged) - 1.0).abs() < EPS);
    }

    #[test]
    fn weighted_merge_unequal_counts_weights_heavier_centroid() {
        // a has weight 3, b has weight 1; result should sit closer to a.
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];

        let merged = weighted_merge(&[(&a, 3), (&b, 1)]);

        // raw mean = [0.75, 0.25]; unit = [3/√10, 1/√10]
        let expected_x = 3.0_f32 / 10.0_f32.sqrt();
        let expected_y = 1.0_f32 / 10.0_f32.sqrt();
        assert!((merged[0] - expected_x).abs() < EPS, "merged[0] = {}", merged[0]);
        assert!((merged[1] - expected_y).abs() < EPS, "merged[1] = {}", merged[1]);
        assert!((norm(&merged) - 1.0).abs() < EPS);
    }

    #[test]
    fn weighted_merge_empty_returns_empty() {
        let result = weighted_merge(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn weighted_merge_zero_count_entry_ignored() {
        // One valid entry with count 2, one dummy with count 0.
        let a = [1.0_f32, 0.0];
        let dummy = [0.0_f32, 1.0];

        let merged = weighted_merge(&[(&a, 2), (&dummy, 0)]);

        // dummy is skipped; result is just unit(a) = a itself.
        assert!((merged[0] - 1.0).abs() < EPS);
        assert!((merged[1] - 0.0).abs() < EPS);
    }

    #[test]
    fn weighted_merge_single_element_is_unit_normalised() {
        // A non-unit input through a single-entry merge is normalised.
        let v = [3.0_f32, 4.0]; // ||v|| = 5
        let merged = weighted_merge(&[(&v, 1)]);
        assert!((merged[0] - 0.6).abs() < EPS);
        assert!((merged[1] - 0.8).abs() < EPS);
        assert!((norm(&merged) - 1.0).abs() < EPS);
    }

    // =========================================================================
    // Additional tests for issue #0003 voiceprint mechanism
    // =========================================================================
    //
    // These tests verify behaviour critical to the refinement + cross-condition
    // identity model (§2.9 of planning/issues/0003-voiceprints-design.md):
    // - idempotence under repeated normalisation
    // - cosine properties on unit vectors
    // - weighted_merge math for incremental refinement and cap-and-merge
    // - the one-bad-meeting bounded-weight poison test

    // --- unit_normalise: idempotence test ------------------------------------

    #[test]
    fn unit_normalise_is_idempotent() {
        // Normalising an already-unit vector should leave it unchanged (within
        // floating-point tolerance). This test verifies the identity property
        // needed during incremental refinement when a contribution's centroid
        // is already unit-normalised.
        let mut v = [0.6_f32, 0.8];

        unit_normalise(&mut v);
        let after_first = v;

        unit_normalise(&mut v);
        let after_second = v;

        // After first normalise: norm should be 1.
        assert!((norm(&after_first) - 1.0).abs() < EPS);

        // After second normalise: should be unchanged.
        assert!((after_first[0] - after_second[0]).abs() < EPS);
        assert!((after_first[1] - after_second[1]).abs() < EPS);
    }

    // --- cosine_unit: full-dim 192-element test (CAM++ embedding dim) --------

    #[test]
    fn cosine_unit_high_dimension_identical_vectors() {
        // CAM++ embeddings are 192-dimensional. Verify that cosine returns ~1.0
        // for identical unit vectors at that scale.
        let dim = 192usize;
        let mut v = vec![1.0_f32 / (dim as f32).sqrt(); dim];
        unit_normalise(&mut v);

        let cosine = cosine_unit(&v, &v);
        assert!(
            (cosine - 1.0).abs() < 1e-4,
            "cosine_unit of identical 192-D unit vector should be ~1.0, got {cosine}"
        );
    }

    #[test]
    fn cosine_unit_partial_slice_match() {
        // If slices are different lengths, shorter governs. Verify both elements
        // are counted correctly, not that the mismatch causes a panic.
        let a = [0.6_f32, 0.8, 0.0];
        let b = [0.6_f32, 0.8]; // Shorter slice

        let cos = cosine_unit(&a, &b);
        // cos = 0.6*0.6 + 0.8*0.8 = 0.36 + 0.64 = 1.0
        assert!((cos - 1.0).abs() < EPS);
    }

    #[test]
    fn cosine_unit_empty_slices() {
        let empty: &[f32] = &[];
        assert_eq!(cosine_unit(empty, empty), 0.0);
        assert_eq!(cosine_unit(&[1.0, 0.0], empty), 0.0);
        assert_eq!(cosine_unit(empty, &[1.0, 0.0]), 0.0);
    }

    // --- weighted_merge: refinement + cap-and-merge scenarios ----------------

    #[test]
    fn weighted_merge_many_equal_contributions() {
        // Refinement scenario: fold several meetings' contributions with equal
        // weight. Result should be their unit-mean.
        let contrib1 = [1.0_f32, 0.0, 0.0];
        let contrib2 = [0.0_f32, 1.0, 0.0];
        let contrib3 = [0.0_f32, 0.0, 1.0];

        let merged = weighted_merge(&[
            (&contrib1, 1),
            (&contrib2, 1),
            (&contrib3, 1),
        ]);

        // Raw mean = [1/3, 1/3, 1/3]; unit = [1/√3, 1/√3, 1/√3]
        let expected = 1.0_f32 / 3.0_f32.sqrt();
        for &elem in &merged {
            assert!((elem - expected).abs() < EPS);
        }
        assert!((norm(&merged) - 1.0).abs() < EPS);
    }

    #[test]
    fn weighted_merge_cap_and_merge_scenario() {
        // When merging two centroids that are already established (cap-and-merge
        // scenario in §2.9.3), weighted_merge correctly produces their
        // weighted-mean representative.
        //
        // Two established centroids: one built from 10 observations,
        // another from 5. Their weighted mean should favour the first.
        let centroid_a = [1.0_f32, 0.0]; // First principal direction
        let centroid_b = [0.0_f32, 1.0]; // Second principal direction

        let merged = weighted_merge(&[(&centroid_a, 10), (&centroid_b, 5)]);

        // Raw weighted mean: (10*[1,0] + 5*[0,1]) / 15 = [2/3, 1/3]
        // Norm = sqrt((2/3)^2 + (1/3)^2) = sqrt(4/9 + 1/9) = sqrt(5/9) = sqrt(5)/3
        // Unit normalised: [2/√5, 1/√5]
        let sqrt5 = 5.0_f32.sqrt();
        let expected_x = 2.0_f32 / sqrt5;
        let expected_y = 1.0_f32 / sqrt5;

        assert!(
            (merged[0] - expected_x).abs() < EPS,
            "merged[0] = {}, expected {expected_x}",
            merged[0]
        );
        assert!(
            (merged[1] - expected_y).abs() < EPS,
            "merged[1] = {}, expected {expected_y}",
            merged[1]
        );
        assert!((norm(&merged) - 1.0).abs() < EPS);
    }

    #[test]
    fn weighted_merge_respects_unit_normalisation() {
        // The result of weighted_merge must always be unit-normalised
        // (within floating-point tolerance), regardless of input magnitudes.
        let non_unit_a = [3.0_f32, 4.0]; // ||v|| = 5
        let non_unit_b = [5.0_f32, 12.0]; // ||v|| = 13

        let merged = weighted_merge(&[(&non_unit_a, 7), (&non_unit_b, 3)]);
        assert!(
            (norm(&merged) - 1.0).abs() < 1e-4,
            "Result must be unit-normalised, norm = {}",
            norm(&merged)
        );
    }

    // --- one-bad-meeting poison defence (§2.9.3 test) -------------------------

    #[test]
    fn weighted_merge_one_bad_meeting_bounded_weight_defence() {
        // §2.9.3: "an established centroid at sample_count = N (large), refined
        // once with an adversarial near-T_accept contribution, must not move
        // enough to cross T_accept for a held-out impostor."
        //
        // Setup: established centroid (many meetings) at [1, 0], an impostor
        // at [0, 1]. Add a single malicious contribution that sits in the
        // T_accept band (sim ~0.60) and use a bounded weight cap to prevent
        // it from poisoning the mean.

        // Established centroid: built from 100 observations of [1, 0].
        let established = [1.0_f32, 0.0];

        // Impostor: orthogonal direction.
        let impostor = [0.0_f32, 1.0];

        // Malicious contribution: near-T_accept similar to both.
        // Using [√2/2, √2/2] which has cosine ~0.707 to both.
        let malicious = [2.0_f32.sqrt() / 2.0, 2.0_f32.sqrt() / 2.0];

        // Without a weight cap, if the malicious contribution had count = 50,
        // it would significantly shift the centroid:
        // raw mean = (100*[1,0] + 50*[√2/2, √2/2]) / 150
        //          ≈ [0.737, 0.245]
        // This would have cosine ~0.737 to both established and impostor.
        //
        // With a weight cap (e.g., min(50, 20) = 20 relative to 100), the
        // shift is much smaller:
        // clamped: (100*[1,0] + 20*[√2/2, √2/2]) / 120
        //        ≈ [0.869, 0.119]
        // Cosine to [0,1] drops to ~0.137, well below T_accept (0.60).

        // Merge with uncapped weight (worst case).
        let uncapped = weighted_merge(&[(&established, 100), (&malicious, 50)]);

        // Cosine of uncapped result to impostor (the test target).
        let uncapped_cos_to_impostor = cosine_unit(&uncapped, &impostor);

        // Merge with bounded weight (defence).
        // REFINE_WEIGHT_CAP: for now, we simulate cap = 20 (per-meeting min).
        let capped = weighted_merge(&[(&established, 100), (&malicious, 20)]);
        let capped_cos_to_impostor = cosine_unit(&capped, &impostor);

        // The bounded version must have a lower cosine to impostor
        // (the centroid is closer to the established direction).
        assert!(
            capped_cos_to_impostor < uncapped_cos_to_impostor,
            "Bounded weight should reduce impostor cosine: capped={}, uncapped={}",
            capped_cos_to_impostor,
            uncapped_cos_to_impostor
        );

        // And critically, the capped version's cosine to the impostor must
        // fall below the T_accept placeholder (0.60), preventing a false accept.
        assert!(
            capped_cos_to_impostor < 0.60,
            "With bounded weight, cosine to impostor {} must be < 0.60",
            capped_cos_to_impostor
        );
    }

    // --- weighted_merge: numerical stability at high counts -------------------

    #[test]
    fn weighted_merge_large_count_values() {
        // Contributions may have large sample_count values (a centroid built
        // from many segments). Verify numerical stability when counts are large.
        let centroid = [0.8_f32, 0.6];
        let merged = weighted_merge(&[(&centroid, 1_000_000), (&centroid, 500_000)]);

        // Result should equal the input (same centroid, different counts).
        assert!((merged[0] - 0.8).abs() < EPS);
        assert!((merged[1] - 0.6).abs() < EPS);
    }

    #[test]
    fn weighted_merge_dissimilar_vectors() {
        // Two very different unit vectors merged with equal weight produce
        // a result that sits between them.
        let head = [1.0_f32, 0.0, 0.0];
        let tail = [-1.0_f32, 0.0, 0.0];

        let merged = weighted_merge(&[(&head, 1), (&tail, 1)]);

        // Weighted mean = [0, 0, 0] → zero vector → no-op normalise → zero.
        assert_eq!(merged, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn weighted_merge_single_large_count() {
        // A single contribution with a very large count behaves like a simple
        // unit-normalise of that contribution.
        let v = [3.0_f32, 4.0]; // Not unit
        let merged = weighted_merge(&[(&v, 1_000_000)]);

        // Should be normalised [0.6, 0.8].
        assert!((merged[0] - 0.6).abs() < EPS);
        assert!((merged[1] - 0.8).abs() < EPS);
        assert!((norm(&merged) - 1.0).abs() < EPS);
    }

    // --- dimension consistency -----------------------------------------------

    #[test]
    fn weighted_merge_all_contributors_same_dim() {
        // If contributors have different dimensions, weighted_merge uses the
        // first's dimension as truth. Verify all are processed consistently.
        let v3d = [1.0_f32, 0.0, 0.0];
        let v2d = [0.0_f32, 1.0]; // Shorter

        let merged = weighted_merge(&[(&v3d, 1), (&v2d, 1)]);

        // Result uses dim of first (3); iteration zips and runs over shorter.
        assert_eq!(merged.len(), 3);
        // Raw mean (using first 2 elements of shorter): [(1+0)/2, (0+1)/2, 0/2]
        // = [0.5, 0.5, 0] → unit [1/√2, 1/√2, 0]
        let expected = 1.0_f32.sqrt() / 2.0_f32.sqrt();
        assert!((merged[0] - expected).abs() < EPS);
        assert!((merged[1] - expected).abs() < EPS);
        assert!(merged[2].abs() < EPS);
    }

    // --- unit_normalise: scale-invariance -----------------------------------

    #[test]
    fn unit_normalise_scale_invariant() {
        // Normalising a scaled vector must produce the same direction
        // (the same unit vector).
        let original = [3.0_f32, 4.0];
        let scaled = [30.0_f32, 40.0];

        let mut u1 = original;
        let mut u2 = scaled;

        unit_normalise(&mut u1);
        unit_normalise(&mut u2);

        // Both should normalize to [0.6, 0.8].
        assert!((u1[0] - u2[0]).abs() < EPS);
        assert!((u1[1] - u2[1]).abs() < EPS);
    }

    // --- cosine_unit: symmetry test ------------------------------------------

    #[test]
    fn cosine_unit_is_symmetric() {
        // cos(a, b) must equal cos(b, a).
        let a = [0.6_f32, 0.8];
        let b = [0.8_f32, -0.6];

        let cos_ab = cosine_unit(&a, &b);
        let cos_ba = cosine_unit(&b, &a);

        assert!((cos_ab - cos_ba).abs() < EPS);
    }
}
