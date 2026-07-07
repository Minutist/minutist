//! Pure, FFI-free online speaker clusterer.
//!
//! Running-mean centroids + cosine similarity + sticky first-seen ordering.
//! Holds NO sherpa state, so the clustering logic is fully unit-testable with
//! synthetic `f32` vectors (mirrors the model-free suite in `lib.rs`). The
//! sherpa `EmbeddingExtractor` lives one layer up in [`crate::online`]; this
//! module never crosses into FFI.
//!
//! ## Why a pure clusterer rather than sherpa's `EmbeddingManager`
//!
//! The `EmbeddingManager` API in sherpa-rs 0.6.8 is unsuitable on three
//! independent counts, all verified in its source (`embedding_manager.rs`):
//! 1. NO running-mean centroids — `add(name, &mut [f32])` stores ONE vector
//!    per name and rejects a duplicate name; there is no exposed remove or
//!    update, so a centroid cannot evolve as more audio for a speaker arrives.
//! 2. `search`/`get_best_matches` take the address of a temporary `Vec`
//!    (`embedding.to_owned().as_mut_ptr()`) that is dropped before/while the C
//!    call reads it — a use-after-free on the hot live path.
//! 3. Every method crosses into `sherpa_rs_sys`, so the clustering logic could
//!    never be exercised with synthetic vectors in the default (no-model) test
//!    suite, which is this crate's established convention.
//!
//! The pure clusterer owns the centroid-update rule the design requires and is
//! deterministic, FFI-free, and unit-testable. Sherpa is used ONLY for the
//! embedding extraction, the one thing that genuinely needs the model.

use minutist_common::AppResult;

use crate::Error;

/// Default cosine-similarity threshold (0.25), chosen from a sweep against the
/// zh-en embedding model on real + synthetic audio (2026-06-05): it is the
/// LOWEST threshold that still keeps two genuinely distinct speakers apart
/// (below it they merge), which maximises single-speaker merging — at 0.25 a
/// real single-speaker recording resolves to 1, two distinct speakers to 2, a
/// single-speaker control to 1. The old 0.5 badly over-split (a single speaker
/// became 6 live labels). NOTE: this is a *similarity* (higher => more
/// speakers), the OPPOSITE orientation to the offline
/// `DiarizerConfig::cluster_threshold` (a *distance*). The greedy online path
/// has little margin here; live labels are provisional and the authoritative
/// on-stop `SherpaDiarizer` pass corrects them.
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.25;

/// Configuration for the online sticky-label clusterer.
#[derive(Debug, Clone)]
pub struct OnlineClustererConfig {
    /// Cosine-similarity threshold in `[0, 1]`. A segment whose best cosine
    /// similarity to an existing centroid is `>=` this joins that centroid;
    /// otherwise it starts a new speaker. Higher => more speakers (stricter).
    pub similarity_threshold: f32,
    /// Hard cap on distinct live speakers. Once reached, further segments are
    /// force-assigned to their nearest centroid even below threshold — a new
    /// speaker is never minted past the cap. The first speaker is always minted,
    /// so `Some(0)` behaves like `Some(1)`. `None` => unbounded.
    pub max_speakers: Option<usize>,
}

impl Default for OnlineClustererConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            max_speakers: None,
        }
    }
}

/// One immutable assignment result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterAssignment {
    /// Zero-based first-seen cluster index (0 => "A"). Map to a label with the
    /// crate's `alpha_label`. Sticky: a given index always maps to the same
    /// label.
    pub index: usize,
    /// True if this segment minted a new cluster (first-seen).
    pub is_new: bool,
}

/// Pure, FFI-free online speaker clusterer: running-mean centroids, cosine
/// similarity, sticky first-seen ordering. Holds NO sherpa state.
#[derive(Debug, Default)]
pub struct OnlineClusterer {
    config: OnlineClustererConfig,
    /// `centroids[i]` is the running mean of the unit-normalised embeddings fed
    /// to cluster `i`; `counts[i]` is how many embeddings fed it. Index order
    /// IS first-seen order, so it never reshuffles => labels are sticky.
    centroids: Vec<Vec<f32>>,
    counts: Vec<u64>,
    dim: Option<usize>,
}

impl OnlineClusterer {
    /// Construct an empty clusterer with the given configuration.
    pub fn new(config: OnlineClustererConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            counts: Vec::new(),
            dim: None,
        }
    }

    /// Feed one embedding; return its sticky assignment. PURE (no I/O, no FFI).
    ///
    /// Errors (`InvalidInput`): empty embedding, dim mismatch vs. the first
    /// embedding seen, or a non-finite/zero-norm vector (cosine undefined). On
    /// the FIRST call, `dim` is locked to `embedding.len()`.
    pub fn assign(&mut self, embedding: &[f32]) -> AppResult<ClusterAssignment> {
        // 1. Validate: non-empty, dim-locked, all-finite, ||x|| > 0.
        if embedding.is_empty() {
            return Err(Error::InvalidInput("empty embedding vector".to_string()).into());
        }
        match self.dim {
            Some(d) if d != embedding.len() => {
                return Err(Error::InvalidInput(format!(
                    "embedding dim mismatch: expected {d}, got {}",
                    embedding.len()
                ))
                .into());
            }
            _ => {}
        }
        let unit = unit_normalise(embedding)?;

        // Lock dim on the first valid embedding.
        if self.dim.is_none() {
            self.dim = Some(embedding.len());
        }

        // 3. No centroids yet => mint cluster 0.
        if self.centroids.is_empty() {
            self.centroids.push(unit);
            self.counts.push(1);
            tracing::debug!(target: "diarizer", index = 0usize, "online clusterer minted new cluster");
            return Ok(ClusterAssignment {
                index: 0,
                is_new: true,
            });
        }

        // 4. Best cosine match over existing centroids; ties => lower index.
        let (best_index, best_sim) = self.best_match(&unit);

        // 5. Decision.
        let at_cap = matches!(self.config.max_speakers, Some(max) if self.centroids.len() >= max);
        let join = best_sim >= self.config.similarity_threshold || at_cap;

        if join {
            self.update_centroid(best_index, &unit);
            tracing::trace!(
                target: "diarizer",
                index = best_index,
                similarity = best_sim,
                "online clusterer joined existing cluster"
            );
            Ok(ClusterAssignment {
                index: best_index,
                is_new: false,
            })
        } else {
            let index = self.centroids.len();
            self.centroids.push(unit);
            self.counts.push(1);
            tracing::debug!(
                target: "diarizer",
                index,
                best_similarity = best_sim,
                "online clusterer minted new cluster"
            );
            Ok(ClusterAssignment {
                index,
                is_new: true,
            })
        }
    }

    /// Cosine similarity of `unit` (already unit-length) against every centroid;
    /// return `(argmax_index, best_sim)`. Ties resolve to the LOWER index
    /// (earlier first-seen, deterministic — mirrors the offline tie-break).
    ///
    /// Centroids are running means of unit vectors and are NOT renormalised
    /// between updates, so the true cosine is `dot(unit, c) / ||c||` (since
    /// `||unit|| == 1`). A zero-norm centroid (cannot occur — every centroid
    /// starts as a unit vector and the mean of unit vectors with `||.|| > 0`
    /// stays non-degenerate for the inputs we accept) scores 0.
    fn best_match(&self, unit: &[f32]) -> (usize, f32) {
        let mut best_index = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_unit_vs_centroid(unit, centroid);
            // Strict `>` keeps the earlier (lower) index on a tie.
            if sim > best_sim {
                best_sim = sim;
                best_index = i;
            }
        }
        (best_index, best_sim)
    }

    /// Incremental running mean (Welford-style) of cluster `index` with the new
    /// unit vector: `c += (u - c) / (n + 1)`; `count = n + 1`. The centroid is
    /// kept a true mean (NOT re-normalised) so a speaker's centroid can migrate
    /// slightly without destabilising the cosine metric.
    fn update_centroid(&mut self, index: usize, unit: &[f32]) {
        let n = self.counts[index];
        let inv = 1.0 / (n as f32 + 1.0);
        let centroid = &mut self.centroids[index];
        for (c, &u) in centroid.iter_mut().zip(unit.iter()) {
            *c += (u - *c) * inv;
        }
        self.counts[index] = n + 1;
    }

    /// Number of distinct clusters minted so far.
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    /// True when no cluster has been minted yet.
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// Read-only snapshot of a centroid (for tests / introspection).
    pub fn centroid(&self, index: usize) -> Option<&[f32]> {
        self.centroids.get(index).map(Vec::as_slice)
    }
}

/// Unit-normalise `x`, rejecting non-finite components or a zero/overflow norm
/// (cosine is undefined there). `x` is assumed non-empty (the caller checks).
///
/// Validates the input fully, then delegates the actual normalisation to
/// [`minutist_common::voiceprint_math::unit_normalise`]. The common function is
/// a no-op on degenerate inputs; the explicit pre-check here converts those
/// cases into `AppError::InvalidInput`.
///
/// `pub(crate)` because [`crate::online::VoiceprintExtractor::centroid`] reuses
/// it (via an `Err` → skip, rather than propagate) to keep a degenerate window
/// out of a re-embedded centroid, with the same reject criteria the online
/// clusterer applies to a live embedding.
pub(crate) fn unit_normalise(x: &[f32]) -> AppResult<Vec<f32>> {
    // Reject non-finite components (NaN, ±Inf): the norm would be undefined.
    for &v in x {
        if !v.is_finite() {
            return Err(
                Error::InvalidInput("embedding contains a non-finite value".to_string()).into(),
            );
        }
    }
    // Compute the norm; reject zero (cosine undefined) and overflow to +inf
    // (very large but finite components — the common function's no-op path).
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let norm = sum_sq.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(
            Error::InvalidInput("embedding has a zero or non-finite norm".to_string()).into(),
        );
    }
    let mut out = x.to_vec();
    minutist_common::voiceprint_math::unit_normalise(&mut out);
    Ok(out)
}

/// True cosine of a unit vector `unit` against a running-mean centroid `c`:
/// `dot(unit, c) / ||c||` (since `||unit|| == 1`). Returns 0 for a degenerate
/// (zero/non-finite-norm) centroid.
///
/// Delegates to [`minutist_common::voiceprint_math::cosine_unit`] for the dot
/// product, then divides by the centroid norm — the centroid is a running mean
/// of unit vectors and is NOT renormalised between updates, so the true cosine
/// requires the extra division.
fn cosine_unit_vs_centroid(unit: &[f32], c: &[f32]) -> f32 {
    let c_sq: f32 = c.iter().map(|&v| v * v).sum();
    let c_norm = c_sq.sqrt();
    if c_norm > 0.0 {
        minutist_common::voiceprint_math::cosine_unit(unit, c) / c_norm
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::AppError;

    fn clusterer(threshold: f32, max: Option<usize>) -> OnlineClusterer {
        OnlineClusterer::new(OnlineClustererConfig {
            similarity_threshold: threshold,
            max_speakers: max,
        })
    }

    #[test]
    fn empty_clusterer_len_is_zero() {
        let c = OnlineClusterer::default();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert!(c.centroid(0).is_none());
    }

    #[test]
    fn first_assign_is_new() {
        let mut c = OnlineClusterer::default();
        let a = c.assign(&[1.0, 0.0, 0.0]).expect("first assign");
        assert_eq!(a.index, 0);
        assert!(a.is_new);
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());
    }

    // Name uses A/B to mirror the sticky-label output; keep it readable.
    #[test]
    #[allow(non_snake_case)]
    fn two_well_separated_clusters_get_A_then_B() {
        // Two orthogonal families, interleaved. First family seen => index 0
        // ("A"), second => index 1 ("B"), regardless of interleave.
        let mut c = clusterer(0.5, None);

        let a0 = c.assign(&[1.0, 0.0, 0.0]).unwrap(); // family X, first-seen
        assert_eq!((a0.index, a0.is_new), (0, true));

        let b0 = c.assign(&[0.0, 1.0, 0.0]).unwrap(); // family Y, second-seen
        assert_eq!((b0.index, b0.is_new), (1, true));

        // More of each family, slightly perturbed; must join their first index,
        // never mint anew.
        let a1 = c.assign(&[0.98, 0.02, 0.0]).unwrap();
        assert_eq!((a1.index, a1.is_new), (0, false));

        let b1 = c.assign(&[0.01, 0.99, 0.0]).unwrap();
        assert_eq!((b1.index, b1.is_new), (1, false));

        // Interleave back to X.
        let a2 = c.assign(&[0.95, 0.05, 0.0]).unwrap();
        assert_eq!((a2.index, a2.is_new), (0, false));

        assert_eq!(c.len(), 2);
    }

    #[test]
    fn sticky_labels_never_reassign() {
        // Cluster 0, then cluster 1, then cluster 0 again => index 0 preserved,
        // proving no retroactive relabel.
        let mut c = clusterer(0.5, None);
        assert_eq!(c.assign(&[1.0, 0.0]).unwrap().index, 0);
        assert_eq!(c.assign(&[0.0, 1.0]).unwrap().index, 1);
        let again = c.assign(&[1.0, 0.0]).unwrap();
        assert_eq!(again.index, 0);
        assert!(!again.is_new);
    }

    #[test]
    fn threshold_governs_split() {
        // Two near-but-distinct vectors. cos(theta) between [1,0] and
        // normalised [1,1] is ~0.707.
        let near = [1.0_f32, 1.0];

        // High threshold (0.9): the second vector is below it => split.
        let mut strict = clusterer(0.9, None);
        strict.assign(&[1.0, 0.0]).unwrap();
        let split = strict.assign(&near).unwrap();
        assert!(split.is_new);
        assert_eq!(strict.len(), 2);

        // Low threshold (0.5): same pair joins one cluster.
        let mut loose = clusterer(0.5, None);
        loose.assign(&[1.0, 0.0]).unwrap();
        let joined = loose.assign(&near).unwrap();
        assert!(!joined.is_new);
        assert_eq!(loose.len(), 1);
    }

    #[test]
    fn running_mean_centroid_drift() {
        // Feed several slightly-varying vectors of one speaker; the centroid
        // must move toward their (unit-normalised) mean and the count track n.
        let mut c = clusterer(0.5, None);
        let samples = [
            [1.0_f32, 0.0],
            [0.96, 0.28],
            [0.94, 0.34],
            [0.92, 0.39],
        ];
        for s in &samples {
            c.assign(s).unwrap();
        }
        assert_eq!(c.len(), 1);

        // Mean of the unit-normalised samples.
        let mut mean = [0.0_f32, 0.0];
        for s in &samples {
            let n = (s[0] * s[0] + s[1] * s[1]).sqrt();
            mean[0] += s[0] / n;
            mean[1] += s[1] / n;
        }
        mean[0] /= samples.len() as f32;
        mean[1] /= samples.len() as f32;

        let centroid = c.centroid(0).unwrap();
        // The centroid sits close to the sample mean (cosine ~ 1).
        let dot = centroid[0] * mean[0] + centroid[1] * mean[1];
        let cn = (centroid[0] * centroid[0] + centroid[1] * centroid[1]).sqrt();
        let mn = (mean[0] * mean[0] + mean[1] * mean[1]).sqrt();
        let cos_to_mean = dot / (cn * mn);
        assert!(
            cos_to_mean > 0.999,
            "centroid should align with the sample mean, cos = {cos_to_mean}"
        );

        // The centroid has drifted off the original [1,0] first sample: it now
        // has a non-trivial second component.
        assert!(
            centroid[1] > 0.1,
            "centroid should have drifted toward later samples, got {centroid:?}"
        );

        // A vector closer to the drifted mean than to the original still joins
        // cluster 0 (no new mint).
        let drifted = c.assign(&[0.9, 0.4]).unwrap();
        assert_eq!(drifted.index, 0);
        assert!(!drifted.is_new);
    }

    #[test]
    fn tie_breaks_to_lower_index() {
        // Two orthonormal centroids, then a probe [1,1,0] (normalised) with equal
        // cosine (~0.707) to both. A high threshold would normally mint; capping
        // at 2 forces it to join, exposing the tie-break to the lower (earlier)
        // index.
        let mut capped = clusterer(0.9, Some(2));
        capped.assign(&[1.0, 0.0, 0.0]).unwrap(); // index 0
        capped.assign(&[0.0, 1.0, 0.0]).unwrap(); // index 1
        let probe = capped.assign(&[1.0, 1.0, 0.0]).unwrap();
        assert_eq!(probe.index, 0, "tie must resolve to the lower (earlier) index");
        assert!(!probe.is_new);
    }

    #[test]
    fn max_speakers_cap_force_joins() {
        // Cap at 2. After two clusters exist, a below-threshold third vector
        // force-joins its nearest centroid — no third cluster minted.
        let mut c = clusterer(0.9, Some(2));
        c.assign(&[1.0, 0.0, 0.0]).unwrap(); // index 0
        c.assign(&[0.0, 1.0, 0.0]).unwrap(); // index 1
        let third = c.assign(&[0.0, 0.0, 1.0]).unwrap(); // orthogonal to both
        assert!(!third.is_new, "cap reached => must force-join, not mint");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn dim_mismatch_and_degenerate_rejected() {
        // First embedding locks dim = 3.
        let mut c = clusterer(0.5, None);
        c.assign(&[1.0, 0.0, 0.0]).unwrap();

        // Later different-length embedding => InvalidInput.
        let mismatch = c.assign(&[1.0, 0.0]).unwrap_err();
        assert!(matches!(mismatch, AppError::InvalidInput { .. }));

        // Zero-norm vector => InvalidInput.
        let zero = c.assign(&[0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(zero, AppError::InvalidInput { .. }));

        // NaN / inf components => InvalidInput.
        let nan = c.assign(&[f32::NAN, 0.0, 0.0]).unwrap_err();
        assert!(matches!(nan, AppError::InvalidInput { .. }));
        let inf = c.assign(&[f32::INFINITY, 0.0, 0.0]).unwrap_err();
        assert!(matches!(inf, AppError::InvalidInput { .. }));

        // Huge-but-finite components: sum-of-squares overflows to +inf, so the
        // norm is non-finite => InvalidInput (not a silent all-zero vector).
        let huge = c.assign(&[1e20, 1e20, 1e20]).unwrap_err();
        assert!(matches!(huge, AppError::InvalidInput { .. }));

        // An empty embedding on a fresh clusterer => InvalidInput.
        let mut fresh = OnlineClusterer::default();
        let empty = fresh.assign(&[]).unwrap_err();
        assert!(matches!(empty, AppError::InvalidInput { .. }));
    }
}
