//! `diarizer` — speaker diarization, offline (authoritative) + live (additive).
//!
//! The OFFLINE [`SherpaDiarizer`] implements [`meeting_app_common::Diarizer`]
//! over a sherpa-onnx (via `sherpa-rs`) two-model pipeline: a **segmentation**
//! model + a **speaker-embedding** model + clustering. It assigns `speaker_id`
//! to the ASR `Segment`s of a finished recording (a post-pass) and is the
//! AUTHORITATIVE labelling for the finished transcript.
//!
//! The LIVE [`OnlineDiarizer`] (module [`online`]) is an ADDITIVE hint: it
//! labels one VAD segment at a time during recording (embedding + a pure online
//! clusterer, no segmentation model), emitting sticky first-seen labels that may
//! disagree with the on-stop pass and are never retroactively relabelled. See
//! `architecture/components.md` — `diarizer`, `cross-cutting.md` — "Live vs.
//! offline diarization", and the Phase-6 plan.
//!
//! Models (license-verified, settings-selected via `model-registry`):
//! - segmentation: pyannote/segmentation-3.0 (MIT)
//! - embedding: 3D-Speaker CAM++ zh-cn 16k-common (Apache-2.0, in-house corpus,
//!   NOT VoxCeleb — the clean commercial-redistribution path).
//!
//! ## Pipeline (Phase 6 Stream S1)
//!
//! [`SherpaDiarizer::open`] constructs the sherpa `Diarize` engine once, holding
//! it behind a `Mutex` (the `common::Diarizer` trait takes `&self` but sherpa's
//! `compute` takes `&mut self`). [`SherpaDiarizer::assign_speakers`] then:
//!
//! 1. asserts the input is 16 kHz (sherpa's pyannote segmentation is fixed at
//!    16 kHz; anything else is `AppError::InvalidInput`),
//! 2. runs `Diarize::compute` to get raw sherpa turns
//!    `{ start_s, end_s, speaker: i32 }`,
//! 3. converts the seconds to milliseconds and overlays a `speaker_id` onto each
//!    ASR `Segment` by **max-overlap interval-join** over `[start_ms, end_ms]`
//!    (no overlap → `None`),
//! 4. relabels the chosen `i32` cluster ids to first-seen-order `"A"`, `"B"`, …
//!    and returns the distinct-label count.
//!
//! The interval-join and the first-seen relabel are pure functions
//! ([`overlay_speakers`]) covered by the default (no-model) test suite; the
//! sherpa `compute` call is exercised by the env-var-gated accuracy test.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use meeting_app_common::{AppResult, Diarizer, Segment};
use sherpa_rs::diarize::{Diarize, DiarizeConfig, Segment as SherpaSegment};

mod error;
pub use error::Error;

mod online;
pub use online::clusterer::{ClusterAssignment, OnlineClusterer, OnlineClustererConfig};
pub use online::{OnlineDiarizer, OnlineDiarizerConfig};

/// Clustering knobs for the diarizer.
///
/// Exactly one of `num_clusters` (known speaker count) or `cluster_threshold`
/// (unknown count; smaller → more speakers) drives sherpa's agglomerative
/// stage. Production builds this via [`Default`] (`num_clusters = None`,
/// threshold mode) because the speaker count is unknown at record time;
/// `num_clusters = Some(n)` is available for callers that genuinely know the
/// count (e.g. fixed-count tests).
#[derive(Debug, Clone)]
pub struct DiarizerConfig {
    /// Fixed speaker count, when known. `None` → use `cluster_threshold`.
    pub num_clusters: Option<u32>,
    /// Agglomerative-clustering distance threshold when the count is unknown.
    pub cluster_threshold: f32,
}

impl Default for DiarizerConfig {
    fn default() -> Self {
        Self {
            num_clusters: None,
            // Conservative default: avoid over-splitting a single speaker.
            cluster_threshold: 0.5,
        }
    }
}

/// The sample rate sherpa's pyannote segmentation model expects. Inputs at any
/// other rate are rejected with `AppError::InvalidInput` — `audio-capture`
/// standardises the whole pipeline on 16 kHz mono, so a mismatch is a wiring
/// bug, not a runtime condition to resample around.
const REQUIRED_SAMPLE_RATE: u32 = 16_000;

/// Reject any input not at [`REQUIRED_SAMPLE_RATE`].
///
/// A pure pre-engine check (no sherpa, no model), factored out so the default
/// test suite covers the guard without constructing a `SherpaDiarizer` (which
/// would need the ONNX models). `audio-capture` standardises the whole pipeline
/// on 16 kHz mono, so a mismatch is a wiring bug, not a runtime condition to
/// resample around.
fn require_supported_sample_rate(sample_rate: u32) -> AppResult<()> {
    if sample_rate != REQUIRED_SAMPLE_RATE {
        return Err(Error::InvalidInput(format!(
            "diarizer requires {REQUIRED_SAMPLE_RATE} Hz mono audio; got {sample_rate} Hz"
        ))
        .into());
    }
    Ok(())
}

/// A speaker diarizer backed by a sherpa-onnx segmentation + embedding pipeline.
///
/// Construct with [`SherpaDiarizer::open`]. The loaded sherpa `Diarize` engine
/// is held behind a `Mutex` so the `&self` `Diarizer::assign_speakers` trait
/// method can drive sherpa's `&mut self` `compute`. Diarization is post-hoc and
/// single-threaded per call, so the mutex is never contended on the hot path.
pub struct SherpaDiarizer {
    segmentation_path: PathBuf,
    embedding_path: PathBuf,
    config: DiarizerConfig,
    engine: Mutex<Diarize>,
}

impl SherpaDiarizer {
    /// Open a diarizer over the segmentation + embedding ONNX models.
    ///
    /// Constructs the sherpa `Diarize` engine, mapping `DiarizerConfig` onto
    /// sherpa's `DiarizeConfig`: a known `num_clusters` drives exact-cluster
    /// mode, otherwise `num_clusters = -1` (sherpa's "use threshold" sentinel,
    /// per Spike 4) selects the agglomerative `cluster_threshold` path. A
    /// sherpa `eyre::Result` error is mapped to `AppError::ModelLoad` at the
    /// boundary (no `eyre` dependency leaks out of this crate).
    pub fn open(
        segmentation_path: &Path,
        embedding_path: &Path,
        config: DiarizerConfig,
    ) -> AppResult<Self> {
        let sherpa_config = sherpa_diarize_config(&config);

        let engine = Diarize::new(segmentation_path, embedding_path, sherpa_config).map_err(|e| {
            Error::ModelLoad {
                path: format!(
                    "segmentation={}, embedding={}",
                    segmentation_path.display(),
                    embedding_path.display()
                ),
                context: format!("{e:?}"),
            }
        })?;

        Ok(Self {
            segmentation_path: segmentation_path.to_path_buf(),
            embedding_path: embedding_path.to_path_buf(),
            config,
            engine: Mutex::new(engine),
        })
    }

    /// The segmentation model path.
    pub fn segmentation_path(&self) -> &Path {
        &self.segmentation_path
    }

    /// The speaker-embedding model path.
    pub fn embedding_path(&self) -> &Path {
        &self.embedding_path
    }

    /// The active clustering configuration.
    pub fn config(&self) -> &DiarizerConfig {
        &self.config
    }
}

/// Map [`DiarizerConfig`] onto sherpa's `DiarizeConfig`.
///
/// `num_clusters = Some(n)` → exact-cluster mode (`n` clusters). `None` →
/// `num_clusters = Some(-1)`, sherpa's "use threshold instead" sentinel (Spike
/// 4), with the agglomerative `cluster_threshold`. `min_duration_*` keep
/// sherpa's defaults. Production (both the on-stop and re-diarize passes) builds
/// the diarizer with [`DiarizerConfig::default`] → threshold mode, since the
/// speaker count is unknown at record time; the conservative `cluster_threshold`
/// is what guards against over-splitting one speaker.
fn sherpa_diarize_config(config: &DiarizerConfig) -> DiarizeConfig {
    DiarizeConfig {
        num_clusters: match config.num_clusters {
            Some(n) => Some(n as i32),
            // -1 is sherpa's contract for "ignore num_clusters, use threshold".
            None => Some(-1),
        },
        threshold: Some(config.cluster_threshold),
        min_duration_on: Some(0.0),
        min_duration_off: Some(0.0),
        provider: None,
        debug: false,
    }
}

impl Diarizer for SherpaDiarizer {
    fn assign_speakers(
        &self,
        audio: &[f32],
        sample_rate: u32,
        segments: &mut [Segment],
    ) -> AppResult<u32> {
        require_supported_sample_rate(sample_rate)?;

        // Nothing to assign — empty transcript or empty audio. Don't invoke
        // sherpa (its `compute` bail!s on zero-length input).
        if segments.is_empty() || audio.is_empty() {
            return Ok(0);
        }

        let turns = {
            let mut engine = self.engine.lock().map_err(|_| {
                Error::Inference("diarizer engine mutex poisoned".to_string())
            })?;
            // sherpa takes ownership of the sample buffer (it mutates the ptr in
            // place); clone the borrowed slice into an owned Vec for the FFI call.
            engine
                .compute(audio.to_vec(), None)
                .map_err(|e| Error::Inference(format!("sherpa Diarize::compute failed: {e:?}")))?
        };

        Ok(overlay_speakers(&turns, segments))
    }
}

/// Overlay first-seen-relabelled speaker ids onto `segments` by max-overlap
/// interval-join, returning the distinct-label count.
///
/// For each ASR `segment`, the sherpa turn (in **seconds**) with the greatest
/// temporal overlap over `[start_ms, end_ms]` wins; its `i32` cluster id is
/// recorded. Segments with no overlapping turn get `speaker_id = None`. Ties
/// (equal overlap) resolve to the **earlier** turn in `turns` order — sherpa
/// sorts turns by start time, so the earlier turn is the lower start, a stable,
/// deterministic tie-break. The chosen `i32` ids are then relabelled to
/// first-seen order `"A"`, `"B"`, … (first-seen across `segments` in slice
/// order), and each segment's `speaker_id` is set. The return value is the
/// number of distinct labels actually assigned (segments left `None` do not
/// count).
///
/// Pure (no FFI, no I/O) so the default test suite covers it without a model.
pub fn overlay_speakers(turns: &[SherpaSegment], segments: &mut [Segment]) -> u32 {
    // First pass: pick the max-overlap sherpa cluster id (if any) per segment.
    let chosen: Vec<Option<i32>> = segments
        .iter()
        .map(|seg| max_overlap_speaker(turns, seg.start_ms, seg.end_ms))
        .collect();

    // Build first-seen-order label map over the chosen cluster ids, in segment
    // slice order (so labels read top-to-bottom of the transcript).
    let mut seen: Vec<i32> = Vec::new();
    for id in chosen.iter().flatten() {
        if !seen.contains(id) {
            seen.push(*id);
        }
    }

    // Second pass: stamp the relabelled id onto each segment.
    for (seg, id) in segments.iter_mut().zip(chosen.iter()) {
        seg.speaker_id = id.map(|cluster| {
            let idx = seen
                .iter()
                .position(|s| *s == cluster)
                .expect("seen contains every chosen cluster id");
            alpha_label(idx)
        });
    }

    seen.len() as u32
}

/// Pick the sherpa cluster id with the greatest overlap over the half-open
/// millisecond window `[start_ms, end_ms)`, or `None` when nothing overlaps.
///
/// sherpa turns carry start/end in **seconds**; they are converted to
/// milliseconds for the overlap computation. A zero-overlap segment (e.g. a
/// segment falling entirely in a silence gap the diarizer dropped) yields
/// `None`. On an exact overlap tie the earlier turn in `turns` order wins
/// (deterministic; sherpa pre-sorts turns by start time).
fn max_overlap_speaker(turns: &[SherpaSegment], start_ms: u64, end_ms: u64) -> Option<i32> {
    let mut best: Option<(i32, u64)> = None;
    for turn in turns {
        let turn_start_ms = seconds_to_ms(turn.start);
        let turn_end_ms = seconds_to_ms(turn.end);
        let overlap = interval_overlap_ms(start_ms, end_ms, turn_start_ms, turn_end_ms);
        if overlap == 0 {
            continue;
        }
        match best {
            // Strict `>` keeps the earlier (already-stored) turn on a tie.
            Some((_, best_overlap)) if overlap <= best_overlap => {}
            _ => best = Some((turn.speaker, overlap)),
        }
    }
    best.map(|(speaker, _)| speaker)
}

/// Overlap length in milliseconds of two half-open intervals `[a0, a1)` and
/// `[b0, b1)`. Zero when they do not overlap.
fn interval_overlap_ms(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    hi.saturating_sub(lo)
}

/// Convert sherpa's seconds (`f32`) to milliseconds, rounding to nearest and
/// clamping negatives to zero.
fn seconds_to_ms(seconds: f32) -> u64 {
    if seconds <= 0.0 {
        0
    } else {
        (seconds * 1000.0).round() as u64
    }
}

/// `0 → "A"`, `1 → "B"`, …, `25 → "Z"`, `26 → "AA"`, `27 → "AB"`, …
/// (FR-12 anonymous-label convention; mirrors the Spike-4 relabeller).
///
/// `pub(crate)` so both the offline overlay ([`overlay_speakers`]) and the
/// online clusterer ([`crate::online`]) share the one A/B/C generator.
pub(crate) fn alpha_label(mut n: usize) -> String {
    let mut out = String::new();
    loop {
        out.insert(0, (b'A' + (n % 26) as u8) as char);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    out
}

// ===========================================================================
// Tests — default suite (no model).
//
// The interval-join + first-seen relabel are pure, so they're fully covered
// here without a sherpa model. The env-var-gated accuracy test lives in
// `tests/accuracy.rs`.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use meeting_app_common::Segment;

    fn turn(start_s: f32, end_s: f32, speaker: i32) -> SherpaSegment {
        SherpaSegment {
            start: start_s,
            end: end_s,
            speaker,
        }
    }

    fn seg(start_ms: u64, end_ms: u64) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: String::new(),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
        }
    }

    #[test]
    fn require_supported_sample_rate_accepts_16khz_rejects_others() {
        // 16 kHz is accepted.
        require_supported_sample_rate(16_000).expect("16 kHz must be accepted");

        // Anything else is rejected as InvalidInput, model-free.
        for bad in [8_000u32, 22_050, 44_100, 48_000, 0] {
            let err = require_supported_sample_rate(bad)
                .expect_err("non-16 kHz must be rejected");
            assert!(
                matches!(err, meeting_app_common::AppError::InvalidInput { .. }),
                "expected InvalidInput for {bad} Hz, got {err}"
            );
        }
    }

    #[test]
    fn alpha_labels_roll_over() {
        assert_eq!(alpha_label(0), "A");
        assert_eq!(alpha_label(1), "B");
        assert_eq!(alpha_label(25), "Z");
        assert_eq!(alpha_label(26), "AA");
        assert_eq!(alpha_label(27), "AB");
    }

    #[test]
    fn seconds_to_ms_rounds_and_clamps() {
        assert_eq!(seconds_to_ms(1.5), 1500);
        assert_eq!(seconds_to_ms(0.0), 0);
        assert_eq!(seconds_to_ms(-1.0), 0);
        assert_eq!(seconds_to_ms(2.0006), 2001);
    }

    #[test]
    fn interval_overlap_basic_and_disjoint() {
        assert_eq!(interval_overlap_ms(0, 1000, 500, 1500), 500);
        assert_eq!(interval_overlap_ms(0, 1000, 1000, 2000), 0); // touching, half-open
        assert_eq!(interval_overlap_ms(0, 1000, 2000, 3000), 0); // disjoint
        assert_eq!(interval_overlap_ms(500, 800, 0, 2000), 300); // fully contained
    }

    #[test]
    fn overlay_two_speakers_first_seen_relabel() {
        // sherpa clusters with arbitrary i32 ids 7 and 3, in that first-seen
        // order across the segments → "A" then "B".
        let turns = vec![
            turn(0.0, 2.0, 7),
            turn(2.0, 4.0, 3),
            turn(4.0, 6.0, 7),
        ];
        let mut segs = vec![
            seg(0, 1_900),     // overlaps the 7-turn → A
            seg(2_100, 3_900), // overlaps the 3-turn → B
            seg(4_100, 5_900), // overlaps the 7-turn → A
        ];
        let count = overlay_speakers(&turns, &mut segs);
        assert_eq!(count, 2);
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(segs[2].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn overlay_no_overlap_yields_none() {
        let turns = vec![turn(0.0, 1.0, 0)];
        // Segment sits entirely after the only turn → no overlap → None.
        let mut segs = vec![seg(5_000, 6_000)];
        let count = overlay_speakers(&turns, &mut segs);
        assert_eq!(count, 0);
        assert_eq!(segs[0].speaker_id, None);
    }

    #[test]
    fn overlay_picks_max_overlap_turn() {
        // The segment straddles two turns; the second turn covers more of it.
        let turns = vec![
            turn(0.0, 1.2, 0), // 0..1200 ms : overlaps [1000,2000) by 200 ms
            turn(1.2, 3.0, 1), // 1200..3000 ms : overlaps [1000,2000) by 800 ms
        ];
        let mut segs = vec![seg(1_000, 2_000)];
        let count = overlay_speakers(&turns, &mut segs);
        assert_eq!(count, 1);
        // Cluster id 1 is the only chosen id → first-seen → "A".
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn overlay_tie_breaks_to_earlier_turn() {
        // Equal overlap (500 ms each) — the earlier turn in slice order wins.
        let turns = vec![
            turn(0.0, 1.5, 9), // overlaps [1000,2000) by 500 ms (1000..1500)
            turn(1.5, 3.0, 4), // overlaps [1000,2000) by 500 ms (1500..2000)
        ];
        let mut segs = vec![seg(1_000, 2_000)];
        let count = overlay_speakers(&turns, &mut segs);
        assert_eq!(count, 1);
        // Earlier turn (cluster 9) wins the tie → first-seen → "A".
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn overlay_single_speaker_one_label() {
        let turns = vec![turn(0.0, 10.0, 2)];
        let mut segs = vec![seg(0, 2_000), seg(3_000, 5_000), seg(6_000, 9_000)];
        let count = overlay_speakers(&turns, &mut segs);
        assert_eq!(count, 1);
        for s in &segs {
            assert_eq!(s.speaker_id.as_deref(), Some("A"));
        }
    }

    #[test]
    fn overlay_mixed_overlap_and_gap() {
        // Three speakers; one segment lands in a gap with no turn → None and is
        // not counted. First-seen order across segments: 5 → A, 8 → B, 1 → C.
        let turns = vec![
            turn(0.0, 1.0, 5),
            turn(1.0, 2.0, 8),
            // gap 2.0..4.0 — no turn
            turn(4.0, 5.0, 1),
        ];
        let mut segs = vec![
            seg(0, 900),       // A
            seg(1_100, 1_900), // B
            seg(2_500, 3_500), // gap → None
            seg(4_100, 4_900), // C
        ];
        let count = overlay_speakers(&turns, &mut segs);
        assert_eq!(count, 3);
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(segs[2].speaker_id, None);
        assert_eq!(segs[3].speaker_id.as_deref(), Some("C"));
    }

    #[test]
    fn overlay_empty_segments_is_zero() {
        let turns = vec![turn(0.0, 1.0, 0)];
        let mut segs: Vec<Segment> = Vec::new();
        assert_eq!(overlay_speakers(&turns, &mut segs), 0);
    }

    #[test]
    fn overlay_clears_stale_speaker_id_on_no_overlap() {
        // A re-diarize pass must overwrite a previously-set label, including
        // back to None when the new turns no longer cover the segment.
        let turns = vec![turn(0.0, 1.0, 0)];
        let mut segs = vec![Segment {
            speaker_id: Some("Z".to_string()),
            ..seg(5_000, 6_000)
        }];
        overlay_speakers(&turns, &mut segs);
        assert_eq!(segs[0].speaker_id, None);
    }
}
