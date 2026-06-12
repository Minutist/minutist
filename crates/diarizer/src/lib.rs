//! `diarizer` — speaker diarization, offline (authoritative) + live (additive).
//!
//! The OFFLINE [`SherpaDiarizer`] implements [`minutist_common::Diarizer`]
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
//! - embedding: 3D-Speaker CAM++ zh-en 16k-common ADVANCED (Apache-2.0, the
//!   "common" corpus, NOT VoxCeleb — the clean commercial-redistribution path).
//!   The zh-en model replaced the Mandarin-only zh-cn one (2026-06-05): on
//!   English audio the zh-cn model's embedding space was too compressed —
//!   distinct speakers and one speaker's natural variation overlapped, so a
//!   single-speaker recording over-split into 3-4 "speakers". The zh-en model
//!   separates them, opening a usable `cluster_threshold` window (see
//!   `DiarizerConfig::default`).
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
//! 4. applies a **post-cluster prune** (and optional cap) — drops clusters that
//!    win a negligible share of the attributed speech and reassigns their
//!    segments to the nearest surviving cluster (issue #63) — then
//! 5. relabels the surviving `i32` cluster ids to first-seen-order `"A"`, `"B"`,
//!    … and returns the distinct-label count.
//!
//! The over-split fix (issue #63, 2026-06-10): on long, acoustically-varied
//! recordings (room coloration + system-audio loopback + a podcast over a
//! loudspeaker) one speaker's embeddings drift past the single distance
//! `cluster_threshold`, minting extra clusters; the user saw 19 / 29 speakers
//! where the truth was a handful. A distance threshold alone cannot tell
//! "same speaker, drifted" from "different speaker", so the robust lever is the
//! duration-share prune in [`overlay_speakers`] (plus sherpa's temporal
//! smoothing, now enabled). See [`DiarizerConfig`] for the knobs and the journal
//! sweep (2026-06-10) for the count-vs-knob trend on the two real meetings.
//!
//! The interval-join, the prune/cap, and the first-seen relabel are pure
//! functions ([`overlay_speakers`]) covered by the default (no-model) test
//! suite; the sherpa `compute` call is exercised by the env-var-gated accuracy
//! test (`tests/accuracy.rs`) and the count-vs-knob eval (`tests/oversplit_eval.rs`).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use minutist_common::{AppResult, Diarizer, Segment};
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
///
/// On top of sherpa's clustering, two post-cluster guards run in
/// [`overlay_speakers`] against the segment→cluster assignment (the only layer
/// that sees per-cluster speech share):
/// - a **min-cluster prune** drops clusters that win a negligible share of the
///   attributed speech (or too few segments) and reassigns their segments to the
///   nearest surviving cluster, and
/// - an optional **max-speaker cap** collapses to the N largest clusters by
///   speech mass when more than `max_speakers` survive.
///
/// Both are pure post-processing over the sherpa turns; sherpa-onnx exposes no
/// such knob (its `FastClustering` returns every cluster it forms). They are the
/// robust lever against the long-recording over-split (issue #63): a single
/// distance `cluster_threshold` cannot tell "same speaker, embedding drifted"
/// from "different speaker", but a tiny-share cluster is almost always the
/// former. See `architecture/cross-cutting.md` — "Live vs. offline diarization".
#[derive(Debug, Clone)]
pub struct DiarizerConfig {
    /// Fixed speaker count, when known. `None` → use `cluster_threshold`.
    pub num_clusters: Option<u32>,
    /// Agglomerative-clustering distance threshold when the count is unknown.
    pub cluster_threshold: f32,
    /// Drop turns shorter than this (seconds) inside sherpa before they reach the
    /// overlay; `0.0` disables. Smooths brief embedding excursions that would
    /// otherwise seed a spurious turn. Maps to sherpa `min_duration_on`.
    pub min_duration_on: f32,
    /// Merge adjacent same-speaker turns separated by a gap shorter than this
    /// (seconds) inside sherpa; `0.0` disables. The single most direct lever
    /// against one speaker fragmenting into many turns. Maps to sherpa
    /// `min_duration_off`.
    pub min_duration_off: f32,
    /// Post-cluster prune: a cluster whose share of total attributed speech
    /// DURATION is below this fraction (`[0.0, 1.0]`) is dropped and its segments
    /// reassigned to the nearest surviving cluster. `0.0` disables the share
    /// prune. Duration-weighted (NOT a raw segment count) so many short segments
    /// of one true speaker do not out-vote a few long ones.
    pub min_cluster_share: f32,
    /// Post-cluster prune: a cluster winning fewer than this many segments is
    /// dropped and its segments reassigned to the nearest surviving cluster.
    /// `0` disables the segment-count prune. Complements `min_cluster_share`
    /// (either condition prunes a cluster).
    pub min_cluster_segments: usize,
    /// Optional hard ceiling on the surviving speaker count. After the prune, if
    /// more than this many clusters remain, keep the N largest by speech mass and
    /// reassign the rest to their nearest survivor. `None` → uncapped.
    pub max_speakers: Option<usize>,
}

impl Default for DiarizerConfig {
    fn default() -> Self {
        Self {
            num_clusters: None,
            // 0.75 chosen from a threshold sweep against the zh-en embedding
            // model (see below): on real data a 175 s single-speaker recording
            // collapses to 1 speaker by 0.70, while two genuinely distinct
            // speakers stay separated until ~0.80 — 0.75 sits in that window. At
            // the old 0.5 (with the Mandarin model) the same recording
            // over-split into 3-4 speakers. Higher => fewer speakers.
            cluster_threshold: 0.75,
            // Temporal smoothing inside sherpa (issue #63). sherpa's own example
            // ships 0.3 / 0.5; both were previously pinned to 0.0 (disabled),
            // which let every brief embedding excursion on long, acoustically-
            // varied audio become its own turn/cluster. `min_duration_off` (0.5)
            // bridges short intra-speaker pauses; `min_duration_on` (0.3) drops
            // sub-300 ms turns.
            min_duration_on: 0.3,
            min_duration_off: 0.5,
            // Post-cluster prune (issue #63). A cluster winning < 2% of the
            // attributed speech duration is treated as a drifted-embedding
            // artefact and folded into the nearest surviving cluster. This mirrors
            // pyannote's production `min_cluster_size` reassignment and the 2026
            // relative-min-cluster-size result (f ≈ 0.01–0.02 of total). On the
            // two real over-split meetings this is what collapses the reported
            // 19 / 29 down to a handful (see the journal sweep, 2026-06-10).
            min_cluster_share: 0.02,
            // The segment-count prune is OFF by default. It is an independent
            // axis (a cluster below it is pruned regardless of share), so a
            // genuine speaker who utters a single long, high-share segment would
            // be wrongly folded away. The duration-share prune above is the
            // robust primary lever; the count floor stays available for the eval
            // / callers that want it.
            min_cluster_segments: 0,
            // No hard cap by default — the share prune is the primary lever and a
            // floor-free cap would only mask a still-misbehaving prune. Callers
            // that need a guaranteed ceiling set this explicitly.
            max_speakers: None,
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
/// 4), with the agglomerative `cluster_threshold`. `min_duration_on` /
/// `min_duration_off` forward the config's temporal smoothing (issue #63 —
/// previously both pinned to `0.0`/disabled). Production (both the on-stop and
/// re-diarize passes) builds the diarizer with [`DiarizerConfig::default`] →
/// threshold mode, since the speaker count is unknown at record time; the
/// conservative `cluster_threshold`, the smoothing, and the post-cluster prune
/// in [`overlay_speakers`] together guard against over-splitting one speaker.
fn sherpa_diarize_config(config: &DiarizerConfig) -> DiarizeConfig {
    DiarizeConfig {
        num_clusters: match config.num_clusters {
            Some(n) => Some(n as i32),
            // -1 is sherpa's contract for "ignore num_clusters, use threshold".
            None => Some(-1),
        },
        threshold: Some(config.cluster_threshold),
        min_duration_on: Some(config.min_duration_on),
        min_duration_off: Some(config.min_duration_off),
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

        Ok(overlay_speakers(&turns, segments, &self.config))
    }
}

/// Overlay first-seen-relabelled speaker ids onto `segments` by max-overlap
/// interval-join, applying the `config`'s post-cluster prune + cap, and return
/// the distinct-label count.
///
/// For each ASR `segment`, the sherpa cluster (turns are in **seconds**) with
/// the greatest total temporal overlap over `[start_ms, end_ms)` wins; its `i32`
/// cluster id is recorded. Segments with no overlapping turn get
/// `speaker_id = None`. Ties (equal overlap) resolve to the **lower** cluster id
/// (deterministic; sherpa numbers clusters by first appearance, so the lower id
/// is the earlier-seen speaker — the same orientation as the previous earlier-
/// turn tie-break, but now over per-cluster totals).
///
/// Then, when the `config` enables it (issue #63):
/// 1. **Prune** — tally each cluster's share of total attributed speech DURATION
///    and its segment count; a cluster below `min_cluster_share` of the duration
///    OR below `min_cluster_segments` is marked spurious. Each segment whose
///    winner is spurious is reassigned to its next-best *surviving* cluster
///    (greatest overlap among survivors), or `None` if it overlaps no survivor.
/// 2. **Cap** — if more clusters still survive than `max_speakers`, keep the N
///    largest by speech duration and reassign the rest the same way.
///
/// The surviving cluster ids are then relabelled to first-seen order `"A"`,
/// `"B"`, … across `segments` in slice order. The return value is the number of
/// distinct surviving labels actually assigned (segments left `None` do not
/// count).
///
/// Pure (no FFI, no I/O) so the default test suite covers it without a model.
pub fn overlay_speakers(
    turns: &[SherpaSegment],
    segments: &mut [Segment],
    config: &DiarizerConfig,
) -> u32 {
    // Per-segment ranked overlaps: (cluster_id, overlap_ms) sorted descending by
    // overlap, lower cluster id breaking ties. Lets the prune/cap reassign a
    // segment to its next-best SURVIVING cluster without re-touching the turns.
    let ranked: Vec<Vec<(i32, u64)>> = segments
        .iter()
        .map(|seg| ranked_overlaps(turns, seg.start_ms, seg.end_ms))
        .collect();

    // Initial winner per segment: the top-ranked cluster (or None).
    let mut chosen: Vec<Option<i32>> = ranked
        .iter()
        .map(|r| r.first().map(|&(id, _)| id))
        .collect();

    // Determine the surviving set after prune + cap, then reassign pruned
    // segments to their next-best survivor.
    let survivors = surviving_clusters(&chosen, &ranked, config);
    if let Some(survivors) = survivors {
        for (slot, r) in chosen.iter_mut().zip(ranked.iter()) {
            let drop = match *slot {
                Some(id) => !survivors.contains(&id),
                None => false,
            };
            if drop {
                *slot = r
                    .iter()
                    .find(|(id, _)| survivors.contains(id))
                    .map(|&(id, _)| id);
            }
        }
    }

    // Build first-seen-order label map over the (post-prune) chosen cluster ids,
    // in segment slice order (so labels read top-to-bottom of the transcript).
    let mut seen: Vec<i32> = Vec::new();
    for id in chosen.iter().flatten() {
        if !seen.contains(id) {
            seen.push(*id);
        }
    }

    // Stamp the relabelled id onto each segment.
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

/// Decide the surviving cluster set after the `config`'s prune + cap, or `None`
/// when neither is active (no reassignment needed — every winner survives).
///
/// Spurious = a winning cluster below `min_cluster_share` of the total attributed
/// speech DURATION, OR below `min_cluster_segments` won. The cap then keeps only
/// the `max_speakers` largest survivors by duration. A pure helper over the
/// initial `chosen` winners; returns the survivor ids (unordered).
fn surviving_clusters(
    chosen: &[Option<i32>],
    ranked: &[Vec<(i32, u64)>],
    config: &DiarizerConfig,
) -> Option<Vec<i32>> {
    let prune_active = config.min_cluster_share > 0.0 || config.min_cluster_segments > 0;
    let cap_active = config.max_speakers.is_some();
    if !prune_active && !cap_active {
        return None;
    }

    // Tally per-cluster duration mass + segment count over the initial winners.
    // Duration = the winning segment's own overlap with its cluster (the speech
    // actually attributed to that cluster), so many tiny segments cannot out-mass
    // a few long ones.
    let mut ids: Vec<i32> = Vec::new();
    let mut dur: Vec<u64> = Vec::new();
    let mut cnt: Vec<usize> = Vec::new();
    for (slot, r) in chosen.iter().zip(ranked.iter()) {
        let Some(id) = *slot else { continue };
        // The winner's overlap is the head of its ranked list.
        let overlap = r
            .iter()
            .find(|(cid, _)| *cid == id)
            .map(|&(_, ov)| ov)
            .unwrap_or(0);
        match ids.iter().position(|x| *x == id) {
            Some(i) => {
                dur[i] += overlap;
                cnt[i] += 1;
            }
            None => {
                ids.push(id);
                dur.push(overlap);
                cnt.push(1);
            }
        }
    }
    if ids.is_empty() {
        return Some(Vec::new());
    }

    let total_dur: u64 = dur.iter().sum();

    // 1. Prune by share + segment count.
    let mut survivors: Vec<i32> = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        let share = if total_dur > 0 {
            dur[i] as f32 / total_dur as f32
        } else {
            0.0
        };
        let below_share = config.min_cluster_share > 0.0 && share < config.min_cluster_share;
        let below_count =
            config.min_cluster_segments > 0 && cnt[i] < config.min_cluster_segments;
        if !(below_share || below_count) {
            survivors.push(id);
        }
    }

    // Edge case: the prune removed everything (e.g. one big cluster but a
    // share floor > 1.0, or every cluster below the count floor). Keep the
    // single largest-duration cluster so we never return zero speakers for a
    // non-empty transcript.
    if survivors.is_empty() {
        if let Some((i, _)) = dur.iter().enumerate().max_by_key(|(_, d)| **d) {
            survivors.push(ids[i]);
        }
    }

    // 2. Cap: keep the `max_speakers` largest survivors by duration.
    if let Some(max) = config.max_speakers {
        if survivors.len() > max {
            // Sort survivors by duration desc, lower id breaking ties, keep N.
            survivors.sort_by(|a, b| {
                let da = dur[ids.iter().position(|x| x == a).unwrap()];
                let db = dur[ids.iter().position(|x| x == b).unwrap()];
                db.cmp(&da).then(a.cmp(b))
            });
            survivors.truncate(max);
        }
    }

    Some(survivors)
}

/// Per-cluster total overlap of `turns` over the half-open millisecond window
/// `[start_ms, end_ms)`, ranked descending by overlap (lower cluster id breaks a
/// tie). Empty when nothing overlaps.
///
/// sherpa turns carry start/end in **seconds**; they are converted to
/// milliseconds for the overlap computation. Overlap is summed PER CLUSTER (a
/// cluster may own several turns touching one segment), which is what the prune
/// reassignment needs to pick a segment's next-best surviving speaker.
fn ranked_overlaps(turns: &[SherpaSegment], start_ms: u64, end_ms: u64) -> Vec<(i32, u64)> {
    let mut totals: Vec<(i32, u64)> = Vec::new();
    for turn in turns {
        let turn_start_ms = seconds_to_ms(turn.start);
        let turn_end_ms = seconds_to_ms(turn.end);
        let overlap = interval_overlap_ms(start_ms, end_ms, turn_start_ms, turn_end_ms);
        if overlap == 0 {
            continue;
        }
        match totals.iter_mut().find(|(id, _)| *id == turn.speaker) {
            Some((_, acc)) => *acc += overlap,
            None => totals.push((turn.speaker, overlap)),
        }
    }
    // Descending overlap; lower cluster id wins a tie (deterministic, and the
    // lower id is the earlier-seen sherpa cluster).
    totals.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    totals
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

/// Re-overlay a prior diarization onto a freshly ASR-transcribed segment slice
/// by maximum-overlap interval-join.
///
/// For each new segment, the prior segment whose interval covers the most of the
/// new segment's time wins; its `speaker_id` is copied onto the new segment. New
/// segments with zero overlap against any prior segment keep `speaker_id = None`.
///
/// The join is pure (no FFI, no I/O) and does not re-letter: the prior `speaker_id`
/// strings ("A", "B", …) are carried forward verbatim, so `MeetingMeta.speaker_names`
/// remains valid without any key remapping.
///
/// # Parameters
///
/// - `new_segments` — the freshly produced ASR segments (mutated in place).
/// - `prior` — `(start_ms, end_ms, speaker_id)` triples extracted from the
///   transcript that existed before the re-transcribe. An empty slice leaves all
///   new segments as `None` without error.
pub fn overlay_speakers_from_prior(
    new_segments: &mut [Segment],
    prior: &[(u64, u64, Option<String>)],
) {
    for seg in new_segments.iter_mut() {
        let seg_start = seg.start_ms;
        let seg_end = seg.end_ms;
        // Find the prior segment with the maximum overlap against this new
        // segment. Among equal overlaps the FIRST prior segment in slice order
        // wins (strict `>` keeps the earliest maximum) — the same earliest-wins
        // tie orientation as the offline join's `ranked_overlaps`.
        let mut winner: Option<(u64, &Option<String>)> = None;
        for (p_start, p_end, label) in prior {
            let ov = interval_overlap_ms(seg_start, seg_end, *p_start, *p_end);
            if ov > 0 && winner.map_or(true, |(best, _)| ov > best) {
                winner = Some((ov, label));
            }
        }
        seg.speaker_id = winner.and_then(|(_, label)| label.clone());
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
    use minutist_common::Segment;

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

    /// A config with the prune + cap fully disabled, so the overlay tests
    /// exercise the raw interval-join / relabel without the post-cluster guards
    /// (those guards have their own dedicated tests below).
    fn no_prune() -> DiarizerConfig {
        DiarizerConfig {
            num_clusters: None,
            cluster_threshold: 0.75,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            min_cluster_share: 0.0,
            min_cluster_segments: 0,
            max_speakers: None,
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
                matches!(err, minutist_common::AppError::InvalidInput { .. }),
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
        let count = overlay_speakers(&turns, &mut segs, &no_prune());
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
        let count = overlay_speakers(&turns, &mut segs, &no_prune());
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
        let count = overlay_speakers(&turns, &mut segs, &no_prune());
        assert_eq!(count, 1);
        // Cluster id 1 is the only chosen id → first-seen → "A".
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn overlay_tie_breaks_to_lower_cluster_id() {
        // Equal per-cluster overlap (500 ms each) — the LOWER cluster id wins
        // (deterministic; the lower id is the earlier-seen sherpa cluster).
        let turns = vec![
            turn(0.0, 1.5, 9), // overlaps [1000,2000) by 500 ms (1000..1500)
            turn(1.5, 3.0, 4), // overlaps [1000,2000) by 500 ms (1500..2000)
        ];
        let mut segs = vec![seg(1_000, 2_000)];
        let count = overlay_speakers(&turns, &mut segs, &no_prune());
        assert_eq!(count, 1);
        // Cluster 4 (the lower id) wins the tie; as the only chosen id it
        // first-seen-relabels to "A".
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn ranked_overlaps_sums_per_cluster_and_ranks() {
        // Cluster 4 owns two turns touching the segment (200 + 300 = 500 ms);
        // cluster 9 owns one (400 ms). Per-cluster summing makes 4 the winner
        // even though no single 4-turn beats the 9-turn.
        let turns = vec![
            turn(0.0, 1.2, 4), // [1000,2000) overlap 200 ms (1000..1200)
            turn(1.2, 1.6, 9), // overlap 400 ms (1200..1600)
            turn(1.6, 1.9, 4), // overlap 300 ms (1600..1900)
        ];
        let ranked = ranked_overlaps(&turns, 1_000, 2_000);
        assert_eq!(ranked, vec![(4, 500), (9, 400)]);
    }

    #[test]
    fn overlay_single_speaker_one_label() {
        let turns = vec![turn(0.0, 10.0, 2)];
        let mut segs = vec![seg(0, 2_000), seg(3_000, 5_000), seg(6_000, 9_000)];
        let count = overlay_speakers(&turns, &mut segs, &no_prune());
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
        let count = overlay_speakers(&turns, &mut segs, &no_prune());
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
        assert_eq!(overlay_speakers(&turns, &mut segs, &no_prune()), 0);
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
        overlay_speakers(&turns, &mut segs, &no_prune());
        assert_eq!(segs[0].speaker_id, None);
    }

    // -----------------------------------------------------------------------
    // Post-cluster prune + cap (issue #63). Model-free: synthetic turns drive
    // the duration-weighted prune and the speech-mass cap directly.
    // -----------------------------------------------------------------------

    /// A config with the share prune at `share`, the segment-count prune at
    /// `min_segs`, and the cap at `cap`. Everything else neutral.
    fn pruned(share: f32, min_segs: usize, cap: Option<usize>) -> DiarizerConfig {
        DiarizerConfig {
            min_cluster_share: share,
            min_cluster_segments: min_segs,
            max_speakers: cap,
            ..no_prune()
        }
    }

    #[test]
    fn prune_drops_tiny_share_cluster_and_reassigns() {
        // Cluster 0 owns ~9.9 s of speech; cluster 1 owns one 100 ms blip
        // (~1% of the total). With a 2% share floor the blip is pruned and its
        // segment reassigns to its next-best surviving cluster (0). One speaker
        // survives.
        let turns = vec![
            turn(0.0, 5.0, 0),
            turn(5.0, 5.1, 1), // tiny spurious turn (100 ms)
            turn(5.1, 10.0, 0),
        ];
        // Segments: long ones on cluster 0, one short segment straddling the
        // blip whose top overlap is cluster 1 but which also overlaps 0.
        let mut segs = vec![
            seg(0, 4_900),
            seg(5_000, 5_300), // top overlap = cluster 1 (100 ms) vs cluster 0 (200 ms)
            seg(5_400, 9_900),
        ];
        // The straddling segment's winner is cluster 0 (200 ms > 100 ms), so the
        // blip cluster only ever wins via a turn no segment maxes on — exercise
        // the share prune with a segment that DOES max on the blip.
        segs[1] = seg(5_000, 5_150); // overlaps cluster 1 by 100 ms, cluster 0 by 50 ms
        let count = overlay_speakers(&turns, &mut segs, &pruned(0.02, 0, None));
        assert_eq!(count, 1, "tiny-share cluster must be pruned away");
        // The straddling segment reassigned to the surviving cluster 0 → "A".
        assert_eq!(segs[1].speaker_id.as_deref(), Some("A"));
        for s in &segs {
            assert_eq!(s.speaker_id.as_deref(), Some("A"));
        }
    }

    #[test]
    fn prune_keeps_genuinely_distinct_speakers() {
        // Two balanced speakers, each with several segments (so neither trips the
        // segment-count floor) and each owning ~half the speech (so neither trips
        // the share floor). Both must survive — the prune must not collapse
        // genuine speakers.
        let turns = vec![turn(0.0, 5.0, 0), turn(5.0, 10.0, 1)];
        let mut segs = vec![
            seg(0, 2_400),
            seg(2_500, 4_900),
            seg(5_100, 7_400),
            seg(7_500, 9_900),
        ];
        let count = overlay_speakers(&turns, &mut segs, &pruned(0.02, 2, None));
        assert_eq!(count, 2);
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[2].speaker_id.as_deref(), Some("B"));
        assert_eq!(segs[3].speaker_id.as_deref(), Some("B"));
    }

    #[test]
    fn prune_by_segment_count_drops_single_segment_cluster() {
        // Cluster 1 wins exactly one segment; cluster 0 wins three. With
        // min_cluster_segments = 2 (and the share floor off), cluster 1 is
        // pruned and its one segment reassigns to its next-best surviving
        // cluster (0), which it also overlaps.
        let turns = vec![
            turn(0.0, 3.0, 0),
            turn(2.9, 3.6, 1), // cluster-1 turn, mostly in the 3.0..3.5 gap
            turn(3.5, 8.0, 0),
        ];
        let mut segs = vec![
            seg(0, 1_000),
            seg(1_000, 2_000),
            // Overlaps cluster 1 by 650 ms (2900..3550) and cluster 0 by 150 ms
            // (2900..3000 + 3500..3550); cluster 1 wins initially, then is pruned
            // and the segment reassigns to cluster 0.
            seg(2_900, 3_550),
            seg(4_100, 5_000),
        ];
        let count = overlay_speakers(&turns, &mut segs, &pruned(0.0, 2, None));
        assert_eq!(count, 1, "single-segment cluster pruned by the count floor");
        assert_eq!(segs[2].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn cap_collapses_to_largest_speakers() {
        // Three clusters, all above the share floor; cap at 2 keeps the two
        // largest by duration (0: 6 s, 1: 4 s) and folds cluster 2 (2 s) into its
        // nearest survivor.
        let turns = vec![
            turn(0.0, 6.0, 0),
            turn(6.0, 10.0, 1),
            turn(10.0, 12.0, 2), // smallest → capped out
        ];
        let mut segs = vec![
            seg(0, 5_900),
            seg(6_100, 9_900),
            seg(10_100, 11_900), // overlaps only cluster 2; after cap, reassigns
        ];
        let count = overlay_speakers(&turns, &mut segs, &pruned(0.0, 0, Some(2)));
        assert_eq!(count, 2, "cap must keep exactly the two largest speakers");
        // The capped-out segment had no overlap with a survivor → None (no
        // surviving turn covers [10100,11900)).
        assert_eq!(segs[2].speaker_id, None);
    }

    #[test]
    fn prune_never_returns_zero_for_nonempty_assignment() {
        // A single cluster owns everything but the share floor is impossibly
        // high (1.0 would prune the lone 100%-share cluster on the strict `<`?
        // No — 1.0 is not < 1.0). Use > 1.0 to force the all-pruned branch and
        // confirm the largest cluster is retained.
        let turns = vec![turn(0.0, 5.0, 0)];
        let mut segs = vec![seg(0, 4_900)];
        let count = overlay_speakers(&turns, &mut segs, &pruned(2.0, 0, None));
        assert_eq!(count, 1, "the largest cluster is retained when all prune");
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn default_config_prunes_a_long_recording_oversplit() {
        // Simulates the real over-split: one dominant speaker (cluster 0, ~30 s
        // of speech) plus a scatter of tiny drifted-embedding clusters, each a
        // single short turn. DiarizerConfig::default() (share 0.02, min segs 2)
        // must fold the scatter back into the dominant speaker.
        let mut turns = vec![turn(0.0, 30.0, 0)];
        let mut segs = vec![seg(0, 29_900)];
        // 8 tiny spurious clusters, each a 200 ms turn + a segment landing on it.
        for k in 0..8u64 {
            let t = 30.0 + k as f32 * 1.0;
            turns.push(turn(t, t + 0.2, (k + 1) as i32));
            let start = (t * 1000.0) as u64;
            segs.push(seg(start, start + 200));
        }
        let count = overlay_speakers(&turns, &mut segs, &DiarizerConfig::default());
        assert_eq!(count, 1, "default prune collapses the tiny-cluster scatter");
    }

    // -----------------------------------------------------------------------
    // Tests for overlay_speakers_from_prior
    // -----------------------------------------------------------------------

    fn prior(start_ms: u64, end_ms: u64, label: &str) -> (u64, u64, Option<String>) {
        (start_ms, end_ms, Some(label.to_string()))
    }
    fn prior_none(start_ms: u64, end_ms: u64) -> (u64, u64, Option<String>) {
        (start_ms, end_ms, None)
    }

    /// A new segment that fully overlaps a prior "A" segment gets "A".
    #[test]
    fn from_prior_full_overlap() {
        let prior_segs = vec![prior(0, 3000, "A")];
        let mut new_segs = vec![seg(0, 3000)];
        overlay_speakers_from_prior(&mut new_segs, &prior_segs);
        assert_eq!(new_segs[0].speaker_id.as_deref(), Some("A"));
    }

    /// A new segment that partially overlaps two prior segments gets the one with
    /// greater overlap.
    #[test]
    fn from_prior_max_overlap_wins() {
        // Prior: A [0, 2000), B [2000, 5000)
        // New: [1500, 4000) — 500 ms overlap with A, 2000 ms overlap with B → B wins.
        let prior_segs = vec![prior(0, 2000, "A"), prior(2000, 5000, "B")];
        let mut new_segs = vec![seg(1500, 4000)];
        overlay_speakers_from_prior(&mut new_segs, &prior_segs);
        assert_eq!(new_segs[0].speaker_id.as_deref(), Some("B"));
    }

    /// On an exact overlap tie, the earliest prior segment in slice order wins
    /// (the documented orientation, matching the offline join).
    #[test]
    fn from_prior_tie_earliest_wins() {
        // Prior: A [0, 1000), B [1000, 2000)
        // New: [500, 1500) — 500 ms overlap with each → A (earlier) wins.
        let prior_segs = vec![prior(0, 1000, "A"), prior(1000, 2000, "B")];
        let mut new_segs = vec![seg(500, 1500)];
        overlay_speakers_from_prior(&mut new_segs, &prior_segs);
        assert_eq!(new_segs[0].speaker_id.as_deref(), Some("A"));
    }

    /// A new segment in a gap between prior segments (no overlap) stays None.
    #[test]
    fn from_prior_gap_stays_none() {
        // Prior: A [0, 1000), B [2000, 3000) — gap at [1000, 2000).
        let prior_segs = vec![prior(0, 1000, "A"), prior(2000, 3000, "B")];
        let mut new_segs = vec![seg(1100, 1900)];
        overlay_speakers_from_prior(&mut new_segs, &prior_segs);
        assert_eq!(new_segs[0].speaker_id, None, "no prior overlap → None");
    }

    /// Labels from prior segments are preserved verbatim (no re-lettering).
    #[test]
    fn from_prior_label_survival() {
        let prior_segs = vec![prior(0, 1000, "A"), prior(1000, 2000, "B")];
        let mut new_segs = vec![seg(0, 1000), seg(1000, 2000)];
        overlay_speakers_from_prior(&mut new_segs, &prior_segs);
        assert_eq!(new_segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(new_segs[1].speaker_id.as_deref(), Some("B"));
    }

    /// An empty prior list leaves all new segments as None.
    #[test]
    fn from_prior_empty_old_list() {
        let mut new_segs = vec![seg(0, 1000), seg(1000, 2000)];
        overlay_speakers_from_prior(&mut new_segs, &[]);
        assert!(
            new_segs.iter().all(|s| s.speaker_id.is_none()),
            "empty prior → all None"
        );
    }

    /// A prior segment that itself had None (meeting was never diarized) leaves
    /// new segments as None.
    #[test]
    fn from_prior_prior_was_none() {
        let prior_segs = vec![prior_none(0, 3000)];
        let mut new_segs = vec![seg(0, 3000)];
        overlay_speakers_from_prior(&mut new_segs, &prior_segs);
        assert_eq!(new_segs[0].speaker_id, None);
    }
}
