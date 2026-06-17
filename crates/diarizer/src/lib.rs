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
    /// Multi-speaker flag (#0002): a SURVIVING cluster other than a segment's
    /// chosen primary contributes its label to `Segment::shared_speakers` when it
    /// overlaps at least this fraction (`[0.0, 1.0]`) of the segment's duration —
    /// and only when the primary is itself that substantial. `0.0` disables the
    /// flag. Presentation only; the segment is not split.
    pub multi_speaker_min_share: f32,
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
            // Flag a segment as multi-speaker when a second surviving speaker
            // covers ≥ 30% of it (and the primary does too). 0.30 keeps the flag
            // to genuinely shared segments — a brief interjection (a short
            // back-channel "mm-hm") stays below it, so single-speaker rows are
            // not noised up.
            multi_speaker_min_share: 0.30,
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
        segments: Vec<Segment>,
    ) -> AppResult<(Vec<Segment>, u32)> {
        require_supported_sample_rate(sample_rate)?;

        // Nothing to assign — empty transcript or empty audio. Don't invoke
        // sherpa (its `compute` bail!s on zero-length input).
        if segments.is_empty() || audio.is_empty() {
            return Ok((segments, 0));
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
/// interval-join, applying the `config`'s post-cluster prune + cap, splitting a
/// mixed word-timestamped segment at its turn boundaries, and return the owned
/// (possibly longer) segment list plus the distinct-label count.
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
/// A segment that spans two or more surviving clusters above
/// `multi_speaker_min_share` is **mixed** ([`is_mixed_segment`]). #0015 phase 3
/// resolves a mixed segment by its backend:
/// - **Parakeet** (non-empty `words`) — [`split_segment_by_words`] regroups the
///   words into one sub-segment per contiguous same-cluster run (no re-ASR, no
///   audio cut). The split grows the list, which is why the owned `Vec` is taken
///   in and returned out.
/// - **Qwen** (empty `words`) — kept as one segment on its dominant cluster and
///   flagged via `shared_speakers` (#0002); Phase 4 will re-ASR it.
///
/// The surviving cluster ids (per kept segment and per split sub-segment) are
/// relabelled to first-seen order `"A"`, `"B"`, … across the OUTPUT segments in
/// order — so a split's sub-segments letter in transcript order. The return
/// value is the number of distinct labels actually assigned (segments left
/// `None` do not count).
///
/// Pure (no FFI, no I/O) so the default test suite covers it without a model.
pub fn overlay_speakers(
    turns: &[SherpaSegment],
    segments: Vec<Segment>,
    config: &DiarizerConfig,
) -> (Vec<Segment>, u32) {
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
    // segments to their next-best survivor. `survivor_set` is the surviving
    // cluster ids whether or not the prune ran (the distinct chosen ids when
    // nothing was pruned), as `is_mixed_segment` needs it either way.
    let survivors = surviving_clusters(&chosen, &ranked, config);
    let survivor_set: Vec<i32> = match &survivors {
        Some(survivors) => {
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
            survivors.clone()
        }
        None => {
            let mut ids: Vec<i32> = Vec::new();
            for id in chosen.iter().flatten() {
                if !ids.contains(id) {
                    ids.push(*id);
                }
            }
            ids
        }
    };

    // Build the intermediate output by consuming the input segments. A mixed
    // Parakeet segment (has words) splits into one sub-segment per same-cluster
    // word run; everything else is kept as one segment on its chosen cluster.
    // `out_cluster[j]` is the cluster id (pre-relabel) backing `out[j]`;
    // `mixed_kept[j]` flags an out segment that is a kept (un-split) mixed
    // segment — the no-words/Qwen case that still needs a `shared_speakers` fill,
    // carrying its original `ranked` index for the fill below.
    let min_share = config.multi_speaker_min_share;
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut out_cluster: Vec<Option<i32>> = Vec::with_capacity(segments.len());
    let mut mixed_kept: Vec<Option<usize>> = Vec::with_capacity(segments.len());
    for (i, mut seg) in segments.into_iter().enumerate() {
        let dur = seg.end_ms.saturating_sub(seg.start_ms);
        let mixed = is_mixed_segment(&ranked[i], &survivor_set, dur, min_share);
        if mixed && !seg.words.is_empty() {
            for (mut sub, cluster) in split_segment_by_words(&seg, turns, &survivor_set) {
                sub.shared_speakers = Vec::new();
                out.push(sub);
                out_cluster.push(Some(cluster));
                mixed_kept.push(None);
            }
        } else {
            // Cleared here; a kept-MIXED (no-words/Qwen) segment may have this
            // re-filled by the #0002 loop below, after the relabel.
            seg.shared_speakers = Vec::new();
            out.push(seg);
            out_cluster.push(chosen[i]);
            mixed_kept.push(if mixed { Some(i) } else { None });
        }
    }

    // First-seen-order label map over the OUTPUT cluster ids, in output order
    // (so a split's sub-segments letter in transcript order, top-to-bottom).
    let mut seen: Vec<i32> = Vec::new();
    for id in out_cluster.iter().flatten() {
        if !seen.contains(id) {
            seen.push(*id);
        }
    }
    let label_for = |cluster: i32| -> String {
        let idx = seen
            .iter()
            .position(|s| *s == cluster)
            .expect("seen contains every out cluster id");
        alpha_label(idx)
    };
    for (seg, cluster) in out.iter_mut().zip(out_cluster.iter()) {
        seg.speaker_id = cluster.map(|c| label_for(c));
    }

    // #0002 `shared_speakers` fill — AFTER the relabel (it needs the label map),
    // and ONLY for kept mixed (no-words/Qwen) segments: a split Parakeet segment
    // is already resolved per-speaker and keeps `shared_speakers` empty. A
    // surviving cluster other than the segment's chosen primary contributes its
    // label when its overlap reaches `multi_speaker_min_share` of the segment
    // duration — and only when the primary is itself that substantial. Restricted
    // to clusters in `seen`, so every shared label matches a `speaker_id` shown
    // elsewhere in the transcript.
    if min_share > 0.0 {
        for (seg, orig) in out.iter_mut().zip(mixed_kept.iter()) {
            let Some(i) = *orig else { continue };
            let Some(primary) = chosen[i] else { continue };
            let dur = seg.end_ms.saturating_sub(seg.start_ms);
            if dur == 0 {
                continue;
            }
            let threshold = (min_share as f64 * dur as f64) as u64;
            let r = &ranked[i];
            let primary_overlap = r
                .iter()
                .find(|(id, _)| *id == primary)
                .map(|&(_, ov)| ov)
                .unwrap_or(0);
            if primary_overlap < threshold {
                continue;
            }
            seg.shared_speakers = r
                .iter()
                .filter(|(id, ov)| *id != primary && *ov >= threshold && seen.contains(id))
                .map(|&(id, _)| label_for(id))
                .collect();
        }
    }

    (out, seen.len() as u32)
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

/// True when a segment spans **two or more distinct surviving speakers**, each
/// covering at least `min_share` of the segment's duration — the predicate the
/// #0015 split uses to decide a segment is "mixed" and must be cut at the turn
/// boundary (rather than collapsed onto one label + flagged).
///
/// It is the same arithmetic as the `shared_speakers` fill in
/// [`overlay_speakers`], factored out so the split and the flag agree by
/// construction. `ranked` is one segment's per-cluster overlaps (descending, as
/// [`ranked_overlaps`] returns); `survivors` is the post-prune surviving
/// cluster-id set (so a pruned drift-cluster never counts); `dur_ms` is the
/// segment duration. `min_share <= 0.0` or a zero-duration segment is never
/// mixed.
fn is_mixed_segment(ranked: &[(i32, u64)], survivors: &[i32], dur_ms: u64, min_share: f32) -> bool {
    if min_share <= 0.0 || dur_ms == 0 {
        return false;
    }
    let threshold = (min_share as f64 * dur_ms as f64) as u64;
    let qualifying = ranked
        .iter()
        .filter(|(id, overlap)| *overlap >= threshold && survivors.contains(id))
        .count();
    qualifying >= 2
}

/// Winning sherpa cluster for a word: the SURVIVING cluster with the greatest
/// total overlap of the word's `[word_start_ms, word_end_ms)` against the `turns`,
/// lower cluster id breaking a tie. `None` when no surviving cluster overlaps the
/// word.
///
/// Reuses [`ranked_overlaps`]' per-cluster summing (via [`interval_overlap_ms`] +
/// [`seconds_to_ms`]), then skips any cluster the issue-63 prune dropped from
/// `survivors` — so the split path respects the prune exactly as the kept-segment
/// reassignment does: a word whose top overlap is a pruned drift-cluster
/// attributes to its next-best surviving cluster, not the dropped one. Parakeet
/// word ENDS are approximate (taken as the next word's start), so a word that
/// straddles a turn boundary attributes by max overlap — the right call.
fn word_turn(
    word_start_ms: u64,
    word_end_ms: u64,
    turns: &[SherpaSegment],
    survivors: &[i32],
) -> Option<i32> {
    ranked_overlaps(turns, word_start_ms, word_end_ms)
        .into_iter()
        .find(|(id, _)| survivors.contains(id))
        .map(|(id, _)| id)
}

/// Split one segment's words into maximal contiguous same-cluster runs, emitting
/// one `(Segment, cluster_id)` per run.
///
/// Each emitted segment's `text` is its run's words joined by a single space;
/// `words` is the run's [`WordTimestamp`] slice; `start_ms`/`end_ms` are the
/// run's first/last word bounds; `confidence` is carried from `seg`;
/// `speaker_id` is left `None` (the caller relabels in output order) and
/// `shared_speakers` is empty.
///
/// A word whose [`word_turn`] is `None` (no SURVIVING cluster overlaps it) joins
/// the PRECEDING run (no orphan sub-segment); when the FIRST word is `None` it
/// seeds a run with the segment's dominant surviving cluster (max overlap over the
/// whole segment). `survivors` is the post-prune surviving cluster set, threaded
/// in so the split never mints a run on a cluster the prune dropped. If `seg.words`
/// is empty or every word maps to ONE cluster, a single (rebuilt) segment is
/// returned, so the caller can split unconditionally on a mixed Parakeet segment.
fn split_segment_by_words(
    seg: &Segment,
    turns: &[SherpaSegment],
    survivors: &[i32],
) -> Vec<(Segment, i32)> {
    // Build a sub-segment from a run of words on one cluster: bounds from the
    // first/last word, text space-joined, words cloned, confidence carried.
    let build = |words: &[minutist_common::WordTimestamp], cluster: i32| -> (Segment, i32) {
        let start_ms = words.first().map(|w| w.start_ms).unwrap_or(seg.start_ms);
        let end_ms = words.last().map(|w| w.end_ms).unwrap_or(seg.end_ms);
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let sub = Segment {
            start_ms,
            end_ms,
            text,
            speaker_id: None,
            confidence: seg.confidence,
            words: words.to_vec(),
            shared_speakers: Vec::new(),
        };
        (sub, cluster)
    };

    if seg.words.is_empty() {
        // No words to regroup — rebuild the whole segment on its dominant cluster
        // (cluster 0 if nothing overlaps, which a mixed segment never hits).
        let cluster = word_turn(seg.start_ms, seg.end_ms, turns, survivors).unwrap_or(0);
        return vec![build(&seg.words, cluster)];
    }

    // The segment's dominant cluster seeds a leading None word (and is the
    // fallback if every word maps to None).
    let dominant = word_turn(seg.start_ms, seg.end_ms, turns, survivors).unwrap_or(0);

    let mut out: Vec<(Segment, i32)> = Vec::new();
    let mut run_cluster: Option<i32> = None;
    let mut run_start: usize = 0;
    for (k, w) in seg.words.iter().enumerate() {
        // A word that maps to no turn joins the current run; the first word seeds
        // the run with the segment's dominant cluster.
        let cluster = word_turn(w.start_ms, w.end_ms, turns, survivors).unwrap_or_else(|| match run_cluster {
            Some(c) => c,
            None => dominant,
        });
        match run_cluster {
            Some(c) if c == cluster => {}
            Some(c) => {
                out.push(build(&seg.words[run_start..k], c));
                run_start = k;
                run_cluster = Some(cluster);
            }
            None => run_cluster = Some(cluster),
        }
    }
    let last = run_cluster.unwrap_or(dominant);
    out.push(build(&seg.words[run_start..], last));
    out
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

/// Merge runs of adjacent segments that share the same `speaker_id` into one
/// segment (#0015 phase 1), so a single speaker fragmented by the VAD — a
/// sub-`gap_threshold_ms` silence, or the 10 s force-split that abuts segments
/// with no gap — reads as one continuous turn rather than several rows.
///
/// Two neighbours fold together when BOTH hold:
/// - identical `speaker_id` (`Some(a) == Some(b)` by string); a `None` label is
///   an opaque hard boundary, never merged with anything — two un-attributable
///   gaps could span a real turn the diarizer did not cover; and
/// - the inter-segment gap `next.start_ms - cur.end_ms` is `<= gap_threshold_ms`
///   (`saturating_sub`, so an abutting/overlapping force-split pair gaps to `0`
///   and always folds).
///
/// Fold mechanics: `text` joins with a single space (an empty side contributes
/// nothing, so no stray padding); `words` concatenate in order (Qwen segments
/// carry none, so the one rule is correct for both backends); `end_ms` takes the
/// union upper bound while `start_ms` stays the first member's; `shared_speakers`
/// is the de-duplicated union minus the run's own label; `confidence` is the
/// member-duration-weighted mean of the `Some` members (an all-`None` run — the
/// only case the live backends produce — stays `None`).
///
/// Segments are assumed start-sorted (the offline transcript is) and are not
/// re-sorted. Pure (no FFI/IO), so the default test suite covers it without a
/// model. The caller recomputes `speaker_count` afterwards; the merge preserves
/// labels, so the count is invariant, but recomputing is the robust choice.
pub fn merge_adjacent_speakers(segments: &mut Vec<Segment>, gap_threshold_ms: u64) {
    if segments.len() < 2 {
        return;
    }

    // Finalise a completed run: resolve the duration-weighted confidence and drop
    // the run's own label from `shared_speakers` (a label is never its own
    // "additional" speaker).
    fn finish_run(seg: &mut Segment, conf_weighted_sum: f64, conf_weight: u64) {
        seg.confidence = if conf_weight > 0 {
            Some((conf_weighted_sum / conf_weight as f64) as f32)
        } else {
            None
        };
        if let Some(own) = seg.speaker_id.clone() {
            seg.shared_speakers.retain(|s| *s != own);
        }
    }

    let dur = |s: &Segment| s.end_ms.saturating_sub(s.start_ms);

    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut iter = std::mem::take(segments).into_iter();
    let mut cur = iter.next().expect("len >= 2 checked above");
    let mut conf_sum = cur.confidence.map_or(0.0, |c| c as f64 * dur(&cur) as f64);
    let mut conf_w = cur.confidence.map_or(0, |_| dur(&cur));

    for next in iter {
        let same_speaker =
            matches!((&cur.speaker_id, &next.speaker_id), (Some(a), Some(b)) if a == b);
        let gap = next.start_ms.saturating_sub(cur.end_ms);
        if same_speaker && gap <= gap_threshold_ms {
            // Fold `next` into the running segment.
            let head = cur.text.trim_end();
            let tail = next.text.trim_start();
            let merged_text = if head.is_empty() {
                tail.to_string()
            } else if tail.is_empty() {
                head.to_string()
            } else {
                format!("{head} {tail}")
            };
            cur.text = merged_text;
            cur.end_ms = next.end_ms.max(cur.end_ms);
            cur.words.extend(next.words);
            for label in next.shared_speakers {
                if !cur.shared_speakers.contains(&label) {
                    cur.shared_speakers.push(label);
                }
            }
            if let Some(c) = next.confidence {
                // Copy fields only — `next` is partially moved (words/shared) above.
                let w = next.end_ms.saturating_sub(next.start_ms);
                conf_sum += c as f64 * w as f64;
                conf_w += w;
            }
        } else {
            finish_run(&mut cur, conf_sum, conf_w);
            out.push(cur);
            cur = next;
            conf_sum = cur.confidence.map_or(0.0, |c| c as f64 * dur(&cur) as f64);
            conf_w = cur.confidence.map_or(0, |_| dur(&cur));
        }
    }
    finish_run(&mut cur, conf_sum, conf_w);
    out.push(cur);
    *segments = out;
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
            shared_speakers: Vec::new(),
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
            // Disabled here so the relabel/prune tests are unaffected; the
            // multi-speaker flag has its own dedicated test with a set share.
            multi_speaker_min_share: 0.0,
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
        let segs = vec![
            seg(0, 1_900),     // overlaps the 7-turn → A
            seg(2_100, 3_900), // overlaps the 3-turn → B
            seg(4_100, 5_900), // overlaps the 7-turn → A
        ];
        let (segs, count) = overlay_speakers(&turns, segs, &no_prune());
        assert_eq!(count, 2);
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(segs[2].speaker_id.as_deref(), Some("A"));
    }

    #[test]
    fn overlay_no_overlap_yields_none() {
        let turns = vec![turn(0.0, 1.0, 0)];
        // Segment sits entirely after the only turn → no overlap → None.
        let segs = vec![seg(5_000, 6_000)];
        let (segs, count) = overlay_speakers(&turns, segs, &no_prune());
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
        let segs = vec![seg(1_000, 2_000)];
        let (segs, count) = overlay_speakers(&turns, segs, &no_prune());
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
        let segs = vec![seg(1_000, 2_000)];
        let (segs, count) = overlay_speakers(&turns, segs, &no_prune());
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
        let segs = vec![seg(0, 2_000), seg(3_000, 5_000), seg(6_000, 9_000)];
        let (segs, count) = overlay_speakers(&turns, segs, &no_prune());
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
        let segs = vec![
            seg(0, 900),       // A
            seg(1_100, 1_900), // B
            seg(2_500, 3_500), // gap → None
            seg(4_100, 4_900), // C
        ];
        let (segs, count) = overlay_speakers(&turns, segs, &no_prune());
        assert_eq!(count, 3);
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(segs[2].speaker_id, None);
        assert_eq!(segs[3].speaker_id.as_deref(), Some("C"));
    }

    #[test]
    fn overlay_empty_segments_is_zero() {
        let turns = vec![turn(0.0, 1.0, 0)];
        let segs: Vec<Segment> = Vec::new();
        let (segs, count) = overlay_speakers(&turns, segs, &no_prune());
        assert_eq!(count, 0);
        assert!(segs.is_empty());
    }

    #[test]
    fn overlay_clears_stale_speaker_id_on_no_overlap() {
        // A re-diarize pass must overwrite a previously-set label, including
        // back to None when the new turns no longer cover the segment.
        let turns = vec![turn(0.0, 1.0, 0)];
        let segs = vec![Segment {
            speaker_id: Some("Z".to_string()),
            ..seg(5_000, 6_000)
        }];
        let (segs, _count) = overlay_speakers(&turns, segs, &no_prune());
        assert_eq!(segs[0].speaker_id, None);
    }

    /// #0002: a config with the multi-speaker flag enabled at the default share.
    fn flag_share() -> DiarizerConfig {
        DiarizerConfig {
            multi_speaker_min_share: 0.30,
            ..no_prune()
        }
    }

    #[test]
    fn overlay_flags_a_segment_spanning_two_speakers() {
        // Segment 0 [0,1000): cluster 0 covers all 1000 ms, cluster 1 covers the
        // last 600 ms (60% ≥ 30%). Segment 1 [1000,2000): cluster 1 alone — this
        // is what gives cluster 1 a label ("B") so it can appear in segment 0's
        // shared list.
        let turns = vec![
            turn(0.0, 1.0, 0),
            turn(0.4, 1.0, 1),
            turn(1.0, 2.0, 1),
        ];
        // Empty `words` (the Qwen path): a mixed segment is NOT split — it keeps
        // its dominant label + the `shared_speakers` flag.
        let segs = vec![seg(0, 1_000), seg(1_000, 2_000)];
        let (segs, _count) = overlay_speakers(&turns, segs, &flag_share());

        // Primary labels by first-seen order.
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[1].speaker_id.as_deref(), Some("B"));
        // Segment 0 is flagged shared with B; segment 1 is single-speaker.
        assert_eq!(segs[0].shared_speakers, vec!["B".to_string()]);
        assert!(segs[1].shared_speakers.is_empty());
    }

    #[test]
    fn overlay_does_not_flag_a_brief_secondary_overlap() {
        // Cluster 1 covers only the last 200 ms of segment 0 (20% < 30%), so the
        // segment is NOT flagged shared even though a second speaker is present.
        let turns = vec![
            turn(0.0, 1.0, 0),
            turn(0.8, 1.0, 1),
            turn(1.0, 2.0, 1),
        ];
        let segs = vec![seg(0, 1_000), seg(1_000, 2_000)];
        let (segs, _count) = overlay_speakers(&turns, segs, &flag_share());

        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert!(
            segs[0].shared_speakers.is_empty(),
            "a 20% secondary overlap is below the 30% share threshold"
        );
    }

    #[test]
    fn overlay_does_not_flag_when_disabled() {
        // `multi_speaker_min_share == 0.0` (no_prune default) disables the flag.
        let turns = vec![
            turn(0.0, 1.0, 0),
            turn(0.4, 1.0, 1),
            turn(1.0, 2.0, 1),
        ];
        let segs = vec![seg(0, 1_000), seg(1_000, 2_000)];
        let (segs, _count) = overlay_speakers(&turns, segs, &no_prune());
        assert!(segs[0].shared_speakers.is_empty());
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
        let (segs, count) = overlay_speakers(&turns, segs, &pruned(0.02, 0, None));
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
        let segs = vec![
            seg(0, 2_400),
            seg(2_500, 4_900),
            seg(5_100, 7_400),
            seg(7_500, 9_900),
        ];
        let (segs, count) = overlay_speakers(&turns, segs, &pruned(0.02, 2, None));
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
        let segs = vec![
            seg(0, 1_000),
            seg(1_000, 2_000),
            // Overlaps cluster 1 by 650 ms (2900..3550) and cluster 0 by 150 ms
            // (2900..3000 + 3500..3550); cluster 1 wins initially, then is pruned
            // and the segment reassigns to cluster 0.
            seg(2_900, 3_550),
            seg(4_100, 5_000),
        ];
        let (segs, count) = overlay_speakers(&turns, segs, &pruned(0.0, 2, None));
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
        let segs = vec![
            seg(0, 5_900),
            seg(6_100, 9_900),
            seg(10_100, 11_900), // overlaps only cluster 2; after cap, reassigns
        ];
        let (segs, count) = overlay_speakers(&turns, segs, &pruned(0.0, 0, Some(2)));
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
        let segs = vec![seg(0, 4_900)];
        let (segs, count) = overlay_speakers(&turns, segs, &pruned(2.0, 0, None));
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
        let (_segs, count) = overlay_speakers(&turns, segs, &DiarizerConfig::default());
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

    // -----------------------------------------------------------------------
    // merge_adjacent_speakers (#0015 phase 1)
    // -----------------------------------------------------------------------

    fn seg_sp(start_ms: u64, end_ms: u64, text: &str, speaker: Option<&str>) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.to_string(),
            speaker_id: speaker.map(|s| s.to_string()),
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        }
    }
    fn word(start_ms: u64, end_ms: u64, text: &str) -> minutist_common::WordTimestamp {
        minutist_common::WordTimestamp {
            start_ms,
            end_ms,
            text: text.to_string(),
        }
    }

    #[test]
    fn merge_force_split_zero_gap() {
        // The 10 s force-split abuts segments (end_ms == next.start_ms) for one
        // speaker → one merged segment; text space-joined, end_ms unioned.
        let mut segs = vec![
            seg_sp(0, 10_000, "first half", Some("A")),
            seg_sp(10_000, 18_000, "second half", Some("A")),
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_ms, 0);
        assert_eq!(segs[0].end_ms, 18_000);
        assert_eq!(segs[0].text, "first half second half");
    }

    #[test]
    fn merge_hangover_gap_below_threshold() {
        let mut below = vec![
            seg_sp(0, 1_000, "one", Some("A")),
            seg_sp(2_400, 3_000, "two", Some("A")), // 1400 ms gap <= 1500
        ];
        merge_adjacent_speakers(&mut below, 1500);
        assert_eq!(below.len(), 1, "1400 ms gap is below threshold");

        let mut above = vec![
            seg_sp(0, 1_000, "one", Some("A")),
            seg_sp(2_600, 3_000, "two", Some("A")), // 1600 ms gap > 1500
        ];
        merge_adjacent_speakers(&mut above, 1500);
        assert_eq!(above.len(), 2, "1600 ms gap exceeds threshold");
    }

    #[test]
    fn merge_stops_at_speaker_change() {
        let mut segs = vec![
            seg_sp(0, 1_000, "a1", Some("A")),
            seg_sp(1_000, 2_000, "a2", Some("A")),
            seg_sp(2_000, 3_000, "b1", Some("B")),
            seg_sp(3_000, 4_000, "b2", Some("B")),
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(segs[0].text, "a1 a2");
        assert_eq!(segs[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(segs[1].text, "b1 b2");
    }

    #[test]
    fn merge_none_is_a_hard_boundary() {
        // A None segment never merges and never bridges the two A's around it.
        let mut segs = vec![
            seg_sp(0, 1_000, "a1", Some("A")),
            seg_sp(1_000, 2_000, "gap", None),
            seg_sp(2_000, 3_000, "a2", Some("A")),
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[1].speaker_id, None);
    }

    #[test]
    fn merge_words_concatenated_in_order() {
        let mut segs = vec![
            Segment {
                words: vec![word(0, 500, "hello"), word(500, 1000, "there")],
                ..seg_sp(0, 1_000, "hello there", Some("A"))
            },
            Segment {
                words: vec![word(1000, 1500, "general"), word(1500, 2000, "kenobi")],
                ..seg_sp(1_000, 2_000, "general kenobi", Some("A"))
            },
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 1);
        let texts: Vec<&str> = segs[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "there", "general", "kenobi"]);
    }

    #[test]
    fn merge_qwen_empty_words_stay_empty() {
        let mut segs = vec![
            seg_sp(0, 1_000, "one", Some("A")),
            seg_sp(1_000, 2_000, "two", Some("A")),
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].words.is_empty());
    }

    #[test]
    fn merge_text_join_handles_empty_sides() {
        let mut segs = vec![
            seg_sp(0, 1_000, "", Some("A")),
            seg_sp(1_000, 2_000, "spoken", Some("A")),
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "spoken", "no stray leading space from the empty side");
    }

    #[test]
    fn merge_shared_speakers_union_dedup_self_removed() {
        let mut segs = vec![
            Segment {
                shared_speakers: vec!["B".to_string(), "A".to_string()],
                ..seg_sp(0, 1_000, "x", Some("A"))
            },
            Segment {
                shared_speakers: vec!["B".to_string(), "C".to_string()],
                ..seg_sp(1_000, 2_000, "y", Some("A"))
            },
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 1);
        // Union {B, A, C} minus the run's own label A, first-seen order.
        assert_eq!(segs[0].shared_speakers, vec!["B".to_string(), "C".to_string()]);
    }

    #[test]
    fn merge_confidence_duration_weighted_mean() {
        // None + None stays None (the only case live backends produce).
        let mut none_run = vec![
            seg_sp(0, 1_000, "a", Some("A")),
            seg_sp(1_000, 2_000, "b", Some("A")),
        ];
        merge_adjacent_speakers(&mut none_run, 1500);
        assert_eq!(none_run[0].confidence, None);

        // Equal-duration Some(0.8) + Some(0.4) → mean 0.6.
        let mut conf_run = vec![
            Segment {
                confidence: Some(0.8),
                ..seg_sp(0, 1_000, "a", Some("A"))
            },
            Segment {
                confidence: Some(0.4),
                ..seg_sp(1_000, 2_000, "b", Some("A"))
            },
        ];
        merge_adjacent_speakers(&mut conf_run, 1500);
        assert_eq!(conf_run.len(), 1);
        let c = conf_run[0].confidence.expect("some");
        assert!((c - 0.6).abs() < 1e-5, "duration-weighted mean = 0.6, got {c}");
    }

    #[test]
    fn merge_single_segment_and_empty_are_noops() {
        let mut one = vec![seg_sp(0, 1_000, "solo", Some("A"))];
        merge_adjacent_speakers(&mut one, 1500);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].text, "solo");

        let mut empty: Vec<Segment> = Vec::new();
        merge_adjacent_speakers(&mut empty, 1500);
        assert!(empty.is_empty());
    }

    #[test]
    fn merge_preserves_start_and_unions_end() {
        // A later fragment whose end_ms is < the running end (overlap) must not
        // shrink the merged end_ms; start stays the first fragment's.
        let mut segs = vec![
            seg_sp(100, 5_000, "long", Some("A")),
            seg_sp(5_000, 4_000, "weird", Some("A")), // degenerate end < run end
        ];
        merge_adjacent_speakers(&mut segs, 1500);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_ms, 100);
        assert_eq!(segs[0].end_ms, 5_000, "union upper bound, not the smaller end");
    }

    // -----------------------------------------------------------------------
    // is_mixed_segment (#0015 phase 2) — mirrors the shared_speakers flag tests
    // -----------------------------------------------------------------------

    #[test]
    fn is_mixed_true_for_two_substantial_survivors() {
        // dur 1000 ms → 30% threshold = 300 ms; clusters 0 (1000) and 1 (600)
        // both clear it and both survive → mixed.
        let ranked = vec![(0i32, 1000u64), (1, 600)];
        assert!(is_mixed_segment(&ranked, &[0, 1], 1000, 0.30));
    }

    #[test]
    fn is_mixed_false_for_brief_secondary() {
        // Secondary covers only 200 ms (< 300 ms) → single substantial speaker.
        let ranked = vec![(0i32, 1000u64), (1, 200)];
        assert!(!is_mixed_segment(&ranked, &[0, 1], 1000, 0.30));
    }

    #[test]
    fn is_mixed_false_for_single_speaker() {
        let ranked = vec![(0i32, 1000u64)];
        assert!(!is_mixed_segment(&ranked, &[0], 1000, 0.30));
    }

    #[test]
    fn is_mixed_ignores_pruned_non_survivor() {
        // Cluster 1 clears the threshold but was pruned (not a survivor) → the
        // segment is attributed to the one survivor, not mixed.
        let ranked = vec![(0i32, 1000u64), (1, 600)];
        assert!(!is_mixed_segment(&ranked, &[0], 1000, 0.30));
    }

    #[test]
    fn is_mixed_disabled_when_share_zero_or_zero_duration() {
        let ranked = vec![(0i32, 1000u64), (1, 600)];
        assert!(!is_mixed_segment(&ranked, &[0, 1], 1000, 0.0));
        assert!(!is_mixed_segment(&ranked, &[0, 1], 0, 0.30));
    }

    // -----------------------------------------------------------------------
    // word_turn + split_segment_by_words + the overlay split path (#0015 phase 3)
    // -----------------------------------------------------------------------

    /// A segment carrying per-word timestamps (the Parakeet path); `text` is the
    /// words space-joined.
    fn seg_words(words: Vec<minutist_common::WordTimestamp>) -> Segment {
        let start_ms = words.first().map(|w| w.start_ms).unwrap_or(0);
        let end_ms = words.last().map(|w| w.end_ms).unwrap_or(0);
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Segment {
            start_ms,
            end_ms,
            text,
            speaker_id: None,
            confidence: None,
            words,
            shared_speakers: Vec::new(),
        }
    }

    #[test]
    fn word_turn_picks_max_overlap_cluster() {
        // The word [1000,2000) overlaps cluster 0 by 200 ms and cluster 1 by
        // 800 ms → cluster 1.
        let turns = vec![turn(0.0, 1.2, 0), turn(1.2, 3.0, 1)];
        assert_eq!(word_turn(1_000, 2_000, &turns, &[0, 1, 4, 9]), Some(1));
    }

    #[test]
    fn word_turn_tie_breaks_to_lower_cluster_id() {
        // Equal 500 ms overlap each → the lower cluster id (4) wins.
        let turns = vec![turn(0.0, 1.5, 9), turn(1.5, 3.0, 4)];
        assert_eq!(word_turn(1_000, 2_000, &turns, &[0, 1, 4, 9]), Some(4));
    }

    #[test]
    fn word_turn_none_when_no_overlap() {
        let turns = vec![turn(0.0, 1.0, 0)];
        assert_eq!(word_turn(5_000, 6_000, &turns, &[0, 1, 4, 9]), None);
    }

    #[test]
    fn split_three_runs_a_a_b_b_a() {
        // Words A A B B A; turns put [0,200)+[400,500) on cluster 0 and
        // [200,400) on cluster 1 → 3 contiguous runs.
        let turns = vec![turn(0.0, 0.2, 0), turn(0.2, 0.4, 1), turn(0.4, 0.5, 0)];
        let seg = seg_words(vec![
            word(0, 100, "a1"),
            word(100, 200, "a2"),
            word(200, 300, "b1"),
            word(300, 400, "b2"),
            word(400, 500, "a3"),
        ]);
        let runs = split_segment_by_words(&seg, &turns, &[0, 1]);
        assert_eq!(runs.len(), 3);
        // Run 0: cluster 0, words a1 a2, [0,200).
        assert_eq!(runs[0].1, 0);
        assert_eq!(runs[0].0.text, "a1 a2");
        assert_eq!((runs[0].0.start_ms, runs[0].0.end_ms), (0, 200));
        // Run 1: cluster 1, words b1 b2, [200,400).
        assert_eq!(runs[1].1, 1);
        assert_eq!(runs[1].0.text, "b1 b2");
        assert_eq!((runs[1].0.start_ms, runs[1].0.end_ms), (200, 400));
        // Run 2: cluster 0 again, word a3, [400,500).
        assert_eq!(runs[2].1, 0);
        assert_eq!(runs[2].0.text, "a3");
        assert_eq!((runs[2].0.start_ms, runs[2].0.end_ms), (400, 500));
        // Sub-segments carry no label (the caller relabels) and no flag.
        assert!(runs.iter().all(|(s, _)| s.speaker_id.is_none()));
        assert!(runs.iter().all(|(s, _)| s.shared_speakers.is_empty()));
    }

    #[test]
    fn split_empty_words_is_single_segment() {
        let turns = vec![turn(0.0, 1.0, 0)];
        let seg = seg(0, 1_000); // no words
        let runs = split_segment_by_words(&seg, &turns, &[0, 1]);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].1, 0);
    }

    #[test]
    fn split_single_cluster_words_is_one_segment() {
        // Every word maps to cluster 0 → a single rebuilt segment.
        let turns = vec![turn(0.0, 1.0, 0)];
        let seg = seg_words(vec![word(0, 300, "one"), word(300, 600, "two")]);
        let runs = split_segment_by_words(&seg, &turns, &[0, 1]);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].1, 0);
        assert_eq!(runs[0].0.text, "one two");
    }

    #[test]
    fn split_first_word_none_seeds_with_dominant_cluster() {
        // The first word [0,100) overlaps no turn (None) → it seeds the run with
        // the segment's dominant cluster (1, which owns most of [0,500)). The
        // remaining words are all cluster 1, so the whole segment stays one run
        // labelled 1.
        let turns = vec![turn(0.1, 0.5, 1)];
        let seg = seg_words(vec![
            word(0, 100, "lead"),
            word(100, 300, "mid"),
            word(300, 500, "end"),
        ]);
        let runs = split_segment_by_words(&seg, &turns, &[0, 1]);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].1, 1, "leading None word seeds with the dominant cluster");
        assert_eq!(runs[0].0.text, "lead mid end");
    }

    #[test]
    fn split_mid_run_none_word_joins_preceding_run() {
        // Words: cluster 0, NONE, cluster 0 — the middle word overlaps no turn so
        // it joins the preceding (cluster 0) run; no orphan run is emitted.
        let turns = vec![turn(0.0, 0.2, 0), turn(0.4, 0.6, 0)];
        let seg = seg_words(vec![
            word(0, 200, "a"),
            word(250, 350, "gap"), // [250,350) overlaps neither turn → None
            word(400, 600, "b"),
        ]);
        let runs = split_segment_by_words(&seg, &turns, &[0, 1]);
        assert_eq!(runs.len(), 1, "the None word folds into the preceding run");
        assert_eq!(runs[0].1, 0);
        assert_eq!(runs[0].0.text, "a gap b");
    }

    #[test]
    fn word_turn_skips_pruned_non_survivor_cluster() {
        // Cluster 1 has the larger overlap but was pruned (not a survivor), so the
        // word attributes to the only survivor (0) — the split path respects the
        // issue-63 prune. With both surviving, the max-overlap cluster (1) wins.
        let turns = vec![turn(0.0, 1.2, 0), turn(1.2, 3.0, 1)];
        assert_eq!(word_turn(1_000, 2_000, &turns, &[0]), Some(0));
        assert_eq!(word_turn(1_000, 2_000, &turns, &[0, 1]), Some(1));
    }

    #[test]
    fn split_respects_prune_pruned_cluster_word_joins_survivor() {
        // Cluster 5 owns [250,500) but was pruned; only cluster 0 survives. The
        // word on [250,500) maps to no survivor, so it folds into the surviving
        // run instead of resurrecting the pruned drift-cluster.
        let turns = vec![turn(0.0, 0.25, 0), turn(0.25, 0.5, 5)];
        let seg = seg_words(vec![word(0, 250, "a"), word(250, 500, "b")]);
        let runs = split_segment_by_words(&seg, &turns, &[0]); // cluster 5 pruned
        assert_eq!(runs.len(), 1, "pruned cluster 5 must not mint its own run");
        assert_eq!(runs[0].1, 0);
        assert_eq!(runs[0].0.text, "a b");
    }

    #[test]
    fn overlay_splits_a_mixed_parakeet_segment() {
        // VAD segment 0 [0,2000) hands over A→B mid-segment; segment 1 [2000,3000)
        // is cluster 1 alone (so cluster 1 wins a top slot and counts as a
        // surviving cluster). Turns: cluster 0 owns [0,1000), cluster 1 owns
        // [1000,3000). Segment 0 is mixed (each cluster 50% ≥ 30%) AND has words →
        // it splits into two single-speaker sub-segments, lettered in transcript
        // order A then B; segment 1 reuses B.
        let turns = vec![turn(0.0, 1.0, 0), turn(1.0, 3.0, 1)];
        let segs = vec![
            seg_words(vec![
                word(0, 400, "hello"),
                word(400, 1_000, "there"),
                word(1_000, 1_500, "general"),
                word(1_500, 2_000, "kenobi"),
            ]),
            seg(2_000, 3_000), // cluster 1 alone (no words)
        ];
        let (out, count) = overlay_speakers(&turns, segs, &flag_share());
        assert_eq!(count, 2, "the split yields two distinct speakers");
        assert_eq!(out.len(), 3, "the mixed segment split into two, plus segment 1");
        assert_eq!(out[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(out[0].text, "hello there");
        assert_eq!(out[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(out[1].text, "general kenobi");
        assert_eq!(out[2].speaker_id.as_deref(), Some("B"));
        // A split segment is resolved per-speaker — never flagged shared.
        assert!(out[0].shared_speakers.is_empty());
        assert!(out[1].shared_speakers.is_empty());
    }

    #[test]
    fn overlay_does_not_split_a_mixed_qwen_segment() {
        // Same mixed geometry but EMPTY words (the Qwen path): the segment is NOT
        // split — it keeps its dominant label and the shared_speakers flag.
        let turns = vec![
            turn(0.0, 1.0, 0),
            turn(0.4, 1.0, 1),
            turn(1.0, 2.0, 1),
        ];
        let segs = vec![seg(0, 1_000), seg(1_000, 2_000)];
        let (out, _count) = overlay_speakers(&turns, segs, &flag_share());
        assert_eq!(out.len(), 2, "no split on the no-words path");
        assert_eq!(out[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(out[0].shared_speakers, vec!["B".to_string()]);
    }
}
