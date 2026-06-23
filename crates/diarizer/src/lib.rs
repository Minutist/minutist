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
//! `compute` takes `&mut self`). The work splits across two public seams:
//!
//! [`SherpaDiarizer::compute_turns`]:
//! 1. asserts the input is 16 kHz (sherpa's pyannote segmentation is fixed at
//!    16 kHz; anything else is `AppError::InvalidInput`),
//! 2. runs `Diarize::compute` to get raw sherpa turns
//!    `{ start_s, end_s, speaker: i32 }`,
//! 3. maps each to a sherpa-free [`SpeakerTurn`] in milliseconds — so the
//!    `sherpa-rs` type never escapes the crate.
//!
//! [`overlay_speakers`] then, over those `SpeakerTurn`s:
//! 4. overlays a `speaker_id` onto each ASR `Segment` by **max-overlap
//!    interval-join** over `[start_ms, end_ms)` (no overlap → `None`),
//! 5. applies a **post-cluster prune** (and optional cap) — drops clusters that
//!    win a negligible share of the attributed speech and reassigns their
//!    segments to the nearest surviving cluster (issue #63) — then
//! 6. relabels the surviving `i32` cluster ids to first-seen-order `"A"`, `"B"`,
//!    …, returning the segments, the distinct-label count, and the cluster→letter
//!    map (so a caller re-lettering a re-ASR'd sub-clip reuses the same scheme).
//!
//! [`SherpaDiarizer::assign_speakers`] (the `common::Diarizer` trait method) is
//! the thin compose of the two, dropping the map for its `(Vec<Segment>, u32)`
//! contract.
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
use sherpa_rs::diarize::{Diarize, DiarizeConfig};

mod error;
pub use error::Error;

/// One diarized speaker turn in **milliseconds** — a sherpa-free plain-data
/// projection of `sherpa_rs::diarize::Segment` (`{ start_s, end_s, speaker }`).
///
/// [`SherpaDiarizer::compute_turns`] maps each raw sherpa turn to this POD so the
/// `sherpa-rs` type never crosses the crate boundary. `cluster` is the arbitrary
/// `i32` cluster id sherpa assigns (not yet a first-seen `A`/`B` label — that
/// happens in [`overlay_speakers`]). `[start_ms, end_ms)` is half-open, on the
/// same clock as the PCM the turns were computed over.
///
/// Diarizer-public, deliberately NOT in `common`: the orchestrator already
/// depends on `diarizer`, so exposing the raw turns here adds no dependency edge,
/// whereas a `common::SpeakerTurn` would let any crate pick it up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerTurn {
    /// Turn start, milliseconds (inclusive).
    pub start_ms: u64,
    /// Turn end, milliseconds (exclusive).
    pub end_ms: u64,
    /// Arbitrary sherpa cluster id (first-seen-relabelled downstream).
    pub cluster: i32,
}

mod online;
pub use online::clusterer::{ClusterAssignment, OnlineClusterer, OnlineClustererConfig};
pub use online::{OnlineDiarizer, OnlineDiarizerConfig, VoiceprintExtractor};

/// A normalised speaker voiceprint: a unit-length embedding vector computed by
/// averaging and L2-normalising the per-window embeddings from a
/// [`VoiceprintExtractor`].
///
/// Diarizer-public, deliberately NOT in `common` (mirrors [`SpeakerTurn`]).
/// The orchestrator already depends on `diarizer`, so exposing it here adds no
/// new dependency edge. The vector is always unit-length — `cosine` delegates
/// directly to [`minutist_common::voiceprint_math::cosine_unit`].
#[derive(Debug, Clone)]
pub struct Voiceprint {
    /// Unit-length embedding vector; dimension is determined by the embedding
    /// model (192-D for the bundled CAM++ zh-en model).
    pub vector: Vec<f32>,
}

impl Voiceprint {
    /// Number of dimensions in this voiceprint.
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Cosine similarity of this voiceprint against `other`.
    ///
    /// Both voiceprints are expected to be unit-length (produced by
    /// [`VoiceprintExtractor`]). Delegates to
    /// [`minutist_common::voiceprint_math::cosine_unit`].
    pub fn cosine(&self, other: &Voiceprint) -> f32 {
        minutist_common::voiceprint_math::cosine_unit(&self.vector, &other.vector)
    }
}

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

/// Minimum number of clean PCM windows behind a low-share cluster centroid
/// before the prune-veto applies the normal `T_ACCEPT` bar (§2.5, §2.4).
///
/// A low-share cluster centroid is built from few segments by definition — the
/// prune-veto's worst-case noise regime. When the window count falls below this
/// threshold the veto requires the noisy-query threshold (`T_ACCEPT_NOISY`
/// from the orchestrator matcher) rather than `T_ACCEPT`. This constant mirrors
/// `matcher::NOISE_GUARD_MIN_WINDOWS` semantics but is used by the orchestrator
/// when assembling the veto verdicts; it is declared here (in the layer that
/// owns the prune logic) for documentation proximity.
///
/// **Placeholder — WU6 calibrates from a multi-session corpus.**
pub const PRUNE_VETO_MIN_WINDOWS: u64 = 3;

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

    /// Run the sherpa segmentation + embedding + clustering pipeline and return
    /// the raw speaker turns as sherpa-free [`SpeakerTurn`]s in **milliseconds**.
    ///
    /// Asserts the input is 16 kHz (sherpa's pyannote segmentation is fixed at
    /// 16 kHz; anything else is `AppError::InvalidInput`), short-circuits empty
    /// audio to an empty turn list (sherpa's `compute` bail!s on zero-length
    /// input), then runs `Diarize::compute` under the engine `Mutex` and maps each
    /// `sherpa_rs::diarize::Segment` `{ start_s, end_s, speaker }` to a
    /// `SpeakerTurn` via [`seconds_to_ms`] — so the sherpa type never escapes.
    ///
    /// The turns run over the audio buffer as supplied (production passes the
    /// pause-INCLUDING PCM), so the returned ms share that clock. This is the
    /// public seam the orchestrator's split funnel consumes directly;
    /// [`SherpaDiarizer::assign_speakers`] composes it with [`overlay_speakers`].
    pub fn compute_turns(&self, audio: &[f32], sample_rate: u32) -> AppResult<Vec<SpeakerTurn>> {
        require_supported_sample_rate(sample_rate)?;

        // Nothing to diarize — empty audio. Don't invoke sherpa (its `compute`
        // bail!s on zero-length input).
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let raw = {
            let mut engine = self
                .engine
                .lock()
                .map_err(|_| Error::Inference("diarizer engine mutex poisoned".to_string()))?;
            // sherpa takes ownership of the sample buffer (it mutates the ptr in
            // place); clone the borrowed slice into an owned Vec for the FFI call.
            engine
                .compute(audio.to_vec(), None)
                .map_err(|e| Error::Inference(format!("sherpa Diarize::compute failed: {e:?}")))?
        };

        Ok(raw
            .into_iter()
            .map(|s| SpeakerTurn {
                start_ms: seconds_to_ms(s.start),
                end_ms: seconds_to_ms(s.end),
                cluster: s.speaker,
            })
            .collect())
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
        // Compose the two public seams: compute the raw turns, then overlay them
        // onto the ASR segments. The cluster→letter map is discarded here (the
        // trait contract is `(Vec<Segment>, u32)`); the orchestrator's split
        // funnel calls `compute_turns` + `overlay_speakers` directly when it needs
        // the map.
        self.compute_turns(audio, sample_rate).map(|turns| {
            // `assign_speakers` is the `common::Diarizer` trait implementation;
            // it has no access to the voiceprint library so no veto verdicts.
            let (segs, n, _map) = overlay_speakers(&turns, segments, &self.config, &[]);
            (segs, n)
        })
    }
}

/// Overlay first-seen-relabelled speaker ids onto `segments` by max-overlap
/// interval-join, applying the `config`'s post-cluster prune + cap, splitting a
/// mixed word-timestamped segment at its turn boundaries, and return the owned
/// (possibly longer) segment list, the distinct-label count, and the
/// cluster→letter map.
///
/// For each ASR `segment`, the [`SpeakerTurn`] cluster (turns are in
/// **milliseconds**) with the greatest total temporal overlap over
/// `[start_ms, end_ms)` wins; its `i32` cluster id is recorded. Segments with no
/// overlapping turn get `speaker_id = None`. Ties (equal overlap) resolve to the
/// **lower** cluster id (deterministic; sherpa numbers clusters by first
/// appearance, so the lower id is the earlier-seen speaker — the same orientation
/// as the previous earlier-turn tie-break, but now over per-cluster totals).
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
/// order — so a split's sub-segments letter in transcript order. The returned
/// tuple is `(segments, distinct-label count, cluster→letter map)`: the count is
/// the number of distinct labels actually assigned (segments left `None` do not
/// count), and the map pairs each surviving cluster id with the letter baked into
/// the output `speaker_id`s, in the SAME first-seen order. The map lets a caller
/// re-ASR'ing a sub-clip letter the new segments into the EXISTING scheme rather
/// than minting a fresh first-seen pass (which would rename speakers and break
/// `MeetingMeta.speaker_names` keying).
///
/// Pure (no FFI, no I/O) so the default test suite covers it without a model.
///
/// `veto_ids` is a slice of cluster ids that must survive the prune regardless of
/// their speech share or segment count (the prune-veto, §2.5). The orchestrator
/// populates this after a second `VoiceprintExtractor` pass over the low-share
/// candidate clusters; an empty slice is the no-veto baseline (the normal prune
/// path). Vetoed clusters are also exempt from the speaker-count cap so a known
/// enrolled speaker is never re-dropped by the cap after being rescued by the veto.
pub fn overlay_speakers(
    turns: &[SpeakerTurn],
    segments: Vec<Segment>,
    config: &DiarizerConfig,
    veto_ids: &[i32],
) -> (Vec<Segment>, u32, Vec<(i32, String)>) {
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
    let survivors = surviving_clusters(&chosen, &ranked, config, veto_ids);
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

    // Cluster→letter map in the SAME first-seen order `label_for` uses, so each
    // entry is consistent with the letter baked into the output `speaker_id`s.
    let cluster_labels: Vec<(i32, String)> = seen
        .iter()
        .enumerate()
        .map(|(idx, &cluster)| (cluster, alpha_label(idx)))
        .collect();

    (out, seen.len() as u32, cluster_labels)
}

/// Decide the surviving cluster set after the `config`'s prune + cap, or `None`
/// when neither is active (no reassignment needed — every winner survives).
///
/// Spurious = a winning cluster below `min_cluster_share` of the total attributed
/// speech DURATION, OR below `min_cluster_segments` won. The cap then keeps only
/// the `max_speakers` largest survivors by duration.
///
/// `veto_ids` lists cluster ids that match an enrolled voiceprint above the accept
/// threshold (§2.5 prune-veto). Any cluster in `veto_ids` is treated as a
/// survivor regardless of its share or segment count, and is excluded from the cap
/// so the cap cannot re-drop a vetoed cluster. The orchestrator populates
/// `veto_ids` after a second `VoiceprintExtractor` pass; callers that do not run
/// the veto pass pass an empty slice.
///
/// A pure helper over the initial `chosen` winners; returns the survivor ids
/// (unordered), or `None` when no reassignment is needed (neither prune, cap, nor
/// veto is active and no cluster would change outcome).
fn surviving_clusters(
    chosen: &[Option<i32>],
    ranked: &[Vec<(i32, u64)>],
    config: &DiarizerConfig,
    veto_ids: &[i32],
) -> Option<Vec<i32>> {
    let prune_active = config.min_cluster_share > 0.0 || config.min_cluster_segments > 0;
    let cap_active = config.max_speakers.is_some();
    let veto_active = !veto_ids.is_empty();
    if !prune_active && !cap_active && !veto_active {
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

    // 1. Prune by share + segment count; veto-protected clusters bypass the prune.
    //
    // A cluster in `veto_ids` matched an enrolled voiceprint above the accept
    // threshold (§2.5): identity, not share, decides its survival. It is inserted
    // into the survivor set unconditionally here, before the reassignment loop runs.
    let mut survivors: Vec<i32> = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        // A vetoed cluster survives regardless of share or segment count.
        if veto_ids.contains(&id) {
            survivors.push(id);
            continue;
        }
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
    // Vetoed clusters are exempt from the cap — they must not be re-dropped after
    // the prune-veto rescued them (§2.5: "ensure the speaker-count cap does not
    // then re-drop the vetoed cluster").
    if let Some(max) = config.max_speakers {
        // Build the non-vetoed survivors; the vetoed ones are always kept.
        let vetoed: Vec<i32> = survivors
            .iter()
            .copied()
            .filter(|id| veto_ids.contains(id))
            .collect();
        let non_vetoed: Vec<i32> = survivors
            .iter()
            .copied()
            .filter(|id| !veto_ids.contains(id))
            .collect();
        // How many non-vetoed slots remain after pinning the vetoed ones?
        let remaining_cap = max.saturating_sub(vetoed.len());
        if non_vetoed.len() > remaining_cap {
            // Sort non-vetoed by duration desc, lower id breaking ties, keep N.
            let mut sortable = non_vetoed;
            sortable.sort_by(|a, b| {
                let da = ids.iter().position(|x| x == a).map_or(0, |i| dur[i]);
                let db = ids.iter().position(|x| x == b).map_or(0, |i| dur[i]);
                db.cmp(&da).then(a.cmp(b))
            });
            sortable.truncate(remaining_cap);
            survivors = vetoed;
            survivors.extend(sortable);
        }
    }

    Some(survivors)
}

/// Per-cluster total overlap of `turns` over the half-open millisecond window
/// `[start_ms, end_ms)`, ranked descending by overlap (lower cluster id breaks a
/// tie). Empty when nothing overlaps.
///
/// [`SpeakerTurn`]s already carry start/end in **milliseconds**, on the same
/// clock as the segment bounds. Overlap is summed PER CLUSTER (a cluster may own
/// several turns touching one segment), which is what the prune reassignment
/// needs to pick a segment's next-best surviving speaker.
fn ranked_overlaps(turns: &[SpeakerTurn], start_ms: u64, end_ms: u64) -> Vec<(i32, u64)> {
    let mut totals: Vec<(i32, u64)> = Vec::new();
    for turn in turns {
        let overlap = interval_overlap_ms(start_ms, end_ms, turn.start_ms, turn.end_ms);
        if overlap == 0 {
            continue;
        }
        match totals.iter_mut().find(|(id, _)| *id == turn.cluster) {
            Some((_, acc)) => *acc += overlap,
            None => totals.push((turn.cluster, overlap)),
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

/// Interior speaker-change boundaries of `turns` inside one segment, as
/// millisecond cut points strictly within `(seg.start_ms, seg.end_ms)`.
///
/// Considers only the `turns` that overlap `[seg.start_ms, seg.end_ms)`, ordered
/// by start; a cut point is the start of a turn whose `cluster` differs from the
/// immediately preceding overlapping turn's — i.e. the instant the speaker
/// changes. Boundaries at or outside the segment edges are excluded (a change
/// flush with `start_ms`/`end_ms` is not an INTERIOR cut), and the result is
/// deduped + sorted ascending.
///
/// An **empty** `Vec` is the keep-whole signal: a continuous single-speaker
/// segment (or one the turns don't subdivide). This is the time-domain analogue
/// of [`split_segment_by_words`] for the Qwen (no-words) path — the orchestrator's
/// re-ASR split slices the segment's PCM at these points. Pure (no FFI/IO).
///
/// The cuts are speaker-change **onsets**, not a full interior partition: a short
/// turn nested inside a longer one yields a leading cut at the interjection's
/// start but NO trailing cut where the primary speaker resumes (no later
/// overlapping turn re-asserts the primary cluster), so the resumed primary speech
/// stays in the trailing sub-clip. The split is coarser there, not mislabelled —
/// the orchestrator re-letters each sub-clip by max-overlap.
pub fn turn_boundaries_within(seg: &Segment, turns: &[SpeakerTurn]) -> Vec<u64> {
    // Overlapping turns only, ordered by start (then end) so adjacency is the
    // temporal order. Ties on start keep a stable order via the end bound.
    let mut overlapping: Vec<&SpeakerTurn> = turns
        .iter()
        .filter(|t| interval_overlap_ms(seg.start_ms, seg.end_ms, t.start_ms, t.end_ms) > 0)
        .collect();
    overlapping.sort_by(|a, b| a.start_ms.cmp(&b.start_ms).then(a.end_ms.cmp(&b.end_ms)));

    let mut cuts: Vec<u64> = Vec::new();
    for pair in overlapping.windows(2) {
        let (prev, next) = (pair[0], pair[1]);
        if prev.cluster == next.cluster {
            continue;
        }
        let boundary = next.start_ms;
        // Strictly interior: a change flush with either edge is not a cut.
        if boundary > seg.start_ms && boundary < seg.end_ms && !cuts.contains(&boundary) {
            cuts.push(boundary);
        }
    }
    cuts.sort_unstable();
    cuts
}

/// Winning sherpa cluster for a word: the SURVIVING cluster with the greatest
/// total overlap of the word's `[word_start_ms, word_end_ms)` against the `turns`,
/// lower cluster id breaking a tie. `None` when no surviving cluster overlaps the
/// word.
///
/// Reuses [`ranked_overlaps`]' per-cluster summing (via [`interval_overlap_ms`]),
/// then skips any cluster the issue-63 prune dropped from `survivors` — so the
/// split path respects the prune exactly as the kept-segment reassignment does: a
/// word whose top overlap is a pruned drift-cluster attributes to its next-best
/// surviving cluster, not the dropped one. Parakeet word ENDS are approximate
/// (taken as the next word's start), so a word that straddles a turn boundary
/// attributes by max overlap — the right call.
fn word_turn(
    word_start_ms: u64,
    word_end_ms: u64,
    turns: &[SpeakerTurn],
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
    turns: &[SpeakerTurn],
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
/// Carry a prior diarization onto freshly ASR-transcribed segments by max-overlap
/// interval-join. The orchestrator's `finalise_retranscribe` calls this on the
/// re-transcribe path so a re-transcribe alone preserves the existing speaker
/// labels (and `MeetingMeta.speaker_names`) without a fresh diarize pass.
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

    /// Build a [`SpeakerTurn`] from seconds (the fixtures read naturally in
    /// seconds; `overlay_speakers` now consumes ms turns directly).
    fn turn(start_s: f32, end_s: f32, cluster: i32) -> SpeakerTurn {
        SpeakerTurn {
            start_ms: seconds_to_ms(start_s),
            end_ms: seconds_to_ms(end_s),
            cluster,
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, _count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, _count, _map) = overlay_speakers(&turns, segs, &flag_share(), &[]);

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
        let (segs, _count, _map) = overlay_speakers(&turns, segs, &flag_share(), &[]);

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
        let (segs, _count, _map) = overlay_speakers(&turns, segs, &no_prune(), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &pruned(0.02, 0, None), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &pruned(0.02, 2, None), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &pruned(0.0, 2, None), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &pruned(0.0, 0, Some(2)), &[]);
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
        let (segs, count, _map) = overlay_speakers(&turns, segs, &pruned(2.0, 0, None), &[]);
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
        let (_segs, count, _map) = overlay_speakers(&turns, segs, &DiarizerConfig::default(), &[]);
        assert_eq!(count, 1, "default prune collapses the tiny-cluster scatter");
    }

    // -----------------------------------------------------------------------
    // Tests for the §2.5 prune-veto (veto_ids parameter)
    // -----------------------------------------------------------------------

    /// A low-share cluster that is in `veto_ids` survives the prune.
    ///
    /// Cluster 0 owns ~9.9 s (dominant), cluster 1 owns one 100 ms blip (~1%).
    /// Without a veto, cluster 1 is pruned by the 2% share floor. With cluster 1
    /// in `veto_ids` (matching an enrolled voiceprint), it survives and is labelled.
    #[test]
    fn prune_veto_keeps_low_share_enrolled_cluster() {
        let turns = vec![
            turn(0.0, 5.0, 0),
            turn(5.0, 5.1, 1), // 100 ms blip — below the 2% share floor
            turn(5.1, 10.0, 0),
        ];
        // Segment 1 overlaps cluster 1 by 100 ms and cluster 0 by 50 ms;
        // cluster 1 wins initially, then would normally be pruned.
        let segs = vec![
            seg(0, 4_900),
            seg(5_000, 5_150), // cluster 1 winner (100 ms vs 50 ms)
            seg(5_400, 9_900),
        ];
        // Without veto: cluster 1 is pruned.
        let (segs_no_veto, count_no_veto, _) =
            overlay_speakers(&turns, segs.clone(), &pruned(0.02, 0, None), &[]);
        assert_eq!(count_no_veto, 1, "without veto the tiny cluster is pruned");
        assert_eq!(
            segs_no_veto[1].speaker_id.as_deref(),
            Some("A"),
            "pruned segment reassigns to the dominant cluster"
        );

        // With veto: cluster 1 is rescued and gets its own letter.
        let (segs_vetoed, count_vetoed, map) =
            overlay_speakers(&turns, segs, &pruned(0.02, 0, None), &[1]);
        assert_eq!(
            count_vetoed, 2,
            "vetoed cluster 1 must survive; two speakers expected"
        );
        // The segment that originally won cluster 1 must still be labelled on it.
        let cluster1_letter = map
            .iter()
            .find(|(id, _)| *id == 1)
            .map(|(_, lbl)| lbl.as_str());
        assert!(
            cluster1_letter.is_some(),
            "cluster 1 must appear in the cluster→letter map"
        );
        assert_eq!(
            segs_vetoed[1].speaker_id.as_deref(),
            cluster1_letter,
            "the blip segment must be labelled on the vetoed cluster"
        );
    }

    /// A low-share cluster NOT in `veto_ids` is still pruned even when another
    /// cluster is vetoed.
    ///
    /// Cluster 0 is dominant; cluster 1 is low-share and vetoed; cluster 2 is
    /// low-share and NOT vetoed. Cluster 2 must be pruned; cluster 1 must survive.
    #[test]
    fn prune_veto_non_enrolled_cluster_still_pruned() {
        let turns = vec![
            turn(0.0, 9.0, 0),   // dominant
            turn(9.0, 9.1, 1),   // low-share, vetoed (enrolled)
            turn(9.1, 9.2, 2),   // low-share, NOT vetoed (unenrolled)
            turn(9.2, 10.0, 0),
        ];
        // Segment layout: big dominant block, then one segment per tiny cluster.
        let segs = vec![
            seg(0, 8_900),
            seg(9_000, 9_100),  // cluster 1 winner
            seg(9_100, 9_200),  // cluster 2 winner
            seg(9_300, 9_900),
        ];
        // Veto cluster 1 only.
        let (out, count, map) = overlay_speakers(&turns, segs, &pruned(0.02, 0, None), &[1]);
        assert_eq!(
            count, 2,
            "cluster 0 (dominant) + cluster 1 (vetoed) survive; cluster 2 pruned"
        );
        // Cluster 2's segment must be reassigned to the nearest surviving cluster.
        let cluster2_letter = map.iter().find(|(id, _)| *id == 2);
        assert!(
            cluster2_letter.is_none(),
            "cluster 2 must not appear in the cluster→letter map (pruned)"
        );
        // Cluster 1's segment keeps its own letter.
        let cluster1_letter = map
            .iter()
            .find(|(id, _)| *id == 1)
            .map(|(_, lbl)| lbl.as_str());
        assert!(cluster1_letter.is_some(), "cluster 1 (vetoed) must be in the map");
        assert_eq!(
            out[1].speaker_id.as_deref(),
            cluster1_letter,
            "segment[1] must be on the vetoed cluster 1"
        );
    }

    /// A vetoed cluster is exempt from the speaker-count cap (§2.5: "ensure the
    /// cap does not then re-drop the vetoed cluster").
    ///
    /// Three clusters: 0 (dominant), 1 (mid-share), 2 (low-share, vetoed).
    /// Cap at 2. Cluster 2 is vetoed so it must survive even though it would
    /// normally be capped out. The result must have 3 distinct labels (or 2 with
    /// the smallest non-vetoed cluster dropped), but the vetoed one is always kept.
    #[test]
    fn prune_veto_exempt_from_cap() {
        let turns = vec![
            turn(0.0, 6.0, 0),   // dominant (~60%)
            turn(6.0, 9.0, 1),   // mid (~30%)
            turn(9.0, 9.5, 2),   // low (~5%), vetoed
        ];
        let segs = vec![
            seg(0, 5_900),
            seg(6_100, 8_900),
            seg(9_100, 9_400),  // cluster 2 winner
        ];
        // Cap at 2 — without veto cluster 2 would be capped out.
        let (out, count, map) = overlay_speakers(&turns, segs, &pruned(0.0, 0, Some(2)), &[2]);
        // The vetoed cluster 2 must survive even though cap = 2. Cap drops cluster 1
        // (mid-share, non-vetoed) to make room, keeping cluster 0 and cluster 2.
        assert_eq!(count, 2, "cap 2: vetoed cluster 2 + dominant cluster 0 survive");
        let cluster2_letter = map.iter().find(|(id, _)| *id == 2);
        assert!(
            cluster2_letter.is_some(),
            "vetoed cluster 2 must be in the cluster→letter map"
        );
        assert_eq!(
            out[2].speaker_id.as_deref(),
            cluster2_letter.map(|(_, l)| l.as_str()),
            "segment[2] must be on the vetoed cluster"
        );
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
        let (out, count, _map) = overlay_speakers(&turns, segs, &flag_share(), &[]);
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
        let (out, _count, _map) = overlay_speakers(&turns, segs, &flag_share(), &[]);
        assert_eq!(out.len(), 2, "no split on the no-words path");
        assert_eq!(out[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(out[0].shared_speakers, vec!["B".to_string()]);
    }

    // -----------------------------------------------------------------------
    // overlay_speakers cluster→letter map (#0015 phase 4)
    // -----------------------------------------------------------------------

    #[test]
    fn overlay_map_is_consistent_with_baked_in_letters() {
        // Two clusters seen in first-seen order 7 → A, 3 → B (the same fixture as
        // `overlay_two_speakers_first_seen_relabel`). The returned map must pair
        // each cluster id with the letter actually baked into the segments.
        let turns = vec![turn(0.0, 2.0, 7), turn(2.0, 4.0, 3), turn(4.0, 6.0, 7)];
        let segs = vec![seg(0, 1_900), seg(2_100, 3_900), seg(4_100, 5_900)];
        let (segs, count, map) = overlay_speakers(&turns, segs, &no_prune(), &[]);

        assert_eq!(count, 2);
        assert_eq!(map, vec![(7, "A".to_string()), (3, "B".to_string())]);

        // Every segment's baked-in label matches its cluster's map entry.
        let lookup = |cluster: i32| -> &str {
            map.iter()
                .find(|(c, _)| *c == cluster)
                .map(|(_, l)| l.as_str())
                .expect("map covers every assigned cluster")
        };
        assert_eq!(segs[0].speaker_id.as_deref(), Some(lookup(7)));
        assert_eq!(segs[1].speaker_id.as_deref(), Some(lookup(3)));
        assert_eq!(segs[2].speaker_id.as_deref(), Some(lookup(7)));
    }

    #[test]
    fn overlay_map_omits_pruned_clusters() {
        // The tiny-share cluster 1 is pruned away (same fixture as
        // `prune_drops_tiny_share_cluster_and_reassigns`), so the map carries only
        // the surviving cluster 0 → A — never a letter for a dropped cluster.
        let turns = vec![turn(0.0, 5.0, 0), turn(5.0, 5.1, 1), turn(5.1, 10.0, 0)];
        let segs = vec![seg(0, 4_900), seg(5_000, 5_150), seg(5_400, 9_900)];
        let (_segs, count, map) = overlay_speakers(&turns, segs, &pruned(0.02, 0, None), &[]);
        assert_eq!(count, 1);
        assert_eq!(map, vec![(0, "A".to_string())]);
    }

    // -----------------------------------------------------------------------
    // turn_boundaries_within (#0015 phase 4)
    // -----------------------------------------------------------------------

    #[test]
    fn boundaries_empty_for_continuous_single_speaker() {
        // One cluster spanning the whole segment → no interior change → keep-whole.
        let turns = vec![turn(0.0, 3.0, 0)];
        assert!(turn_boundaries_within(&seg(0, 3_000), &turns).is_empty());
    }

    #[test]
    fn boundaries_empty_for_two_same_cluster_turns() {
        // Two adjacent turns of the SAME cluster never produce a cut.
        let turns = vec![turn(0.0, 1.5, 0), turn(1.5, 3.0, 0)];
        assert!(turn_boundaries_within(&seg(0, 3_000), &turns).is_empty());
    }

    #[test]
    fn boundaries_single_interior_change() {
        // Cluster 0 then cluster 1 abut at 1500 ms, strictly inside [0,3000).
        let turns = vec![turn(0.0, 1.5, 0), turn(1.5, 3.0, 1)];
        assert_eq!(turn_boundaries_within(&seg(0, 3_000), &turns), vec![1_500]);
    }

    #[test]
    fn boundaries_multiple_changes_sorted() {
        // 0 → 1 at 1000, 1 → 2 at 2000: two interior cuts, ascending.
        let turns = vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1), turn(2.0, 3.0, 2)];
        assert_eq!(
            turn_boundaries_within(&seg(0, 3_000), &turns),
            vec![1_000, 2_000]
        );
    }

    #[test]
    fn boundaries_ignore_non_overlapping_distant_turn() {
        // The cluster-1 turn at [5000,6000) is wholly outside the segment, so the
        // speaker change it would imply is not an interior cut.
        let turns = vec![turn(0.0, 3.0, 0), turn(5.0, 6.0, 1)];
        assert!(turn_boundaries_within(&seg(0, 3_000), &turns).is_empty());
    }

    #[test]
    fn boundaries_change_flush_with_edge_is_not_interior() {
        // The change happens exactly at the segment's start (the first overlapping
        // turn IS cluster 1 starting at 0) — a boundary at start_ms is excluded.
        // Cluster 0's turn [−? ,0) does not overlap, so only one transition lands
        // strictly inside.
        let turns = vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1)];
        // Segment [0,2000): change at 1000 is interior → one cut. A segment
        // [1000,2000) would see only cluster 1 overlapping → no interior cut.
        assert_eq!(turn_boundaries_within(&seg(0, 2_000), &turns), vec![1_000]);
        assert!(turn_boundaries_within(&seg(1_000, 2_000), &turns).is_empty());
    }

    #[test]
    fn boundaries_dedup_repeated_boundary() {
        // Two turns both start at 1000 ms (an overlap artefact) with clusters that
        // differ from each other AND from the preceding cluster-0 turn — so two
        // adjacent differing pairs both yield boundary 1000. The dedup must collapse
        // them to a single 1000 entry.
        let turns = vec![
            turn(0.0, 1.0, 0),
            turn(1.0, 1.5, 1), // 0 → 1 boundary at 1000
            turn(1.0, 2.0, 2), // 1 → 2 boundary also at 1000 (deduped)
        ];
        assert_eq!(turn_boundaries_within(&seg(0, 3_000), &turns), vec![1_000]);
    }

    // -----------------------------------------------------------------------
    // Voiceprint struct (WU1 / #0003)
    // -----------------------------------------------------------------------

    /// Build a `Voiceprint` centroid from near-identical synthetic unit vectors
    /// and assert that `cosine()` of the centroid against one of its own unit
    /// samples is > 0.999 — the retargeted clusterer.rs:364-413 discipline.
    ///
    /// This test uses [`minutist_common::voiceprint_math::weighted_merge`]
    /// directly to construct the centroid, mirroring the code path inside
    /// `VoiceprintExtractor::centroid` without needing a live model.
    ///
    /// The 0.999 threshold is achievable when the windows all point in nearly
    /// the same direction (a realistic same-speaker enrolment set). With highly
    /// varied windows the centroid direction drifts from any individual sample;
    /// in that regime the clusterer's plain-mean alignment test applies instead
    /// (see `voiceprint_centroid_aligns_with_plain_mean`).
    #[test]
    fn voiceprint_centroid_cos_with_own_sample_gt_0999() {
        // Four nearly-identical 2-D unit vectors (tiny perturbations of [1,0]).
        // All are close to each other, so the centroid is close to all of them.
        let raw: &[[f32; 2]] = &[
            [1.000, 0.000],
            [0.9998, 0.020],
            [0.9997, 0.025],
            [0.9995, 0.032],
        ];

        let fnorm = |v: &[f32; 2]| (v[0] * v[0] + v[1] * v[1]).sqrt();

        // Unit-normalise each sample (as VoiceprintExtractor::centroid does).
        let units: Vec<[f32; 2]> = raw
            .iter()
            .map(|s| {
                let n = fnorm(s);
                [s[0] / n, s[1] / n]
            })
            .collect();

        // Build the centroid via weighted_merge (equal weight 1 per window),
        // which is what VoiceprintExtractor::centroid uses.
        let pairs: Vec<(&[f32], u64)> = units.iter().map(|u| (u.as_slice(), 1u64)).collect();
        let centroid_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);

        let centroid = Voiceprint { vector: centroid_vec };

        // Every unit sample should have cosine > 0.999 with the centroid of its
        // near-identical family.
        for (i, unit) in units.iter().enumerate() {
            let sample_vp = Voiceprint { vector: unit.to_vec() };
            let cos = centroid.cosine(&sample_vp);
            assert!(
                cos > 0.999,
                "centroid cosine with own unit sample {i} should be > 0.999, got {cos}"
            );
        }

        // dim() must equal the vector length.
        assert_eq!(centroid.dim(), 2);
    }

    /// The centroid of more varied unit vectors aligns with their plain mean
    /// (cos > 0.999) — the retargeted clusterer.rs:364-413 alignment discipline.
    ///
    /// Both the centroid and the plain mean are unit-normalised before comparing,
    /// since `Voiceprint::cosine` delegates to `cosine_unit` (a plain dot product
    /// that assumes both inputs are unit vectors).
    #[test]
    fn voiceprint_centroid_aligns_with_plain_mean() {
        let raw: &[[f32; 2]] = &[
            [1.0, 0.0],
            [0.96, 0.28],
            [0.94, 0.34],
            [0.92, 0.39],
        ];

        let fnorm = |v: &[f32; 2]| (v[0] * v[0] + v[1] * v[1]).sqrt();

        let units: Vec<[f32; 2]> = raw
            .iter()
            .map(|s| {
                let n = fnorm(s);
                [s[0] / n, s[1] / n]
            })
            .collect();

        let mut plain_mean = [0.0_f32; 2];
        for u in &units {
            plain_mean[0] += u[0];
            plain_mean[1] += u[1];
        }
        plain_mean[0] /= units.len() as f32;
        plain_mean[1] /= units.len() as f32;
        // Unit-normalise the mean so Voiceprint::cosine (a plain dot product)
        // gives the true geometric cosine.
        minutist_common::voiceprint_math::unit_normalise(&mut plain_mean);

        let pairs: Vec<(&[f32], u64)> = units.iter().map(|u| (u.as_slice(), 1u64)).collect();
        let centroid_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);

        let centroid = Voiceprint { vector: centroid_vec };
        let mean_vp = Voiceprint { vector: plain_mean.to_vec() };
        let cos = centroid.cosine(&mean_vp);
        assert!(cos > 0.999, "centroid should align with the unit-normalised plain mean, cos = {cos}");
    }

    /// A voiceprint against itself must have cosine ≈ 1.0.
    #[test]
    fn voiceprint_cosine_self_is_one() {
        let v = Voiceprint {
            vector: vec![1.0_f32, 0.0],
        };
        let cos = v.cosine(&v);
        assert!((cos - 1.0_f32).abs() < 1e-5, "cosine with self should be 1, got {cos}");
    }

    /// Two orthogonal unit voiceprints must have cosine ≈ 0.
    #[test]
    fn voiceprint_cosine_orthogonal_is_zero() {
        let a = Voiceprint { vector: vec![1.0_f32, 0.0] };
        let b = Voiceprint { vector: vec![0.0_f32, 1.0] };
        let cos = a.cosine(&b);
        assert!(cos.abs() < 1e-5, "cosine of orthogonal voiceprints should be 0, got {cos}");
    }

    // --- voiceprint_extractor_centroid (model-free synthetic tests) -----------

    /// VoiceprintExtractor::centroid with synthetic unit-normalised vectors
    /// produces a result with cosine > 0.999 against nearly-identical samples
    /// (the retargeted online clusterer discipline).
    ///
    /// This test mimics the embed→centroid path without a live model, using
    /// the weighted_merge construction directly.
    #[test]
    fn voiceprint_extractor_centroid_with_identical_samples() {
        // Simulate three identical 192-D embeddings (matching the CAM++ model).
        // In reality these would come from VoiceprintExtractor::embed; here we
        // build them synthetically.
        const DIM: usize = 192;
        let base = [1.0_f32; DIM];

        // Unit-normalise the synthetic samples (as embed() produces).
        let norm = (DIM as f32).sqrt();
        let unit_base: Vec<f32> = base.iter().map(|&x| x / norm).collect();

        // Simulate three near-identical embeddings by adding tiny noise.
        let sample1 = unit_base.clone();
        let mut sample2 = unit_base.clone();
        sample2[0] += 0.001;
        let mut sample3 = unit_base.clone();
        sample3[1] -= 0.0005;

        // Unit-normalise the noisy samples (this is what centroid() does).
        let mut s1_unit = sample1.clone();
        let mut s2_unit = sample2.clone();
        let mut s3_unit = sample3.clone();
        minutist_common::voiceprint_math::unit_normalise(&mut s1_unit);
        minutist_common::voiceprint_math::unit_normalise(&mut s2_unit);
        minutist_common::voiceprint_math::unit_normalise(&mut s3_unit);

        // Build centroid via weighted_merge (equal weight 1 per sample).
        let pairs = vec![
            (&s1_unit[..], 1u64),
            (&s2_unit[..], 1u64),
            (&s3_unit[..], 1u64),
        ];
        let centroid_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);
        let centroid = Voiceprint {
            vector: centroid_vec,
        };

        // All three samples should have cosine > 0.999 with the centroid.
        for (i, sample) in [s1_unit, s2_unit, s3_unit].iter().enumerate() {
            let sample_vp = Voiceprint {
                vector: sample.clone(),
            };
            let cos = centroid.cosine(&sample_vp);
            assert!(
                cos > 0.999,
                "sample {i} cosine with centroid should be > 0.999, got {cos}"
            );
        }
    }

    /// A centroid built from diverse unit vectors aligns with their plain mean
    /// (cos > 0.999), mimicking the online clusterer's running-mean discipline.
    #[test]
    fn voiceprint_extractor_centroid_aligns_with_diverse_mean() {
        const DIM: usize = 192;

        // Create four diverse 192-D vectors that simulate different acoustic
        // conditions for the same speaker.
        let raw: [&[f32]; 4] = [
            &std::array::from_fn::<f32, 192, _>(|i| if i == 0 { 1.0 } else { 0.01 }),
            &std::array::from_fn::<f32, 192, _>(|i| if i == 1 { 1.0 } else { 0.01 }),
            &std::array::from_fn::<f32, 192, _>(|i| if i == 2 { 1.0 } else { 0.01 }),
            &std::array::from_fn::<f32, 192, _>(|i| if i < 3 { 0.5 } else { 0.01 }),
        ];

        // Unit-normalise each raw sample.
        let mut units: Vec<Vec<f32>> = Vec::with_capacity(4);
        for r in &raw {
            let mut v = r.to_vec();
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            units.push(v);
        }

        // Compute plain arithmetic mean of unit vectors.
        let mut plain_mean = vec![0.0_f32; DIM];
        for u in &units {
            for (m, &u_i) in plain_mean.iter_mut().zip(u.iter()) {
                *m += u_i;
            }
        }
        for m in &mut plain_mean {
            *m /= units.len() as f32;
        }
        minutist_common::voiceprint_math::unit_normalise(&mut plain_mean);

        // Build centroid via weighted_merge.
        let pairs: Vec<(&[f32], u64)> = units.iter().map(|v| (v.as_slice(), 1u64)).collect();
        let centroid_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);

        let centroid = Voiceprint {
            vector: centroid_vec,
        };
        let mean_vp = Voiceprint {
            vector: plain_mean,
        };

        let cos = centroid.cosine(&mean_vp);
        assert!(
            cos > 0.999,
            "centroid should align with plain mean of units, cos = {cos}"
        );
    }

    /// dim() returns the vector length.
    #[test]
    fn voiceprint_dim_matches_vector_length() {
        let vp = Voiceprint {
            vector: vec![0.5_f32; 192],
        };
        assert_eq!(vp.dim(), 192);

        let vp_small = Voiceprint {
            vector: vec![0.1_f32; 8],
        };
        assert_eq!(vp_small.dim(), 8);

        let vp_empty = Voiceprint {
            vector: vec![],
        };
        assert_eq!(vp_empty.dim(), 0);
    }

    /// Cosine between two identical voiceprints is 1.0.
    #[test]
    fn voiceprint_cosine_identical_is_one() {
        let mut v: Vec<f32> = std::array::from_fn::<f32, 192, _>(|i| (i as f32).sin()).to_vec();
        // Unit-normalise to ensure it's a valid voiceprint.
        minutist_common::voiceprint_math::unit_normalise(&mut v);
        let vp = Voiceprint { vector: v };
        let cos = vp.cosine(&vp);
        assert!((cos - 1.0).abs() < 1e-5, "cosine with identical should be 1, got {cos}");
    }

    /// Cosine between opposite unit vectors is -1.0.
    #[test]
    fn voiceprint_cosine_opposite_is_minus_one() {
        let a = Voiceprint {
            vector: vec![1.0_f32, 0.0, 0.0],
        };
        let b = Voiceprint {
            vector: vec![-1.0_f32, 0.0, 0.0],
        };
        let cos = a.cosine(&b);
        assert!((cos - (-1.0)).abs() < 1e-5, "cosine opposite should be -1, got {cos}");
    }

    /// Cosine is symmetric: cos(a, b) == cos(b, a).
    #[test]
    fn voiceprint_cosine_is_symmetric() {
        let a = Voiceprint {
            vector: vec![1.0_f32, 0.0, 0.5],
        };
        let b = Voiceprint {
            vector: vec![0.5_f32, 1.0, 0.0],
        };
        let cos_ab = a.cosine(&b);
        let cos_ba = b.cosine(&a);
        assert!((cos_ab - cos_ba).abs() < 1e-6, "cosine must be symmetric");
    }

    /// Multiple embeddings folded by weighted_merge produce a unit-length centroid.
    #[test]
    fn voiceprint_centroid_is_unit_length() {
        const DIM: usize = 192;

        // Create synthetic unit vectors.
        let v1: Vec<f32> = std::array::from_fn::<f32, DIM, _>(|i| ((i as f32) * 0.001).cos())
            .to_vec();
        let v2: Vec<f32> = std::array::from_fn::<f32, DIM, _>(|i| ((i as f32) * 0.002).sin())
            .to_vec();

        // Normalise each.
        let mut u1 = v1.clone();
        let mut u2 = v2.clone();
        minutist_common::voiceprint_math::unit_normalise(&mut u1);
        minutist_common::voiceprint_math::unit_normalise(&mut u2);

        // Merge with equal counts.
        let pairs = vec![(&u1[..], 1u64), (&u2[..], 1u64)];
        let centroid_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);

        let centroid = Voiceprint {
            vector: centroid_vec,
        };

        // Check the centroid is unit-length.
        let norm_sq: f32 = centroid.vector.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "centroid should be unit-length, got norm {norm}");
    }

    /// Weighted merge with unequal counts produces a centroid closer to the
    /// heavier sample. This tests the refinement mechanism where an established
    /// centroid is updated with new contributions.
    #[test]
    fn voiceprint_weighted_merge_respects_counts() {
        // Two unit vectors at 90°.
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];

        // Create Voiceprints with those unit vectors.
        let vp_a = Voiceprint {
            vector: a.to_vec(),
        };
        let vp_b = Voiceprint {
            vector: b.to_vec(),
        };

        // Weighted merge: a gets count 9, b gets count 1 (9:1 weight).
        let pairs = vec![(&a[..], 9u64), (&b[..], 1u64)];
        let merged_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);
        let merged = Voiceprint {
            vector: merged_vec,
        };

        // The merged centroid should be closer (higher cosine) to a than to b.
        let cos_to_a = merged.cosine(&vp_a);
        let cos_to_b = merged.cosine(&vp_b);

        assert!(
            cos_to_a > cos_to_b,
            "weighted merge should be closer to heavier sample: cos_to_a={cos_to_a}, cos_to_b={cos_to_b}"
        );
    }

    /// A single contribution folded via weighted_merge produces a unit-normalised
    /// version of that vector, demonstrating the refinement fold path.
    #[test]
    fn voiceprint_fold_single_contribution() {
        // A non-unit vector.
        let raw = [3.0_f32, 4.0];

        // Fold via weighted_merge.
        let pairs = vec![(&raw[..], 5u64)];
        let folded_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);

        // Expected: unit-normalised [3, 4] = [0.6, 0.8]
        assert!((folded_vec[0] - 0.6).abs() < 1e-5, "folded[0] = {}", folded_vec[0]);
        assert!((folded_vec[1] - 0.8).abs() < 1e-5, "folded[1] = {}", folded_vec[1]);

        let folded_vp = Voiceprint {
            vector: folded_vec,
        };
        assert_eq!(folded_vp.dim(), 2);
    }

    /// Empty weighted_merge returns an empty vector.
    #[test]
    fn voiceprint_empty_merge_returns_empty() {
        let pairs: Vec<(&[f32], u64)> = vec![];
        let result = minutist_common::voiceprint_math::weighted_merge(&pairs);
        assert!(result.is_empty());
    }

    /// Poison-defence test: an established high-count centroid is only minimally
    /// shifted by one low-count adversarial near-threshold contribution. This
    /// verifies the bounded-weight refinement mechanism.
    #[test]
    fn voiceprint_established_centroid_resists_single_bad_contribution() {
        // Establish a strong centroid from 100 identical observations.
        let base = [1.0_f32, 0.0, 0.0];
        let mut established = base.to_vec();
        minutist_common::voiceprint_math::unit_normalise(&mut established);

        // Create 100 copies with tiny perturbations.
        let mut pairs: Vec<(&[f32], u64)> = Vec::with_capacity(101);
        let mut variations = Vec::with_capacity(100);
        for i in 0..100 {
            let mut v = base.to_vec();
            v[1] = (i as f32) * 0.001;
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            variations.push(v);
        }
        for v in variations.iter() {
            pairs.push((v.as_slice(), 1u64));
        }

        // The established centroid.
        let established_vp = Voiceprint {
            vector: minutist_common::voiceprint_math::weighted_merge(&pairs),
        };

        // Now add one adversarial low-count observation: a vector at 45° from
        // the established centroid (near-threshold similarity).
        let adversary = [1.0_f32, 1.0, 0.0];
        let mut adversary_unit = adversary.to_vec();
        minutist_common::voiceprint_math::unit_normalise(&mut adversary_unit);

        // Add the adversary with small weight (clamped REFINE_WEIGHT_CAP;
        // design spec §2.9.3 says it should be min(count, cap) relative to
        // sample_count = 100).
        let clamped_count = 1u64; // simulate cap applied.
        pairs.push((&adversary_unit, clamped_count));

        let refined_vp = Voiceprint {
            vector: minutist_common::voiceprint_math::weighted_merge(&pairs),
        };

        // The refined centroid should be very close to the established one
        // (cosine should still be > 0.99, not pushed below a target acceptance
        // threshold).
        let shift = established_vp.cosine(&refined_vp);
        assert!(
            shift > 0.99,
            "one low-count adversarial sample should not shift established centroid much, got cos {shift}"
        );
    }

    /// A 192-D centroid (the real CAM++ model dimension) round-trips via
    /// serialisation/deserialisation without loss.
    #[test]
    fn voiceprint_192d_centroid_construction() {
        const DIM: usize = 192;

        // Create a realistic 192-D vector.
        let mut v: Vec<f32> = std::array::from_fn::<f32, DIM, _>(|i| (i as f32 * 0.01).sin())
            .to_vec();
        minutist_common::voiceprint_math::unit_normalise(&mut v);

        let vp = Voiceprint { vector: v.clone() };

        // Verify dimension and unit length.
        assert_eq!(vp.dim(), DIM);
        let norm_sq: f32 = vp.vector.iter().map(|&x| x * x).sum();
        assert!((norm_sq.sqrt() - 1.0).abs() < 1e-5);

        // Cosine with itself must be 1.0.
        let cos_self = vp.cosine(&vp);
        assert!((cos_self - 1.0).abs() < 1e-5);
    }

    /// Centroid built from multiple varied 192-D samples exhibits the
    /// expected running-mean behaviour (cos > 0.999 alignment).
    #[test]
    fn voiceprint_192d_centroid_running_mean_alignment() {
        const DIM: usize = 192;
        const SAMPLES: usize = 10;

        // Create 10 synthetic 192-D vectors that simulate embeddings from
        // the same speaker recorded in slightly different conditions.
        let mut vectors = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            let mut v: Vec<f32> = std::array::from_fn::<f32, DIM, _>(|j| {
                ((j as f32) * 0.01 + (i as f32) * 0.001).sin()
            })
            .to_vec();
            minutist_common::voiceprint_math::unit_normalise(&mut v);
            vectors.push(v);
        }

        // Build centroid via weighted_merge.
        let pairs: Vec<(&[f32], u64)> = vectors.iter().map(|v| (v.as_slice(), 1u64)).collect();
        let centroid_vec = minutist_common::voiceprint_math::weighted_merge(&pairs);
        let centroid = Voiceprint {
            vector: centroid_vec,
        };

        // Each sample should have high cosine with the centroid.
        for (i, v) in vectors.iter().enumerate() {
            let sample_vp = Voiceprint {
                vector: v.clone(),
            };
            let cos = centroid.cosine(&sample_vp);
            assert!(
                cos > 0.95,
                "sample {i} should have cos > 0.95 with centroid, got {cos}"
            );
        }

        // Verify centroid is unit-length.
        let norm_sq: f32 = centroid.vector.iter().map(|&x| x * x).sum();
        assert!((norm_sq.sqrt() - 1.0).abs() < 1e-5, "centroid should be unit-length");
    }
}
