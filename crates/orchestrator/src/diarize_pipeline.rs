//! Diarization split/merge/veto pipeline — shared by the user-triggered
//! re-diarize pass and the on-stop pass.
//!
//! Turns a [`SpeakerTurn`] list + ASR transcript into speaker-lettered
//! [`Segment`]s: overlays labels, collapses adjacent same-speaker fragments,
//! re-ASRs mixed-speaker segments split by turn boundaries (§0015 phase 4),
//! and applies the §2.5 prune-veto + #0023 library-informed merge when a
//! voiceprint gallery is available. [`run_diarization_blocking`] is the
//! entry point; everything else here is its decomposition.
//!
//! Stays inside `orchestrator` rather than the `diarizer` crate because it is
//! genuinely an orchestration layer: it reads `persistence` (audio + the
//! stored voiceprint gallery), emits `AppEvent`s as it re-ASRs mixed
//! segments, and drives `orchestrator`'s own pause-clock helpers
//! (`runner::pause_excluding_segments` et al.) and voiceprint matcher
//! (`crate::matcher`). `diarizer` itself stays free of `persistence` and
//! `AppEvent` (see `architecture/components.md`, "diarizer"); this module is
//! the orchestrator-side caller that supplies those dependencies to
//! `diarizer`'s pure primitives (`overlay_speakers`, `merge_adjacent_speakers`,
//! `turn_boundaries_within`).

use std::path::PathBuf;

use diarizer::SpeakerTurn;
use minutist_common::{AppError, AppEvent, AppResult, MeetingId, Segment};
use tokio::sync::broadcast;

use crate::runner;

/// #0015 phase 1 merge threshold: a same-speaker inter-segment gap up to this
/// many ms is rejoined into one segment. Kept strictly below the live
/// accumulator's `MAX_GAP_MS` (3 s) and far below `PAUSE_MIN_MS` (4 s) so a merge
/// never bridges a region the timeline treats as a pause; it comfortably covers
/// the 720 ms VAD hangover and the zero-gap 10 s force-split.
const MERGE_GAP_MS: u64 = 1500;

/// Minimum Jaccard temporal overlap required between a fresh cluster's total
/// speech span and an old named cluster's total speech span for the
/// timeline-coherence fallback in `apply_ephemeral_remap` to accept the name
/// transfer. A value of 0.50 means the intersection must cover at least half the
/// union of the two spans.
///
/// **Placeholder — WU6 calibrates** this alongside the cosine thresholds once a
/// multi-session corpus is available. It is intentionally named as a constant
/// (not a magic literal) so WU6 can swap the value without changing call sites.
pub(crate) const TIMELINE_JACCARD_THRESHOLD: f64 = 0.50;

/// Total duration of non-overlapping speech attributed to `label` in `segments` (ms).
///
/// Sums the duration of every segment whose `speaker_id == label`. Does not
/// de-duplicate overlapping time ranges (the diarizer produces non-overlapping
/// segments per label in practice, so this is exact for the use case).
pub(crate) fn total_speech_ms(segments: &[Segment], label: &str) -> u64 {
    segments
        .iter()
        .filter(|s| s.speaker_id.as_deref() == Some(label))
        .map(|s| s.end_ms.saturating_sub(s.start_ms))
        .sum()
}

/// Jaccard temporal overlap between the speech spans of `new_label` (in
/// `new_segs`) and `old_label` (in `old_segs`).
///
/// Computes the intersection and union of the two sets of `[start_ms, end_ms)`
/// intervals (as continuous coverage), then returns `intersection / union`.
/// Returns `0.0` when either set is empty or the union is zero.
///
/// Uses a merge-then-scan approach: each set of intervals is sorted and merged
/// into non-overlapping coverage; intersection and union are derived from the
/// merged representations.
pub(crate) fn timeline_jaccard(
    new_segs: &[Segment],
    new_label: &str,
    old_segs: &[Segment],
    old_label: &str,
) -> f64 {
    fn merged_intervals(segs: &[Segment], label: &str) -> Vec<(u64, u64)> {
        let mut ivs: Vec<(u64, u64)> = segs
            .iter()
            .filter(|s| s.speaker_id.as_deref() == Some(label) && s.end_ms > s.start_ms)
            .map(|s| (s.start_ms, s.end_ms))
            .collect();
        ivs.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in ivs {
            match merged.last_mut() {
                Some((_, prev_hi)) if lo <= *prev_hi => {
                    if hi > *prev_hi {
                        *prev_hi = hi;
                    }
                }
                _ => merged.push((lo, hi)),
            }
        }
        merged
    }

    let new_ivs = merged_intervals(new_segs, new_label);
    let old_ivs = merged_intervals(old_segs, old_label);
    if new_ivs.is_empty() || old_ivs.is_empty() {
        return 0.0;
    }

    // Compute total coverage of each set.
    let new_total: u64 = new_ivs.iter().map(|(a, b)| b - a).sum();
    let old_total: u64 = old_ivs.iter().map(|(a, b)| b - a).sum();

    // Intersection: sorted two-pointer scan.
    let mut intersection: u64 = 0;
    let mut ni = 0;
    let mut oi = 0;
    while ni < new_ivs.len() && oi < old_ivs.len() {
        let (nlo, nhi) = new_ivs[ni];
        let (olo, ohi) = old_ivs[oi];
        let lo = nlo.max(olo);
        let hi = nhi.min(ohi);
        if hi > lo {
            intersection += hi - lo;
        }
        if nhi < ohi {
            ni += 1;
        } else {
            oi += 1;
        }
    }

    let union = new_total + old_total - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// How a diarization+split pass obtains its turns + re-ASR backend (#0015 phase
/// 4). Both variants converge on [`diarize_split_merge`] (the model-free core);
/// the variant only decides where the turns + backend come from.
///
/// `Production` carries the bundled `SherpaDiarizer` (its `compute_turns` runs on
/// the decoded PCM, on the blocking thread) + the best-effort routed Qwen
/// backend (`None` when the model is absent → degrade to keep-whole). `Stub`
/// supplies the turns + backend + config directly, the seam the default suite
/// uses to exercise the split with no `SherpaDiarizer` and no Qwen GGUF.
pub(crate) enum DiarizationJob {
    Production {
        diarizer: diarizer::SherpaDiarizer,
        backend: Option<Box<dyn minutist_common::AsrBackend + Send>>,
    },
    #[cfg(any(test, feature = "test-source"))]
    Stub {
        turns: Vec<SpeakerTurn>,
        backend: Option<Box<dyn minutist_common::AsrBackend + Send>>,
        config: diarizer::DiarizerConfig,
    },
}

/// Decode the meeting's PCM + transcript, resolve the turns + config + backend
/// from `job`, and run the [`diarize_split_merge`] core, all on a
/// `spawn_blocking` thread.
///
/// Returns the (possibly split) segments with `speaker_id` overlaid, the distinct
/// speaker count, and any vetoed-cluster names `(letter, display_name)` that the
/// caller must write to `speaker_names` after `finalise_diarization` clears the map.
///
/// `prune_veto_extractor` is `Some` when the embedding model is locally available;
/// `gallery` is the active-model voiceprint library. When either is absent, the
/// veto pass is skipped (graceful degradation — the prune runs normally). When
/// both are present, a second `VoiceprintExtractor` pass runs over each low-share
/// candidate cluster's PCM windows (after `compute_turns`, before `diarize_split_merge`)
/// to build a centroid and match against the gallery via `matcher::assign_identities`.
/// Accept-band matches (with the query-side noise guard) produce veto verdicts that
/// are passed into `diarize_split_merge` so those clusters survive the prune.
///
/// The `job` carries either the production `SherpaDiarizer` (+ best-effort Qwen
/// backend) or stub-supplied turns + backend (the default-suite seam), so a
/// `SherpaDiarizer` and a model-free stub both drive the SAME split core.
///
/// VRAM sequencing (§2.5): the `SherpaDiarizer` holds the segmentation model; the
/// `prune_veto_extractor` holds the embedding model (same as the diarizer's internal
/// one, but a separate instance opened before the diarizer drop). The Qwen re-ASR
/// `backend` is the VRAM-heavy component; it is moved into `diarize_split_merge` and
/// dropped there. The extractor is a lightweight embedding-only model and is dropped
/// at the end of the veto pass, before `diarize_split_merge` starts — so the peak
/// VRAM budget is (sherpa-seg + sherpa-emb + extractor) for the veto pass, then
/// (sherpa-seg + sherpa-emb + Qwen) for the split — never all three simultaneously.
pub(crate) async fn run_diarization_blocking(
    meeting_dir: PathBuf,
    job: DiarizationJob,
    prune_veto_extractor: Option<diarizer::VoiceprintExtractor>,
    gallery: Vec<persistence::StoredVoiceprint>,
    event_tx: broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
) -> AppResult<(Vec<Segment>, u32, Vec<(String, String)>)> {
    tokio::task::spawn_blocking(move || -> AppResult<(Vec<Segment>, u32, Vec<(String, String)>)> {
        let pcm = persistence::read_audio_pcm(&meeting_dir)?;
        let segments = persistence::read_transcript(&meeting_dir)?;

        let (turns, config, backend): (
            Vec<SpeakerTurn>,
            diarizer::DiarizerConfig,
            Option<Box<dyn minutist_common::AsrBackend + Send>>,
        ) = match job {
            DiarizationJob::Production { diarizer, backend } => {
                // `compute_turns` runs over the pause-INCLUDING PCM → turn ms are
                // on the INCLUDING clock the split funnel maps onto.
                let turns = diarizer.compute_turns(&pcm, 16_000)?;
                (turns, diarizer.config().clone(), backend)
            }
            #[cfg(any(test, feature = "test-source"))]
            DiarizationJob::Stub {
                turns,
                backend,
                config,
            } => (turns, config, backend),
        };

        // §2.5 prune-veto + #0023 library-informed merge: when the embedding model
        // is available, run a single extractor pass over all clusters to build
        // centroids and match against the gallery. The pass produces:
        // - veto_verdicts: low-share clusters that match an enrolled identity (rescued
        //   from the prune), passed to diarize_split_merge as veto_ids.
        // - merge_map: (source→canonical) pairs for clusters that both match the same
        //   enrolled identity; passed to overlay_speakers so the prune/cap sees the
        //   combined speech mass.
        let (veto_verdicts, merge_map): (Vec<(i32, String)>, Vec<(i32, i32)>) =
            if let Some(extractor) = prune_veto_extractor {
                compute_prune_veto_verdicts(
                    &turns, &pcm, &config, &extractor, &gallery, meeting_id,
                )
            } else {
                (Vec::new(), Vec::new())
            };
        // Drop the extractor before the split: its VRAM is freed before the Qwen
        // backend enters (§2.5 VRAM sequencing note above).
        // (prune_veto_extractor is moved into `extractor` above or already None)

        // `Box<dyn AsrBackend + Send>` → `Box<dyn AsrBackend>` for the core (the
        // split runs on this one thread; the `Send` bound is only needed to move
        // the backend into the closure).
        let backend = backend.map(|b| b as Box<dyn minutist_common::AsrBackend>);
        let ctx = SplitMergeContext {
            turns: &turns,
            pcm: &pcm,
            event_tx: &event_tx,
            meeting_id,
        };
        diarize_split_merge(&ctx, segments, backend, &config, &veto_verdicts, &merge_map)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("diarization spawn_blocking join failed: {e}"),
    })?
}

/// Distinct-label count over the segments (`speaker_id` first-seen order).
///
/// The merge + split preserve labels, but recomputing from the final list is
/// robust and self-documenting — never trust an upstream count after the list
/// has been transformed.
fn distinct_label_count(segments: &[Segment]) -> u32 {
    let mut seen: Vec<&str> = Vec::new();
    for seg in segments {
        if let Some(label) = seg.speaker_id.as_deref() {
            if !seen.contains(&label) {
                seen.push(label);
            }
        }
    }
    seen.len() as u32
}

/// Dominant cluster of `turns` over `[start_ms, end_ms)`: the cluster id with the
/// greatest total temporal overlap, lower id breaking a tie (matching
/// `diarizer::overlay_speakers`' tie orientation). `None` when no turn overlaps.
///
/// Used to letter a re-ASR'd sub-clip via the WU1 cluster→letter map. The
/// orchestrator computes this from the public [`SpeakerTurn`] fields rather than
/// reaching into the diarizer's private overlap helper.
fn dominant_cluster(turns: &[SpeakerTurn], start_ms: u64, end_ms: u64) -> Option<i32> {
    if end_ms <= start_ms {
        return None;
    }
    // Per-cluster overlap totals, then argmax (greatest overlap; lower id on a tie).
    let mut totals: Vec<(i32, u64)> = Vec::new();
    for t in turns {
        let lo = start_ms.max(t.start_ms);
        let hi = end_ms.min(t.end_ms);
        let overlap = hi.saturating_sub(lo);
        if overlap == 0 {
            continue;
        }
        match totals.iter_mut().find(|(id, _)| *id == t.cluster) {
            Some((_, sum)) => *sum += overlap,
            None => totals.push((t.cluster, overlap)),
        }
    }
    totals
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(id, _)| id)
}

/// Read-only per-meeting context shared by [`diarize_split_merge`] and
/// [`split_mixed_qwen_segment`]: the pause-INCLUDING [`SpeakerTurn`]s + decoded
/// PCM, the event bus, and the meeting id. Bundled into one struct rather than
/// four parallel parameters so a future meeting-wide input (like the
/// per-segment cache work in B2) does not grow either function's argument
/// list further.
struct SplitMergeContext<'a> {
    /// Raw [`SpeakerTurn`]s from `compute_turns`, on the pause-INCLUDING clock
    /// `pcm` shares.
    turns: &'a [SpeakerTurn],
    /// The pause-INCLUDING decoded audio.
    pcm: &'a [f32],
    event_tx: &'a broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
}

/// Re-ASR split core (#0015 phase 4) — the BLOCKING, model-free-testable funnel.
///
/// A free fn taking EXPLICIT params (mirroring [`crate::transcribe_pcm_window_blocking`]
/// rather than dispatching through the `common::Diarizer` trait) so the default
/// suite can drive the whole split with a stub-supplied `turns` + stub `AsrBackend` —
/// no `SherpaDiarizer`, no Qwen GGUF:
/// - `ctx.turns` are the raw [`SpeakerTurn`]s from `compute_turns`, on the
///   pause-INCLUDING clock `ctx.pcm` shares.
/// - `segments` is the ASR transcript (pause-EXCLUDING `start_ms`).
/// - `ctx.pcm` is the pause-INCLUDING decoded audio.
/// - `backend` is the routed Qwen re-ASR backend, or `None` (model absent /
///   degrade to keep-whole — no regression vs. the pre-split behaviour).
/// - `config` is the `DiarizerConfig` the overlay + flag use.
///
/// Steps:
/// 1. [`diarizer::overlay_speakers`] labels segments + flags mixed Qwen segments
///    (keep-whole, `shared_speakers` set, empty `words`) + returns the
///    cluster→letter map.
/// 2. [`diarizer::merge_adjacent_speakers`] collapses VAD/force-split fragments.
/// 3. For each KEPT mixed Qwen segment (non-empty `shared_speakers` AND empty
///    `words`) with a `backend`: take [`diarizer::turn_boundaries_within`] cuts
///    on the SAME pause-INCLUDING clock (mapped via
///    [`runner::excluding_range_to_pcm_slice`] against one meeting-wide
///    [`runner::pause_excluding_segments`] scan), energy-snap each cut, slice the
///    PCM, re-ASR each single-speaker sub-clip, letter it from the map by its
///    dominant [`SpeakerTurn`] cluster, and stamp its `start_ms` on the EXCLUDING
///    clock via [`runner::excluding_ms_for_pcm_sample_in_regions`]. Keep-whole if the cuts
///    are empty, any snap returns `None`, or `backend` is `None`.
/// 4. Re-run [`diarizer::merge_adjacent_speakers`] (the split may have produced
///    adjacent same-letter sub-clips across segments) and recompute the count.
///
/// The clock discipline is the #1 blocking fix: turn cuts are taken on the
/// pause-INCLUDING clock the turns + PCM share, and a sub-clip's `start_ms` is
/// mapped back to the EXCLUDING transcript clock by the inverse. INCLUDING-clock
/// turns are NEVER compared against EXCLUDING-clock segment bounds.
/// `veto_verdicts` is a list of `(cluster_id, display_name)` pairs for low-share
/// clusters the orchestrator has vetoed from pruning (§2.5 prune-veto). Each
/// cluster in the list matched an enrolled voiceprint above `T_ACCEPT` (with the
/// query-side noise guard) and must survive the share prune + cap. After
/// `overlay_speakers` assigns letters, the cluster→letter map is used to convert
/// each vetoed cluster id to its final letter and collect `(letter, name)` pairs
/// returned as the third tuple element so the caller can write them to
/// `speaker_names` after `finalise_diarization` clears the map.
///
/// An empty `veto_verdicts` is the no-veto baseline (the prune runs normally).
///
/// `merge_map` is a slice of `(source, canonical)` pairs from the library-informed
/// merge pass (#0023). An empty slice is the no-merge baseline (bit-identical to
/// the pre-merge behaviour).
fn diarize_split_merge(
    ctx: &SplitMergeContext<'_>,
    segments: Vec<Segment>,
    mut backend: Option<Box<dyn minutist_common::AsrBackend>>,
    config: &diarizer::DiarizerConfig,
    veto_verdicts: &[(i32, String)],
    merge_map: &[(i32, i32)],
) -> AppResult<(Vec<Segment>, u32, Vec<(String, String)>)> {
    // Extract cluster ids for the veto; names are resolved below once the
    // cluster→letter map is available.
    let veto_ids: Vec<i32> = veto_verdicts.iter().map(|(id, _)| *id).collect();

    // 1. Overlay labels + flag mixed Qwen segments; keep the cluster→letter map.
    // Pass veto_ids (enrolled cluster rescue) and merge_map (same-identity cluster
    // unification, #0023) — applied together in a single overlay pass.
    let (mut segments, _count, cluster_letters) =
        diarizer::overlay_speakers(ctx.turns, segments, config, &veto_ids, merge_map);

    // 2. Collapse fragments so a turn reads as one row (#0015 phase 1).
    diarizer::merge_adjacent_speakers(&mut segments, MERGE_GAP_MS);

    // 3. Re-ASR split, only when a backend is present. The pause-excluding kept
    // regions are computed ONCE for the whole meeting here — `pause_excluding_segments`
    // is an O(samples) scan — rather than once per segment (or per interior cut)
    // inside `split_mixed_qwen_segment`, which is what made this pass
    // O(segments × cuts × samples) for a long recording (B2).
    if backend.is_some() {
        let regions = runner::pause_excluding_segments(ctx.pcm);
        let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
        for seg in segments.into_iter() {
            // A kept mixed Qwen segment: flagged by `overlay_speakers` with
            // non-empty `shared_speakers` and no per-word timestamps. Everything
            // else passes through unchanged.
            let is_kept_mixed_qwen = !seg.shared_speakers.is_empty() && seg.words.is_empty();
            if !is_kept_mixed_qwen {
                out.push(seg);
                continue;
            }

            match split_mixed_qwen_segment(
                ctx,
                &seg,
                &regions,
                backend.as_deref_mut().expect("backend present in this branch"),
                &cluster_letters,
            )? {
                Some(sub_segments) => out.extend(sub_segments),
                // Keep-whole: empty cuts, a snap with no clear minimum, or a
                // re-ASR that produced nothing — leave the overlay's dominant
                // label + `shared_speakers` flag intact.
                None => out.push(seg),
            }
        }
        segments = out;
    }

    // 4. Re-merge (the split can yield adjacent same-letter sub-clips that should
    // read as one row) and recompute the distinct-label count.
    diarizer::merge_adjacent_speakers(&mut segments, MERGE_GAP_MS);
    let count = distinct_label_count(&segments);

    // Drop the re-ASR backend promptly: the Qwen GGUF is co-resident with the
    // sherpa diarizer models, so free its VRAM as soon as the split loop is done
    // (it lives no longer than this fn).
    drop(backend);

    // 5. Resolve vetoed cluster ids to their assigned letters so the caller can
    // write the pre-matched names to speaker_names after finalise_diarization clears
    // the map. Clusters that did not survive (e.g. were absent from the turns and
    // got no segments) are silently dropped — only present letters carry names.
    let veto_names: Vec<(String, String)> = veto_verdicts
        .iter()
        .filter_map(|(cluster_id, name)| {
            cluster_letters
                .iter()
                .find(|(cid, _)| *cid == *cluster_id)
                .map(|(_, letter)| (letter.clone(), name.clone()))
        })
        .collect();

    Ok((segments, count, veto_names))
}

/// Minimum "clean" segment duration for voiceprint extraction (§2.3.1): long
/// enough that a single short interjection does not distort a speaker's
/// average embedding. Shared by every "segments → PCM windows → centroid"
/// extraction site: enrolment, cross-meeting matching, and ephemeral
/// name-carry matching.
const MIN_CLEAN_VOICEPRINT_DURATION_MS: u64 = 1000;

/// Segments belonging to `label`: excludes mixed (non-empty `shared_speakers`)
/// segments and any shorter than [`MIN_CLEAN_VOICEPRINT_DURATION_MS`] (the
/// §2.3.1 cleanliness filter every voiceprint-extraction call site applies
/// before mapping segments to PCM windows).
pub(crate) fn clean_segments_for_label<'a>(segments: &'a [Segment], label: &str) -> Vec<&'a Segment> {
    segments
        .iter()
        .filter(|seg| {
            seg.speaker_id.as_deref() == Some(label)
                && seg.shared_speakers.is_empty()
                && seg.end_ms.saturating_sub(seg.start_ms) >= MIN_CLEAN_VOICEPRINT_DURATION_MS
        })
        .collect()
}

/// Map `segs` through the pause-excl → incl clock mapper against the caller's
/// precomputed `regions`, collecting the non-empty PCM windows.
///
/// `regions` is one [`runner::pause_excluding_segments`] scan for the whole
/// meeting, computed ONCE by the caller and shared across every label's
/// mapping — the O(samples) scan must not re-run per segment (B2). `segs`
/// should already be filtered to one label/cleanliness criterion (see
/// [`clean_segments_for_label`]); a segment the mapper cannot resolve (out of
/// range) is silently skipped, matching
/// [`runner::excluding_range_to_pcm_slice`]'s documented behaviour.
pub(crate) fn segment_windows(
    pcm: &[f32],
    regions: &[runner::KeptRegion],
    segs: &[&Segment],
) -> Vec<Vec<f32>> {
    segs.iter()
        .filter_map(|seg| {
            let range = runner::excluding_range_to_pcm_slice(regions, seg.start_ms, seg.end_ms)?;
            let window = pcm[range].to_vec();
            (!window.is_empty()).then_some(window)
        })
        .collect()
}

/// Build a speaker centroid from non-empty PCM `windows`.
///
/// The common tail of every "segments/turns → PCM windows → centroid"
/// extraction site (voiceprint enrolment, cross-meeting matching, ephemeral
/// name-carry matching, and the library-merge prune-veto pass) — those sites
/// differ only in how `windows` is gathered and in what they do with an
/// extraction `Err` (propagate vs. log-and-skip), both left to the caller.
/// The caller checks `windows.is_empty()` itself: the two possible "no usable
/// audio" cases (no clean segments vs. every clean segment mapping to nothing)
/// log distinct messages at several call sites.
pub(crate) fn centroid_from_windows(
    extractor: &diarizer::VoiceprintExtractor,
    windows: &[Vec<f32>],
    sample_rate: u32,
) -> AppResult<(diarizer::Voiceprint, u64)> {
    let window_count = windows.len() as u64;
    let window_refs: Vec<&[f32]> = windows.iter().map(|w| w.as_slice()).collect();
    let centroid = extractor.centroid(&window_refs, sample_rate)?;
    Ok((centroid, window_count))
}

/// Second embedding pass: compute prune-veto verdicts AND the library-informed
/// merge map in a single extractor pass.
///
/// Embeds every cluster from `turns` (not only low-share candidates) so that
/// `match_each_cluster` can detect when two clusters match the same enrolled
/// identity. The single extractor pass is reused for both outputs:
///
/// - **Veto verdicts** `Vec<(i32, String)>`: clusters that ARE low-share (would
///   be pruned) AND match an enrolled identity above the accept threshold. These
///   are returned as `(cluster_id, display_name)` pairs and forwarded to
///   `overlay_speakers` as `veto_ids`.
/// - **Merge map** `Vec<(i32, i32)>`: `(source, canonical)` pairs from the
///   library-informed merge pass (issue #0023). For each enrolled identity matched
///   by ≥2 clusters, the group merges: canonical = the member with the greatest
///   speech mass (tie-break: lowest cluster id). Non-canonical members become
///   sources. The invariant that only same-identity clusters share a canonical is
///   enforced here: groups are keyed by `identity_id`, so two different identities
///   can never share a canonical.
///
/// `veto_ids` and `merge_map` are derived from the SAME per-cluster match results;
/// no third extractor pass is added.
///
/// When the prune is not active, no cluster is at risk of pruning — veto verdicts
/// are skipped, but the merge pass still runs (a dominant cluster and another
/// cluster can both match the same identity even without a prune floor).
///
/// Errors in individual cluster embeddings are logged and skipped (best-effort).
fn compute_prune_veto_verdicts(
    turns: &[SpeakerTurn],
    pcm: &[f32],
    config: &diarizer::DiarizerConfig,
    extractor: &diarizer::VoiceprintExtractor,
    gallery: &[persistence::StoredVoiceprint],
    meeting_id: MeetingId,
) -> (Vec<(i32, String)>, Vec<(i32, i32)>) {
    use crate::matcher::{match_each_cluster, QueryCluster};

    if gallery.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Tally per-cluster total turn duration + turn count (pause-INCLUDING ms).
    // All clusters are tallied — not only low-share ones — because the merge pass
    // needs centroids for every cluster that might share an identity.
    let mut cluster_ids: Vec<i32> = Vec::new();
    let mut cluster_dur: Vec<u64> = Vec::new();
    let mut cluster_turn_count: Vec<usize> = Vec::new();
    for t in turns {
        let dur = t.end_ms.saturating_sub(t.start_ms);
        match cluster_ids.iter().position(|&id| id == t.cluster) {
            Some(i) => {
                cluster_dur[i] += dur;
                cluster_turn_count[i] += 1;
            }
            None => {
                cluster_ids.push(t.cluster);
                cluster_dur.push(dur);
                cluster_turn_count.push(1);
            }
        }
    }

    if cluster_ids.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let total_dur: u64 = cluster_dur.iter().sum();

    // Identify which clusters are low-share (would be pruned without a veto).
    let prune_active = config.min_cluster_share > 0.0 || config.min_cluster_segments > 0;
    let is_low_share: Vec<bool> = if prune_active {
        cluster_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let share = if total_dur > 0 {
                    cluster_dur[i] as f32 / total_dur as f32
                } else {
                    0.0
                };
                let below_share =
                    config.min_cluster_share > 0.0 && share < config.min_cluster_share;
                let below_count = config.min_cluster_segments > 0
                    && cluster_turn_count[i] < config.min_cluster_segments;
                below_share || below_count
            })
            .collect()
    } else {
        vec![false; cluster_ids.len()]
    };

    tracing::debug!(
        target: "orchestrator",
        meeting_id = %meeting_id.0,
        total_clusters = cluster_ids.len(),
        low_share_count = is_low_share.iter().filter(|&&b| b).count(),
        "library-merge/prune-veto: running embedding pass for all clusters"
    );

    // Embed every cluster (not only low-share ones) in one extractor pass.
    // Turns are on the pause-INCLUDING clock; PCM is pause-INCLUDING — no clock mapper.
    const SR: u32 = 16_000;
    let mut queries: Vec<QueryCluster> = Vec::new();

    for &cluster_id in &cluster_ids {
        let windows: Vec<Vec<f32>> = turns
            .iter()
            .filter(|t| t.cluster == cluster_id)
            .filter_map(|t| {
                let start_sample = (t.start_ms as usize * SR as usize) / 1000;
                let end_sample = (t.end_ms as usize * SR as usize) / 1000;
                if start_sample >= end_sample || end_sample > pcm.len() {
                    return None;
                }
                let window = pcm[start_sample..end_sample].to_vec();
                if window.is_empty() {
                    None
                } else {
                    Some(window)
                }
            })
            .collect();

        if windows.is_empty() {
            tracing::debug!(
                target: "orchestrator",
                meeting_id = %meeting_id.0,
                cluster_id,
                "library-merge/prune-veto: no usable PCM windows for cluster; skipping"
            );
            continue;
        }

        // The noise guard threshold (T_ACCEPT_NOISY vs T_ACCEPT) depends on
        // window_count; it is applied inside match_each_cluster via the
        // per-cluster QueryCluster below.
        let (centroid, window_count) = match centroid_from_windows(extractor, &windows, SR) {
            Ok(pair) => pair,
            Err(e) => {
                tracing::debug!(
                    target: "orchestrator",
                    meeting_id = %meeting_id.0,
                    cluster_id,
                    error = %e,
                    "library-merge/prune-veto: centroid extraction failed; skipping"
                );
                continue;
            }
        };

        queries.push(QueryCluster {
            label: cluster_id.to_string(),
            centroid: centroid.vector,
            window_count,
        });
    }

    if queries.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Per-cluster best-identity match (collisions allowed) for the merge pass.
    let per_cluster = match_each_cluster(&queries, gallery);

    // -------------------------------------------------------------------
    // Derive veto verdicts: low-share clusters with an accept-band match.
    // -------------------------------------------------------------------
    // `is_low_share` is indexed by cluster_ids; look up by cluster_id.
    let veto_verdicts: Vec<(i32, String)> = per_cluster
        .iter()
        .filter_map(|(label, matched)| {
            let (identity_id, _sim) = matched.as_ref()?;
            let cluster_id: i32 = label.parse().ok()?;
            // Only veto if this cluster is low-share (would be pruned).
            let idx = cluster_ids.iter().position(|&id| id == cluster_id)?;
            if !is_low_share[idx] {
                return None;
            }
            // Resolve display name from gallery.
            let display_name = gallery
                .iter()
                .find(|e| e.identity_id == *identity_id)
                .map(|e| e.display_name.clone())?;
            Some((cluster_id, display_name))
        })
        .collect();

    if !veto_verdicts.is_empty() {
        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            vetoed_count = veto_verdicts.len(),
            "prune-veto: enrolled quiet speakers rescued from pruning"
        );
    }

    // -------------------------------------------------------------------
    // Derive merge map: group clusters by matched identity; emit
    // (source, canonical) pairs for non-canonical members of each group.
    //
    // Canonical = the group member with the greatest speech mass
    // (turn-duration sum from cluster_dur); tie-break = lowest cluster id.
    // Invariant: groups are keyed by identity_id, so only same-identity
    // clusters are ever merged.
    // -------------------------------------------------------------------
    use minutist_common::VoiceprintIdentityId;
    // Build identity → Vec<cluster_id> groups from accept-band matches.
    let mut identity_groups: Vec<(VoiceprintIdentityId, Vec<i32>)> = Vec::new();
    for (label, matched) in &per_cluster {
        let Some((identity_id, _)) = matched else { continue };
        let cluster_id: i32 = match label.parse() {
            Ok(id) => id,
            Err(_) => continue,
        };
        match identity_groups.iter_mut().find(|(id, _)| id == identity_id) {
            Some((_, members)) => members.push(cluster_id),
            None => identity_groups.push((*identity_id, vec![cluster_id])),
        }
    }

    let mut merge_map: Vec<(i32, i32)> = Vec::new();
    for (_identity_id, members) in &identity_groups {
        if members.len() < 2 {
            // Single cluster matched this identity — nothing to merge.
            continue;
        }
        // Canonical = largest speech mass; tie-break = lowest cluster id.
        let canonical = *members
            .iter()
            .max_by(|&&a, &&b| {
                let dur_a = cluster_ids
                    .iter()
                    .position(|&id| id == a)
                    .map(|i| cluster_dur[i])
                    .unwrap_or(0);
                let dur_b = cluster_ids
                    .iter()
                    .position(|&id| id == b)
                    .map(|i| cluster_dur[i])
                    .unwrap_or(0);
                dur_a.cmp(&dur_b).then(b.cmp(&a)) // tie: lower id wins (b.cmp(&a) = a < b)
            })
            .expect("members is non-empty");
        for &source in members {
            if source != canonical {
                merge_map.push((source, canonical));
            }
        }
    }

    if !merge_map.is_empty() {
        tracing::info!(
            target: "orchestrator",
            meeting_id = %meeting_id.0,
            merge_count = merge_map.len(),
            "library-merge: merging same-identity diarizer clusters"
        );
    }

    (veto_verdicts, merge_map)
}

/// Split one kept mixed Qwen segment into single-speaker sub-segments by
/// re-ASR'ing each speaker turn's audio (#0015 phase 4), or `None` to keep-whole.
///
/// Returns `None` (caller keeps the segment whole) when:
/// - the segment maps to no pause-INCLUDING PCM range, or
/// - [`diarizer::turn_boundaries_within`] yields no interior cut, or
/// - any cut's [`runner::snap_to_energy_min`] finds no clear minimum (continuous
///   / overlapping speech), or
/// - the resulting sub-clips re-ASR to nothing.
///
/// Otherwise returns one sub-segment per single-speaker slice, lettered from
/// `cluster_letters` by its dominant [`SpeakerTurn`] cluster, with empty
/// `shared_speakers` (no longer mixed) and `start_ms` on the EXCLUDING clock.
fn split_mixed_qwen_segment(
    ctx: &SplitMergeContext<'_>,
    seg: &Segment,
    regions: &[runner::KeptRegion],
    backend: &mut dyn minutist_common::AsrBackend,
    cluster_letters: &[(i32, String)],
) -> AppResult<Option<Vec<Segment>>> {
    let turns = ctx.turns;
    let pcm = ctx.pcm;
    // Map the segment's pause-EXCLUDING [start_ms, end_ms) to the single
    // pause-INCLUDING PCM range the turns share (the clamp matches the offline
    // pause model — a mixed Qwen segment never straddles a ≥4 s pause). `regions`
    // is the caller's ONE pause scan for the whole meeting (B2) — this is the
    // cheap per-segment half, [`runner::excluding_range_to_pcm_slice`].
    let seg_range = match runner::excluding_range_to_pcm_slice(regions, seg.start_ms, seg.end_ms) {
        Some(r) => r,
        None => return Ok(None),
    };

    // Interior speaker-change cuts on the SAME pause-INCLUDING clock the turns +
    // PCM share. `turn_boundaries_within` takes a synthetic segment whose bounds
    // are on the INCLUDING clock (the PCM range's ms), NEVER the excluding bounds.
    let incl_start_ms = (seg_range.start as u64 * 1000) / 16_000;
    let incl_end_ms = (seg_range.end as u64 * 1000) / 16_000;
    let incl_seg = Segment {
        start_ms: incl_start_ms,
        end_ms: incl_end_ms,
        ..seg.clone()
    };
    let cut_ms = diarizer::turn_boundaries_within(&incl_seg, turns);
    if cut_ms.is_empty() {
        return Ok(None);
    }

    // Convert each interior cut (INCLUDING ms) to a PCM sample, then energy-snap
    // it. Any snap with no clear minimum abandons the whole split (keep-whole).
    let mut cut_samples: Vec<usize> = Vec::with_capacity(cut_ms.len());
    for ms in &cut_ms {
        let sample = (*ms as usize * 16_000) / 1000;
        match runner::snap_to_energy_min(pcm, sample, SNAP_SEARCH_WINDOW_MS) {
            Some(snapped) => cut_samples.push(snapped),
            None => return Ok(None),
        }
    }
    cut_samples.sort_unstable();
    cut_samples.dedup();

    // Slice boundaries inside the segment's PCM range: [seg_start, c0, c1, …, seg_end].
    let mut bounds: Vec<usize> = Vec::with_capacity(cut_samples.len() + 2);
    bounds.push(seg_range.start);
    for c in &cut_samples {
        // A snapped cut can land just outside the segment range; clamp + skip a
        // degenerate slice.
        let c = (*c).clamp(seg_range.start, seg_range.end);
        if c > *bounds.last().unwrap() && c < seg_range.end {
            bounds.push(c);
        }
    }
    bounds.push(seg_range.end);
    if bounds.len() < 3 {
        // No usable interior cut survived the clamp — keep-whole.
        return Ok(None);
    }

    let mut sub_segments: Vec<Segment> = Vec::with_capacity(bounds.len() - 1);
    for pair in bounds.windows(2) {
        let (lo, hi) = (pair[0], pair[1]);
        if hi <= lo {
            continue;
        }
        // Stamp each sub-clip's start on the EXCLUDING transcript clock (the
        // inverse map); the chunk's own clock is the INCLUDING ms so the backend's
        // word offsets stay self-consistent within the clip.
        let excl_start_ms = runner::excluding_ms_for_pcm_sample_in_regions(regions, lo);
        let chunk_incl_start_ms = (lo as u64 * 1000) / 16_000;
        let chunk_incl_end_ms = (hi as u64 * 1000) / 16_000;
        let chunk = minutist_common::AudioChunk {
            samples: pcm[lo..hi].to_vec(),
            sample_rate: 16_000,
            start_ms: chunk_incl_start_ms,
            end_ms: chunk_incl_end_ms,
        };
        let re_asr = backend.transcribe_chunk(&chunk)?;

        // Letter this sub-clip by its dominant turn cluster via the WU1 map, so it
        // lands in the EXISTING scheme (no rename). A cluster the overlay pruned
        // away has no map entry → leave `None`.
        let cluster = dominant_cluster(turns, chunk_incl_start_ms, chunk_incl_end_ms);
        let letter = cluster.and_then(|c| {
            cluster_letters
                .iter()
                .find(|(id, _)| *id == c)
                .map(|(_, l)| l.clone())
        });

        let text: String = re_asr
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let sub = Segment {
            start_ms: excl_start_ms,
            // The sub-clip's excluding end is the next slice's excluding start;
            // compute it from `hi` directly (the inverse clamps a trailing edge).
            end_ms: runner::excluding_ms_for_pcm_sample_in_regions(regions, hi),
            text,
            speaker_id: letter,
            confidence: seg.confidence,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        };
        let _ = ctx.event_tx.send(AppEvent::TranscriptSegment {
            meeting_id: ctx.meeting_id,
            segment: sub.clone(),
        });
        sub_segments.push(sub);
    }

    if sub_segments.is_empty() {
        return Ok(None);
    }
    Ok(Some(sub_segments))
}

/// `± window_ms` energy-snap search span for a speaker-change cut (#0015 phase 4).
const SNAP_SEARCH_WINDOW_MS: u64 = 150;
