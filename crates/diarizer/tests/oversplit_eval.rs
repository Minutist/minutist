//! Env-var-gated over-split count-vs-knob eval (issue #63).
//!
//! This is a *diagnostic* harness, not a pass/fail accuracy test. It decodes a
//! real meeting's audio + transcript and reports the distinct `speaker_count`
//! the offline diarizer produces across a sweep of `cluster_threshold`, of the
//! sherpa temporal-smoothing knobs (`min_duration_on` / `min_duration_off`), and
//! of the post-cluster prune (`min_cluster_share`) — so the chosen tuning can be
//! grounded in the count trend on the long, acoustically-varied recordings that
//! over-split in the field (reported 19 / 29 speakers).
//!
//! It is gated on three env vars (any unset → skip + return `Ok`, so the default
//! `cargo test -p diarizer` suite never touches the models or the user's audio):
//!   * `MINUTIST_DIARIZE_SEG_PATH`     — pyannote segmentation ONNX
//!   * `MINUTIST_DIARIZE_EMB_PATH`     — speaker-embedding ONNX
//!   * `MINUTIST_DIARIZE_EVAL_MEETING` — a meeting folder (audio.opus +
//!     transcript.json)
//!
//! The recording is long (~26-31 min); to keep the sweep tractable the harness
//! uses only a leading time-bounded slice (`MINUTIST_DIARIZE_EVAL_SECS`,
//! default 360 s). The model is run ONCE per `cluster_threshold` (the expensive
//! step), and the cheap pure `overlay_speakers` prune stage is re-swept over the
//! cached turns without re-invoking the model.
//!
//! To run:
//!   MINUTIST_DIARIZE_SEG_PATH=/path/seg.onnx \
//!   MINUTIST_DIARIZE_EMB_PATH=/path/emb.onnx \
//!   MINUTIST_DIARIZE_EVAL_MEETING=/path/to/meetings/<uuid> \
//!   cargo test -p diarizer --test oversplit_eval -- --nocapture

use std::path::PathBuf;

use diarizer::{overlay_speakers, DiarizerConfig, SpeakerTurn};
use minutist_common::Segment;
use sherpa_rs::diarize::{Diarize, DiarizeConfig};

const SAMPLE_RATE: u32 = 16_000;
const DEFAULT_SLICE_SECS: u64 = 360;

/// Resolve the gated inputs, or `None` (→ skip) when any required var is unset.
fn eval_inputs() -> Option<(PathBuf, PathBuf, PathBuf, u64)> {
    let non_empty = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
    let seg = non_empty("MINUTIST_DIARIZE_SEG_PATH")?;
    let emb = non_empty("MINUTIST_DIARIZE_EMB_PATH")?;
    let meeting = non_empty("MINUTIST_DIARIZE_EVAL_MEETING")?;
    let secs = non_empty("MINUTIST_DIARIZE_EVAL_SECS")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_SLICE_SECS);
    Some((
        PathBuf::from(seg),
        PathBuf::from(emb),
        PathBuf::from(meeting),
        secs,
    ))
}

/// Run sherpa `Diarize::compute` once with the given threshold + smoothing,
/// returning the turns as diarizer [`SpeakerTurn`]s (seconds → ms, mirroring
/// `SherpaDiarizer::compute_turns`). The model loads + computes here (the
/// expensive part).
fn compute_turns(
    seg_path: &std::path::Path,
    emb_path: &std::path::Path,
    pcm: &[f32],
    cluster_threshold: f32,
    min_duration_on: f32,
    min_duration_off: f32,
) -> Vec<SpeakerTurn> {
    let config = DiarizeConfig {
        num_clusters: Some(-1), // threshold mode (production sentinel)
        threshold: Some(cluster_threshold),
        min_duration_on: Some(min_duration_on),
        min_duration_off: Some(min_duration_off),
        provider: None,
        debug: false,
    };
    let mut engine =
        Diarize::new(seg_path, emb_path, config).expect("open Diarize with the real models");
    let raw = engine
        .compute(pcm.to_vec(), None)
        .expect("Diarize::compute on the meeting slice");
    raw.into_iter()
        .map(|s| SpeakerTurn {
            start_ms: (s.start.max(0.0) * 1000.0).round() as u64,
            end_ms: (s.end.max(0.0) * 1000.0).round() as u64,
            cluster: s.speaker,
        })
        .collect()
}

/// Count the distinct speakers `overlay_speakers` reports for a given prune
/// `share` over already-computed `turns`. Pure + cheap (no model).
fn count_with_prune(turns: &[SpeakerTurn], base: &[Segment], share: f32) -> u32 {
    let cfg = DiarizerConfig {
        min_cluster_share: share,
        min_cluster_segments: 0,
        max_speakers: None,
        ..DiarizerConfig::default()
    };
    // Clear words so the #0015 word-split path is inert here: this eval isolates
    // the over-split PRUNE (the split is covered by the diarizer unit tests, and
    // with words present it could move the count in either direction).
    let mut segs = base.to_vec();
    for s in &mut segs {
        s.words.clear();
    }
    let (_segs, count, _map) = overlay_speakers(turns, segs, &cfg);
    count
}

#[test]
fn oversplit_count_vs_knob_sweep() {
    let Some((seg_path, emb_path, meeting_dir, secs)) = eval_inputs() else {
        eprintln!(
            "skipping oversplit_count_vs_knob_sweep: set MINUTIST_DIARIZE_SEG_PATH, \
             MINUTIST_DIARIZE_EMB_PATH and MINUTIST_DIARIZE_EVAL_MEETING to run"
        );
        return;
    };

    // Decode the full pause-including PCM, then take the leading slice.
    let full = persistence::read_audio_pcm(&meeting_dir).expect("decode audio.opus");
    let slice_samples = (secs * SAMPLE_RATE as u64) as usize;
    let pcm: Vec<f32> = full.iter().take(slice_samples).copied().collect();
    let slice_ms = (pcm.len() as u64 * 1000) / SAMPLE_RATE as u64;

    // Read the transcript and keep the segments inside the slice window.
    let all_segments = persistence::read_transcript(&meeting_dir).expect("read transcript.json");
    let base: Vec<Segment> = all_segments
        .into_iter()
        .filter(|s| s.start_ms < slice_ms)
        .map(|mut s| {
            // Clear any persisted (live) label so the count reflects this pass.
            s.speaker_id = None;
            s
        })
        .collect();

    eprintln!("=== over-split eval: {} ===", meeting_dir.display());
    eprintln!(
        "slice = {} s ({} samples), segments in window = {}",
        slice_ms / 1000,
        pcm.len(),
        base.len()
    );

    // Sweep cluster_threshold (distance: higher => fewer speakers). For each,
    // run the model once with the production smoothing, then re-sweep the prune.
    let thresholds = [0.50_f32, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95];
    let prune_shares = [0.0_f32, 0.01, 0.02, 0.05];

    // First, the smoothing on/off contrast at the production threshold (0.75).
    eprintln!("\n-- smoothing contrast @ cluster_threshold = 0.75, prune OFF --");
    for (label, on, off) in [
        ("smoothing OFF (old 0.0/0.0)", 0.0_f32, 0.0_f32),
        ("smoothing ON  (new 0.3/0.5)", 0.3, 0.5),
    ] {
        let turns = compute_turns(&seg_path, &emb_path, &pcm, 0.75, on, off);
        let count = count_with_prune(&turns, &base, 0.0);
        eprintln!("  {label}: turns = {}, speaker_count = {count}", turns.len());
    }

    // Main grid: threshold x prune-share, with production smoothing on.
    eprintln!("\n-- speaker_count grid (rows: cluster_threshold, cols: min_cluster_share) --");
    eprint!("  thr \\ share ");
    for s in &prune_shares {
        eprint!("| {s:>5.2} ");
    }
    eprintln!();
    for &thr in &thresholds {
        let turns = compute_turns(&seg_path, &emb_path, &pcm, thr, 0.3, 0.5);
        eprint!("  {thr:>10.2} ");
        for &share in &prune_shares {
            let count = count_with_prune(&turns, &base, share);
            eprint!("| {count:>5} ");
        }
        eprintln!();
    }

    // The headline before/after at the SHIPPED config: old (0.75, no smoothing,
    // no prune) vs new (DiarizerConfig::default()).
    let old_turns = compute_turns(&seg_path, &emb_path, &pcm, 0.75, 0.0, 0.0);
    let old_count = count_with_prune(&old_turns, &base, 0.0);
    let new_turns = compute_turns(&seg_path, &emb_path, &pcm, 0.75, 0.3, 0.5);
    let mut new_segs = base.clone();
    for s in &mut new_segs {
        s.words.clear();
    }
    let (_new_segs, new_count, _map) =
        overlay_speakers(&new_turns, new_segs, &DiarizerConfig::default());
    eprintln!(
        "\n-- BEFORE/AFTER on this slice --\n  OLD (thr 0.75, no smoothing, no prune): {old_count}\
         \n  NEW (DiarizerConfig::default()):        {new_count}"
    );

    // The eval is diagnostic; the only hard invariant is that the new default
    // does not produce MORE speakers than the old config. Both sides run with
    // words cleared (above), so the #0015 word-split is inert and this compares
    // the prune + smoothing in isolation — which can only merge, never split,
    // relative to the unsmoothed/unpruned baseline at the same threshold.
    assert!(
        new_count <= old_count,
        "new default ({new_count}) must not over-split worse than the old config ({old_count})"
    );
}
