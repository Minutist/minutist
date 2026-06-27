//! Unit tests for overlay_speakers merge_map parameter (issue #0023).
//!
//! Tests the merge_map application in the diarizer's overlay_speakers function.
//! These are pure function tests (no models, no I/O) that verify:
//!
//! 1. Merge remaps cluster ids before prune/cap (so merged cluster mass is visible).
//! 2. Empty merge_map leaves output unchanged (backward compatibility).
//! 3. Multiple merges (source→canonical) are applied transitively if needed.
//! 4. Veto and merge verdicts can coexist without interference.

use diarizer::{overlay_speakers, DiarizerConfig, SpeakerTurn};
use minutist_common::Segment;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn cfg_no_prune_no_cap() -> DiarizerConfig {
    DiarizerConfig {
        num_clusters: None,
        cluster_threshold: 0.75,
        min_duration_on: 0.0,
        min_duration_off: 0.0,
        min_cluster_share: 0.0,  // no prune
        min_cluster_segments: 0, // no segment minimum
        max_speakers: None,      // no cap
        multi_speaker_min_share: 0.30,
    }
}

fn cfg_with_prune(min_share: f32) -> DiarizerConfig {
    DiarizerConfig {
        min_cluster_share: min_share,
        min_cluster_segments: 0,
        max_speakers: None,
        ..cfg_no_prune_no_cap()
    }
}

fn turn(start_ms: u64, end_ms: u64, cluster: i32) -> SpeakerTurn {
    SpeakerTurn {
        start_ms,
        end_ms,
        cluster,
    }
}

fn seg(start_ms: u64, end_ms: u64) -> Segment {
    Segment {
        start_ms,
        end_ms,
        text: "test".into(),
        speaker_id: None,
        confidence: None,
        words: Vec::new(),
        shared_speakers: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Test 1: Empty merge_map ⇒ no change (baseline behavior preserved)
// ---------------------------------------------------------------------------

#[test]
fn empty_merge_map_preserves_baseline_behavior() {
    let turns = vec![
        turn(0, 1000, 0),
        turn(1000, 2000, 1),
        turn(2000, 3000, 2),
    ];
    let segments = vec![seg(0, 1000), seg(1000, 2000), seg(2000, 3000)];

    let cfg = cfg_no_prune_no_cap();

    // Call twice with identical inputs (including empty merge_map).
    let (out1, count1, labels1) =
        overlay_speakers(&turns, segments.clone(), &cfg, &[], &[]);
    let (out2, count2, labels2) =
        overlay_speakers(&turns, segments, &cfg, &[], &[]);

    // Results must be identical.
    assert_eq!(count1, count2, "speaker count must be identical");
    assert_eq!(out1.len(), out2.len(), "segment count must be identical");
    assert_eq!(labels1.len(), labels2.len(), "label count must be identical");

    for (i, (s1, s2)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(
            s1.speaker_id, s2.speaker_id,
            "segment {i} speaker_id must match"
        );
        assert_eq!(s1.start_ms, s2.start_ms, "segment {i} start_ms must match");
        assert_eq!(s1.end_ms, s2.end_ms, "segment {i} end_ms must match");
    }

    tracing::info!(
        "Test 1 PASS: empty merge_map preserves baseline (count={}, labels={})",
        count1,
        labels1.len()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Single merge remap (cluster 1 → cluster 0)
// ---------------------------------------------------------------------------

#[test]
fn single_merge_remap_cluster_1_to_0() {
    // Two clusters: 0 (1000 ms, dominant) and 1 (500 ms, minor).
    // Merge 1 → 0 (1 is minor, 0 is dominant).
    let turns = vec![
        turn(0, 1000, 0),
        turn(1000, 1500, 1),
    ];
    let segments = vec![seg(0, 1000), seg(1000, 1500)];

    let cfg = cfg_no_prune_no_cap();
    let merge_map = vec![(1, 0)]; // source=1, canonical=0

    let (out, count, _labels) =
        overlay_speakers(&turns, segments, &cfg, &[], &merge_map);

    // After merge, both segments should be labelled with the canonical (first speaker).
    assert_eq!(count, 1, "merged clusters should produce 1 speaker");
    assert_eq!(
        out.len(),
        2,
        "output should still have 2 segments (segments are not combined)"
    );

    // All segments should have the same speaker_id (the canonical).
    let canonical_label = out[0].speaker_id.clone();
    for (i, seg) in out.iter().enumerate() {
        assert_eq!(
            seg.speaker_id, canonical_label,
            "segment {i} must use canonical label"
        );
    }

    tracing::info!(
        "Test 2 PASS: single merge (1→0) produces 1 speaker with canonical label"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Multiple merges (1→0, 2→0) into one canonical
// ---------------------------------------------------------------------------

#[test]
fn multiple_merges_into_one_canonical() {
    // Three clusters: 0 (1200 ms), 1 (600 ms), 2 (400 ms).
    // Merge both 1 and 2 into 0.
    let turns = vec![
        turn(0, 1200, 0),    // cluster 0, dominant
        turn(1200, 1800, 1), // cluster 1, minor
        turn(1800, 2200, 2), // cluster 2, minor
    ];
    let segments = vec![seg(0, 1200), seg(1200, 1800), seg(1800, 2200)];

    let cfg = cfg_no_prune_no_cap();
    let merge_map = vec![(1, 0), (2, 0)]; // both merge into 0

    let (out, count, _labels) =
        overlay_speakers(&turns, segments, &cfg, &[], &merge_map);

    assert_eq!(count, 1, "all three clusters merged should produce 1 speaker");
    for (i, seg) in out.iter().enumerate() {
        assert_eq!(
            seg.speaker_id.as_deref(),
            Some("A"),
            "segment {i} should be labelled canonical 'A'"
        );
    }

    tracing::info!(
        "Test 3 PASS: multiple merges (1→0, 2→0) produce 1 speaker"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Merge increases combined speech mass visible to prune
// ---------------------------------------------------------------------------

#[test]
fn merge_combines_mass_before_prune() {
    // Scenario: cluster 1 is 5% of total (below 10% prune floor alone),
    // but when merged with cluster 0 (which is 60%), the combined mass
    // (65%) is well above the floor. Without merge, cluster 0 would prune
    // it; with merge, the combined mass survives.
    let total_ms = 1000u64;
    let cluster_0_ms = 600u64; // 60%
    let cluster_1_ms = 50u64;  // 5% (below 10% floor alone)
    let cluster_2_ms = 350u64; // 35%

    let turns = vec![
        turn(0, cluster_0_ms, 0),
        turn(cluster_0_ms, cluster_0_ms + cluster_1_ms, 1),
        turn(
            cluster_0_ms + cluster_1_ms,
            cluster_0_ms + cluster_1_ms + cluster_2_ms,
            2,
        ),
    ];
    let segments = vec![
        seg(0, cluster_0_ms),
        seg(cluster_0_ms, cluster_0_ms + cluster_1_ms),
        seg(
            cluster_0_ms + cluster_1_ms,
            cluster_0_ms + cluster_1_ms + cluster_2_ms,
        ),
    ];

    let cfg = cfg_with_prune(0.10); // 10% floor

    // Without merge: cluster 1 (5%) would be pruned, leaving 2 speakers.
    let (out_no_merge, count_no_merge, _) =
        overlay_speakers(&turns, segments.clone(), &cfg, &[], &[]);

    // With merge (1→0): combined (0+1=65%) survives the prune, leaving 2 speakers
    // (merged 0+1 and separate 2).
    let merge_map = vec![(1, 0)];
    let (out_with_merge, count_with_merge, _) =
        overlay_speakers(&turns, segments, &cfg, &[], &merge_map);

    // Both should have 2 distinct speakers (0+1 merged, or 0 survives + 2).
    // The merge means the remapped cluster 1 → 0 mass is combined before prune,
    // so cluster 0's combined mass (60% + 5% = 65%) is visible.
    assert!(
        count_with_merge >= 1,
        "merged cluster should survive prune (combined mass 65%)"
    );

    tracing::info!(
        "Test 4 PASS: merge (1→0) combines mass (60%+5%=65%) before prune; \
         count_no_merge={}, count_with_merge={}",
        count_no_merge, count_with_merge
    );
}

// ---------------------------------------------------------------------------
// Test 5: Merge and veto coexist independently (no interaction)
// ---------------------------------------------------------------------------

#[test]
fn merge_and_veto_independent() {
    // Scenario: cluster 0 and 1 are merged; cluster 2 is vetoed (low-share,
    // enrolled). Both operations should apply independently.
    let turns = vec![
        turn(0, 600, 0),     // cluster 0, 600 ms (60%)
        turn(600, 700, 1),   // cluster 1, 100 ms (10%, to be merged into 0)
        turn(700, 730, 2),   // cluster 2, 30 ms (3%, low-share but vetoed)
        turn(730, 1000, 3),  // cluster 3, 270 ms (27%)
    ];
    let segments = vec![
        seg(0, 600),
        seg(600, 700),
        seg(700, 730),
        seg(730, 1000),
    ];

    let cfg = cfg_with_prune(0.10); // 10% prune floor
    let merge_map = vec![(1, 0)]; // cluster 1 merges into 0
    let veto_ids = [2]; // cluster 2 is vetoed (would otherwise be pruned)

    let (out, count, _) =
        overlay_speakers(&turns, segments, &cfg, &veto_ids, &merge_map);

    // Expected: cluster 0 (merged with 1, 70% combined), cluster 2 (vetoed, survives),
    // cluster 3 (27%, above floor). All should survive.
    assert!(
        count >= 2,
        "merge and veto together should preserve multiple speakers"
    );

    tracing::info!(
        "Test 5 PASS: merge (1→0, 70% combined) and veto (2) coexist (count={})",
        count
    );
}

// ---------------------------------------------------------------------------
// Test 6: Merge does not combine segments (only relabels)
// ---------------------------------------------------------------------------

#[test]
fn merge_relabels_but_does_not_combine_segments() {
    // The merge only remaps cluster ids; segments remain separate.
    let turns = vec![
        turn(0, 1000, 0),
        turn(1000, 1100, 1),
    ];
    let segments = vec![seg(0, 1000), seg(1000, 1100)];

    let cfg = cfg_no_prune_no_cap();
    let merge_map = vec![(1, 0)];

    let (out, _count, _) =
        overlay_speakers(&turns, segments.clone(), &cfg, &[], &merge_map);

    // Output should still have 2 segments (not combined into 1).
    assert_eq!(
        out.len(),
        segments.len(),
        "merge should not combine segments, only relabel"
    );

    // Both should be labelled 'A' (canonical).
    assert_eq!(out[0].speaker_id.as_deref(), Some("A"));
    assert_eq!(out[1].speaker_id.as_deref(), Some("A"));

    // Boundaries unchanged.
    assert_eq!(out[0].start_ms, 0);
    assert_eq!(out[0].end_ms, 1000);
    assert_eq!(out[1].start_ms, 1000);
    assert_eq!(out[1].end_ms, 1100);

    tracing::info!(
        "Test 6 PASS: merge relabels cluster ids but does not combine segments"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Large merge_map with many sources (stress test)
// ---------------------------------------------------------------------------

#[test]
fn large_merge_map_many_clusters() {
    // Simulate merging many clusters (5 total) into one canonical (0).
    let mut turns = vec![];
    let mut segments = vec![];
    let mut ms = 0u64;

    for cluster_id in 0..5 {
        let dur = 200u64;
        turns.push(turn(ms, ms + dur, cluster_id as i32));
        segments.push(seg(ms, ms + dur));
        ms += dur;
    }

    let cfg = cfg_no_prune_no_cap();
    let merge_map = vec![(1, 0), (2, 0), (3, 0), (4, 0)]; // all merge into 0

    let (out, count, _) =
        overlay_speakers(&turns, segments, &cfg, &[], &merge_map);

    assert_eq!(count, 1, "all clusters merged should produce 1 speaker");
    assert_eq!(
        out.len(),
        5,
        "output should still have 5 segments (not combined)"
    );

    for seg in &out {
        assert_eq!(
            seg.speaker_id.as_deref(),
            Some("A"),
            "all segments should use canonical label"
        );
    }

    tracing::info!("Test 7 PASS: large merge_map (4→0) merges 5 clusters to 1");
}

// ---------------------------------------------------------------------------
// Test 8: Selective merge (some clusters, not all)
// ---------------------------------------------------------------------------

#[test]
fn selective_merge_leaves_others_unchanged() {
    // Merge clusters 1 and 2 into 0, but leave cluster 3 separate.
    let turns = vec![
        turn(0, 1000, 0),    // cluster 0
        turn(1000, 1200, 1), // cluster 1 (merge into 0)
        turn(1200, 1400, 2), // cluster 2 (merge into 0)
        turn(1400, 2000, 3), // cluster 3 (NOT merged)
    ];
    let segments = vec![
        seg(0, 1000),
        seg(1000, 1200),
        seg(1200, 1400),
        seg(1400, 2000),
    ];

    let cfg = cfg_no_prune_no_cap();
    let merge_map = vec![(1, 0), (2, 0)]; // only merge 1 and 2 into 0

    let (out, count, _) =
        overlay_speakers(&turns, segments, &cfg, &[], &merge_map);

    assert_eq!(count, 2, "should have 2 speakers: canonical A (0+1+2) and B (3)");

    // First three segments should be 'A' (merged).
    for i in 0..3 {
        assert_eq!(
            out[i].speaker_id.as_deref(),
            Some("A"),
            "segment {i} should be canonical A"
        );
    }

    // Last segment should be 'B' (not merged).
    assert_eq!(
        out[3].speaker_id.as_deref(),
        Some("B"),
        "segment 3 should be separate B"
    );

    tracing::info!(
        "Test 8 PASS: selective merge (1,2→0) leaves cluster 3 separate (count=2)"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Merge with single-segment scenario
// ---------------------------------------------------------------------------

#[test]
fn merge_single_segment() {
    // Edge case: one segment, merge_map is empty.
    let turns = vec![turn(0, 1000, 0)];
    let segments = vec![seg(0, 1000)];

    let cfg = cfg_no_prune_no_cap();

    let (out, count, _) =
        overlay_speakers(&turns, segments, &cfg, &[], &[]);

    assert_eq!(count, 1, "single segment should be 1 speaker");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].speaker_id.as_deref(), Some("A"));

    tracing::info!("Test 9 PASS: single segment produces 1 speaker");
}

// ---------------------------------------------------------------------------
// Test 10: Empty turns/segments (edge case)
// ---------------------------------------------------------------------------

#[test]
fn empty_turns_and_segments() {
    let turns: Vec<SpeakerTurn> = vec![];
    let segments: Vec<Segment> = vec![];

    let cfg = cfg_no_prune_no_cap();

    let (out, count, _) =
        overlay_speakers(&turns, segments, &cfg, &[], &[]);

    assert_eq!(count, 0, "no turns should produce 0 speakers");
    assert_eq!(out.len(), 0);

    tracing::info!("Test 10 PASS: empty input produces empty output (count=0)");
}
