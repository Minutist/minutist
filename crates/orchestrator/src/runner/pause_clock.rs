//! The pause-excluding clock: maps between the pause-INCLUDING decoded PCM and the pause-EXCLUDING clock the live transcript, re-transcribe, and re-listen paths share.

use super::*;


/// A contiguous non-pause region of the decoded PCM, tagged with where it sits
/// on the pause-EXCLUDING clock (TIMELINE-DRIFT #4). `src_start..src_end` index
/// the original (pause-INCLUDING) `pcm`; `excl_start_ms` is the region's start
/// on the pause-excluding timeline (the cumulative duration of all kept audio
/// before it).
pub(crate) struct KeptRegion {
    pub(super) src_start: usize,
    pub(super) src_end: usize,
    pub(super) excl_start_ms: u64,
}

/// Amplitude at or below which a sample counts as "silent" for pause detection.
/// The Opus encoder synthesises exact-zero frames for a pause; the lossy decode
/// reconstructs them as very-low-amplitude samples (the existing persistence
/// pause tests assert the pause region decodes to `< 0.02` peak), so a small
/// epsilon reliably catches encoder pause padding without flagging speech.
const PAUSE_SILENCE_EPS: f32 = 0.02;

/// Minimum length of a near-silent run to be treated as a recording **pause**
/// (and excluded from the pause-excluding timeline). Set comfortably above the
/// live accumulator's `MAX_GAP_MS` (3 s) inter-utterance cap so only genuine
/// user pauses — never a natural quiet gap the live capture clock would have
/// counted — are skipped.
const PAUSE_MIN_MS: u64 = 4000;

/// Split `pcm` into the non-pause regions to feed the offline VAD, excluding
/// encoder-synthesised pause padding from the timeline (TIMELINE-DRIFT #4).
///
/// A "pause" is a run of at least `PAUSE_MIN_MS` of near-silent
/// (`|x| <= PAUSE_SILENCE_EPS`) samples. Each returned [`KeptRegion`] carries
/// the source index range of kept audio plus its start offset on the
/// pause-EXCLUDING clock, which advances only over kept audio. Short quiet gaps
/// (below the threshold) are NOT split out — they remain part of a kept region,
/// matching the live capture clock which counts them.
pub(crate) fn pause_excluding_segments(pcm: &[f32]) -> Vec<KeptRegion> {
    let pause_min_samples = (PAUSE_MIN_MS as usize * SAMPLE_RATE_HZ as usize) / 1000;

    let mut regions: Vec<KeptRegion> = Vec::new();
    let mut excl_start_ms: u64 = 0;
    let mut region_start = 0usize; // start of the current kept region
    let mut i = 0usize;

    while i < pcm.len() {
        // Measure a near-silent run starting at `i`.
        if pcm[i].abs() <= PAUSE_SILENCE_EPS {
            let run_start = i;
            let mut j = i;
            while j < pcm.len() && pcm[j].abs() <= PAUSE_SILENCE_EPS {
                j += 1;
            }
            let run_len = j - run_start;

            if run_len >= pause_min_samples {
                // This silent run is a pause. Close the current kept region at
                // `run_start` (it ends just before the pause) and skip the run.
                if run_start > region_start {
                    let kept_len = run_start - region_start;
                    regions.push(KeptRegion {
                        src_start: region_start,
                        src_end: run_start,
                        excl_start_ms,
                    });
                    excl_start_ms += (kept_len as u64 * 1000) / SAMPLE_RATE_HZ;
                }
                // The next kept region begins after the pause; its
                // pause-excluding start is unchanged (the pause contributed no
                // pause-excluding time), exactly as the live capture clock froze
                // during the pause.
                region_start = j;
            }
            // Else: a short quiet gap — keep it in the current region.
            i = j;
        } else {
            i += 1;
        }
    }

    // Close the trailing kept region.
    if pcm.len() > region_start {
        regions.push(KeptRegion {
            src_start: region_start,
            src_end: pcm.len(),
            excl_start_ms,
        });
    }

    regions
}

/// Map a pause-EXCLUDING window `[start_ms, end_ms)` (transcript-clock
/// timestamps) onto a single slice of the pause-INCLUDING decoded `pcm`
/// (Phase 9 — backs `Orchestrator::transcribe_pcm_window`).
///
/// `Segment::start_ms`/`end_ms` live on the pause-EXCLUDING capture clock, but
/// `read_audio_pcm` returns the pause-INCLUDING buffer. To slice the right
/// audio for a re-listen we walk the [`pause_excluding_segments`] kept regions
/// (the same pause model the offline re-transcribe uses) and translate the
/// requested excluding interval back into PCM sample indices.
///
/// **Straddling a pause (W1 decision — clamp, do NOT concatenate).** A
/// pause-excluding window can span a skipped pause, which on the
/// pause-INCLUDING clock maps to two disjoint PCM ranges separated by the
/// pause's synthesised silence. Concatenating them would seam two non-adjacent
/// audio spans and make the re-transcribed timestamps impossible to map cleanly
/// back onto the meeting timeline. v1 therefore **clamps to the single kept
/// region that contains `start_ms`** and slices within it up to `end_ms` (or the
/// region end, whichever is first). The tool's `description` states this caveat.
/// Returns `None` when `start_ms` falls past the last kept region's
/// excluding-clock extent (an out-of-range request).
pub(crate) fn pcm_window_for_excluding_range(
    pcm: &[f32],
    start_ms: u64,
    end_ms: u64,
) -> Option<std::ops::Range<usize>> {
    excluding_range_to_pcm_slice(&pause_excluding_segments(pcm), start_ms, end_ms)
}

/// The per-request half of [`pcm_window_for_excluding_range`]: map the
/// pause-EXCLUDING window onto a PCM slice given an already-computed kept-region
/// table. Split out so the re-listen path can compute the O(samples)
/// [`pause_excluding_segments`] scan once per meeting (cached alongside the
/// decoded PCM) and run only this cheap region math per clip request.
pub(crate) fn excluding_range_to_pcm_slice(
    regions: &[KeptRegion],
    start_ms: u64,
    end_ms: u64,
) -> Option<std::ops::Range<usize>> {
    for region in regions {
        let region_kept_samples = region.src_end - region.src_start;
        let region_excl_len_ms =
            (region_kept_samples as u64 * 1000) / SAMPLE_RATE_HZ;
        let region_excl_end_ms = region.excl_start_ms + region_excl_len_ms;

        // The first region whose excluding-clock extent contains `start_ms`.
        if start_ms < region_excl_end_ms {
            // Offset of `start_ms` within this region, in pause-excluding ms.
            let into_region_ms = start_ms.saturating_sub(region.excl_start_ms);
            let start_off = (into_region_ms as usize * SAMPLE_RATE_HZ as usize) / 1000;
            // Clamp `end_ms` to this region (W1 clamp decision).
            let clamped_end_ms = end_ms.min(region_excl_end_ms);
            let end_into_region_ms = clamped_end_ms.saturating_sub(region.excl_start_ms);
            let end_off = (end_into_region_ms as usize * SAMPLE_RATE_HZ as usize) / 1000;

            let lo = region.src_start + start_off.min(region_kept_samples);
            let hi = region.src_start + end_off.min(region_kept_samples);
            if hi <= lo {
                return None;
            }
            return Some(lo..hi);
        }
    }
    None
}

/// Encode a 16 kHz mono f32 PCM slice as a self-contained little-endian PCM16
/// WAV byte buffer (44-byte RIFF/WAVE header + samples). Backs
/// [`crate::Orchestrator::extract_segment_wav`], which hands the webview a
/// pre-cut, seek-free clip for the transcript "play segment" affordance:
/// serving WAV (not the source Ogg/Opus) keeps the granule-position
/// non-conformance (#0024) and all container quirks out of the playback path.
pub(crate) fn pcm16_wav(samples: &[f32]) -> Vec<u8> {
    // The caller (`extract_segment_wav`) caps the slice well under 4 GiB; assert
    // it so the u32 RIFF / `data` length fields below cannot silently truncate.
    debug_assert!(
        samples
            .len()
            .checked_mul(2)
            .is_some_and(|n| n <= u32::MAX as usize),
        "pcm16_wav slice too large for a 32-bit WAV length field",
    );
    let sr = SAMPLE_RATE_HZ as u32;
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt-chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate (mono, 16-bit)
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    out
}

/// Inverse of [`pcm_window_for_excluding_range`]: map a pause-INCLUDING PCM
/// sample index back to its position on the pause-EXCLUDING transcript clock,
/// in milliseconds (#0015 phase 4), given an already-computed [`KeptRegion`]
/// table (see [`pause_excluding_segments`]).
///
/// [`pcm_window_for_excluding_range`] is forward-only (excluding-ms → PCM
/// range), so the re-ASR split needs this inverse to stamp each single-speaker
/// sub-clip's `start_ms` on the transcript clock. Without it a sub-clip cut at a
/// pause-INCLUDING sample would inherit the cumulative pre-segment pause padding
/// and drift forward by every pause before it. Takes `regions` rather than
/// `pcm` directly because the re-ASR split calls this once per interior cut —
/// the caller computes the O(samples) [`pause_excluding_segments`] scan ONCE
/// per segment and shares the table across every cut (B2).
///
/// Walks the kept regions and inverts the per-region accounting: for the
/// region that contains `sample`, the excluding-clock position is the
/// region's `excl_start_ms` plus the kept-audio duration from the region start
/// up to `sample`. A `sample` that falls inside a skipped pause (between two
/// kept regions) clamps to the END of the preceding kept region — the same
/// instant the excluding clock froze when the pause began — so a cut
/// energy-snapped a few ms into pause padding still lands coherently. A
/// `sample` past the last kept region clamps to the total kept duration. Pure
/// (no FFI/IO) so the inverse round-trips under unit test.
pub(crate) fn excluding_ms_for_pcm_sample_in_regions(regions: &[KeptRegion], sample: usize) -> u64 {
    let mut last_region_excl_end_ms: u64 = 0;
    for region in regions {
        let region_kept_samples = region.src_end - region.src_start;
        let region_excl_len_ms = (region_kept_samples as u64 * 1000) / SAMPLE_RATE_HZ;
        let region_excl_end_ms = region.excl_start_ms + region_excl_len_ms;

        if sample < region.src_start {
            // The sample sits in the pause that precedes this kept region (or in
            // the leading pause before the first region): the excluding clock was
            // frozen at the previous region's end (0 before the first region).
            return last_region_excl_end_ms;
        }
        if sample < region.src_end {
            // Inside this kept region: excluding position = region start +
            // kept-audio duration from the region start to `sample`.
            let into_region = sample - region.src_start;
            let into_region_ms = (into_region as u64 * 1000) / SAMPLE_RATE_HZ;
            return region.excl_start_ms + into_region_ms;
        }
        last_region_excl_end_ms = region_excl_end_ms;
    }
    // Past the last kept region (trailing pause or out of range): the total kept
    // duration.
    last_region_excl_end_ms
}

/// Length of an RMS analysis window when snapping a cut to a local energy
/// minimum (~5 ms at 16 kHz).
const SNAP_RMS_WINDOW_MS: u64 = 5;

/// Relative-RMS floor below which a candidate cut is accepted as a genuine
/// low-energy boundary. The argmin window's RMS must be at most this fraction of
/// the search span's mean RMS; otherwise the span is continuous/overlapping
/// speech with no clear gap and [`snap_to_energy_min`] returns `None` so the
/// caller keeps the segment whole.
///
/// Calibrated conservatively: a real inter-speaker boundary in meeting audio
/// dips well under half the local mean energy, while a hand-off mid-phrase with
/// no breath does not. Too strict here silently forces keep-whole; the abandon
/// is logged so a real mixed clip can recalibrate it.
const SNAP_REL_FLOOR: f32 = 0.5;

/// Snap a speaker-change cut to the lowest-energy sample near `cut_sample`,
/// searching `± window_ms` (#0015 phase 4).
///
/// `samples` is the pause-INCLUDING PCM the cut indexes. The search slides a
/// `SNAP_RMS_WINDOW_MS` RMS window across `[cut_sample - window_ms, cut_sample +
/// window_ms]` (clamped to the buffer) and returns the start sample of the
/// minimum-RMS window — the quietest instant, where cutting the audio least
/// disturbs either speaker's words.
///
/// Returns `None` when there is **no clear minimum**: if the argmin window's RMS
/// is not at most [`SNAP_REL_FLOOR`] of the search span's mean RMS, the span is
/// continuous or overlapping speech with no real gap, and the caller keeps the
/// segment whole rather than cutting mid-word. Also `None` when the search span
/// is degenerate (empty or shorter than one RMS window). Emits `tracing::debug`
/// when a snap is abandoned. Pure (no FFI/IO).
pub(crate) fn snap_to_energy_min(
    samples: &[f32],
    cut_sample: usize,
    window_ms: u64,
) -> Option<usize> {
    let rms_win = (SNAP_RMS_WINDOW_MS as usize * SAMPLE_RATE_HZ as usize) / 1000;
    if rms_win == 0 || samples.is_empty() {
        return None;
    }
    let window_samples = (window_ms as usize * SAMPLE_RATE_HZ as usize) / 1000;

    let lo = cut_sample.saturating_sub(window_samples);
    // The last valid RMS-window start keeps the whole window inside the span.
    let hi = (cut_sample + window_samples).min(samples.len());
    if hi <= lo || hi - lo < rms_win {
        tracing::debug!(
            target: "orchestrator",
            cut_sample,
            window_ms,
            "snap_to_energy_min: search span too short; keeping segment whole"
        );
        return None;
    }

    let rms_at = |start: usize| -> f32 {
        let end = (start + rms_win).min(samples.len());
        let win = &samples[start..end];
        if win.is_empty() {
            return f32::MAX;
        }
        let sum_sq: f32 = win.iter().map(|s| s * s).sum();
        (sum_sq / win.len() as f32).sqrt()
    };

    let last_start = hi - rms_win;
    let mut best_start = lo;
    let mut best_rms = f32::MAX;
    let mut rms_sum = 0.0f32;
    let mut rms_count = 0u32;
    for start in lo..=last_start {
        let rms = rms_at(start);
        rms_sum += rms;
        rms_count += 1;
        if rms < best_rms {
            best_rms = rms;
            best_start = start;
        }
    }

    let mean_rms = if rms_count > 0 {
        rms_sum / rms_count as f32
    } else {
        0.0
    };
    // A genuine boundary dips well below the local mean; continuous/overlapping
    // speech does not. `mean_rms == 0` (digital silence everywhere) is itself a
    // clear minimum, so accept it.
    if mean_rms > 0.0 && best_rms > SNAP_REL_FLOOR * mean_rms {
        tracing::debug!(
            target: "orchestrator",
            cut_sample,
            best_rms,
            mean_rms,
            "snap_to_energy_min: no clear minimum (continuous speech); keeping segment whole"
        );
        return None;
    }

    Some(best_start)
}
