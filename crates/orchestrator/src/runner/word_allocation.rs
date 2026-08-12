//! Proportional word allocation: re-splits one ASR transcript across the VAD sub-segments it was flushed from.

use super::*;

/// Split the transcript `text` across `vad_segments` proportionally by
/// per-segment audio duration.
///
/// Returns one `Segment` per VAD segment. If `text` is empty, each segment
/// carries an empty text string (timestamps are preserved from the VAD layer).
/// The last segment absorbs any word-count rounding remainder.
///
/// `speaker_ids` carries the live provisional speaker label for each VAD
/// segment (Phase B), indexed in lockstep with `vad_segments`. The output
/// `Segment` at index `i` inherits `speaker_ids[i]` via `.get(i)`, so a
/// (theoretically impossible) shorter `speaker_ids` slice yields `None` for the
/// missing tail rather than panicking on the worker thread.
pub(crate) fn emit_segments_proportional(
    text: &str,
    vad_segments: &[(u64, u64)],
    speaker_ids: &[Option<String>],
) -> Vec<Segment> {
    if vad_segments.is_empty() {
        return Vec::new();
    }

    let total_ms: u64 = vad_segments
        .iter()
        .map(|(s, e)| e.saturating_sub(*s))
        .sum();

    let words: Vec<&str> = text.split_whitespace().collect();
    let n = vad_segments.len();
    let mut out = Vec::with_capacity(n);
    let mut word_idx = 0usize;

    for (i, (start_ms, end_ms)) in vad_segments.iter().enumerate() {
        let seg_ms = end_ms.saturating_sub(*start_ms);
        let take = if words.is_empty() {
            0
        } else if i == n - 1 {
            words.len() - word_idx
        } else if total_ms == 0 {
            0
        } else {
            let proportion = seg_ms as f64 / total_ms as f64;
            let count = (proportion * words.len() as f64).round() as usize;
            count.min(words.len() - word_idx)
        };

        let seg_text = words[word_idx..word_idx + take].join(" ");
        word_idx += take;

        out.push(Segment {
            start_ms: *start_ms,
            end_ms: *end_ms,
            text: seg_text,
            speaker_id: speaker_ids.get(i).cloned().flatten(),
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        });
    }

    out
}

