//! Markdown / transcript chunking for retrieval.
//!
//! Fixed-size character windows with overlap, each extended to the next newline
//! so chunks break on line boundaries (paragraphs, list items, speaker turns)
//! rather than mid-sentence. Char-boundary-safe: never splits a multi-byte
//! UTF-8 sequence. ~`chunk_chars` ≈ `chunk_chars / 4` tokens; the SP-LIVE E6
//! benchmark used ~256-token (≈1024-char) chunks with ~200-char overlap.

use serde::{Deserialize, Serialize};

/// One retrievable chunk of a source document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// The chunk text.
    pub text: String,
    /// Byte offset of the chunk start within the source document (for
    /// provenance / re-anchoring back to the original).
    pub byte_offset: usize,
}

/// Largest byte index `<= idx` that is a char boundary of `s`.
fn floor_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// How far past the target window to scan for a newline so a chunk ends on a line
/// boundary (paragraph, list item, speaker turn) instead of mid-sentence — a small
/// fixed slack, large enough to reach the end of a typical prose/markdown line yet
/// small enough not to meaningfully skew the ~256-token chunk budget.
const NEWLINE_LOOKAHEAD_BYTES: usize = 400;

/// Split `text` into ~`chunk_chars`-char windows (each extended to the next
/// newline) with `overlap` characters of overlap between consecutive chunks.
///
/// `overlap` is clamped to `chunk_chars / 2`, so the step between chunks is at
/// least `chunk_chars / 2`. This guarantees forward progress and bounds total
/// output to ~2× the input (an unclamped `overlap == chunk_chars - 1` would make
/// the step ~1 char and blow output up to O(n · chunk_chars)). A single chunk is
/// at most `chunk_chars` + `NEWLINE_LOOKAHEAD_BYTES` (the newline-lookahead
/// window). Returns an empty vec for empty input.
pub fn chunk_text(text: &str, chunk_chars: usize, overlap: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let n = text.len();
    if n == 0 || chunk_chars == 0 {
        return chunks;
    }
    let overlap = overlap.min(chunk_chars / 2);
    let mut start = 0usize;
    while start < n {
        let target = floor_boundary(text, (start + chunk_chars).min(n));
        // Extend to the next newline within a small lookahead so chunks end on
        // a line boundary; otherwise hard-cut at the char-boundary target.
        let mut end = target;
        if end < n {
            // Clamp the lookahead's upper bound to a char boundary too — `end` is
            // already aligned, but `(end + NEWLINE_LOOKAHEAD_BYTES).min(n)` can land
            // mid-codepoint and panic the slice on multi-byte (e.g. CJK) text.
            let hi = floor_boundary(text, (end + NEWLINE_LOOKAHEAD_BYTES).min(n));
            if let Some(nl) = text[end..hi].find('\n') {
                end = floor_boundary(text, end + nl + 1);
            }
        }
        if end <= start {
            end = target.max(start + 1);
        }
        chunks.push(Chunk {
            text: text[start..end].to_string(),
            byte_offset: start,
        });
        if end >= n {
            break;
        }
        // Step forward by (window − overlap), char-boundary-aligned, always > 0.
        let step_back = floor_boundary(text, end.saturating_sub(overlap));
        start = step_back.max(start + 1);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_text("", 100, 10).is_empty());
        assert!(chunk_text("abc", 0, 0).is_empty());
    }

    #[test]
    fn covers_all_text_in_order() {
        let text = "line one\nline two\nline three\nline four\n";
        let chunks = chunk_text(text, 12, 4);
        assert!(!chunks.is_empty());
        // Offsets are non-decreasing and within bounds.
        for w in chunks.windows(2) {
            assert!(w[0].byte_offset < w[1].byte_offset, "offsets strictly increase");
        }
        // Every byte of the source appears in at least one chunk (coverage).
        let mut covered = vec![false; text.len()];
        for c in &chunks {
            for slot in covered.iter_mut().skip(c.byte_offset).take(c.text.len()) {
                *slot = true;
            }
        }
        assert!(covered.iter().all(|&b| b), "all source bytes covered");
    }

    #[test]
    fn chunks_are_valid_utf8_on_multibyte() {
        // Greek letters are 2 bytes each; chunking must not split them.
        let text = "α β γ δ ε\nζ η θ ι κ\nλ μ ν ξ ο\n".repeat(4);
        for c in chunk_text(&text, 10, 3) {
            assert!(std::str::from_utf8(c.text.as_bytes()).is_ok());
        }
    }

    #[test]
    fn overlap_is_clamped_below_window() {
        // overlap >= chunk_chars must still make progress (no infinite loop).
        let text = "aaaa\nbbbb\ncccc\ndddd\n";
        let chunks = chunk_text(text, 5, 999);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].byte_offset, 0);
    }

    #[test]
    fn cjk_without_newline_in_lookahead_does_not_panic() {
        // 3-byte CJK chars, no '\n' anywhere: the newline-lookahead slice's upper
        // bound `(end + 400).min(n)` lands mid-codepoint and panics unless clamped
        // to a char boundary. Regression for the chunk.rs char-boundary bug.
        let text = "中".repeat(5000); // 15_000 bytes, no newlines
        let chunks = chunk_text(&text, 6, 2);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(std::str::from_utf8(c.text.as_bytes()).is_ok());
        }
    }

    #[test]
    fn output_bounded_on_no_newline_adversarial_overlap() {
        // overlap ~= chunk_chars would (unclamped) make the step ~1 char and blow
        // output up to O(n · chunk_chars). Clamped to chunk_chars/2, total output
        // stays near ~2x the input.
        let n = 100_000;
        let text = "a".repeat(n); // no newlines, no extension
        let chunks = chunk_text(&text, 4000, 3999);
        let total: usize = chunks.iter().map(|c| c.text.len()).sum();
        assert!(total <= 3 * n, "output {total} bytes must stay near ~2x input ({n})");
    }
}
