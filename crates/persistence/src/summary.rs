//! `summary.md` write/read helpers.
//!
//! Phase 5's `summariser` produces the markdown summary; Phase 4 lands only
//! the [`crate::MeetingFolder::summary_path`] helper and these I/O functions so
//! the path and storage seam exist before Phase 5 wires the producer.
//!
//! Writes are **atomic** (write to a sibling `*.tmp` in the same dir, fsync,
//! rename into place), matching [`crate::NotesStore`]'s durability story so a
//! crash mid-write never leaves a truncated `summary.md`.

use std::io::Write;
use std::path::Path;

use minutist_common::AppResult;

use crate::error::Error;

/// Atomically write `summary_md` to `summary.md` inside the existing meeting
/// folder.
///
/// Writes to a sibling `summary.md.tmp` then renames into place. Does **not**
/// create the meeting folder — the folder is owned by [`crate::MeetingWriter`]
/// and is expected to exist.
pub fn write_summary(meeting_dir: &Path, summary_md: &str) -> AppResult<()> {
    let path = meeting_dir.join("summary.md");
    let tmp_path = meeting_dir.join("summary.md.tmp");

    let write_result = (|| -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(summary_md.as_bytes())?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }

    if let Err(e) = std::fs::rename(&tmp_path, &path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(Error::Io(e).into());
    }

    tracing::debug!(
        target: "persistence",
        folder = %meeting_dir.display(),
        "summary.md written"
    );

    Ok(())
}

/// Read `summary.md` from the meeting folder.
///
/// Returns `Ok(None)` when `summary.md` does not exist (a meeting that has not
/// been summarised yet).
pub fn read_summary(meeting_dir: &Path) -> AppResult<Option<String>> {
    let path = meeting_dir.join("summary.md");
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Io(e).into()),
    }
}

/// Max length (chars) of a meeting-list excerpt blurb derived from a summary
/// (live-test UX T6). ~140 chars fits a single list row.
const BLURB_MAX_CHARS: usize = 140;

/// Whether a trimmed summary line is structural markdown (heading / list item /
/// blockquote) rather than the opening prose sentence we want for the blurb.
///
/// A list marker is the marker FOLLOWED BY A SPACE (`- `, `* `, `+ `, `1. `), so
/// a line that merely STARTS with `*` for inline bold/emphasis (e.g.
/// `**Decision:** …`) is treated as prose, not a bullet.
fn is_skippable_summary_line(line: &str) -> bool {
    // ATX headings.
    if line.starts_with('#') {
        return true;
    }
    // Blockquote.
    if line.starts_with('>') {
        return true;
    }
    // Unordered list item: `- `, `* `, or `+ ` (marker + space). A bare `**`
    // (bold) does NOT match because the char after `*` is not a space.
    let bytes = line.as_bytes();
    if matches!(bytes.first(), Some(b'-' | b'*' | b'+')) && bytes.get(1) == Some(&b' ') {
        return true;
    }
    // Ordered list item: `<digits>. ` or `<digits>) `.
    let after_digits = line.trim_start_matches(|c: char| c.is_ascii_digit());
    if after_digits.len() < line.len()
        && (after_digits.starts_with(". ") || after_digits.starts_with(") "))
    {
        return true;
    }
    false
}

/// Derive a one-line blurb from a `summary.md` for the meeting-list excerpt
/// (live-test UX T6).
///
/// The summary is markdown with a leading `# …` heading and section headings;
/// the blurb is the FIRST non-heading, non-empty, non-list-marker line — the
/// summary's opening overview sentence — trimmed of inline markdown emphasis and
/// truncated to ~140 chars (on a char boundary, ellipsis appended when cut).
/// Returns `None` when the summary has no usable prose line (e.g. headings only),
/// so the caller falls back to the first transcript segment.
///
/// Pure + unit-tested so the derivation is verified without a meeting folder.
pub fn summary_blurb(summary_md: &str) -> Option<String> {
    let line = summary_md
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !is_skippable_summary_line(l))?;

    // Strip the most common inline markdown emphasis so the blurb reads as plain
    // prose (the list row is plain text). Bold/italic/code markers only.
    let cleaned: String = line
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '`'))
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return None;
    }

    if cleaned.chars().count() <= BLURB_MAX_CHARS {
        Some(cleaned.to_string())
    } else {
        let truncated: String = cleaned.chars().take(BLURB_MAX_CHARS).collect();
        Some(format!("{}…", truncated.trim_end()))
    }
}

#[cfg(test)]
mod tests {
    use super::summary_blurb;

    #[test]
    fn blurb_skips_headings_and_uses_first_prose_line() {
        let md = "# Meeting summary\n\n## Overview\n\nThe team agreed to ship the \
                  beta on Friday and assigned the rollout tasks.\n\n## Action items\n\
                  - Alice: docs\n";
        assert_eq!(
            summary_blurb(md).as_deref(),
            Some("The team agreed to ship the beta on Friday and assigned the rollout tasks.")
        );
    }

    #[test]
    fn blurb_strips_inline_emphasis() {
        let md = "# S\n\n**Decision:** ship _now_ with the `vulkan` build.\n";
        assert_eq!(
            summary_blurb(md).as_deref(),
            Some("Decision: ship now with the vulkan build.")
        );
    }

    #[test]
    fn blurb_truncates_long_line_on_char_boundary() {
        let long = "word ".repeat(60); // 300 chars
        let md = format!("# Title\n\n{long}\n");
        let blurb = summary_blurb(&md).expect("a prose line exists");
        assert!(blurb.ends_with('…'), "long blurb must be ellipsised");
        // 140 kept chars + the ellipsis (a single char).
        assert!(
            blurb.chars().count() <= 141,
            "blurb must be bounded to ~140 chars, got {}",
            blurb.chars().count()
        );
    }

    #[test]
    fn blurb_none_when_only_headings() {
        let md = "# Title\n\n## Section\n\n### Sub\n";
        assert_eq!(summary_blurb(md), None);
    }

    #[test]
    fn blurb_none_when_empty() {
        assert_eq!(summary_blurb(""), None);
        assert_eq!(summary_blurb("   \n\n  "), None);
    }
}
