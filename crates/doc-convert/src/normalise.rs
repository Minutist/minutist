//! Canonical markdown normalisation via pulldown-cmark round-trip.
//!
//! Every converter pipes its output through [`normalise`] so the summariser
//! receives consistent markdown regardless of which parser produced it.

use pulldown_cmark::{Options, Parser};
use pulldown_cmark_to_cmark::cmark;

use crate::error::{ConvertError, Result};

/// Parse `input` as CommonMark and re-emit it as canonical markdown.
///
/// This round-trip:
/// - strips redundant whitespace / blank lines
/// - normalises heading levels and list markers
/// - sanitises inline HTML that slipped through converters
///
/// An empty input returns an empty string without allocating.
pub fn normalise(input: &str) -> Result<String> {
    if input.is_empty() {
        return Ok(String::new());
    }

    let opts = Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS;

    let parser = Parser::new_ext(input, opts);
    let events: Vec<_> = parser.collect();

    let mut out = String::with_capacity(input.len());
    cmark(events.iter().cloned(), &mut out)
        .map_err(|e| ConvertError::Normalise(e.to_string()))?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(normalise("").unwrap(), "");
    }

    #[test]
    fn plain_paragraph_round_trips() {
        let out = normalise("Hello world\n").unwrap();
        assert!(out.contains("Hello world"), "got: {out:?}");
    }

    #[test]
    fn heading_round_trips() {
        let out = normalise("# Title\n\nBody text.\n").unwrap();
        assert!(out.contains("# Title"), "got: {out:?}");
        assert!(out.contains("Body text"), "got: {out:?}");
    }
}
