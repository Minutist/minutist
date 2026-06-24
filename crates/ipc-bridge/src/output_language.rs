//! Output-language resolution for generated text (summaries and chat replies).
//!
//! The `Settings::output_language` field is a `String` with two forms:
//!
//! - The sentinel `"auto"` — resolve from the host system locale via
//!   [`sys_locale::get_locale`], extract the primary BCP-47 subtag, and map it
//!   through [`SUBTAG_TO_LANGUAGE`] to a full English language name.
//! - Any other non-empty string — treated as a full English language name and
//!   passed through verbatim (the UI constrains this to names in `OUTPUT_LANGUAGES`).
//!
//! [`resolve_output_language`] is the single call site; callers append the
//! returned name to the system prompt as `"\n\nRespond entirely in {lang}."`.
//! A `None` return means "no instruction" — the LLM is left to choose its
//! output language naturally.

/// Maps BCP-47 primary language subtags to full English language names.
///
/// Covers the 15 languages the output-language picker exposes. Subtags are
/// lowercase ASCII (BCP-47 §2.2.1); entries sorted by subtag.
static SUBTAG_TO_LANGUAGE: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("de", "German"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

/// Resolve the output-language setting to a full English language name, or
/// `None` when no language instruction should be appended.
///
/// - `"auto"` (case-insensitive): queries the host system locale via
///   [`sys_locale::get_locale`], extracts the primary BCP-47 subtag (the
///   portion before the first `-` or `_`), and maps it through
///   [`SUBTAG_TO_LANGUAGE`]. Returns `None` when the locale is unavailable or
///   the subtag is not in the table — the LLM is left to choose naturally.
/// - Empty or whitespace-only: returns `None` (no instruction).
/// - Any other value: returned verbatim (trimmed). The UI constrains these to
///   full English names from `OUTPUT_LANGUAGES`, but an arbitrary name
///   passes through so the user can set a language outside the picker list by
///   editing `settings.store` directly.
pub fn resolve_output_language(setting: &str) -> Option<String> {
    let trimmed = setting.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.eq_ignore_ascii_case("auto") {
        let locale = sys_locale::get_locale()?;
        let subtag = primary_subtag(&locale);
        return subtag_to_language(subtag).map(str::to_string);
    }
    Some(trimmed.to_string())
}

/// Extract the primary BCP-47 language subtag (before the first `-` or `_`),
/// lowercased.
fn primary_subtag(locale: &str) -> &str {
    let end = locale
        .find(['-', '_'])
        .unwrap_or(locale.len());
    &locale[..end]
}

/// Look up a primary BCP-47 subtag in the static table; case-insensitive.
fn subtag_to_language(subtag: &str) -> Option<&'static str> {
    SUBTAG_TO_LANGUAGE
        .iter()
        .find(|(s, _)| s.eq_ignore_ascii_case(subtag))
        .map(|(_, lang)| *lang)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // subtag extraction
    // -----------------------------------------------------------------------

    #[test]
    fn primary_subtag_plain() {
        assert_eq!(primary_subtag("en"), "en");
    }

    #[test]
    fn primary_subtag_with_hyphen() {
        assert_eq!(primary_subtag("en-AU"), "en");
    }

    #[test]
    fn primary_subtag_with_underscore() {
        assert_eq!(primary_subtag("en_AU"), "en");
    }

    #[test]
    fn primary_subtag_complex_bcp47() {
        assert_eq!(primary_subtag("zh-Hans-CN"), "zh");
    }

    // -----------------------------------------------------------------------
    // subtag → language mapping
    // -----------------------------------------------------------------------

    #[test]
    fn known_subtags_map_to_expected_names() {
        let cases = [
            ("en", "English"),
            ("zh", "Chinese"),
            ("es", "Spanish"),
            ("fr", "French"),
            ("de", "German"),
            ("it", "Italian"),
            ("pt", "Portuguese"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
            ("ru", "Russian"),
            ("nl", "Dutch"),
            ("ar", "Arabic"),
            ("hi", "Hindi"),
            ("pl", "Polish"),
            ("tr", "Turkish"),
        ];
        for (subtag, expected) in cases {
            assert_eq!(
                subtag_to_language(subtag),
                Some(expected),
                "subtag {subtag:?} should map to {expected:?}"
            );
        }
    }

    #[test]
    fn unknown_subtag_returns_none() {
        assert_eq!(subtag_to_language("xx"), None);
        assert_eq!(subtag_to_language("tlh"), None); // Klingon
    }

    // -----------------------------------------------------------------------
    // resolve_output_language
    // -----------------------------------------------------------------------

    #[test]
    fn explicit_language_name_passes_through_verbatim() {
        assert_eq!(
            resolve_output_language("French"),
            Some("French".to_string())
        );
        assert_eq!(
            resolve_output_language("German"),
            Some("German".to_string())
        );
    }

    #[test]
    fn explicit_name_is_trimmed() {
        assert_eq!(
            resolve_output_language("  Japanese  "),
            Some("Japanese".to_string())
        );
    }

    #[test]
    fn empty_setting_returns_none() {
        assert_eq!(resolve_output_language(""), None);
        assert_eq!(resolve_output_language("   "), None);
    }

    #[test]
    fn en_au_locale_tag_resolves_to_english() {
        // Simulate the "auto" path with a known locale by testing the
        // internal helpers directly (the auto path calls sys_locale which
        // we cannot mock without a shim).
        let subtag = primary_subtag("en-AU");
        assert_eq!(subtag, "en");
        assert_eq!(subtag_to_language(subtag), Some("English"));
    }

    #[test]
    fn zh_hans_cn_locale_tag_resolves_to_chinese() {
        let subtag = primary_subtag("zh-Hans-CN");
        assert_eq!(subtag, "zh");
        assert_eq!(subtag_to_language(subtag), Some("Chinese"));
    }

    #[test]
    fn unknown_subtag_from_locale_returns_none() {
        // A locale whose primary subtag is not in the table.
        let subtag = primary_subtag("xx-Latn");
        assert_eq!(subtag_to_language(subtag), None);
    }
}
