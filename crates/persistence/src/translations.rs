//! Translations sidecar — `translations.json`.
//!
//! Holds per-language translations of transcript segments, indexed by segment
//! position (zero-based) and target language name (full English name, e.g.
//! `"Spanish"`). The file is a derived view: the verbatim `transcript.json`
//! remains authoritative; translations are generated post-hoc and cleared
//! automatically when `write_transcript` replaces the segment array (since
//! segment indices shift on a full retranscription).
//!
//! # Invariant
//!
//! `write_transcript` MUST clear `translations.json` for the same meeting
//! directory. That call-site invariant is documented on `write_transcript` and
//! enforced there, so callers that replace the transcript (orchestrator
//! `finalise_retranscribe`, user-triggered re-transcribe) do not need separate
//! cleanup calls. Re-diarize does NOT clear translations (only `speaker_id`s
//! change; segment indices and text are unchanged).
//!
//! # File format
//!
//! JSON object: `{ "<language>": { "<index>": "<translated_text>", … }, … }`.
//! Written atomically (tmp + fsync + rename). Absent file is treated as an
//! empty map (no translations yet).
//!
//! # Merge semantics
//!
//! `merge_translations` adds or overwrites entries for one language without
//! touching other languages, supporting incremental progress: each translated
//! segment is persisted as it arrives, so partial progress survives an
//! interruption.

use std::collections::HashMap;
use std::path::Path;

use minutist_common::AppResult;

use crate::error::Error;

/// The on-disk shape: outer key is language name, inner key is segment index
/// (as a decimal string, since JSON object keys are always strings).
type TranslationsMap = HashMap<String, HashMap<String, String>>;

/// Path to `translations.json` inside a meeting folder.
pub fn translations_path(meeting_dir: &Path) -> std::path::PathBuf {
    meeting_dir.join("translations.json")
}

/// Read `translations.json` from `meeting_dir`.
///
/// Returns a `HashMap<language, HashMap<segment_index, text>>`.
/// An absent file returns an empty map (no translations yet); a corrupt file
/// returns an I/O error.
pub fn read_translations(
    meeting_dir: &Path,
) -> AppResult<HashMap<String, HashMap<usize, String>>> {
    let path = translations_path(meeting_dir);
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let raw: TranslationsMap =
                serde_json::from_str(&s).map_err(Error::Serialise)?;
            // Convert string keys back to `usize` indices.
            let converted = raw
                .into_iter()
                .map(|(lang, inner)| {
                    let parsed: HashMap<usize, String> = inner
                        .into_iter()
                        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v)))
                        .collect();
                    (lang, parsed)
                })
                .collect();
            Ok(converted)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(HashMap::new()),
        Err(e) => Err(Error::Io(e).into()),
    }
}

/// Merge translated segments for `language` into `translations.json`.
///
/// Reads the existing sidecar (absent = empty), merges/overwrites entries for
/// `language` only (other languages are untouched), and writes the result
/// atomically. The caller is responsible for flush cadence; batching multiple
/// segments per call reduces I/O on long meetings while still allowing the
/// caller to flush at any checkpoint so partial progress survives an
/// interruption.
pub fn merge_translations(
    meeting_dir: &Path,
    language: &str,
    translations: &HashMap<usize, String>,
) -> AppResult<()> {
    let path = translations_path(meeting_dir);

    // Read existing map (absent = empty).
    let mut raw: TranslationsMap = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).map_err(Error::Serialise)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
        Err(e) => return Err(Error::Io(e).into()),
    };

    // Merge new entries for this language (usize keys → string keys).
    let lang_map = raw.entry(language.to_string()).or_default();
    for (idx, text) in translations {
        lang_map.insert(idx.to_string(), text.clone());
    }

    // Atomic write.
    let json = serde_json::to_vec_pretty(&raw).map_err(Error::Serialise)?;
    minutist_common::fs::write_atomic(&path, &json)?;

    tracing::debug!(
        target: "persistence",
        path = %path.display(),
        language,
        count = translations.len(),
        "translations.json updated"
    );

    Ok(())
}

/// Remove `translations.json` from `meeting_dir`.
///
/// Called by `write_transcript` to enforce the invariant that a full
/// retranscription (which renumbers segment indices) does not leave stale
/// translations pointing at the wrong segments. An absent file is treated as
/// already-cleared (idempotent).
pub fn clear_translations(meeting_dir: &Path) -> AppResult<()> {
    let path = translations_path(meeting_dir);
    match std::fs::remove_file(&path) {
        Ok(()) => {
            tracing::debug!(
                target: "persistence",
                path = %path.display(),
                "translations.json cleared (transcript replaced)"
            );
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::Io(e).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmpdir() -> TempDir {
        tempfile::TempDir::new().expect("tempdir")
    }

    #[test]
    fn absent_file_returns_empty_map() {
        let dir = tmpdir();
        let result = read_translations(dir.path()).expect("read");
        assert!(result.is_empty());
    }

    #[test]
    fn round_trip_single_language() {
        let dir = tmpdir();
        let mut entries = HashMap::new();
        entries.insert(0usize, "Hola mundo".to_string());
        entries.insert(1usize, "Esto es una prueba".to_string());

        merge_translations(dir.path(), "Spanish", &entries).expect("merge");

        let read = read_translations(dir.path()).expect("read");
        let spanish = read.get("Spanish").expect("Spanish key present");
        assert_eq!(spanish.get(&0), Some(&"Hola mundo".to_string()));
        assert_eq!(spanish.get(&1), Some(&"Esto es una prueba".to_string()));
    }

    #[test]
    fn merge_adds_to_existing_without_clobbering_other_languages() {
        let dir = tmpdir();

        // Write Spanish.
        let mut es = HashMap::new();
        es.insert(0usize, "Hola".to_string());
        merge_translations(dir.path(), "Spanish", &es).expect("merge es");

        // Write French.
        let mut fr = HashMap::new();
        fr.insert(0usize, "Bonjour".to_string());
        merge_translations(dir.path(), "French", &fr).expect("merge fr");

        let read = read_translations(dir.path()).expect("read");
        assert_eq!(read["Spanish"][&0], "Hola");
        assert_eq!(read["French"][&0], "Bonjour");
    }

    #[test]
    fn merge_overwrites_entry_within_language() {
        let dir = tmpdir();

        let mut v1 = HashMap::new();
        v1.insert(0usize, "v1".to_string());
        merge_translations(dir.path(), "Spanish", &v1).expect("first merge");

        let mut v2 = HashMap::new();
        v2.insert(0usize, "v2".to_string());
        merge_translations(dir.path(), "Spanish", &v2).expect("second merge");

        let read = read_translations(dir.path()).expect("read");
        assert_eq!(read["Spanish"][&0], "v2");
    }

    #[test]
    fn clear_removes_file() {
        let dir = tmpdir();
        let mut entries = HashMap::new();
        entries.insert(0usize, "test".to_string());
        merge_translations(dir.path(), "Spanish", &entries).expect("merge");

        assert!(translations_path(dir.path()).exists());
        clear_translations(dir.path()).expect("clear");
        assert!(!translations_path(dir.path()).exists());
    }

    #[test]
    fn clear_on_absent_file_is_noop() {
        let dir = tmpdir();
        clear_translations(dir.path()).expect("clear absent is ok");
    }

    #[test]
    fn write_transcript_clears_translations() {
        // Verify the invariant: write_transcript calls clear_translations so
        // stale translations are removed when the segment array is replaced.
        use minutist_common::Segment;

        let dir = tmpdir();
        // Seed a translations file.
        let mut entries = HashMap::new();
        entries.insert(0usize, "Hola".to_string());
        merge_translations(dir.path(), "Spanish", &entries).expect("merge");
        assert!(translations_path(dir.path()).exists());

        // write_transcript with new segments must clear it.
        let segments = vec![Segment {
            start_ms: 0,
            end_ms: 1000,
            text: "Hello world".to_string(),
            speaker_id: None,
            confidence: None,
            words: vec![],
            shared_speakers: Vec::new(),
        }];
        crate::transcript::write_transcript(dir.path(), &segments).expect("write_transcript");

        assert!(
            !translations_path(dir.path()).exists(),
            "translations.json must be cleared after write_transcript"
        );
    }
}
