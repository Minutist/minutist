//! Document-to-markdown conversion for the summariser reference-material feed.
//!
//! Owner role: **data-engineer** (byte parsing, NOT ML). Depends only on
//! `minutist-common` (AppError/AppResult at the boundary, and the [`DocVlm`]
//! trait it calls through) and third-party parser crates — no other
//! workspace-component edge. `image` (decode an image attachment and
//! re-encode it to PNG for the VLM OCR path) is likewise a third-party dep,
//! not a workspace edge, so the `common`-only rule holds.
//!
//! # Public surface
//!
//! ```text
//! pub fn convert_to_markdown(bytes: &[u8], ext: &str, vlm: Option<&dyn DocVlm>) -> AppResult<String>
//! pub fn supported_exts() -> &'static [&'static str]
//! ```
//!
//! [`dispatch`] routes each extension to one [`converters`] function, one
//! match arm per format:
//!
//! | Extension | Converter |
//! |---|---|
//! | `txt`, `md` | passthrough (UTF-8 lossy, then markdown normalisation) |
//! | `csv`, `tsv` | `csv` crate → a markdown pipe-table (first row = header) |
//! | `json`, `yaml`/`yml`, `xml`, `log` | wrapped verbatim in a fenced code block — these are not markdown, so re-emitting them through `pulldown-cmark` would mangle them |
//! | `xlsx`, `ods` | `calamine` → a markdown pipe-table per sheet; date cells render as ISO via `ExcelDateTime::as_datetime()`, not the raw serial number |
//! | `html`, `htm` | `dom_smoothie` readability extraction → `htmd` markdown |
//! | `eml` | `mail-parser` → HTML body through the `html` path, or a plain-text body passthrough |
//! | `pdf` | `pdf_oxide` (pure-Rust digital-text extraction, no native lib; a page that fails to decode is skipped, not fatal) |
//! | `pptx` | `zip` + `quick-xml` walk of `ppt/slides/slideN.xml` `<a:t>` runs (incl. table cells), one `## Slide N` per slide, with per-slide speaker notes appended as a `### Notes` block |
//! | `docx` | `zip` + `quick-xml` walk of `word/document.xml` (`<w:p>`/`<w:t>`/`<w:tbl>`); no `docx-rs` production dependency (it is a dev-dependency only, used to synthesise the DOCX test fixture) |
//! | `png`, `jpg`, `jpeg`, `tiff` | VLM OCR only — no pure-Rust text path |
//!
//! For `pptx`/`docx` the bar is textual content for the summariser, not
//! faithful structure: paragraph/list/cell text is captured; numbering
//! glyphs and exact layout are not reconstructed. Every converter's output is
//! normalised through `pulldown-cmark` (parse → re-emit) before it is
//! returned, so the stored markdown is canonical.
//!
//! Integration tests (`tests/integration_tests.rs`) drive every converter
//! against a small committed fixture file per format under `tests/fixtures/`
//! — the smallest valid document of each type — asserting structural
//! properties of the converted markdown rather than exact byte output.
//!
//! # Robustness contract
//!
//! Every conversion path runs inside [`std::panic::catch_unwind`]. Parser
//! panics on malformed input surface as [`AppError::InvalidInput`] rather than
//! crashing the worker (cross-cutting.md §155 — "panic in spawn_blocking must
//! surface as recoverable AppError"). Hard size limits are checked BEFORE the
//! parser sees the bytes.
//!
//! # VLM OCR for image attachments
//!
//! Direct image attachments (`png`/`jpg`/`jpeg`/`tiff`) have no pure-Rust text
//! path, so they convert through the injected [`minutist_common::DocVlm`]
//! (vision OCR). When `vlm` is `None` they return [`AppError::Unsupported`];
//! every digital-text path ignores it. Scanned / image-only PDFs are not
//! handled here — they need PDF-page rasterisation, tracked in planning issue
//! 0019; a near-empty PDF extraction returns [`AppError::Unsupported`].

use std::panic::AssertUnwindSafe;

use minutist_common::{AppError, AppResult, DocVlm};

mod error;
mod normalise;
mod converters;

pub use error::ConvertError;

/// Maximum input size (50 MiB). Inputs above this are rejected before any
/// parser sees the bytes.
///
/// Exposed so callers (e.g. `ipc-bridge`) can reject oversize input at the IPC
/// boundary before reading the whole file into memory, sharing the single cap
/// rather than duplicating the constant.
pub const MAX_INPUT_BYTES: usize = 50 * 1024 * 1024;

/// Zip-decompression bomb guard: cumulative uncompressed bytes across all
/// entries in a single archive.
const MAX_ZIP_UNCOMPRESSED_BYTES: u64 = 500 * 1024 * 1024; // 500 MiB

/// Maximum number of zip entries to process (prevents zip-entry-count bombs).
const MAX_ZIP_ENTRIES: usize = 10_000;

/// Raster-image decode-bomb guard: the maximum pixel-buffer allocation
/// [`converters::image`]'s decode permits. Mirrors [`MAX_ZIP_UNCOMPRESSED_BYTES`]'s
/// reasoning — [`MAX_INPUT_BYTES`] bounds only the COMPRESSED input, so a small
/// file declaring huge dimensions could otherwise force an unbounded decode
/// allocation before this limit is checked.
const MAX_IMAGE_DECODED_BYTES: u64 = 500 * 1024 * 1024; // 500 MiB

/// The file extensions this crate can convert. Lower-cased, dot-less.
///
/// The image extensions (`png`/`jpg`/`jpeg`/`tiff`) have no pure-Rust text
/// path: they convert ONLY when a [`DocVlm`] is injected into
/// [`convert_to_markdown`]; without one they return `AppError::Unsupported`.
pub fn supported_exts() -> &'static [&'static str] {
    &[
        "txt", "md", "csv", "tsv", "json", "yaml", "yml", "xml", "log", "xlsx", "ods", "html",
        "htm", "eml", "pdf", "pptx", "docx", "png", "jpg", "jpeg", "tiff",
    ]
}

/// Convert `bytes` (the raw content of a file with extension `ext`) to a
/// markdown string.
///
/// `ext` must be lower-cased and dot-less (e.g. `"pdf"`, `"xlsx"`); the caller
/// (`ipc-bridge`) normalises it before handing to this function.
///
/// `ext` is a ROUTING HINT only — it selects the converter but is NOT a
/// validated assertion about `bytes`. The bytes may not match the extension
/// (the caller passes them separately and does not sniff content), so every
/// converter must self-defend on its own input rather than trusting `ext` to
/// imply a safe shape. In particular the zip-container converters
/// (`xlsx`/`ods`/`pptx`/`docx`) run their own bomb guard; a payload mislabelled
/// to a non-zip extension simply reaches a converter that treats it as that
/// format (bounded by `MAX_INPUT_BYTES`), never the zip guard.
///
/// `vlm` is the optional vision-OCR backend. It is consulted only for direct
/// image attachments (`png`/`jpg`/`jpeg`/`tiff`), which have no pure-Rust text
/// path. When `None`, those paths return `AppError::Unsupported`; all
/// digital-text paths ignore it.
///
/// Returns `AppError::InvalidInput` for oversize input, pathological zip
/// archives, an unreadable/corrupt PDF, or an unsupported extension. Returns
/// `AppError::Internal` when a parser returns a genuine error on otherwise
/// valid-looking input. Parser panics are caught and mapped to
/// `AppError::InvalidInput`.
///
/// Empty input is allowed and returns an empty string.
pub fn convert_to_markdown(
    bytes: &[u8],
    ext: &str,
    vlm: Option<&dyn DocVlm>,
) -> AppResult<String> {
    // Hard size guard — checked before any parser is invoked.
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(AppError::InvalidInput {
            context: format!(
                "input too large: {} bytes (max {} MiB)",
                bytes.len(),
                MAX_INPUT_BYTES / 1024 / 1024,
            ),
        });
    }

    // Wrap in catch_unwind so a panicking parser cannot crash the conversion
    // worker. `bytes` and `ext` are both `UnwindSafe`; the closure only reads
    // them.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        dispatch(bytes, ext, vlm)
    }));

    match result {
        Ok(inner) => inner,
        Err(_panic) => {
            tracing::warn!(
                target: "doc-convert",
                ext = ext,
                bytes = bytes.len(),
                "converter panicked; treating as invalid input"
            );
            Err(AppError::InvalidInput {
                context: format!("conversion panicked for .{ext}"),
            })
        }
    }
}

/// Route to the per-extension converter.
fn dispatch(bytes: &[u8], ext: &str, vlm: Option<&dyn DocVlm>) -> AppResult<String> {
    match ext {
        "txt" | "md" => converters::passthrough(bytes),
        "csv" => converters::csv(bytes),
        "tsv" => converters::tsv(bytes),
        "json" => converters::fenced_text(bytes, "json"),
        "yaml" | "yml" => converters::fenced_text(bytes, "yaml"),
        "xml" => converters::fenced_text(bytes, "xml"),
        "log" => converters::fenced_text(bytes, ""),
        "xlsx" | "ods" => converters::spreadsheet(bytes, ext),
        "html" | "htm" => converters::html(bytes),
        "eml" => converters::eml(bytes),
        "pdf" => converters::pdf(bytes),
        "pptx" => converters::pptx(bytes),
        "docx" => converters::docx(bytes),
        "png" | "jpg" | "jpeg" | "tiff" => converters::image(bytes, ext, vlm),
        // Test-only hook: drive the `catch_unwind` arm in `convert_to_markdown`
        // with a real parser panic (no third-party input panics deterministically).
        #[cfg(test)]
        "__panic__" => panic!("intentional panic for catch_unwind coverage"),
        other => Err(AppError::InvalidInput {
            context: format!("unsupported extension: .{other}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversize_input() {
        // One byte over the limit — must be rejected before any parser runs.
        let big = vec![0u8; MAX_INPUT_BYTES + 1];
        let err = convert_to_markdown(&big, "txt", None).unwrap_err();
        assert!(
            matches!(err, AppError::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn rejects_unknown_extension() {
        let err = convert_to_markdown(b"hello", "rtf", None).unwrap_err();
        assert!(
            matches!(err, AppError::InvalidInput { .. }),
            "expected InvalidInput for unknown ext, got {err:?}"
        );
    }

    #[test]
    fn empty_input_returns_empty_string() {
        let out = convert_to_markdown(b"", "txt", None).expect("empty txt should succeed");
        assert_eq!(out, "");
    }

    #[test]
    fn supported_exts_are_lowercase_dotless() {
        for ext in supported_exts() {
            assert_eq!(*ext, ext.to_lowercase(), "ext {ext:?} must be lowercase");
            assert!(!ext.starts_with('.'), "ext {ext:?} must not have a leading dot");
        }
    }

    #[test]
    fn converter_panic_is_caught_as_invalid_input() {
        // The test-only `__panic__` ext panics inside `dispatch`; the
        // `catch_unwind` wrapper must convert that to InvalidInput rather than
        // unwind into the caller — the conversion worker must survive a malformed
        // input that panics a parser.
        let err = convert_to_markdown(b"anything", "__panic__", None).unwrap_err();
        assert!(
            matches!(err, AppError::InvalidInput { .. }),
            "a panicking converter must map to InvalidInput, got {err:?}"
        );
    }
}
