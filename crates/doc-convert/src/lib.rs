//! Document-to-markdown conversion for the summariser reference-material feed.
//!
//! Owner role: **data-engineer** (byte parsing, NOT ML). Depends only on
//! `minutist-common` (AppError/AppResult at the boundary) and third-party parser
//! crates; no other workspace-component edges.
//!
//! # Public surface
//!
//! ```text
//! pub fn convert_to_markdown(bytes: &[u8], ext: &str) -> AppResult<String>
//! pub fn supported_exts() -> &'static [&'static str]
//! ```
//!
//! # Robustness contract
//!
//! Every conversion path runs inside [`std::panic::catch_unwind`]. Parser
//! panics on malformed input surface as [`AppError::InvalidInput`] rather than
//! crashing the worker (cross-cutting.md §155 — "panic in spawn_blocking must
//! surface as recoverable AppError"). Hard size limits are checked BEFORE the
//! parser sees the bytes.
//!
//! # VLM fallback seam
//!
//! A private stub [`vlm_fallback`] is the documented insertion point for a
//! future vision-model path (validated separately in `spikes/doc-vlm`). The
//! production build returns `AppError::Unsupported` from that stub; the spike's
//! result can graduate here without changing the public surface.

use std::panic::AssertUnwindSafe;

use minutist_common::{AppError, AppResult};

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

/// The file extensions this crate can convert. Lower-cased, dot-less.
pub fn supported_exts() -> &'static [&'static str] {
    &["txt", "md", "xlsx", "ods", "html", "htm", "eml", "pdf", "pptx"]
}

/// Convert `bytes` (the raw content of a file with extension `ext`) to a
/// markdown string.
///
/// `ext` must be lower-cased and dot-less (e.g. `"pdf"`, `"xlsx"`); the caller
/// (`ipc-bridge`) normalises it before handing to this function.
///
/// Returns `AppError::InvalidInput` for oversize input, pathological zip
/// archives, or an unsupported extension. Returns `AppError::Internal` when a
/// parser returns a genuine error on otherwise valid-looking input. Parser
/// panics are caught and mapped to `AppError::InvalidInput`.
///
/// Empty input is allowed and returns an empty string.
pub fn convert_to_markdown(bytes: &[u8], ext: &str) -> AppResult<String> {
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
        dispatch(bytes, ext)
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
fn dispatch(bytes: &[u8], ext: &str) -> AppResult<String> {
    match ext {
        "txt" | "md" => converters::passthrough(bytes),
        "xlsx" | "ods" => converters::spreadsheet(bytes, ext),
        "html" | "htm" => converters::html(bytes),
        "eml" => converters::eml(bytes),
        "pdf" => converters::pdf(bytes),
        "pptx" => converters::pptx(bytes),
        // Test-only hook: drive the `catch_unwind` arm in `convert_to_markdown`
        // with a real parser panic (no third-party input panics deterministically).
        #[cfg(test)]
        "__panic__" => panic!("intentional panic for catch_unwind coverage"),
        other => Err(AppError::InvalidInput {
            context: format!("unsupported extension: .{other}"),
        }),
    }
}

/// Future VLM fallback insertion point (NOT built in production).
///
/// The `spikes/doc-vlm` spike validates Gemma 4 multimodal vision as a
/// fallback for scanned PDFs (where `pdf-extract` returns near-empty text).
/// When that spike graduates, its implementation replaces this stub without
/// changing the public surface of `convert_to_markdown`.
///
/// Called internally by the `pdf` converter when the extracted text is
/// near-empty (heuristic: fewer than 100 non-whitespace chars).
fn vlm_fallback(_bytes: &[u8], ext: &str) -> AppResult<String> {
    // User-facing reason (surfaced on the attachment row as the Failed message),
    // so a scanned PDF reads clearly rather than as a generic "unsupported".
    let context = match ext {
        "pdf" => "scanned or image-only PDF — no extractable text \
                  (image-document conversion is not yet available)"
            .to_string(),
        _ => {
            format!("no text-extraction path for .{ext} (VLM fallback not built; see spikes/doc-vlm)")
        }
    };
    Err(AppError::Unsupported { context })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversize_input() {
        // One byte over the limit — must be rejected before any parser runs.
        let big = vec![0u8; MAX_INPUT_BYTES + 1];
        let err = convert_to_markdown(&big, "txt").unwrap_err();
        assert!(
            matches!(err, AppError::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn rejects_unknown_extension() {
        let err = convert_to_markdown(b"hello", "docx").unwrap_err();
        assert!(
            matches!(err, AppError::InvalidInput { .. }),
            "expected InvalidInput for unknown ext, got {err:?}"
        );
    }

    #[test]
    fn empty_input_returns_empty_string() {
        let out = convert_to_markdown(b"", "txt").expect("empty txt should succeed");
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
        let err = convert_to_markdown(b"anything", "__panic__").unwrap_err();
        assert!(
            matches!(err, AppError::InvalidInput { .. }),
            "a panicking converter must map to InvalidInput, got {err:?}"
        );
    }
}
