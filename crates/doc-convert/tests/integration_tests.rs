//! Integration tests for `doc-convert`.
//!
//! Each test loads a tiny fixture file from `tests/fixtures/` and asserts
//! basic structural properties of the converted markdown. The fixtures are
//! committed alongside the test; they are the smallest valid documents of
//! each type.

use doc_convert::{convert_to_markdown, supported_exts};
use minutist_common::AppError;

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#[test]
fn supported_exts_covers_all_converters() {
    let exts = supported_exts();
    for required in &["txt", "md", "xlsx", "ods", "html", "htm", "eml", "pdf", "pptx"] {
        assert!(
            exts.contains(required),
            "missing ext {required:?} in supported_exts()"
        );
    }
}

// ---------------------------------------------------------------------------
// txt / md passthrough
// ---------------------------------------------------------------------------

#[test]
fn txt_passthrough_preserves_content() {
    let bytes = fixture("sample.txt");
    let out = convert_to_markdown(&bytes, "txt").expect("txt conversion");
    assert!(out.contains("Plain text"), "got: {out:?}");
    assert!(out.contains("Line two"), "got: {out:?}");
}

#[test]
fn md_passthrough_preserves_heading() {
    let bytes = fixture("sample.md");
    let out = convert_to_markdown(&bytes, "md").expect("md conversion");
    assert!(out.contains("Meeting Notes"), "got: {out:?}");
    assert!(out.contains("Revenue"), "got: {out:?}");
}

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

#[test]
fn html_converter_extracts_body_text() {
    let bytes = fixture("sample.html");
    let out = convert_to_markdown(&bytes, "html").expect("html conversion");
    assert!(
        out.contains("Quarterly Report") || out.contains("Revenue"),
        "expected article content, got: {out:?}"
    );
}

#[test]
fn htm_extension_also_works() {
    // .htm must be treated identically to .html.
    let bytes = fixture("sample.html");
    let out = convert_to_markdown(&bytes, "htm").expect("htm conversion");
    assert!(!out.is_empty(), "htm output must not be empty");
}

// ---------------------------------------------------------------------------
// XLSX
// ---------------------------------------------------------------------------

#[test]
fn xlsx_converter_produces_table() {
    let bytes = fixture("sample.xlsx");
    let out = convert_to_markdown(&bytes, "xlsx").expect("xlsx conversion");
    // The output should contain a markdown table with pipe characters.
    assert!(out.contains('|'), "expected table separators, got: {out:?}");
    assert!(
        out.contains("Name") || out.contains("Revenue"),
        "expected header or data cell, got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// EML
// ---------------------------------------------------------------------------

#[test]
fn eml_converter_extracts_body() {
    let bytes = fixture("sample.eml");
    let out = convert_to_markdown(&bytes, "eml").expect("eml conversion");
    // Should contain some body text from either the HTML or plain-text part.
    assert!(
        out.contains("Milestone") || out.contains("Project"),
        "expected body text, got: {out:?}"
    );
}

#[test]
fn eml_no_body_renders_headers_without_debug_formatting() {
    // A headers-only message (no html/text body) hits the headers-summary
    // fallback. The From line must read as "Name <addr>", not the parser's
    // internal `Address` Debug shape (no "List(", "Addr {", "Some(" leakage).
    let bytes = fixture("headers-only.eml");
    let out = convert_to_markdown(&bytes, "eml").expect("eml conversion");
    assert!(out.contains("**From:**"), "expected From header, got: {out:?}");
    assert!(out.contains("Jane Doe"), "expected sender name, got: {out:?}");
    assert!(
        out.contains("jane.doe@example.com"),
        "expected sender address, got: {out:?}"
    );
    assert!(
        out.contains("Jane Doe <jane.doe@example.com>"),
        "expected 'Name <addr>' formatting, got: {out:?}"
    );
    assert!(
        out.contains("Quarterly planning sync"),
        "expected subject, got: {out:?}"
    );
    for leak in &["Addr {", "Address::", "List(", "Some(", "name:", "address:"] {
        assert!(
            !out.contains(leak),
            "Debug struct leakage {leak:?} in output: {out:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// PDF
// ---------------------------------------------------------------------------

/// Build a single-page PDF containing `text` using printpdf, returning the
/// serialised bytes. Generated at test time so no opaque binary fixture is
/// committed.
fn build_text_pdf(text: &str) -> Vec<u8> {
    use printpdf::{BuiltinFont, Mm, PdfDocument};
    let (doc, page1, layer1) = PdfDocument::new("doc-convert test", Mm(210.0), Mm(297.0), "Layer 1");
    let font = doc
        .add_builtin_font(BuiltinFont::Helvetica)
        .expect("add builtin font");
    let layer = doc.get_page(page1).get_layer(layer1);
    layer.use_text(text, 14.0, Mm(20.0), Mm(270.0), &font);
    doc.save_to_bytes().expect("serialise pdf")
}

#[test]
fn pdf_digital_text_is_extracted() {
    // A PDF with a real text layer (>= 100 non-whitespace chars) must extract
    // the text rather than reach the VLM fallback seam.
    let body = "Quarterly revenue increased by twelve percent across all regions, \
                and the board approved the proposed expansion plan for next year.";
    let bytes = build_text_pdf(body);
    let out = convert_to_markdown(&bytes, "pdf").expect("digital-text pdf conversion");
    assert!(
        out.contains("Quarterly revenue") || out.contains("expansion plan"),
        "expected extracted PDF text, got: {out:?}"
    );
}

#[test]
fn pdf_near_empty_reaches_vlm_fallback() {
    // A PDF whose extractable text is under the 100 non-whitespace-char
    // threshold (here: a near-empty page) reaches the VLM fallback seam, which
    // in the production build returns AppError::Unsupported.
    let bytes = build_text_pdf("ok");
    let err = convert_to_markdown(&bytes, "pdf").unwrap_err();
    assert!(
        matches!(err, AppError::Unsupported { .. }),
        "expected Unsupported from the VLM fallback seam, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// ODS
// ---------------------------------------------------------------------------

/// Build a minimal valid OpenDocument Spreadsheet (`.ods`) in memory: a STORED
/// `mimetype` entry followed by a `content.xml` with one sheet of text cells.
/// Generated at test time rather than committing an opaque binary.
fn build_ods(cells: &[&[&str]]) -> Vec<u8> {
    use std::io::Cursor;
    use zip::write::FileOptions;
    use zip::{CompressionMethod, ZipWriter};

    let mut buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(&mut buf);

    // The mimetype entry must be first and STORED (uncompressed) per the ODF
    // package spec; calamine relies on it to detect the format.
    let stored: FileOptions<'_, ()> =
        FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("mimetype", stored).unwrap();
    {
        use std::io::Write;
        zip.write_all(b"application/vnd.oasis.opendocument.spreadsheet")
            .unwrap();
    }

    let mut content = String::new();
    content.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    content.push_str(
        r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">"#,
    );
    content.push_str("<office:body><office:spreadsheet>");
    content.push_str(r#"<table:table table:name="Sheet1">"#);
    for row in cells {
        content.push_str("<table:table-row>");
        for cell in *row {
            content.push_str("<table:table-cell office:value-type=\"string\"><text:p>");
            content.push_str(cell);
            content.push_str("</text:p></table:table-cell>");
        }
        content.push_str("</table:table-row>");
    }
    content.push_str("</table:table></office:spreadsheet></office:body>");
    content.push_str("</office:document-content>");

    let deflated: FileOptions<'_, ()> =
        FileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file("content.xml", deflated).unwrap();
    {
        use std::io::Write;
        zip.write_all(content.as_bytes()).unwrap();
    }

    // calamine's ODS reader requires META-INF/manifest.xml (it inspects it for
    // password protection before reading content.xml).
    let manifest = r#"<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">
 <manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/>
 <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"#;
    zip.start_file("META-INF/manifest.xml", deflated).unwrap();
    {
        use std::io::Write;
        zip.write_all(manifest.as_bytes()).unwrap();
    }

    zip.finish().unwrap();
    buf.into_inner()
}

#[test]
fn ods_converter_produces_table() {
    let bytes = build_ods(&[
        &["Name", "Revenue"],
        &["North", "1200"],
        &["South", "950"],
    ]);
    let out = convert_to_markdown(&bytes, "ods").expect("ods conversion");
    assert!(out.contains('|'), "expected table separators, got: {out:?}");
    assert!(out.contains("Name"), "expected header cell, got: {out:?}");
    assert!(out.contains("Revenue"), "expected header cell, got: {out:?}");
    assert!(out.contains("North"), "expected data cell, got: {out:?}");
}

// ---------------------------------------------------------------------------
// PPTX
// ---------------------------------------------------------------------------

#[test]
fn pptx_converter_extracts_slide_text() {
    let bytes = fixture("sample.pptx");
    let out = convert_to_markdown(&bytes, "pptx").expect("pptx conversion");
    // The converter emits "## Slide N" headings.
    assert!(
        out.contains("## Slide") || out.contains("Results") || out.contains("Action"),
        "expected slide content, got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// Size and zip limits
// ---------------------------------------------------------------------------

#[test]
fn oversize_input_is_rejected() {
    // 50 MiB + 1 byte — rejected before the parser runs.
    let big = vec![0u8; 50 * 1024 * 1024 + 1];
    let err = convert_to_markdown(&big, "txt").unwrap_err();
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn zip_bomb_entry_count_rejected() {
    // Build a zip with more entries than MAX_ZIP_ENTRIES (10_000) by
    // constructing a synthetic zip with a fabricated entry-count header.
    // The simplest way: use the `zip` crate to write exactly MAX+1 tiny entries.
    use std::io::Cursor;
    use zip::write::FileOptions;
    use zip::ZipWriter;

    let mut buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(&mut buf);
    let opts: FileOptions<'_, ()> = FileOptions::default();

    // Write enough entries to trip the count limit (10_001).
    for i in 0..10_001usize {
        let name = format!("ppt/slides/slide{i}.xml");
        zip.start_file(&name, opts).unwrap();
    }
    zip.finish().unwrap();
    let bytes = buf.into_inner();

    let err = convert_to_markdown(&bytes, "pptx").unwrap_err();
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput for entry-count bomb, got {err:?}"
    );
}

#[test]
fn malformed_pptx_zip_is_rejected() {
    // Garbage bytes that are not a valid zip archive.
    let garbage = b"this is not a zip file at all!!!";
    let err = convert_to_markdown(garbage, "pptx").unwrap_err();
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput for malformed zip, got {err:?}"
    );
}

#[test]
fn unknown_extension_is_rejected() {
    let err = convert_to_markdown(b"hello", "docx").unwrap_err();
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput for unsupported ext, got {err:?}"
    );
}

#[test]
fn empty_input_is_ok() {
    let out = convert_to_markdown(b"", "txt").expect("empty txt");
    assert_eq!(out, "");
}
