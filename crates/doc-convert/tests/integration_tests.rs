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
    for required in &["txt", "md", "xlsx", "ods", "html", "htm", "eml", "pdf", "pptx", "docx"] {
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
    let out = convert_to_markdown(&bytes, "txt", None).expect("txt conversion");
    assert!(out.contains("Plain text"), "got: {out:?}");
    assert!(out.contains("Line two"), "got: {out:?}");
}

#[test]
fn md_passthrough_preserves_heading() {
    let bytes = fixture("sample.md");
    let out = convert_to_markdown(&bytes, "md", None).expect("md conversion");
    assert!(out.contains("Meeting Notes"), "got: {out:?}");
    assert!(out.contains("Revenue"), "got: {out:?}");
}

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

#[test]
fn html_converter_extracts_body_text() {
    let bytes = fixture("sample.html");
    let out = convert_to_markdown(&bytes, "html", None).expect("html conversion");
    assert!(
        out.contains("Quarterly Report") || out.contains("Revenue"),
        "expected article content, got: {out:?}"
    );
}

#[test]
fn htm_extension_also_works() {
    // .htm must be treated identically to .html.
    let bytes = fixture("sample.html");
    let out = convert_to_markdown(&bytes, "htm", None).expect("htm conversion");
    assert!(!out.is_empty(), "htm output must not be empty");
}

// ---------------------------------------------------------------------------
// XLSX
// ---------------------------------------------------------------------------

#[test]
fn xlsx_converter_produces_table() {
    let bytes = fixture("sample.xlsx");
    let out = convert_to_markdown(&bytes, "xlsx", None).expect("xlsx conversion");
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
    let out = convert_to_markdown(&bytes, "eml", None).expect("eml conversion");
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
    let out = convert_to_markdown(&bytes, "eml", None).expect("eml conversion");
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

/// Build a single-page PDF laid out as two side-by-side text columns, each a
/// vertical stack of the given words. Generated at test time via printpdf.
fn build_two_column_pdf(left: &[&str], right: &[&str]) -> Vec<u8> {
    use printpdf::{BuiltinFont, Mm, PdfDocument};
    let (doc, page1, layer1) =
        PdfDocument::new("doc-convert test", Mm(210.0), Mm(297.0), "Layer 1");
    let font = doc
        .add_builtin_font(BuiltinFont::Helvetica)
        .expect("add builtin font");
    let layer = doc.get_page(page1).get_layer(layer1);

    // Left column at x=20mm, right column at x=120mm; both stack downward from
    // the top so the two columns share the same vertical band.
    let mut y = 270.0;
    for word in left {
        layer.use_text(*word, 14.0, Mm(20.0), Mm(y), &font);
        y -= 10.0;
    }
    let mut y = 270.0;
    for word in right {
        layer.use_text(*word, 14.0, Mm(120.0), Mm(y), &font);
        y -= 10.0;
    }
    doc.save_to_bytes().expect("serialise pdf")
}

/// Build a minimal single-page PDF whose content stream embeds an inline image
/// (`BI ... ID <raw bytes> EI`) alongside text. This is the content-stream class
/// that made the PREVIOUS engine (`pdf-extract`, via `lopdf`) panic — lopdf could
/// not skip the inline-image binary payload to `EI` and failed the whole
/// document. printpdf cannot emit inline images, so this is hand-built with a
/// correct xref (a valid PDF, not a repair case). The surrounding text is long
/// enough to clear the near-empty threshold so it takes the real extraction path.
fn build_inline_image_pdf() -> Vec<u8> {
    let text = "InlineImageProbe this digital pdf carries an inline image in its \
                content stream alongside enough surrounding text to clear the near \
                empty threshold so it converts as a real document not the fallback";
    // 2x2 RGB inline-image payload: 12 raw bytes, including a NUL, to be the
    // adversarial binary blob a naive content tokenizer walks into.
    let img: [u8; 12] = [
        0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x00,
    ];

    let mut content: Vec<u8> = Vec::new();
    content.extend_from_slice(format!("BT /F1 14 Tf 36 720 Td ({text}) Tj ET\n").as_bytes());
    content.extend_from_slice(b"q 24 0 0 24 36 600 cm\n");
    content.extend_from_slice(b"BI /W 2 /H 2 /CS /RGB /BPC 8 ID ");
    content.extend_from_slice(&img);
    content.extend_from_slice(b"\nEI\nQ\n");

    let objs: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        {
            let mut s = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
            s.extend_from_slice(&content);
            s.extend_from_slice(b"\nendstream");
            s
        },
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ];

    let mut pdf: Vec<u8> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.7\n");
    let mut offsets = vec![0usize; objs.len() + 1];
    for (i, body) in objs.iter().enumerate() {
        offsets[i + 1] = pdf.len();
        pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref_pos = pdf.len();
    let n = objs.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {n}\n").as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for off in offsets.iter().skip(1) {
        pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF").as_bytes(),
    );
    pdf
}

#[test]
fn pdf_multi_column_captures_all_text() {
    // Content-completeness bar: a two-column digital PDF must yield every word
    // from BOTH columns. Reading order across columns is NOT asserted —
    // pdf_oxide sorts spans into row bands (by Y then X), so two columns sharing
    // a vertical band are interleaved left/right rather than read each
    // top-to-bottom. This is a documented known limitation (see
    // architecture/cross-cutting.md); the summariser tolerates imperfect
    // ordering, so capturing all the text is the requirement here.
    // Use distinctive multi-syllable words; their combined non-whitespace length
    // must clear the 100-char near-empty threshold so the real extractor runs
    // rather than the VLM fallback seam.
    let left = [
        "alphabetical",
        "bravissimo",
        "charcuterie",
        "delicatessen",
        "echolocation",
        "foxgloves",
        "golfcourse",
        "hotelier",
    ];
    let right = [
        "indianapolis",
        "julienne",
        "kilometres",
        "limousine",
        "microphone",
        "novemberfest",
        "oscillator",
        "paparazzi",
    ];
    let bytes = build_two_column_pdf(&left, &right);
    let out = convert_to_markdown(&bytes, "pdf", None).expect("two-column pdf conversion");
    for word in left.iter().chain(right.iter()) {
        assert!(
            out.contains(word),
            "expected column word {word:?} in extracted text, got: {out:?}"
        );
    }
}

#[test]
fn pdf_with_inline_image_extracts_surrounding_text() {
    // Regression lock-in for the pdf-extract -> pdf_oxide swap. An inline image
    // (BI/ID <raw bytes> EI) in the content stream is the class that PANICKED the
    // previous engine (lopdf could not skip the binary payload to EI, failing the
    // whole document). pdf_oxide must extract the surrounding text without
    // panicking — a panic would unwind and fail this test.
    let bytes = build_inline_image_pdf();
    let out = convert_to_markdown(&bytes, "pdf", None).expect("inline-image pdf converts");
    assert!(
        out.contains("InlineImageProbe"),
        "expected surrounding text from a content stream with an inline image, got: {out:?}"
    );
}

#[test]
fn corrupt_pdf_is_classified_as_bad_input_not_internal() {
    // A non-PDF / corrupt attachment is bad INPUT. The pdf() open / page_count
    // failure path maps to AppError::InvalidInput (like a malformed zip and a
    // caught converter panic), never AppError::Internal (an internal-fault signal).
    let err = convert_to_markdown(b"\x89 not a pdf at all \x00\x01\x02 garbage", "pdf", None)
        .unwrap_err();
    assert!(
        !matches!(err, AppError::Internal { .. }),
        "a corrupt PDF must not classify as an Internal fault, got {err:?}"
    );
    assert!(
        matches!(err, AppError::InvalidInput { .. } | AppError::Unsupported { .. }),
        "expected InvalidInput (open failure) or Unsupported (near-empty), got {err:?}"
    );
}

#[test]
fn pdf_digital_text_is_extracted() {
    // A PDF with a real text layer (>= 100 non-whitespace chars) must extract
    // the text rather than reach the VLM fallback seam.
    let body = "Quarterly revenue increased by twelve percent across all regions, \
                and the board approved the proposed expansion plan for next year.";
    let bytes = build_text_pdf(body);
    let out = convert_to_markdown(&bytes, "pdf", None).expect("digital-text pdf conversion");
    assert!(
        out.contains("Quarterly revenue") || out.contains("expansion plan"),
        "expected extracted PDF text, got: {out:?}"
    );
}

#[test]
fn pdf_near_empty_without_vlm_is_unsupported() {
    // A PDF whose extractable text is under the 100 non-whitespace-char
    // threshold (here: a near-empty page) reaches the VLM fallback. With no VLM
    // injected, the fallback returns AppError::Unsupported rather than guessing.
    let bytes = build_text_pdf("ok");
    let err = convert_to_markdown(&bytes, "pdf", None).unwrap_err();
    assert!(
        matches!(err, AppError::Unsupported { .. }),
        "expected Unsupported from the VLM fallback with no model, got {err:?}"
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
    let out = convert_to_markdown(&bytes, "ods", None).expect("ods conversion");
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
    let out = convert_to_markdown(&bytes, "pptx", None).expect("pptx conversion");
    // The converter emits "## Slide N" headings.
    assert!(
        out.contains("## Slide") || out.contains("Results") || out.contains("Action"),
        "expected slide content, got: {out:?}"
    );
}

/// Build a minimal PPTX in memory: a `[Content_Types].xml`, one slide carrying
/// shape `<a:t>` text, and a matching `notesSlide1.xml` carrying speaker-note
/// `<a:t>` text. Constructed with the `zip` crate (no committed binary),
/// mirroring how `build_ods` synthesises an ODS package. Only the parts the
/// `pptx` converter reads (`ppt/slides/*` and `ppt/notesSlides/*`) are
/// populated; relationship and presentation parts are omitted because the
/// converter pairs slide↔notes by numeric ordinal, not via `.rels`.
fn build_pptx_with_notes(slide_text: &str, notes_text: &str) -> Vec<u8> {
    use std::io::{Cursor, Write};
    use zip::write::FileOptions;
    use zip::{CompressionMethod, ZipWriter};

    let mut buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(&mut buf);
    let opts: FileOptions<'_, ()> =
        FileOptions::default().compression_method(CompressionMethod::Deflated);

    let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="xml" ContentType="application/xml"/>
</Types>"#;
    zip.start_file("[Content_Types].xml", opts).unwrap();
    zip.write_all(content_types.as_bytes()).unwrap();

    // A slide part: one shape with a single paragraph/run.
    let slide = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{slide_text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"#
    );
    zip.start_file("ppt/slides/slide1.xml", opts).unwrap();
    zip.write_all(slide.as_bytes()).unwrap();

    // A notes-slide part carrying the speaker-note text.
    let notes = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{notes_text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:notes>"#
    );
    zip.start_file("ppt/notesSlides/notesSlide1.xml", opts).unwrap();
    zip.write_all(notes.as_bytes()).unwrap();

    zip.finish().unwrap();
    buf.into_inner()
}

#[test]
fn pptx_appends_speaker_notes_per_slide() {
    let bytes = build_pptx_with_notes("Roadmap overview", "Remember to mention the Q3 budget");
    let out = convert_to_markdown(&bytes, "pptx", None).expect("pptx conversion");

    assert!(out.contains("Roadmap overview"), "expected slide body, got: {out:?}");
    assert!(out.contains("Notes"), "expected a Notes block, got: {out:?}");
    assert!(
        out.contains("Remember to mention the Q3 budget"),
        "expected speaker-note text, got: {out:?}"
    );
}

/// Build a minimal valid DOCX in memory using the `docx-rs` writer (a DEV-only
/// dependency): a heading paragraph, a body paragraph, two bullet-list item
/// paragraphs, and a 2x2 table with known cell text. Generated at test time so
/// no opaque binary fixture is committed; the production `docx` converter never
/// depends on `docx-rs` (it reads the OOXML with `zip` + `quick-xml`).
fn build_docx() -> Vec<u8> {
    use docx_rs::{Docx, Paragraph, Run, Table, TableCell, TableRow};
    use std::io::Cursor;

    fn cell(text: &str) -> TableCell {
        TableCell::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text(text)))
    }

    let docx = Docx::new()
        .add_paragraph(
            Paragraph::new()
                .style("Heading1")
                .add_run(Run::new().add_text("Project Kickoff")),
        )
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Intro body paragraph text.")))
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("First bullet item")))
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Second bullet item")))
        .add_table(Table::new(vec![
            TableRow::new(vec![cell("Region"), cell("Owner")]),
            TableRow::new(vec![cell("North"), cell("Alice")]),
        ]));

    let mut buf = Cursor::new(Vec::new());
    docx.build().pack(&mut buf).expect("pack docx");
    buf.into_inner()
}

#[test]
fn docx_extracts_paragraph_list_and_table_cells() {
    let bytes = build_docx();
    let out = convert_to_markdown(&bytes, "docx", None).expect("docx conversion");

    // Heading + body + list-item text all surface as plain paragraph text
    // (the bullet glyph itself is not reconstructed — the bar is content).
    assert!(out.contains("Project Kickoff"), "expected heading text, got: {out:?}");
    assert!(
        out.contains("Intro body paragraph text"),
        "expected body paragraph, got: {out:?}"
    );
    assert!(out.contains("First bullet item"), "expected list item, got: {out:?}");
    assert!(out.contains("Second bullet item"), "expected list item, got: {out:?}");

    // Table cells render as a markdown pipe-table carrying every cell's text.
    assert!(out.contains('|'), "expected table separators, got: {out:?}");
    for cell in &["Region", "Owner", "North", "Alice"] {
        assert!(out.contains(cell), "expected table cell {cell:?}, got: {out:?}");
    }
}

// ---------------------------------------------------------------------------
// Size and zip limits
// ---------------------------------------------------------------------------

#[test]
fn oversize_input_is_rejected() {
    // 50 MiB + 1 byte — rejected before the parser runs.
    let big = vec![0u8; 50 * 1024 * 1024 + 1];
    let err = convert_to_markdown(&big, "txt", None).unwrap_err();
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

    let err = convert_to_markdown(&bytes, "pptx", None).unwrap_err();
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput for entry-count bomb, got {err:?}"
    );
}

#[test]
fn malformed_pptx_zip_is_rejected() {
    // Garbage bytes that are not a valid zip archive.
    let garbage = b"this is not a zip file at all!!!";
    let err = convert_to_markdown(garbage, "pptx", None).unwrap_err();
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput for malformed zip, got {err:?}"
    );
}

#[test]
fn unknown_extension_is_rejected() {
    let err = convert_to_markdown(b"hello", "rtf", None).unwrap_err();
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput for unsupported ext, got {err:?}"
    );
}

#[test]
fn empty_input_is_ok() {
    let out = convert_to_markdown(b"", "txt", None).expect("empty txt");
    assert_eq!(out, "");
}

// ---------------------------------------------------------------------------
// VLM OCR fallback (stub DocVlm — no real model)
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicUsize, Ordering};

/// A `DocVlm` test double: counts how many times it is invoked and returns a
/// fixed markdown string echoing the call index, so concatenation order and
/// per-page invocation can be asserted without loading any model.
#[derive(Default)]
struct StubVlm {
    calls: AtomicUsize,
}

impl minutist_common::DocVlm for StubVlm {
    fn image_to_markdown(&self, png: &[u8]) -> Result<String, AppError> {
        // Sanity: the fallback must hand us a valid PNG (magic bytes).
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "DocVlm must receive a PNG");
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(format!("STUB-OCR call {n}"))
    }
}

/// Build a tiny in-memory PNG (no opaque binary committed). The pixels are
/// irrelevant — the stub asserts the PNG header, not the content.
fn build_png() -> Vec<u8> {
    let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 200, 200]));
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .expect("encode png");
    buf.into_inner()
}

#[test]
fn image_attachment_routes_to_vlm_stub() {
    let png = build_png();
    let stub = StubVlm::default();
    let out = convert_to_markdown(&png, "png", Some(&stub)).expect("png via stub");
    assert!(out.contains("STUB-OCR call 1"), "stub output missing: {out:?}");
    assert_eq!(stub.calls.load(Ordering::SeqCst), 1, "stub called once");
}

#[test]
fn image_attachment_without_vlm_is_unsupported() {
    let png = build_png();
    let err = convert_to_markdown(&png, "png", None).unwrap_err();
    assert!(
        matches!(err, AppError::Unsupported { .. }),
        "image with no VLM must be Unsupported, got {err:?}"
    );
}

#[test]
fn image_attachment_jpeg_reencoded_to_png_for_vlm() {
    // A JPEG must be decoded and re-encoded to PNG before the VLM sees it —
    // the stub asserts the PNG header, so a JPEG passed through unchanged would
    // trip that assertion.
    let rgb = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(rgb)
        .write_to(&mut buf, image::ImageFormat::Jpeg)
        .expect("encode jpeg");
    let jpeg = buf.into_inner();

    let stub = StubVlm::default();
    let out = convert_to_markdown(&jpeg, "jpeg", Some(&stub)).expect("jpeg via stub");
    assert!(out.contains("STUB-OCR call 1"), "stub output missing: {out:?}");
}

#[test]
fn near_empty_pdf_is_unsupported_even_with_vlm() {
    // A near-empty PDF (under the 100 non-ws-char threshold) is a scanned /
    // image-only document. PDF-page OCR is deferred (planning issue 0019): the
    // VLM is never consulted for PDFs, so this returns Unsupported and the stub
    // stays untouched even when one is injected.
    let bytes = build_text_pdf("ok");
    let stub = StubVlm::default();
    let err = convert_to_markdown(&bytes, "pdf", Some(&stub)).unwrap_err();
    assert!(
        matches!(err, AppError::Unsupported { .. }),
        "near-empty PDF must be Unsupported, got {err:?}"
    );
    assert_eq!(
        stub.calls.load(Ordering::SeqCst),
        0,
        "the VLM must not be invoked for PDFs (no rasterisation path)"
    );
}
