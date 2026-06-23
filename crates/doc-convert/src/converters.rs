//! Per-format converter implementations.
//!
//! Each public function takes raw bytes and returns a markdown string, using
//! [`crate::normalise::normalise`] as the final step. All errors are
//! [`crate::error::ConvertError`]; the caller in [`crate`] maps them to
//! [`minutist_common::AppError`] via `?`.

use crate::error::{ConvertError, Result};
use crate::normalise::normalise;
use crate::{MAX_ZIP_ENTRIES, MAX_ZIP_UNCOMPRESSED_BYTES};

// ---------------------------------------------------------------------------
// txt / md passthrough
// ---------------------------------------------------------------------------

/// Plain text and markdown: interpret bytes as UTF-8 (lossy) and normalise.
pub fn passthrough(bytes: &[u8]) -> minutist_common::AppResult<String> {
    let text = String::from_utf8_lossy(bytes);
    normalise(&text).map_err(minutist_common::AppError::from)
}

// ---------------------------------------------------------------------------
// XLSX / ODS via calamine
// ---------------------------------------------------------------------------

/// Spreadsheet converter for `.xlsx` and `.ods`.
///
/// Renders each sheet as a Markdown table (pipe-separated). Each row becomes
/// one table row; the first row becomes the header. Empty cells become an
/// empty cell. Sheets with no data are skipped.
pub fn spreadsheet(bytes: &[u8], ext: &str) -> minutist_common::AppResult<String> {
    use calamine::{open_workbook_auto_from_rs, Reader};
    use std::io::Cursor;

    // xlsx/ods are zip containers and calamine decompresses them internally with
    // no size bound — `MAX_INPUT_BYTES` caps only the COMPRESSED input, so a small
    // high-ratio archive could still inflate to many GB (and `catch_unwind` does
    // not catch an OOM). Validate the archive before calamine touches it. See the
    // doc-convert sandboxing rule in architecture/cross-cutting.md.
    enforce_zip_bounds(bytes)?;

    let cursor = Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor)
        .map_err(|e| ConvertError::Spreadsheet(e.to_string()))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();

    let mut md = String::new();

    for sheet_name in &sheet_names {
        let range = match workbook.worksheet_range(sheet_name) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    target: "doc-convert",
                    sheet = %sheet_name,
                    ext = ext,
                    "failed to read sheet: {e}"
                );
                continue;
            }
        };

        if range.is_empty() {
            continue;
        }

        md.push_str(&format!("## Sheet: {sheet_name}\n\n"));

        let rows: Vec<Vec<String>> = range
            .rows()
            .map(|row| {
                row.iter()
                    .map(|cell| cell_to_str(cell))
                    .collect()
            })
            .collect();

        if rows.is_empty() {
            continue;
        }

        // Find the maximum column count across all rows for consistent tables.
        let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        if col_count == 0 {
            continue;
        }

        // Header row
        let header = &rows[0];
        let padded_header: Vec<String> = (0..col_count)
            .map(|i| header.get(i).cloned().unwrap_or_default())
            .collect();
        md.push('|');
        for h in &padded_header {
            md.push_str(&format!(" {} |", escape_pipe(h)));
        }
        md.push('\n');

        // Separator row
        md.push('|');
        for _ in 0..col_count {
            md.push_str(" --- |");
        }
        md.push('\n');

        // Data rows (starting from index 1)
        for row in rows.iter().skip(1) {
            md.push('|');
            for i in 0..col_count {
                let cell = row.get(i).map(String::as_str).unwrap_or("");
                md.push_str(&format!(" {} |", escape_pipe(cell)));
            }
            md.push('\n');
        }

        md.push('\n');
    }

    normalise(&md).map_err(minutist_common::AppError::from)
}

fn cell_to_str(cell: &calamine::Data) -> String {
    use calamine::Data;
    match cell {
        Data::Empty => String::new(),
        Data::String(s) => s.clone(),
        Data::Float(f) => {
            // Avoid spurious decimal points for integer-valued floats.
            if f.fract() == 0.0 && f.abs() < 1e15 {
                format!("{}", *f as i64)
            } else {
                format!("{f}")
            }
        }
        Data::Int(i) => format!("{i}"),
        Data::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        Data::Error(e) => format!("#ERR:{e}"),
        // `ExcelDateTime`'s Display writes the raw serial number, not a date, so
        // render via `as_datetime()` (the `dates` feature). Durations and any
        // value that does not resolve to a datetime fall back to the serial.
        Data::DateTime(dt) => dt
            .as_datetime()
            .map(|ndt| ndt.to_string())
            .unwrap_or_else(|| format!("{dt}")),
        Data::DateTimeIso(s) | Data::DurationIso(s) => s.clone(),
    }
}

fn escape_pipe(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ").replace('\r', "")
}

// ---------------------------------------------------------------------------
// HTML via dom_smoothie + htmd
// ---------------------------------------------------------------------------

/// HTML converter: readability extraction then HTML → Markdown.
///
/// Unlike the zip-container formats, the readability/DOM pass has no structural
/// bound (entry count, expansion ratio) — its only backstop against pathological
/// markup is the `MAX_INPUT_BYTES` (50 MiB) input cap enforced in
/// [`crate::convert_to_markdown`] before this runs. That ceiling is the
/// deliberate limit for the non-zip parsers (`html`/`eml`); it is acceptable for
/// a local single-user app where the single conversion worker bounds the blast
/// radius to one background job.
pub fn html(bytes: &[u8]) -> minutist_common::AppResult<String> {
    let html_str = String::from_utf8_lossy(bytes);
    html_str_to_markdown(&html_str)
}

fn html_str_to_markdown(html_str: &str) -> minutist_common::AppResult<String> {
    use dom_smoothie::{Config, Readability};

    // Readability extraction: reduces boilerplate nav/footer HTML to article body.
    // On failure (e.g. the document has no extractable article) fall back to the
    // raw html.
    let readable_html = match Readability::new(html_str, None, Some(Config::default())) {
        Ok(mut reader) => match reader.parse() {
            Ok(article) => article.content.to_string(),
            Err(_) => html_str.to_string(),
        },
        Err(_) => html_str.to_string(),
    };

    let md = htmd::convert(&readable_html).map_err(|e| ConvertError::Html(e.to_string()))?;
    normalise(&md).map_err(minutist_common::AppError::from)
}

// ---------------------------------------------------------------------------
// .eml via mail-parser
// ---------------------------------------------------------------------------

/// Email converter: parse the MIME message, extract the HTML body (preferred)
/// or plain-text body, convert to markdown.
pub fn eml(bytes: &[u8]) -> minutist_common::AppResult<String> {
    use mail_parser::MessageParser;

    let message = MessageParser::default()
        .parse(bytes)
        .ok_or_else(|| ConvertError::Email("failed to parse email message".to_string()))?;

    // Prefer HTML body; fall back to plain text.
    if let Some(html_body) = message.body_html(0) {
        html_str_to_markdown(&html_body)
    } else if let Some(text_body) = message.body_text(0) {
        normalise(&text_body).map_err(minutist_common::AppError::from)
    } else {
        // No body at all — return a headers summary. Format address fields
        // explicitly ("Name <addr>") rather than via the `Address` Debug impl,
        // which would leak the parser's internal struct shape into the markdown.
        let mut md = String::new();
        if let Some(from) = message.from() {
            md.push_str(&format!("**From:** {}\n\n", format_address(from)));
        }
        if let Some(subject) = message.subject() {
            md.push_str(&format!("**Subject:** {subject}\n\n"));
        }
        normalise(&md).map_err(minutist_common::AppError::from)
    }
}

/// Render a mail-parser [`mail_parser::Address`] as a comma-separated list of
/// `Name <addr>` / `addr` / `Name` fragments, skipping entries with neither a
/// name nor an address. Used only for the headers-only `.eml` fallback.
fn format_address(address: &mail_parser::Address) -> String {
    let parts: Vec<String> = address
        .iter()
        .filter_map(|addr| match (addr.name(), addr.address()) {
            (Some(name), Some(email)) => Some(format!("{name} <{email}>")),
            (None, Some(email)) => Some(email.to_string()),
            (Some(name), None) => Some(name.to_string()),
            (None, None) => None,
        })
        .collect();
    parts.join(", ")
}

// ---------------------------------------------------------------------------
// PDF via pdf-extract (digital text only)
// ---------------------------------------------------------------------------

/// Digital-text PDF extractor. Returns extracted text as plain paragraphs.
///
/// If the extracted text is near-empty (< 100 non-whitespace chars), the PDF is
/// treated as scanned / image-only and rejected with `AppError::Unsupported`:
/// this build has no PDF-page rasterisation path, so only PDFs with a usable
/// digital text layer convert. (PDF page OCR is tracked in planning issue 0019.)
pub fn pdf(bytes: &[u8]) -> minutist_common::AppResult<String> {
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| ConvertError::Pdf(e.to_string()))?;

    if is_near_empty_text(&text) {
        let non_ws = non_whitespace_count(&text);
        tracing::debug!(
            target: "doc-convert",
            non_ws,
            "pdf-extract returned near-empty text; scanned/image-only PDF is unsupported"
        );
        return Err(minutist_common::AppError::Unsupported {
            context: "scanned or image-only PDF — no extractable text \
                      (PDF page OCR is not available in this build)"
                .to_string(),
        });
    }

    normalise(&text).map_err(minutist_common::AppError::from)
}

/// Threshold below which extracted PDF text is treated as "no usable text
/// layer" and the VLM fallback seam is reached.
const PDF_MIN_NON_WHITESPACE_CHARS: usize = 100;

/// Count the non-whitespace characters in `text`.
fn non_whitespace_count(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

/// Whether extracted PDF text is near-empty: fewer than
/// [`PDF_MIN_NON_WHITESPACE_CHARS`] non-whitespace characters. The threshold is
/// exclusive — exactly the threshold count is treated as usable text.
fn is_near_empty_text(text: &str) -> bool {
    non_whitespace_count(text) < PDF_MIN_NON_WHITESPACE_CHARS
}

// ---------------------------------------------------------------------------
// Direct image attachments via the VLM (png / jpg / jpeg / tiff)
// ---------------------------------------------------------------------------

/// Direct image attachment converter.
///
/// Images have no pure-Rust text path, so the injected
/// [`minutist_common::DocVlm`] is the only route: the bytes are decoded and
/// re-encoded to PNG (the format the trait expects) and handed to
/// [`minutist_common::DocVlm::image_to_markdown`]. Returns
/// `AppError::Unsupported` when no VLM is injected, and `AppError::InvalidInput`
/// when the bytes do not decode as `ext`.
pub fn image(
    bytes: &[u8],
    ext: &str,
    vlm: Option<&dyn minutist_common::DocVlm>,
) -> minutist_common::AppResult<String> {
    let Some(vlm) = vlm else {
        return Err(minutist_common::AppError::Unsupported {
            context: format!(
                "image attachment .{ext} — document OCR is unavailable (no vision model)"
            ),
        });
    };

    let png = reencode_to_png(bytes).map_err(minutist_common::AppError::from)?;
    let md = vlm.image_to_markdown(&png)?;
    normalise(&md).map_err(minutist_common::AppError::from)
}

/// Decode an image (any format `image` recognises by content) and re-encode it
/// as PNG, so the [`minutist_common::DocVlm`] always receives a valid PNG.
fn reencode_to_png(bytes: &[u8]) -> Result<Vec<u8>> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| ConvertError::Image(format!("decoding image: {e}")))?;
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| ConvertError::Image(format!("PNG-encoding image: {e}")))?;
    Ok(buf.into_inner())
}

// ---------------------------------------------------------------------------
// Shared zip-container guard (pptx / xlsx / ods)
// ---------------------------------------------------------------------------

/// Decompression-bomb guard for zip-container formats (`pptx`, `xlsx`, `ods`).
///
/// Reads only the archive's central directory (no extraction) and rejects an
/// archive whose entry count or cumulative *uncompressed* size exceeds the
/// limits. Run BEFORE handing bytes to a parser that decompresses the zip
/// internally (calamine for `xlsx`/`ods`): [`crate::MAX_INPUT_BYTES`] bounds only
/// the COMPRESSED input, so a small high-ratio archive can still inflate to many
/// GB, and `catch_unwind` does not catch an OOM.
///
/// The two checks differ in strength. The entry-count cap and the 50 MiB
/// compressed-input cap ([`crate::MAX_INPUT_BYTES`], enforced by the caller) are
/// hard bounds: an attacker cannot exceed them without sending more bytes or
/// more entries. The cumulative *uncompressed* cap is advisory — it trusts the
/// uncompressed sizes declared in the central-directory records, which a crafted
/// archive can understate. It catches honest large archives and naive bombs, but
/// is not a substitute for bounding decompression at extraction time; the hard
/// entry-count and compressed-size caps are the real ceiling.
fn enforce_zip_bounds(bytes: &[u8]) -> Result<()> {
    enforce_zip_bounds_with(bytes, MAX_ZIP_ENTRIES, MAX_ZIP_UNCOMPRESSED_BYTES)
}

/// [`enforce_zip_bounds`] with explicit limits, so tests can trip the guard
/// without constructing a multi-GB archive.
fn enforce_zip_bounds_with(bytes: &[u8], max_entries: usize, max_uncompressed: u64) -> Result<()> {
    use std::io::Cursor;
    use zip::ZipArchive;

    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).map_err(ConvertError::Zip)?;

    let total_count = archive.len();
    if total_count > max_entries {
        return Err(ConvertError::ZipBomb(format!(
            "entry count {total_count} exceeds limit {max_entries}"
        )));
    }

    let mut total_uncompressed: u64 = 0;
    for i in 0..total_count {
        let entry = archive.by_index_raw(i).map_err(ConvertError::Zip)?;
        total_uncompressed += entry.size();
        if total_uncompressed > max_uncompressed {
            return Err(ConvertError::ZipBomb(format!(
                "cumulative uncompressed size {total_uncompressed} exceeds limit {max_uncompressed}"
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// PPTX via zip + quick-xml
// ---------------------------------------------------------------------------

/// PPTX converter: extract text runs from each slide's OOXML XML.
///
/// Iterates `ppt/slides/slideN.xml` entries in the zip in numeric order,
/// emits `## Slide N` headings, and collects `<a:t>` text runs (which include
/// table-cell text — see [`extract_slide_text`]). After each slide's body, the
/// corresponding `ppt/notesSlides/notesSlideN.xml` speaker notes, if any, are
/// appended under a `### Notes` sub-heading.
///
/// The slide↔notes pairing is by numeric suffix (`slideN` ↔ `notesSlideN`).
/// OOXML defines the binding through the slide's `.rels`, but in practice
/// PowerPoint names the notes part with the same ordinal as its slide, so the
/// suffix match is sufficient for a text-content extraction (the bar here is
/// capturing the notes text, not reconstructing the relationship graph).
pub fn pptx(bytes: &[u8]) -> minutist_common::AppResult<String> {
    use std::io::{Cursor, Read};
    use zip::ZipArchive;

    // Decompression-bomb guard (shared with the xlsx/ods path). Run it before
    // opening the archive for parsing — match the spreadsheet path so a
    // pathological archive is rejected before any decompression seam.
    enforce_zip_bounds(bytes)?;

    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).map_err(ConvertError::Zip)?;

    let total_count = archive.len();

    // Collect slide entry names (ppt/slides/slideN.xml) and, separately, the
    // notes-slide names (ppt/notesSlides/notesSlideN.xml), keyed by ordinal.
    let mut slide_names: Vec<String> = Vec::new();
    let mut notes_by_num: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    for i in 0..total_count {
        if let Ok(e) = archive.by_index_raw(i) {
            let name = e.name().to_string();
            if name.contains("/_rels/") || !name.ends_with(".xml") {
                continue;
            }
            if name.starts_with("ppt/slides/slide") {
                slide_names.push(name);
            } else if name.starts_with("ppt/notesSlides/notesSlide") {
                notes_by_num.insert(notes_slide_number(&name), name);
            }
        }
    }

    // Sort numerically by slide number (slide1.xml < slide2.xml ... < slide10.xml).
    slide_names.sort_by(|a, b| slide_number(a).cmp(&slide_number(b)));

    let mut md = String::new();

    for (slide_idx, slide_name) in slide_names.iter().enumerate() {
        let slide_num = slide_idx + 1;
        let ordinal = slide_number(slide_name);

        let mut xml_bytes = Vec::new();
        {
            let mut file = archive.by_name(slide_name).map_err(ConvertError::Zip)?;
            file.read_to_end(&mut xml_bytes).map_err(ConvertError::Io)?;
        }
        let slide_text = extract_slide_text(&xml_bytes)?;

        // Speaker notes for this slide, matched by numeric ordinal.
        let notes_text = match notes_by_num.get(&ordinal) {
            Some(notes_name) => {
                let mut notes_bytes = Vec::new();
                {
                    let mut file = archive.by_name(notes_name).map_err(ConvertError::Zip)?;
                    file.read_to_end(&mut notes_bytes).map_err(ConvertError::Io)?;
                }
                extract_slide_text(&notes_bytes)?
            }
            None => String::new(),
        };

        if slide_text.trim().is_empty() && notes_text.trim().is_empty() {
            continue;
        }

        md.push_str(&format!("## Slide {slide_num}\n\n"));
        if !slide_text.trim().is_empty() {
            md.push_str(&slide_text);
            md.push('\n');
        }
        if !notes_text.trim().is_empty() {
            md.push_str("### Notes\n\n");
            md.push_str(&notes_text);
            md.push('\n');
        }
    }

    normalise(&md).map_err(minutist_common::AppError::from)
}

/// Extract the `<a:t>` text runs from a slide XML buffer.
///
/// Text runs within the same paragraph (`<a:p>`) are concatenated; paragraphs
/// are separated by newlines. A blank paragraph yields an empty line. This walk
/// is tag-name driven on `a:p` / `a:t` only, so it captures text wherever it
/// appears — including table cells, which in a slide are an `a:graphicFrame`
/// containing an `a:tbl` whose `a:tc` cells hold ordinary `a:p`/`a:t` runs.
fn extract_slide_text(xml: &[u8]) -> Result<String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_reader(xml);
    reader.config_mut().trim_text(true);

    let mut text = String::new();
    let mut current_para = String::new();
    let mut in_a_t = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                match e.name().as_ref() {
                    b"a:t" => in_a_t = true,
                    b"a:p" => {
                        // New paragraph: flush the previous one.
                        if !current_para.is_empty() {
                            text.push_str(&current_para);
                            text.push('\n');
                            current_para.clear();
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                if e.name().as_ref() == b"a:t" {
                    in_a_t = false;
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_a_t {
                    let s = e
                        .decode()
                        .map_err(|e| ConvertError::Xml(e.to_string()))?;
                    current_para.push_str(&s);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(ConvertError::Xml(e.to_string()));
            }
            _ => {}
        }
    }

    // Flush the last paragraph.
    if !current_para.is_empty() {
        text.push_str(&current_para);
        text.push('\n');
    }

    Ok(text)
}

/// Parse the slide number from a name like `"ppt/slides/slide3.xml"`.
/// Slides that don't match the pattern sort to the end (usize::MAX).
fn slide_number(name: &str) -> usize {
    // Basename is e.g. "slide3.xml"
    let basename = name.rsplit('/').next().unwrap_or(name);
    // Strip prefix "slide" and suffix ".xml"
    let inner = basename
        .strip_prefix("slide")
        .and_then(|s| s.strip_suffix(".xml"))
        .unwrap_or("");
    inner.parse::<usize>().unwrap_or(usize::MAX)
}

/// Parse the notes-slide number from a name like
/// `"ppt/notesSlides/notesSlide3.xml"`. Non-matching names sort to `usize::MAX`.
fn notes_slide_number(name: &str) -> usize {
    let basename = name.rsplit('/').next().unwrap_or(name);
    let inner = basename
        .strip_prefix("notesSlide")
        .and_then(|s| s.strip_suffix(".xml"))
        .unwrap_or("");
    inner.parse::<usize>().unwrap_or(usize::MAX)
}

// ---------------------------------------------------------------------------
// DOCX via zip + quick-xml
// ---------------------------------------------------------------------------

/// DOCX converter: extract textual content from `word/document.xml`.
///
/// DOCX is a zip container directly analogous to PPTX: the body text lives in
/// `word/document.xml` as `<w:t>` runs inside `<w:p>` paragraphs and
/// `<w:tbl>`/`<w:tr>`/`<w:tc>` tables — the same shape as a slide's
/// `<a:p>`/`<a:t>`, so the same zip + `quick-xml` approach applies.
///
/// The bar is textual content for the summariser, not faithful structure:
/// paragraphs become blank-line-separated text, list items become text (the
/// numbering/bullet glyph is not reconstructed), and tables become markdown
/// pipe-tables built from each cell's concatenated `<w:t>` text. The first table
/// row is emitted as the header row (the source rarely marks a header
/// explicitly; the summariser does not depend on the distinction).
pub fn docx(bytes: &[u8]) -> minutist_common::AppResult<String> {
    use std::io::{Cursor, Read};
    use zip::ZipArchive;

    // Decompression-bomb guard, identical to the pptx/xlsx/ods path.
    enforce_zip_bounds(bytes)?;

    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).map_err(ConvertError::Zip)?;

    let mut xml_bytes = Vec::new();
    {
        let mut file = archive
            .by_name("word/document.xml")
            .map_err(ConvertError::Zip)?;
        file.read_to_end(&mut xml_bytes).map_err(ConvertError::Io)?;
    }

    let md = extract_docx_text(&xml_bytes)?;
    normalise(&md).map_err(minutist_common::AppError::from)
}

/// Walk `word/document.xml`, emitting markdown-ish text.
///
/// State machine over the OOXML WordprocessingML body:
/// - `<w:p>` boundaries flush the current paragraph (blank line between).
/// - `<w:t>` runs contribute the paragraph's text.
/// - Inside a `<w:tbl>`, paragraph flushes are diverted into the current table
///   cell instead of the document body; `<w:tr>`/`<w:tc>` build a row/cell
///   matrix that is rendered as a markdown pipe-table when the table closes.
///   Tables may nest paragraphs and runs arbitrarily deep; cell text is the
///   concatenation of every `<w:t>` reached while that `<w:tc>` is open, with
///   paragraph breaks inside a cell collapsed to spaces (pipe-table cells are
///   single-line).
fn extract_docx_text(xml: &[u8]) -> Result<String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_reader(xml);
    reader.config_mut().trim_text(true);

    let mut md = String::new();
    let mut current_para = String::new();
    let mut in_w_t = false;

    // Table state: depth of nested w:tbl, the row/cell matrix being built, and
    // the cell currently accumulating text.
    let mut tbl_depth: usize = 0;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut current_cell = String::new();

    /// Append `para` as a body paragraph (blank-line separated), trimming.
    fn flush_body_para(md: &mut String, para: &mut String) {
        let trimmed = para.trim();
        if !trimmed.is_empty() {
            md.push_str(trimmed);
            md.push_str("\n\n");
        }
        para.clear();
    }

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => match e.name().as_ref() {
                b"w:t" => in_w_t = true,
                b"w:tbl" => {
                    // Entering a table: any pending body paragraph is flushed
                    // first so table output starts on its own block.
                    if tbl_depth == 0 {
                        flush_body_para(&mut md, &mut current_para);
                        table_rows.clear();
                    }
                    tbl_depth += 1;
                }
                b"w:tr" if tbl_depth > 0 => {
                    current_row.clear();
                }
                b"w:tc" if tbl_depth > 0 => {
                    current_cell.clear();
                }
                b"w:p" => {
                    // Paragraph boundary. Inside a cell, a paragraph break is a
                    // space (cells are single-line); in the body it flushes the
                    // accumulated paragraph.
                    if tbl_depth > 0 {
                        if !current_cell.trim().is_empty() {
                            current_cell.push(' ');
                        }
                    } else {
                        flush_body_para(&mut md, &mut current_para);
                    }
                }
                _ => {}
            },
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"w:t" => in_w_t = false,
                b"w:tc" if tbl_depth > 0 => {
                    current_row.push(current_cell.trim().to_string());
                    current_cell.clear();
                }
                b"w:tr" if tbl_depth > 0 => {
                    table_rows.push(std::mem::take(&mut current_row));
                }
                b"w:tbl" if tbl_depth > 0 => {
                    tbl_depth -= 1;
                    if tbl_depth == 0 {
                        render_pipe_table(&mut md, &table_rows);
                        table_rows.clear();
                    }
                }
                _ => {}
            },
            Ok(Event::Text(ref e)) => {
                if in_w_t {
                    let s = e.decode().map_err(|e| ConvertError::Xml(e.to_string()))?;
                    if tbl_depth > 0 {
                        current_cell.push_str(&s);
                    } else {
                        current_para.push_str(&s);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(ConvertError::Xml(e.to_string())),
            _ => {}
        }
    }

    // Flush a trailing body paragraph (no closing w:p before EOF is unusual but
    // handled for robustness).
    flush_body_para(&mut md, &mut current_para);

    Ok(md)
}

/// Render a collected row/cell matrix as a markdown pipe-table. The first row
/// is treated as the header. Rows are padded to the widest row's column count
/// so the table is rectangular. A matrix with no rows emits nothing.
fn render_pipe_table(md: &mut String, rows: &[Vec<String>]) {
    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if col_count == 0 {
        return;
    }

    let emit_row = |md: &mut String, row: &[String]| {
        md.push('|');
        for i in 0..col_count {
            let cell = row.get(i).map(String::as_str).unwrap_or("");
            md.push_str(&format!(" {} |", escape_pipe(cell)));
        }
        md.push('\n');
    };

    emit_row(md, &rows[0]);

    md.push('|');
    for _ in 0..col_count {
        md.push_str(" --- |");
    }
    md.push('\n');

    for row in rows.iter().skip(1) {
        emit_row(md, row);
    }

    md.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_plain_text() {
        let out = passthrough(b"Hello world").unwrap();
        assert!(out.contains("Hello world"), "got: {out:?}");
    }

    #[test]
    fn datetime_cell_renders_iso_not_serial() {
        use calamine::{Data, ExcelDateTime, ExcelDateTimeType};
        // Serial 45000 (1900 date system) is a real calendar date. It must
        // render as an ISO-ish date string, never the raw serial number that
        // `ExcelDateTime`'s Display would emit.
        let dt = ExcelDateTime::new(45000.0, ExcelDateTimeType::DateTime, false);
        let rendered = cell_to_str(&Data::DateTime(dt));
        assert!(
            !rendered.contains("45000"),
            "date rendered as raw serial: {rendered:?}"
        );
        assert!(
            rendered.contains('-') && rendered.len() >= 10,
            "expected an ISO-ish date, got: {rendered:?}"
        );
    }

    #[test]
    fn passthrough_markdown() {
        let input = "# Heading\n\nSome text.\n";
        let out = passthrough(input.as_bytes()).unwrap();
        assert!(out.contains("Heading"), "got: {out:?}");
        assert!(out.contains("Some text"), "got: {out:?}");
    }

    #[test]
    fn html_converter_basic() {
        let html = b"<html><body><h1>Title</h1><p>Body text</p></body></html>";
        let out = super::html(html).unwrap();
        assert!(out.contains("Title"), "got: {out:?}");
    }

    #[test]
    fn slide_number_parse() {
        assert_eq!(slide_number("ppt/slides/slide1.xml"), 1);
        assert_eq!(slide_number("ppt/slides/slide10.xml"), 10);
        assert_eq!(slide_number("ppt/slides/slide2.xml"), 2);
        assert_eq!(slide_number("ppt/slides/_rels/slide1.xml.rels"), usize::MAX);
    }

    #[test]
    fn escape_pipe_in_cell() {
        assert_eq!(escape_pipe("a|b"), "a\\|b");
        assert_eq!(escape_pipe("no pipe"), "no pipe");
    }

    #[test]
    fn zip_bounds_reject_excess_entry_count() {
        let xlsx = include_bytes!("../tests/fixtures/sample.xlsx");
        // A real xlsx has several zip entries; capping at 1 must be rejected,
        // while the production limits accept the same archive.
        assert!(enforce_zip_bounds_with(xlsx, 1, u64::MAX).is_err());
        assert!(enforce_zip_bounds_with(xlsx, MAX_ZIP_ENTRIES, MAX_ZIP_UNCOMPRESSED_BYTES).is_ok());
    }

    #[test]
    fn zip_bounds_reject_excess_uncompressed_size() {
        let xlsx = include_bytes!("../tests/fixtures/sample.xlsx");
        assert!(enforce_zip_bounds_with(xlsx, MAX_ZIP_ENTRIES, 1).is_err());
    }

    #[test]
    fn pdf_near_empty_threshold_is_exclusive_at_100() {
        // The VLM fallback fires strictly below 100 non-whitespace chars.
        // 99 non-ws chars -> near-empty (fallback); exactly 100 -> usable text.
        let just_under = "a".repeat(99);
        assert!(is_near_empty_text(&just_under), "99 non-ws must be near-empty");
        assert_eq!(non_whitespace_count(&just_under), 99);

        let at_threshold = "a".repeat(100);
        assert!(
            !is_near_empty_text(&at_threshold),
            "exactly 100 non-ws must not be near-empty"
        );

        // Whitespace is excluded from the count: padding 99 letters with many
        // spaces/newlines stays near-empty regardless of total length.
        let padded = format!("{}{}", " \n\t".repeat(50), "a".repeat(99));
        assert!(is_near_empty_text(&padded), "whitespace must not count toward the threshold");
        assert_eq!(non_whitespace_count(&padded), 99);
    }

    #[test]
    fn spreadsheet_guards_against_non_zip_input() {
        // The zip guard runs before calamine: a non-zip blob is rejected up front
        // rather than handed to calamine's internal decompressor.
        assert!(spreadsheet(b"not a zip at all", "xlsx").is_err());
    }
}
