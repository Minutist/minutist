//! Self-contained synthetic fixtures with EXACT ground truth.
//!
//! Every page is rendered in-process from a known Rust string (and the table
//! from a known grid of cell strings) via a pure-Rust monospaced rasteriser
//! (`ab_glyph` + `image`). Ground truth is therefore the source text itself —
//! no committed binary fixture, no hand-authored reference file.
//!
//! These pages test the mtmd IMAGE plumbing plus clean-text and simple-table
//! transcription accuracy. They are NOT a substitute for dense real-world
//! layout — point `--input` at a hard document for that.

use anyhow::{Context, Result};
use ab_glyph::{point, Font, FontRef, Glyph, PxScale, ScaleFont};
use image::{ImageBuffer, ImageFormat, Rgba, RgbaImage};

/// DejaVu Sans Mono — Bitstream-Vera-derived permissive licence; embeddable in
/// commercial software. Monospaced so the table columns line up exactly.
static FONT_TTF: &[u8] = include_bytes!("../assets/DejaVuSansMono.ttf");

const FONT_PX: f32 = 28.0;
const MARGIN: f32 = 32.0;

/// One synthetic page: PNG bytes for inference + its exact ground-truth text.
pub struct SyntheticPage {
    pub name: &'static str,
    /// PNG-encoded page image (fed to `MtmdBitmap::from_buffer`).
    pub png: Vec<u8>,
    /// Exact, normalised ground-truth markdown the model should reproduce.
    pub ground_truth: String,
}

/// Build the standard synthetic page set: a clean multi-paragraph text page and
/// a simple bordered table page.
pub fn build_pages() -> Result<Vec<SyntheticPage>> {
    Ok(vec![text_page()?, table_page()?])
}

// ---------------------------------------------------------------------------
// Page (a): clean multi-paragraph text
// ---------------------------------------------------------------------------

fn text_page() -> Result<SyntheticPage> {
    // Known source lines. Plain ASCII keeps the ground truth unambiguous and
    // the monospaced layout exact.
    let lines = [
        "Meeting Notes",
        "",
        "The project review covered three topics. First, the",
        "transcription pipeline now produces per-speaker turns.",
        "Second, the summariser runs after recording stops.",
        "",
        "Action items were assigned to each owner. The next",
        "review is scheduled for the following sprint.",
    ];
    let img = render_lines(&lines)?;
    let png = encode_png(&img).context("encoding text page PNG")?;
    let ground_truth = normalise(&lines.join("\n"));
    Ok(SyntheticPage {
        name: "clean-text",
        png,
        ground_truth,
    })
}

// ---------------------------------------------------------------------------
// Page (b): simple bordered table
// ---------------------------------------------------------------------------

fn table_page() -> Result<SyntheticPage> {
    // Known cells. Ground truth is the canonical pipe-table rendering of the
    // same cells, so the model's markdown table can be scored directly.
    let header = ["Item", "Owner", "Status"];
    let rows = [
        ["Transcript", "Alice", "Done"],
        ["Summary", "Bob", "Open"],
        ["Voiceprint", "Carol", "Open"],
    ];

    let img = render_table(&header, &rows)?;
    let png = encode_png(&img).context("encoding table page PNG")?;

    // Canonical pipe-table ground truth.
    let mut gt = String::new();
    gt.push_str(&pipe_row(&header));
    gt.push('\n');
    gt.push_str(&pipe_sep(header.len()));
    gt.push('\n');
    for r in &rows {
        gt.push_str(&pipe_row(r));
        gt.push('\n');
    }

    Ok(SyntheticPage {
        name: "table",
        png,
        ground_truth: normalise(&gt),
    })
}

fn pipe_row(cells: &[&str]) -> String {
    format!("| {} |", cells.join(" | "))
}

fn pipe_sep(n: usize) -> String {
    let mut s = String::from("|");
    for _ in 0..n {
        s.push_str(" --- |");
    }
    s
}

// ---------------------------------------------------------------------------
// Pure-Rust monospaced text rasterisation
// ---------------------------------------------------------------------------

fn font() -> Result<FontRef<'static>> {
    FontRef::try_from_slice(FONT_TTF).context("parsing embedded DejaVuSansMono.ttf")
}

/// Render a block of monospaced text lines to a white-background image.
fn render_lines(lines: &[&str]) -> Result<RgbaImage> {
    let font = font()?;
    let scale = PxScale::from(FONT_PX);
    let sf = font.as_scaled(scale);
    let line_h = (sf.ascent() - sf.descent() + sf.line_gap()).ceil();
    let adv = sf.h_advance(font.glyph_id('M'));
    let max_cols = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as f32;

    let w = (MARGIN * 2.0 + adv * max_cols).ceil().max(1.0) as u32;
    let h = (MARGIN * 2.0 + line_h * lines.len() as f32).ceil().max(1.0) as u32;
    let mut img: RgbaImage = ImageBuffer::from_pixel(w, h, Rgba([255, 255, 255, 255]));

    for (row, text) in lines.iter().enumerate() {
        let baseline_y = MARGIN + sf.ascent() + line_h * row as f32;
        draw_text(&mut img, &font, scale, MARGIN, baseline_y, text, adv);
    }
    Ok(img)
}

/// Render a bordered table: monospaced cells on a fixed grid with 1px rules.
fn render_table(header: &[&str; 3], rows: &[[&str; 3]; 3]) -> Result<RgbaImage> {
    let font = font()?;
    let scale = PxScale::from(FONT_PX);
    let sf = font.as_scaled(scale);
    let line_h = (sf.ascent() - sf.descent() + sf.line_gap()).ceil();
    let adv = sf.h_advance(font.glyph_id('M'));

    let n_cols = 3usize;
    let n_rows = 1 + rows.len(); // header + body

    // Column width = widest cell in that column, in monospace cells, + padding.
    let mut col_chars = [0usize; 3];
    for c in 0..n_cols {
        col_chars[c] = header[c].chars().count();
    }
    for r in rows {
        for c in 0..n_cols {
            col_chars[c] = col_chars[c].max(r[c].chars().count());
        }
    }
    let pad = 2usize; // cells of padding each side
    let col_w: Vec<f32> = col_chars
        .iter()
        .map(|&c| adv * (c + pad * 2) as f32)
        .collect();
    let row_h = line_h + 12.0;

    let table_w: f32 = col_w.iter().sum();
    let w = (MARGIN * 2.0 + table_w).ceil().max(1.0) as u32;
    let h = (MARGIN * 2.0 + row_h * n_rows as f32).ceil().max(1.0) as u32;
    let mut img: RgbaImage = ImageBuffer::from_pixel(w, h, Rgba([255, 255, 255, 255]));

    let x0 = MARGIN;
    let y0 = MARGIN;

    // Cell text.
    for r in 0..n_rows {
        let cells: [&str; 3] = if r == 0 { *header } else { rows[r - 1] };
        let baseline_y = y0 + row_h * r as f32 + sf.ascent() + 4.0;
        let mut cx = x0;
        for c in 0..n_cols {
            let text_x = cx + adv * pad as f32;
            draw_text(&mut img, &font, scale, text_x, baseline_y, cells[c], adv);
            cx += col_w[c];
        }
    }

    // Rules.
    let table_right = (x0 + table_w).round() as u32;
    let table_bottom = (y0 + row_h * n_rows as f32).round() as u32;
    for r in 0..=n_rows {
        let y = (y0 + row_h * r as f32).round() as u32;
        draw_hline(&mut img, x0.round() as u32, table_right, y);
    }
    let mut cx = x0;
    draw_vline(&mut img, y0.round() as u32, table_bottom, cx.round() as u32);
    for c in 0..n_cols {
        cx += col_w[c];
        draw_vline(&mut img, y0.round() as u32, table_bottom, cx.round() as u32);
    }

    Ok(img)
}

fn draw_text(
    img: &mut RgbaImage,
    font: &FontRef<'_>,
    scale: PxScale,
    start_x: f32,
    baseline_y: f32,
    text: &str,
    adv: f32,
) {
    let mut pen_x = start_x;
    for ch in text.chars() {
        let g: Glyph = font
            .glyph_id(ch)
            .with_scale_and_position(scale, point(pen_x, baseline_y));
        if let Some(outlined) = font.outline_glyph(g) {
            let bb = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                let px = bb.min.x as i32 + gx as i32;
                let py = bb.min.y as i32 + gy as i32;
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    let v = (255.0 * (1.0 - coverage)) as u8; // black text on white
                    img.put_pixel(px as u32, py as u32, Rgba([v, v, v, 255]));
                }
            });
        }
        pen_x += adv; // fixed advance (monospace) -> predictable layout
    }
}

fn draw_hline(img: &mut RgbaImage, x0: u32, x1: u32, y: u32) {
    if y >= img.height() {
        return;
    }
    for x in x0..=x1.min(img.width().saturating_sub(1)) {
        img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
    }
}

fn draw_vline(img: &mut RgbaImage, y0: u32, y1: u32, x: u32) {
    if x >= img.width() {
        return;
    }
    for y in y0..=y1.min(img.height().saturating_sub(1)) {
        img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
    }
}

fn encode_png(img: &RgbaImage) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img.clone())
        .write_to(&mut buf, ImageFormat::Png)
        .context("PNG encode")?;
    Ok(buf.into_inner())
}

// ---------------------------------------------------------------------------
// Normalisation (shared by ground truth and prediction before scoring)
// ---------------------------------------------------------------------------

/// Light normalisation matching the OmniDocBench approach: trim, collapse all
/// runs of whitespace (incl. newlines) to a single space. Applied identically
/// to ground truth and model output so spacing noise does not dominate CER.
pub fn normalise(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_have_nonempty_png_and_truth() {
        let pages = build_pages().expect("build synthetic pages");
        assert_eq!(pages.len(), 2);
        for p in &pages {
            assert!(p.png.len() > 100, "{} png too small", p.name);
            // PNG magic bytes.
            assert_eq!(&p.png[..4], &[0x89, b'P', b'N', b'G'], "{} not a PNG", p.name);
            assert!(!p.ground_truth.is_empty(), "{} empty truth", p.name);
        }
    }

    #[test]
    fn normalise_collapses_whitespace() {
        assert_eq!(normalise("  a\n\n b\t c  "), "a b c");
    }

    #[test]
    fn table_ground_truth_is_pipe_table() {
        let pages = build_pages().unwrap();
        let table = pages.iter().find(|p| p.name == "table").unwrap();
        assert!(table.ground_truth.contains("| Item | Owner | Status |"));
        assert!(table.ground_truth.contains("--- |"));
    }
}
