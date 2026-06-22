//! doc-vlm-spike — fully-automated Gemma-4 vision doc-to-markdown go/no-go.
//!
//! Validates the VLM fallback seam that `crates/doc-convert` deliberately
//! omits from its production path. Throwaway code (see
//! `architecture/cross-cutting.md` §6-10): `anyhow`, `eprintln!`, blocking
//! `ureq`, a single sync `main`. NOT wired into the app, and it deliberately
//! does NOT touch the production model-registry manifest.
//!
//! # Zero-config run
//!
//! ```text
//! cargo run -p spike-doc-vlm --release            # CPU
//! cargo run -p spike-doc-vlm --release --features vulkan   # GPU
//! ```
//!
//! No model paths, no PDFium path, no fixture, no required flags. The spike
//! self-acquires the Gemma-4 E4B vision LM + mmproj GGUFs and a PDFium prebuilt
//! into the OS cache dir, renders known synthetic pages with exact ground
//! truth, runs each through mtmd IMAGE inference, scores CER + latency, prints a
//! table, and exits 0 (PASS) / non-zero (FAIL).
//!
//! # mtmd IMAGE path
//!
//! `MtmdBitmap::from_buffer(png)` -> `tokenize(text, &[&bitmap])` with the media
//! marker -> `eval_chunks` prefill -> greedy decode with EOG stop. Same C
//! bindings as the audio path in `crates/asr-runtime`, fed a page image.
//!
//! # Caveat (record on first Vulkan run)
//!
//! Gemma-4 PLE forward-graph issue (llama.cpp #22243) may affect the vision
//! graph. If vision inference crashes on Vulkan, a Gemma-3 multimodal GGUF
//! (`ggml-org/gemma-3-4b-it-GGUF`, no PLE) is the control.

mod acquire;
mod synth;

use std::ffi::CString;
use std::fs;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;

// ---------------------------------------------------------------------------
// Acceptance thresholds
// ---------------------------------------------------------------------------

const CER_GATE: f64 = 0.15;
const LATENCY_GATE_SECS: f64 = 30.0;

/// `--features vulkan` (or metal/cuda/rocm) compiles a GPU backend into
/// llama.cpp; in that case offload layers and run the mtmd encoder on GPU.
const GPU_BUILD: bool = cfg!(any(
    feature = "vulkan",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
));

// ---------------------------------------------------------------------------
// CLI — every flag optional; the default run needs none.
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "spike-doc-vlm",
    about = "Self-acquiring Gemma-4 vision doc-to-markdown go/no-go spike"
)]
struct Cli {
    /// Optional real document to transcribe (PDF rasterised via the
    /// auto-acquired PDFium, or a plain image). No ground truth -> prints the
    /// markdown + latency only; does NOT affect the PASS/FAIL gate.
    #[arg(short = 'i', long, value_name = "PATH")]
    input: Option<PathBuf>,

    /// Override the auto-acquired LM GGUF (default: self-downloaded).
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Override the auto-acquired vision mmproj GGUF (default: self-downloaded).
    #[arg(long, value_name = "PATH")]
    mmproj: Option<PathBuf>,

    /// Override the auto-acquired PDFium shared library (default:
    /// self-downloaded). Only used for PDF `--input`.
    #[arg(long, value_name = "PATH")]
    pdfium_lib: Option<PathBuf>,

    /// Rasterisation DPI for PDF `--input` pages.
    #[arg(long, default_value_t = 150)]
    dpi: u16,

    /// Which PDF page(s) of `--input` to process (1-indexed, comma list, or
    /// 'all'). Ignored for image input.
    #[arg(long, default_value = "1")]
    pages: String,

    /// Instruction prompt injected before the image marker.
    #[arg(
        long,
        default_value = "Convert this document page to clean, well-structured markdown. \
                          Preserve headings, lists, and tables. For tables use GitHub \
                          pipe-table syntax. Output only the markdown content, no preamble."
    )]
    prompt: String,

    /// CPU threads for inference.
    #[arg(short = 't', long, default_value_t = 8)]
    threads: i32,

    /// Maximum tokens to generate per page.
    #[arg(long, default_value_t = 1024)]
    max_tokens: i32,

    /// Context window size in tokens.
    #[arg(long, default_value_t = NonZeroU32::new(8192).unwrap())]
    n_ctx: NonZeroU32,

    /// Decode batch size.
    #[arg(long, default_value_t = 512)]
    n_batch: i32,

    /// GPU layers to offload. Defaults to 99 on a GPU build, 0 otherwise.
    #[arg(long)]
    n_gpu_layers: Option<u32>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    // Any error here is a clean, actionable message + non-zero exit, never a
    // bare panic on the network/acquisition path.
    if let Err(e) = run() {
        eprintln!("\nspike-doc-vlm FAILED: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let n_gpu_layers = cli.n_gpu_layers.unwrap_or(if GPU_BUILD { 99 } else { 0 });

    eprintln!("spike-doc-vlm: GPU build={GPU_BUILD} n_gpu_layers={n_gpu_layers}");

    // --- Acquire artifacts (cache + skip when present) ---
    let model_path = match &cli.model {
        Some(p) => {
            if !p.exists() {
                bail!("--model {} does not exist", p.display());
            }
            p.clone()
        }
        None => acquire::ensure_models()
            .context("acquiring Gemma-4 vision LM GGUF")?
            .lm,
    };
    let mmproj_path = match &cli.mmproj {
        Some(p) => {
            if !p.exists() {
                bail!("--mmproj {} does not exist", p.display());
            }
            p.clone()
        }
        // ensure_models() also fetches the LM; when --model is given but
        // --mmproj is not, we still need the mmproj from the cache.
        None => acquire::ensure_models()
            .context("acquiring Gemma-4 vision mmproj GGUF")?
            .mmproj,
    };

    eprintln!("LM     = {}", model_path.display());
    eprintln!("mmproj = {}", mmproj_path.display());

    // --- llama backend + model + mtmd context ---
    let load_t0 = Instant::now();
    let backend = LlamaBackend::init().map_err(|e| anyhow!("LlamaBackend::init: {e}"))?;
    eprintln!("backend init: {:?}", load_t0.elapsed());

    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let model_t0 = Instant::now();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .map_err(|e| anyhow!("LlamaModel::load_from_file ({}): {e}", model_path.display()))?;
    eprintln!("model load: {:?}", model_t0.elapsed());

    let mtmd_params = MtmdContextParams {
        use_gpu: n_gpu_layers > 0,
        print_timings: true,
        n_threads: cli.threads,
        media_marker: CString::new(mtmd_default_marker())
            .map_err(|e| anyhow!("media marker CString: {e}"))?,
    };
    let mmproj_str = mmproj_path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF8 mmproj path"))?;

    let mtmd_t0 = Instant::now();
    let mtmd_ctx = MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params)
        .map_err(|e| anyhow!("MtmdContext::init_from_file ({mmproj_str}): {e}"))?;
    eprintln!("mtmd init: {:?}", mtmd_t0.elapsed());
    eprintln!(
        "mtmd: supports_vision={} supports_audio={}",
        mtmd_ctx.support_vision(),
        mtmd_ctx.support_audio(),
    );

    // Hard bail: image inference is meaningless without a vision encoder.
    if !mtmd_ctx.support_vision() {
        bail!(
            "passed mmproj does not advertise vision support — did you point at the \
             audio projector? Expected the Gemma-4 vision mmproj \
             (mmproj-gemma-4-E4B-it-Q8_0.gguf)."
        );
    }

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(cli.n_ctx))
        .with_n_batch(cli.n_batch as u32)
        .with_n_threads(cli.threads)
        .with_n_threads_batch(cli.threads);

    // --- Real-document mode: print output only, no gate ---
    if let Some(input) = &cli.input {
        return run_real_input(
            input, &backend, &model, &mtmd_ctx, &ctx_params, &cli, n_gpu_layers,
        );
    }

    // --- Synthetic gate mode ---
    let pages = synth::build_pages().context("building synthetic fixtures")?;
    eprintln!("synthetic pages: {}", pages.len());

    let mut results: Vec<GateRow> = Vec::new();
    for page in &pages {
        eprintln!("--- page '{}' ({} bytes PNG) ---", page.name, page.png.len());
        let out = infer_page(
            &backend, &model, &mtmd_ctx, &ctx_params, &page.png, &cli,
        )
        .with_context(|| format!("inference on synthetic page '{}'", page.name))?;

        let pred = synth::normalise(&out.markdown);
        let cer = char_error_rate(&page.ground_truth, &pred);
        eprintln!(
            "  page '{}': CER={:.3} latency={:.2}s tokens={}",
            page.name, cer, out.inference_secs, out.tokens
        );
        results.push(GateRow {
            name: page.name,
            cer,
            latency: out.inference_secs,
            ground_truth: page.ground_truth.clone(),
            prediction: pred,
        });
    }

    report_and_gate(&results)
}

// ---------------------------------------------------------------------------
// Real --input document mode (no ground truth)
// ---------------------------------------------------------------------------

fn run_real_input(
    input: &std::path::Path,
    backend: &LlamaBackend,
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    ctx_params: &LlamaContextParams,
    cli: &Cli,
    _n_gpu_layers: u32,
) -> Result<()> {
    if !input.exists() {
        bail!("--input {} does not exist", input.display());
    }
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let page_pngs: Vec<(u32, Vec<u8>)> = if ext == "pdf" {
        let pdfium_lib = match &cli.pdfium_lib {
            Some(p) => p.clone(),
            None => acquire::ensure_pdfium().context("acquiring PDFium prebuilt")?,
        };
        rasterise_pdf(input, &pdfium_lib, cli).context("rasterising PDF input")?
    } else {
        let bytes = fs::read(input)
            .with_context(|| format!("reading image {}", input.display()))?;
        vec![(1, bytes)]
    };

    if page_pngs.is_empty() {
        bail!("no pages to process from {}", input.display());
    }

    let mut all = String::new();
    for (page, png) in &page_pngs {
        eprintln!("--- input page {page} ({} bytes) ---", png.len());
        let out = infer_page(backend, model, mtmd_ctx, ctx_params, png, cli)
            .with_context(|| format!("inference on input page {page}"))?;
        eprintln!("  page {page}: latency={:.2}s tokens={}", out.inference_secs, out.tokens);
        if page_pngs.len() > 1 {
            all.push_str(&format!("## Page {page}\n\n"));
        }
        all.push_str(out.markdown.trim());
        all.push_str("\n\n");
    }

    // markdown -> stdout; diagnostics -> stderr.
    print!("{}", all.trim_end());
    println!();
    eprintln!("--- real-document mode: no CER gate applied ---");
    Ok(())
}

// ---------------------------------------------------------------------------
// mtmd IMAGE inference (mirrors crates/asr-runtime, image instead of audio)
// ---------------------------------------------------------------------------

struct PageOutput {
    markdown: String,
    inference_secs: f64,
    tokens: usize,
}

fn infer_page(
    backend: &LlamaBackend,
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    ctx_params: &LlamaContextParams,
    png_bytes: &[u8],
    cli: &Cli,
) -> Result<PageOutput> {
    // Image bitmap from raw PNG bytes (stb_image decode inside mtmd). This is
    // the image analogue of MtmdBitmap::from_audio_data.
    let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, png_bytes)
        .map_err(|e| anyhow!("MtmdBitmap::from_buffer: {e:?}"))?;
    eprintln!(
        "  bitmap: is_audio={} {}x{}",
        bitmap.is_audio(),
        bitmap.nx(),
        bitmap.ny()
    );

    // Prompt: instruction + media marker in one user turn.
    let media_marker = mtmd_default_marker();
    let user_content = format!("{}\n{}", cli.prompt, media_marker);
    let prompt_text = build_prompt(model, &user_content)?;

    let input_text = MtmdInputText {
        text: prompt_text,
        add_special: true,
        parse_special: true,
    };

    // Fresh context per page (clean KV cache); MtmdContext reused.
    let mut llama_ctx = model
        .new_context(backend, ctx_params.clone())
        .map_err(|e| anyhow!("LlamaContext init: {e}"))?;

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .map_err(|e| anyhow!("mtmd tokenize: {e}"))?;
    eprintln!(
        "  tokenize: {} chunks, {} tokens, {} positions",
        chunks.len(),
        chunks.total_tokens(),
        chunks.total_positions(),
    );

    let infer_t0 = Instant::now();

    // Prefill: image chunk encoded via mtmd_encode inside eval_chunks, then the
    // text chunks via llama_decode.
    let mut n_past = chunks
        .eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, cli.n_batch, true)
        .map_err(|e| anyhow!("eval_chunks (PLE forward-graph crash? see llama.cpp #22243): {e}"))?;

    // Greedy decode with EOG stop.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut batch = LlamaBatch::new(cli.n_batch as usize, 1);
    let mut decoder = UTF_8.new_decoder();
    let mut markdown = String::new();
    let mut tokens = 0usize;

    for _ in 0..cli.max_tokens {
        let token = sampler.sample(&llama_ctx, -1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow!("token_to_piece: {e}"))?;
        markdown.push_str(&piece);
        tokens += 1;

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| anyhow!("batch.add: {e}"))?;
        n_past += 1;
        llama_ctx
            .decode(&mut batch)
            .map_err(|e| anyhow!("llama decode: {e}"))?;
    }

    Ok(PageOutput {
        markdown: markdown.trim().to_string(),
        inference_secs: infer_t0.elapsed().as_secs_f64(),
        tokens,
    })
}

fn build_prompt(model: &LlamaModel, user_content: &str) -> Result<String> {
    use llama_cpp_2::model::LlamaChatMessage;

    let msg = LlamaChatMessage::new("user".to_string(), user_content.to_string())
        .map_err(|e| anyhow!("LlamaChatMessage::new: {e}"))?;

    match model.chat_template(None::<&str>) {
        Ok(template) => match model.apply_chat_template(&template, &[msg], true) {
            Ok(rendered) => return Ok(rendered),
            Err(e) => eprintln!("apply_chat_template failed, ChatML fallback: {e}"),
        },
        Err(e) => eprintln!("no chat template ({e:?}); ChatML fallback"),
    }
    Ok(format!(
        "<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    ))
}

// ---------------------------------------------------------------------------
// PDF rasterisation (real --input only) via the auto-acquired PDFium
// ---------------------------------------------------------------------------

fn rasterise_pdf(
    input: &std::path::Path,
    pdfium_lib: &std::path::Path,
    cli: &Cli,
) -> Result<Vec<(u32, Vec<u8>)>> {
    use pdfium_render::prelude::*;

    let bindings = Pdfium::bind_to_library(pdfium_lib)
        .map_err(|e| anyhow!("Pdfium::bind_to_library({}): {e:?}", pdfium_lib.display()))?;
    let pdfium = Pdfium::new(bindings);

    let doc = pdfium
        .load_pdf_from_file(
            input.to_str().ok_or_else(|| anyhow!("non-UTF8 PDF path"))?,
            None,
        )
        .map_err(|e| anyhow!("opening PDF {}: {e:?}", input.display()))?;

    let total = doc.pages().len() as u32;
    eprintln!("PDF: {total} page(s)");
    let indices = parse_page_selection(&cli.pages, total)?;

    let render_config = PdfRenderConfig::new()
        .set_target_width((8.5 * cli.dpi as f64) as i32)
        .set_maximum_height((11.0 * cli.dpi as f64) as i32)
        .rotate_if_landscape(PdfPageRenderRotation::None, true);

    let mut out = Vec::new();
    for &idx in &indices {
        if idx >= total {
            bail!("page {} requested but PDF only has {} page(s)", idx + 1, total);
        }
        let page = doc
            .pages()
            .get(idx as i32)
            .map_err(|e| anyhow!("loading page {}: {e:?}", idx + 1))?;
        let img = page
            .render_with_config(&render_config)
            .map_err(|e| anyhow!("rendering page {}: {e:?}", idx + 1))?
            .as_image()
            .map_err(|e| anyhow!("page {} bitmap to image: {e:?}", idx + 1))?;
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .with_context(|| format!("PNG-encoding page {}", idx + 1))?;
        out.push((idx + 1, buf.into_inner()));
    }
    Ok(out)
}

fn parse_page_selection(spec: &str, total: u32) -> Result<Vec<u32>> {
    if spec.eq_ignore_ascii_case("all") {
        return Ok((0..total).collect());
    }
    let mut indices = Vec::new();
    for part in spec.split(',') {
        let n: u32 = part
            .trim()
            .parse()
            .with_context(|| format!("invalid page number: {part:?}"))?;
        if n == 0 {
            bail!("page numbers are 1-indexed; got 0");
        }
        indices.push(n - 1);
    }
    Ok(indices)
}

// ---------------------------------------------------------------------------
// CER + gate
// ---------------------------------------------------------------------------

struct GateRow {
    name: &'static str,
    cer: f64,
    latency: f64,
    ground_truth: String,
    prediction: String,
}

/// OmniDocBench-style character error rate: Levenshtein at the character level
/// normalised by max(|gt|, |pred|) so over-generation caps CER at 1.0.
fn char_error_rate(gt: &str, pred: &str) -> f64 {
    let denom = gt.chars().count().max(pred.chars().count()).max(1);
    strsim::levenshtein(gt, pred) as f64 / denom as f64
}

fn report_and_gate(rows: &[GateRow]) -> Result<()> {
    println!();
    println!("==== doc-vlm-spike synthetic gate ====");
    println!(
        "{:<14} {:>8} {:>10}  {}",
        "page", "CER", "latency_s", "verdict"
    );
    let mut all_pass = true;
    for r in rows {
        let pass = r.cer < CER_GATE && r.latency < LATENCY_GATE_SECS;
        all_pass &= pass;
        println!(
            "{:<14} {:>8.3} {:>10.2}  {}",
            r.name,
            r.cer,
            r.latency,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!(
        "thresholds: CER < {:.2}  latency < {:.0}s",
        CER_GATE, LATENCY_GATE_SECS
    );

    // Diagnostics for any failing page (truncated) to aid debugging.
    for r in rows {
        if !(r.cer < CER_GATE) {
            eprintln!("--- page '{}' diff ---", r.name);
            eprintln!("  ground truth: {}", truncate(&r.ground_truth, 200));
            eprintln!("  prediction  : {}", truncate(&r.prediction, 200));
        }
    }

    if all_pass {
        println!("RESULT: PASS (go)");
        Ok(())
    } else {
        println!("RESULT: FAIL (no-go)");
        // Non-zero exit via the run() -> main() error path.
        bail!("acceptance gate failed (see table above)")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cer_identical() {
        assert_eq!(char_error_rate("hello world", "hello world"), 0.0);
    }

    #[test]
    fn cer_single_substitution() {
        // 1 edit / 11 chars.
        let cer = char_error_rate("hello world", "hello_world");
        assert!((cer - 1.0 / 11.0).abs() < 1e-9, "got {cer}");
    }

    #[test]
    fn cer_over_generation_caps_at_one() {
        // Prediction much longer than gt; denominator = max -> capped.
        let cer = char_error_rate("abc", "abcdefghij");
        assert!(cer <= 1.0, "got {cer}");
        assert!((cer - 7.0 / 10.0).abs() < 1e-9, "got {cer}");
    }

    #[test]
    fn cer_empty_both() {
        assert_eq!(char_error_rate("", ""), 0.0);
    }

    #[test]
    fn parse_page_selection_all() {
        assert_eq!(parse_page_selection("all", 5).unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn parse_page_selection_comma_list() {
        assert_eq!(parse_page_selection("1,3,5", 10).unwrap(), vec![0, 2, 4]);
    }

    #[test]
    fn parse_page_selection_zero_is_error() {
        assert!(parse_page_selection("0", 5).is_err());
    }
}
