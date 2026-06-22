//! doc-vlm-spike — fully-automated head-to-head doc-to-markdown go/no-go.
//!
//! Benchmarks Gemma-4-E4B (a generic chat VLM) against PaddleOCR-VL-1.6 (a
//! doc-OCR specialist) on the SAME synthetic pages, to decide which model
//! should fill the VLM fallback seam that `crates/doc-convert` deliberately
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
//! self-acquires each registered model's LM + mmproj GGUFs and a PDFium prebuilt
//! into the OS cache dir, renders known synthetic pages with exact ground
//! truth, runs EVERY page through EVERY model via mtmd IMAGE inference, scores
//! CER + latency, prints a side-by-side comparison + per-model PASS/FAIL +
//! the winning model, and exits 0 (every model passes) / non-zero otherwise.
//!
//! # Per-model prompting (load-bearing)
//!
//! The two models need OPPOSITE image-marker placement. Gemma takes a verbose
//! "convert to markdown" instruction with the `<__media__>` marker AFTER it,
//! rendered through its chat template. PaddleOCR-VL is trained on bare task
//! prefixes (`OCR:`, `Table Recognition:`) with the marker BEFORE the prefix
//! inside an ERNIE-4.5 turn. Each `ModelSpec` in `models.rs` carries its own
//! instruction + prompt-assembly so the two coexist.
//!
//! # mtmd IMAGE path
//!
//! `MtmdBitmap::from_buffer(png)` -> `tokenize(text, &[&bitmap])` with the media
//! marker -> `eval_chunks` prefill -> greedy decode with EOG stop. Same C
//! bindings as the audio path in `crates/asr-runtime`, fed a page image.
//!
//! # Caveats (record on first Vulkan run)
//!
//! * Gemma-4 PLE forward-graph issue (llama.cpp #22243) may affect the vision
//!   graph. If vision inference crashes on Vulkan, a Gemma-3 multimodal GGUF
//!   (`ggml-org/gemma-3-4b-it-GGUF`, no PLE) is the control.
//! * PaddleOCR-VL support landed via llama.cpp PR #18825 (mrope + the
//!   `<__media__>OCR:` template). If `MtmdContext::init_from_file` rejects the
//!   projector or `support_vision()` is false, the vendored llama.cpp predates
//!   the PR and `llama-cpp-2` must be bumped.

mod acquire;
mod models;
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

use crate::models::{ModelSpec, REGISTRY};

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

    /// Override the auto-acquired LM GGUF for the FIRST registered model
    /// (Gemma-4); other models still self-acquire. Mostly for local debugging.
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Override the auto-acquired vision mmproj GGUF for the FIRST registered
    /// model (Gemma-4); other models still self-acquire.
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

    eprintln!(
        "spike-doc-vlm: GPU build={GPU_BUILD} n_gpu_layers={n_gpu_layers} models={}",
        REGISTRY.len()
    );

    // --- Acquire every model up front (cache + skip when present) ---
    // Resolve all artifacts before loading anything so a download failure on
    // model 2 does not strand us mid-run after model 1's inference.
    let mut resolved: Vec<(&'static ModelSpec, acquire::ModelPaths)> = Vec::new();
    for (i, spec) in REGISTRY.iter().enumerate() {
        let paths = resolve_model_paths(spec, &cli, i == 0)
            .with_context(|| format!("acquiring artifacts for {}", spec.display_name))?;
        eprintln!(
            "[{}] LM={} mmproj={}",
            spec.display_name,
            paths.lm.display(),
            paths.mmproj.display()
        );
        resolved.push((spec, paths));
    }

    let backend = LlamaBackend::init().map_err(|e| anyhow!("LlamaBackend::init: {e}"))?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(cli.n_ctx))
        .with_n_batch(cli.n_batch as u32)
        .with_n_threads(cli.threads)
        .with_n_threads_batch(cli.threads);

    // --- Real-document mode: run through EVERY model, print output only ---
    if let Some(input) = &cli.input {
        return run_real_input(input, &backend, &ctx_params, &resolved, &cli, n_gpu_layers);
    }

    // --- Synthetic gate mode: every page through every model ---
    let pages = synth::build_pages().context("building synthetic fixtures")?;
    eprintln!("synthetic pages: {}", pages.len());

    let mut results: Vec<GateRow> = Vec::new();
    for (spec, paths) in &resolved {
        eprintln!("\n==== model: {} ====", spec.display_name);
        let loaded = load_model(&backend, spec, paths, &cli, n_gpu_layers)
            .with_context(|| format!("loading {}", spec.display_name))?;

        for page in &pages {
            eprintln!(
                "--- [{}] page '{}' ({} bytes PNG) ---",
                spec.display_name,
                page.name,
                page.png.len()
            );
            let out = infer_page(
                &backend,
                &loaded.model,
                &loaded.mtmd_ctx,
                &ctx_params,
                spec,
                page.name,
                &page.png,
                &cli,
            )
            .with_context(|| {
                format!("inference on '{}' with {}", page.name, spec.display_name)
            })?;

            let pred = synth::normalise(&out.markdown);
            let cer = char_error_rate(&page.ground_truth, &pred);
            eprintln!(
                "  [{}] '{}': CER={:.3} latency={:.2}s tokens={}",
                spec.display_name, page.name, cer, out.inference_secs, out.tokens
            );
            results.push(GateRow {
                model: spec.display_name,
                page: page.name,
                instruction: spec.instruction_for(page.name),
                cer,
                latency: out.inference_secs,
                ground_truth: page.ground_truth.clone(),
                prediction: pred,
            });
        }
        // `loaded` (model + mtmd context) drops here, freeing VRAM/RAM before
        // the next model loads — the two LMs are not co-resident.
    }

    report_and_gate(&pages, &results)
}

/// Resolve a model's LM + mmproj paths, honouring the `--model`/`--mmproj`
/// debugging overrides for the first registered model only.
fn resolve_model_paths(
    spec: &ModelSpec,
    cli: &Cli,
    is_first: bool,
) -> Result<acquire::ModelPaths> {
    if is_first && (cli.model.is_some() || cli.mmproj.is_some()) {
        // Overrides apply to the first model; fall back to self-acquired paths
        // for whichever of LM/mmproj is not overridden.
        let acquired = spec.acquire()?;
        let lm = match &cli.model {
            Some(p) => {
                if !p.exists() {
                    bail!("--model {} does not exist", p.display());
                }
                p.clone()
            }
            None => acquired.lm,
        };
        let mmproj = match &cli.mmproj {
            Some(p) => {
                if !p.exists() {
                    bail!("--mmproj {} does not exist", p.display());
                }
                p.clone()
            }
            None => acquired.mmproj,
        };
        Ok(acquire::ModelPaths { lm, mmproj })
    } else {
        spec.acquire()
    }
}

/// A loaded model: the LM plus its mtmd vision context. The two are kept
/// together so one model's resources drop as a unit before the next loads.
struct LoadedModel {
    model: LlamaModel,
    mtmd_ctx: MtmdContext,
}

/// Load one model's LM + mtmd context and assert it advertises vision support.
fn load_model(
    backend: &LlamaBackend,
    spec: &ModelSpec,
    paths: &acquire::ModelPaths,
    cli: &Cli,
    n_gpu_layers: u32,
) -> Result<LoadedModel> {
    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let model_t0 = Instant::now();
    let model = LlamaModel::load_from_file(backend, &paths.lm, &model_params)
        .map_err(|e| anyhow!("LlamaModel::load_from_file ({}): {e}", paths.lm.display()))?;
    eprintln!("[{}] model load: {:?}", spec.display_name, model_t0.elapsed());

    let mtmd_params = MtmdContextParams {
        // Per-model GPU affinity, gated on actually having a GPU build/offload.
        use_gpu: spec.use_gpu && n_gpu_layers > 0,
        print_timings: true,
        n_threads: cli.threads,
        // The default marker `<__media__>` is correct for BOTH models (Gemma
        // appends it, PaddleOCR prepends it); only its placement differs, and
        // that is handled in `ModelSpec::build_prompt`, not here.
        media_marker: CString::new(mtmd_default_marker())
            .map_err(|e| anyhow!("media marker CString: {e}"))?,
    };
    let mmproj_str = paths
        .mmproj
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF8 mmproj path"))?;

    let mtmd_t0 = Instant::now();
    let mtmd_ctx = MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params)
        .map_err(|e| anyhow!("MtmdContext::init_from_file ({mmproj_str}): {e}"))?;
    eprintln!("[{}] mtmd init: {:?}", spec.display_name, mtmd_t0.elapsed());
    eprintln!(
        "[{}] mtmd: supports_vision={} supports_audio={}",
        spec.display_name,
        mtmd_ctx.support_vision(),
        mtmd_ctx.support_audio(),
    );

    // Hard bail: image inference is meaningless without a vision encoder.
    // (PaddleOCR support needs a post-PR-#18825 llama.cpp; a false here on that
    //  model means the vendored llama.cpp predates the PR.)
    if !mtmd_ctx.support_vision() {
        bail!(
            "{}: projector {} does not advertise vision support — either it is an \
             audio projector, or (for PaddleOCR-VL) the vendored llama.cpp predates \
             PR #18825 and llama-cpp-2 must be bumped.",
            spec.display_name,
            paths.mmproj.display()
        );
    }

    Ok(LoadedModel { model, mtmd_ctx })
}

// ---------------------------------------------------------------------------
// Real --input document mode (no ground truth)
// ---------------------------------------------------------------------------

fn run_real_input(
    input: &std::path::Path,
    backend: &LlamaBackend,
    ctx_params: &LlamaContextParams,
    resolved: &[(&'static ModelSpec, acquire::ModelPaths)],
    cli: &Cli,
    n_gpu_layers: u32,
) -> Result<()> {
    if !input.exists() {
        bail!("--input {} does not exist", input.display());
    }
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // Rasterise once (model-independent), then run every model over the pages.
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

    let mut out_doc = String::new();
    for (spec, paths) in resolved {
        eprintln!("\n==== model: {} ====", spec.display_name);
        let loaded = load_model(backend, spec, paths, cli, n_gpu_layers)
            .with_context(|| format!("loading {}", spec.display_name))?;

        out_doc.push_str(&format!("# {} — {}\n\n", spec.display_name, input.display()));
        for (page, png) in &page_pngs {
            eprintln!("--- [{}] input page {page} ({} bytes) ---", spec.display_name, png.len());
            // No ground truth -> the page name is not "table", so each model
            // uses its general instruction (`OCR:` for PaddleOCR).
            let out = infer_page(
                backend,
                &loaded.model,
                &loaded.mtmd_ctx,
                ctx_params,
                spec,
                "input",
                png,
                cli,
            )
            .with_context(|| format!("inference on input page {page} with {}", spec.display_name))?;
            eprintln!(
                "  [{}] page {page}: latency={:.2}s tokens={}",
                spec.display_name, out.inference_secs, out.tokens
            );
            if page_pngs.len() > 1 {
                out_doc.push_str(&format!("## Page {page}\n\n"));
            }
            out_doc.push_str(out.markdown.trim());
            out_doc.push_str("\n\n");
        }
    }

    // markdown -> stdout; diagnostics -> stderr.
    print!("{}", out_doc.trim_end());
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

#[allow(clippy::too_many_arguments)]
fn infer_page(
    backend: &LlamaBackend,
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    ctx_params: &LlamaContextParams,
    spec: &ModelSpec,
    page_name: &str,
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

    // Per-model prompt assembly. Gemma puts the marker AFTER its verbose
    // instruction (via chat template); PaddleOCR puts it BEFORE the bare task
    // prefix inside an ERNIE turn. `tokenize` splits the text on the marker and
    // inserts the image chunk at that position, so placement is what differs.
    let media_marker = mtmd_default_marker();
    let prompt_text = spec.build_prompt(model, page_name, media_marker)?;
    eprintln!("  prompt[{}]: {}", spec.display_name, truncate(&prompt_text, 160));

    let input_text = MtmdInputText {
        text: prompt_text,
        add_special: spec.add_special(),
        parse_special: spec.parse_special(),
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

/// One (model x page) measurement.
struct GateRow {
    model: &'static str,
    page: &'static str,
    /// The instruction prefix this model used on this page (documents the
    /// like-for-like vs on-spec prompt choice in the comparison).
    instruction: &'static str,
    cer: f64,
    latency: f64,
    ground_truth: String,
    prediction: String,
}

impl GateRow {
    /// A page passes the gate iff CER is under the gate AND it ran in under the
    /// per-page latency budget.
    fn pass(&self) -> bool {
        self.cer < CER_GATE && self.latency < LATENCY_GATE_SECS
    }
}

/// OmniDocBench-style character error rate: Levenshtein at the character level
/// normalised by max(|gt|, |pred|) so over-generation caps CER at 1.0.
fn char_error_rate(gt: &str, pred: &str) -> f64 {
    let denom = gt.chars().count().max(pred.chars().count()).max(1);
    strsim::levenshtein(gt, pred) as f64 / denom as f64
}

/// Per-model rollup over the synthetic pages.
struct ModelSummary {
    model: &'static str,
    mean_cer: f64,
    mean_latency: f64,
    /// PASS iff EVERY synthetic page passed the gate for this model.
    pass: bool,
}

/// Side-by-side report: a model x page CER/latency grid, per-model PASS/FAIL,
/// and the winning model on the synthetic pages (lower mean CER, then lower
/// mean latency). Returns Ok iff every model passes; otherwise bails so the
/// process exits non-zero.
fn report_and_gate(pages: &[synth::SyntheticPage], rows: &[GateRow]) -> Result<()> {
    let model_names: Vec<&'static str> = REGISTRY.iter().map(|m| m.display_name).collect();

    let lookup = |model: &str, page: &str| -> Option<&GateRow> {
        rows.iter().find(|r| r.model == model && r.page == page)
    };

    println!();
    println!("==== doc-vlm-spike head-to-head (model x page) ====");

    // One column per model; rows are pages.
    let col_w = 30usize;
    print!("{:<14}", "page");
    for m in &model_names {
        print!(" | {:<width$}", m, width = col_w);
    }
    println!();
    for page in pages {
        print!("{:<14}", page.name);
        for m in &model_names {
            let cell = match lookup(m, page.name) {
                Some(r) => format!(
                    "CER={:.3} {:.1}s {} [{}]",
                    r.cer,
                    r.latency,
                    if r.pass() { "PASS" } else { "FAIL" },
                    r.instruction.trim_end_matches(':')
                ),
                None => "—".to_string(),
            };
            print!(" | {cell:<col_w$}");
        }
        println!();
    }
    println!(
        "thresholds: CER < {:.2}  latency < {:.0}s/page  (synthetic pages only)",
        CER_GATE, LATENCY_GATE_SECS
    );

    // Per-model summary over the synthetic pages.
    let mut summaries: Vec<ModelSummary> = Vec::new();
    for m in &model_names {
        let model_rows: Vec<&GateRow> = rows.iter().filter(|r| r.model == *m).collect();
        if model_rows.is_empty() {
            continue;
        }
        let n = model_rows.len() as f64;
        let mean_cer = model_rows.iter().map(|r| r.cer).sum::<f64>() / n;
        let mean_latency = model_rows.iter().map(|r| r.latency).sum::<f64>() / n;
        let pass = model_rows.iter().all(|r| r.pass());
        summaries.push(ModelSummary {
            model: m,
            mean_cer,
            mean_latency,
            pass,
        });
    }

    println!();
    println!("---- per-model summary (synthetic pages) ----");
    println!("{:<18} {:>9} {:>11}  {}", "model", "mean_CER", "mean_lat_s", "verdict");
    for s in &summaries {
        println!(
            "{:<18} {:>9.3} {:>11.2}  {}",
            s.model,
            s.mean_cer,
            s.mean_latency,
            if s.pass { "PASS" } else { "FAIL" }
        );
    }

    // Diagnostics for any failing (model x page) cell to aid debugging.
    for r in rows {
        if !(r.cer < CER_GATE) {
            eprintln!("--- [{}] page '{}' diff (prompt: {:?}) ---", r.model, r.page, r.instruction);
            eprintln!("  ground truth: {}", truncate(&r.ground_truth, 200));
            eprintln!("  prediction  : {}", truncate(&r.prediction, 200));
        }
    }

    // Winner: lowest mean CER, tie-broken by lowest mean latency.
    if let Some(winner) = summaries.iter().min_by(|a, b| {
        a.mean_cer
            .partial_cmp(&b.mean_cer)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.mean_latency
                    .partial_cmp(&b.mean_latency)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    }) {
        println!();
        println!(
            "WINNER (synthetic pages): {} (mean CER={:.3}, mean latency={:.2}s)",
            winner.model, winner.mean_cer, winner.mean_latency
        );
    }

    let all_pass = !summaries.is_empty() && summaries.iter().all(|s| s.pass);
    if all_pass {
        println!("RESULT: PASS (every model meets the gate)");
        Ok(())
    } else {
        println!("RESULT: FAIL (at least one model misses the gate)");
        // Non-zero exit via the run() -> main() error path.
        bail!("acceptance gate failed for at least one model (see tables above)")
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
