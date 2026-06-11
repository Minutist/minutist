//! Phase 0 Spike 2 - llama-cpp-2 text-only summarisation.
//!
//! End-to-end CLI: load a small instruct-tuned text-only GGUF, format a chat
//! prompt that asks for a markdown summary of a fixed paragraph + fake
//! transcript, greedy-decode, and emit the response on stdout.
//!
//! Guard-rail spike (Phase 0 Spike 2 / Q-P0-4): the information goal is
//! "does the same `llama-cpp-2` crate that Spike 1 uses for mtmd ASR also
//! cover the text-only inference path Phase 5 will need for the
//! summariser?" — not to pick the final summarisation model.
//!
//! Spike code; intentionally uses `anyhow`, `eprintln!`, sync `main`. See
//! `architecture/cross-cutting.md` — `spikes/` are exempt from production
//! rules.
//!
//! Acceptance: the Phase 0 Spike 2 gate. The output must
//! contain at least one markdown heading (`#`) and one bullet (`-` / `*`)
//! and wall-clock must be under 60 s.

use std::fs;
use std::io::Read;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Fixture. Phase 0 Spike 2 specifies "300-word paragraph plus a 5-segment
// fake transcript stored as Rust const / JSON literal. No external file."
//
// Subject matter is not load-bearing for the spike's information goal; this
// fixture imitates the structure the real summariser will see (free-form
// notes + diarised segments with timestamps) so the generated summary
// exercises both heading + bullet formatting.
// ---------------------------------------------------------------------------

/// 300-word paragraph. Hand-typed meeting note context; the model is meant
/// to treat this as the user's running notes for the meeting.
const FIXTURE_PARAGRAPH: &str = "\
Internal sync covering the Q3 backlog re-prioritisation for the minutist \
project. Andrew opened by reframing the goal: the team is shipping a \
local-first desktop notes app that pairs hand-typed notes with a Whisper-style \
transcript, and the v1 release must run end-to-end on a single laptop without \
network access. The current sprint is focused on closing Phase 0 gating \
spikes. Spike 1 (mtmd ASR via llama-cpp-2) has landed and produced correct \
transcripts on the librispeech and JFK fixtures at zero WER, with a peak \
RSS of about two gigabytes. Spike 2 (text-only LLM through the same crate) \
is the topic of this meeting. Spike 3 (VAD-driven streaming) is blocked \
behind Spike 2 only by reviewer bandwidth. Diarisation work for Spike 4 is \
parked until next week. Risks raised: the workspace Cargo.lock currently \
holds two majors of an unrelated crate; vendoring decisions for sherpa-onnx \
are still open; and the binary footprint estimate of the eventual bundled \
desktop app is unknown because phases 5 and 6 have not been measured yet. \
Beth flagged that the user-facing first-run download flow is still \
unscoped — we agreed it would land in Phase 1 alongside the model registry \
crate. Action items: Andrew to write up Spike 2 results in the planning \
journal once the spike is green; Carl to pull the latest llama.cpp release \
notes into the journal and check for relevant fixes since 0.1.146; Beth to \
draft a download-progress UI mock in the design doc; Dee to chase the \
sherpa-onnx upstream maintainer about the bindgen build issue she found \
last Friday. Next sync is the regular Thursday slot.";

/// Five-segment fake transcript, JSON-array literal. Each segment carries
/// `start_ms`, `end_ms`, a `speaker` label, and the `text`. This mirrors
/// the production `Segment` shape closely enough for the spike — the real
/// summariser in Phase 5 will receive `minutist_common::Segment` values
/// serialised with `serde_json`, not this exact shape.
const FIXTURE_TRANSCRIPT_JSON: &str = r#"[
  {"start_ms":     0, "end_ms":  6800, "speaker": "Andrew",
   "text": "Welcome back. The agenda for today is short: we land Spike 2 results so Spike 3 can unblock, then we walk the open risks."},
  {"start_ms":  7100, "end_ms": 18400, "speaker": "Beth",
   "text": "I want to call out the first-run download flow. Right now nobody owns it. If we leave it open into Phase 1 we'll repeat the registry argument from last sprint, so I'd like it scoped in writing this week."},
  {"start_ms": 18600, "end_ms": 30100, "speaker": "Carl",
   "text": "Two upstream points. First, llama.cpp shipped a streaming mtmd PR last week. Second, the 0.1.146 crate doesn't record the embedded llama.cpp commit. Worth investigating if there's a newer pin that exposes it."},
  {"start_ms": 30400, "end_ms": 42700, "speaker": "Dee",
   "text": "Sherpa-onnx binding is the risk I keep coming back to. The Rust wrapper hasn't been updated in three months. I'm going to email the upstream maintainer; if no answer by Tuesday we should plan a bindgen wrapper instead."},
  {"start_ms": 43000, "end_ms": 56200, "speaker": "Andrew",
   "text": "Agreed on all three. Action items: I'll write up Spike 2; Carl checks llama.cpp release notes; Beth drafts the download UI mock; Dee chases sherpa-onnx. Same Thursday slot next week."}
]"#;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "spike-llm",
    about = "Phase 0 Spike 2: llama-cpp-2 text-only summarisation"
)]
struct Cli {
    /// Path to the instruct-tuned GGUF (e.g. Qwen3.5-9B Q8_0).
    #[arg(short = 'm', long, value_name = "PATH")]
    model: PathBuf,

    /// Maximum tokens to generate during decode. The acceptance run uses
    /// 256; longer is fine but counts against the 60 s wall-clock budget.
    #[arg(long, default_value_t = 256)]
    max_tokens: i32,

    /// Context size (tokens). The fixture prompt is well under 2 k tokens
    /// for any reasonable tokeniser; 4096 leaves headroom.
    #[arg(long, default_value_t = NonZeroU32::new(4096).unwrap())]
    n_ctx: NonZeroU32,

    /// Decode batch size for prefill.
    #[arg(long, default_value_t = 512)]
    n_batch: i32,

    /// Number of CPU threads for inference.
    #[arg(short = 't', long, default_value_t = 8)]
    threads: i32,

    /// Disable GPU. Default on WSL Linux (WSL2 paravirt GPU is unreliable).
    #[arg(long, default_value_t = true)]
    no_gpu: bool,

    /// If set, bypass the model's baked-in chat template and use a
    /// hand-built ChatML scaffold instead. Lets us characterise Q-P0-4 by
    /// running both paths from the command line.
    #[arg(long, default_value_t = false)]
    force_manual_chatml: bool,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    // Spike-grade logging: stderr, INFO by default, overridable via RUST_LOG.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    eprintln!(
        "spike-llm: model={} max_tokens={} n_ctx={} threads={}",
        cli.model.display(),
        cli.max_tokens,
        cli.n_ctx,
        cli.threads,
    );

    if !cli.model.exists() {
        bail!("model GGUF does not exist: {}", cli.model.display());
    }

    // SHA-256 of the GGUF for README provenance.
    let model_sha = sha256_of(&cli.model).context("hashing model gguf")?;
    eprintln!("model sha256 = {model_sha}");

    // -- backend / model ----------------------------------------------------
    let load_t0 = Instant::now();
    let backend = LlamaBackend::init().map_err(|e| anyhow!("LlamaBackend::init failed: {e}"))?;
    eprintln!("backend init: {:?}", load_t0.elapsed());

    let mut model_params = LlamaModelParams::default();
    if cli.no_gpu {
        // Explicit 0 GPU layers; on the CPU-only build this is the default
        // but pinning it documents the intent.
        model_params = model_params.with_n_gpu_layers(0);
    }

    let model_t0 = Instant::now();
    let model = LlamaModel::load_from_file(&backend, &cli.model, &model_params)
        .map_err(|e| anyhow!("LlamaModel::load_from_file failed: {e}"))?;
    let model_load_secs = model_t0.elapsed();
    eprintln!("model load: {:?}", model_load_secs);

    // -- chat-template handling (Q-P0-4) -----------------------------------
    //
    // The system prompt asks for markdown with a heading and a bullet list.
    // The user turn includes the paragraph + the JSON transcript so the
    // model has both modalities of meeting context.
    let system_prompt = "You are a meeting-note assistant. Read the user's running notes and the diarised transcript. \
Produce a concise markdown summary. The output MUST include at least one Markdown heading (a line starting with '#') \
and at least one bullet list (lines starting with '-' or '*'). Use the heading for the meeting topic and bullets for \
the agreed action items. Do not invent facts that are not in the input.";

    let user_content = format!(
        "## Notes\n\n{paragraph}\n\n## Transcript (JSON segments)\n\n```json\n{transcript}\n```\n\n\
Please summarise the meeting in Markdown.",
        paragraph = FIXTURE_PARAGRAPH,
        transcript = FIXTURE_TRANSCRIPT_JSON,
    );

    let (prompt_text, chat_template_path) = build_prompt(
        &model,
        system_prompt,
        &user_content,
        cli.force_manual_chatml,
    )?;
    eprintln!("chat template path: {chat_template_path}");
    eprintln!("prompt length (chars): {}", prompt_text.len());

    // -- context ------------------------------------------------------------
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(cli.n_ctx))
        .with_n_batch(cli.n_batch as u32)
        .with_n_threads(cli.threads)
        .with_n_threads_batch(cli.threads);

    let ctx_t0 = Instant::now();
    let mut llama_ctx: LlamaContext = model
        .new_context(&backend, ctx_params)
        .map_err(|e| anyhow!("LlamaContext init failed: {e}"))?;
    eprintln!("context init: {:?}", ctx_t0.elapsed());

    // -- tokenise + prefill --------------------------------------------------
    //
    // AddBos::Never because the chat template (or the manual ChatML fallback)
    // already places the BOS or its analogue. Letting the tokeniser add a
    // second BOS shifts position 0 and Qwen will sometimes emit garbage.
    let prompt_tokens = model
        .str_to_token(&prompt_text, AddBos::Never)
        .map_err(|e| anyhow!("str_to_token failed: {e}"))?;
    eprintln!("prompt tokenised: {} tokens", prompt_tokens.len());

    let n_ctx_usize = usize::try_from(u32::from(cli.n_ctx)).unwrap();
    if prompt_tokens.len() >= n_ctx_usize {
        bail!(
            "prompt ({} tokens) does not fit in n_ctx={}",
            prompt_tokens.len(),
            n_ctx_usize
        );
    }

    let infer_t0 = Instant::now();

    // Prefill the prompt in `n_batch`-sized chunks. llama.cpp asserts
    // `n_tokens <= cparams.n_batch` per decode call (see llama-context.cpp
    // 1599); a single batch covering the whole prompt fails for any
    // realistic prompt size. Only the very last token of the last chunk
    // gets `logits = true` because that is the position we sample from
    // first.
    let n_batch = cli.n_batch as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let prefill_t0 = Instant::now();
    let mut pos: i32 = 0;
    for chunk in prompt_tokens.chunks(n_batch) {
        batch.clear();
        let chunk_end_pos = pos + chunk.len() as i32; // exclusive
        for tok in chunk {
            let is_last_token_of_prompt = pos + 1 == prompt_tokens.len() as i32;
            batch
                .add(*tok, pos, &[0], is_last_token_of_prompt)
                .map_err(|e| anyhow!("batch.add (prefill): {e}"))?;
            pos += 1;
        }
        debug_assert_eq!(pos, chunk_end_pos);
        llama_ctx
            .decode(&mut batch)
            .map_err(|e| anyhow!("decode (prefill chunk): {e}"))?;
    }
    let prefill_secs = prefill_t0.elapsed().as_secs_f64();
    eprintln!(
        "prefill: {} tokens in {:.3} s ({:.1} tok/s)",
        prompt_tokens.len(),
        prefill_secs,
        prompt_tokens.len() as f64 / prefill_secs.max(1e-6),
    );

    // -- generation loop ----------------------------------------------------
    //
    // Greedy decoding. Summarisation is largely deterministic when the prompt
    // is concrete; greedy keeps the spike reproducible and side-steps a
    // separate seed question.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut decoder = UTF_8.new_decoder();
    let mut output = String::new();
    let mut tokens_generated = 0usize;
    let mut n_past = pos; // position of the next token to be inserted

    let gen_t0 = Instant::now();
    for _ in 0..cli.max_tokens {
        let token = sampler.sample(&llama_ctx, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        // `special: true` so we can spot any leaked control tokens in the
        // README; if they appear it means the template path is wrong.
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow!("token_to_piece: {e}"))?;
        output.push_str(&piece);
        tokens_generated += 1;

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| anyhow!("batch.add (gen): {e}"))?;
        n_past += 1;

        llama_ctx
            .decode(&mut batch)
            .map_err(|e| anyhow!("decode (gen): {e}"))?;
    }
    let gen_secs = gen_t0.elapsed().as_secs_f64();
    let inference_secs = infer_t0.elapsed().as_secs_f64();

    eprintln!(
        "generation: {} tokens in {:.3} s ({:.1} tok/s)",
        tokens_generated,
        gen_secs,
        tokens_generated as f64 / gen_secs.max(1e-6),
    );
    eprintln!(
        "inference wall-clock (prefill+gen): {:.3} s",
        inference_secs
    );

    // The summary itself goes to stdout; everything else goes to stderr so a
    // future `--quiet` flag could pipe just the markdown.
    println!("{}", output.trim_end());

    // -- measurements --------------------------------------------------------
    let peak_rss = read_peak_rss_kib();
    if let Some(kib) = peak_rss {
        eprintln!("peak RSS (VmHWM): {:.1} MiB", kib as f64 / 1024.0);
    }

    // -- acceptance gate -----------------------------------------------------
    let mut failures = Vec::new();
    if inference_secs > 60.0 {
        failures.push(format!(
            "inference wall-clock {:.3}s > 60s budget",
            inference_secs
        ));
    }
    if !contains_markdown_heading(&output) {
        failures.push(
            "output does not contain a markdown heading line (one starting with '#')".to_string(),
        );
    }
    if !contains_markdown_bullet(&output) {
        failures.push(
            "output does not contain a markdown bullet line (one starting with '-' or '*')"
                .to_string(),
        );
    }

    eprintln!("---");
    if !failures.is_empty() {
        eprintln!("ACCEPTANCE FAIL:");
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(2);
    }

    eprintln!("ACCEPTANCE PASS");
    Ok(())
}

// ---------------------------------------------------------------------------
// Chat-template handling. Q-P0-4: `llama-cpp-2 = 0.1.146` exposes both
// `LlamaModel::chat_template(name: Option<&str>)` (returns the model's
// baked-in template, an FFI wrapper around `llama_model_chat_template`) and
// `LlamaModel::apply_chat_template(template, messages, add_ass)` (wraps
// `llama_chat_apply_template`). The spike prefers that path; if the model
// has no template baked in or apply fails, it falls back to a hand-built
// ChatML scaffold.
// ---------------------------------------------------------------------------

/// Returns `(prompt_text, path_label)` where `path_label` is one of
/// `"baked-template"` or `"manual-chatml"` for README accounting.
fn build_prompt(
    model: &LlamaModel,
    system_prompt: &str,
    user_content: &str,
    force_manual: bool,
) -> Result<(String, &'static str)> {
    if !force_manual {
        match model.chat_template(None::<&str>) {
            Ok(template) => {
                let messages = vec![
                    LlamaChatMessage::new("system".to_string(), system_prompt.to_string())
                        .map_err(|e| anyhow!("LlamaChatMessage::new(system): {e}"))?,
                    LlamaChatMessage::new("user".to_string(), user_content.to_string())
                        .map_err(|e| anyhow!("LlamaChatMessage::new(user): {e}"))?,
                ];
                match model.apply_chat_template(&template, &messages, /* add_ass */ true) {
                    Ok(rendered) => return Ok((rendered, "baked-template")),
                    Err(e) => {
                        eprintln!(
                            "apply_chat_template failed ({e}); falling back to manual ChatML"
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "model has no baked-in chat template ({e:?}); falling back to manual ChatML"
                );
            }
        }
    } else {
        eprintln!("--force-manual-chatml set; skipping baked-in template");
    }

    // ChatML fallback. Qwen models follow this exactly; Gemma uses a
    // different scheme so the fallback is best-effort, not guaranteed.
    let rendered = format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n",
        system = system_prompt,
        user = user_content,
    );
    Ok((rendered, "manual-chatml"))
}

// ---------------------------------------------------------------------------
// Acceptance helpers
// ---------------------------------------------------------------------------

/// Acceptance check: at least one line starts with `#` (optionally after
/// whitespace) and is followed by space + content. `####### no` counts;
/// `#hashtag` (no space) doesn't because that's not how Markdown headings
/// render.
fn contains_markdown_heading(s: &str) -> bool {
    s.lines().any(|line| {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('#') {
            return false;
        }
        // Skip the run of '#' chars and require at least one space-delimited
        // word after them.
        let after_hash = trimmed.trim_start_matches('#');
        after_hash.starts_with(' ')
            && after_hash
                .trim_start()
                .contains(|c: char| !c.is_whitespace())
    })
}

/// Acceptance check: at least one line is a Markdown bullet — starts with
/// `-`, `*`, or `+` followed by a space and content. `*emphasis*` doesn't
/// count; `* item` does.
fn contains_markdown_bullet(s: &str) -> bool {
    s.lines().any(|line| {
        let trimmed = line.trim_start();
        let after = if let Some(rest) = trimmed.strip_prefix("- ") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("* ") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("+ ") {
            rest
        } else {
            return false;
        };
        after.contains(|c: char| !c.is_whitespace())
    })
}

// ---------------------------------------------------------------------------
// Misc utilities: SHA-256 and /proc/self/status peak RSS. Both copied
// verbatim from spikes/asr/src/main.rs so the two spikes report identically.
// ---------------------------------------------------------------------------

fn sha256_of(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).unwrap();
    }
    Ok(s)
}

fn read_peak_rss_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            // Format: "VmHWM:\t   12345 kB"
            let s = rest.trim();
            let kib_str = s.strip_suffix(" kB").unwrap_or(s).trim();
            return kib_str.parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_detector_accepts_h1() {
        assert!(contains_markdown_heading("# Title\nbody"));
    }

    #[test]
    fn heading_detector_accepts_indented_h2() {
        assert!(contains_markdown_heading("  ## Sub-heading\nbody"));
    }

    #[test]
    fn heading_detector_rejects_hashtag() {
        assert!(!contains_markdown_heading("#hashtag without space"));
    }

    #[test]
    fn heading_detector_rejects_plain_text() {
        assert!(!contains_markdown_heading("no markdown at all"));
    }

    #[test]
    fn bullet_detector_accepts_dash() {
        assert!(contains_markdown_bullet("- item one\n- item two"));
    }

    #[test]
    fn bullet_detector_accepts_star() {
        assert!(contains_markdown_bullet("* item one"));
    }

    #[test]
    fn bullet_detector_accepts_plus() {
        assert!(contains_markdown_bullet("+ item one"));
    }

    #[test]
    fn bullet_detector_rejects_emphasis() {
        assert!(!contains_markdown_bullet("*emphasis* in a paragraph"));
    }

    #[test]
    fn bullet_detector_rejects_subtraction() {
        assert!(!contains_markdown_bullet("3 - 1 = 2"));
    }

    #[test]
    fn fixture_paragraph_has_300ish_words() {
        // Phase 0 Spike 2: "300-word paragraph". Allow some slack so the
        // exact value isn't load-bearing on tweaks to the fixture text.
        let words = FIXTURE_PARAGRAPH.split_whitespace().count();
        assert!(
            (250..=350).contains(&words),
            "fixture paragraph is {words} words; expected ~300"
        );
    }

    #[test]
    fn fixture_transcript_has_five_segments() {
        // Document the fixture invariant. If the JSON is hand-edited later,
        // this test catches accidental drops.
        let v: serde_json::Value = serde_json::from_str(FIXTURE_TRANSCRIPT_JSON)
            .expect("fixture transcript must parse as JSON");
        let arr = v
            .as_array()
            .expect("fixture transcript must be a JSON array");
        assert_eq!(arr.len(), 5, "fixture transcript must have 5 segments");
    }
}
