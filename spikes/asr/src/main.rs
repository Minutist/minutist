//! Phase 0 Spike 1 - llama-cpp-2 mtmd ASR.
//!
//! End-to-end CLI: feed a ~10 s 16 kHz mono WAV into the llama.cpp mtmd
//! audio path backed by a Qwen3-ASR GGUF + mmproj pair, decode greedily,
//! print the transcript, and report WER + wall-clock + peak RSS.
//!
//! This is throwaway spike code (see `architecture/cross-cutting.md`); it intentionally uses `anyhow`,
//! `eprintln!`, and a single sync `main` rather than the production
//! cross-cutting rules.
//!
//! Acceptance criteria: the Phase 0 Spike 1 gate.
//! Open questions answered in `spikes/asr/README.md`.

use std::ffi::CString;
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
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "spike-asr",
    about = "Phase 0 Spike 1: llama-cpp-2 mtmd Qwen3-ASR end-to-end"
)]
struct Cli {
    /// Path to the Qwen3-ASR main GGUF (e.g. Qwen3-ASR-0.6B-Q8_0).
    #[arg(short = 'm', long, value_name = "PATH")]
    model: PathBuf,

    /// Path to the multimodal projector GGUF (mmproj).
    #[arg(long, value_name = "PATH")]
    mmproj: PathBuf,

    /// 16 kHz mono WAV file. PCM s16 or f32 accepted.
    #[arg(short = 'w', long, value_name = "PATH")]
    wav: PathBuf,

    /// Truncate the input audio to this many seconds before feeding mtmd.
    /// llama.cpp's mtmd path pads sub-30s audio internally to 30s, so this
    /// only bounds compute, not the encoder window.
    #[arg(long, value_name = "SEC", default_value_t = 10.0)]
    max_seconds: f32,

    /// Optional reference transcript file for WER measurement. If omitted
    /// and the WAV name matches a known LibriSpeech fixture, a built-in
    /// reference is used.
    #[arg(long, value_name = "PATH")]
    reference: Option<PathBuf>,

    /// Optional text prompt (Qwen3-ASR is largely zero-shot; this can be
    /// empty). The media marker is appended automatically.
    #[arg(long, default_value = "")]
    prompt: String,

    /// Number of CPU threads for inference.
    #[arg(short = 't', long, default_value_t = 8)]
    threads: i32,

    /// Maximum tokens to generate during decode.
    #[arg(long, default_value_t = 256)]
    max_tokens: i32,

    /// Context size (tokens). Qwen3-ASR audio prefill is ~400 tokens for
    /// a 30 s chunk; 4096 is generous and leaves headroom for repeat runs.
    #[arg(long, default_value_t = NonZeroU32::new(4096).unwrap())]
    n_ctx: NonZeroU32,

    /// Decode batch size used by mtmd_helper_eval_chunks.
    #[arg(long, default_value_t = 512)]
    n_batch: i32,

    /// If set, transcribe the same audio twice (fresh LlamaContext, same
    /// MtmdContext) to characterise repeated-call behaviour (Q-P0-2).
    #[arg(long, default_value_t = false)]
    repeat: bool,

    /// Disable GPU. Default on WSL Linux (WSL2 paravirt GPU is unreliable).
    #[arg(long, default_value_t = true)]
    no_gpu: bool,
}

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
        "spike-asr: model={} mmproj={} wav={} max_seconds={}",
        cli.model.display(),
        cli.mmproj.display(),
        cli.wav.display(),
        cli.max_seconds,
    );

    // -------- inputs ---------------------------------------------------
    for p in [&cli.model, &cli.mmproj, &cli.wav] {
        if !p.exists() {
            bail!("input file does not exist: {}", p.display());
        }
    }

    // SHA-256 of the GGUFs for the README's model-provenance record.
    let model_sha = sha256_of(&cli.model).context("hashing model gguf")?;
    let mmproj_sha = sha256_of(&cli.mmproj).context("hashing mmproj gguf")?;
    eprintln!("model sha256 = {model_sha}");
    eprintln!("mmproj sha256 = {mmproj_sha}");

    // -------- audio ----------------------------------------------------
    let (samples_full, sample_rate, declared_duration) =
        load_wav_mono_f32(&cli.wav).context("loading WAV")?;
    eprintln!(
        "wav: sample_rate={} samples={} duration={:.3}s",
        sample_rate,
        samples_full.len(),
        declared_duration,
    );
    if sample_rate != 16_000 {
        bail!(
            "Qwen3-ASR mtmd requires 16 kHz mono; got {sample_rate}. \
             Resample the WAV before feeding the spike."
        );
    }

    // Truncate to max_seconds. The spike's information value is whether
    // 10 s round-trips through mtmd, not throughput.
    let max_samples = (cli.max_seconds * sample_rate as f32) as usize;
    let samples: Vec<f32> = if samples_full.len() > max_samples {
        eprintln!(
            "truncating audio to first {:.3}s ({} samples)",
            cli.max_seconds, max_samples
        );
        samples_full[..max_samples].to_vec()
    } else {
        samples_full
    };
    let audio_duration_s = samples.len() as f32 / sample_rate as f32;
    eprintln!(
        "audio fed to mtmd: {:.3}s ({} samples)",
        audio_duration_s,
        samples.len()
    );

    // -------- reference (for WER) -------------------------------------
    let reference_text = resolve_reference(&cli)?;
    match reference_text.as_ref() {
        Some(s) => eprintln!("reference: {} chars", s.len()),
        None => eprintln!("reference: (none provided)"),
    }

    // -------- llama backend, model ------------------------------------
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
    eprintln!("model load: {:?}", model_t0.elapsed());

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(cli.n_ctx))
        .with_n_batch(cli.n_batch as u32)
        .with_n_threads(cli.threads)
        .with_n_threads_batch(cli.threads);

    // -------- mtmd context --------------------------------------------
    let mtmd_params = MtmdContextParams {
        use_gpu: !cli.no_gpu,
        print_timings: true,
        n_threads: cli.threads,
        media_marker: CString::new(mtmd_default_marker()).unwrap(),
    };

    let mtmd_t0 = Instant::now();
    let mtmd_ctx = MtmdContext::init_from_file(
        cli.mmproj
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF8 mmproj path"))?,
        &model,
        &mtmd_params,
    )
    .map_err(|e| anyhow!("MtmdContext init failed: {e}"))?;
    eprintln!("mtmd init: {:?}", mtmd_t0.elapsed());

    eprintln!(
        "mtmd: supports_audio={} supports_vision={} audio_sample_rate={:?} \
         decode_use_non_causal={} decode_use_mrope={}",
        mtmd_ctx.support_audio(),
        mtmd_ctx.support_vision(),
        mtmd_ctx.get_audio_sample_rate(),
        mtmd_ctx.decode_use_non_causal(),
        mtmd_ctx.decode_use_mrope(),
    );

    if !mtmd_ctx.support_audio() {
        bail!(
            "loaded mmproj does not advertise audio support; \
             check that the Q8_0 mmproj path is correct"
        );
    }

    // -------- run ------------------------------------------------------
    let transcript = run_once(
        &backend,
        &model,
        &mtmd_ctx,
        &samples,
        &ctx_params,
        &cli,
        "first",
    )?;
    let clean = strip_asr_wrapper(&transcript.text);
    // The transcript itself goes to stdout; everything else goes to stderr
    // so a `--quiet` future caller can pipe just the transcript.
    println!("{}", clean.trim());

    if cli.repeat {
        eprintln!("--- repeat: same audio, fresh LlamaContext, same MtmdContext ---");
        let transcript2 = run_once(
            &backend,
            &model,
            &mtmd_ctx,
            &samples,
            &ctx_params,
            &cli,
            "second",
        )?;
        let clean2 = strip_asr_wrapper(&transcript2.text);
        if clean2.trim() == clean.trim() {
            eprintln!("repeat: identical output (MtmdContext reusable across calls)");
        } else {
            eprintln!(
                "repeat: outputs differ.\n  first : {}\n  second: {}",
                clean.trim(),
                clean2.trim()
            );
        }
    }

    // -------- measurements --------------------------------------------
    let wer = reference_text.as_ref().map(|r| word_error_rate(r, &clean));
    let rtf = transcript.inference_secs / audio_duration_s as f64;
    let peak_rss = read_peak_rss_kib();

    eprintln!("---");
    eprintln!("inference wall-clock: {:.3} s", transcript.inference_secs);
    eprintln!("audio duration: {:.3} s", audio_duration_s);
    eprintln!(
        "real-time factor (lower is better, <1 = faster than realtime): {:.3}",
        rtf
    );
    eprintln!("tokens decoded: {}", transcript.tokens_decoded);
    if let Some(kib) = peak_rss {
        eprintln!("peak RSS (VmHWM): {:.1} MiB", kib as f64 / 1024.0);
    }
    if let Some(ref r) = reference_text {
        let wer_val = wer.unwrap();
        eprintln!("reference text: {r}");
        eprintln!("WER: {:.2}% (acceptance threshold: <=15%)", wer_val * 100.0);
    }

    // -------- acceptance gate -----------------------------------------
    let mut failures = Vec::new();
    if transcript.inference_secs > 60.0 {
        failures.push(format!(
            "inference wall-clock {:.3}s > 60s budget",
            transcript.inference_secs
        ));
    }
    if let Some(w) = wer {
        if w > 0.15 {
            failures.push(format!("WER {:.2}% > 15% threshold", w * 100.0));
        }
    }

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

struct RunResult {
    text: String,
    inference_secs: f64,
    tokens_decoded: usize,
}

fn run_once(
    backend: &LlamaBackend,
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    samples: &[f32],
    ctx_params: &LlamaContextParams,
    cli: &Cli,
    label: &str,
) -> Result<RunResult> {
    // A fresh LlamaContext per run gives a clean KV cache (cheap relative
    // to model/mtmd init) and lets us compare run-to-run behaviour without
    // worrying about cache state leakage between runs.
    let mut llama_ctx: LlamaContext = model
        .new_context(backend, ctx_params.clone())
        .map_err(|e| anyhow!("LlamaContext init failed: {e}"))?;

    // -- Build the audio bitmap. Q-P0-1: the mtmd Rust surface accepts a
    //    raw f32 slice at 16 kHz. The mtmd context owns the spectrogram
    //    step. Signature:
    //      MtmdBitmap::from_audio_data(data: &[f32]) -> Result<Self, _>
    let bitmap_t0 = Instant::now();
    let bitmap =
        MtmdBitmap::from_audio_data(samples).map_err(|e| anyhow!("from_audio_data: {e}"))?;
    eprintln!(
        "[{label}] bitmap built: is_audio={} elapsed={:?}",
        bitmap.is_audio(),
        bitmap_t0.elapsed()
    );

    // -- Build the prompt. We use the Qwen3-ASR chat template if exposed;
    //    if not we fall back to a minimal ChatML scaffold. Either way the
    //    prompt ends with the assistant turn open and contains a single
    //    <__media__> marker.
    let media_marker = mtmd_default_marker();
    let user_content = if cli.prompt.is_empty() {
        media_marker.to_string()
    } else {
        format!("{}\n{}", cli.prompt, media_marker)
    };
    let prompt_text = build_prompt(model, &user_content)?;
    eprintln!("[{label}] prompt: {:?}", prompt_text);

    let input_text = MtmdInputText {
        text: prompt_text,
        add_special: true,
        parse_special: true,
    };

    let tok_t0 = Instant::now();
    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .map_err(|e| anyhow!("mtmd tokenize: {e}"))?;
    eprintln!(
        "[{label}] tokenize: {} chunks, {} tokens total, {} positions; elapsed={:?}",
        chunks.len(),
        chunks.total_tokens(),
        chunks.total_positions(),
        tok_t0.elapsed()
    );
    for i in 0..chunks.len() {
        if let Some(c) = chunks.get(i) {
            eprintln!(
                "[{label}]   chunk[{i}] type={:?} n_tokens={} n_positions={}",
                c.chunk_type(),
                c.n_tokens(),
                c.n_positions()
            );
        }
    }

    let infer_t0 = Instant::now();

    // -- Prefill: mtmd_helper_eval_chunks runs llama_decode on text chunks
    //    and mtmd_encode + llama_decode on the audio chunk, returning the
    //    new n_past. logits_last=true so logits are ready for sampling.
    let mut n_past = chunks
        .eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, cli.n_batch, true)
        .map_err(|e| anyhow!("eval_chunks: {e}"))?;
    eprintln!("[{label}] prefill done: n_past={n_past}");

    // -- Greedy decode. Qwen3-ASR is a transcription task, not a creative
    //    one; greedy is the right call.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut batch = LlamaBatch::new(cli.n_batch as usize, 1);
    let mut decoder = UTF_8.new_decoder();
    let mut text = String::new();
    let mut tokens_decoded = 0usize;

    for _ in 0..cli.max_tokens {
        let token = sampler.sample(&llama_ctx, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow!("token_to_piece: {e}"))?;
        text.push_str(&piece);
        tokens_decoded += 1;

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| anyhow!("batch.add: {e}"))?;
        n_past += 1;

        llama_ctx
            .decode(&mut batch)
            .map_err(|e| anyhow!("decode: {e}"))?;
    }

    let inference_secs = infer_t0.elapsed().as_secs_f64();
    eprintln!(
        "[{label}] decoded {} tokens in {:.3}s",
        tokens_decoded, inference_secs
    );

    Ok(RunResult {
        text,
        inference_secs,
        tokens_decoded,
    })
}

/// Qwen3-ASR wraps its output as
/// `language English<asr_text>...transcript...</asr_text>` (the language
/// tag, the `<asr_text>` open, the transcript, and an optional closing
/// `</asr_text>` if the model emits one before EOG). Strip the wrapper so
/// WER measures the transcript itself, not the schema bookkeeping.
fn strip_asr_wrapper(s: &str) -> String {
    let mut out = s.trim().to_string();

    // Drop "language XXX" prefix before <asr_text>.
    if let Some(open) = out.find("<asr_text>") {
        let after = out[open + "<asr_text>".len()..].to_string();
        out = after;
    } else if let Some(open) = out.find("language ") {
        // The model occasionally emits the prefix without the tag (older
        // checkpoints, or when generation is truncated). Strip the first
        // two whitespace-separated tokens ("language English" / "language
        // Chinese" / etc.) and trust the rest.
        let after = &out[open + "language ".len()..];
        // Skip the language name word.
        if let Some(sp) = after.find(char::is_whitespace) {
            out = after[sp..].trim_start().to_string();
        }
    }

    // Drop trailing </asr_text> if present.
    if let Some(close) = out.find("</asr_text>") {
        out.truncate(close);
    }
    out.trim().to_string()
}

/// Apply the model's chat template if it has one; otherwise fall back to a
/// hand-built ChatML user turn. Both paths produce a string that ends with
/// the assistant tag open, so the model's next token completes the reply.
fn build_prompt(model: &LlamaModel, user_content: &str) -> Result<String> {
    use llama_cpp_2::model::LlamaChatMessage;

    let msg = LlamaChatMessage::new("user".to_string(), user_content.to_string())
        .map_err(|e| anyhow!("LlamaChatMessage::new: {e}"))?;

    // Try the model's baked-in template first. `chat_template(None)` asks
    // for the default template embedded in the GGUF.
    match model.chat_template(None::<&str>) {
        Ok(template) => match model.apply_chat_template(&template, &[msg], true) {
            Ok(rendered) => return Ok(rendered),
            Err(e) => {
                eprintln!("apply_chat_template failed, falling back to ChatML: {e}");
            }
        },
        Err(e) => {
            eprintln!("model has no chat template ({e:?}); using ChatML fallback");
        }
    }

    // ChatML fallback. The trailing newline matters; without it the model
    // sometimes prepends a leading newline of its own.
    Ok(format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        user_content
    ))
}

// ---------------------------------------------------------------------------
// WAV loading. Accepts s16 PCM, i32 PCM, or f32; rejects multi-channel and
// rates other than 16 kHz (Qwen3-ASR's mtmd requires that exact rate).
// ---------------------------------------------------------------------------

fn load_wav_mono_f32(path: &Path) -> Result<(Vec<f32>, u32, f32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!(
            "expected mono WAV; got {} channels. Provide a 16 kHz mono file.",
            spec.channels
        );
    }
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()
            .context("reading i16 samples")?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<_, _>>()
            .context("reading i32 samples")?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .context("reading f32 samples")?,
        (fmt, bits) => bail!("unsupported WAV sample format: {:?} {} bits", fmt, bits),
    };
    let duration = samples.len() as f32 / sample_rate as f32;
    Ok((samples, sample_rate, duration))
}

// ---------------------------------------------------------------------------
// Reference lookup. CLI --reference wins. Otherwise: built-in transcripts
// for the LibriSpeech fixtures we know about (see
// ~/qwen3-asr-onnx/tests/fixtures/README.md for sources).
// ---------------------------------------------------------------------------

fn resolve_reference(cli: &Cli) -> Result<Option<String>> {
    if let Some(p) = &cli.reference {
        let s = fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        return Ok(Some(s));
    }
    let name = cli
        .wav
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let builtin = match name {
        // The 10s truncation of librispeech_30s.wav covers approximately
        // the first ~25 words of the transcript. We carry the whole-clip
        // reference; word_error_rate truncates the reference to a slack-ed
        // prefix matching the hypothesis length.
        "librispeech_30s.wav" => Some(
            "LAW SEEMED TO HIM WELL ENOUGH AS A SCIENCE BUT HE NEVER COULD \
             DISCOVER A PRACTICAL CASE WHERE IT APPEARED TO HIM WORTH WHILE TO \
             GO TO LAW AND ALL THE CLIENTS WHO STOPPED WITH THIS NEW CLERK IN \
             THE ANTE ROOM OF THE LAW OFFICE WHERE HE WAS WRITING PHILIP \
             INVARIABLY ADVISED TO SETTLE NO MATTER HOW BUT SETTLE GREATLY TO \
             THE DISGUST OF HIS EMPLOYER WHO KNEW THAT JUSTICE BETWEEN MAN AND \
             MAN COULD ONLY BE ATTAINED BY THE RECOGNIZED PROCESSES WITH THE \
             ATTENDANT FEES",
        ),
        "librispeech_32s.wav" => Some(
            "BUT THERE IS ALWAYS A STRONGER SENSE OF LIFE WHEN THE SUN IS \
             BRILLIANT AFTER RAIN AND NOW HE IS POURING DOWN HIS BEAMS AND \
             MAKING SPARKLES AMONG THE WET STRAW AND LIGHTING UP EVERY PATCH \
             OF VIVID GREEN MOSS ON THE RED TILES OF THE COW SHED AND TURNING \
             EVEN THE MUDDY WATER THAT IS HURRYING ALONG THE CHANNEL TO THE \
             DRAIN INTO A MIRROR FOR THE YELLOW BILLED DUCKS WHO ARE SEIZING \
             THE OPPORTUNITY OF GETTING A DRINK WITH AS MUCH BODY IN IT AS \
             POSSIBLE",
        ),
        "librispeech_35s.wav" => Some(
            "YESTERDAY YOU WERE TREMBLING FOR A HEALTH THAT IS DEAR TO YOU TO \
             DAY YOU FEAR FOR YOUR OWN TO MORROW IT WILL BE ANXIETY ABOUT \
             MONEY THE DAY AFTER TO MORROW THE DIATRIBE OF A SLANDERER THE \
             DAY AFTER THAT THE MISFORTUNE OF SOME FRIEND THEN THE PREVAILING \
             WEATHER THEN SOMETHING THAT HAS BEEN BROKEN OR LOST THEN A \
             PLEASURE WITH WHICH YOUR CONSCIENCE AND YOUR VERTEBRAL COLUMN \
             REPROACH YOU AGAIN THE COURSE OF PUBLIC AFFAIRS",
        ),
        _ => None,
    };
    Ok(builtin.map(|s| s.to_string()))
}

// ---------------------------------------------------------------------------
// WER. Word-level Levenshtein, normalised by reference word count. Token
// normalisation: lowercase, strip ASCII punctuation, collapse whitespace.
//
// Because the input is truncated to ~10s but the LibriSpeech reference
// covers ~30s, we compute WER over the prefix of the reference that has
// the same word count as the hypothesis (+25% slack). This is documented
// as a limitation in the README.
// ---------------------------------------------------------------------------

fn normalise_words(s: &str) -> Vec<String> {
    // Step 1: lowercase + strip punctuation (keep apostrophes).
    let cleaned: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || c == '\'' {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();

    // Step 2: expand a small set of orthographic equivalences. LibriSpeech
    // references are SHOUT-CASE and write "MISTER" / "MISSUS"; the model
    // emits "Mr." / "Mrs." which normalise to "mr" / "mrs". Without
    // expansion this becomes a substitution error per token.
    let mut out = Vec::new();
    for w in cleaned.split_whitespace() {
        match w {
            "mr" => out.push("mister".to_string()),
            "mrs" => out.push("missus".to_string()),
            "ms" => out.push("miss".to_string()),
            "dr" => out.push("doctor".to_string()),
            "st" => out.push("saint".to_string()),
            other => out.push(other.to_string()),
        }
    }
    out
}

fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let hyp = normalise_words(hypothesis);
    let ref_all = normalise_words(reference);
    if hyp.is_empty() {
        return if ref_all.is_empty() { 0.0 } else { 1.0 };
    }
    // Trim the reference to the prefix that matches the hypothesis length
    // (+25% slack) so a 10 s truncation isn't measured against the full
    // 30 s transcript.
    let prefix_len = ((hyp.len() as f64) * 1.25).ceil() as usize;
    let prefix_len = prefix_len.min(ref_all.len());
    let ref_words = &ref_all[..prefix_len];

    let n = ref_words.len();
    let m = hyp.len();
    if n == 0 {
        return 1.0;
    }
    // Word-level edit distance, O(n*m) time, O(m) space.
    let mut prev = (0..=m).collect::<Vec<usize>>();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if ref_words[i - 1] == hyp[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m] as f64 / n as f64
}

// ---------------------------------------------------------------------------
// Misc utilities: SHA-256 and /proc/self/status peak RSS.
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
    fn wer_identical_is_zero() {
        let r = "the quick brown fox jumps over the lazy dog";
        let h = "the quick brown fox jumps over the lazy dog";
        assert_eq!(word_error_rate(r, h), 0.0);
    }

    #[test]
    fn wer_one_substitution() {
        let r = "the quick brown fox";
        let h = "the slow brown fox";
        // 1 substitution / 4 reference words = 0.25.
        assert!((word_error_rate(r, h) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn wer_prefix_truncation() {
        // Hypothesis covers only the prefix; the reference is longer. The
        // reference should be trimmed to ~1.25x the hypothesis length
        // before scoring.
        let r = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let h = "alpha beta gamma delta";
        // Trimmed reference is "alpha beta gamma delta epsilon" (5 words),
        // hypothesis is 4 words: 1 deletion / 5 = 0.2.
        let wer = word_error_rate(r, h);
        assert!((wer - 0.2).abs() < 1e-9, "got {wer}");
    }

    #[test]
    fn wer_normalisation_strips_punctuation_and_case() {
        let r = "Hello, world!";
        let h = "hello world";
        assert_eq!(word_error_rate(r, h), 0.0);
    }

    #[test]
    fn wer_expands_mr_abbreviation() {
        let r = "MISTER QUILTER IS";
        let h = "Mr. Quilter is";
        assert_eq!(word_error_rate(r, h), 0.0);
    }

    #[test]
    fn strip_asr_wrapper_removes_language_prefix() {
        let raw = "language English<asr_text>hello world</asr_text>";
        assert_eq!(strip_asr_wrapper(raw), "hello world");
    }

    #[test]
    fn strip_asr_wrapper_handles_missing_close_tag() {
        let raw = "language English<asr_text>hello world";
        assert_eq!(strip_asr_wrapper(raw), "hello world");
    }

    #[test]
    fn strip_asr_wrapper_handles_no_wrapper() {
        let raw = "just a transcript";
        assert_eq!(strip_asr_wrapper(raw), "just a transcript");
    }

    #[test]
    fn strip_asr_wrapper_strips_language_only_prefix() {
        // Some checkpoints emit "language English" without the tag.
        let raw = "language English hello world";
        assert_eq!(strip_asr_wrapper(raw), "hello world");
    }
}
