//! ASR runtime — llama-cpp-2 mtmd Qwen3-ASR binding.
//!
//! Loads the Qwen3-ASR GGUF + mmproj pair once per process (via a module-level
//! `OnceLock` for the `LlamaBackend`), allocates a fresh `LlamaContext` per
//! `transcribe_chunk` call (cheap; sub-100 ms per Spike 1 Q-P0-2), and reuses
//! the shared `MtmdContext` across calls.
//!
//! # Stop conditions
//!
//! Generation stops on either:
//! 1. `model.is_eog_token(token)` — the model emits an end-of-generation token.
//! 2. The concatenated detokenised text contains `</asr_text>` — Qwen3-ASR's
//!    explicit transcript-terminator. The model does not always emit EOG when
//!    audio is shorter than the 30 s encoder window (Spike 3 API surprise #4).
//!    The check is against the full concatenated string so the tag is detected
//!    even when it spans token boundaries.
//!
//! See `architecture/cross-cutting.md` — "ASR chunking constraint" and
//! `spikes/asr/README.md` + `spikes/vad-loop/README.md` for findings.

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::Path;

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use meeting_app_common::{AppError, AppResult, AsrBackend, AudioChunk, Segment};

// ---------------------------------------------------------------------------
// Per-crate error
// ---------------------------------------------------------------------------

/// Errors originating inside `asr-runtime`.
///
/// All variants convert to `AppError` via the `From` impl below so callers
/// only see the shared boundary type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ASR backend init failed: {0}")]
    BackendInit(String),

    #[error("model load failed from {path}: {context}")]
    ModelLoad { path: String, context: String },

    #[error("mtmd context init failed from {path}: {context}")]
    MtmdInit { path: String, context: String },

    #[error("tokenization failed: {0}")]
    Tokenize(String),

    #[error("decode failed: {0}")]
    Decode(String),

    #[error("output parse failed: {0}")]
    OutputParse(String),

    #[error("invalid ASR input: {0}")]
    InvalidInput(String),
}

impl From<Error> for AppError {
    fn from(e: Error) -> Self {
        match e {
            Error::BackendInit(ctx) => AppError::ModelLoad {
                model_id: "llama-backend".to_string(),
                context: ctx,
            },
            Error::ModelLoad { path, context } => AppError::ModelLoad {
                model_id: path,
                context,
            },
            Error::MtmdInit { path, context } => AppError::ModelLoad {
                model_id: path,
                context,
            },
            Error::Tokenize(ctx) | Error::Decode(ctx) | Error::OutputParse(ctx) => {
                AppError::Inference {
                    backend: "mtmd".to_string(),
                    context: ctx,
                }
            }
            Error::InvalidInput(ctx) => AppError::InvalidInput { context: ctx },
        }
    }
}

// ---------------------------------------------------------------------------
// Input contract
// ---------------------------------------------------------------------------

/// Sample rate the mtmd audio encoder is fixed at.
///
/// Qwen3-ASR's mtmd encoder resamples nothing: it treats the incoming `f32`
/// samples as 16 kHz mono. The whole pipeline (`audio-capture` → VAD → ASR) is
/// standardised on 16 kHz mono, so a different rate is a wiring bug that would
/// otherwise be transcribed as garbage rather than surfaced as an error.
const REQUIRED_SAMPLE_RATE: u32 = 16_000;

/// Reject any chunk not at [`REQUIRED_SAMPLE_RATE`].
///
/// A pure pre-engine check (no model required), factored out so the default test
/// suite covers the guard without loading the GGUF. Mirrors the diarizer's
/// `require_supported_sample_rate`.
fn require_supported_sample_rate(sample_rate: u32) -> AppResult<()> {
    if sample_rate != REQUIRED_SAMPLE_RATE {
        return Err(Error::InvalidInput(format!(
            "asr-runtime requires {REQUIRED_SAMPLE_RATE} Hz mono audio; got {sample_rate} Hz"
        ))
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LlamaBackend singleton
// ---------------------------------------------------------------------------

/// Return the process-wide `LlamaBackend`.
///
/// Delegates to the SHARED singleton in `common` rather than owning a private
/// `OnceLock` here: `LlamaBackend::init()` is global (once per process), and
/// `summariser` also loads a GGUF model in the same app process, so a private
/// per-crate cell would make whichever crate inits second fail. See
/// `meeting_app_common::llama_backend`.
fn get_or_init_backend() -> Result<&'static LlamaBackend, Error> {
    meeting_app_common::llama_backend::shared_llama_backend()
        .map_err(|e| Error::BackendInit(e.to_string()))
}

// ---------------------------------------------------------------------------
// GPU offload policy
// ---------------------------------------------------------------------------

/// Number of model layers to offload to the GPU, decided at compile time.
///
/// The crate's `vulkan`/`metal`/`cuda`/`rocm` Cargo features each forward to the
/// matching `llama-cpp-2` backend feature, compiling a GPU-capable llama.cpp. If
/// the build selects one of those backends but the runtime still loads the model
/// with `n_gpu_layers = 0`, every layer runs on the CPU and the compiled backend
/// offloads nothing — the GPU feature is dead weight.
///
/// - **Any GPU feature enabled** → `u32::MAX`. `LlamaModelParams::with_n_gpu_layers`
///   clamps an out-of-`i32` value to `i32::MAX`, which llama.cpp interprets as
///   "offload all layers". llama.cpp falls back to CPU at runtime when no device
///   is present, so this is safe on machines without a GPU.
/// - **No GPU feature (default build)** → `0`. CPU-only; no GPU SDK required.
const fn default_n_gpu_layers() -> u32 {
    #[cfg(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm"))]
    {
        u32::MAX
    }
    #[cfg(not(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm")))]
    {
        0
    }
}

/// Whether the mtmd encoder should use the GPU, decided at compile time.
///
/// Mirrors [`default_n_gpu_layers`]: `true` when any GPU feature is enabled (so
/// the audio encoder runs on the same device as the model), `false` otherwise.
const fn default_mtmd_use_gpu() -> bool {
    cfg!(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm"))
}

// ---------------------------------------------------------------------------
// Public config
// ---------------------------------------------------------------------------

/// Runtime configuration for [`AsrRuntime`].
///
/// All fields have sensible defaults; construct with `AsrRuntimeConfig::default()`
/// and override individual fields as needed.
pub struct AsrRuntimeConfig {
    /// CPU threads for llama.cpp inference. Default: `(num_cpus::get() / 2).min(8)`, min 1.
    pub threads: i32,
    /// KV-cache context size in tokens. Default: 4096.
    pub n_ctx: NonZeroU32,
    /// Decode batch size used by `mtmd_helper_eval_chunks`. Default: 512.
    pub n_batch: u32,
    /// Maximum tokens to generate per call. Default: 256.
    pub max_tokens: i32,
}

impl Default for AsrRuntimeConfig {
    fn default() -> Self {
        let threads = ((num_cpus::get() / 2) as i32).clamp(1, 8);
        Self {
            threads,
            n_ctx: NonZeroU32::new(4096).expect("4096 is non-zero"),
            n_batch: 512,
            max_tokens: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// AsrRuntime
// ---------------------------------------------------------------------------

/// Production ASR backend backed by llama-cpp-2's mtmd audio path.
///
/// Load once per app session with [`AsrRuntime::new`]; the model and mtmd
/// context are retained for the lifetime of the struct. `Drop` releases them.
///
/// Per-call: a fresh `LlamaContext` is allocated for each `transcribe_chunk`
/// invocation so the KV cache is always clean without requiring an explicit
/// reset. This is cheap (sub-100 ms; see Spike 1 Q-P0-2).
pub struct AsrRuntime {
    model: LlamaModel,
    mtmd_ctx: MtmdContext,
    config: AsrRuntimeConfig,
}

impl AsrRuntime {
    /// Load the ASR model from disk.
    ///
    /// This is heavy: it initialises the `LlamaBackend` singleton (first call
    /// only), loads the GGUF weights, and constructs the `MtmdContext` from the
    /// mmproj file. Call once at application start.
    ///
    /// # Errors
    ///
    /// Returns `AppError::ModelLoad` if either GGUF cannot be loaded, or
    /// `AppError::Inference` if the mmproj doesn't advertise audio support.
    pub fn new(
        model_path: &Path,
        mmproj_path: &Path,
        config: AsrRuntimeConfig,
    ) -> AppResult<Self> {
        let backend = get_or_init_backend().map_err(AppError::from)?;

        // Check existence before calling into llama.cpp — the C library panics on
        // a missing path rather than returning an error.
        if !model_path.exists() {
            return Err(AppError::from(Error::ModelLoad {
                path: model_path.display().to_string(),
                context: "file not found".to_string(),
            }));
        }

        let model_params =
            LlamaModelParams::default().with_n_gpu_layers(default_n_gpu_layers());
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| {
                AppError::from(Error::ModelLoad {
                    path: model_path.display().to_string(),
                    context: e.to_string(),
                })
            })?;

        tracing::info!(
            target = "asr-runtime",
            model = %model_path.display(),
            "model loaded"
        );

        let mmproj_str = mmproj_path
            .to_str()
            .ok_or_else(|| AppError::from(Error::MtmdInit {
                path: mmproj_path.display().to_string(),
                context: "path is not valid UTF-8".to_string(),
            }))?;

        let mtmd_params = MtmdContextParams {
            use_gpu: default_mtmd_use_gpu(),
            print_timings: false,
            n_threads: config.threads,
            media_marker: CString::new(mtmd_default_marker())
                .map_err(|e| AppError::from(Error::MtmdInit {
                    path: mmproj_path.display().to_string(),
                    context: format!("invalid media marker: {e}"),
                }))?,
        };

        let mtmd_ctx = MtmdContext::init_from_file(mmproj_str, &model, &mtmd_params)
            .map_err(|e| {
                AppError::from(Error::MtmdInit {
                    path: mmproj_path.display().to_string(),
                    context: e.to_string(),
                })
            })?;

        if !mtmd_ctx.support_audio() {
            return Err(AppError::from(Error::MtmdInit {
                path: mmproj_path.display().to_string(),
                context: "mmproj does not advertise audio support".to_string(),
            }));
        }

        tracing::info!(
            target = "asr-runtime",
            mmproj = %mmproj_path.display(),
            audio_sample_rate = ?mtmd_ctx.get_audio_sample_rate(),
            "mtmd context initialised"
        );

        Ok(Self { model, mtmd_ctx, config })
    }
}

// ---------------------------------------------------------------------------
// AsrBackend implementation
// ---------------------------------------------------------------------------

impl AsrBackend for AsrRuntime {
    /// Transcribe one VAD-bounded `AudioChunk` into zero or more `Segment`s.
    ///
    /// Returns exactly one `Segment` per call (one segment per chunk). The
    /// segment carries the chunk's `start_ms` and `end_ms`; `speaker_id` is
    /// left `None` (diarization is a separate post-hoc pass).
    ///
    /// # Errors
    ///
    /// Returns `AppError::InvalidInput` when `chunk.sample_rate` is not 16 kHz:
    /// the mtmd encoder resamples nothing, so any other rate produces garbage.
    ///
    /// # Stop conditions
    ///
    /// - `model.is_eog_token(token)` → stop immediately.
    /// - The full concatenated detokenised text contains `</asr_text>` →
    ///   stop immediately. This handles the case where Qwen3-ASR does not
    ///   emit EOG for sub-window audio (Spike 3, API surprise #4). The check
    ///   is against the complete string, not per-token, so the tag is caught
    ///   even when it spans a token boundary.
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
        // The mtmd encoder is fixed at 16 kHz and resamples nothing; a mismatch
        // would be silently transcribed as garbage, so reject it up front.
        require_supported_sample_rate(chunk.sample_rate)?;

        let result = transcribe_inner(
            &self.model,
            &self.mtmd_ctx,
            &self.config,
            &chunk.samples,
        )?;

        let text = strip_asr_wrapper(&result);

        tracing::debug!(
            target = "asr-runtime",
            start_ms = chunk.start_ms,
            end_ms = chunk.end_ms,
            text = %text,
            "transcribed chunk"
        );

        Ok(vec![Segment {
            start_ms: chunk.start_ms,
            end_ms: chunk.end_ms,
            text,
            speaker_id: None,
            confidence: None,
            words: vec![],
        }])
    }
}

// ---------------------------------------------------------------------------
// Core inference logic
// ---------------------------------------------------------------------------

/// Run one ASR pass. Returns the raw model output (including the
/// `language English<asr_text>...</asr_text>` wrapper).
fn transcribe_inner(
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    config: &AsrRuntimeConfig,
    samples: &[f32],
) -> AppResult<String> {
    let backend = get_or_init_backend().map_err(AppError::from)?;

    // --- Step 1: build audio bitmap ---
    let bitmap = MtmdBitmap::from_audio_data(samples)
        .map_err(|e| AppError::from(Error::Tokenize(format!("MtmdBitmap::from_audio_data: {e}"))))?;

    // --- Step 2: allocate fresh LlamaContext ---
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(config.n_ctx))
        .with_n_batch(config.n_batch)
        .with_n_threads(config.threads)
        .with_n_threads_batch(config.threads);

    let mut llama_ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| AppError::from(Error::Decode(format!("LlamaContext init: {e}"))))?;

    // --- Step 3: apply chat template with <__media__> user message ---
    let media_marker = mtmd_default_marker();
    let prompt_text = build_prompt(model, media_marker)?;

    let input_text = MtmdInputText {
        text: prompt_text,
        // Step 4: AddBos::Never equivalent — the template already includes BOS.
        // MtmdInputText.add_special controls whether the tokenizer adds BOS/EOS.
        // Setting add_special=false, parse_special=true: don't add extra specials,
        // but do interpret special tokens already present in the template string.
        add_special: false,
        parse_special: true,
    };

    // --- Tokenise (the <__media__> marker is replaced with audio chunk inline) ---
    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .map_err(|e| AppError::from(Error::Tokenize(format!("mtmd tokenize: {e}"))))?;

    // --- Step 5: prefill via mtmd_helper_eval_chunks ---
    let mut n_past = chunks
        .eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, config.n_batch as i32, true)
        .map_err(|e| AppError::from(Error::Decode(format!("eval_chunks: {e}"))))?;

    // --- Step 6: greedy generation loop ---
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut batch = LlamaBatch::new(config.n_batch as usize, 1);
    let mut decoder = UTF_8.new_decoder();
    let mut text = String::new();

    for _ in 0..config.max_tokens {
        let token = sampler.sample(&llama_ctx, -1);
        sampler.accept(token);

        // Stop condition 1: EOG token.
        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| AppError::from(Error::Decode(format!("token_to_piece: {e}"))))?;
        text.push_str(&piece);

        // Stop condition 2: </asr_text> tag in the full concatenated output.
        // Check after appending — the tag may span token boundaries, so
        // per-token checking would miss it.
        if text.contains("</asr_text>") {
            break;
        }

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| AppError::from(Error::Decode(format!("batch.add: {e}"))))?;
        n_past += 1;

        llama_ctx
            .decode(&mut batch)
            .map_err(|e| AppError::from(Error::Decode(format!("llama decode: {e}"))))?;
    }

    Ok(text)
}

/// Apply the model's baked chat template and build the prompt string.
///
/// Uses the GGUF-embedded template (`model.chat_template(None)` + `apply_chat_template`)
/// with a single user message containing the `<__media__>` marker. The model's
/// template embeds BOS itself, so the tokenizer step uses `add_special=false`.
fn build_prompt(model: &LlamaModel, media_marker: &str) -> AppResult<String> {
    let msg = LlamaChatMessage::new("user".to_string(), media_marker.to_string())
        .map_err(|e| AppError::from(Error::Tokenize(format!("LlamaChatMessage::new: {e}"))))?;

    let template = model
        .chat_template(None::<&str>)
        .map_err(|e| AppError::from(Error::Tokenize(format!("chat_template: {e}"))))?;

    let rendered = model
        .apply_chat_template(&template, &[msg], true)
        .map_err(|e| AppError::from(Error::Tokenize(format!("apply_chat_template: {e}"))))?;

    Ok(rendered)
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// Strip the Qwen3-ASR `language English<asr_text>...</asr_text>` wrapper.
///
/// Returns the inner transcript text. Handles:
/// - Full wrapper: `language English<asr_text>hello world</asr_text>` → `hello world`
/// - Open tag only (EOG before close): `language English<asr_text>hello world` → `hello world`
/// - Language prefix without tag: `language English hello world` → `hello world`
/// - No wrapper at all: returned as-is.
fn strip_asr_wrapper(s: &str) -> String {
    let mut out = s.trim().to_string();

    // Strip content up through <asr_text> open tag if present.
    if let Some(open) = out.find("<asr_text>") {
        out = out[open + "<asr_text>".len()..].to_string();
    } else if let Some(open) = out.find("language ") {
        // Some checkpoints emit "language English" without the tag.
        let after = &out[open + "language ".len()..];
        if let Some(sp) = after.find(char::is_whitespace) {
            out = after[sp..].trim_start().to_string();
        }
    }

    // Strip trailing </asr_text> close tag if present.
    if let Some(close) = out.find("</asr_text>") {
        out.truncate(close);
    }

    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Unit tests — always run, no model required
    // -----------------------------------------------------------------------

    #[test]
    fn strip_wrapper_full() {
        let raw = "language English<asr_text>hello world</asr_text>";
        assert_eq!(strip_asr_wrapper(raw), "hello world");
    }

    #[test]
    fn strip_wrapper_open_only() {
        let raw = "language English<asr_text>hello world";
        assert_eq!(strip_asr_wrapper(raw), "hello world");
    }

    #[test]
    fn strip_wrapper_language_prefix_only() {
        let raw = "language English hello world";
        assert_eq!(strip_asr_wrapper(raw), "hello world");
    }

    #[test]
    fn strip_wrapper_no_wrapper() {
        let raw = "just a transcript";
        assert_eq!(strip_asr_wrapper(raw), "just a transcript");
    }

    #[test]
    fn strip_wrapper_empty() {
        assert_eq!(strip_asr_wrapper(""), "");
    }

    #[test]
    fn config_default_threads_is_at_least_one() {
        let cfg = AsrRuntimeConfig::default();
        assert!(cfg.threads >= 1);
    }

    #[test]
    fn config_default_threads_is_at_most_eight() {
        let cfg = AsrRuntimeConfig::default();
        assert!(cfg.threads <= 8);
    }

    #[test]
    fn config_default_n_ctx_is_4096() {
        let cfg = AsrRuntimeConfig::default();
        assert_eq!(cfg.n_ctx.get(), 4096);
    }

    #[test]
    fn config_default_n_batch_is_512() {
        let cfg = AsrRuntimeConfig::default();
        assert_eq!(cfg.n_batch, 512);
    }

    #[test]
    fn config_default_max_tokens_is_256() {
        let cfg = AsrRuntimeConfig::default();
        assert_eq!(cfg.max_tokens, 256);
    }

    #[test]
    fn require_supported_sample_rate_accepts_16khz_rejects_others() {
        // 16 kHz is accepted.
        require_supported_sample_rate(16_000).expect("16 kHz must be accepted");

        // Anything else is rejected as InvalidInput, model-free.
        for bad in [8_000u32, 22_050, 44_100, 48_000, 0] {
            let err = require_supported_sample_rate(bad)
                .expect_err("non-16 kHz must be rejected");
            assert!(
                matches!(err, AppError::InvalidInput { .. }),
                "expected InvalidInput for {bad} Hz, got {err}"
            );
        }
    }

    #[test]
    fn default_n_gpu_layers_matches_build_features() {
        let n = default_n_gpu_layers();
        if cfg!(any(
            feature = "vulkan",
            feature = "metal",
            feature = "cuda",
            feature = "rocm"
        )) {
            // GPU build: offload all layers (clamped to i32::MAX downstream).
            assert_eq!(n, u32::MAX, "GPU build must offload all layers");
            assert!(default_mtmd_use_gpu(), "GPU build must enable mtmd GPU");
        } else {
            // Default build: CPU-only.
            assert_eq!(n, 0, "default build must keep model on CPU");
            assert!(!default_mtmd_use_gpu(), "default build must disable mtmd GPU");
        }
    }

    /// (no-model, always runs) — loading from a nonexistent path must
    /// return `AppError::ModelLoad`, not panic.
    #[test]
    fn new_nonexistent_path_returns_model_load_error() {
        let result = AsrRuntime::new(
            Path::new("/nonexistent/model.gguf"),
            Path::new("/nonexistent/mmproj.gguf"),
            AsrRuntimeConfig::default(),
        );
        match result {
            Err(AppError::ModelLoad { .. }) => { /* expected */ }
            Err(other) => panic!("expected AppError::ModelLoad, got: {other}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    // -----------------------------------------------------------------------
    // Gated integration tests — skip when model env vars are absent.
    //
    // To run:
    //   MEETING_APP_ASR_MODEL_PATH=/path/to/model.gguf \
    //   MEETING_APP_ASR_MMPROJ_PATH=/path/to/mmproj.gguf \
    //   cargo test -p asr-runtime -- --include-ignored
    // -----------------------------------------------------------------------

    fn gated_runtime() -> Option<AsrRuntime> {
        let model_path = std::env::var("MEETING_APP_ASR_MODEL_PATH").ok()?;
        let mmproj_path = std::env::var("MEETING_APP_ASR_MMPROJ_PATH").ok()?;
        Some(
            AsrRuntime::new(
                Path::new(&model_path),
                Path::new(&mmproj_path),
                AsrRuntimeConfig::default(),
            )
            .expect("model load must succeed with valid paths"),
        )
    }

    fn load_librispeech_0() -> AudioChunk {
        // tests/fixtures/librispeech_0.wav — 5.86 s, 16 kHz mono.
        // Path is relative to the workspace root at test time.
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/librispeech_0.wav");
        let mut reader = hound::WavReader::open(&fixture)
            .unwrap_or_else(|e| panic!("cannot open {:?}: {e}", fixture));
        let spec = reader.spec();
        assert_eq!(spec.channels, 1, "fixture must be mono");
        assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()
            .expect("reading samples");
        let duration_ms = (samples.len() as u64 * 1000) / spec.sample_rate as u64;
        AudioChunk {
            samples,
            sample_rate: spec.sample_rate,
            start_ms: 0,
            end_ms: duration_ms,
        }
    }

    fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
        let normalise = |s: &str| -> Vec<String> {
            s.chars()
                .map(|c| {
                    if c.is_alphanumeric() || c.is_whitespace() || c == '\'' {
                        c.to_ascii_lowercase()
                    } else {
                        ' '
                    }
                })
                .collect::<String>()
                .split_whitespace()
                .map(|w| match w {
                    "mr" => "mister".to_string(),
                    "mrs" => "missus".to_string(),
                    _ => w.to_string(),
                })
                .collect()
        };
        let hyp = normalise(hypothesis);
        let ref_words = normalise(reference);
        if ref_words.is_empty() {
            return if hyp.is_empty() { 0.0 } else { 1.0 };
        }
        if hyp.is_empty() {
            return 1.0;
        }
        let n = ref_words.len();
        let m = hyp.len();
        let mut prev = (0..=m).collect::<Vec<usize>>();
        let mut curr = vec![0usize; m + 1];
        for i in 1..=n {
            curr[0] = i;
            for j in 1..=m {
                let cost = if ref_words[i - 1] == hyp[j - 1] { 0 } else { 1 };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[m] as f64 / n as f64
    }

    /// (gated) — transcribe librispeech_0.wav and verify WER <= 5%.
    #[test]
    #[ignore = "requires MEETING_APP_ASR_MODEL_PATH and MEETING_APP_ASR_MMPROJ_PATH"]
    fn transcribe_librispeech_0_within_5pct_wer() {
        let mut runtime = match gated_runtime() {
            Some(r) => r,
            None => return,
        };
        let chunk = load_librispeech_0();
        let segments = runtime
            .transcribe_chunk(&chunk)
            .expect("transcription must succeed");

        assert_eq!(segments.len(), 1, "must return exactly one segment");
        let seg = &segments[0];
        assert_eq!(seg.start_ms, chunk.start_ms);
        assert_eq!(seg.end_ms, chunk.end_ms);
        assert!(seg.speaker_id.is_none());

        // Reference from Spike 1 README.
        let reference =
            "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL";
        let wer = word_error_rate(reference, &seg.text);
        assert!(
            wer <= 0.05,
            "WER {:.2}% exceeds 5% threshold; transcript: {:?}",
            wer * 100.0,
            seg.text
        );
    }

    /// (gated) — verify that </asr_text> early-stop prevents max_tokens runaway.
    #[test]
    #[ignore = "requires MEETING_APP_ASR_MODEL_PATH and MEETING_APP_ASR_MMPROJ_PATH"]
    fn early_stop_fires_before_max_tokens() {
        let mut runtime = match gated_runtime() {
            Some(r) => r,
            None => return,
        };

        // Use a generous max_tokens that would produce garbage if early-stop
        // didn't work. The librispeech_0 clip is ~5.86 s; the real transcript
        // is ~18 words. If we set max_tokens=1000 and early-stop doesn't fire,
        // the model would fill the remaining 30-s silence window with ~180+
        // hallucinated tokens.
        runtime.config.max_tokens = 1000;

        let chunk = load_librispeech_0();
        let segments = runtime
            .transcribe_chunk(&chunk)
            .expect("transcription must succeed");

        assert_eq!(segments.len(), 1);
        let text = &segments[0].text;

        // A 5.86 s clip cannot produce more than ~30 words of real transcript.
        // If early-stop fired correctly the word count should be well under 50.
        let word_count = text.split_whitespace().count();
        assert!(
            word_count < 50,
            "word count {word_count} suggests early-stop did not fire; text: {:?}",
            text
        );
    }
}
