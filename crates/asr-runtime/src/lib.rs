//! ASR runtime — llama-cpp-2 mtmd Qwen3-ASR binding.
//!
//! Implements `minutist_common::AsrBackend`: takes an `AudioChunk`, returns
//! `Vec<Segment>`. Loads the Qwen3-ASR GGUF + mmproj pair once per process —
//! `LlamaModel` and `MtmdContext` are built once in `AsrRuntime::new`, using
//! the shared process-wide `LlamaBackend` singleton in `minutist_common`
//! (`summariser` also loads a GGUF in the same process, so a private
//! per-crate `OnceLock` here would make whichever crate inits second fail).
//! Each `transcribe_chunk` call allocates a fresh `LlamaContext` (cheap,
//! sub-100 ms) to guarantee a clean KV cache, reusing the shared
//! `MtmdContext` across calls.
//!
//! This crate also drives **Qwen3-ASR-1.7B** (same mtmd path, official
//! `ggml-org` GGUF + mmproj) as a higher-accuracy, better-multilingual tier
//! alongside the 0.6B default; the orchestrator's VRAM clamp
//! (`resolve_gpu_plan`) decides whether the 1.7B fits alongside the
//! summariser, falling back to the 0.6B CPU default otherwise. Both tiers
//! share the encoder-window constraint below, so batched-VAD chunking
//! upstream is mandatory for either.
//!
//! # Encoder-window constraint
//!
//! mtmd's audio encoder uses a fixed 30 s window: sub-30 s input is
//! silence-padded internally and the model continues generating into the
//! padded region, hallucinating words that were never in the audio. This
//! crate's `AsrBackend` implementation does not itself work around the
//! constraint — the orchestrator is responsible for shaping calls into it,
//! via the batched-VAD strategy (collect VAD segments into a ≥25 s buffer,
//! keeping the original inter-utterance silences zero-padded and capped at
//! ~3 s each, before dispatch; see `architecture/cross-cutting.md` — "ASR
//! chunking constraint").
//!
//! # Stop conditions
//!
//! Generation stops on either:
//! 1. `model.is_eog_token(token)` — the model emits an end-of-generation token.
//! 2. The concatenated detokenised text contains `</asr_text>` — Qwen3-ASR's
//!    explicit transcript-terminator, which this crate MUST also check for:
//!    the model does not always emit EOG for sub-window audio. The check is
//!    against the full concatenated string so the tag is detected even when
//!    it spans a token boundary.
//!
//! # Language hint
//!
//! `AsrRuntimeConfig.language: Option<String>` (full English name, e.g.
//! `Some("English")`) prefix-forces the output language. `None` (the
//! default) is auto-detect and produces a prompt byte-identical to the
//! no-language-hint path — a locked guarantee. When `Some(name)`, the prompt
//! is prefix-forced via an assistant-turn prefill appended AFTER
//! `apply_chat_template` (never inside the user message): the rendered
//! prompt ends with `language <name><asr_text>`, exactly the wrapper
//! Qwen3-ASR emits itself, so the model only generates the transcript. The
//! orchestrator resolves the hint from `settings.transcription_language` at
//! start.
//!
//! # Auto-language spurious-CJK guard
//!
//! When `config.language` is `None` (auto-detect), the runtime tracks a
//! rolling history of per-chunk script observations (`ScriptHistory`, last 8
//! chunks: Latin / CJK / Other). If the current chunk's text is majority-CJK
//! (> 50% of non-whitespace chars) but the recent session history is
//! majority-Latin with at least two prior Latin observations, the chunk is
//! re-run once with `language` forced to `"English"`. The retry is accepted
//! only if it is plausible (non-empty, non-degenerate, CJK-free) AND its
//! mean-token-logprob (computed from sample-time logits, log-sum-exp for
//! numerical stability) exceeds the auto run's by more than a small epsilon
//! (0.05). If the retry loses or is within epsilon, the auto result is kept
//! and a warning is emitted — genuine Chinese utterances in a mixed room are
//! never silently dropped. In a non-English Latin-script room the forced
//! retry typically scores worse and is rejected, which is self-correcting,
//! though the forced language is currently hardcoded to English rather than
//! derived from a user locale setting. The guard is inert (and
//! `ScriptHistory` is not updated) when `config.language` is `Some(..)`.
//!
//! See `architecture/cross-cutting.md` — "ASR chunking constraint".

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
use minutist_common::{AppError, AppResult, AsrBackend, AudioChunk, Segment};

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

/// Reject a model path that is missing, not a regular file, or empty, before it
/// is handed to llama.cpp's `load_from_file` / mtmd `init_from_file` FFI.
///
/// The native library aborts (rather than returning an error) on a missing or
/// truncated path, and a foreign C++ abort cannot be caught by
/// `std::panic::catch_unwind` — so validation must happen before the call. A
/// self-contained `std::fs` check (no `model-registry` dependency edge): it
/// catches an absent file and a zero-byte file (the shape a truncated
/// in-progress download leaves), but not a size-complete-yet-corrupt file — the
/// model-registry hash check upstream is the authoritative guard for that.
fn require_model_file(path: &Path) -> AppResult<()> {
    let meta = std::fs::metadata(path).map_err(|e| -> AppError {
        Error::ModelLoad {
            path: path.display().to_string(),
            context: format!("model file missing or unreadable: {e}"),
        }
        .into()
    })?;
    if !meta.is_file() {
        return Err(Error::ModelLoad {
            path: path.display().to_string(),
            context: "model path is not a regular file".to_string(),
        }
        .into());
    }
    if meta.len() == 0 {
        return Err(Error::ModelLoad {
            path: path.display().to_string(),
            context: "model file is empty (incomplete download?)".to_string(),
        }
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
/// `minutist_common::llama_backend`.
fn get_or_init_backend() -> Result<&'static LlamaBackend, Error> {
    minutist_common::llama_backend::shared_llama_backend()
        .map_err(|e| Error::BackendInit(e.to_string()))
}

// ---------------------------------------------------------------------------
// GPU offload policy
// ---------------------------------------------------------------------------

/// Compile-time GPU-offload ceiling for this build.
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
///
/// This is the **compile-time ceiling** and the `Default` source for
/// [`AsrRuntimeConfig::n_gpu_layers`]. The actual per-run layer count is now a
/// **runtime** decision: the orchestrator sets `config.n_gpu_layers` from this
/// value when the `gpu_acceleration` setting is on, or `0` to force CPU even in a
/// GPU build (see `architecture/cross-cutting.md` — "GPU portability").
pub const fn default_n_gpu_layers() -> u32 {
    #[cfg(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm"))]
    {
        u32::MAX
    }
    #[cfg(not(any(feature = "vulkan", feature = "metal", feature = "cuda", feature = "rocm")))]
    {
        0
    }
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
    /// Number of model layers to offload to the GPU.
    ///
    /// Default is the compile-time [`default_n_gpu_layers`] (`u32::MAX` in a
    /// GPU-feature build, `0` otherwise). The mtmd encoder's `use_gpu` flag is
    /// derived from this at model-open time (`> 0` → GPU). The caller (the
    /// orchestrator) sets this at **runtime** from the `gpu_acceleration`
    /// setting, passing `0` to force CPU even in a GPU build.
    pub n_gpu_layers: u32,
    /// Forcing language full-English-name (e.g. `Some("English")`,
    /// `Some("Spanish")`); `None` = auto-detect, no prefix = pre-feature
    /// behaviour.
    ///
    /// When `Some(name)`, [`build_prompt`] appends `language <name><asr_text>`
    /// to the assistant-turn opening (prefix-force). When `None`, the prompt is
    /// byte-identical to the pre-feature auto-detect prompt. `Default` is `None`
    /// (auto-detect), so the no-arg/test path is unchanged; the orchestrator sets
    /// the real value at start, exactly like `n_gpu_layers`.
    pub language: Option<String>,
}

impl Default for AsrRuntimeConfig {
    fn default() -> Self {
        let threads = ((num_cpus::get() / 2) as i32).clamp(1, 8);
        Self {
            threads,
            n_ctx: NonZeroU32::new(4096).expect("4096 is non-zero"),
            n_batch: 512,
            max_tokens: 256,
            n_gpu_layers: default_n_gpu_layers(),
            language: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Script classification
// ---------------------------------------------------------------------------

/// Classify a character as CJK (broad), Latin, or Other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScriptClass {
    /// CJK or related East/Southeast-Asian script.
    Cjk,
    /// Latin (basic + extended), ASCII letters.
    Latin,
    /// Everything else (digits, punctuation, Cyrillic, Arabic, …).
    Other,
}

/// Return `true` when `c` belongs to a CJK or closely related script.
///
/// Covered ranges (Unicode):
/// - Hiragana:                            U+3040–U+309F
/// - Katakana:                            U+30A0–U+30FF
/// - CJK Unified Ideographs Extension A: U+3400–U+4DBF
/// - CJK Unified Ideographs:             U+4E00–U+9FFF
/// - Hangul Syllables:                    U+AC00–U+D7AF
/// - Fullwidth Latin / Forms:             U+FF01–U+FFEE (catches
///   fullwidth digits/letters that appear in CJK-mode model output)
///
/// Ranges are conservative: a false negative (non-Latin codepoint missed here)
/// is safer than a false positive that mislabels a genuine non-CJK utterance.
pub(crate) fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    matches!(
        cp,
        0x3040..=0x309F  // Hiragana
        | 0x30A0..=0x30FF  // Katakana
        | 0x3400..=0x4DBF  // CJK Extension A
        | 0x4E00..=0x9FFF  // CJK Unified Ideographs
        | 0xAC00..=0xD7AF  // Hangul Syllables
        | 0xFF01..=0xFFEE  // Fullwidth Forms
    )
}

/// Classify a text string by its dominant non-whitespace script.
///
/// Returns `(cjk_fraction, class)` where `cjk_fraction` is the fraction of
/// non-whitespace characters that are CJK, and `class` is:
/// - `Cjk` when `cjk_fraction > CJK_MAJORITY_THRESHOLD`
/// - `Latin` when no CJK majority AND at least one ASCII letter is present
/// - `Other` otherwise (mixed or all-punctuation/digit)
pub(crate) fn classify_script(text: &str) -> (f32, ScriptClass) {
    let non_ws: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if non_ws.is_empty() {
        return (0.0, ScriptClass::Other);
    }
    let total = non_ws.len() as f32;
    let cjk_count = non_ws.iter().filter(|&&c| is_cjk(c)).count() as f32;
    let latin_count = non_ws
        .iter()
        .filter(|&&c| c.is_ascii_alphabetic())
        .count() as f32;

    let cjk_frac = cjk_count / total;
    let class = if cjk_frac > CJK_MAJORITY_THRESHOLD {
        ScriptClass::Cjk
    } else if latin_count > 0.0 {
        ScriptClass::Latin
    } else {
        ScriptClass::Other
    };
    (cjk_frac, class)
}

/// Fraction of non-whitespace characters that must be CJK for a chunk to
/// trigger the spurious-CJK guard.
const CJK_MAJORITY_THRESHOLD: f32 = 0.5;

/// Epsilon for mean-logprob comparison: if the forced run's score exceeds the
/// auto run's by at most this margin, scores are considered equivalent and the
/// auto result is kept.
const LOGPROB_EPSILON: f64 = 0.05;

// ---------------------------------------------------------------------------
// Session-history ring buffer
// ---------------------------------------------------------------------------

/// Number of recent chunk observations retained in the ring buffer.
const RING_SIZE: usize = 8;

/// Minimum Latin observations required in the ring buffer before the CJK guard
/// considers the session to be Latin-dominant.
const MIN_LATIN_PRIOR: usize = 2;

/// Rolling history of script-class observations, one entry per emitted chunk.
///
/// Fixed-capacity ring buffer (`RING_SIZE` entries). The session is considered
/// Latin-dominant when the majority of recent observations are `Latin` with at
/// least `MIN_LATIN_PRIOR` Latin entries — the condition under which a sudden
/// CJK chunk is suspicious and eligible for the retry guard.
pub(crate) struct ScriptHistory {
    buf: [ScriptClass; RING_SIZE],
    /// Write position (mod RING_SIZE).
    head: usize,
    /// Number of entries written so far (saturates at RING_SIZE).
    count: usize,
}

impl ScriptHistory {
    pub(crate) fn new() -> Self {
        Self {
            buf: [ScriptClass::Other; RING_SIZE],
            head: 0,
            count: 0,
        }
    }

    /// Record the script class of the most-recently emitted chunk.
    pub(crate) fn push(&mut self, class: ScriptClass) {
        self.buf[self.head] = class;
        self.head = (self.head + 1) % RING_SIZE;
        self.count = (self.count + 1).min(RING_SIZE);
    }

    /// Return the number of Latin-class observations in the current window.
    pub(crate) fn latin_count(&self) -> usize {
        self.buf[..self.count]
            .iter()
            .filter(|&&c| c == ScriptClass::Latin)
            .count()
    }

    /// Return `true` when the session history is majority-Latin with at least
    /// `MIN_LATIN_PRIOR` Latin observations.
    pub(crate) fn is_latin_session(&self) -> bool {
        let n = self.count;
        if n == 0 {
            return false;
        }
        let latin = self.latin_count();
        latin >= MIN_LATIN_PRIOR && latin * 2 > n
    }

    /// Number of observations stored (≤ `RING_SIZE`).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// Plausibility check for forced-English retry output
// ---------------------------------------------------------------------------

/// Minimum non-whitespace characters for a retry output to be considered
/// plausible.
const MIN_PLAUSIBLE_LEN: usize = 1;

/// Word-level repetition threshold: if any single word accounts for more than
/// this fraction of all words AND there are at least `DEGENERATE_MIN_WORDS`
/// words, the output is considered degenerate (model runaway).
const DEGENERATE_WORD_REPEAT_THRESHOLD: f32 = 0.5;

/// Minimum word count before the word-repetition check fires. Short phrases
/// like "hello hello" (2 words) are not degenerate even if one word dominates.
const DEGENERATE_MIN_WORDS: usize = 5;

/// Return `true` when the forced-English output is plausible.
///
/// A plausible output:
/// - Has at least one non-whitespace character.
/// - Contains no CJK codepoints (force failed or is mixed).
/// - Is not degenerate: no single word accounts for more than
///   `DEGENERATE_WORD_REPEAT_THRESHOLD` of all words in a string that is
///   at least `DEGENERATE_MIN_WORDS` words long.
pub(crate) fn is_plausible_english(text: &str) -> bool {
    let non_ws: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if non_ws.len() < MIN_PLAUSIBLE_LEN {
        return false;
    }
    if non_ws.iter().any(|&c| is_cjk(c)) {
        return false;
    }
    // Degenerate repetition check: a model in runaway mode emits the same
    // word/token repeatedly. Count word frequencies and reject if one word
    // dominates a string that is long enough to judge.
    let words: Vec<&str> = text.split_whitespace().collect();
    let total_words = words.len();
    if total_words >= DEGENERATE_MIN_WORDS {
        // Find the most-frequent word.
        let mut max_count = 0usize;
        for &w in &words {
            let count = words.iter().filter(|&&x| x == w).count();
            if count > max_count {
                max_count = count;
            }
        }
        if (max_count as f32) / (total_words as f32) > DEGENERATE_WORD_REPEAT_THRESHOLD {
            return false;
        }
    }
    true
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
///
/// Session state: a rolling `ScriptHistory` ring buffer tracks script-class
/// observations across chunks to drive the auto-language CJK guard (active only
/// when `config.language` is `None`).
pub struct AsrRuntime {
    model: LlamaModel,
    mtmd_ctx: MtmdContext,
    config: AsrRuntimeConfig,
    /// Rolling script-class history, updated after every successfully emitted chunk.
    script_history: ScriptHistory,
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

        // Pre-flight validation before calling into llama.cpp / mtmd. The native
        // C++ library aborts (rather than returning an error) when handed a
        // missing or truncated path, and a foreign C++ abort/exception cannot be
        // caught by `std::panic::catch_unwind` on the Rust side — so the only
        // viable defence is refusing the call up front. This catches an absent
        // file and a zero-byte file (the shape a truncated in-progress download
        // leaves before it reaches its expected size); the model-registry hash
        // check upstream is the authoritative guard for a size-complete-but-
        // corrupt file. Both the GGUF and the mmproj are validated because the
        // mtmd context init is a second FFI entry point with the same failure
        // mode.
        require_model_file(model_path)?;
        require_model_file(mmproj_path)?;

        // GPU offload is a RUNTIME decision driven by `config.n_gpu_layers`
        // (the orchestrator sets it from the `gpu_acceleration` setting; the
        // `Default` is the compile-time ceiling). `0` forces CPU even in a
        // GPU-feature build. See `architecture/cross-cutting.md` — "GPU
        // portability".
        let model_params =
            LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| {
                AppError::from(Error::ModelLoad {
                    path: model_path.display().to_string(),
                    context: e.to_string(),
                })
            })?;

        tracing::info!(
            target: "asr-runtime",
            model = %model_path.display(),
            "model loaded"
        );

        let mmproj_str = mmproj_path
            .to_str()
            .ok_or_else(|| AppError::from(Error::MtmdInit {
                path: mmproj_path.display().to_string(),
                context: "path is not valid UTF-8".to_string(),
            }))?;

        // The mtmd encoder follows the same runtime decision as the model: it
        // uses the GPU iff layers are being offloaded (`n_gpu_layers > 0`), so
        // forcing CPU (`n_gpu_layers = 0`) keeps the encoder on CPU too.
        let mtmd_params = MtmdContextParams {
            use_gpu: config.n_gpu_layers > 0,
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
            target: "asr-runtime",
            mmproj = %mmproj_path.display(),
            audio_sample_rate = ?mtmd_ctx.get_audio_sample_rate(),
            "mtmd context initialised"
        );

        Ok(Self {
            model,
            mtmd_ctx,
            config,
            script_history: ScriptHistory::new(),
        })
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
    /// # Auto-language CJK guard (active when `config.language` is `None`)
    ///
    /// When the current chunk's text is majority-CJK but the session history is
    /// majority-Latin with at least two prior Latin observations, the chunk is
    /// re-run once with `language = Some("English")`. The retry is accepted when:
    /// - it passes the plausibility check (non-empty, CJK-free, non-degenerate), AND
    /// - its mean-token-logprob exceeds the auto run by more than `LOGPROB_EPSILON`.
    ///
    /// If the forced run fails plausibility or the score difference is within
    /// epsilon, the original auto result is kept and a `tracing::warn` is emitted.
    ///
    /// **v1 limitation:** in non-English Latin-script rooms the forced-English
    /// retry will typically score WORSE and be rejected (self-correcting), and the
    /// forced language will later become locale/settings-driven.
    ///
    /// # Errors
    ///
    /// Returns `AppError::InvalidInput` when `chunk.sample_rate` is not 16 kHz.
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
        require_supported_sample_rate(chunk.sample_rate)?;

        let auto_result = transcribe_inner(
            &self.model,
            &self.mtmd_ctx,
            &self.config,
            &chunk.samples,
        )?;

        let forced = self.config.language.is_some();
        let auto_text = strip_asr_wrapper(&auto_result.text, forced);

        // CJK guard is active only in auto-detect mode.
        let final_text = if !forced {
            self.cjk_guard(chunk, &auto_text, &auto_result)?
        } else {
            auto_text
        };

        // Log the length, never the text: transcript content must not enter the
        // log stream, because the crash-capture ring buffer lifts log lines into
        // a user-facing diagnostic report (see cross-cutting.md "Crash capture").
        tracing::debug!(
            target: "asr-runtime",
            start_ms = chunk.start_ms,
            end_ms = chunk.end_ms,
            text_chars = final_text.chars().count(),
            "transcribed chunk"
        );

        Ok(vec![Segment {
            start_ms: chunk.start_ms,
            end_ms: chunk.end_ms,
            text: final_text,
            speaker_id: None,
            confidence: None,
            words: vec![],
            shared_speakers: Vec::new(),
        }])
    }
}

// ---------------------------------------------------------------------------
// CJK guard
// ---------------------------------------------------------------------------

impl AsrRuntime {
    /// Apply the auto-language spurious-CJK guard and update session history.
    ///
    /// Called from `transcribe_chunk` only when `config.language` is `None`.
    fn cjk_guard(
        &mut self,
        chunk: &AudioChunk,
        auto_text: &str,
        auto_result: &InnerResult,
    ) -> AppResult<String> {
        let (cjk_frac, script_class) = classify_script(auto_text);

        let final_text = if script_class == ScriptClass::Cjk
            && self.script_history.is_latin_session()
        {
            // Current chunk is majority-CJK; session history is Latin-dominant.
            // Re-run with language forced to "English".
            let forced_config = AsrRuntimeConfig {
                language: Some("English".to_string()),
                threads: self.config.threads,
                n_ctx: self.config.n_ctx,
                n_batch: self.config.n_batch,
                max_tokens: self.config.max_tokens,
                n_gpu_layers: self.config.n_gpu_layers,
            };

            let forced_result = transcribe_inner(
                &self.model,
                &self.mtmd_ctx,
                &forced_config,
                &chunk.samples,
            )?;
            let forced_text = strip_asr_wrapper(&forced_result.text, true);

            let plausible = is_plausible_english(&forced_text);
            let score_diff = forced_result.mean_logprob - auto_result.mean_logprob;
            let forced_wins = plausible && score_diff > LOGPROB_EPSILON;

            tracing::debug!(
                target: "asr-runtime",
                start_ms = chunk.start_ms,
                end_ms = chunk.end_ms,
                cjk_fraction = %format!("{:.2}", cjk_frac),
                auto_logprob = %format!("{:.4}", auto_result.mean_logprob),
                forced_logprob = %format!("{:.4}", forced_result.mean_logprob),
                forced_plausible = %plausible,
                score_diff = %format!("{:.4}", score_diff),
                decision = if forced_wins { "accept_forced" } else { "keep_auto" },
                "CJK guard triggered"
            );

            if forced_wins {
                self.script_history.push(ScriptClass::Latin);
                forced_text
            } else {
                tracing::warn!(
                    target: "asr-runtime",
                    start_ms = chunk.start_ms,
                    auto_logprob = %format!("{:.4}", auto_result.mean_logprob),
                    forced_logprob = %format!("{:.4}", forced_result.mean_logprob),
                    forced_plausible = %plausible,
                    "CJK guard: keeping auto result (forced run lost or implausible); \
                     may be genuine CJK in a mixed room"
                );
                self.script_history.push(ScriptClass::Cjk);
                auto_text.to_string()
            }
        } else {
            self.script_history.push(script_class);
            auto_text.to_string()
        };

        Ok(final_text)
    }
}

// ---------------------------------------------------------------------------
// Core inference logic
// ---------------------------------------------------------------------------

/// Return value from [`transcribe_inner`].
struct InnerResult {
    /// Raw model output (including the `language English<asr_text>…</asr_text>`
    /// wrapper — caller strips it).
    text: String,
    /// Mean log-probability of emitted tokens, computed from sample-time logits.
    ///
    /// Arithmetic mean of `log_softmax(logits)[token_id]` over all generated
    /// tokens. The O(n_vocab) sum per token is trivial next to the decode itself.
    /// When no tokens were emitted (e.g. pure EOG on a silent chunk), this is
    /// `0.0`.
    ///
    /// **Comparability caveat:** the logprob is prompt-conditioned — two runs with
    /// different prompts (auto vs forced) yield values on slightly different
    /// distributions. The comparison in the CJK guard is a heuristic, not an
    /// exact likelihood ratio.
    mean_logprob: f64,
}

/// Run one ASR pass.
///
/// Returns the raw model output (including the wrapper) and the mean
/// token log-probability computed from sample-time logits.
fn transcribe_inner(
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    config: &AsrRuntimeConfig,
    samples: &[f32],
) -> AppResult<InnerResult> {
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
    let prompt_text = build_prompt(model, media_marker, config.language.as_deref())?;

    let input_text = MtmdInputText {
        text: prompt_text,
        // AddBos::Never equivalent — the template already includes BOS.
        // add_special=false: don't add extra specials.
        // parse_special=true: interpret special tokens already in the template.
        add_special: false,
        parse_special: true,
    };

    // --- Tokenise (the <__media__> marker is replaced with audio chunk inline) ---
    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .map_err(|e| AppError::from(Error::Tokenize(format!("mtmd tokenize: {e}"))))?;

    // --- Step 4: prefill via mtmd_helper_eval_chunks ---
    let mut n_past = chunks
        .eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, config.n_batch as i32, true)
        .map_err(|e| AppError::from(Error::Decode(format!("eval_chunks: {e}"))))?;

    // --- Step 5: greedy generation loop ---
    //
    // Logprob timing: after prefill (and after each decode), `get_logits()` holds
    // the raw next-token distribution. `sampler.sample` draws from those logits.
    // The logprob of the SAMPLED token must be read BEFORE calling decode for that
    // token — after decode the logits hold the distribution for the FOLLOWING token.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut batch = LlamaBatch::new(config.n_batch as usize, 1);
    let mut decoder = UTF_8.new_decoder();
    let mut text = String::new();

    let n_vocab = model.n_vocab() as usize;
    let mut logprob_sum: f64 = 0.0;
    let mut token_count: usize = 0;

    for _ in 0..config.max_tokens {
        let token = sampler.sample(&llama_ctx, -1);
        sampler.accept(token);

        // Stop condition 1: EOG token.
        if model.is_eog_token(token) {
            break;
        }

        // Compute log-softmax at sample time (before this token's decode).
        // get_logits() returns the distribution from which `token` was just
        // sampled. After decode() below it will hold the next-token distribution.
        {
            let logits = llama_ctx.get_logits();
            debug_assert_eq!(logits.len(), n_vocab, "logit vector length mismatch");
            // log-sum-exp for numerical stability.
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = logits
                .iter()
                .map(|&l| ((l - max_logit) as f64).exp())
                .sum();
            let log_sum = sum_exp.ln() + max_logit as f64;
            let token_logprob = (logits[token.0 as usize] as f64) - log_sum;
            logprob_sum += token_logprob;
            token_count += 1;
        }

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| AppError::from(Error::Decode(format!("token_to_piece: {e}"))))?;
        text.push_str(&piece);

        // Stop condition 2: </asr_text> tag in the full concatenated output.
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

    let mean_logprob = if token_count > 0 {
        logprob_sum / token_count as f64
    } else {
        0.0
    };

    Ok(InnerResult { text, mean_logprob })
}

/// Apply the model's baked chat template and build the prompt string.
///
/// Uses the GGUF-embedded template (`model.chat_template(None)` + `apply_chat_template`)
/// with a single user message containing the `<__media__>` marker. The model's
/// template embeds BOS itself, so the tokenizer step uses `add_special=false`.
///
/// When `language` is `Some(name)`, the language hint is forced via an
/// assistant-turn prefill appended AFTER `apply_chat_template` (never inside the
/// user message): the rendered string ends with the assistant-turn opening, and
/// [`apply_language_prefix`] appends `language <name><asr_text>` to it so the
/// model only GENERATES the transcript. When `language` is `None`, the prompt is
/// byte-identical to the pre-feature auto-detect prompt (no prefix).
fn build_prompt(
    model: &LlamaModel,
    media_marker: &str,
    language: Option<&str>,
) -> AppResult<String> {
    let msg = LlamaChatMessage::new("user".to_string(), media_marker.to_string())
        .map_err(|e| AppError::from(Error::Tokenize(format!("LlamaChatMessage::new: {e}"))))?;

    let template = model
        .chat_template(None::<&str>)
        .map_err(|e| AppError::from(Error::Tokenize(format!("chat_template: {e}"))))?;

    let rendered = model
        .apply_chat_template(&template, &[msg], true)
        .map_err(|e| AppError::from(Error::Tokenize(format!("apply_chat_template: {e}"))))?;

    Ok(apply_language_prefix(rendered, language))
}

/// Append the Qwen3-ASR language-force prefix to an already-rendered prompt.
///
/// Pure (no model) so it is unit-tested on its own. The forced string is exactly
/// `language <Name><asr_text>` (single space after `language`, no space before
/// `<asr_text>`), matching the wrapper Qwen3-ASR emits itself and the open-side
/// parse in [`strip_asr_wrapper`] (`find("<asr_text>")`).
///
/// `None` returns `rendered` unchanged — the byte-identical auto-detect guard.
fn apply_language_prefix(rendered: String, language: Option<&str>) -> String {
    match language {
        Some(name) => format!("{rendered}language {name}<asr_text>"),
        None => rendered,
    }
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// Strip the Qwen3-ASR `language English<asr_text>...</asr_text>` wrapper.
///
/// `forced` says whether the language was prefix-FORCED. The generated text
/// differs by mode, so the open-side parse MUST differ:
/// - **Auto-detect** (`forced = false`): the model emits the full open wrapper
///   (`language English<asr_text>…`), so strip it. `language English<asr_text>hello
///   world</asr_text>` → `hello world`; tagless `language English hello world` →
///   `hello world`.
/// - **Forced** (`forced = true`): the open wrapper is in the PROMPT (prefilled),
///   so the model generates ONLY the transcript + close tag. There is no open
///   wrapper to strip — doing so would corrupt a transcript that legitimately
///   starts with or contains "language …". So only the close tag is removed.
///
/// Either way the trailing `</asr_text>` is truncated.
fn strip_asr_wrapper(s: &str, forced: bool) -> String {
    let mut out = s.trim();

    if !forced {
        // Auto-detect: remove the model-emitted open wrapper. Anchor the tagless
        // fallback to the START (`strip_prefix`), never a mid-transcript "language".
        if let Some(open) = out.find("<asr_text>") {
            out = &out[open + "<asr_text>".len()..];
        } else if let Some(rest) = out.strip_prefix("language ") {
            // Tagless checkpoint: "language <Name> <transcript>" — drop one name token.
            if let Some(sp) = rest.find(char::is_whitespace) {
                out = rest[sp..].trim_start();
            }
        }
    }

    // Both modes: truncate at the close tag if present.
    if let Some(close) = out.find("</asr_text>") {
        out = &out[..close];
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
        assert_eq!(strip_asr_wrapper(raw, false), "hello world");
    }

    #[test]
    fn strip_wrapper_open_only() {
        let raw = "language English<asr_text>hello world";
        assert_eq!(strip_asr_wrapper(raw, false), "hello world");
    }

    #[test]
    fn strip_wrapper_language_prefix_only() {
        let raw = "language English hello world";
        assert_eq!(strip_asr_wrapper(raw, false), "hello world");
    }

    #[test]
    fn strip_wrapper_no_wrapper() {
        let raw = "just a transcript";
        assert_eq!(strip_asr_wrapper(raw, false), "just a transcript");
    }

    #[test]
    fn strip_wrapper_empty() {
        assert_eq!(strip_asr_wrapper("", false), "");
    }

    /// Auto path must NOT mistake a mid-transcript "language" for the tagless
    /// wrapper — the open-tag branch handles the real wrapper, and the tagless
    /// fallback is start-anchored.
    #[test]
    fn strip_wrapper_auto_preserves_mid_transcript_language_word() {
        let raw = "language English<asr_text>the programming language Rust is great</asr_text>";
        assert_eq!(
            strip_asr_wrapper(raw, false),
            "the programming language Rust is great"
        );
    }

    #[test]
    fn apply_language_prefix_some() {
        assert_eq!(
            apply_language_prefix("PREFIX".to_string(), Some("English")),
            "PREFIXlanguage English<asr_text>"
        );
        assert_eq!(
            apply_language_prefix("PREFIX".to_string(), Some("Spanish")),
            "PREFIXlanguage Spanish<asr_text>"
        );
    }

    #[test]
    fn apply_language_prefix_none_is_identity() {
        assert_eq!(
            apply_language_prefix("PREFIX".to_string(), None),
            "PREFIX"
        );
    }

    #[test]
    fn strip_wrapper_forced_only_strips_close_tag() {
        assert_eq!(strip_asr_wrapper("hello world</asr_text>", true), "hello world");
        assert_eq!(strip_asr_wrapper("hello world", true), "hello world");
        assert_eq!(
            strip_asr_wrapper("the programming language Rust is great</asr_text>", true),
            "the programming language Rust is great"
        );
        assert_eq!(
            strip_asr_wrapper("language models are reshaping search</asr_text>", true),
            "language models are reshaping search"
        );
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
        require_supported_sample_rate(16_000).expect("16 kHz must be accepted");
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
            assert_eq!(n, u32::MAX, "GPU build must offload all layers");
        } else {
            assert_eq!(n, 0, "default build must keep model on CPU");
        }
    }

    #[test]
    fn config_default_n_gpu_layers_matches_build_features() {
        let cfg = AsrRuntimeConfig::default();
        assert_eq!(cfg.n_gpu_layers, default_n_gpu_layers());
        if cfg!(any(
            feature = "vulkan",
            feature = "metal",
            feature = "cuda",
            feature = "rocm"
        )) {
            assert_eq!(cfg.n_gpu_layers, u32::MAX, "GPU build offloads all layers");
            assert!(cfg.n_gpu_layers > 0, "GPU build derives mtmd use_gpu = true");
        } else {
            assert_eq!(cfg.n_gpu_layers, 0, "default build stays on CPU");
            let use_gpu = cfg.n_gpu_layers > 0;
            assert!(!use_gpu, "default build derives mtmd use_gpu = false");
        }
    }

    #[test]
    fn forced_zero_gpu_layers_yields_use_gpu_false() {
        let cfg = AsrRuntimeConfig {
            n_gpu_layers: 0,
            ..AsrRuntimeConfig::default()
        };
        assert_eq!(cfg.n_gpu_layers, 0);
        let use_gpu = cfg.n_gpu_layers > 0;
        assert!(!use_gpu, "n_gpu_layers = 0 must keep the mtmd encoder on CPU");
        let cfg_gpu = AsrRuntimeConfig {
            n_gpu_layers: u32::MAX,
            ..AsrRuntimeConfig::default()
        };
        assert!(cfg_gpu.n_gpu_layers > 0, "non-zero layers enable mtmd GPU");
    }

    #[test]
    fn require_model_file_rejects_missing_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.gguf");
        match require_model_file(&path) {
            Err(AppError::ModelLoad { .. }) => {}
            other => panic!("expected AppError::ModelLoad, got: {other:?}"),
        }
    }

    #[test]
    fn require_model_file_rejects_empty_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("truncated.gguf");
        std::fs::write(&path, []).expect("write empty file");
        match require_model_file(&path) {
            Err(AppError::ModelLoad { .. }) => {}
            other => panic!("expected AppError::ModelLoad, got: {other:?}"),
        }
    }

    #[test]
    fn require_model_file_accepts_nonempty_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, [0u8; 16]).expect("write file");
        require_model_file(&path).expect("non-empty file must be accepted");
    }

    #[test]
    fn new_nonexistent_path_returns_model_load_error() {
        let result = AsrRuntime::new(
            Path::new("/nonexistent/model.gguf"),
            Path::new("/nonexistent/mmproj.gguf"),
            AsrRuntimeConfig::default(),
        );
        match result {
            Err(AppError::ModelLoad { .. }) => {}
            Err(other) => panic!("expected AppError::ModelLoad, got: {other}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    // -----------------------------------------------------------------------
    // is_cjk — promoted runtime helper
    // -----------------------------------------------------------------------

    #[test]
    fn is_cjk_unified() {
        assert!(is_cjk('\u{4E00}'), "U+4E00 start of CJK Unified");
        assert!(is_cjk('\u{9FFF}'), "U+9FFF end of CJK Unified");
        assert!(is_cjk('中'), "CJK ideograph");
    }

    #[test]
    fn is_cjk_extension_a() {
        assert!(is_cjk('\u{3400}'), "U+3400 start of Extension A");
        assert!(is_cjk('\u{4DBF}'), "U+4DBF end of Extension A");
    }

    #[test]
    fn is_cjk_hiragana_katakana() {
        assert!(is_cjk('\u{3040}'), "U+3040 Hiragana start");
        assert!(is_cjk('\u{30FF}'), "U+30FF Katakana end");
        assert!(is_cjk('の'), "Hiragana");
        assert!(is_cjk('ア'), "Katakana");
    }

    #[test]
    fn is_cjk_hangul() {
        assert!(is_cjk('\u{AC00}'), "U+AC00 Hangul start");
        assert!(is_cjk('\u{D7AF}'), "U+D7AF Hangul end");
        assert!(is_cjk('한'), "Hangul syllable");
    }

    #[test]
    fn is_cjk_fullwidth_forms() {
        assert!(is_cjk('\u{FF01}'), "U+FF01 fullwidth exclamation");
        assert!(is_cjk('\u{FFEE}'), "U+FFEE halfwidth white circle");
    }

    #[test]
    fn is_cjk_rejects_latin_and_common() {
        for c in ['a', 'Z', '0', '9', ' ', '.', ',', '!', '\n'] {
            assert!(!is_cjk(c), "'{c}' must not be CJK");
        }
    }

    // -----------------------------------------------------------------------
    // classify_script
    // -----------------------------------------------------------------------

    #[test]
    fn classify_script_all_latin() {
        let (frac, class) = classify_script("hello world");
        assert_eq!(class, ScriptClass::Latin);
        assert!(frac < 0.01, "no CJK in Latin text");
    }

    #[test]
    fn classify_script_majority_cjk() {
        let (frac, class) = classify_script("你好世界 hi");
        assert_eq!(class, ScriptClass::Cjk, "majority CJK must classify as Cjk");
        assert!(frac > 0.5);
    }

    #[test]
    fn classify_script_mixed_below_threshold() {
        // 'a','b','c','d','e','f','好','好' = 2/8 = 25% CJK < 50%
        let (frac, class) = classify_script("ab cd ef 好 好");
        assert!(frac < 0.5);
        assert_eq!(class, ScriptClass::Latin);
    }

    #[test]
    fn classify_script_empty() {
        let (frac, class) = classify_script("");
        assert_eq!(class, ScriptClass::Other);
        assert_eq!(frac, 0.0);
    }

    #[test]
    fn classify_script_whitespace_only() {
        let (frac, class) = classify_script("   \t\n");
        assert_eq!(class, ScriptClass::Other);
        assert_eq!(frac, 0.0);
    }

    // -----------------------------------------------------------------------
    // ScriptHistory ring buffer
    // -----------------------------------------------------------------------

    #[test]
    fn script_history_new_is_not_latin_session() {
        let h = ScriptHistory::new();
        assert!(!h.is_latin_session(), "empty history is not a Latin session");
        assert_eq!(h.latin_count(), 0);
    }

    #[test]
    fn script_history_single_latin_not_session() {
        let mut h = ScriptHistory::new();
        h.push(ScriptClass::Latin);
        assert!(!h.is_latin_session());
    }

    #[test]
    fn script_history_two_latin_is_session() {
        let mut h = ScriptHistory::new();
        h.push(ScriptClass::Latin);
        h.push(ScriptClass::Latin);
        assert!(h.is_latin_session(), "two Latin observations make a Latin session");
    }

    #[test]
    fn script_history_majority_check() {
        let mut h = ScriptHistory::new();
        for _ in 0..3 {
            h.push(ScriptClass::Latin);
        }
        for _ in 0..4 {
            h.push(ScriptClass::Cjk);
        }
        assert!(!h.is_latin_session(), "CJK majority must not be Latin session");
    }

    #[test]
    fn script_history_ring_wraps() {
        let mut h = ScriptHistory::new();
        for _ in 0..RING_SIZE {
            h.push(ScriptClass::Latin);
        }
        assert_eq!(h.len(), RING_SIZE);
        h.push(ScriptClass::Cjk);
        assert_eq!(h.len(), RING_SIZE);
        assert_eq!(h.latin_count(), RING_SIZE - 1);
    }

    // -----------------------------------------------------------------------
    // is_plausible_english
    // -----------------------------------------------------------------------

    #[test]
    fn plausible_normal_english() {
        assert!(is_plausible_english("hello world"));
        assert!(is_plausible_english("the quick brown fox"));
    }

    #[test]
    fn plausible_rejects_empty() {
        assert!(!is_plausible_english(""));
        assert!(!is_plausible_english("   "));
    }

    #[test]
    fn plausible_rejects_cjk() {
        assert!(!is_plausible_english("你好世界"));
        assert!(!is_plausible_english("hello 世界"));
    }

    #[test]
    fn plausible_rejects_degenerate_repetition() {
        let rep = "the ".repeat(30);
        assert!(!is_plausible_english(&rep));
    }

    #[test]
    fn plausible_accepts_short_non_degenerate() {
        assert!(is_plausible_english("hello hello"));
    }

    // -----------------------------------------------------------------------
    // Gated integration tests — skip when model env vars are absent.
    //
    // To run:
    //   MINUTIST_ASR_MODEL_PATH=/path/to/model.gguf \
    //   MINUTIST_ASR_MMPROJ_PATH=/path/to/mmproj.gguf \
    //   cargo test -p asr-runtime -- --include-ignored
    // -----------------------------------------------------------------------

    fn gated_runtime() -> Option<AsrRuntime> {
        gated_runtime_with_language(None)
    }

    fn gated_runtime_with_language(language: Option<String>) -> Option<AsrRuntime> {
        let model_path = std::env::var("MINUTIST_ASR_MODEL_PATH").ok()?;
        let mmproj_path = std::env::var("MINUTIST_ASR_MMPROJ_PATH").ok()?;
        let config = AsrRuntimeConfig {
            language,
            ..AsrRuntimeConfig::default()
        };
        Some(
            AsrRuntime::new(
                Path::new(&model_path),
                Path::new(&mmproj_path),
                config,
            )
            .expect("model load must succeed with valid paths"),
        )
    }

    fn load_librispeech_0() -> AudioChunk {
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
    #[ignore = "requires MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH"]
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
    #[ignore = "requires MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH"]
    fn early_stop_fires_before_max_tokens() {
        let mut runtime = match gated_runtime() {
            Some(r) => r,
            None => return,
        };
        runtime.config.max_tokens = 1000;
        let chunk = load_librispeech_0();
        let segments = runtime
            .transcribe_chunk(&chunk)
            .expect("transcription must succeed");

        assert_eq!(segments.len(), 1);
        let text = &segments[0].text;
        let word_count = text.split_whitespace().count();
        assert!(
            word_count < 50,
            "word count {word_count} suggests early-stop did not fire; text: {:?}",
            text
        );
    }

    /// (gated, DIAGNOSTIC — issue 0015) — Qwen short-clip behaviour sweep.
    ///
    /// Measures how Qwen3-ASR degrades as the audio clip shrinks, to decide the
    /// re-ASR floor for splitting a mixed segment into per-speaker sub-clips.
    /// Cuts the first N ms of VOICED audio from librispeech_0 (skipping the
    /// ~570 ms leading silence) and transcribes each, printing words / loop-rep
    /// / fraction-of-words-in-reference vs duration. No hard assertions beyond a
    /// sanity check that the longest clip transcribes — read the printed table.
    /// Run with `--ignored --nocapture` (the Windows runner does both).
    #[test]
    #[ignore = "requires MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH"]
    fn short_clip_floor_sweep_0015() {
        let mut runtime = match gated_runtime() {
            Some(r) => r,
            None => return,
        };
        let full = load_librispeech_0();
        let sr = full.sample_rate as usize;

        let normalise = |text: &str| -> Vec<String> {
            text.chars()
                .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c.to_ascii_lowercase() } else { ' ' })
                .collect::<String>()
                .split_whitespace()
                .map(|w| w.to_string())
                .collect()
        };
        let reference: std::collections::HashSet<String> = normalise(
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
        )
        .into_iter()
        .collect();

        let transcribe = |rt: &mut AsrRuntime, samples: Vec<f32>, dur_ms: u64| -> String {
            let chunk = AudioChunk { samples, sample_rate: sr as u32, start_ms: 0, end_ms: dur_ms };
            rt.transcribe_chunk(&chunk)
                .expect("transcription must succeed")
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };

        // Baseline: the full fixture, full context — the "good" end of the curve.
        let baseline = transcribe(&mut runtime, full.samples.clone(), full.end_ms);
        eprintln!("\n=== Qwen short-clip sweep (issue 0015) ===");
        eprintln!("baseline (full {} ms): {:?}", full.end_ms, baseline);
        assert!(!baseline.trim().is_empty(), "baseline transcription must be non-empty");
        eprintln!(
            "{:>7} | {:>5} | {:>7} | {:>8} | text",
            "dur_ms", "words", "max_rep", "in_ref%"
        );

        // Skip leading silence so short clips actually contain speech.
        const LEAD_SILENCE_MS: usize = 570;
        let start = (LEAD_SILENCE_MS * sr) / 1000;
        for dur_ms in [3000usize, 2000, 1500, 1000, 750, 500, 300] {
            let n = (dur_ms * sr) / 1000;
            let end = (start + n).min(full.samples.len());
            if end <= start {
                continue;
            }
            let text = transcribe(&mut runtime, full.samples[start..end].to_vec(), dur_ms as u64);
            let words = normalise(&text);
            let wc = words.len();
            // Crude loop detector: highest count of any single word.
            let max_rep = words
                .iter()
                .map(|w| words.iter().filter(|x| *x == w).count())
                .max()
                .unwrap_or(0);
            let in_ref = if wc == 0 {
                0.0
            } else {
                100.0 * words.iter().filter(|w| reference.contains(*w)).count() as f64 / wc as f64
            };
            eprintln!("{:>7} | {:>5} | {:>7} | {:>7.0}% | {:?}", dur_ms, wc, max_rep, in_ref, text);
        }
        eprintln!("=== end sweep ===\n");
    }

    /// (gated) — forced English must yield no CJK codepoints.
    #[test]
    #[ignore = "requires MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH"]
    fn forced_english_output_has_no_cjk() {
        let mut runtime = match gated_runtime_with_language(Some("English".to_string())) {
            Some(r) => r,
            None => return,
        };
        let chunk = load_librispeech_0();
        let segments = runtime
            .transcribe_chunk(&chunk)
            .expect("transcription must succeed");

        assert_eq!(segments.len(), 1, "must return exactly one segment");
        let text = &segments[0].text;
        assert!(
            !text.trim().is_empty(),
            "forced-English transcript must be non-empty; got: {text:?}"
        );
        assert!(
            !text.chars().any(is_cjk),
            "forced English must not yield CJK codepoints; transcript: {text:?}"
        );
    }

    /// (gated) — auto-detect must not be regressed: English fixture within 5% WER.
    #[test]
    #[ignore = "requires MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH"]
    fn auto_detect_unchanged() {
        let mut runtime = match gated_runtime_with_language(None) {
            Some(r) => r,
            None => return,
        };
        let chunk = load_librispeech_0();
        let segments = runtime
            .transcribe_chunk(&chunk)
            .expect("transcription must succeed");

        assert_eq!(segments.len(), 1);
        let reference =
            "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL";
        let wer = word_error_rate(reference, &segments[0].text);
        assert!(
            wer <= 0.05,
            "auto-detect WER {:.2}% exceeds 5% threshold; transcript: {:?}",
            wer * 100.0,
            segments[0].text
        );
    }

    /// (gated) — CJK guard must be INERT when `config.language` is `Some(..)`.
    /// In forced mode `transcribe_chunk` skips the guard entirely. The English
    /// fixture must return a clean, low-WER transcript.
    #[test]
    #[ignore = "requires MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH"]
    fn cjk_guard_inert_when_language_forced() {
        let mut runtime = match gated_runtime_with_language(Some("English".to_string())) {
            Some(r) => r,
            None => return,
        };
        let chunk = load_librispeech_0();
        let segments = runtime
            .transcribe_chunk(&chunk)
            .expect("transcription must succeed");
        assert_eq!(segments.len(), 1);
        let text = &segments[0].text;
        assert!(!text.trim().is_empty());
        assert!(!text.chars().any(is_cjk));
        let reference =
            "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL";
        let wer = word_error_rate(reference, text);
        assert!(
            wer <= 0.05,
            "forced-language WER {:.2}% exceeds 5%; transcript: {:?}",
            wer * 100.0,
            text
        );
    }

    /// (gated) — Auto-mode on English audio for multiple chunks. After two English
    /// chunks the history is Latin-dominant, but the English fixture is not
    /// majority-CJK so the guard never triggers. WER stays within 5%.
    #[test]
    #[ignore = "requires MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH"]
    fn cjk_guard_does_not_degrade_english_session() {
        let mut runtime = match gated_runtime_with_language(None) {
            Some(r) => r,
            None => return,
        };
        let chunk = load_librispeech_0();
        let reference =
            "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL";

        for _ in 0..3 {
            let segments = runtime
                .transcribe_chunk(&chunk)
                .expect("transcription must succeed");
            assert_eq!(segments.len(), 1);
            let wer = word_error_rate(reference, &segments[0].text);
            assert!(
                wer <= 0.05,
                "guard must not degrade English session; WER {:.2}%; text: {:?}",
                wer * 100.0,
                segments[0].text,
            );
        }
    }
}
