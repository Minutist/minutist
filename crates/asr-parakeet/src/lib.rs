//! `asr-parakeet` — NVIDIA Parakeet TDT 0.6B v3 via sherpa-onnx, implementing
//! [`minutist_common::AsrBackend`] with per-word timestamps.
//!
//! # Why a separate crate from `asr-runtime`
//!
//! Keeps the single-domain rule: `asr-runtime` is the llama-cpp-2/Qwen
//! domain, this is the sherpa-onnx/Parakeet domain (sherpa-onnx already
//! enters the workspace via `diarizer`; this is its second consumer, FFI via
//! `sherpa-rs`, the same `=0.6.8` pin). The two backends are interchangeable
//! behind `Box<dyn AsrBackend + Send>`; the orchestrator selects one per the
//! resolved transcription language (`runner::build_asr_backend`).
//!
//! # Language routing
//!
//! Parakeet TDT v3 covers 25 European languages (English + EU). Languages
//! outside that set, and `Auto-detect`, route to the Qwen `asr-runtime`
//! tiers instead (broadest coverage). The mapping is a pure function in
//! `common` (`asr_engine_for_language`) so the UI and the orchestrator agree
//! on it.
//!
//! # Binding note
//!
//! sherpa-rs 0.6.8 `TransducerRecognizer::transcribe()` returns only the text
//! and drops the per-token timestamps the C result carries
//! (`SherpaOnnxGetOfflineStreamResult` → `timestamps` + `tokens`). This crate
//! enables the `sherpa-rs` `sys` feature and calls the C API directly to read
//! text + tokens + timestamps, then groups Parakeet's sub-word tokens into
//! words on the leading-space boundary to fill `Segment.words` — the
//! per-word timestamps the mtmd path cannot produce. The recogniser
//! (including the ~650 MB encoder) is loaded once in [`ParakeetBackend::new`];
//! each `transcribe_chunk` allocates a fresh offline stream (cheap).
//!
//! # Output guard
//!
//! A chunk whose decode is a degenerate repetition runaway — one word
//! exceeding 50% of the output (over ≥ 5 words), or a distinct-word ratio
//! below 0.35 (over ≥ 8 words) — yields no segment rather than a hallucinated
//! transcript. Discontinuous or starved audio (a dropped-frame burst) drives
//! the transducer to loop a word or clause; dropping the window keeps the
//! loop out of the transcript, summary, and RAG index. This is the Parakeet
//! counterpart to `asr-runtime`'s plausibility check.
//!
//! License: the Parakeet model is CC-BY-4.0, distinct from the Apache-2.0
//! Qwen models — attribution is shipped in the About dialog.

use std::ffi::{CStr, CString};
use std::mem;
use std::path::{Path, PathBuf};

use minutist_common::{AppError, AppResult, AudioChunk, Segment, WordTimestamp};
use sherpa_rs::sherpa_rs_sys as sys;

/// Sample rate Parakeet (and our pipeline) operates at; sherpa does not resample.
const SAMPLE_RATE: u32 = 16_000;
/// Feature dimension for the FastConformer front-end (80 log-mel bins).
const FEATURE_DIM: i32 = 80;
/// Manifest id, used only in error context.
const MODEL_ID: &str = "parakeet-tdt-0.6b-v3-int8";

/// Minimum word count before the single-word-runaway branch of
/// [`is_degenerate_repetition`] fires — short utterances ("yeah yeah yeah") are
/// legitimate even when one word dominates.
const REPEAT_MIN_WORDS: usize = 5;
/// Single-word runaway: one word accounting for more than this fraction of a
/// string of at least `REPEAT_MIN_WORDS` words is degenerate.
const SINGLE_WORD_DOMINANCE: f32 = 0.5;
/// Minimum word count before the repeated-phrase branch fires; a healthy clause
/// is rarely this long with so few distinct words.
const PHRASE_MIN_WORDS: usize = 8;
/// Repeated-phrase runaway: a distinct-word-to-total ratio below this (over at
/// least `PHRASE_MIN_WORDS` words) means a whole clause is looping even though
/// no single word dominates.
const PHRASE_DISTINCT_RATIO: f32 = 0.35;

/// The four files [`ParakeetBackend::new`] requires in `model_dir`.
const REQUIRED_MODEL_FILES: [&str; 4] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

/// Reject a model file that is missing, not a regular file, or empty, before
/// it is handed to `SherpaOnnxCreateOfflineRecognizer`.
///
/// onnxruntime throws a C++ exception across the `extern "C"` boundary when
/// handed a nonexistent or truncated model file; `std::panic::catch_unwind`
/// cannot catch a foreign exception, so once that call is made there is no way
/// to recover — the only viable defence is refusing the call up front. This is
/// a cheap, self-contained `std::fs` check (no `model-registry` dependency
/// edge); it catches an absent file and a zero-byte file (the shape a
/// truncated in-progress download leaves before it reaches its expected size),
/// but not a file that is the right size yet still corrupt — the
/// model-registry hash check upstream is the authoritative guard for that.
fn require_model_file(path: &Path) -> AppResult<()> {
    let meta = std::fs::metadata(path).map_err(|e| AppError::ModelLoad {
        model_id: MODEL_ID.into(),
        context: format!("model file missing or unreadable at {}: {e}", path.display()),
    })?;
    if !meta.is_file() {
        return Err(AppError::ModelLoad {
            model_id: MODEL_ID.into(),
            context: format!("model path is not a regular file: {}", path.display()),
        });
    }
    if meta.len() == 0 {
        return Err(AppError::ModelLoad {
            model_id: MODEL_ID.into(),
            context: format!(
                "model file is empty (incomplete download?): {}",
                path.display()
            ),
        });
    }
    Ok(())
}

/// Construction inputs for [`ParakeetBackend`].
#[derive(Debug, Clone)]
pub struct ParakeetConfig {
    /// Directory holding `encoder.int8.onnx`, `decoder.int8.onnx`,
    /// `joiner.int8.onnx`, and `tokens.txt`.
    pub model_dir: PathBuf,
    /// ONNX Runtime intra-op thread count (CPU EP).
    pub num_threads: i32,
}

impl ParakeetConfig {
    /// Derive the ONNX Runtime intra-op thread count from the host: the CPU core
    /// count, capped at 6. A fixed 4 left the 0.6B int8 transducer riding the
    /// real-time edge on this class of APU; 6 keeps comfortable headroom while
    /// leaving cores for VAD, diarisation, the UI, and the OS, and is the point
    /// of diminishing returns for a model this size.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 6) as i32;
        Self {
            model_dir: model_dir.into(),
            num_threads,
        }
    }
}

/// Parakeet TDT offline-transducer backend. Loads the recogniser once.
pub struct ParakeetBackend {
    recognizer: *const sys::SherpaOnnxOfflineRecognizer,
}

// The recogniser pointer is only ever used behind `&mut self` (one chunk at a
// time, on whichever thread owns the backend). `AsrBackend: Send` and the
// orchestrator never shares it across threads concurrently.
unsafe impl Send for ParakeetBackend {}

impl ParakeetBackend {
    /// Build the recogniser from the four model files in `config.model_dir`.
    ///
    /// Every file is pre-flight validated ([`require_model_file`]) before the
    /// sherpa FFI call — a recoverable `AppError::ModelLoad` on a missing or
    /// empty file, never a call into sherpa with a bad path.
    pub fn new(config: ParakeetConfig) -> AppResult<Self> {
        let dir = &config.model_dir;
        for name in REQUIRED_MODEL_FILES {
            require_model_file(&dir.join(name))?;
        }
        let cpath = |name: &str| -> AppResult<CString> {
            let p = dir.join(name);
            let s = p.to_str().ok_or_else(|| AppError::ModelLoad {
                model_id: MODEL_ID.into(),
                context: format!("non-UTF-8 model path: {}", p.display()),
            })?;
            CString::new(s).map_err(|e| AppError::ModelLoad {
                model_id: MODEL_ID.into(),
                context: format!("interior NUL in model path: {e}"),
            })
        };
        let encoder = cpath("encoder.int8.onnx")?;
        let decoder = cpath("decoder.int8.onnx")?;
        let joiner = cpath("joiner.int8.onnx")?;
        let tokens = cpath("tokens.txt")?;
        let provider = CString::new("cpu").unwrap();
        // Empty model_type → sherpa auto-detects from the ONNX metadata (NeMo
        // exports carry `model_type=nemo_transducer`).
        let model_type = CString::new("").unwrap();
        let modeling_unit = CString::new("").unwrap();
        let bpe_vocab = CString::new("").unwrap();
        let hotwords_file = CString::new("").unwrap();
        let decoding_method = CString::new("greedy_search").unwrap();

        let recognizer = unsafe {
            let model_config = sys::SherpaOnnxOfflineModelConfig {
                transducer: sys::SherpaOnnxOfflineTransducerModelConfig {
                    encoder: encoder.as_ptr(),
                    decoder: decoder.as_ptr(),
                    joiner: joiner.as_ptr(),
                },
                tokens: tokens.as_ptr(),
                num_threads: config.num_threads.max(1),
                debug: 0,
                provider: provider.as_ptr(),
                model_type: model_type.as_ptr(),
                modeling_unit: modeling_unit.as_ptr(),
                bpe_vocab: bpe_vocab.as_ptr(),
                // NULL the model variants we are not using.
                telespeech_ctc: mem::zeroed(),
                paraformer: mem::zeroed(),
                tdnn: mem::zeroed(),
                nemo_ctc: mem::zeroed(),
                whisper: mem::zeroed(),
                sense_voice: mem::zeroed(),
                moonshine: mem::zeroed(),
                fire_red_asr: mem::zeroed(),
                dolphin: mem::zeroed(),
                zipformer_ctc: mem::zeroed(),
                canary: mem::zeroed(),
            };
            let recognizer_config = sys::SherpaOnnxOfflineRecognizerConfig {
                model_config,
                feat_config: sys::SherpaOnnxFeatureConfig {
                    sample_rate: SAMPLE_RATE as i32,
                    feature_dim: FEATURE_DIM,
                },
                hotwords_file: hotwords_file.as_ptr(),
                blank_penalty: 0.0,
                decoding_method: decoding_method.as_ptr(),
                hotwords_score: 0.0,
                lm_config: mem::zeroed(),
                rule_fsts: mem::zeroed(),
                rule_fars: mem::zeroed(),
                max_active_paths: mem::zeroed(),
                hr: mem::zeroed(),
            };
            sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config)
        };

        if recognizer.is_null() {
            return Err(AppError::ModelLoad {
                model_id: MODEL_ID.into(),
                context: format!(
                    "SherpaOnnxCreateOfflineRecognizer returned null for {}",
                    dir.display()
                ),
            });
        }
        Ok(Self { recognizer })
    }

    /// Run one offline decode, returning (text, tokens, per-token start times in
    /// seconds). Reads the full C result so the timestamps survive.
    ///
    /// Every pointer sherpa-onnx hands back across the FFI boundary is
    /// null-checked before it is dereferenced: none of these calls are
    /// documented as infallible, and reading through a null pointer here
    /// would segfault the process rather than surface a recoverable error.
    fn transcribe_raw(&self, samples: &[f32]) -> AppResult<(String, Vec<String>, Vec<f32>)> {
        unsafe {
            let stream = sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            require_non_null(stream, "SherpaOnnxCreateOfflineStream")?;
            sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                SAMPLE_RATE as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result_ptr = sys::SherpaOnnxGetOfflineStreamResult(stream);
            if let Err(e) = require_non_null(result_ptr, "SherpaOnnxGetOfflineStreamResult") {
                sys::SherpaOnnxDestroyOfflineStream(stream);
                return Err(e);
            }
            let raw = result_ptr.read();

            let text = cptr_to_string(raw.text);
            let count = raw.count as usize;
            let timestamps = if raw.timestamps.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(raw.timestamps, count).to_vec()
            };
            let tokens = if count == 0 {
                Vec::new()
            } else if let Err(e) =
                require_non_null(raw.tokens, "SherpaOnnxOfflineRecognizerResult.tokens")
            {
                sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
                sys::SherpaOnnxDestroyOfflineStream(stream);
                return Err(e);
            } else {
                let mut tokens = Vec::with_capacity(count);
                let mut next = raw.tokens;
                for _ in 0..count {
                    let t = CStr::from_ptr(next);
                    tokens.push(t.to_string_lossy().into_owned());
                    next = next.wrapping_byte_offset(t.to_bytes_with_nul().len() as isize);
                }
                tokens
            };

            sys::SherpaOnnxDestroyOfflineRecognizerResult(result_ptr);
            sys::SherpaOnnxDestroyOfflineStream(stream);
            Ok((text, tokens, timestamps))
        }
    }
}

/// Validate a pointer sherpa-onnx returned across the FFI boundary before it
/// is dereferenced, naming the C API call that produced it in the error so a
/// null failure is diagnosable instead of a silent segfault.
fn require_non_null<T>(ptr: *const T, source: &str) -> AppResult<()> {
    if ptr.is_null() {
        Err(AppError::Inference {
            backend: "asr-parakeet".into(),
            context: format!("{source} returned a null pointer"),
        })
    } else {
        Ok(())
    }
}

impl Drop for ParakeetBackend {
    fn drop(&mut self) {
        unsafe {
            sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer);
        }
    }
}

impl minutist_common::AsrBackend for ParakeetBackend {
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
        if chunk.sample_rate != SAMPLE_RATE {
            return Err(AppError::InvalidInput {
                context: format!(
                    "asr-parakeet requires {SAMPLE_RATE} Hz mono, got {} Hz",
                    chunk.sample_rate
                ),
            });
        }
        if chunk.samples.is_empty() {
            return Ok(vec![]);
        }

        let (text, tokens, timestamps) = self.transcribe_raw(&chunk.samples)?;
        let text = text.trim().to_string();
        if text.is_empty() {
            return Ok(vec![]);
        }
        // Drop a runaway-repetition chunk rather than pour a hallucinated loop
        // into the transcript: starved/discontinuous audio (e.g. a dropped-frame
        // burst) makes the decoder repeat a word or clause indefinitely.
        if is_degenerate_repetition(&text) {
            tracing::warn!(
                target: "asr-parakeet",
                start_ms = chunk.start_ms,
                end_ms = chunk.end_ms,
                text_chars = text.chars().count(),
                "dropping degenerate (repetition-runaway) ASR output — likely starved/discontinuous audio"
            );
            return Ok(vec![]);
        }
        let words = aggregate_words(&tokens, &timestamps, chunk.start_ms, chunk.end_ms);

        // Log the length, never the text: transcript content must not enter the
        // log stream, because the crash-capture ring buffer lifts log lines into
        // a user-facing diagnostic report (see cross-cutting.md "Crash capture").
        tracing::debug!(
            target: "asr-parakeet",
            start_ms = chunk.start_ms,
            end_ms = chunk.end_ms,
            words = words.len(),
            text_chars = text.chars().count(),
            "transcribed chunk"
        );

        // One segment per chunk (mirrors `asr-runtime`), but with word timing
        // populated; the orchestrator runner handles any re-splitting.
        Ok(vec![Segment {
            start_ms: chunk.start_ms,
            end_ms: chunk.end_ms,
            text,
            speaker_id: None,
            confidence: None,
            words,
            shared_speakers: Vec::new(),
        }])
    }
}

/// Detect degenerate ASR output — model runaway that emits the same word or
/// phrase repeatedly. Returns `true` when the chunk should be dropped rather
/// than poured into the transcript (and thence the summary and RAG index).
///
/// Two failure modes, both seen when the audio feed is discontinuous (dropped
/// frames starve the decoder): a single word looping ("the the the …") and a
/// whole clause looping ("scope of work is significant. scope of work is
/// significant. …"). The first is caught by single-word dominance; the second
/// by a low ratio of distinct words to total — which a single healthy sentence
/// never trips. Words are lower-cased and stripped of surrounding punctuation
/// so "word," and "word" count as one token.
fn is_degenerate_repetition(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    let total = words.len();
    if total < REPEAT_MIN_WORDS {
        return false;
    }

    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in &words {
        *counts.entry(w.as_str()).or_insert(0) += 1;
    }

    // Single-word runaway.
    let max_count = counts.values().copied().max().unwrap_or(0);
    if (max_count as f32) / (total as f32) > SINGLE_WORD_DOMINANCE {
        return true;
    }

    // Repeated-phrase runaway: few distinct words relative to the total.
    if total >= PHRASE_MIN_WORDS && (counts.len() as f32) / (total as f32) < PHRASE_DISTINCT_RATIO {
        return true;
    }

    false
}

unsafe fn cptr_to_string(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Group Parakeet's sub-word tokens into words on the leading-space boundary
/// (space or the `▁` SentencePiece marker), assigning each word a start (its
/// first token's timestamp) and an end (the next word's start, or `chunk_end_ms`
/// for the final word). Output timestamps are recording-clock absolute (offset
/// by `chunk_start_ms`). sherpa exposes per-token START times only, so word ends
/// are approximated by the following word's start.
fn aggregate_words(
    tokens: &[String],
    timestamps: &[f32],
    chunk_start_ms: u64,
    chunk_end_ms: u64,
) -> Vec<WordTimestamp> {
    let abs = |t: f32| chunk_start_ms.saturating_add((t.max(0.0) * 1000.0).round() as u64);
    let is_marker = |c: char| c == ' ' || c == '\u{2581}';

    // (word text, start_ms)
    let mut words: Vec<(String, u64)> = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        let t = timestamps.get(i).copied().unwrap_or(0.0);
        let starts_word = tok.starts_with(is_marker);
        if starts_word || words.is_empty() {
            words.push((tok.trim_start_matches(is_marker).to_string(), abs(t)));
        } else if let Some(last) = words.last_mut() {
            last.0.push_str(tok);
        }
    }

    let mut out: Vec<WordTimestamp> = Vec::with_capacity(words.len());
    for idx in 0..words.len() {
        let start = words[idx].1;
        let end = words
            .get(idx + 1)
            .map(|w| w.1)
            .unwrap_or(chunk_end_ms)
            .max(start);
        let text = words[idx].0.trim();
        if text.is_empty() {
            continue;
        }
        out.push(WordTimestamp {
            start_ms: start,
            end_ms: end,
            text: text.to_string(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_subword_tokens_into_words() {
        // "Hello world." across BPE tokens; words begin on a leading space.
        let tokens = vec!["H".into(), "ello".into(), " world".into(), ".".into()];
        let ts = vec![1.0f32, 1.2, 1.6, 2.0];
        let words = aggregate_words(&tokens, &ts, 10_000, 13_000);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[0].start_ms, 11_000); // 10_000 + 1.0 s
        assert_eq!(words[0].end_ms, 11_600); // next word's start (10_000 + 1.6 s)
        assert_eq!(words[1].text, "world.");
        assert_eq!(words[1].start_ms, 11_600);
        assert_eq!(words[1].end_ms, 13_000); // last word → chunk end
    }

    #[test]
    fn handles_sentencepiece_marker() {
        let tokens = vec!["\u{2581}Hi".into(), "\u{2581}there".into()];
        let ts = vec![0.0f32, 0.5];
        let words = aggregate_words(&tokens, &ts, 0, 1_000);
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            vec!["Hi", "there"]
        );
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[1].start_ms, 500);
    }

    #[test]
    fn empty_tokens_yield_no_words() {
        assert!(aggregate_words(&[], &[], 0, 1_000).is_empty());
    }

    #[test]
    fn require_non_null_rejects_null_pointer() {
        let ptr: *const u8 = std::ptr::null();
        match require_non_null(ptr, "SomeSherpaCall") {
            Err(AppError::Inference { backend, context }) => {
                assert_eq!(backend, "asr-parakeet");
                assert!(context.contains("SomeSherpaCall"));
            }
            other => panic!("expected AppError::Inference, got: {other:?}"),
        }
    }

    #[test]
    fn require_non_null_accepts_non_null_pointer() {
        let value = 42u8;
        assert!(require_non_null(&value as *const u8, "SomeSherpaCall").is_ok());
    }

    #[test]
    fn require_model_file_rejects_missing_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("does-not-exist.onnx");
        match require_model_file(&path) {
            Err(AppError::ModelLoad { .. }) => {}
            other => panic!("expected AppError::ModelLoad, got: {other:?}"),
        }
    }

    #[test]
    fn require_model_file_rejects_empty_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("truncated.onnx");
        std::fs::write(&path, []).expect("write empty file");
        match require_model_file(&path) {
            Err(AppError::ModelLoad { .. }) => {}
            other => panic!("expected AppError::ModelLoad, got: {other:?}"),
        }
    }

    #[test]
    fn require_model_file_accepts_nonempty_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.onnx");
        std::fs::write(&path, [0u8; 16]).expect("write file");
        assert!(require_model_file(&path).is_ok());
    }

    #[test]
    fn new_rejects_incomplete_model_dir_without_panicking() {
        // A model dir with only some of the required files present (the shape
        // an in-progress download leaves) must surface as an `AppError`, never
        // reach the sherpa FFI call, and never panic/abort.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("encoder.int8.onnx"), [0u8; 16]).unwrap();
        // decoder.int8.onnx, joiner.int8.onnx, tokens.txt intentionally absent.
        let result = ParakeetBackend::new(ParakeetConfig::new(dir.path()));
        match result {
            Err(AppError::ModelLoad { .. }) => {}
            Err(other) => panic!("expected AppError::ModelLoad, got: {other:?}"),
            Ok(_) => panic!("expected an error for an incomplete model dir"),
        }
    }

    #[test]
    fn degenerate_rejects_single_word_runaway() {
        // The classic decoder loop: one word emitted over and over.
        assert!(is_degenerate_repetition(&"the ".repeat(10)));
        // Punctuation- and case-insensitive: "X. x. X." is still one word.
        assert!(is_degenerate_repetition("Yeah. yeah. YEAH. yeah. yeah. yeah."));
    }

    #[test]
    fn degenerate_rejects_repeated_phrase_runaway() {
        // A whole clause looping — no single word exceeds 50%, but the distinct
        // ratio is far below threshold (5 distinct words across 25).
        assert!(is_degenerate_repetition(
            &"scope of work is significant. ".repeat(5)
        ));
    }

    #[test]
    fn degenerate_accepts_healthy_speech() {
        // A normal varied sentence is not degenerate.
        assert!(!is_degenerate_repetition(
            "we should confirm the staging budget before the load test runs next week"
        ));
        // Legitimate incidental repetition ("the" twice) stays under threshold.
        assert!(!is_degenerate_repetition(
            "the cat sat on the mat while we talked"
        ));
    }

    #[test]
    fn degenerate_ignores_short_utterances() {
        // Below REPEAT_MIN_WORDS: short confirmations are never degenerate even
        // when one word dominates.
        assert!(!is_degenerate_repetition("okay okay sure"));
    }
}
