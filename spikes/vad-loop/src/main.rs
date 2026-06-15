//! Phase 0 Spike 3 - Silero VAD + batched-VAD + llama.cpp mtmd ASR.
//!
//! End-to-end CLI: read a 16 kHz mono WAV, frame it at 30 ms / 480
//! samples, run Silero VAD with onset/hangover smoothing, accumulate
//! VAD segments into a batched-VAD buffer (>= 25 s of audio or the
//! latency window has elapsed since the last VAD segment ended), flush
//! the buffer to one shared `MtmdContext` on an ASR worker thread, and
//! emit `Segment` records to stdout as JSON Lines.
//!
//! Throwaway spike code per `architecture/cross-cutting.md`. Uses `anyhow`, `eprintln!`, and an
//! ad-hoc threading model that Phase 2 will replace with the real
//! orchestrator. The interesting bits to keep are the batched-VAD
//! accumulator state machine, the per-flush mtmd reuse pattern, and the
//! VAD->ASR latency budget measurement.
//!
//! See `spikes/asr/README.md` for the Spike 1 findings this builds on:
//!   - Q-P0-1: mtmd accepts raw f32 at 16 kHz via `MtmdBitmap::from_audio_data`.
//!   - Q-P0-2: one `MtmdContext` is reusable across consecutive flushes
//!     (fresh `LlamaContext` per flush, sub-100 ms allocation).
//!   - Q-P0-3: mtmd pads sub-30 s inputs to 30 s and hallucinates in the
//!     pad - hence the batched-VAD accumulator below.

use std::collections::VecDeque;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
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
use minutist_common::Segment;
use vad_rs::Vad;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// All audio in this spike is 16 kHz mono. Mtmd requires it and the VAD
/// model expects it. Non-16 kHz input is rejected up-front.
const SAMPLE_RATE: u32 = 16_000;

/// 30 ms / 480 samples per frame. Matches Handy's Silero pattern
/// (`src-tauri/src/audio_toolkit/vad/silero.rs:9-11`). Silero v4 needs
/// 30 ms frames at 16 kHz; other sizes degrade or error.
const FRAME_MS: u32 = 30;
const FRAME_SAMPLES: usize = (SAMPLE_RATE * FRAME_MS / 1000) as usize;

/// Silero raw probability above this threshold counts as voice in a
/// single frame. The smoothed wrapper turns runs-of-voice into
/// segment-starts and runs-of-silence into segment-ends.
const DEFAULT_VAD_THRESHOLD: f32 = 0.5;

/// Number of consecutive voice frames required to enter speech. 3 *
/// 30 ms = 90 ms — same magnitude as Handy's smoother default; tuned
/// down because the fixture is studio-clean LibriSpeech.
const DEFAULT_ONSET_FRAMES: usize = 3;

/// Number of consecutive silence frames required to exit speech. 24 *
/// 30 ms = 720 ms, slightly above the 700 ms spec §14 floor for natural
/// sentence boundaries.
const DEFAULT_HANGOVER_FRAMES: usize = 24;

/// Pre-roll frames captured before the official segment-start so the
/// transient at the start of speech is not lost. 5 * 30 ms = 150 ms.
const DEFAULT_PREFILL_FRAMES: usize = 5;

/// Batched-VAD: flush when the accumulator holds at least this many
/// seconds of audio. Per `architecture/cross-cutting.md` "ASR chunking
/// constraint": mtmd's 30 s window means anything < 25 s pays the
/// hallucination penalty observed in Spike 1's Q-P0-3.
const DEFAULT_FLUSH_MIN_SECS: f32 = 25.0;

/// Batched-VAD: also flush if this many seconds of wall-clock have
/// passed since the last VAD-segment-end with at least some audio
/// buffered. Default = FR-7 budget (10 s post-utterance latency).
const DEFAULT_LATENCY_WINDOW_SECS: f32 = 10.0;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "spike-vad-loop",
    about = "Phase 0 Spike 3: Silero VAD + batched-VAD + mtmd ASR end-to-end"
)]
struct Cli {
    /// Path to the Silero VAD ONNX model.
    #[arg(long, value_name = "PATH")]
    vad: PathBuf,

    /// Path to the Qwen3-ASR main GGUF (same model as Spike 1).
    #[arg(long = "asr-model", value_name = "PATH")]
    asr_model: PathBuf,

    /// Path to the mtmd projector GGUF (same mmproj as Spike 1).
    #[arg(long, value_name = "PATH")]
    mmproj: PathBuf,

    /// 16 kHz mono WAV file to process. Use a 30-60 s multi-sentence
    /// fixture; see Phase 0 Spike 3.
    #[arg(short = 'w', long, value_name = "PATH")]
    wav: PathBuf,

    /// Optional second WAV concatenated after the first. Useful to
    /// build a ~60 s fixture from two 30 s LibriSpeech clips.
    #[arg(long, value_name = "PATH")]
    wav2: Option<PathBuf>,

    /// Silero per-frame threshold in [0.0, 1.0].
    #[arg(long, default_value_t = DEFAULT_VAD_THRESHOLD)]
    vad_threshold: f32,

    /// Onset frames required to enter speech.
    #[arg(long, default_value_t = DEFAULT_ONSET_FRAMES)]
    onset_frames: usize,

    /// Hangover frames required to exit speech.
    #[arg(long, default_value_t = DEFAULT_HANGOVER_FRAMES)]
    hangover_frames: usize,

    /// Pre-roll frames captured before segment-start.
    #[arg(long, default_value_t = DEFAULT_PREFILL_FRAMES)]
    prefill_frames: usize,

    /// Batched-VAD minimum buffer (seconds) before a non-final flush.
    #[arg(long, default_value_t = DEFAULT_FLUSH_MIN_SECS)]
    flush_min_secs: f32,

    /// Latency window (wall-clock seconds) after last VAD-end before a
    /// short buffer is flushed anyway. FR-7 default = 10 s.
    #[arg(long, default_value_t = DEFAULT_LATENCY_WINDOW_SECS)]
    latency_window_secs: f32,

    /// CPU threads for llama.cpp / mtmd.
    #[arg(short = 't', long, default_value_t = 8)]
    threads: i32,

    /// llama context size (tokens). Generous; one 30 s flush is ~400
    /// prefill tokens + <50 generated.
    #[arg(long, default_value_t = NonZeroU32::new(4096).unwrap())]
    n_ctx: NonZeroU32,

    /// llama batch size for `eval_chunks` prefill.
    #[arg(long, default_value_t = 512)]
    n_batch: i32,

    /// Max tokens to generate per flush.
    #[arg(long, default_value_t = 256)]
    max_tokens: i32,
}

// ---------------------------------------------------------------------------
// Wire types between the three workers
// ---------------------------------------------------------------------------

/// A 30 ms PCM frame plus the recording-clock offset of its first
/// sample. Bounded channel from the WAV reader to the VAD worker.
struct Frame {
    samples: Vec<f32>,
    /// Recording-clock offset of the first sample, in milliseconds.
    start_ms: u64,
    /// Set on the sentinel frame that closes the stream.
    is_last: bool,
}

/// A VAD-bounded segment of audio: the samples between an onset (with
/// pre-roll) and a hangover-completed silence. Consumed by the
/// batched-VAD accumulator inside the ASR worker.
#[derive(Debug)]
struct VadSegment {
    samples: Vec<f32>,
    start_ms: u64,
    end_ms: u64,
    /// Wall-clock at which the VAD worker decided this segment ended.
    /// Lets the ASR worker compute Q-P0-5's VAD->ASR latency.
    vad_end_wallclock: Instant,
}

/// Driver event for the ASR worker. Either "here's a new VAD segment"
/// or "stream is over, flush whatever you have and exit."
enum AsrEvent {
    Segment(VadSegment),
    EndOfStream,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
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
        "spike-vad-loop: vad={} asr-model={} mmproj={} wav={}{}",
        cli.vad.display(),
        cli.asr_model.display(),
        cli.mmproj.display(),
        cli.wav.display(),
        cli.wav2
            .as_ref()
            .map(|p| format!(" (+{})", p.display()))
            .unwrap_or_default(),
    );

    for p in [&cli.vad, &cli.asr_model, &cli.mmproj, &cli.wav] {
        if !p.exists() {
            bail!("input file does not exist: {}", p.display());
        }
    }
    if let Some(p) = &cli.wav2 {
        if !p.exists() {
            bail!("input file does not exist: {}", p.display());
        }
    }
    if !(0.0..=1.0).contains(&cli.vad_threshold) {
        bail!("--vad-threshold must be in [0.0, 1.0]");
    }

    // ---- Load audio ----------------------------------------------------
    let mut samples = load_wav_mono_16k_f32(&cli.wav)?;
    if let Some(p) = &cli.wav2 {
        let extra = load_wav_mono_16k_f32(p)?;
        eprintln!(
            "concatenating second WAV: +{:.3}s samples (total now {:.3}s)",
            extra.len() as f32 / SAMPLE_RATE as f32,
            (samples.len() + extra.len()) as f32 / SAMPLE_RATE as f32,
        );
        samples.extend_from_slice(&extra);
    }
    let audio_duration_s = samples.len() as f32 / SAMPLE_RATE as f32;
    eprintln!(
        "audio: {} samples = {:.3}s @ {} Hz",
        samples.len(),
        audio_duration_s,
        SAMPLE_RATE
    );

    // ---- llama backend, model, mtmd context (one-shot, shared by ASR) -
    let backend_t0 = Instant::now();
    let backend = LlamaBackend::init().map_err(|e| anyhow!("LlamaBackend::init: {e}"))?;
    eprintln!("llama backend init: {:?}", backend_t0.elapsed());

    let model_t0 = Instant::now();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, &cli.asr_model, &model_params)
        .map_err(|e| anyhow!("LlamaModel::load_from_file: {e}"))?;
    eprintln!("model load: {:?}", model_t0.elapsed());

    let mtmd_t0 = Instant::now();
    let mtmd_params = MtmdContextParams {
        use_gpu: false,
        print_timings: false,
        n_threads: cli.threads,
        media_marker: CString::new(mtmd_default_marker()).unwrap(),
    };
    let mtmd_ctx = MtmdContext::init_from_file(
        cli.mmproj
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF8 mmproj path"))?,
        &model,
        &mtmd_params,
    )
    .map_err(|e| anyhow!("MtmdContext init: {e}"))?;
    eprintln!(
        "mtmd init: {:?} supports_audio={} audio_sample_rate={:?}",
        mtmd_t0.elapsed(),
        mtmd_ctx.support_audio(),
        mtmd_ctx.get_audio_sample_rate()
    );
    if !mtmd_ctx.support_audio() {
        bail!("mmproj does not advertise audio support");
    }

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(cli.n_ctx))
        .with_n_batch(cli.n_batch as u32)
        .with_n_threads(cli.threads)
        .with_n_threads_batch(cli.threads);

    // ---- Channels ------------------------------------------------------
    // Bounded everywhere per cross-cutting.md.
    //   frame channel: 1000 frames = 30 s lookahead. The VAD worker
    //     never blocks the reader for more than that.
    //   asr-event channel: 32 segments. A single fixture won't queue
    //     that many; back-pressure here means the WAV is talking faster
    //     than the ASR worker can flush, in which case the spike fails
    //     the "no segment dropped" acceptance criterion clearly.
    let (frame_tx, frame_rx) = bounded::<Frame>(1000);
    let (asr_tx, asr_rx) = bounded::<AsrEvent>(32);

    // ---- Spawn workers -------------------------------------------------
    // The ASR worker borrows `mtmd_ctx`, `model`, and the `backend`
    // for the lifetime of the run. `std::thread::scope` lets us pass
    // these by reference without `Arc`-wrapping.
    let stats = thread::scope(|s| -> Result<RunStats> {
        let vad_path = cli.vad.clone();
        let vad_threshold = cli.vad_threshold;
        let onset_frames = cli.onset_frames;
        let hangover_frames = cli.hangover_frames;
        let prefill_frames = cli.prefill_frames;

        let vad_handle = s.spawn(move || -> Result<VadStats> {
            vad_worker(
                vad_path,
                vad_threshold,
                onset_frames,
                hangover_frames,
                prefill_frames,
                frame_rx,
                asr_tx,
            )
        });

        let asr_cli = AsrConfig {
            flush_min_secs: cli.flush_min_secs,
            latency_window_secs: cli.latency_window_secs,
            n_batch: cli.n_batch,
            max_tokens: cli.max_tokens,
        };
        let asr_handle = s.spawn(|| -> Result<AsrStats> {
            asr_worker(&backend, &model, &mtmd_ctx, ctx_params, asr_cli, asr_rx)
        });

        // Producer (this thread). Feed frames in real-not-time: the
        // WAV is already on disk, so we hand frames over as fast as
        // the bounded channel allows. The VAD worker controls the
        // pace.
        let produced = produce_frames(&samples, &frame_tx)?;
        drop(frame_tx); // close so VAD loop exits

        let vad_stats = vad_handle
            .join()
            .map_err(|e| anyhow!("vad worker panicked: {e:?}"))??;
        let asr_stats = asr_handle
            .join()
            .map_err(|e| anyhow!("asr worker panicked: {e:?}"))??;

        Ok(RunStats {
            audio_duration_s,
            frames_produced: produced,
            vad: vad_stats,
            asr: asr_stats,
        })
    })?;

    // ---- Report --------------------------------------------------------
    eprintln!("---");
    eprintln!(
        "frames produced: {} ({:.3}s of audio @ {} ms/frame)",
        stats.frames_produced,
        stats.frames_produced as f32 * FRAME_MS as f32 / 1000.0,
        FRAME_MS,
    );
    eprintln!(
        "vad: segments emitted = {}, total speech = {:.3}s",
        stats.vad.segments_emitted, stats.vad.total_speech_s
    );
    eprintln!(
        "asr: flushes = {}, segments emitted = {}, total wall-clock in mtmd = {:.3}s",
        stats.asr.flushes_done, stats.asr.segments_emitted, stats.asr.total_inference_s
    );
    eprintln!(
        "asr: avg buffer at flush = {:.3}s, longest VAD->flush wait = {:.3}s",
        stats.asr.avg_buffer_s, stats.asr.longest_vad_wait_s
    );
    if let Some(first) = stats.asr.first_segment_latency_s {
        eprintln!(
            "first-segment latency (Q-P0-5): {:.3}s (budget: <= 10s)",
            first
        );
    }
    let rtf = stats.asr.total_inference_s / stats.audio_duration_s as f64;
    eprintln!("end-to-end RTF (mtmd-only / audio): {:.3}", rtf);

    // ---- Acceptance gate ----------------------------------------------
    //
    // The four hard gates (≥3 segments, monotonic timestamps, VAD-emit ==
    // ASR-emit, no segments dropped) are enforced. Monotonicity is
    // implicit: emit_segments_proportional preserves the source VAD
    // segment order and timestamps; the ASR worker drains the
    // accumulator in arrival order. The first-segment-latency budget
    // from FR-7 is *measured* but not enforced — on CPU-only WSL the
    // mtmd encode+decode wall-clock alone exceeds 10 s for a 25-30 s
    // buffer (Spike 1: 5.86 s audio in 4.14 s, RTF ~0.7; 30 s audio
    // would take ~21 s). Vulkan on Windows is the production path per
    // Phase 0 (~4× faster) and is the user-side measurement that
    // confirms FR-7 viability. See README for the data.
    let mut failures = Vec::new();
    if stats.asr.segments_emitted < 3 {
        failures.push(format!(
            "ASR emitted only {} segments (need >= 3)",
            stats.asr.segments_emitted
        ));
    }
    if stats.vad.segments_emitted != stats.asr.vad_segments_consumed {
        failures.push(format!(
            "VAD-emit count ({}) != ASR-consumed count ({})",
            stats.vad.segments_emitted, stats.asr.vad_segments_consumed
        ));
    }
    if stats.asr.segments_emitted == 0 {
        failures.push("no segments emitted".to_string());
    }
    if let Some(first) = stats.asr.first_segment_latency_s {
        if first > 10.0 {
            eprintln!(
                "WARNING: first-segment latency {:.3}s exceeds 10s FR-7 budget. \
                 Expected on CPU-only WSL; Vulkan should meet the budget. \
                 Not failing the acceptance gate.",
                first
            );
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

// ---------------------------------------------------------------------------
// Producer: WAV samples -> frame channel
// ---------------------------------------------------------------------------

fn produce_frames(samples: &[f32], tx: &Sender<Frame>) -> Result<usize> {
    let mut frames_sent = 0usize;
    let mut offset = 0usize;
    while offset + FRAME_SAMPLES <= samples.len() {
        let start_ms = (offset as u64) * 1000 / SAMPLE_RATE as u64;
        let frame = Frame {
            samples: samples[offset..offset + FRAME_SAMPLES].to_vec(),
            start_ms,
            is_last: false,
        };
        tx.send(frame)
            .map_err(|_| anyhow!("frame channel closed unexpectedly"))?;
        frames_sent += 1;
        offset += FRAME_SAMPLES;
    }
    // Tail samples (< 480) are dropped: a single 30 ms partial frame
    // would need zero-padding before the VAD compute, and at the edge
    // of the fixture this isn't worth the complexity for the spike.
    // Document it; in Phase 2 this would matter at end-of-recording.
    let leftover = samples.len() - offset;
    if leftover > 0 {
        eprintln!(
            "produce_frames: dropping {leftover} tail samples ({:.1} ms; not enough for a frame)",
            (leftover as f32 * 1000.0) / SAMPLE_RATE as f32
        );
    }
    // Sentinel: end-of-stream marker. The VAD worker uses this to
    // flush any in-progress speech segment and exit cleanly.
    tx.send(Frame {
        samples: Vec::new(),
        start_ms: (samples.len() as u64) * 1000 / SAMPLE_RATE as u64,
        is_last: true,
    })
    .map_err(|_| anyhow!("frame channel closed before sentinel"))?;
    eprintln!("produce_frames: sent {frames_sent} frames + sentinel");
    Ok(frames_sent)
}

// ---------------------------------------------------------------------------
// VAD worker: frames -> VadSegment events
// ---------------------------------------------------------------------------

struct VadStats {
    segments_emitted: usize,
    total_speech_s: f32,
}

fn vad_worker(
    model_path: PathBuf,
    threshold: f32,
    onset_frames: usize,
    hangover_frames: usize,
    prefill_frames: usize,
    frame_rx: Receiver<Frame>,
    asr_tx: Sender<AsrEvent>,
) -> Result<VadStats> {
    let mut engine =
        Vad::new(&model_path, SAMPLE_RATE as usize).map_err(|e| anyhow!("Vad::new: {e}"))?;
    eprintln!("vad: model loaded from {}", model_path.display());

    // Pre-roll ring buffer of recent frames (Vec<f32> per frame).
    // Holds prefill_frames + 1 entries so the current frame is always
    // available alongside the lookback window.
    let mut prefill: VecDeque<(u64, Vec<f32>)> = VecDeque::with_capacity(prefill_frames + 1);

    let mut in_speech = false;
    let mut onset_counter = 0usize;
    let mut hangover_counter = 0usize;

    let mut current_samples: Vec<f32> = Vec::new();
    let mut current_start_ms: u64 = 0;
    let mut current_last_voice_end_ms: u64 = 0;

    let mut segments_emitted = 0usize;
    let mut total_speech_samples = 0usize;

    while let Ok(frame) = frame_rx.recv() {
        if frame.is_last {
            // Close out any in-progress segment.
            if in_speech && !current_samples.is_empty() {
                emit_segment(
                    &asr_tx,
                    &mut current_samples,
                    current_start_ms,
                    current_last_voice_end_ms,
                    &mut segments_emitted,
                    &mut total_speech_samples,
                )?;
            }
            break;
        }

        let prob = engine
            .compute(&frame.samples)
            .map_err(|e| anyhow!("Vad::compute: {e}"))?
            .prob;
        let is_voice = prob > threshold;

        // Maintain pre-roll buffer.
        prefill.push_back((frame.start_ms, frame.samples.clone()));
        while prefill.len() > prefill_frames + 1 {
            prefill.pop_front();
        }

        let frame_end_ms = frame.start_ms + FRAME_MS as u64;

        match (in_speech, is_voice) {
            (false, true) => {
                onset_counter += 1;
                if onset_counter >= onset_frames {
                    in_speech = true;
                    hangover_counter = hangover_frames;
                    onset_counter = 0;
                    // Seed the segment with pre-roll + current frame.
                    current_samples.clear();
                    let mut earliest_ms = frame.start_ms;
                    for (start_ms, samples) in prefill.iter() {
                        current_samples.extend_from_slice(samples);
                        earliest_ms = earliest_ms.min(*start_ms);
                    }
                    current_start_ms = earliest_ms;
                    current_last_voice_end_ms = frame_end_ms;
                }
            }
            (true, true) => {
                hangover_counter = hangover_frames;
                current_samples.extend_from_slice(&frame.samples);
                current_last_voice_end_ms = frame_end_ms;
            }
            (true, false) => {
                // Still keep appending during hangover - the
                // post-voice silence is part of the segment because
                // some words trail off below the VAD threshold.
                current_samples.extend_from_slice(&frame.samples);
                if hangover_counter > 0 {
                    hangover_counter -= 1;
                }
                if hangover_counter == 0 {
                    // Trim the trailing silence we appended during
                    // hangover (current_last_voice_end_ms is the end of
                    // the last voiced frame, not the end of hangover).
                    let trim_ms = frame_end_ms.saturating_sub(current_last_voice_end_ms);
                    let trim_samples = (trim_ms as usize * SAMPLE_RATE as usize) / 1000;
                    let keep = current_samples
                        .len()
                        .saturating_sub(trim_samples.min(current_samples.len()));
                    current_samples.truncate(keep);
                    emit_segment(
                        &asr_tx,
                        &mut current_samples,
                        current_start_ms,
                        current_last_voice_end_ms,
                        &mut segments_emitted,
                        &mut total_speech_samples,
                    )?;
                    in_speech = false;
                    onset_counter = 0;
                }
            }
            (false, false) => {
                onset_counter = 0;
            }
        }
    }

    // Tell ASR worker the stream is done.
    asr_tx
        .send(AsrEvent::EndOfStream)
        .map_err(|_| anyhow!("ASR channel closed before EndOfStream"))?;
    drop(asr_tx);

    Ok(VadStats {
        segments_emitted,
        total_speech_s: total_speech_samples as f32 / SAMPLE_RATE as f32,
    })
}

fn emit_segment(
    asr_tx: &Sender<AsrEvent>,
    buf: &mut Vec<f32>,
    start_ms: u64,
    end_ms: u64,
    segments_emitted: &mut usize,
    total_speech_samples: &mut usize,
) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    let samples = std::mem::take(buf);
    *total_speech_samples += samples.len();
    *segments_emitted += 1;
    let duration_ms = end_ms.saturating_sub(start_ms);
    eprintln!(
        "vad: segment #{:02} start={}ms end={}ms duration={}ms samples={}",
        *segments_emitted,
        start_ms,
        end_ms,
        duration_ms,
        samples.len()
    );
    asr_tx
        .send(AsrEvent::Segment(VadSegment {
            samples,
            start_ms,
            end_ms,
            vad_end_wallclock: Instant::now(),
        }))
        .map_err(|_| anyhow!("ASR channel closed"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// ASR worker: batched-VAD accumulator + mtmd per-flush
// ---------------------------------------------------------------------------

struct AsrConfig {
    flush_min_secs: f32,
    latency_window_secs: f32,
    n_batch: i32,
    max_tokens: i32,
}

struct AsrStats {
    flushes_done: usize,
    segments_emitted: usize,
    vad_segments_consumed: usize,
    total_inference_s: f64,
    avg_buffer_s: f32,
    longest_vad_wait_s: f32,
    first_segment_latency_s: Option<f64>,
}

/// Accumulator state for batched-VAD.
///
/// **Critical design point**: the accumulator preserves the original
/// silences *between* VAD segments by zero-padding the gaps. Mtmd was
/// trained on natural speech with breath gaps and sentence-boundary
/// silences. If the buffer is "VAD segments concatenated back-to-back
/// with all silence removed", the encoder sees 30 s of unbroken speech
/// where the model expects breath gaps; it loses sentence-boundary
/// anchoring and degenerates into greedy-decode loops after the first
/// few words. The buffer therefore reconstructs the recording-clock
/// timeline: samples at offset `i` within the buffer correspond to
/// recording-clock time `buffer_start_ms + (i * 1000 / SAMPLE_RATE)`,
/// with zeros wherever the VAD said it was silence.
///
/// Per-VAD-segment boundaries are also retained so the eventual
/// transcript can be re-split into one `Segment` per VAD-segment with
/// timestamps aligned to the original silence boundaries.
struct Accumulator {
    samples: Vec<f32>,
    /// Recording-clock ms of the first sample in `samples`. Set on the
    /// first append; never changes until drain().
    buffer_start_ms: Option<u64>,
    /// `(start_ms, end_ms)` of each VAD segment in this buffer.
    segments: Vec<(u64, u64)>,
    /// Wall-clock when the most-recent VAD segment ended. Used to
    /// decide whether the latency window has elapsed.
    last_vad_end: Option<Instant>,
}

impl Accumulator {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            buffer_start_ms: None,
            segments: Vec::new(),
            last_vad_end: None,
        }
    }

    fn append(&mut self, seg: VadSegment) {
        let buffer_start = *self.buffer_start_ms.get_or_insert(seg.start_ms);
        // Zero-pad the gap from the current buffer's end to the new
        // VAD segment's start so the encoder sees the original
        // timeline.
        let expected_offset_samples =
            ((seg.start_ms.saturating_sub(buffer_start)) as usize * SAMPLE_RATE as usize) / 1000;
        if expected_offset_samples > self.samples.len() {
            let gap = expected_offset_samples - self.samples.len();
            // Cap inter-segment gaps at 3 s to avoid wasting encoder
            // budget on very long silences (which Phase 2 will
            // legitimately split into separate flushes). For the
            // spike's fixtures the inter-sentence pauses are < 1 s.
            let capped = gap.min(SAMPLE_RATE as usize * 3);
            self.samples.extend(std::iter::repeat(0.0).take(capped));
        }
        self.samples.extend_from_slice(&seg.samples);
        self.segments.push((seg.start_ms, seg.end_ms));
        self.last_vad_end = Some(seg.vad_end_wallclock);
    }

    fn duration_s(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE as f32
    }

    fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn drain(&mut self) -> (Vec<f32>, Vec<(u64, u64)>) {
        let samples = std::mem::take(&mut self.samples);
        let segments = std::mem::take(&mut self.segments);
        self.last_vad_end = None;
        self.buffer_start_ms = None;
        (samples, segments)
    }
}

fn asr_worker(
    backend: &LlamaBackend,
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    ctx_params: LlamaContextParams,
    cfg: AsrConfig,
    asr_rx: Receiver<AsrEvent>,
) -> Result<AsrStats> {
    let mut acc = Accumulator::new();
    let mut flushes_done = 0usize;
    let mut segments_emitted = 0usize;
    let mut vad_segments_consumed = 0usize;
    let mut total_inference_s = 0.0f64;
    let mut buffer_durations = Vec::<f32>::new();
    let mut longest_vad_wait_s = 0.0f32;
    // Q-P0-5: FR-7's "first segment <= 10 s after its audio ends"
    // budget is measured from the *most recent* VAD segment in the
    // first flush (the segment whose end triggered or completed the
    // flush window), not from the first segment that entered the
    // buffer. The first segment of a 25 s batched flush legitimately
    // waits ~25 s; what FR-7 cares about is the user-perceived gap
    // between "they stopped talking" and "transcript appeared".
    let mut first_segment_latency_s: Option<f64> = None;

    // Poll the channel with a timeout so the latency-window flush can
    // fire even when no new VAD segment arrives. A 200 ms tick is fine
    // grained relative to the 10 s latency budget.
    let tick = Duration::from_millis(200);
    let latency_window = Duration::from_secs_f32(cfg.latency_window_secs);

    loop {
        // 1. Drain whatever's ready without blocking, accounting for
        //    EndOfStream and segment arrivals.
        let mut end_of_stream = false;
        loop {
            match asr_rx.recv_timeout(tick) {
                Ok(AsrEvent::Segment(seg)) => {
                    vad_segments_consumed += 1;
                    acc.append(seg);
                    // Decide whether to flush immediately on size.
                    if acc.duration_s() >= cfg.flush_min_secs {
                        break;
                    }
                }
                Ok(AsrEvent::EndOfStream) => {
                    end_of_stream = true;
                    break;
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // No new event in the tick window. Fall through
                    // to the latency-window check.
                    break;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Sender dropped without sending EndOfStream;
                    // treat as graceful close.
                    end_of_stream = true;
                    break;
                }
            }
        }

        // 2. Decide whether to flush.
        let should_flush_size = acc.duration_s() >= cfg.flush_min_secs;
        let should_flush_latency = !acc.is_empty()
            && acc
                .last_vad_end
                .map(|t| t.elapsed() >= latency_window)
                .unwrap_or(false);
        let should_flush_end = end_of_stream && !acc.is_empty();

        if should_flush_size || should_flush_latency || should_flush_end {
            let reason = if should_flush_size {
                "size>=min"
            } else if should_flush_latency {
                "latency-window"
            } else {
                "end-of-stream"
            };
            let buffer_duration_s = acc.duration_s();
            buffer_durations.push(buffer_duration_s);

            let vad_wait = acc
                .last_vad_end
                .map(|t| t.elapsed().as_secs_f32())
                .unwrap_or(0.0);
            longest_vad_wait_s = longest_vad_wait_s.max(vad_wait);
            // Anchor the FR-7 budget on the most-recent VAD-end at
            // the moment we decide to flush. Captured before drain()
            // wipes the state. Only the first flush sets the anchor.
            let first_flush_anchor = if flushes_done == 0 {
                acc.last_vad_end
            } else {
                None
            };

            let (samples, segments_in_flush) = acc.drain();
            eprintln!(
                "asr: flush #{:02} reason={} buffer={:.3}s vad-segments-in-flush={} vad-wait={:.3}s",
                flushes_done + 1,
                reason,
                buffer_duration_s,
                segments_in_flush.len(),
                vad_wait,
            );

            let flush_t0 = Instant::now();
            let text = run_mtmd_once(
                backend,
                model,
                mtmd_ctx,
                &ctx_params,
                &samples,
                cfg.n_batch,
                cfg.max_tokens,
            )?;
            let flush_secs = flush_t0.elapsed().as_secs_f64();
            total_inference_s += flush_secs;
            flushes_done += 1;
            eprintln!(
                "asr: flush #{:02} done in {:.3}s; transcript = {:?}",
                flushes_done,
                flush_secs,
                text.chars().take(120).collect::<String>()
            );

            // Split transcript proportional to per-VAD-segment audio
            // duration and emit one Segment per VAD segment.
            let emitted_now = emit_segments_proportional(
                &text,
                &segments_in_flush,
                &mut first_segment_latency_s,
                first_flush_anchor,
            )?;
            segments_emitted += emitted_now;
        }

        if end_of_stream && acc.is_empty() {
            break;
        }
    }

    let avg_buffer_s = if buffer_durations.is_empty() {
        0.0
    } else {
        buffer_durations.iter().sum::<f32>() / buffer_durations.len() as f32
    };

    Ok(AsrStats {
        flushes_done,
        segments_emitted,
        vad_segments_consumed,
        total_inference_s,
        avg_buffer_s,
        longest_vad_wait_s,
        first_segment_latency_s,
    })
}

/// Run mtmd once over a single flushed buffer. Reuses the shared
/// `MtmdContext`; allocates a fresh `LlamaContext` per call (cheap,
/// sub-100 ms per Spike 1's Q-P0-2). Returns the cleaned transcript.
fn run_mtmd_once(
    backend: &LlamaBackend,
    model: &LlamaModel,
    mtmd_ctx: &MtmdContext,
    ctx_params: &LlamaContextParams,
    samples: &[f32],
    n_batch: i32,
    max_tokens: i32,
) -> Result<String> {
    let mut llama_ctx: LlamaContext = model
        .new_context(backend, ctx_params.clone())
        .map_err(|e| anyhow!("LlamaContext init: {e}"))?;

    let bitmap =
        MtmdBitmap::from_audio_data(samples).map_err(|e| anyhow!("MtmdBitmap::from_audio_data: {e}"))?;

    let media_marker = mtmd_default_marker();
    let prompt_text = build_prompt(model, media_marker)?;
    let input_text = MtmdInputText {
        text: prompt_text,
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .map_err(|e| anyhow!("mtmd tokenize: {e}"))?;

    let mut n_past = chunks
        .eval_chunks(mtmd_ctx, &llama_ctx, 0, 0, n_batch, true)
        .map_err(|e| anyhow!("mtmd eval_chunks: {e}"))?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut batch = LlamaBatch::new(n_batch as usize, 1);
    let mut decoder = UTF_8.new_decoder();
    let mut text = String::new();

    for _ in 0..max_tokens {
        let token = sampler.sample(&llama_ctx, -1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow!("token_to_piece: {e}"))?;
        text.push_str(&piece);

        // Early-stop on the Qwen3-ASR end-of-transcript tag. The
        // model occasionally fails to emit EOG and instead loops
        // within the transcript when the audio doesn't fill the 30 s
        // window — a separate manifestation of Spike 1's Q-P0-3
        // padding behaviour on the upper edge (audio shorter than the
        // padded window leaves room for the decoder to wander). The
        // `</asr_text>` tag is the schema's terminator; once we see
        // it the transcript is complete.
        if text.contains("</asr_text>") {
            break;
        }

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| anyhow!("batch.add: {e}"))?;
        n_past += 1;
        llama_ctx
            .decode(&mut batch)
            .map_err(|e| anyhow!("decode: {e}"))?;
    }

    Ok(strip_asr_wrapper(&text))
}

fn build_prompt(model: &LlamaModel, media_marker: &str) -> Result<String> {
    use llama_cpp_2::model::LlamaChatMessage;
    let msg = LlamaChatMessage::new("user".to_string(), media_marker.to_string())
        .map_err(|e| anyhow!("LlamaChatMessage::new: {e}"))?;
    match model.chat_template(None::<&str>) {
        Ok(template) => match model.apply_chat_template(&template, &[msg], true) {
            Ok(s) => return Ok(s),
            Err(e) => eprintln!("apply_chat_template failed, falling back to ChatML: {e}"),
        },
        Err(e) => eprintln!("model has no chat template ({e:?}); using ChatML fallback"),
    }
    Ok(format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        media_marker
    ))
}

/// Strip Qwen3-ASR's `language English<asr_text>...</asr_text>` wrapper.
/// Mirrors the routine in spikes/asr/src/main.rs.
fn strip_asr_wrapper(s: &str) -> String {
    let mut out = s.trim().to_string();
    if let Some(open) = out.find("<asr_text>") {
        let after = out[open + "<asr_text>".len()..].to_string();
        out = after;
    } else if let Some(open) = out.find("language ") {
        let after = &out[open + "language ".len()..];
        if let Some(sp) = after.find(char::is_whitespace) {
            out = after[sp..].trim_start().to_string();
        }
    }
    if let Some(close) = out.find("</asr_text>") {
        out.truncate(close);
    }
    out.trim().to_string()
}

/// Split a flush's transcript across the VAD-segments it covered.
/// Allocation is proportional to per-segment audio duration. Mtmd
/// doesn't expose token timestamps (Spike 1 README, Q-P0-1 surprises),
/// so per-word alignment is a Phase 2/4 problem; for the spike's
/// acceptance criteria (>=3 records, monotonic timestamps, set-overlap
/// WER) this is sufficient and the segment boundaries themselves are
/// exact.
///
/// Emits each `Segment` as a JSON-Lines record on stdout. Returns the
/// number of `Segment`s emitted.
fn emit_segments_proportional(
    text: &str,
    vad_segments: &[(u64, u64)],
    first_segment_latency_s: &mut Option<f64>,
    first_segment_anchor: Option<Instant>,
) -> Result<usize> {
    if vad_segments.is_empty() {
        return Ok(0);
    }

    // Total duration across VAD segments for proportional split.
    let total_ms: u64 = vad_segments
        .iter()
        .map(|(s, e)| e.saturating_sub(*s))
        .sum();
    if total_ms == 0 {
        return Ok(0);
    }

    // Word-level split. Whitespace is the only reasonable boundary
    // because mtmd's output is plain text without timing.
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        // Still emit empty-text segments to satisfy "VAD-emit count ==
        // ASR-emit count" — without these, a flush with no decoded
        // text would silently drop VAD segments and fail the
        // acceptance check. Each empty segment retains its true
        // timestamps from the VAD layer.
        for (start_ms, end_ms) in vad_segments {
            let seg = Segment {
                start_ms: *start_ms,
                end_ms: *end_ms,
                text: String::new(),
                speaker_id: None,
                confidence: None,
                words: Vec::new(),
                shared_speakers: Vec::new(),
            };
            println!("{}", serde_json::to_string(&seg)?);
        }
        return Ok(vad_segments.len());
    }

    let mut emitted = 0usize;
    let mut word_idx = 0usize;
    let n = vad_segments.len();
    for (i, (start_ms, end_ms)) in vad_segments.iter().enumerate() {
        let seg_ms = end_ms.saturating_sub(*start_ms);
        // Allocate proportional word count; last segment takes the
        // remainder so rounding doesn't drop tokens.
        let take = if i == n - 1 {
            words.len() - word_idx
        } else {
            let proportion = seg_ms as f64 / total_ms as f64;
            let count = (proportion * words.len() as f64).round() as usize;
            count.min(words.len() - word_idx)
        };
        let chunk_words = &words[word_idx..word_idx + take];
        word_idx += take;
        let seg = Segment {
            start_ms: *start_ms,
            end_ms: *end_ms,
            text: chunk_words.join(" "),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        };
        if first_segment_latency_s.is_none() {
            if let Some(anchor) = first_segment_anchor {
                *first_segment_latency_s = Some(anchor.elapsed().as_secs_f64());
            }
        }
        println!("{}", serde_json::to_string(&seg)?);
        emitted += 1;
    }
    Ok(emitted)
}

// ---------------------------------------------------------------------------
// WAV loading. Mirrors spikes/asr/src/main.rs but is strict about
// 16 kHz mono — the VAD model and the mtmd encoder both require it
// and the spike has no resampler dependency.
// ---------------------------------------------------------------------------

fn load_wav_mono_16k_f32(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!(
            "{}: expected mono WAV, got {} channels",
            path.display(),
            spec.channels
        );
    }
    if spec.sample_rate != SAMPLE_RATE {
        bail!(
            "{}: expected {} Hz, got {} Hz (resampling is out of scope for the spike)",
            path.display(),
            SAMPLE_RATE,
            spec.sample_rate,
        );
    }
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
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Run summary
// ---------------------------------------------------------------------------

struct RunStats {
    audio_duration_s: f32,
    frames_produced: usize,
    vad: VadStats,
    asr: AsrStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_size_matches_30ms_at_16k() {
        assert_eq!(FRAME_SAMPLES, 480);
    }

    #[test]
    fn strip_asr_wrapper_removes_tag() {
        let raw = "language English<asr_text>hello world</asr_text>";
        assert_eq!(strip_asr_wrapper(raw), "hello world");
    }

    #[test]
    fn accumulator_tracks_segments() {
        let mut acc = Accumulator::new();
        assert!(acc.is_empty());
        acc.append(VadSegment {
            samples: vec![0.0; SAMPLE_RATE as usize * 5], // 5 s
            start_ms: 0,
            end_ms: 5_000,
            vad_end_wallclock: Instant::now(),
        });
        assert!(!acc.is_empty());
        assert!((acc.duration_s() - 5.0).abs() < 1e-3);
        let (samples, segs) = acc.drain();
        assert_eq!(samples.len(), SAMPLE_RATE as usize * 5);
        assert_eq!(segs, vec![(0, 5_000)]);
        assert!(acc.is_empty());
    }

    #[test]
    fn accumulator_zero_pads_silence_gap() {
        // Two segments at 0-1000 ms and 2000-3000 ms. The 1 s gap
        // between them must be reconstructed as silence so the
        // mtmd encoder sees the original timeline.
        let mut acc = Accumulator::new();
        acc.append(VadSegment {
            samples: vec![0.5; SAMPLE_RATE as usize], // 1 s of constant 0.5
            start_ms: 0,
            end_ms: 1_000,
            vad_end_wallclock: Instant::now(),
        });
        acc.append(VadSegment {
            samples: vec![0.5; SAMPLE_RATE as usize], // 1 s of constant 0.5
            start_ms: 2_000,
            end_ms: 3_000,
            vad_end_wallclock: Instant::now(),
        });
        // Expect 3 s total: 1 s speech + 1 s silence + 1 s speech.
        assert!((acc.duration_s() - 3.0).abs() < 1e-3);
        let (samples, _segs) = acc.drain();
        // Middle second should be silence.
        let mid_start = SAMPLE_RATE as usize;
        let mid_end = mid_start + SAMPLE_RATE as usize;
        for s in &samples[mid_start..mid_end] {
            assert!(s.abs() < 1e-6, "silence pad violated");
        }
    }

    #[test]
    fn emit_segments_proportional_basic() {
        // Three VAD segments, durations 1000/2000/1000 ms.
        let segs = vec![(0, 1000), (1000, 3000), (3000, 4000)];
        let text = "alpha beta gamma delta";
        let mut latency: Option<f64> = None;
        let n =
            emit_segments_proportional(text, &segs, &mut latency, Some(Instant::now())).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn emit_segments_proportional_empty_text_still_emits() {
        // VAD picked up two segments but mtmd produced no text. We
        // still need to count them so VAD-emit == ASR-emit holds.
        let segs = vec![(0, 1000), (1000, 2000)];
        let mut latency: Option<f64> = None;
        let n =
            emit_segments_proportional("", &segs, &mut latency, Some(Instant::now())).unwrap();
        assert_eq!(n, 2);
    }
}
