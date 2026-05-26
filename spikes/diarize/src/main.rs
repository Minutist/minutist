//! Phase 0 Spike 4 — sherpa-onnx offline speaker diarization.
//!
//! End-to-end CLI: read a 16 kHz mono WAV, drive `sherpa_rs::diarize::Diarize`
//! with a pyannote segmentation ONNX + a speaker-embedding ONNX, emit
//! `{start_ms, end_ms, speaker_id}` segments as a JSON array on stdout,
//! and (optionally) score Diarization Error Rate (DER) against a reference
//! RTTM.
//!
//! This is throwaway spike code (see `architecture/cross-cutting.md`). Spike crates are exempt from
//! production cross-cutting rules: `anyhow`, `eprintln!`, and `println!`
//! on stdout are all fine here.
//!
//! The primary information goal is **Q-P0-6**: is the Rust binding mature
//! enough to ship in Phase 6, or does Phase 6 need to wrap sherpa-onnx's
//! C API directly via `bindgen`? See `spikes/diarize/README.md` for the
//! verdict.
//!
//! Acceptance criteria mirror Phase 0 Spike 4.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use hound::{SampleFormat, WavReader};
use serde_json::json;
use sha2::{Digest, Sha256};
use sherpa_rs::diarize::{Diarize, DiarizeConfig, Segment as SherpaSegment};

#[derive(Debug, Parser)]
#[command(
    name = "spike-diarize",
    about = "Phase 0 Spike 4: sherpa-onnx offline speaker diarization end-to-end"
)]
struct Cli {
    /// Path to the pyannote segmentation ONNX (e.g.
    /// `sherpa-onnx-pyannote-segmentation-3-0/model.onnx`).
    #[arg(long, value_name = "PATH")]
    segmentation: PathBuf,

    /// Path to the speaker embedding ONNX (e.g.
    /// `nemo_en_titanet_small.onnx`).
    #[arg(long, value_name = "PATH")]
    embedding: PathBuf,

    /// 16 kHz mono WAV file. PCM s16 or f32 accepted; multi-channel inputs
    /// are downmixed to mono (channel 0 only, matching the upstream
    /// `offline-speaker-diarization.py` example).
    #[arg(short = 'w', long, value_name = "PATH")]
    wav: PathBuf,

    /// Known number of speakers. If positive, exact-cluster mode is used;
    /// if omitted or <= 0, threshold-based clustering is used instead.
    #[arg(long, value_name = "N")]
    num_clusters: Option<i32>,

    /// Cluster threshold for auto-cluster mode (smaller -> more speakers).
    /// Default 0.5 matches sherpa-onnx's `OfflineSpeakerDiarizationConfig`
    /// default; only consulted when `--num-clusters` is unset.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Minimum on-duration for a speaker turn (s). Default 0.3 matches
    /// the upstream example. Set to 0 to keep all segments.
    #[arg(long, default_value_t = 0.3)]
    min_duration_on: f32,

    /// Minimum off-duration before a new turn (s). Smaller -> finer
    /// segmentation. Default 0.5 matches the upstream example.
    #[arg(long, default_value_t = 0.5)]
    min_duration_off: f32,

    /// Optional RTTM reference for DER scoring. Format: one speech turn
    /// per line, fields:
    ///   `SPEAKER <file_id> 1 <start_s> <dur_s> <NA> <NA> <speaker> <NA> <NA>`
    /// (NIST md-eval RTTM v1.0). When provided, DER is printed to stderr.
    #[arg(long, value_name = "PATH")]
    reference_rttm: Option<PathBuf>,

    /// Optional path to write the binding's output as an RTTM. Useful for
    /// piping into `pyannote.metrics` or `dscore` for an independent DER
    /// check.
    #[arg(long, value_name = "PATH")]
    output_rttm: Option<PathBuf>,

    /// Speaker-label format. `numeric` -> `0`, `1`, ...; `alpha` -> `A`,
    /// `B`, ... (matches the FR-12 anonymous-label convention).
    #[arg(long, default_value = "alpha")]
    label_format: String,

    /// Override the sherpa-onnx ONNX-Runtime execution provider. Default
    /// is auto (CPU on Linux WSL); other choices: `cpu`, `cuda`, `coreml`,
    /// `directml`.
    #[arg(long, value_name = "PROVIDER")]
    provider: Option<String>,
}

#[derive(Debug, Clone)]
struct SpeakerTurn {
    start_s: f32,
    end_s: f32,
    speaker: String,
}

/// Output segment for the JSON-on-stdout contract.
#[derive(Debug, Clone)]
struct OutSegment {
    start_ms: u64,
    end_ms: u64,
    speaker_id: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,sherpa_rs=warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    // Resolve and report file hashes — pinning model identity is part of
    // the spike's deliverables.
    let seg_sha = sha256_of(&cli.segmentation)?;
    let emb_sha = sha256_of(&cli.embedding)?;
    let wav_sha = sha256_of(&cli.wav)?;
    tracing::info!(
        path = %cli.segmentation.display(),
        sha256 = %seg_sha,
        "segmentation model"
    );
    tracing::info!(
        path = %cli.embedding.display(),
        sha256 = %emb_sha,
        "embedding model"
    );
    tracing::info!(path = %cli.wav.display(), sha256 = %wav_sha, "input wav");

    // ---- Load + decode WAV ----------------------------------------------
    let load_start = Instant::now();
    let (samples, sample_rate, duration_s) = read_wav_mono_f32(&cli.wav)
        .with_context(|| format!("reading WAV {}", cli.wav.display()))?;
    tracing::info!(
        n_samples = samples.len(),
        sample_rate,
        duration_s,
        elapsed_ms = load_start.elapsed().as_millis(),
        "decoded wav"
    );

    if sample_rate != 16_000 {
        bail!(
            "sherpa-onnx pyannote segmentation expects 16 kHz; got {} Hz. \
             Resample externally (sox / ffmpeg) and re-run.",
            sample_rate
        );
    }

    // ---- Configure + construct Diarize ----------------------------------
    let cfg = DiarizeConfig {
        // -1 in the binding means "use threshold instead". Map clap's
        // Option<i32> to that contract.
        num_clusters: cli.num_clusters.filter(|n| *n > 0).or(Some(-1)),
        threshold: Some(cli.threshold),
        min_duration_on: Some(cli.min_duration_on),
        min_duration_off: Some(cli.min_duration_off),
        provider: cli.provider.clone(),
        debug: false,
    };
    tracing::info!(?cfg, "DiarizeConfig");

    let construct_start = Instant::now();
    let mut diarizer = Diarize::new(&cli.segmentation, &cli.embedding, cfg)
        .map_err(|e| anyhow!("Diarize::new failed: {e:?}"))?;
    tracing::info!(
        elapsed_ms = construct_start.elapsed().as_millis(),
        "Diarize constructed"
    );

    // ---- Run inference --------------------------------------------------
    let infer_start = Instant::now();
    let raw_segments = diarizer
        .compute(samples.clone(), None)
        .map_err(|e| anyhow!("Diarize::compute failed: {e:?}"))?;
    let infer_elapsed = infer_start.elapsed();
    let rtf = infer_elapsed.as_secs_f64() / duration_s as f64;
    tracing::info!(
        n_segments = raw_segments.len(),
        elapsed_ms = infer_elapsed.as_millis(),
        rtf = format!("{:.3}", rtf),
        "diarization complete"
    );

    if raw_segments.is_empty() {
        bail!("sherpa-onnx returned zero segments — model output empty");
    }

    // ---- Normalise + relabel --------------------------------------------
    let labelled = relabel(&raw_segments, &cli.label_format)?;

    let distinct: std::collections::BTreeSet<&str> =
        labelled.iter().map(|s| s.speaker_id.as_str()).collect();
    tracing::info!(distinct_speakers = distinct.len(), "speakers");

    // ---- DER (if reference provided) ------------------------------------
    if let Some(ref_path) = cli.reference_rttm.as_ref() {
        let reference = load_rttm(ref_path)
            .with_context(|| format!("reading reference RTTM {}", ref_path.display()))?;
        let der = der_score(&reference, &turns_from(&labelled), duration_s);
        tracing::info!(
            der_pct = format!("{:.2}", der.der * 100.0),
            miss_pct = format!("{:.2}", der.miss * 100.0),
            false_alarm_pct = format!("{:.2}", der.false_alarm * 100.0),
            confusion_pct = format!("{:.2}", der.confusion * 100.0),
            total_ref_speech_s = format!("{:.2}", der.total_ref_speech_s),
            "DER scored"
        );
        eprintln!(
            "DER {:.2}%  (miss {:.2}%, FA {:.2}%, confusion {:.2}%)",
            der.der * 100.0,
            der.miss * 100.0,
            der.false_alarm * 100.0,
            der.confusion * 100.0
        );
    }

    // ---- Optional RTTM output -------------------------------------------
    if let Some(out) = cli.output_rttm.as_ref() {
        let file_id = cli
            .wav
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio")
            .to_string();
        write_rttm(out, &file_id, &labelled).with_context(|| {
            format!("writing output RTTM {}", out.display())
        })?;
        tracing::info!(path = %out.display(), "wrote RTTM");
    }

    // ---- JSON output on stdout (the acceptance contract) ----------------
    let json_segments: Vec<_> = labelled
        .iter()
        .map(|s| {
            json!({
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "speaker_id": s.speaker_id,
            })
        })
        .collect();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer_pretty(&mut out, &json_segments)?;
    writeln!(out)?;

    // ---- Final spike-grade banner to stderr -----------------------------
    eprintln!(
        "ok n_segments={} distinct_speakers={} duration_s={:.2} inference_s={:.3} rtf={:.3}",
        labelled.len(),
        distinct.len(),
        duration_s,
        infer_elapsed.as_secs_f64(),
        rtf
    );

    Ok(())
}

// ===========================================================================
// WAV decoding
// ===========================================================================

fn read_wav_mono_f32(path: &Path) -> Result<(Vec<f32>, u32, f32)> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => {
            let r = reader;
            r.into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?
        }
        (SampleFormat::Int, 16) => {
            let r = reader;
            r.into_samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32_768.0))
                .collect::<Result<Vec<_>, _>>()?
        }
        (SampleFormat::Int, 32) => {
            let r = reader;
            r.into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 2_147_483_648.0))
                .collect::<Result<Vec<_>, _>>()?
        }
        other => bail!("unsupported WAV format {:?}", other),
    };

    // Downmix multi-channel to channel-0 only (matches upstream example).
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|frame| frame[0])
            .collect()
    };

    let duration_s = mono.len() as f32 / sample_rate as f32;
    Ok((mono, sample_rate, duration_s))
}

// ===========================================================================
// Speaker-label normalisation
// ===========================================================================

fn relabel(raw: &[SherpaSegment], format: &str) -> Result<Vec<OutSegment>> {
    // Preserve first-seen order so labels are stable across runs on the
    // same input. sherpa returns speakers as i32 ids (cluster ids); these
    // are not guaranteed to be 0..N-1 ordered by appearance.
    let mut seen: Vec<i32> = Vec::new();
    for s in raw {
        if !seen.contains(&s.speaker) {
            seen.push(s.speaker);
        }
    }

    let labels: Vec<String> = match format {
        "alpha" => (0..seen.len())
            .map(|i| alpha_label(i))
            .collect(),
        "numeric" => (0..seen.len()).map(|i| i.to_string()).collect(),
        other => bail!("unknown --label-format: {other} (use alpha|numeric)"),
    };

    let mut out = Vec::with_capacity(raw.len());
    for s in raw {
        let idx = seen
            .iter()
            .position(|x| *x == s.speaker)
            .expect("seen contains every speaker id");
        out.push(OutSegment {
            start_ms: (s.start * 1000.0).round() as u64,
            end_ms: (s.end * 1000.0).round() as u64,
            speaker_id: labels[idx].clone(),
        });
    }
    Ok(out)
}

/// 0->"A", 1->"B", ..., 25->"Z", 26->"AA", 27->"AB", ...
fn alpha_label(mut n: usize) -> String {
    let mut out = String::new();
    loop {
        out.insert(0, (b'A' + (n % 26) as u8) as char);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    out
}

// ===========================================================================
// DER scoring (frame-level, with greedy speaker-mapping)
// ===========================================================================

#[derive(Debug, Clone)]
struct DerResult {
    der: f32,
    miss: f32,
    false_alarm: f32,
    confusion: f32,
    total_ref_speech_s: f32,
}

fn turns_from(segs: &[OutSegment]) -> Vec<SpeakerTurn> {
    segs.iter()
        .map(|s| SpeakerTurn {
            start_s: s.start_ms as f32 / 1000.0,
            end_s: s.end_ms as f32 / 1000.0,
            speaker: s.speaker_id.clone(),
        })
        .collect()
}

/// Frame-quantised DER. Quantises both reference and hypothesis to a 10 ms
/// grid, builds a (ref-speaker × hyp-speaker) overlap matrix, greedily
/// assigns hyp -> ref to maximise overlap (one-to-one), then sums:
///   miss          = frames where ref has speech but hyp does not
///   false_alarm   = frames where hyp has speech but ref does not
///   confusion     = frames where both have speech but hyp's mapped speaker
///                   != ref's speaker
///   DER           = (miss + FA + confusion) / total_ref_speech
///
/// This is a simplified md-eval pulled inline — accurate to roughly 10 ms
/// per turn, sufficient to gate the §2 Spike 4 acceptance (DER ≤ 25%).
/// No collar or overlap detection — the fixtures are non-overlapping.
fn der_score(reference: &[SpeakerTurn], hypothesis: &[SpeakerTurn], duration_s: f32) -> DerResult {
    const FRAME_S: f32 = 0.01;
    let n_frames = (duration_s / FRAME_S).ceil() as usize + 1;

    let (ref_labels, ref_grid) = paint(reference, n_frames, FRAME_S);
    let (hyp_labels, hyp_grid) = paint(hypothesis, n_frames, FRAME_S);

    // overlap[r][h] = number of frames where ref has speaker r AND hyp has speaker h
    let mut overlap = vec![vec![0u32; hyp_labels.len()]; ref_labels.len()];
    for i in 0..n_frames {
        if let (Some(r), Some(h)) = (ref_grid[i], hyp_grid[i]) {
            overlap[r][h] += 1;
        }
    }

    // Greedy best-pair assignment. The strict NIST mapping uses Hungarian;
    // greedy is sufficient for ≤ ~6 speakers (the v1 ceiling) and easier
    // to audit. Picks the largest overlap cell, assigns, removes the row
    // and column, repeats until exhausted.
    let r_count = ref_labels.len();
    let h_count = hyp_labels.len();
    let mut hyp_to_ref: Vec<Option<usize>> = vec![None; h_count];
    let mut used_ref = vec![false; r_count];
    let mut used_hyp = vec![false; h_count];
    loop {
        let mut best: Option<(usize, usize, u32)> = None;
        for r in 0..r_count {
            if used_ref[r] {
                continue;
            }
            for h in 0..h_count {
                if used_hyp[h] {
                    continue;
                }
                if overlap[r][h] == 0 {
                    continue;
                }
                let val = overlap[r][h];
                if best.map_or(true, |(_, _, v)| val > v) {
                    best = Some((r, h, val));
                }
            }
        }
        match best {
            Some((r, h, _)) => {
                hyp_to_ref[h] = Some(r);
                used_ref[r] = true;
                used_hyp[h] = true;
            }
            None => break,
        }
    }

    let mut miss_frames: u32 = 0;
    let mut fa_frames: u32 = 0;
    let mut conf_frames: u32 = 0;
    let mut ref_speech_frames: u32 = 0;
    for i in 0..n_frames {
        let r = ref_grid[i];
        let h = hyp_grid[i];
        if r.is_some() {
            ref_speech_frames += 1;
        }
        match (r, h) {
            (Some(_), None) => miss_frames += 1,
            (None, Some(_)) => fa_frames += 1,
            (Some(r), Some(h)) => {
                if hyp_to_ref[h] != Some(r) {
                    conf_frames += 1;
                }
            }
            (None, None) => {}
        }
    }

    let total = ref_speech_frames.max(1) as f32;
    let total_ref_speech_s = ref_speech_frames as f32 * FRAME_S;
    DerResult {
        der: (miss_frames + fa_frames + conf_frames) as f32 / total,
        miss: miss_frames as f32 / total,
        false_alarm: fa_frames as f32 / total,
        confusion: conf_frames as f32 / total,
        total_ref_speech_s,
    }
}

/// Paint turns onto a 10 ms grid. Returns (label list, per-frame -> label index or None).
/// Later turns overwrite earlier ones on overlap (not relevant for these fixtures).
fn paint(turns: &[SpeakerTurn], n_frames: usize, frame_s: f32) -> (Vec<String>, Vec<Option<usize>>) {
    let mut labels: Vec<String> = Vec::new();
    let mut grid: Vec<Option<usize>> = vec![None; n_frames];
    for t in turns {
        let lbl_idx = match labels.iter().position(|l| l == &t.speaker) {
            Some(i) => i,
            None => {
                labels.push(t.speaker.clone());
                labels.len() - 1
            }
        };
        let start_f = (t.start_s / frame_s).round() as usize;
        let end_f = ((t.end_s / frame_s).round() as usize).min(n_frames);
        for i in start_f..end_f {
            grid[i] = Some(lbl_idx);
        }
    }
    (labels, grid)
}

// ===========================================================================
// RTTM I/O
// ===========================================================================

/// Parse a NIST md-eval RTTM v1.0 file's SPEAKER lines.
/// Format: `SPEAKER <file_id> 1 <start_s> <dur_s> <NA> <NA> <speaker> <NA> <NA>`.
fn load_rttm(path: &Path) -> Result<Vec<SpeakerTurn>> {
    let mut s = String::new();
    File::open(path)?.read_to_string(&mut s)?;
    let mut out = Vec::new();
    for (lineno, raw) in s.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 8 || toks[0] != "SPEAKER" {
            // Skip non-SPEAKER lines (e.g. SPKR-INFO) silently.
            continue;
        }
        let start_s: f32 = toks[3]
            .parse()
            .with_context(|| format!("RTTM line {}: bad start", lineno + 1))?;
        let dur_s: f32 = toks[4]
            .parse()
            .with_context(|| format!("RTTM line {}: bad duration", lineno + 1))?;
        let speaker = toks[7].to_string();
        out.push(SpeakerTurn {
            start_s,
            end_s: start_s + dur_s,
            speaker,
        });
    }
    if out.is_empty() {
        bail!("no SPEAKER turns parsed from {}", path.display());
    }
    Ok(out)
}

fn write_rttm(path: &Path, file_id: &str, segs: &[OutSegment]) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    for s in segs {
        let start_s = s.start_ms as f32 / 1000.0;
        let dur_s = (s.end_ms - s.start_ms) as f32 / 1000.0;
        writeln!(
            w,
            "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>",
            file_id, start_s, dur_s, s.speaker_id
        )?;
    }
    w.flush()?;
    Ok(())
}

// ===========================================================================
// File hashing — included so the README's SHAs can be regenerated on demand.
// ===========================================================================

fn sha256_of(path: &Path) -> Result<String> {
    let mut f = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(hasher.finalize().as_slice()))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(hex_nib((b >> 4) & 0xf));
        s.push(hex_nib(b & 0xf));
    }
    s
}

fn hex_nib(n: u8) -> char {
    if n < 10 {
        (b'0' + n) as char
    } else {
        (b'a' + n - 10) as char
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(s: f32, e: f32, spk: &str) -> SpeakerTurn {
        SpeakerTurn {
            start_s: s,
            end_s: e,
            speaker: spk.to_string(),
        }
    }

    #[test]
    fn alpha_labels_roll_over() {
        assert_eq!(alpha_label(0), "A");
        assert_eq!(alpha_label(1), "B");
        assert_eq!(alpha_label(25), "Z");
        assert_eq!(alpha_label(26), "AA");
        assert_eq!(alpha_label(27), "AB");
    }

    #[test]
    fn der_is_zero_when_hypothesis_matches_reference() {
        // Two speakers, no overlap. Identical turn list.
        let r = vec![turn(0.0, 5.0, "A"), turn(5.0, 10.0, "B")];
        let h = r.clone();
        let d = der_score(&r, &h, 10.0);
        assert!(d.der < 0.005, "DER should be ~0 on identical input, got {}", d.der);
    }

    #[test]
    fn der_handles_relabelled_speakers() {
        // Same turn boundaries, different speaker names — DER should
        // still be ~0 because greedy mapping resolves the permutation.
        let r = vec![turn(0.0, 5.0, "A"), turn(5.0, 10.0, "B")];
        let h = vec![turn(0.0, 5.0, "X"), turn(5.0, 10.0, "Y")];
        let d = der_score(&r, &h, 10.0);
        assert!(d.der < 0.005, "DER should be ~0 under permutation, got {}", d.der);
    }

    #[test]
    fn der_detects_total_swap_confusion() {
        // Speakers fully swapped relative to reference — confusion DER.
        let r = vec![turn(0.0, 5.0, "A"), turn(5.0, 10.0, "B")];
        let h = vec![turn(0.0, 5.0, "B"), turn(5.0, 10.0, "A")];
        let d = der_score(&r, &h, 10.0);
        // Greedy permutation MUST still resolve this to ~0 — A->B, B->A.
        assert!(d.der < 0.005, "DER should be ~0 under full swap, got {}", d.der);
    }

    #[test]
    fn der_penalises_missing_turn() {
        let r = vec![turn(0.0, 5.0, "A"), turn(5.0, 10.0, "B")];
        let h = vec![turn(0.0, 5.0, "A")]; // missing second speaker
        let d = der_score(&r, &h, 10.0);
        // 5s of speech missed out of 10s -> ~50% miss.
        assert!(d.miss > 0.45 && d.miss < 0.55, "miss should be ~0.5, got {}", d.miss);
        assert!(d.der > 0.45);
    }

    #[test]
    fn der_penalises_false_alarm() {
        let r = vec![turn(0.0, 5.0, "A")];
        let h = vec![turn(0.0, 5.0, "A"), turn(5.0, 10.0, "B")]; // extra
        let d = der_score(&r, &h, 10.0);
        assert!(d.false_alarm > 0.9, "FA should be ~1.0, got {}", d.false_alarm);
        assert!(d.der > 0.9);
    }

    #[test]
    fn rttm_round_trip() {
        let tmp = std::env::temp_dir().join("spike-diarize-test.rttm");
        let segs = vec![
            OutSegment {
                start_ms: 100,
                end_ms: 2500,
                speaker_id: "A".into(),
            },
            OutSegment {
                start_ms: 2600,
                end_ms: 4900,
                speaker_id: "B".into(),
            },
        ];
        write_rttm(&tmp, "fixture", &segs).unwrap();
        let back = load_rttm(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(back.len(), 2);
        assert!((back[0].start_s - 0.1).abs() < 1e-3);
        assert!((back[0].end_s - 2.5).abs() < 1e-3);
        assert_eq!(back[1].speaker, "B");
    }

    #[test]
    fn relabel_preserves_first_seen_order() {
        let raw = vec![
            SherpaSegment { start: 0.0, end: 1.0, speaker: 7 },
            SherpaSegment { start: 1.0, end: 2.0, speaker: 3 },
            SherpaSegment { start: 2.0, end: 3.0, speaker: 7 },
            SherpaSegment { start: 3.0, end: 4.0, speaker: 3 },
        ];
        let out = relabel(&raw, "alpha").unwrap();
        assert_eq!(out[0].speaker_id, "A");
        assert_eq!(out[1].speaker_id, "B");
        assert_eq!(out[2].speaker_id, "A");
        assert_eq!(out[3].speaker_id, "B");
    }
}
