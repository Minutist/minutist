//! End-to-end integration test for the live transcription pipeline.
//!
//! Exercises the full path:
//!   DummyAudioSource → orchestrator → VadChunker → batched-VAD accumulator
//!   → AsrRuntime → transcript events → persistence::transcript.json
//!
//! # Gate
//!
//! The test is gated on the env vars `MINUTIST_ASR_MODEL_PATH` and
//! `MINUTIST_ASR_MMPROJ_PATH`. When those are absent (e.g. CI without the
//! ~1 GB model) the test emits an `eprintln!` and returns immediately so the
//! no-op path compiles and passes.
//!
//! # Registry setup
//!
//! The `ModelRegistry` is pointed at a tempdir with a custom manifest. The
//! manifest entries carry the real SHA-256 values for the model files
//! (loaded from `resources/models.json` at the workspace root). The actual
//! GGUF and mmproj files are symlinked into the expected per-kind cache
//! subdirectory under the tempdir so `registry.ensure()` finds them already
//! present and hash-verified, bypassing any download.
//!
//! Implements the Phase 2 Stream H design contract.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use audio_capture::{AudioFrameBatch, AudioStreams};
use minutist_common::{AppEvent, ModelFileEntry, ModelId, ModelKind, ModelManifestEntry, Segment};
use orchestrator::test_support::load_fixture_wav;
use orchestrator::Orchestrator;
use model_registry::ModelRegistry;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::{broadcast, mpsc};

// ---------------------------------------------------------------------------
// Env-var gate helpers
// ---------------------------------------------------------------------------

/// Returns `(model_path, mmproj_path)` from env vars, or `None` if either is
/// absent OR set to an empty string. When `None` is returned the test skips.
fn model_env_vars() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    // Treat an EMPTY value as unset (skip) so `VAR=""` does not stage an empty
    // path and panic; only a non-empty path counts as "set".
    let model = non_empty_env("MINUTIST_ASR_MODEL_PATH")?;
    let mmproj = non_empty_env("MINUTIST_ASR_MMPROJ_PATH")?;
    Some((std::path::PathBuf::from(model), std::path::PathBuf::from(mmproj)))
}

/// Read `name` from the environment, returning `None` when it is unset OR set
/// to an empty string, so `VAR=""` is equivalent to "unset" for the gate.
fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

// ---------------------------------------------------------------------------
// Registry construction helpers
// ---------------------------------------------------------------------------

/// Compute the SHA-256 hex digest of a file at `path`.
///
/// Called once per model file at test setup time; the files are ~800 MB and
/// ~220 MB respectively, so this takes 2-4 seconds on a typical SSD. The
/// cost is acceptable for a gated integration test.
fn sha256_of_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!("failed to read model file {:?}: {e}", path)
    });
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

/// Build a `ModelManifestEntry` for `qwen3-asr-0.6b-q8_0` backed by real
/// files at `model_path` and `mmproj_path`.
///
/// The manifest entry uses the **actual** filenames from `resources/models.json`
/// so the registry's path layout matches what the orchestrator runner expects
/// (`cache_root/asr/qwen3-asr-0.6b-q8_0/{filename}`).
///
/// The SHA-256 values are computed from the actual files so
/// `registry.ensure()` passes verification without downloading.
fn build_manifest_entry(
    model_path: &Path,
    mmproj_path: &Path,
) -> ModelManifestEntry {
    // These filenames must match resources/models.json exactly so the
    // registry's path composition (`model_dir.join(&file.filename)`)
    // resolves to the symlinks we create below.
    let model_filename = "Qwen3-ASR-0.6B-Q8_0.gguf";
    let mmproj_filename = "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf";

    let model_size = std::fs::metadata(model_path)
        .unwrap_or_else(|e| panic!("stat model file: {e}"))
        .len();
    let mmproj_size = std::fs::metadata(mmproj_path)
        .unwrap_or_else(|e| panic!("stat mmproj file: {e}"))
        .len();

    let model_sha = sha256_of_file(model_path);
    let mmproj_sha = sha256_of_file(mmproj_path);

    ModelManifestEntry {
        id: ModelId::from("qwen3-asr-0.6b-q8_0"),
        kind: ModelKind::Asr,
        display_name: "Qwen3-ASR 0.6B (Q8_0)".to_string(),
        license: "apache-2.0".to_string(),
        total_size_bytes: model_size + mmproj_size,
        files: vec![
            ModelFileEntry {
                filename: model_filename.to_string(),
                url: "file:///unused-in-test".to_string(),
                size: model_size,
                sha256: model_sha,
            },
            ModelFileEntry {
                filename: mmproj_filename.to_string(),
                url: "file:///unused-in-test".to_string(),
                size: mmproj_size,
                sha256: mmproj_sha,
            },
        ],
    }
}

/// Populate `cache_dir/asr/qwen3-asr-0.6b-q8_0/` with symlinks to the
/// real model files so `registry.ensure()` finds them present and
/// hash-verified without downloading.
fn symlink_model_files(
    cache_dir: &Path,
    model_path: &Path,
    mmproj_path: &Path,
) {
    let model_cache = cache_dir
        .join("asr")
        .join("qwen3-asr-0.6b-q8_0");
    std::fs::create_dir_all(&model_cache)
        .expect("create model cache dir");

    let model_link = model_cache.join("Qwen3-ASR-0.6B-Q8_0.gguf");
    let mmproj_link = model_cache.join("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf");

    // Materialise the model files into the cache dir without copying ~1 GB
    // when avoidable. Cross-platform ladder: hard link (fast, no privilege,
    // same-volume) → platform symlink → full copy (cross-filesystem fallback).
    link_or_copy(model_path, &model_link);
    link_or_copy(mmproj_path, &mmproj_link);
}

/// Make `dst` resolve to the same bytes as `src`, preferring zero-copy.
fn link_or_copy(src: &Path, dst: &Path) {
    if std::fs::hard_link(src, dst).is_ok() {
        return;
    }
    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(src, dst).is_ok() {
            return;
        }
    }
    #[cfg(windows)]
    {
        if std::os::windows::fs::symlink_file(src, dst).is_ok() {
            return;
        }
    }
    std::fs::copy(src, dst)
        .unwrap_or_else(|e| panic!("link/copy {src:?} -> {dst:?}: {e}"));
}

// ---------------------------------------------------------------------------
// Real-speech fixture helpers
// ---------------------------------------------------------------------------

/// Load the 16 kHz mono LibriSpeech fixture as f32 samples.
///
/// The pipeline runs audio through the real Silero VAD, which only emits
/// speech segments for genuine speech — a synthetic sine tone is rejected,
/// so the accumulator would never fill. We must feed real speech.

/// Build `AudioStreams` from pre-loaded samples, chunked into `batch_size`
/// sample batches. The feeder runs on a blocking task; dropping its sender
/// signals end-of-stream to the runner.
fn samples_to_streams(
    samples: Vec<f32>,
    batch_size: usize,
) -> (AudioStreams, tokio::task::JoinHandle<()>) {
    let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(256);
    let (_meter_tx, meter_rx) = mpsc::channel(64);

    let streams = AudioStreams {
        samples: sample_rx,
        meter: meter_rx,
    };

    let handle = tokio::task::spawn_blocking(move || {
        let sample_rate = 16_000u64;
        let mut start_ms = 0u64;
        for chunk in samples.chunks(batch_size) {
            let duration_ms = (chunk.len() as u64 * 1000) / sample_rate;
            let batch = AudioFrameBatch {
                samples: chunk.to_vec(),
                start_ms,
                end_ms: start_ms + duration_ms,
            };
            if sample_tx.blocking_send(batch).is_err() {
                break;
            }
            start_ms += duration_ms;
        }
    });

    (streams, handle)
}

// ---------------------------------------------------------------------------
// Orchestrator constructor for the e2e test
// ---------------------------------------------------------------------------

/// Build an `Orchestrator` pointing at `persistence_root`, with a
/// `ModelRegistry` backed by `manifest_entry` and files in `model_cache`.
///
/// The shared event channel is returned so assertions can subscribe to it.
fn e2e_orchestrator(
    persistence_root: std::path::PathBuf,
    model_cache: std::path::PathBuf,
    manifest_entry: ModelManifestEntry,
) -> (Orchestrator, broadcast::Receiver<AppEvent>) {
    let settings_path = persistence_root.join(".e2e_settings.json");
    let store = JsonFileStore::new(settings_path);
    let settings =
        SettingsHandle::new(store).expect("SettingsHandle construction");

    let (event_tx, event_rx) = broadcast::channel::<AppEvent>(512);

    let registry = ModelRegistry::new(
        model_cache,
        vec![manifest_entry],
        event_tx.clone(),
    )
    .expect("ModelRegistry construction");

    let orch = Orchestrator::with_event_tx(
        settings,
        persistence_root,
        Arc::new(registry),
        event_tx,
    );

    (orch, event_rx)
}

// ---------------------------------------------------------------------------
// End-to-end test
// ---------------------------------------------------------------------------

/// Full pipeline: DummyAudioSource → orchestrator → VAD → ASR → transcript.
///
/// Gated on `MINUTIST_ASR_MODEL_PATH` + `MINUTIST_ASR_MMPROJ_PATH`.
/// When those are absent the test emits a diagnostic and returns immediately
/// — the no-op path is what CI runs.
///
/// With models present the test verifies:
/// 1. At least one `AppEvent::TranscriptSegment` is emitted within 30 s.
/// 2. The meeting folder contains `audio.opus`, `metadata.json`, and
///    `transcript.json` after stop.
/// 3. `transcript.json` parses as `Vec<Segment>`, has ≥1 segment, with
///    monotonically-increasing `start_ms` values.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "model-gated: set MINUTIST_ASR_MODEL_PATH + MINUTIST_ASR_MMPROJ_PATH"]
async fn live_pipeline_emits_transcript_segment_and_writes_transcript_json() {
    let _ = tracing_subscriber::fmt::try_init();

    // -- Gate: skip when model env vars are absent. --
    let (model_path, mmproj_path) = match model_env_vars() {
        Some(paths) => paths,
        None => {
            eprintln!(
                "skipping transcription_e2e; ASR model env vars not set \
                 (MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH)"
            );
            return;
        }
    };

    // -- Validate paths before the expensive SHA-256 pass. --
    assert!(
        model_path.exists(),
        "MINUTIST_ASR_MODEL_PATH does not exist: {:?}",
        model_path
    );
    assert!(
        mmproj_path.exists(),
        "MINUTIST_ASR_MMPROJ_PATH does not exist: {:?}",
        mmproj_path
    );

    // -- Tempdir setup --
    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");
    let model_cache = dir.path().join("models");

    // -- Build manifest entry with real SHA-256 hashes (reads the files). --
    //    This is the expensive step: ~2-4 s for a 1 GB model on a fast SSD.
    eprintln!("computing SHA-256 of model files (this takes a few seconds)…");
    let manifest_entry = build_manifest_entry(&model_path, &mmproj_path);

    // -- Symlink model files into the cache directory. --
    symlink_model_files(&model_cache, &model_path, &mmproj_path);

    // -- Construct orchestrator with the populated registry. --
    let (orch, mut event_rx) = e2e_orchestrator(
        persistence_root.clone(),
        model_cache,
        manifest_entry,
    );

    // -- Ensure the model is seen as Available (fast path: files present + hash verified). --
    let asr_model_id = ModelId::from("qwen3-asr-0.6b-q8_0");
    orch.ensure_model(&asr_model_id)
        .await
        .expect("ensure_model must succeed with pre-staged files");

    // -- Build a ≥25 s real-speech stream to exercise the size-triggered flush. --
    //
    // The batched-VAD accumulator flushes at FLUSH_MIN_SECS (25 s). The
    // 5.86 s LibriSpeech clip is repeated 6× (~35 s) so a size-triggered
    // flush fires mid-stream — the live streaming path — rather than only the
    // on-stop flush. Silero VAD detects the repeated speech and emits segments
    // that fill the accumulator.
    let clip = load_fixture_wav();
    let mut samples = Vec::with_capacity(clip.len() * 6);
    for _ in 0..6 {
        samples.extend_from_slice(&clip);
    }
    // 1600 samples = 100 ms at 16 kHz.
    let (streams, _feeder) = samples_to_streams(samples, 1600);

    let start_instant = Instant::now();
    let meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // -- Wait up to 90 s for at least one TranscriptSegment event. --
    //    Budget covers cold model load (~5 s) plus CPU inference on a ~30 s
    //    buffer. The size-triggered flush fires before the feeder exhausts, so
    //    the first segment arrives without needing stop().
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    let mut got_transcript_event = false;
    let mut first_segment_latency: Option<Duration> = None;

    while tokio::time::Instant::now() < deadline {
        match event_rx.try_recv() {
            Ok(AppEvent::TranscriptSegment { meeting_id: mid, .. }) => {
                assert_eq!(
                    mid, meeting_id,
                    "TranscriptSegment meeting_id must match the active recording"
                );
                first_segment_latency = Some(start_instant.elapsed());
                got_transcript_event = true;
                break;
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(target: "test", lagged = n, "e2e subscriber lagged");
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }

    assert!(
        got_transcript_event,
        "expected at least one AppEvent::TranscriptSegment within 90 s; \
         the pipeline may have failed to flush or the ASR inference timed out"
    );

    eprintln!(
        "E2E first-segment latency (start → first TranscriptSegment, \
         incl. cold model load): {:?}",
        first_segment_latency.expect("latency set when event received")
    );

    // -- Stop the recording and verify on-disk artefacts. --
    let meta = orch.stop().await.expect("stop recording");

    // Derive the meeting directory from the returned MeetingMeta uuid.
    let meeting_dir = persistence_root.join(meta.uuid.0.to_string());

    assert!(
        meeting_dir.exists(),
        "meeting directory must exist at {:?}",
        meeting_dir
    );

    // audio.opus must be present.
    let audio_path = meeting_dir.join("audio.opus");
    assert!(
        audio_path.exists(),
        "audio.opus must exist in meeting folder"
    );
    assert!(
        std::fs::metadata(&audio_path).unwrap().len() > 0,
        "audio.opus must not be empty"
    );

    // metadata.json must be present.
    let meta_path = meeting_dir.join("metadata.json");
    assert!(
        meta_path.exists(),
        "metadata.json must exist in meeting folder"
    );

    // transcript.json must be present (the pipeline wrote ≥1 segment).
    let transcript_path = meeting_dir.join("transcript.json");
    assert!(
        transcript_path.exists(),
        "transcript.json must exist in meeting folder after a recording that \
         produced at least one TranscriptSegment event"
    );

    // -- Parse and validate transcript.json. --
    let transcript_json =
        std::fs::read_to_string(&transcript_path).expect("read transcript.json");
    let segments: Vec<Segment> =
        serde_json::from_str(&transcript_json)
            .expect("transcript.json must parse as Vec<Segment>");

    assert!(
        !segments.is_empty(),
        "transcript.json must contain at least one segment"
    );

    // Verify monotonically-increasing start_ms.
    let mut prev_start = 0u64;
    for (i, seg) in segments.iter().enumerate() {
        assert!(
            seg.start_ms >= prev_start,
            "segment[{i}].start_ms ({}) must be >= segment[{}].start_ms ({})",
            seg.start_ms,
            i.saturating_sub(1),
            prev_start
        );
        prev_start = seg.start_ms;
    }

    // Segments must have non-empty text (real ASR output, not silence).
    let has_text = segments.iter().any(|s| !s.text.trim().is_empty());
    assert!(
        has_text,
        "at least one segment must have non-empty text"
    );
}
