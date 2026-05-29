//! End-to-end integration test for the live transcription pipeline.
//!
//! Exercises the full path:
//!   DummyAudioSource → orchestrator → VadChunker → batched-VAD accumulator
//!   → AsrRuntime → transcript events → persistence::transcript.json
//!
//! # Gate
//!
//! The test is gated on the env vars `MEETING_APP_ASR_MODEL_PATH` and
//! `MEETING_APP_ASR_MMPROJ_PATH`. When those are absent (e.g. CI without the
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
use std::time::Duration;

use audio_capture::test_source::DummyAudioSource;
use meeting_app_common::{AppEvent, ModelFileEntry, ModelId, ModelKind, ModelManifestEntry, Segment};
use orchestrator::Orchestrator;
use model_registry::ModelRegistry;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// Env-var gate helpers
// ---------------------------------------------------------------------------

/// Returns `(model_path, mmproj_path)` from env vars, or `None` if either is
/// absent. When `None` is returned the test skips.
fn model_env_vars() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let model = std::env::var("MEETING_APP_ASR_MODEL_PATH").ok()?;
    let mmproj = std::env::var("MEETING_APP_ASR_MMPROJ_PATH").ok()?;
    Some((std::path::PathBuf::from(model), std::path::PathBuf::from(mmproj)))
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

    // Use symlinks so we don't copy ~1 GB of data.
    std::os::unix::fs::symlink(model_path, &model_link)
        .unwrap_or_else(|e| panic!("symlink model file: {e}"));
    std::os::unix::fs::symlink(mmproj_path, &mmproj_link)
        .unwrap_or_else(|e| panic!("symlink mmproj file: {e}"));
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
/// Gated on `MEETING_APP_ASR_MODEL_PATH` + `MEETING_APP_ASR_MMPROJ_PATH`.
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
async fn live_pipeline_emits_transcript_segment_and_writes_transcript_json() {
    let _ = tracing_subscriber::fmt::try_init();

    // -- Gate: skip when model env vars are absent. --
    let (model_path, mmproj_path) = match model_env_vars() {
        Some(paths) => paths,
        None => {
            eprintln!(
                "skipping transcription_e2e; ASR model env vars not set \
                 (MEETING_APP_ASR_MODEL_PATH and MEETING_APP_ASR_MMPROJ_PATH)"
            );
            return;
        }
    };

    // -- Validate paths before the expensive SHA-256 pass. --
    assert!(
        model_path.exists(),
        "MEETING_APP_ASR_MODEL_PATH does not exist: {:?}",
        model_path
    );
    assert!(
        mmproj_path.exists(),
        "MEETING_APP_ASR_MMPROJ_PATH does not exist: {:?}",
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

    // -- Generate a multi-utterance sine-with-silences pattern. --
    //
    // The batched-VAD accumulator requires ≥25 s of audio before flushing to
    // ASR. We generate enough batches to exceed that threshold:
    //   - speech_samples = 16_000 (1 s speech per batch)
    //   - silence_samples = 8_000 (0.5 s silence per batch)
    //   - batch_count = 20 → 20 × 1.5 s = 30 s of audio
    //
    // The accumulator collects VAD segments and flushes at ~25 s. With 30 s
    // provided, at least one flush occurs, triggering the ASR worker.
    //
    // The 30 s permissive wall-clock timeout below accounts for CPU-bound
    // inference latency (WSL CPU: 6-39 s per flush per Spike 3 measurements).
    let source = DummyAudioSource::new(
        16_000, // 1 s of 440 Hz sine at 16 kHz
        8_000,  // 0.5 s of silence
    );
    // sample_capacity=256 is generous (20 batches × ~2 sub-batches per batch);
    // meter_capacity=512 gives the broadcast channel headroom.
    let streams = source.generate_streams(20, 256, 512);

    let meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // -- Wait up to 30 s for at least one TranscriptSegment event. --
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut got_transcript_event = false;

    while tokio::time::Instant::now() < deadline {
        match event_rx.try_recv() {
            Ok(AppEvent::TranscriptSegment { meeting_id: mid, .. }) => {
                assert_eq!(
                    mid, meeting_id,
                    "TranscriptSegment meeting_id must match the active recording"
                );
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
        "expected at least one AppEvent::TranscriptSegment within 30 s; \
         the pipeline may have failed to flush or the ASR inference timed out"
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
