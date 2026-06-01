//! Integration tests for the Phase 4 offline `Orchestrator::re_transcribe`.
//!
//! Three scenarios, per `architecture/cross-cutting.md` — Automated-testing
//! policy:
//!
//! 1. **save → drop → open round-trip** (DEFAULT suite, no model). A synthetic
//!    meeting folder (metadata + transcript) is written, the orchestrator state
//!    is dropped, and `persistence::read_meeting_state` (the `open_meeting`
//!    backing) returns a `MeetingState` matching what was written.
//! 2. **`re_transcribe` over the LibriSpeech fixture** (GATED on
//!    `MEETING_APP_ASR_MODEL_PATH` + `MEETING_APP_ASR_MMPROJ_PATH`). Records a
//!    real-speech meeting through the live pipeline so `audio.opus` exists, then
//!    re-runs transcription offline and asserts `transcript.json` is rewritten
//!    and at least one `AppEvent::TranscriptSegment` is emitted. Synthetic tones
//!    fail the real Silero VAD, so real speech is required — hence the gate.
//! 3. **`re_transcribe` refuses during a live recording** (DEFAULT suite, no
//!    model). With the orchestrator not `Idle`, `re_transcribe` returns
//!    `AppError::InvalidInput` without touching the meeting.
//!
//! The model-staging helpers (manifest with real SHA-256, hard-link/symlink the
//! model files into the registry cache) mirror `transcription_e2e.rs`.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use audio_capture::{AudioFrameBatch, AudioStreams};
use meeting_app_common::{
    AppError, AppEvent, AudioFormat, MeetingId, MeetingMeta, ModelFileEntry, ModelId, ModelKind,
    ModelManifestEntry, Segment,
};
use model_registry::ModelRegistry;
use orchestrator::test_support::test_orchestrator;
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::{broadcast, mpsc};

// ---------------------------------------------------------------------------
// Env-var gate
// ---------------------------------------------------------------------------

fn model_env_vars() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let model = std::env::var("MEETING_APP_ASR_MODEL_PATH").ok()?;
    let mmproj = std::env::var("MEETING_APP_ASR_MMPROJ_PATH").ok()?;
    Some((
        std::path::PathBuf::from(model),
        std::path::PathBuf::from(mmproj),
    ))
}

// ---------------------------------------------------------------------------
// Synthetic meeting folder (no audio) for the save → open round-trip
// ---------------------------------------------------------------------------

/// Write a synthetic meeting folder (`metadata.json` + `transcript.json`) under
/// `root`, returning its id. Mirrors the on-disk layout `persistence` produces.
fn write_synthetic_meeting(root: &Path, title: &str, first_text: &str) -> MeetingId {
    let meeting_id = MeetingId::new();
    let folder = root.join(meeting_id.0.to_string());
    std::fs::create_dir_all(&folder).expect("create meeting folder");

    let meta = MeetingMeta {
        uuid: meeting_id,
        title: title.to_string(),
        started_at: "2026-06-02T10:00:00Z".to_string(),
        ended_at: Some("2026-06-02T10:30:00Z".to_string()),
        duration_ms: 1_800_000,
        speaker_count: 1,
        audio_format: AudioFormat {
            codec: "opus".into(),
            sample_rate: 16_000,
            channels: 1,
            bitrate_kbps: Some(32),
        },
        asr_model: None,
        llm_model: None,
        diarizer: None,
        app_version: "0.0.0".into(),
    };
    std::fs::write(
        folder.join("metadata.json"),
        serde_json::to_vec_pretty(&meta).unwrap(),
    )
    .expect("write metadata.json");

    let segments = vec![Segment {
        start_ms: 0,
        end_ms: 1_000,
        text: first_text.to_string(),
        speaker_id: None,
        confidence: None,
        words: Vec::new(),
    }];
    std::fs::write(
        folder.join("transcript.json"),
        serde_json::to_vec_pretty(&segments).unwrap(),
    )
    .expect("write transcript.json");

    meeting_id
}

// ---------------------------------------------------------------------------
// 1. save → drop → open round-trip (DEFAULT suite)
// ---------------------------------------------------------------------------

/// A meeting written to disk, after the writing orchestrator state is dropped,
/// is reopened to a `MeetingState` that matches what was written. This is the
/// `open_meeting` restore guarantee at the persistence layer (no model needed).
#[tokio::test]
async fn saved_meeting_reopens_to_matching_state() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    let meeting_id = {
        // Build an orchestrator over this root, then drop it — the meeting state
        // must survive the orchestrator going away (it lives on disk).
        let _orch = test_orchestrator(root.clone());
        write_synthetic_meeting(&root, "Launch sync", "hello world")
    }; // _orch dropped here

    // Reopen via the `open_meeting` backing (`read_meeting_state`).
    let meeting_dir = root.join(meeting_id.0.to_string());
    let state = persistence::read_meeting_state(&meeting_dir).expect("read_meeting_state");

    assert_eq!(state.meta.uuid, meeting_id);
    assert_eq!(state.meta.title, "Launch sync");
    assert_eq!(state.transcript.len(), 1);
    assert_eq!(state.transcript[0].text, "hello world");
    assert!(state.notes.is_none(), "no notes saved → None");
}

// ---------------------------------------------------------------------------
// 3. re_transcribe refuses during a live recording (DEFAULT suite)
// ---------------------------------------------------------------------------

/// `re_transcribe` must refuse with `AppError::InvalidInput` when the recorder
/// is not `Idle` (a live recording holds the ASR model). Uses the no-model test
/// orchestrator + an in-memory index; no audio device required.
#[tokio::test]
async fn re_transcribe_refused_while_recording() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let orch = test_orchestrator(root.clone());

    // A synthetic meeting to point re_transcribe at (it should never be touched).
    let meeting_id = write_synthetic_meeting(&root, "Busy", "original");

    let index = MeetingIndex::open(":memory:")
        .await
        .expect("open in-memory index");

    // Drive the orchestrator into a non-Idle state with a live (empty) stream.
    let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(8);
    let (_meter_tx, meter_rx) = mpsc::channel(8);
    let streams = AudioStreams {
        samples: sample_rx,
        meter: meter_rx,
    };
    let _recording_id = orch
        .start_with_streams(streams)
        .await
        .expect("start recording");

    // re_transcribe must refuse while not Idle.
    let err = orch
        .re_transcribe(&index, meeting_id)
        .await
        .expect_err("re_transcribe must be refused while recording");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );

    // The meeting's transcript must be untouched.
    let meeting_dir = root.join(meeting_id.0.to_string());
    let transcript = persistence::read_transcript(&meeting_dir).expect("read transcript");
    assert_eq!(transcript.len(), 1);
    assert_eq!(transcript[0].text, "original");

    // Tidy up: dropping the sender ends the stream; stop the recording.
    drop(sample_tx);
    let _ = orch.stop().await;
}

// ---------------------------------------------------------------------------
// Model-staging helpers (mirror transcription_e2e.rs)
// ---------------------------------------------------------------------------

fn sha256_of_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read model file {path:?}: {e}"));
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

fn build_manifest_entry(model_path: &Path, mmproj_path: &Path) -> ModelManifestEntry {
    let model_filename = "Qwen3-ASR-0.6B-Q8_0.gguf";
    let mmproj_filename = "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf";

    let model_size = std::fs::metadata(model_path).expect("stat model").len();
    let mmproj_size = std::fs::metadata(mmproj_path).expect("stat mmproj").len();

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
                sha256: sha256_of_file(model_path),
            },
            ModelFileEntry {
                filename: mmproj_filename.to_string(),
                url: "file:///unused-in-test".to_string(),
                size: mmproj_size,
                sha256: sha256_of_file(mmproj_path),
            },
        ],
    }
}

fn symlink_model_files(cache_dir: &Path, model_path: &Path, mmproj_path: &Path) {
    let model_cache = cache_dir.join("asr").join("qwen3-asr-0.6b-q8_0");
    std::fs::create_dir_all(&model_cache).expect("create model cache dir");
    link_or_copy(model_path, &model_cache.join("Qwen3-ASR-0.6B-Q8_0.gguf"));
    link_or_copy(
        mmproj_path,
        &model_cache.join("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"),
    );
}

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
    std::fs::copy(src, dst).unwrap_or_else(|e| panic!("link/copy {src:?} -> {dst:?}: {e}"));
}

fn load_fixture_wav() -> Vec<f32> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/librispeech_0.wav");
    let mut reader = hound::WavReader::open(&fixture)
        .unwrap_or_else(|e| panic!("cannot open {fixture:?}: {e}"));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "fixture must be mono");
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
    reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<_, _>>()
        .expect("reading samples")
}

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

fn e2e_orchestrator(
    persistence_root: std::path::PathBuf,
    model_cache: std::path::PathBuf,
    manifest_entry: ModelManifestEntry,
) -> (Orchestrator, broadcast::Receiver<AppEvent>) {
    let settings_path = persistence_root.join(".e2e_settings.json");
    let store = JsonFileStore::new(settings_path);
    let settings = SettingsHandle::new(store).expect("SettingsHandle construction");

    let (event_tx, event_rx) = broadcast::channel::<AppEvent>(512);
    let registry = ModelRegistry::new(model_cache, vec![manifest_entry], event_tx.clone())
        .expect("ModelRegistry construction");

    let orch =
        Orchestrator::with_event_tx(settings, persistence_root, Arc::new(registry), event_tx);
    (orch, event_rx)
}

// ---------------------------------------------------------------------------
// 2. re_transcribe over the LibriSpeech fixture (GATED)
// ---------------------------------------------------------------------------

/// Record a real-speech meeting, then re-transcribe it offline. Verifies that
/// `re_transcribe` rewrites `transcript.json` and emits at least one
/// `AppEvent::TranscriptSegment`. Gated on the ASR model env vars; the no-op
/// skip path is what CI runs.
#[tokio::test(flavor = "multi_thread")]
async fn re_transcribe_rewrites_transcript_over_fixture() {
    let _ = tracing_subscriber::fmt::try_init();

    let (model_path, mmproj_path) = match model_env_vars() {
        Some(paths) => paths,
        None => {
            eprintln!(
                "skipping re_transcribe_rewrites_transcript_over_fixture; ASR model env vars not set \
                 (MEETING_APP_ASR_MODEL_PATH and MEETING_APP_ASR_MMPROJ_PATH)"
            );
            return;
        }
    };

    assert!(model_path.exists(), "model path missing: {model_path:?}");
    assert!(mmproj_path.exists(), "mmproj path missing: {mmproj_path:?}");

    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");
    let model_cache = dir.path().join("models");

    eprintln!("computing SHA-256 of model files (this takes a few seconds)…");
    let manifest_entry = build_manifest_entry(&model_path, &mmproj_path);
    symlink_model_files(&model_cache, &model_path, &mmproj_path);

    let (orch, mut event_rx) =
        e2e_orchestrator(persistence_root.clone(), model_cache, manifest_entry);

    let asr_model_id = ModelId::from("qwen3-asr-0.6b-q8_0");
    orch.ensure_model(&asr_model_id)
        .await
        .expect("ensure_model must succeed with pre-staged files");

    // -- Record a real-speech meeting so audio.opus exists. --
    let clip = load_fixture_wav();
    let mut samples = Vec::with_capacity(clip.len() * 6);
    for _ in 0..6 {
        samples.extend_from_slice(&clip);
    }
    let (streams, _feeder) = samples_to_streams(samples, 1600);
    let _meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // Wait for the live pipeline to emit at least one transcript segment, then
    // stop and finalise so audio.opus + transcript.json are on disk.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    let mut got_live_segment = false;
    while tokio::time::Instant::now() < deadline {
        match event_rx.try_recv() {
            Ok(AppEvent::TranscriptSegment { .. }) => {
                got_live_segment = true;
                break;
            }
            Ok(_) => {}
            Err(broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {}
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }
    assert!(got_live_segment, "live pipeline must emit a transcript segment");

    let meta = orch.stop().await.expect("stop recording");
    let meeting_id = meta.uuid;
    let meeting_dir = persistence_root.join(meeting_id.0.to_string());
    assert!(
        meeting_dir.join("audio.opus").exists(),
        "audio.opus must exist for re_transcribe to decode"
    );

    // Capture the live transcript, then deliberately corrupt transcript.json so
    // we can prove re_transcribe REWRITES it.
    let live_transcript = persistence::read_transcript(&meeting_dir).expect("read live transcript");
    assert!(!live_transcript.is_empty(), "live transcript must be non-empty");
    std::fs::write(
        meeting_dir.join("transcript.json"),
        serde_json::to_vec_pretty(&Vec::<Segment>::new()).unwrap(),
    )
    .expect("overwrite transcript.json with []");

    // Drain any backlog so we can observe re_transcribe's fresh events.
    while event_rx.try_recv().is_ok() {}

    // -- Re-transcribe offline. --
    let index = MeetingIndex::open(":memory:").await.expect("open index");
    index
        .rebuild_from_disk(&persistence_root)
        .await
        .expect("seed index");

    orch.re_transcribe(&index, meeting_id)
        .await
        .expect("re_transcribe must succeed with the staged model");

    // transcript.json must have been rewritten with real segments.
    let rewritten = persistence::read_transcript(&meeting_dir).expect("read rewritten transcript");
    assert!(
        !rewritten.is_empty(),
        "re_transcribe must rewrite transcript.json with at least one segment"
    );
    assert!(
        rewritten.iter().any(|s| !s.text.trim().is_empty()),
        "rewritten transcript must carry real text"
    );

    // At least one TranscriptSegment event must have been emitted by re_transcribe.
    let mut got_retranscribe_event = false;
    let drain_deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < drain_deadline {
        match event_rx.try_recv() {
            Ok(AppEvent::TranscriptSegment { meeting_id: mid, .. }) => {
                assert_eq!(mid, meeting_id, "event meeting_id must match");
                got_retranscribe_event = true;
                break;
            }
            Ok(_) => {}
            Err(broadcast::error::TryRecvError::Empty) => break,
            Err(_) => break,
        }
    }
    assert!(
        got_retranscribe_event,
        "re_transcribe must emit at least one AppEvent::TranscriptSegment"
    );

    // The index excerpt must reflect the refreshed transcript.
    let rows = index.list_meetings().await.expect("list");
    let row = rows.iter().find(|r| r.id == meeting_id).expect("indexed");
    assert!(row.excerpt.is_some(), "index excerpt must be refreshed");
}
