//! Integration tests for the Phase 4 offline `Orchestrator::re_transcribe`.
//!
//! Four scenarios, per `architecture/cross-cutting.md` — Automated-testing
//! policy:
//!
//! 1. **save → drop → open round-trip** (DEFAULT suite, no model). A synthetic
//!    meeting folder (metadata + transcript) is written, the orchestrator state
//!    is dropped, and `persistence::read_meeting_state` (the `open_meeting`
//!    backing) returns a `MeetingState` matching what was written.
//! 2. **`re_transcribe` over the LibriSpeech fixture** (GATED on
//!    `MINUTIST_ASR_MODEL_PATH` + `MINUTIST_ASR_MMPROJ_PATH`). Records a
//!    real-speech meeting through the live pipeline so `audio.opus` exists, then
//!    re-runs transcription offline and asserts `transcript.json` is rewritten
//!    and at least one `AppEvent::TranscriptSegment` is emitted. Synthetic tones
//!    fail the real Silero VAD, so real speech is required — hence the gate.
//! 3. **`re_transcribe` refuses during a live recording** (DEFAULT suite, no
//!    model). With the orchestrator not `Idle`, `re_transcribe` returns
//!    `AppError::InvalidInput` without touching the meeting.
//! 4. **`re_transcribe_with_backend` over the LibriSpeech fixture, stub ASR**
//!    (DEFAULT suite, no model). The offline counterpart of scenario 2 with the
//!    ASR model replaced by a `StubAsrBackend` (the same seam
//!    `pipeline_stub_test.rs` uses for the live path). A synthetic meeting
//!    folder whose `audio.opus` is the committed LibriSpeech fixture encoded via
//!    the persistence Opus encoder is built, its `transcript.json` is emptied,
//!    and the offline re-transcribe drives the **real** Silero VAD over the real
//!    speech and asserts `transcript.json` is rewritten with the stub text, an
//!    `AppEvent::TranscriptSegment` is emitted, and the index excerpt is
//!    refreshed. This is the model-free coverage of the offline
//!    `re_transcribe_buffer` → `transcribe_one_flush` → `write_transcript` →
//!    index-upsert path that scenario 2 gates on a real model.
//!
//! The model-staging helpers (manifest with real SHA-256, hard-link/symlink the
//! model files into the registry cache) mirror `transcription_e2e.rs`.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use audio_capture::{AudioFrameBatch, AudioStreams};
use minutist_common::{
    AppError, AppEvent, AudioFormat, MeetingId, MeetingMeta, ModelFileEntry, ModelId, ModelKind,
    ModelManifestEntry, Segment,
};
use model_registry::ModelRegistry;
use orchestrator::test_support::{build_meeting, load_fixture_wav, test_orchestrator, StubAsrBackend};
use orchestrator::Orchestrator;
use persistence::MeetingIndex;
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::{broadcast, mpsc};

// ---------------------------------------------------------------------------
// Env-var gate
// ---------------------------------------------------------------------------

fn model_env_vars() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    // Treat an EMPTY value as unset (skip) so `VAR=""` does not stage an empty
    // path and panic; only a non-empty path counts as "set".
    let model = non_empty_env("MINUTIST_ASR_MODEL_PATH")?;
    let mmproj = non_empty_env("MINUTIST_ASR_MMPROJ_PATH")?;
    Some((
        std::path::PathBuf::from(model),
        std::path::PathBuf::from(mmproj),
    ))
}

/// Read `name` from the environment, returning `None` when it is unset OR set
/// to an empty string, so `VAR=""` is equivalent to "unset" for the gate.
fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
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
        speaker_names: std::collections::BTreeMap::new(),
        notes_format: 0,
        processing: Default::default(),
        collection_id: None,
        recording_started: true,
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
        shared_speakers: Vec::new(),
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
#[ignore = "model-gated: set MINUTIST_ASR_MODEL_PATH + MINUTIST_ASR_MMPROJ_PATH"]
async fn re_transcribe_rewrites_transcript_over_fixture() {
    let _ = tracing_subscriber::fmt::try_init();

    let (model_path, mmproj_path) = match model_env_vars() {
        Some(paths) => paths,
        None => {
            eprintln!(
                "skipping re_transcribe_rewrites_transcript_over_fixture; ASR model env vars not set \
                 (MINUTIST_ASR_MODEL_PATH and MINUTIST_ASR_MMPROJ_PATH)"
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

// ---------------------------------------------------------------------------
// 4. re_transcribe_with_backend over the LibriSpeech fixture, stub ASR
//    (DEFAULT suite — no model env vars)
// ---------------------------------------------------------------------------

/// Offline re-transcribe with a stub ASR backend over a real-speech fixture.
///
/// This test ALWAYS runs — it is NOT gated on any env var. It exercises the
/// full offline path (decode → real Silero VAD → batched-VAD accumulator →
/// `transcribe_one_flush` → `write_transcript` → index upsert) with a stub
/// backend, so the model-free CI suite covers everything scenario 2 gates on a
/// real model.
///
/// Verifies:
/// 1. `transcript.json` is rewritten with the stub text (was emptied first).
/// 2. At least one `AppEvent::TranscriptSegment` is emitted for the meeting.
/// 3. The index excerpt is refreshed to the new first segment.
#[tokio::test(flavor = "multi_thread")]
async fn re_transcribe_with_stub_backend_rewrites_transcript_over_fixture() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    // A no-model test orchestrator is sufficient: the stub backend is injected,
    // so the registry is never consulted for an ASR model.
    let orch = test_orchestrator(root.clone());
    let mut event_rx = orch.subscribe_events();

    // Real speech, repeated so the VAD reliably yields ≥1 segment (the same
    // approach the gated fixture test uses).
    let clip = load_fixture_wav();
    let mut samples = Vec::with_capacity(clip.len() * 4);
    for _ in 0..4 {
        samples.extend_from_slice(&clip);
    }

    let meeting_id = build_meeting(&root, "Stub re-transcribe", &samples, &[], &[]);

    // transcript.json starts empty.
    let meeting_dir = root.join(meeting_id.0.to_string());
    let before = persistence::read_transcript(&meeting_dir).expect("read transcript before");
    assert!(before.is_empty(), "transcript.json must start empty");

    // Seed an in-memory index from disk (excerpt is None while transcript empty).
    let index = MeetingIndex::open(":memory:").await.expect("open index");
    index
        .rebuild_from_disk(&root)
        .await
        .expect("seed index from disk");
    let seeded = index.list_meetings().await.expect("list seeded");
    let seeded_row = seeded.iter().find(|r| r.id == meeting_id).expect("seeded row");
    assert!(
        seeded_row.excerpt.is_none(),
        "index excerpt must be None before re-transcribe (transcript empty)"
    );

    // Run the offline re-transcribe with the stub backend.
    let stub = Box::new(StubAsrBackend::new("stub transcript text"));
    orch.re_transcribe_with_backend(&index, meeting_id, stub)
        .await
        .expect("re_transcribe_with_backend must succeed");

    // 1. transcript.json rewritten with the stub text.
    let rewritten = persistence::read_transcript(&meeting_dir).expect("read transcript after");
    assert!(
        !rewritten.is_empty(),
        "re_transcribe must rewrite transcript.json with at least one segment"
    );
    assert!(
        rewritten.iter().any(|s| s.text.contains("stub")),
        "rewritten transcript must carry the stub text; got: {rewritten:?}"
    );

    // 2. At least one TranscriptSegment event for this meeting was emitted.
    let mut got_event = false;
    loop {
        match event_rx.try_recv() {
            Ok(AppEvent::TranscriptSegment { meeting_id: mid, segment }) => {
                assert_eq!(mid, meeting_id, "event meeting_id must match");
                if segment.text.contains("stub") {
                    got_event = true;
                }
            }
            Ok(_) => {}
            Err(broadcast::error::TryRecvError::Empty) => break,
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }
    assert!(
        got_event,
        "re_transcribe_with_backend must emit at least one AppEvent::TranscriptSegment \
         (check the Silero VAD model at resources/silero/silero_vad_v4.onnx)"
    );

    // 3. Index excerpt refreshed to the new first segment.
    let rows = index.list_meetings().await.expect("list after");
    let row = rows.iter().find(|r| r.id == meeting_id).expect("indexed");
    assert!(
        row.excerpt.as_deref().is_some_and(|e| e.contains("stub")),
        "index excerpt must be refreshed to the rewritten transcript; got: {:?}",
        row.excerpt
    );
}

// ---------------------------------------------------------------------------
// 5. transcribe_pcm_window_with_backend over the fixture, stub ASR
//    (DEFAULT suite — no model env vars; Phase 9 `relisten_section` backing)
// ---------------------------------------------------------------------------

/// `Orchestrator::transcribe_pcm_window` (driven via the `*_with_backend`
/// test seam) re-runs ASR over a bounded window and returns the segments WITHOUT
/// rewriting `transcript.json` — proving it is a read-only compute op. It also
/// does NOT take the offline claim (it runs without any `Idle` gate handshake).
///
/// This ALWAYS runs (no env gate): the stub backend stands in for the ASR model,
/// and the real Silero VAD is not on this path (the window is fed to the backend
/// as one chunk).
#[tokio::test(flavor = "multi_thread")]
async fn transcribe_pcm_window_returns_segments_without_rewriting_transcript() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let orch = test_orchestrator(root.clone());

    // ~2 s of real speech so the decoded PCM has content to slice.
    let clip = load_fixture_wav();
    let mut samples = Vec::with_capacity(clip.len() * 2);
    for _ in 0..2 {
        samples.extend_from_slice(&clip);
    }
    let meeting_id = build_meeting(&root, "Relisten window", &samples, &[], &[]);
    let meeting_dir = root.join(meeting_id.0.to_string());

    // transcript.json starts empty; the window re-listen must NOT change it.
    let before = persistence::read_transcript(&meeting_dir).expect("read transcript before");
    assert!(before.is_empty(), "transcript.json must start empty");

    // A window inside the recording (no pause present → single kept region).
    let stub = Box::new(StubAsrBackend::new("relisten text"));
    let segments = orch
        .transcribe_pcm_window_with_backend(meeting_id, 200, 800, stub)
        .await
        .expect("transcribe_pcm_window_with_backend must succeed");

    assert!(
        !segments.is_empty(),
        "the window must transcribe to at least one segment"
    );
    assert!(
        segments.iter().all(|s| s.text.contains("relisten")),
        "returned segments must carry the stub text; got: {segments:?}"
    );
    // The returned segments sit on the requested window (the chunk start was
    // 200 ms), proving the timestamps are mapped onto the meeting timeline.
    assert!(
        segments.first().map(|s| s.start_ms).unwrap_or(0) >= 200,
        "segment start must be at/after the requested window start"
    );

    // Read-only: transcript.json untouched (still empty).
    let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
    assert!(
        after.is_empty(),
        "transcribe_pcm_window must NOT rewrite transcript.json"
    );

    // The recorder is still Idle (no claim taken / released cycle wedged it).
    assert!(matches!(
        orch.state().await,
        minutist_common::RecordingState::Idle
    ));
}
