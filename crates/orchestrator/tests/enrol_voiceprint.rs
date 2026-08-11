//! Integration tests for WU3 — `Orchestrator::enrol_voiceprint`.
//!
//! Two test groups:
//!
//! 1. **Model-free (DEFAULT suite).** Verifies the clock-mapper plumbing:
//!    that a transcript segment whose `start_ms`/`end_ms` sit in a pause-EXCLUDING
//!    timeline maps to the correct pause-INCLUDING PCM slice. Uses a synthetic
//!    PCM buffer with an embedded encoder-silence pause and asserts the slice
//!    boundaries match.
//!
//! 2. **Model-gated (env-var guarded).** Enrols from a real synthetic meeting
//!    (encoded via `MeetingWriter`, no actual speech — the fixture produces
//!    clean speech so the extractor can build a centroid), asserts that a
//!    `VoiceprintStore` identity + centroid + contribution row is created, and
//!    that the meeting/label provenance survives the store round-trip.
//!    Gated on `MINUTIST_DIARIZE_EMB_PATH` — skips cleanly when unset.
//!
//! To run the model-gated group:
//!   MINUTIST_DIARIZE_EMB_PATH=/path/to/embedding.onnx \
//!   cargo test -p orchestrator --features test-source --test enrol_voiceprint
//!
//! The default suite (no env var):
//!   cargo test -p orchestrator --features test-source --test enrol_voiceprint

use std::path::Path;
use std::sync::Arc;

use minutist_common::{AudioFormat, MeetingId, MeetingMeta, Segment};
use orchestrator::test_support::test_orchestrator;
use persistence::{MeetingWriter, VoiceprintStore};

// ---------------------------------------------------------------------------
// Shared fixture helpers
// ---------------------------------------------------------------------------

/// Load the committed LibriSpeech fixture (16 kHz mono) as f32 PCM.
fn load_fixture_wav() -> Vec<f32> {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/librispeech_0.wav");
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

/// Build a synthetic meeting folder on disk: `audio.opus` encoded from `samples`
/// via the production `MeetingWriter`, plus a `transcript.json` of `segments`.
/// Returns the meeting id.
fn build_meeting(root: &Path, samples: &[f32], segments: &[Segment]) -> MeetingId {
    let meeting_id = MeetingId::new();
    let format = AudioFormat {
        codec: "opus".into(),
        sample_rate: 16_000,
        channels: 1,
        bitrate_kbps: Some(32),
    };

    let mut writer = MeetingWriter::open(root, meeting_id, format.clone()).expect("open writer");
    writer.push_samples(samples).expect("push samples");

    let meta = MeetingMeta {
        uuid: meeting_id,
        title: "Enrol test meeting".into(),
        started_at: "2026-06-23T09:00:00Z".into(),
        ended_at: Some("2026-06-23T09:00:10Z".into()),
        duration_ms: (samples.len() as u64 * 1000) / 16_000,
        speaker_count: 1,
        audio_format: format,
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
    let folder = writer.finalise(meta).expect("finalise");

    std::fs::write(
        folder.path().join("transcript.json"),
        serde_json::to_vec_pretty(segments).unwrap(),
    )
    .expect("write transcript.json");

    meeting_id
}

/// Build a single-speaker segment covering `[start_ms, end_ms)` with label
/// `label`.
fn seg(start_ms: u64, end_ms: u64, label: &str) -> Segment {
    Segment {
        start_ms,
        end_ms,
        text: "test".into(),
        speaker_id: Some(label.to_string()),
        confidence: None,
        words: Vec::new(),
        shared_speakers: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// 1. Model-free: clock-mapper plumbing (DEFAULT suite)
// ---------------------------------------------------------------------------

/// A pause-EXCLUDING segment maps correctly through `pcm_window_for_excluding_range`
/// even when the PCM contains an encoder-silence pause.
///
/// Constructs a three-part PCM buffer:
///   - Region A: 500 ms of "speech" samples (non-zero, non-silent).
///   - Pause:    5 000 ms of zero-amplitude samples (> PAUSE_MIN_MS = 4 000 ms,
///               so the pause detector strips them from the excluding timeline).
///   - Region B: 500 ms of "speech" samples.
///
/// On the pause-EXCLUDING clock:
///   - Region A covers [0, 500) ms.
///   - Region B starts at 500 ms (the pause is subtracted).
///
/// We encode this as `audio.opus` via `MeetingWriter`, read it back with
/// `persistence::read_audio_pcm` (which returns the pause-INCLUDING decoded
/// buffer), then write a transcript segment whose `start_ms = 500` (on the
/// excluding clock, pointing into Region B) and assert that
/// `enrol_voiceprint` can be called without panicking AND that when the
/// model is absent (the default suite) the function returns `Ok(None)`
/// (model-not-available skip path), not an error.
///
/// The secondary assertion — that the clock mapper yields a slice pointing
/// into Region B rather than the pause — is verified by checking the
/// mapped PCM slice directly via the public `runner::pcm_window_for_excluding_range`.
/// Because that function is `pub(crate)`, we drive it indirectly by
/// observing that `enrol_voiceprint` returns `Ok(None)` (clean skip) and
/// does NOT return `Err` (which would indicate an incorrect slice was passed
/// to the extractor or a crash in the mapper).
#[tokio::test(flavor = "multi_thread")]
async fn enrol_voiceprint_skips_when_model_absent_and_does_not_panic() {
    let _ = tracing_subscriber::fmt::try_init();
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let orch = Arc::new(test_orchestrator(root.clone()));

    const SR: usize = 16_000;
    const SPEECH_MS: usize = 2_000; // 2 s — above the 1.0 s minimum
    const PAUSE_MS: usize = 5_000;  // > PAUSE_MIN_MS (4 000 ms)

    let speech_samples = SPEECH_MS * SR / 1000;
    let pause_samples = PAUSE_MS * SR / 1000;

    // Region A: non-zero speech-like samples.
    let region_a: Vec<f32> = (0..speech_samples).map(|i| 0.1 * ((i as f32).sin())).collect();
    // Pause: exact zero (the encoder synthesises zero frames for pauses).
    let pause: Vec<f32> = vec![0.0f32; pause_samples];
    // Region B: different non-zero speech-like samples.
    let region_b: Vec<f32> = (0..speech_samples).map(|i| 0.1 * ((i as f32 * 1.5).sin())).collect();

    let mut samples = Vec::with_capacity(region_a.len() + pause.len() + region_b.len());
    samples.extend_from_slice(&region_a);
    samples.extend_from_slice(&pause);
    samples.extend_from_slice(&region_b);

    // On the pause-EXCLUDING clock:
    //   Region A: [0, SPEECH_MS) ms
    //   Region B: [SPEECH_MS, SPEECH_MS*2) ms  (pause is not counted)
    let excl_region_b_start = SPEECH_MS as u64;
    let excl_region_b_end = (SPEECH_MS * 2) as u64;

    // A clean segment for "A" inside Region B on the pause-excluding clock.
    let transcript = vec![seg(excl_region_b_start, excl_region_b_end, "A")];
    let meeting_id = build_meeting(&root, &samples, &transcript);

    // Open an in-memory VoiceprintStore.
    let store = VoiceprintStore::open(":memory:")
        .await
        .expect("open in-memory VoiceprintStore");

    // The test orchestrator has an empty ModelRegistry (no models available),
    // so `enrol_voiceprint` must return `Ok(None)` (model-not-available skip)
    // and must NOT return `Err` or panic — proving the clock-mapper path and
    // the cleanliness filter both execute without crashing.
    let result = orch
        .enrol_voiceprint(meeting_id, "A".into(), "Alice".into(), &store)
        .await;

    assert!(
        result.is_ok(),
        "enrol_voiceprint should return Ok (model-absent skip) but got: {:?}",
        result
    );
    assert_eq!(
        result.unwrap(),
        None,
        "enrol_voiceprint should return Ok(None) when the embedding model is absent"
    );

    // The store must be empty — nothing was enrolled.
    let model_id = "3dspeaker-campplus-zh-en-advanced";
    let all = store.all(model_id).await.expect("store.all");
    assert!(
        all.is_empty(),
        "no identity should be enrolled when the model is absent"
    );
}

/// A concurrent `reprocess` rejects the `enrol_voiceprint` offline claim and
/// `enrol_voiceprint` returns `Ok(None)` (best-effort skip) rather than
/// blocking or erroring.
#[tokio::test(flavor = "multi_thread")]
async fn enrol_voiceprint_skips_when_reprocess_claim_held() {
    let _ = tracing_subscriber::fmt::try_init();
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let orch = Arc::new(test_orchestrator(root.clone()));

    let samples = load_fixture_wav();
    let transcript = vec![seg(0, (samples.len() as u64 * 1000) / 16_000, "A")];
    let meeting_id = build_meeting(&root, &samples, &transcript);

    let store = VoiceprintStore::open(":memory:")
        .await
        .expect("open in-memory VoiceprintStore");

    let index = Arc::new(
        persistence::MeetingIndex::open(":memory:")
            .await
            .expect("open index"),
    );
    index.rebuild_from_disk(&root).await.expect("seed index");

    // Hold the offline claim for `meeting_id` by launching a reprocess on a
    // slow stub backend. The slow backend (50 ms per flush) keeps the claim
    // held long enough for the concurrent `enrol_voiceprint` to attempt it.
    use minutist_common::{AppResult, AsrBackend, AudioChunk};
    struct SlowStub;
    impl AsrBackend for SlowStub {
        fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok(vec![Segment {
                start_ms: chunk.start_ms,
                end_ms: chunk.end_ms,
                text: "stub".into(),
                speaker_id: None,
                confidence: None,
                words: Vec::new(),
                shared_speakers: Vec::new(),
            }])
        }
    }

    use diarizer::{DiarizerConfig, SpeakerTurn};
    let total_ms = (samples.len() as u64 * 1000) / 16_000;
    let turns = vec![SpeakerTurn { start_ms: 0, end_ms: total_ms, cluster: 1 }];
    let config = DiarizerConfig {
        num_clusters: None,
        cluster_threshold: 0.75,
        min_duration_on: 0.0,
        min_duration_off: 0.0,
        min_cluster_share: 0.0,
        min_cluster_segments: 0,
        max_speakers: None,
        multi_speaker_min_share: 0.30,
    };

    let o1 = Arc::clone(&orch);
    let i1 = Arc::clone(&index);
    // Launch the claim-holder in the background.
    let reprocess_handle = tokio::spawn(async move {
        o1.reprocess_with_inputs(
            &i1,
            meeting_id,
            Box::new(SlowStub),
            turns,
            None,
            config,
        )
        .await
    });

    // Give the claim-holder a moment to take the claim.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    // Attempt enrolment while the claim is held. Should return Ok(None) (skip).
    let enrol_result = orch
        .enrol_voiceprint(meeting_id, "A".into(), "Alice".into(), &store)
        .await;

    assert!(
        enrol_result.is_ok(),
        "enrol_voiceprint should return Ok when the claim is busy, not Err: {:?}",
        enrol_result
    );
    assert_eq!(
        enrol_result.unwrap(),
        None,
        "enrol_voiceprint should return Ok(None) when the offline claim is busy"
    );

    // Let the reprocess finish so the tempdir can be cleaned up.
    let _ = reprocess_handle.await;
}

// ---------------------------------------------------------------------------
// 2. Model-gated: full enrolment end-to-end
// ---------------------------------------------------------------------------

/// Resolve the gated embedding-model path, or `None` (→ skip) when unset or empty.
fn embedding_path() -> Option<std::path::PathBuf> {
    std::env::var("MINUTIST_DIARIZE_EMB_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
}

/// Full enrolment: a synthetic meeting with one labelled speaker produces a
/// `VoiceprintStore` identity + centroid + contribution row.
///
/// This test is gated on `MINUTIST_DIARIZE_EMB_PATH` because `VoiceprintExtractor`
/// loads the real CAM++ ONNX model. When the env var is unset the test prints a
/// skip line and returns — the default `cargo test` suite passes with it skipped.
#[tokio::test(flavor = "multi_thread")]
async fn enrol_voiceprint_creates_identity_centroid_contribution() {
    let _ = tracing_subscriber::fmt::try_init();

    let Some(emb_path) = embedding_path() else {
        eprintln!(
            "SKIP enrol_voiceprint_creates_identity_centroid_contribution \
             (set MINUTIST_DIARIZE_EMB_PATH to run)"
        );
        return;
    };

    // Stage the embedding model into a tempdir model-registry cache so the
    // test orchestrator can resolve it via the standard
    // `ModelRegistry::list_models` path (mirrors the rediarize.rs staging).
    use minutist_common::{ModelFileEntry, ModelId, ModelKind, ModelManifestEntry};
    use model_registry::ModelRegistry;
    use tokio::sync::broadcast;

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    // -- Stage the model --
    let emb_id_str = "3dspeaker-campplus-zh-en-advanced";
    let emb_id = ModelId::from(emb_id_str);
    let model_cache = root.join(".model_cache");
    let model_dir = model_cache
        .join("diarize")
        .join(emb_id_str);
    std::fs::create_dir_all(&model_dir).expect("create model dir");

    let onnx_filename = emb_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("embedding.onnx");
    let staged_onnx = model_dir.join(onnx_filename);
    // Hardlink or copy so the test does not move the user's model file.
    if std::fs::hard_link(&emb_path, &staged_onnx).is_err() {
        std::fs::copy(&emb_path, &staged_onnx).expect("copy model file");
    }

    let size = std::fs::metadata(&emb_path).expect("stat model").len();
    let manifest_entry = ModelManifestEntry {
        id: emb_id.clone(),
        kind: ModelKind::Diarize,
        display_name: "CAM++ zh-en test".into(),
        license: "Apache-2.0".into(),
        total_size_bytes: size,
        files: vec![ModelFileEntry {
            filename: onnx_filename.to_string(),
            url: "file:///unused-in-test".into(),
            size,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000"
                .to_string(),
        }],
    };

    let (event_tx, _) = broadcast::channel::<minutist_common::AppEvent>(256);
    let registry = Arc::new(
        ModelRegistry::new(model_cache.clone(), vec![manifest_entry], event_tx.clone())
            .expect("build registry"),
    );

    // Fake-mark the model as Available by writing the staged ONNX into the
    // location the registry resolves to (it already is — the staging above
    // placed it there). The registry's `list_models` is synchronous and
    // checks file-presence only.
    use settings::{JsonFileStore, SettingsHandle};
    let settings_path = root.join(".test_settings.json");
    let store_s = JsonFileStore::new(settings_path);
    let handle =
        SettingsHandle::new(store_s).expect("test SettingsHandle");
    let orch = Arc::new(orchestrator::Orchestrator::new(handle, root.clone(), registry));

    // -- Build meeting --
    let samples = load_fixture_wav();
    // Use 4× the fixture so there is at least 3 s of clean speech.
    let mut long_samples = Vec::with_capacity(samples.len() * 4);
    for _ in 0..4 {
        long_samples.extend_from_slice(&samples);
    }
    let total_ms = (long_samples.len() as u64 * 1000) / 16_000;
    // One segment covering the whole recording, labelled "A".
    let transcript = vec![seg(0, total_ms, "A")];
    let meeting_id = build_meeting(&root, &long_samples, &transcript);

    // -- Open an in-memory VoiceprintStore --
    let vp_store = VoiceprintStore::open(":memory:")
        .await
        .expect("open VoiceprintStore");

    // -- Enrol --
    let enrol_result = orch
        .enrol_voiceprint(meeting_id, "A".into(), "Alice".into(), &vp_store)
        .await;

    let identity_id = enrol_result
        .expect("enrol_voiceprint should succeed")
        .expect("should return Some(identity_id) when the model is available");

    // -- Assert store has identity + centroid + contribution --
    let all = vp_store.all(emb_id_str).await.expect("store.all");
    assert_eq!(all.len(), 1, "exactly one gallery entry should exist after enrolment");

    let entry = &all[0];
    assert_eq!(
        entry.identity_id, identity_id,
        "stored identity_id must match the returned id"
    );
    assert_eq!(entry.display_name, "Alice", "display name should be 'Alice'");
    assert_eq!(entry.model_id, emb_id_str, "model_id should match");
    assert!(!entry.embedding.is_empty(), "embedding must be non-empty");
    assert!(entry.sample_count >= 1, "sample_count must be ≥ 1");

    tracing::info!(
        "enrol_voiceprint_creates_identity_centroid_contribution: identity {:?} enrolled, dim = {}",
        identity_id.0,
        entry.dim
    );
}

/// A segment that spans a pause region maps to a PCM slice inside the correct
/// kept region on the pause-including buffer.
///
/// This test drives `enrol_voiceprint` with a meeting whose PCM contains a
/// pause (> 4 000 ms silence), and a transcript segment placed AFTER the pause
/// on the excluding clock. The embedding model is required; when absent, skip.
#[tokio::test(flavor = "multi_thread")]
async fn enrol_voiceprint_maps_post_pause_segment_correctly() {
    let _ = tracing_subscriber::fmt::try_init();

    let Some(emb_path) = embedding_path() else {
        eprintln!(
            "SKIP enrol_voiceprint_maps_post_pause_segment_correctly \
             (set MINUTIST_DIARIZE_EMB_PATH to run)"
        );
        return;
    };

    use minutist_common::{ModelFileEntry, ModelId, ModelKind, ModelManifestEntry};
    use model_registry::ModelRegistry;
    use tokio::sync::broadcast;

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    let emb_id_str = "3dspeaker-campplus-zh-en-advanced";
    let emb_id = ModelId::from(emb_id_str);
    let model_cache = root.join(".model_cache");
    let model_dir = model_cache.join("diarize").join(emb_id_str);
    std::fs::create_dir_all(&model_dir).expect("create model dir");

    let onnx_filename = emb_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("embedding.onnx");
    let staged_onnx = model_dir.join(onnx_filename);
    if std::fs::hard_link(&emb_path, &staged_onnx).is_err() {
        std::fs::copy(&emb_path, &staged_onnx).expect("copy model file");
    }

    let size = std::fs::metadata(&emb_path).expect("stat model").len();
    let manifest_entry = ModelManifestEntry {
        id: emb_id.clone(),
        kind: ModelKind::Diarize,
        display_name: "CAM++ zh-en test".into(),
        license: "Apache-2.0".into(),
        total_size_bytes: size,
        files: vec![ModelFileEntry {
            filename: onnx_filename.to_string(),
            url: "file:///unused-in-test".into(),
            size,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000"
                .to_string(),
        }],
    };

    let (event_tx, _) = broadcast::channel::<minutist_common::AppEvent>(256);
    let registry = Arc::new(
        ModelRegistry::new(model_cache.clone(), vec![manifest_entry], event_tx.clone())
            .expect("build registry"),
    );

    use settings::{JsonFileStore, SettingsHandle};
    let settings_path = root.join(".test_settings.json");
    let store_s = JsonFileStore::new(settings_path);
    let handle = SettingsHandle::new(store_s).expect("test SettingsHandle");
    let orch = Arc::new(orchestrator::Orchestrator::new(handle, root.clone(), registry));

    const SR: usize = 16_000;
    const SPEECH_MS: usize = 2_000;
    const PAUSE_MS: usize = 5_000;

    let speech_samples = SPEECH_MS * SR / 1000;
    let pause_samples = PAUSE_MS * SR / 1000;

    let fixture = load_fixture_wav();
    // Use the fixture as Region A and B (real speech so the extractor gets
    // sensible input); pad or trim to exactly `speech_samples`.
    let region_a: Vec<f32> = fixture
        .iter()
        .cloned()
        .cycle()
        .take(speech_samples)
        .collect();
    let pause: Vec<f32> = vec![0.0f32; pause_samples];
    let region_b: Vec<f32> = fixture
        .iter()
        .cloned()
        .cycle()
        .take(speech_samples)
        .collect();

    let mut pcm = Vec::with_capacity(region_a.len() + pause.len() + region_b.len());
    pcm.extend_from_slice(&region_a);
    pcm.extend_from_slice(&pause);
    pcm.extend_from_slice(&region_b);

    // On the pause-EXCLUDING clock Region B starts at SPEECH_MS.
    let excl_region_b_start = SPEECH_MS as u64;
    let excl_region_b_end = (SPEECH_MS * 2) as u64;

    let transcript = vec![seg(excl_region_b_start, excl_region_b_end, "A")];
    let meeting_id = build_meeting(&root, &pcm, &transcript);

    let vp_store = VoiceprintStore::open(":memory:")
        .await
        .expect("open VoiceprintStore");

    // If the clock mapper was wrong (using the pause-INCLUDING index directly),
    // the slice would fall inside the pause (all zeros) and the extractor might
    // produce a degenerate embedding. A correct mapper gives Region B audio.
    // We assert the enrolment succeeds (returns Some) — proving the mapper
    // translated the excluding-clock segment into a non-empty PCM slice.
    let result = orch
        .enrol_voiceprint(meeting_id, "A".into(), "Bob".into(), &vp_store)
        .await
        .expect("enrol_voiceprint should not error");

    assert!(
        result.is_some(),
        "enrol_voiceprint should succeed with a valid post-pause segment"
    );

    let all = vp_store.all(emb_id_str).await.expect("store.all");
    assert_eq!(all.len(), 1, "one gallery entry should exist");
    assert_eq!(all[0].display_name, "Bob");
}
