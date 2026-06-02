//! GATED integration test for the Phase 6 offline `Orchestrator::rediarize`.
//!
//! Per `architecture/cross-cutting.md` — Automated-testing policy, the
//! default-suite (no-model) wiring of the diarization path is covered by the
//! `StubDiarizer` test in `src/tests.rs` (the re-diarize inner path + the
//! toggle-OFF `stop()` pass). This file holds the **env-var-gated** end-to-end
//! run over real models + the S1 2-speaker fixture, with a no-op skip path so
//! CI passes with the models absent.
//!
//! To run:
//!   MEETING_APP_DIARIZE_SEG_PATH=/path/to/segmentation.onnx \
//!   MEETING_APP_DIARIZE_EMB_PATH=/path/to/embedding.onnx \
//!   cargo test -p orchestrator --features test-source --test rediarize
//!
//! It stages the two diarize models into a tempdir model-registry cache (mirror
//! of `re_transcribe.rs`'s ASR staging), encodes the committed S1 two-speaker
//! fixture into a meeting's `audio.opus` via the persistence Opus encoder,
//! writes a transcript covering both speakers, then runs `orch.rediarize` and
//! asserts the transcript is rewritten with `speaker_id`s, `metadata.json`'s
//! `speaker_count` is updated, the index row is refreshed, and an
//! `AppEvent::DiarizationComplete` is emitted.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use meeting_app_common::{
    AppEvent, AudioFormat, MeetingId, MeetingMeta, ModelFileEntry, ModelId, ModelKind,
    ModelManifestEntry, Segment,
};
use model_registry::ModelRegistry;
use orchestrator::Orchestrator;
use persistence::{MeetingIndex, MeetingWriter};
use settings::{JsonFileStore, SettingsHandle};
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// Env-var gate
// ---------------------------------------------------------------------------

const SEG_MODEL_ID: &str = "pyannote-segmentation-3-0";
const EMB_MODEL_ID: &str = "3dspeaker-campplus-zh-cn-16k-common";

fn diarize_model_env_vars() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let seg = std::env::var("MEETING_APP_DIARIZE_SEG_PATH").ok()?;
    let emb = std::env::var("MEETING_APP_DIARIZE_EMB_PATH").ok()?;
    Some((std::path::PathBuf::from(seg), std::path::PathBuf::from(emb)))
}

// ---------------------------------------------------------------------------
// Model-staging helpers (mirror re_transcribe.rs)
// ---------------------------------------------------------------------------

fn sha256_of_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read model file {path:?}: {e}"));
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

fn diarize_entry(id: &str, display: &str, license: &str, filename: &str, path: &Path) -> ModelManifestEntry {
    let size = std::fs::metadata(path).expect("stat model").len();
    ModelManifestEntry {
        id: ModelId::from(id),
        kind: ModelKind::Diarize,
        display_name: display.to_string(),
        license: license.to_string(),
        total_size_bytes: size,
        files: vec![ModelFileEntry {
            filename: filename.to_string(),
            url: "file:///unused-in-test".to_string(),
            size,
            sha256: sha256_of_file(path),
        }],
    }
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

/// Stage one diarize model's single file into `{cache}/diarize/{id}/{filename}`.
fn stage_model(cache_dir: &Path, id: &str, filename: &str, src: &Path) {
    let model_cache = cache_dir.join("diarize").join(id);
    std::fs::create_dir_all(&model_cache).expect("create model cache dir");
    link_or_copy(src, &model_cache.join(filename));
}

// ---------------------------------------------------------------------------
// Fixture (S1 two-speaker synthetic real-speech clip)
// ---------------------------------------------------------------------------

/// Decode the committed S1 two-speaker fixture (16 kHz mono s16 WAV).
fn load_two_speaker_fixture() -> Vec<f32> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/diarizer/tests/fixtures/two_speakers_synth.wav");
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

/// Build a meeting folder whose `audio.opus` is `samples` encoded via the
/// production Opus encoder, plus a `transcript.json` with `segments` (all
/// `speaker_id = None`) and a `metadata.json` with `speaker_count = 0`.
fn build_meeting(root: &Path, samples: &[f32], segments: &[Segment]) -> MeetingId {
    let meeting_id = MeetingId::new();
    let format = AudioFormat {
        codec: "opus".into(),
        sample_rate: 16_000,
        channels: 1,
        bitrate_kbps: Some(32),
    };

    let mut writer = MeetingWriter::open(root, meeting_id, format.clone()).expect("open writer");
    writer.push_samples(samples).expect("push fixture samples");

    let meta = MeetingMeta {
        uuid: meeting_id,
        title: "Gated rediarize".into(),
        started_at: "2026-06-02T09:00:00Z".into(),
        ended_at: Some("2026-06-02T09:00:12Z".into()),
        duration_ms: (samples.len() as u64 * 1000) / 16_000,
        speaker_count: 0,
        audio_format: format,
        asr_model: None,
        llm_model: None,
        diarizer: None,
        app_version: "0.0.0".into(),
    };
    let folder = writer.finalise(meta).expect("finalise writer");

    std::fs::write(
        folder.path().join("transcript.json"),
        serde_json::to_vec_pretty(segments).unwrap(),
    )
    .expect("write transcript.json");

    meeting_id
}

fn diarize_orchestrator(
    persistence_root: std::path::PathBuf,
    model_cache: std::path::PathBuf,
    manifest: Vec<ModelManifestEntry>,
) -> (Orchestrator, broadcast::Receiver<AppEvent>) {
    let settings_path = persistence_root.join(".diarize_settings.json");
    let store = JsonFileStore::new(settings_path);
    let settings = SettingsHandle::new(store).expect("SettingsHandle construction");

    let (event_tx, event_rx) = broadcast::channel::<AppEvent>(512);
    let registry = ModelRegistry::new(model_cache, manifest, event_tx.clone())
        .expect("ModelRegistry construction");

    let orch =
        Orchestrator::with_event_tx(settings, persistence_root, Arc::new(registry), event_tx);
    (orch, event_rx)
}

// ---------------------------------------------------------------------------
// Gated end-to-end re-diarize
// ---------------------------------------------------------------------------

/// Re-diarize a meeting whose audio is the S1 two-speaker fixture, using the
/// real sherpa models. Verifies that `transcript.json` is rewritten with
/// `speaker_id`s, `metadata.json`'s `speaker_count` is updated (≥ 1), the index
/// row is refreshed, and an `AppEvent::DiarizationComplete` is emitted. Gated on
/// the diarize-model env vars; the no-op skip path is what CI runs.
#[tokio::test(flavor = "multi_thread")]
async fn rediarize_assigns_speakers_over_two_speaker_fixture() {
    let _ = tracing_subscriber::fmt::try_init();

    let (seg_path, emb_path) = match diarize_model_env_vars() {
        Some(paths) => paths,
        None => {
            eprintln!(
                "skipping rediarize_assigns_speakers_over_two_speaker_fixture; diarize model env \
                 vars not set (MEETING_APP_DIARIZE_SEG_PATH and MEETING_APP_DIARIZE_EMB_PATH)"
            );
            return;
        }
    };
    assert!(seg_path.exists(), "segmentation model missing: {seg_path:?}");
    assert!(emb_path.exists(), "embedding model missing: {emb_path:?}");

    let dir = tempfile::tempdir().expect("tempdir");
    let persistence_root = dir.path().join("meetings");
    std::fs::create_dir_all(&persistence_root).expect("create persistence root");
    let model_cache = dir.path().join("models");

    let seg_filename = seg_path.file_name().unwrap().to_string_lossy().to_string();
    let emb_filename = emb_path.file_name().unwrap().to_string_lossy().to_string();

    let manifest = vec![
        diarize_entry(SEG_MODEL_ID, "pyannote segmentation 3.0", "mit", &seg_filename, &seg_path),
        diarize_entry(
            EMB_MODEL_ID,
            "3D-Speaker CAM++",
            "apache-2.0",
            &emb_filename,
            &emb_path,
        ),
    ];
    stage_model(&model_cache, SEG_MODEL_ID, &seg_filename, &seg_path);
    stage_model(&model_cache, EMB_MODEL_ID, &emb_filename, &emb_path);

    let (orch, mut event_rx) =
        diarize_orchestrator(persistence_root.clone(), model_cache, manifest);

    // Encode the two-speaker fixture as a meeting + a transcript that tiles the
    // clip in 1 s segments (covering both speakers across the recording).
    let samples = load_two_speaker_fixture();
    let total_ms = (samples.len() as u64 * 1000) / 16_000;
    let mut segments = Vec::new();
    let mut start = 0u64;
    while start + 1000 <= total_ms {
        segments.push(Segment {
            start_ms: start,
            end_ms: start + 800,
            text: format!("segment at {start} ms"),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
        });
        start += 1000;
    }
    assert!(segments.len() >= 6, "fixture must yield several segments");

    let meeting_id = build_meeting(&persistence_root, &samples, &segments);
    let meeting_dir = persistence_root.join(meeting_id.0.to_string());

    let index = MeetingIndex::open(":memory:").await.expect("open index");
    index
        .rebuild_from_disk(&persistence_root)
        .await
        .expect("seed index");

    orch.rediarize(&index, meeting_id)
        .await
        .expect("rediarize must succeed with the staged models");

    // transcript.json rewritten with at least one speaker_id.
    let rewritten = persistence::read_transcript(&meeting_dir).expect("read transcript");
    assert_eq!(rewritten.len(), segments.len());
    assert!(
        rewritten.iter().any(|s| s.speaker_id.is_some()),
        "re-diarize must overlay at least one speaker_id"
    );

    // metadata.json speaker_count updated (≥ 1) + diarizer recorded.
    let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
    assert!(meta.speaker_count >= 1, "speaker_count must be updated");
    assert!(meta.diarizer.is_some(), "diarizer descriptor must be recorded");

    // index row speaker_count refreshed.
    let rows = index.list_meetings().await.expect("list");
    let row = rows.iter().find(|r| r.id == meeting_id).expect("row");
    assert_eq!(row.speaker_count, meta.speaker_count);

    // DiarizationComplete emitted.
    let mut got = false;
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        match event_rx.try_recv() {
            Ok(AppEvent::DiarizationComplete { meeting_id: mid, speaker_count }) => {
                assert_eq!(mid, meeting_id);
                assert_eq!(speaker_count, meta.speaker_count);
                got = true;
                break;
            }
            Ok(_) => {}
            Err(broadcast::error::TryRecvError::Empty) => break,
            Err(_) => break,
        }
    }
    assert!(got, "rediarize must emit AppEvent::DiarizationComplete");
}
