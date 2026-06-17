//! GATED integration test for the Phase 6 offline `Orchestrator::rediarize`.
//!
//! Per `architecture/cross-cutting.md` — Automated-testing policy, the
//! default-suite (no-model) wiring of the diarization + #0015-phase-4 re-ASR
//! split is covered by the `rediarize_with_split_inputs` tests in `src/tests.rs`
//! (synthetic turns + a stub `AsrBackend` drive the re-diarize inner path, the
//! split, and the toggle-OFF `stop()` pass). This file holds the
//! **env-var-gated** end-to-end run over real models + the S1 2-speaker fixture,
//! with a no-op skip path so CI passes with the models absent.
//!
//! To run:
//!   MINUTIST_DIARIZE_SEG_PATH=/path/to/segmentation.onnx \
//!   MINUTIST_DIARIZE_EMB_PATH=/path/to/embedding.onnx \
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

use minutist_common::{
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
const EMB_MODEL_ID: &str = "3dspeaker-campplus-zh-en-advanced";

fn diarize_model_env_vars() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    // Treat an EMPTY value as unset (skip) so `VAR=""` does not stage an empty
    // path and panic; only a non-empty path counts as "set".
    let seg = non_empty_env("MINUTIST_DIARIZE_SEG_PATH")?;
    let emb = non_empty_env("MINUTIST_DIARIZE_EMB_PATH")?;
    Some((std::path::PathBuf::from(seg), std::path::PathBuf::from(emb)))
}

/// Read `name` from the environment, returning `None` when it is unset OR set
/// to an empty string. This keeps `VAR=""` equivalent to "unset" so the gated
/// tests skip cleanly instead of staging an empty model path.
fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
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

/// Decode a committed 16 kHz mono s16 WAV diarizer fixture into f32 PCM.
fn load_diarizer_fixture(filename: &str) -> Vec<f32> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/diarizer/tests/fixtures")
        .join(filename);
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
        speaker_names: std::collections::BTreeMap::new(),
        notes_format: 0,
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
// Gated end-to-end re-diarize (shared core)
// ---------------------------------------------------------------------------

/// Run the gated end-to-end `rediarize` over a diarizer fixture under the
/// production `DiarizerConfig::default()` (threshold/auto-count mode — the only
/// shipped config), asserting `metadata.json`'s `speaker_count` equals
/// `expected_speakers`.
///
/// Stages the real sherpa models into a tempdir registry cache, encodes the
/// fixture as a meeting whose `transcript.json` tiles the clip in 1 s segments,
/// runs `orch.rediarize`, and asserts: `transcript.json` rewritten with at least
/// one `speaker_id`, `metadata.json`'s `speaker_count == expected_speakers`,
/// the index row refreshed, and `AppEvent::DiarizationComplete` emitted with the
/// same count. The segment tiling derives from audio length, so no boundary
/// constants are needed and the assertion holds against the genuine-distinct /
/// genuine-single fixture contract.
async fn run_gated_rediarize_over_fixture(fixture_filename: &str, expected_speakers: u32) {
    let _ = tracing_subscriber::fmt::try_init();

    let (seg_path, emb_path) = match diarize_model_env_vars() {
        Some(paths) => paths,
        None => {
            eprintln!(
                "skipping gated rediarize over {fixture_filename}; diarize model env vars not set \
                 (MINUTIST_DIARIZE_SEG_PATH and MINUTIST_DIARIZE_EMB_PATH)"
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

    // Encode the fixture as a meeting + a transcript that tiles the clip in 1 s
    // segments (derived from audio length, covering the whole recording).
    let samples = load_diarizer_fixture(fixture_filename);
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
            shared_speakers: Vec::new(),
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

    // metadata.json speaker_count equals the genuine fixture speaker count under
    // the production DiarizerConfig::default() + diarizer recorded.
    let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
    assert_eq!(
        meta.speaker_count, expected_speakers,
        "speaker_count must equal the fixture's genuine speaker count under the production config"
    );
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

// ---------------------------------------------------------------------------
// Gated end-to-end re-diarize (per-fixture)
// ---------------------------------------------------------------------------

/// Re-diarize a meeting whose audio is the two-speaker fixture (two genuinely
/// distinct concatenated real readers), using the real sherpa models under the
/// production `DiarizerConfig::default()`. Asserts `speaker_count == 2`. Gated on
/// the diarize-model env vars; the no-op skip path is what CI runs.
#[tokio::test(flavor = "multi_thread")]
async fn rediarize_assigns_speakers_over_two_speaker_fixture() {
    run_gated_rediarize_over_fixture("two_speakers_synth.wav", 2).await;
}

/// Single-speaker control: re-diarize a meeting whose audio is one real speaker
/// repeated, under the same production `DiarizerConfig::default()` + staging +
/// re-diarize path. Asserts `speaker_count == 1`, proving the auto-count
/// threshold mode does not over-segment a single speaker into multiple. Gated on
/// the diarize-model env vars; the no-op skip path is what CI runs.
#[tokio::test(flavor = "multi_thread")]
async fn rediarize_reports_single_speaker_over_control_fixture() {
    run_gated_rediarize_over_fixture("single_speaker_control.wav", 1).await;
}
