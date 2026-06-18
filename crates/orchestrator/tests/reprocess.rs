//! Integration tests for the #0015-phase-5 offline `Orchestrator::reprocess` —
//! the merged re-transcribe + diarize one-shot under a SINGLE offline claim.
//!
//! Model-free (DEFAULT suite): driven through the `reprocess_with_inputs`
//! test-source seam, which composes the SAME re-transcribe → diarize/split/merge
//! → finalise-once path the production `reprocess` uses, but with a stub ASR
//! backend (for the re-transcribe step) and stub `SpeakerTurn`s + a stub split
//! backend (for the diarize step). The real Silero VAD (vendored at
//! `resources/silero/silero_vad_v4.onnx`) drives the re-transcribe segmentation
//! over a committed real-speech fixture, so no ASR / sherpa / Qwen model is
//! required.
//!
//! Coverage (WU3):
//! 1. **One claim / release, no `Idle` window between the sub-steps.** While a
//!    `reprocess` is in flight (held open by a slow stub ASR), a concurrent
//!    offline op is rejected with `AppError::InvalidInput` — the claim spans the
//!    WHOLE pass, so no gap opens between re-transcribe and diarize.
//! 2. **Order: re-transcribe FIRST, then split + label.** The final
//!    `transcript.json` carries the FRESH re-transcribed text AND is
//!    speaker-labelled, and the STALE seed text is gone — proving the fresh
//!    transcript was written, then diarized (not diarize-first then clobbered by
//!    a re-transcribe finalise, the lost-update the verifiers flagged).
//! 3. **`speaker_names` cleared.** A reprocess always diarizes, so a seeded
//!    `speaker_names` map is cleared every run (accept-and-warn product default).

use std::path::Path;
use std::sync::Arc;

use diarizer::{DiarizerConfig, SpeakerTurn};
use minutist_common::{
    AppError, AppResult, AsrBackend, AudioChunk, AudioFormat, MeetingId, MeetingMeta, Segment,
};
use orchestrator::test_support::test_orchestrator;
use persistence::{MeetingIndex, MeetingWriter};

// ---------------------------------------------------------------------------
// Stub backends + fixture helpers
// ---------------------------------------------------------------------------

/// A stub ASR backend returning canned text per chunk, with an optional
/// per-call delay so a test can hold the re-transcribe step open long enough to
/// race a concurrent offline op against the single claim.
struct DelayingStubBackend {
    text: String,
    delay_ms: u64,
}

impl AsrBackend for DelayingStubBackend {
    fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
        if self.delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
        }
        Ok(vec![Segment {
            start_ms: chunk.start_ms,
            end_ms: chunk.end_ms,
            text: self.text.clone(),
            speaker_id: None,
            confidence: None,
            words: Vec::new(),
            shared_speakers: Vec::new(),
        }])
    }
}

/// A `DiarizerConfig` with only the multi-speaker flag active (no prune/cap), so
/// `overlay_speakers` labels every segment by its dominant cluster without
/// folding any cluster away. Single-cluster turns therefore letter every
/// re-transcribed segment "A".
fn label_only_config() -> DiarizerConfig {
    DiarizerConfig {
        num_clusters: None,
        cluster_threshold: 0.75,
        min_duration_on: 0.0,
        min_duration_off: 0.0,
        min_cluster_share: 0.0,
        min_cluster_segments: 0,
        max_speakers: None,
        multi_speaker_min_share: 0.30,
    }
}

/// Load the committed LibriSpeech real-speech fixture (16 kHz mono) as f32 PCM.
/// Synthetic tones fail the real Silero VAD, so the re-transcribe step needs
/// real speech to yield segments.
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

/// Build a meeting folder on disk: `audio.opus` encoded from `samples` via the
/// production `MeetingWriter`, a `metadata.json` (with `speaker_names` seeded
/// from `seed_names`), and a `transcript.json` of `stale_segments` (the text the
/// reprocess must REPLACE). Returns the meeting id.
fn build_meeting(
    root: &Path,
    samples: &[f32],
    stale_segments: &[Segment],
    seed_names: &[(&str, &str)],
) -> MeetingId {
    let meeting_id = MeetingId::new();
    let format = AudioFormat {
        codec: "opus".into(),
        sample_rate: 16_000,
        channels: 1,
        bitrate_kbps: Some(32),
    };

    let mut writer = MeetingWriter::open(root, meeting_id, format.clone()).expect("open writer");
    writer.push_samples(samples).expect("push samples");

    let mut speaker_names = std::collections::BTreeMap::new();
    for (k, v) in seed_names {
        speaker_names.insert((*k).to_string(), (*v).to_string());
    }

    let meta = MeetingMeta {
        uuid: meeting_id,
        title: "Reprocess me".into(),
        started_at: "2026-06-02T09:00:00Z".into(),
        ended_at: Some("2026-06-02T09:00:06Z".into()),
        duration_ms: (samples.len() as u64 * 1000) / 16_000,
        speaker_count: 0,
        audio_format: format,
        asr_model: None,
        llm_model: None,
        diarizer: None,
        speaker_names,
        notes_format: 0,
        collection_id: None,
        app_version: "0.0.0".into(),
    };
    let folder = writer.finalise(meta).expect("finalise");

    std::fs::write(
        folder.path().join("transcript.json"),
        serde_json::to_vec_pretty(stale_segments).unwrap(),
    )
    .expect("write transcript.json");

    meeting_id
}

/// A single stale ASR segment (Qwen shape: no words) covering `[s, e)`.
fn stale_seg(s: u64, e: u64, text: &str) -> Segment {
    Segment {
        start_ms: s,
        end_ms: e,
        text: text.to_string(),
        speaker_id: None,
        confidence: None,
        words: Vec::new(),
        shared_speakers: Vec::new(),
    }
}

/// Real speech repeated so the VAD reliably yields several segments.
fn fixture_samples() -> Vec<f32> {
    let clip = load_fixture_wav();
    let mut samples = Vec::with_capacity(clip.len() * 4);
    for _ in 0..4 {
        samples.extend_from_slice(&clip);
    }
    samples
}

/// One turn covering the whole recording on a single cluster, so every
/// re-transcribed segment is labelled "A" (the order test only needs labels to
/// be assigned over the FRESH transcript, not a split).
fn single_cluster_turns(samples: &[f32]) -> Vec<SpeakerTurn> {
    let total_ms = (samples.len() as u64 * 1000) / 16_000;
    vec![SpeakerTurn {
        start_ms: 0,
        end_ms: total_ms,
        cluster: 1,
    }]
}

// ---------------------------------------------------------------------------
// 1. ONE claim / release — no Idle window between re-transcribe and diarize
// ---------------------------------------------------------------------------

/// A `reprocess` holds a SINGLE offline claim for the WHOLE serial pass —
/// re-transcribe THEN diarize — with no `Idle` window between the sub-steps.
///
/// Proven two ways. (1) Exactly-one-of-two: two concurrent `reprocess` calls on
/// the same meeting are launched; the offline claim lets exactly one through and
/// rejects the other with `AppError::InvalidInput`. Were there an `Idle` window
/// between the re-transcribe and diarize sub-steps, the loser could slip in and
/// BOTH could succeed (and race the transcript). (2) The losing op observes the
/// claim held while the winner is mid-pass; the slow stub ASR keeps the winner's
/// claim held across multiple flushes, so the rejection is not a one-instant
/// coincidence. After both settle, the recorder is `Idle` again.
#[tokio::test(flavor = "multi_thread")]
async fn reprocess_holds_one_claim_with_no_idle_window() {
    let _ = tracing_subscriber::fmt::try_init();
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let orch = Arc::new(test_orchestrator(root.clone()));

    let samples = fixture_samples();
    let stale = vec![stale_seg(0, 1_000, "STALE")];
    let meeting_id = build_meeting(&root, &samples, &stale, &[]);

    let index = Arc::new(MeetingIndex::open(":memory:").await.expect("open index"));
    index.rebuild_from_disk(&root).await.expect("seed index");

    // Two concurrent reprocess calls on the SAME meeting. A per-flush delay keeps
    // each winner's single claim held across the whole re-transcribe + diarize
    // pass, so the two calls genuinely overlap.
    let o1 = Arc::clone(&orch);
    let i1 = Arc::clone(&index);
    let s1 = samples.clone();
    let h1 = tokio::spawn(async move {
        o1.reprocess_with_inputs(
            &i1,
            meeting_id,
            Box::new(DelayingStubBackend { text: "op-1".into(), delay_ms: 50 }),
            single_cluster_turns(&s1),
            None,
            label_only_config(),
        )
        .await
    });
    let o2 = Arc::clone(&orch);
    let i2 = Arc::clone(&index);
    let s2 = samples.clone();
    let h2 = tokio::spawn(async move {
        o2.reprocess_with_inputs(
            &i2,
            meeting_id,
            Box::new(DelayingStubBackend { text: "op-2".into(), delay_ms: 50 }),
            single_cluster_turns(&s2),
            None,
            label_only_config(),
        )
        .await
    });

    let r1 = h1.await.expect("join 1");
    let r2 = h2.await.expect("join 2");

    let oks = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
    let invalid = [&r1, &r2]
        .iter()
        .filter(|r| matches!(r, Err(AppError::InvalidInput { .. })))
        .count();
    assert_eq!(
        oks, 1,
        "exactly one concurrent reprocess must hold the single claim; r1={r1:?} r2={r2:?}"
    );
    assert_eq!(
        invalid, 1,
        "the losing concurrent reprocess must be rejected (no Idle window mid-pass); \
         r1={r1:?} r2={r2:?}"
    );

    // After both settle, the recorder is Idle again — a fresh reprocess is accepted.
    orch.reprocess_with_inputs(
        &index,
        meeting_id,
        Box::new(DelayingStubBackend { text: "op-3".into(), delay_ms: 0 }),
        single_cluster_turns(&samples),
        None,
        label_only_config(),
    )
    .await
    .expect("a reprocess after release must be accepted (claim freed)");
}

// ---------------------------------------------------------------------------
// 2. ORDER — re-transcribe FIRST, then split + label (lost-update guard)
// ---------------------------------------------------------------------------

/// The final transcript reflects the RE-TRANSCRIBED text labelled by the diarize
/// step — not the stale seed, and not an un-labelled re-transcribe. This guards
/// the diarize-first lost-update: had diarize run first and re-transcribe
/// finalised last, the labels would be gone (clobbered) and/or the text stale.
#[tokio::test(flavor = "multi_thread")]
async fn reprocess_retranscribes_then_labels_not_diarize_first() {
    let _ = tracing_subscriber::fmt::try_init();
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let orch = test_orchestrator(root.clone());

    let samples = fixture_samples();
    // Seed a STALE transcript whose text must NOT survive the reprocess.
    let stale = vec![stale_seg(0, 1_000, "STALE-TEXT"), stale_seg(1_000, 2_000, "STALE-TEXT")];
    let meeting_id = build_meeting(&root, &samples, &stale, &[]);
    let meeting_dir = root.join(meeting_id.0.to_string());

    let index = MeetingIndex::open(":memory:").await.expect("open index");
    index.rebuild_from_disk(&root).await.expect("seed index");

    let asr = Box::new(DelayingStubBackend {
        text: "FRESH-TEXT".into(),
        delay_ms: 0,
    });

    orch.reprocess_with_inputs(
        &index,
        meeting_id,
        asr,
        single_cluster_turns(&samples),
        None,
        label_only_config(),
    )
    .await
    .expect("reprocess must succeed");

    let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
    assert!(!after.is_empty(), "reprocess must rewrite the transcript");

    // The stale seed text is gone (the fresh transcript replaced it BEFORE the
    // diarize step read it).
    assert!(
        after.iter().all(|s| !s.text.contains("STALE")),
        "stale seed text must be replaced by the re-transcribe; got {after:?}"
    );
    // The fresh re-transcribed text survives the final (diarize) write — i.e. the
    // diarize step ran OVER the fresh transcript, then finalised once.
    assert!(
        after.iter().any(|s| s.text.contains("FRESH-TEXT")),
        "re-transcribed text must survive the diarize finalise; got {after:?}"
    );
    // Every segment is speaker-labelled "A" (single cluster) — the labels written
    // by the diarize step are present, NOT clobbered by a trailing re-transcribe
    // finalise (the diarize-first lost-update).
    assert!(
        after.iter().all(|s| s.speaker_id.as_deref() == Some("A")),
        "diarize labels must be present on the fresh transcript; got {after:?}"
    );

    // metadata: speaker_count + diarizer descriptor written by the single finalise.
    let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
    assert_eq!(meta.speaker_count, 1, "single cluster → one labelled speaker");
    assert!(meta.diarizer.is_some(), "the diarizer descriptor must be recorded");
}

// ---------------------------------------------------------------------------
// 3. speaker_names cleared on every reprocess (accept-and-warn default)
// ---------------------------------------------------------------------------

/// A reprocess always diarizes, so a user-seeded `speaker_names` map is cleared
/// by the single `finalise_diarization` write — the accepted product default
/// (the durable fix is embedding-anchored retention, #0003).
#[tokio::test(flavor = "multi_thread")]
async fn reprocess_clears_speaker_names() {
    let _ = tracing_subscriber::fmt::try_init();
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let orch = test_orchestrator(root.clone());

    let samples = fixture_samples();
    let stale = vec![stale_seg(0, 1_000, "STALE")];
    // Seed a user-assigned name on the OLD letter scheme.
    let meeting_id = build_meeting(&root, &samples, &stale, &[("A", "Alice")]);
    let meeting_dir = root.join(meeting_id.0.to_string());

    // Sanity: the seed is present before reprocess.
    let before = persistence::read_metadata(&meeting_dir).expect("read metadata before");
    assert_eq!(before.speaker_names.get("A").map(String::as_str), Some("Alice"));

    let index = MeetingIndex::open(":memory:").await.expect("open index");
    index.rebuild_from_disk(&root).await.expect("seed index");

    // Single cluster → no split; the reprocess still diarizes, so it clears names.
    orch.reprocess_with_inputs(
        &index,
        meeting_id,
        Box::new(DelayingStubBackend {
            text: "fresh".into(),
            delay_ms: 0,
        }),
        single_cluster_turns(&samples),
        None,
        label_only_config(),
    )
    .await
    .expect("reprocess must succeed");

    let after = persistence::read_metadata(&meeting_dir).expect("read metadata after");
    assert!(
        after.speaker_names.is_empty(),
        "reprocess must clear speaker_names (re-lettering invalidates the old map); got {:?}",
        after.speaker_names
    );
}
