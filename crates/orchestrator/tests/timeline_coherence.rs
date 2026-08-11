//! Timeline-coherence regression tests for the TIMELINE-DRIFT cluster.
//!
//! These exercise the pause-EXCLUDING-timeline contract end to end
//! (`architecture/cross-cutting.md` — "Notes paragraph-anchor clock"):
//!
//! - **Coherence (#1 + #3 + #4):** a meeting recorded WITH a pause produces live
//!   `Segment::start_ms` on the pause-EXCLUDING capture clock. Re-transcribing
//!   the same meeting (decoding the pause-INCLUDING `audio.opus`) must produce
//!   timestamps on that SAME timeline — i.e. the offline pre-skip trim (#1), VAD
//!   re-alignment (#3), and pause-exclusion (#4) together keep the offline
//!   transcript anchored where the live one was.
//! - **Offline-op claim race (#5):** two concurrent `re_transcribe` calls — only
//!   one runs; the other is rejected with `AppError::InvalidInput`.
//! - **Stop-from-Paused flush (#6):** stopping while paused still flushes the
//!   end-of-stream VAD segment, so the last utterance is not lost.
//!
//! All tests run in the DEFAULT suite (no model env vars): the live + offline
//! paths use the real Silero VAD over the committed LibriSpeech fixture with a
//! `StubAsrBackend`, whose returned `Segment::start_ms` equals the VAD chunk
//! start — so the timeline math is asserted directly, model-free.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use audio_capture::{AudioFrameBatch, AudioStreams};
use minutist_common::{AppError, AudioFormat, MeetingId, MeetingMeta, Segment};
use orchestrator::test_support::{test_orchestrator, StubAsrBackend};
use persistence::{MeetingIndex, MeetingWriter};
use tokio::sync::mpsc;

const SAMPLE_RATE: u64 = 16_000;
/// 100 ms batch, matching the live capture forwarder's granularity.
const BATCH_SAMPLES: usize = 1600;

fn load_fixture_wav() -> Vec<f32> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/librispeech_0.wav");
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

fn live_streams() -> (
    mpsc::Sender<AudioFrameBatch>,
    mpsc::Sender<minutist_common::AudioMeterFrame>,
    AudioStreams,
) {
    let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(256);
    let (meter_tx, meter_rx) = mpsc::channel(64);
    (
        sample_tx,
        meter_tx,
        AudioStreams {
            samples: sample_rx,
            meter: meter_rx,
        },
    )
}

/// Feed `samples` to the runner as 100 ms batches whose `start_ms` runs on the
/// supplied pause-EXCLUDING clock beginning at `clock_ms`. Returns the clock
/// value one past the last fed sample (so the caller can continue contiguously
/// after a pause).
async fn feed_pause_excluding(
    tx: &mpsc::Sender<AudioFrameBatch>,
    samples: &[f32],
    mut clock_ms: u64,
) -> u64 {
    for chunk in samples.chunks(BATCH_SAMPLES) {
        let dur_ms = (chunk.len() as u64 * 1000) / SAMPLE_RATE;
        let batch = AudioFrameBatch {
            samples: chunk.to_vec(),
            start_ms: clock_ms,
            end_ms: clock_ms + dur_ms,
        };
        tx.send(batch).await.expect("send batch");
        clock_ms += dur_ms;
        // Pace lightly so the runner drains as it would live.
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    clock_ms
}

// ---------------------------------------------------------------------------
// Coherence: live pause-EXCLUDING timestamps == re-transcribed timestamps
// (TIMELINE-DRIFT #1 + #3 + #4)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn paused_meeting_retranscribe_matches_live_pause_excluding_timeline() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    // Two distinct speech blocks: the fixture, and the fixture repeated, so each
    // block reliably yields VAD segments. The PAUSE between them is real
    // wall-clock so the encoder pads silence into audio.opus (pause-INCLUDING),
    // while the fed capture clock stays pause-EXCLUDING (contiguous start_ms).
    let clip = load_fixture_wav();
    let block_a = clip.clone();
    let block_b: Vec<f32> = clip.iter().chain(clip.iter()).copied().collect();

    // The pause must exceed the offline pause-detection floor (PAUSE_MIN_MS =
    // 4 s) so the offline feeder recognises and excludes it. 4.5 s gives margin.
    const PAUSE_WALL_MS: u64 = 4_500;

    // ----- LIVE PASS -----
    let orch = test_orchestrator(root.clone());
    let (sample_tx, _meter_tx, streams) = live_streams();

    let stub = Box::new(StubAsrBackend::new("coherence stub text"));
    let meeting_id = orch
        .start_with_streams_and_backend(streams, stub, None)
        .await
        .expect("start_with_streams_and_backend");

    // Pre-pause block at clock 0.
    let clock_after_a = feed_pause_excluding(&sample_tx, &block_a, 0).await;

    // Pause → wall-clock elapses (encoder pads ~PAUSE_WALL_MS of silence) →
    // resume. The capture clock does NOT advance during the pause.
    orch.pause().await.expect("pause");
    tokio::time::sleep(Duration::from_millis(PAUSE_WALL_MS)).await;
    orch.resume().await.expect("resume");

    // Post-pause block continues CONTIGUOUSLY on the pause-excluding clock.
    let _ = feed_pause_excluding(&sample_tx, &block_b, clock_after_a).await;

    drop(sample_tx);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _meta = orch.stop().await.expect("stop");

    // The live transcript timestamps are the ground truth (pause-EXCLUDING).
    let meeting_dir = root.join(meeting_id.0.to_string());
    let live: Vec<Segment> = persistence::read_transcript(&meeting_dir).expect("read live transcript");
    assert!(
        live.len() >= 2,
        "live recording must yield ≥2 segments (one per speech block); got {}",
        live.len()
    );
    // A post-pause segment must exist on the pause-excluding clock (i.e. its
    // start is at/after the pre-pause block duration, NOT inflated by the pause).
    let block_a_ms = (block_a.len() as u64 * 1000) / SAMPLE_RATE;
    let live_post_pause = live
        .iter()
        .find(|s| s.start_ms >= block_a_ms)
        .expect("a post-pause live segment must exist");
    // It must be well BELOW the pause-INCLUDING position (block_a + pause).
    assert!(
        live_post_pause.start_ms < block_a_ms + PAUSE_WALL_MS,
        "live post-pause start {} must be on the pause-EXCLUDING clock (< {} + {})",
        live_post_pause.start_ms,
        block_a_ms,
        PAUSE_WALL_MS
    );

    // ----- OFFLINE RE-TRANSCRIBE PASS -----
    let index = MeetingIndex::open(":memory:").await.expect("open index");
    index.rebuild_from_disk(&root).await.expect("seed index");

    let stub2 = Box::new(StubAsrBackend::new("coherence stub text"));
    orch.re_transcribe_with_backend(&index, meeting_id, stub2)
        .await
        .expect("re_transcribe_with_backend");

    let offline: Vec<Segment> =
        persistence::read_transcript(&meeting_dir).expect("read offline transcript");
    assert!(
        !offline.is_empty(),
        "re_transcribe must produce segments"
    );

    // ----- COHERENCE ASSERTION (the #4 contract: pause-EXCLUDING timeline) -----
    // We deliberately do NOT require per-segment correspondence between the two
    // passes, for TWO independent reasons:
    //
    //  1. Streaming-vs-batch VAD: the live path accumulates contiguous
    //     utterances differently from the offline VAD re-run over decoded
    //     regions, so segment COUNTS and split points legitimately differ.
    //  2. Pre-pause trailing silence: the live capture clock counted block_a's
    //     trailing silence (it was captured BEFORE `pause()` was pressed), so
    //     live places block_b's onset at ~block_a_ms. The offline pause detector
    //     CANNOT tell that trailing silence from the synthesised pause padding —
    //     it skips the whole near-silent run — so it places block_b's onset a
    //     few hundred ms EARLIER, just below block_a_ms. That offset is inherent
    //     to offline reconstruction (the pause boundary is not persisted), not a
    //     bug, so the per-segment `live ≈ offline` correspondence the earlier
    //     version asserted was wrong and made this test flaky.
    //
    // What #4 is actually about is the TOTAL TIMELINE SPAN: a pause-INCLUDING
    // offline decode would run the VAD over the full pause-INCLUDING buffer and
    // shift the whole post-pause tail LATER by ~PAUSE_WALL_MS. The robust
    // discriminator is therefore the span and the post-pause placement, not a
    // single segment's start. TOL covers Opus lossy-decode boundary jitter +
    // frame quantisation.
    const TOL_MS: u64 = 300;
    let block_b_ms = (block_b.len() as u64 * 1000) / SAMPLE_RATE;

    // (a) Pre-pause: both passes place block_a's segment in the pre-pause region
    // (< block_a_ms). Their exact starts are intentionally NOT compared: as the
    // block comment above explains, the live clock counts block_a's trailing
    // silence while the offline pause detector skips it, so the two first-segment
    // starts diverge by a few hundred ms BY DESIGN (not a bug). Asserting
    // per-segment start correspondence here is precisely what made this test
    // flaky, so this is an existence-only check; the pause-EXCLUDING contract is
    // enforced by the span/placement assertions (b) and (c) below.
    let live_first = live.first().unwrap().start_ms;
    let offline_first = offline.first().unwrap().start_ms;
    assert!(
        live_first < block_a_ms && offline_first < block_a_ms,
        "both passes must have a pre-pause segment (< {block_a_ms} ms); \
         live_first={live_first} offline_first={offline_first}"
    );

    // (b) #4 core — the offline timeline stays on the pause-EXCLUDING clock. The
    // total pause-EXCLUDING audio is at most block_a + block_b; a pause-INCLUDING
    // decode would push the tail past block_a + PAUSE_WALL_MS + block_b. Assert
    // the LAST offline segment ENDS before the pause-EXCLUDING ceiling (with
    // tolerance) — a #4 regression inflates this by ~PAUSE_WALL_MS and fails.
    let excl_ceiling_ms = block_a_ms + block_b_ms;
    let offline_last_end = offline.iter().map(|s| s.end_ms).max().unwrap();
    assert!(
        offline_last_end <= excl_ceiling_ms + TOL_MS,
        "offline timeline left the pause-EXCLUDING clock: last segment ends at {offline_last_end} ms, \
         past the pause-EXCLUDING ceiling {excl_ceiling_ms} ms (a pause-INCLUDING decode would push it \
         toward ~{} ms); live={:?} offline={:?}",
        excl_ceiling_ms + PAUSE_WALL_MS,
        live.iter().map(|s| (s.start_ms, s.end_ms)).collect::<Vec<_>>(),
        offline.iter().map(|s| (s.start_ms, s.end_ms)).collect::<Vec<_>>(),
    );

    // (c) Post-pause content WAS transcribed and sits on the pause-EXCLUDING
    // clock — an offline segment must start after block_a's utterance but BELOW
    // where a pause-INCLUDING decode would place block_b's onset
    // (block_a_ms + PAUSE_WALL_MS). This also guards the region-boundary VAD
    // flush (runner::re_transcribe_buffer): without it the pre- and post-pause
    // utterances MERGE into one segment whose only post-block_a member sits past
    // this floor, failing here.
    let pause_including_floor = block_a_ms + PAUSE_WALL_MS;
    assert!(
        offline
            .iter()
            .any(|s| s.start_ms > offline_first + 1000 && s.start_ms < pause_including_floor),
        "offline must transcribe post-pause content on the pause-EXCLUDING clock (a segment starting \
         between {} and {pause_including_floor} ms); offline starts were {:?}",
        offline_first + 1000,
        offline.iter().map(|s| s.start_ms).collect::<Vec<_>>(),
    );
}

// ---------------------------------------------------------------------------
// #5: two concurrent re_transcribe — exactly one runs, the other InvalidInput
// ---------------------------------------------------------------------------

/// Build a meeting folder whose `audio.opus` encodes `samples`, with an empty
/// `transcript.json`. Returns the meeting id.
fn build_meeting_with_audio(root: &Path, samples: &[f32]) -> MeetingId {
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
        title: "Concurrency".into(),
        started_at: "2026-06-03T09:00:00Z".into(),
        ended_at: Some("2026-06-03T09:00:10Z".into()),
        duration_ms: (samples.len() as u64 * 1000) / SAMPLE_RATE,
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
        serde_json::to_vec_pretty(&Vec::<Segment>::new()).unwrap(),
    )
    .expect("empty transcript.json");
    meeting_id
}

#[tokio::test(flavor = "multi_thread")]
async fn two_concurrent_re_transcribe_only_one_runs() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    // Real speech so the VAD produces work and the offline op takes long enough
    // for the two calls to overlap.
    let clip = load_fixture_wav();
    let mut samples = Vec::new();
    for _ in 0..4 {
        samples.extend_from_slice(&clip);
    }
    let meeting_id = build_meeting_with_audio(&root, &samples);

    let orch = Arc::new(test_orchestrator(root.clone()));
    let index = Arc::new(MeetingIndex::open(":memory:").await.expect("open index"));
    index.rebuild_from_disk(&root).await.expect("seed");

    // Launch two re_transcribe calls concurrently on the SAME meeting. The
    // offline claim must let exactly one through; the other gets InvalidInput.
    let o1 = Arc::clone(&orch);
    let i1 = Arc::clone(&index);
    let o2 = Arc::clone(&orch);
    let i2 = Arc::clone(&index);

    let h1 = tokio::spawn(async move {
        let stub = Box::new(StubAsrBackend::new("op-1"));
        o1.re_transcribe_with_backend(&i1, meeting_id, stub).await
    });
    let h2 = tokio::spawn(async move {
        let stub = Box::new(StubAsrBackend::new("op-2"));
        o2.re_transcribe_with_backend(&i2, meeting_id, stub).await
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
        "exactly one concurrent re_transcribe must succeed; r1={r1:?} r2={r2:?}"
    );
    assert_eq!(
        invalid, 1,
        "the losing concurrent re_transcribe must be rejected with InvalidInput; r1={r1:?} r2={r2:?}"
    );

    // After both settle, the recorder is Idle again (the claim was released) —
    // a subsequent re_transcribe must succeed.
    let stub = Box::new(StubAsrBackend::new("op-3"));
    orch.re_transcribe_with_backend(&index, meeting_id, stub)
        .await
        .expect("recorder must be Idle again after the concurrent ops settle");
}

// ---------------------------------------------------------------------------
// #6: stop while Paused still flushes the end-of-stream VAD segment
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn stop_while_paused_flushes_last_utterance() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    let orch = test_orchestrator(root.clone());
    let (sample_tx, _meter_tx, streams) = live_streams();

    let stub = Box::new(StubAsrBackend::new("paused-stop utterance"));
    let meeting_id = orch
        .start_with_streams_and_backend(streams, stub, None)
        .await
        .expect("start");

    // Feed a single block of real speech. With hangover not yet exhausted at the
    // feed end, the VAD segment is still IN PROGRESS — only the end-of-stream
    // flush closes it. We then pause and stop WITHOUT resuming: the stop must
    // still run the VAD flush + accumulator drain so the utterance is captured.
    let clip = load_fixture_wav();
    let _ = feed_pause_excluding(&sample_tx, &clip, 0).await;

    // Pause (do NOT drop the sender — we want to stop FROM the paused state).
    orch.pause().await.expect("pause");
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Stop while paused.
    drop(sample_tx);
    let _meta = orch.stop().await.expect("stop while paused");

    // The last utterance must be in transcript.json — the paused-stop path ran
    // the end-of-stream flush (TIMELINE-DRIFT #6). Before the fix this was lost.
    let meeting_dir = root.join(meeting_id.0.to_string());
    let transcript: Vec<Segment> =
        persistence::read_transcript(&meeting_dir).expect("read transcript");
    assert!(
        transcript.iter().any(|s| s.text.contains("paused-stop")),
        "stop-from-Paused must flush the in-progress VAD segment; transcript was {transcript:?}"
    );
}
