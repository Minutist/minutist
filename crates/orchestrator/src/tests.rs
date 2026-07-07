//! Unit tests for `orchestrator`.
//!
//! All tests use `DummyAudioSource` + `start_with_streams` to avoid needing a
//! real microphone. Tests run on the tokio multi-thread scheduler via the
//! `#[tokio::test]` macro.

use std::time::Duration;

use audio_capture::test_source::DummyAudioSource;
use minutist_common::{AppError, AppEvent, RecordingState};
use tempfile::TempDir;

use crate::test_support::test_orchestrator;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a tempdir-backed orchestrator.
fn make_orchestrator() -> (crate::Orchestrator, TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());
    (orch, dir)
}

/// Return a `DummyAudioSource` that generates enough audio to trigger meter
/// events and produce some sample data.
///
/// 1 batch × 1600 speech samples + 800 silence = ~150 ms of audio at 16 kHz.
fn dummy_source() -> DummyAudioSource {
    DummyAudioSource::new(1600, 800)
}

// ---------------------------------------------------------------------------
// Test 1: State machine transitions emit expected StateChanged events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn state_machine_happy_path_emits_state_changed_events() {
    let _ = tracing_subscriber::fmt::try_init();
    let (orch, _dir) = make_orchestrator();
    let mut rx = orch.subscribe_events();

    // --- Idle ---
    assert_eq!(orch.state().await, RecordingState::Idle);

    // --- start → Recording ---
    let source = dummy_source();
    let streams = source.generate_streams(4, 32, 64);
    let meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    let ev = rx.try_recv().expect("StateChanged(Recording) expected");
    match ev {
        AppEvent::StateChanged {
            state: RecordingState::Recording {
                meeting_id: mid, ..
            },
        } => assert_eq!(mid, meeting_id),
        other => panic!("expected StateChanged(Recording), got {other:?}"),
    }
    match orch.state().await {
        RecordingState::Recording {
            meeting_id: mid, ..
        } => assert_eq!(mid, meeting_id),
        s => panic!("expected Recording state, got {s:?}"),
    }

    // --- pause → Paused ---
    orch.pause().await.expect("pause");
    let ev = rx.try_recv().expect("StateChanged(Paused) expected");
    match ev {
        AppEvent::StateChanged {
            state: RecordingState::Paused {
                meeting_id: mid, ..
            },
        } => assert_eq!(mid, meeting_id),
        other => panic!("expected StateChanged(Paused), got {other:?}"),
    }

    // --- resume → Recording ---
    orch.resume().await.expect("resume");
    let ev = rx.try_recv().expect("StateChanged(Recording) expected");
    match ev {
        AppEvent::StateChanged {
            state: RecordingState::Recording { .. },
        } => {}
        other => panic!("expected StateChanged(Recording), got {other:?}"),
    }

    // --- stop → Stopping → Finalising → Idle ---
    // `stop()` emits StateChanged(Stopping) before awaiting the runner, then
    // StateChanged(Finalising) while the runner finalises on its thread, then
    // StateChanged(Idle) + MeetingFinalised once finalise completes. All are
    // broadcast synchronously from within the stop() future, so by the time
    // stop() returns they're already in the channel.
    let meta = orch.stop().await.expect("stop");

    // Drain pending events to find the Stopping → Finalising → Idle sequence.
    let mut saw_stopping = false;
    let mut saw_finalising = false;
    let mut saw_idle = false;

    // Use recv() with a short timeout to collect all queued events.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
    loop {
        if std::time::Instant::now() > deadline {
            break;
        }
        match rx.try_recv() {
            Ok(AppEvent::StateChanged {
                state: RecordingState::Stopping { .. },
            }) => {
                saw_stopping = true;
            }
            Ok(AppEvent::StateChanged {
                state: RecordingState::Finalising { .. },
            }) => {
                saw_finalising = true;
            }
            Ok(AppEvent::StateChanged {
                state: RecordingState::Idle,
            }) => {
                saw_idle = true;
            }
            Ok(AppEvent::MeetingFinalised { .. }) => {}
            Ok(AppEvent::AudioMeter { .. }) => {}
            // The finalise drain emits an indeterminate OperationProgress (T4(c));
            // it rides the same bus and is not part of the state sequence.
            Ok(AppEvent::OperationProgress { .. }) => {}
            Ok(other) => panic!("unexpected event: {other:?}"),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                // Brief yield to allow any in-flight events to arrive.
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!("test receiver lagged by {n}");
            }
        }
        if saw_stopping && saw_finalising && saw_idle {
            break;
        }
    }

    assert!(saw_stopping, "expected StateChanged(Stopping)");
    assert!(saw_finalising, "expected StateChanged(Finalising)");
    assert!(saw_idle, "expected StateChanged(Idle)");
    assert_eq!(orch.state().await, RecordingState::Idle);
    assert_eq!(meta.uuid, meeting_id);
    assert_eq!(meta.speaker_count, 0);
}

/// `diarize_timeout` / `retranscribe_timeout` / `relisten_timeout` clamp their
/// length-relative budget at the documented floors/caps (sub-floor → floor,
/// proportional mid-range, supra-cap → cap). re-transcribe + relisten run ~3×
/// real-time vs diarize's ~1×; relisten is bounded by the WINDOW length (S2).
#[test]
fn timeout_helpers_clamp_to_documented_bounds() {
    // diarize: ~1× real-time, floor 120 s, cap 600 s.
    assert_eq!(crate::diarize_timeout(0), Duration::from_secs(120));
    assert_eq!(crate::diarize_timeout(300_000), Duration::from_secs(300));
    assert_eq!(crate::diarize_timeout(3_600_000), Duration::from_secs(600));
    // re-transcribe: ~3× real-time, floor 300 s, cap 1800 s.
    assert_eq!(crate::retranscribe_timeout(0), Duration::from_secs(300));
    assert_eq!(crate::retranscribe_timeout(300_000), Duration::from_secs(900));
    assert_eq!(crate::retranscribe_timeout(3_600_000), Duration::from_secs(1800));
    // relisten (S2): ~3× the WINDOW length, floor 60 s, cap 300 s.
    assert_eq!(crate::relisten_timeout(0), Duration::from_secs(60)); // sub-floor → floor
    assert_eq!(crate::relisten_timeout(30_000), Duration::from_secs(90)); // 30 s window × 3
    assert_eq!(crate::relisten_timeout(600_000), Duration::from_secs(300)); // supra-cap → cap
}

/// A clean recording (live ASR kept up, no drop-oldest, drained in time) is NOT
/// flagged incomplete through `stop()`, and `take_transcript_incomplete()`
/// read-resets. (The drop→incomplete detection itself is covered by the
/// runner-level `dispatch_flush_drops_oldest_when_queue_full` test.)
#[tokio::test]
async fn clean_recording_is_not_flagged_transcript_incomplete() {
    let _ = tracing_subscriber::fmt::try_init();
    let (orch, _dir) = make_orchestrator();
    let source = DummyAudioSource::new(3200, 1600);
    let streams = source.generate_streams(5, 32, 64);
    orch.start_with_streams(streams).await.expect("start");
    tokio::time::sleep(Duration::from_millis(100)).await;
    orch.stop().await.expect("stop");
    assert!(
        !orch.take_transcript_incomplete(),
        "a clean recording must not be flagged transcript-incomplete"
    );
    assert!(
        !orch.take_transcript_incomplete(),
        "take_transcript_incomplete must read-reset (still false)"
    );
}

/// B1 — a failure inside the `stop()` finalise handshake must NOT wedge the
/// orchestrator in `Stopping`/`Finalising`. Fault-inject by removing the
/// meeting folder out from under the writer right after `start`: the audio
/// encoder's already-open fd keeps accepting writes (an unlink does not
/// invalidate an open fd on Linux), so recording proceeds normally, but
/// `MeetingWriter::finalise`'s `metadata.json` write opens a FRESH file in the
/// now-missing directory and fails with a real I/O error — the same failure
/// shape as a disk unmount or permission change mid-recording.
///
/// Before the B1 fix this error propagated out of `stop()` via a bare `?`
/// that skipped the `transition_idle` + `StateChanged(Idle)` steps, leaving
/// the recorder stuck reporting `Finalising` forever (no further recording
/// possible without a process restart). This test starts a SECOND recording
/// immediately after the failed `stop()` to prove that symptom is gone.
#[tokio::test]
async fn stop_failure_still_returns_to_idle_and_emits_state_changed() {
    let _ = tracing_subscriber::fmt::try_init();
    let (orch, dir) = make_orchestrator();
    let mut rx = orch.subscribe_events();

    let source = dummy_source();
    let streams = source.generate_streams(4, 32, 64);
    let meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // `MeetingWriter::open` (awaited inside `start_with_streams`) has already
    // created the folder by the time `start_with_streams` returns.
    let meeting_dir = dir.path().join(meeting_id.0.to_string());
    std::fs::remove_dir_all(&meeting_dir).expect("remove meeting dir to fault-inject");

    let err = orch
        .stop()
        .await
        .expect_err("finalise must fail once its metadata write has no directory to write into");
    assert!(
        matches!(err, AppError::Io { .. }),
        "expected the metadata write's I/O error to surface from stop(), got {err:?}"
    );

    // Not wedged: the internal state machine is back to Idle...
    assert_eq!(orch.state().await, RecordingState::Idle);

    // ...and a StateChanged(Idle) was broadcast (not just the internal field —
    // this is what lets a listening UI un-stick its busy indicator).
    let mut saw_idle = false;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    while std::time::Instant::now() < deadline && !saw_idle {
        match rx.try_recv() {
            Ok(AppEvent::StateChanged {
                state: RecordingState::Idle,
            }) => saw_idle = true,
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }
    assert!(
        saw_idle,
        "expected a StateChanged(Idle) broadcast after the finalise failure"
    );

    // The actual "wedged" symptom: recording must be possible again right
    // away, with no process restart.
    let source2 = dummy_source();
    let streams2 = source2.generate_streams(4, 32, 64);
    orch.start_with_streams(streams2)
        .await
        .expect("a new recording must be startable immediately after a failed finalise");
}

// ---------------------------------------------------------------------------
// Test 2: Invalid transitions return AppError::InvalidInput
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invalid_transitions_return_invalid_input_error() {
    let _ = tracing_subscriber::fmt::try_init();
    let (orch, _dir) = make_orchestrator();

    // pause() from Idle
    let err = orch.pause().await.expect_err("pause from Idle should fail");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );

    // resume() from Idle
    let err = orch
        .resume()
        .await
        .expect_err("resume from Idle should fail");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );

    // stop() from Idle
    let err = orch.stop().await.expect_err("stop from Idle should fail");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );

    // Start a session.
    let source = dummy_source();
    let streams = source.generate_streams(2, 32, 64);
    orch.start_with_streams(streams).await.expect("start");

    // start() again while Recording
    let source2 = dummy_source();
    let streams2 = source2.generate_streams(2, 32, 64);
    let err = orch
        .start_with_streams(streams2)
        .await
        .expect_err("start while Recording should fail");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );

    // resume() from Recording (not Paused)
    let err = orch
        .resume()
        .await
        .expect_err("resume from Recording should fail");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected InvalidInput, got {err:?}"
    );

    // Clean up.
    orch.stop().await.expect("stop");
}

// ---------------------------------------------------------------------------
// Test 3: stop() produces valid meeting folder with expected files + metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_produces_valid_meeting_folder_with_expected_files() {
    let _ = tracing_subscriber::fmt::try_init();
    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());

    // Generate a few batches of audio so there's actual data to encode.
    let source = DummyAudioSource::new(3200, 1600); // ~200 ms speech + 100 ms silence
    let streams = source.generate_streams(5, 32, 64);

    let meeting_id = orch.start_with_streams(streams).await.expect("start");

    // Let the runner process samples before stopping.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let meta = orch.stop().await.expect("stop");

    // 1. meeting folder exists.
    let meeting_dir = dir.path().join(meeting_id.0.to_string());
    assert!(
        meeting_dir.exists(),
        "meeting directory should exist at {meeting_dir:?}"
    );

    // 2. audio.opus exists.
    let audio_path = meeting_dir.join("audio.opus");
    assert!(
        audio_path.exists(),
        "audio.opus should exist at {audio_path:?}"
    );
    assert!(
        std::fs::metadata(&audio_path).unwrap().len() > 0,
        "audio.opus should not be empty"
    );

    // 3. metadata.json exists and deserialises into MeetingMeta.
    let meta_path = meeting_dir.join("metadata.json");
    assert!(meta_path.exists(), "metadata.json should exist");
    let json = std::fs::read_to_string(&meta_path).expect("read metadata.json");
    let loaded: minutist_common::MeetingMeta =
        serde_json::from_str(&json).expect("deserialise metadata.json");

    // 4. Verify expected fields.
    assert_eq!(loaded.uuid, meeting_id);
    assert_eq!(loaded.speaker_count, 0);
    assert!(
        loaded.title.starts_with("Recording "),
        "title should start with 'Recording ', got {:?}",
        loaded.title
    );
    assert!(loaded.ended_at.is_some(), "ended_at should be populated");
    assert_eq!(loaded.audio_format.codec, "opus");
    assert_eq!(loaded.audio_format.sample_rate, 16_000);
    assert_eq!(loaded.audio_format.channels, 1);
    assert_eq!(loaded.audio_format.bitrate_kbps, Some(32));
    assert!(loaded.asr_model.is_none());
    assert!(loaded.llm_model.is_none());
    assert!(loaded.diarizer.is_none());

    // The returned MeetingMeta from stop() should match what's on disk.
    assert_eq!(meta.uuid, loaded.uuid);
    assert_eq!(meta.title, loaded.title);
}

// ---------------------------------------------------------------------------
// Test 4: Audio meter events are received after start
// ---------------------------------------------------------------------------

#[tokio::test]
async fn audio_meter_events_arrive_within_one_second() {
    let _ = tracing_subscriber::fmt::try_init();
    let (orch, _dir) = make_orchestrator();
    let mut rx = orch.subscribe_events();

    // Generate many batches so meter frames definitely get emitted.
    let source = DummyAudioSource::new(1600, 512);
    let streams = source.generate_streams(20, 64, 128);

    orch.start_with_streams(streams).await.expect("start");

    // Wait up to 1 second for at least one AudioMeter event.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
    let mut got_meter = false;

    while tokio::time::Instant::now() < deadline {
        match rx.try_recv() {
            Ok(AppEvent::AudioMeter { .. }) => {
                got_meter = true;
                break;
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(target: "orchestrator", lagged = n, "test subscriber lagged");
            }
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }

    orch.stop().await.expect("stop");

    assert!(
        got_meter,
        "expected at least one AudioMeter event within 1 second"
    );
}

// ---------------------------------------------------------------------------
// Phase 6 — diarization wiring (DEFAULT suite, no model)
//
// The model-free `rediarize_with_split_inputs` seam supplies synthetic
// `SpeakerTurn`s + a stub `AsrBackend` to drive the re-diarize + #0015-phase-4
// re-ASR split over a synthetic meeting folder, and a toggle-OFF `stop()` test
// proves the on-stop pass is gated. No sherpa model, no Qwen GGUF.
// ---------------------------------------------------------------------------

mod diarization {
    use super::*;
    use diarizer::{DiarizerConfig, SpeakerTurn};
    use minutist_common::{
        AppResult, AsrBackend, AudioChunk, AudioFormat, MeetingId, MeetingMeta, Segment,
    };
    use persistence::{MeetingIndex, MeetingWriter};

    /// A model-free [`AsrBackend`]: returns one segment per chunk whose text
    /// encodes the chunk's INCLUDING-clock window, so a split test can assert each
    /// sub-clip was re-ASR'd over the expected audio slice. Records every chunk it
    /// was handed so a test can count the re-ASR passes.
    struct RecordingStubBackend {
        chunks: std::sync::Arc<std::sync::Mutex<Vec<(u64, u64)>>>,
    }

    impl AsrBackend for RecordingStubBackend {
        fn transcribe_chunk(&mut self, chunk: &AudioChunk) -> AppResult<Vec<Segment>> {
            self.chunks
                .lock()
                .unwrap()
                .push((chunk.start_ms, chunk.end_ms));
            Ok(vec![Segment {
                start_ms: chunk.start_ms,
                end_ms: chunk.end_ms,
                text: format!("reasr[{}-{}]", chunk.start_ms, chunk.end_ms),
                speaker_id: None,
                confidence: None,
                words: Vec::new(),
                shared_speakers: Vec::new(),
            }])
        }
    }

    /// A `DiarizerConfig` whose multi-speaker flag is the only post-processing
    /// active: no prune, no cap, flag a segment as mixed when a second cluster
    /// covers ≥ 30% of it. This makes `overlay_speakers` flag a two-cluster Qwen
    /// segment without folding away a small cluster the split needs.
    fn split_config() -> DiarizerConfig {
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

    /// Build a synthetic meeting folder on disk: `audio.opus` encoded from
    /// `samples` via the production `MeetingWriter`, a `metadata.json` with
    /// `speaker_count = 0` / `diarizer = None`, and a `transcript.json` of the
    /// supplied `segments`. Returns the id.
    fn build_meeting_with_segments(
        root: &std::path::Path,
        samples: &[f32],
        segments: &[Segment],
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

        let meta = MeetingMeta {
            uuid: meeting_id,
            title: "Diarize me".into(),
            started_at: "2026-06-02T09:00:00Z".into(),
            ended_at: Some("2026-06-02T09:00:06Z".into()),
            duration_ms: (samples.len() as u64 * 1000) / 16_000,
            speaker_count: 0,
            audio_format: format,
            asr_model: None,
            llm_model: None,
            diarizer: None,
            speaker_names: std::collections::BTreeMap::new(),
            notes_format: 0,
            processing: Default::default(),
            collection_id: None,
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

    /// A single ASR segment with no words (the Qwen shape) covering `[s, e)`.
    fn qwen_seg(s: u64, e: u64, text: &str) -> Segment {
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

    /// Two-speaker speech buffer with a low-energy gap at the boundary so the
    /// energy-snap finds a clear minimum: `[0, gap_ms)` loud, `[gap_ms-pad,
    /// gap_ms+pad)` near-silent, `[gap_ms, total_ms)` loud. No ≥4 s pause, so the
    /// whole buffer is one kept region (excluding clock == including clock).
    fn two_speaker_pcm(gap_ms: u64, total_ms: u64) -> Vec<f32> {
        let n = (total_ms as usize * 16_000) / 1000;
        let gap = (gap_ms as usize * 16_000) / 1000;
        let pad = (20 * 16_000) / 1000; // ±20 ms quiet around the boundary
        let mut pcm = vec![0.5f32; n];
        let lo = gap.saturating_sub(pad);
        let hi = (gap + pad).min(n);
        for s in pcm.iter_mut().take(hi).skip(lo) {
            *s = 0.0;
        }
        pcm
    }

    /// The re-diarize + split inner path (driven via the
    /// `rediarize_with_split_inputs` seam) overlays first-seen labels onto
    /// single-speaker Qwen segments, rewrites `transcript.json`, updates
    /// `metadata.json`'s `speaker_count` + `diarizer`, refreshes the index row,
    /// and emits `DiarizationComplete` — all without a sherpa model.
    #[tokio::test]
    async fn rediarize_with_stub_writes_speaker_ids_and_emits() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());
        let mut event_rx = orch.subscribe_events();

        // 3 s of loud PCM; three single-speaker Qwen segments, one per second.
        let samples = vec![0.5f32; 16_000 * 3];
        let segs = vec![
            qwen_seg(0, 900, "hello"),
            qwen_seg(1_000, 1_900, "there"),
            qwen_seg(2_000, 2_900, "again"),
        ];
        let meeting_id = build_meeting_with_segments(&root, &samples, &segs);
        let meeting_dir = root.join(meeting_id.0.to_string());

        // Turns: speaker 7 → seg0 + seg2, speaker 3 → seg1. Each segment overlaps
        // exactly one cluster, so none is mixed; first-seen order 7→A, 3→B.
        let turns = vec![
            SpeakerTurn { start_ms: 0, end_ms: 900, cluster: 7 },
            SpeakerTurn { start_ms: 1_000, end_ms: 1_900, cluster: 3 },
            SpeakerTurn { start_ms: 2_000, end_ms: 2_900, cluster: 7 },
        ];

        // Seed a user-set speaker_names map (mapping the OLD letters). A
        // re-diarization can re-letter speakers, so the map must be cleared by
        // the rediarize metadata write (Phase 9 §4.4).
        {
            let mut meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
            meta.speaker_names.insert("A".to_string(), "Alice".to_string());
            persistence::write_metadata(&meeting_dir, &meta).expect("seed speaker_names");
        }

        let index = MeetingIndex::open(":memory:").await.expect("open index");
        index.rebuild_from_disk(&root).await.expect("seed index");

        let chunks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let backend = Box::new(RecordingStubBackend { chunks: chunks.clone() });
        orch.rediarize_with_split_inputs(
            &index,
            meeting_id,
            turns,
            Some(backend),
            split_config(),
        )
        .await
        .expect("rediarize_with_split_inputs must succeed");

        // No mixed Qwen segment → the re-ASR backend is never invoked.
        assert!(
            chunks.lock().unwrap().is_empty(),
            "single-speaker segments must not be re-ASR'd"
        );

        // 1. transcript.json rewritten with first-seen speaker_ids (7→A, 3→B).
        let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
        assert_eq!(after.len(), 3);
        assert_eq!(after[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(after[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(after[2].speaker_id.as_deref(), Some("A"));

        // 2. metadata.json speaker_count updated + diarizer descriptor set, and
        //    the seeded speaker_names map cleared (§4.4 — re-lettering
        //    invalidates the old map).
        let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
        assert_eq!(meta.speaker_count, 2);
        assert!(meta.diarizer.is_some(), "diarizer descriptor must be recorded");
        assert!(
            meta.speaker_names.is_empty(),
            "rediarize must clear speaker_names (re-lettering invalidates the map)"
        );

        // 3. index row speaker_count refreshed.
        let rows = index.list_meetings().await.expect("list");
        let row = rows.iter().find(|r| r.id == meeting_id).expect("row");
        assert_eq!(row.speaker_count, 2);

        // 4. DiarizationComplete emitted with the right meeting_id + count.
        let mut got = false;
        loop {
            match event_rx.try_recv() {
                Ok(AppEvent::DiarizationComplete { meeting_id: mid, speaker_count }) => {
                    assert_eq!(mid, meeting_id);
                    assert_eq!(speaker_count, 2);
                    got = true;
                    break;
                }
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
                Err(_) => break,
            }
        }
        assert!(got, "DiarizationComplete must be emitted");
    }

    /// A mixed Qwen segment (two clusters, no words) is split at the
    /// speaker-change boundary: each single-speaker sub-clip is re-ASR'd, lettered
    /// from the cluster→letter map, carries an EXCLUDING-clock `start_ms`, and has
    /// empty `shared_speakers`.
    #[tokio::test]
    async fn rediarize_splits_mixed_qwen_segment_via_stub_backend() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());

        // 6 s buffer with a gap at 2 s (the mixed segment's interior cut). A mixed
        // Qwen segment over [0, 4000) plus a solo speaker-9 segment near the end,
        // so cluster 9 is a primary somewhere and earns letter B in the map.
        let samples = two_speaker_pcm(2_000, 6_000);
        let segs = vec![
            qwen_seg(0, 4_000, "alpha beta"),
            qwen_seg(5_600, 6_000, "tail"),
        ];
        let meeting_id = build_meeting_with_segments(&root, &samples, &segs);
        let meeting_dir = root.join(meeting_id.0.to_string());

        // Turns: cluster 5 on [0, 2000), cluster 9 on [2000, 6000). The mixed
        // segment overlaps both ≥ 30%, so `overlay_speakers` flags it mixed; the
        // interior cut is at 2000 ms. First-seen order 5→A, 9→B.
        let turns = vec![
            SpeakerTurn { start_ms: 0, end_ms: 2_000, cluster: 5 },
            SpeakerTurn { start_ms: 2_000, end_ms: 6_000, cluster: 9 },
        ];

        let index = MeetingIndex::open(":memory:").await.expect("open index");
        index.rebuild_from_disk(&root).await.expect("seed index");

        let chunks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let backend = Box::new(RecordingStubBackend { chunks: chunks.clone() });
        orch.rediarize_with_split_inputs(
            &index,
            meeting_id,
            turns,
            Some(backend),
            split_config(),
        )
        .await
        .expect("split must succeed");

        // The mixed segment splits into A then B; the trailing solo-B row stays
        // separate (gap > MERGE_GAP_MS) → three rows A, B, B.
        let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
        assert_eq!(after.len(), 3, "mixed segment splits; trailing solo-B is kept");
        assert_eq!(after[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(after[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(after[2].speaker_id.as_deref(), Some("B"));
        // The two split sub-clips have empty shared_speakers (no longer mixed) and
        // are re-ASR'd text.
        assert!(after[0].shared_speakers.is_empty());
        assert!(after[1].shared_speakers.is_empty());
        assert!(after[0].text.starts_with("reasr["));
        assert!(after[1].text.starts_with("reasr["));

        // Two re-ASR passes, one per split sub-clip (the solo-B row is untouched).
        assert_eq!(chunks.lock().unwrap().len(), 2);

        // EXCLUDING-clock start_ms: no ≥4 s pause, so excluding == including. The
        // first sub-clip starts at 0; the second at the snapped ~2000 ms boundary.
        assert_eq!(after[0].start_ms, 0);
        assert!(
            (1_800..=2_050).contains(&after[1].start_ms),
            "second sub-clip starts at the snapped boundary (~2000 ms; the Opus \
             round-trip shifts the energy valley earlier), got {}",
            after[1].start_ms
        );

        assert_eq!(meta_speaker_count(&meeting_dir), 2);
    }

    /// Keep-whole when the backend is `None` (no re-ASR model): the mixed Qwen
    /// segment stays one segment on its dominant label with `shared_speakers`
    /// retained — the documented no-regression degrade.
    #[tokio::test]
    async fn rediarize_keeps_mixed_segment_whole_when_backend_absent() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());

        // The mixed [0,4000) segment needs the secondary cluster 9 to be a
        // survivor, so add a trailing solo-9 segment (mirroring the split test);
        // otherwise the segment is never flagged mixed and never reaches the
        // keep-whole branch.
        let samples = two_speaker_pcm(2_000, 6_000);
        let segs = vec![qwen_seg(0, 4_000, "alpha beta"), qwen_seg(5_600, 6_000, "tail")];
        let meeting_id = build_meeting_with_segments(&root, &samples, &segs);
        let meeting_dir = root.join(meeting_id.0.to_string());

        let turns = vec![
            SpeakerTurn { start_ms: 0, end_ms: 2_000, cluster: 5 },
            SpeakerTurn { start_ms: 2_000, end_ms: 6_000, cluster: 9 },
        ];

        let index = MeetingIndex::open(":memory:").await.expect("open index");
        index.rebuild_from_disk(&root).await.expect("seed index");

        orch.rediarize_with_split_inputs(&index, meeting_id, turns, None, split_config())
            .await
            .expect("keep-whole must succeed");

        // backend None → the mixed [0,4000) is kept whole (dominant A + the B
        // shared flag); the trailing solo-B stays its own row → two rows.
        let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
        assert_eq!(after.len(), 2, "backend None → keep-whole, no split");
        assert_eq!(after[0].speaker_id.as_deref(), Some("A"));
        assert!(
            !after[0].shared_speakers.is_empty(),
            "the mixed flag is retained when kept whole"
        );
    }

    /// Keep-whole when the energy-snap finds no clear minimum: a constant-energy
    /// buffer at the boundary means no real gap, so the split is abandoned and the
    /// mixed segment stays whole even with a backend present.
    #[tokio::test]
    async fn rediarize_keeps_mixed_segment_whole_when_snap_finds_no_minimum() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());

        // Constant-energy buffer (no gap) → snap_to_energy_min returns None. The
        // trailing solo-9 segment makes cluster 9 a survivor so [0,4000) is
        // genuinely flagged mixed and reaches the split branch (where snap then
        // abandons it).
        let samples = vec![0.5f32; 16_000 * 6];
        let segs = vec![qwen_seg(0, 4_000, "alpha beta"), qwen_seg(5_600, 6_000, "tail")];
        let meeting_id = build_meeting_with_segments(&root, &samples, &segs);
        let meeting_dir = root.join(meeting_id.0.to_string());

        let turns = vec![
            SpeakerTurn { start_ms: 0, end_ms: 2_000, cluster: 5 },
            SpeakerTurn { start_ms: 2_000, end_ms: 6_000, cluster: 9 },
        ];

        let index = MeetingIndex::open(":memory:").await.expect("open index");
        index.rebuild_from_disk(&root).await.expect("seed index");

        let chunks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let backend = Box::new(RecordingStubBackend { chunks: chunks.clone() });
        orch.rediarize_with_split_inputs(
            &index,
            meeting_id,
            turns,
            Some(backend),
            split_config(),
        )
        .await
        .expect("keep-whole must succeed");

        // snap None at the interior cut abandons the split BEFORE any re-ASR; the
        // mixed [0,4000) stays whole + flagged, the trailing solo-B is its own row.
        let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
        assert_eq!(after.len(), 2, "no clear energy minimum → keep-whole");
        assert!(
            chunks.lock().unwrap().is_empty(),
            "an abandoned split must not re-ASR anything"
        );
        assert!(!after[0].shared_speakers.is_empty());
    }

    /// The post-split re-merge collapses an adjacent SAME-speaker pair but never
    /// bridges the NEW speaker-change boundary the split introduced.
    ///
    /// Layout: a leading A segment immediately before a mixed A/B Qwen segment.
    /// The split turns the mixed segment into A then B; the re-merge folds the
    /// leading A into the split's leading A (same label, sub-`MERGE_GAP_MS` gap)
    /// but leaves the A|B boundary intact → exactly two rows, A then B.
    #[tokio::test]
    async fn merge_after_split_does_not_bridge_the_new_boundary() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());

        // 8 s buffer: a clear gap at 4 s (the mixed segment's interior cut). The
        // 0-2 s leading-A region and the 2-4 s first half of the mixed segment are
        // the same speaker, so they re-merge. A trailing solo-9 segment makes
        // cluster 9 a primary so it earns letter B in the cluster→letter map.
        let samples = two_speaker_pcm(4_000, 8_000);
        let segs = vec![
            // Leading single-speaker A segment [0, 2000).
            qwen_seg(0, 2_000, "intro"),
            // Mixed A/B Qwen segment [2000, 6000), cut at 4000.
            qwen_seg(2_000, 6_000, "alpha beta"),
            // Trailing solo-B segment, far enough not to re-merge.
            qwen_seg(7_600, 8_000, "tail"),
        ];
        let meeting_id = build_meeting_with_segments(&root, &samples, &segs);
        let meeting_dir = root.join(meeting_id.0.to_string());

        // Cluster 5 (→A) spans [0, 4000); cluster 9 (→B) spans [4000, 8000).
        let turns = vec![
            SpeakerTurn { start_ms: 0, end_ms: 4_000, cluster: 5 },
            SpeakerTurn { start_ms: 4_000, end_ms: 8_000, cluster: 9 },
        ];

        let index = MeetingIndex::open(":memory:").await.expect("open index");
        index.rebuild_from_disk(&root).await.expect("seed index");

        let chunks = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let backend = Box::new(RecordingStubBackend { chunks: chunks.clone() });
        orch.rediarize_with_split_inputs(
            &index,
            meeting_id,
            turns,
            Some(backend),
            split_config(),
        )
        .await
        .expect("split + merge must succeed");

        let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
        // Rows: A (leading A folded into the split's leading A) | B (split's
        // second half) | B (trailing solo, gap > MERGE_GAP_MS so not folded). The
        // A|B boundary the split introduced is never bridged.
        assert_eq!(after.len(), 3);
        assert_eq!(after[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(after[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(after[2].speaker_id.as_deref(), Some("B"));
        // The leading A row spans [0, ~4000): the leading-A segment merged with
        // the split's leading-A sub-clip (no boundary between them).
        assert_eq!(after[0].start_ms, 0);
        assert!(
            after[0].end_ms >= 3_700,
            "leading A must extend through the split's first sub-clip (~4000 ms; \
             Opus smear pulls the snapped boundary earlier), got {}",
            after[0].end_ms
        );
    }

    /// Read `metadata.json`'s `speaker_count`.
    fn meta_speaker_count(meeting_dir: &std::path::Path) -> u32 {
        persistence::read_metadata(meeting_dir)
            .expect("read metadata")
            .speaker_count
    }

    /// With `diarization_enabled = true`, `stop()` is now DECOUPLED from
    /// diarization: it returns the meeting un-diarized (`speaker_count` 0,
    /// `diarizer` None), does NOT rewrite the transcript, and emits no
    /// `DiarizationComplete`. The on-stop pass is run in the BACKGROUND by
    /// `ipc-bridge` (via [`Orchestrator::rediarize`]) AFTER the meeting is
    /// indexed, so a slow or hung diarization can never wedge `stop()` or hide
    /// the meeting (the original failure mode). `stop()` only surfaces the
    /// toggle via [`Orchestrator::diarization_enabled`] for ipc to act on; the
    /// actual diarize + persist + index path is covered by the `rediarize`
    /// tests.
    #[tokio::test]
    async fn stop_with_diarization_enabled_is_decoupled_from_stop() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());
        let mut event_rx = orch.subscribe_events();

        orch.settings_handle_for_test()
            .update(|s| s.diarization_enabled = true)
            .await
            .expect("enable diarization_enabled");

        let source = DummyAudioSource::new(3200, 1600);
        let streams = source.generate_streams(5, 32, 64);
        let meeting_id = orch.start_with_streams(streams).await.expect("start");

        // Seed unlabelled segments; stop() must leave them untouched.
        let meeting_dir = root.join(meeting_id.0.to_string());
        let seeded: Vec<Segment> = ["alpha", "beta", "gamma"]
            .iter()
            .enumerate()
            .map(|(i, t)| Segment {
                start_ms: i as u64 * 1000,
                end_ms: i as u64 * 1000 + 800,
                text: (*t).to_string(),
                speaker_id: None,
                confidence: None,
                words: Vec::new(),
                shared_speakers: Vec::new(),
            })
            .collect();
        persistence::write_transcript(&meeting_dir, &seeded).expect("seed transcript");

        tokio::time::sleep(Duration::from_millis(100)).await;
        let meta = orch.stop().await.expect("stop");

        // The toggle is surfaced for ipc, but stop() itself does not diarize.
        assert!(orch.diarization_enabled(), "accessor reflects the enabled toggle");
        assert_eq!(
            meta.speaker_count, 0,
            "stop() returns un-diarized regardless of the toggle (diarization is decoupled)"
        );
        assert!(meta.diarizer.is_none(), "stop() does not set the diarizer");

        // Transcript untouched (still unlabelled) and no DiarizationComplete.
        let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
        assert!(
            after.iter().all(|s| s.speaker_id.is_none()),
            "stop() must not rewrite the transcript with speaker ids"
        );
        let mut saw_diarization = false;
        while let Ok(ev) = event_rx.try_recv() {
            if matches!(ev, AppEvent::DiarizationComplete { .. }) {
                saw_diarization = true;
            }
        }
        assert!(
            !saw_diarization,
            "stop() must not emit DiarizationComplete (the background pass does)"
        );
    }

    /// A title typed during the live recording (via `set_pending_title`) is
    /// captured and consumed by `stop()` in place of the `Recording <timestamp>`
    /// default — trimmed — and a set against a non-live id is a no-op.
    #[tokio::test]
    async fn stop_uses_the_live_pending_title_when_set() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());

        let source = DummyAudioSource::new(3200, 1600);
        let streams = source.generate_streams(5, 32, 64);
        let meeting_id = orch.start_with_streams(streams).await.expect("start");

        // A stale id must NOT clobber the live title.
        orch.set_pending_title(MeetingId::new(), "Wrong".into())
            .await
            .expect("stale id is a no-op");
        // The live meeting's title is captured (and trimmed).
        orch.set_pending_title(meeting_id, "  Quarterly planning  ".into())
            .await
            .expect("set live title");

        tokio::time::sleep(Duration::from_millis(100)).await;
        let meta = orch.stop().await.expect("stop");
        assert_eq!(
            meta.title, "Quarterly planning",
            "stop() must use the trimmed live title"
        );
    }

    /// An unnamed recording falls back to the synthesized `Recording <timestamp>`
    /// default title.
    #[tokio::test]
    async fn stop_falls_back_to_default_title_when_unnamed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());

        let source = DummyAudioSource::new(3200, 1600);
        let streams = source.generate_streams(5, 32, 64);
        orch.start_with_streams(streams).await.expect("start");

        tokio::time::sleep(Duration::from_millis(100)).await;
        let meta = orch.stop().await.expect("stop");
        assert!(
            meta.title.starts_with("Recording "),
            "unnamed recording keeps the default title, got {:?}",
            meta.title
        );
    }

    /// With `diarization_enabled = false`, `stop()` runs no diarization pass:
    /// the returned `MeetingMeta` and the on-disk transcript keep every
    /// `speaker_id == None`, `speaker_count == 0`, and NO `DiarizationComplete`
    /// event is emitted. (Diarization is on by default since 2026-06-25, so this
    /// OFF-path test disables it explicitly.)
    #[tokio::test]
    async fn stop_with_diarization_disabled_leaves_segments_unlabelled() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let orch = test_orchestrator(dir.path().to_path_buf());
        orch.settings_handle_for_test()
            .update(|s| s.diarization_enabled = false)
            .await
            .expect("disable diarization_enabled");
        let mut event_rx = orch.subscribe_events();

        let source = DummyAudioSource::new(3200, 1600);
        let streams = source.generate_streams(5, 32, 64);
        let meeting_id = orch.start_with_streams(streams).await.expect("start");
        tokio::time::sleep(Duration::from_millis(100)).await;

        let meta = orch.stop().await.expect("stop");

        // Returned meta carries no speaker labels.
        assert_eq!(meta.speaker_count, 0, "toggle-OFF stop must not diarize");
        assert!(meta.diarizer.is_none(), "toggle-OFF stop must not set diarizer");

        // On-disk transcript (if any) has no speaker_ids.
        let meeting_dir = dir.path().join(meeting_id.0.to_string());
        let transcript = persistence::read_transcript(&meeting_dir).expect("read transcript");
        assert!(
            transcript.iter().all(|s| s.speaker_id.is_none()),
            "no segment may carry a speaker_id when diarization is disabled"
        );

        // No DiarizationComplete event was emitted.
        let mut saw_diarization = false;
        while let Ok(ev) = event_rx.try_recv() {
            if matches!(ev, AppEvent::DiarizationComplete { .. }) {
                saw_diarization = true;
            }
        }
        assert!(
            !saw_diarization,
            "no DiarizationComplete event may be emitted with diarization disabled"
        );
    }
}

// ---------------------------------------------------------------------------
// Voiceprint refinement-on-confirm (WU3b) — model-free store boundary tests
//
// These tests exercise the enrol-vs-refine dispatch logic that
// `enrol_voiceprint_claimed` now performs: when an identity already exists for
// a given display_name + model_id, the orchestrator calls
// `VoiceprintStore::refine` rather than `VoiceprintStore::enrol`. Because
// the actual centroid-building step (VoiceprintExtractor) requires the
// embedding model and real audio, the tests operate at the persistence store
// boundary — the layer that `enrol_voiceprint_claimed` hands off to after
// the centroid is produced.
//
// Two disciplines from the acceptance criteria:
// 1. A second confirmed association for the same name+model_id routes to
//    refine, and the resulting centroid equals the count-weighted mean of
//    both contributions.
// 2. An adversarial near-threshold refine (mirroring the WU2 poison fixture
//    at the orchestrator boundary) does not push an established centroid past
//    T_accept for a held-out impostor.
// ---------------------------------------------------------------------------

mod voiceprints {
    use minutist_common::{MeetingId, VoiceprintIdentityId};
    use persistence::VoiceprintStore;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    async fn open_mem() -> VoiceprintStore {
        VoiceprintStore::open(":memory:").await.unwrap()
    }

    fn mid() -> MeetingId {
        MeetingId::new()
    }

    /// Synthetic unit-normalised embedding: `v[0] = signal`, `v[1] = sqrt(1 -
    /// signal²)` (the rest zero), then L2-normalised. Produces a deterministic
    /// direction for the given `signal` value.
    fn embed(dim: usize, signal: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[0] = signal;
        if dim > 1 {
            v[1] = (1.0f32 - signal * signal).abs().sqrt();
        }
        minutist_common::voiceprint_math::unit_normalise(&mut v);
        v
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        minutist_common::voiceprint_math::cosine_unit(a, b)
    }

    // -----------------------------------------------------------------------
    // Test 1: second confirmed association routes to refine, not enrol
    //
    // Simulates what `enrol_voiceprint_claimed` does after building the centroid:
    //   - first association → find_identity_by_name_and_model returns None → enrol
    //   - second association → find_identity_by_name_and_model returns Some(id) → refine
    //
    // After both calls, the resulting gallery centroid must equal the
    // count-weighted mean of both contributions (§2.9.1 invariant).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn second_confirmed_association_calls_refine_not_enrol() {
        let store = open_mem().await;
        let model = "cam-test-v1";

        // First association: no prior identity → enrol.
        let emb1 = embed(4, 0.95);
        let existing = store
            .find_identity_by_name_and_model("Alice", model)
            .await
            .unwrap();
        assert!(existing.is_none(), "no identity before first enrol");

        let id1: VoiceprintIdentityId = store
            .enrol("Alice", &emb1, 4, model, mid(), "A")
            .await
            .unwrap();

        // Verify a second lookup finds the identity just created.
        let found = store
            .find_identity_by_name_and_model("Alice", model)
            .await
            .unwrap();
        assert_eq!(found, Some(id1), "identity must be found after first enrol");

        // Second association: identity exists → refine (as the orchestrator now does).
        let emb2 = embed(4, 0.92);
        store
            .refine(id1, &emb2, 2, model, mid(), "A")
            .await
            .unwrap();

        // Verify the orchestrator path did NOT create a second identity.
        let all = store.all(model).await.unwrap();
        let identity_ids: std::collections::HashSet<_> =
            all.iter().map(|s| s.identity_id).collect();
        assert_eq!(
            identity_ids.len(),
            1,
            "refine must not create a second identity; still exactly one identity"
        );
        assert_eq!(identity_ids.iter().next().copied(), Some(id1));

        // The gallery centroid must equal the count-weighted mean of both
        // contributions (§2.9.1 invariant): contribution 1 has count=1, contribution
        // 2 has count=2 (clamped from 2 — existing_sample_count=1, cap=0.30*1→0,
        // so cap=0 means count is used as-is for a first-time existing store with
        // sample_count=1 and REFINE_WEIGHT_CAP=0.30 → cap = ceil(1*0.30)=1, so
        // clamped_count = min(2,1) = 1).
        // Recompute expected centroid: weighted_merge({emb1: count=1, emb2: count=1}).
        let expected = minutist_common::voiceprint_math::weighted_merge(&[
            (emb1.as_slice(), 1u64),
            (emb2.as_slice(), 1u64),
        ]);
        let gallery = store.all(model).await.unwrap();
        assert_eq!(gallery.len(), 1);
        let actual = &gallery[0].embedding;
        assert!(
            cos(actual, &expected) > 0.999,
            "gallery centroid must equal the count-weighted mean of both contributions \
             (cos={:.4})",
            cos(actual, &expected)
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: adversarial near-threshold refine at the orchestrator boundary
    //
    // Mirrors the WU2 `bounded_weight_poison_test` at the persistence boundary.
    // An established centroid with large sample_count, refined with an
    // adversarial contribution near T_accept (0.60), must not shift enough
    // to exceed T_accept for a held-out impostor. REFINE_WEIGHT_CAP = 0.30
    // clamps the adversarial contribution weight, bounding the drift.
    //
    // T_accept placeholder: 0.60 (§2.4, documented in cross-cutting.md).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn adversarial_near_threshold_refine_does_not_breach_t_accept_for_impostor() {
        const T_ACCEPT: f32 = 0.60;
        let store = open_mem().await;
        let model = "cam-poison-v1";
        let dim = 8;

        // Alice's established centroid: a direction close to (1,0,0,...).
        // First, enrol with a large batch of contributions that build up the
        // sample_count (simulating a well-established profile). We do this by
        // enrolling once then refining many times with the same direction.
        let alice_dir = embed(dim, 0.99);
        let identity_id = store
            .enrol("Alice", &alice_dir, dim, model, mid(), "A")
            .await
            .unwrap();

        // Build up sample_count to a large N via repeated refines to make the
        // REFINE_WEIGHT_CAP binding. Use a near-identical embedding to keep the
        // centroid stable (same condition → fold, not add-condition).
        for i in 0u8..10 {
            let v = embed(dim, 0.98 + 0.001 * (i as f32));
            store
                .refine(identity_id, &v, 10, model, mid(), "A")
                .await
                .unwrap();
        }

        // Confirm Alice's gallery centroid is close to alice_dir.
        let pre_gallery = store.all(model).await.unwrap();
        assert_eq!(pre_gallery.len(), 1);
        let pre_cos = cos(&pre_gallery[0].embedding, &alice_dir);
        assert!(
            pre_cos > 0.999,
            "established centroid must be close to its constituent embeddings (cos={pre_cos:.4})"
        );

        // Held-out impostor direction: orthogonal to alice_dir (dim-1 axis only).
        // This represents a distinct speaker that Alice's centroid must not match.
        let mut impostor = vec![0.0f32; dim];
        impostor[dim - 1] = 1.0;
        minutist_common::voiceprint_math::unit_normalise(&mut impostor);
        let pre_impostor_sim = cos(&pre_gallery[0].embedding, &impostor);
        assert!(
            pre_impostor_sim < T_ACCEPT,
            "impostor must be below T_accept before poisoning (sim={pre_impostor_sim:.4})"
        );

        // Adversarial contribution: a direction mixed between Alice (~0.82 cosine,
        // above FOLD_GATE=0.70 so it folds into Alice's centroid) and the impostor
        // axis. The attacker supplies count=1000 (extremely large) to try to dominate
        // the centroid. REFINE_WEIGHT_CAP clamps the contribution to at most
        // 30% of the existing sample_count.
        //
        // Mix: 0.82 × alice_dir + 0.57 × impostor, normalised.
        // cosine(adversary, alice_dir) ≈ 0.82 > FOLD_GATE ✓
        let mut adversary = vec![0.0f32; dim];
        for (i, (&a, &b)) in alice_dir.iter().zip(impostor.iter()).enumerate() {
            adversary[i] = 0.82 * a + 0.57 * b;
        }
        minutist_common::voiceprint_math::unit_normalise(&mut adversary);
        let adv_sim_with_alice = cos(&adversary, &alice_dir);
        assert!(
            adv_sim_with_alice >= 0.70,
            "adversary must clear FOLD_GATE so it folds into the centroid \
             (adv_sim_with_alice={adv_sim_with_alice:.4})"
        );

        let existing_sample_count: u64 = pre_gallery[0].sample_count;
        store
            .refine(identity_id, &adversary, 1000, model, mid(), "A")
            .await
            .unwrap();

        // After the adversarial refine, every centroid in Alice's gallery must still
        // be below T_accept for the impostor. The held-out impostor test: even with
        // an adversarial fold at REFINE_WEIGHT_CAP, Alice's centroid stays safe.
        let post_gallery = store.all(model).await.unwrap();
        // max cosine of any gallery centroid against the impostor:
        let post_max_sim = post_gallery
            .iter()
            .map(|g| cos(&g.embedding, &impostor))
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            post_max_sim < T_ACCEPT,
            "adversarial refine (count=1000, capped by REFINE_WEIGHT_CAP={}) must not push \
             any gallery centroid above T_accept for the impostor \
             (post_max_sim={post_max_sim:.4}, T_accept={T_ACCEPT}, \
             existing_sample_count={existing_sample_count})",
            0.30
        );
    }
}
