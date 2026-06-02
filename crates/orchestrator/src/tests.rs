//! Unit tests for `orchestrator`.
//!
//! All tests use `DummyAudioSource` + `start_with_streams` to avoid needing a
//! real microphone. Tests run on the tokio multi-thread scheduler via the
//! `#[tokio::test]` macro.

use std::time::Duration;

use audio_capture::test_source::DummyAudioSource;
use meeting_app_common::{AppError, AppEvent, RecordingState};
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

    // --- stop → Stopping then Idle ---
    // The `stop()` call emits StateChanged(Stopping) *before* awaiting the runner,
    // and then StateChanged(Idle) after the runner finalises. Both events are
    // broadcast synchronously from within the stop() future, so by the time
    // stop() returns they're already in the channel.
    let meta = orch.stop().await.expect("stop");

    // Drain all pending events to find Stopping and Idle.
    let mut saw_stopping = false;
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
                state: RecordingState::Idle,
            }) => {
                saw_idle = true;
            }
            Ok(AppEvent::AudioMeter { .. }) => {}
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
        if saw_stopping && saw_idle {
            break;
        }
    }

    assert!(saw_stopping, "expected StateChanged(Stopping)");
    assert!(saw_idle, "expected StateChanged(Idle)");
    assert_eq!(orch.state().await, RecordingState::Idle);
    assert_eq!(meta.uuid, meeting_id);
    assert_eq!(meta.speaker_count, 0);
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
    let loaded: meeting_app_common::MeetingMeta =
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
// A `StubDiarizer` (a `common::Diarizer` that assigns speaker_ids
// deterministically, no sherpa model) drives the re-diarize inner path over a
// synthetic meeting folder, and a toggle-OFF `stop()` test proves the on-stop
// pass is gated.
// ---------------------------------------------------------------------------

mod diarization {
    use super::*;
    use meeting_app_common::{
        AppResult, AudioFormat, Diarizer, MeetingId, MeetingMeta, Segment,
    };
    use persistence::{MeetingIndex, MeetingWriter};

    /// A `common::Diarizer` with no model: it assigns first-seen-order labels
    /// `"A"`, `"B"`, … round-robin across the segments and reports the distinct
    /// count, so the orchestrator's persistence + index + event wiring can be
    /// exercised without sherpa or any ONNX file.
    struct StubDiarizer {
        /// Number of distinct speakers to spread across the segments.
        speakers: u32,
    }

    impl Diarizer for StubDiarizer {
        fn assign_speakers(
            &self,
            audio: &[f32],
            sample_rate: u32,
            segments: &mut [Segment],
        ) -> AppResult<u32> {
            assert_eq!(sample_rate, 16_000, "orchestrator must call with 16 kHz");
            assert!(!audio.is_empty(), "orchestrator must decode non-empty PCM");
            if segments.is_empty() || self.speakers == 0 {
                return Ok(0);
            }
            let labels = ["A", "B", "C", "D"];
            let used = (self.speakers as usize).min(labels.len()).min(segments.len());
            for (i, seg) in segments.iter_mut().enumerate() {
                seg.speaker_id = Some(labels[i % used].to_string());
            }
            Ok(used as u32)
        }
    }

    /// Build a synthetic meeting folder on disk: `audio.opus` encoded from
    /// `samples` via the production `MeetingWriter`, a `metadata.json` with
    /// `speaker_count = 0` / `diarizer = None`, and a `transcript.json` with
    /// `segment_texts.len()` segments (all `speaker_id = None`). Returns the id.
    fn build_meeting(
        root: &std::path::Path,
        samples: &[f32],
        segment_texts: &[&str],
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
            app_version: "0.0.0".into(),
        };
        let folder = writer.finalise(meta).expect("finalise");

        let segments: Vec<Segment> = segment_texts
            .iter()
            .enumerate()
            .map(|(i, t)| Segment {
                start_ms: i as u64 * 1000,
                end_ms: i as u64 * 1000 + 800,
                text: (*t).to_string(),
                speaker_id: None,
                confidence: None,
                words: Vec::new(),
            })
            .collect();
        std::fs::write(
            folder.path().join("transcript.json"),
            serde_json::to_vec_pretty(&segments).unwrap(),
        )
        .expect("write transcript.json");

        meeting_id
    }

    /// The re-diarize inner path (driven via the `rediarize_with_diarizer` stub
    /// seam) rewrites `transcript.json` with overlaid `speaker_id`s, updates
    /// `metadata.json`'s `speaker_count` + `diarizer`, refreshes the index row,
    /// and emits `DiarizationComplete` — all without a sherpa model.
    #[tokio::test]
    async fn rediarize_with_stub_writes_speaker_ids_and_emits() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        let orch = test_orchestrator(root.clone());
        let mut event_rx = orch.subscribe_events();

        // Non-zero PCM (the stub asserts non-empty; content is irrelevant).
        let samples = vec![0.25f32; 16_000];
        let meeting_id = build_meeting(&root, &samples, &["hello", "there", "again"]);
        let meeting_dir = root.join(meeting_id.0.to_string());

        // Before: every segment is unlabelled, metadata speaker_count is 0.
        let before = persistence::read_transcript(&meeting_dir).expect("read transcript");
        assert!(before.iter().all(|s| s.speaker_id.is_none()));

        let index = MeetingIndex::open(":memory:").await.expect("open index");
        index.rebuild_from_disk(&root).await.expect("seed index");

        let stub = Box::new(StubDiarizer { speakers: 2 });
        orch.rediarize_with_diarizer(&index, meeting_id, stub)
            .await
            .expect("rediarize_with_diarizer must succeed");

        // 1. transcript.json rewritten with speaker_ids (A/B round-robin).
        let after = persistence::read_transcript(&meeting_dir).expect("read transcript after");
        assert_eq!(after.len(), 3);
        assert_eq!(after[0].speaker_id.as_deref(), Some("A"));
        assert_eq!(after[1].speaker_id.as_deref(), Some("B"));
        assert_eq!(after[2].speaker_id.as_deref(), Some("A"));

        // 2. metadata.json speaker_count updated + diarizer descriptor set.
        let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
        assert_eq!(meta.speaker_count, 2);
        assert!(meta.diarizer.is_some(), "diarizer descriptor must be recorded");

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

    /// With `diarization_enabled = false` (the default), `stop()` runs no
    /// diarization pass: the returned `MeetingMeta` and the on-disk transcript
    /// keep every `speaker_id == None`, `speaker_count == 0`, and NO
    /// `DiarizationComplete` event is emitted.
    #[tokio::test]
    async fn stop_with_diarization_disabled_leaves_segments_unlabelled() {
        let _ = tracing_subscriber::fmt::try_init();
        let dir = tempfile::tempdir().expect("tempdir");
        let orch = test_orchestrator(dir.path().to_path_buf());
        // The default SettingsHandle has diarization_enabled = false.
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
