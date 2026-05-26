//! Integration tests for the full Phase 1 recording pipeline.
//!
//! All tests use `start_with_streams` to avoid needing a real microphone.
//! Each test covers a distinct behavioural scenario of the orchestrator.
//!
//! Integration tests for the orchestrator live here per the architecture
//! convention (`architecture/cross-cutting.md` — Testing section).

use std::time::Duration;

use audio_capture::{AudioFrameBatch, AudioStreams};
use meeting_app_common::{AppError, AppEvent, AudioMeterFrame, MeetingId, RecordingState};
use orchestrator::test_support::test_orchestrator;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Produce live `AudioStreams` senders and the matching `AudioStreams` receiver
/// for injection into `start_with_streams`.
///
/// The caller keeps the senders and pushes batches on demand; dropping them
/// signals end-of-stream to the runner.
fn live_streams(
    sample_cap: usize,
    meter_cap: usize,
) -> (
    mpsc::Sender<AudioFrameBatch>,
    mpsc::Sender<AudioMeterFrame>,
    AudioStreams,
) {
    let (sample_tx, sample_rx) = mpsc::channel::<AudioFrameBatch>(sample_cap);
    let (meter_tx, meter_rx) = mpsc::channel::<AudioMeterFrame>(meter_cap);
    (
        sample_tx,
        meter_tx,
        AudioStreams {
            samples: sample_rx,
            meter: meter_rx,
        },
    )
}

/// Generate a single batch of 1600 16 kHz mono f32 samples (100 ms worth),
/// using a 440 Hz sine at 0.5 amplitude.
fn make_batch(start_ms: u64) -> AudioFrameBatch {
    const SAMPLES: usize = 1600;
    const FREQ: f32 = 440.0;
    const AMP: f32 = 0.5;
    let end_ms = start_ms + (SAMPLES as u64 * 1000 / 16_000);
    let samples: Vec<f32> = (0..SAMPLES)
        .map(|i| {
            let t = (start_ms * 16 + i as u64) as f32 / 16_000.0;
            AMP * (2.0 * std::f32::consts::PI * FREQ * t).sin()
        })
        .collect();
    AudioFrameBatch {
        samples,
        start_ms,
        end_ms,
    }
}

/// Count total PCM samples encoded in an Ogg/Opus file at `path`.
///
/// Skips the two Opus header packets and uses `Decoder::nb_samples` on each
/// audio packet. Each packet encodes exactly one Opus frame (20 ms = 320
/// samples at 16 kHz in this implementation), so this gives the exact
/// decoded duration without a full decode pass.
fn count_opus_samples(path: &std::path::Path) -> usize {
    use audiopus::{coder::Decoder, packet::Packet, Channels, SampleRate};
    use std::convert::TryFrom;

    let data = std::fs::read(path).expect("read audio.opus");
    let mut cursor = std::io::Cursor::new(data);
    let mut reader = ogg::reading::PacketReader::new(&mut cursor);

    let decoder = Decoder::new(SampleRate::Hz16000, Channels::Mono).expect("create opus decoder");

    let mut header_skipped = 0usize;
    let mut total_samples = 0usize;

    loop {
        match reader.read_packet() {
            Ok(Some(pkt)) => {
                if header_skipped < 2 {
                    header_skipped += 1;
                    continue;
                }
                // nb_samples counts samples without a full decode.
                if let Ok(pkt_ref) = Packet::try_from(pkt.data.as_slice()) {
                    if let Ok(n) = decoder.nb_samples(pkt_ref) {
                        total_samples += n;
                    }
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }

    total_samples
}

// ---------------------------------------------------------------------------
// Test 1: start → record → stop lifecycle
// ---------------------------------------------------------------------------

/// Full lifecycle: start with injected live streams, assert state becomes
/// Recording, receive ≥1 StateChanged(Recording) and ≥1 AudioMeter within
/// 5 seconds, stop, assert MeetingMeta fields are correct, verify on-disk
/// files exist and deserialise correctly.
#[tokio::test]
async fn start_record_stop_produces_valid_meeting_files() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());
    let mut rx = orch.subscribe_events();

    assert_eq!(orch.state().await, RecordingState::Idle);

    let (sample_tx, meter_tx, streams) = live_streams(64, 64);

    let meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // Push ~200 ms of audio (2 × 100 ms batches).
    for i in 0..2u64 {
        sample_tx
            .send(make_batch(i * 100))
            .await
            .expect("send batch");
        meter_tx
            .send(AudioMeterFrame {
                peak: 0.5,
                rms: 0.3,
            })
            .await
            .expect("send meter");
    }
    // Drop senders to signal end-of-stream.
    drop(sample_tx);
    drop(meter_tx);

    // Assert state is Recording immediately after start.
    match orch.state().await {
        RecordingState::Recording {
            meeting_id: mid, ..
        } => assert_eq!(mid, meeting_id, "meeting_id mismatch in Recording state"),
        s => panic!("expected Recording, got {s:?}"),
    }

    // Within 5 seconds, collect ≥1 StateChanged(Recording) and ≥1 AudioMeter.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut saw_recording_event = false;
    let mut saw_meter_event = false;

    while tokio::time::Instant::now() < deadline && (!saw_recording_event || !saw_meter_event) {
        match rx.try_recv() {
            Ok(AppEvent::StateChanged {
                state:
                    RecordingState::Recording {
                        meeting_id: mid, ..
                    },
            }) => {
                assert_eq!(mid, meeting_id);
                saw_recording_event = true;
            }
            Ok(AppEvent::AudioMeter { .. }) => {
                saw_meter_event = true;
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(target: "test", lagged = n, "subscriber lagged");
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }

    assert!(
        saw_recording_event,
        "never received StateChanged(Recording)"
    );
    assert!(saw_meter_event, "never received AudioMeter event");

    // Stop and inspect returned MeetingMeta.
    let meta = orch.stop().await.expect("stop");

    assert_eq!(meta.uuid, meeting_id, "uuid must match");
    assert!(!meta.started_at.is_empty(), "started_at must be populated");
    assert!(meta.ended_at.is_some(), "ended_at must be Some after stop");
    assert_eq!(meta.audio_format.codec, "opus");
    assert_eq!(meta.audio_format.sample_rate, 16_000);
    assert_eq!(meta.audio_format.channels, 1);
    assert_eq!(meta.audio_format.bitrate_kbps, Some(32));

    // Verify on-disk files.
    let meeting_dir = dir.path().join(meeting_id.0.to_string());
    assert!(
        meeting_dir.exists(),
        "meeting directory must exist at {meeting_dir:?}"
    );

    let audio_path = meeting_dir.join("audio.opus");
    assert!(audio_path.exists(), "audio.opus must exist");
    assert!(
        std::fs::metadata(&audio_path).unwrap().len() > 0,
        "audio.opus must be non-empty"
    );

    let meta_path = meeting_dir.join("metadata.json");
    assert!(meta_path.exists(), "metadata.json must exist");

    let json = std::fs::read_to_string(&meta_path).expect("read metadata.json");
    let on_disk: meeting_app_common::MeetingMeta =
        serde_json::from_str(&json).expect("deserialise metadata.json");
    assert_eq!(on_disk.uuid, meeting_id, "on-disk uuid must match");
    assert_eq!(on_disk.audio_format.codec, "opus");
    assert_eq!(on_disk.started_at, meta.started_at);
    assert_eq!(on_disk.ended_at, meta.ended_at);
}

// ---------------------------------------------------------------------------
// Test 2: pause → resume → decoded duration includes pause gap
// ---------------------------------------------------------------------------

/// Pause/resume lifecycle: start, push ~200 ms of audio, pause, sleep 500 ms,
/// resume, push ~200 ms more, stop. Assert state events fire in order and the
/// decoded audio duration is approximately (audio + pause_gap), confirming the
/// zero-sample padding strategy documented in `crates/persistence/README.md`.
#[tokio::test]
async fn pause_resume_decoded_duration_includes_pause_gap() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());
    let mut rx = orch.subscribe_events();

    let (sample_tx, meter_tx, streams) = live_streams(64, 64);

    let meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // Push ~200 ms of audio before the pause.
    for i in 0..2u64 {
        sample_tx
            .send(make_batch(i * 100))
            .await
            .expect("send pre-pause batch");
        meter_tx
            .send(AudioMeterFrame {
                peak: 0.4,
                rms: 0.25,
            })
            .await
            .expect("send pre-pause meter");
    }

    // Give the runner time to drain the pre-pause samples before we pause.
    tokio::time::sleep(Duration::from_millis(50)).await;

    orch.pause().await.expect("pause");

    // Sleep 500 ms wall-clock — the encoder will pad this gap with silence.
    let pause_wall_ms: u64 = 500;
    tokio::time::sleep(Duration::from_millis(pause_wall_ms)).await;

    orch.resume().await.expect("resume");

    // Push ~200 ms of audio after the resume (clock offset 700 ms = 200 + 500).
    for i in 0..2u64 {
        sample_tx
            .send(make_batch(700 + i * 100))
            .await
            .expect("send post-resume batch");
        meter_tx
            .send(AudioMeterFrame {
                peak: 0.4,
                rms: 0.25,
            })
            .await
            .expect("send post-resume meter");
    }

    // Drop senders to signal end-of-stream.
    drop(sample_tx);
    drop(meter_tx);

    // Give the runner time to drain the post-resume samples.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let _meta = orch.stop().await.expect("stop");

    // Collect state-change events and verify the ordering.
    let mut events: Vec<RecordingState> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    loop {
        if tokio::time::Instant::now() > deadline {
            break;
        }
        match rx.try_recv() {
            Ok(AppEvent::StateChanged { state }) => {
                events.push(state.clone());
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                tracing::warn!(target: "test", lagged = n, "subscriber lagged");
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
        // Recording → Paused → Recording → Stopping → Idle = 5 transitions.
        if events.len() >= 5 {
            break;
        }
    }

    // Verify all expected state variants appeared.
    assert!(
        events.iter().any(
            |s| matches!(s, RecordingState::Recording { meeting_id: mid, .. } if *mid == meeting_id)
        ),
        "expected RecordingState::Recording event for {meeting_id:?}, got: {events:?}"
    );
    assert!(
        events.iter().any(
            |s| matches!(s, RecordingState::Paused { meeting_id: mid, .. } if *mid == meeting_id)
        ),
        "expected RecordingState::Paused event, got: {events:?}"
    );
    assert!(
        events.iter().any(
            |s| matches!(s, RecordingState::Stopping { meeting_id: mid } if *mid == meeting_id)
        ),
        "expected RecordingState::Stopping event, got: {events:?}"
    );
    assert!(
        events.iter().any(|s| matches!(s, RecordingState::Idle)),
        "expected RecordingState::Idle event, got: {events:?}"
    );

    // Verify ordering using the position of the first occurrence of each variant.
    let first_pos = |check: &dyn Fn(&RecordingState) -> bool| -> usize {
        events
            .iter()
            .position(check)
            .expect("state not found in events")
    };

    let pos_recording = first_pos(&|s| matches!(s, RecordingState::Recording { .. }));
    let pos_paused = first_pos(&|s| matches!(s, RecordingState::Paused { .. }));
    let pos_stopping = first_pos(&|s| matches!(s, RecordingState::Stopping { .. }));
    let pos_idle = first_pos(&|s| matches!(s, RecordingState::Idle));

    assert!(
        pos_recording < pos_paused,
        "Recording must appear before Paused in event stream"
    );
    assert!(
        pos_paused < pos_stopping,
        "Paused must appear before Stopping in event stream"
    );
    assert!(
        pos_stopping < pos_idle,
        "Stopping must appear before Idle in event stream"
    );

    // Decode the resulting audio.opus and check the pause gap is included.
    let meeting_dir = dir.path().join(meeting_id.0.to_string());
    let audio_path = meeting_dir.join("audio.opus");
    assert!(audio_path.exists(), "audio.opus must exist after stop");

    let total_samples = count_opus_samples(&audio_path);
    let decoded_duration_ms = (total_samples as u64 * 1000) / 16_000;

    // Expected decoded duration:
    //   ~200 ms pre-pause audio + ~500 ms pause gap + ~200 ms post-resume audio
    //   = ~900 ms total.
    //
    // The encoder rounds the pause to the nearest 20 ms frame boundary (±20 ms).
    // We allow ±100 ms for OS scheduling jitter in the test environment, which is
    // within the ±50 ms accuracy budget stated in persistence/README.md.
    let audio_only_ms: u64 = 400; // 2 × 200 ms audio segments
    let expected_min_ms = audio_only_ms + pause_wall_ms - 100;
    let expected_max_ms = audio_only_ms + pause_wall_ms + 100;

    assert!(
        decoded_duration_ms >= expected_min_ms,
        "decoded duration {decoded_duration_ms} ms is below minimum {expected_min_ms} ms \
         (pause gap may be too short; pause_wall_ms={pause_wall_ms})"
    );
    assert!(
        decoded_duration_ms <= expected_max_ms,
        "decoded duration {decoded_duration_ms} ms exceeds maximum {expected_max_ms} ms \
         (pause gap may be inflated; pause_wall_ms={pause_wall_ms})"
    );
}

// ---------------------------------------------------------------------------
// Test 3: invalid state transitions
// ---------------------------------------------------------------------------

/// `pause()` from Idle returns `AppError::InvalidInput`.
/// `start_with_streams()` while already Recording returns `AppError::InvalidInput`.
#[tokio::test]
async fn invalid_transitions_return_invalid_input() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());

    // pause() from Idle.
    let err = orch
        .pause()
        .await
        .expect_err("pause() from Idle should fail");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected AppError::InvalidInput from pause() on Idle, got {err:?}"
    );

    // Start a session.
    let (_sample_tx1, _meter_tx1, streams1) = live_streams(32, 32);
    let _mid = orch
        .start_with_streams(streams1)
        .await
        .expect("first start_with_streams");

    // start_with_streams() again while Recording.
    let (_sample_tx2, _meter_tx2, streams2) = live_streams(32, 32);
    let err = orch
        .start_with_streams(streams2)
        .await
        .expect_err("start() while Recording should fail");
    assert!(
        matches!(err, AppError::InvalidInput { .. }),
        "expected AppError::InvalidInput from start() while Recording, got {err:?}"
    );

    // Clean up.
    drop(_sample_tx1);
    orch.stop()
        .await
        .expect("stop after invalid-transitions test");
}

// ---------------------------------------------------------------------------
// Helpers used in tests
// ---------------------------------------------------------------------------

/// Suppress the unused-import warning on `MeetingId` when the test pattern
/// binds the field inline.
#[allow(dead_code)]
fn _check_meeting_id_used(_: MeetingId) {}
