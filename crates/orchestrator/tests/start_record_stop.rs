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

    // Pause ~1 s, then resume. The encoder pads the pause with synthesised
    // silence. This test asserts only that the orchestrator pause/resume
    // commands drive the encoder to insert SOME pause silence (pause-INCLUDING,
    // not zero) — NOT an exact duration. `orch.pause()`/`resume()` are async
    // commands, so the encoder's measured pause window (runner-processes-pause →
    // runner-processes-resume) is offset from the test thread's wall-clock by a
    // variable command-dispatch latency; any tight wall-clock window is
    // structurally flaky under parallel-binary load (observed 50-60% of the
    // nominal pause "lost" to dispatch under contention). The EXACT pause
    // inclusion is guarded deterministically, with no wall-clock, by
    // `persistence::tests::test_read_audio_pcm_includes_silent_gap_deterministic`
    // (the `resume_with_pause_frames` seam).
    let nominal_pause_ms: u64 = 1_000;
    tokio::time::sleep(Duration::from_millis(nominal_pause_ms)).await;
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
    //
    // All six transitions (… → Stopping → Finalising → Idle) are already emitted
    // onto the broadcast channel by the orchestrator calls above; this loop only
    // drains them. The deadline
    // must therefore tolerate a saturated scheduler — when the gated
    // transcription_e2e binary loads a model and runs CPU inference in a
    // parallel test process, a tight 500 ms window starved this drain loop and
    // it missed already-queued events. 5 s is ample for draining without
    // weakening the assertion (it still requires all five variants to appear).
    let mut events: Vec<RecordingState> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
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
        // Recording → Paused → Recording → Stopping → Finalising → Idle = 6.
        if events.len() >= 6 {
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
        events.iter().any(
            |s| matches!(s, RecordingState::Finalising { meeting_id: mid } if *mid == meeting_id)
        ),
        "expected RecordingState::Finalising event, got: {events:?}"
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
    let pos_finalising = first_pos(&|s| matches!(s, RecordingState::Finalising { .. }));
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
        pos_stopping < pos_finalising,
        "Stopping must appear before Finalising in event stream"
    );
    assert!(
        pos_finalising < pos_idle,
        "Finalising must appear before Idle in event stream"
    );

    // Decode the resulting audio.opus and check the pause gap is included.
    let meeting_dir = dir.path().join(meeting_id.0.to_string());
    let audio_path = meeting_dir.join("audio.opus");
    assert!(audio_path.exists(), "audio.opus must exist after stop");

    let total_samples = count_opus_samples(&audio_path);
    let decoded_duration_ms = (total_samples as u64 * 1000) / 16_000;

    // Integration assertion (deliberately loose — see the pause comment above):
    // a pause-INCLUDING encoder decodes to clearly MORE than the audio-only
    // duration (silence was inserted); a pause-EXCLUDING one decodes to
    // ~audio_only. The floor is set well above audio_only but far below a
    // typical ~1 s pause's contribution, so it survives heavy dispatch loss
    // while still failing if pause silence is dropped entirely. The ceiling
    // catches absurd inflation. Exact duration is the persistence test's job.
    let audio_only_ms: u64 = 400; // 2 × 200 ms audio segments
    let floor_ms = audio_only_ms + 250; // > audio_only ⇒ pause silence present
    let ceil_ms = audio_only_ms + nominal_pause_ms * 3; // generous upper sanity bound

    assert!(
        decoded_duration_ms >= floor_ms,
        "decoded duration {decoded_duration_ms} ms must exceed {floor_ms} ms — the orchestrator \
         pause/resume did not insert pause silence (pause-EXCLUDING regression)"
    );
    assert!(
        decoded_duration_ms <= ceil_ms,
        "decoded duration {decoded_duration_ms} ms exceeds {ceil_ms} ms — pause gap absurdly inflated"
    );
}

// ---------------------------------------------------------------------------
// Test 3: RecordingClock is the pause-EXCLUDING sample clock (binding A4)
// ---------------------------------------------------------------------------

/// Regression for binding correction A4 (`architecture/cross-cutting.md` —
/// "Notes paragraph-anchor clock").
///
/// The runner emits `AppEvent::RecordingClock { clock_ms: batch.end_ms }` on
/// the sample-batch receive path, throttled to ~5 Hz. The notes editor stamps
/// paragraph anchors from this value, so the clock MUST track the
/// pause-EXCLUDING capture-sample timeline (the same origin as
/// `Segment::start_ms`), never pause-including wall-clock.
///
/// This test drives the orchestrator through start → record → pause → (wall
/// time elapses) → resume → record → stop with deterministic batches whose
/// `end_ms` lie on the sample timeline, collecting every emitted
/// `RecordingClock { clock_ms, .. }`. It asserts:
///
/// 1. `RecordingClock` IS emitted while recording.
/// 2. `clock_ms` is monotonically non-decreasing.
/// 3. Every emitted `clock_ms` equals one of the fed `batch.end_ms` values
///    (the pause-EXCLUDING clock) — it is never wall-clock and never a value
///    inflated by the pause wall-time.
/// 4. The LOAD-BEARING property: across the pause, the clock does not advance
///    by the pause duration. The post-resume batches continue contiguously on
///    the sample timeline (their `end_ms` resume from the pre-pause tail, NOT
///    from `tail + pause_wall`), and no emitted `clock_ms` ever equals a
///    pause-inflated offset.
///
/// Batches are spaced by > `RECORDING_CLOCK_MIN_INTERVAL_MS` of wall time so
/// each crosses the ~5 Hz throttle window and is emitted; the assertions are
/// nonetheless written against the *sequence* of emitted values (a subset of
/// the fed `end_ms` set in order), not a fixed count, so they remain correct
/// if the throttle coalesces some emissions.
///
/// `DummyAudioSource`-style synchronous generation would deliver every batch
/// before the runner drains the first one, collapsing the throttle to a single
/// emission and making the pause boundary unobservable in the event stream.
/// The live-streams harness with paced sends is the correct fixture for the
/// clock/metering path here (this asserts the clock timeline, not VAD speech
/// output — valid per `cross-cutting.md` §Testing).
#[tokio::test]
async fn recording_clock_is_pause_excluding_across_pause_resume() {
    let _ = tracing_subscriber::fmt::try_init();

    // Comfortably larger than RECORDING_CLOCK_MIN_INTERVAL_MS (200 ms) so each
    // paced batch lands in its own throttle window and emits a RecordingClock.
    const SEND_SPACING_MS: u64 = 300;
    const PAUSE_WALL_MS: u64 = 800;

    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());
    let mut rx = orch.subscribe_events();

    let (sample_tx, meter_tx, streams) = live_streams(64, 64);

    let meeting_id = orch
        .start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // Drives one paced batch into the runner and drains every RecordingClock
    // that has accumulated on the broadcast bus so far, appending each
    // `clock_ms` (in arrival order) to `clocks`.
    //
    // The sample-clock offset advances by exactly 100 ms per batch (the batch
    // duration), independent of how much wall time passes — this is what makes
    // the post-pause clock pause-EXCLUDING.
    async fn feed_and_collect(
        sample_tx: &mpsc::Sender<AudioFrameBatch>,
        meter_tx: &mpsc::Sender<AudioMeterFrame>,
        rx: &mut tokio::sync::broadcast::Receiver<AppEvent>,
        sample_offset_ms: u64,
        clocks: &mut Vec<u64>,
        expected_meeting_id: MeetingId,
    ) {
        sample_tx
            .send(make_batch(sample_offset_ms))
            .await
            .expect("send batch");
        meter_tx
            .send(AudioMeterFrame {
                peak: 0.4,
                rms: 0.25,
            })
            .await
            .expect("send meter");
        // Let the runner drain this batch and emit the throttled clock.
        tokio::time::sleep(Duration::from_millis(SEND_SPACING_MS)).await;
        drain_recording_clocks(rx, clocks, expected_meeting_id);
    }

    // Drain all currently-queued RecordingClock events into `clocks`.
    fn drain_recording_clocks(
        rx: &mut tokio::sync::broadcast::Receiver<AppEvent>,
        clocks: &mut Vec<u64>,
        expected_meeting_id: MeetingId,
    ) {
        loop {
            match rx.try_recv() {
                Ok(AppEvent::RecordingClock {
                    meeting_id: mid,
                    clock_ms,
                }) => {
                    assert_eq!(
                        mid, expected_meeting_id,
                        "RecordingClock meeting_id must match the active recording"
                    );
                    clocks.push(clock_ms);
                }
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            }
        }
    }

    let mut clocks: Vec<u64> = Vec::new();

    // --- Pre-pause: feed batches at sample offsets 0, 100, 200 ms.
    // Their end_ms values are 100, 200, 300 ms (pause-EXCLUDING sample clock).
    let pre_pause_offsets = [0u64, 100, 200];
    for off in pre_pause_offsets {
        feed_and_collect(&sample_tx, &meter_tx, &mut rx, off, &mut clocks, meeting_id)
            .await;
    }

    // Snapshot every clock emitted up to (and including) the moment we pause.
    let clocks_before_pause = clocks.clone();

    // The last sample end_ms before the pause. Post-resume must continue from
    // here on the sample timeline, NOT from here + PAUSE_WALL_MS.
    let pre_pause_tail_end_ms = pre_pause_offsets.last().copied().unwrap() + 100; // 300 ms

    orch.pause().await.expect("pause");

    // Wall time elapses during the pause. A pause-INCLUDING (wall-clock) clock
    // would advance by ~PAUSE_WALL_MS here; the pause-EXCLUDING sample clock
    // must not. No batches are fed while paused, and the runner's paused branch
    // never reaches the RecordingClock emit site, so no clock must be emitted.
    tokio::time::sleep(Duration::from_millis(PAUSE_WALL_MS)).await;
    drain_recording_clocks(&mut rx, &mut clocks, meeting_id);
    let clocks_at_pause_end = clocks.clone();

    orch.resume().await.expect("resume");

    // --- Post-resume: continue CONTIGUOUSLY on the sample timeline.
    // Offsets 300, 400, 500 ms → end_ms 400, 500, 600 ms. Crucially these do
    // NOT skip ahead by PAUSE_WALL_MS; the sample clock only counts captured
    // audio, not pause wall-time.
    let post_resume_offsets = [
        pre_pause_tail_end_ms,       // 300 ms — contiguous with pre-pause tail
        pre_pause_tail_end_ms + 100, // 400 ms
        pre_pause_tail_end_ms + 200, // 500 ms
    ];
    for off in post_resume_offsets {
        feed_and_collect(&sample_tx, &meter_tx, &mut rx, off, &mut clocks, meeting_id)
            .await;
    }

    drop(sample_tx);
    drop(meter_tx);

    tokio::time::sleep(Duration::from_millis(50)).await;
    drain_recording_clocks(&mut rx, &mut clocks, meeting_id);

    let _meta = orch.stop().await.expect("stop");
    drain_recording_clocks(&mut rx, &mut clocks, meeting_id);

    // -------------------- Assertions --------------------

    // (1) RecordingClock IS emitted while recording.
    assert!(
        !clocks.is_empty(),
        "expected at least one AppEvent::RecordingClock while recording, got none"
    );

    // (2) Monotonically non-decreasing.
    for w in clocks.windows(2) {
        assert!(
            w[1] >= w[0],
            "RecordingClock clock_ms must be monotonically non-decreasing; \
             saw {} after {} in sequence {clocks:?}",
            w[1],
            w[0]
        );
    }

    // (3) Every emitted clock_ms is a fed batch end_ms (the pause-EXCLUDING
    //     sample clock) — never wall-clock, never a pause-inflated value.
    let fed_end_ms: std::collections::BTreeSet<u64> = pre_pause_offsets
        .iter()
        .chain(post_resume_offsets.iter())
        .map(|off| off + 100) // make_batch sets end_ms = start_ms + 100
        .collect();
    for &c in &clocks {
        assert!(
            fed_end_ms.contains(&c),
            "clock_ms {c} is not one of the fed sample-batch end_ms values \
             {fed_end_ms:?} — the clock leaked wall-clock or a pause-inflated \
             offset; full sequence: {clocks:?}"
        );
    }

    // (4a) LOAD-BEARING: no RecordingClock was emitted during the paused
    //      interval. The paused branch must never reach the emit site, so the
    //      snapshot taken at the END of the pause window (after PAUSE_WALL_MS of
    //      wall time elapsed, before resume) must be byte-for-byte identical to
    //      the snapshot taken at the START of the pause. Any growth here means a
    //      clock leaked out while paused.
    assert_eq!(
        clocks_at_pause_end, clocks_before_pause,
        "RecordingClock was emitted during the pause; the emit site must be \
         unreachable while paused. before pause: {clocks_before_pause:?}, \
         after {PAUSE_WALL_MS} ms paused: {clocks_at_pause_end:?}"
    );

    // (4b) LOAD-BEARING: the clock never advanced by the pause wall-time. The
    //      maximum emitted clock_ms must equal the maximum fed sample end_ms
    //      (600 ms), NOT a value inflated toward (600 + PAUSE_WALL_MS). A
    //      pause-INCLUDING clock would necessarily emit a value above the
    //      sample-timeline maximum once post-resume batches flowed.
    let max_sample_end_ms = post_resume_offsets.last().copied().unwrap() + 100; // 600 ms
    let max_clock = clocks.iter().copied().max().unwrap();
    assert!(
        max_clock <= max_sample_end_ms,
        "max RecordingClock {max_clock} exceeds the max sample-timeline end_ms \
         {max_sample_end_ms}; the clock advanced across the pause \
         (pause-INCLUDING). PAUSE_WALL_MS={PAUSE_WALL_MS}; sequence: {clocks:?}"
    );

    // (4c) The post-resume clock continues contiguously from the pre-pause
    //      tail: at least one emitted value lies at or beyond the pre-pause
    //      tail end_ms, confirming recording genuinely resumed and advanced the
    //      sample clock — without the pause gap. (Guards against the test
    //      trivially passing because post-resume audio never reached the emit
    //      site at all.)
    assert!(
        clocks.iter().any(|&c| c > pre_pause_tail_end_ms),
        "no RecordingClock advanced past the pre-pause tail {pre_pause_tail_end_ms} ms; \
         post-resume sample audio did not advance the clock as expected; \
         sequence: {clocks:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: invalid state transitions
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
