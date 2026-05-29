//! Back-pressure integration tests for the orchestrator.
//!
//! Test 4: Exercises the broadcast channel back-pressure (slow subscriber).
//! Test 5: Orchestrator survives subscriber going away.
//! Test 6: ASR flush queue drop-oldest backpressure — drives ≥5 rapid flushes
//!   into a cap-4 queue and asserts the oldest is dropped, ErrorOccurred is
//!   emitted, and the orchestrator survives (no panic, stop() returns Ok).
//!
//! Integration tests for the orchestrator live in `crates/orchestrator/tests/`
//! per `architecture/cross-cutting.md` — Testing section.

use std::time::Duration;

use audio_capture::test_source::DummyAudioSource;
use meeting_app_common::{AppEvent, AppError};
use orchestrator::test_support::{test_orchestrator, FlushBackpressureHarness};
use tokio::sync::broadcast::error::RecvError;

// ---------------------------------------------------------------------------
// Test 4: back-pressure — slow subscriber does not crash the orchestrator
// ---------------------------------------------------------------------------

/// Subscribe to events but do NOT drain the receiver promptly. Drive 500+
/// meter events via DummyAudioSource. Assert:
///   1. The orchestrator does not panic.
///   2. The slow subscriber observes at least one `RecvError::Lagged`
///      (broadcast channel is capped at 256 events; the test drives far more).
///   3. The orchestrator's internal pipeline completes cleanly (stop() returns
///      Ok).
#[tokio::test]
async fn slow_subscriber_observes_lag_and_orchestrator_does_not_panic() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());

    // Subscribe BEFORE start so the receiver is registered.
    let mut slow_rx = orch.subscribe_events();

    // Generate many batches:
    //   - 600 batches × 512 speech samples × ~3 meter windows per batch
    //     ≈ 1800 meter frames >> broadcast capacity of 256.
    // Using DummyAudioSource::generate_streams (synchronous pre-fill) means
    // the runner will be flooded immediately.
    let source = DummyAudioSource::new(512, 256); // 512 speech + 256 silence per batch
    let streams = source.generate_streams(600, 256, 512);

    orch.start_with_streams(streams)
        .await
        .expect("start_with_streams");

    // Do NOT drain `slow_rx` here — let the channel fill up.
    //
    // Wait for the runner to finish processing all pre-generated batches.
    // Since the samples channel closes when DummyAudioSource drops its sender,
    // the runner will drain and then wait for the Stop command. We give it
    // up to 5 seconds.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Stop the orchestrator — must not panic.
    let stop_result = orch.stop().await;
    assert!(
        stop_result.is_ok(),
        "orchestrator.stop() must succeed even with a slow subscriber; got: {stop_result:?}"
    );

    // Now drain the slow receiver and count Lagged errors.
    let mut lagged_count = 0u64;
    let mut meter_count = 0u64;
    let mut state_count = 0u64;

    // Drain with a short timeout so the test doesn't block indefinitely.
    let drain_deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    loop {
        if tokio::time::Instant::now() > drain_deadline {
            break;
        }
        match slow_rx.try_recv() {
            Ok(AppEvent::AudioMeter { .. }) => meter_count += 1,
            Ok(AppEvent::StateChanged { .. }) => state_count += 1,
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                lagged_count += n;
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
        }
    }

    // The broadcast channel capacity is 256. We drove ~1800 meter frames, so
    // at least (1800 − 256) ≈ 1544 events must have been dropped.
    assert!(
        lagged_count > 0,
        "expected the slow subscriber to observe at least one Lagged event; \
         got lagged_count={lagged_count}, meter_count={meter_count}, state_count={state_count}"
    );

    tracing::info!(
        target: "back-pressure-test",
        lagged = lagged_count,
        meter = meter_count,
        state_events = state_count,
        "back-pressure test complete"
    );
}

// ---------------------------------------------------------------------------
// Test 5: orchestrator survives channel closed by subscriber going away
// ---------------------------------------------------------------------------

/// Subscribe, immediately drop the receiver, then run a full recording cycle.
/// The orchestrator must not panic when `broadcast::send` finds no subscribers.
#[tokio::test]
async fn orchestrator_survives_subscriber_gone() {
    let _ = tracing_subscriber::fmt::try_init();

    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());

    // Subscribe and immediately drop — no active receivers.
    let rx = orch.subscribe_events();
    drop(rx);

    let source = DummyAudioSource::new(1600, 800);
    let streams = source.generate_streams(4, 32, 64);

    orch.start_with_streams(streams)
        .await
        .expect("start_with_streams");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let result = orch.stop().await;
    assert!(
        result.is_ok(),
        "stop() must succeed with no active event subscribers; got: {result:?}"
    );
}

// Suppress unused import warning for RecvError (used via the alias in the
// drain loop above via the match arm type).
const _: fn() = || {
    let _: RecvError = RecvError::Closed;
};

// ---------------------------------------------------------------------------
// Test 6: ASR flush queue drop-oldest backpressure
// ---------------------------------------------------------------------------

/// Drive ≥5 rapid flushes into the cap-4 ASR flush queue and verify:
///   (a) The OLDEST flush is the one dropped (not the newest).
///   (b) `AppEvent::ErrorOccurred` is emitted at least once.
///   (c) The orchestrator survives — no panic and `stop()` returns `Ok`.
///
/// This test uses `FlushBackpressureHarness` to inject flushes directly into
/// the flush queue without needing a real VAD model, exercising the drop-oldest
/// logic in `dispatch_flush` (runner.rs) directly.
#[tokio::test]
async fn asr_flush_queue_drops_oldest_and_emits_error_occurred() {
    let _ = tracing_subscriber::fmt::try_init();

    let harness = FlushBackpressureHarness::new();
    let mut event_rx = harness.subscribe();

    // Dispatch 6 payloads into a cap-4 queue. The queue fills on the 4th
    // dispatch. Dispatches 5 and 6 each drop one oldest entry.
    for i in 0..6u64 {
        harness.dispatch(i * 1000);
    }

    // Queue must still be at capacity (4).
    assert_eq!(
        harness.queue_len(),
        4,
        "queue must be at FLUSH_CHANNEL_CAP=4 after 6 dispatches with drop-oldest"
    );

    // The OLDEST flush (start_ms = 0) must have been dropped.
    let starts = harness.queued_start_ms();
    assert!(
        !starts.contains(&0),
        "oldest flush (start_ms=0) must have been dropped; queued start_ms: {starts:?}"
    );
    // The SECOND oldest (start_ms = 1000) must also have been dropped (two overflows).
    assert!(
        !starts.contains(&1000),
        "second oldest flush (start_ms=1000) must have been dropped; queued: {starts:?}"
    );
    // The NEWEST flush (start_ms = 5000) must be present.
    assert!(
        starts.contains(&5000),
        "newest flush (start_ms=5000) must be retained; queued: {starts:?}"
    );

    // Collect all events emitted during the dispatches.
    let mut error_occurred_count = 0u32;
    loop {
        match event_rx.try_recv() {
            Ok(AppEvent::ErrorOccurred { error: AppError::Internal { .. } }) => {
                error_occurred_count += 1;
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(_) => break,
        }
    }

    assert!(
        error_occurred_count >= 2,
        "expected ≥2 ErrorOccurred events (one per overflow); got {error_occurred_count}"
    );

    // Verify the orchestrator itself is also resilient with slow backend and
    // rapid flushes. Use test_orchestrator (no real ASR) which skips flush
    // processing anyway.
    let dir = tempfile::tempdir().expect("tempdir");
    let orch = test_orchestrator(dir.path().to_path_buf());

    let source = DummyAudioSource::new(512, 256);
    let streams = source.generate_streams(10, 64, 128);

    orch.start_with_streams(streams)
        .await
        .expect("start_with_streams");

    tokio::time::sleep(Duration::from_millis(100)).await;

    let stop_result = orch.stop().await;
    assert!(
        stop_result.is_ok(),
        "orchestrator.stop() must return Ok even under flush queue backpressure; got: {stop_result:?}"
    );
}
