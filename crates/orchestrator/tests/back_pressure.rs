//! Back-pressure integration test for the orchestrator's broadcast channel.
//!
//! The test subscribes a slow receiver (does not drain promptly) while driving
//! 500+ meter events via DummyAudioSource at a high-throughput rate.
//! The orchestrator must not panic; some events will be dropped (broadcast
//! semantics). The slow receiver should observe `RecvError::Lagged` from
//! tokio, confirming that back-pressure is surfaced rather than silently lost.
//!
//! Integration tests for the orchestrator live in `crates/orchestrator/tests/`
//! per `architecture/cross-cutting.md` — Testing section.

use std::time::Duration;

use audio_capture::test_source::DummyAudioSource;
use meeting_app_common::AppEvent;
use orchestrator::test_support::test_orchestrator;
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
