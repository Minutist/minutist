//! Supervised reconnect loop around [`run_tunnel`] (WS4-A S5b, tc-reconnect).
//!
//! [`run_tunnel`] performs one connect-and-run; this wraps it in a long-lived
//! loop that survives transient relay drops with capped exponential backoff +
//! jitter, and that surfaces the connection state through a callback so the UI
//! can show `Connecting` / `Online` / `RelayUnreachable` rather than error
//! spam.
//!
//! # The revoked-credential rule (S5a open-question #2)
//!
//! A tunnel `HelloErr { AuthFailed }` after the credential has *previously*
//! worked means the device was revoked (or the credential rotated) — retrying
//! cannot succeed and would hot-loop. So the loop tracks whether the current
//! credential has ever completed a handshake:
//!
//! - `AuthFailed` **before** any successful session → a config problem (bad
//!   credential from the start). The loop stops with [`ReconnectExit::AuthFailed`].
//! - `AuthFailed` **after** a successful session → the credential was revoked.
//!   The loop stops with [`ReconnectExit::NeedsRepair`] so the app prompts a
//!   re-pair instead of retrying.
//!
//! Every other error (transport, connect, protocol) is transient: back off and
//! redial. A clean close also redials (the relay may cycle a connection).
//!
//! # Cancellation
//!
//! The loop is cancellable via a [`tokio::sync::watch`] receiver. When it flips
//! to `true` the loop stops at the next await point (between or during a
//! session); the in-session teardown (the `JoinSet` abort + writer drain in
//! [`run_tunnel`]) handles in-flight work. The cancel is checked both during the
//! backoff sleep and concurrently with a running session.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;

use crate::run::{run_tunnel_with_observer, TunnelConfig, TunnelError};

/// Initial backoff after the first failed attempt.
pub const BACKOFF_INITIAL: Duration = Duration::from_secs(1);

/// Ceiling on the backoff delay (RFC-8628-style capped exponential).
pub const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// The live connection state, reported to the caller's callback as the loop
/// transitions. Maps directly onto the UI's `TunnelStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Dialing / handshaking a fresh session.
    Connecting,
    /// A session is established (the handshake was acknowledged).
    Online,
    /// The last attempt failed; waiting out the backoff before redialing. The
    /// app stays fully functional locally — this is not an error.
    Reconnecting,
}

/// Why the reconnect loop exited (it only exits on cancel or a terminal
/// auth outcome; transient errors never exit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconnectExit {
    /// The cancel signal was raised (the connector was disabled / the app is
    /// shutting down). The clean stop.
    Cancelled,
    /// The relay rejected the credential and it had never worked — a config
    /// problem (bad/blank credential). Stop; do not retry.
    AuthFailed,
    /// The relay rejected a credential that had previously worked — the device
    /// was revoked. The app must re-pair; stop retrying.
    NeedsRepair,
}

/// Run the supervised reconnect loop until cancelled or a terminal auth outcome.
///
/// `config` is redialed each attempt. `cancel` is the stop signal (set to `true`
/// to stop). `on_state` is called on every state transition so the caller can
/// surface the live status; it is a plain `Fn` (no async) — keep it cheap.
pub async fn reconnect_loop<F>(
    config: TunnelConfig,
    mut cancel: watch::Receiver<bool>,
    on_state: F,
) -> ReconnectExit
where
    F: Fn(ConnectionState),
{
    // True once *this credential* has completed at least one handshake. Gates
    // the revoked-vs-bad-credential decision on AuthFailed. Set inside the
    // per-session handshake observer (run_tunnel_with_observer), which fires the
    // instant the relay acknowledges the Hello — so even a session that later
    // dies with a transport error is correctly counted as "the credential
    // worked".
    let mut ever_connected = false;
    let mut backoff = BACKOFF_INITIAL;

    loop {
        if *cancel.borrow() {
            return ReconnectExit::Cancelled;
        }

        on_state(ConnectionState::Connecting);

        // Per-session handshake flag: the observer flips it and we report Online
        // exactly when the relay acks the Hello, not by inferring from runtime.
        let handshook = Arc::new(AtomicBool::new(false));

        // Race the session against the cancel signal so a disable stops a live
        // session promptly (run_tunnel's own teardown aborts in-flight work).
        let outcome = {
            let session = {
                let flag = Arc::clone(&handshook);
                run_tunnel_with_observer(config.clone(), move || {
                    flag.store(true, Ordering::SeqCst);
                })
            };
            tokio::pin!(session);
            loop {
                tokio::select! {
                    result = &mut session => break SessionResult::Ended(result),
                    changed = cancel.changed() => {
                        if changed.is_err() || *cancel.borrow() {
                            break SessionResult::Cancelled;
                        }
                    }
                }
            }
        };

        // If the handshake landed this session, the credential demonstrably
        // works and we are (were) Online — record both before classifying the
        // exit, and reset the backoff on a session that actually connected.
        if handshook.load(Ordering::SeqCst) {
            if !ever_connected {
                ever_connected = true;
            }
            on_state(ConnectionState::Online);
            backoff = BACKOFF_INITIAL;
        }

        match outcome {
            SessionResult::Cancelled => return ReconnectExit::Cancelled,
            SessionResult::Ended(Ok(())) => {
                // Clean close. Redial (the relay may cycle a connection).
                tracing::info!("tunnel: session closed cleanly; reconnecting");
            }
            SessionResult::Ended(Err(TunnelError::HelloRejected(reason))) => {
                use crate::frame::HelloErrReason;
                match reason {
                    HelloErrReason::AuthFailed if ever_connected => {
                        tracing::warn!(
                            "tunnel: credential rejected after a working session — device revoked; needs re-pair"
                        );
                        return ReconnectExit::NeedsRepair;
                    }
                    HelloErrReason::AuthFailed => {
                        tracing::warn!(
                            "tunnel: credential rejected and never worked — bad credential; stopping"
                        );
                        return ReconnectExit::AuthFailed;
                    }
                    HelloErrReason::UnsupportedVersion => {
                        // The relay speaks a different protocol version. Retrying
                        // the same version cannot help, but this is not a
                        // credential problem — back off (a relay rollback/forward
                        // may restore compatibility) rather than demand a re-pair.
                        tracing::warn!("tunnel: relay rejected the protocol version; backing off");
                    }
                }
            }
            SessionResult::Ended(Err(error)) => {
                // Transient: connect/transport/protocol/decode. A successful
                // handshake during this session would have logged "online"; the
                // error here means the session dropped or never connected.
                tracing::warn!(%error, "tunnel: session ended with error; backing off");
            }
        }

        // Back off before the next attempt, but wake immediately on cancel.
        on_state(ConnectionState::Reconnecting);
        let sleep_for = with_jitter(backoff);
        tokio::select! {
            _ = tokio::time::sleep(sleep_for) => {}
            changed = cancel.changed() => {
                if changed.is_err() || *cancel.borrow() {
                    return ReconnectExit::Cancelled;
                }
            }
        }
        backoff = next_backoff(backoff);
    }
}

/// Internal session outcome: the run finished (with a result) or was cancelled.
enum SessionResult {
    Ended(Result<(), TunnelError>),
    Cancelled,
}

/// The next backoff: double, capped at [`BACKOFF_MAX`].
fn next_backoff(current: Duration) -> Duration {
    let doubled = current.saturating_mul(2);
    doubled.min(BACKOFF_MAX)
}

/// Apply ±25% jitter to a backoff so many clients reconnecting after a relay
/// restart spread their retries (thundering-herd avoidance). Deterministic
/// per-call jitter derived from the system clock's nanoseconds — no RNG
/// dependency, and the spread does not need to be cryptographic.
fn with_jitter(base: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // jitter factor in [0.75, 1.25): map subsec_nanos (0..1e9) to [-0.25, 0.25).
    let frac = (nanos as f64) / 1_000_000_000.0; // [0, 1)
    let factor = 0.75 + frac * 0.5; // [0.75, 1.25)
    let millis = (base.as_millis() as f64 * factor) as u64;
    Duration::from_millis(millis.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopback::{InternalBearer, LoopbackTarget};

    fn test_config(relay_url: &str) -> TunnelConfig {
        TunnelConfig {
            relay_url: relay_url.to_string(),
            device_credential: "mdc_dev.secret".to_string(),
            account_id: "acct".to_string(),
            loopback: LoopbackTarget::new("http://127.0.0.1:8765", InternalBearer::new("tok")),
        }
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let mut b = BACKOFF_INITIAL;
        assert_eq!(b, Duration::from_secs(1));
        b = next_backoff(b);
        assert_eq!(b, Duration::from_secs(2));
        b = next_backoff(b);
        assert_eq!(b, Duration::from_secs(4));
        // Walk it up and confirm it never exceeds the cap.
        for _ in 0..20 {
            b = next_backoff(b);
        }
        assert_eq!(b, BACKOFF_MAX);
    }

    #[test]
    fn jitter_stays_within_band() {
        let base = Duration::from_secs(8);
        for _ in 0..50 {
            let j = with_jitter(base);
            assert!(j >= Duration::from_millis(6000), "jitter below -25%: {j:?}");
            assert!(j < Duration::from_millis(10001), "jitter above +25%: {j:?}");
        }
    }

    #[tokio::test]
    async fn cancel_before_first_attempt_exits_clean() {
        let (tx, rx) = watch::channel(true); // already cancelled
        let exit = reconnect_loop(test_config("wss://relay.example/tunnel"), rx, |_| {}).await;
        assert_eq!(exit, ReconnectExit::Cancelled);
        drop(tx);
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_during_backoff_exits_clean() {
        // A cleartext remote relay fails the scheme check inside run_tunnel
        // (TunnelError::Config) → transient → the loop enters backoff. Raising
        // cancel during the backoff must exit promptly.
        let (tx, rx) = watch::channel(false);
        let config = test_config("ws://relay.example/tunnel"); // refused → transient error
        let handle = tokio::spawn(reconnect_loop(config, rx, |_| {}));
        // Let the first attempt fail and enter backoff.
        tokio::time::advance(Duration::from_millis(10)).await;
        tx.send(true).expect("send cancel");
        let exit = handle.await.expect("loop join");
        assert_eq!(exit, ReconnectExit::Cancelled);
    }
}
