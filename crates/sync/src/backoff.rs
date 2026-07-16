//! Failed-dial backoff for out-of-band peers.
//!
//! Decouples "stop dialling an unreachable peer" from "re-add it from the
//! directory": [`BackoffRegistry`] tracks per-peer dial failures and, once a
//! threshold is crossed, suppresses further dials for an exponentially growing
//! window — independent of whether the peer is still present in a
//! [`crate::address_lookup::PeerDirectory`] (account-reconcile add/remove is a
//! separate concern; the two must never fight over the same peer).
//!
//! The dial site ([`crate::endpoint::SyncEngine::dial`]) feeds every outcome via
//! [`BackoffRegistry::on_dial_outcome`], so all consumers' dials — desktop,
//! headless, and the phone's syncs (which flow through the same engine dial) —
//! participate automatically. `is_suppressed` is read by
//! [`crate::account::RefreshSink::is_suppressed`] (gating the loop's
//! first-contact dial-kick) and, via [`crate::endpoint::SyncEngine::is_suppressed`],
//! by `sync-ffi` for the phone's own TS discovery loop.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The failure threshold and exponential-backoff shape a [`BackoffRegistry`]
/// applies. Values are policy owned by the consumer (desktop/headless) and
/// passed in at [`crate::SyncEngine`] construction via
/// [`crate::SyncConfig::backoff_policy`]; the registry only implements the
/// counting/suppress/clear mechanism.
#[derive(Debug, Clone)]
pub struct BackoffPolicy {
    /// Consecutive failures before a peer is suppressed.
    pub max_fails: u32,
    /// The backoff window after crossing `max_fails`; doubles per failure beyond
    /// that, saturating and capped at [`Self::cap`].
    pub base: Duration,
    /// The maximum backoff window, regardless of how many failures accumulate.
    pub cap: Duration,
}

impl Default for BackoffPolicy {
    /// Placeholder values — the desktop/headless owner overrides these via
    /// [`crate::SyncConfig::with_backoff_policy`] once the product-level policy
    /// is set.
    fn default() -> Self {
        Self {
            max_fails: 3,
            base: Duration::from_secs(30),
            cap: Duration::from_secs(3600),
        }
    }
}

/// Per-peer backoff state. `retry_after` is only set once `fails` crosses
/// `policy.max_fails` — below the threshold a peer is never suppressed, however
/// many times it has failed.
#[derive(Debug, Clone, Default)]
struct State {
    fails: u32,
    retry_after: Option<Instant>,
}

/// Feed every dial's outcome; consult whether a peer is currently suppressed.
/// Internally shared (cheap to clone) so it can be held alongside
/// [`crate::address_lookup::PeerDirectory`] on [`crate::SyncEngine`] and
/// referenced from a production [`crate::account::RefreshSink`] impl.
#[derive(Debug, Clone)]
pub struct BackoffRegistry {
    states: Arc<Mutex<HashMap<String, State>>>,
    policy: BackoffPolicy,
}

impl BackoffRegistry {
    /// A fresh, empty registry applying `policy`.
    pub fn new(policy: BackoffPolicy) -> Self {
        Self {
            states: Arc::new(Mutex::new(HashMap::new())),
            policy,
        }
    }

    /// Record a dial's outcome for `endpoint_id` (hex).
    ///
    /// A success clears the entry entirely (fail count and any suppression
    /// reset). A failure increments the fail count; once it reaches
    /// [`BackoffPolicy::max_fails`], `retry_after` is set to `now +
    /// backoff(fails)` — recomputed (and extended) on every further failure
    /// while still suppressed.
    pub fn on_dial_outcome(&self, endpoint_id: &str, success: bool) {
        let mut states = self.states.lock().expect("backoff registry poisoned");
        if success {
            states.remove(endpoint_id);
            return;
        }
        let state = states.entry(endpoint_id.to_string()).or_default();
        state.fails = state.fails.saturating_add(1);
        if state.fails >= self.policy.max_fails {
            state.retry_after = Some(Instant::now() + backoff_for(state.fails, &self.policy));
        }
    }

    /// Whether `endpoint_id` is currently suppressed: an entry exists with a
    /// `retry_after` still in the future. A peer below the failure threshold, or
    /// one whose backoff window has elapsed, is not suppressed.
    pub fn is_suppressed(&self, endpoint_id: &str) -> bool {
        let states = self.states.lock().expect("backoff registry poisoned");
        match states.get(endpoint_id).and_then(|s| s.retry_after) {
            Some(retry_after) => Instant::now() < retry_after,
            None => false,
        }
    }
}

/// The exponential-backoff window for `fails` consecutive failures:
/// `base * 2^(fails - max_fails)`, saturating on overflow and capped at
/// `policy.cap`. Crate-private so the growth curve is directly unit-testable
/// without going through the full `on_dial_outcome` timing dance.
fn backoff_for(fails: u32, policy: &BackoffPolicy) -> Duration {
    let exponent = fails.saturating_sub(policy.max_fails);
    // `checked_mul` guards `2^exponent` overflowing u32; either overflow path
    // (the shift or the duration multiply) saturates to the cap rather than
    // panicking or wrapping.
    let multiplier = 1u32.checked_shl(exponent).unwrap_or(u32::MAX);
    policy
        .base
        .checked_mul(multiplier)
        .unwrap_or(policy.cap)
        .min(policy.cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> BackoffPolicy {
        BackoffPolicy {
            max_fails: 3,
            base: Duration::from_secs(30),
            cap: Duration::from_secs(3600),
        }
    }

    #[test]
    fn below_threshold_is_not_suppressed() {
        let reg = BackoffRegistry::new(policy());
        reg.on_dial_outcome("peer", false);
        reg.on_dial_outcome("peer", false);
        assert!(
            !reg.is_suppressed("peer"),
            "two failures below max_fails=3 must not suppress"
        );
    }

    #[test]
    fn crossing_threshold_suppresses() {
        let reg = BackoffRegistry::new(policy());
        reg.on_dial_outcome("peer", false);
        reg.on_dial_outcome("peer", false);
        reg.on_dial_outcome("peer", false);
        assert!(
            reg.is_suppressed("peer"),
            "the third failure crosses max_fails=3 and must suppress"
        );
    }

    #[test]
    fn success_clears_suppression() {
        let reg = BackoffRegistry::new(policy());
        for _ in 0..3 {
            reg.on_dial_outcome("peer", false);
        }
        assert!(reg.is_suppressed("peer"));
        reg.on_dial_outcome("peer", true);
        assert!(
            !reg.is_suppressed("peer"),
            "a success must clear the fail count and any suppression"
        );
    }

    #[test]
    fn unknown_peer_is_not_suppressed() {
        let reg = BackoffRegistry::new(policy());
        assert!(!reg.is_suppressed("never-seen"));
    }

    #[tokio::test]
    async fn suppression_expires_after_the_backoff_window_elapses() {
        let short = BackoffPolicy {
            max_fails: 1,
            base: Duration::from_millis(10),
            cap: Duration::from_secs(3600),
        };
        let reg = BackoffRegistry::new(short);
        reg.on_dial_outcome("peer", false);
        assert!(reg.is_suppressed("peer"), "one failure crosses max_fails=1");

        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !reg.is_suppressed("peer"),
            "suppression must expire once retry_after elapses"
        );
    }

    #[test]
    fn backoff_grows_exponentially_and_saturates_at_the_cap() {
        let p = policy();
        // At the threshold, the window is exactly `base`.
        assert_eq!(backoff_for(3, &p), Duration::from_secs(30));
        // Each failure beyond the threshold doubles it.
        assert_eq!(backoff_for(4, &p), Duration::from_secs(60));
        assert_eq!(backoff_for(5, &p), Duration::from_secs(120));
        // A huge fail count must saturate at the cap, not overflow/panic.
        assert_eq!(backoff_for(1_000, &p), p.cap);
    }
}
