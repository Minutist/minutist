//! Precedence merge for the processing-lifecycle field of a meeting's
//! `metadata.json`.
//!
//! The inbound (synced) lifecycle apply path must not write a peer-advertised
//! state verbatim: a stale advertisement (e.g. a capture device still announcing
//! `PendingProcessing` after a host has `Claimed` or `Processed` the meeting, or
//! a 5-minute discovery sweep replaying an old view) would walk the local state
//! *backwards* under last-writer-wins. [`merge_processing`] resolves an inbound
//! state against the current one by a deterministic precedence so the apply is
//! convergent. It lives in this leaf crate so every metadata.json writer that can
//! reach `notes-crdt` — `persistence` and the mobile sync FFI — shares one rule.
//!
//! Convergence: for any two states authored by DISTINCT hosts the result is
//! order-independent (both `Claimed` and `Processed` tie-break on the lowest
//! `HostRef`, never on a clock — §7 decision 2). A same-host re-advertisement is
//! idempotent (for `Claimed`, the later lease wins so an out-of-order replay of an
//! older lease cannot shorten reap timing).

use chrono::DateTime;
use minutist_common::ProcessingLifecycle;

/// Merge an inbound (peer-advertised) `ProcessingLifecycle` into the `current`
/// local one, returning the winner by precedence — so applying a peer's state can
/// never walk the local state backwards.
///
/// Precedence, highest wins: `Processed` > `Claimed` > `PendingProcessing` >
/// `Local`. Ties within the two host-stamped states are broken on the **lowest
/// `HostRef`** (lexicographic) — deterministic and clock-independent, per
/// `planning/DESIGN_processing-lifecycle.md` §7 decision 2 (never tie-break on
/// `claimed_at`/`lease_expires_at` for the cross-host winner; lease expiry is the
/// election loop's reap concern). A same-`HostRef` `Claimed` re-advertisement (a
/// lease refresh) keeps the LATER `lease_expires_at` (same-clock comparison, not a
/// cross-host tiebreak). At equal precedence among the unit states the incoming
/// value is taken (idempotent for a same-state re-advertisement).
///
/// This is the SYNCED/inbound merge only. A local authoritative write — a host
/// setting its OWN state (a claim, a renewal, `Processed` on completion) — is
/// unconditional and does not pass through here.
///
/// Deferred (D6): a deliberate reprocess request (`Processed` →
/// `PendingProcessing`) is a *backwards* move this precedence correctly rejects;
/// distinguishing it from a stale replay needs a monotonic epoch on the state,
/// which lands with that feature, not here.
pub fn merge_processing(
    current: &ProcessingLifecycle,
    incoming: ProcessingLifecycle,
) -> ProcessingLifecycle {
    use ProcessingLifecycle::{Claimed, Processed};

    // Two competing claims: lowest `HostRef` wins (deterministic, clock-
    // independent). For the same host (a re-advertised claim / lease refresh),
    // keep the LATER lease so an out-of-order replay of an older lease cannot
    // shorten reap timing.
    if let (Claimed { claim: cur }, Claimed { claim: inc }) = (current, &incoming) {
        return match inc.host.0.cmp(&cur.host.0) {
            std::cmp::Ordering::Less => incoming,
            std::cmp::Ordering::Greater => current.clone(),
            std::cmp::Ordering::Equal => {
                if lease_ge(&inc.lease_expires_at, &cur.lease_expires_at) {
                    incoming
                } else {
                    current.clone()
                }
            }
        };
    }

    // Two completed states: lowest `processed_by` HostRef wins — convergent and
    // matching the claim tiebreak + the Artifacts authority rule (so the
    // lifecycle edge and the on-disk derived bytes name one authoritative host).
    if let (Processed { processed_by: cur, .. }, Processed { processed_by: inc, .. }) =
        (current, &incoming)
    {
        return if inc.0 <= cur.0 {
            incoming
        } else {
            current.clone()
        };
    }

    if rank(&incoming) >= rank(current) {
        incoming
    } else {
        current.clone()
    }
}

/// Precedence rank: higher wins. The two host-stamped states (`Claimed`,
/// `Processed`) are disambiguated by `HostRef` in [`merge_processing`]; this rank
/// orders the four variants against each other.
fn rank(p: &ProcessingLifecycle) -> u8 {
    match p {
        ProcessingLifecycle::Local => 0,
        ProcessingLifecycle::PendingProcessing => 1,
        ProcessingLifecycle::Claimed { .. } => 2,
        ProcessingLifecycle::Processed { .. } => 3,
    }
}

/// `a >= b` for two RFC 3339 lease timestamps, parsed to instants — robust to
/// mixed fractional precision or offsets that a raw string compare would sort
/// wrong. A malformed timestamp is treated as NOT newer, so a parse glitch can
/// only keep the current lease, never shorten it.
fn lease_ge(a: &str, b: &str) -> bool {
    match (DateTime::parse_from_rfc3339(a), DateTime::parse_from_rfc3339(b)) {
        (Ok(a), Ok(b)) => a >= b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::{HostRef, ProcessingClaim};

    fn claim_with(host: &str, lease: &str) -> ProcessingLifecycle {
        ProcessingLifecycle::Claimed {
            claim: ProcessingClaim {
                host: HostRef(host.to_string()),
                claimed_at: "2026-06-30T10:00:00Z".to_string(),
                lease_expires_at: lease.to_string(),
            },
        }
    }

    fn claim(host: &str) -> ProcessingLifecycle {
        claim_with(host, "2026-06-30T10:30:00Z")
    }

    fn processed(host: &str) -> ProcessingLifecycle {
        ProcessingLifecycle::Processed {
            processed_by: HostRef(host.to_string()),
            at: "2026-06-30T10:25:00Z".to_string(),
        }
    }

    /// The headline invariant: a stale inbound advertisement never walks the
    /// local state backwards.
    #[test]
    fn inbound_never_regresses_local() {
        use ProcessingLifecycle::*;
        assert_eq!(merge_processing(&claim("a"), PendingProcessing), claim("a"));
        assert_eq!(merge_processing(&processed("a"), PendingProcessing), processed("a"));
        assert_eq!(merge_processing(&claim("a"), Local), claim("a"));
        assert_eq!(merge_processing(&processed("a"), Local), processed("a"));
        assert_eq!(merge_processing(&PendingProcessing, Local), PendingProcessing);
        assert_eq!(merge_processing(&processed("a"), claim("b")), processed("a"));
    }

    /// Forward transitions and equal-precedence idempotence.
    #[test]
    fn forward_transitions_and_idempotence() {
        use ProcessingLifecycle::*;
        assert_eq!(merge_processing(&Local, PendingProcessing), PendingProcessing);
        assert_eq!(merge_processing(&PendingProcessing, claim("a")), claim("a"));
        assert_eq!(merge_processing(&claim("a"), processed("a")), processed("a"));
        assert_eq!(merge_processing(&Local, Local), Local);
        assert_eq!(merge_processing(&PendingProcessing, PendingProcessing), PendingProcessing);
    }

    /// Two competing claims: lowest `HostRef` wins; the SAME host keeps the later
    /// lease (a refresh wins; an out-of-order older-lease replay does not).
    #[test]
    fn claimed_tiebreak_is_lowest_hostref_then_later_lease() {
        assert_eq!(merge_processing(&claim("b"), claim("a")), claim("a"));
        assert_eq!(merge_processing(&claim("a"), claim("b")), claim("a"));
        // Same host, NEWER incoming lease → take the refresh.
        let newer = claim_with("a", "2026-06-30T11:00:00Z");
        assert_eq!(merge_processing(&claim("a"), newer.clone()), newer);
        // Same host, OLDER incoming lease (out-of-order replay) → keep current.
        let current_fresh = claim_with("a", "2026-06-30T11:00:00Z");
        let older_replay = claim_with("a", "2026-06-30T10:30:00Z");
        assert_eq!(merge_processing(&current_fresh, older_replay), current_fresh);
    }

    /// Two distinct completed states: lowest `processed_by` HostRef wins —
    /// convergent (order-independent), not last-writer-wins.
    #[test]
    fn processed_tiebreak_is_lowest_hostref_convergent() {
        assert_eq!(merge_processing(&processed("b"), processed("a")), processed("a"));
        assert_eq!(merge_processing(&processed("a"), processed("b")), processed("a"));
    }
}
