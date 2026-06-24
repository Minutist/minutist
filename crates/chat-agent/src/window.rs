//! Context budget + the sliding-window trim policy (§6.2, "until context
//! full").
//!
//! Pure functions — no llama.cpp state — so the policy is unit-tested without a
//! model. The DRIVER (ipc-bridge, a later phase) owns the `Vec<ChatMessage>`
//! history and APPLIES this policy before each engine call; it lives here as a
//! pure helper so the policy is co-located with the engine that documents the
//! "until context full" behaviour, and so the same arithmetic is exercised by
//! the default test suite.
//!
//! # Policy (the Phase 9 recommendation)
//!
//! - **Pin the head:** message `0` is the system prompt (turn 0 — persona +
//!   meeting context + the tool list). It is never evicted.
//! - **Re-measure each turn:** the driver re-tokenises the whole windowed
//!   history (the authoritative length — BOS/turn markers make an incremental
//!   running counter unreliable) and checks `prompt_tokens + max_tokens <=
//!   n_ctx - reserve` via [`fits_budget`].
//! - **Evict oldest non-pinned first:** on overflow, drop the oldest message
//!   AFTER the pinned head, one at a time, until it fits. The driver should
//!   evict a user+assistant(+tool) group together to keep template alternation
//!   valid; [`trim_to_budget`] returns the count to drop so the driver can snap
//!   that to a group boundary.
//! - **Hard floor:** if even `[pinned head] + [the most-recent message] +
//!   max_tokens > n_ctx`, the single turn is genuinely too large — the driver
//!   rejects it with `AppError::InvalidInput { context: "message too large for
//!   context window" }` ([`HARD_FLOOR_REJECT`]). Eviction cannot help; this is
//!   not evictable history.

/// The driver's reject message when a single turn cannot fit even with the
/// whole evictable history dropped (the hard floor). Surfaced as
/// `AppError::InvalidInput { context: HARD_FLOOR_REJECT.into() }` — there is no
/// `AppError::ContextOverflow` variant (§0 correction).
pub const HARD_FLOOR_REJECT: &str = "message too large for context window";

/// The result of applying the sliding-window trim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimOutcome {
    /// The windowed history fits the budget; drop the first `drop_after_head`
    /// non-pinned messages (those immediately after the pinned head). `0` means
    /// nothing was trimmed.
    Fits { drop_after_head: usize },
    /// Even after dropping ALL evictable history, the pinned head plus the
    /// most-recent message plus the generation reserve do not fit. The driver
    /// rejects the turn ([`HARD_FLOOR_REJECT`]).
    HardFloor,
}

/// Whether `prompt_tokens + max_tokens + reserve` fits `n_ctx`.
///
/// Reserves both the generation headroom (`max_tokens`, like
/// `summariser::check_context_budget`) AND a fixed `reserve` for template
/// markers the running count might miss. Saturating to stay total.
pub fn fits_budget(prompt_tokens: usize, max_tokens: usize, reserve: usize, n_ctx: usize) -> bool {
    prompt_tokens
        .saturating_add(max_tokens)
        .saturating_add(reserve)
        <= n_ctx
}

/// Decide the sliding-window trim for a history whose per-message token
/// estimates are `token_lens` (index `0` = the pinned head / system prompt).
///
/// Returns how many messages immediately AFTER the pinned head to drop so the
/// remaining windowed history fits `prompt_tokens(remaining) + max_tokens +
/// reserve <= n_ctx`, or [`TrimOutcome::HardFloor`] when no amount of eviction
/// makes the most-recent message fit alongside the pinned head.
///
/// Pure: takes the per-message lengths the driver measured, returns a count.
/// The driver applies it (and snaps the count to a user+assistant group
/// boundary so alternation stays valid). An empty or single-message history
/// never trims (`drop_after_head: 0`); the hard-floor check still applies to a
/// single over-budget message.
pub fn trim_to_budget(
    token_lens: &[usize],
    max_tokens: usize,
    reserve: usize,
    n_ctx: usize,
) -> TrimOutcome {
    let total: usize = token_lens.iter().copied().fold(0, usize::saturating_add);
    if fits_budget(total, max_tokens, reserve, n_ctx) {
        return TrimOutcome::Fits { drop_after_head: 0 };
    }

    // The pinned head (index 0) is never dropped. With no head at all there is
    // nothing to pin; treat the whole slice as evictable-after-an-empty-head.
    let head_len = token_lens.first().copied().unwrap_or(0);
    let n = token_lens.len();
    if n <= 1 {
        // A single (head-only) message that does not fit is a hard floor.
        return TrimOutcome::HardFloor;
    }

    // The minimal surviving window is [head] + [most-recent message]. If that
    // already overflows, eviction cannot help.
    let last_len = token_lens[n - 1];
    if !fits_budget(
        head_len.saturating_add(last_len),
        max_tokens,
        reserve,
        n_ctx,
    ) {
        return TrimOutcome::HardFloor;
    }

    // Drop the oldest non-pinned messages (indices 1.., front-first) until the
    // remaining prompt fits.
    let mut remaining = total;
    for (dropped, &len) in token_lens[1..].iter().enumerate() {
        if fits_budget(remaining, max_tokens, reserve, n_ctx) {
            return TrimOutcome::Fits {
                drop_after_head: dropped,
            };
        }
        // The last evictable message (index n-1) is the most-recent turn and is
        // never dropped — it is part of the minimal window guaranteed to fit by
        // the hard-floor check above.
        if dropped + 1 == n - 1 {
            break;
        }
        remaining = remaining.saturating_sub(len);
    }

    // After dropping everything evictable, the [head + last] window fits (the
    // hard-floor check passed), so report the maximal drop.
    TrimOutcome::Fits {
        drop_after_head: n - 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_budget_reserves_generation_and_markers() {
        // 100 prompt + 50 gen + 10 reserve = 160 <= 200 → fits.
        assert!(fits_budget(100, 50, 10, 200));
        // 160 prompt + 50 + 10 = 220 > 200 → does not fit.
        assert!(!fits_budget(160, 50, 10, 200));
        // Exact boundary fits.
        assert!(fits_budget(140, 50, 10, 200));
    }

    #[test]
    fn fits_budget_saturates_on_absurd_inputs() {
        // Must not panic on overflow; an absurd prompt+reserve saturates to
        // `usize::MAX` and does not fit a finite window.
        assert!(!fits_budget(usize::MAX, usize::MAX, usize::MAX, 1_000));
    }

    #[test]
    fn trim_no_op_when_history_fits() {
        // head=20, two turns of 20 each; 60 + 50 + 10 = 120 <= 200.
        let lens = [20, 20, 20];
        assert_eq!(
            trim_to_budget(&lens, 50, 10, 200),
            TrimOutcome::Fits { drop_after_head: 0 }
        );
    }

    #[test]
    fn trim_evicts_oldest_non_pinned_until_it_fits() {
        // head=20 (pinned), then four 40-token turns = 180 total.
        // budget: max=50, reserve=10, n_ctx=200 → need prompt <= 140.
        // Dropping the single oldest non-pinned (index 1) leaves 20+40+40+40 =
        // 140 == 140 (fits) — the policy drops the MINIMUM needed.
        let lens = [20, 40, 40, 40, 40];
        match trim_to_budget(&lens, 50, 10, 200) {
            TrimOutcome::Fits { drop_after_head } => {
                assert!(
                    drop_after_head >= 1,
                    "must evict at least the oldest non-pinned; dropped {drop_after_head}"
                );
                // Verify the surviving prompt actually fits, and that the policy
                // dropped no more than necessary.
                let surviving: usize = lens[0] + lens[1 + drop_after_head..].iter().sum::<usize>();
                assert!(fits_budget(surviving, 50, 10, 200), "survivors must fit");
                let one_fewer: usize = lens[0]
                    + lens[1 + drop_after_head.saturating_sub(1)..]
                        .iter()
                        .sum::<usize>();
                if drop_after_head > 1 {
                    assert!(
                        !fits_budget(one_fewer, 50, 10, 200),
                        "policy must drop the minimum; {drop_after_head} was more than needed"
                    );
                }
            }
            other => panic!("expected Fits, got {other:?}"),
        }
    }

    #[test]
    fn trim_pins_the_head_and_keeps_the_most_recent() {
        // The head and the most-recent message are never dropped.
        let lens = [30, 40, 40, 40];
        match trim_to_budget(&lens, 50, 10, 200) {
            // Surviving = head(30) + tail. Dropping the two middle leaves
            // 30 + 40 = 70 <= 140. drop_after_head must be <= n-2 = 2.
            TrimOutcome::Fits { drop_after_head } => {
                assert!(drop_after_head <= 2, "never drops head or most-recent");
            }
            other => panic!("expected Fits, got {other:?}"),
        }
    }

    #[test]
    fn trim_hard_floor_when_single_turn_too_large() {
        // head(50) + most-recent(120) + max(50) + reserve(10) = 230 > 200.
        // No eviction of the (empty) middle helps → HardFloor.
        let lens = [50, 120];
        assert_eq!(trim_to_budget(&lens, 50, 10, 200), TrimOutcome::HardFloor);
    }

    #[test]
    fn trim_hard_floor_when_head_alone_too_large() {
        // A single over-budget head message is a hard floor.
        let lens = [400];
        assert_eq!(trim_to_budget(&lens, 50, 10, 200), TrimOutcome::HardFloor);
    }

    #[test]
    fn trim_empty_history_fits_trivially() {
        assert_eq!(
            trim_to_budget(&[], 50, 10, 200),
            TrimOutcome::Fits { drop_after_head: 0 }
        );
    }
}
