//! Confirming that a peer is a device the user owns, and the record of having
//! done so (`planning/DESIGN_sync-encryption.md` §5).
//!
//! Account-mediated discovery hands this device every endpoint the account
//! service publishes, and the ed25519 membership check then admits all of them.
//! That makes the account service a trusted party for authorisation: publish an
//! endpoint it controls and it becomes an authorised peer. Confirmation is what
//! removes that, by requiring a human to agree that a specific endpoint is a
//! device they own before it can hold the account content key.
//!
//! # The code
//!
//! [`safety_code`] is a fingerprint over the two devices' ed25519 identity keys,
//! sorted so both sides compute the same six digits without negotiating who is
//! first. Static, not per-session, and deliberately so. Its job is not to defeat
//! a wire attacker (TLS already does that, and a rogue cannot hold another
//! device's ed25519 secret) but to let the *user* check that the endpoint the
//! directory is offering is a device they actually own. They read the code off
//! the real device's screen; a rogue endpoint has a different identity key, so
//! its code cannot match. Same shape as an SSH host-key fingerprint or a Signal
//! safety number.
//!
//! # The record
//!
//! [`EnrolmentStore`] persists one [`Verdict`] per peer, keyed on the peer's
//! ed25519 key rather than on any address, so it survives a restart and survives
//! the peer being re-advertised at a new address. A refusal is kept, not
//! forgotten: it is the alarm state for an endpoint the directory offered and
//! the user rejected, and re-prompting for it every poll would train the user to
//! dismiss the prompt.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

/// The file under `{app-data}` holding the per-peer verdicts.
const STORE_FILE: &str = "enrolled_peers.json";

/// Domain separator for [`safety_code`]. Changing it invalidates every code a
/// user has ever compared, so it is part of the wire contract in practice even
/// though it never travels.
const SAS_LABEL: &[u8] = b"minutist/sync/sas/v1";

/// Digits in a safety code. Six is the familiar length for a code read aloud or
/// compared across two screens.
const SAS_DIGITS: u32 = 6;

/// What the user decided about a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The user confirmed this is a device they own. It may hold the account
    /// content key, and may receive it from us.
    Confirmed,
    /// The user rejected it. It must never be sent the key, and is not
    /// re-prompted for.
    Refused,
}

/// One peer's decision, as persisted.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    verdict: Verdict,
    /// When the user decided, RFC 3339. Recorded for the UI and for support: a
    /// refusal months ago and one from this morning warrant different reactions.
    /// Not load-bearing, so a missing value does not invalidate the verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    decided_at: Option<String>,
}

/// The six-digit code both devices show, derived from their two ed25519 identity
/// keys.
///
/// Sorting the keys makes the value independent of which side computes it, so
/// neither device has to be told whether it is "first". Returned zero-padded,
/// because a code that renders as `4213` on one screen and `04213` on another is
/// a code users will mis-compare.
pub fn safety_code(a: &[u8; 32], b: &[u8; 32]) -> String {
    let (first, second) = if a <= b { (a, b) } else { (b, a) };
    let mut hasher = Sha256::new();
    hasher.update(SAS_LABEL);
    hasher.update(first);
    hasher.update(second);
    let digest = hasher.finalize();
    // Fold the leading 4 bytes; 10^6 fits comfortably inside a u32, so the
    // modulo bias across 2^32 is far below anything a human comparison notices.
    let n = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let modulus = 10u32.pow(SAS_DIGITS);
    format!("{:0width$}", n % modulus, width = SAS_DIGITS as usize)
}

/// The persisted per-peer verdicts, at `{app-data}/enrolled_peers.json`.
///
/// Loaded whole and rewritten whole: the file holds one small entry per device
/// on the account, so the simple thing is also the right one. An unreadable or
/// corrupt file reads as empty rather than failing the engine: the effect is
/// re-prompting, which is safe, where refusing to start is not.
#[derive(Debug, Clone, Default)]
pub struct EnrolmentStore {
    path: PathBuf,
    records: BTreeMap<String, Record>,
}

impl EnrolmentStore {
    /// Path to the store under `app_data_dir`.
    pub fn path(app_data_dir: &Path) -> PathBuf {
        app_data_dir.join(STORE_FILE)
    }

    /// Load the store, treating an absent, unreadable or corrupt file as empty.
    ///
    /// A corrupt file loses recorded verdicts, which re-prompts the user rather
    /// than admitting anyone: [`Self::verdict`] returns `None` for an unknown
    /// peer and every caller treats `None` as "not confirmed".
    pub fn load(app_data_dir: &Path) -> Self {
        let path = Self::path(app_data_dir);
        let records = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        Self { path, records }
    }

    /// This peer's recorded verdict, or `None` if the user has not decided.
    pub fn verdict(&self, peer: &str) -> Option<Verdict> {
        self.records.get(peer).map(|r| r.verdict)
    }

    /// Whether the user has confirmed this peer. The question every caller
    /// actually asks, so it does not have to remember that `None` and `Refused`
    /// are both "no".
    pub fn is_confirmed(&self, peer: &str) -> bool {
        self.verdict(peer) == Some(Verdict::Confirmed)
    }

    /// Record a decision and persist it.
    ///
    /// `decided_at` is an RFC 3339 timestamp supplied by the caller: this crate
    /// has no clock dependency and does not want one for a display field.
    pub fn record(
        &mut self,
        peer: &str,
        verdict: Verdict,
        decided_at: Option<String>,
    ) -> Result<()> {
        self.records.insert(
            peer.to_string(),
            Record {
                verdict,
                decided_at,
            },
        );
        self.persist()
    }

    /// Every peer the user has decided about, with the decision.
    pub fn all(&self) -> Vec<(String, Verdict)> {
        self.records
            .iter()
            .map(|(peer, r)| (peer.clone(), r.verdict))
            .collect()
    }

    fn persist(&self) -> Result<()> {
        let json = serde_json::to_vec_pretty(&self.records)
            .map_err(|e| Error::Identity(format!("serialising the enrolment store: {e}")))?;
        std::fs::write(&self.path, json).map_err(|e| {
            Error::Identity(format!(
                "writing the enrolment store to {:?}: {e}",
                self.path
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: [u8; 32] = [0x11; 32];
    const B: [u8; 32] = [0x22; 32];

    #[test]
    fn both_sides_compute_the_same_code() {
        assert_eq!(safety_code(&A, &B), safety_code(&B, &A));
    }

    #[test]
    fn a_different_peer_gives_a_different_code() {
        // The whole point: a rogue endpoint holds a different ed25519 key, so
        // the code it produces cannot match the one on the real device.
        let rogue = [0x33; 32];
        assert_ne!(safety_code(&A, &B), safety_code(&A, &rogue));
    }

    #[test]
    fn the_code_is_always_six_digits() {
        // A code that renders shorter on one screen than the other is one users
        // mis-compare, so leading zeros must survive. Sweep enough pairs to hit
        // a low value rather than trusting one sample.
        for i in 0..=255u8 {
            let code = safety_code(&[i; 32], &B);
            assert_eq!(code.len(), 6, "code {code:?} is not six digits");
            assert!(code.chars().all(|c| c.is_ascii_digit()), "code {code:?}");
        }
    }

    #[test]
    fn the_code_is_stable_across_runs() {
        // Both devices derive it independently at different times, so it must be
        // a pure function of the two keys and nothing else.
        assert_eq!(safety_code(&A, &B), safety_code(&A, &B));
    }

    #[test]
    fn an_unknown_peer_has_no_verdict_and_is_not_confirmed() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let store = EnrolmentStore::load(dir.path());
        assert_eq!(store.verdict("unknown"), None);
        assert!(!store.is_confirmed("unknown"));
    }

    #[test]
    fn a_decision_survives_a_reload() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let mut store = EnrolmentStore::load(dir.path());
        store
            .record(
                "peer-a",
                Verdict::Confirmed,
                Some("2026-08-25T00:00:00Z".into()),
            )
            .expect("record");
        store
            .record("peer-b", Verdict::Refused, None)
            .expect("record");

        let reloaded = EnrolmentStore::load(dir.path());
        assert!(reloaded.is_confirmed("peer-a"));
        assert_eq!(reloaded.verdict("peer-b"), Some(Verdict::Refused));
        assert!(
            !reloaded.is_confirmed("peer-b"),
            "a refusal must never read as confirmation"
        );
    }

    #[test]
    fn a_decision_can_be_changed() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let mut store = EnrolmentStore::load(dir.path());
        store
            .record("peer", Verdict::Refused, None)
            .expect("record");
        store
            .record("peer", Verdict::Confirmed, None)
            .expect("record");
        assert!(EnrolmentStore::load(dir.path()).is_confirmed("peer"));
    }

    #[test]
    fn a_corrupt_store_reads_as_empty_rather_than_confirming_anyone() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(EnrolmentStore::path(dir.path()), b"{not json").expect("seed");
        let store = EnrolmentStore::load(dir.path());
        assert!(store.all().is_empty());
        assert!(
            !store.is_confirmed("anyone"),
            "losing the file must re-prompt, never admit"
        );
    }

    #[test]
    fn a_record_written_without_a_timestamp_still_loads() {
        // `decided_at` is a display field; a peer without one is still decided.
        // Guards the serde-default discipline issue 0068 is about.
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(
            EnrolmentStore::path(dir.path()),
            br#"{"peer":{"verdict":"confirmed"}}"#,
        )
        .expect("seed");
        assert!(EnrolmentStore::load(dir.path()).is_confirmed("peer"));
    }
}
