//! Peers-file pairing persistence — the shared mechanism both frontends use to
//! persist and reload paired-device tickets outside the in-memory
//! [`PeerDirectory`](crate::endpoint).
//!
//! The [`SyncEngine`]'s peer set is in-memory and process-scoped; it persists no
//! peer list itself. This module is the one place that gives a caller a durable
//! peers store: `{root}/peers`, one [`EndpointTicket`] per line (blank lines and
//! `#`-comments ignored). Both callers own their own data root (the single-writer
//! rule applies per root):
//!
//! - the headless hub (`minutist-hub`) roots it at its `--data-dir`;
//! - the desktop app (`ConnectedSync`) roots it at the app-data base, beside the
//!   device key.
//!
//! [`append`] adds one validated, deduplicated ticket; [`reload_into`] authorises
//! every not-yet-applied ticket against a bound engine. An external writer (an
//! operator's `add-peer`, or an agent appending a line) is picked up by the next
//! [`reload_into`] the caller runs — at startup and, if the caller polls, while
//! running.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use iroh_tickets::endpoint::EndpointTicket;

use crate::{Error, Result, SyncEngine};

/// The peers file path under a data root.
pub fn peers_path(root: &Path) -> PathBuf {
    root.join("peers")
}

/// Parse `{root}/peers`: one pairing ticket per line; blank lines and
/// `#`-prefixed comments are ignored. A missing (or unreadable) file yields an
/// empty list — the absence of any paired peer is the normal initial state.
pub fn read_peer_tickets(root: &Path) -> Vec<String> {
    let contents = match std::fs::read_to_string(peers_path(root)) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

/// Outcome of [`append`]: whether the ticket was newly written or was already
/// present (so a caller can report "registered" vs "already registered").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendOutcome {
    Added,
    AlreadyPresent,
}

/// Validate `ticket` and append it to `{root}/peers` unless it is already
/// present (line-exact, after trimming). A malformed ticket is rejected as
/// [`Error::Protocol`] BEFORE any write, so the file never accumulates a line the
/// engine would later reject; a filesystem failure is [`Error::Io`].
pub fn append(root: &Path, ticket: &str) -> Result<AppendOutcome> {
    let ticket = ticket.trim();
    ticket
        .parse::<EndpointTicket>()
        .map_err(|e| Error::Protocol(format!("not a valid pairing ticket: {e}")))?;

    let path = peers_path(root);
    if read_peer_tickets(root).iter().any(|line| line == ticket) {
        return Ok(AppendOutcome::AlreadyPresent);
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(file, "{ticket}")?;
    Ok(AppendOutcome::Added)
}

/// Read the peers file and authorise every ticket not already applied this run
/// (tracked in `seen`, so a repeated poll neither re-adds nor re-warns). A
/// malformed line is logged and skipped (and marked seen). Returns the number of
/// peers newly authorised on this call.
///
/// `add_peer_from_ticket` is idempotent at the [`PeerDirectory`](crate::endpoint)
/// level, so a ticket also added via a live command is harmless to re-apply; the
/// `seen` set is purely to avoid redundant work and duplicate log lines.
pub fn reload_into(engine: &SyncEngine, root: &Path, seen: &mut HashSet<String>) -> usize {
    let mut added = 0;
    for ticket in read_peer_tickets(root) {
        if !seen.insert(ticket.clone()) {
            continue;
        }
        match engine.add_peer_from_ticket(&ticket) {
            Ok(id) => {
                tracing::info!(target: "sync", peer = %id, "authorised paired peer from peers file");
                added += 1;
            }
            Err(e) => {
                tracing::warn!(target: "sync", error = %e, "skipping malformed peer ticket")
            }
        }
    }
    added
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_ignores_blank_and_comment_lines() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            peers_path(dir.path()),
            "# a comment\n\n  ticketA  \nticketB\n\n# trailing\n",
        )
        .unwrap();
        assert_eq!(read_peer_tickets(dir.path()), vec!["ticketA", "ticketB"]);
    }

    #[test]
    fn read_missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_peer_tickets(dir.path()).is_empty());
    }

    #[test]
    fn append_rejects_malformed_ticket_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let err = append(dir.path(), "not-a-ticket").unwrap_err();
        assert!(matches!(err, Error::Protocol(_)));
        // Nothing was written — the file must not exist after a rejected append.
        assert!(!peers_path(dir.path()).exists());
    }

    #[test]
    fn append_dedups_line_exact() {
        let dir = tempfile::tempdir().unwrap();
        // A syntactically valid EndpointTicket produced from a fixed EndpointAddr,
        // so the parse guard passes without a bound engine.
        let ticket = sample_ticket();
        assert_eq!(
            append(dir.path(), &ticket).unwrap(),
            AppendOutcome::Added
        );
        assert_eq!(
            append(dir.path(), &ticket).unwrap(),
            AppendOutcome::AlreadyPresent
        );
        assert_eq!(read_peer_tickets(dir.path()), vec![ticket]);
    }

    /// A valid `EndpointTicket` string built from a freshly generated key — enough
    /// to pass [`append`]'s parse guard (no network / bound engine needed).
    fn sample_ticket() -> String {
        let id = iroh::SecretKey::generate().public();
        EndpointTicket::new(iroh::EndpointAddr::new(id)).to_string()
    }
}
