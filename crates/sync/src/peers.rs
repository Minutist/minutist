//! Peers-file pairing persistence: the shared mechanism both frontends use to
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
//! [`reload_into`] the caller runs, at startup and, if the caller polls, while
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
/// empty list: the absence of any paired peer is the normal initial state.
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
/// [`Error::Protocol`] before any write, so the file never accumulates a line the
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

/// Outcome of [`remove`]: whether a matching ticket line was dropped or the
/// ticket was not in the file (a missing file counts as [`RemoveOutcome::NotPresent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveOutcome {
    Removed,
    NotPresent,
}

/// Remove `ticket` from `{root}/peers`, the inverse of [`append`]. Every line
/// whose trimmed content equals `ticket` is dropped; all other lines
/// (`#`-comments, blank lines, other peers) are preserved verbatim, and the
/// rewrite is atomic (temp-file + rename via the shared
/// [`minutist_common::fs::write_atomic`]), so a crash never leaves the peers
/// file truncated. A missing file or an absent ticket yields
/// [`RemoveOutcome::NotPresent`] without writing.
///
/// This is the file side of an explicit, consumer-driven unpair (a Settings
/// "remove device" action): dropping a peer's ticket here, together with
/// removing it from the engine's in-memory `PeerDirectory`, un-pairs it
/// durably so the next [`reload_into`] does not re-authorise it. It differs
/// from the failed-dial path, where a transiently-offline paired device is
/// suppressed with backoff and re-dialled when it returns, never silently
/// unpaired. Account-sourced peers never live in this file, so they are
/// untouched here (their removal is the engine's account-reconcile).
pub fn remove(root: &Path, ticket: &str) -> Result<RemoveOutcome> {
    let ticket = ticket.trim();
    let path = peers_path(root);
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(RemoveOutcome::NotPresent),
        Err(e) => return Err(Error::Io(e)),
    };

    let mut removed = false;
    let kept: Vec<&str> = contents
        .lines()
        .filter(|line| {
            if line.trim() == ticket {
                removed = true;
                false
            } else {
                true
            }
        })
        .collect();

    if !removed {
        return Ok(RemoveOutcome::NotPresent);
    }

    // Preserve a single trailing newline when content remains; an emptied file is
    // written empty (it reads back as no peers, same as an absent file).
    let mut out = kept.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    minutist_common::fs::write_atomic(&path, out.as_bytes())
        .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
    Ok(RemoveOutcome::Removed)
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
        // Nothing was written: the file must not exist after a rejected append.
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

    #[test]
    fn remove_drops_only_the_named_ticket() {
        let dir = tempfile::tempdir().unwrap();
        let a = sample_ticket();
        let b = sample_ticket();
        append(dir.path(), &a).unwrap();
        append(dir.path(), &b).unwrap();
        assert_eq!(remove(dir.path(), &a).unwrap(), RemoveOutcome::Removed);
        assert_eq!(read_peer_tickets(dir.path()), vec![b]);
    }

    #[test]
    fn remove_absent_ticket_is_not_present_and_leaves_the_file_intact() {
        let dir = tempfile::tempdir().unwrap();
        let a = sample_ticket();
        append(dir.path(), &a).unwrap();
        assert_eq!(
            remove(dir.path(), &sample_ticket()).unwrap(),
            RemoveOutcome::NotPresent
        );
        assert_eq!(read_peer_tickets(dir.path()), vec![a]);
    }

    #[test]
    fn remove_on_missing_file_is_not_present() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            remove(dir.path(), &sample_ticket()).unwrap(),
            RemoveOutcome::NotPresent
        );
    }

    #[test]
    fn remove_preserves_comments_blank_lines_and_other_peers() {
        let dir = tempfile::tempdir().unwrap();
        let a = sample_ticket();
        let b = sample_ticket();
        std::fs::write(peers_path(dir.path()), format!("# header\n{a}\n\n{b}\n")).unwrap();
        assert_eq!(remove(dir.path(), &a).unwrap(), RemoveOutcome::Removed);
        // The other peer still parses out; the comment survives in the raw file.
        assert_eq!(read_peer_tickets(dir.path()), vec![b.clone()]);
        let raw = std::fs::read_to_string(peers_path(dir.path())).unwrap();
        assert!(raw.contains("# header"), "comment must be preserved: {raw:?}");
        assert!(!raw.contains(&a), "removed ticket must be gone: {raw:?}");
    }

    #[test]
    fn remove_last_ticket_leaves_an_empty_readable_file() {
        let dir = tempfile::tempdir().unwrap();
        let a = sample_ticket();
        append(dir.path(), &a).unwrap();
        assert_eq!(remove(dir.path(), &a).unwrap(), RemoveOutcome::Removed);
        assert!(read_peer_tickets(dir.path()).is_empty());
    }

    #[test]
    fn remove_leaves_no_tmp_residue() {
        let dir = tempfile::tempdir().unwrap();
        let a = sample_ticket();
        let b = sample_ticket();
        append(dir.path(), &a).unwrap();
        append(dir.path(), &b).unwrap();
        remove(dir.path(), &a).unwrap();
        let residue: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(residue.is_empty(), "expected no .tmp residue, found: {residue:?}");
    }

    /// A valid `EndpointTicket` string built from a freshly generated key: enough
    /// to pass [`append`]'s parse guard (no network / bound engine needed).
    fn sample_ticket() -> String {
        let id = iroh::SecretKey::generate().public();
        EndpointTicket::new(iroh::EndpointAddr::new(id)).to_string()
    }
}
