//! `get_diagnostic_report` — assemble + redact the snapshot the no-telemetry
//! "Report a problem" flow pre-fills into a GitHub issue (issue #0014).
//!
//! Decision O1/U6: no telemetry. This command never sends anything; it builds a
//! [`DiagnosticReport`] the webview turns into a pre-filled issue URL the user
//! reviews and submits from their own browser. Log-excerpt / backtrace
//! redaction is owned HERE (where the data is read), per the architecture note.
//!
//! Sources, in priority order:
//! - `{logs}/last-crash.txt` (written by `app-main`'s panic hook) — when present
//!   it supplies the backtrace and the recent-lines excerpt, and the error class
//!   is `"panic"`. The panic hook already redacted it; we redact again
//!   defensively.
//! - otherwise, the tail of the rolling `minutist.log*` file — error class
//!   `"diagnostic report"`, no backtrace.
//!
//! Privacy: every free-text field is passed through [`redact`] (meeting-id-UUID
//! strip). By the #0014 audit no meeting *content* is logged at any level, so
//! the excerpt holds none; the strip is the boundary guard for paths.

use std::path::Path;

use minutist_common::DiagnosticReport;
use tauri::State;

use crate::{error::IpcError, IpcState};

/// How many trailing log lines to attach when there is no crash file.
const LOG_TAIL_LINES: usize = 200;

/// Strip meeting-id UUIDs (`8-4-4-4-12` hex) from `text`, each → `<redacted-id>`.
///
/// Mirrors the webview's `redactMeetingPaths` and `app-main`'s `crash::redact`
/// (each crate owns its own copy — `ipc-bridge` cannot import `app-main`). A
/// meeting id in a logged path is the one piece of meeting identity that can
/// leak through an otherwise content-free log line. No regex dependency.
pub fn redact(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = uuid_at(bytes, i) {
            out.push_str("<redacted-id>");
            i = end;
        } else {
            let ch = text[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// If a UUID starts at `bytes[i]`, return the index past its end; else `None`.
fn uuid_at(bytes: &[u8], i: usize) -> Option<usize> {
    const GROUPS: [usize; 5] = [8, 4, 4, 4, 12];
    let mut pos = i;
    for (g, &len) in GROUPS.iter().enumerate() {
        for _ in 0..len {
            let b = *bytes.get(pos)?;
            if !b.is_ascii_hexdigit() {
                return None;
            }
            pos += 1;
        }
        if g < GROUPS.len() - 1 {
            if bytes.get(pos) != Some(&b'-') {
                return None;
            }
            pos += 1;
        }
    }
    Some(pos)
}

/// Resolve the newest rolling log file under `logs_dir`.
///
/// `tracing_appender::rolling::daily(.., "minutist.log")` writes
/// `minutist.log.YYYY-MM-DD`; pick the lexicographically greatest such name
/// (dates sort chronologically). Returns `None` when none exist.
fn newest_log_file(logs_dir: &Path) -> Option<std::path::PathBuf> {
    let mut newest: Option<(String, std::path::PathBuf)> = None;
    for entry in std::fs::read_dir(logs_dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("minutist.log") {
            continue;
        }
        match &newest {
            Some((best, _)) if *best >= name => {}
            _ => newest = Some((name, entry.path())),
        }
    }
    newest.map(|(_, path)| path)
}

/// Read the last `max_lines` lines of `path`, joined by `\n`. Best-effort: an
/// unreadable file yields an empty string.
fn tail_lines(path: &Path, max_lines: usize) -> String {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let lines: Vec<&str> = contents.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

/// Split a `last-crash.txt` body into `(backtrace, recent_lines)`.
///
/// The crash file (see `app-main`'s `crash::render_crash_report`) has labelled
/// `Backtrace:` and `Recent log lines:` sections. Pull each out; if a label is
/// missing, fall back to attaching the whole file as the excerpt. Pure so it is
/// unit-tested.
pub fn parse_crash_file(body: &str) -> (Option<String>, String) {
    let backtrace = section(body, "Backtrace:", "Recent log lines:");
    let recent = section(body, "Recent log lines:", "");
    match (backtrace, recent) {
        (bt, Some(lines)) => (bt, lines),
        // No "Recent log lines:" label — attach the whole file as the excerpt.
        (bt, None) => (bt, body.trim().to_string()),
    }
}

/// Extract the text between `start_label` and `end_label` (or end of string when
/// `end_label` is empty / not found). `None` if `start_label` is absent.
///
/// Splits on the FIRST occurrence of each label. A log line that itself contained
/// the literal label text would mis-section the report; the failure mode is benign
/// (a still-redacted, slightly mis-split excerpt), and the labels are fixed crash-
/// file headers we emit, so first-occurrence is sufficient here.
fn section(body: &str, start_label: &str, end_label: &str) -> Option<String> {
    let start = body.find(start_label)? + start_label.len();
    let rest = &body[start..];
    let slice = if end_label.is_empty() {
        rest
    } else {
        match rest.find(end_label) {
            Some(e) => &rest[..e],
            None => rest,
        }
    };
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Assemble the redacted [`DiagnosticReport`] from its already-gathered parts.
///
/// PURE (no I/O) so the redaction + field mapping is unit-tested without a Tauri
/// app or filesystem. Every free-text field is redacted here.
pub fn assemble_report(
    app_version: String,
    platform: String,
    gpu: String,
    error_class: &str,
    log_excerpt: &str,
    backtrace: Option<&str>,
) -> DiagnosticReport {
    DiagnosticReport {
        app_version,
        platform,
        gpu,
        error_class: error_class.to_string(),
        log_excerpt: redact(log_excerpt),
        backtrace: backtrace.map(redact),
    }
}

/// Best-effort GPU string for the report: the resolved plan summary for the
/// current settings, or `"cpu"` when no usable GPU. Reuses the same plan
/// resolution as the startup `log_gpu_probe`.
fn gpu_string(state: &IpcState) -> String {
    let s = state.settings.current();
    let probe = minutist_common::probe_primary_gpu();
    let plan = minutist_common::resolve_gpu_plan(probe.as_ref(), s.gpu_acceleration, true);
    match probe {
        Some(p) => format!(
            "{} (mode={:?}, summariser_gpu={}, asr_gpu={})",
            p.name, s.gpu_acceleration, plan.summariser_gpu, plan.asr_gpu
        ),
        None => format!("cpu (mode={:?})", s.gpu_acceleration),
    }
}

/// Assemble the diagnostic report (#0014). See the module docs for the source
/// priority (`last-crash.txt` then the rolling log tail) and the privacy
/// invariant.
///
/// `probe_primary_gpu` can block (it inits the llama backend), so the work runs
/// on `spawn_blocking` rather than the async command thread.
#[tauri::command]
#[specta::specta]
pub async fn get_diagnostic_report(
    state: State<'_, IpcState>,
) -> Result<DiagnosticReport, IpcError> {
    let logs_dir = state.logs_dir.clone();
    let app_version = state.app_version.clone();
    let platform = state.platform.clone();
    let gpu = gpu_string(&state);

    let report = tokio::task::spawn_blocking(move || {
        let crash_path = logs_dir.join("last-crash.txt");
        if let Ok(crash_body) = std::fs::read_to_string(&crash_path) {
            let (backtrace, recent) = parse_crash_file(&crash_body);
            assemble_report(
                app_version,
                platform,
                gpu,
                "panic",
                &recent,
                backtrace.as_deref(),
            )
        } else {
            let excerpt = newest_log_file(&logs_dir)
                .map(|p| tail_lines(&p, LOG_TAIL_LINES))
                .unwrap_or_default();
            assemble_report(
                app_version,
                platform,
                gpu,
                "diagnostic report",
                &excerpt,
                None,
            )
        }
    })
    .await
    .map_err(|e| IpcError::Internal {
        context: format!("diagnostic report assembly task failed: {e}"),
    })?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_strips_meeting_id_uuids() {
        let line = "ERROR persistence: write failed for \
                    meetings/3f2504e0-4f89-41d3-9a0c-0305e82c3301/transcript.json";
        let out = redact(line);
        assert!(!out.contains("3f2504e0-4f89-41d3-9a0c-0305e82c3301"));
        assert!(out.contains("<redacted-id>"));
        assert!(out.contains("/transcript.json"));
    }

    #[test]
    fn redact_leaves_non_uuid_text_intact() {
        let line = "deadbeef not-a-uuid 1234-5678";
        assert_eq!(redact(line), line);
    }

    #[test]
    fn parse_crash_file_splits_backtrace_and_recent_lines() {
        let body = "Minutist crash report\nMinutist version: 0.0.0\n\n\
                    Panic: boom\n\n\
                    Backtrace:\n0: a\n1: b\n\n\
                    Recent log lines:\nINFO app-main: started\nWARN x: y\n";
        let (bt, recent) = parse_crash_file(body);
        assert_eq!(bt.as_deref(), Some("0: a\n1: b"));
        assert_eq!(recent, "INFO app-main: started\nWARN x: y");
    }

    #[test]
    fn parse_crash_file_without_labels_uses_whole_body() {
        let body = "some unstructured crash dump without sections";
        let (bt, recent) = parse_crash_file(body);
        assert!(bt.is_none());
        assert_eq!(recent, body);
    }

    #[test]
    fn assemble_report_redacts_every_text_field_and_omits_absent_backtrace() {
        let r = assemble_report(
            "0.0.0".to_string(),
            "Linux / x86_64 / connected".to_string(),
            "cpu".to_string(),
            "diagnostic report",
            "INFO x: meetings/3f2504e0-4f89-41d3-9a0c-0305e82c3301/a",
            None,
        );
        assert_eq!(r.error_class, "diagnostic report");
        assert!(r.log_excerpt.contains("<redacted-id>"));
        assert!(!r.log_excerpt.contains("3f2504e0-4f89-41d3-9a0c-0305e82c3301"));
        assert!(r.backtrace.is_none());

        let r = assemble_report(
            "0.0.0".to_string(),
            "p".to_string(),
            "cpu".to_string(),
            "panic",
            "lines",
            Some("0: f for 11111111-2222-3333-4444-555555555555"),
        );
        let bt = r.backtrace.unwrap();
        assert!(bt.contains("<redacted-id>"));
        assert!(!bt.contains("11111111-2222-3333-4444-555555555555"));
    }

    #[test]
    fn newest_log_file_picks_latest_date_suffix() {
        let dir = tempfile::TempDir::new().unwrap();
        for name in ["minutist.log.2026-06-10", "minutist.log.2026-06-14", "other.txt"] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        let picked = newest_log_file(dir.path()).unwrap();
        assert!(picked.ends_with("minutist.log.2026-06-14"));
    }

    #[test]
    fn tail_lines_returns_last_n() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("minutist.log.2026-06-14");
        let body: String = (0..50).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&path, body).unwrap();
        let tail = tail_lines(&path, 3);
        assert_eq!(tail, "line 47\nline 48\nline 49");
    }
}
