//! Crash capture for the no-telemetry "Report a problem" flow (issue #0014).
//!
//! Two pieces, both owned by `app-main` (this is the logging-bootstrap domain):
//!
//! 1. [`RingBufferLayer`] — a `tracing` layer that keeps the last
//!    [`RING_CAPACITY`] formatted log lines in a process-wide static, so a
//!    crash report (and the on-demand diagnostic report) can attach recent
//!    context without re-reading the rolling log file.
//! 2. [`install_panic_hook`] — a `std::panic::set_hook` that, on a panic,
//!    writes a **redacted** `last-crash.txt` to the logs dir (app version,
//!    OS/arch, best-effort GPU plan, the panic message + location, a backtrace,
//!    and the recent ring lines). `ipc-bridge::get_diagnostic_report` reads this
//!    file when present.
//!
//! Privacy: every line written to `last-crash.txt` passes through [`redact`],
//! which strips meeting-id UUIDs (the one piece of meeting identity that can
//! appear in a logged path). By design no meeting *content* (transcript / notes
//! / title / speaker text) is logged at any level (audited in #0014), so the
//! ring never holds it; the UUID strip is the defensive boundary pass for paths.
//!
//! Nothing here sends anything off the machine — the file is local, and the
//! user reviews + submits the report from their own browser.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Mutex;

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// How many recent log lines the ring buffer retains. Enough to give a crash
/// report useful context without bloating the URL/clipboard report.
pub const RING_CAPACITY: usize = 200;

/// The crash file name written under `{app-data}/logs/`.
pub const LAST_CRASH_FILE: &str = "last-crash.txt";

/// Process-wide ring buffer of recent formatted log lines (newest last).
///
/// A plain `Mutex<VecDeque>` — no new dependency. The lock is held only for the
/// duration of a push or a snapshot copy, both O(1)/O(n) and non-blocking.
static LOG_RING: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

/// Push one formatted line into the ring, evicting the oldest when full.
fn ring_push(line: String) {
    if let Ok(mut ring) = LOG_RING.lock() {
        if ring.len() >= RING_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(line);
    }
    // A poisoned lock (a panic while another thread held it) is ignored: losing
    // a log line is never worth a cascading panic from inside the logger.
}

/// Snapshot the recent log lines as a single newline-joined, **redacted** block.
///
/// Used by both the panic hook and the on-demand diagnostic report. Returns at
/// most [`RING_CAPACITY`] lines, oldest first.
pub fn recent_log_lines() -> String {
    let lines: Vec<String> = match LOG_RING.lock() {
        Ok(ring) => ring.iter().cloned().collect(),
        Err(poisoned) => poisoned.into_inner().iter().cloned().collect(),
    };
    redact(&lines.join("\n"))
}

/// Strip meeting-id UUIDs from `text`, replacing each with `<redacted-id>`.
///
/// Mirrors the webview's `redactMeetingPaths` boundary pass. A meeting id in a
/// logged path is the one piece of meeting identity that can leak through an
/// otherwise content-free log line. No regex dependency: a UUID is a fixed
/// 8-4-4-4-12 hex-with-hyphens shape, scanned by hand.
pub fn redact(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if let Some(end) = uuid_at(bytes, i) {
            out.push_str("<redacted-id>");
            i = end;
        } else {
            // Push the single char starting at `i`. `text` is valid UTF-8; step
            // by the char's byte length so multi-byte chars are not split.
            let ch = text[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// If a UUID (`8-4-4-4-12` lowercase/uppercase hex) starts at `bytes[i]`,
/// return the index one past its end; otherwise `None`.
fn uuid_at(bytes: &[u8], i: usize) -> Option<usize> {
    // Group lengths in a canonical UUID.
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

/// A `tracing` layer that appends each event to the [`LOG_RING`].
///
/// Formats `LEVEL target: field=value … message` — compact, ASCII, and
/// dependency-free. Field values are rendered via the event's [`Visit`]
/// callbacks; the conventional `message` field is appended last.
pub struct RingBufferLayer;

impl<S: Subscriber> Layer<S> for RingBufferLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let mut visitor = LineVisitor::new(*meta.level(), meta.target());
        event.record(&mut visitor);
        ring_push(visitor.finish());
    }
}

/// Builds one ring line from an event's fields.
struct LineVisitor {
    /// `LEVEL target:` prefix.
    prefix: String,
    /// Accumulated `key=value` fields (excluding the message).
    fields: String,
    /// The conventional `message` field, rendered separately so it trails.
    message: String,
}

impl LineVisitor {
    fn new(level: tracing::Level, target: &str) -> Self {
        Self {
            prefix: format!("{level:>5} {target}:"),
            fields: String::new(),
            message: String::new(),
        }
    }

    fn finish(self) -> String {
        let mut line = self.prefix;
        if !self.message.is_empty() {
            line.push(' ');
            line.push_str(&self.message);
        }
        if !self.fields.is_empty() {
            line.push_str(&self.fields);
        }
        line
    }
}

impl Visit for LineVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
            // `record_debug` of a &str message wraps it in quotes; tracing's own
            // fmt does the same for non-string Debug, so leave as-is.
            if self.message.starts_with('"') && self.message.ends_with('"') {
                self.message = self.message[1..self.message.len() - 1].to_string();
            }
        } else {
            let _ = write!(self.fields, " {}={value:?}", field.name());
        }
    }
}

/// Install a panic hook that writes a redacted [`LAST_CRASH_FILE`] to `logs_dir`.
///
/// Chains the previous hook so the default abort/print behaviour is preserved.
/// `version`, `platform`, and `gpu` are resolved by the caller (app-main has the
/// app version + GPU probe); they are formatted into the file header. The file
/// is overwritten on each panic (only the most recent crash is retained).
pub fn install_panic_hook(logs_dir: &Path, version: String, platform: String, gpu: String) {
    let crash_path = logs_dir.join(LAST_CRASH_FILE);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let body = render_crash_report(
            &version,
            &platform,
            &gpu,
            &panic_message(info),
            &backtrace.to_string(),
            &recent_log_lines(),
        );
        // Best-effort write; a failure here must never mask the panic.
        let _ = std::fs::write(&crash_path, body);
        // Preserve default behaviour (print + abort/unwind as configured).
        previous(info);
    }));
}

/// Extract a human-readable message + location from a [`std::panic::PanicHookInfo`].
fn panic_message(info: &std::panic::PanicHookInfo<'_>) -> String {
    let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "panic (non-string payload)".to_string()
    };
    match info.location() {
        Some(loc) => format!("{payload}\n  at {}:{}:{}", loc.file(), loc.line(), loc.column()),
        None => payload,
    }
}

/// Render the redacted crash-report body. Pure (no I/O) so it is unit-tested.
///
/// Every free-text section is passed through [`redact`] before joining.
pub fn render_crash_report(
    version: &str,
    platform: &str,
    gpu: &str,
    panic_message: &str,
    backtrace: &str,
    recent_lines: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Minutist crash report");
    let _ = writeln!(out, "Minutist version: {version}");
    let _ = writeln!(out, "Platform: {platform}");
    let _ = writeln!(out, "GPU: {gpu}");
    let _ = writeln!(out);
    let _ = writeln!(out, "Panic: {}", redact(panic_message));
    let _ = writeln!(out);
    let _ = writeln!(out, "Backtrace:");
    let _ = writeln!(out, "{}", redact(backtrace).trim_end());
    let _ = writeln!(out);
    let _ = writeln!(out, "Recent log lines:");
    // `recent_lines` is already redacted by `recent_log_lines`, but redact again
    // defensively in case a caller passes raw text.
    let _ = writeln!(out, "{}", redact(recent_lines).trim_end());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_strips_meeting_id_uuids() {
        let line =
            "ERROR persistence: write failed for meetings/3f2504e0-4f89-41d3-9a0c-0305e82c3301/transcript.json";
        let out = redact(line);
        assert!(!out.contains("3f2504e0-4f89-41d3-9a0c-0305e82c3301"));
        assert!(out.contains("<redacted-id>"));
        assert!(out.contains("meetings/"));
        assert!(out.contains("/transcript.json"));
    }

    #[test]
    fn redact_handles_uppercase_and_multiple_ids() {
        let line = "3F2504E0-4F89-41D3-9A0C-0305E82C3301 and 11111111-2222-3333-4444-555555555555";
        let out = redact(line);
        assert_eq!(out, "<redacted-id> and <redacted-id>");
    }

    #[test]
    fn redact_leaves_non_uuid_text_intact() {
        // A short hex run, and a hyphenated non-UUID, must survive untouched.
        let line = "deadbeef not-a-uuid 1234-5678";
        assert_eq!(redact(line), line);
    }

    #[test]
    fn redact_preserves_multibyte_text() {
        let line = "café — note for 3f2504e0-4f89-41d3-9a0c-0305e82c3301";
        let out = redact(line);
        assert!(out.starts_with("café — note for "));
        assert!(out.ends_with("<redacted-id>"));
    }

    #[test]
    fn ring_push_evicts_oldest_when_full() {
        // Drive the real static; assert capacity is honoured and order kept.
        // (Other tests in this module do not push to the ring.)
        for i in 0..(RING_CAPACITY + 5) {
            ring_push(format!("line {i}"));
        }
        let snapshot = recent_log_lines();
        let lines: Vec<&str> = snapshot.lines().collect();
        assert_eq!(lines.len(), RING_CAPACITY);
        // The first five (0..5) were evicted; the window starts at "line 5".
        assert_eq!(lines.first().copied(), Some("line 5"));
        assert_eq!(
            lines.last().copied(),
            Some(format!("line {}", RING_CAPACITY + 4)).as_deref()
        );
    }

    #[test]
    fn render_crash_report_redacts_and_has_no_meeting_content() {
        let body = render_crash_report(
            "0.0.0",
            "Linux / x86_64 / connected",
            "cpu",
            "called Option::unwrap() on a None for meetings/3f2504e0-4f89-41d3-9a0c-0305e82c3301",
            "0: minutist::crash\n1: core::panic",
            "INFO app-main: started\nWARN persistence: x for 11111111-2222-3333-4444-555555555555",
        );
        assert!(body.contains("Minutist version: 0.0.0"));
        assert!(body.contains("Platform: Linux / x86_64 / connected"));
        assert!(body.contains("Backtrace:"));
        assert!(body.contains("minutist::crash"));
        // Both meeting ids stripped from their respective sections.
        assert!(!body.contains("3f2504e0-4f89-41d3-9a0c-0305e82c3301"));
        assert!(!body.contains("11111111-2222-3333-4444-555555555555"));
        assert_eq!(body.matches("<redacted-id>").count(), 2);
    }
}
