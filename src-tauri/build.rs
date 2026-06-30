use std::process::Command;

/// Embed a short git commit SHA so dev builds are identifiable (matching a crash
/// report or the title-bar build to an exact commit). Resolution order:
///   1. `MINUTIST_GIT_SHA` env — set by the Windows build path, whose mirrored
///      source tree excludes `.git` (see `scripts/build-windows-app.ps1`).
///   2. `git rev-parse --short HEAD` — works for local/WSL builds with a `.git`.
///   3. `"unknown"` — never fails the build.
fn git_sha() -> String {
    if let Ok(sha) = std::env::var("MINUTIST_GIT_SHA") {
        let sha = sha.trim();
        if !sha.is_empty() {
            return sha.to_string();
        }
    }
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    println!("cargo:rustc-env=MINUTIST_GIT_SHA={}", git_sha());
    // Re-run when the env override changes or HEAD moves. Watch `logs/HEAD` (the
    // reflog), which is appended on every commit/checkout/reset — unlike `HEAD`,
    // whose symbolic-ref content is invariant across same-branch commits. Both
    // paths are absent on the Windows mirror (which excludes .git), where the
    // env override drives this instead. (rerun-if-changed on a missing path is a
    // harmless no-op.)
    println!("cargo:rerun-if-env-changed=MINUTIST_GIT_SHA");
    println!("cargo:rerun-if-changed=../.git/logs/HEAD");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    tauri_build::build()
}
