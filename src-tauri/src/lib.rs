//! `meeting-app` library entry point.
//!
//! Tauri convention: the app logic lives in a `lib` crate so it can be
//! linked into mobile targets (iOS = cdylib, Android = cdylib) as well as
//! the desktop binary.  The `main.rs` binary simply calls `run()` from here.
//!
//! For Phase 1 the mobile targets are not in scope; this file exists to
//! satisfy the `[lib]` declaration in `Cargo.toml` which Tauri's toolchain
//! requires for the crate-type split.

/// Called by `main.rs`.  All runtime logic is in `main.rs` for Phase 1.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Delegate to the binary entry point.  In Phase 1 the full run() logic
    // lives in main.rs; this stub satisfies the lib requirement without
    // duplication.
}
