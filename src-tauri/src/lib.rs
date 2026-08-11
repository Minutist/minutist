//! `minutist_lib` — satisfies the `[lib]` crate-type split
//! (`staticlib`/`cdylib`/`rlib`) Tauri's toolchain requires for a mobile
//! build. The desktop binary (`main.rs`) does not call anything here; its own
//! `run()` holds all runtime logic and this crate is otherwise unused on
//! desktop.
//!
//! This app does not build a mobile target today (no `gen/apple`/`gen/android`
//! project, no mobile CI leg). `#[tauri::mobile_entry_point]` below is real —
//! it IS what a mobile build would call — but the function body is an empty
//! stub, so a mobile build of this crate as it stands would launch and do
//! nothing. Give this function main.rs's logic before attempting one.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {}
