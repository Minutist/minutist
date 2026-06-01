//! `persistence` — Phase 1/2 write surface.
//!
//! Writes per-meeting `audio.opus`, `metadata.json`, and (Phase 2)
//! `transcript.json` to `{app-data}/meetings/{uuid}/`. This is the **only**
//! crate in the workspace allowed to write under that directory
//! (`architecture/cross-cutting.md` — Filesystem layout).
//!
//! Phase 3 adds [`NotesStore`], a standalone reader/writer for `notes.json`
//! (opaque Tiptap document) + `notes.md`, independent of [`MeetingWriter`].
//!
//! Phase 4 will add the libsql index, summary storage, and meeting-list
//! queries. Do **not** preempt Phase 4 here.
//!
//! # Tracing
//!
//! All log calls use `target: "persistence"` for filtered log output.
//!
//! # No Tauri
//!
//! This crate does **not** import `tauri::*`. The path to `{app-data}` is
//! passed in by the caller (typically the orchestrator).

pub mod error;
pub mod folder;
pub mod metadata;
pub mod notes;
pub mod opus_encoder;
pub mod transcript;
pub mod writer;

// Public re-exports for the crate's primary surface.
pub use error::Error;
pub use folder::MeetingFolder;
pub use notes::{NotesData, NotesStore};
pub use transcript::TranscriptWriter;
pub use writer::MeetingWriter;

#[cfg(test)]
mod tests;
