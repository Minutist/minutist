//! `persistence` — full Phase-4 read/write surface.
//!
//! Writes per-meeting `audio.opus`, `metadata.json`, `transcript.json`,
//! `notes.json` / `notes.md`, and `summary.md` to `{app-data}/meetings/{uuid}/`,
//! and owns the libsql `index.db` meeting index. This is the **only** crate in
//! the workspace allowed to write under those paths
//! (`architecture/cross-cutting.md` — Filesystem layout).
//!
//! # Write surface
//!
//! - [`MeetingWriter`] (Phase 1/2): `audio.opus`, `metadata.json`,
//!   `transcript.json` while a recording is in flight.
//! - [`NotesStore`] (Phase 3): standalone reader/writer for the opaque
//!   `notes.json` + `notes.md`, independent of [`MeetingWriter`].
//! - [`summary`] (Phase 4): `summary.md` write/read I/O (the producer lands in
//!   Phase 5; the path + I/O seam is here).
//! - [`ChatStore`] (Phase 9): standalone reader/writer for a meeting's chat
//!   sessions under `chat/{session_id}.json` (atomic tmp+rename), independent of
//!   [`MeetingWriter`]; the chat driver in `ipc-bridge` persists a session at
//!   turn end through it.
//!
//! # Read surface (Phase 4)
//!
//! [`reader`] holds the synchronous folder readers — [`reader::read_metadata`],
//! [`reader::read_transcript`], [`reader::read_audio_pcm`] (the graduated
//! pause-INCLUDING Opus decoder), and [`reader::read_meeting_state`] (the
//! `open_meeting` restore-payload assembler).
//!
//! # Index (Phase 4)
//!
//! [`MeetingIndex`] is the libsql `index.db` — a **derived cache** over the
//! per-meeting folders, rebuildable via [`MeetingIndex::rebuild_from_disk`].
//! libsql is async (tokio); the index methods are `async fn` and the crate
//! never calls `block_on` (see `architecture/cross-cutting.md` — Async runtime).
//! [`meeting_ops`] holds the rename/delete operations that keep the on-disk
//! folder and the index row consistent.
//!
//! # Tracing
//!
//! All log calls use `target: "persistence"` for filtered log output.
//!
//! # No Tauri
//!
//! This crate does **not** import `tauri::*`. The paths to `{app-data}/meetings/`
//! and `{app-data}/index.db` are passed in by the caller (orchestrator /
//! app-main).

pub mod chat;
pub mod error;
pub mod folder;
pub mod index;
pub mod meeting_ops;
pub mod metadata;
pub mod migrations;
pub mod notes;
pub mod opus_encoder;
pub mod reader;
pub mod summary;
pub mod transcript;
pub mod writer;

// Public re-exports for the crate's primary surface.
pub use chat::ChatStore;
pub use error::Error;
pub use folder::MeetingFolder;
pub use index::MeetingIndex;
pub use metadata::write_metadata;
pub use notes::{NotesData, NotesStore};
pub use reader::{read_audio_pcm, read_meeting_state, read_metadata, read_transcript};
pub use summary::{read_summary, summary_blurb, write_summary};
pub use transcript::{write_transcript, TranscriptWriter};
pub use writer::MeetingWriter;

#[cfg(test)]
mod tests;
