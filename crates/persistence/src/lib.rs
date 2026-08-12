//! `persistence` — the sole owner of on-disk state under
//! `{app-data}/meetings/{uuid}/`, `{app-data}/index.db`,
//! `{app-data}/collections.json`, and `{app-data}/voiceprints.db`. No other
//! crate in the workspace reads or writes those paths
//! (`architecture/cross-cutting.md` — Filesystem layout).
//!
//! Inputs are typed write commands from `orchestrator` and `ipc-bridge`;
//! outputs are typed read responses. The crate emits no events of its own.
//!
//! # Dependencies
//!
//! `persistence` depends on `common` and on the leaf `notes-crdt` crate,
//! which holds the `yrs`-based (Yjs) CRDT machinery for the authoritative
//! `notes.ydoc` and the per-meeting folder layout (`MeetingFolder`).
//! `persistence` re-exports that surface at its historical paths so its
//! public API is unchanged; the split exists so the `sync` crate can
//! transport the CRDT without pulling in `persistence`'s C-heavy dependency
//! graph (libsql, Opus, Ogg). `yrs` itself is pure Rust with no network
//! surface and is embedded in both build variants — only the sync
//! *transport* is `connected`-gated. Durable whole-state CRDT blobs use the
//! lib0 v2 encoding; the editor's incremental update stream uses lib0 v1 —
//! the two encodings are not interchangeable (feeding a v2 blob to JS
//! `Y.applyUpdate` silently corrupts the doc).
//!
//! # Write surface
//!
//! - [`MeetingWriter`]: `audio.opus`, `metadata.json`, `transcript.json`
//!   while a recording is in flight. `metadata.json` is written as an
//!   in-progress stub (`duration_ms = 0`, a synthesised title, `started_at`)
//!   as the last step of `open`, not only at `finalise`, so a killed process
//!   still leaves a self-heal-recoverable meeting folder; the write is
//!   guarded on the file not already existing, so it never downgrades an
//!   already-finalised record. `finalise` overwrites the stub with the
//!   complete record. Audio is Opus, 16 kHz mono, 32 kbps, Ogg container
//!   (`audiopus = "0.3.0-rc.0"`; the explicit pre-release tag is required at
//!   workspace level because Cargo's semver does not resolve pre-releases
//!   from a `"0.3"` constraint). Pause/resume inserts zero-sample (silent)
//!   Opus frames so decoded duration equals wall-clock duration including
//!   pauses.
//! - [`write_transcript`] (`transcript` module): rewrites `transcript.json`
//!   wholesale from a slice, atomically (tmp + fsync + rename), for the
//!   offline re-transcribe path. An empty slice removes the file rather than
//!   writing `[]`, preserving the absent-means-empty invariant. It always
//!   calls [`clear_translations`] after writing the segment array — every
//!   caller that replaces the array (a full re-transcription renumbers
//!   segment indices; diarization splits/merges segments at speaker-turn
//!   boundaries) invalidates the `translations.json` sidecar rather than
//!   leaving it silently misaligned.
//! - `notes_crdt::NotesStore`: standalone reader/writer for `notes.ydoc`
//!   (the authoritative lib0-v2 whole-state blob) plus the `notes.json` /
//!   `notes.md` projections derived from it, independent of
//!   [`MeetingWriter`] — no shared file handle, which is what lets the
//!   editor autosave run concurrently with an active recording. `save`
//!   rebuilds the doc from JSON; `apply_update` merges a lib0-v1 Yjs update
//!   from the editor's `Y.Doc` `'update'` event onto the stored doc and is
//!   the primary write path for an open Yjs-native editor (it preserves CRDT
//!   history that `save`'s rebuild-from-JSON discards). `notes.json` is
//!   stored as an opaque `serde_json::Value` — the document shape is never
//!   modelled in Rust, so unknown/custom node types (e.g. the transcript-chip
//!   atom) round-trip losslessly; this opacity guarantee must not be broken
//!   by introducing a typed Tiptap model. The `ydoc` submodule's JSON↔Yjs
//!   walk is generic (element tags, attributes, marks, and nesting round-trip
//!   by structure) and mirrors y-prosemirror's mapping (a top-level
//!   `XmlFragment` named `"prosemirror"`) so the editor-side Yjs binding
//!   interops.
//! - [`assets`]: content-addressed note image files under
//!   `{meeting}/assets/<sha256(bytes)>.<ext>`, written by [`save_note_asset`]
//!   and read back by [`read_note_asset`], which rejects any filename
//!   containing a path separator or `..` before reading. `notes.json` stores
//!   only the bare filename (a portable reference), so the opacity guarantee
//!   holds and a meeting folder can be copied to another machine with notes
//!   still resolving. Auto-cleaned by `meeting_ops::delete_meeting`'s
//!   `remove_dir_all`.
//! - [`attachments`]: `{meeting}/attachments/` holds a JSON manifest
//!   (`attachments.json`, atomic tmp+rename), the content-addressed original
//!   (`<sha256>.<ext>`), and a converted markdown sibling (`<sha256>.md`)
//!   once conversion succeeds — a separate subdirectory from `assets/` so the
//!   two content-addressed namespaces don't collide. Manifest
//!   read-modify-write operations (add / set-conversion / remove) serialise
//!   on a process-wide per-meeting `Mutex` registry so concurrent writers
//!   cannot lose-update. Auto-cleaned by `delete_meeting` along with the rest
//!   of the folder.
//! - [`summary`]: `summary.md` write/read I/O; the `summariser` crate
//!   produces the content.
//! - [`ChatStore`]: standalone reader/writer for a meeting's chat sessions
//!   under `chat/{session_id}.json` (one file per session, atomic
//!   tmp+rename), independent of [`MeetingWriter`]; the chat driver in
//!   `ipc-bridge` persists a session at turn end through it.
//! - [`translations`]: `translations.json`, a derived sidecar of
//!   per-language translations of transcript segments, indexed by
//!   `(language, segment_index)`. [`merge_translations`] is an atomic
//!   read-modify-write that adds/overwrites one language's entries, leaving
//!   other languages untouched.
//! - `notes_crdt::write_metadata(meeting_dir, &MeetingMeta)` is a public
//!   atomic writer (tmp + fsync + rename) that rewrites `metadata.json`
//!   wholesale inside an existing meeting folder; it's the seam
//!   `orchestrator` uses to update `{ speaker_count, diarizer }` after
//!   diarization, and the same implementation `MeetingWriter::finalise` and
//!   `meeting_ops::rename_meeting` delegate to, so `persistence` stays the
//!   sole writer under `meetings/{uuid}/`.
//!
//! Every read-modify-write of `metadata.json` (rename, delete,
//! `set_speaker_name`, processing-lifecycle updates, `orchestrator`'s
//! post-processing writes, `agent-tools`' write tools) routes through the
//! guarded primitive `meeting_ops::update_metadata` /
//! `update_metadata_if_present`, which takes the per-meeting
//! `notes_crdt::metadata_lock`, reads, applies the caller's closure, and
//! writes atomically — the single lock every in-process writer of a
//! meeting's `metadata.json` serialises on, rather than each keeping its own
//! mutex. These ops log the meeting id (and the diarizer label) but never
//! `new_title` or speaker `name` — both are user content that must not reach
//! a log line.
//!
//! ## "New meeting" prep drafts
//!
//! A meeting can exist before its audio does. [`writer::create_draft`]
//! builds the on-disk shell — the meeting folder, a `metadata.json` with an
//! empty title and `recording_started: false`, and a `notes.ydoc` seeded via
//! `notes_crdt::meta_crdt::initialise_notes_with_meta` — with no audio file.
//! The Attachments feature and the notes editor both require a real
//! `MeetingId` and folder to exist, so a prep draft is a durable meeting on
//! disk, not a client-side-only placeholder.
//!
//! [`MeetingWriter`] has two entry points that begin capturing into a
//! meeting, both funnelling into a shared private `start_capturing` body
//! (opens `audio.opus` and the encoder and the transcript writer, then RMWs
//! `metadata.json` — never a raw overwrite, since a promoted draft may carry
//! a real user-typed title or collection set during the prep phase — to set
//! `recording_started: true` and stamp the real capture `started_at` /
//! `audio_format`, mirroring the same two fields into the meta CRDT via the
//! granular `meta_crdt::set_started_at` / `set_audio_format` setters):
//!
//! - [`MeetingWriter::open`] composes `create_draft` with `start_capturing`
//!   in one call — the auto-start-immediately path.
//! - [`MeetingWriter::open_for_recording`] promotes an existing draft,
//!   opened via `notes_crdt::MeetingFolder::open_existing`, which returns an
//!   error instead of creating the folder when it is absent.
//!
//! `finalise` never (re-)initialises the meta CRDT map — `notes.ydoc` exists
//! by the time it runs, seeded at draft creation for every meeting — and
//! mirrors `ended_at`, `duration_ms`, `speaker_count`, `asr_model`,
//! `llm_model`, `diarizer`, and `speaker_names` via `edit_meta_ydoc` and the
//! granular setters, using the same un-nested-after-the-RMW lock ordering as
//! `meeting_ops::rename_meeting`. It also re-affirms `title` through the
//! granular `set_title` setter: `meta` at that point holds whatever is
//! currently the best-known title (live-typed, else the on-disk value, else
//! a synthesised default), and a draft's `notes.ydoc` carries no title key
//! at all until a real title exists, so this is what puts a title into the
//! CRDT for a meeting that was never renamed during prep. `started_at` and
//! `audio_format` are untouched at `finalise` — they are set only at
//! draft-creation/promotion time — so a concurrent rename during a live
//! recording (possible, since a draft's `notes.ydoc` syncs during prep,
//! before this device's own recording starts) cannot clobber them.
//!
//! `MeetingMeta::recording_started` is the "never recorded" signal on this
//! device. `#[serde(default = "default_recording_started")]` defaults it to
//! `true`, so a `metadata.json` lacking the field, the sync-arrival
//! placeholder (`notes_crdt::MeetingFolder::ensure`), and orphan recovery
//! (`index::synthesize_metadata`, which only fires when a folder holds real
//! audio or transcript data) all read as "not a draft". It is not carried in
//! the meta CRDT and so does not converge: a peer that syncs an origin's
//! still-unpromoted draft (title/notes with no audio) reads `true`
//! regardless of the origin's real `false` — a known gap, alongside the open
//! question of whether an unpromoted draft should sync at all before it has
//! real content (today it does, unconditionally, from `create_draft`
//! onward). The field is mirrored onto `MeetingListEntry` (`index.db`
//! migration 3: `recording_started INTEGER NOT NULL DEFAULT 1`) and
//! surfaced in the UI as a "Draft" chip (`MeetingList.tsx`:
//! `meeting.recording_started === false`), so an abandoned or unpromoted
//! draft is at least identifiable in the list — nothing yet garbage-collects
//! one.
//!
//! ## Authored-metadata CRDT mirroring
//!
//! Every authoritative write of `MeetingMeta`'s descriptive fields (capture
//! finalise, `meeting_ops::rename_meeting`, `meeting_ops::set_speaker_name`,
//! and `orchestrator`'s re-transcribe/re-diarize post-processing writes)
//! also mirrors into `notes.ydoc`'s meta CRDT map (`notes_crdt::meta_crdt`,
//! owned by the sync domain) via `meeting_ops::meta_crdt`, a re-export
//! following the same pattern as the guarded-RMW functions — so
//! `orchestrator` doesn't need its own `notes-crdt` dependency edge. This is
//! what lets a peer that only ever received a sync-arrival placeholder
//! folder converge onto the real `title` / `started_at` / `duration_ms`
//! instead of being stuck with invented values.
//!
//! Calls to `meta_crdt::edit_meta_ydoc` happen **after** the corresponding
//! `update_metadata` RMW returns, never nested inside it:
//! `edit_meta_ydoc` takes `notes_lock`, `update_metadata` takes
//! `metadata_lock`. Keeping the two calls un-nested avoids a lock-ordering
//! deadlock against the sync-receive projection path
//! (`project_ydoc_meta_into_metadata`), which takes only `metadata_lock`.
//!
//! # Read surface
//!
//! [`reader`] holds the synchronous, blocking `std::fs` folder readers;
//! callers in an async context drive them via `spawn_blocking`.
//!
//! - [`read_metadata`], [`read_transcript`] (an absent `transcript.json`
//!   reads as an empty `Vec`, not an error).
//! - [`read_audio_pcm`]: the graduated Opus decoder. Returns the full
//!   pause-INCLUDING 16 kHz mono `f32` buffer — the silent frames written
//!   for pause gaps decode to real zero samples, so buffer duration equals
//!   wall-clock recording duration. This is what diarization and
//!   re-transcribe consume, and why the orchestrator sources audio through
//!   this reader rather than `diarizer` depending on `persistence` directly.
//!   The decoder is dispatched by file extension
//!   (`minutist_common::resolve_audio_path`): Opus for the desktop's own
//!   recordings, AAC via `symphonia` for a synced phone `.m4a` recording,
//!   falling back to the AAC decoder on an Opus-decode failure for a
//!   `.opus` path (rescuing a mislabelled-AAC backlog without a migration).
//! - [`read_meeting_state`]: assembles `meta` + `transcript` + optional
//!   `notes` — the `open_meeting` restore payload. Also the lazy
//!   notes-CRDT migration trigger: on open, when `notes.ydoc` is absent but
//!   `notes.json` exists, it seeds `notes.ydoc` from the JSON and flips
//!   `MeetingMeta::notes_format` to `1`. The seed is idempotent, build
//!   invariant (both build variants seed), and per-meeting — a never-opened
//!   meeting is never touched and stays JSON-readable; after seeding,
//!   `notes.ydoc` is authoritative.
//! - [`read_note_blocks`]: projects `notes.json` into `common::NoteBlock`s
//!   for the summariser (one block per non-empty paragraph, carrying its
//!   `data-anchor-ms` anchor when present) — a best-effort read projection
//!   that does not weaken the `NotesStore` opacity guarantee.
//!
//! # Index
//!
//! [`MeetingIndex`] is the libsql `index.db` — a **derived cache** over the
//! per-meeting folders, holding one row per meeting list entry. libsql is
//! async (tokio); index methods are `async fn` and the crate never calls
//! `block_on` (see `architecture/cross-cutting.md` — Async runtime). A
//! forward-only migration runner (`migrations::run`) converges both an
//! empty DB and a prior-schema DB onto the current schema additively (each
//! step is `CREATE TABLE/INDEX IF NOT EXISTS`), so a derived-cache rebuild
//! never loses reconstructable rows. [`MeetingIndex::rebuild_from_disk`]
//! clears and repopulates the whole index by scanning every meeting folder.
//! [`MeetingIndex::reconcile_orphans`] is the in-session self-heal: an
//! ADD-only (never deletes) counterpart that indexes folders present on disk
//! but missing from the cache, so a meeting can't stay hidden after a missed
//! upsert without waiting for the next startup rebuild. A folder with
//! recording data (`audio.opus` and/or `transcript.json`) but no
//! `metadata.json` is recovered by synthesising a minimal metadata record
//! before indexing; a folder with neither is left alone.
//!
//! [`meeting_ops`] keeps the on-disk folder and the index row consistent —
//! the folder is authoritative: `rename_meeting` rewrites `metadata.json`
//! then updates the index row; `delete_meeting` removes the folder then the
//! row. A crash between the two steps leaves the index stale but
//! rebuildable. `apply_processing_lifecycle` / `apply_synced_lifecycle_if_present`
//! apply an inbound peer's processing state (host-authoritative, arriving
//! over the sync lifecycle exchange, and applied only when the meeting is
//! held locally); `apply_own_processing_if_not_superseded` is the guarded
//! counterpart `election` uses for a host's own terminal `Processed` write,
//! committing only when it wins the CRDT merge against the current on-disk
//! state, so a host that finishes processing after a peer's stronger/tied
//! state cannot regress it.
//!
//! # Collections
//!
//! [`CollectionStore`] (`collections` module) is the authoritative
//! reader/writer for the flat `Vec<Collection>` in `collections.json`
//! (atomic tmp+fsync+rename), stored beside — not under — `index.db`,
//! since the index is a wiped-and-rebuilt cache while the folder list must
//! survive a rebuild. Membership is authoritative on each meeting's
//! `metadata.json` (`MeetingMeta::collection_id`, written by
//! `meeting_ops::set_meeting_collection`, which also refreshes the index
//! row); the index carries a derived, nullable `collection_id` column for
//! filtered listing. `collections::delete_collection` clears every member
//! meeting's `collection_id` before removing the collection definition, so
//! no `metadata.json` keeps a dangling reference.
//!
//! # RAG store
//!
//! [`RagStore`] (`rag` module) is a per-meeting retrieval cache in
//! `{meeting}/meeting.db` (libsql) — derived and rebuildable (re-chunk +
//! re-embed), never authoritative. Tables: `rag_chunk`, `rag_embedding`, and
//! a self-content FTS5 index `rag_chunk_fts`. `retrieve_dense` and
//! `retrieve_lexical` are the two retrieval legs (brute-force cosine via
//! `common::voiceprint_math`; `bm25()` full-text over sanitised, quoted
//! query tokens), fused by the caller (Reciprocal Rank Fusion). The dense
//! leg scores only vectors stored under the query's `model_id` and of
//! matching dimension — a foreign vector is skipped, not truncate-scored, so
//! a model swap degrades to "no comparable vectors" instead of corrupting
//! the ranking. `RagStore::open` sets a `busy_timeout` so overlapping
//! indexers retry rather than error.
//!
//! # Voiceprint library
//!
//! [`VoiceprintStore`] (`voiceprints` module) is backed by a separate
//! durable libsql database, `{app-data}/voiceprints.db`
//! ([`voiceprints_db_path`]), with its own forward-only migration runner
//! (`voiceprints_migrations::run`) run before any query; a migration or open
//! error is returned as an `Error` and the caller maps it to
//! enrolment-OFF (the corruption degrade-to-off contract). Three tables:
//! `voiceprint_identity` (one row per enrolled speaker, stable across
//! renames and merges), `voiceprint_centroid` (one acquisition-condition
//! gallery entry per identity — a cached, weighted-merged unit-normalised
//! embedding, `ON DELETE CASCADE` from identity), and
//! `voiceprint_contribution` (one `(meeting_id, label)` that fed a centroid,
//! retaining its own centroid vector so the gallery centroid is
//! recomputable and refinement reversible by dropping contributions and
//! re-merging over survivors, `ON DELETE CASCADE` from centroid). Any
//! operation that changes a centroid's contribution set must call the
//! private `recompute_centroid` helper in the same transaction. Public
//! operations include `enrol`, `refine` (nearest-centroid match with a
//! similarity fold gate, a gallery cap, and a bounded-weight poison
//! defence), `merge_identities`, `rename_identity`, `delete_identity`,
//! `forget_meeting` (meeting-granularity erasure — drops every contribution
//! for the meeting, recomputes affected centroids, drops zero-contribution
//! centroids and zero-centroid identities), `find_identity_by_name_and_model`,
//! `all` (returns zero rows for a foreign `model_id` — hard invalidation),
//! and `identities_with_gallery` (management-UI query). [`StoredVoiceprint`]
//! and its embedding bytes never cross IPC; `IdentityWithGallery` /
//! `CentroidSummary` are the embedding-free query results that are IPC-safe.
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

pub mod assets;
pub mod attachments;
mod blob;
pub mod chat;
pub mod collections;
pub mod error;
pub mod index;
pub mod meeting_ops;
pub mod metadata;
pub mod migrations;
pub mod opus_encoder;
pub mod rag;
pub mod reader;
pub mod summary;
pub mod transcript;
pub mod translations;
pub mod voiceprints;
pub mod voiceprints_migrations;
pub mod writer;

// The notes-CRDT primitives (`folder`, `notes`, `ydoc` and their public items —
// `MeetingFolder`, `NotesStore`, `NotesData`, `write_metadata`, `metadata_lock`,
// `note_blocks_from_json`) live in the leaf `notes-crdt` crate; every consumer
// imports them from `notes_crdt::*` directly. `persistence` remains the sole
// writer under `{app-data}/meetings/` — it depends on `notes-crdt` for its own
// implementation but no longer re-exports these symbols at its own paths.

// Public re-exports for the crate's primary surface.
pub use assets::{read_note_asset, save_note_asset};
pub use attachments::{
    add_manifest_entry, attachment_original_path, read_attachment_markdown,
    read_attachment_original, read_attachments_markdown_parts, read_manifest,
    remove_manifest_entry, save_attachment_markdown, save_attachment_original,
    set_entry_conversion, unlink_attachment_files, unlink_orphan_attachment_markdown,
};
pub use chat::ChatStore;
pub use collections::{collections_path, delete_collection, CollectionStore};
pub use error::Error;
pub use index::MeetingIndex;
pub use rag::{meeting_db_path, NewChunk, RagStore, RetrievedChunk};
pub use reader::{
    read_audio_pcm, read_meeting_state, read_metadata, read_note_blocks, read_transcript,
};
pub use summary::{read_summary, summary_blurb, write_summary};
pub use transcript::{write_transcript, TranscriptWriter};
pub use translations::{clear_translations, merge_translations, read_translations};
pub use voiceprints::{voiceprints_db_path, StoredVoiceprint, VoiceprintStore};
pub use writer::MeetingWriter;

#[cfg(test)]
mod tests;
