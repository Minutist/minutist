//! Tauri command handlers for the Phase 1 IPC surface.
//!
//! Every command is annotated with both `#[tauri::command]` (wires it into
//! Tauri's invoke handler) and `#[specta::specta]` (registers its signature
//! for TypeScript generation).
//!
//! Commands are `async fn` because the orchestrator's methods are async.
//!
//! `list_devices` routes through `Orchestrator::list_devices` (which wraps
//! the cpal enumeration in `spawn_blocking`), preserving the dependency-table
//! invariant that `ipc-bridge` depends only on `orchestrator + settings +
//! common`.
//!
//! All commands return `Result<T, IpcError>`.  The `?` operator on
//! `AppResult<T>` automatically converts via `IpcError::from(AppError)`.
//!
//! ## Specta types
//!
//! `common` and `settings` derive `specta::Type` directly (gated on the
//! `specta` feature, which this crate enables on both deps). The mirror
//! layer that Phase 1 carried in `specta_types.rs` was removed in P0a;
//! commands return `common` / `settings` types directly.
//!
//! ## Tauri State
//!
//! Each command that needs the orchestrator or settings receives its handles
//! as `tauri::State<'_, IpcState>`.

use std::path::Path;

use meeting_app_common::{
    AppError, AudioDevice, MeetingId, MeetingListEntry, MeetingMeta, MeetingState, ModelId,
    ModelStatus, RecordingState,
};
use persistence::{meeting_ops, NotesStore};
use serde::{Deserialize, Serialize};
use settings::Settings;
use specta::Type;
use tauri::State;

use crate::{error::IpcError, IpcState};

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

/// List all available audio-input devices.
///
/// Routes through `Orchestrator::list_devices`, which wraps the FFI-bound
/// cpal enumeration in `spawn_blocking`. This keeps `ipc-bridge`'s
/// dependency table honest: it depends on `orchestrator`, not directly on
/// `audio-capture`.
#[tauri::command]
#[specta::specta]
pub async fn list_devices(state: State<'_, IpcState>) -> Result<Vec<AudioDevice>, IpcError> {
    state
        .orchestrator
        .list_devices()
        .await
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Recording lifecycle
// ---------------------------------------------------------------------------

/// Start a new recording session.
///
/// `device_id = None` → use the device configured in settings, or the OS
/// default if none is configured.
///
/// Returns the new `MeetingId` on success.
#[tauri::command]
#[specta::specta]
pub async fn start_recording(
    device_id: Option<String>,
    state: State<'_, IpcState>,
) -> Result<MeetingId, IpcError> {
    state
        .orchestrator
        .start(device_id)
        .await
        .map_err(IpcError::from)
}

/// Pause the current recording.
#[tauri::command]
#[specta::specta]
pub async fn pause_recording(state: State<'_, IpcState>) -> Result<(), IpcError> {
    state.orchestrator.pause().await.map_err(IpcError::from)
}

/// Resume after a pause.
#[tauri::command]
#[specta::specta]
pub async fn resume_recording(state: State<'_, IpcState>) -> Result<(), IpcError> {
    state.orchestrator.resume().await.map_err(IpcError::from)
}

/// Stop the current recording and finalise the meeting.
///
/// Returns the completed `MeetingMeta` on success.
///
/// After the orchestrator finalises the meeting folder, this also **upserts the
/// libsql index** (FR-33) so the just-recorded meeting appears in
/// `list_meetings` **in the same session** — without waiting for the next
/// app-start `rebuild_from_disk`. The orchestrator stays decoupled from the
/// index (it does not own one); the `ipc-bridge` owns the index handle in
/// `IpcState`, so the upsert lives here. See `architecture/components.md`,
/// `ipc-bridge` "Phase 4 additions — stop-upsert".
///
/// The blocking transcript read (for the list excerpt) runs on `spawn_blocking`
/// per the threading model; the async index `upsert` is awaited, never
/// `block_on`'d (the no-`block_on`-in-command-handlers rule).
///
/// An index-upsert failure is logged and swallowed — the recording itself is
/// safely persisted on disk and the index is a derived cache that the next
/// startup `rebuild_from_disk` will reconcile, so a failed upsert must not turn
/// a successful stop into an error.
#[tauri::command]
#[specta::specta]
pub async fn stop_recording(state: State<'_, IpcState>) -> Result<MeetingMeta, IpcError> {
    let meta = state.orchestrator.stop().await.map_err(IpcError::from)?;

    // Build the meeting-list row for the freshly-stopped meeting. The excerpt is
    // the first transcript segment (if any), read on a blocking thread.
    let meetings_dir = state.meetings_dir.clone();
    let meta_for_entry = meta.clone();
    let entry = tokio::task::spawn_blocking(move || {
        meeting_list_entry_for_meta(&meetings_dir, &meta_for_entry)
    })
    .await;

    match entry {
        Ok(entry) => {
            if let Err(e) = state.index.upsert(&entry).await {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meta.uuid.0,
                    "index upsert after stop_recording failed: {e}; the recording is on disk \
                     and will be re-indexed on next startup"
                );
            }
        }
        Err(join_err) => {
            tracing::warn!(
                target: "ipc-bridge",
                meeting_id = %meta.uuid.0,
                "building index entry after stop_recording failed (join error): {join_err}; \
                 the recording is on disk and will be re-indexed on next startup"
            );
        }
    }

    Ok(meta)
}

/// Build the [`MeetingListEntry`] for a just-stopped meeting from its
/// [`MeetingMeta`] plus the first transcript segment (the list excerpt).
///
/// Blocking `std::fs` read of `transcript.json` (via
/// `persistence::reader::read_transcript`); an absent/empty transcript yields
/// `excerpt: None`. Extracted so it can be unit-tested without a Tauri runtime.
fn meeting_list_entry_for_meta(meetings_dir: &Path, meta: &MeetingMeta) -> MeetingListEntry {
    let meeting_dir = meetings_dir.join(meta.uuid.0.to_string());
    let excerpt = persistence::read_transcript(&meeting_dir)
        .ok()
        .and_then(|segs| segs.first().map(|s| s.text.clone()));

    MeetingListEntry {
        id: meta.uuid,
        title: meta.title.clone(),
        started_at: meta.started_at.clone(),
        duration_ms: meta.duration_ms,
        speaker_count: meta.speaker_count,
        excerpt,
    }
}

/// Return a snapshot of the current recording state.
#[tauri::command]
#[specta::specta]
pub async fn get_recording_state(
    state: State<'_, IpcState>,
) -> Result<RecordingState, IpcError> {
    Ok(state.orchestrator.state().await)
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/// List all known models with their current runtime status.
///
/// Routes through `Orchestrator::list_models`, which wraps `ModelRegistry::list_models`
/// so that `ipc-bridge` does not need a direct `model-registry` dependency.
#[tauri::command]
#[specta::specta]
pub async fn list_models(state: State<'_, IpcState>) -> Result<Vec<ModelStatus>, IpcError> {
    Ok(state.orchestrator.list_models())
}

/// Ensure a model is downloaded and hash-verified.
///
/// Returns `Ok(())` when the model is ready for use. Starts a download if the
/// model is absent; the webview tracks granular progress via
/// `AppEvent::ModelDownloadProgress` events emitted on the broadcast channel.
///
/// Routes through `Orchestrator::ensure_model`, preserving the dependency-table
/// invariant that `ipc-bridge` does not depend directly on `model-registry`.
#[tauri::command]
#[specta::specta]
pub async fn ensure_model(
    model_id: ModelId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    state
        .orchestrator
        .ensure_model(&model_id)
        .await
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Return the current application settings.
#[tauri::command]
#[specta::specta]
pub async fn get_settings(state: State<'_, IpcState>) -> Result<Settings, IpcError> {
    Ok(state.settings.current())
}

/// Persist updated application settings.
///
/// Broadcasts the change to all `SettingsHandle` subscribers.
#[tauri::command]
#[specta::specta]
pub async fn update_settings(
    settings: Settings,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    state
        .settings
        .update(|s| *s = settings)
        .await
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Notes persistence (Phase 3)
// ---------------------------------------------------------------------------

/// A persisted notes document returned by [`load_notes`].
///
/// `notes_json` carries the Tiptap/ProseMirror document **as a `String`**, not
/// a `serde_json::Value`: a bare `serde_json::Value` does not derive
/// `specta::Type`, so it cannot cross the tauri-specta boundary directly. The
/// webview owns the (de)serialisation of this opaque document; `persistence`
/// stores it verbatim (the Phase-4 transcript-chip opacity guarantee). The
/// `String`-over-the-wire choice keeps the IPC contract typed without forcing a
/// Rust-side Tiptap model.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct NotesDoc {
    pub notes_json: String,
    pub notes_markdown: String,
}

/// Persist a meeting's notes (`notes.json` + `notes.md`).
///
/// Routes **directly** to `persistence::NotesStore` against
/// `IpcState::meetings_dir` — notes I/O is independent of the live recording
/// pipeline (see `architecture/components.md`, `persistence` "Phase 3 surface
/// growth — notes"), so the orchestrator is not involved. The blocking
/// filesystem write runs on `spawn_blocking` per the threading model.
///
/// `notes_json` is parsed from a `String` into a `serde_json::Value`; an
/// invalid JSON string is rejected as `AppError::InvalidInput`.
#[tauri::command]
#[specta::specta]
pub async fn save_notes(
    meeting_id: MeetingId,
    notes_json: String,
    notes_markdown: String,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || {
        save_notes_inner(&meetings_dir, meeting_id, &notes_json, &notes_markdown)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("save_notes task join failed: {e}"),
    })?
    .map_err(IpcError::from)
}

/// Load a meeting's persisted notes, or `None` when no notes have been saved.
///
/// Routes directly to `persistence::NotesStore`; the loaded opaque
/// `serde_json::Value` is re-serialised back to a `String` for the wire (see
/// [`NotesDoc`]).
#[tauri::command]
#[specta::specta]
pub async fn load_notes(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<Option<NotesDoc>, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || load_notes_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("load_notes task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

// ---------------------------------------------------------------------------
// Notes command bodies — extracted so they can be unit-tested without a
// running Tauri runtime (the round-trip test calls these directly).
// ---------------------------------------------------------------------------

/// Inner body of [`save_notes`]: parse the JSON string and write via
/// `NotesStore`. Returns `AppError` so both the command and the unit test
/// share one error path.
fn save_notes_inner(
    meetings_dir: &std::path::Path,
    meeting_id: MeetingId,
    notes_json: &str,
    notes_markdown: &str,
) -> Result<(), AppError> {
    let value: serde_json::Value =
        serde_json::from_str(notes_json).map_err(|e| AppError::InvalidInput {
            context: format!("notes_json is not valid JSON: {e}"),
        })?;
    NotesStore::save(meetings_dir, meeting_id, &value, notes_markdown)
}

/// Inner body of [`load_notes`]: read via `NotesStore` and re-serialise the
/// opaque document back to a `String` for the wire.
fn load_notes_inner(
    meetings_dir: &std::path::Path,
    meeting_id: MeetingId,
) -> Result<Option<NotesDoc>, AppError> {
    let loaded = NotesStore::load(meetings_dir, meeting_id)?;
    match loaded {
        None => Ok(None),
        Some(data) => {
            let notes_json = serde_json::to_string(&data.json).map_err(|e| AppError::Internal {
                context: format!("failed to re-serialise loaded notes.json: {e}"),
            })?;
            Ok(Some(NotesDoc {
                notes_json,
                notes_markdown: data.markdown,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Meeting list + open + actions (Phase 4)
// ---------------------------------------------------------------------------

/// List all meetings for the meeting-list view (FR-33), most-recent first.
///
/// Reads straight from the libsql `index.db` ([`MeetingIndex::list_meetings`])
/// — a cheap projection that never loads a meeting's full transcript. The index
/// is async (libsql/tokio); the future is awaited here, never `block_on`'d.
#[tauri::command]
#[specta::specta]
pub async fn list_meetings(state: State<'_, IpcState>) -> Result<Vec<MeetingListEntry>, IpcError> {
    state.index.list_meetings().await.map_err(IpcError::from)
}

/// Open a meeting, returning its full restorable [`MeetingState`]
/// (metadata + transcript + optional notes).
///
/// Resolves the per-meeting folder under `IpcState::meetings_dir` and assembles
/// the state via `persistence::read_meeting_state`. The blocking `std::fs` reads
/// run on `spawn_blocking` per the threading model — the index is not consulted
/// (the folder is authoritative for the full state).
#[tauri::command]
#[specta::specta]
pub async fn open_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<MeetingState, IpcError> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || open_meeting_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("open_meeting task join failed: {e}"),
        })?
        .map_err(IpcError::from)
}

/// Rename a meeting: updates `metadata.json` (authoritative) then the index row.
///
/// Routes to `persistence::meeting_ops::rename_meeting`, which keeps the on-disk
/// folder and the index consistent (see `architecture/components.md`,
/// `persistence` "Phase 4 surface growth — meeting ops").
#[tauri::command]
#[specta::specta]
pub async fn rename_meeting(
    meeting_id: MeetingId,
    title: String,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    meeting_ops::rename_meeting(&state.meetings_dir, &state.index, meeting_id, &title)
        .await
        .map_err(IpcError::from)
}

/// Delete a meeting: removes the folder then the index row.
///
/// Routes to `persistence::meeting_ops::delete_meeting`.
#[tauri::command]
#[specta::specta]
pub async fn delete_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    meeting_ops::delete_meeting(&state.meetings_dir, &state.index, meeting_id)
        .await
        .map_err(IpcError::from)
}

/// Re-run transcription for a meeting offline (FR-33 action).
///
/// Routes to `Orchestrator::re_transcribe`, which decodes the meeting audio and
/// drives the same batched-VAD + ASR pipeline the live recorder uses, rewrites
/// `transcript.json`, refreshes the index row, and emits
/// `AppEvent::TranscriptSegment` events. Refused while a recording is in
/// progress (the orchestrator returns `AppError::InvalidInput`). The index
/// handle is shared from `IpcState` so the orchestrator need not own one.
#[tauri::command]
#[specta::specta]
pub async fn re_transcribe(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    state
        .orchestrator
        .re_transcribe(&state.index, meeting_id)
        .await
        .map_err(IpcError::from)
}

/// Re-run summarisation for a meeting (FR-33 action) — **Phase 5 stub**.
///
/// The `summariser` crate (Phase 5) produces `summary.md` and emits
/// `AppEvent::SummaryReady`. Until then this command returns
/// `AppError::Unsupported` so the command surface + generated binding exist for
/// Stream B's meeting-list action without a backing implementation.
#[tauri::command]
#[specta::specta]
pub async fn re_summarise(
    meeting_id: MeetingId,
    _state: State<'_, IpcState>,
) -> Result<(), IpcError> {
    tracing::info!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        "re_summarise requested; not yet implemented (Phase 5)"
    );
    Err(IpcError::from(AppError::Unsupported {
        context: "re-summarise is not implemented until Phase 5".into(),
    }))
}

/// Inner body of [`open_meeting`]: assemble the [`MeetingState`] from the
/// meeting folder under `meetings_dir`. Extracted so it can be unit-tested
/// without a Tauri runtime.
fn open_meeting_inner(
    meetings_dir: &Path,
    meeting_id: MeetingId,
) -> Result<MeetingState, AppError> {
    let meeting_dir = meetings_dir.join(meeting_id.0.to_string());
    persistence::read_meeting_state(&meeting_dir)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use meeting_app_common::MeetingId;
    use persistence::{MeetingFolder, MeetingIndex};
    use tempfile::TempDir;

    /// `save_notes` → `load_notes` round-trip through a tempdir `meetings_dir`,
    /// exercising the command bodies directly (no Tauri runtime needed).
    #[test]
    fn save_then_load_round_trips_through_meetings_dir() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id = MeetingId::new();
        // NotesStore writes into an *existing* meeting folder; create it via
        // the owning type so the layout matches production exactly.
        MeetingFolder::create(root, meeting_id).expect("create meeting folder");

        let notes_json = r#"{"type":"doc","content":[{"type":"paragraph","attrs":{"data-anchor-ms":1234},"content":[{"type":"text","text":"hello"}]}]}"#;
        let notes_markdown = "# Notes\n\nhello\n";

        save_notes_inner(root, meeting_id, notes_json, notes_markdown).expect("save");

        let loaded = load_notes_inner(root, meeting_id)
            .expect("load")
            .expect("notes present after save");

        // The markdown round-trips verbatim.
        assert_eq!(loaded.notes_markdown, notes_markdown);
        // The JSON round-trips structurally (re-serialised string may differ in
        // whitespace, so compare parsed values).
        let expected: serde_json::Value = serde_json::from_str(notes_json).unwrap();
        let actual: serde_json::Value = serde_json::from_str(&loaded.notes_json).unwrap();
        assert_eq!(actual, expected, "notes_json must round-trip structurally");
    }

    /// `load_notes` returns `None` when no notes have been saved for a meeting.
    #[test]
    fn load_returns_none_when_no_notes_saved() {
        let tempdir = TempDir::new().expect("tempdir");
        let meeting_id = MeetingId::new();
        let loaded = load_notes_inner(tempdir.path(), meeting_id).expect("load");
        assert!(loaded.is_none(), "absent notes must yield None");
    }

    /// Invalid `notes_json` is rejected as `AppError::InvalidInput`, not written.
    #[test]
    fn save_rejects_invalid_json() {
        let tempdir = TempDir::new().expect("tempdir");
        let meeting_id = MeetingId::new();
        MeetingFolder::create(tempdir.path(), meeting_id).expect("folder");
        let err = save_notes_inner(tempdir.path(), meeting_id, "not json", "")
            .expect_err("invalid JSON must error");
        assert!(matches!(err, AppError::InvalidInput { .. }));
    }

    // -----------------------------------------------------------------------
    // Phase 4 meeting list/open/rename/delete round-trips (no Tauri runtime,
    // no model — a synthetic meeting folder + in-memory libsql index).
    // -----------------------------------------------------------------------

    use meeting_app_common::{AudioFormat, MeetingMeta, Segment};

    /// Write a synthetic meeting folder (`metadata.json` + optional
    /// `transcript.json`) under `root` and return its `MeetingId`. Mirrors the
    /// on-disk layout `persistence` produces so the readers + index agree.
    fn write_synthetic_meeting(
        root: &Path,
        title: &str,
        started_at: &str,
        first_segment_text: Option<&str>,
    ) -> MeetingId {
        let meeting_id = MeetingId::new();
        let folder = MeetingFolder::create(root, meeting_id).expect("create meeting folder");

        let meta = MeetingMeta {
            uuid: meeting_id,
            title: title.to_string(),
            started_at: started_at.to_string(),
            ended_at: Some(started_at.to_string()),
            duration_ms: 60_000,
            speaker_count: 1,
            audio_format: AudioFormat {
                codec: "opus".into(),
                sample_rate: 16_000,
                channels: 1,
                bitrate_kbps: Some(32),
            },
            asr_model: None,
            llm_model: None,
            diarizer: None,
            app_version: "0.0.0".into(),
        };
        let meta_json = serde_json::to_vec_pretty(&meta).expect("serialise metadata");
        std::fs::write(folder.metadata_path(), meta_json).expect("write metadata.json");

        if let Some(text) = first_segment_text {
            let segments = vec![Segment {
                start_ms: 0,
                end_ms: 1_000,
                text: text.to_string(),
                speaker_id: None,
                confidence: None,
                words: Vec::new(),
            }];
            let seg_json = serde_json::to_vec_pretty(&segments).expect("serialise transcript");
            std::fs::write(folder.transcript_path(), seg_json).expect("write transcript.json");
        }

        meeting_id
    }

    /// Open an in-memory index seeded by rebuilding from the meeting folders.
    async fn seeded_index(meetings_root: &Path) -> MeetingIndex {
        let index = MeetingIndex::open(":memory:")
            .await
            .expect("open in-memory index");
        index
            .rebuild_from_disk(meetings_root)
            .await
            .expect("rebuild index from disk");
        index
    }

    /// `list_meetings` returns every indexed meeting, most-recent first, with the
    /// first transcript segment as the excerpt.
    #[tokio::test]
    async fn list_meetings_returns_indexed_rows_most_recent_first() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();

        let _older = write_synthetic_meeting(
            root,
            "Older meeting",
            "2026-06-01T09:00:00Z",
            Some("older excerpt"),
        );
        let _newer = write_synthetic_meeting(
            root,
            "Newer meeting",
            "2026-06-02T09:00:00Z",
            Some("newer excerpt"),
        );

        let index = seeded_index(root).await;
        let rows = index.list_meetings().await.expect("list_meetings");

        assert_eq!(rows.len(), 2, "both meetings must be indexed");
        assert_eq!(rows[0].title, "Newer meeting", "most-recent first");
        assert_eq!(rows[0].excerpt.as_deref(), Some("newer excerpt"));
        assert_eq!(rows[1].title, "Older meeting");
    }

    /// `open_meeting_inner` assembles a `MeetingState` matching what was written
    /// to the synthetic folder (metadata + transcript; no notes saved → None).
    #[test]
    fn open_meeting_returns_meeting_state_matching_disk() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Launch sync", "2026-06-02T10:00:00Z", Some("hello world"));

        let state = open_meeting_inner(root, meeting_id).expect("open_meeting");

        assert_eq!(state.meta.uuid, meeting_id);
        assert_eq!(state.meta.title, "Launch sync");
        assert_eq!(state.transcript.len(), 1);
        assert_eq!(state.transcript[0].text, "hello world");
        assert!(state.notes.is_none(), "no notes saved → None");
    }

    /// `open_meeting_inner` errors for a meeting folder that does not exist.
    #[test]
    fn open_meeting_errors_for_missing_meeting() {
        let tempdir = TempDir::new().expect("tempdir");
        let missing = MeetingId::new();
        let err = open_meeting_inner(tempdir.path(), missing).expect_err("missing meeting must error");
        assert!(matches!(err, AppError::Io { .. }));
    }

    /// `rename_meeting` rewrites `metadata.json` and refreshes the index row so a
    /// subsequent `list_meetings` shows the new title.
    #[tokio::test]
    async fn rename_meeting_updates_disk_and_index() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Old title", "2026-06-02T11:00:00Z", Some("excerpt"));

        let index = seeded_index(root).await;
        meeting_ops::rename_meeting(root, &index, meeting_id, "New title")
            .await
            .expect("rename");

        // On-disk metadata reflects the new title.
        let meeting_dir = root.join(meeting_id.0.to_string());
        let meta = persistence::read_metadata(&meeting_dir).expect("read metadata");
        assert_eq!(meta.title, "New title");

        // Index row reflects the new title.
        let rows = index.list_meetings().await.expect("list");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].title, "New title");
    }

    /// Stop-upsert (FR-33): after a meeting folder is written and its index row
    /// is upserted (the stop-equivalent the `stop_recording` command performs),
    /// `list_meetings` returns it **in the same session** — without a
    /// `rebuild_from_disk`. This is the in-session visibility guarantee that
    /// `Orchestrator::stop` alone does not provide (it finalises the folder but
    /// never touches the index).
    #[tokio::test]
    async fn stop_upsert_makes_meeting_visible_in_session() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();

        // A fresh, EMPTY in-memory index — no rebuild_from_disk. This models a
        // running session where the index was opened at startup and the meeting
        // recorded afterwards.
        let index = MeetingIndex::open(":memory:")
            .await
            .expect("open in-memory index");
        assert!(
            index.list_meetings().await.expect("list").is_empty(),
            "index must start empty (no rebuild)"
        );

        // Write a meeting folder + transcript, exactly as a finished recording
        // leaves on disk.
        let meeting_id = write_synthetic_meeting(
            root,
            "In-session meeting",
            "2026-06-02T13:00:00Z",
            Some("first words of the meeting"),
        );

        // The stop-equivalent: build the list entry from metadata + first
        // transcript segment, then upsert into the live index — exactly what the
        // `stop_recording` command does after `orchestrator.stop()`.
        let meta = persistence::read_metadata(&root.join(meeting_id.0.to_string()))
            .expect("read metadata");
        let entry = meeting_list_entry_for_meta(root, &meta);
        index.upsert(&entry).await.expect("upsert after stop");

        // list_meetings now returns the meeting in the SAME session.
        let rows = index.list_meetings().await.expect("list after upsert");
        assert_eq!(rows.len(), 1, "stopped meeting must be visible without a rebuild");
        assert_eq!(rows[0].id, meeting_id);
        assert_eq!(rows[0].title, "In-session meeting");
        assert_eq!(
            rows[0].excerpt.as_deref(),
            Some("first words of the meeting"),
            "excerpt must be the first transcript segment"
        );
    }

    /// `meeting_list_entry_for_meta` yields `excerpt: None` when the meeting has
    /// no transcript (a zero-segment meeting writes no `transcript.json`).
    #[test]
    fn stop_upsert_entry_has_no_excerpt_without_transcript() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Silent meeting", "2026-06-02T14:00:00Z", None);

        let meta = persistence::read_metadata(&root.join(meeting_id.0.to_string()))
            .expect("read metadata");
        let entry = meeting_list_entry_for_meta(root, &meta);

        assert_eq!(entry.id, meeting_id);
        assert_eq!(entry.title, "Silent meeting");
        assert!(entry.excerpt.is_none(), "no transcript → excerpt None");
    }

    /// `delete_meeting` removes the on-disk folder and the index row.
    #[tokio::test]
    async fn delete_meeting_removes_folder_and_index_row() {
        let tempdir = TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let meeting_id =
            write_synthetic_meeting(root, "Doomed", "2026-06-02T12:00:00Z", Some("excerpt"));

        let index = seeded_index(root).await;
        assert_eq!(index.list_meetings().await.expect("list").len(), 1);

        meeting_ops::delete_meeting(root, &index, meeting_id)
            .await
            .expect("delete");

        let meeting_dir = root.join(meeting_id.0.to_string());
        assert!(!meeting_dir.exists(), "folder must be removed");
        assert!(
            index.list_meetings().await.expect("list").is_empty(),
            "index row must be removed"
        );
    }
}
