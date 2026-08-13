//! Meeting list, open, and per-meeting action commands (Phase 4).
use super::*;


/// List all meetings for the meeting-list view (FR-33), most-recent first.
///
/// Reads straight from the libsql `index.db` ([`MeetingIndex::list_meetings`])
/// — a cheap projection that never loads a meeting's full transcript. The index
/// is async (libsql/tokio); the future is awaited here, never `block_on`'d.
#[tauri::command]
#[specta::specta]
pub async fn list_meetings(state: State<'_, IpcState>) -> AppResult<Vec<MeetingListEntry>> {
    // Self-heal: index any meeting present on disk but missing from the cache
    // (e.g. the process killed between finalise and the stop-time upsert) so it
    // can never stay hidden until the next startup `rebuild_from_disk`. Cheap
    // (a readdir + set-diff; only NEW folders are read). Best-effort — a
    // reconcile failure must not break listing, so log and serve the cache.
    if let Err(e) = state.index.reconcile_orphans(&state.meetings_dir).await {
        tracing::warn!(
            target: "ipc-bridge",
            "meeting-list self-heal reconcile failed: {e}; listing cached rows as-is"
        );
    }
    state.index.list_meetings().await
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
) -> AppResult<MeetingState> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || open_meeting_inner(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("open_meeting task join failed: {e}"),
        })?
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
) -> AppResult<()> {
    meeting_ops::rename_meeting(&state.meetings_dir, &state.index, meeting_id, &title)
        .await
}

/// Set a speaker's display name on a saved meeting.
///
/// Maps a diarizer label (e.g. `"A"`) to a display name in `metadata.json`'s
/// `speaker_names`; an empty `name` clears the mapping. Returns the updated
/// map so the caller can re-render without re-reading the meeting. Names are
/// keyed by the diarizer label, so re-running diarization (which re-letters
/// speakers) resets them. Routes to `persistence::meeting_ops::set_speaker_name`.
///
/// The label and name are each capped at `MAX_SPEAKER_NAME_LEN` characters so
/// the UI cannot persist an unbounded value (mirrors the `set_speaker_name`
/// chat tool's bound).
#[tauri::command]
#[specta::specta]
pub async fn set_speaker_name(
    meeting_id: MeetingId,
    label: String,
    name: String,
    state: State<'_, IpcState>,
) -> AppResult<std::collections::BTreeMap<String, String>> {
    const MAX_SPEAKER_NAME_LEN: usize = 512;
    let name = name.trim().to_string();
    if label.chars().count() > MAX_SPEAKER_NAME_LEN || name.chars().count() > MAX_SPEAKER_NAME_LEN {
        return Err(minutist_common::AppError::InvalidInput {
            context: format!("speaker label/name too long (max {MAX_SPEAKER_NAME_LEN} characters)"),
        });
    }
    let result = meeting_ops::set_speaker_name(&state.meetings_dir, meeting_id, &label, &name)
        .await?;

    // Best-effort voiceprint enrolment: gate on the settings flag and a live
    // VoiceprintStore. Errors are logged and swallowed so a rename never fails
    // because of an enrolment problem (§2.3 asymmetry — accepted and documented).
    // The MCP/agent-tools set_speaker_name path (tools.rs) does NOT enrol:
    // it has no audio/diarizer access (§2.3 path asymmetry, intended).
    if state.settings.current().voiceprint_enrolment_enabled {
        if let Some(store) = state.voiceprints.as_ref().as_ref() {
            match state
                .orchestrator
                .enrol_voiceprint(meeting_id, label.clone(), name.clone(), store)
                .await
            {
                Ok(Some(id)) => {
                    tracing::debug!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        label = %label,
                        identity_id = %id.0,
                        "voiceprint enrolled after speaker rename"
                    );
                }
                Ok(None) => {
                    tracing::debug!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        label = %label,
                        "voiceprint enrolment skipped (no clean windows or model unavailable)"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "ipc-bridge",
                        meeting_id = %meeting_id.0,
                        label = %label,
                        "voiceprint enrolment failed (best-effort, ignoring): {e}"
                    );
                }
            }
        }
    }

    Ok(result)
}

/// Move a meeting to the trash: it stays fully recoverable — the folder,
/// voiceprint contributions, and blobs are all left untouched — until
/// [`restore_meeting`] brings it back or [`purge_meeting`] (manual or the
/// 7-day auto-purge sweep) removes it for good.
///
/// Routes to `persistence::meeting_ops::soft_delete_meeting`.
#[tauri::command]
#[specta::specta]
pub async fn delete_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    meeting_ops::soft_delete_meeting(
        &state.meetings_dir,
        &state.index,
        meeting_id,
        state.connected.sync.host_ref().await,
    )
    .await
}

/// Restore a meeting out of the trash — the mirror image of [`delete_meeting`].
///
/// Routes to `persistence::meeting_ops::restore_meeting`.
#[tauri::command]
#[specta::specta]
pub async fn restore_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    meeting_ops::restore_meeting(
        &state.meetings_dir,
        &state.index,
        meeting_id,
        state.connected.sync.host_ref().await,
    )
    .await
}

/// Permanently remove a meeting ("Delete forever" on a trashed row): the
/// folder, its index row, its voiceprint contributions, and its blobs — then
/// records a purged tombstone so a hub replica can never resurrect it from a
/// slow peer (see `persistence::purged`). Unlike a soft delete, this is NOT
/// reversible.
///
/// The voiceprint purge is best-effort: if the `VoiceprintStore` is not open
/// (degraded-to-off) the step is skipped silently. The folder/index deletion
/// runs first so a crash between the two steps leaves at most an orphaned
/// voiceprint entry, not an orphaned meeting folder.
///
/// Routes to `persistence::meeting_ops::purge_meeting`.
#[tauri::command]
#[specta::specta]
pub async fn purge_meeting(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    meeting_ops::purge_meeting(
        &state.meetings_dir,
        app_data_root(&state),
        &state.index,
        state.voiceprints.as_ref().as_ref(),
        meeting_id,
    )
    .await?;

    // Best-effort: unpin this meeting's blobs from the local blob store (a no-op
    // on the free build, or before the sync engine has started).
    if let Err(e) = state.connected.sync.delete_meeting_blobs(meeting_id).await {
        tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "unpinning purged meeting's blobs failed (best-effort): {e}"
        );
    }

    Ok(())
}

/// Open a meeting's on-disk directory (`{meetings_dir}/{uuid}/`) in the host
/// OS file explorer.
///
/// Mirrors [`open_attachment`]'s host hand-off: the path is resolved
/// server-side from `meeting_id` alone (never crosses the IPC boundary) and
/// handed to `tauri-plugin-opener`'s Rust API, so no opener capability scope
/// is required. The existence check runs on `spawn_blocking`; an absent
/// directory is `AppError::InvalidInput` rather than a silent no-op, since
/// calling the opener on a missing path would surface as a confusing OS-level
/// error instead of an app-level one.
#[tauri::command]
#[specta::specta]
pub async fn open_meeting_folder(
    meeting_id: MeetingId,
    app: tauri::AppHandle,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let meetings_dir = state.meetings_dir.clone();
    let path = tokio::task::spawn_blocking(move || -> Result<std::path::PathBuf, AppError> {
        let dir = meetings_dir.join(meeting_id.0.to_string());
        if !dir.is_dir() {
            return Err(AppError::InvalidInput {
                context: format!("meeting {meeting_id:?} has no on-disk directory"),
            });
        }
        Ok(dir)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("open_meeting_folder task join failed: {e}"),
    })??;

    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_path(path.to_string_lossy().into_owned(), None::<&str>)
        .map_err(|e| AppError::Internal {
            context: format!("opening the meeting folder in the host file explorer failed: {e}"),
        })?;
    Ok(())
}

