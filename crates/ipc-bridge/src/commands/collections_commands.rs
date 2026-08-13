//! Collection ("folder") commands: list / create / rename / delete + meeting filing.
//!
//! A collection is a user-facing folder that groups meetings (UI label: "Folders");
//! distinct from the per-meeting on-disk `notes_crdt::MeetingFolder`. Definitions live in
//! `{app-data}/collections.json` (authoritative); membership is on each meeting's
//! `metadata.json` with a derived `collection_id` mirror in the index for filtered
//! listing. The app-data root is derived from the index path (its parent), which is where
//! `collections.json` sits.
use super::*;

/// Max length for a user-set collection (folder) name. Bounds the persisted
/// value so the UI cannot store an unbounded string (mirrors the speaker-name cap).
const MAX_COLLECTION_NAME_LEN: usize = 128;

/// Validate + normalise a collection name: trim, reject empty / over-long.
fn normalise_collection_name(name: String) -> AppResult<String> {
    let name = name.trim().to_string();
    if name.is_empty() || name.chars().count() > MAX_COLLECTION_NAME_LEN {
        return Err(AppError::InvalidInput {
            context: format!("collection name must be 1..={MAX_COLLECTION_NAME_LEN} characters"),
        });
    }
    Ok(name)
}

/// List all collections (folders), ordered by position. Reads the authoritative
/// `collections.json` (blocking file I/O on `spawn_blocking`).
#[tauri::command]
#[specta::specta]
pub async fn list_collections(state: State<'_, IpcState>) -> AppResult<Vec<Collection>> {
    let root = app_data_root(&state).to_path_buf();
    tokio::task::spawn_blocking(move || collections::CollectionStore::load(&root))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("list_collections task join failed: {e}"),
        })?
}

/// Create a collection named `name`; returns the created [`Collection`].
#[tauri::command]
#[specta::specta]
pub async fn create_collection(
    name: String,
    state: State<'_, IpcState>,
) -> AppResult<Collection> {
    let name = normalise_collection_name(name)?;
    let root = app_data_root(&state).to_path_buf();
    tokio::task::spawn_blocking(move || collections::CollectionStore::create(&root, &name))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("create_collection task join failed: {e}"),
        })?
}

/// Rename the collection `collection_id` to `name`.
#[tauri::command]
#[specta::specta]
pub async fn rename_collection(
    collection_id: CollectionId,
    name: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let name = normalise_collection_name(name)?;
    let root = app_data_root(&state).to_path_buf();
    tokio::task::spawn_blocking(move || {
        collections::CollectionStore::rename(&root, collection_id, &name)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("rename_collection task join failed: {e}"),
    })?
}

/// Delete a collection: clears the membership of every meeting filed under it
/// (those meetings become unfiled), then removes the definition.
#[tauri::command]
#[specta::specta]
pub async fn delete_collection(
    collection_id: CollectionId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let root = app_data_root(&state).to_path_buf();
    collections::delete_collection(&root, &state.meetings_dir, &state.index, collection_id)
        .await
}

/// File a meeting into a collection (`Some(id)`) or unfile it (`None`). Updates
/// `metadata.json` (authoritative) then the index row's derived mirror.
#[tauri::command]
#[specta::specta]
pub async fn set_meeting_collection(
    meeting_id: MeetingId,
    collection_id: Option<CollectionId>,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    meeting_ops::set_meeting_collection(
        &state.meetings_dir,
        &state.index,
        meeting_id,
        collection_id,
    )
    .await
}

/// Reprocess a meeting offline (FR-33 + FR-11 action): re-transcribe THEN
/// re-diarize under a single offline claim.
///
/// Routes to `Orchestrator::reprocess`, which takes ONE `claim_offline`, re-runs
/// the live ASR pipeline over the complete `audio.opus` (rewriting the
/// transcript), then diarizes/splits/merges over that FRESH transcript and
/// finalises ONCE — rewriting `transcript.json` with overlaid `speaker_id`s,
/// updating `metadata.json`'s `{ speaker_count, diarizer }`, refreshing the
/// index row, and emitting the same `AppEvent::TranscriptSegment` +
/// `AppEvent::DiarizationComplete` events the two former passes emitted (the
/// `DiarizationComplete` is emitted by the **orchestrator**, on the shared bus
/// the forwarder subscribes to). The re-diarize step clears
/// `metadata.json`'s `speaker_names` (re-lettering can change who each label
/// is), so any user-assigned speaker names are reset.
///
/// Refused while a recording is in progress (the orchestrator returns
/// `AppError::InvalidInput`). The `model-registry` edge stays inside the
/// orchestrator (both the ASR backend and the diarizer are built there) — there
/// is **no** `ipc-bridge → diarizer` Cargo edge; `ipc-bridge` routes via the
/// orchestrator. The shared `IpcState::index` handle is passed into the call so
/// the orchestrator refreshes the index row without owning one.
#[tauri::command]
#[specta::specta]
pub async fn reprocess(meeting_id: MeetingId, state: State<'_, IpcState>) -> AppResult<()> {
    state
        .orchestrator
        .reprocess(&state.index, meeting_id)
        .await?;

    // After a completed reprocess the diariser has re-lettered speakers, so
    // speaker_names was cleared by finalise_diarization. If voiceprint
    // enrolment is ON, run the §2.4 matcher: accepted matches restore names,
    // uncertain matches emit VoiceprintSuggestions for the UI affordance.
    // Best-effort: a failed match is logged and the meeting is left with
    // cleared names rather than propagating a match error.
    if state.settings.current().voiceprint_enrolment_enabled {
        if let Some(store) = state.voiceprints.as_ref().as_ref() {
            if let Err(e) = state
                .orchestrator
                .apply_voiceprint_matches(meeting_id, store)
                .await
            {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    error = %e,
                    "reprocess: voiceprint matching failed (best-effort)"
                );
            }
        }
    }

    // Re-index the (now-repaired) transcript for retrieval so meeting.db doesn't
    // hold stale chunks after a manual reprocess (best-effort; mirrors the
    // post-stop transcript-index pass — the transcript is finalised here too).
    if let Err(e) = crate::rag_index::index_transcript(&state.chat_handles(), meeting_id).await {
        tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            error = %e,
            "reprocess: transcript RAG re-index failed (best-effort)"
        );
    }

    Ok(())
}

