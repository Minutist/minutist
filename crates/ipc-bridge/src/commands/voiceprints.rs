//! Voiceprint correction + identity-library management commands (WU5, WU8).
use super::*;


/// Reject a speaker-identity match for a specific meeting and label.
///
/// This is the §2.4 correction path: when the user indicates "this isn't
/// them" for an auto-applied or confirmed name, this command (a) clears the
/// `speaker_names` entry for `label` in `meeting_id` (the label reverts to
/// its bare diarizer letter) and (b) drops the `(meeting_id, label)`
/// contribution from `identity_id`'s gallery, recomputing the affected
/// centroid so the rejected observation no longer influences future matches.
///
/// Silently succeeds when `voiceprint_enrolment_enabled` is OFF or when no
/// `VoiceprintStore` is open (degraded-to-off at startup). Errors from the
/// store or the name-clear are propagated to the caller.
///
/// The `model_id` parameter identifies which embedding model the identity
/// belongs to (required to look up the gallery; use the active model id,
/// e.g. `runner::DIARIZE_EMB_MODEL_ID`). Passing the wrong `model_id`
/// results in a no-op contribution drop (the gallery lookup returns no rows
/// for that model).
#[tauri::command]
#[specta::specta]
pub async fn reject_match(
    meeting_id: MeetingId,
    label: String,
    identity_id: VoiceprintIdentityId,
    model_id: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    if let Some(store) = state.voiceprints.as_ref().as_ref() {
        state
            .orchestrator
            .reject_match(meeting_id, label, identity_id, &model_id, store)
            .await?;
    }
    Ok(())
}

/// Clear the entire voiceprint library (§4 privacy — right to erasure).
///
/// Deletes every identity, centroid, and contribution from `voiceprints.db`.
/// This is the local clear; a full erasure across synced peers is a separate
/// operation (see §4 design — the E2E sync path must also purge replicas).
///
/// Silently succeeds when the `VoiceprintStore` is not open (degraded-to-off).
/// Always succeeds on an already-empty library (idempotent).
#[tauri::command]
#[specta::specta]
pub async fn clear_all_voiceprints(state: State<'_, IpcState>) -> AppResult<()> {
    if let Some(store) = state.voiceprints.as_ref().as_ref() {
        store.clear_all().await?;
        tracing::info!(target: "ipc-bridge", "voiceprint library cleared by user");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// WU8 — identity management commands
// ---------------------------------------------------------------------------

/// One per-centroid summary returned by `list_voiceprints`.
///
/// No embedding vectors — safe for IPC (§2.2: embedding bytes must not cross
/// the IPC boundary).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct CentroidInfo {
    pub centroid_id: String,
    pub sample_count: u64,
    pub condition_label: Option<String>,
}

/// One identity with its gallery, returned by `list_voiceprints`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct VoiceprintIdentityInfo {
    pub identity_id: VoiceprintIdentityId,
    pub display_name: String,
    pub model_id: String,
    pub centroids: Vec<CentroidInfo>,
}

/// Return every enrolled speaker identity with per-condition gallery summaries.
///
/// Returns all identities regardless of embedding model (so the management UI
/// can display and delete identities from previous models). Silently returns an
/// empty list when the `VoiceprintStore` is not open (degraded-to-off).
#[tauri::command]
#[specta::specta]
pub async fn list_voiceprints(
    state: State<'_, IpcState>,
) -> AppResult<Vec<VoiceprintIdentityInfo>> {
    let Some(store) = state.voiceprints.as_ref().as_ref() else {
        return Ok(Vec::new());
    };

    let identities = store.identities_with_gallery().await?;

    Ok(identities
        .into_iter()
        .map(|i| VoiceprintIdentityInfo {
            identity_id: i.identity_id,
            display_name: i.display_name,
            model_id: i.model_id,
            centroids: i
                .centroids
                .into_iter()
                .map(|c| CentroidInfo {
                    centroid_id: c.centroid_id.0.to_string(),
                    sample_count: c.sample_count,
                    condition_label: c.condition_label,
                })
                .collect(),
        })
        .collect())
}

/// Merge two speaker identities: re-home every centroid from `merged_id` to
/// `keep_id`, cap-and-merge the resulting gallery, then delete `merged_id`.
///
/// The caller's UI is responsible for showing a confirmation (including a
/// "which name survives" choice — it must have renamed `keep_id` before
/// calling if the desired name differs). This operation is not reversible
/// once cap-and-merge collapses contributions.
///
/// Silently succeeds when the `VoiceprintStore` is not open (degraded-to-off).
#[tauri::command]
#[specta::specta]
pub async fn merge_voiceprint_identities(
    keep_id: VoiceprintIdentityId,
    merged_id: VoiceprintIdentityId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    if let Some(store) = state.voiceprints.as_ref().as_ref() {
        store
            .merge_identities(keep_id, merged_id)
            .await?;
        tracing::info!(
            target: "ipc-bridge",
            keep_id = %keep_id.0,
            merged_id = %merged_id.0,
            "voiceprint identities merged by user"
        );
    }
    Ok(())
}

/// Rename a speaker identity's display name.
///
/// The new name is trimmed of whitespace; passing a blank name is an error.
/// Silently succeeds when the `VoiceprintStore` is not open (degraded-to-off).
#[tauri::command]
#[specta::specta]
pub async fn rename_voiceprint_identity(
    identity_id: VoiceprintIdentityId,
    new_name: String,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    if let Some(store) = state.voiceprints.as_ref().as_ref() {
        store
            .rename_identity(identity_id, &new_name)
            .await?;
        tracing::info!(
            target: "ipc-bridge",
            identity_id = %identity_id.0,
            "voiceprint identity renamed by user"
        );
    }
    Ok(())
}

/// Delete one speaker identity and all its centroids and contributions.
///
/// This is the per-identity variant of `clear_all_voiceprints`. The deleted
/// identity's data cannot be recovered.
///
/// Silently succeeds when the `VoiceprintStore` is not open (degraded-to-off).
#[tauri::command]
#[specta::specta]
pub async fn delete_voiceprint_identity(
    identity_id: VoiceprintIdentityId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    if let Some(store) = state.voiceprints.as_ref().as_ref() {
        store
            .delete_identity(identity_id)
            .await?;
        tracing::info!(
            target: "ipc-bridge",
            identity_id = %identity_id.0,
            "voiceprint identity deleted by user"
        );
    }
    Ok(())
}

/// Purge all voiceprint contributions from a specified meeting, recomputing
/// affected centroids, dropping zero-contribution centroids, and dropping
/// zero-centroid identities (§4 meeting-granularity erasure — issue #0003).
///
/// `delete_meeting` calls this automatically; this command is exposed for
/// explicit invocation (e.g. clearing acoustic traces from a meeting whose
/// audio was removed via an external path or by a future bulk-erase flow).
///
/// Silently succeeds when the `VoiceprintStore` is not open (degraded-to-off).
#[tauri::command]
#[specta::specta]
pub async fn forget_meeting_voiceprints(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    if let Some(store) = state.voiceprints.as_ref().as_ref() {
        store
            .forget_meeting(meeting_id)
            .await?;
        tracing::info!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            "voiceprint contributions for deleted meeting purged"
        );
    }
    Ok(())
}

