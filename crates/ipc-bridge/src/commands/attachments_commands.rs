//! Attachment commands (#0016): reference material (files) attached to a meeting and fed
//! to the post-hoc summariser.
use super::*;


/// Upper bound on a stored `original_filename` (characters).
///
/// Mirrors `MAX_SPEAKER_NAME_LEN`. The filename is attacker-influenced (it comes
/// from the OS file picker and is echoed back in the `## Attachment: <name>`
/// reference-material header), so an unbounded one could inflate the manifest
/// and the summariser prompt. 512 is far above any real filename.
pub(crate) const MAX_ATTACHMENT_FILENAME_LEN: usize = 512;

/// Validate + normalise an attachment extension against
/// `doc_convert::supported_exts()`.
///
/// Strips a single leading dot, lower-cases, and rejects anything not in the
/// supported set as `AppError::InvalidInput` (mirrors [`normalise_image_ext`]).
/// Pure so the allowlist gate is verified without a Tauri runtime.
pub(crate) fn normalise_attachment_ext(ext: &str) -> Result<String, AppError> {
    let cleaned = ext.trim().trim_start_matches('.').to_ascii_lowercase();
    if doc_convert::supported_exts().contains(&cleaned.as_str()) {
        Ok(cleaned)
    } else {
        Err(AppError::InvalidInput {
            context: format!(
                "unsupported attachment extension {ext:?}; allowed: {:?}",
                doc_convert::supported_exts()
            ),
        })
    }
}

/// Validate the size/length caps on a candidate attachment, independent of the
/// extension allowlist (see [`normalise_attachment_ext`]). Returns
/// `AppError::InvalidInput` for an over-long filename or oversize bytes.
pub(crate) fn check_attachment_limits(original_filename: &str, byte_len: usize) -> Result<(), AppError> {
    if original_filename.chars().count() > MAX_ATTACHMENT_FILENAME_LEN {
        return Err(AppError::InvalidInput {
            context: format!(
                "attachment filename too long (max {MAX_ATTACHMENT_FILENAME_LEN} characters)"
            ),
        });
    }
    if byte_len > doc_convert::MAX_INPUT_BYTES {
        return Err(AppError::InvalidInput {
            context: format!(
                "attachment exceeds the {} MiB size limit",
                doc_convert::MAX_INPUT_BYTES / 1024 / 1024
            ),
        });
    }
    Ok(())
}

/// Add a reference-material attachment to a meeting.
///
/// Routes DIRECTLY to `persistence` (no orchestrator — attachments are
/// pipeline-independent, like note assets): stores the original bytes under
/// `attachments/<hash>.<ext>`, writes a `Pending` manifest row, emits
/// [`AppEvent::AttachmentAdded`], and enqueues a background conversion job. The
/// blocking filesystem work runs on `spawn_blocking`. Returns the new entry so
/// the webview can insert the row without a re-list.
///
/// `ext` is validated against `doc_convert::supported_exts()` (a non-supported
/// extension is rejected as `AppError::InvalidInput`). The conversion runs on the
/// shared bounded worker; if its queue is full the row is marked `Failed`
/// (back-pressure surfaced) rather than blocking this command.
#[tauri::command]
#[specta::specta]
pub async fn add_attachment(
    meeting_id: MeetingId,
    bytes: Vec<u8>,
    ext: String,
    original_filename: String,
    state: State<'_, IpcState>,
) -> AppResult<AttachmentEntry> {
    let ext = normalise_attachment_ext(&ext)?;
    check_attachment_limits(&original_filename, bytes.len())?;
    let byte_len = bytes.len() as u64;
    let meetings_dir = state.meetings_dir.clone();

    // Store the original + write the Pending manifest row on a blocking thread.
    let entry_ext = ext.clone();
    let entry = tokio::task::spawn_blocking(move || -> Result<AttachmentEntry, AppError> {
        let hash = persistence::save_attachment_original(&meetings_dir, meeting_id, &bytes, &ext)?;
        let entry = AttachmentEntry {
            id: AttachmentId::new(),
            hash,
            original_filename,
            ext,
            byte_len,
            added_at: chrono::Utc::now().to_rfc3339(),
            conversion: ConversionState::Pending,
            converted_md_filename: None,
            awareness: None,
        };
        persistence::add_manifest_entry(&meetings_dir, meeting_id, entry.clone())?;
        Ok(entry)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("add_attachment task join failed: {e}"),
    })??;

    // The row is durable — tell the webview before kicking off conversion.
    let _ = state.event_tx.send(AppEvent::AttachmentAdded {
        meeting_id,
        attachment: entry.clone(),
    });

    // Enqueue conversion on the bounded worker. `try_send` (never blocks the
    // command): a full queue is back-pressure → mark the row Failed + emit so the
    // user sees a clear state rather than a silent stall.
    let job = ConvertJob {
        meeting_id,
        attachment_id: entry.id,
        hash: entry.hash.clone(),
        ext: entry_ext,
    };
    crate::attachments::enqueue_or_mark_failed(
        &state.attachment_convert_tx,
        &state.meetings_dir,
        &state.event_tx,
        meeting_id,
        entry.id,
        job,
    );

    Ok(entry)
}

/// List a meeting's attachments in manifest order.
///
/// Routes directly to `persistence::read_manifest` on `spawn_blocking`. An
/// absent manifest is an empty list.
#[tauri::command]
#[specta::specta]
pub async fn list_attachments(
    meeting_id: MeetingId,
    state: State<'_, IpcState>,
) -> AppResult<Vec<AttachmentEntry>> {
    let meetings_dir = state.meetings_dir.clone();
    tokio::task::spawn_blocking(move || persistence::read_manifest(&meetings_dir, meeting_id))
        .await
        .map_err(|e| AppError::Internal {
            context: format!("list_attachments task join failed: {e}"),
        })?
}

/// Open an attachment original in the HOST OS default application.
///
/// The stored original (`attachments/<hash>.<ext>`) is a real file on disk, so
/// it is handed to the platform opener (`tauri-plugin-opener`) — the OS launches
/// the user's PDF reader / Word / Excel / image viewer for it. The open happens
/// server-side: this command holds the `persistence` edge to resolve the path and
/// passes it to the opener via its Rust API (so no filesystem path crosses the
/// IPC boundary and no opener capability scope is needed). The webview only ever
/// asks "open attachment X"; it never navigates to the file itself.
///
/// An absent attachment id is `AppError::InvalidInput`.
#[tauri::command]
#[specta::specta]
pub async fn open_attachment(
    meeting_id: MeetingId,
    attachment_id: AttachmentId,
    app: tauri::AppHandle,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let meetings_dir = state.meetings_dir.clone();
    let path = tokio::task::spawn_blocking(move || -> Result<std::path::PathBuf, AppError> {
        let manifest = persistence::read_manifest(&meetings_dir, meeting_id)?;
        let entry = manifest.iter().find(|e| e.id == attachment_id).ok_or_else(|| {
            AppError::InvalidInput {
                context: format!("attachment {attachment_id:?} not found in meeting {meeting_id:?}"),
            }
        })?;
        let filename = format!("{}.{}", entry.hash, entry.ext);
        persistence::attachment_original_path(&meetings_dir, meeting_id, &filename)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("open_attachment task join failed: {e}"),
    })??;

    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_path(path.to_string_lossy().into_owned(), None::<&str>)
        .map_err(|e| AppError::Internal {
            context: format!("opening attachment in the host application failed: {e}"),
        })?;
    Ok(())
}

/// Remove an attachment from a meeting.
///
/// Routes directly to `persistence::remove_manifest_entry` (which performs the
/// dedup-safe unlink internally — the `<hash>.<ext>` / `<hash>.md` files are only
/// deleted when no surviving row shares the hash) on `spawn_blocking`, then emits
/// [`AppEvent::AttachmentRemoved`]. Idempotent: removing an absent id is a no-op
/// that still emits (the webview drops the row either way).
#[tauri::command]
#[specta::specta]
pub async fn remove_attachment(
    meeting_id: MeetingId,
    attachment_id: AttachmentId,
    state: State<'_, IpcState>,
) -> AppResult<()> {
    let meetings_dir = state.meetings_dir.clone();
    let dir = meetings_dir.clone();
    // remove_manifest_entry returns the removed entry plus whether its content hash
    // is now orphaned (computed under the manifest lock — the same decision that
    // gates the markdown unlink).
    let removed = tokio::task::spawn_blocking(move || {
        persistence::remove_manifest_entry(&dir, meeting_id, attachment_id)
    })
    .await
    .map_err(|e| AppError::Internal {
        context: format!("remove_attachment task join failed: {e}"),
    })??;

    // RAG (best-effort): once the source content is fully gone (hash orphaned), drop
    // its retrieval chunks so a removed attachment can no longer surface in retrieval.
    if let Some((entry, orphaned)) = &removed {
        if *orphaned {
            crate::rag_index::forget_attachment(&meetings_dir, meeting_id, &entry.hash).await;
        }
    }

    let _ = state.event_tx.send(AppEvent::AttachmentRemoved {
        meeting_id,
        attachment_id,
    });
    Ok(())
}

