//! Meeting attachments — the bounded background conversion worker.
//!
//! Attachment originals live under `{meetings_dir}/<uuid>/attachments/` (NOT the
//! notes `assets/` subdir — see `architecture/cross-cutting.md`, "Attachments").
//! Opening an original hands its on-disk path to the host OS default application
//! (`open_attachment` → `tauri-plugin-opener`); the bytes never cross the webview.
//!
//! ## Conversion worker
//!
//! Conversion runs as ONE long-lived worker draining a BOUNDED queue (bounded
//! channels only — `architecture/cross-cutting.md`), NOT N detached tasks.
//! `app-main` constructs the `(tx, rx)` pair, stores `tx` on [`crate::IpcState`],
//! and spawns the worker via [`spawn_attachment_convert_worker`]. Each job reads
//! the original, runs `doc_convert::convert_to_markdown` (its own
//! `catch_unwind` + size/zip limits keep a malformed input from crashing the
//! worker), then persists `<hash>.md` and flips the manifest row to
//! [`ConversionState::Ready`] — or records [`ConversionState::Failed`] on error.
//! Every outcome is logged; the worker never panics or exits on a job error
//! (best-effort).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use minutist_common::{AppError, AppEvent, AttachmentId, ConversionState, DocVlm, MeetingId};
use tokio::sync::{broadcast, mpsc};

use crate::chat_runtime::ChatHandles;
use crate::commands;

// ---------------------------------------------------------------------------
// GemmaVlm — the held-summariser-backed DocVlm fallback
// ---------------------------------------------------------------------------

/// The [`DocVlm`] implementation injected into `doc-convert`'s OCR fallback,
/// backed by the **already-held** Gemma-4 summariser.
///
/// `doc-convert` is a `common`-only leaf and reaches the VLM solely through the
/// [`DocVlm`] trait; this type is the concrete `ipc-bridge` side. It carries the
/// shared [`ChatHandles`] so that resolving the model and the vision projector
/// reuses the SAME lazily-loaded `LlamaSummariser` the chat / summarise paths
/// hold — no second model and no second GGUF.
///
/// **Lazy.** Nothing loads at construction. The summariser GGUF and the vision
/// `MtmdContext` come up only on the first OCR call (the first image attachment
/// actually reaching the converter), via [`ChatHandles::ensure_summariser`] and
/// [`summariser::LlamaSummariser::ensure_vision`].
///
/// **Async-in-sync.** [`DocVlm::image_to_markdown`] is synchronous (it is called
/// from `doc_convert::convert_to_markdown` inside the worker's `spawn_blocking`),
/// but resolving the model directory and loading the summariser are async. The
/// implementation bridges with `Handle::block_on` on the ambient Tokio runtime —
/// valid because the call always originates on a `spawn_blocking` thread, never
/// on a runtime worker thread.
#[derive(Clone)]
pub struct GemmaVlm {
    handles: ChatHandles,
}

impl GemmaVlm {
    /// Construct a `GemmaVlm` over the shared chat-runtime handles. Loads
    /// nothing — see the type doc.
    pub fn new(handles: ChatHandles) -> Self {
        Self { handles }
    }
}

/// Locate the multimodal vision projector (`mmproj-*.gguf`) inside a resolved
/// model directory.
///
/// The vision projector ships as a sibling file of the Gemma-4 weights (the
/// model-registry provisions it at onboarding). `find_gguf_weights` in
/// `commands` deliberately SKIPS the `mmproj-*` file when loading the text
/// weights; this is its mirror image — the single `mmproj-*.gguf`.
fn find_mmproj_in_dir(model_dir: &Path) -> Result<PathBuf, AppError> {
    let read_dir = std::fs::read_dir(model_dir).map_err(|e| AppError::ModelLoad {
        model_id: model_dir.display().to_string(),
        context: format!("cannot read model directory: {e}"),
    })?;

    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let is_gguf = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        if is_gguf && name.to_ascii_lowercase().starts_with("mmproj") {
            candidates.push(path);
        }
    }

    match candidates.len() {
        1 => Ok(candidates.pop().expect("len == 1")),
        0 => Err(AppError::ModelLoad {
            model_id: model_dir.display().to_string(),
            context: "no mmproj-*.gguf vision projector found in model directory; \
                      document OCR requires the Gemma-4 vision projector sibling file"
                .into(),
        }),
        n => Err(AppError::ModelLoad {
            model_id: model_dir.display().to_string(),
            context: format!("expected one mmproj-*.gguf vision projector, found {n}"),
        }),
    }
}

impl DocVlm for GemmaVlm {
    fn image_to_markdown(&self, png: &[u8]) -> Result<String, AppError> {
        // Resolve the held summariser + the mmproj sibling path on the ambient
        // runtime. This runs on a `spawn_blocking` thread (the conversion
        // worker's), so a blocking `block_on` is safe (never on a runtime worker
        // thread). A missing runtime is a programming error (the VLM is only
        // ever reached from the worker), surfaced as `Internal`.
        let handle = tokio::runtime::Handle::try_current().map_err(|e| AppError::Internal {
            context: format!("GemmaVlm::image_to_markdown called outside a Tokio runtime: {e}"),
        })?;
        let handles = self.handles.clone();
        let (summariser, mmproj_path) = handle.block_on(async move {
            let model_id = commands::resolve_llm_model_id(&handles.settings.current());
            let model_dir = handles.orchestrator.ensure_model_path(&model_id).await?;
            let mmproj_path = find_mmproj_in_dir(&model_dir)?;
            // Loads the GGUF once (shared with summarise + chat); subsequent OCR
            // jobs reuse the cached handle.
            let summariser = handles.ensure_summariser().await.map_err(AppError::from)?;
            Ok::<_, AppError>((summariser, mmproj_path))
        })?;

        // Build (once) and lock the vision projector, then OCR the page. Both
        // are synchronous; `ensure_vision` is idempotent so repeated pages reuse
        // the cached context.
        summariser.ensure_vision(&mmproj_path)?;
        summariser.image_to_markdown(png)
    }
}

// ---------------------------------------------------------------------------
// Bounded conversion worker
// ---------------------------------------------------------------------------

/// The bound on the conversion job queue. A full queue means `add_attachment`
/// surfaces back-pressure by marking the new row `Failed` rather than blocking
/// the command (see `add_attachment`'s `try_send`).
pub const ATTACHMENT_CONVERT_QUEUE_BOUND: usize = 64;

/// A queued attachment-conversion job. The worker reloads the original bytes
/// from disk (by `<hash>.<ext>`) rather than carrying them in the job so the
/// queue stays small under back-pressure.
#[derive(Debug, Clone)]
pub struct ConvertJob {
    pub meeting_id: MeetingId,
    pub attachment_id: AttachmentId,
    /// Hex SHA-256 of the original — names both `<hash>.<ext>` and `<hash>.md`.
    pub hash: String,
    /// Lower-cased, dot-less extension (drives the `<hash>.<ext>` filename and
    /// the `doc_convert` converter selection).
    pub ext: String,
}

/// Spawn the single long-lived attachment-conversion worker.
///
/// Drains `rx` (the receive half of the bounded `ConvertJob` channel `app-main`
/// constructs). For each job it reads the original, converts it to markdown on a
/// blocking thread, persists `<hash>.md`, flips the manifest row to `Ready`, and
/// emits [`AppEvent::AttachmentConverted`]; on any failure it records
/// `Failed(reason)` and emits [`AppEvent::AttachmentConversionFailed`]. The
/// worker never crashes on a job error (best-effort) and exits cleanly when the
/// sender is dropped.
///
/// `vlm` is the optional image-OCR backend (a [`GemmaVlm`] in production,
/// `None` when no held model is wired). It is cloned into each job and passed to
/// `doc_convert::convert_to_markdown`, which consults it only for direct image
/// attachments — every digital-text path ignores it.
///
/// Uses `tauri::async_runtime::spawn` (like [`crate::spawn_event_forwarder`])
/// because `app-main` calls this from Tauri's `setup()` hook, which has no
/// entered Tokio runtime for a bare `tokio::spawn`.
pub fn spawn_attachment_convert_worker(
    mut rx: mpsc::Receiver<ConvertJob>,
    meetings_dir: PathBuf,
    event_tx: broadcast::Sender<AppEvent>,
    vlm: Option<Arc<dyn DocVlm>>,
) {
    tauri::async_runtime::spawn(async move {
        while let Some(job) = rx.recv().await {
            run_convert_job(&meetings_dir, &event_tx, vlm.clone(), job).await;
        }
        tracing::info!(
            target: "ipc-bridge",
            "attachment conversion channel closed; worker exiting"
        );
    });
}

/// Re-enqueue a `ConvertJob` for every attachment still `Pending` across all
/// meetings, so conversions stranded by a crash (or a clean shutdown with jobs
/// still queued) resume on the next launch.
///
/// Converge-on-startup, mirroring the meeting-index rebuild: `app-main` drives
/// this from `setup()` on a background task after the worker is spawned. It is
/// best-effort — every error is logged and swallowed, it never panics, and a
/// full queue is treated as "already enough work pending" (the remaining rows
/// stay `Pending` and a later launch picks them up). The `tx` is the same
/// bounded sender `add_attachment` uses, so back-pressure behaves identically.
///
/// `meetings_dir` is scanned for per-meeting subdirectories; each manifest is
/// read via `persistence::read_manifest` and every `ConversionState::Pending`
/// row is re-`try_send`. A missing or non-UUID subdirectory is skipped.
pub fn requeue_pending(meetings_dir: &Path, tx: &mpsc::Sender<ConvertJob>) {
    let read_dir = match std::fs::read_dir(meetings_dir) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::warn!(
                target: "ipc-bridge",
                "requeue_pending: cannot read meetings dir: {e}"
            );
            return;
        }
    };

    let mut requeued = 0usize;
    for dir_entry in read_dir.flatten() {
        let Some(name) = dir_entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(uuid) = uuid::Uuid::parse_str(&name) else {
            continue; // not a meeting folder
        };
        let meeting_id = MeetingId(uuid);

        let manifest = match persistence::read_manifest(meetings_dir, meeting_id) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    target: "ipc-bridge",
                    meeting_id = %meeting_id.0,
                    "requeue_pending: skipping unreadable manifest: {e}"
                );
                continue;
            }
        };

        for entry in manifest {
            if !matches!(entry.conversion, ConversionState::Pending) {
                continue;
            }
            let job = ConvertJob {
                meeting_id,
                attachment_id: entry.id,
                hash: entry.hash,
                ext: entry.ext,
            };
            match tx.try_send(job) {
                Ok(()) => requeued += 1,
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::info!(
                        target: "ipc-bridge",
                        "requeue_pending: queue full; remaining Pending rows resume on a later launch"
                    );
                    return;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    tracing::warn!(
                        target: "ipc-bridge",
                        "requeue_pending: conversion worker channel closed; aborting"
                    );
                    return;
                }
            }
        }
    }

    tracing::info!(
        target: "ipc-bridge",
        requeued,
        "requeue_pending: re-enqueued stranded Pending conversions on startup"
    );
}

/// Run one conversion job: read original → convert → persist `.md` → flip the
/// manifest row + emit. Synchronous file/convert work runs on `spawn_blocking`.
/// Best-effort: a failure to convert is recorded on the row + emitted, never
/// propagated (the worker keeps draining the queue).
async fn run_convert_job(
    meetings_dir: &Path,
    event_tx: &broadcast::Sender<AppEvent>,
    vlm: Option<Arc<dyn DocVlm>>,
    job: ConvertJob,
) {
    let ConvertJob {
        meeting_id,
        attachment_id,
        hash,
        ext,
    } = job;

    let meetings_dir_owned = meetings_dir.to_path_buf();
    let original_filename = format!("{hash}.{ext}");
    // The blocking task returns `true` when the manifest row was flipped to
    // Ready, `false` when the row was removed mid-conversion (so the just-written
    // `<hash>.md` was cleaned up and no `Ready` event should be emitted).
    let convert = tokio::task::spawn_blocking(move || -> Result<bool, AppError> {
        let bytes = persistence::read_attachment_original(
            &meetings_dir_owned,
            meeting_id,
            &original_filename,
        )?;
        // The VLM is reached only for image attachments; `convert_to_markdown`
        // ignores it on every digital-text path. `as_deref()` turns the
        // `Option<Arc<dyn DocVlm>>` into the `Option<&dyn DocVlm>` the converter
        // takes.
        let md = doc_convert::convert_to_markdown(&bytes, &ext, vlm.as_deref())?;
        let md_filename =
            persistence::save_attachment_markdown(&meetings_dir_owned, meeting_id, &hash, &md)?;
        // Flip the row to Ready inside the same blocking task (the manifest RMW
        // is brief synchronous std::fs under the per-meeting lock).
        let found = persistence::set_entry_conversion(
            &meetings_dir_owned,
            meeting_id,
            attachment_id,
            ConversionState::Ready,
            Some(md_filename),
        )?;
        if !found {
            // The attachment was removed while this conversion was in flight.
            // `remove_manifest_entry` ran its dedup-safe unlink BEFORE the `.md`
            // existed, so the markdown we just wrote is now orphaned. Clean it up
            // (dedup-safe: only when no surviving row shares the hash) so the
            // remove-then-convert ordering leaves nothing behind.
            persistence::unlink_orphan_attachment_markdown(&meetings_dir_owned, meeting_id, &hash)?;
        }
        Ok(found)
    })
    .await;

    match convert {
        Ok(Ok(true)) => {
            let _ = event_tx.send(AppEvent::AttachmentConverted {
                meeting_id,
                attachment_id,
            });
            tracing::info!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                attachment_id = %attachment_id.0,
                "attachment converted to markdown"
            );
        }
        Ok(Ok(false)) => {
            // Row removed mid-conversion: the markdown was cleaned up above and
            // there is nothing for the webview to act on (it already dropped the
            // row on AttachmentRemoved). No event, no Failed state.
            tracing::info!(
                target: "ipc-bridge",
                meeting_id = %meeting_id.0,
                attachment_id = %attachment_id.0,
                "attachment removed during conversion; discarded converted markdown"
            );
        }
        Ok(Err(e)) => mark_failed(
            meetings_dir,
            event_tx,
            meeting_id,
            attachment_id,
            e.to_string(),
        ),
        Err(join_err) => mark_failed(
            meetings_dir,
            event_tx,
            meeting_id,
            attachment_id,
            format!("conversion task join failed: {join_err}"),
        ),
    }
}

/// Record a failed conversion on the manifest row and emit
/// [`AppEvent::AttachmentConversionFailed`]. Best-effort — a failure to write
/// the Failed state is logged but not propagated (the in-flight conversion is
/// already lost; the event still tells the webview).
pub(crate) fn mark_failed(
    meetings_dir: &Path,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    attachment_id: AttachmentId,
    reason: String,
) {
    tracing::warn!(
        target: "ipc-bridge",
        meeting_id = %meeting_id.0,
        attachment_id = %attachment_id.0,
        "attachment conversion failed: {reason}"
    );
    if let Err(e) = persistence::set_entry_conversion(
        meetings_dir,
        meeting_id,
        attachment_id,
        ConversionState::Failed(reason.clone()),
        None,
    ) {
        tracing::warn!(
            target: "ipc-bridge",
            meeting_id = %meeting_id.0,
            attachment_id = %attachment_id.0,
            "recording Failed conversion state failed: {e}"
        );
    }
    let _ = event_tx.send(AppEvent::AttachmentConversionFailed {
        meeting_id,
        attachment_id,
        reason,
    });
}

/// Enqueue a conversion job, or record the attachment as Failed if the bounded
/// worker queue cannot accept it. `try_send` never blocks the caller: a full
/// queue is back-pressure and a closed channel means the worker is gone — both
/// surface as a Failed row + [`AppEvent::AttachmentConversionFailed`] so the UI
/// shows a clear state rather than a permanent `Pending`.
pub(crate) fn enqueue_or_mark_failed(
    tx: &mpsc::Sender<ConvertJob>,
    meetings_dir: &Path,
    event_tx: &broadcast::Sender<AppEvent>,
    meeting_id: MeetingId,
    attachment_id: AttachmentId,
    job: ConvertJob,
) {
    if let Err(e) = tx.try_send(job) {
        let reason = match e {
            mpsc::error::TrySendError::Full(_) => {
                "conversion queue is full; try again shortly".to_string()
            }
            mpsc::error::TrySendError::Closed(_) => {
                "conversion worker is not running".to_string()
            }
        };
        mark_failed(meetings_dir, event_tx, meeting_id, attachment_id, reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minutist_common::AttachmentEntry;

    #[test]
    fn find_mmproj_in_dir_picks_the_single_projector() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let dir = tempdir.path();
        std::fs::write(dir.join("gemma-4-E4B-it-Q8_0.gguf"), b"weights").expect("write weights");
        std::fs::write(dir.join("mmproj-gemma-4-E4B-it-Q8_0.gguf"), b"proj").expect("write proj");

        let found = find_mmproj_in_dir(dir).expect("mmproj found");
        assert_eq!(
            found.file_name().and_then(|n| n.to_str()),
            Some("mmproj-gemma-4-E4B-it-Q8_0.gguf")
        );
    }

    #[test]
    fn find_mmproj_in_dir_errors_when_absent() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let dir = tempdir.path();
        std::fs::write(dir.join("gemma-4-E4B-it-Q8_0.gguf"), b"weights").expect("write weights");

        assert!(
            matches!(find_mmproj_in_dir(dir), Err(AppError::ModelLoad { .. })),
            "a directory with no mmproj projector must be a ModelLoad error"
        );
    }

    /// The conversion worker marks a row Failed + emits the failure event when
    /// the converter errors (here: an unsupported ext reaches `doc_convert`).
    #[tokio::test]
    async fn run_convert_job_marks_failed_on_convert_error() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        persistence::MeetingFolder::create(root, id).expect("folder");

        // Save an original whose ext `doc_convert` cannot convert so the
        // conversion errors and the worker records Failed.
        let bytes = b"not a real doc".to_vec();
        let hash = persistence::save_attachment_original(root, id, &bytes, "zzz").expect("save");
        let attachment_id = AttachmentId::new();
        persistence::add_manifest_entry(
            root,
            id,
            AttachmentEntry {
                id: attachment_id,
                hash: hash.clone(),
                original_filename: "x.zzz".to_string(),
                ext: "zzz".to_string(),
                byte_len: bytes.len() as u64,
                added_at: chrono::Utc::now().to_rfc3339(),
                conversion: ConversionState::Pending,
                converted_md_filename: None,
            },
        )
        .expect("add manifest");

        let (event_tx, mut event_rx) = broadcast::channel(8);
        run_convert_job(
            root,
            &event_tx,
            None,
            ConvertJob {
                meeting_id: id,
                attachment_id,
                hash,
                ext: "zzz".to_string(),
            },
        )
        .await;

        // The row is now Failed.
        let manifest = persistence::read_manifest(root, id).expect("read manifest");
        let row = manifest
            .iter()
            .find(|e| e.id == attachment_id)
            .expect("row present");
        assert!(
            matches!(row.conversion, ConversionState::Failed(_)),
            "row must be Failed, got {:?}",
            row.conversion
        );

        // And a failure event was emitted.
        let evt = event_rx.try_recv().expect("event emitted");
        assert!(matches!(
            evt,
            AppEvent::AttachmentConversionFailed { attachment_id: aid, .. } if aid == attachment_id
        ));
    }

    /// Seed a saved original + a Pending manifest row, returning its id.
    fn seed_pending(root: &Path, id: MeetingId, ext: &str) -> AttachmentId {
        let bytes = b"seed bytes".to_vec();
        let hash = persistence::save_attachment_original(root, id, &bytes, ext).expect("save");
        let attachment_id = AttachmentId::new();
        persistence::add_manifest_entry(
            root,
            id,
            AttachmentEntry {
                id: attachment_id,
                hash,
                original_filename: format!("seed.{ext}"),
                ext: ext.to_string(),
                byte_len: bytes.len() as u64,
                added_at: chrono::Utc::now().to_rfc3339(),
                conversion: ConversionState::Pending,
                converted_md_filename: None,
            },
        )
        .expect("add manifest");
        attachment_id
    }

    fn job_for(id: MeetingId, attachment_id: AttachmentId) -> ConvertJob {
        ConvertJob {
            meeting_id: id,
            attachment_id,
            hash: String::new(),
            ext: "txt".to_string(),
        }
    }

    #[test]
    fn enqueue_or_mark_failed_marks_failed_when_queue_closed() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        persistence::MeetingFolder::create(root, id).expect("folder");
        let attachment_id = seed_pending(root, id, "txt");

        // A dropped receiver → try_send fails Closed.
        let (tx, rx) = mpsc::channel::<ConvertJob>(4);
        drop(rx);
        let (event_tx, mut event_rx) = broadcast::channel(8);

        enqueue_or_mark_failed(&tx, root, &event_tx, id, attachment_id, job_for(id, attachment_id));

        let manifest = persistence::read_manifest(root, id).expect("manifest");
        let row = manifest.iter().find(|e| e.id == attachment_id).expect("row");
        assert!(
            matches!(row.conversion, ConversionState::Failed(_)),
            "closed queue must mark the row Failed, got {:?}",
            row.conversion
        );
        let evt = event_rx.try_recv().expect("event emitted");
        assert!(matches!(
            evt,
            AppEvent::AttachmentConversionFailed { attachment_id: aid, .. } if aid == attachment_id
        ));
    }

    #[test]
    fn enqueue_or_mark_failed_marks_failed_when_queue_full() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        persistence::MeetingFolder::create(root, id).expect("folder");
        let attachment_id = seed_pending(root, id, "txt");

        // Capacity-1 channel, pre-filled so the next try_send is Full.
        let (tx, _rx) = mpsc::channel::<ConvertJob>(1);
        tx.try_send(job_for(id, attachment_id)).expect("prime the queue");
        let (event_tx, mut event_rx) = broadcast::channel(8);

        enqueue_or_mark_failed(&tx, root, &event_tx, id, attachment_id, job_for(id, attachment_id));

        let manifest = persistence::read_manifest(root, id).expect("manifest");
        let row = manifest.iter().find(|e| e.id == attachment_id).expect("row");
        assert!(
            matches!(row.conversion, ConversionState::Failed(_)),
            "full queue must mark the row Failed, got {:?}",
            row.conversion
        );
        assert!(event_rx.try_recv().is_ok(), "a failure event must be emitted");
    }

    #[test]
    fn requeue_pending_reenqueues_only_pending_and_skips_non_uuid_dirs() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        persistence::MeetingFolder::create(root, id).expect("folder");

        // One Pending row (must requeue) + one Ready row (must NOT).
        let pending_id = seed_pending(root, id, "txt");
        let ready_bytes = b"ready".to_vec();
        let ready_hash =
            persistence::save_attachment_original(root, id, &ready_bytes, "txt").expect("save");
        persistence::add_manifest_entry(
            root,
            id,
            AttachmentEntry {
                id: AttachmentId::new(),
                hash: ready_hash,
                original_filename: "ready.txt".to_string(),
                ext: "txt".to_string(),
                byte_len: ready_bytes.len() as u64,
                added_at: chrono::Utc::now().to_rfc3339(),
                conversion: ConversionState::Ready,
                converted_md_filename: Some("x.md".to_string()),
            },
        )
        .expect("add ready");

        // A non-UUID sibling dir must be skipped, not panic.
        std::fs::create_dir(root.join("not-a-meeting")).expect("mkdir");

        let (tx, mut rx) = mpsc::channel::<ConvertJob>(8);
        requeue_pending(root, &tx);
        drop(tx); // close so the drain terminates

        let mut jobs = Vec::new();
        while let Ok(job) = rx.try_recv() {
            jobs.push(job);
        }
        assert_eq!(jobs.len(), 1, "exactly one Pending row requeued, got {jobs:?}");
        assert_eq!(
            jobs[0].attachment_id, pending_id,
            "the requeued job must be the Pending row, not the Ready one"
        );
    }

    /// Remove-then-convert ordering: when the manifest row is removed BEFORE its
    /// in-flight conversion finishes, the worker must clean up the markdown it
    /// wrote (no orphan `<hash>.md`) and emit NO event — the row is already gone.
    #[tokio::test]
    async fn run_convert_job_removed_mid_flight_leaves_no_orphan_and_no_event() {
        let tempdir = tempfile::TempDir::new().expect("tempdir");
        let root = tempdir.path();
        let id = MeetingId::new();
        persistence::MeetingFolder::create(root, id).expect("folder");

        // A convertible original (plain text → markdown succeeds).
        let bytes = b"hello reference material".to_vec();
        let hash = persistence::save_attachment_original(root, id, &bytes, "txt").expect("save");
        let attachment_id = AttachmentId::new();
        persistence::add_manifest_entry(
            root,
            id,
            AttachmentEntry {
                id: attachment_id,
                hash: hash.clone(),
                original_filename: "ref.txt".to_string(),
                ext: "txt".to_string(),
                byte_len: bytes.len() as u64,
                added_at: chrono::Utc::now().to_rfc3339(),
                conversion: ConversionState::Pending,
                converted_md_filename: None,
            },
        )
        .expect("add manifest");

        // Simulate the remove landing before the conversion completes: drop the
        // row (and its files) up front, then run the job against the now-absent
        // row. The conversion still reads the original it captured by name — but
        // we removed it, so re-save it so the read half succeeds and the race we
        // are pinning is purely the post-write `set_entry_conversion` miss.
        persistence::remove_manifest_entry(root, id, attachment_id).expect("remove");
        persistence::save_attachment_original(root, id, &bytes, "txt").expect("re-save original");

        let (event_tx, mut event_rx) = broadcast::channel(8);
        run_convert_job(
            root,
            &event_tx,
            None,
            ConvertJob {
                meeting_id: id,
                attachment_id,
                hash: hash.clone(),
                ext: "txt".to_string(),
            },
        )
        .await;

        // No `<hash>.md` orphan was left behind (the row is gone).
        let md_path = root
            .join(id.0.to_string())
            .join("attachments")
            .join(format!("{hash}.md"));
        assert!(
            !md_path.exists(),
            "orphaned markdown must be cleaned up after a remove-then-convert race"
        );

        // And NO event was emitted (not Converted, not Failed).
        assert!(
            event_rx.try_recv().is_err(),
            "no event should fire for a row removed mid-conversion"
        );
    }

}
