//! Core `ModelRegistry` implementation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::sync::{broadcast, watch, Mutex};

use minutist_common::{
    AppError, AppEvent, AppResult, ModelFileEntry, ModelId, ModelKind, ModelManifestEntry,
    ModelStatus, ModelStatusState,
};

use crate::error::Error;

/// Minimum interval between `ModelDownloadProgress` events — enforces 10 Hz cap.
const PROGRESS_MIN_INTERVAL: Duration = Duration::from_millis(100);

/// Outcome of an in-flight download, published by the owning task to all
/// waiters. `None` is the initial (in-progress) state; `Some(Ok(()))` on
/// success, or `Some(Err(..))` carrying the owner's actual error.
///
/// `AppError` is `Clone`, so each waiter receives a faithful copy of the
/// owner's error rather than a fabricated not-found.
type DownloadOutcome = Option<Result<(), AppError>>;

pub struct ModelRegistry {
    cache_root: PathBuf,
    manifest: Vec<ModelManifestEntry>,
    event_tx: broadcast::Sender<AppEvent>,
    /// Map of model id → a `watch::Receiver` of the in-flight download's
    /// outcome.  A `watch` channel is used (rather than a bare `Notify`)
    /// because it is not subject to lost-wakeup races: a waiter that subscribes
    /// *after* the owner publishes still observes the final value via
    /// `borrow()`, and the stored value carries the owner's real `Result`.
    in_flight: Mutex<HashMap<ModelId, watch::Receiver<DownloadOutcome>>>,
    http: reqwest::Client,
}

impl ModelRegistry {
    /// `cache_root` is `{app-data}/models/`; per-kind subdirs are created on
    /// demand.  `event_tx` is shared with the orchestrator's broadcast channel.
    pub fn new(
        cache_root: PathBuf,
        manifest: Vec<ModelManifestEntry>,
        event_tx: broadcast::Sender<AppEvent>,
    ) -> AppResult<Self> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(Error::Http)?;

        Ok(Self {
            cache_root,
            manifest,
            event_tx,
            in_flight: Mutex::new(HashMap::new()),
            http,
        })
    }

    /// Where this `ModelRegistry` stores files.  Used by `app-main` to seed
    /// pre-downloaded models in dev / CI.
    pub fn cache_root(&self) -> &Path {
        &self.cache_root
    }

    /// Snapshot status of all known models.
    pub fn list_models(&self) -> Vec<ModelStatus> {
        self.manifest
            .iter()
            .map(|entry| self.status_for(entry))
            .collect()
    }

    /// Snapshot status of one model.
    pub fn get_status(&self, model_id: &ModelId) -> AppResult<ModelStatus> {
        let entry = self.find_entry(model_id)?;
        Ok(self.status_for(entry))
    }

    /// Ensures the model's files are present locally and hashes match.
    /// Returns the per-model cache directory.
    ///
    /// If two callers race on the same `model_id`, only one download runs;
    /// the second waits on a shared `watch` channel and adopts the owner's
    /// outcome — the same `Ok(PathBuf)` on success, or the owner's real
    /// `AppError` on failure.
    pub async fn ensure(&self, model_id: &ModelId) -> AppResult<PathBuf> {
        let entry = self.find_entry(model_id)?.clone();
        let model_dir = self.model_dir(&entry);

        // Fast path: files already present and hashes match.
        if self.all_files_valid(&entry, &model_dir).await {
            return Ok(model_dir);
        }

        // In-flight dedup: check whether another caller is already downloading.
        let outcome_tx = {
            let mut in_flight = self.in_flight.lock().await;
            if let Some(existing) = in_flight.get(model_id) {
                // Another download is running — wait for it and adopt its
                // outcome.  Clone the receiver under the lock, then release it
                // before awaiting so the owner can make progress.
                let mut rx = existing.clone();
                drop(in_flight);

                // Wait until the owner publishes a terminal outcome. `watch`
                // has no lost-wakeup race: if the owner already published before
                // we cloned, the current `borrow()` is already `Some`. Otherwise
                // `changed()` resolves when the owner sends.
                let outcome = loop {
                    if let Some(outcome) = rx.borrow().clone() {
                        break outcome;
                    }
                    // The sender is dropped only after publishing, so a
                    // `changed()` error means the value was already set (handled
                    // by the borrow above on the next iteration / below).
                    if rx.changed().await.is_err() {
                        break rx.borrow().clone().unwrap_or(Ok(()));
                    }
                };

                // Adopt the owner's result verbatim so a download/network
                // failure surfaces the owner's actual `AppError` instead of a
                // fabricated not-found.
                return outcome.map(|()| model_dir);
            } else {
                let (tx, rx) = watch::channel::<DownloadOutcome>(None);
                in_flight.insert(model_id.clone(), rx);
                tx
            }
        };

        // This caller owns the download.  Ensure directory exists.
        let result = match tokio::fs::create_dir_all(&model_dir).await {
            Ok(()) => self
                .download_entry(&entry, &model_dir)
                .await
                .map_err(AppError::from),
            Err(e) => Err(Error::Io(e).into()),
        };

        // Publish the owner's real outcome to any waiters, then remove the
        // entry from the in-flight map. Publishing before removal guarantees a
        // concurrent caller either sees the live receiver (and waits for this
        // value) or, after removal, starts its own attempt.
        let _ = outcome_tx.send(Some(result.clone()));
        {
            let mut in_flight = self.in_flight.lock().await;
            in_flight.remove(model_id);
        }
        drop(outcome_tx);

        result.map(|()| model_dir)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn find_entry(&self, model_id: &ModelId) -> AppResult<&ModelManifestEntry> {
        self.manifest
            .iter()
            .find(|e| &e.id == model_id)
            .ok_or_else(|| {
                Error::ManifestEntryNotFound {
                    model_id: model_id.0.clone(),
                }
                .into()
            })
    }

    fn model_dir(&self, entry: &ModelManifestEntry) -> PathBuf {
        let kind_subdir = match entry.kind {
            ModelKind::Asr => "asr",
            ModelKind::Llm => "llm",
            ModelKind::Diarize => "diarize",
            ModelKind::Embed => "embed",
        };
        self.cache_root.join(kind_subdir).join(&entry.id.0)
    }

    fn status_for(&self, entry: &ModelManifestEntry) -> ModelStatus {
        let model_dir = self.model_dir(entry);
        let status = compute_status_sync(entry, &model_dir);
        ModelStatus {
            id: entry.id.clone(),
            kind: entry.kind,
            display_name: entry.display_name.clone(),
            status,
            license: entry.license.clone(),
        }
    }

    /// Returns `true` only when every file in the manifest entry is present
    /// on disk and its SHA-256 matches.
    async fn all_files_valid(&self, entry: &ModelManifestEntry, model_dir: &Path) -> bool {
        for file in &entry.files {
            let path = model_dir.join(&file.filename);
            // Cheap pre-check: an absent or wrong-size file is definitely not
            // valid, so skip hashing it. This avoids reading a multi-GB *partial*
            // download in full just to fail the hash — the slow part of the
            // delay between clicking Download and the resume actually starting.
            match tokio::fs::metadata(&path).await {
                Ok(m) if m.len() == file.size => {}
                _ => return false,
            }
            match verify_file_hash(&path, &file.sha256).await {
                Ok(true) => {}
                _ => return false,
            }
        }
        true
    }

    /// Download all files for `entry` into `model_dir`.
    ///
    /// Progress is reported against the *aggregate* byte total of every file in
    /// the entry (`base_offset` is the sum of fully-fetched prior files), so a
    /// multi-file model (e.g. the ASR gguf + mmproj pair) advances a single
    /// monotonic bar to 100% rather than resetting between files. A final
    /// `bytes_done == grand_total` event is emitted on success so the webview's
    /// completion check fires deterministically — the per-chunk emits are
    /// throttled and the last one always lands short of the total.
    async fn download_entry(
        &self,
        entry: &ModelManifestEntry,
        model_dir: &Path,
    ) -> Result<(), Error> {
        let grand_total: u64 = entry.files.iter().map(|f| f.size).sum();
        let mut base_offset: u64 = 0;
        for file in &entry.files {
            self.download_file(&entry.id, file, model_dir, base_offset, grand_total)
                .await?;
            base_offset += file.size;
        }

        // Terminal 100% event — guarantees the UI sees `bytes_done >= total`.
        let _ = self.event_tx.send(AppEvent::ModelDownloadProgress {
            model_id: entry.id.clone(),
            bytes_done: grand_total,
            bytes_total: Some(grand_total),
        });
        Ok(())
    }

    /// Download a single file, supporting HTTP Range resume.
    ///
    /// Progress events are rate-limited to 10 Hz (100 ms minimum gap).
    async fn download_file(
        &self,
        model_id: &ModelId,
        file: &ModelFileEntry,
        model_dir: &Path,
        base_offset: u64,
        grand_total: u64,
    ) -> Result<(), Error> {
        let dest = model_dir.join(&file.filename);

        // Determine how many bytes are already on disk (for resume).
        let existing_bytes = match tokio::fs::metadata(&dest).await {
            Ok(m) => m.len(),
            Err(_) => 0,
        };

        let mut hasher = Sha256::new();
        let mut bytes_done: u64 = 0;

        let (mut fs_file, request_builder) = if existing_bytes > 0
            && existing_bytes < file.size
        {
            tracing::info!(
                target: "model-registry",
                model_id = %model_id.0,
                filename = %file.filename,
                existing_bytes,
                "resuming partial download"
            );

            // Seed the hasher with the existing partial bytes.
            let partial = tokio::fs::read(&dest).await.map_err(Error::Io)?;
            hasher.update(&partial);
            bytes_done = partial.len() as u64;

            let file_handle = tokio::fs::OpenOptions::new()
                .append(true)
                .open(&dest)
                .await
                .map_err(Error::Io)?;

            let req = self
                .http
                .get(&file.url)
                .header("Range", format!("bytes={}-", existing_bytes));

            (file_handle, req)
        } else {
            // Fresh download (or existing_bytes == file.size handled below).
            if existing_bytes == file.size {
                // File is full-size; verify hash rather than re-downloading.
                match verify_file_hash(&dest, &file.sha256).await {
                    Ok(true) => return Ok(()),
                    Ok(false) => {
                        // Hash mismatch on a "complete" file — delete and restart.
                        tokio::fs::remove_file(&dest).await.map_err(Error::Io)?;
                    }
                    Err(_) => {
                        tokio::fs::remove_file(&dest).await.map_err(Error::Io)?;
                    }
                }
            }

            let file_handle = tokio::fs::File::create(&dest).await.map_err(Error::Io)?;
            let req = self.http.get(&file.url);
            (file_handle, req)
        };

        let response = request_builder.send().await.map_err(Error::Http)?;

        // If we asked for a range but got 200 (not 206), the server doesn't
        // honour Range — discard the partial, restart from scratch.
        if existing_bytes > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            tracing::warn!(
                target: "model-registry",
                model_id = %model_id.0,
                filename = %file.filename,
                "server did not honour Range request; restarting download"
            );
            drop(fs_file);
            tokio::fs::remove_file(&dest).await.map_err(Error::Io)?;
            // Restart without a Range header.
            return Box::pin(self.download_file_fresh(
                model_id,
                file,
                model_dir,
                base_offset,
                grand_total,
            ))
            .await;
        }

        if !response.status().is_success() {
            return Err(Error::Http(
                response.error_for_status().unwrap_err(),
            ));
        }

        tracing::info!(
            target: "model-registry",
            model_id = %model_id.0,
            filename = %file.filename,
            size = file.size,
            "downloading model file"
        );

        let mut last_emit = Instant::now();
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(Error::Http)?;
            hasher.update(&chunk);
            fs_file.write_all(&chunk).await.map_err(Error::Io)?;
            bytes_done += chunk.len() as u64;

            // Emit progress at most 10 Hz.
            let now = Instant::now();
            if now.duration_since(last_emit) >= PROGRESS_MIN_INTERVAL {
                last_emit = now;
                let _ = self.event_tx.send(AppEvent::ModelDownloadProgress {
                    model_id: model_id.clone(),
                    bytes_done: base_offset + bytes_done,
                    bytes_total: Some(grand_total),
                });
            }
        }

        fs_file.flush().await.map_err(Error::Io)?;
        drop(fs_file);

        let actual_hex = format!("{:x}", hasher.finalize());
        if actual_hex != file.sha256 {
            tokio::fs::remove_file(&dest).await.map_err(Error::Io)?;
            return Err(Error::HashMismatch {
                filename: file.filename.clone(),
                expected: file.sha256.clone(),
                actual: actual_hex,
            });
        }

        tracing::info!(
            target: "model-registry",
            model_id = %model_id.0,
            filename = %file.filename,
            "model file download complete and verified"
        );

        Ok(())
    }

    /// Fresh (no Range) download of a single file — used when the server
    /// rejected our Range request.
    async fn download_file_fresh(
        &self,
        model_id: &ModelId,
        file: &ModelFileEntry,
        model_dir: &Path,
        base_offset: u64,
        grand_total: u64,
    ) -> Result<(), Error> {
        let dest = model_dir.join(&file.filename);
        let mut fs_file = tokio::fs::File::create(&dest).await.map_err(Error::Io)?;
        let response = self
            .http
            .get(&file.url)
            .send()
            .await
            .map_err(Error::Http)?
            .error_for_status()
            .map_err(Error::Http)?;

        let mut hasher = Sha256::new();
        let mut bytes_done: u64 = 0;
        let mut last_emit = Instant::now();
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(Error::Http)?;
            hasher.update(&chunk);
            fs_file.write_all(&chunk).await.map_err(Error::Io)?;
            bytes_done += chunk.len() as u64;

            let now = Instant::now();
            if now.duration_since(last_emit) >= PROGRESS_MIN_INTERVAL {
                last_emit = now;
                let _ = self.event_tx.send(AppEvent::ModelDownloadProgress {
                    model_id: model_id.clone(),
                    bytes_done: base_offset + bytes_done,
                    bytes_total: Some(grand_total),
                });
            }
        }

        fs_file.flush().await.map_err(Error::Io)?;
        drop(fs_file);

        let actual_hex = format!("{:x}", hasher.finalize());
        if actual_hex != file.sha256 {
            tokio::fs::remove_file(&dest).await.map_err(Error::Io)?;
            return Err(Error::HashMismatch {
                filename: file.filename.clone(),
                expected: file.sha256.clone(),
                actual: actual_hex,
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute a synchronous status snapshot for `entry` given `model_dir`.
///
/// This does a stat-only check (no hash read) — suitable for the `list_models`
/// and `get_status` snapshot paths where we need speed, not absolute accuracy.
/// The accurate hash check lives in `all_files_valid` (async, reads bytes).
fn compute_status_sync(entry: &ModelManifestEntry, model_dir: &Path) -> ModelStatusState {
    let mut bytes_present: u64 = 0;
    let bytes_total = entry.total_size_bytes;
    let mut all_full = true;

    for file in &entry.files {
        let path = model_dir.join(&file.filename);
        match std::fs::metadata(&path) {
            Ok(m) => {
                let on_disk = m.len();
                bytes_present += on_disk;
                if on_disk < file.size {
                    all_full = false;
                }
            }
            Err(_) => {
                all_full = false;
            }
        }
    }

    if all_full && bytes_present >= bytes_total {
        ModelStatusState::Available {
            local_dir: model_dir.to_string_lossy().into_owned(),
        }
    } else {
        ModelStatusState::Missing {
            bytes_present,
            bytes_total,
        }
    }
}

/// Async hash verification of a single file.  Returns `Ok(true)` if the file
/// exists and its SHA-256 matches `expected_hex`.
pub(crate) async fn verify_file_hash(path: &Path, expected_hex: &str) -> Result<bool, Error> {
    let bytes = match tokio::fs::read(path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(Error::Io(e)),
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual = format!("{:x}", hasher.finalize());
    Ok(actual == expected_hex)
}
