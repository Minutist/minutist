//! Content-addressed media-blob store (WS4-B S4).
//!
//! Wraps an [`iroh_blobs`] [`FsStore`] so a device can import its meeting media —
//! `audio.opus` and each note asset under `assets/` — into a BLAKE3-addressed
//! store, advertise the resulting [`Hash`]es to a paired peer over the notes
//! channel, and pull the blobs it is missing from that peer over the blobs ALPN,
//! exporting each to the correct per-meeting path.
//!
//! # On-disk location
//!
//! The store lives at `{meetings_root}/.blobs`. The meetings root holds the
//! per-meeting `{uuid}` directories; a meeting UUID's string form never begins
//! with a dot, so a dot-prefixed sibling cannot collide with any `{uuid}` folder.
//! Co-locating the store under the meetings root (rather than elsewhere in the
//! app-data tree) keeps the blob cache in the same XDG data subtree as the
//! meetings it backs, so a backup or wipe of the meetings tree carries the blob
//! cache with it. The store is the device's own redb-backed deduplicating cache;
//! the authoritative media files remain the per-meeting `audio.opus` / `assets/*`
//! that `persistence` owns — the store is a transport-and-dedup layer beside them.
//!
//! # Retention (GC) and reclamation
//!
//! Every payload is pinned with a PERSISTENT, deterministically named tag
//! (`meeting/{id}/audio`, `meeting/{id}/asset/{filename}`,
//! `meeting/{id}/artifact/{rel}`), so an imported or downloaded blob is retained
//! regardless of GC state and can always be found by meeting id. [`BlobStore::open`]
//! enables `iroh-blobs`' periodic mark-and-sweep GC ([`GC_INTERVAL`]): the mark
//! phase roots every blob still referenced by SOME live tag (across every
//! meeting), so a hash that is content-identical across two meetings (e.g. the
//! same pasted image) survives as long as either meeting still tags it — the
//! only reclamation strategy that is safe under content-addressed dedup, since
//! directly deleting a blob's bytes on one meeting's deletion could not tell
//! whether another meeting's tag still points at the same hash. Two paths remove
//! a tag so its blob becomes sweep-eligible:
//!
//! - [`BlobStore::delete_meeting_blobs`] unpins every tag for a deleted meeting
//!   (media + derived artifacts), called from the meeting-deletion path
//!   (`ipc-bridge`'s `delete_meeting` command, via [`crate::SyncEngine`]).
//! - re-tagging a superseded derived artifact ([`BlobStore::import_artifacts`])
//!   overwrites the old tag -> hash mapping with the new one (`tags().set` is a
//!   single-value-per-name map), so the superseded hash is no longer rooted by
//!   that tag once the overwrite lands — no separate unpin step is needed.
//!
//! Named tags survive restart; temp-tags (in-flight import protection) do not.
//! Every persistent tag this crate ever sets starts with [`meeting_tag_prefix`]
//! (`meeting/{id}/...`) — never the `auto-` prefix `iroh-blobs`' own
//! `Tags::create` mints for an un-named tag (see [`Self::import_path`]'s doc for
//! why this crate never lets that path run). [`BlobStore::open`] therefore
//! deletes every `auto-`-prefixed tag on load ([`reclaim_stray_auto_tags`]): a
//! store from before this distinction was enforced could hold one, and the
//! prefix match can only ever remove that stray kind, never a deterministic
//! meeting tag.
//!
//! # Transport
//!
//! [`Self::download`] dials the peer through the SAME [`iroh::Endpoint`] the
//! [`crate::SyncEngine`] already owns (via `store.downloader(&endpoint)`), so the
//! relay path is shared and no second socket is opened. The hash is exchanged
//! out-of-band over the notes channel ([`crate::media_proto`]); there is no blob
//! discovery — the peer's [`EndpointId`] plus the hash is all the downloader needs.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use chrono::{DateTime, FixedOffset};
use futures_util::StreamExt;
use iroh::{Endpoint, EndpointId};
use iroh_blobs::api::downloader::{DownloadProgressItem, Downloader, Shuffled};
use iroh_blobs::api::{Store, TempTag};
use iroh_blobs::store::fs::FsStore;
use iroh_blobs::store::GcConfig;
use iroh_blobs::HashAndFormat;
use minutist_common::{resolve_audio_path, HostRef, MeetingId, SUPPORTED_AUDIO_EXTS};

use crate::timeouts::BLOB_DOWNLOAD_TIMEOUT;
use crate::{Error, Result};

/// The BLAKE3 content hash type, re-exported from [`iroh_blobs`] so callers and
/// the media protocol name it through this crate.
pub use iroh_blobs::Hash;

/// The dot-prefixed store directory under the meetings root. A meeting UUID's
/// string form never starts with a dot, so this cannot collide with a `{uuid}`
/// per-meeting folder.
const STORE_DIR: &str = ".blobs";

/// Interval between the blob store's periodic mark-and-sweep GC runs (see the
/// module doc — "Retention (GC) and reclamation"). A few minutes is generous
/// enough that GC is never the bottleneck on a busy sync burst, while short
/// enough that an unpinned meeting's blobs do not linger indefinitely.
const GC_INTERVAL: Duration = Duration::from_secs(5 * 60);

/// Upper bound on a single downloaded blob's size, in bytes. Meeting audio is the
/// largest legitimate payload; Opus at typical voice bitrates keeps even a
/// multi-hour meeting well under this. Enforced against the downloader's own
/// running byte-offset progress ([`download_capped`]), not a self-reported size —
/// a hostile paired peer fully controls its own advertised manifest (it can
/// declare any size it likes for a hash it also freely chose), so only real-time
/// enforcement during the transfer actually bounds the bytes a peer can force
/// onto this device's disk.
pub(crate) const MAX_BLOB_BYTES: u64 = 2 * 1024 * 1024 * 1024; // 2 GiB

/// One file's `(size, mtime)` memoised against the [`Hash`] `sync` computed for
/// it last time — see [`BlobStore::import_path`].
#[derive(Debug, Clone, Copy)]
struct HashMemoEntry {
    size: u64,
    mtime: std::time::SystemTime,
    hash: Hash,
}

/// A content-addressed store for a device's meeting media, backed by an
/// [`iroh_blobs`] [`FsStore`] at `{meetings_root}/.blobs`.
///
/// Holds the owned [`FsStore`] (kept alive for the lifetime of the engine; the
/// [`iroh_blobs::BlobsProtocol`] registered on the router borrows from a clone of
/// the inner [`Store`]) and is cheap to clone (the store is internally an RPC
/// client handle; the hash memo is an `Arc`, shared by every clone).
#[derive(Debug, Clone)]
pub struct BlobStore {
    store: FsStore,
    /// `(path, size, mtime)` -> hash memo so [`Self::import_path`] can skip
    /// re-hashing (and re-importing into the `iroh-blobs` store) a file whose
    /// metadata hasn't changed since the last reconciliation — every peer
    /// arrival (`push_all_to`) and every `sync_now` re-imports each meeting's
    /// media/artifacts, so an unchanged multi-megabyte `audio.opus` would
    /// otherwise be re-hashed from scratch on each pass. In-memory only, live for
    /// this `BlobStore`'s process lifetime (a restart re-hashes once, which is
    /// cheap relative to the reconciliation traffic this saves); grows by one
    /// entry per distinct path ever imported and is never reclaimed, the same
    /// bounded-by-meeting-count growth `notes_crdt::metadata_lock`'s registry
    /// and the artifact-authority lock registry ([`ARTIFACT_AUTHORITY_LOCKS`])
    /// already accept.
    hash_memo: Arc<Mutex<HashMap<PathBuf, HashMemoEntry>>>,
}

impl BlobStore {
    /// Open (or create) the blob store at `{meetings_root}/.blobs`, with periodic
    /// GC enabled ([`GC_INTERVAL`]) so a tag unpinned via
    /// [`Self::delete_meeting_blobs`] or superseded by a re-tag actually reclaims
    /// its blob's bytes — see the module doc, "Retention (GC) and reclamation".
    /// Also reconciles away any stray `auto-`-prefixed tag left in the store
    /// ([`reclaim_stray_auto_tags`]).
    ///
    /// Creates the directory tree if absent. The meetings root is expected to be
    /// absolute (the app passes the XDG meetings root; tests pass a tempdir).
    pub async fn open(meetings_root: &Path) -> Result<Self> {
        let path = meetings_root.join(STORE_DIR);
        std::fs::create_dir_all(&path)?;
        let db_path = path.join("blobs.db");
        let mut options = iroh_blobs::store::fs::options::Options::new(&path);
        options.gc = Some(GcConfig {
            interval: GC_INTERVAL,
            add_protected: None,
        });
        let store = FsStore::load_with_opts(db_path, options)
            .await
            .map_err(|e| Error::Endpoint(format!("opening blob store at {path:?}: {e}")))?;
        reclaim_stray_auto_tags(&store).await?;
        Ok(Self {
            store,
            hash_memo: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// The inner [`iroh_blobs`] [`Store`], used to construct the
    /// [`iroh_blobs::BlobsProtocol`] handler registered on the router and the
    /// downloader. Crate-internal: the protocol wiring lives in
    /// [`crate::endpoint`].
    pub(crate) fn inner(&self) -> &Store {
        &self.store
    }

    /// Import a meeting's `audio.opus` and every file under its `assets/` into the
    /// store, pinning each with a persistent named tag, and return the resulting
    /// media [`Manifest`].
    ///
    /// A meeting with no `audio.opus` (e.g. notes-only) contributes no audio
    /// entry; an absent `assets/` directory contributes no asset entries. Import
    /// is content-addressed and idempotent: re-importing identical bytes yields
    /// the same [`Hash`] and re-sets the same tag, touching no new storage.
    pub async fn import_meeting(
        &self,
        meetings_root: &Path,
        meeting_id: MeetingId,
    ) -> Result<Manifest> {
        let folder = meetings_root.join(meeting_id.0.to_string());
        let mut entries = Vec::new();

        // Resolve the meeting's actual audio file rather than assuming a fixed
        // container: a desktop recording is `audio.opus`, a synced phone one is
        // `audio.m4a` (AAC-in-MP4, no hardware Opus encoder) — the shared
        // `minutist_common::resolve_audio_path` contract (0048). The manifest
        // carries the real filename so the receiving side writes it under the
        // same name and `persistence` decode-by-extension finds it.
        if let Some(audio) = resolve_audio_path(&folder) {
            let rel = audio
                .file_name()
                .and_then(|n| n.to_str())
                .expect("resolve_audio_path always yields an `audio.<ext>` filename");
            let hash = self.import_path(&audio, &audio_tag(meeting_id)).await?;
            entries.push(ManifestEntry {
                rel_path: rel.to_string(),
                hash,
            });
        }

        let assets_dir = folder.join(ASSETS_DIR);
        if assets_dir.is_dir() {
            // Sort by filename so the manifest order is deterministic across
            // devices (the asset filenames are content hashes, so the order is
            // stable and independent of directory-read order).
            let mut names: Vec<String> = std::fs::read_dir(&assets_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .filter_map(|e| e.file_name().to_str().map(str::to_owned))
                // Skip the atomic-write temp residue persistence may leave behind.
                .filter(|n| !n.ends_with(".tmp"))
                .collect();
            names.sort();
            for name in names {
                let path = assets_dir.join(&name);
                let hash = self
                    .import_path(&path, &asset_tag(meeting_id, &name))
                    .await?;
                entries.push(ManifestEntry {
                    rel_path: format!("{ASSETS_DIR}/{name}"),
                    hash,
                });
            }
        }

        Ok(Manifest { entries })
    }

    /// Download `hash` from `peer` and export it to `{meetings_root}/{uuid}/{rel}`,
    /// pinning it with the per-meeting persistent tag.
    ///
    /// `rel` is one of the relative paths a [`Manifest`] carries (`audio.opus` or
    /// `assets/{filename}`); it is re-validated here (defence in depth — the
    /// protocol already validated the whole manifest) so it cannot escape the
    /// meeting folder. The download dials `peer` through the engine's existing
    /// [`Endpoint`], bounded by [`BLOB_DOWNLOAD_TIMEOUT`] and [`MAX_BLOB_BYTES`]
    /// ([`download_capped`]) so a stalled or hostile peer cannot pin this call or
    /// fill the disk. The export-and-tag step ([`Self::export_and_tag_downloaded`])
    /// keeps the downloaded blob continuously GC-rooted from the moment the
    /// transfer completes through to the deterministic tag landing, and is atomic
    /// with respect to the target file ([`Self::export_atomic`]): a crash or two
    /// concurrent syncs offering different hashes for the same `rel` cannot leave a
    /// torn file at the real path. The returned path is the absolute export target.
    ///
    /// [`Self::export_atomic`] creates the meeting folder itself if absent —
    /// no caller needs to pre-create it.
    pub async fn download(
        &self,
        endpoint: &Endpoint,
        peer: EndpointId,
        meetings_root: &Path,
        meeting_id: MeetingId,
        rel: &str,
        hash: Hash,
    ) -> Result<PathBuf> {
        if !is_safe_rel(rel) {
            return Err(Error::Protocol(format!("unsafe manifest path: {rel:?}")));
        }

        let downloader = self.store.downloader(endpoint);
        tokio::time::timeout(
            BLOB_DOWNLOAD_TIMEOUT,
            download_capped(&downloader, hash, peer, MAX_BLOB_BYTES),
        )
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                %hash, %peer,
                timeout = ?BLOB_DOWNLOAD_TIMEOUT,
                "downloading a media blob timed out"
            );
            Error::Protocol(format!(
                "downloading blob {hash} from {peer} timed out after {BLOB_DOWNLOAD_TIMEOUT:?}"
            ))
        })??;

        let target = meetings_root.join(meeting_id.0.to_string()).join(rel);
        self.export_and_tag_downloaded(hash, &target, &tag_for_rel(meeting_id, rel))
            .await?;
        Ok(target)
    }

    /// Import a meeting's derived artifacts (`transcript.json`, `summary.md`) that
    /// exist on disk into the store — pinning each with a persistent artifact tag —
    /// and return the [`ArtifactManifest`] this device advertises, every entry
    /// stamped with the authority (`produced_by` host + `produced_at`) for those
    /// exact bytes.
    ///
    /// Authority is content-bound, NEVER re-derived from `metadata.json` at
    /// exchange time (the relay-clobber — `planning/DESIGN_artifacts.md` §2 C1):
    ///
    /// - if the per-meeting authority record holds an entry whose hash equals the
    ///   on-disk file's hash, that recorded `(produced_by, produced_at)` is used —
    ///   the device faithfully relays the authority that arrived WITH the bytes,
    ///   and a present `producer_authority` is IGNORED (a received record always
    ///   wins, so a consumer's own stale `Processed` cannot re-stamp bytes it did
    ///   not produce);
    /// - otherwise the bytes were not received over sync (a relay always records on
    ///   receive — see [`record_artifact_authority`]), so THIS device produced
    ///   them: `producer_authority` (the local `Processed { processed_by, at }`
    ///   read from `metadata.json`) supplies the stamp, recorded so a later relay
    ///   is faithful.
    ///
    /// The producer fallback is byte-coherent only under the single-producer-
    /// per-meeting topology this cut targets: a producer's own `metadata.json`
    /// `Processed` matches the bytes it wrote. It rests on the record-on-receive
    /// invariant — which holds because every sync receive records authority under
    /// the per-meeting lock (so a consumer's bytes always carry a matching record
    /// and never reach the fallback). The fully-robust guard (mint only when the
    /// local `Processed.processed_by` is THIS device's own host) needs the
    /// producer-gate's `processed_by` convention, which is unbuilt (DESIGN §7) —
    /// no production path flips a meeting to `Processed` yet, so the fallback is
    /// exercised only by tests and the future producer-gate.
    ///
    /// An artifact present on disk for which neither path establishes authority
    /// (bytes exist, no matching record, the meeting is not locally `Processed`) is
    /// NOT advertised — the device will not stamp bytes it cannot date. A meeting
    /// with no artifact files yields an empty manifest (a zero-segment or
    /// not-yet-summarised `Processed` meeting has nothing to send — not an error).
    ///
    /// The two awaited blob imports run first; the authority read-modify-write then
    /// runs under the per-meeting artifact-authority lock (off the await points), so
    /// two concurrent exchanges for the same meeting cannot lose each other's record
    /// — see [`with_artifact_authority`].
    pub async fn import_artifacts(
        &self,
        meetings_root: &Path,
        meeting_id: MeetingId,
        producer_authority: Option<(HostRef, String)>,
    ) -> Result<ArtifactManifest> {
        let folder = meetings_root.join(meeting_id.0.to_string());

        // Phase 1 (awaited): import each present artifact into the blob store and
        // pin its tag, collecting the (rel, hash) pairs. No authority decision yet.
        let mut imported: Vec<(&'static str, Hash)> = Vec::new();
        for rel in ARTIFACT_RELS {
            let path = folder.join(rel);
            if !path.is_file() {
                continue;
            }
            let hash = self
                .import_path(&path, &artifact_tag(meeting_id, rel))
                .await?;
            imported.push((rel, hash));
        }

        // Phase 2 (synchronous, under the per-meeting lock): resolve each imported
        // artifact's authority against the stored record, falling back to the
        // producer authority, and persist any newly-minted records atomically.
        let entries = with_artifact_authority(meetings_root, meeting_id, |authority| {
            let mut dirty = false;
            let mut entries = Vec::new();
            for &(rel, hash) in &imported {
                let (produced_by, produced_at) = match authority.recorded_for(rel, hash) {
                    Some(rec) => (rec.produced_by.clone(), rec.produced_at.clone()),
                    None => match &producer_authority {
                        Some((by, at)) => {
                            authority.record(rel, hash, by.clone(), at.clone());
                            dirty = true;
                            (by.clone(), at.clone())
                        }
                        None => {
                            tracing::debug!(
                                target: "sync",
                                meeting_id = %meeting_id.0,
                                rel,
                                "artifact present but no provable authority (not synced-in, not locally Processed); not advertising"
                            );
                            continue;
                        }
                    },
                };
                entries.push(ArtifactEntry {
                    rel_path: rel.to_string(),
                    hash,
                    produced_by,
                    produced_at,
                });
            }
            (dirty, entries)
        })?;

        Ok(ArtifactManifest { entries })
    }

    /// Download an artifact `hash` from `peer` and export it to
    /// `{meetings_root}/{uuid}/{rel}` (where `rel` is an [`is_artifact_rel`] path),
    /// pinning it with the per-meeting artifact tag.
    ///
    /// `rel` is re-validated against the artifact allow-list (defence in depth — the
    /// manifest was validated whole). The download is bounded by
    /// [`BLOB_DOWNLOAD_TIMEOUT`] and [`MAX_BLOB_BYTES`] ([`download_capped`]), and
    /// the export-and-tag step ([`Self::export_and_tag_downloaded`]) keeps the
    /// downloaded blob continuously GC-rooted through to the deterministic tag
    /// landing and is atomic with respect to the target file
    /// ([`Self::export_atomic`], which creates the meeting folder itself if
    /// absent): a concurrent reader of `transcript.json` (read far more often
    /// than `audio.opus`) never observes a partial file and a crash cannot
    /// commit a truncated one. The caller records the received authority (it
    /// holds the peer entry's `produced_by`/`produced_at`); this writes only
    /// the artifact file.
    pub async fn download_artifact(
        &self,
        endpoint: &Endpoint,
        peer: EndpointId,
        meetings_root: &Path,
        meeting_id: MeetingId,
        rel: &str,
        hash: Hash,
    ) -> Result<PathBuf> {
        if !is_artifact_rel(rel) {
            return Err(Error::Protocol(format!("not an artifact path: {rel:?}")));
        }

        let downloader = self.store.downloader(endpoint);
        tokio::time::timeout(
            BLOB_DOWNLOAD_TIMEOUT,
            download_capped(&downloader, hash, peer, MAX_BLOB_BYTES),
        )
        .await
        .map_err(|_| {
            tracing::warn!(
                target: "sync",
                %hash, %peer,
                timeout = ?BLOB_DOWNLOAD_TIMEOUT,
                "downloading an artifact blob timed out"
            );
            Error::Protocol(format!(
                "downloading artifact {hash} from {peer} timed out after {BLOB_DOWNLOAD_TIMEOUT:?}"
            ))
        })??;

        let target = meetings_root.join(meeting_id.0.to_string()).join(rel);
        self.export_and_tag_downloaded(hash, &target, &artifact_tag(meeting_id, rel))
            .await?;
        Ok(target)
    }

    /// Export `hash`'s content to `target` atomically: write to a hash-suffixed
    /// sibling `{target-name}.{hash}.tmp` (so two concurrent pulls of the same
    /// target at DIFFERENT hashes never share a tmp and tear each other's export;
    /// same-hash concurrent pulls write byte-identical content to one name,
    /// harmless), fsync it, then atomically rename over `target`. So a crash or
    /// two concurrent syncs offering different hashes for the same path cannot
    /// leave a torn file at `target` — it is always either the previous complete
    /// content or the new complete content, never a partial write. Shared by
    /// [`Self::download`] (media) and [`Self::download_artifact`].
    async fn export_atomic(&self, hash: Hash, target: &Path) -> Result<()> {
        // Own the meeting folder's existence at this boundary rather than
        // relying on `iroh-blobs`' internal export creating it — that is a
        // private implementation detail of a pinned third-party crate, not a
        // contract this codebase can assert on. Idempotent and cheap.
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Protocol(format!("creating export directory {parent:?}: {e}")))?;
        }
        let ext = target.extension().and_then(|e| e.to_str()).unwrap_or("");
        let tmp = target.with_extension(format!("{ext}.{hash}.tmp"));
        self.store
            .blobs()
            .export(hash, &tmp)
            .await
            .map_err(|e| Error::Protocol(format!("exporting blob {hash} to {tmp:?}: {e}")))?;
        // Best-effort fsync of the exported tmp: on a clean host this makes the
        // rename below commit durable bytes (matching persistence's atomic
        // writers), but it is NOT required — the authoritative, content-verified
        // copy lives in the blob store, so a lost export self-heals on the next
        // media sync. On Windows an antivirus real-time scan of the freshly-written
        // tmp frequently holds its handle and fails THIS open with a sharing
        // violation (the observed failure); treat that as best-effort and proceed
        // to the rename rather than fail the whole media/artifacts connection over
        // a durability nicety. A genuine (non-lock) fsync error still fails fast.
        match std::fs::File::open(&tmp).and_then(|f| f.sync_all()) {
            Ok(()) => {}
            Err(e) if is_transient_export_lock(&e) => tracing::debug!(
                target: "sync",
                tmp = ?tmp,
                error = %e,
                "export tmp fsync skipped under a transient lock; relying on blob-store durability"
            ),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(Error::Protocol(format!("fsyncing export tmp {tmp:?}: {e}")));
            }
        }
        // Atomically rename the tmp over `target`, retrying on a transient Windows
        // sharing violation (an antivirus scan holding the tmp/target handle) with
        // exponential backoff over a wide (~9.5s) budget — a real-time scan of a
        // fresh file can hold it for several seconds, and delivering the file a few
        // seconds late beats tearing down the connection and delivering nothing.
        // Terminal errors fail fast and leave no tmp residue (the next media sync
        // re-exports regardless). The retry never fires on non-Windows platforms
        // (those raw codes are not transient there — see `is_transient_export_lock`).
        const RENAME_RETRY_BACKOFF_MS: [u64; 8] = [50, 100, 200, 400, 800, 1500, 2500, 4000];
        retry_on_transient(
            || std::fs::rename(&tmp, target),
            is_transient_export_lock,
            &RENAME_RETRY_BACKOFF_MS,
        )
        .await
        .map_err(|e| {
            // Terminal or budget-exhausted failure: leave no tmp residue — the next
            // media sync re-exports regardless.
            let _ = std::fs::remove_file(&tmp);
            Error::Protocol(format!("renaming export {tmp:?} → {target:?}: {e}"))
        })
    }

    /// Export a just-downloaded blob to `target` and pin it under `tag_name`,
    /// keeping the blob continuously GC-rooted across the whole export.
    /// [`download_capped`] (via [`Self::download`] / [`Self::download_artifact`])
    /// writes the blob's bytes into the store but sets no tag of its own, so a
    /// mark-and-sweep landing between the transfer completing and the
    /// deterministic tag below would otherwise find the blob unrooted and
    /// reclaim it. A [`TempTag`] taken on the already-present hash roots it for
    /// the duration of [`Self::export_atomic`]; [`Self::commit_deterministic_tag`]
    /// then sets `tag_name` while that temp tag still holds and only drops it
    /// once the deterministic tag is live — see the module doc, "Retention (GC)
    /// and reclamation".
    async fn export_and_tag_downloaded(
        &self,
        hash: Hash,
        target: &Path,
        tag_name: &str,
    ) -> Result<()> {
        let temp_tag = self
            .store
            .tags()
            .temp_tag(HashAndFormat::raw(hash))
            .await
            .map_err(|e| {
                Error::Protocol(format!("protecting downloaded blob {hash} for export: {e}"))
            })?;
        self.export_atomic(hash, target).await?;
        self.commit_deterministic_tag(temp_tag, tag_name).await?;
        Ok(())
    }

    /// Set `tag_name` on `temp_tag`'s hash while the temp tag still roots it
    /// against GC, then release the temp tag — so the blob stays continuously
    /// GC-rooted across the handoff from iroh-blobs' own ephemeral protection to
    /// this crate's persistent, deterministically-named tag. Shared by
    /// [`Self::import_path`] (a temp tag from an `add_path`) and
    /// [`Self::export_and_tag_downloaded`] (a temp tag taken explicitly on an
    /// already-downloaded hash) — see the module doc, "Retention (GC) and
    /// reclamation".
    async fn commit_deterministic_tag(&self, temp_tag: TempTag, tag_name: &str) -> Result<Hash> {
        let hash = temp_tag.hash();
        self.tag(tag_name, hash).await?;
        drop(temp_tag);
        Ok(hash)
    }

    /// Unpin every blob tag this device holds for `meeting_id` — its media
    /// (`meeting/{id}/audio`, `meeting/{id}/asset/*`) and any derived-artifact
    /// tags (`meeting/{id}/artifact/*`) — so the underlying bytes become
    /// GC-eligible on the store's next periodic sweep ([`GC_INTERVAL`]). Does not
    /// force an immediate sweep; a hash still tagged by another meeting (dedup)
    /// survives regardless, since GC roots from every remaining tag, not just this
    /// meeting's. Called from the meeting-deletion path
    /// (`ipc-bridge::delete_meeting` via [`crate::SyncEngine::delete_meeting_blobs`]);
    /// idempotent (deleting an already-untagged prefix is a no-op).
    pub async fn delete_meeting_blobs(&self, meeting_id: MeetingId) -> Result<()> {
        let prefix = meeting_tag_prefix(meeting_id);
        self.store
            .tags()
            .delete_prefix(prefix.as_bytes())
            .await
            .map_err(|e| {
                Error::Protocol(format!("unpinning blobs for meeting {meeting_id:?}: {e}"))
            })?;
        Ok(())
    }

    /// Import a single file into the store under the deterministic `tag_name`
    /// and return its [`Hash`].
    ///
    /// Consults [`Self::hash_memo`] first: when `path`'s current `(size, mtime)`
    /// matches a memoised entry, the memoised hash is returned WITHOUT re-hashing
    /// or re-tagging — content-addressing stays correct because a real content
    /// change almost always changes the size and/or mtime, invalidating the memo
    /// and falling through to a real re-import. (The residual risk is the same
    /// one any mtime-based change check accepts — e.g. `rsync`'s quick check,
    /// `make`'s timestamp rule — a same-size rewrite landing within one
    /// filesystem mtime tick; not a concern for the audio/artifact files this
    /// memoises, which are written once by `persistence`/`orchestrator` and never
    /// rewritten in place.) Skipping the re-tag on a memo hit is sound because
    /// `path` is unique per (meeting, rel) across every call site, so a memo
    /// entry can only exist because a PRIOR call already imported and tagged it
    /// under this exact `tag_name`.
    ///
    /// A memo miss imports via `iroh-blobs`' `add_path(path).temp_tag()` rather
    /// than awaiting `add_path(path)` bare: a bare await resolves through
    /// iroh-blobs' `IntoFuture` impl to `with_tag()`, which mints its own
    /// PERSISTENT, uniquely-named `auto-<timestamp>` tag — a second, permanent
    /// root for the same hash that this crate never unpins, defeating
    /// [`Self::delete_meeting_blobs`] for every locally-imported blob.
    /// [`Self::commit_deterministic_tag`] instead sets ONLY `tag_name` while the
    /// import's own temp tag still roots the blob, then drops the temp tag — so
    /// the blob is continuously GC-rooted and no auto tag is ever created. See
    /// the module doc, "Retention (GC) and reclamation".
    async fn import_path(&self, path: &Path, tag_name: &str) -> Result<Hash> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| Error::Protocol(format!("stat-ing {path:?} before import: {e}")))?;
        let size = metadata.len();
        let mtime = metadata
            .modified()
            .map_err(|e| Error::Protocol(format!("reading mtime of {path:?}: {e}")))?;

        if let Some(entry) = self.hash_memo.lock().expect("hash memo poisoned").get(path) {
            if entry.size == size && entry.mtime == mtime {
                return Ok(entry.hash);
            }
        }

        let temp_tag = self
            .store
            .blobs()
            .add_path(path)
            .temp_tag()
            .await
            .map_err(|e| Error::Protocol(format!("importing {path:?} into blob store: {e}")))?;
        let hash = self.commit_deterministic_tag(temp_tag, tag_name).await?;
        self.hash_memo
            .lock()
            .expect("hash memo poisoned")
            .insert(path.to_path_buf(), HashMemoEntry { size, mtime, hash });
        Ok(hash)
    }

    /// Set a persistent named tag pinning `hash` (as a raw blob) against GC.
    /// Overwrites any prior value under `name` — a re-tag (e.g. a superseded
    /// derived artifact in [`Self::import_artifacts`]) un-roots the old hash from
    /// this tag in the same call, so the old bytes are reclaimed on the next GC
    /// sweep unless another tag still pins them (dedup-safe).
    async fn tag(&self, name: &str, hash: Hash) -> Result<()> {
        self.store
            .tags()
            .set(name.as_bytes(), HashAndFormat::raw(hash))
            .await
            .map_err(|e| Error::Protocol(format!("tagging blob {hash} as {name}: {e}")))
    }

    /// Test-only seam: like [`Self::download`]'s size-capped transfer, but with an
    /// explicit `max_bytes` in place of the production [`MAX_BLOB_BYTES`], so a
    /// test can prove the size-cap rejection without needing a multi-gigabyte
    /// payload. Gated behind `test-support`, mirroring [`crate::SyncEngine`]'s
    /// other test seams.
    #[cfg(feature = "test-support")]
    pub(crate) async fn download_capped_for_test(
        &self,
        endpoint: &Endpoint,
        peer: EndpointId,
        hash: Hash,
        max_bytes: u64,
    ) -> Result<()> {
        let downloader = self.store.downloader(endpoint);
        download_capped(&downloader, hash, peer, max_bytes).await
    }
}

/// The tag-name prefix shared by every tag [`BlobStore`] pins for `meeting_id` —
/// media ([`audio_tag`] / [`asset_tag`]) and derived artifacts ([`artifact_tag`])
/// all start with it — so [`BlobStore::delete_meeting_blobs`] can unpin all of
/// them in one `delete_prefix` call.
fn meeting_tag_prefix(meeting_id: MeetingId) -> String {
    format!("meeting/{}/", meeting_id.0)
}

/// The tag-name prefix iroh-blobs' own `Tags::create` (reached via `with_tag()`,
/// `store::util::Tag::auto`) mints for an un-named tag: `auto-<RFC-3339-ish
/// timestamp>`. This crate never sets a tag under this prefix itself — every
/// persistent tag it creates starts with [`meeting_tag_prefix`] — so a tag under
/// it can only be a stray left by an import that (pre-fix) awaited `add_path`
/// bare instead of going through [`BlobStore::import_path`]'s
/// `temp_tag()`-then-[`BlobStore::commit_deterministic_tag`] path.
const AUTO_TAG_PREFIX: &str = "auto-";

/// Delete every tag under [`AUTO_TAG_PREFIX`], called once from
/// [`BlobStore::open`] — see the module doc, "Retention (GC) and reclamation".
/// Scoped to exactly that prefix via `delete_prefix`, so it can only ever
/// remove a stray auto-named tag and can never touch a deterministic
/// `meeting/{id}/...` tag. A no-op (and not an error) when no such tag exists,
/// which is the steady-state case for a store that has only ever been written
/// by the fixed [`BlobStore::import_path`].
async fn reclaim_stray_auto_tags(store: &Store) -> Result<()> {
    let removed = store
        .tags()
        .delete_prefix(AUTO_TAG_PREFIX)
        .await
        .map_err(|e| Error::Protocol(format!("reclaiming stray auto-* blob tags: {e}")))?;
    if removed > 0 {
        tracing::info!(
            target: "sync",
            removed,
            "reclaimed stray auto-named blob tags left by a pre-fix import"
        );
    }
    Ok(())
}

/// Pull `hash` from `peer` through `downloader`, aborting with
/// [`Error::Protocol`] the moment the transfer's running byte offset exceeds
/// `max_bytes`. Watches the downloader's own progress stream rather than
/// checking the blob's size only after a (potentially oversized) transfer has
/// already completed — a hostile peer fully controls the manifest entry it
/// advertises (both the hash and any size it claims for it), so only a live
/// check during the transfer actually bounds what it can write to this device's
/// disk. Dropping the progress stream on a cap breach stops the downloader's
/// background task from writing further bytes.
async fn download_capped(
    downloader: &Downloader,
    hash: Hash,
    peer: EndpointId,
    max_bytes: u64,
) -> Result<()> {
    let mut progress = downloader
        .download(hash, Shuffled::new(vec![peer]))
        .stream()
        .await
        .map_err(|e| Error::Protocol(format!("starting download of {hash} from {peer}: {e}")))?;
    while let Some(item) = progress.next().await {
        match item {
            DownloadProgressItem::Progress(offset) if offset > max_bytes => {
                tracing::warn!(
                    target: "sync",
                    %hash, %peer, offset, cap = max_bytes,
                    "peer's blob exceeded the per-blob size cap; aborting download"
                );
                return Err(Error::Protocol(format!(
                    "blob {hash} from {peer} exceeded the {max_bytes}-byte cap"
                )));
            }
            DownloadProgressItem::Error(e) => {
                return Err(Error::Protocol(format!(
                    "downloading blob {hash} from {peer}: {e}"
                )));
            }
            DownloadProgressItem::DownloadError => {
                return Err(Error::Protocol(format!(
                    "downloading blob {hash} from {peer}: download failed"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Relative path of a meeting's primary audio file within its folder.
/// Name of the note-assets subdirectory within a meeting folder.
pub(crate) const ASSETS_DIR: &str = "assets";

/// Whether `rel` names a meeting's audio file — `audio.<ext>` for any ext in
/// [`minutist_common::SUPPORTED_AUDIO_EXTS`] (`opus` for a desktop recording,
/// `m4a` for a synced phone one). The single-audio-per-meeting invariant means
/// the stem is always `audio`; only the container differs by capture platform.
fn is_audio_rel(rel: &str) -> bool {
    matches!(rel.strip_prefix("audio."), Some(ext) if SUPPORTED_AUDIO_EXTS.contains(&ext))
}

/// The persistent tag name pinning a meeting's audio blob.
fn audio_tag(meeting_id: MeetingId) -> String {
    format!("{}audio", meeting_tag_prefix(meeting_id))
}

/// The persistent tag name pinning one of a meeting's asset blobs.
fn asset_tag(meeting_id: MeetingId, filename: &str) -> String {
    format!("{}asset/{filename}", meeting_tag_prefix(meeting_id))
}

/// The persistent tag name for a (meeting, relative-path) pair: the audio tag for
/// `audio.opus`, an asset tag for `assets/{filename}`. `rel` must already be a
/// safe manifest path.
fn tag_for_rel(meeting_id: MeetingId, rel: &str) -> String {
    match rel.strip_prefix(&format!("{ASSETS_DIR}/")) {
        Some(filename) => asset_tag(meeting_id, filename),
        None => audio_tag(meeting_id),
    }
}

/// A relative path is safe iff it is an audio file (`audio.<ext>` for a
/// supported ext) or `assets/<single-component>` with no separator-escaping or
/// `..` components. Mirrors the persistence asset-filename guard so a hostile
/// manifest cannot direct an export outside the meeting folder.
pub(crate) fn is_safe_rel(rel: &str) -> bool {
    if is_audio_rel(rel) {
        return true;
    }
    let Some(filename) = rel.strip_prefix(&format!("{ASSETS_DIR}/")) else {
        return false;
    };
    if filename.is_empty()
        || filename.contains('/')
        || filename.contains('\\')
        || filename.contains("..")
    {
        return false;
    }
    // Exactly one normal path component after the prefix.
    let mut components = Path::new(filename).components();
    matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(c)), None) if c.to_str() == Some(filename)
    )
}

/// Relative path of a meeting's derived transcript within its folder.
pub(crate) const TRANSCRIPT_REL: &str = "transcript.json";
/// Relative path of a meeting's derived summary within its folder.
pub(crate) const SUMMARY_REL: &str = "summary.md";
/// The derived artifacts the [`crate::artifacts_proto`] exchange syncs, in a fixed
/// order so a manifest built on two devices lists them identically.
pub(crate) const ARTIFACT_RELS: [&str; 2] = [TRANSCRIPT_REL, SUMMARY_REL];

/// Whether `rel` names a derived artifact carried on the [`crate::artifacts_proto`]
/// exchange — exactly `transcript.json` or `summary.md`. An allow-list kept
/// DISJOINT from [`is_safe_rel`] (media) by construction: a derived file must never
/// ride the media union path, whose last-write-wins overwrite is correct for
/// immutable content-addressed media but would clobber a mutable derived output
/// (`planning/DESIGN_artifacts.md` §2).
pub(crate) fn is_artifact_rel(rel: &str) -> bool {
    rel == TRANSCRIPT_REL || rel == SUMMARY_REL
}

/// The persistent tag name pinning one of a meeting's derived-artifact blobs.
/// `rel` must already be an [`is_artifact_rel`] path. A sub-namespace distinct
/// from the media tags (`meeting/{id}/audio|asset/...`) — both still share the
/// [`meeting_tag_prefix`] so [`BlobStore::delete_meeting_blobs`] unpins media and
/// artifacts together for a deleted meeting in one `delete_prefix` call.
fn artifact_tag(meeting_id: MeetingId, rel: &str) -> String {
    format!("{}artifact/{rel}", meeting_tag_prefix(meeting_id))
}

/// Upper bound on the number of entries in a manifest received from a peer. A
/// real meeting carries one `audio.opus` plus its note assets — tens of files in
/// practice — so this ceiling is far above any legitimate manifest while bounding
/// the work a hostile paired peer can force: without it an 8 MiB manifest frame
/// (the [`crate::frame::MAX_FRAME`] cap) could pack on the order of 10^5 minimal
/// entries, each driving a download attempt. The frame cap already bounds
/// allocation; this bounds the per-entry fan-out.
const MAX_MANIFEST_ENTRIES: usize = 4096;

/// A media manifest: the `(relative-path, hash)` pairs for a meeting's
/// `audio.opus` and each note asset. Exchanged over the notes channel
/// ([`crate::media_proto`]); the receiver pulls the entries whose hashes it lacks.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    /// One entry per media file, ordered deterministically (audio first, then
    /// assets sorted by filename).
    pub entries: Vec<ManifestEntry>,
}

/// One `(relative-path, BLAKE3 hash)` pair in a [`Manifest`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestEntry {
    /// Path relative to the meeting folder: `audio.opus` or `assets/{filename}`.
    pub rel_path: String,
    /// BLAKE3 content hash of the file, serialised as its 32 raw bytes.
    #[serde(with = "hash_bytes")]
    pub hash: Hash,
}

impl Manifest {
    /// Validate a manifest received from a peer before any export. Rejects, in
    /// order, a manifest that:
    ///
    /// - carries more than [`MAX_MANIFEST_ENTRIES`] entries (bounds the download
    ///   fan-out a hostile paired peer can force);
    /// - contains an entry whose relative path could escape the meeting folder
    ///   ([`is_safe_rel`]);
    /// - lists the same `rel_path` twice. A local import produces at most one
    ///   entry per path (audio once, each asset filename once), so a duplicate is
    ///   definitionally malformed; left unchecked, two entries for one path would
    ///   each be pulled and the later export would overwrite the earlier at the
    ///   same target — a nondeterministic, peer-controlled outcome.
    pub fn validate(&self) -> Result<()> {
        if self.entries.len() > MAX_MANIFEST_ENTRIES {
            return Err(Error::Protocol(format!(
                "manifest has {} entries, over the cap {MAX_MANIFEST_ENTRIES}",
                self.entries.len()
            )));
        }
        let mut seen = std::collections::HashSet::with_capacity(self.entries.len());
        for entry in &self.entries {
            if !is_safe_rel(&entry.rel_path) {
                return Err(Error::Protocol(format!(
                    "manifest entry has unsafe path: {:?}",
                    entry.rel_path
                )));
            }
            if !seen.insert(entry.rel_path.as_str()) {
                return Err(Error::Protocol(format!(
                    "manifest lists relative path more than once: {:?}",
                    entry.rel_path
                )));
            }
        }
        Ok(())
    }
}

/// Serialise an [`iroh_blobs::Hash`] as its raw 32 bytes (a `Hash` is BLAKE3, so
/// the byte form is canonical and avoids depending on the crate's string
/// encoding on the wire).
mod hash_bytes {
    use super::Hash;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(hash: &Hash, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_bytes(hash.as_bytes())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Hash, D::Error> {
        let bytes = <[u8; 32]>::deserialize(d)?;
        Ok(Hash::from(bytes))
    }
}

/// Upper bound on the number of entries in an artifact manifest received from a
/// peer. A meeting's derived outputs are exactly `transcript.json` + `summary.md`
/// ([`ARTIFACT_RELS`]); the small headroom bounds a hostile peer's pull fan-out
/// while never constraining a legitimate manifest.
const MAX_ARTIFACT_ENTRIES: usize = 8;

/// A derived-artifact manifest: the entries a device advertises for a meeting's
/// `transcript.json` / `summary.md`, each carrying the authority for its exact
/// bytes. Exchanged over the sync ALPN ([`crate::artifacts_proto`]); the receiver
/// pulls each entry that strictly supersedes its own (or that it lacks). Mirrors
/// [`Manifest`] but with per-entry, content-bound authority (`planning/DESIGN_artifacts.md` §2).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactManifest {
    /// One entry per derived artifact present on the advertising device, in
    /// [`ARTIFACT_RELS`] order.
    pub entries: Vec<ArtifactEntry>,
}

/// One derived artifact's `(relative-path, hash)` plus the authority that produced
/// those exact bytes. Unlike a media [`ManifestEntry`], the authority travels WITH
/// the bytes, so the pull decision never consults `metadata.json` (the
/// relay-clobber — DESIGN §2 C1).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactEntry {
    /// Path relative to the meeting folder: `transcript.json` or `summary.md`.
    pub rel_path: String,
    /// BLAKE3 content hash of the file, serialised as its 32 raw bytes.
    #[serde(with = "hash_bytes")]
    pub hash: Hash,
    /// The host that produced these bytes (a meeting's `Processed.processed_by`).
    pub produced_by: HostRef,
    /// When these bytes were produced (a meeting's `Processed.at`), RFC 3339.
    pub produced_at: String,
}

impl ArtifactEntry {
    /// Whether `self` should overwrite `other` for the same `rel_path`: strictly
    /// newer by `produced_at` (parsed to an instant), ties broken by the LOWEST
    /// `produced_by` HostRef. NEVER `>=` (a same-`produced_at`, different-content
    /// pair must not pull on both sides — DESIGN §2 C2) and NEVER a raw-string
    /// compare (mixed fractional precision / offset sorts wrong). A malformed
    /// timestamp on EITHER side is treated as not-superseding, so a parse glitch
    /// can only keep local bytes, never clobber them. Callers short-circuit equal
    /// hashes (identical bytes never need pulling) before consulting this.
    ///
    /// This is the BYTES order: newest `produced_at` wins, so a reprocess by ANY
    /// host (even a higher HostRef) supersedes an older copy. It shares only the
    /// lowest-`HostRef` TIEBREAK with `notes_crdt::merge_processing`'s two-`Processed`
    /// rule — NOT the whole order: that merge is clock-INDEPENDENT (lowest HostRef
    /// regardless of timestamp, §7 D2, so the lifecycle state converges without
    /// trusting clocks), whereas this is clock-dependent by design (newest bytes
    /// win). Under a single producer per meeting the two never disagree (one host
    /// stamps both the lifecycle and the bytes); only a cross-host reprocess —
    /// which the unbuilt producer-gate gates against — could make `metadata.json`'s
    /// `processed_by` name a different host than the on-disk `produced_by`. The
    /// pull is byte-authoritative and never consults `metadata.json`, so that
    /// divergence can never cause a clobber here.
    pub(crate) fn supersedes(&self, other: &ArtifactEntry) -> bool {
        match (
            parse_produced_at(&self.produced_at),
            parse_produced_at(&other.produced_at),
        ) {
            (Some(a), Some(b)) => match a.cmp(&b) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => self.produced_by.0 < other.produced_by.0,
            },
            _ => false,
        }
    }
}

impl ArtifactManifest {
    /// Validate a manifest received from a peer before any export. Rejects, in
    /// order, a manifest that: carries more than [`MAX_ARTIFACT_ENTRIES`] entries;
    /// contains an entry whose path is not an [`is_artifact_rel`] derived artifact;
    /// carries an empty `produced_by` or an unparseable `produced_at` (an unusable
    /// authority would defeat the strict-`>` pull rule and be recorded as garbage);
    /// or lists the same `rel_path` twice (a local import produces at most one entry
    /// per path).
    pub fn validate(&self) -> Result<()> {
        if self.entries.len() > MAX_ARTIFACT_ENTRIES {
            return Err(Error::Protocol(format!(
                "artifact manifest has {} entries, over the cap {MAX_ARTIFACT_ENTRIES}",
                self.entries.len()
            )));
        }
        let mut seen = std::collections::HashSet::with_capacity(self.entries.len());
        for entry in &self.entries {
            if !is_artifact_rel(&entry.rel_path) {
                return Err(Error::Protocol(format!(
                    "artifact manifest entry has a non-artifact path: {:?}",
                    entry.rel_path
                )));
            }
            if entry.produced_by.0.is_empty() {
                return Err(Error::Protocol(format!(
                    "artifact manifest entry {:?} has an empty produced_by",
                    entry.rel_path
                )));
            }
            if parse_produced_at(&entry.produced_at).is_none() {
                return Err(Error::Protocol(format!(
                    "artifact manifest entry {:?} has an unparseable produced_at {:?}",
                    entry.rel_path, entry.produced_at
                )));
            }
            if !seen.insert(entry.rel_path.as_str()) {
                return Err(Error::Protocol(format!(
                    "artifact manifest lists relative path more than once: {:?}",
                    entry.rel_path
                )));
            }
        }
        Ok(())
    }

    /// The entry for `rel`, if present — the local side the receiver compares a
    /// peer entry against via [`ArtifactEntry::supersedes`].
    pub(crate) fn entry(&self, rel: &str) -> Option<&ArtifactEntry> {
        self.entries.iter().find(|e| e.rel_path == rel)
    }
}

/// Parse an RFC 3339 `produced_at` to an instant for the strict-`>` authority
/// comparison; `None` for a malformed timestamp (treated as not-newer by
/// [`ArtifactEntry::supersedes`], so it can only keep local bytes).
fn parse_produced_at(s: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(s).ok()
}

/// Subdirectory under the blob store ([`STORE_DIR`]) holding the per-meeting
/// artifact-authority records. Under `.blobs` (already sync-owned), NOT the
/// meeting folder, so the authority store never widens the set of files sync
/// writes into a meeting's visible namespace beyond the two artifact files.
const ARTIFACT_AUTHORITY_SUBDIR: &str = "artifacts";

/// Per-meeting record binding each synced artifact's content hash to the authority
/// that produced those exact bytes. Persisted at
/// `{meetings_root}/.blobs/artifacts/{meeting_id}.json`, written whenever an
/// artifact's bytes are written (received over sync, or first advertised by the
/// producer), so a device re-advertises the authority that arrived WITH the bytes
/// rather than one re-derived from `metadata.json` (DESIGN §2 C1). Every
/// read-modify-write goes through [`with_artifact_authority`] under the per-meeting
/// lock, so concurrent exchanges for one meeting cannot lose each other's record.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct ArtifactAuthority {
    /// `rel_path` -> the authority recorded for the bytes last written at that path.
    by_rel: BTreeMap<String, RecordedAuthority>,
}

/// The authority recorded for one artifact's current bytes. The `hash` binds the
/// record to specific content, so a byte change (a producer reprocess) invalidates
/// it via a hash mismatch and the authority is re-established.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RecordedAuthority {
    #[serde(with = "hash_bytes")]
    hash: Hash,
    produced_by: HostRef,
    produced_at: String,
}

impl ArtifactAuthority {
    /// The recorded authority for `rel` IFF it was recorded against `hash` (the
    /// bytes currently on disk). A hash mismatch (bytes changed under the record)
    /// returns `None` so the caller re-establishes authority.
    fn recorded_for(&self, rel: &str, hash: Hash) -> Option<&RecordedAuthority> {
        self.by_rel.get(rel).filter(|rec| rec.hash == hash)
    }

    /// Record (or replace) the authority for `rel`'s current bytes.
    fn record(&mut self, rel: &str, hash: Hash, produced_by: HostRef, produced_at: String) {
        self.by_rel.insert(
            rel.to_string(),
            RecordedAuthority {
                hash,
                produced_by,
                produced_at,
            },
        );
    }
}

/// Path of a meeting's artifact-authority record under the blob store.
fn artifact_authority_path(meetings_root: &Path, meeting_id: MeetingId) -> PathBuf {
    meetings_root
        .join(STORE_DIR)
        .join(ARTIFACT_AUTHORITY_SUBDIR)
        .join(format!("{}.json", meeting_id.0))
}

/// Load a meeting's artifact-authority record, defaulting to empty when absent or
/// unreadable. A corrupt record is logged and treated as empty. The worst case is
/// that the artifact is not advertised this round (no provable authority); the pull
/// guard then KEEPS the still-present local bytes rather than overwriting them with
/// a peer copy it cannot prove is newer (see `artifacts_proto::pull_superseding`),
/// so a lost record degrades to fetch-pending, never a clobber.
fn load_artifact_authority(meetings_root: &Path, meeting_id: MeetingId) -> ArtifactAuthority {
    let path = artifact_authority_path(meetings_root, meeting_id);
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            tracing::warn!(
                target: "sync",
                meeting_id = %meeting_id.0,
                %error,
                "artifact-authority record parse failed; treating as empty"
            );
            ArtifactAuthority::default()
        }),
        Err(_) => ArtifactAuthority::default(),
    }
}

/// Persist a meeting's artifact-authority record, creating `.blobs/artifacts/` if
/// needed. Written to a sibling `.tmp` then atomically renamed so a concurrent
/// reader never sees a partial record.
fn save_artifact_authority(
    meetings_root: &Path,
    meeting_id: MeetingId,
    authority: &ArtifactAuthority,
) -> Result<()> {
    let path = artifact_authority_path(meetings_root, meeting_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Error::Protocol(format!("creating artifact-authority dir {parent:?}: {e}"))
        })?;
    }
    let bytes = serde_json::to_vec(authority)
        .map_err(|e| Error::Protocol(format!("encoding artifact-authority record: {e}")))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes)
        .map_err(|e| Error::Protocol(format!("writing artifact-authority record {tmp:?}: {e}")))?;
    std::fs::rename(&tmp, &path)
        .map_err(|e| Error::Protocol(format!("atomically renaming {tmp:?} -> {path:?}: {e}")))
}

/// Process-wide registry of per-meeting artifact-authority-store mutexes — the
/// `.blobs/artifacts/{id}.json` analogue of `notes_crdt::metadata_lock`'s
/// `metadata.json` registry. A separate registry (not a reuse of the metadata
/// lock) so an artifact exchange and a `metadata.json` RMW for the same meeting do
/// not needlessly serialise against each other.
///
/// Each `MeetingId` gets its own `Mutex<()>`; the map grows by one tiny entry per
/// meeting touched and is never reclaimed (the same bounded-by-meeting-count growth
/// the metadata-lock registry accepts).
static ARTIFACT_AUTHORITY_LOCKS: OnceLock<Mutex<HashMap<MeetingId, Arc<Mutex<()>>>>> =
    OnceLock::new();

/// Return (or lazily create) the per-meeting artifact-authority-store lock.
fn artifact_authority_lock(meeting_id: MeetingId) -> Arc<Mutex<()>> {
    ARTIFACT_AUTHORITY_LOCKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("ARTIFACT_AUTHORITY_LOCKS registry poisoned")
        .entry(meeting_id)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Run a read-modify-write on a meeting's artifact-authority record under the
/// per-meeting lock, so two concurrent artifact exchanges for the same meeting (two
/// inbound pulls on a hub, or an inbound pull racing an outbound push) cannot lose
/// each other's authority update via a stale load. `f` receives the loaded record
/// and returns `(dirty, value)`; the record is saved (atomically) iff `dirty`.
///
/// The lock is a [`std::sync::Mutex`] held only across the synchronous load →
/// mutate → save — never across an `.await` (the awaited blob download/import
/// happens before this is called), so it adds no async-runtime entanglement.
fn with_artifact_authority<R>(
    meetings_root: &Path,
    meeting_id: MeetingId,
    f: impl FnOnce(&mut ArtifactAuthority) -> (bool, R),
) -> Result<R> {
    let lock = artifact_authority_lock(meeting_id);
    let _guard = lock.lock().expect("artifact-authority lock poisoned");
    let mut authority = load_artifact_authority(meetings_root, meeting_id);
    let (dirty, value) = f(&mut authority);
    if dirty {
        save_artifact_authority(meetings_root, meeting_id, &authority)?;
    }
    Ok(value)
}

/// Record the authority for an artifact's received bytes — called after
/// [`BlobStore::download_artifact`] with the peer entry's `produced_by` /
/// `produced_at`, so a future [`BlobStore::import_artifacts`] on this device
/// faithfully re-advertises the authority that arrived with the bytes rather than
/// re-deriving it from `metadata.json`. The read-modify-write runs under the
/// per-meeting lock ([`with_artifact_authority`]).
pub(crate) fn record_artifact_authority(
    meetings_root: &Path,
    meeting_id: MeetingId,
    rel: &str,
    hash: Hash,
    produced_by: HostRef,
    produced_at: String,
) -> Result<()> {
    with_artifact_authority(meetings_root, meeting_id, |authority| {
        authority.record(rel, hash, produced_by, produced_at);
        (true, ())
    })
}

/// Whether an export fsync/rename io error is a transient Windows sharing/access
/// violation worth retrying (an antivirus scan or indexer briefly holding the
/// tmp handle): `ERROR_ACCESS_DENIED` (5), `ERROR_SHARING_VIOLATION` (32),
/// `ERROR_LOCK_VIOLATION` (33). This mitigation is Windows-only — off Windows the
/// same raw codes mean unrelated, genuine errors (notably `EIO` is also errno 5),
/// so a fsync/rename failure there is not transient and the export never retries.
#[cfg(windows)]
fn is_transient_export_lock(err: &std::io::Error) -> bool {
    matches!(err.raw_os_error(), Some(5) | Some(32) | Some(33))
}

#[cfg(not(windows))]
fn is_transient_export_lock(_err: &std::io::Error) -> bool {
    false
}

/// Run a fallible filesystem finalize step (`op`), retrying while `is_retryable`
/// says its error is transient — sleeping `backoff_ms[attempt]` between tries and
/// giving up once the schedule is exhausted (returning the last error) or the
/// error is terminal. Used for the export rename under a Windows AV lock; the
/// retry predicate is injected (rather than calling [`is_transient_export_lock`]
/// directly) so the loop is testable independently of the platform-gated
/// classifier.
async fn retry_on_transient<F, P>(
    mut op: F,
    is_retryable: P,
    backoff_ms: &[u64],
) -> std::io::Result<()>
where
    F: FnMut() -> std::io::Result<()>,
    P: Fn(&std::io::Error) -> bool,
{
    let mut attempt = 0usize;
    loop {
        match op() {
            Ok(()) => return Ok(()),
            Err(e) if is_retryable(&e) && attempt < backoff_ms.len() => {
                tracing::warn!(
                    target: "sync",
                    attempt,
                    error = %e,
                    "export rename hit a transient sharing violation; retrying"
                );
                tokio::time::sleep(Duration::from_millis(backoff_ms[attempt])).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_export_lock_is_windows_only() {
        // The three Win32 codes an AV/indexer lock surfaces on the tmp handle are
        // transient ONLY on Windows; off Windows the same raw codes mean unrelated
        // genuine errors (e.g. EIO = 5) and must NOT be retried.
        #[cfg(windows)]
        for code in [5, 32, 33] {
            assert!(
                is_transient_export_lock(&std::io::Error::from_raw_os_error(code)),
                "os error {code} should be retryable on Windows"
            );
        }
        #[cfg(not(windows))]
        for code in [5, 32, 33] {
            assert!(
                !is_transient_export_lock(&std::io::Error::from_raw_os_error(code)),
                "os error {code} must not be retried off Windows (genuine error)"
            );
        }
        // A clearly non-lock error is terminal everywhere.
        assert!(!is_transient_export_lock(&std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no os code"
        )));
    }

    #[tokio::test]
    async fn retry_on_transient_succeeds_after_transient_failures() {
        // Fails twice (transient), then succeeds — the export rename recovering
        // once the AV lock releases.
        let calls = std::cell::Cell::new(0u32);
        let res = retry_on_transient(
            || {
                let n = calls.get();
                calls.set(n + 1);
                if n < 2 {
                    Err(std::io::Error::from_raw_os_error(5))
                } else {
                    Ok(())
                }
            },
            |_| true,
            &[1, 1, 1, 1],
        )
        .await;
        assert!(res.is_ok());
        assert_eq!(calls.get(), 3, "two failures then success = 3 calls");
    }

    #[tokio::test]
    async fn retry_on_transient_gives_up_after_the_bound() {
        // Always transiently fails: initial attempt + one retry per backoff entry,
        // then the last error surfaces.
        let calls = std::cell::Cell::new(0u32);
        let res = retry_on_transient(
            || {
                calls.set(calls.get() + 1);
                Err(std::io::Error::from_raw_os_error(5))
            },
            |_| true,
            &[1, 1, 1],
        )
        .await;
        assert!(res.is_err());
        assert_eq!(calls.get(), 4, "initial + 3 retries, then give up");
    }

    #[tokio::test]
    async fn retry_on_transient_does_not_retry_terminal_errors() {
        // A non-retryable error surfaces immediately, no retries.
        let calls = std::cell::Cell::new(0u32);
        let res = retry_on_transient(
            || {
                calls.set(calls.get() + 1);
                Err(std::io::Error::from_raw_os_error(2))
            },
            |_| false,
            &[1, 1, 1],
        )
        .await;
        assert!(res.is_err());
        assert_eq!(calls.get(), 1, "terminal error is not retried");
    }

    #[test]
    fn safe_rel_accepts_audio_and_assets() {
        assert!(is_safe_rel("audio.opus"));
        assert!(is_safe_rel("audio.m4a"), "a synced phone recording is audio.m4a");
        assert!(is_safe_rel("assets/abc123.png"));
        // The stem must be exactly `audio`, the ext one of the supported set.
        assert!(!is_safe_rel("audio.wav"), "unsupported container");
        assert!(!is_safe_rel("audio."), "no ext");
        assert!(!is_safe_rel("track.m4a"), "wrong stem");
    }

    #[tokio::test]
    async fn import_meeting_picks_up_an_m4a_audio_file() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let store = BlobStore::open(dir.path()).await.expect("store");
        let meeting = MeetingId(uuid::Uuid::new_v4());
        let folder = dir.path().join(meeting.0.to_string());
        std::fs::create_dir_all(&folder).expect("mkdir");
        // A phone recording: AAC-in-MP4 stored under the honest .m4a extension.
        std::fs::write(folder.join("audio.m4a"), b"fake-aac-bytes").expect("write audio");

        let manifest = store
            .import_meeting(dir.path(), meeting)
            .await
            .expect("import");
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(
            manifest.entries[0].rel_path, "audio.m4a",
            "the manifest carries the real container filename, not a fixed audio.opus"
        );
    }

    #[test]
    fn safe_rel_rejects_traversal() {
        for evil in [
            "../secret",
            "assets/../../etc/passwd",
            "assets/sub/dir.png",
            "assets/",
            "assets/..",
            "/etc/passwd",
            "assets\\win.png",
            "notes.json",
            "metadata.json",
            "",
        ] {
            assert!(!is_safe_rel(evil), "expected {evil:?} rejected");
        }
    }

    #[test]
    fn manifest_validate_rejects_unsafe_entry() {
        let m = Manifest {
            entries: vec![ManifestEntry {
                rel_path: "../escape".to_string(),
                hash: Hash::from([0u8; 32]),
            }],
        };
        assert!(matches!(m.validate(), Err(Error::Protocol(_))));
    }

    #[test]
    fn manifest_validate_rejects_too_many_entries() {
        // One past the cap is rejected; a manifest at the cap of distinct safe
        // paths is accepted (the per-entry path safety is already covered
        // elsewhere, so use audio + distinct asset filenames here).
        let over = MAX_MANIFEST_ENTRIES + 1;
        let m = Manifest {
            entries: (0..over)
                .map(|i| ManifestEntry {
                    rel_path: format!("assets/a{i}.png"),
                    hash: Hash::from([0u8; 32]),
                })
                .collect(),
        };
        assert!(matches!(m.validate(), Err(Error::Protocol(_))));

        let at_cap = Manifest {
            entries: (0..MAX_MANIFEST_ENTRIES)
                .map(|i| ManifestEntry {
                    rel_path: format!("assets/a{i}.png"),
                    hash: Hash::from([0u8; 32]),
                })
                .collect(),
        };
        assert!(at_cap.validate().is_ok());
    }

    #[test]
    fn manifest_validate_rejects_duplicate_rel_path() {
        // Two entries for the same path (here two `audio.opus` rows with different
        // hashes) is malformed — a local import never produces it — and would
        // otherwise let the peer pick the final on-disk bytes by entry order.
        let m = Manifest {
            entries: vec![
                ManifestEntry {
                    rel_path: "audio.opus".to_string(),
                    hash: Hash::from([1u8; 32]),
                },
                ManifestEntry {
                    rel_path: "audio.opus".to_string(),
                    hash: Hash::from([2u8; 32]),
                },
            ],
        };
        assert!(matches!(m.validate(), Err(Error::Protocol(_))));
    }

    #[test]
    fn manifest_serde_round_trips_hash_bytes() {
        let m = Manifest {
            entries: vec![ManifestEntry {
                rel_path: "audio.opus".to_string(),
                hash: Hash::from([7u8; 32]),
            }],
        };
        let bytes = serde_json::to_vec(&m).expect("encode");
        let back: Manifest = serde_json::from_slice(&bytes).expect("decode");
        assert_eq!(m, back);
    }

    fn art_entry(rel: &str, hash: u8, by: &str, at: &str) -> ArtifactEntry {
        ArtifactEntry {
            rel_path: rel.to_string(),
            hash: Hash::from([hash; 32]),
            produced_by: HostRef(by.to_string()),
            produced_at: at.to_string(),
        }
    }

    #[test]
    fn artifact_rel_is_disjoint_from_media() {
        assert!(is_artifact_rel("transcript.json"));
        assert!(is_artifact_rel("summary.md"));
        // Disjoint from the media allow-list: neither set accepts the other's
        // paths, so a derived file can never ride the media union path.
        assert!(!is_artifact_rel("audio.opus"));
        assert!(!is_artifact_rel("assets/x.png"));
        assert!(!is_safe_rel("transcript.json"));
        assert!(!is_safe_rel("summary.md"));
        // Traversal / unknowns rejected.
        assert!(!is_artifact_rel("../etc/passwd"));
        assert!(!is_artifact_rel("metadata.json"));
        assert!(!is_artifact_rel(""));
    }

    #[test]
    fn supersedes_is_strict_then_lowest_hostref() {
        let older = art_entry("transcript.json", 1, "b", "2026-06-30T10:00:00Z");
        let newer = art_entry("transcript.json", 2, "b", "2026-06-30T10:05:00Z");
        assert!(newer.supersedes(&older));
        assert!(!older.supersedes(&newer));
        // Equal produced_at, distinct content: NOT >= — lowest HostRef wins and the
        // higher HostRef does not supersede, so the same pair never pulls both ways.
        let a = art_entry("transcript.json", 3, "a", "2026-06-30T10:00:00Z");
        let b = art_entry("transcript.json", 4, "b", "2026-06-30T10:00:00Z");
        assert!(a.supersedes(&b));
        assert!(!b.supersedes(&a));
        // Equal produced_at AND host: neither supersedes (no pull either way).
        let a2 = art_entry("transcript.json", 5, "a", "2026-06-30T10:00:00Z");
        assert!(!a.supersedes(&a2));
        assert!(!a2.supersedes(&a));
        // A malformed timestamp on EITHER side never supersedes (keeps local).
        let bad = art_entry("transcript.json", 6, "a", "not-a-time");
        assert!(!bad.supersedes(&newer));
        assert!(!newer.supersedes(&bad));
    }

    #[test]
    fn artifact_manifest_validate_rejects_malformed() {
        let over = ArtifactManifest {
            entries: (0..MAX_ARTIFACT_ENTRIES + 1)
                .map(|_| art_entry("transcript.json", 1, "a", "2026-06-30T10:00:00Z"))
                .collect(),
        };
        assert!(matches!(over.validate(), Err(Error::Protocol(_))));
        let bad_path = ArtifactManifest {
            entries: vec![art_entry("audio.opus", 1, "a", "2026-06-30T10:00:00Z")],
        };
        assert!(matches!(bad_path.validate(), Err(Error::Protocol(_))));
        let no_by = ArtifactManifest {
            entries: vec![art_entry("transcript.json", 1, "", "2026-06-30T10:00:00Z")],
        };
        assert!(matches!(no_by.validate(), Err(Error::Protocol(_))));
        let bad_at = ArtifactManifest {
            entries: vec![art_entry("summary.md", 1, "a", "yesterday")],
        };
        assert!(matches!(bad_at.validate(), Err(Error::Protocol(_))));
        let dup = ArtifactManifest {
            entries: vec![
                art_entry("transcript.json", 1, "a", "2026-06-30T10:00:00Z"),
                art_entry("transcript.json", 2, "a", "2026-06-30T10:00:00Z"),
            ],
        };
        assert!(matches!(dup.validate(), Err(Error::Protocol(_))));
        let ok = ArtifactManifest {
            entries: vec![
                art_entry("transcript.json", 1, "a", "2026-06-30T10:00:00Z"),
                art_entry("summary.md", 2, "a", "2026-06-30T10:01:00Z"),
            ],
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn artifact_manifest_serde_round_trips() {
        let m = ArtifactManifest {
            entries: vec![art_entry(
                "transcript.json",
                7,
                "endpoint-a",
                "2026-06-30T10:00:00Z",
            )],
        };
        let bytes = serde_json::to_vec(&m).expect("encode");
        let back: ArtifactManifest = serde_json::from_slice(&bytes).expect("decode");
        assert_eq!(m, back);
    }

    /// `import_artifacts` establishes authority from the producer fallback on first
    /// advertise, then faithfully re-advertises the recorded stamp (NOT
    /// `metadata.json`) even with no producer authority — and refuses to advertise
    /// bytes that change with neither a matching record nor producer authority.
    #[tokio::test]
    async fn import_artifacts_producer_fallback_then_faithful_relay() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");
        let id = MeetingId::new();
        let folder = root.join(id.0.to_string());
        std::fs::create_dir_all(&folder).expect("mk folder");
        std::fs::write(folder.join("transcript.json"), b"[]").expect("write transcript");

        // No record yet + a local Processed → producer fallback stamps the entry
        // from the metadata authority AND records it.
        let producer = Some((HostRef("host-a".to_string()), "2026-06-30T10:00:00Z".to_string()));
        let m = store
            .import_artifacts(root, id, producer)
            .await
            .expect("import");
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0].rel_path, "transcript.json");
        assert_eq!(m.entries[0].produced_by, HostRef("host-a".to_string()));

        // A second import with NO producer authority (a relay) still advertises the
        // SAME stamp — read from the recorded authority, not metadata.
        let again = store
            .import_artifacts(root, id, None)
            .await
            .expect("reimport");
        assert_eq!(again.entries.len(), 1);
        assert_eq!(again.entries[0].produced_by, HostRef("host-a".to_string()));
        assert_eq!(again.entries[0].produced_at, "2026-06-30T10:00:00Z");

        // Bytes that change with neither a matching record (hash mismatch) nor a
        // producer authority cannot be stamped → not advertised.
        std::fs::write(folder.join("transcript.json"), b"[{}]").expect("rewrite");
        let changed = store
            .import_artifacts(root, id, None)
            .await
            .expect("reimport changed");
        assert!(
            changed.entries.is_empty(),
            "changed bytes with no provable authority must not be advertised"
        );
    }

    /// A faithful authority record (the state a device is in after RECEIVING bytes)
    /// overrides a present `producer_authority`. This is the W-2 guard: a consumer
    /// whose `metadata.json` advanced to `Processed` via discovery must NOT re-stamp
    /// the bytes it received from its own (stale, not byte-coherent) `Processed` —
    /// the recorded authority that arrived WITH the bytes wins.
    #[tokio::test]
    async fn recorded_authority_overrides_producer_authority() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");
        let id = MeetingId::new();
        let folder = root.join(id.0.to_string());
        std::fs::create_dir_all(&folder).expect("mk folder");
        std::fs::write(folder.join("transcript.json"), b"[]").expect("write transcript");

        // First import records (host-a, T1) for the on-disk bytes (producer
        // fallback) — standing in for the authority a receive would have recorded.
        store
            .import_artifacts(
                root,
                id,
                Some((HostRef("host-a".to_string()), "2026-06-30T10:00:00Z".to_string())),
            )
            .await
            .expect("seed record");

        // A later import offering a DIFFERENT producer_authority (host-b, T2) — e.g.
        // a device whose metadata.json advanced via discovery — must NOT re-stamp:
        // the recorded (host-a, T1) wins because it matches the on-disk bytes.
        let m = store
            .import_artifacts(
                root,
                id,
                Some((HostRef("host-b".to_string()), "2026-06-30T11:00:00Z".to_string())),
            )
            .await
            .expect("reimport");
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0].produced_by, HostRef("host-a".to_string()));
        assert_eq!(m.entries[0].produced_at, "2026-06-30T10:00:00Z");
    }

    /// F3(a): deleting a meeting unpins every blob tag it holds — media AND
    /// derived artifacts — so a subsequent GC sweep can reclaim the bytes. A
    /// second delete on an already-untagged meeting is a no-op, not an error.
    #[tokio::test]
    async fn delete_meeting_blobs_unpins_media_and_artifact_tags() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");
        let id = MeetingId::new();
        let folder = root.join(id.0.to_string());
        std::fs::create_dir_all(&folder).expect("mk folder");
        std::fs::write(folder.join("audio.opus"), b"fake-opus-bytes").expect("write audio");
        std::fs::write(folder.join("transcript.json"), b"[]").expect("write transcript");

        store.import_meeting(root, id).await.expect("import media");
        store
            .import_artifacts(
                root,
                id,
                Some((
                    HostRef("host-a".to_string()),
                    "2026-06-30T10:00:00Z".to_string(),
                )),
            )
            .await
            .expect("import artifact");

        assert!(
            store
                .store
                .tags()
                .get(audio_tag(id))
                .await
                .expect("get audio tag")
                .is_some(),
            "audio tag must exist before deletion"
        );
        assert!(
            store
                .store
                .tags()
                .get(artifact_tag(id, "transcript.json"))
                .await
                .expect("get artifact tag")
                .is_some(),
            "artifact tag must exist before deletion"
        );

        store
            .delete_meeting_blobs(id)
            .await
            .expect("delete meeting blobs");

        assert!(
            store
                .store
                .tags()
                .get(audio_tag(id))
                .await
                .expect("get audio tag after delete")
                .is_none(),
            "audio tag must be unpinned after delete_meeting_blobs"
        );
        assert!(
            store
                .store
                .tags()
                .get(artifact_tag(id, "transcript.json"))
                .await
                .expect("get artifact tag after delete")
                .is_none(),
            "artifact tag must be unpinned after delete_meeting_blobs"
        );

        // Idempotent: an already-untagged meeting deletes cleanly again.
        store
            .delete_meeting_blobs(id)
            .await
            .expect("second delete is a no-op, not an error");
    }

    /// The minor "media export is non-atomic" fix, unit-tested directly against
    /// the shared [`BlobStore::export_atomic`] helper (both [`BlobStore::download`]
    /// and [`BlobStore::download_artifact`] route through it): exporting new
    /// content over an existing target file replaces it completely and leaves no
    /// `.tmp` residue, whether or not a stale file was already present.
    #[tokio::test]
    async fn export_atomic_replaces_target_and_leaves_no_tmp_residue() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");

        // Seed a stale target as if from an earlier (non-atomic) write or crash.
        let target = root.join("export-target.bin");
        std::fs::write(&target, b"stale-content").expect("seed stale target");

        let src = root.join("source.bin");
        std::fs::write(&src, b"fresh-content").expect("write source");
        let hash = store
            .import_path(&src, "test/export-atomic")
            .await
            .expect("import path");

        store
            .export_atomic(hash, &target)
            .await
            .expect("atomic export");

        assert_eq!(
            std::fs::read(&target).expect("read target"),
            b"fresh-content",
            "export_atomic must fully replace the target with the new content"
        );
        let stray_tmp: Vec<_> = std::fs::read_dir(root)
            .expect("read root dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(
            stray_tmp.is_empty(),
            "no .tmp file must remain after export_atomic, found {stray_tmp:?}"
        );
    }

    /// `export_atomic` owns creating its target's parent directory rather than
    /// relying on the underlying `iroh-blobs` export doing it implicitly — the
    /// contract `media_proto`/`artifacts_proto`'s responders now depend on
    /// (they no longer pre-create the meeting folder via `MeetingFolder::ensure`).
    #[tokio::test]
    async fn export_atomic_creates_a_missing_parent_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");

        let src = root.join("source.bin");
        std::fs::write(&src, b"fresh-content").expect("write source");
        let hash = store
            .import_path(&src, "test/export-atomic-mkdir")
            .await
            .expect("import path");

        // The target's parent (a meeting folder) does not exist yet.
        let target = root.join("11111111-1111-1111-1111-111111111111").join("audio.opus");
        assert!(!target.parent().unwrap().exists(), "parent must not pre-exist");

        store
            .export_atomic(hash, &target)
            .await
            .expect("atomic export must create the missing parent directory");

        assert_eq!(
            std::fs::read(&target).expect("read target"),
            b"fresh-content",
            "export_atomic must write the file once the parent exists"
        );
    }

    /// The re-hash memo ([`BlobStore::import_path`]) returns the memoised hash
    /// without re-hashing when a file's `(size, mtime)` is unchanged — proven by
    /// planting a deliberately WRONG hash under a matching memo entry and
    /// observing `import_path` returns THAT wrong value (so the skip genuinely
    /// happens, not just "happens to recompute the same value") — and correctly
    /// invalidates and re-hashes once the file changes.
    #[tokio::test]
    async fn import_path_memo_skips_unchanged_files_and_invalidates_on_change() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");

        let path = root.join("audio.opus");
        std::fs::write(&path, b"original-bytes").expect("write file");
        let real_hash = store
            .import_path(&path, "test/memo")
            .await
            .expect("first import");

        // Plant a deliberately wrong hash under the file's CURRENT (size, mtime).
        // If the memo is genuinely consulted (rather than the file re-hashed for
        // real), import_path must return this wrong value.
        let metadata = std::fs::metadata(&path).expect("stat");
        let planted = Hash::from([0xAAu8; 32]);
        assert_ne!(
            planted, real_hash,
            "the planted hash must differ from the real one for this test to be meaningful"
        );
        store.hash_memo.lock().expect("memo lock").insert(
            path.clone(),
            HashMemoEntry {
                size: metadata.len(),
                mtime: metadata.modified().expect("mtime"),
                hash: planted,
            },
        );
        let memoised = store
            .import_path(&path, "test/memo")
            .await
            .expect("second import");
        assert_eq!(
            memoised, planted,
            "an unchanged (size, mtime) must return the memoised hash without re-hashing"
        );

        // Changing the file's content (and size) invalidates the memo: import_path
        // must re-hash for real, never returning the stale planted value.
        std::fs::write(&path, b"different-length-content-here").expect("rewrite file");
        let rehashed = store
            .import_path(&path, "test/memo")
            .await
            .expect("third import");
        assert_ne!(
            rehashed, planted,
            "a changed file must invalidate the memo and be re-hashed for real"
        );
    }

    /// The producer-gate merge gate for the auto-tag leak: a LOCALLY-IMPORTED
    /// blob (via [`BlobStore::import_meeting`], the dominant case — a
    /// recorded/produced meeting, not a downloaded one) is fully GC-eligible
    /// after [`BlobStore::delete_meeting_blobs`] — no tag, named or temporary,
    /// still roots its hash. Before the [`BlobStore::import_path`] fix, a bare
    /// `add_path(path).await` minted a PERSISTENT `auto-*` tag alongside the
    /// deterministic `meeting/{id}/audio` tag; `delete_meeting_blobs` only
    /// deletes the `meeting/{id}/...` prefix, so the `auto-*` tag survived and
    /// this test's tag-list scan found it — this test fails against that code
    /// and passes once `import_path` leaves no such orphan.
    #[tokio::test]
    async fn local_import_is_fully_reclaimable_after_meeting_delete() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");
        let id = MeetingId::new();
        let folder = root.join(id.0.to_string());
        std::fs::create_dir_all(&folder).expect("mk folder");
        std::fs::write(folder.join("audio.opus"), b"locally-recorded-audio").expect("write audio");

        let manifest = store
            .import_meeting(root, id)
            .await
            .expect("import meeting");
        let hash = manifest.entries[0].hash;

        store
            .delete_meeting_blobs(id)
            .await
            .expect("delete meeting blobs");

        // No NAMED tag — including any stray `auto-*` tag a bare
        // `add_path(...).await` would have left behind — still roots the hash.
        let mut tags = store.store.tags().list().await.expect("list tags");
        while let Some(tag) = tags.next().await {
            let info = tag.expect("tag info");
            assert_ne!(
                info.hash, hash,
                "tag {:?} still roots the deleted meeting's blob",
                info.name
            );
        }

        // No TEMP tag roots the hash either: the transient protection
        // `import_path` holds while it sets the deterministic tag must not
        // outlive that call.
        let mut temp_tags = store
            .store
            .tags()
            .list_temp_tags()
            .await
            .expect("list temp tags");
        while let Some(haf) = temp_tags.next().await {
            assert_ne!(
                haf.hash, hash,
                "a temp tag still roots the deleted meeting's blob"
            );
        }
    }

    /// [`reclaim_stray_auto_tags`] (run from [`BlobStore::open`]) removes a
    /// pre-existing `auto-*` tag — the kind a bare `add_path(...).await` used to
    /// leave behind — while leaving a deterministic `meeting/{id}/...` tag
    /// untouched, proving the prefix scoping can only ever hit the stray kind.
    #[tokio::test]
    async fn reclaim_stray_auto_tags_removes_auto_leaves_deterministic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let store = BlobStore::open(root).await.expect("open store");
        let id = MeetingId::new();

        // Seed a stray auto tag directly (standing in for one left by a pre-fix
        // import) alongside a legitimate deterministic tag.
        let stray_hash = Hash::from([0x11u8; 32]);
        let kept_hash = Hash::from([0x22u8; 32]);
        store
            .store
            .tags()
            .set(
                "auto-2026-01-01T00:00:00.000Z",
                HashAndFormat::raw(stray_hash),
            )
            .await
            .expect("seed stray auto tag");
        store
            .tag(&audio_tag(id), kept_hash)
            .await
            .expect("seed deterministic tag");

        reclaim_stray_auto_tags(&store.store)
            .await
            .expect("reconcile");

        assert!(
            store
                .store
                .tags()
                .get("auto-2026-01-01T00:00:00.000Z")
                .await
                .expect("get stray tag")
                .is_none(),
            "the stray auto-* tag must be removed"
        );
        assert!(
            store
                .store
                .tags()
                .get(audio_tag(id))
                .await
                .expect("get deterministic tag")
                .is_some(),
            "the deterministic meeting tag must survive the reconciliation"
        );
    }
}
