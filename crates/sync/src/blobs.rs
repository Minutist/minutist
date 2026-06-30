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
//! # Retention (GC)
//!
//! [`FsStore::load`] starts with garbage collection DISABLED (its `Options` carry
//! `gc: None` and no GC task is spawned), so an imported or downloaded blob is
//! retained until explicitly deleted. We nonetheless pin every payload with a
//! PERSISTENT, deterministically named tag (`meeting/{id}/audio`,
//! `meeting/{id}/asset/{filename}`) so that retention survives a future decision
//! to enable GC and so a payload can be found and unpinned when its meeting is
//! deleted. Named tags survive restart; temp-tags (in-flight import protection)
//! do not.
//!
//! # Transport
//!
//! [`Self::download`] dials the peer through the SAME [`iroh::Endpoint`] the
//! [`crate::SyncEngine`] already owns (via `store.downloader(&endpoint)`), so the
//! relay path is shared and no second socket is opened. The hash is exchanged
//! out-of-band over the notes channel ([`crate::media_proto`]); there is no blob
//! discovery — the peer's [`EndpointId`] plus the hash is all the downloader needs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset};
use iroh::{Endpoint, EndpointId};
use iroh_blobs::api::downloader::Shuffled;
use iroh_blobs::api::Store;
use iroh_blobs::store::fs::FsStore;
use iroh_blobs::HashAndFormat;
use minutist_common::{HostRef, MeetingId};

use crate::{Error, Result};

/// The BLAKE3 content hash type, re-exported from [`iroh_blobs`] so callers and
/// the media protocol name it through this crate.
pub use iroh_blobs::Hash;

/// The dot-prefixed store directory under the meetings root. A meeting UUID's
/// string form never starts with a dot, so this cannot collide with a `{uuid}`
/// per-meeting folder.
const STORE_DIR: &str = ".blobs";

/// A content-addressed store for a device's meeting media, backed by an
/// [`iroh_blobs`] [`FsStore`] at `{meetings_root}/.blobs`.
///
/// Holds the owned [`FsStore`] (kept alive for the lifetime of the engine; the
/// [`iroh_blobs::BlobsProtocol`] registered on the router borrows from a clone of
/// the inner [`Store`]) and is cheap to clone (the store is internally an RPC
/// client handle).
#[derive(Debug, Clone)]
pub struct BlobStore {
    store: FsStore,
}

impl BlobStore {
    /// Open (or create) the blob store at `{meetings_root}/.blobs`.
    ///
    /// Creates the directory tree if absent. The meetings root is expected to be
    /// absolute (the app passes the XDG meetings root; tests pass a tempdir).
    pub async fn open(meetings_root: &Path) -> Result<Self> {
        let path = meetings_root.join(STORE_DIR);
        std::fs::create_dir_all(&path)?;
        let store = FsStore::load(&path)
            .await
            .map_err(|e| Error::Endpoint(format!("opening blob store at {path:?}: {e}")))?;
        Ok(Self { store })
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

        let audio = folder.join(AUDIO_REL);
        if audio.is_file() {
            let hash = self.import_path(&audio).await?;
            self.tag(&audio_tag(meeting_id), hash).await?;
            entries.push(ManifestEntry {
                rel_path: AUDIO_REL.to_string(),
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
                let hash = self.import_path(&path).await?;
                self.tag(&asset_tag(meeting_id, &name), hash).await?;
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
    /// [`Endpoint`] and blocks until the blob is local; the export then writes a
    /// real copy to the per-meeting path, creating the immediate parent
    /// (`assets/`) if needed. The returned path is the absolute export target.
    ///
    /// The caller ensures the meeting folder via `notes_crdt::MeetingFolder`
    /// before reconciling media; this method writes only the media file itself.
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
        downloader
            .download(hash, Shuffled::new(vec![peer]))
            .await
            .map_err(|e| Error::Protocol(format!("downloading blob {hash} from {peer}: {e}")))?;

        let target = meetings_root.join(meeting_id.0.to_string()).join(rel);
        self.store
            .blobs()
            .export(hash, &target)
            .await
            .map_err(|e| Error::Protocol(format!("exporting blob {hash} to {target:?}: {e}")))?;
        self.tag(&tag_for_rel(meeting_id, rel), hash).await?;
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
    ///   the device faithfully relays the authority that arrived WITH the bytes;
    /// - otherwise the bytes were not received over sync (a relay always records on
    ///   receive — see [`record_artifact_authority`]), so THIS device produced
    ///   them: `producer_authority` (the local `Processed { processed_by, at }`
    ///   read from `metadata.json`) supplies the stamp, recorded so a later relay
    ///   is faithful. This fallback is byte-coherent only because, in the
    ///   single-producer-per-meeting topology this cut targets, a producer's own
    ///   `metadata.json` `Processed` matches the bytes it wrote (the producer-gate
    ///   that would otherwise admit a second producer is unbuilt — DESIGN §7).
    ///
    /// An artifact present on disk for which neither path establishes authority
    /// (bytes exist, no matching record, the meeting is not locally `Processed`) is
    /// NOT advertised — the device will not stamp bytes it cannot date. A meeting
    /// with no artifact files yields an empty manifest (a zero-segment or
    /// not-yet-summarised `Processed` meeting has nothing to send — not an error).
    pub async fn import_artifacts(
        &self,
        meetings_root: &Path,
        meeting_id: MeetingId,
        producer_authority: Option<(HostRef, String)>,
    ) -> Result<ArtifactManifest> {
        let folder = meetings_root.join(meeting_id.0.to_string());
        let mut authority = load_artifact_authority(meetings_root, meeting_id);
        let mut dirty = false;
        let mut entries = Vec::new();

        for rel in ARTIFACT_RELS {
            let path = folder.join(rel);
            if !path.is_file() {
                continue;
            }
            let hash = self.import_path(&path).await?;
            self.tag(&artifact_tag(meeting_id, rel), hash).await?;

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

        if dirty {
            save_artifact_authority(meetings_root, meeting_id, &authority)?;
        }
        Ok(ArtifactManifest { entries })
    }

    /// Download an artifact `hash` from `peer` and export it to
    /// `{meetings_root}/{uuid}/{rel}` (where `rel` is an [`is_artifact_rel`] path),
    /// pinning it with the per-meeting artifact tag.
    ///
    /// `rel` is re-validated against the artifact allow-list (defence in depth — the
    /// manifest was validated whole). The export goes to a sibling `{rel}.tmp` then
    /// atomically renames over the target, so a concurrent reader of
    /// `transcript.json` (read far more often than `audio.opus`) never observes a
    /// partial file. The caller ensures the meeting folder via
    /// `notes_crdt::MeetingFolder::ensure` and records the received authority (it
    /// holds the peer entry's `produced_by`/`produced_at`); this writes only the
    /// artifact file.
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
        downloader
            .download(hash, Shuffled::new(vec![peer]))
            .await
            .map_err(|e| {
                Error::Protocol(format!("downloading artifact {hash} from {peer}: {e}"))
            })?;

        let target = meetings_root.join(meeting_id.0.to_string()).join(rel);
        let tmp = target.with_extension(format!(
            "{}.tmp",
            target.extension().and_then(|e| e.to_str()).unwrap_or("")
        ));
        self.store
            .blobs()
            .export(hash, &tmp)
            .await
            .map_err(|e| Error::Protocol(format!("exporting artifact {hash} to {tmp:?}: {e}")))?;
        std::fs::rename(&tmp, &target).map_err(|e| {
            Error::Protocol(format!("atomically renaming {tmp:?} -> {target:?}: {e}"))
        })?;
        self.tag(&artifact_tag(meeting_id, rel), hash).await?;
        Ok(target)
    }

    /// Import a single file into the store and return its [`Hash`].
    async fn import_path(&self, path: &Path) -> Result<Hash> {
        let tag = self
            .store
            .blobs()
            .add_path(path)
            .await
            .map_err(|e| Error::Protocol(format!("importing {path:?} into blob store: {e}")))?;
        Ok(tag.hash)
    }

    /// Set a persistent named tag pinning `hash` (as a raw blob) against GC.
    async fn tag(&self, name: &str, hash: Hash) -> Result<()> {
        self.store
            .tags()
            .set(name.as_bytes(), HashAndFormat::raw(hash))
            .await
            .map_err(|e| Error::Protocol(format!("tagging blob {hash} as {name}: {e}")))
    }
}

/// Relative path of a meeting's primary audio file within its folder.
pub(crate) const AUDIO_REL: &str = "audio.opus";
/// Name of the note-assets subdirectory within a meeting folder.
pub(crate) const ASSETS_DIR: &str = "assets";

/// The persistent tag name pinning a meeting's audio blob.
fn audio_tag(meeting_id: MeetingId) -> String {
    format!("meeting/{}/audio", meeting_id.0)
}

/// The persistent tag name pinning one of a meeting's asset blobs.
fn asset_tag(meeting_id: MeetingId, filename: &str) -> String {
    format!("meeting/{}/asset/{filename}", meeting_id.0)
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

/// A relative path is safe iff it is `audio.opus` or `assets/<single-component>`
/// with no separator-escaping or `..` components. Mirrors the persistence
/// asset-filename guard so a hostile manifest cannot direct an export outside the
/// meeting folder.
pub(crate) fn is_safe_rel(rel: &str) -> bool {
    if rel == AUDIO_REL {
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
/// `rel` must already be an [`is_artifact_rel`] path. A namespace distinct from
/// the media tags (`meeting/{id}/audio|asset/...`) so unpinning media never
/// touches an artifact and vice versa.
fn artifact_tag(meeting_id: MeetingId, rel: &str) -> String {
    format!("meeting/{}/artifact/{rel}", meeting_id.0)
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
    /// `produced_by` HostRef. Clock-independent on the tiebreak and matching the
    /// lifecycle merge's two-`Processed` rule (`notes_crdt::merge_processing`), so
    /// the lifecycle edge and the on-disk bytes name one authoritative host. NEVER
    /// `>=` (a same-`produced_at`, different-content pair must not pull on both
    /// sides — DESIGN §2 C2) and NEVER a raw-string compare (mixed fractional
    /// precision / offset sorts wrong). A malformed timestamp on EITHER side is
    /// treated as not-superseding, so a parse glitch can only keep local bytes,
    /// never clobber them. Callers short-circuit equal hashes (identical bytes
    /// never need pulling) before consulting this.
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
/// rather than one re-derived from `metadata.json` (DESIGN §2 C1).
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
/// unreadable. A corrupt record is logged and treated as empty: the worst case is
/// re-establishing authority from the producer fallback, never a clobber.
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

/// Record the authority for an artifact's received bytes — called after
/// [`BlobStore::download_artifact`] with the peer entry's `produced_by` /
/// `produced_at`, so a future [`BlobStore::import_artifacts`] on this device
/// faithfully re-advertises the authority that arrived with the bytes rather than
/// re-deriving it from `metadata.json`.
pub(crate) fn record_artifact_authority(
    meetings_root: &Path,
    meeting_id: MeetingId,
    rel: &str,
    hash: Hash,
    produced_by: HostRef,
    produced_at: String,
) -> Result<()> {
    let mut authority = load_artifact_authority(meetings_root, meeting_id);
    authority.record(rel, hash, produced_by, produced_at);
    save_artifact_authority(meetings_root, meeting_id, &authority)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_rel_accepts_audio_and_assets() {
        assert!(is_safe_rel("audio.opus"));
        assert!(is_safe_rel("assets/abc123.png"));
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
}
