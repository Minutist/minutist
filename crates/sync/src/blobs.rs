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

use std::path::{Path, PathBuf};

use iroh::{Endpoint, EndpointId};
use iroh_blobs::api::downloader::Shuffled;
use iroh_blobs::api::Store;
use iroh_blobs::store::fs::FsStore;
use iroh_blobs::HashAndFormat;
use minutist_common::MeetingId;

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
    /// The caller ensures the meeting folder via `persistence::MeetingFolder`
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
}
