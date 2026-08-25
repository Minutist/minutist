//! The account content key and the frame cipher derived from it
//! (`planning/DESIGN_sync-encryption.md` §3, §4).
//!
//! One 32-byte symmetric key per account, held by every enrolled device and sent
//! to neither the relay nor the account service. Everything the sync protocols
//! put on a frame is sealed under a subkey of it, so a peer that passes the
//! ed25519 membership check but was never confirmed by the user reads nothing.
//!
//! The key is persisted at `{app-data}/sync_content_key`, `0600`, beside the
//! ed25519 device key ([`crate::identity`]). It is not zeroized in memory: the
//! meeting files it protects on the wire are plaintext on disk by product
//! decision, so a threat that can read this process's memory can read the
//! content directly and the scrub buys nothing.
//!
//! # Two devices that both minted a key
//!
//! A device with no key mints one, because it cannot ask the account service
//! whether the account already has one (the service holds no key, by design). So
//! two devices can independently mint and then fail to talk: each sees the
//! other's frames fail to open.
//!
//! That state is benign and self-healing, which is a direct consequence of
//! plaintext-at-rest. Nothing is lost, because each device holds its own
//! meetings in plain files; enrolment ([`replace`](ContentKey::replace)) then
//! overwrites one side's key with the other's and the next sweep re-syncs. The
//! overwrite is safe only because it is reachable exclusively from the
//! user-confirmed enrolment exchange, never from an unconfirmed peer.

use std::path::{Path, PathBuf};

use chacha20poly1305::aead::{AeadInOut, KeyInit};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use sha2::Sha256;

use crate::{key_file, Error, Result};

/// The file under `{app-data}` holding the raw 32 key bytes. `0600` on Unix.
const KEY_FILE: &str = "sync_content_key";

/// HKDF label for the frame subkey. The key bytes are never used as an AEAD key
/// directly, so a second use of the account key gets its own subkey rather than
/// sharing this one. Changing this string is a wire break.
const FRAME_KEY_LABEL: &[u8] = b"minutist/sync/frame/v1";

/// XChaCha20-Poly1305 nonce length. `pub(crate)` because [`crate::frame`]
/// reads the nonce off the wire ahead of the body.
pub(crate) const NONCE_LEN: usize = 24;

/// Poly1305 authentication tag length.
pub(crate) const TAG_LEN: usize = 16;

/// Bytes a sealed frame adds over its plaintext: the prepended nonce plus the
/// trailing tag. [`crate::frame`] accounts for this against `MAX_FRAME`.
pub(crate) const SEAL_OVERHEAD: usize = NONCE_LEN + TAG_LEN;

/// The account content key: the shared secret every enrolled device holds.
///
/// `Debug` is hand-written to redact the bytes, matching
/// [`crate::SyncConfig`]'s treatment of the relay token: the key must never
/// reach a log line through a derived `Debug`.
#[derive(Clone)]
pub struct ContentKey {
    bytes: [u8; 32],
}

impl ContentKey {
    /// Path to the persisted key under `app_data_dir`.
    pub fn path(app_data_dir: &Path) -> PathBuf {
        app_data_dir.join(KEY_FILE)
    }

    /// Load the key, or `None` when this device holds none and so is not yet
    /// enrolled. A short or long file is an error rather than a silent remint,
    /// because reminting over a truncated key would strand the device from every
    /// peer while looking like a fresh install.
    pub fn load(app_data_dir: &Path) -> Result<Option<Self>> {
        Ok(key_file::read(&Self::path(app_data_dir), "content key")?.map(|bytes| Self { bytes }))
    }

    /// Mint and persist a fresh key, or return the existing one.
    ///
    /// The CALLER decides this device is the first on its account; see
    /// [`SyncEngine::note_account_peers`](crate::SyncEngine::note_account_peers),
    /// which owns that policy and is the only production caller. Minting on a
    /// device that is not first is not a leak, only a wasted key that no peer
    /// holds, but it strands the device until enrolment replaces it, so the
    /// decision belongs at the one place that has the directory's answer rather
    /// than being re-expressed here.
    pub fn load_or_mint(app_data_dir: &Path) -> Result<Self> {
        if let Some(key) = Self::load(app_data_dir)? {
            return Ok(key);
        }
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes)
            .map_err(|e| Error::Identity(format!("generating a content key: {e}")))?;
        let key = Self { bytes };
        key.persist(app_data_dir)?;
        tracing::info!(
            target: "sync",
            "minted a new account content key (first device on this account)"
        );
        Ok(key)
    }

    /// Adopt a key received from an enrolled peer, overwriting any local one.
    ///
    /// Only reachable from the user-confirmed enrolment exchange. An
    /// unconfirmed peer must never reach this: it is the one path that lets a
    /// remote party choose this device's key.
    pub fn replace(app_data_dir: &Path, bytes: [u8; 32]) -> Result<Self> {
        let key = Self { bytes };
        key.persist(app_data_dir)?;
        tracing::info!(
            target: "sync",
            "adopted the account content key from a confirmed peer"
        );
        Ok(key)
    }

    /// An in-memory key with no file behind it, for a test that needs two
    /// engines to hold deliberately DIFFERENT keys. Gated behind `test-support`
    /// so no production path can conjure a key from constant bytes.
    #[cfg(feature = "test-support")]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    /// The key a co-enrolled pair shares in tests. The value is arbitrary; only
    /// agreement between the engines matters, which is why every test that just
    /// wants a working pair should use this rather than its own constant.
    #[cfg(feature = "test-support")]
    pub fn for_tests() -> Self {
        Self { bytes: [0x5a; 32] }
    }

    /// Persist [`Self::for_tests`] at `app_data_dir`, for a test that has to
    /// co-enrol a separate process it cannot hand a [`ContentKey`] to (the hub
    /// daemon spawned by `headless`' end-to-end tests). Writing the key here
    /// keeps the file's name and format owned by this module.
    #[cfg(feature = "test-support")]
    pub fn seed_for_tests(app_data_dir: &Path) -> Result<Self> {
        let key = Self::for_tests();
        key.persist(app_data_dir)?;
        Ok(key)
    }

    /// The raw key bytes, for the enrolment transfer only
    /// ([`crate::enrolment_proto`]), which is the one place the key legitimately
    /// leaves this device. Every other consumer wants [`Self::frame_cipher`].
    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// The AEAD that seals and opens frames, keyed by HKDF-SHA256 over this key
    /// under [`FRAME_KEY_LABEL`].
    pub(crate) fn frame_cipher(&self) -> FrameCipher {
        let hkdf = Hkdf::<Sha256>::new(None, &self.bytes);
        let mut subkey = [0u8; 32];
        hkdf.expand(FRAME_KEY_LABEL, &mut subkey)
            .expect("32 bytes is a valid HKDF-SHA256 output length");
        FrameCipher {
            aead: XChaCha20Poly1305::new(&subkey.into()),
        }
    }

    /// Write the key `0600` via the shared [`crate::key_file`] writer.
    fn persist(&self, app_data_dir: &Path) -> Result<()> {
        let path = Self::path(app_data_dir);
        key_file::write(&path, &self.bytes)
            .map_err(|e| Error::Identity(format!("writing content key to {path:?}: {e}")))
    }
}

impl std::fmt::Debug for ContentKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentKey")
            .field("bytes", &"<redacted>")
            .finish()
    }
}

/// Seals and opens one frame body under the frame subkey.
///
/// The nonce is 24 random bytes per frame, prepended to the ciphertext. A
/// 192-bit random nonce carries no counter state, which matters because frames
/// are written from several tasks across several connections: a counter would be
/// one shared-state bug away from reuse, and nonce reuse under a stream cipher
/// loses confidentiality outright.
/// `Debug` is derived: `chacha20poly1305`'s own impl renders
/// `ChaChaPoly1305 { .. }` with no key material, so a struct holding a
/// `FrameCipher` can derive `Debug` without leaking the key.
#[derive(Clone, Debug)]
pub(crate) struct FrameCipher {
    aead: XChaCha20Poly1305,
}

impl FrameCipher {
    /// Seal `plaintext`, returning `nonce || ciphertext || tag`.
    ///
    /// `aad` is the stream's [`crate::notes_proto::StreamKind`] tag, bound as
    /// additional data so a frame lifted from one protocol cannot be replayed
    /// into another.
    ///
    /// Builds the wire buffer once and encrypts in place. The convenience
    /// `Aead::encrypt` would allocate its own `Vec`, copy the plaintext into it,
    /// and leave a second copy into the output buffer here: two allocations and
    /// two passes over what can be an 8 MiB frame ([`crate::frame::MAX_FRAME`]),
    /// where one of each suffices.
    pub(crate) fn seal(&self, aad: u8, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce = [0u8; NONCE_LEN];
        getrandom::fill(&mut nonce)
            .map_err(|e| Error::Protocol(format!("generating a frame nonce: {e}")))?;

        let mut out = Vec::with_capacity(NONCE_LEN + plaintext.len() + TAG_LEN);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(plaintext);
        let tag = self
            .aead
            .encrypt_inout_detached(&XNonce::from(nonce), &[aad], (&mut out[NONCE_LEN..]).into())
            .map_err(|_| Error::Protocol("sealing a frame failed".to_string()))?;
        out.extend_from_slice(&tag);
        Ok(out)
    }

    /// Open a body sealed by [`Self::seal`] under the same `aad`, decrypting
    /// `ciphertext` in place and returning it as the plaintext.
    ///
    /// The nonce arrives separately because the caller reads it off the wire
    /// ahead of the body ([`crate::frame`]), which keeps this a single buffer
    /// with no copy: `Aead::decrypt` would clone the whole ciphertext first.
    ///
    /// A failure is authentication failure, not corruption: the peer holds a
    /// different content key, or the bytes were tampered with, or the frame came
    /// from a different protocol. All three are [`Error::Unauthenticated`], kept
    /// distinct from [`Error::Protocol`] so a key mismatch is diagnosable rather
    /// than looking like a malformed peer.
    pub(crate) fn open(
        &self,
        aad: u8,
        nonce: &[u8; NONCE_LEN],
        mut ciphertext: Vec<u8>,
    ) -> Result<Vec<u8>> {
        self.aead
            .decrypt_in_place(&XNonce::from(*nonce), &[aad], &mut ciphertext)
            .map_err(|_| {
                Error::Unauthenticated(
                    "opening a frame failed: wrong content key, wrong protocol, or tampered bytes"
                        .to_string(),
                )
            })?;
        Ok(ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAG: u8 = 1;

    /// Split a `seal` output the way `frame::read_sealed` splits it off the wire:
    /// the nonce is read ahead of the body so `open` can work in place.
    fn split(sealed: &[u8]) -> ([u8; NONCE_LEN], Vec<u8>) {
        let (nonce, body) = sealed.split_at(NONCE_LEN);
        (nonce.try_into().expect("nonce width"), body.to_vec())
    }

    /// Seal, split, and open again under `open_aad`.
    fn round_trip(cipher: &FrameCipher, seal_aad: u8, open_aad: u8, msg: &[u8]) -> Result<Vec<u8>> {
        let sealed = cipher.seal(seal_aad, msg).expect("seal");
        let (nonce, body) = split(&sealed);
        cipher.open(open_aad, &nonce, body)
    }

    #[test]
    fn mints_once_and_reloads_the_same_key() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let first = ContentKey::load_or_mint(dir.path()).expect("mint");
        let reloaded = ContentKey::load_or_mint(dir.path()).expect("reload");
        assert_eq!(
            first.bytes, reloaded.bytes,
            "a minted key must persist, not be re-minted on the next load"
        );
    }

    #[test]
    fn absent_key_loads_as_not_enrolled() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        assert!(
            ContentKey::load(dir.path()).expect("load").is_none(),
            "no key file means not enrolled, which is not an error"
        );
    }

    #[test]
    fn key_file_is_owner_only_and_32_bytes() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        ContentKey::load_or_mint(dir.path()).expect("mint");
        let meta = std::fs::metadata(ContentKey::path(dir.path())).expect("stat");
        assert_eq!(meta.len(), 32);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                meta.permissions().mode() & 0o777,
                0o600,
                "the content key must be owner-only"
            );
        }
    }

    #[test]
    fn rejects_a_truncated_key_file() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(ContentKey::path(dir.path()), b"short").expect("write");
        // Not a silent remint: that would strand the device from every peer
        // while looking like a fresh install.
        assert!(ContentKey::load(dir.path()).is_err());
        assert!(ContentKey::load_or_mint(dir.path()).is_err());
    }

    #[test]
    fn replace_overwrites_at_0600() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let minted = ContentKey::load_or_mint(dir.path()).expect("mint");
        let adopted = ContentKey::replace(dir.path(), [7u8; 32]).expect("replace");
        assert_ne!(minted.bytes, adopted.bytes);
        assert_eq!(
            ContentKey::load(dir.path())
                .expect("load")
                .expect("present")
                .bytes,
            [7u8; 32],
            "the adopted key must be the one on disk afterwards"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = std::fs::metadata(ContentKey::path(dir.path())).expect("stat");
            assert_eq!(meta.permissions().mode() & 0o777, 0o600);
        }
    }

    #[test]
    fn seal_then_open_round_trips() {
        let key = ContentKey { bytes: [3u8; 32] };
        let cipher = key.frame_cipher();
        let plaintext = b"a yjs state vector, say";
        let sealed = cipher.seal(TAG, plaintext).expect("seal");
        assert_eq!(
            sealed.len(),
            plaintext.len() + SEAL_OVERHEAD,
            "the overhead constant must match what seal actually adds"
        );
        let (nonce, body) = split(&sealed);
        assert_eq!(cipher.open(TAG, &nonce, body).expect("open"), plaintext);
    }

    #[test]
    fn the_same_plaintext_seals_differently_each_time() {
        let cipher = ContentKey { bytes: [3u8; 32] }.frame_cipher();
        let a = cipher.seal(TAG, b"same").expect("seal");
        let b = cipher.seal(TAG, b"same").expect("seal");
        assert_ne!(a, b, "a fresh nonce per frame must make the bytes differ");
    }

    #[test]
    fn a_different_key_cannot_open_it() {
        let sealed = ContentKey { bytes: [3u8; 32] }
            .frame_cipher()
            .seal(TAG, b"secret")
            .expect("seal");
        let (nonce, body) = split(&sealed);
        let other = ContentKey { bytes: [4u8; 32] }.frame_cipher();
        assert!(matches!(
            other.open(TAG, &nonce, body),
            Err(Error::Unauthenticated(_))
        ));
    }

    #[test]
    fn a_frame_cannot_be_replayed_into_another_protocol() {
        let cipher = ContentKey { bytes: [3u8; 32] }.frame_cipher();
        // Same key, different StreamKind tag as AAD: the frame must not open.
        assert!(matches!(
            round_trip(&cipher, TAG, TAG + 1, b"a media manifest"),
            Err(Error::Unauthenticated(_))
        ));
    }

    #[test]
    fn a_tampered_frame_does_not_open() {
        let cipher = ContentKey { bytes: [3u8; 32] }.frame_cipher();
        let mut sealed = cipher.seal(TAG, b"a yjs diff").expect("seal");
        let last = sealed.len() - 1;
        sealed[last] ^= 0x01;
        let (nonce, body) = split(&sealed);
        assert!(matches!(
            cipher.open(TAG, &nonce, body),
            Err(Error::Unauthenticated(_))
        ));
    }

    #[test]
    fn debug_redacts_the_key() {
        let rendered = format!("{:?}", ContentKey { bytes: [9u8; 32] });
        assert!(rendered.contains("<redacted>"));
        assert!(
            !rendered.contains('9'),
            "no key byte may reach a Debug rendering: {rendered}"
        );
    }
}
