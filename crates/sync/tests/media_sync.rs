//! Round-trip proof for the WS4-B S4 media-blob sync protocol.
//!
//! Two `SyncEngine`s, each backed by its own temp meetings root sharing the SAME
//! `meeting_id`, reconcile a meeting's MEDIA — `audio.opus` plus a note asset —
//! over a real (relay-less, loopback) iroh connection. The blobs travel over the
//! blobs ALPN, named by hashes exchanged over the sync ALPN.
//!
//! The exchange runs over `SyncEngine::start_direct` with relays disabled, the
//! same loopback addressing the notes-sync test uses.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use iroh::EndpointAddr;
use minutist_common::MeetingId;
use notes_crdt::MeetingFolder;
use persistence::save_note_asset;
use sync::identity::DeviceIdentity;
use sync::SyncEngine;

/// Loopback `EndpointAddr` for `engine` (its id + each bound port against
/// `127.0.0.1`).
fn direct_addr(engine: &SyncEngine) -> EndpointAddr {
    let mut addr = EndpointAddr::new(engine.endpoint_id());
    for sock in engine.bound_sockets() {
        let loopback = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), sock.port());
        addr = addr.with_ip_addr(loopback);
    }
    addr
}

/// Start two relay-less engines bound to `root_a` / `root_b` and cross-inject
/// their loopback addresses (the out-of-band MUTUAL pairing the account service
/// does).
async fn paired_engines(root_a: &Path, root_b: &Path) -> (SyncEngine, SyncEngine) {
    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, root_a.to_path_buf())
        .await
        .expect("engine a");
    let b = SyncEngine::start_direct(id_b, root_b.to_path_buf())
        .await
        .expect("engine b");
    a.add_peer(direct_addr(&b));
    b.add_peer(direct_addr(&a));
    (a, b)
}

/// Write `audio.opus` into `{root}/{id}/`.
fn write_audio(root: &Path, id: MeetingId, bytes: &[u8]) {
    let folder = MeetingFolder::create(root, id).expect("create meeting folder");
    std::fs::write(folder.audio_path(), bytes).expect("write audio.opus");
}

#[tokio::test]
async fn media_round_trips_audio_and_an_asset() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    // A has the media; B has only the (empty) meeting folder.
    let meeting = MeetingId::new();
    let audio = b"OggS-fake-opus-audio-payload".repeat(64);
    write_audio(root_a, meeting, &audio);
    let asset_bytes = b"\x89PNG\r\n\x1a\nfake-pasted-image".repeat(8);
    let asset_name = save_note_asset(root_a, meeting, &asset_bytes, "png").expect("seed asset");
    MeetingFolder::create(root_b, meeting).expect("create folder b");

    let (a, b) = paired_engines(root_a, root_b).await;

    // A initiates a media reconciliation with B.
    a.sync_media(direct_addr(&b), meeting)
        .await
        .expect("reconcile media a -> b");

    // B received byte-identical audio.
    let b_audio = std::fs::read(root_b.join(meeting.0.to_string()).join("audio.opus"))
        .expect("B must have audio.opus after sync");
    assert_eq!(b_audio, audio, "B's audio must be byte-identical to A's");

    // B received the asset, byte-identical, under the same portable filename.
    let b_asset = persistence::read_note_asset(root_b, meeting, &asset_name)
        .expect("B must have the asset after sync");
    assert_eq!(
        b_asset, asset_bytes,
        "B's asset must be byte-identical to A's"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn identical_bytes_dedupe_to_one_hash_on_both_sides() {
    // The same audio bytes on both devices must import to the SAME content hash
    // (BLAKE3 dedup): the manifests agree, so no transfer is needed.
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    let audio = b"identical-audio-bytes".repeat(32);
    write_audio(root_a, meeting, &audio);
    write_audio(root_b, meeting, &audio);

    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, root_a.to_path_buf())
        .await
        .expect("engine a");
    let b = SyncEngine::start_direct(id_b, root_b.to_path_buf())
        .await
        .expect("engine b");

    let manifest_a = a.import_media(meeting).await.expect("import a");
    let manifest_b = b.import_media(meeting).await.expect("import b");

    let audio_hash_a = manifest_a
        .entries
        .iter()
        .find(|e| e.rel_path == "audio.opus")
        .map(|e| e.hash);
    let audio_hash_b = manifest_b
        .entries
        .iter()
        .find(|e| e.rel_path == "audio.opus")
        .map(|e| e.hash);
    assert_eq!(
        audio_hash_a, audio_hash_b,
        "identical bytes must import to the same content hash on both devices"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn media_sync_is_a_noop_when_both_sides_already_hold_the_blobs() {
    // Both devices already have byte-identical audio + asset for the same
    // meeting. A full `sync_media` must converge cleanly (a no-op): every
    // manifest entry is already held by content hash, so nothing is pulled and
    // both files stay byte-identical. Proves the content-addressed skip path and
    // that a re-run after convergence does not corrupt or re-fetch.
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    let audio = b"already-converged-audio-bytes".repeat(48);
    write_audio(root_a, meeting, &audio);
    write_audio(root_b, meeting, &audio);
    // Same asset bytes on both sides → same content-addressed filename.
    let asset_bytes = b"already-converged-asset-bytes".repeat(8);
    let name_a = save_note_asset(root_a, meeting, &asset_bytes, "png").expect("seed asset a");
    let name_b = save_note_asset(root_b, meeting, &asset_bytes, "png").expect("seed asset b");
    assert_eq!(
        name_a, name_b,
        "identical asset bytes must dedupe to one name"
    );

    let (a, b) = paired_engines(root_a, root_b).await;

    a.sync_media(direct_addr(&b), meeting)
        .await
        .expect("no-op media reconcile a -> b");

    // Both files are unchanged and still byte-identical on both sides.
    let a_audio = std::fs::read(root_a.join(meeting.0.to_string()).join("audio.opus"))
        .expect("A keeps its audio");
    let b_audio = std::fs::read(root_b.join(meeting.0.to_string()).join("audio.opus"))
        .expect("B keeps its audio");
    assert_eq!(a_audio, audio, "A's audio is unchanged by the no-op sync");
    assert_eq!(b_audio, audio, "B's audio is unchanged by the no-op sync");
    assert_eq!(
        persistence::read_note_asset(root_a, meeting, &name_a).expect("A keeps its asset"),
        asset_bytes
    );
    assert_eq!(
        persistence::read_note_asset(root_b, meeting, &name_b).expect("B keeps its asset"),
        asset_bytes
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn an_unpaired_peer_is_rejected_on_the_blobs_alpn() {
    // B does NOT pair A (only A pairs B). B holds a blob; A learns its hash and
    // tries to download it directly over the blobs ALPN. B's blobs accept side
    // must reject the unpaired peer, so the download fails — proving the blobs
    // ALPN carries the same MUTUAL-pairing guard as the sync ALPN (the H1 fix is
    // not reintroduced on the new ALPN).
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    let audio = b"secret-audio-only-on-B".repeat(16);
    write_audio(root_b, meeting, &audio);
    MeetingFolder::create(root_a, meeting).expect("create folder a");

    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, root_a.to_path_buf())
        .await
        .expect("engine a");
    let b = SyncEngine::start_direct(id_b, root_b.to_path_buf())
        .await
        .expect("engine b");

    // B imports its audio so the blob exists in B's store; A learns the hash.
    let manifest_b = b.import_media(meeting).await.expect("import b");
    let audio_hash = manifest_b
        .entries
        .iter()
        .find(|e| e.rel_path == "audio.opus")
        .map(|e| e.hash)
        .expect("B's manifest must carry audio");

    // ONE-directional pairing: A can resolve B's address, but B has NOT paired A.
    a.add_peer(direct_addr(&b));

    // A attempts to pull B's blob over the blobs ALPN. B rejects the unpaired
    // remote, so the download must fail and A must not end up with the audio.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        a.download_blob(b.endpoint_id(), meeting, "audio.opus", audio_hash),
    )
    .await;
    assert!(
        matches!(result, Ok(Err(_))),
        "downloading from a peer that has not paired this device must fail, got {result:?}"
    );
    assert!(
        !root_a
            .join(meeting.0.to_string())
            .join("audio.opus")
            .exists(),
        "an unpaired peer must not be able to pull B's audio over the blobs ALPN"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}
