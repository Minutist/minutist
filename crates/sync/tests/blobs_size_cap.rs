//! Per-blob size-cap rejection proof (the "no per-blob byte-size cap" minor).
//!
//! Two relay-less `SyncEngine`s pair over loopback, mirroring `media_sync.rs`'s
//! helper. A imports a modest media payload into its blob store; B pulls it
//! through the test-support `download_blob_capped` seam with a cap far below the
//! payload's real size and must be rejected, proving the cap is enforced against
//! the transfer's own running byte count rather than a self-reported size.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use iroh::EndpointAddr;
use minutist_common::MeetingId;
use notes_crdt::MeetingFolder;
use sync::identity::DeviceIdentity;
use sync::SyncEngine;

/// Loopback `EndpointAddr` for `engine` (its id + each bound port against
/// `127.0.0.1`) — same helper as `media_sync.rs` / `endpoint_round_trip.rs`.
fn direct_addr(engine: &SyncEngine) -> EndpointAddr {
    let mut addr = EndpointAddr::new(engine.endpoint_id());
    for sock in engine.bound_sockets() {
        let loopback = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), sock.port());
        addr = addr.with_ip_addr(loopback);
    }
    addr
}

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

#[tokio::test]
async fn oversized_blob_is_rejected_under_a_tiny_cap() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    let folder = MeetingFolder::create(root_a, meeting).expect("create meeting folder");
    // A modest payload (a few KB) — the cap under test is tinier still, so no
    // multi-gigabyte fixture is needed to prove the real-time enforcement works.
    let audio = b"OggS-fake-opus-audio-payload".repeat(256);
    std::fs::write(folder.audio_path(), &audio).expect("write audio.opus");

    let (a, b) = paired_engines(root_a, root_b).await;

    // A imports its media so the blob is staged in its store and B can fetch it
    // by hash over the blobs ALPN.
    let manifest = a.import_media(meeting).await.expect("import media on A");
    let hash = manifest.entries[0].hash;

    let cap = 64u64;
    assert!(
        audio.len() as u64 > cap,
        "test payload must exceed the tiny cap for this test to be meaningful"
    );
    let err = b
        .download_blob_capped(a.endpoint_id(), hash, cap)
        .await
        .expect_err("a blob exceeding the cap must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("exceeded"),
        "rejection error should mention the size cap, got: {message}"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

#[tokio::test]
async fn blob_under_the_cap_downloads_normally() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    let meeting = MeetingId::new();
    let folder = MeetingFolder::create(root_a, meeting).expect("create meeting folder");
    let audio = b"OggS-fake-opus-audio-payload".repeat(4);
    std::fs::write(folder.audio_path(), &audio).expect("write audio.opus");

    let (a, b) = paired_engines(root_a, root_b).await;
    let manifest = a.import_media(meeting).await.expect("import media on A");
    let hash = manifest.entries[0].hash;

    // A generous cap (well over the payload) must not interfere with a normal
    // transfer — the test-support seam still routes through the same
    // `download_capped` logic the production cap uses.
    b.download_blob_capped(a.endpoint_id(), hash, 1024 * 1024)
        .await
        .expect("a blob under the cap must download successfully");

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}
