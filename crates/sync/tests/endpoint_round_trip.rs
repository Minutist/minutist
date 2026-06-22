//! End-to-end check of the device identity + iroh endpoint + SYNC_ALPN, with no
//! deployed relay.
//!
//! Two `SyncEngine`s bind on localhost with relays disabled, inject each other's
//! direct `EndpointAddr` (id + loopback socket), then the dial side reconciles a
//! meeting's notes with the accept side over the sync ALPN. A successful exchange
//! proves the persisted ed25519 identity, the bound endpoint, and ALPN
//! negotiation work together — without the homelab relay or its access token.
//! The notes-convergence assertions live in `notes_sync.rs`; this test only
//! exercises the transport handshake against an empty meeting (an empty exchange
//! still completes the four-message protocol).

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use iroh::EndpointAddr;
use minutist_common::MeetingId;
use sync::identity::DeviceIdentity;
use sync::SyncEngine;

/// Build the peer's direct `EndpointAddr` from its id plus a loopback rewrite of
/// each bound socket. The bound sockets report the unspecified address
/// (`0.0.0.0`); the test reaches the peer over loopback, so the port is reused
/// against `127.0.0.1`.
fn direct_addr(engine: &SyncEngine) -> EndpointAddr {
    let mut addr = EndpointAddr::new(engine.endpoint_id());
    for sock in engine.bound_sockets() {
        let loopback = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), sock.port());
        addr = addr.with_ip_addr(loopback);
    }
    addr
}

#[tokio::test]
async fn two_engines_reconcile_over_sync_alpn() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");

    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");

    // Each engine's meetings root is its own temp dir; the empty meeting has no
    // notes on either side, so the exchange completes with empty diffs.
    let engine_a = SyncEngine::start_direct(id_a, dir_a.path().to_path_buf())
        .await
        .expect("engine a");
    let engine_b = SyncEngine::start_direct(id_b, dir_b.path().to_path_buf())
        .await
        .expect("engine b");

    // Inject each other's direct address (the out-of-band step the account
    // service performs in production).
    engine_a.add_peer(direct_addr(&engine_b));
    engine_b.add_peer(direct_addr(&engine_a));

    // Identity check: a bare dial resolves to B's endpoint id over the ALPN.
    let conn = engine_a
        .connect(direct_addr(&engine_b))
        .await
        .expect("dial b on sync alpn");
    assert_eq!(
        conn.remote_id(),
        engine_b.endpoint_id(),
        "the dialled peer identity must match B's endpoint id"
    );
    conn.close(0u32.into(), b"id-check");

    // A reconciles an (empty) meeting with B end-to-end over the protocol.
    let meeting = MeetingId::new();
    engine_a
        .sync_notes(direct_addr(&engine_b), meeting)
        .await
        .expect("notes reconciliation over sync alpn");

    engine_a.shutdown().await.expect("shutdown a");
    engine_b.shutdown().await.expect("shutdown b");
}
