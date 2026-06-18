//! End-to-end check of the device identity + iroh endpoint + SYNC_ALPN, with no
//! deployed relay.
//!
//! Two `SyncEngine`s bind on localhost with relays disabled, inject each other's
//! direct `EndpointAddr` (id + loopback socket), then the dial side opens a bi
//! stream on the sync ALPN and the accept side (the S2 hook) echoes the bytes
//! back. A successful round-trip proves the persisted ed25519 identity, the bound
//! endpoint, and ALPN negotiation work together — without the homelab relay or its
//! access token.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use iroh::EndpointAddr;
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
async fn two_engines_round_trip_bytes_over_sync_alpn() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");

    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");

    let engine_a = SyncEngine::start_direct(id_a).await.expect("engine a");
    let engine_b = SyncEngine::start_direct(id_b).await.expect("engine b");

    // Inject each other's direct address (the out-of-band step the account
    // service performs in production).
    engine_a.add_peer(direct_addr(&engine_b));
    engine_b.add_peer(direct_addr(&engine_a));

    // A dials B on the sync ALPN and round-trips a probe payload through the
    // accept-side echo hook.
    let conn = engine_a
        .connect(direct_addr(&engine_b))
        .await
        .expect("dial b on sync alpn");
    assert_eq!(
        conn.remote_id(),
        engine_b.endpoint_id(),
        "the dialled peer identity must match B's endpoint id"
    );

    let probe = b"minutist-sync-probe";
    let (mut send, mut recv) = conn.open_bi().await.expect("open bi");
    send.write_all(probe).await.expect("write probe");
    send.finish().expect("finish send");

    let echoed = recv.read_to_end(probe.len()).await.expect("read echo");
    assert_eq!(echoed, probe, "the accept hook must echo the probe bytes");

    conn.close(0u32.into(), b"done");
    engine_a.shutdown().await.expect("shutdown a");
    engine_b.shutdown().await.expect("shutdown b");
}
