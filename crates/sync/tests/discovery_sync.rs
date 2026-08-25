//! Round-trip proof for the WS-lifecycle discovery protocol
//! ([`StreamKind::Discovery`]).
//!
//! Two relay-less `SyncEngine`s, paired, exchange the `(MeetingId,
//! ProcessingLifecycle)` of every meeting they hold over a real (relay-less,
//! loopback) iroh connection. Asserts:
//!
//! - the initiator learns the responder's meeting list (the deferred meeting-list
//!   discovery) via [`SyncEngine::discover_with`]'s return value;
//! - each side receives the *other's* meeting lifecycle on its
//!   [`SyncEngine::subscribe_lifecycle_events`] surface — including a
//!   non-`Local` (`Processed`) state, proving the host-authoritative lifecycle
//!   crosses the wire, not just the id.
//!
//! `sync` never writes `metadata.json`; the seed states are written via
//! `persistence` (the authoritative writer), and `sync` reads them to advertise.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::Duration;

use iroh::EndpointAddr;
use minutist_common::{HostRef, MeetingId, ProcessingLifecycle};
use notes_crdt::MeetingFolder;
use sync::identity::DeviceIdentity;
use sync::ContentKey;
use sync::SyncEngine;
use tokio::time::timeout;

/// Loopback `EndpointAddr` for `engine` (mirrors the notes/round-trip helper).
fn direct_addr(engine: &SyncEngine) -> EndpointAddr {
    let mut addr = EndpointAddr::new(engine.endpoint_id());
    for sock in engine.bound_sockets() {
        let loopback = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), sock.port());
        addr = addr.with_ip_addr(loopback);
    }
    addr
}

/// Two relay-less engines, mutually paired (the out-of-band pairing the account
/// service performs).
async fn paired_engines(root_a: &Path, root_b: &Path) -> (SyncEngine, SyncEngine) {
    let id_a = DeviceIdentity::load_or_generate(root_a).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(root_b).expect("identity b");
    let a = SyncEngine::start_direct(id_a, Some(ContentKey::for_tests()), root_a.to_path_buf())
        .await
        .expect("engine a");
    let b = SyncEngine::start_direct(id_b, Some(ContentKey::for_tests()), root_b.to_path_buf())
        .await
        .expect("engine b");
    a.add_peer(direct_addr(&b));
    b.add_peer(direct_addr(&a));
    (a, b)
}

#[tokio::test]
async fn discovery_exchanges_meeting_list_and_lifecycle() {
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let root_a = dir_a.path();
    let root_b = dir_b.path();

    // B holds a meeting flagged `Processed` (a non-Local, host-authoritative
    // state) so the test proves the lifecycle — not just the id — crosses.
    let b_meeting = MeetingId::new();
    // `ensure` seeds the placeholder `metadata.json` (Local) that
    // `apply_processing_lifecycle` then overwrites; `create` makes only the folder.
    MeetingFolder::ensure(root_b, b_meeting).expect("ensure b meeting");
    let processed = ProcessingLifecycle::Processed {
        processed_by: HostRef("endpoint-b".into()),
        at: "2026-06-27T10:00:00Z".into(),
    };
    persistence::meeting_ops::apply_processing_lifecycle(root_b, b_meeting, processed.clone())
        .await
        .expect("flag b meeting processed");

    // A holds a different meeting, left at the placeholder `Local`.
    let a_meeting = MeetingId::new();
    MeetingFolder::create(root_a, a_meeting).expect("create a meeting");

    let (a, b) = paired_engines(root_a, root_b).await;

    // Subscribe both sides BEFORE the exchange so neither event is missed.
    let mut rx_a = a.subscribe_lifecycle_events();
    let mut rx_b = b.subscribe_lifecycle_events();

    // A initiates discovery with B.
    let discovered = a
        .discover_with(direct_addr(&b))
        .await
        .expect("discover a -> b");

    // A learned B's meeting list (the meeting-list discovery).
    assert_eq!(discovered, vec![b_meeting], "A must learn B's meeting list");

    // A received B's meeting's lifecycle on the event surface — the Processed
    // state, proving the host-authoritative lifecycle crossed the wire.
    let (id, processing, _deletion) = timeout(Duration::from_secs(5), rx_a.recv())
        .await
        .expect("A lifecycle event within timeout")
        .expect("A lifecycle event");
    assert_eq!(id, b_meeting);
    assert_eq!(processing, processed, "A must receive B's Processed state");

    // B (the responder) symmetrically learned A's meeting (placeholder `Local`).
    let (id_b_side, proc_b_side, _deletion_b_side) = timeout(Duration::from_secs(5), rx_b.recv())
        .await
        .expect("B lifecycle event within timeout")
        .expect("B lifecycle event");
    assert_eq!(id_b_side, a_meeting);
    assert_eq!(proc_b_side, ProcessingLifecycle::Local);

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}
