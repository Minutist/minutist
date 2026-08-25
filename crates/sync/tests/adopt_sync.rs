//! `adopt_from_peer` backfill: a hub-role engine pulls every meeting a peer holds
//! that it LACKS, and skips those it already has. Runs relay-brokered through an
//! in-process iroh test relay (the hub's id+relay addressing shape), mirroring
//! `relay_live`'s local-by-default harness.

use std::path::Path;
use std::time::Duration;

use iroh::{EndpointAddr, RelayUrl};
use minutist_common::MeetingId;
use notes_crdt::{MeetingFolder, NotesStore};
use serde_json::json;
use sync::ContentKey;
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use uuid::Uuid;

fn seed(root: &Path, id: MeetingId, text: &str) {
    MeetingFolder::ensure(root, id).expect("ensure meeting folder");
    let doc = json!({
        "type": "doc",
        "content": [{ "type": "paragraph",
            "content": [{ "type": "text", "text": text }] }]
    });
    NotesStore::save(root, id, &doc, text).expect("seed notes.ydoc");
}

/// Project a meeting's `notes.ydoc` to comparable JSON, independent of encoding.
fn projected(root: &Path, id: MeetingId) -> serde_json::Value {
    let v1 = NotesStore::read_ydoc_state(root, id)
        .expect("read ydoc state")
        .expect("meeting has a notes.ydoc");
    let doc = notes_crdt::ydoc::new_ydoc();
    notes_crdt::ydoc::apply_update_v1(&doc, &v1).expect("apply v1 state");
    notes_crdt::ydoc::ydoc_to_json(&doc)
}

#[tokio::test]
async fn adopt_pulls_lacked_meetings_and_skips_held() {
    let (_relay_map, relay_url, _relay_guard) = iroh::test_utils::run_relay_server()
        .await
        .expect("spawn local test relay");
    let relay_url = relay_url.to_string();
    let relay: RelayUrl = relay_url.parse().expect("relay url parses");

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");

    let cfg = |dir: &Path| SyncConfig {
        app_data_dir: dir.to_path_buf(),
        relay_url: relay_url.clone(),
        relay_auth_token: None,
        meetings_root: dir.to_path_buf(),
        backoff_policy: Default::default(),
        relay_ips: Vec::new(),
    };
    let a = SyncEngine::start_insecure(cfg(dir_a.path()), id_a, Some(ContentKey::for_tests()))
        .await
        .expect("engine A binds");
    let b = SyncEngine::start_insecure(cfg(dir_b.path()), id_b, Some(ContentKey::for_tests()))
        .await
        .expect("engine B binds");

    // Pair relay-only (id + relay), the hub addressing shape. Pairing is also the
    // accept-side authorisation.
    a.add_peer(EndpointAddr::new(b.endpoint_id()).with_relay_url(relay.clone()));
    b.add_peer(EndpointAddr::new(a.endpoint_id()).with_relay_url(relay.clone()));
    tokio::time::sleep(Duration::from_secs(3)).await;

    // A (the "device") holds two meetings; B (the "hub") already holds one of them.
    let m1 = MeetingId(Uuid::new_v4());
    let m2 = MeetingId(Uuid::new_v4());
    seed(dir_a.path(), m1, "meeting one");
    seed(dir_a.path(), m2, "meeting two");
    seed(dir_b.path(), m1, "meeting one"); // B already has m1

    // B adopts A: pulls m2 (lacked), skips m1 (held).
    let never_purged = |_: MeetingId| false;
    let adopted = tokio::time::timeout(
        Duration::from_secs(30),
        b.adopt_from_peer(&a.endpoint_id().to_string(), &never_purged),
    )
    .await
    .expect("adopt timed out")
    .expect("adopt failed");
    assert_eq!(
        adopted, 1,
        "adopts only the meeting B lacks (m2), skips the held m1"
    );
    assert_eq!(
        projected(dir_b.path(), m2),
        projected(dir_a.path(), m2),
        "B converged to A's m2 notes"
    );

    // Idempotent: a second adopt pulls nothing new (B now holds all of A's).
    let again = b
        .adopt_from_peer(&a.endpoint_id().to_string(), &never_purged)
        .await
        .expect("second adopt failed");
    assert_eq!(
        again, 0,
        "a second adopt pulls nothing once B holds all of A's meetings"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}

/// A meeting B already purged (recorded in its local purged set) must NOT be
/// resurrected just because A still advertises it — the whole point of the
/// purge-guard (`crates/persistence/src/purged.rs`'s design rationale).
#[tokio::test]
async fn adopt_never_resurrects_a_locally_purged_meeting() {
    let (_relay_map, relay_url, _relay_guard) = iroh::test_utils::run_relay_server()
        .await
        .expect("spawn local test relay");
    let relay_url = relay_url.to_string();
    let relay: RelayUrl = relay_url.parse().expect("relay url parses");

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");

    let cfg = |dir: &Path| SyncConfig {
        app_data_dir: dir.to_path_buf(),
        relay_url: relay_url.clone(),
        relay_auth_token: None,
        meetings_root: dir.to_path_buf(),
        backoff_policy: Default::default(),
        relay_ips: Vec::new(),
    };
    let a = SyncEngine::start_insecure(cfg(dir_a.path()), id_a, Some(ContentKey::for_tests()))
        .await
        .expect("engine A binds");
    let b = SyncEngine::start_insecure(cfg(dir_b.path()), id_b, Some(ContentKey::for_tests()))
        .await
        .expect("engine B binds");

    a.add_peer(EndpointAddr::new(b.endpoint_id()).with_relay_url(relay.clone()));
    b.add_peer(EndpointAddr::new(a.endpoint_id()).with_relay_url(relay.clone()));
    tokio::time::sleep(Duration::from_secs(3)).await;

    // A still holds `m` (it hasn't caught up to the purge); B does not — and
    // remembers, via `is_purged`, that its absence is deliberate.
    let m = MeetingId(Uuid::new_v4());
    seed(dir_a.path(), m, "purged meeting");
    let purged_locally_on_b = |id: MeetingId| id == m;

    let adopted = tokio::time::timeout(
        Duration::from_secs(30),
        b.adopt_from_peer(&a.endpoint_id().to_string(), &purged_locally_on_b),
    )
    .await
    .expect("adopt timed out")
    .expect("adopt failed");
    assert_eq!(adopted, 0, "a purged id must never be re-adopted");
    assert!(
        !dir_b.path().join(m.0.to_string()).exists(),
        "B must not have resurrected the purged meeting's folder"
    );

    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
}
