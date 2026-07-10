//! Live end-to-end check against the DEPLOYED relay (`sync.minutist.ai`).
//!
//! Two `SyncEngine`s bind with `RelayMode::Custom(sync.minutist.ai)` carrying the
//! relay access token, then one dials the other **addressed by relay URL only**
//! (no direct socket addresses) so the connection must be brokered by the relay
//! rather than going direct over loopback. A note seeded on A is reconciled to B
//! over that relay-routed connection. Passing proves, against the real
//! deployment: relay reachability, the shared-token `AccessControl`, and the
//! notes-sync protocol — the things the in-process tests (which use
//! `RelayMode::Disabled`) cannot exercise.
//!
//! GATED: skips unless `MINUTIST_SYNC_TOKEN` is set, so a normal `cargo test`
//! and CI never touch the network. Run:
//!
//! ```sh
//! MINUTIST_SYNC_TOKEN=<relay-access-token> \
//!   cargo test -p sync --features test-support --test relay_live -- --nocapture
//! ```
//! `MINUTIST_SYNC_RELAY` overrides the relay URL (default the crate's
//! `DEFAULT_RELAY_URL`).

use std::time::Duration;

use iroh::{EndpointAddr, RelayUrl};
use minutist_common::MeetingId;
use notes_crdt::NotesStore;
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use uuid::Uuid;

/// Project a meeting's authoritative `notes.ydoc` to ProseMirror JSON, using only
/// public `persistence` APIs, so the two devices' converged state can be compared
/// independent of v1 encoding details.
fn projected(root: &std::path::Path, meeting: MeetingId) -> serde_json::Value {
    let v1 = NotesStore::read_ydoc_state(root, meeting)
        .expect("read ydoc state")
        .expect("meeting has a notes.ydoc");
    let doc = notes_crdt::ydoc::new_ydoc();
    notes_crdt::ydoc::apply_update_v1(&doc, &v1).expect("apply v1 state");
    notes_crdt::ydoc::ydoc_to_json(&doc)
}

#[tokio::test]
async fn notes_converge_through_the_deployed_relay() {
    let token = match std::env::var("MINUTIST_SYNC_TOKEN") {
        Ok(t) if !t.is_empty() => t,
        _ => {
            eprintln!("SKIP relay_live: set MINUTIST_SYNC_TOKEN to run the live relay test");
            return;
        }
    };
    let relay_url = std::env::var("MINUTIST_SYNC_RELAY")
        .unwrap_or_else(|_| SyncConfig::DEFAULT_RELAY_URL.into());
    eprintln!("relay_live: using relay {relay_url}");

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");

    let cfg = |dir: &std::path::Path| SyncConfig {
        relay_url: relay_url.clone(),
        relay_auth_token: Some(token.clone()),
        // Live test conflates the device-key base and the meetings root onto one
        // temp dir: the key and the per-meeting folders both sit directly under
        // it. Production keeps them distinct (base vs `base/meetings`).
        meetings_root: dir.to_path_buf(),
    };

    let engine_a = SyncEngine::start(cfg(dir_a.path()), id_a)
        .await
        .expect("engine A binds");
    let engine_b = SyncEngine::start(cfg(dir_b.path()), id_b)
        .await
        .expect("engine B binds");
    eprintln!(
        "relay_live: A={} B={}",
        engine_a.endpoint_id(),
        engine_b.endpoint_id()
    );

    // Mutual pairing: each device adds the other's ticket. Sync requires it —
    // B's accept side rejects an inbound connection from a peer it has not paired,
    // so A must be in B's directory for the dial below to be served.
    engine_a
        .add_peer_from_ticket(&engine_b.my_ticket())
        .expect("A pairs B");
    engine_b
        .add_peer_from_ticket(&engine_a.my_ticket())
        .expect("B pairs A");

    // Let both endpoints home to the relay (the token-gated relay handshake).
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Seed a note on A only.
    let meeting = MeetingId(Uuid::new_v4());
    let json = serde_json::json!({
        "type": "doc",
        "content": [{ "type": "paragraph",
            "content": [{ "type": "text", "text": "hello over the relay" }] }]
    });
    notes_crdt::MeetingFolder::ensure(dir_a.path(), meeting).expect("ensure A meeting folder");
    NotesStore::save(dir_a.path(), meeting, &json, "hello over the relay").expect("seed A");

    // Address B by RELAY ONLY — no direct IPs — so the dial is brokered by the
    // deployed relay rather than connecting directly over loopback.
    let relay: RelayUrl = relay_url.parse().expect("relay url parses");
    let b_relay_only = EndpointAddr::new(engine_b.endpoint_id()).with_relay_url(relay);

    // A dials B through the relay and reconciles the meeting (30s guard so a relay
    // failure surfaces as a timeout, not a hang).
    tokio::time::timeout(
        Duration::from_secs(30),
        engine_a.sync_notes(b_relay_only, meeting),
    )
    .await
    .expect("relay-routed notes sync timed out")
    .expect("relay-routed notes sync failed");

    // B must now hold A's note, byte-for-byte at the projection level.
    let a_json = projected(dir_a.path(), meeting);
    let b_json = projected(dir_b.path(), meeting);
    assert_eq!(a_json, b_json, "B must converge to A's note via the relay");
    assert!(
        serde_json::to_string(&b_json)
            .unwrap()
            .contains("hello over the relay"),
        "B's converged note must carry the seeded text"
    );

    engine_a.shutdown().await.expect("shutdown a");
    engine_b.shutdown().await.expect("shutdown b");
    eprintln!("relay_live: PASS — notes converged through {relay_url}");
}
