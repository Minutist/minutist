//! Relay-only convergence: two `SyncEngine`s connect through a relay — dialled
//! by relay URL alone, no direct socket address — and reconcile a note.
//!
//! Local-by-default: a normal `cargo test` spins an in-process iroh test relay
//! (`iroh::test_utils::run_relay_server`, `AllowAll` access — no token needed)
//! and runs ungated, with no network dependency. This proves the relay-only
//! DATA flow (the account-directory addressing shape) without touching
//! anything deployed.
//!
//! Live-on-demand: when both `MINUTIST_SYNC_TOKEN` and `MINUTIST_SYNC_RELAY`
//! are set, the test instead runs against the deployed relay
//! (`sync.minutist.ai` by convention), carrying the token through
//! `RelayConfig::with_auth_token`. That is the smoke path for validating the
//! real deployment; it is never required for a plain `cargo test` or CI. Run
//! it with:
//!
//! ```sh
//! MINUTIST_SYNC_TOKEN=<relay-access-token> \
//! MINUTIST_SYNC_RELAY=https://sync.minutist.ai \
//!   cargo test -p sync --features test-support --test relay_live -- --nocapture
//! ```

use std::time::Duration;

use iroh::{EndpointAddr, RelayUrl};
use minutist_common::MeetingId;
use notes_crdt::NotesStore;
use sync::ContentKey;
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
async fn notes_converge_through_the_relay() {
    // Resolve the relay: live (deployed) only when BOTH env vars are set;
    // otherwise spin an in-process local relay so the test runs ungated with no
    // live dependency. `_relay_guard` is declared here (before the engines) so
    // it drops LAST, keeping the local relay alive for the whole test; it is
    // `None` on the live path, where nothing local needs to stay alive.
    let live_token = std::env::var("MINUTIST_SYNC_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let live_relay = std::env::var("MINUTIST_SYNC_RELAY")
        .ok()
        .filter(|r| !r.is_empty());
    let (relay_url, token, insecure, _relay_guard) = match (live_token, live_relay) {
        (Some(token), Some(relay_url)) => {
            eprintln!("relay_live: using LIVE relay {relay_url}");
            (relay_url, Some(token), false, None)
        }
        _ => {
            let (_relay_map, relay_url, guard) = iroh::test_utils::run_relay_server()
                .await
                .expect("spawn local test relay");
            let relay_url = relay_url.to_string();
            eprintln!("relay_live: using LOCAL relay {relay_url}");
            (relay_url, None, true, Some(guard))
        }
    };

    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");

    let cfg = |dir: &std::path::Path| SyncConfig {
        app_data_dir: dir.to_path_buf(),
        relay_url: relay_url.clone(),
        relay_auth_token: token.clone(),
        // The test conflates the device-key base and the meetings root onto one
        // temp dir: the key and the per-meeting folders both sit directly under
        // it. Production keeps them distinct (base vs `base/meetings`).
        meetings_root: dir.to_path_buf(),
        backoff_policy: Default::default(),
        relay_ips: Vec::new(),
    };

    // `start_insecure` trusts the local relay's self-signed certificate instead
    // of verifying it; the live path verifies normally via `start`.
    let (engine_a, engine_b) = if insecure {
        let a = SyncEngine::start_insecure(cfg(dir_a.path()), id_a, Some(ContentKey::for_tests()))
            .await
            .expect("engine A binds");
        let b = SyncEngine::start_insecure(cfg(dir_b.path()), id_b, Some(ContentKey::for_tests()))
            .await
            .expect("engine B binds");
        (a, b)
    } else {
        let a = SyncEngine::start(cfg(dir_a.path()), id_a, Some(ContentKey::for_tests()))
            .await
            .expect("engine A binds");
        let b = SyncEngine::start(cfg(dir_b.path()), id_b, Some(ContentKey::for_tests()))
            .await
            .expect("engine B binds");
        (a, b)
    };
    eprintln!(
        "relay_live: A={} B={}",
        engine_a.endpoint_id(),
        engine_b.endpoint_id()
    );

    // Mutual pairing — RELAY-ONLY (id + relay url, NO direct addrs). This is what
    // makes the test a genuine relay-data-plane proof: `my_ticket()` carries direct
    // socket addrs, so ticket-pairing would let the two engines holepunch directly
    // on localhost and converge EVEN IF THE RELAY WERE DOWN (the false positive an
    // earlier pass hit). Pairing by id+relay only — the account-directory addressing
    // shape — means a broken relay is no route at all, so the 30s guard below fires.
    // (Pairing is also the authorisation: B's accept side rejects an inbound
    // connection from a peer it has not paired.)
    let relay: RelayUrl = relay_url.parse().expect("relay url parses");
    engine_a.add_peer(EndpointAddr::new(engine_b.endpoint_id()).with_relay_url(relay.clone()));
    engine_b.add_peer(EndpointAddr::new(engine_a.endpoint_id()).with_relay_url(relay.clone()));

    // Let both endpoints home to the relay.
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

    // Address B by RELAY ONLY — no direct IPs — so the dial must be brokered by
    // the relay rather than connecting directly. (On localhost iroh may still
    // upgrade the brokered connection to a direct path once it discovers one;
    // that is expected and does not undermine the addressing proof — the point
    // is that the peer is reachable and dialable from relay-only address
    // information, the account-directory addressing shape.)
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
