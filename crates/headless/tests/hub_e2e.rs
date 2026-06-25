//! Gated live end-to-end check: two devices converge THROUGH a real, running
//! `minutist-hub` daemon — the parked-peer model the hub deployment relies on.
//!
//! Spawns the actual `minutist-hub` binary as the always-on middle peer, pairs two
//! `SyncEngine`s (A and B) with it via the relay, seeds a note on A and pushes it
//! to the hub, then has B pull it from the hub. B converging proves the hub
//! authorises paired peers, holds the merged CRDT, and serves it back — over the
//! deployed relay (`sync.minutist.ai`), addressed relay-only so the path is
//! genuinely brokered.
//!
//! GATED: skips unless `MINUTIST_SYNC_TOKEN` is set, so a normal `cargo test` and
//! CI never touch the network. Run:
//!
//! ```sh
//! MINUTIST_SYNC_TOKEN=<relay-access-token> \
//!   cargo test -p headless --test hub_e2e -- --nocapture
//! ```
//! `MINUTIST_SYNC_RELAY` overrides the relay URL (default `DEFAULT_RELAY_URL`).

use std::process::Stdio;
use std::time::Duration;

use iroh::{EndpointAddr, RelayUrl};
use minutist_common::MeetingId;
use persistence::NotesStore;
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use uuid::Uuid;

/// Project a meeting's authoritative `notes.ydoc` to ProseMirror JSON via public
/// `persistence` APIs, so the two devices' converged state compares independent of
/// v1 encoding details.
fn projected(root: &std::path::Path, meeting: MeetingId) -> serde_json::Value {
    let v1 = NotesStore::read_ydoc_state(root, meeting)
        .expect("read ydoc state")
        .expect("meeting has a notes.ydoc");
    let doc = persistence::ydoc::new_ydoc();
    persistence::ydoc::apply_update_v1(&doc, &v1).expect("apply v1 state");
    persistence::ydoc::ydoc_to_json(&doc)
}

#[tokio::test]
async fn notes_converge_through_a_running_hub() {
    let token = match std::env::var("MINUTIST_SYNC_TOKEN") {
        Ok(t) if !t.is_empty() => t,
        _ => {
            eprintln!("SKIP hub_e2e: set MINUTIST_SYNC_TOKEN to run the live hub test");
            return;
        }
    };
    let relay_url = std::env::var("MINUTIST_SYNC_RELAY")
        .unwrap_or_else(|_| SyncConfig::DEFAULT_RELAY_URL.into());
    eprintln!("hub_e2e: using relay {relay_url}");

    let hub_dir = tempfile::TempDir::new().expect("hub tempdir");
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");

    // Pre-create the hub's device identity so we know its EndpointId without
    // parsing the daemon's logs; the daemon reloads this exact key on startup.
    let hub_id = DeviceIdentity::load_or_generate(hub_dir.path())
        .expect("hub identity")
        .endpoint_id();

    let cfg = |dir: &std::path::Path| SyncConfig {
        relay_url: relay_url.clone(),
        relay_auth_token: Some(token.clone()),
        meetings_root: dir.to_path_buf(),
    };
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");
    let engine_a = SyncEngine::start(cfg(dir_a.path()), id_a)
        .await
        .expect("engine A binds");
    let engine_b = SyncEngine::start(cfg(dir_b.path()), id_b)
        .await
        .expect("engine B binds");

    // The hub must authorise A and B on startup — write their tickets to its peers
    // file before launching it (the daemon loads the file as it comes up).
    std::fs::write(
        hub_dir.path().join("peers"),
        format!("{}\n{}\n", engine_a.my_ticket(), engine_b.my_ticket()),
    )
    .expect("write hub peers file");

    // Launch the real daemon against the hub data dir (it reloads `hub_id` and the
    // peers). kill_on_drop guarantees no orphan if an assertion panics.
    let bin = env!("CARGO_BIN_EXE_minutist-hub");
    let mut hub = tokio::process::Command::new(bin)
        .args([
            "--data-dir",
            hub_dir.path().to_str().expect("utf8 hub dir"),
            "--relay-url",
            &relay_url,
        ])
        .env("MINUTIST_SYNC_TOKEN", &token)
        .env("RUST_LOG", "hub=info,iroh=error,iroh_relay=error")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn minutist-hub");

    // Let A, B, and the hub all bind and home to the token-gated relay.
    tokio::time::sleep(Duration::from_secs(8)).await;

    // Address the hub by relay only so the dial is brokered, not direct.
    let relay: RelayUrl = relay_url.parse().expect("relay url parses");
    let hub_addr = EndpointAddr::new(hub_id).with_relay_url(relay);
    engine_a.add_peer(hub_addr.clone());
    engine_b.add_peer(hub_addr.clone());

    // Seed a note on A and ensure B's folder exists for the inbound apply.
    let meeting = MeetingId(Uuid::new_v4());
    let json = serde_json::json!({
        "type": "doc",
        "content": [{ "type": "paragraph",
            "content": [{ "type": "text", "text": "hello through the hub" }] }]
    });
    persistence::MeetingFolder::ensure(dir_a.path(), meeting).expect("ensure A meeting folder");
    persistence::MeetingFolder::ensure(dir_b.path(), meeting).expect("ensure B meeting folder");
    NotesStore::save(dir_a.path(), meeting, &json, "hello through the hub").expect("seed A");

    // A pushes the note to the hub (the hub merges it as the responder).
    tokio::time::timeout(
        Duration::from_secs(30),
        engine_a.sync_notes(hub_addr.clone(), meeting),
    )
    .await
    .expect("A->hub notes sync timed out")
    .expect("A->hub notes sync failed");

    // B pulls it from the hub (B is the initiator; it also applies the hub's diff).
    tokio::time::timeout(
        Duration::from_secs(30),
        engine_b.sync_notes(hub_addr.clone(), meeting),
    )
    .await
    .expect("B<-hub notes sync timed out")
    .expect("B<-hub notes sync failed");

    // B must now hold A's note, converged through the hub.
    let a_json = projected(dir_a.path(), meeting);
    let b_json = projected(dir_b.path(), meeting);
    assert_eq!(
        a_json, b_json,
        "B must converge to A's note through the hub"
    );
    assert!(
        serde_json::to_string(&b_json)
            .unwrap()
            .contains("hello through the hub"),
        "B's converged note must carry the seeded text"
    );

    engine_a.shutdown().await.expect("shutdown a");
    engine_b.shutdown().await.expect("shutdown b");
    let _ = hub.kill().await;
    eprintln!("hub_e2e: PASS — A's note converged to B through the running hub");
}

/// Poll until `meeting`'s `notes.ydoc` exists under `root`, or `deadline` elapses.
async fn wait_for_meeting(root: &std::path::Path, meeting: MeetingId, deadline: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if matches!(NotesStore::read_ydoc_state(root, meeting), Ok(Some(_))) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// S5 — push-on-reconnect: when a peer ARRIVES (opens an authorised inbound
/// connection), the hub reciprocally pushes everything it holds. A deposits a
/// meeting while B is away; later B connects (to deposit its own, unrelated
/// meeting) and must end up with A's meeting WITHOUT ever asking for it.
#[tokio::test]
async fn hub_pushes_a_meeting_to_an_arriving_peer() {
    let token = match std::env::var("MINUTIST_SYNC_TOKEN") {
        Ok(t) if !t.is_empty() => t,
        _ => {
            eprintln!("SKIP hub_e2e push: set MINUTIST_SYNC_TOKEN to run");
            return;
        }
    };
    let relay_url = std::env::var("MINUTIST_SYNC_RELAY")
        .unwrap_or_else(|_| SyncConfig::DEFAULT_RELAY_URL.into());

    let hub_dir = tempfile::TempDir::new().expect("hub tempdir");
    let dir_a = tempfile::TempDir::new().expect("tempdir a");
    let dir_b = tempfile::TempDir::new().expect("tempdir b");

    let hub_id = DeviceIdentity::load_or_generate(hub_dir.path())
        .expect("hub identity")
        .endpoint_id();

    let cfg = |dir: &std::path::Path| SyncConfig {
        relay_url: relay_url.clone(),
        relay_auth_token: Some(token.clone()),
        meetings_root: dir.to_path_buf(),
    };
    let engine_a = SyncEngine::start(
        cfg(dir_a.path()),
        DeviceIdentity::load_or_generate(dir_a.path()).expect("id a"),
    )
    .await
    .expect("engine A");
    let engine_b = SyncEngine::start(
        cfg(dir_b.path()),
        DeviceIdentity::load_or_generate(dir_b.path()).expect("id b"),
    )
    .await
    .expect("engine B");

    std::fs::write(
        hub_dir.path().join("peers"),
        format!("{}\n{}\n", engine_a.my_ticket(), engine_b.my_ticket()),
    )
    .expect("write hub peers");

    let bin = env!("CARGO_BIN_EXE_minutist-hub");
    let mut hub = tokio::process::Command::new(bin)
        .args([
            "--data-dir",
            hub_dir.path().to_str().expect("utf8"),
            "--relay-url",
            &relay_url,
        ])
        .env("MINUTIST_SYNC_TOKEN", &token)
        .env("RUST_LOG", "hub=info,iroh=error,iroh_relay=error")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn minutist-hub");

    tokio::time::sleep(Duration::from_secs(8)).await;

    let relay: RelayUrl = relay_url.parse().expect("relay url");
    let hub_addr = EndpointAddr::new(hub_id).with_relay_url(relay);
    // Both must have the hub in their directory: A/B to dial it, and B to AUTHORISE
    // the hub's reciprocal push-back (an inbound connection from the hub).
    engine_a.add_peer(hub_addr.clone());
    engine_b.add_peer(hub_addr.clone());

    // A deposits meeting X with the hub (while B is "away").
    let meeting_x = MeetingId(Uuid::new_v4());
    let jx = serde_json::json!({"type":"doc","content":[{"type":"paragraph",
        "content":[{"type":"text","text":"deposited by A while B was away"}]}]});
    persistence::MeetingFolder::ensure(dir_a.path(), meeting_x).expect("ensure A/X");
    NotesStore::save(
        dir_a.path(),
        meeting_x,
        &jx,
        "deposited by A while B was away",
    )
    .expect("seed A/X");
    tokio::time::timeout(
        Duration::from_secs(30),
        engine_a.sync_notes(hub_addr.clone(), meeting_x),
    )
    .await
    .expect("A->hub timed out")
    .expect("A->hub failed");

    // B ARRIVES: it connects to deposit its OWN unrelated meeting Y. That inbound
    // connection fires the hub's peer-arrived event, which pushes ALL meetings
    // (including X) back to B.
    let meeting_y = MeetingId(Uuid::new_v4());
    let jy = serde_json::json!({"type":"doc","content":[{"type":"paragraph",
        "content":[{"type":"text","text":"B's own meeting"}]}]});
    persistence::MeetingFolder::ensure(dir_b.path(), meeting_y).expect("ensure B/Y");
    NotesStore::save(dir_b.path(), meeting_y, &jy, "B's own meeting").expect("seed B/Y");
    tokio::time::timeout(
        Duration::from_secs(30),
        engine_b.sync_notes(hub_addr.clone(), meeting_y),
    )
    .await
    .expect("B->hub timed out")
    .expect("B->hub failed");

    // B must receive X from the hub's reciprocal push — without ever syncing X.
    let got_x = wait_for_meeting(dir_b.path(), meeting_x, Duration::from_secs(45)).await;
    assert!(got_x, "the hub must push A's meeting X to B on arrival");
    assert!(
        serde_json::to_string(&projected(dir_b.path(), meeting_x))
            .unwrap()
            .contains("deposited by A while B was away"),
        "B's pushed copy of X must carry A's text"
    );

    engine_a.shutdown().await.expect("shutdown a");
    engine_b.shutdown().await.expect("shutdown b");
    let _ = hub.kill().await;
    eprintln!("hub_e2e push: PASS — hub pushed A's meeting to B on arrival");
}
