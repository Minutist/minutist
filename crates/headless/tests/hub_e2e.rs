//! End-to-end check: two devices converge THROUGH a real, running `minutist-hub`
//! daemon — the parked-peer model the hub deployment relies on.
//!
//! Spawns the actual `minutist-hub` binary as the always-on middle peer, pairs two
//! `SyncEngine`s (A and B) with it via the relay, seeds a note on A and pushes it
//! to the hub, then has B pull it from the hub. B converging proves the hub
//! authorises paired peers, holds the merged CRDT, and serves it back over a
//! relay, addressed relay-only so the path is genuinely brokered.
//!
//! `notes_converge_through_a_running_hub` is LOCAL-BY-DEFAULT: a plain
//! `cargo test` spins an in-process iroh test relay
//! (`iroh::test_utils::run_relay_server`) and runs ungated, with no live
//! dependency. The spawned `minutist-hub` binary is told to trust that relay's
//! self-signed certificate via `MINUTIST_HUB_INSECURE_RELAY_TLS` (see
//! `bind_sync_engine` in `src/main.rs`, compiled only under this crate's
//! `test-support` feature — never in a production build). Live-on-demand: when
//! both `MINUTIST_SYNC_TOKEN` and `MINUTIST_SYNC_RELAY` are set, it instead runs
//! against the deployed relay, the real smoke-test path:
//!
//! ```sh
//! MINUTIST_SYNC_TOKEN=<relay-access-token> \
//! MINUTIST_SYNC_RELAY=https://sync.minutist.ai \
//!   cargo test -p headless --test hub_e2e -- --nocapture
//! ```
//!
//! The other two hub_e2e cases (`hub_pushes_a_meeting_to_an_arriving_peer`,
//! `hub_records_a_peers_processing_lifecycle_via_discovery`) remain GATED: they
//! skip unless `MINUTIST_SYNC_TOKEN` is set, so a normal `cargo test` and CI
//! never touch the network for them.
//!
//! `hub_discovers_and_syncs_an_account_listed_peer` is LOCAL-BY-DEFAULT like
//! `notes_converge_through_a_running_hub`, but covers the production
//! account-mediated discovery path instead of the `MINUTIST_HUB_TEST_PEERS`
//! seam the other cases use: a local mock account-service (`wiremock`) serves
//! a peer's endpoint, and the hub (seeded with a credential, no test peers)
//! must acquire it purely through `StoredCredential` -> `AccountDirectoryClient`
//! -> the account-refresh loop.
//!
//! `create_meeting_seeds_a_meeting_folder_with_notes` is a local, ungated
//! one-shot that invokes the built binary's `create-meeting` and asserts the
//! on-disk meeting + its status digest; it needs no network and always runs.

use std::process::Stdio;
use std::time::Duration;

use iroh::{EndpointAddr, RelayUrl};
use minutist_common::{HostRef, MeetingId, ProcessingLifecycle};
use notes_crdt::NotesStore;
use sha2::Digest;
use sync::{DeviceIdentity, SyncConfig, SyncEngine};
use uuid::Uuid;

/// Project a meeting's authoritative `notes.ydoc` to ProseMirror JSON via public
/// `persistence` APIs, so the two devices' converged state compares independent of
/// v1 encoding details.
fn projected(root: &std::path::Path, meeting: MeetingId) -> serde_json::Value {
    let v1 = NotesStore::read_ydoc_state(root, meeting)
        .expect("read ydoc state")
        .expect("meeting has a notes.ydoc");
    let doc = notes_crdt::ydoc::new_ydoc();
    notes_crdt::ydoc::apply_update_v1(&doc, &v1).expect("apply v1 state");
    notes_crdt::ydoc::ydoc_to_json(&doc)
}

/// Spawn the real minutist-hub daemon for a test: collapsed sub-second timers and
/// piped stderr drained in the background, returning the child plus a signal that
/// fires once the daemon logs its readiness marker. `kill_on_drop` cleans up.
///
/// `token` is `None` for the local in-process test relay (which needs none);
/// `insecure_relay_tls` sets `MINUTIST_HUB_INSECURE_RELAY_TLS` so the daemon
/// trusts that relay's self-signed certificate instead of verifying it — the
/// `test-support`-gated escape hatch in `src/main.rs`'s `bind_sync_engine`.
/// `test_peers` are the peers' own tickets (`SyncEngine::my_ticket`), fed to the
/// daemon via `MINUTIST_HUB_TEST_PEERS` (`src/main.rs`'s `seed_test_peers`,
/// also `test-support`-gated) since the daemon has no account to discover them
/// through in this local test. `account_api_url`, when `Some`, points the
/// daemon's account-mediated discovery at a mock account-service instead of
/// the real one (see `hub_discovers_and_syncs_an_account_listed_peer`); `None`
/// leaves the default, irrelevant for a test that never seeds a credential.
fn spawn_hub(
    hub_dir: &std::path::Path,
    relay_url: &str,
    token: Option<&str>,
    insecure_relay_tls: bool,
    test_peers: &[String],
    account_api_url: Option<&str>,
) -> (tokio::process::Child, tokio::sync::oneshot::Receiver<()>) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_minutist-hub"));
    command
        .args([
            "--data-dir",
            hub_dir.to_str().expect("utf8 hub dir"),
            "--relay-url",
            relay_url,
        ])
        .env("RUST_LOG", "hub=info,iroh=error,iroh_relay=error")
        // Collapse the hub's timers so reconnect/push/discovery scenarios run
        // sub-second.
        .env("MINUTIST_HUB_PUSH_DEBOUNCE_MS", "200")
        .env("MINUTIST_HUB_DISCOVERY_MS", "500")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(token) = token {
        command.env("MINUTIST_SYNC_TOKEN", token);
    }
    if insecure_relay_tls {
        command.env("MINUTIST_HUB_INSECURE_RELAY_TLS", "1");
    }
    if !test_peers.is_empty() {
        command.env("MINUTIST_HUB_TEST_PEERS", test_peers.join(","));
    }
    if let Some(url) = account_api_url {
        command.env("MINUTIST_HUB_API_URL", url);
    }
    let mut child = command.spawn().expect("spawn minutist-hub");
    let stderr = child.stderr.take().expect("hub stderr piped");
    let (tx, rx) = tokio::sync::oneshot::channel();
    // Drain stderr (so the daemon never blocks on a full pipe) and fire once ready.
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        let mut tx = Some(tx);
        while let Ok(Some(line)) = lines.next_line().await {
            if line.contains("minutist-hub ready") {
                if let Some(tx) = tx.take() {
                    let _ = tx.send(());
                }
            }
        }
    });
    (child, rx)
}

/// Reconcile a meeting with `addr`, retrying until the relay-routed path succeeds
/// or a budget elapses — replaces a fixed homing sleep (peers home at their pace).
async fn sync_with_retry(
    engine: &SyncEngine,
    addr: &EndpointAddr,
    meeting: MeetingId,
    label: &str,
) {
    let start = std::time::Instant::now();
    loop {
        match tokio::time::timeout(
            Duration::from_secs(15),
            engine.sync_notes(addr.clone(), meeting),
        )
        .await
        {
            Ok(Ok(())) => return,
            other => {
                assert!(
                    start.elapsed() < Duration::from_secs(45),
                    "{label} did not succeed within 45s: {other:?}"
                );
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
}

/// The hub-side oracle: run `minutist-hub status` against a data dir and return a
/// meeting's content digest (if held).
fn status_digest(
    data_dir: &std::path::Path,
    relay_url: &str,
    meeting: MeetingId,
) -> Option<String> {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_minutist-hub"))
        .args([
            "--data-dir",
            data_dir.to_str().expect("utf8 data dir"),
            "--relay-url",
            relay_url,
            "status",
        ])
        .output()
        .expect("run status");
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).expect("parse status json");
    let id = meeting.0.to_string();
    json["meetings"]
        .as_array()?
        .iter()
        .find(|m| m["id"] == id)
        .and_then(|m| m["digest"].as_str().map(str::to_owned))
}

#[tokio::test]
async fn notes_converge_through_a_running_hub() {
    // Resolve the relay: live (deployed) only when BOTH env vars are set;
    // otherwise spin an in-process local relay so the test runs ungated with no
    // live dependency. `_relay_guard` is bound in this, the function's FIRST
    // `let`, so it drops LAST (Rust drops locals in reverse declaration order),
    // keeping the local relay alive for the whole test. `insecure` marks the
    // local path, where both the in-process engines and the spawned hub daemon
    // must trust the local relay's self-signed certificate instead of
    // verifying it.
    let live_token = std::env::var("MINUTIST_SYNC_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let live_relay = std::env::var("MINUTIST_SYNC_RELAY")
        .ok()
        .filter(|r| !r.is_empty());
    let (relay_url, token, insecure, _relay_guard) = match (live_token, live_relay) {
        (Some(token), Some(relay_url)) => {
            eprintln!("hub_e2e: using LIVE relay {relay_url}");
            (relay_url, Some(token), false, None)
        }
        _ => {
            let (_relay_map, relay_url, guard) = iroh::test_utils::run_relay_server()
                .await
                .expect("spawn local test relay");
            let relay_url = relay_url.to_string();
            eprintln!("hub_e2e: using LOCAL relay {relay_url}");
            (relay_url, None, true, Some(guard))
        }
    };

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
        relay_auth_token: token.clone(),
        meetings_root: dir.to_path_buf(),
        backoff_policy: Default::default(),
        relay_ips: Vec::new(),
    };
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let id_b = DeviceIdentity::load_or_generate(dir_b.path()).expect("identity b");
    // `start_insecure` trusts the local relay's self-signed certificate instead
    // of verifying it; the live path verifies normally via `start`.
    let (engine_a, engine_b) = if insecure {
        let a = SyncEngine::start_insecure(cfg(dir_a.path()), id_a)
            .await
            .expect("engine A binds");
        let b = SyncEngine::start_insecure(cfg(dir_b.path()), id_b)
            .await
            .expect("engine B binds");
        (a, b)
    } else {
        let a = SyncEngine::start(cfg(dir_a.path()), id_a)
            .await
            .expect("engine A binds");
        let b = SyncEngine::start(cfg(dir_b.path()), id_b)
            .await
            .expect("engine B binds");
        (a, b)
    };

    // The hub must authorise A and B on startup — feed their tickets in via
    // MINUTIST_HUB_TEST_PEERS (see `spawn_hub`'s doc); the daemon has no account
    // to discover them through here.
    //
    // These are FULL tickets (`my_ticket()` carries direct socket addrs), so the hub
    // learns A/B's direct addrs. This test's relay-requirement rests on the hub NEVER
    // pulling note content — it only serves inbound A/B→hub connections (relay-only
    // from A/B's side, added below) and pushes its OWN meetings; it never dials A/B to
    // fetch notes. If a future change makes the hub reconcile missing meetings by
    // DIALING peers, harden these to relay-only tickets
    // (`EndpointTicket::new(EndpointAddr::new(id).with_relay_url(relay))`), or the hub
    // could fetch over a direct localhost path with the relay down — a false positive.
    // relay_live is the structurally relay-only proof (both ends withhold direct
    // addrs); this test's role is the local-relay + hub-binary integration.
    let test_peers = [engine_a.my_ticket(), engine_b.my_ticket()];

    // Launch the real daemon (collapsed timers; readiness via its stderr marker).
    // It reloads `hub_id` and adds the seeded peers; kill_on_drop cleans up on
    // panic. On the local path it is told (via `insecure`) to trust the relay's
    // self-signed certificate the same way the in-process engines above do.
    let (mut hub, ready) = spawn_hub(
        hub_dir.path(),
        &relay_url,
        token.as_deref(),
        insecure,
        &test_peers,
        None,
    );
    tokio::time::timeout(Duration::from_secs(20), ready)
        .await
        .expect("hub did not become ready within 20s")
        .expect("hub ready signal dropped");

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
    notes_crdt::MeetingFolder::ensure(dir_a.path(), meeting).expect("ensure A meeting folder");
    notes_crdt::MeetingFolder::ensure(dir_b.path(), meeting).expect("ensure B meeting folder");
    NotesStore::save(dir_a.path(), meeting, &json, "hello through the hub").expect("seed A");

    // A pushes the note to the hub; B pulls it back (retry until all three home).
    sync_with_retry(&engine_a, &hub_addr, meeting, "A->hub").await;
    sync_with_retry(&engine_b, &hub_addr, meeting, "B<-hub").await;

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

    // Hub-as-oracle: `minutist-hub status` reports the converged meeting with a
    // content digest. (Content convergence is asserted above via `projected`; this
    // confirms the status command surfaces the held meeting for a harness. A/B use
    // the relay_live flat `meetings_root`, so status — which expects the daemon's
    // `{data-dir}/meetings` layout — is meaningful only against the hub.)
    let hub_dig = status_digest(hub_dir.path(), &relay_url, meeting);
    assert!(
        hub_dig.is_some(),
        "hub status must report the converged meeting with a digest"
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
        backoff_policy: Default::default(),
        relay_ips: Vec::new(),
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

    let test_peers = [engine_a.my_ticket(), engine_b.my_ticket()];

    let (mut hub, ready) = spawn_hub(hub_dir.path(), &relay_url, Some(&token), false, &test_peers, None);
    tokio::time::timeout(Duration::from_secs(20), ready)
        .await
        .expect("hub did not become ready within 20s")
        .expect("hub ready signal dropped");

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
    notes_crdt::MeetingFolder::ensure(dir_a.path(), meeting_x).expect("ensure A/X");
    NotesStore::save(
        dir_a.path(),
        meeting_x,
        &jx,
        "deposited by A while B was away",
    )
    .expect("seed A/X");
    sync_with_retry(&engine_a, &hub_addr, meeting_x, "A->hub X").await;

    // B ARRIVES: it connects to deposit its OWN unrelated meeting Y. That inbound
    // connection fires the hub's peer-arrived event, which pushes ALL meetings
    // (including X) back to B.
    let meeting_y = MeetingId(Uuid::new_v4());
    let jy = serde_json::json!({"type":"doc","content":[{"type":"paragraph",
        "content":[{"type":"text","text":"B's own meeting"}]}]});
    notes_crdt::MeetingFolder::ensure(dir_b.path(), meeting_y).expect("ensure B/Y");
    NotesStore::save(dir_b.path(), meeting_y, &jy, "B's own meeting").expect("seed B/Y");
    sync_with_retry(&engine_b, &hub_addr, meeting_y, "B->hub Y").await;

    // B must receive X from the hub's reciprocal push — without ever syncing X.
    let got_x = wait_for_meeting(dir_b.path(), meeting_x, Duration::from_secs(45)).await;
    assert!(got_x, "the hub must push A's meeting X to B on arrival");
    assert!(
        serde_json::to_string(&projected(dir_b.path(), meeting_x))
            .unwrap()
            .contains("deposited by A while B was away"),
        "B's pushed copy of X must carry A's text"
    );

    // Hub-as-oracle: the hub's status reports BOTH meetings it accumulated.
    assert!(
        status_digest(hub_dir.path(), &relay_url, meeting_x).is_some(),
        "hub status must report meeting X"
    );
    assert!(
        status_digest(hub_dir.path(), &relay_url, meeting_y).is_some(),
        "hub status must report meeting Y"
    );

    engine_a.shutdown().await.expect("shutdown a");
    engine_b.shutdown().await.expect("shutdown b");
    let _ = hub.kill().await;
    eprintln!("hub_e2e push: PASS — hub pushed A's meeting to B on arrival");
}

/// Discovery-scheduling: a peer's processing-lifecycle state reaches the hub's
/// `metadata.json` via the hub's discovery — the §7 ride-along on the peer-arrival
/// push and the periodic recovery sweep (both collapsed sub-second by `spawn_hub`).
/// A flags a meeting `Processed` and pushes only its NOTES to the hub; the
/// `Processed` state has no transport of its own, so the hub recording it proves
/// discovery carried it.
#[tokio::test]
async fn hub_records_a_peers_processing_lifecycle_via_discovery() {
    let token = match std::env::var("MINUTIST_SYNC_TOKEN") {
        Ok(t) if !t.is_empty() => t,
        _ => {
            eprintln!("SKIP hub_e2e lifecycle: set MINUTIST_SYNC_TOKEN to run");
            return;
        }
    };
    let relay_url = std::env::var("MINUTIST_SYNC_RELAY")
        .unwrap_or_else(|_| SyncConfig::DEFAULT_RELAY_URL.into());

    let hub_dir = tempfile::TempDir::new().expect("hub tempdir");
    let dir_a = tempfile::TempDir::new().expect("tempdir a");

    let hub_id = DeviceIdentity::load_or_generate(hub_dir.path())
        .expect("hub identity")
        .endpoint_id();

    let cfg = |dir: &std::path::Path| SyncConfig {
        relay_url: relay_url.clone(),
        relay_auth_token: Some(token.clone()),
        meetings_root: dir.to_path_buf(),
        backoff_policy: Default::default(),
        relay_ips: Vec::new(),
    };
    let engine_a = SyncEngine::start(
        cfg(dir_a.path()),
        DeviceIdentity::load_or_generate(dir_a.path()).expect("id a"),
    )
    .await
    .expect("engine A");

    let test_peers = [engine_a.my_ticket()];

    let (mut hub, ready) = spawn_hub(hub_dir.path(), &relay_url, Some(&token), false, &test_peers, None);
    tokio::time::timeout(Duration::from_secs(20), ready)
        .await
        .expect("hub did not become ready within 20s")
        .expect("hub ready signal dropped");

    let relay: RelayUrl = relay_url.parse().expect("relay url");
    let hub_addr = EndpointAddr::new(hub_id).with_relay_url(relay);
    engine_a.add_peer(hub_addr.clone());

    // A seeds a meeting, flags it Processed, and pushes its NOTES to the hub (which
    // seeds the meeting folder on the hub). The Processed state has no transport of
    // its own — it reaches the hub only through discovery.
    let meeting = MeetingId(Uuid::new_v4());
    let jx = serde_json::json!({"type":"doc","content":[{"type":"paragraph",
        "content":[{"type":"text","text":"processed by A"}]}]});
    notes_crdt::MeetingFolder::ensure(dir_a.path(), meeting).expect("ensure A meeting");
    NotesStore::save(dir_a.path(), meeting, &jx, "processed by A").expect("seed A");
    let processed = ProcessingLifecycle::Processed {
        processed_by: HostRef("endpoint-a".into()),
        at: "2026-06-29T10:00:00Z".into(),
    };
    persistence::meeting_ops::apply_processing_lifecycle(dir_a.path(), meeting, processed.clone())
        .await
        .expect("flag A meeting processed");
    sync_with_retry(&engine_a, &hub_addr, meeting, "A->hub notes").await;

    // The hub must end up recording Processed for the meeting — applied from A's
    // discovery (the ride-along on A's arrival and/or the periodic sweep).
    let hub_meeting_dir = hub_dir.path().join("meetings").join(meeting.0.to_string());
    let got = wait_for_processing(&hub_meeting_dir, &processed, Duration::from_secs(45)).await;
    assert!(
        got,
        "the hub must record the peer's Processed lifecycle via discovery"
    );

    engine_a.shutdown().await.expect("shutdown a");
    let _ = hub.kill().await;
    eprintln!("hub_e2e lifecycle: PASS — A's Processed state reached the hub via discovery");
}

/// Poll until `dir`'s `metadata.json` records `expected` processing, or `deadline`.
async fn wait_for_processing(
    dir: &std::path::Path,
    expected: &ProcessingLifecycle,
    deadline: Duration,
) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if let Ok(meta) = persistence::read_metadata(dir) {
            if &meta.processing == expected {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Production discovery-path coverage: unlike every other case in this file
/// (which seed the hub's peer directory directly via `MINUTIST_HUB_TEST_PEERS`,
/// a test-only escape hatch), this drives the REAL chain a deployed hub uses:
/// `StoredCredential::load` -> `AccountDirectoryClient` -> the account-refresh
/// loop -> `upsert_account_peer`. A's endpoint is served from a local mock
/// account-service (`wiremock`, no live dependency); the hub is seeded with a
/// credential and pointed at the mock via `MINUTIST_HUB_API_URL`, with an EMPTY
/// `test_peers` list, so A pushing successfully to the hub can only mean the
/// account-refresh loop actually authorised it.
///
/// LOCAL-BY-DEFAULT like `notes_converge_through_a_running_hub`: an in-process
/// test relay when `MINUTIST_SYNC_TOKEN`/`MINUTIST_SYNC_RELAY` are unset, the
/// deployed relay otherwise.
#[tokio::test]
async fn hub_discovers_and_syncs_an_account_listed_peer() {
    let live_token = std::env::var("MINUTIST_SYNC_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    let live_relay = std::env::var("MINUTIST_SYNC_RELAY")
        .ok()
        .filter(|r| !r.is_empty());
    let (relay_url, token, insecure, _relay_guard) = match (live_token, live_relay) {
        (Some(token), Some(relay_url)) => (relay_url, Some(token), false, None),
        _ => {
            let (_relay_map, relay_url, guard) = iroh::test_utils::run_relay_server()
                .await
                .expect("spawn local test relay");
            (relay_url.to_string(), None, true, Some(guard))
        }
    };

    let hub_dir = tempfile::TempDir::new().expect("hub tempdir");
    let dir_a = tempfile::TempDir::new().expect("tempdir a");

    let hub_id = DeviceIdentity::load_or_generate(hub_dir.path())
        .expect("hub identity")
        .endpoint_id();

    let cfg = SyncConfig {
        relay_url: relay_url.clone(),
        relay_auth_token: token.clone(),
        meetings_root: dir_a.path().to_path_buf(),
        backoff_policy: Default::default(),
        relay_ips: Vec::new(),
    };
    let id_a = DeviceIdentity::load_or_generate(dir_a.path()).expect("identity a");
    let engine_a = if insecure {
        SyncEngine::start_insecure(cfg, id_a)
            .await
            .expect("engine A binds")
    } else {
        SyncEngine::start(cfg, id_a).await.expect("engine A binds")
    };

    // The mock account-service: `GET /v1/account/devices` lists A's endpoint;
    // `PUT .../self/endpoint` (the hub registering itself) just needs to succeed.
    let mock = wiremock::MockServer::start().await;
    let devices = serde_json::json!([{
        "device_id": "device-a",
        "endpoint_id": engine_a.endpoint_id().to_string(),
        "relay_url": relay_url,
        "direct_addrs": Vec::<String>::new(),
    }]);
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/v1/account/devices"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(&devices))
        .mount(&mock)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("PUT"))
        .and(wiremock::matchers::path("/v1/account/devices/self/endpoint"))
        .respond_with(wiremock::ResponseTemplate::new(204))
        .mount(&mock)
        .await;

    // Seed the hub's credential so `start_engine` wires up account-mediated
    // discovery against the mock (`spawn_hub`'s `account_api_url`) instead of
    // logging "no device credential found" and discovering nothing.
    std::fs::write(
        hub_dir.path().join("tunnel_device.json"),
        r#"{"device_credential":"mdc_test.secret","account_id":"acct-test","device_id":"hub-test"}"#,
    )
    .expect("seed hub credential");

    let (mut hub, ready) = spawn_hub(
        hub_dir.path(),
        &relay_url,
        token.as_deref(),
        insecure,
        &[], // no MINUTIST_HUB_TEST_PEERS: peer acquisition must come from discovery
        Some(&mock.uri()),
    );
    tokio::time::timeout(Duration::from_secs(20), ready)
        .await
        .expect("hub did not become ready within 20s")
        .expect("hub ready signal dropped");

    let relay: RelayUrl = relay_url.parse().expect("relay url parses");
    let hub_addr = EndpointAddr::new(hub_id).with_relay_url(relay);
    // A must still authorise the hub to dial it back; the hub authorising A is
    // exactly the property under test (via the account-refresh loop, not this).
    engine_a.add_peer(hub_addr.clone());

    let meeting = MeetingId(Uuid::new_v4());
    let json = serde_json::json!({"type":"doc","content":[{"type":"paragraph",
        "content":[{"type":"text","text":"discovered via the account service"}]}]});
    notes_crdt::MeetingFolder::ensure(dir_a.path(), meeting).expect("ensure A meeting folder");
    NotesStore::save(dir_a.path(), meeting, &json, "discovered via the account service")
        .expect("seed A");

    // Succeeds only if the hub's PeerDirectory already authorises A inbound —
    // which the account-refresh loop's first (immediate) tick must have done.
    sync_with_retry(&engine_a, &hub_addr, meeting, "A->hub via account discovery").await;

    assert!(
        status_digest(hub_dir.path(), &relay_url, meeting).is_some(),
        "the hub must hold the meeting pushed by an account-discovered peer"
    );

    engine_a.shutdown().await.expect("shutdown a");
    let _ = hub.kill().await;
    eprintln!(
        "hub_e2e account-discovery: PASS — A synced to the hub purely via account-mediated discovery"
    );
}

/// `create-meeting` subcommand test: originate a meeting in a data directory,
/// verify the meeting folder and notes.ydoc are created, and confirm the
/// `status` command reports the meeting with a non-empty digest.
#[test]
fn create_meeting_seeds_a_meeting_folder_with_notes() {
    let data_dir = tempfile::TempDir::new().expect("tempdir");
    let dir = data_dir.path();

    // Run `minutist-hub --data-dir <dir> create-meeting --title "Test Meeting"`
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_minutist-hub"))
        .args([
            "--data-dir",
            dir.to_str().expect("utf8 data dir"),
            "create-meeting",
            "--title",
            "Test Meeting",
        ])
        .output()
        .expect("run create-meeting");

    assert!(
        output.status.success(),
        "create-meeting must exit cleanly; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The command prints the UUID to stdout.
    let uuid_str = String::from_utf8(output.stdout)
        .expect("parse stdout as utf8")
        .trim()
        .to_string();
    let meeting_id = MeetingId(
        uuid::Uuid::parse_str(&uuid_str).expect("parse UUID from create-meeting output"),
    );

    // Verify the meeting folder exists.
    let meetings_root = dir.join("meetings");
    let meeting_dir = meetings_root.join(meeting_id.0.to_string());
    assert!(meeting_dir.exists(), "meeting folder must be created");

    // Verify metadata.json exists (created by MeetingFolder::ensure).
    let metadata_path = meeting_dir.join("metadata.json");
    assert!(metadata_path.exists(), "metadata.json must be created");

    // Verify notes.ydoc exists (created by NotesStore::save).
    let notes_path = meeting_dir.join("notes.ydoc");
    assert!(notes_path.exists(), "notes.ydoc must be created");

    // Verify the notes have content by projecting to JSON and checking digest.
    let v1 = NotesStore::read_ydoc_state(&meetings_root, meeting_id)
        .expect("read ydoc state")
        .expect("meeting has a notes.ydoc");
    let doc = notes_crdt::ydoc::new_ydoc();
    notes_crdt::ydoc::apply_update_v1(&doc, &v1).expect("apply v1 state");
    let json = notes_crdt::ydoc::ydoc_to_json(&doc);
    let bytes = serde_json::to_vec(&json).expect("serialize json");
    let digest = format!("{:x}", sha2::Sha256::digest(&bytes));
    assert!(!digest.is_empty(), "digest must be non-empty");
    assert_eq!(digest.len(), 64, "sha256 hex digest must be 64 chars");

    // Verify `status` reports the meeting with the same digest.
    let status_output = std::process::Command::new(env!("CARGO_BIN_EXE_minutist-hub"))
        .args([
            "--data-dir",
            dir.to_str().expect("utf8 data dir"),
            "--relay-url",
            "wss://relay.example.invalid",
            "status",
        ])
        .output()
        .expect("run status");

    assert!(
        status_output.status.success(),
        "status must exit cleanly"
    );

    let status_json: serde_json::Value = serde_json::from_slice(&status_output.stdout)
        .expect("parse status json");
    let meetings_arr = status_json["meetings"]
        .as_array()
        .expect("status must have meetings array");
    let created_meeting = meetings_arr
        .iter()
        .find(|m| m["id"] == uuid_str)
        .expect("created meeting must be listed in status");

    assert_eq!(
        created_meeting["ydoc_present"], true,
        "status must report ydoc_present=true"
    );
    assert_eq!(
        created_meeting["digest"].as_str().expect("digest must be string"),
        digest,
        "status digest must match the projected digest"
    );
}
